"""Deep, provider-specific Nebius Serverless AI job-contact mechanics.

This module deliberately stops at provider contact.  It does not interpret worker
markers, scientific tools, object storage, artifacts, residue, or manifests.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from bio_neb.agentic_journey.failure_cause import cause_identity
from bio_neb.content import canonical_json
from bio_neb.providers.job_observations import (
    FetchedProviderEnvelope,
    ProviderCreateObserved,
    ProviderLogWindowFailureEvidence,
    ProviderStateObserved,
    bounded_log_observation,
)
from bio_neb.providers.nebius_authentication import (
    NebiusAuthenticationProbe,
)
from bio_neb.providers.nebius_common import (
    AmbiguousNebiusCreate,
    NebiusCliError,
    NebiusTenantNotActive,
    ProviderCreateFailureReason,
    nebius_common_arguments,
)
from bio_neb.providers.runtime import NebiusCliTarget

# The authoritative vocabulary is JobStatus.State in nebius/ai/v1/job.proto:
# STATE_UNSPECIFIED, PROVISIONING, STARTING, RUNNING, CANCELLING, DELETING, COMPLETED,
# FAILED, CANCELLED, ERROR, IMAGE_PULLING.  Names outside it -- SUCCEEDED, QUEUED,
# PENDING, TIMED_OUT, TIMEOUT and CANCELED -- must never appear in the terminal sets
# below, where they would decide real outcomes.
_TERMINAL_SUCCESS = frozenset({"COMPLETED"})
_TERMINAL_FAILURE = frozenset({"FAILED", "CANCELLED", "ERROR"})
# These two closed sets are the whole classification.  There is no allowlist of in-flight
# states: any state outside both sets is in flight, recorded with its literal string, and
# polled again.  IMAGE_PULLING is the concrete reason this is generalized rather than widened
# by one name -- it is a real documented state that a poll can easily miss (one sampled run
# left a 147.2 s unobserved window exactly where it would have appeared), and it must not be
# able to crash a Run the first time it does.
# In-flight states have no order either: the provider guarantees none, and a live job has
# been seen stepping IMAGE_PULLING back to an earlier stage and then completing.  A step back
# is polled through; only a change after a terminal fails, and its reason names both states.
# Absence has two halves and needs both.  `ai job get-by-name` exits 13 with `rpc
# error: code = NotFound` for a name that does not exist, but docs.nebius.com/cli/exit-codes
# states that an error raised by an *extra* call the CLI makes internally reports the target
# code plus 40, and the identical NotFound condition has surfaced as 53 from `ai job get`.
# So the numeric half of the proof is a set, not a constant, and a `get-by-name` that
# resolves through a secondary lookup must not be read as a hard failure.
_NAME_NOT_FOUND_EXIT_CODES = frozenset({13, 53})
_JOB_ID_PATTERN = re.compile(r"aijob-[a-z0-9]+(?:-[a-z0-9]+)*\Z")

ContactErrorCategory = Literal[
    "authentication",
    "runtime_invalid",
    "name_not_absent",
    "lookup_failed",
    "create_rejected",
    "reconciliation_failed",
    "contact_failed",
    "envelope_drift",
    "payload_invalid",
    "provider_failed",
    "log_invalid",
]


class NebiusJobContactError(RuntimeError):
    """Secret-free failure with no recovery or mutation authority."""

    def __init__(
        self,
        operation: str,
        category: ContactErrorCategory,
        *,
        reason: str | None = None,
        provider_reason: ProviderCreateFailureReason | None = None,
        provider_job_id: str | None = None,
        name_match_count: int | None = None,
        log_window: ProviderLogWindowFailureEvidence | None = None,
    ) -> None:
        super().__init__(reason or category)
        self.operation = operation
        self.category = category
        self.reason = reason or category
        self.provider_reason = provider_reason
        self.provider_job_id = provider_job_id
        self.name_match_count = name_match_count
        # Set only when the failure is about one log window.
        self.log_window = log_window


@dataclass(frozen=True, slots=True)
class ProviderReadyJobRequest:
    """Secret-free provider-ready projection supplied by either stage."""

    target: NebiusCliTarget
    provider_job_name: str
    provider_image_reference: str
    platform: str
    preset: str
    disk_size: str
    disk_size_bytes: int
    shared_memory_size: str | None
    shared_memory_bytes: int
    # There is deliberately no timeout field.  The APP authors no deadline; the
    # provider's own spec.timeout is observed off the fetched envelope rather than
    # requested, so there is nothing here to carry.
    restart_policy: Literal["never"]
    normalized_arguments: tuple[str, ...]
    injection_provider_values: tuple[str, ...]
    injected_paths: tuple[str, ...]
    environment_names: tuple[str, ...]
    version_bound_secret_environment_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.provider_job_name
            or self.provider_job_name.strip() != self.provider_job_name
        ):
            raise ValueError("provider job name must be canonical")
        if self.disk_size_bytes < 0 or self.shared_memory_bytes < 0:
            raise ValueError("provider byte sizes cannot be negative")
        if len(self.injection_provider_values) != len(self.injected_paths):
            raise ValueError("provider injection values changed requested paths")
        if set(self.version_bound_secret_environment_names) - set(
            self.environment_names
        ):
            raise ValueError("secret environment names are absent")
        if len(set(self.environment_names)) != len(self.environment_names):
            raise ValueError("environment names contain duplicates")


@dataclass(frozen=True, slots=True)
class CreatedJobContact:
    """Provider identity taken by name lookup, plus the verbatim create response.

    ``create_response`` is the create call's exact stdout.  Ticket 140 section 3 keeps it
    as Run proof and never as an identity source: the caller retains it as an immutable
    content-addressed Journey artifact, while ``create`` carries its structured facts.
    Raw stderr deliberately does not appear here -- Ticket 138 observed the CLI's auth
    prompt carrying a live OAuth URL with code_challenge and state on that stream, so it
    goes to the operator record and Ticket 88's pool, never into the Journey.
    """

    create: ProviderCreateObserved
    create_response: str
    envelope: FetchedProviderEnvelope
    initial_state: ProviderStateObserved

    @property
    def provider_job_id(self) -> str:
        return self.create.provider_job_id


@dataclass(slots=True)
class _PrivateContact:
    request: ProviderReadyJobRequest
    expected_environment: tuple[dict[str, Any], ...]
    initial_envelope: FetchedProviderEnvelope
    state: ProviderStateObserved


def _nested(value: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _valid_job_id(value: Any) -> bool:
    return isinstance(value, str) and _JOB_ID_PATTERN.fullmatch(value) is not None


def _secret_selector(value: str) -> dict[str, str]:
    secret_id, separator, version_id = value.rpartition("@")
    if not separator or not secret_id or not version_id:
        raise ValueError("SecretStash selector must bind one secret and version")
    return {"secret_id": secret_id, "version_id": version_id}


def _binding(value: str, *, secret: bool) -> dict[str, Any]:
    name, separator, selected = value.partition("=")
    if (
        not separator
        or not name
        or not selected
        or name.strip() != name
        or selected.strip() != selected
    ):
        raise ValueError("environment binding must be one canonical assignment")
    return {
        "name": name,
        **(
            {"mysterybox_secret": _secret_selector(selected)}
            if secret
            else {"value": selected}
        ),
    }


class NebiusServerlessJobContact:
    """Own one exact Nebius Serverless AI mechanical contact state machine."""

    def __init__(
        self,
        *,
        cli_runner: Any,
        authentication: NebiusAuthenticationProbe | None = None,
    ) -> None:
        self._runner = cli_runner
        self._contacts: dict[str, _PrivateContact] = {}
        self._attempted_names: set[tuple[str, str]] = set()
        self._create_active = False
        # One owner for `iam whoami`.  When the Parabricks adapter supplies
        # its entry probe, the proof made there covers this preflight and no second call
        # happens; a standalone construction gets an equivalent probe of its own.
        self._authentication = authentication or NebiusAuthenticationProbe(
            cli_runner=cli_runner
        )
        self._preflighted: NebiusCliTarget | None = None

    @staticmethod
    def common_arguments(target: NebiusCliTarget) -> tuple[str, ...]:
        """Return the shared argv tail; Ticket 144 moved the body to nebius_common."""

        return nebius_common_arguments(target)

    def preflight(self, target: NebiusCliTarget) -> None:
        """Verify the exact native runner and complete selected job-parent topology.

        Ticket 144 memoized this against the *exact* target.  The capsule describes
        readiness as **one** bounded discovery chain before any mutation; the code had
        drifted into running the whole chain once per stage, so a Run made six identical
        reads where three were meant.  A stage carrying a different selection still gets
        its own chain rather than a silent pass.
        """

        if self._preflighted == target:
            return
        try:
            preflight = getattr(self._runner, "preflight", None)
            if not callable(preflight):
                raise NebiusJobContactError(
                    "preflight",
                    "runtime_invalid",
                    reason="cli_identity_preflight_missing",
                )
            try:
                identity = preflight()
            except Exception as error:
                raise NebiusJobContactError(
                    "preflight",
                    "runtime_invalid",
                    reason="cli_identity_preflight_failed",
                ) from error
            try:
                mismatch = (
                    identity.executable != target.executable
                    or identity.resolved_executable != target.resolved_executable
                    or identity.version != target.cli_version
                    or identity.byte_count != target.executable_byte_count
                    or identity.sha256 != target.executable_sha256
                )
            except Exception as error:
                raise NebiusJobContactError(
                    "preflight",
                    "runtime_invalid",
                    reason="cli_identity_preflight_invalid",
                ) from error
            if mismatch:
                raise NebiusJobContactError(
                    "preflight", "runtime_invalid", reason="cli_identity_mismatch"
                )
            common = self.common_arguments(target)
            try:
                self._authentication.prove(target)
            except NebiusTenantNotActive as error:
                raise NebiusJobContactError(
                    "preflight", "authentication", reason="tenant_not_active"
                ) from error
            project = self._runner.json(
                "iam", "project", "get", "--id", target.project_id, *common
            )
            subnet = self._runner.json(
                "vpc", "subnet", "get", "--id", target.subnet_id, *common
            )
            if (
                _nested(project, "metadata", "id") != target.project_id
                or _nested(project, "metadata", "parent_id") != target.tenant_id
                or _nested(subnet, "metadata", "id") != target.subnet_id
                or _nested(subnet, "metadata", "parent_id") != target.project_id
                or _nested(subnet, "status", "state") != "READY"
            ):
                raise NebiusJobContactError(
                    "preflight", "runtime_invalid", reason="job_parent_topology_invalid"
                )
            self._preflighted = target
        except NebiusJobContactError:
            raise
        except NebiusCliError as error:
            raise NebiusJobContactError(
                "preflight", "runtime_invalid", reason=error.journey_reason()
            ) from error
        except Exception as error:
            raise NebiusJobContactError(
                "preflight", "payload_invalid", reason=cause_identity(error)
            ) from error

    def _lookup_by_name(
        self, request: ProviderReadyJobRequest, *, operation: str
    ) -> dict[str, Any] | None:
        """Return the one job with this exact name, or None when it does not exist.

        This is the project's identity path, and Ticket 140 section 4 renamed it to say
        so.  It was previously reached through `ai job list` and described as a recovery
        after a lost create response; it is neither.  Every job id this project has ever
        held came through a name lookup, because on payload 0.12.235 the create response
        was structurally unparseable, and under Fork A the create response is 33 bytes of
        prose that we deliberately do not parse at all.  The lookup is the only reachable
        identity source, not a fallback from one.

        A NotFound answer is the only absence proof the CLI offers -- Tickets 146 and
        146a both measured it -- and Ticket 145 requires both halves of it before any
        create is authorized: an exit code in the NotFound set *and* a provider error
        stderr already classified `not_found`.  Neither half alone decides.  The number
        alone is not enough because exit 7 is any authentication error, and reading an
        expired session as "the job does not exist" would falsely prove absence and
        authorize a duplicate H100 create.  The classification alone is not enough
        because it is a substring match over prose, which a message about some other
        missing thing could satisfy.  Every other nonzero exit refuses.
        """

        try:
            payload = self._runner.json(
                "ai",
                "job",
                "get-by-name",
                "--parent-id",
                request.target.project_id,
                "--name",
                request.provider_job_name,
                *self.common_arguments(request.target),
            )
        except NebiusCliError as error:
            evidence = error.failure_evidence
            if (
                evidence is not None
                and evidence.child_exit_code in _NAME_NOT_FOUND_EXIT_CODES
                and evidence.stderr_classification == "not_found"
            ):
                return None
            raise NebiusJobContactError(
                operation, "lookup_failed", reason=error.journey_reason()
            ) from error
        name = _nested(payload, "metadata", "name")
        if name != request.provider_job_name:
            raise NebiusJobContactError(
                operation, "payload_invalid", reason="job_lookup_name_mismatch"
            )
        return payload

    @staticmethod
    def _create_arguments(
        request: ProviderReadyJobRequest,
        environment_bindings: Sequence[str],
        secret_environment_bindings: Sequence[str],
    ) -> tuple[str, ...]:
        arguments: list[str] = [
            "ai",
            "job",
            "create",
            "--parent-id",
            request.target.project_id,
            "--name",
            request.provider_job_name,
            "--image",
            request.provider_image_reference,
            "--args",
            shlex.join(request.normalized_arguments),
        ]
        for injection in request.injection_provider_values:
            arguments.extend(("--inject-file", injection))
        arguments.extend(
            (
                "--platform",
                request.platform,
                "--preset",
                request.preset,
                "--disk-size",
                request.disk_size,
            )
        )
        if request.shared_memory_size is not None:
            arguments.extend(("--shm-size", request.shared_memory_size))
        # No --timeout: the APP authors no deadline.  Any compiled ceiling here would be
        # invented; a four-hour one is roughly 25x a real 569.010 s germline run.  This is
        # *not* "no timer": ai job create --timeout cannot be opted out of, and omitting it
        # makes the server inject its own spec.timeout ("86400s" when observed).  A provider
        # deadline always exists and it runs from created_at, covering provisioning and the
        # image pull as well as the workload -- it has killed a real job during PROVISIONING.
        # Only who authors it differs.  The value the server chose is read back off the
        # fetched envelope and recorded; it is never asserted against one of ours.
        arguments.extend(
            (
                "--subnet-id",
                request.target.subnet_id,
                "--restart-policy",
                request.restart_policy,
                # create --async returns during PROVISIONING (+2.9 s to +12.65 s after
                # created_at) and observe owns everything afterwards.  The submitted job
                # does not need this process alive to keep running.
                "--async",
            )
        )
        for binding in environment_bindings:
            arguments.extend(("--env", binding))
        for binding in secret_environment_bindings:
            arguments.extend(("--env-secret", binding))
        arguments.extend(NebiusServerlessJobContact.common_arguments(request.target))
        return tuple(arguments)

    def create(
        self,
        request: ProviderReadyJobRequest,
        *,
        environment_bindings: tuple[str, ...],
        secret_environment_bindings: tuple[str, ...],
        reattest_before_create: Callable[[], None],
        record_create_invoked: Callable[[], None],
        reuse_named: bool = False,
    ) -> CreatedJobContact:
        """Prove absence, reattest, record once, create once, and validate one job."""

        if self._create_active:
            raise NebiusJobContactError(
                "create", "contact_failed", reason="create_operation_in_progress"
            )
        self._create_active = True
        try:
            return self._create_once(
                request,
                environment_bindings=environment_bindings,
                secret_environment_bindings=secret_environment_bindings,
                reattest_before_create=reattest_before_create,
                record_create_invoked=record_create_invoked,
                reuse_named=reuse_named,
            )
        finally:
            self._create_active = False

    def _create_once(
        self,
        request: ProviderReadyJobRequest,
        *,
        environment_bindings: tuple[str, ...],
        secret_environment_bindings: tuple[str, ...],
        reattest_before_create: Callable[[], None],
        record_create_invoked: Callable[[], None],
        reuse_named: bool = False,
    ) -> CreatedJobContact:
        """Run one create operation while the engine-global guard is held."""

        attempt = (request.target.project_id, request.provider_job_name)
        if not reuse_named and attempt in self._attempted_names:
            raise NebiusJobContactError(
                "create", "name_not_absent", reason="contact_already_attempted"
            )
        if not reuse_named:
            self._attempted_names.add(attempt)
        # Absence is proved before create, and a NotFound answer is the only proof the
        # CLI offers.  What counts as one: an exit code in the NotFound set together with a
        # provider error classified `not_found`, never a bare number, because falsely proving
        # absence here is what authorizes a duplicate H100 create.
        existing = self._lookup_by_name(request, operation="create")
        if existing is not None and not reuse_named:
            raise NebiusJobContactError(
                "create",
                "name_not_absent",
                reason="exact_name_preexisting",
                name_match_count=1,
            )
        try:
            environment = tuple(
                sorted(
                    (
                        *(
                            _binding(value, secret=False)
                            for value in environment_bindings
                        ),
                        *(
                            _binding(value, secret=True)
                            for value in secret_environment_bindings
                        ),
                    ),
                    key=lambda item: item["name"],
                )
            )
            if tuple(item["name"] for item in environment) != tuple(
                sorted(request.environment_names)
            ):
                raise ValueError("private environment changed public projection")
            derived_secret_names = tuple(
                item["name"] for item in environment if "mysterybox_secret" in item
            )
            if derived_secret_names != tuple(
                sorted(request.version_bound_secret_environment_names)
            ):
                raise ValueError("private secret bindings changed public projection")
        except Exception as error:
            raise NebiusJobContactError(
                "create", "payload_invalid", reason=cause_identity(error)
            ) from error

        if existing is not None:
            job_id, envelope, state = self._decode_job(
                existing, request=request, expected_environment=environment
            )
            self._contacts[job_id] = _PrivateContact(
                request=request,
                expected_environment=environment,
                initial_envelope=envelope,
                state=state,
            )
            return CreatedJobContact(
                create=ProviderCreateObserved(
                    provider_job_id=job_id,
                    provider_job_name=request.provider_job_name,
                    disposition="reused",
                    invocation_count=0,
                ),
                create_response="",
                envelope=envelope,
                initial_state=state,
            )

        # These callbacks belong to the stage application. Their exact exception
        # identity crosses this seam unchanged, and the attempted name is already
        # consumed before either callback can re-enter contact.
        reattest_before_create()
        record_create_invoked()
        try:
            ambiguous = False
            response_text = ""
            try:
                # The create response is never parsed.  Under --async it is 33 bytes of
                # prose -- `Job ID: aijob-...` -- and `create` ignores --format entirely.
                # Identity comes from the name lookup instead, so create ignoring --format
                # does not matter.  The bytes are retained verbatim as Run proof rather
                # than read.
                response_text = self._runner(
                    *self._create_arguments(
                        request, environment_bindings, secret_environment_bindings
                    )
                ).stdout
            except AmbiguousNebiusCreate:
                # The create call gave no usable answer.  This is not a lost response to
                # recover from: the name lookup below is the same identity path a clean
                # create uses, so only the disposition differs.
                ambiguous = True
            except NebiusCliError as error:
                raise NebiusJobContactError(
                    "create",
                    "create_rejected",
                    reason=error.journey_reason(),
                    provider_reason=error.provider_reason,
                ) from error
            payload = self._lookup_by_name(request, operation="create")
            if payload is None:
                raise NebiusJobContactError(
                    "create",
                    "lookup_failed",
                    reason="created_name_absent_after_create",
                    name_match_count=0,
                )
            job_id, envelope, state = self._decode_job(
                payload, request=request, expected_environment=environment
            )
            contact = CreatedJobContact(
                create=ProviderCreateObserved(
                    provider_job_id=job_id,
                    provider_job_name=request.provider_job_name,
                    disposition=(
                        "created_with_ambiguous_create_response"
                        if ambiguous
                        else "created"
                    ),
                    response_byte_count=len(response_text.encode("utf-8")),
                    response_sha256=hashlib.sha256(
                        response_text.encode("utf-8")
                    ).hexdigest(),
                ),
                create_response=response_text,
                envelope=envelope,
                initial_state=state,
            )
            self._contacts[job_id] = _PrivateContact(
                request=request,
                expected_environment=environment,
                initial_envelope=envelope,
                state=state,
            )
            return contact
        except NebiusJobContactError:
            raise
        except Exception as error:
            raise NebiusJobContactError(
                "create", "payload_invalid", reason=cause_identity(error)
            ) from error

    def _decode_job(
        self,
        payload: dict[str, Any],
        *,
        request: ProviderReadyJobRequest,
        expected_environment: Sequence[dict[str, Any]],
        expected_job_id: str | None = None,
    ) -> tuple[str, FetchedProviderEnvelope, ProviderStateObserved]:
        try:
            job_id = _nested(payload, "metadata", "id")
            name = _nested(payload, "metadata", "name")
            parent = _nested(payload, "metadata", "parent_id")
            spec = payload.get("spec")
            status = payload.get("status")
            if not isinstance(spec, dict):
                raise TypeError("provider job omitted spec")
            # Runtime state is not demanded here: identity and the spec envelope belong
            # to submission, state belongs to observe.  A job that has been created but has
            # not started is a real provider answer, not a malformed one.
            if status is None:
                status = {}
            if not isinstance(status, dict):
                raise TypeError("provider job status is not an object")
            environment = spec.get("environment_variables", [])
            injected = spec.get("injected_files")
            disk = spec.get("disk")
            if (
                not _valid_job_id(job_id)
                or (expected_job_id is not None and job_id != expected_job_id)
                or name != request.provider_job_name
                or parent != request.target.project_id
                or not isinstance(environment, list)
                or not all(isinstance(item, dict) for item in environment)
                or sorted(environment, key=lambda item: item.get("name", ""))
                != list(expected_environment)
                or not isinstance(injected, list)
                or not all(
                    isinstance(item, dict)
                    and set(item) == {"container_path"}
                    and isinstance(item["container_path"], str)
                    for item in injected
                )
                or tuple(sorted(item["container_path"] for item in injected))
                != tuple(sorted(request.injected_paths))
                or spec.get("image") != request.provider_image_reference
                or spec.get("container_command") is not None
                or spec.get("args") != shlex.join(request.normalized_arguments)
                or spec.get("volumes") not in (None, [])
                or spec.get("platform") != request.platform
                or spec.get("preset") != request.preset
                or not isinstance(disk, dict)
                or disk.get("size_bytes")
                not in (request.disk_size_bytes, str(request.disk_size_bytes))
                or spec.get("shm_size_bytes", 0)
                not in (request.shared_memory_bytes, str(request.shared_memory_bytes))
                or spec.get("subnet_id") != request.target.subnet_id
                or spec.get("restart_attempts", 0) not in (None, 0, "0")
                or spec.get("restart_policy", "never")
                not in (None, "", request.restart_policy)
            ):
                raise NebiusJobContactError(
                    "get",
                    "envelope_drift",
                    reason="provider_envelope_mismatch",
                    provider_job_id=(job_id if isinstance(job_id, str) else None),
                )
            # spec.timeout is read, never asserted.  The APP authors no deadline, so there
            # is no requested value to compare against and the server's injected one is simply
            # recorded.  With the flag omitted the server has injected "86400s"; whether that is
            # a fixed default or derived from platform, preset or quota is unknown from one
            # observation, so nothing here may assume its value.
            provider_timeout = spec.get("timeout")
            if not isinstance(provider_timeout, str) or not provider_timeout:
                raise TypeError("provider omitted its own spec timeout")
            # An unrecognised state is a recorded observation carrying the provider's
            # literal string, classified only by terminality, and never fatal.  Raising on a
            # state outside a fixed allowlist would fail real jobs on states the list misses,
            # PROVISIONING among them -- the state every real job starts in.
            state_name = status.get("state")
            if state_name is None:
                # The proto's own name for "not set", which is what an envelope with no
                # state key is actually saying.  It is in flight by terminality.
                state_name = "STATE_UNSPECIFIED"
            if not isinstance(state_name, str) or not state_name:
                raise TypeError("provider state is not a state name")
            normalized_state = state_name.upper()
            envelope = FetchedProviderEnvelope(
                provider_job_id=job_id,
                provider_job_name=name,
                project_id=parent,
                provider_image_reference=request.provider_image_reference,
                platform=request.platform,
                preset=request.preset,
                disk_size_bytes=request.disk_size_bytes,
                subnet_id=request.target.subnet_id,
                # The provider's own deadline, observed rather than requested.
                timeout=provider_timeout,
                container_command_requested="image-entrypoint",
                container_command_fetched=spec.get("container_command"),
                normalized_arguments=request.normalized_arguments,
                injected_paths=tuple(sorted(request.injected_paths)),
                environment_names=tuple(sorted(item["name"] for item in environment)),
                version_bound_secret_environment_names=tuple(
                    sorted(
                        item["name"]
                        for item in environment
                        if "mysterybox_secret" in item
                    )
                ),
                shared_memory_bytes=request.shared_memory_bytes,
                restart_fetched=(
                    "never" if spec.get("restart_policy") == "never" else "omitted"
                ),
            )
            state = ProviderStateObserved(
                provider_job_id=job_id,
                provider_state=normalized_state,
                # state_details is a terminal-failure explanation and nothing else.
                # Live jobs populate it only in the terminal FAILED envelope: it was absent in
                # all 41 PROVISIONING samples of a failed job and through an entire successful
                # run, including the terminal COMPLETED.  It is not a progress channel, so it
                # is read exactly where it means something and left alone everywhere else.
                state_details=(
                    self._safe_details(status.get("state_details"), environment)
                    if normalized_state in _TERMINAL_FAILURE
                    else ()
                ),
                instances=self._safe_instances(status.get("instances"), environment),
                created_at=_nested(payload, "metadata", "created_at"),
                started_at=status.get("started_at"),
                finished_at=status.get("finished_at"),
            )
            return job_id, envelope, state
        except NebiusJobContactError:
            raise
        except Exception as error:
            raise NebiusJobContactError(
                "get",
                "payload_invalid",
                reason=cause_identity(error),
                provider_job_id=(
                    job_id if isinstance(job_id, str) else expected_job_id
                ),
            ) from error

    @staticmethod
    def _safe_details(
        details: Any, environment: Sequence[dict[str, Any]]
    ) -> tuple[str, ...]:
        def canonical(value: str) -> str:
            if (
                not value
                or value.strip() != value
                or any(character in value for character in "\r\n\0")
            ):
                raise ValueError("provider detail is not one canonical line")
            return value

        def validate_mapping_strings(value: Any) -> None:
            if isinstance(value, str):
                canonical(value)
            elif isinstance(value, Mapping):
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TypeError("provider detail keys must be strings")
                    canonical(key)
                    validate_mapping_strings(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    validate_mapping_strings(item)

        if details is None:
            retained: tuple[str, ...] = ()
        elif isinstance(details, str):
            retained = (canonical(details),)
        elif isinstance(details, Mapping):
            validate_mapping_strings(details)
            retained = (canonical(canonical_json(dict(details))),)
        elif isinstance(details, (list, tuple)) and all(
            isinstance(item, str) for item in details
        ):
            retained = tuple(canonical(item) for item in details)
        else:
            raise TypeError("provider details have an unsupported shape")
        if any(
            marker in detail
            for detail in retained
            for marker in NebiusServerlessJobContact._forbidden_material(environment)
        ):
            raise ValueError("provider details reflected authorization material")
        return retained

    @staticmethod
    def _forbidden_material(environment: Sequence[dict[str, Any]]) -> set[str]:
        forbidden: set[str] = set()
        for item in environment:
            value = item.get("value")
            if isinstance(value, str) and item.get("name") == "AWS_ACCESS_KEY_ID":
                forbidden.add(value)
            secret = item.get("mysterybox_secret")
            if isinstance(secret, dict):
                secret_id = secret.get("secret_id")
                version_id = secret.get("version_id")
                if isinstance(secret_id, str) and isinstance(version_id, str):
                    forbidden.update(
                        (secret_id, version_id, f"{secret_id}@{version_id}")
                    )
        return forbidden

    @staticmethod
    def _safe_instances(
        instances: Any, environment: Sequence[dict[str, Any]]
    ) -> tuple[str, ...]:
        """Retain status.instances[] verbatim as diagnostic context, classifying nothing.

        Ticket 146 watched this key change shape three times inside one job: absent for
        338 s, then entries carrying state/compute_instance_id/compute_instance_state/
        private_ip, then a terminal envelope whose entries had no ``state`` key at all.
        Ticket 146a saw an entry with no ``state`` while the job was STARTING.  So the
        shape is decoded defensively and whatever arrives is recorded; ``state`` is
        optional, absence means nothing, and no value here is read as an outcome.
        """

        if instances is None:
            return ()
        entries = instances if isinstance(instances, (list, tuple)) else [instances]
        retained: list[str] = []
        for entry in entries:
            line = entry if isinstance(entry, str) else canonical_json(entry)
            if (
                not line
                or line.strip() != line
                or any(character in line for character in "\r\n\0")
            ):
                raise ValueError("provider instance is not one canonical line")
            retained.append(line)
        if any(
            marker in line
            for line in retained
            for marker in NebiusServerlessJobContact._forbidden_material(environment)
        ):
            raise ValueError("provider instances reflected authorization material")
        return tuple(retained)

    @staticmethod
    def _validate_transition(
        previous: ProviderStateObserved, current: ProviderStateObserved
    ) -> None:
        # Any in-flight state may follow another; poll_state refuses a change
        # after a terminal before it gets here.
        if current.created_at != previous.created_at:
            raise ValueError("provider created_at changed")
        if (
            previous.started_at is not None
            and current.started_at != previous.started_at
        ):
            raise ValueError("provider started_at changed or disappeared")
        if (
            previous.finished_at is not None
            and current.finished_at != previous.finished_at
        ):
            raise ValueError("provider finished_at changed or disappeared")
        for value in (current.created_at, current.started_at, current.finished_at):
            if value is not None:
                datetime.fromisoformat(value)

    def poll_state(self, job_id: str):
        """Check one exact retained job envelope and state, without reading logs."""
        private = self._contacts[job_id]
        try:
            payload = self._runner.json(
                "ai",
                "job",
                "get",
                "--id",
                job_id,
                *self.common_arguments(private.request.target),
            )
        except NebiusCliError as error:
            raise NebiusJobContactError(
                "get",
                "contact_failed",
                reason=error.journey_reason(),
                provider_job_id=job_id,
            ) from error
        _, envelope, state = self._decode_job(
            payload,
            request=private.request,
            expected_environment=private.expected_environment,
            expected_job_id=job_id,
        )
        if envelope != private.initial_envelope:
            raise NebiusJobContactError(
                "get", "envelope_drift", reason="fetched_projection_changed"
            )
        try:
            if private.state.provider_state in _TERMINAL_SUCCESS | _TERMINAL_FAILURE:
                if (
                    state.provider_state,
                    state.created_at,
                    state.started_at,
                    state.finished_at,
                ) != (
                    private.state.provider_state,
                    private.state.created_at,
                    private.state.started_at,
                    private.state.finished_at,
                ):
                    raise ValueError(
                        "terminal provider state changed: "
                        f"{private.state.provider_state} -> {state.provider_state}"
                    )
            else:
                self._validate_transition(private.state, state)
        except ValueError as error:
            raise NebiusJobContactError(
                "get", "payload_invalid", reason=str(error), provider_job_id=job_id
            ) from error
        changed = state.provider_state != private.state.provider_state
        private.state = state
        return state, changed

    def read_log(self, job_id: str, *, window: int):
        """Read one bounded diagnostic window, without interpreting worker markers."""
        if not 1 <= window <= 1000:
            raise ValueError("log window must contain 1..1000 lines")
        private = self._contacts[job_id]
        try:
            output = self._runner(
                "ai",
                "job",
                "logs",
                job_id,
                "--timestamps",
                "--tail",
                str(window),
                "--profile",
                private.request.target.profile,
                "--retries",
                "1",
                "--no-check-update",
                "--no-progress",
            ).stdout
        except NebiusCliError:
            return None
        if (
            any(
                value in output
                for value in self._forbidden_material(private.expected_environment)
            )
            or len(output.splitlines()) > window
        ):
            return None
        bounded_log_observation(job_id, output)
        return output

    def poll(self, job_id: str, *, window: int):
        """One bounded sample; cadence and worker closure belong to the caller."""
        if not 1 <= window <= 1000:
            raise ValueError("log window must contain 1..1000 lines")
        state, changed = self.poll_state(job_id)
        output = (
            self.read_log(job_id, window=window)
            if state.provider_state
            in {
                "RUNNING",
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "ERROR",
            }
            else None
        )
        return state, changed, output


__all__ = [
    "CreatedJobContact",
    "NebiusJobContactError",
    "NebiusServerlessJobContact",
    "ProviderReadyJobRequest",
]
