from __future__ import annotations

# Shared provider-shaped offline contact fixtures.
#
# A GREEN RESULT HERE IS NOT EVIDENCE ABOUT NEBIUS.  _NativeCli builds every response by
# parsing the argv it was just handed, so it can only replay what this repository already
# believes and can never discover a provider behaviour.  This limit is deliberate: every
# provider fact the fake encodes -- the 33-byte create response, exit 13 NotFound, the
# server-injected 86400s spec.timeout -- came from a live run, never from a passing test.
# When the provider changes, these fixtures keep passing.  Only a live attempt can say
# otherwise.
# ruff: noqa: F401
import copy
import unittest
from dataclasses import replace

from bio_neb.observation import AdapterHeartbeat
from bio_neb.parabricks.execution_observations import ProviderTerminalContact
from bio_neb.providers.job_observations import (
    BoundedProviderLogObserved,
    ProviderStateObserved,
)
from bio_neb.providers.nebius_common import (
    AmbiguousNebiusCreate,
    NebiusCliError,
    NebiusCliFailureEvidence,
    NebiusCliInstallIdentity,
    NebiusCliResponse,
)
from bio_neb.providers.nebius_serverless_job_contact import (
    NebiusJobContactError,
    NebiusServerlessJobContact,
    ProviderReadyJobRequest,
)
from bio_neb.providers.runtime import NebiusCliTarget


def _not_found(
    *, child_exit_code: int = 13, stderr_classification: str = "not_found"
) -> NebiusCliError:
    """The exact shape `ai job get-by-name` returns for a name that does not exist.

    The defaults are what Tickets 146 and 146a measured: exit 13 with `rpc error: code =
    NotFound`.  Both halves are parameters because Ticket 145 decides absence from both,
    and a helper that can only build the accepted combination proves nothing about the
    near misses -- the same NotFound reported through the CLI's extra-call exit code, a
    NotFound-shaped number carrying some other error, or an authentication failure.
    """

    return NebiusCliError(
        "provider_command_failed",
        failure_evidence=NebiusCliFailureEvidence(
            operation="ai.job.get-by-name",
            child_exit_code=child_exit_code,
            stdout_byte_count=0,
            stdout_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            stderr_byte_count=48,
            stderr_sha256="f" * 64,
            stderr_classification=stderr_classification,
        ),
    )


def _target() -> NebiusCliTarget:
    return NebiusCliTarget(
        controller_runtime="native_ubuntu",
        executable="/opt/nebius/bin/nebius",
        resolved_executable="/opt/nebius/bin/nebius",
        cli_version="0.12.263",
        executable_byte_count=1,
        executable_sha256="a" * 64,
        profile="accepted-profile",
        tenant_id="tenant-exact",
        project_id="project-exact",
        subnet_id="vpcsubnet-exact",
        registry_id="registry-exact",
    )


def _request() -> ProviderReadyJobRequest:
    return ProviderReadyJobRequest(
        target=_target(),
        provider_job_name="job-exact",
        provider_image_reference="registry.example/bio-neb:accepted",
        platform="cpu-d3",
        preset="4vcpu-16gb",
        disk_size="100Gi",
        disk_size_bytes=100 * 1024**3,
        shared_memory_size=None,
        shared_memory_bytes=0,
        restart_policy="never",
        normalized_arguments=("/opt/bio-neb/request.json",),
        injection_provider_values=(
            "/run/bio-neb/request.json:/opt/bio-neb/request.json",
        ),
        injected_paths=("/opt/bio-neb/request.json",),
        environment_names=(
            "AWS_ENDPOINT_URL",
            "AWS_REGION",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        version_bound_secret_environment_names=("AWS_SECRET_ACCESS_KEY",),
    )


def _h100_request() -> ProviderReadyJobRequest:
    return ProviderReadyJobRequest(
        target=_target(),
        provider_job_name="job-exact",
        provider_image_reference="registry.example/bio-neb:accepted",
        platform="gpu-h100-sxm",
        preset="1gpu-16vcpu-200gb",
        disk_size="250Gi",
        disk_size_bytes=250 * 1024**3,
        shared_memory_size="16Gi",
        shared_memory_bytes=16 * 1024**3,
        restart_policy="never",
        normalized_arguments=("/opt/bio-neb/request.json",),
        injection_provider_values=("/run/bio-neb/h100.json:/opt/bio-neb/request.json",),
        injected_paths=("/opt/bio-neb/request.json",),
        environment_names=(
            "AWS_ENDPOINT_URL",
            "AWS_REGION",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        version_bound_secret_environment_names=("AWS_SECRET_ACCESS_KEY",),
    )


_ENVIRONMENT = (
    "AWS_ENDPOINT_URL=https://storage.example",
    "AWS_REGION=eu-north1",
    "AWS_ACCESS_KEY_ID=contact-access",
)
_SECRET_ENVIRONMENT = ("AWS_SECRET_ACCESS_KEY=secret-exact@version-exact",)


class _NativeCli:
    def __init__(
        self,
        *,
        preexisting=False,
        ambiguous=False,
        non_json=False,
        create_error=None,
        reconciliation_count=1,
        match_id="aijob-exact",
        lookup_error=False,
        absence_error=None,
        get_error=False,
        create_mutator=None,
        get_mutators=(),
        log_outputs=("",),
        initial_state="STARTING",
        preflight_identity=None,
        preflight_failure=None,
        states=("COMPLETED",),
    ):
        self.preexisting = preexisting
        self.ambiguous = ambiguous
        self.non_json = non_json
        self.create_error = create_error
        self.reconciliation_count = reconciliation_count
        self.match_id = match_id
        self.lookup_error = lookup_error
        self.absence_error = absence_error
        self.get_error = get_error
        self.create_mutator = create_mutator
        self.get_mutators = iter(get_mutators)
        self.log_outputs = iter(log_outputs)
        self.last_log_output = ""
        self.initial_state = initial_state
        self.preflight_identity = preflight_identity or NebiusCliInstallIdentity(
            executable="/opt/nebius/bin/nebius",
            resolved_executable="/opt/nebius/bin/nebius",
            version="0.12.263",
            byte_count=1,
            sha256="a" * 64,
        )
        self.preflight_failure = preflight_failure
        self.states = iter(states)
        self.created = False
        self.calls = []
        self.spec = None

    def preflight(self):
        if self.preflight_failure is not None:
            raise self.preflight_failure
        return self.preflight_identity

    def json(self, *arguments):
        self.calls.append(arguments)
        if arguments[:2] == ("iam", "whoami"):
            return {
                "user_profile": {
                    "tenants": [
                        {
                            "tenant_id": "tenant-exact",
                            "tenant_user_account_state": "ACTIVE",
                        }
                    ]
                }
            }
        if arguments[:3] == ("iam", "project", "get"):
            return {"metadata": {"id": "project-exact", "parent_id": "tenant-exact"}}
        if arguments[:3] == ("vpc", "subnet", "get"):
            return {
                "metadata": {"id": "vpcsubnet-exact", "parent_id": "project-exact"},
                "status": {"state": "READY"},
            }
        if arguments[:3] == ("ai", "job", "get-by-name"):
            if self.lookup_error:
                raise NebiusCliError(
                    self.lookup_error
                    if isinstance(self.lookup_error, str)
                    else "provider_command_failed"
                )
            if not (self.preexisting or self.created):
                # A NotFound answer is the only absence proof the CLI offers, measured in
                # Tickets 146 and 146a.  `absence_error` replaces it so a test can hand
                # the decode a near miss -- a NotFound reported through the CLI's
                # extra-call exit code, or an authentication failure -- and see which
                # ones it accepts as absence.
                raise self.absence_error or _not_found()
            if self.spec is None:
                return {
                    "metadata": {
                        "id": self.match_id,
                        "name": "job-exact",
                        "parent_id": "project-exact",
                        "created_at": "2026-08-14T10:00:00+00:00",
                    }
                }
            payload = self._job(self.initial_state)
            payload["metadata"]["id"] = self.match_id
            return self._mutate(payload, self.create_mutator)
        if arguments[:3] == ("ai", "job", "get"):
            if self.get_error:
                raise NebiusCliError(
                    self.get_error
                    if isinstance(self.get_error, str)
                    else "provider_command_failed"
                )
            payload = self._job(next(self.states))
            return self._mutate(payload, next(self.get_mutators, None))
        raise AssertionError(arguments)

    def __call__(self, *arguments):
        self.calls.append(arguments)
        if arguments[:3] == ("ai", "job", "create"):
            # Create is invoked through the plain runner, never json(): its response is
            # prose and is deliberately never parsed.
            return self._create(arguments)
        if arguments[:3] == ("ai", "job", "logs"):
            selected = next(self.log_outputs, self.last_log_output)
            if selected is None:
                raise NebiusCliError("provider_command_failed")
            self.last_log_output = selected
            return NebiusCliResponse(stdout=selected)
        raise AssertionError(arguments)

    def _create(self, arguments):
        self.created = True
        values = {
            arguments[index]: arguments[index + 1]
            for index in range(3, len(arguments) - 1)
            if arguments[index].startswith("--")
            and arguments[index] not in {"--env", "--env-secret"}
        }
        environment = []
        for index, argument in enumerate(arguments[:-1]):
            if argument == "--env":
                name, value = arguments[index + 1].split("=", 1)
                environment.append({"name": name, "value": value})
            elif argument == "--env-secret":
                name, selector = arguments[index + 1].split("=", 1)
                secret_id, version_id = selector.rsplit("@", 1)
                environment.append(
                    {
                        "name": name,
                        "mysterybox_secret": {
                            "secret_id": secret_id,
                            "version_id": version_id,
                        },
                    }
                )
        self.spec = {
            "image": values["--image"],
            "args": values["--args"],
            "environment_variables": environment,
            "platform": values["--platform"],
            "preset": values["--preset"],
            "disk": {
                "size_bytes": str(
                    {"100Gi": 100, "250Gi": 250}[values["--disk-size"]] * 1024**3
                )
            },
            "shm_size_bytes": str(
                16 * 1024**3 if values.get("--shm-size") == "16Gi" else 0
            ),
            "subnet_id": values["--subnet-id"],
            # Server-injected, because no --timeout was passed.
            "timeout": "86400s",
            "restart_policy": values["--restart-policy"],
            "injected_files": [
                {"container_path": values["--inject-file"].split(":", 1)[1]}
            ],
        }
        if self.ambiguous:
            raise AmbiguousNebiusCreate()
        if self.create_error is not None:
            raise NebiusCliError(
                "provider_command_failed", provider_reason=self.create_error
            )
        # 33 bytes of prose, ignoring --format, exactly as measured three times.
        return NebiusCliResponse(stdout=f"Job ID: {self.match_id}\n")

    def _job(self, state):
        terminal = state in {"COMPLETED", "FAILED"}
        return {
            "metadata": {
                "id": "aijob-exact",
                "name": "job-exact",
                "parent_id": "project-exact",
                "created_at": "2026-08-14T10:00:00+00:00",
            },
            "spec": self.spec,
            "status": {
                "state": state,
                "started_at": "2026-08-14T10:00:01+00:00",
                **({"finished_at": "2026-08-14T10:00:02+00:00"} if terminal else {}),
            },
        }

    @staticmethod
    def _mutate(payload, mutator):
        retained = copy.deepcopy(payload)
        if mutator is not None:
            mutator(retained)
        return retained


def observe_contact(contact, created, *, heartbeat_interval_seconds):
    from bio_neb.parabricks.execution_observations import observe_job

    observations = []
    terminal = observe_job(
        contact,
        created,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        progress=observations.append,
        pause=lambda _: None,
    )
    return (*observations, terminal)
