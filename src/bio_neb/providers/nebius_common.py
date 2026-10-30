"""Contact-only CLI and credential primitives shared by Nebius adapters."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from pydantic import Field, StrictInt, StrictStr

from bio_neb.contracts import StrictFrozenContract
from bio_neb.providers.runtime import NebiusCliTarget

ProviderCreateFailureReason = Literal[
    "authentication",
    "invalid_argument",
    "capacity",
    "conflict",
    "unavailable",
    "unknown",
]
NebiusCliFailureClassification = Literal[
    "interactive_authentication_required",
    "authentication_failed",
    "authorization_denied",
    "invalid_argument",
    "not_found",
    "unavailable",
    "unknown",
]


# Every failing command returns this exact shape:
#
#   Error: rpc error: code = NotFound desc = not found request = <uuid>
#   ...
#     Trace ID: ba6fd61a3f05c84c639836764059c44e
#
# Only the two identifiers are lifted out.  `desc =` is the provider's free text and the
# request id is deliberately left with it: the exit code, the rpc code and the Trace ID
# are exactly what the Journey may carry.
_RPC_CODE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_TRACE_ID_VALUE = re.compile(r"[0-9a-f]{32}")
_RPC_CODE_MARKER = re.compile(r"rpc error: code = ([A-Za-z][A-Za-z0-9_]{0,63})")
_TRACE_ID_MARKER = re.compile(r"Trace ID:[ \t]*([0-9a-fA-F]{32})")
# The operator record's delimiters.  They exist so a later reader can find one
# command's complete raw stderr in an operator stream without guessing where it
# started or ended, and so the bytes carry their own identity.
PROVIDER_STDERR_OPEN = "[bio-neb.provider-stderr"
PROVIDER_STDERR_CLOSE = "[/bio-neb.provider-stderr]"


def _canonical_line(value: str, field_name: str) -> str:
    if not value or value.strip() != value or any(item in value for item in "\r\n\0"):
        raise ValueError(f"{field_name} must be one non-empty canonical line")
    return value


def _absolute_posix_path(value: str, field_name: str) -> str:
    _canonical_line(value, field_name)
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or value == "/"
        or str(path) != value
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field_name} must be an absolute canonical POSIX path")
    return value


@dataclass(frozen=True, slots=True)
class NebiusCliFailureEvidence:
    """Bounded nonzero-child evidence; raw provider streams never cross the boundary.

    Ticket 139 added `rpc_code` and `trace_id`.  Ticket 138 measured that the CLI never
    writes JSON on failure: under `--format json` every failing command left stdout
    empty and wrote plain text to stderr, carrying the provider's own rpc code and the
    vendor's support handle.  Both are extracted here as bounded identifiers and nothing
    else crosses -- the raw text stays on the operator record, because Ticket 138 also
    observed the CLI's authentication prompt arriving on that same stream with a live
    OAuth URL carrying `code_challenge` and `state`.
    """

    operation: str
    child_exit_code: int
    stdout_byte_count: int
    stdout_sha256: str
    stderr_byte_count: int
    stderr_sha256: str
    stderr_classification: NebiusCliFailureClassification
    rpc_code: str | None = None
    trace_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.operation
            or self.operation.strip() != self.operation
            or any(item in self.operation for item in "\r\n\x00 :")
        ):
            raise ValueError("operation must be one secret-safe identifier")
        if self.child_exit_code == 0:
            raise ValueError("child_exit_code must be nonzero")
        if self.stdout_byte_count < 0 or self.stderr_byte_count < 0:
            raise ValueError("stream byte counts cannot be negative")
        if self.stderr_classification not in {
            "interactive_authentication_required",
            "authentication_failed",
            "authorization_denied",
            "invalid_argument",
            "not_found",
            "unavailable",
            "unknown",
        }:
            raise ValueError("stderr_classification is unsupported")
        for value in (self.stdout_sha256, self.stderr_sha256):
            if (
                len(value) != 64
                or value.casefold() != value
                or any(item not in "0123456789abcdef" for item in value)
            ):
                raise ValueError("stream digest must be canonical SHA-256")
        if self.rpc_code is not None and not _RPC_CODE_NAME.fullmatch(self.rpc_code):
            raise ValueError("rpc_code must be one bounded provider code name")
        if self.trace_id is not None and not _TRACE_ID_VALUE.fullmatch(self.trace_id):
            raise ValueError("trace_id must be one canonical lowercase hex handle")

    def public_fields(self) -> list[str]:
        """Render the exact value-safe fields retained by the public command."""

        return [
            f"operation:{self.operation}",
            f"child_exit_code:{self.child_exit_code}",
            f"stdout_byte_count:{self.stdout_byte_count}",
            f"stdout_sha256:{self.stdout_sha256}",
            f"stderr_byte_count:{self.stderr_byte_count}",
            f"stderr_sha256:{self.stderr_sha256}",
            f"stderr_classification:{self.stderr_classification}",
            *(() if self.rpc_code is None else (f"rpc_code:{self.rpc_code}",)),
            *(() if self.trace_id is None else (f"trace_id:{self.trace_id}",)),
        ]

    def journey_extract(self) -> str:
        """Render the provider's own diagnosis as identifiers, never as its text.

        The exit code is retained raw.  Ticket 138 measured the CLI's documented rule
        that an error raised by an *extra* call reports the target code plus 40, so 13
        and 53 are the same NOT_FOUND answer; keeping the literal number here lets a
        reader apply that rule instead of hiding which call actually failed.
        """

        return ";".join(
            (
                f"exit={self.child_exit_code}",
                *(() if self.rpc_code is None else (f"rpc={self.rpc_code}",)),
                *(() if self.trace_id is None else (f"trace={self.trace_id}",)),
            )
        )


class NebiusCliError(RuntimeError):
    """A secret-free CLI boundary failure."""

    def __init__(
        self,
        code: str,
        *,
        provider_reason: ProviderCreateFailureReason | None = None,
        failure_evidence: NebiusCliFailureEvidence | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.provider_reason = provider_reason
        self.failure_evidence = failure_evidence

    def journey_reason(self) -> str:
        """Name this failure with the provider's own diagnosis appended, never its text.

        Ticket 139: `Class@module.py:line` names *our* failing site and this names
        *theirs*.  Ticket 137 needed both and had neither.  The exception deliberately
        never carries the raw stderr, so no later handler can put it in the Journey by
        accident -- the operator record already has those bytes, bound to the
        `stderr_sha256` this evidence retains.
        """

        if self.failure_evidence is None:
            return self.code
        return f"{self.code};{self.failure_evidence.journey_extract()}"


def _failure_classification(stderr: str) -> NebiusCliFailureClassification:
    normalized = stderr.casefold()
    categories: tuple[tuple[NebiusCliFailureClassification, tuple[str, ...]], ...] = (
        (
            "interactive_authentication_required",
            (
                "switch to your browser",
                "complete the authentication process",
                "open browser",
            ),
        ),
        (
            "authentication_failed",
            (
                "unauthenticated",
                "authentication failed",
                "invalid token",
                "token expired",
                "credential",
            ),
        ),
        (
            "authorization_denied",
            (
                "permission denied",
                "permissiondenied",
                "access denied",
                "forbidden",
                "not authorized",
            ),
        ),
        (
            "invalid_argument",
            (
                "invalid argument",
                "invalid_argument",
                "validation failed",
                "bad request",
                "unknown flag",
            ),
        ),
        ("not_found", ("not found", "not_found", "notfound")),
        (
            "unavailable",
            (
                "temporarily unavailable",
                "service unavailable",
                "connection refused",
                "connection reset",
                "deadline exceeded",
                "timeout",
                "unavailable",
            ),
        ),
    )
    return next(
        (
            classification
            for classification, markers in categories
            if any(marker in normalized for marker in markers)
        ),
        "unknown",
    )


def _provider_rpc_code(stderr: str) -> str | None:
    match = _RPC_CODE_MARKER.search(stderr)
    return None if match is None else match.group(1)


def _provider_trace_id(stderr: str) -> str | None:
    match = _TRACE_ID_MARKER.search(stderr)
    return None if match is None else match.group(1).casefold()


def record_provider_stderr(evidence: NebiusCliFailureEvidence, stderr: str) -> None:
    """Retain one failing command's raw stderr on the operator record only.

    Ticket 139's owner decision: capture everything, but the Journey and the operator
    record get different bytes.  This process's stderr *is* the operator record -- the
    native capsule opens a distinct live `stderr.log` for the child and hashes it at
    finish -- so writing here retains the vendor's complete text where an operator and
    Ticket 88's pool can read it, while the Journey keeps only the identifiers in
    `journey_extract`.  A scrubber was considered and rejected: a pattern list silently
    rots, and structured extraction keeps the Journey clean by construction instead.
    """

    print(
        f"{PROVIDER_STDERR_OPEN} operation:{evidence.operation}"
        f" child_exit_code:{evidence.child_exit_code}"
        f" stderr_byte_count:{evidence.stderr_byte_count}"
        f" stderr_sha256:{evidence.stderr_sha256}]",
        file=sys.stderr,
    )
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
    print(PROVIDER_STDERR_CLOSE, file=sys.stderr, flush=True)


def _operation_identifier(arguments: tuple[str, ...]) -> str:
    boundary = next(
        (index for index, item in enumerate(arguments) if item.startswith("-")),
        len(arguments),
    )
    operation = arguments[:boundary]
    return ".".join(operation) or "unknown"


def _stream_identity(value: str) -> tuple[int, str]:
    payload = value.encode("utf-8")
    return len(payload), hashlib.sha256(payload).hexdigest()


def _provider_create_failure_reason(stderr: str) -> ProviderCreateFailureReason:
    """Reduce provider stderr to one bounded diagnostic without retaining its text."""

    normalized = stderr.casefold()
    categories: tuple[tuple[ProviderCreateFailureReason, tuple[str, ...]], ...] = (
        (
            "authentication",
            (
                "unauthenticated",
                "authentication",
                "unauthorized",
                "permission denied",
                "permissiondenied",
                "access denied",
                "invalid token",
                "credential",
            ),
        ),
        (
            "invalid_argument",
            (
                "invalid argument",
                "invalid_argument",
                "invalidargument",
                "validation failed",
                "bad request",
                "unknown flag",
            ),
        ),
        (
            "capacity",
            (
                "insufficient capacity",
                "resource exhausted",
                "resource_exhausted",
                "resourceexhausted",
                "quota exceeded",
                "capacity",
            ),
        ),
        (
            "conflict",
            ("already exists", "alreadyexists", "conflict", "duplicate"),
        ),
        (
            "unavailable",
            (
                "temporarily unavailable",
                "service unavailable",
                "connection refused",
                "connection reset",
                "deadline exceeded",
                "timeout",
                "unavailable",
            ),
        ),
    )
    return next(
        (
            reason
            for reason, markers in categories
            if any(marker in normalized for marker in markers)
        ),
        "unknown",
    )


class AmbiguousNebiusCreate(NebiusCliError):
    """The provider may have accepted one timed-out create request."""

    def __init__(self) -> None:
        super().__init__("provider_create_ambiguous")


# docs.nebius.com/cli/exit-codes: 7 is "any authentication error".  The same page states
# that an error raised by an *extra* call the CLI makes internally reports the target
# code plus 40, so 47 is the same answer; the NotFound set {13, 53} applies the same
# rule.  47 has never been observed and is inference from the documented rule -- but it
# is the safe direction of that inference.  An extra halt only asks a human to log in,
# while a missed one restores an unbounded auth stall (one was measured at 654.249 s).
# The NotFound set is the mirror image: there the unsafe direction is proving absence,
# so both halves of the proof are required.
AUTHENTICATION_EXIT_CODES = frozenset({7, 47})


class NebiusAuthenticationRequired(NebiusCliError):
    """A human must complete the CLI's own browser flow before this Run continues.

    Ticket 144.  This is deliberately its own name.  Ticket 138 measured what the
    alternatives cost: an expired credential surfaced as a 654-second silent subprocess,
    and under the then-compiled 180-second bound it surfaced as `AmbiguousNebiusCreate`
    -- a lost create response -- for a human who had merely stepped away from the
    browser.  Neither name tells a reader that the one action required is a person
    logging in.
    """

    def __init__(self, evidence: NebiusCliFailureEvidence) -> None:
        super().__init__("provider_authentication_required", failure_evidence=evidence)


class NebiusTenantNotActive(NebiusCliError):
    """Authentication succeeded, and the selected tenant is still not usable.

    This is not the same condition as `NebiusAuthenticationRequired` and no browser
    login resolves it, so it keeps its own name rather than sharing one.
    """

    def __init__(self) -> None:
        super().__init__("provider_tenant_not_active")


def _nested(value: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def authenticated_profile_kind(
    payload: Any, *, tenant_id: str, project_id: str
) -> Literal["user_profile", "service_account_profile"] | None:
    """Name the `iam whoami` profile that proves the selected tenant and project.

    Issue 35.  Both `iam whoami` gates (the storage adapter's `authenticate` and the
    Parabricks entry probe) used to accept only a federation `user_profile`, which
    tied every live step to the 12-hour browser login.  A key-based service-account
    profile answers with a different shape -- no `user_profile`, no tenants list, just
    the account's own metadata and status -- so it proves the selection differently:
    it must be active and parented by the selected project.  The project's own parent
    tenant is what `iam project get` checks right after the storage gate.

    Returns the accepted kind, or None when the payload proves neither; the caller
    owns the refusal's name.
    """

    user = _nested(payload, "user_profile")
    if isinstance(user, dict):
        tenants = user.get("tenants")
        if isinstance(tenants, list) and any(
            isinstance(item, dict)
            and item.get("tenant_id") == tenant_id
            and item.get("tenant_user_account_state") == "ACTIVE"
            for item in tenants
        ):
            return "user_profile"
        return None
    if (
        _nested(payload, "service_account_profile", "info", "status", "active") is True
        and _nested(payload, "service_account_profile", "info", "metadata", "parent_id")
        == project_id
    ):
        return "service_account_profile"
    return None


# The CLI's global defaults, as `nebius --help` lists them: --auth-timeout defaults
# to 15m0s.  Passing it explicitly makes the CLI's own outer bound a stated decision
# rather than an unexamined default, which is the point of having no subprocess bound
# of our own.
NEBIUS_AUTH_TIMEOUT = "15m0s"


@dataclass(frozen=True, slots=True)
class NebiusAccessScope:
    """Selected authentication scope, independent of job-only runtime fields."""

    profile: str
    tenant_id: str
    project_id: str


def nebius_common_arguments(
    target: NebiusCliTarget | NebiusAccessScope,
) -> tuple[str, ...]:
    """Render the one shared argv tail every Nebius contact uses.

    Ticket 144 moved this here from the job-contact adapter so the authentication
    precondition, storage operations and job contacts cannot drift into different
    flag dialects. Callers supply their already selected scope.

    --retries 1 keeps --per-retry-timeout (default 20s) moot, so it stays unset.  The
    CLI's global --timeout (default 1m0s) is passed nowhere, and the create site is where
    that would bite.  Ticket 138's synchronous create blocked for 189.903 s and still
    succeeded -- three times that default -- so either the global bound does not cover
    long-running operation polling or a subcommand flag overrode it.  Both readings are
    consistent with everything measured, and nothing has distinguished them.  Under
    --async the question is moot: no CLI call in Tickets 143, 146 or 146a ran longer than
    15.625 s.  Measure the global --timeout before restoring any blocking call; do not
    resolve it by assumption.
    """

    return (
        "--profile",
        target.profile,
        "--retries",
        "1",
        "--auth-timeout",
        NEBIUS_AUTH_TIMEOUT,
        "--no-check-update",
        "--no-progress",
        "--format",
        "json",
    )


@dataclass(frozen=True, slots=True)
class NebiusCliResponse:
    """Successful stdout only; provider stderr never crosses the boundary."""

    stdout: str


class RegistryResponseEvidence(StrictFrozenContract):
    """Bounded Registry HTTP facts; never headers, URLs, token bodies or credentials."""

    request_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    redirect_from_sha256: StrictStr | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    status: StrictInt | None = Field(default=None, ge=100, le=599)
    body_digest: StrictStr | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    declared_digest: StrictStr | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    expected_digest: StrictStr | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )


class ExecutionContactObserver(Protocol):
    """Receive safe lifecycle facts without becoming part of Adapter mechanics."""

    def start(
        self,
        *,
        category: Literal[
            "local_runtime", "provider", "storage", "credential", "source"
        ],
        operation: str,
        executable_identity: str,
        credential_names: tuple[str, ...] = (),
        selected_identity: str | None = None,
    ) -> str: ...

    def finish(
        self,
        unit_id: str | None,
        *,
        disposition: Literal["succeeded", "failed", "unknown"],
        exit_status: int | None = None,
        parse_classification: str | None = None,
        registry_response: RegistryResponseEvidence | None = None,
    ) -> None: ...


def observe_execution_contact_start(
    observer: ExecutionContactObserver | None,
    *,
    category: Literal["local_runtime", "provider", "storage", "credential", "source"],
    operation: str,
    executable_identity: str,
    credential_names: tuple[str, ...] = (),
    selected_identity: str | None = None,
) -> str | None:
    """Fail soft: diagnostic persistence never replaces an Adapter operation."""

    if observer is None:
        return None
    try:
        return observer.start(
            category=category,
            operation=operation,
            executable_identity=executable_identity,
            credential_names=credential_names,
            selected_identity=selected_identity,
        )
    except Exception:
        return None


def observe_execution_contact_finish(
    observer: ExecutionContactObserver | None,
    unit_id: str | None,
    *,
    disposition: Literal["succeeded", "failed", "unknown"],
    exit_status: int | None = None,
    parse_classification: str | None = None,
    registry_response: RegistryResponseEvidence | Mapping[str, Any] | None = None,
) -> None:
    """Fail soft: a terminal diagnostic cannot change the Adapter result."""

    if observer is None:
        return
    try:
        observer.finish(
            unit_id,
            disposition=disposition,
            exit_status=exit_status,
            parse_classification=parse_classification,
            **(
                {
                    "registry_response": RegistryResponseEvidence.model_validate(
                        registry_response
                    )
                }
                if registry_response is not None
                else {}
            ),
        )
    except Exception:
        return


_CREATE_COMMANDS = {
    ("ai", "job", "create"),
    ("storage", "bucket", "create"),
}


def _run_nebius_cli(
    *,
    command: tuple[str, ...],
    arguments: tuple[str, ...],
    process_runner: Callable[..., subprocess.CompletedProcess[str]],
    input_text: str | None = None,
    contact_observer: ExecutionContactObserver | None = None,
    executable_identity: str = "nebius-cli:unverified",
    success_classification: str = "text",
) -> NebiusCliResponse:
    # No local bound.  A compiled bound below the CLI's own --auth-timeout default of
    # 15 minutes would pre-empt the provider tool's contract with a smaller invented number
    # on every call: a synchronous create was measured at 189.903 s, so a 180 s bound would
    # cut off a wait that was about to succeed.  A CLI that is still waiting is not a failed
    # state.  The CLI's own --auth-timeout is stated explicitly in argv instead, and under
    # --async no measured call has run longer than 15.625 s.
    process_options: dict[str, Any] = {
        "check": False,
        "shell": False,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "strict",
    }
    if input_text is not None:
        process_options["input"] = input_text
    category: Literal["local_runtime", "provider"] = (
        "local_runtime" if arguments == ("version",) else "provider"
    )
    operation = _execution_contact_operation(arguments)
    unit_id = observe_execution_contact_start(
        contact_observer,
        category=category,
        operation=operation,
        executable_identity=executable_identity,
    )
    try:
        completed = process_runner(command, **process_options)
    except subprocess.TimeoutExpired:
        observe_execution_contact_finish(
            contact_observer,
            unit_id,
            disposition="failed",
            parse_classification="timeout",
        )
        if arguments[:3] in _CREATE_COMMANDS:
            raise AmbiguousNebiusCreate() from None
        raise NebiusCliError("provider_command_timeout") from None
    except UnicodeError:
        observe_execution_contact_finish(
            contact_observer,
            unit_id,
            disposition="failed",
            parse_classification="invalid_utf8",
        )
        raise NebiusCliError("provider_output_invalid_utf8") from None
    except OSError as error:
        observe_execution_contact_finish(
            contact_observer,
            unit_id,
            disposition="failed",
            parse_classification="unavailable",
        )
        raise NebiusCliError("provider_command_unavailable") from error
    except BaseException:
        observe_execution_contact_finish(
            contact_observer,
            unit_id,
            disposition="unknown",
            parse_classification="runtime_interrupted",
        )
        raise
    if completed.returncode != 0:
        stdout_byte_count, stdout_sha256 = _stream_identity(completed.stdout)
        stderr_byte_count, stderr_sha256 = _stream_identity(completed.stderr)
        evidence = NebiusCliFailureEvidence(
            operation=_operation_identifier(arguments),
            child_exit_code=completed.returncode,
            stdout_byte_count=stdout_byte_count,
            stdout_sha256=stdout_sha256,
            stderr_byte_count=stderr_byte_count,
            stderr_sha256=stderr_sha256,
            stderr_classification=_failure_classification(completed.stderr),
            rpc_code=_provider_rpc_code(completed.stderr),
            trace_id=_provider_trace_id(completed.stderr),
        )
        # The raw text is retained here and nowhere else.  It never rides on the
        # exception, so no later handler can carry it into a Journey event by accident.
        record_provider_stderr(evidence, completed.stderr)
        observe_execution_contact_finish(
            contact_observer,
            unit_id,
            disposition="failed",
            exit_status=completed.returncode,
            parse_classification=evidence.stderr_classification,
        )
        raise NebiusCliError(
            "provider_command_failed",
            provider_reason=(
                _provider_create_failure_reason(completed.stderr)
                if arguments[:3] in _CREATE_COMMANDS
                else None
            ),
            failure_evidence=evidence,
        )
    observe_execution_contact_finish(
        contact_observer,
        unit_id,
        disposition="succeeded",
        exit_status=0,
        parse_classification=success_classification,
    )
    return NebiusCliResponse(stdout=completed.stdout)


_CONTACT_OPERATION_PREFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("iam", "project", "get"), "iam.project.get"),
    (("iam", "project", "list"), "iam.project.list"),
    (("iam", "tenant", "list"), "iam.tenant.list"),
    (("iam", "v2", "access-key", "list"), "iam.v2.access-key.list"),
    (
        ("iam", "group-membership", "list-member-of"),
        "iam.group-membership.list-member-of",
    ),
    (("iam", "group-membership", "list-members"), "iam.group-membership.list-members"),
    (("iam", "access-permit", "list"), "iam.access-permit.list"),
    (("mysterybox", "payload", "get"), "mysterybox.payload.get"),
    (
        ("mysterybox", "secret-version", "payload", "get"),
        "mysterybox.secret-version.payload.get",
    ),
    (("mysterybox", "secret-version", "get"), "mysterybox.secret-version.get"),
    (("mysterybox", "secret", "get"), "mysterybox.secret.get"),
    (("iam", "v2", "access-key", "get"), "iam.v2.access-key.get"),
    (("iam", "group-membership", "get"), "iam.group-membership.get"),
    (("iam", "service-account", "get"), "iam.service-account.get"),
    (("iam", "access-permit", "get"), "iam.access-permit.get"),
    (("iam", "group", "get"), "iam.group.get"),
    (("iam", "whoami"), "iam.whoami"),
    (("storage", "bucket", "create"), "storage.bucket.create"),
    (("storage", "bucket", "get"), "storage.bucket.get"),
    (("storage", "bucket", "list"), "storage.bucket.list"),
    (("registry", "v2", "repository", "get"), "registry.v2.repository.get"),
    (("ai", "job", "create"), "ai.job.create"),
    (("ai", "job", "get"), "ai.job.get"),
)


def _execution_contact_operation(arguments: tuple[str, ...]) -> str:
    if arguments == ("version",):
        return "nebius.version"
    for prefix, operation in _CONTACT_OPERATION_PREFIXES:
        if arguments[: len(prefix)] == prefix:
            return operation
    return "nebius.allowlisted-other"


@dataclass(frozen=True, slots=True)
class NebiusCliInstallIdentity:
    """Secret-free exact identity for one accepted CLI payload."""

    executable: str
    resolved_executable: str
    version: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _absolute_posix_path(self.executable, "executable")
        _absolute_posix_path(self.resolved_executable, "resolved_executable")
        _canonical_line(self.version, "version")
        if self.byte_count <= 0:
            raise ValueError("byte_count must be positive")
        if (
            len(self.sha256) != 64
            or self.sha256.casefold() != self.sha256
            or any(item not in "0123456789abcdef" for item in self.sha256)
        ):
            raise ValueError("sha256 must be one canonical lowercase SHA-256 digest")

    def to_receipt(self) -> dict[str, str | int]:
        """Render the stable, secret-free identity retained by public preflight."""

        return {
            "schema_version": "bio-neb.nebius-cli-install-identity/1",
            "executable": self.executable,
            "resolved_executable": self.resolved_executable,
            "version": self.version,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _ExecutablePayloadIdentity:
    resolved_executable: str
    byte_count: int
    sha256: str


def _inspect_executable_payload(executable: str) -> _ExecutablePayloadIdentity:
    try:
        requested = Path(executable)
        resolved = requested.resolve(strict=True)
        resolved_executable = _absolute_posix_path(
            resolved.as_posix(), "resolved_executable"
        )
        before = resolved.stat()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise OSError("not an executable file")
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = resolved.stat()
    except (OSError, ValueError):
        raise NebiusCliError("provider_cli_identity_unavailable") from None
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise NebiusCliError("provider_cli_identity_changed")
    return _ExecutablePayloadIdentity(
        resolved_executable=resolved_executable,
        byte_count=after.st_size,
        sha256=digest.hexdigest(),
    )


class NebiusCliRunner:
    """Invoke one exactly pinned Nebius executable directly, without a shell."""

    def __init__(
        self,
        *,
        expected_identity: NebiusCliInstallIdentity,
        process_runner: Callable[
            ..., subprocess.CompletedProcess[str]
        ] = subprocess.run,
        executable_inspector: Callable[
            [str], _ExecutablePayloadIdentity
        ] = _inspect_executable_payload,
        execution_contact_observer: ExecutionContactObserver | None = None,
    ) -> None:
        # There is deliberately no timeout_seconds.  See _run_nebius_cli above: the bound
        # is removed rather than tuned.  Guardrails against a genuinely infinite wait are
        # deferred to a hardening pass; the one unbounded-hang risk actually measured, a
        # stalled browser auth, is removed by proving auth at adapter entry.  A cached
        # `iam whoami` returns in about 0.45 s with no browser, which is that happy path.
        self._expected_identity = expected_identity
        self._process_runner = process_runner
        self._executable_inspector = executable_inspector
        self._execution_contact_observer = execution_contact_observer
        self._verified_identity: NebiusCliInstallIdentity | None = None

    def argv(self, *arguments: str) -> tuple[str, ...]:
        if not arguments:
            raise ValueError("Nebius CLI invocation requires exact arguments")
        for index, argument in enumerate(arguments):
            _canonical_line(argument, f"arguments[{index}]")
        return (self._expected_identity.executable, *arguments)

    def preflight(self) -> NebiusCliInstallIdentity:
        if self._verified_identity is not None:
            return self._verified_identity
        expected = self._expected_identity
        observed = self._executable_inspector(expected.executable)
        if (
            observed.resolved_executable != expected.resolved_executable
            or observed.byte_count != expected.byte_count
            or observed.sha256 != expected.sha256
        ):
            raise NebiusCliError("provider_cli_identity_mismatch")
        response = _run_nebius_cli(
            command=(expected.executable, "version"),
            arguments=("version",),
            process_runner=self._process_runner,
            contact_observer=self._execution_contact_observer,
            executable_identity=f"nebius-cli:{expected.sha256}",
            success_classification="version_text",
        )
        if response.stdout not in {expected.version, expected.version + "\n"}:
            raise NebiusCliError("provider_cli_version_mismatch")
        self._verified_identity = expected
        return self._verified_identity

    def __call__(self, *arguments: str) -> NebiusCliResponse:
        result = self._invoke(arguments, input_text=None, parse_json=False)
        if not isinstance(result, NebiusCliResponse):
            raise AssertionError("plain Nebius invocation returned parsed JSON")
        return result

    def json(self, *arguments: str) -> dict[str, Any]:
        result = self._invoke(arguments, input_text=None, parse_json=True)
        if not isinstance(result, dict):
            raise AssertionError("JSON Nebius invocation returned plain text")
        return result

    def json_from_stdin(self, input_text: str, *arguments: str) -> dict[str, Any]:
        """Run one pinned command with exact in-memory stdin and parse JSON stdout."""

        if not isinstance(input_text, str) or not input_text:
            raise ValueError("input_text must be non-empty text")
        result = self._invoke(arguments, input_text=input_text, parse_json=True)
        if not isinstance(result, dict):
            raise AssertionError("JSON Nebius invocation returned plain text")
        return result

    def _invoke(
        self,
        arguments: tuple[str, ...],
        *,
        input_text: str | None,
        parse_json: bool,
    ) -> NebiusCliResponse | dict[str, Any]:
        command = self.argv(*arguments)
        self.preflight()
        expected = self._expected_identity
        response = _run_nebius_cli(
            command=command,
            arguments=arguments,
            process_runner=self._process_runner,
            input_text=input_text,
            contact_observer=self._execution_contact_observer,
            executable_identity=f"nebius-cli:{expected.sha256}",
            # The observed unit is the direct subprocess.  A zero exit succeeds that
            # unit before JSON parsing begins; this classification deliberately makes
            # the still-pending application parse explicit.
            success_classification="json_parse_pending" if parse_json else "text",
        )
        return self._json_response(response) if parse_json else response

    @staticmethod
    def _json_response(response: NebiusCliResponse) -> dict[str, Any]:
        try:
            payload = json.loads(response.stdout)
        except json.JSONDecodeError as error:
            raise NebiusCliError("provider_json_invalid") from error
        if not isinstance(payload, dict):
            raise NebiusCliError("provider_json_invalid")
        return payload


class AwsCredentialMaterial:
    """Contact-only credential values with an always-redacted representation."""

    __slots__ = (
        "access_key_id",
        "credentials_profile",
        "secret_access_key",
        "session_token",
        "worker_secret_selector",
    )

    def __init__(
        self,
        *,
        access_key_id: str,
        secret_access_key: str,
        session_token: str | None = None,
        worker_secret_selector: str | None = None,
        credentials_profile: str = "default",
    ) -> None:
        self.access_key_id = _canonical_line(access_key_id, "access_key_id")
        self.secret_access_key = _canonical_line(secret_access_key, "secret_access_key")
        self.session_token = (
            None
            if session_token is None
            else _canonical_line(session_token, "session_token")
        )
        self.worker_secret_selector = (
            None
            if worker_secret_selector is None
            else _canonical_line(worker_secret_selector, "worker_secret_selector")
        )
        self.credentials_profile = _canonical_line(
            credentials_profile, "credentials_profile"
        )

    def __repr__(self) -> str:
        return "AwsCredentialMaterial([REDACTED])"


S3FailureKind = Literal[
    "credentials_unavailable",
    "source_unavailable",
    "source_mismatch",
    "transfer_failed",
    "object_absent",
    "object_mismatch",
    "download_failed",
]


class S3ContactError(RuntimeError):
    """A secret-free S3 boundary failure."""

    def __init__(self, kind: S3FailureKind) -> None:
        super().__init__(kind)
        self.kind = kind


class EnvironmentAwsCredentialResolver:
    """Resolve exact credential values from environment state only at contact."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = os.environ if environ is None else environ

    def __call__(self) -> AwsCredentialMaterial:
        try:
            access_key_id = self._environ["AWS_ACCESS_KEY_ID"]
            secret_access_key = self._environ["AWS_SECRET_ACCESS_KEY"]
            credentials_profile = self._environ["BIO_NEB_S3_CREDENTIALS_PROFILE"]
        except KeyError as error:
            raise S3ContactError("credentials_unavailable") from error
        try:
            return AwsCredentialMaterial(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                session_token=self._environ.get("AWS_SESSION_TOKEN"),
                worker_secret_selector=self._environ.get(
                    "BIO_NEB_WORKER_SECRET_SELECTOR"
                ),
                credentials_profile=credentials_profile,
            )
        except ValueError as error:
            raise S3ContactError("credentials_unavailable") from error


__all__ = [
    "AUTHENTICATION_EXIT_CODES",
    "NEBIUS_AUTH_TIMEOUT",
    "PROVIDER_STDERR_CLOSE",
    "PROVIDER_STDERR_OPEN",
    "AmbiguousNebiusCreate",
    "AwsCredentialMaterial",
    "ExecutionContactObserver",
    "EnvironmentAwsCredentialResolver",
    "NebiusAuthenticationRequired",
    "NebiusCliRunner",
    "NebiusCliError",
    "NebiusCliFailureEvidence",
    "NebiusCliInstallIdentity",
    "NebiusCliResponse",
    "NebiusTenantNotActive",
    "ProviderCreateFailureReason",
    "S3ContactError",
    "S3FailureKind",
    "authenticated_profile_kind",
    "nebius_common_arguments",
    "record_provider_stderr",
    "observe_execution_contact_finish",
    "observe_execution_contact_start",
]
