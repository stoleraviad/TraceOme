"""Supported secret-free execution choices retained before human review."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import (
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from bio_neb.application.verification_policy import (
    DEFAULT_OBJECT_VERIFICATION_MODE,
    ObjectVerificationOverrides,
)
from bio_neb.contracts import FrozenContract
from bio_neb.parabricks.capability_policy import ExecutionRuntimeSelection
from bio_neb.providers.nebius_context import NebiusExecutionContextSelection
from bio_neb.providers.runtime import NebiusCliTarget, RunObjectStorageTarget


class LocalReferenceSource(FrozenContract):
    """A selected local replica of the approved compressed reference.

    It carries no byte identity: upload checks the approved publisher md5 and size
    while streaming, as for the public URL.
    """

    path: StrictStr

    @model_validator(mode="after")
    def absolute_path(self):
        if not Path(self.path).is_absolute():
            raise ValueError("local reference source requires an absolute path")
        return self


WORKER_MOUNT_PATH = "/mnt/object-storage"


class ParabricksLaunch(FrozenContract):
    """Explicit secret-free launch choices; recipe, resources and layout are derived."""

    object_scope: StrictStr = Field(
        pattern=r"^[a-z0-9][a-z0-9-]*(?:/[a-z0-9][a-z0-9-]*)*$"
    )
    nebius: NebiusCliTarget
    object_storage: RunObjectStorageTarget
    context: NebiusExecutionContextSelection
    local_reference_source: LocalReferenceSource | None = None
    work_directory: StrictStr
    cleanup_owner: StrictStr
    heartbeat_interval_seconds: StrictFloat | StrictInt = Field(
        gt=0, allow_inf_nan=False
    )
    bounded_get_bytes: StrictInt = Field(ge=1024, le=64 * 1024 * 1024)
    # The effective verification selection, including defaults, is retained in
    # pre-review configuration. Per-slot full reads carry an explicit reason.
    verification_mode: Literal["independent_full_read", "reuse_creation_proof"] = (
        DEFAULT_OBJECT_VERIFICATION_MODE
    )
    verification_overrides: ObjectVerificationOverrides = ObjectVerificationOverrides()

    @model_validator(mode="after")
    def worker_mount(self):
        if self.object_storage.mount_path != WORKER_MOUNT_PATH:
            raise ValueError("Parabricks requires its worker object-storage mount")
        return self


class ParabricksConfiguration(FrozenContract):
    launch: ParabricksLaunch
    observation_interval_seconds: StrictFloat | StrictInt = Field(
        default=5, gt=0, allow_inf_nan=False
    )
    runtime_selection: ExecutionRuntimeSelection

    @model_validator(mode="after")
    def exact_provider_scope(self):
        launch = self.launch
        context, target = launch.context, launch.nebius
        if (
            context.controller_runtime,
            context.nebius_executable,
            context.resolved_nebius_executable,
            context.nebius_cli_version,
            context.nebius_executable_byte_count,
            context.nebius_executable_sha256,
            context.topology.profile,
            context.topology.tenant_id,
            context.topology.project_id,
        ) != (
            target.controller_runtime,
            target.executable,
            target.resolved_executable,
            target.cli_version,
            target.executable_byte_count,
            target.executable_sha256,
            target.profile,
            target.tenant_id,
            target.project_id,
        ):
            raise ValueError("execution context differs from supplied provider scope")
        if not Path(launch.work_directory).is_absolute():
            raise ValueError("execution working location must be absolute")
        return self


class ParabricksConfigurationRegistered(FrozenContract):
    event_type: Literal["ParabricksConfigurationRegistered"] = (
        "ParabricksConfigurationRegistered"
    )
    configuration: ParabricksConfiguration


class ParabricksAccountObjectStorage(FrozenContract):
    endpoint_url: StrictStr
    region: StrictStr


class ParabricksAccountLaunch(FrozenContract):
    nebius: NebiusCliTarget
    context: NebiusExecutionContextSelection
    object_storage: ParabricksAccountObjectStorage
    cleanup_owner: StrictStr


class ParabricksAccount(FrozenContract):
    """One account's Parabricks values, in the section's own layout (owner, #454).

    It names no prepared reference: a Run's index is the Execution Review's bound
    prebuilt index or one CPU preparation job (owner, 2026-10-29, #464), so a file
    that still names #107's account prepared candidate is refused as an unknown
    field.
    Identifiers, not secrets; they stay in the operator home, never in source. The
    build supplies every other leaf but the Run's own.
    """

    schema_version: Literal["bio-neb.parabricks-account/1"]
    launch: ParabricksAccountLaunch
    runtime_selection: ExecutionRuntimeSelection


class ParabricksAccountRefused(ValueError):
    """The account file gives no section; the Run keeps its preparation Review."""

    def __init__(self, code: str, fields: tuple[str, ...]):
        super().__init__(code)
        self.code = code
        self.fields = fields


# The same for every account and Run.
_BUILD_LAUNCH_DEFAULTS = {
    "heartbeat_interval_seconds": 5,
    "bounded_get_bytes": 8 * 1024 * 1024,
    "verification_mode": DEFAULT_OBJECT_VERIFICATION_MODE,
}
_BUCKET_LIMIT = 63


def _run_token(run_root: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", Path(run_root).name.lower()).strip("-")
    return token or "run-" + hashlib.sha256(run_root.encode()).hexdigest()[:12]


def run_launch_selections(run_root: str) -> dict[str, str]:
    """The Run's own launch leaves, from its root's name: chosen afresh per Run."""
    token = _run_token(run_root)
    bucket = f"bio-neb-{token}"
    if len(bucket) > _BUCKET_LIMIT:
        # A shortened name keeps the whole name's identity in its suffix.
        digest = hashlib.sha256(token.encode()).hexdigest()[:12]
        bucket = f"{bucket[: _BUCKET_LIMIT - 13].rstrip('-')}-{digest}"
    return {
        "object_scope": token,
        "bucket": bucket,
        "work_directory": str(Path(run_root) / "runs" / "execution" / "work"),
    }


def _fields(error: ValidationError) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                ".".join(str(part) for part in item["loc"]) or "$"
                for item in error.errors(include_url=False, include_input=False)
            }
        )
    )


def _require_this_build(selection: ExecutionRuntimeSelection) -> None:
    """The account's runtime runs this build's worker (#464 live a2 F4): every image
    states this build's request schema and worker sha256, and every runtime profile
    this worker; an older image refuses the request only after a paid job starts."""
    from bio_neb.parabricks.execution_runtime import ATTESTED_WORKER_SHA256
    from bio_neb.parabricks.worker_exchange import REQUEST_SCHEMA_VERSION

    capability = selection.model_dump(mode="json")["capability"]
    stale = [
        f"capability.images.{index}"
        for index, image in enumerate(capability["images"])
        if image["worker_request_schema"] != REQUEST_SCHEMA_VERSION
        or image["worker_sha256"] != ATTESTED_WORKER_SHA256
    ] + [
        f"capability.runtime_profiles.{index}.{role}"
        for index, profile in enumerate(capability["runtime_profiles"])
        for role in ("cpu", "h100")
        if profile[role]["worker_sha256"] != ATTESTED_WORKER_SHA256
    ]
    if stale:
        raise ParabricksAccountRefused("runtime_not_this_build", tuple(stale))


def default_configuration(account_file: Path, run_root: str) -> ParabricksConfiguration:
    """The section a Run registers when it is given none (owner, #454).

    The build's defaults, the account file's values and the Run's own leaves.
    """
    try:
        document = json.loads(Path(account_file).read_bytes().decode("utf-8"))
    except FileNotFoundError as error:
        raise ParabricksAccountRefused("parabricks_account_missing", ("$",)) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ParabricksAccountRefused(
            "parabricks_account_unreadable", ("$",)
        ) from error
    try:
        account = ParabricksAccount.model_validate(document)
    except ValidationError as error:
        raise ParabricksAccountRefused(
            "parabricks_account_invalid", _fields(error)
        ) from error
    _require_this_build(account.runtime_selection)
    launch = account.launch.model_dump(mode="json")
    selected = run_launch_selections(run_root)
    try:
        return ParabricksConfiguration.model_validate(
            {
                "launch": {
                    **launch,
                    **_BUILD_LAUNCH_DEFAULTS,
                    "object_scope": selected["object_scope"],
                    "work_directory": selected["work_directory"],
                    "object_storage": {
                        **launch["object_storage"],
                        "bucket": selected["bucket"],
                        "mount_path": WORKER_MOUNT_PATH,
                    },
                },
                "runtime_selection": account.runtime_selection.model_dump(mode="json"),
            }
        )
    except ValidationError as error:
        raise ParabricksAccountRefused(
            "parabricks_account_invalid", _fields(error)
        ) from error


def current_configuration(events):
    """Select immutable supplied settings; the checked writer owns their ancestry."""
    for event in reversed(list(events)):
        if event["event_type"] == "ParabricksConfigurationRegistered":
            return event, ParabricksConfigurationRegistered.model_validate(
                event["payload"]
            ).configuration
        if (
            event["event_type"] == "PreparationInformationAnswered"
            and event["payload"]["response"]["action"] == "supply"
        ):
            return event, ParabricksConfiguration.model_validate(
                event["payload"]["response"]["configuration"]
            )
    return None, None
