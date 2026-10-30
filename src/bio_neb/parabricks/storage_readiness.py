"""Direct public APP tracer for one fresh Nebius Object Storage bucket."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import (
    Field,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from bio_neb.content import canonical_json
from bio_neb.contracts import FrozenContract
from bio_neb.providers.nebius_context import (
    NebiusExecutionContextSelection,
)
from bio_neb.providers.nebius_controller_runtime import (
    accepted_nebius_cli_identity,
)
from bio_neb.storage_contracts import (
    ExecutionInputStorageAdapterError,
    ExecutionStorageReadinessConfig,
    FreshBucketNameAvailable,
    FreshBucketReceipt,
    InputStorageAuthenticationReceipt,
    InputStorageBootstrapMutationCounts,
    InputStorageInventory,
    InputStorageNebiusTarget,
    InputStorageObjectStorageTarget,
    InputStorageReadinessWriteReceipt,
)


class DirectStorageReadinessRefused(ValueError):
    def __init__(self, code: str, *fields: str) -> None:
        super().__init__(code)
        self.code = code
        self.fields = fields


class DirectStorageReadinessCommand(FrozenContract):
    selection_file: StrictStr
    bucket_name: StrictStr
    receipt_file: StrictStr
    region: StrictStr = "eu-north1"


class DirectStorageReadinessCeilings(FrozenContract):
    service_account_create_ceiling: Literal[0] = 0
    group_create_ceiling: Literal[0] = 0
    group_membership_create_ceiling: Literal[0] = 0
    access_permit_create_ceiling: Literal[0] = 0
    access_key_create_ceiling: Literal[0] = 0
    bucket_create_ceiling: Literal[1] = 1
    readiness_write_ceiling: Literal[1] = 1
    reference_put_ceiling: Literal[0] = 0
    fastq_put_ceiling: Literal[0] = 0
    prepared_candidate_contact_ceiling: Literal[0] = 0
    reference_preparation_create_ceiling: Literal[0] = 0
    h100_create_ceiling: Literal[0] = 0
    cleanup_ceiling: Literal[0] = 0


class DirectStorageReadinessPlan(FrozenContract):
    schema_version: Literal["bio-neb.direct-storage-readiness-plan/1"] = (
        "bio-neb.direct-storage-readiness-plan/1"
    )
    selection: NebiusExecutionContextSelection
    configuration: ExecutionStorageReadinessConfig
    ceilings: DirectStorageReadinessCeilings = DirectStorageReadinessCeilings()
    scientific_agent_authority_count: Literal[0] = 0
    model_authority_count: Literal[0] = 0
    later_stage_authority_count: Literal[0] = 0


class DirectStorageReadinessCounts(FrozenContract):
    adapter_operation_invocation_count: StrictInt = Field(ge=0, le=6)
    bucket_create_claim_count: StrictInt = Field(ge=0, le=1)
    bucket_create_invocation_count: StrictInt = Field(ge=0, le=1)
    readiness_write_claim_count: StrictInt = Field(ge=0, le=1)
    readiness_write_invocation_count: StrictInt = Field(ge=0, le=1)
    reference_put_count: Literal[0] = 0
    fastq_put_count: Literal[0] = 0
    prepared_candidate_contact_count: Literal[0] = 0
    reference_preparation_create_count: Literal[0] = 0
    h100_create_count: Literal[0] = 0
    retry_count: Literal[0] = 0
    resume_count: Literal[0] = 0
    recreate_count: Literal[0] = 0
    cleanup_count: Literal[0] = 0


class DirectStorageReadinessStageTiming(FrozenContract):
    stage: StrictStr
    observed_at_utc: datetime
    elapsed_ms: StrictInt = Field(ge=0)
    duration_since_prior_stage_ms: StrictInt = Field(ge=0)


class DirectStorageReadinessReceipt(FrozenContract):
    schema_version: Literal["bio-neb.direct-storage-readiness-receipt/1"] = (
        "bio-neb.direct-storage-readiness-receipt/1"
    )
    status: Literal["execution_storage_ready", "execution_storage_readiness_failed"]
    plan: DirectStorageReadinessPlan
    counts: DirectStorageReadinessCounts
    started_at_utc: datetime
    finished_at_utc: datetime
    duration_ms: StrictInt = Field(ge=0)
    stage_timings: tuple[DirectStorageReadinessStageTiming, ...] = Field(min_length=1)
    authentication: InputStorageAuthenticationReceipt | None = None
    bucket_name_available: FreshBucketNameAvailable | None = None
    bucket: FreshBucketReceipt | None = None
    initial_empty_inventory: InputStorageInventory | None = None
    readiness_write: InputStorageReadinessWriteReceipt | None = None
    final_empty_inventory: InputStorageInventory | None = None
    partial_inventory: InputStorageInventory | None = None
    failure_phase: StrictStr | None = None
    failure_reason: StrictStr | None = None

    @model_validator(mode="after")
    def exact_terminal_shape(self) -> "DirectStorageReadinessReceipt":
        if self.status == "execution_storage_ready":
            if (
                self.authentication is None
                or self.bucket_name_available is None
                or self.bucket is None
                or self.initial_empty_inventory is None
                or self.readiness_write is None
                or self.final_empty_inventory is None
                or self.failure_phase is not None
                or self.failure_reason is not None
                or self.counts.adapter_operation_invocation_count != 6
                or self.counts.bucket_create_claim_count != 1
                or self.counts.bucket_create_invocation_count != 1
                or self.counts.readiness_write_claim_count != 1
                or self.counts.readiness_write_invocation_count != 1
                or self.initial_empty_inventory.contact_complete != 1
                or self.initial_empty_inventory.objects
                or self.initial_empty_inventory.multipart_uploads
                or self.final_empty_inventory.contact_complete != 1
                or self.final_empty_inventory.objects
                or self.final_empty_inventory.multipart_uploads
                or self.partial_inventory is not None
            ):
                raise ValueError("successful readiness receipt is incomplete")
        elif self.failure_phase is None or self.failure_reason is None:
            raise ValueError("failed readiness receipt lacks exact failure")
        return self


class DirectStorageReadinessReview(FrozenContract):
    schema_version: Literal["bio-neb.direct-storage-readiness-review/1"] = (
        "bio-neb.direct-storage-readiness-review/1"
    )
    status: Literal["execution_storage_ready", "execution_storage_readiness_failed"]
    receipt: DirectStorageReadinessReceipt
    receipt_byte_count: StrictInt = Field(gt=0)
    receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    original_receipt_bytes_unchanged: Literal[True] = True
    reopen_adapter_contact_count: Literal[0] = 0
    reopen_nebius_cli_contact_count: Literal[0] = 0
    reopen_object_storage_contact_count: Literal[0] = 0
    reopen_write_count: Literal[0] = 0
    reopen_cleanup_count: Literal[0] = 0


class StorageReadinessAdapter(Protocol):
    def authenticate(
        self, config: ExecutionStorageReadinessConfig
    ) -> InputStorageAuthenticationReceipt: ...

    def precheck_fresh_bucket(
        self, config: ExecutionStorageReadinessConfig
    ) -> FreshBucketNameAvailable: ...

    def create_fresh_bucket(
        self,
        config: ExecutionStorageReadinessConfig,
        record_create_invoked: Callable[[], None],
    ) -> FreshBucketReceipt: ...

    def inspect_inventory(
        self, config: ExecutionStorageReadinessConfig, bucket: FreshBucketReceipt
    ) -> InputStorageInventory: ...

    def verify_readiness_write(
        self,
        config: ExecutionStorageReadinessConfig,
        bucket: FreshBucketReceipt,
        record_write_claimed: Callable[[], None],
        record_write_invoked: Callable[[], None],
    ) -> InputStorageReadinessWriteReceipt: ...


def _load_plan(command: DirectStorageReadinessCommand) -> DirectStorageReadinessPlan:
    selection_path = Path(command.selection_file)
    receipt_path = Path(command.receipt_file)
    if not selection_path.is_absolute():
        raise DirectStorageReadinessRefused(
            "direct_storage_readiness_selection_invalid", "selection_file"
        )
    if (
        not receipt_path.is_absolute()
        or receipt_path.exists()
        or not receipt_path.parent.is_dir()
    ):
        raise DirectStorageReadinessRefused(
            "direct_storage_readiness_receipt_invalid", "receipt_file"
        )
    try:
        selection = TypeAdapter(NebiusExecutionContextSelection).validate_json(
            selection_path.read_text(encoding="utf-8")
        )
        accepted = accepted_nebius_cli_identity(selection.nebius_executable)
        if (
            selection.resolved_nebius_executable != accepted.resolved_executable
            or selection.nebius_cli_version != accepted.version
            or selection.nebius_executable_byte_count != accepted.byte_count
            or selection.nebius_executable_sha256 != accepted.sha256
        ):
            raise ValueError("selection changed the accepted native CLI identity")
        domain = f"storage.{command.region}.nebius.cloud"
        selected = selection.topology
        config = ExecutionStorageReadinessConfig(
            nebius=InputStorageNebiusTarget(
                controller_runtime=selection.controller_runtime,
                executable=selection.nebius_executable,
                resolved_executable=selection.resolved_nebius_executable,
                cli_version=selection.nebius_cli_version,
                executable_byte_count=selection.nebius_executable_byte_count,
                executable_sha256=selection.nebius_executable_sha256,
                nebius_cli_profile=selected.profile,
                tenant_id=selected.tenant_id,
                project_id=selected.project_id,
            ),
            object_storage=InputStorageObjectStorageTarget(
                endpoint_url=f"https://{domain}", region=command.region
            ),
            bucket_name=command.bucket_name,
            bucket_domain=domain,
        )
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        raise DirectStorageReadinessRefused(
            "direct_storage_readiness_selection_invalid",
            "selection_file",
            "bucket_name",
        ) from error
    return DirectStorageReadinessPlan(selection=selection, configuration=config)


def _publish_receipt(receipt: DirectStorageReadinessReceipt, target: Path) -> None:
    descriptor: int | None = None
    staging: str | None = None
    directory_descriptor: int | None = None
    linked = False
    try:
        descriptor, staging = tempfile.mkstemp(
            prefix=".bio-neb-storage-readiness-", dir=target.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            descriptor = None
            handle.write(canonical_json(receipt.model_dump(mode="json")))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, target)
        linked = True
        os.unlink(staging)
        staging = None
        directory_descriptor = os.open(
            target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        os.fsync(directory_descriptor)
        os.close(directory_descriptor)
        directory_descriptor = None
    except (OSError, UnicodeError) as error:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if staging is not None:
            try:
                os.unlink(staging)
            except OSError:
                pass
        if directory_descriptor is not None:
            try:
                os.close(directory_descriptor)
            except OSError:
                pass
        if linked:
            try:
                os.unlink(target)
            except OSError:
                pass
        raise DirectStorageReadinessRefused(
            "direct_storage_readiness_receipt_write_failed", "receipt_file"
        ) from error


def _empty_inventory(inventory: InputStorageInventory) -> None:
    if inventory.contact_complete != 1:
        raise ExecutionInputStorageAdapterError(
            "authentication_lost", "bucket_readiness", inventory=inventory
        )
    if inventory.objects:
        raise ExecutionInputStorageAdapterError(
            "bucket_not_empty", "bucket_readiness", inventory=inventory
        )
    if inventory.multipart_uploads:
        raise ExecutionInputStorageAdapterError(
            "multipart_residue", "bucket_readiness", inventory=inventory
        )


def run_direct_storage_readiness(  # noqa: PLR0915 - Existing orchestration complexity; contract-base migration preserves this flow.
    command: DirectStorageReadinessCommand,
    *,
    adapter_factory: Callable[
        [ExecutionStorageReadinessConfig, NebiusExecutionContextSelection],
        StorageReadinessAdapter,
    ],
    emit: Callable[[dict[str, Any]], None],
    utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = time.monotonic,
) -> DirectStorageReadinessReceipt:
    """Run one fail-stop readiness attempt and never contact after its first error."""

    started_at = utc_now()
    started_monotonic = monotonic()
    prior_monotonic = started_monotonic
    timings: list[DirectStorageReadinessStageTiming] = []

    def stamp(stage: str) -> DirectStorageReadinessStageTiming:
        nonlocal prior_monotonic
        observed_monotonic = monotonic()
        timing = DirectStorageReadinessStageTiming(
            stage=stage,
            observed_at_utc=utc_now(),
            elapsed_ms=max(0, round((observed_monotonic - started_monotonic) * 1000)),
            duration_since_prior_stage_ms=max(
                0, round((observed_monotonic - prior_monotonic) * 1000)
            ),
        )
        prior_monotonic = observed_monotonic
        timings.append(timing)
        return timing

    def emit_stage(stage: str, payload: dict[str, Any]) -> None:
        timing = stamp(stage)
        emit({"stage": stage, **payload, "timing": timing.model_dump(mode="json")})

    plan = _load_plan(command)
    emit_stage(
        "precontact_readiness_summary",
        {
            "status": "ready_for_contact",
            "plan": plan.model_dump(mode="json"),
        },
    )
    phase = "authentication"
    authentication = None
    available = None
    bucket = None
    initial_inventory = None
    readiness_write = None
    final_inventory = None
    adapter_invocations = 0
    bucket_claims = 0
    bucket_invocations = 0
    readiness_claims = 0
    readiness_invocations = 0

    def counts() -> DirectStorageReadinessCounts:
        return DirectStorageReadinessCounts(
            adapter_operation_invocation_count=adapter_invocations,
            bucket_create_claim_count=bucket_claims,
            bucket_create_invocation_count=bucket_invocations,
            readiness_write_claim_count=readiness_claims,
            readiness_write_invocation_count=readiness_invocations,
        )

    def record_create_invoked() -> None:
        nonlocal bucket_invocations
        if bucket_claims != 1 or bucket_invocations != 0:
            raise ValueError("Adapter bucket create was unclaimed or repeated")
        bucket_invocations = 1
        emit_stage(
            "bucket_create_invoked",
            {
                "status": "invocation_acknowledged",
                "bucket_name": plan.configuration.bucket_name,
                "bucket_create_invocation_count": 1,
            },
        )

    def record_readiness_claimed() -> None:
        nonlocal readiness_claims
        if readiness_claims != 0 or readiness_invocations != 0:
            raise ValueError("Adapter readiness write claim repeated")
        readiness_claims = 1
        emit_stage(
            "readiness_write_claimed",
            {
                "status": "claimed",
                "bucket_name": plan.configuration.bucket_name,
                "readiness_write_claim_count": 1,
            },
        )

    def record_readiness_invoked() -> None:
        nonlocal readiness_invocations
        if readiness_claims != 1 or readiness_invocations != 0:
            raise ValueError("Adapter readiness write was unclaimed or repeated")
        readiness_invocations = 1
        emit_stage(
            "readiness_write_invoked",
            {
                "status": "invocation_acknowledged",
                "bucket_name": plan.configuration.bucket_name,
                "readiness_write_invocation_count": 1,
            },
        )

    try:
        adapter = adapter_factory(plan.configuration, plan.selection)
        adapter_invocations += 1
        authentication = InputStorageAuthenticationReceipt.model_validate(
            adapter.authenticate(plan.configuration)
        )
        if (
            authentication.nebius_cli_profile
            != plan.configuration.nebius.nebius_cli_profile
            or authentication.tenant_id != plan.configuration.nebius.tenant_id
            or authentication.project_id != plan.configuration.nebius.project_id
            or authentication.object_storage_credential.bootstrap_claim_counts
            != InputStorageBootstrapMutationCounts()
            or authentication.object_storage_credential.bootstrap_invocation_counts
            != InputStorageBootstrapMutationCounts()
        ):
            raise ValueError("Adapter changed selected authenticated topology")
        selected_credential = authentication.object_storage_credential
        selected_topology = plan.selection.topology
        if (
            selected_credential.service_account_id
            != selected_topology.service_account_id
            or selected_credential.access_key_resource_id
            != selected_topology.access_key_resource_id
            or selected_credential.access_key_id != selected_topology.access_key_id
        ):
            raise ValueError(
                "Adapter changed selected Object Storage credential identity"
            )
        emit_stage(
            "authenticated_context",
            {
                "status": "verified",
                "receipt": authentication.model_dump(mode="json"),
            },
        )

        phase = "bucket_freshness"
        adapter_invocations += 1
        available = FreshBucketNameAvailable.model_validate(
            adapter.precheck_fresh_bucket(plan.configuration)
        )
        if available.bucket_name != plan.configuration.bucket_name:
            raise ValueError("Adapter changed the exact absent bucket name")
        emit_stage(
            "bucket_name_absent",
            {
                "status": "verified",
                "receipt": available.model_dump(mode="json"),
            },
        )

        bucket_claims = 1
        emit_stage(
            "bucket_create_claimed",
            {
                "status": "claimed",
                "bucket_name": plan.configuration.bucket_name,
                "bucket_create_claim_count": 1,
            },
        )
        phase = "bucket_create"
        adapter_invocations += 1
        bucket = FreshBucketReceipt.model_validate(
            adapter.create_fresh_bucket(plan.configuration, record_create_invoked)
        )
        if bucket_invocations != 1:
            raise ValueError("Adapter omitted fresh bucket create invocation")
        if (
            bucket.name != plan.configuration.bucket_name
            or bucket.parent_id != plan.configuration.nebius.project_id
            or bucket.domain != plan.configuration.bucket_domain
            or bucket.region != plan.configuration.object_storage.region
        ):
            raise ValueError("Adapter changed selected bucket identity or placement")
        emit_stage(
            "bucket_active",
            {
                "status": "verified",
                "receipt": bucket.model_dump(mode="json"),
            },
        )

        phase = "bucket_readiness"
        adapter_invocations += 1
        initial_inventory = InputStorageInventory.model_validate(
            adapter.inspect_inventory(plan.configuration, bucket)
        )
        _empty_inventory(initial_inventory)
        emit_stage(
            "empty_inventory_before_readiness_write",
            {
                "status": "verified",
                "inventory": initial_inventory.model_dump(mode="json"),
            },
        )

        adapter_invocations += 1
        readiness_write = InputStorageReadinessWriteReceipt.model_validate(
            adapter.verify_readiness_write(
                plan.configuration,
                bucket,
                record_readiness_claimed,
                record_readiness_invoked,
            )
        )
        if readiness_claims != 1 or readiness_invocations != 1:
            raise ValueError("Adapter omitted readiness write mutation evidence")
        emit_stage(
            "readiness_lifecycle_readback",
            {
                "status": "verified",
                "receipt": readiness_write.model_dump(mode="json"),
            },
        )

        adapter_invocations += 1
        final_inventory = InputStorageInventory.model_validate(
            adapter.inspect_inventory(plan.configuration, bucket)
        )
        _empty_inventory(final_inventory)
        emit_stage(
            "empty_object_inventory",
            {
                "status": "verified",
                "objects": [],
                "contact_complete": 1,
            },
        )
        emit_stage(
            "zero_multipart_inventory",
            {
                "status": "verified",
                "multipart_uploads": [],
                "contact_complete": 1,
            },
        )
        terminal_timing = stamp("execution_storage_ready")
        receipt = DirectStorageReadinessReceipt(
            status="execution_storage_ready",
            plan=plan,
            counts=counts(),
            started_at_utc=started_at,
            finished_at_utc=terminal_timing.observed_at_utc,
            duration_ms=terminal_timing.elapsed_ms,
            stage_timings=tuple(timings),
            authentication=authentication,
            bucket_name_available=available,
            bucket=bucket,
            initial_empty_inventory=initial_inventory,
            readiness_write=readiness_write,
            final_empty_inventory=final_inventory,
        )
    except ExecutionInputStorageAdapterError as error:
        terminal_timing = stamp("execution_storage_readiness_failed")
        supplied_inventory = error.inventory
        partial_inventory = (
            supplied_inventory
            if supplied_inventory.contact_complete
            or supplied_inventory.objects
            or supplied_inventory.multipart_uploads
            else final_inventory or initial_inventory or supplied_inventory
        )
        receipt = DirectStorageReadinessReceipt(
            status="execution_storage_readiness_failed",
            plan=plan,
            counts=counts(),
            started_at_utc=started_at,
            finished_at_utc=terminal_timing.observed_at_utc,
            duration_ms=terminal_timing.elapsed_ms,
            stage_timings=tuple(timings),
            authentication=authentication,
            bucket_name_available=available,
            bucket=bucket,
            initial_empty_inventory=initial_inventory,
            readiness_write=readiness_write,
            final_empty_inventory=final_inventory,
            partial_inventory=partial_inventory,
            failure_phase=error.phase,
            failure_reason=error.reason,
        )
    except (ValidationError, ValueError):
        terminal_timing = stamp("execution_storage_readiness_failed")
        receipt = DirectStorageReadinessReceipt(
            status="execution_storage_readiness_failed",
            plan=plan,
            counts=counts(),
            started_at_utc=started_at,
            finished_at_utc=terminal_timing.observed_at_utc,
            duration_ms=terminal_timing.elapsed_ms,
            stage_timings=tuple(timings),
            authentication=authentication,
            bucket_name_available=available,
            bucket=bucket,
            initial_empty_inventory=initial_inventory,
            readiness_write=readiness_write,
            final_empty_inventory=final_inventory,
            partial_inventory=final_inventory or initial_inventory,
            failure_phase=phase,
            failure_reason="adapter_output_invalid",
        )
    _publish_receipt(receipt, Path(command.receipt_file))
    emit(receipt.model_dump(mode="json"))
    return receipt


def review_direct_storage_readiness(receipt_file: str) -> DirectStorageReadinessReview:
    """Reopen one immutable canonical receipt without constructing or calling an Adapter."""

    path = Path(receipt_file)
    if not path.is_absolute():
        raise DirectStorageReadinessRefused(
            "direct_storage_readiness_receipt_invalid", "receipt_file"
        )
    try:
        receipt_bytes = path.read_bytes()
        document = receipt_bytes.decode("utf-8")
        receipt = DirectStorageReadinessReceipt.model_validate_json(document)
    except (OSError, UnicodeError, ValidationError) as error:
        raise DirectStorageReadinessRefused(
            "direct_storage_readiness_receipt_invalid", "receipt_file"
        ) from error
    if document != canonical_json(receipt.model_dump(mode="json")):
        raise DirectStorageReadinessRefused(
            "direct_storage_readiness_receipt_noncanonical", "receipt_file"
        )
    return DirectStorageReadinessReview(
        status=receipt.status,
        receipt=receipt,
        receipt_byte_count=len(receipt_bytes),
        receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
    )


__all__ = [
    "DirectStorageReadinessCeilings",
    "DirectStorageReadinessCommand",
    "DirectStorageReadinessCounts",
    "DirectStorageReadinessPlan",
    "DirectStorageReadinessReceipt",
    "DirectStorageReadinessReview",
    "DirectStorageReadinessRefused",
    "StorageReadinessAdapter",
    "review_direct_storage_readiness",
    "run_direct_storage_readiness",
]
