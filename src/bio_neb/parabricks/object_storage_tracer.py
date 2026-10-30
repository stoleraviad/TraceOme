"""Bounded public APP tracer for shared Nebius Object Storage mechanics."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

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
from bio_neb.parabricks.storage_readiness import (
    StorageReadinessAdapter,
)
from bio_neb.providers.nebius_context import (
    NebiusExecutionContextSelection,
)
from bio_neb.providers.nebius_controller_runtime import (
    accepted_nebius_cli_identity,
)
from bio_neb.providers.nebius_object_storage import (
    ListedObject,
    MultipartResidue,
    NebiusObjectStorageError,
    NebiusObjectStorageMechanics,
    ObjectIdentity,
    ReadContact,
)
from bio_neb.storage_contracts import (
    ExecutionInputStorageAdapterError,
    ExecutionStorageReadinessConfig,
    FreshBucketReceipt,
    InputStorageAuthenticationReceipt,
    InputStorageInventory,
    InputStorageNebiusTarget,
    InputStorageObjectStorageTarget,
    InputStorageReadinessWriteReceipt,
)

_DATA_KEY = "object-storage-tracer/deterministic-bytes.txt"
_MANIFEST_KEY = "object-storage-tracer/manifest.json"
_DATA_BYTES = b"bio-neb-object-storage-tracer/v1\n"
_UNRELATED_METADATA = {"tracer-kind": "bounded-live-acceptance"}
_OFFLINE_RETRY_EVIDENCE = (
    "tests.test_nebius_object_storage.ReadPolicyTests.test_transient_head_retries_once_and_retains_both_contacts",
    "tests.test_nebius_object_storage.ReadPolicyTests.test_transient_get_and_list_are_allowlisted",
    "tests.test_nebius_object_storage.ReadPolicyTests.test_transient_exhaustion_has_exactly_two_contacts",
    "tests.test_nebius_object_storage.ReadPolicyTests.test_nontransient_read_failure_is_not_retried",
)


class ObjectStorageTracerRefused(ValueError):
    def __init__(self, code: str, *fields: str) -> None:
        super().__init__(code)
        self.code = code
        self.fields = fields


class ObjectStorageTracerCommand(FrozenContract):
    selection_file: StrictStr
    bucket_name: StrictStr
    receipt_file: StrictStr
    region: StrictStr = "eu-north1"


class ObjectStorageTracerCounts(FrozenContract):
    bucket_create_claim_count: StrictInt = Field(ge=0, le=1)
    bucket_create_invocation_count: StrictInt = Field(ge=0, le=1)
    lifecycle_write_claim_count: StrictInt = Field(ge=0, le=1)
    lifecycle_write_invocation_count: StrictInt = Field(ge=0, le=1)
    data_put_claim_count: StrictInt = Field(ge=0, le=1)
    data_put_invocation_count: StrictInt = Field(ge=0, le=1)
    manifest_put_claim_count: StrictInt = Field(ge=0, le=1)
    manifest_put_invocation_count: StrictInt = Field(ge=0, le=1)
    serverless_job_create_count: Literal[0] = 0
    biological_operation_count: Literal[0] = 0
    cleanup_count: Literal[0] = 0
    abort_count: Literal[0] = 0
    delete_count: Literal[0] = 0


class ObjectStorageTracerContact(FrozenContract):
    operation: StrictStr
    physical_contact_ordinal: StrictInt = Field(ge=1)
    classification: StrictStr
    provider_status: StrictInt | None = None
    provider_code: StrictStr | None = None


class ObjectStorageTracerIdentity(FrozenContract):
    object_key: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class ObjectStorageTracerPartialObject(FrozenContract):
    """Inventory-only residue without fabricated content authority."""

    object_key: StrictStr
    byte_count: StrictInt = Field(ge=0)


class ObjectStorageTracerClientPolicy(FrozenContract):
    connect_timeout_seconds: Literal[30] = 30
    read_timeout_seconds: Literal[300] = 300
    sdk_total_max_attempts: Literal[1] = 1
    default_profile_chain_used: Literal[False] = False
    default_region_chain_used: Literal[False] = False
    default_credential_chain_used: Literal[False] = False


class ObjectStorageTracerVerificationPolicy(FrozenContract):
    policy: Literal["transition_based"] = "transition_based"
    data_upload_stream_full_byte_pass_count: Literal[1] = 1
    routine_host_data_full_read_count: Literal[0] = 0
    audit_host_data_full_read_count: Literal[1] = 1
    manifest_exact_readback_count: Literal[1] = 1


class ObjectStorageTracerClosure(FrozenContract):
    objects: tuple[ObjectStorageTracerIdentity, ...]
    multipart: tuple[tuple[StrictStr, StrictStr], ...]


class ObjectStorageTracerReceipt(FrozenContract):
    schema_version: Literal["bio-neb.object-storage-tracer-receipt/2"] = (
        "bio-neb.object-storage-tracer-receipt/2"
    )
    status: Literal["object_storage_tracer_succeeded", "object_storage_tracer_failed"]
    selection: NebiusExecutionContextSelection
    configuration_id: StrictStr
    bucket_name: StrictStr
    client_policy: ObjectStorageTracerClientPolicy = ObjectStorageTracerClientPolicy()
    verification_policy: ObjectStorageTracerVerificationPolicy = (
        ObjectStorageTracerVerificationPolicy()
    )
    counts: ObjectStorageTracerCounts
    authentication: InputStorageAuthenticationReceipt | None = None
    bucket: FreshBucketReceipt | None = None
    lifecycle: InputStorageReadinessWriteReceipt | None = None
    data_object: ObjectStorageTracerIdentity | None = None
    data_custom_metadata: dict[StrictStr, StrictStr] | None = None
    data_audit_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    manifest_object: ObjectStorageTracerIdentity | None = None
    pre_manifest_closure: ObjectStorageTracerClosure | None = None
    final_closure: ObjectStorageTracerClosure | None = None
    read_contacts: tuple[ObjectStorageTracerContact, ...] = ()
    live_transient_retry_exercised: bool = False
    live_transient_retry_disposition: StrictStr
    offline_retry_evidence: tuple[StrictStr, ...] = _OFFLINE_RETRY_EVIDENCE
    failure_phase: StrictStr | None = None
    failure_reason: StrictStr | None = None
    partial_objects: tuple[ObjectStorageTracerPartialObject, ...] = ()
    multipart_residue: tuple[tuple[StrictStr, StrictStr], ...] = ()

    @model_validator(mode="after")
    def exact_terminal_shape(self) -> "ObjectStorageTracerReceipt":
        if self.status == "object_storage_tracer_succeeded":
            if (
                self.authentication is None
                or self.bucket is None
                or self.lifecycle is None
                or self.data_object is None
                or self.data_custom_metadata != _UNRELATED_METADATA
                or self.data_audit_sha256 != self.data_object.sha256
                or self.manifest_object is None
                or self.pre_manifest_closure is None
                or self.final_closure is None
                or self.failure_phase is not None
                or self.failure_reason is not None
                or self.multipart_residue
                or self.counts
                != ObjectStorageTracerCounts(
                    bucket_create_claim_count=1,
                    bucket_create_invocation_count=1,
                    lifecycle_write_claim_count=1,
                    lifecycle_write_invocation_count=1,
                    data_put_claim_count=1,
                    data_put_invocation_count=1,
                    manifest_put_claim_count=1,
                    manifest_put_invocation_count=1,
                )
            ):
                raise ValueError(
                    "successful object-storage tracer receipt is incomplete"
                )
        elif self.failure_phase is None or self.failure_reason is None:
            raise ValueError("failed object-storage tracer receipt lacks exact failure")
        return self


class ObjectStorageTracerReview(FrozenContract):
    schema_version: Literal["bio-neb.object-storage-tracer-review/2"] = (
        "bio-neb.object-storage-tracer-review/2"
    )
    status: Literal["object_storage_tracer_succeeded", "object_storage_tracer_failed"]
    receipt: ObjectStorageTracerReceipt
    receipt_byte_count: StrictInt = Field(gt=0)
    receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    original_receipt_bytes_unchanged: Literal[True] = True
    reopen_adapter_contact_count: Literal[0] = 0
    reopen_nebius_cli_contact_count: Literal[0] = 0
    reopen_object_storage_contact_count: Literal[0] = 0


class ObjectStorageTracerAdapterSet(Protocol):
    readiness: StorageReadinessAdapter
    objects: NebiusObjectStorageMechanics


def _load_command(
    command: ObjectStorageTracerCommand,
) -> tuple[NebiusExecutionContextSelection, ExecutionStorageReadinessConfig, Path]:
    selection_path = Path(command.selection_file)
    receipt_path = Path(command.receipt_file)
    if not selection_path.is_absolute():
        raise ObjectStorageTracerRefused(
            "object_storage_tracer_selection_invalid", "selection_file"
        )
    if (
        not receipt_path.is_absolute()
        or receipt_path.exists()
        or not receipt_path.parent.is_dir()
    ):
        raise ObjectStorageTracerRefused(
            "object_storage_tracer_receipt_invalid", "receipt_file"
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
        raise ObjectStorageTracerRefused(
            "object_storage_tracer_selection_invalid", "selection_file", "bucket_name"
        ) from error
    return selection, config, receipt_path


def _contact(value: ReadContact) -> ObjectStorageTracerContact:
    return ObjectStorageTracerContact(
        operation=value.operation,
        physical_contact_ordinal=value.ordinal,
        classification=value.classification,
        provider_status=value.detail.status,
        provider_code=value.detail.code,
    )


def _inventory_residue(
    inventory: InputStorageInventory | None,
) -> tuple[tuple[ObjectStorageTracerPartialObject, ...], tuple[tuple[str, str], ...]]:
    if inventory is None:
        return (), ()
    return (
        tuple(
            ObjectStorageTracerPartialObject(
                object_key=item.object_key, byte_count=item.byte_count
            )
            for item in inventory.objects
        ),
        tuple(
            (item.object_key, item.upload_id) for item in inventory.multipart_uploads
        ),
    )


def _publish(receipt: ObjectStorageTracerReceipt, target: Path) -> None:
    descriptor: int | None = None
    staging: str | None = None
    directory_descriptor: int | None = None
    linked = False
    try:
        descriptor, staging = tempfile.mkstemp(
            prefix=".bio-neb-object-storage-tracer-", dir=target.parent
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
        if linked:
            try:
                os.unlink(target)
            except OSError:
                pass
        raise ObjectStorageTracerRefused(
            "object_storage_tracer_receipt_write_failed", "receipt_file"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        if staging is not None:
            try:
                os.unlink(staging)
            except FileNotFoundError:
                pass


def run_object_storage_tracer(
    command: ObjectStorageTracerCommand,
    *,
    adapter_factory: Callable[
        [
            ExecutionStorageReadinessConfig,
            NebiusExecutionContextSelection,
            Callable[[ReadContact], None],
        ],
        ObjectStorageTracerAdapterSet,
    ],
    emit: Callable[[dict[str, object]], None] = lambda _: None,
) -> ObjectStorageTracerReceipt:
    """Run one fail-stop, cleanup-free storage mechanics tracer."""

    selection, config, receipt_path = _load_command(command)
    contacts: list[ReadContact] = []
    adapters = adapter_factory(config, selection, contacts.append)
    readiness, objects = adapters.readiness, adapters.objects
    bucket = None
    authentication = None
    lifecycle = None
    data_identity = None
    manifest_identity = None
    pre_closure = None
    final_closure = None
    inventory: InputStorageInventory | None = None
    phase = "authentication"
    failure_reason = None
    residue: tuple[MultipartResidue, ...] = ()
    values = dict(
        bucket_create_claim_count=0,
        bucket_create_invocation_count=0,
        lifecycle_write_claim_count=0,
        lifecycle_write_invocation_count=0,
        data_put_claim_count=0,
        data_put_invocation_count=0,
        manifest_put_claim_count=0,
        manifest_put_invocation_count=0,
    )

    def mark(name: str) -> None:
        if values[name] != 0:
            raise ValueError(f"repeated consequence marker: {name}")
        values[name] = 1

    try:
        authentication = readiness.authenticate(config)
        phase = "bucket_freshness"
        available = readiness.precheck_fresh_bucket(config)
        if available.bucket_name != config.bucket_name:
            raise ValueError("Adapter changed fresh bucket identity")
        mark("bucket_create_claim_count")
        phase = "bucket_create"
        bucket = readiness.create_fresh_bucket(
            config, lambda: mark("bucket_create_invocation_count")
        )
        phase = "bucket_readiness"
        inventory = readiness.inspect_inventory(config, bucket)
        if (
            not inventory.contact_complete
            or inventory.objects
            or inventory.multipart_uploads
        ):
            raise ValueError("fresh tracer bucket is not empty")
        lifecycle = readiness.verify_readiness_write(
            config,
            bucket,
            lambda: mark("lifecycle_write_claim_count"),
            lambda: mark("lifecycle_write_invocation_count"),
        )
        inventory = readiness.inspect_inventory(config, bucket)
        if (
            not inventory.contact_complete
            or inventory.objects
            or inventory.multipart_uploads
        ):
            raise ValueError("tracer bucket changed before object publication")
        phase = "data_publication"
        mark("data_put_claim_count")
        mark("data_put_invocation_count")
        written = objects.put_conditional_stream(
            _DATA_KEY,
            io.BytesIO(_DATA_BYTES),
            expected=ObjectIdentity(
                len(_DATA_BYTES), hashlib.sha256(_DATA_BYTES).hexdigest()
            ),
            content_type="text/plain",
            custom_metadata=_UNRELATED_METADATA,
        )
        headed = objects.head_metadata(
            _DATA_KEY, expected_sha256=written.identity.sha256
        )
        if (
            headed.value.identity != written.identity
            or headed.value.custom_metadata
            != tuple(sorted(_UNRELATED_METADATA.items()))
        ):
            raise ValueError("published tracer data identity changed")
        audited = objects.get_bytes(_DATA_KEY, maximum_bytes=len(_DATA_BYTES))
        if audited.value != _DATA_BYTES:
            raise ValueError("tracer audit read changed bytes")
        data_identity = ObjectStorageTracerIdentity(
            object_key=_DATA_KEY,
            byte_count=written.identity.byte_count,
            sha256=written.identity.sha256,
        )
        phase = "manifest_publication"
        expected_pre = (ListedObject(_DATA_KEY, len(_DATA_BYTES)),)
        pre_closure = ObjectStorageTracerClosure(objects=(data_identity,), multipart=())
        manifest_bytes = canonical_json(
            {
                "schema_version": "bio-neb.object-storage-tracer-manifest/1",
                "data_object": data_identity.model_dump(mode="json"),
                "unrelated_custom_metadata": _UNRELATED_METADATA,
            }
        ).encode("utf-8")
        mark("manifest_put_claim_count")
        mark("manifest_put_invocation_count")
        published = objects.publish_manifest_last(
            prefix="object-storage-tracer/",
            manifest_key=_MANIFEST_KEY,
            body=manifest_bytes,
            expected_pre_inventory=expected_pre,
        )
        manifest_identity = ObjectStorageTracerIdentity(
            object_key=_MANIFEST_KEY,
            byte_count=published.identity.byte_count,
            sha256=published.identity.sha256,
        )
        expected_final = tuple(
            sorted(
                (
                    ListedObject(_DATA_KEY, data_identity.byte_count),
                    ListedObject(_MANIFEST_KEY, manifest_identity.byte_count),
                ),
                key=lambda item: item.object_key,
            )
        )
        final_closure = ObjectStorageTracerClosure(
            objects=tuple(
                sorted(
                    (data_identity, manifest_identity),
                    key=lambda item: item.object_key,
                )
            ),
            multipart=(),
        )
        if tuple(item.object_key for item in final_closure.objects) != tuple(
            item.object_key for item in expected_final
        ):
            raise ValueError("final tracer inventory changed")
        status = "object_storage_tracer_succeeded"
    except ExecutionInputStorageAdapterError as error:
        status = "object_storage_tracer_failed"
        phase = error.phase
        failure_reason = error.reason
        inventory = error.inventory
    except NebiusObjectStorageError as error:
        status = "object_storage_tracer_failed"
        phase = error.operation
        failure_reason = error.kind
        residue = error.residue
    except (TypeError, ValueError) as error:
        status = "object_storage_tracer_failed"
        failure_reason = type(error).__name__

    partial_objects, inventory_residue = _inventory_residue(inventory)
    retained_residue = tuple((item.object_key, item.upload_id) for item in residue)
    transient_contacts = [
        item for item in contacts if item.classification == "transient"
    ]
    retry_exercised = bool(transient_contacts)
    disposition = (
        "eligible transient read was observed and every physical contact is retained"
        if retry_exercised
        else "no natural eligible transient occurred; live failure induction would fabricate provider evidence; offline retry matrix is controlling"
    )
    receipt = ObjectStorageTracerReceipt(
        status=status,
        selection=selection,
        configuration_id=config.configuration_id,
        bucket_name=config.bucket_name,
        counts=ObjectStorageTracerCounts(**values),
        authentication=authentication,
        bucket=bucket,
        lifecycle=lifecycle,
        data_object=data_identity,
        data_custom_metadata=_UNRELATED_METADATA if data_identity is not None else None,
        data_audit_sha256=data_identity.sha256 if data_identity is not None else None,
        manifest_object=manifest_identity,
        pre_manifest_closure=pre_closure,
        final_closure=final_closure,
        read_contacts=tuple(_contact(item) for item in contacts),
        live_transient_retry_exercised=retry_exercised,
        live_transient_retry_disposition=disposition,
        failure_phase=phase if status == "object_storage_tracer_failed" else None,
        failure_reason=failure_reason,
        partial_objects=partial_objects,
        multipart_residue=retained_residue or inventory_residue,
    )
    _publish(receipt, receipt_path)
    emit(receipt.model_dump(mode="json"))
    return receipt


def review_object_storage_tracer(receipt_file: str) -> ObjectStorageTracerReview:
    """Reopen canonical tracer evidence without constructing an Adapter."""

    path = Path(receipt_file)
    try:
        body = path.read_bytes()
        document = body.decode("utf-8")
        receipt = ObjectStorageTracerReceipt.model_validate_json(document)
    except (OSError, UnicodeError, ValidationError) as error:
        raise ObjectStorageTracerRefused(
            "object_storage_tracer_receipt_invalid", "receipt_file"
        ) from error
    if document != canonical_json(receipt.model_dump(mode="json")):
        raise ObjectStorageTracerRefused(
            "object_storage_tracer_receipt_noncanonical", "receipt_file"
        )
    return ObjectStorageTracerReview(
        status=receipt.status,
        receipt=receipt,
        receipt_byte_count=len(body),
        receipt_sha256=hashlib.sha256(body).hexdigest(),
    )


__all__ = [
    "ObjectStorageTracerAdapterSet",
    "ObjectStorageTracerCommand",
    "ObjectStorageTracerReceipt",
    "ObjectStorageTracerRefused",
    "ObjectStorageTracerReview",
    "review_object_storage_tracer",
    "run_object_storage_tracer",
]
