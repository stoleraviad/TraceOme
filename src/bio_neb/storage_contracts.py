"""Secret-free credential and mutation evidence shared by storage consumers."""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from bio_neb.content import canonical_json_digest
from bio_neb.contracts import FrozenContract


def _line(value: str, field: str) -> str:
    if not value or value.strip() != value or any(char in value for char in "\r\n\0"):
        raise ValueError(f"{field} must be one canonical line")
    return value


class InputStorageBootstrapMutationCounts(FrozenContract):
    """Zero mutation authority for the accepted existing-credential path."""

    service_account: Literal[0] = 0
    group: Literal[0] = 0
    group_membership: Literal[0] = 0
    access_permit: Literal[0] = 0
    access_key: Literal[0] = 0


class ObjectStorageCredentialReuseReceipt(FrozenContract):
    """Secret-free proof of one exact reusable Object Storage identity."""

    service_account_id: StrictStr
    access_key_resource_id: StrictStr
    access_key_id: StrictStr
    exact_topology_match_count: Literal[1]
    access_key_state: Literal["ACTIVE"]
    secret_state: Literal["ACTIVE"]
    secret_delivery_mode: Literal["mystery_box"]
    host_secret_resolved: Literal[True]
    worker_secret_version_bound: Literal[True]
    selector_identifiers_retained: Literal[False] = False
    bootstrap_claim_counts: InputStorageBootstrapMutationCounts = (
        InputStorageBootstrapMutationCounts()
    )
    bootstrap_invocation_counts: InputStorageBootstrapMutationCounts = (
        InputStorageBootstrapMutationCounts()
    )

    @field_validator("service_account_id", "access_key_resource_id", "access_key_id")
    @classmethod
    def secret_free_identifiers(cls, value: str, info: Any) -> str:
        return _line(value, info.field_name)


class InputStorageNebiusTarget(FrozenContract):
    """Exact Ubuntu CLI identity and selected control-plane scope."""

    controller_runtime: Literal["native_ubuntu"]
    executable: StrictStr
    resolved_executable: StrictStr
    cli_version: Literal["0.12.263"]
    executable_byte_count: StrictInt = Field(gt=0)
    executable_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    nebius_cli_profile: StrictStr
    tenant_id: StrictStr
    project_id: StrictStr

    @field_validator("nebius_cli_profile", "tenant_id", "project_id")
    @classmethod
    def canonical_lines(cls, value: str, info: Any) -> str:
        return _line(value, info.field_name)

    @field_validator("executable", "resolved_executable")
    @classmethod
    def absolute_executable(cls, value: str, info: Any) -> str:
        _line(value, info.field_name)
        if not value.startswith("/") or "//" in value or "/../" in value:
            raise ValueError(
                f"{info.field_name} must be an absolute canonical POSIX path"
            )
        return value


class InputStorageObjectStorageTarget(FrozenContract):
    endpoint_url: StrictStr
    region: StrictStr

    @field_validator("region")
    @classmethod
    def canonical_lines(cls, value: str, info: Any) -> str:
        return _line(value, info.field_name)

    @field_validator("endpoint_url")
    @classmethod
    def exact_https_origin(cls, value: str) -> str:
        selected = urlsplit(value)
        try:
            _ = selected.port
        except ValueError as error:
            raise ValueError("endpoint_url must be one exact HTTPS origin") from error
        if (
            selected.scheme != "https"
            or not selected.hostname
            or selected.username is not None
            or selected.password is not None
            or selected.path
            or selected.query
            or selected.fragment
            or selected.geturl() != value
        ):
            raise ValueError("endpoint_url must be one exact HTTPS origin")
        return value


class ExecutionStorageReadinessConfig(FrozenContract):
    """Secret-free operational selection with no scientific placement authority."""

    schema_version: Literal["nebius-execution-storage-readiness/1"] = (
        "nebius-execution-storage-readiness/1"
    )
    nebius: InputStorageNebiusTarget
    object_storage: InputStorageObjectStorageTarget
    bucket_name: StrictStr = Field(pattern=r"^[a-z0-9](?:[a-z0-9-]{1,61}[a-z0-9])$")
    bucket_domain: StrictStr

    @field_validator("bucket_domain")
    @classmethod
    def canonical_domain(cls, value: str) -> str:
        _line(value, "bucket_domain")
        if value.casefold() != value or ":" in value or "/" in value:
            raise ValueError("bucket_domain must be one canonical DNS name")
        return value

    @property
    def configuration_id(self) -> str:
        digest = canonical_json_digest(self.model_dump(mode="json"))
        return f"execution-storage-readiness-configuration:{digest}"


class InputStorageAuthenticationReceipt(FrozenContract):
    nebius_cli_profile: StrictStr
    # Which `iam whoami` shape authenticated. Defaults to the federation user so
    # receipts recorded before service accounts were accepted still validate.
    profile_kind: Literal["user_profile", "service_account_profile"] = "user_profile"
    tenant_id: StrictStr
    project_id: StrictStr
    control_plane_identity_observed: Literal[True] = True
    object_storage_credential: ObjectStorageCredentialReuseReceipt
    authenticated: Literal[True] = True
    host_object_storage_credentials_resolved: Literal[True] = True
    job_object_storage_secret_version_bound: Literal[True] = True
    credential_selectors_distinct: Literal[True] = True


class FreshBucketReceipt(FrozenContract):
    bucket_id: StrictStr
    parent_id: StrictStr
    name: StrictStr
    state: Literal["ACTIVE"]
    suspended: Literal[False]
    domain: StrictStr
    region: StrictStr
    reconciled_after_ambiguous_create: StrictInt = Field(ge=0, le=1)


class FreshBucketNameAvailable(FrozenContract):
    bucket_name: StrictStr
    exact_match_count: Literal[0] = 0


class InputStorageReadinessWriteReceipt(FrozenContract):
    """Exact readback of the bucket lifecycle written at readiness.

    The configuration holds the disabled readiness rule named here plus the enabled
    rule that aborts incomplete multipart uploads; the exact readback covers both.
    """

    lifecycle_rule_id: StrictStr
    status: Literal["Disabled"]
    exact_readback: Literal[True]
    reconciled_after_ambiguous_write: StrictInt = Field(ge=0, le=1)

    @field_validator("lifecycle_rule_id")
    @classmethod
    def canonical_rule_id(cls, value: str) -> str:
        return _line(value, "lifecycle_rule_id")


class InventoryObject(FrozenContract):
    object_key: StrictStr
    byte_count: StrictInt = Field(gt=0)


class MultipartUpload(FrozenContract):
    object_key: StrictStr
    upload_id: StrictStr


class InputStorageInventory(FrozenContract):
    contact_complete: StrictInt = Field(ge=0, le=1)
    objects: tuple[InventoryObject, ...] = ()
    multipart_uploads: tuple[MultipartUpload, ...] = ()

    @model_validator(mode="after")
    def unique_inventory(self) -> InputStorageInventory:
        keys = [item.object_key for item in self.objects]
        uploads = [(item.object_key, item.upload_id) for item in self.multipart_uploads]
        if len(keys) != len(set(keys)) or len(uploads) != len(set(uploads)):
            raise ValueError("inventory repeated an object or multipart upload")
        return self


InputStoragePhase = Literal[
    "authentication",
    "bucket_freshness",
    "bucket_create",
    "bucket_readiness",
    "source_stream",
    "fastq_upload",
    "object_verification",
    "inventory_closure",
]

InputStorageFailureReason = Literal[
    "authentication_unavailable",
    "authentication_lost",
    "bucket_name_exists",
    "bucket_freshness_unknown",
    "bucket_create_ambiguous",
    "bucket_create_rejected",
    "bucket_create_unavailable",
    "bucket_identity_mismatch",
    "bucket_not_active",
    "bucket_suspended",
    "bucket_domain_mismatch",
    "bucket_region_mismatch",
    "bucket_not_empty",
    "readiness_write_rejected",
    "readiness_write_ambiguous",
    "readiness_write_mismatch",
    "source_unavailable",
    "source_mismatch",
    "object_collision",
    "transfer_failed",
    "object_missing",
    "object_unexpected",
    "hash_drift",
    "multipart_residue",
    "adapter_output_invalid",
]

InputStorageSafeErrorCategory = Literal[
    "transport_interrupted",
    "provider_rejected",
    "checksum_mismatch",
]


class InputStorageFailureEvidence(FrozenContract):
    """Bounded placement evidence; never provider text, headers, bodies, or URLs."""

    object_role: StrictStr
    object_key: StrictStr
    bytes_read: StrictInt | None = Field(default=None, ge=0)
    safe_error_category: InputStorageSafeErrorCategory
    http_status: StrictInt | None = Field(default=None, ge=100, le=599)


class ExecutionInputStorageAdapterError(RuntimeError):
    """Secret-free fail-stop result with the best bounded partial inventory."""

    def __init__(
        self,
        reason: InputStorageFailureReason,
        phase: InputStoragePhase,
        *,
        inventory: InputStorageInventory | None = None,
        failure_evidence: InputStorageFailureEvidence | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.phase = phase
        self.inventory = inventory or InputStorageInventory(contact_complete=0)
        self.failure_evidence = failure_evidence
