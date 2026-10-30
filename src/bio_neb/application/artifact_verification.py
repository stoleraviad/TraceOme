"""Reusable producer proof and explicit audits; transport remains provider-owned."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from bio_neb.application.verification_policy import (
    ObjectVerificationCheckpoints,
    ObjectVerificationMode,
)
from bio_neb.journey import canonical_json_digest

if TYPE_CHECKING:
    from bio_neb.parabricks.execution_runtime import CpuReferenceRuntime, H100Runtime
    from bio_neb.parabricks.h100_recipe import H100ExecutionRequest
    from bio_neb.parabricks.reference_recipe import ReferencePreparationRequest
    from bio_neb.parabricks.worker_request import ExactWorkerObject
    from bio_neb.providers.runtime import RunObjectStorageTarget
    from bio_neb.verified_artifacts import VerifiedObject


def use_produced_object(store, item, *, proof_ref, mode):
    """Use validated producer evidence; an opted-in audit never replaces its origin.

    Callers validate worker/request/authority meaning before supplying the object.
    Only the explicit audit performs another remote read and records its own contact.
    """
    from bio_neb.failure import ProviderError
    from bio_neb.providers.ports import Receipt

    if mode not in {"reuse_creation_proof", "independent_full_read"}:
        raise ValueError("unknown artifact verification mode")
    if not proof_ref:
        raise ProviderError("semantic", "artifact_producer_proof_missing")
    values = {"bytes": item.byte_count, "sha256": item.sha256}
    if mode == "independent_full_read":
        audit = store.head(item.bucket, item.object_key, verification=mode)
        if (audit.values.get("bytes"), audit.values.get("sha256")) != (
            item.byte_count,
            item.sha256,
        ):
            raise ProviderError(
                "semantic",
                "artifact_identity_mismatch",
                {
                    "expected_bytes": item.byte_count,
                    "expected_sha256": item.sha256,
                    "observed_bytes": audit.values.get("bytes"),
                    "observed_sha256": audit.values.get("sha256"),
                },
            )
        values["audit_proof_ref"] = audit.proof_ref
    return Receipt(values=values, proof_ref=proof_ref)


class WorkerTransferProof(BaseModel):
    """Host-reconciled worker evidence; byte totals describe the attested contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proof_id: StrictStr
    assurance: Literal["attested_worker_stream_and_controlled_run_storage"]
    endpoint_url: StrictStr
    region: StrictStr
    bucket: StrictStr
    configuration_id: StrictStr
    request_id: StrictStr
    worker_runtime_image: StrictStr
    worker_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    worker_request_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    worker_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    source_proof_ids: tuple[StrictStr, ...] = Field(min_length=1)
    inputs_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    outputs_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    verification_mode: ObjectVerificationMode
    checkpoints: ObjectVerificationCheckpoints
    accounting_basis: Literal["derived_from_attested_worker_contract"]
    download_bytes: StrictInt = Field(gt=0)
    producer_admission_bytes: StrictInt = Field(gt=0)
    upload_bytes: StrictInt = Field(gt=0)
    host_audit_bytes: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def exact_proof(self) -> WorkerTransferProof:
        if self.proof_id != "worker-transfer-proof:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"proof_id"})
        ):
            raise ValueError("worker transfer proof changed")
        if (
            self.upload_bytes != self.producer_admission_bytes
            or self.host_audit_bytes
            != (
                self.upload_bytes
                if self.verification_mode == "independent_full_read"
                else 0
            )
        ):
            raise ValueError("worker transfer checkpoint accounting changed")
        return self

    @classmethod
    def bind(
        cls,
        *,
        storage: RunObjectStorageTarget,
        runtime: H100Runtime | CpuReferenceRuntime,
        configuration_id: str,
        request: H100ExecutionRequest | ReferencePreparationRequest,
        source_proof_ids: tuple[str, ...],
        inputs: tuple[ExactWorkerObject, ...] | tuple[VerifiedObject, ...],
        outputs: tuple[VerifiedObject, ...],
        worker_receipt: VerifiedObject,
        mode: ObjectVerificationMode,
    ) -> WorkerTransferProof:
        output_bytes = sum(item.byte_count for item in outputs)
        material = {
            "assurance": "attested_worker_stream_and_controlled_run_storage",
            "endpoint_url": storage.endpoint_url,
            "region": storage.region,
            "bucket": storage.bucket,
            "configuration_id": configuration_id,
            "request_id": request.request_id,
            "worker_runtime_image": runtime.worker_runtime_image,
            "worker_sha256": runtime.worker_sha256,
            "worker_request_sha256": request.worker_request_identity.sha256,
            "worker_receipt_sha256": worker_receipt.sha256,
            "source_proof_ids": list(source_proof_ids),
            "inputs_sha256": canonical_json_digest(
                [item.model_dump(mode="json") for item in inputs]
            ),
            "outputs_sha256": canonical_json_digest(
                [item.model_dump(mode="json") for item in outputs]
            ),
            "verification_mode": mode,
            "checkpoints": ObjectVerificationCheckpoints().model_dump(),
            "accounting_basis": "derived_from_attested_worker_contract",
            "download_bytes": sum(item.byte_count for item in inputs),
            "producer_admission_bytes": output_bytes,
            "upload_bytes": output_bytes,
            "host_audit_bytes": output_bytes if mode == "independent_full_read" else 0,
        }
        return cls(
            proof_id="worker-transfer-proof:" + canonical_json_digest(material),
            **material,
        )
