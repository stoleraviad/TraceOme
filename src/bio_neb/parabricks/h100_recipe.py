"""Typed worker recipe, manifest and closure contracts for current stages."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Literal

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from bio_neb.application.artifact_verification import WorkerTransferProof
from bio_neb.contracts import FrozenContract
from bio_neb.journey import (
    canonical_json,
    canonical_json_digest,
)
from bio_neb.parabricks.execution_observations import (
    ToolExited,
    ToolStarted,
    TruthfulExecutionObservation,
    UploadCompleted,
    UploadStarted,
    WorkerInputVerifiedObserved,
    WorkerPreparedReferenceVerifiedObserved,
    WorkerReceiptObserved,
    WorkerReferenceSupportObserved,
    WorkerRequestObserved,
    WorkerStartupObserved,
    WorkerToolsObserved,
    validate_truthful_observation_order,
)
from bio_neb.parabricks.execution_runtime import H100Runtime
from bio_neb.parabricks.worker_request import (
    PREPARED_ROLES,
    ConsumePreparedReferenceParameters,
    ExactWorkerObject,
    WorkerObjectStorage,
    WorkerObjectTarget,
    WorkerRequestIdentity,
    WorkerResources,
    compile_worker_request,
)
from bio_neb.parabricks.worker_request_placement import (
    reconcile_fetched_request_route,
    reconcile_observed_worker_request,
)
from bio_neb.providers.job_observations import (
    FetchedProviderEnvelope,
    ProviderCreateObserved,
)
from bio_neb.providers.runtime import NebiusCliTarget, RunObjectStorageTarget
from bio_neb.verified_artifacts import VerifiedObject


def _line(value: str, field: str) -> str:
    if not value or value.strip() != value or any(c in value for c in "\r\n\0"):
        raise ValueError(f"{field} must be one canonical line")
    return value


def _key(value: str, field: str) -> str:
    _line(value, field)
    path = PurePosixPath(value)
    if value.startswith("/") or value.endswith("/") or str(path) != value:
        raise ValueError(f"{field} must be one exact object key")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} contains a relative segment")
    return value


class H100OutputTarget(FrozenContract):
    role: Literal["aligned_bam", "VCF", "worker_receipt"]
    object_key: StrictStr
    local_target: StrictStr

    @field_validator("object_key")
    @classmethod
    def exact_key(cls, value: str) -> str:
        return _key(value, "object_key")

    @field_validator("local_target")
    @classmethod
    def exact_target(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or str(path) != value
            or not path.is_relative_to(PurePosixPath("/work"))
        ):
            raise ValueError("local_target must be an exact /work path")
        return value


class H100ExecutionRequest(FrozenContract):
    request_id: StrictStr = Field(pattern=r"^h100-execution-request:[0-9a-f]{64}$")
    journey_id: StrictStr
    authority_id: StrictStr
    proposal_id: StrictStr
    configuration_id: StrictStr
    prepared_reference_receipt_id: StrictStr
    execution_package_id: StrictStr
    deterministic_job_name: StrictStr = Field(pattern=r"^bio-neb-h100-[0-9a-f]{16}$")
    inputs: tuple[ExactWorkerObject, ...] = Field(min_length=10, max_length=10)
    outputs: tuple[H100OutputTarget, ...] = Field(min_length=3, max_length=3)
    parabricks_argv: tuple[StrictStr, ...]
    worker_request_identity: WorkerRequestIdentity
    source_proof_ids: tuple[StrictStr, StrictStr, StrictStr]

    @model_validator(mode="after")
    def exact_identity(self) -> H100ExecutionRequest:
        try:
            canonical_journey = str(uuid.UUID(self.journey_id))
        except ValueError as error:
            raise ValueError("journey_id must be a UUID") from error
        material = self.model_dump(
            mode="json",
            exclude={"request_id", "deterministic_job_name", "worker_request_identity"},
        )
        expected = "h100-execution-request:" + canonical_json_digest(material)
        if canonical_journey != self.journey_id or self.request_id != expected:
            raise ValueError("H100 request changed deterministic identity")
        return self


def output_targets(plan) -> tuple[H100OutputTarget, ...]:
    """Recipe-local worker destinations paired with the plan's object layout."""
    roles_and_paths = (
        ("aligned_bam", "/work/outputs/aligned.bam"),
        ("VCF", "/work/outputs/variants.vcf"),
        ("worker_receipt", "/work/receipts/worker.json"),
    )
    return tuple(
        H100OutputTarget(role=role, object_key=key, local_target=path)
        for (role, path), key in zip(
            roles_and_paths, plan.layout.h100_output_keys, strict=True
        )
    )


def configuration_id(plan) -> str:
    """Bind effective recipe inputs without retaining another configuration bundle."""
    return "h100-execution-configuration:" + canonical_json_digest(
        {
            "scope": plan.scope.model_dump(mode="json"),
            "storage": plan.storage.model_dump(mode="json"),
            "runtime": plan.runtimes.h100.model_dump(mode="json"),
            "outputs": [
                target.model_dump(mode="json") for target in output_targets(plan)
            ],
            "manifest_object_key": plan.layout.h100_manifest_object_key,
            "verification_mode": plan.verification.effective.h100_outputs,
        }
    )


def _worker_paths() -> dict[str, str]:
    fasta = "/work/reference/reference.fa"
    return {
        "read_1": "/work/inputs/read-1.fastq.gz",
        "read_2": "/work/inputs/read-2.fastq.gz",
        "genomic_fasta": fasta,
        "fasta_fai": fasta + ".fai",
        "bwa_amb": fasta + ".amb",
        "bwa_ann": fasta + ".ann",
        "bwa_bwt": fasta + ".bwt",
        "bwa_pac": fasta + ".pac",
        "bwa_sa": fasta + ".sa",
        "reference_dict": "/work/reference/reference.dict",
    }


def _compile_worker_request(
    storage: RunObjectStorageTarget,
    runtime: H100Runtime,
    output_prefix: str,
    request: H100ExecutionRequest,
):
    targets = {item.role: item for item in request.outputs}
    return compile_worker_request(
        ConsumePreparedReferenceParameters(
            request_id=request.request_id,
            storage=WorkerObjectStorage(
                bucket=storage.bucket,
                endpoint_url=storage.endpoint_url,
                region=storage.region,
            ),
            output_prefix=output_prefix,
            resources=WorkerResources(
                gpu_count=runtime.gpu_count,
                shm_size_bytes=runtime.shared_memory_bytes,
            ),
            inputs=request.inputs,
            outputs=tuple(
                WorkerObjectTarget(
                    role=targets[role].role,
                    object_key=targets[role].object_key,
                    local_target=targets[role].local_target,
                )
                for role in ("aligned_bam", "VCF")
            ),
            receipt_output=WorkerObjectTarget(
                role="worker_receipt",
                object_key=targets["worker_receipt"].object_key,
                local_target=targets["worker_receipt"].local_target,
            ),
            parabricks_argv=request.parabricks_argv,
        )
    )


def compile_h100_worker_request(
    storage: RunObjectStorageTarget,
    runtime: H100Runtime,
    output_prefix: str,
    request: H100ExecutionRequest,
):
    """Return the canonical schema-4 bytes bound to the retained request identity."""

    compiled = _compile_worker_request(storage, runtime, output_prefix, request)
    if compiled.identity != request.worker_request_identity:
        raise ValueError("H100 worker request identity drifted")
    return compiled


class H100SubmissionReceipt(FrozenContract):
    provider_job_id: StrictStr
    deterministic_job_name: StrictStr
    create_count: Literal[1] = 1
    # The name lookup is the identity path, never a recovery, so this records only
    # whether the create call's own response was usable.
    create_response_ambiguous: StrictBool = False
    # Identity of the retained create response.  The bytes themselves are kept
    # as immutable content-addressed Journey enrichment after the submission event;
    # only their identity belongs in its payload.
    create_response_byte_count: StrictInt = Field(default=0, ge=0)
    create_response_sha256: StrictStr = Field(
        default=hashlib.sha256(b"").hexdigest(), pattern=r"^[0-9a-f]{64}$"
    )


class H100OutputClosure(FrozenContract):
    outputs: tuple[VerifiedObject, ...] = Field(min_length=2, max_length=2)
    worker_receipt: VerifiedObject
    pre_manifest_inventory: tuple[StrictStr, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def exact_inventory(self) -> H100OutputClosure:
        if tuple(item.role for item in self.outputs) != ("aligned_bam", "VCF"):
            raise ValueError("output closure changed BAM/VCF order")
        expected = tuple(
            sorted(
                (
                    *[item.object_key for item in self.outputs],
                    self.worker_receipt.object_key,
                )
            )
        )
        if (
            self.worker_receipt.role != "worker_receipt"
            or self.pre_manifest_inventory != expected
        ):
            raise ValueError("output closure changed exact pre-manifest inventory")
        return self


class H100ExecutionManifest(FrozenContract):
    schema_version: Literal["bio-neb.execution-success-manifest/1"]
    authority: Literal["application"] = "application"
    request_id: StrictStr
    configuration_id: StrictStr
    prepared_reference_receipt_id: StrictStr
    provider_job_id: StrictStr
    deterministic_job_name: StrictStr
    worker_runtime_image: StrictStr
    worker_sha256: StrictStr
    pbrun_path: StrictStr
    pbrun_version: StrictStr
    parabricks_argv: tuple[StrictStr, ...]
    inputs: tuple[ExactWorkerObject, ...] = Field(min_length=10, max_length=10)
    outputs: tuple[VerifiedObject, ...] = Field(min_length=2, max_length=2)
    worker_receipt: VerifiedObject
    pre_manifest_inventory: tuple[StrictStr, ...] = Field(min_length=3, max_length=3)
    output_verification_mode: Literal["independent_full_read", "reuse_creation_proof"]
    transfer_proof: WorkerTransferProof


def validate_h100_observation_closure(
    scope: NebiusCliTarget,
    storage: RunObjectStorageTarget,
    runtime: H100Runtime,
    request: H100ExecutionRequest,
    submission: H100SubmissionReceipt | None,
    observations: Iterable[TruthfulExecutionObservation],
    *,
    durable: bool = False,
    container_path: str = "/opt/bio-neb/request.json",
) -> WorkerReceiptObserved:
    if submission is None:
        raise ValueError("H100 closure lacks submission")
    items = tuple(observations)
    validate_truthful_observation_order(items, terminal=True, durable=durable)
    creates = [item for item in items if isinstance(item, ProviderCreateObserved)]
    envelopes = [item for item in items if isinstance(item, FetchedProviderEnvelope)]
    startups = [item for item in items if isinstance(item, WorkerStartupObserved)]
    requests = [item for item in items if isinstance(item, WorkerRequestObserved)]
    tools = [item for item in items if isinstance(item, WorkerToolsObserved)]
    tool_identity = (
        {item.name: item for item in tools[0].tools} if len(tools) == 1 else {}
    )
    verified_inputs = [
        item for item in items if isinstance(item, WorkerInputVerifiedObserved)
    ]
    support = [
        item for item in items if isinstance(item, WorkerReferenceSupportObserved)
    ]
    prepared = [
        item
        for item in items
        if isinstance(item, WorkerPreparedReferenceVerifiedObserved)
    ]
    receipts = [item for item in items if isinstance(item, WorkerReceiptObserved)]
    scoped = [item for item in items if hasattr(item, "provider_job_id")]
    if len(envelopes) == 1:
        reconcile_fetched_request_route(
            expected_container_path=container_path,
            normalized_arguments=envelopes[0].normalized_arguments,
            injected_paths=envelopes[0].injected_paths,
        )
    if len(startups) == 1 and len(requests) == 1:
        reconcile_observed_worker_request(
            request.worker_request_identity,
            request_schema=startups[0].request_schema,
            byte_count=requests[0].byte_count,
            request_sha256=requests[0].request_sha256,
        )
    if (
        len(creates) != 1
        or any(item.provider_job_id != submission.provider_job_id for item in scoped)
        or creates[0].provider_job_name != request.deterministic_job_name
        or len(envelopes) != 1
        or envelopes[0].provider_job_name != request.deterministic_job_name
        or envelopes[0].project_id != scope.project_id
        or envelopes[0].provider_image_reference != runtime.provider_image_reference
        or envelopes[0].platform != runtime.platform
        or envelopes[0].preset != runtime.preset
        or envelopes[0].disk_size_bytes != runtime.disk_size_bytes
        or envelopes[0].subnet_id != scope.subnet_id
        or envelopes[0].container_command_requested != "image-entrypoint"
        or envelopes[0].container_command_fetched is not None
        or envelopes[0].environment_names
        != (
            "AWS_ACCESS_KEY_ID",
            "AWS_ENDPOINT_URL",
            "AWS_REGION",
            "AWS_SECRET_ACCESS_KEY",
        )
        or envelopes[0].version_bound_secret_environment_names
        != ("AWS_SECRET_ACCESS_KEY",)
        or envelopes[0].shared_memory_bytes != runtime.shared_memory_bytes
        or envelopes[0].restart_requested != "never"
        or envelopes[0].restart_fetched not in {"never", "omitted"}
        or len(startups) != 1
        or startups[0].interpreter != runtime.python_path
        or len(requests) != 1
        or requests[0].worker_sha256 != runtime.worker_sha256
        or len(tools) != 1
        or set(tool_identity) != {"python", "pbrun"}
        or tool_identity["python"].path != runtime.python_path
        or tool_identity["python"].resolved_path != startups[0].interpreter_resolved
        or tool_identity["python"].version != runtime.python_version
        or tool_identity["pbrun"].path != runtime.pbrun_path
        or tool_identity["pbrun"].resolved_path not in {None, runtime.pbrun_path}
        or tool_identity["pbrun"].version != runtime.pbrun_version
        or len(verified_inputs) != 10
        or tuple(item.role for item in verified_inputs)
        != ("read_1", "read_2", *PREPARED_ROLES)
        or tuple(
            (item.bucket, item.object_key, item.byte_count, item.sha256)
            for item in verified_inputs
        )
        != tuple(
            (
                item.bucket or storage.bucket,
                item.object_key,
                item.byte_count,
                item.sha256,
            )
            for item in request.inputs
        )
        or len(support) != 1
        or support[0].mode != "verify"
        or len(prepared) != 1
        or tuple(
            (item.role, item.object_key, item.byte_count, item.sha256)
            for item in support[0].objects
        )
        != tuple(
            (
                item.role,
                item.object_key,
                item.byte_count,
                item.sha256,
            )
            for item in request.inputs[2:]
        )
        or prepared[0].objects != support[0].objects
        or len(receipts) != 1
    ):
        raise ValueError("H100 provider, startup, worker, or tool identity changed")
    expected_argv = (runtime.pbrun_path, *request.parabricks_argv[1:])
    command_facts = [
        item for item in items if isinstance(item, (ToolStarted, ToolExited))
    ]
    support_index = items.index(support[0]) if support else -1
    prepared_index = items.index(prepared[0]) if prepared else -1
    tool_index = next(
        (index for index, item in enumerate(items) if isinstance(item, ToolStarted)),
        -1,
    )
    if not (support_index < prepared_index < tool_index):
        raise ValueError("H100 prepared-reference proof did not precede pbrun")
    if command_facts != [
        ToolStarted(command_id="pbrun-germline", argv=expected_argv),
        ToolExited(command_id="pbrun-germline", returncode=0),
    ]:
        raise ValueError("H100 worker changed the sole pbrun vector or actual exit")
    starts = [item for item in items if isinstance(item, UploadStarted)]
    completions = [item for item in items if isinstance(item, UploadCompleted)]
    targets = {item.role: item for item in request.outputs}
    roles = (
        ("aligned_bam", "VCF") if durable else ("aligned_bam", "VCF", "worker_receipt")
    )
    if (
        tuple(item.role for item in starts) != roles
        or tuple(item.role for item in completions) != roles
        or tuple((item.bucket, item.object_key) for item in starts)
        != tuple((storage.bucket, targets[role].object_key) for role in roles)
        or tuple((item.byte_count, item.sha256) for item in starts)
        != tuple((item.byte_count, item.sha256) for item in completions)
    ):
        raise ValueError("H100 worker changed BAM, VCF, receipt upload order")
    receipt = receipts[0]
    receipt_bytes = canonical_json(receipt.receipt).encode()
    if (
        receipt.receipt.get("schema_version")
        != (
            "bio-neb.parabricks-worker-receipt/5"
            if durable
            else "bio-neb.parabricks-worker-receipt/3"
        )
        or receipt.receipt.get("operation") != "consume_prepared_reference"
        or receipt.receipt.get("workflow") != "nvidia-parabricks-germline"
        or receipt.receipt.get("request_id") != request.request_id
        or receipt.receipt.get("prepared_inputs")
        != [
            {
                "role": item.role,
                "object_key": item.object_key,
                "byte_count": item.byte_count,
                "sha256": item.sha256,
            }
            for item in request.inputs[2:]
        ]
        or receipt.receipt_object.bucket != storage.bucket
        or receipt.receipt_object.object_key != targets["worker_receipt"].object_key
        or (
            not durable
            and (receipt.receipt_object.byte_count, receipt.receipt_object.sha256)
            != (len(receipt_bytes), hashlib.sha256(receipt_bytes).hexdigest())
        )
    ):
        raise ValueError("H100 worker receipt changed request identity or bytes")
    return receipt


def compile_h100_manifest(
    storage: RunObjectStorageTarget,
    runtime: H100Runtime,
    output_verification_mode: str,
    request: H100ExecutionRequest,
    submission: H100SubmissionReceipt,
    closure: H100OutputClosure,
) -> H100ExecutionManifest:
    return H100ExecutionManifest(
        schema_version="bio-neb.execution-success-manifest/1",
        request_id=request.request_id,
        configuration_id=request.configuration_id,
        prepared_reference_receipt_id=request.prepared_reference_receipt_id,
        provider_job_id=submission.provider_job_id,
        deterministic_job_name=request.deterministic_job_name,
        worker_runtime_image=runtime.worker_runtime_image,
        worker_sha256=runtime.worker_sha256,
        pbrun_path=runtime.pbrun_path,
        pbrun_version=runtime.pbrun_version,
        parabricks_argv=request.parabricks_argv,
        inputs=request.inputs,
        outputs=closure.outputs,
        worker_receipt=closure.worker_receipt,
        pre_manifest_inventory=closure.pre_manifest_inventory,
        output_verification_mode=output_verification_mode,
        transfer_proof=WorkerTransferProof.bind(
            storage=storage,
            runtime=runtime,
            configuration_id=request.configuration_id,
            request=request,
            source_proof_ids=request.source_proof_ids,
            inputs=request.inputs,
            outputs=closure.outputs,
            worker_receipt=closure.worker_receipt,
            mode=output_verification_mode,
        ),
    )
