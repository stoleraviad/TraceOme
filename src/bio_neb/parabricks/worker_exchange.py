"""Provider-free backend/worker exchange definitions.

Wire shape and canonical JSON live here. Compilation and independent worker
admission retain their own semantic checks; result types describe observations,
never artifact verification or execution authority.
"""

import json
from typing import Annotated, Literal

from pydantic import (
    Field,
    JsonValue,
    StrictInt,
    StrictStr,
    TypeAdapter,
    model_validator,
)

from bio_neb.contracts import StrictFrozenContract

REQUEST_SCHEMA_VERSION = "bio-neb.parabricks-worker-request/7"
RECEIPT_SCHEMA_VERSION = "bio-neb.parabricks-worker-receipt/5"
MAX_REQUEST_BYTES = 64 * 1024 - 1
PREPARED_ROLES = (
    "genomic_fasta",
    "fasta_fai",
    "reference_dict",
    "bwa_amb",
    "bwa_ann",
    "bwa_bwt",
    "bwa_pac",
    "bwa_sa",
)
GPU_INPUT_ROLES = ("read_1", "read_2", *PREPARED_ROLES)
PREPARATION_COMMANDS = (
    "bwa-index",
    "samtools-faidx",
    "samtools-dict",
)


class ExactWorkerObject(StrictFrozenContract):
    role: StrictStr
    bucket: StrictStr | None = None
    object_key: StrictStr
    local_target: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr


class WorkerObjectTarget(StrictFrozenContract):
    role: StrictStr
    object_key: StrictStr
    local_target: StrictStr


class PlainFastaIdentity(StrictFrozenContract):
    local_target: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr


class AcceptedPlainFasta(PlainFastaIdentity):
    role: Literal["genomic_fasta"]


class PlainFastaTarget(StrictFrozenContract):
    role: Literal["genomic_fasta"]
    local_target: StrictStr


class PublicReferenceDownload(StrictFrozenContract):
    """The approved publisher URL the worker fetches and writes to its object key."""

    role: Literal["reference_source"]
    url: StrictStr
    byte_count: StrictInt = Field(gt=0)
    md5: StrictStr = Field(pattern=r"^[0-9a-f]{32}$")
    object_key: StrictStr
    local_target: StrictStr


class PreparationLineage(StrictFrozenContract):
    """A stored source carries its accepted plain identity; a download measures it.
    The plain FASTA is the decompressed source without the records of
    ``excluded_sequences`` (`fasta_selection.FastaRecordSelection`, #464)."""

    source_role: Literal["reference_source"]
    compression: Literal["gzip"]
    excluded_sequences: list[StrictStr]
    accepted_plain_fasta: AcceptedPlainFasta | None = None
    plain_fasta: PlainFastaTarget | None = None
    commands: list[StrictStr] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def one_plain_fasta(self) -> "PreparationLineage":
        if (self.accepted_plain_fasta is None) == (self.plain_fasta is None):
            raise ValueError("preparation names exactly one plain FASTA form")
        return self


class WorkerResources(StrictFrozenContract):
    gpu_count: StrictInt = Field(gt=0)
    shm_size_bytes: StrictInt = Field(gt=0)


class RuntimeSmokeRequest(StrictFrozenContract):
    schema_version: Literal["bio-neb.parabricks-worker-request/7"]
    operation: Literal["runtime_smoke"]
    request_id: StrictStr
    workflow: Literal["nvidia-parabricks-germline"]


class PrepareReferenceRequest(StrictFrozenContract):
    schema_version: Literal["bio-neb.parabricks-worker-request/7"]
    operation: Literal["prepare_reference"]
    request_id: StrictStr
    bucket: StrictStr
    endpoint_url: StrictStr
    region: StrictStr
    output_prefix: StrictStr
    workflow: Literal["bwa-samtools-reference-preparation"]
    inputs: list[ExactWorkerObject] = Field(max_length=1)
    download: PublicReferenceDownload | None = None
    reference: PreparationLineage
    prepared_outputs: list[WorkerObjectTarget] = Field(min_length=8, max_length=8)
    receipt_output: WorkerObjectTarget

    @model_validator(mode="after")
    def one_source(self) -> "PrepareReferenceRequest":
        stored = len(self.inputs) == 1
        if stored == (self.download is not None) or stored != (
            self.reference.accepted_plain_fasta is not None
        ):
            raise ValueError("preparation names one stored or downloaded source")
        return self


class ConsumePreparedReferenceRequest(StrictFrozenContract):
    schema_version: Literal["bio-neb.parabricks-worker-request/7"]
    operation: Literal["consume_prepared_reference"]
    request_id: StrictStr
    bucket: StrictStr
    endpoint_url: StrictStr
    region: StrictStr
    output_prefix: StrictStr
    workflow: Literal["nvidia-parabricks-germline"]
    resources: WorkerResources
    inputs: list[ExactWorkerObject] = Field(min_length=10, max_length=10)
    outputs: list[WorkerObjectTarget] = Field(min_length=2, max_length=2)
    receipt_output: WorkerObjectTarget
    parabricks_argv: list[StrictStr] = Field(min_length=2)


WorkerRequest = Annotated[
    RuntimeSmokeRequest | PrepareReferenceRequest | ConsumePreparedReferenceRequest,
    Field(discriminator="operation"),
]
REQUEST_ADAPTER = TypeAdapter(WorkerRequest)


class ObservedObject(StrictFrozenContract):
    role: StrictStr
    object_key: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class UploadedObject(ObservedObject):
    bucket: StrictStr


class ObservedTool(StrictFrozenContract):
    path: StrictStr
    version: StrictStr
    resolved_path: StrictStr | None = None


class _WorkerReceipt(StrictFrozenContract):
    schema_version: Literal["bio-neb.parabricks-worker-receipt/5"]
    authority: Literal["non-authoritative-worker-observation"]
    request_id: StrictStr
    tools: dict[StrictStr, ObservedTool]
    validation: list[dict[str, JsonValue]] = Field(min_length=1, max_length=128)


class AcquiredSource(ObservedObject):
    """The source bytes the worker read: their md5 and where they came from."""

    md5: StrictStr = Field(pattern=r"^[0-9a-f]{32}$")
    origin: StrictStr


class PreparedReferenceReceipt(_WorkerReceipt):
    operation: Literal["prepare_reference"]
    workflow: Literal["bwa-samtools-reference-preparation"]
    source: AcquiredSource
    prepared_uploads: list[UploadedObject] = Field(min_length=8, max_length=8)


class ConsumedReferenceReceipt(_WorkerReceipt):
    operation: Literal["consume_prepared_reference"]
    workflow: Literal["nvidia-parabricks-germline"]
    prepared_inputs: list[ObservedObject] = Field(min_length=8, max_length=8)
    uploads: list[UploadedObject] = Field(min_length=2, max_length=2)


WorkerReceipt = Annotated[
    PreparedReferenceReceipt | ConsumedReferenceReceipt,
    Field(discriminator="operation"),
]
RECEIPT_ADAPTER = TypeAdapter(WorkerReceipt)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
