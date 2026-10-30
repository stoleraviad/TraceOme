"""Typed compact request contract for the baked reusable worker.

This domain module compiles ordinary typed parameters into the baked worker's
schema-7 canonical bytes and exact identity. Native placement and provider-route
reconciliation are separate Adapter-side responsibilities.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from bio_neb.parabricks.worker_exchange import (
    GPU_INPUT_ROLES,
    MAX_REQUEST_BYTES,
    PREPARATION_COMMANDS,
    PREPARED_ROLES,
    REQUEST_SCHEMA_VERSION,
    ConsumePreparedReferenceRequest,
    ExactWorkerObject,
    PreparationLineage,
    PrepareReferenceRequest,
    PublicReferenceDownload,
    RuntimeSmokeRequest,
    WorkerObjectTarget,
    WorkerResources,
    canonical_bytes,
)
from bio_neb.parabricks.worker_exchange import (
    AcceptedPlainFasta as WireAcceptedPlainFasta,
)
from bio_neb.parabricks.worker_exchange import (
    PlainFastaIdentity as AcceptedPlainFasta,
)
from bio_neb.parabricks.worker_exchange import (
    PlainFastaTarget as WirePlainFastaTarget,
)

_COMPILED_REQUEST_TOKEN = object()

WorkerOperation = Literal[
    "prepare_reference", "consume_prepared_reference", "runtime_smoke"
]


class WorkerRequestContractError(ValueError):
    """A bounded adapter-side request or provider-boundary failure."""


class _Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class WorkerObjectStorage(_Parameters):
    bucket: StrictStr
    endpoint_url: StrictStr
    region: StrictStr


class PlainFastaTarget(_Parameters):
    """Where a downloaded source's plain FASTA lands; the worker measures it."""

    local_target: StrictStr


class PrepareReferenceParameters(_Parameters):
    """A stored source carries its accepted plain FASTA; a public URL does not.
    ``excluded_sequences`` are the records the plain FASTA leaves out (#464)."""

    request_id: StrictStr
    storage: WorkerObjectStorage
    output_prefix: StrictStr
    source: ExactWorkerObject | PublicReferenceDownload
    plain_fasta: AcceptedPlainFasta | PlainFastaTarget
    excluded_sequences: tuple[StrictStr, ...]
    prepared_outputs: tuple[WorkerObjectTarget, ...] = Field(min_length=8, max_length=8)
    receipt_output: WorkerObjectTarget


class ConsumePreparedReferenceParameters(_Parameters):
    request_id: StrictStr
    storage: WorkerObjectStorage
    output_prefix: StrictStr
    resources: WorkerResources
    inputs: tuple[ExactWorkerObject, ...] = Field(min_length=10, max_length=10)
    outputs: tuple[WorkerObjectTarget, ...] = Field(min_length=2, max_length=2)
    receipt_output: WorkerObjectTarget
    parabricks_argv: tuple[StrictStr, ...] = Field(min_length=2)


class RuntimeSmokeParameters(_Parameters):
    request_id: StrictStr


WorkerRequestParameters = (
    PrepareReferenceParameters
    | ConsumePreparedReferenceParameters
    | RuntimeSmokeParameters
)


class WorkerRequestIdentity(_Parameters):
    schema_version: Literal["bio-neb.parabricks-worker-request/7"]
    operation: WorkerOperation
    byte_count: StrictInt = Field(gt=0, lt=64 * 1024)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, init=False)
class CompiledWorkerRequest:
    """Canonical retained request bytes and their exact identity."""

    request_bytes: bytes
    identity: WorkerRequestIdentity

    def __new__(cls, token: object = None) -> "CompiledWorkerRequest":
        if token is not _COMPILED_REQUEST_TOKEN:
            raise TypeError("compiled worker requests are created only by the compiler")
        return super().__new__(cls)

    @classmethod
    def _create(
        cls, request_bytes: bytes, identity: WorkerRequestIdentity
    ) -> "CompiledWorkerRequest":
        compiled = cls(_COMPILED_REQUEST_TOKEN)
        object.__setattr__(compiled, "request_bytes", request_bytes)
        object.__setattr__(compiled, "identity", identity)
        compiled._require_exact_identity()
        return compiled

    def _require_exact_identity(self) -> None:
        try:
            document = json.loads(self.request_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WorkerRequestContractError("worker_request_invalid") from error
        if (
            not self.request_bytes
            or len(self.request_bytes) > MAX_REQUEST_BYTES
            or canonical_bytes(document) != self.request_bytes
            or not isinstance(document, dict)
            or document.get("schema_version") != self.identity.schema_version
            or document.get("operation") != self.identity.operation
            or len(self.request_bytes) != self.identity.byte_count
            or hashlib.sha256(self.request_bytes).hexdigest() != self.identity.sha256
        ):
            raise WorkerRequestContractError("worker_request_identity_invalid")


def _model_json(value: BaseModel) -> dict[str, object]:
    # Optional per-input buckets are omitted for the single-bucket wire identity
    # and emitted only for a cross-bucket immutable link; so are unused source forms.
    return value.model_dump(mode="json", exclude_none=True)


def _line(value: str) -> str:
    if (
        not value
        or value.strip() != value
        or any(character in value for character in "\r\n\x00")
    ):
        raise WorkerRequestContractError("worker_request_value_invalid")
    return value


def _object_key(value: str, *, prefix: bool = False) -> str:
    key = _line(value)
    if key.startswith("/") or "\\" in key or "//" in key or key.endswith("/") != prefix:
        raise WorkerRequestContractError("worker_request_object_key_invalid")
    path_value = key[:-1] if prefix else key
    path = PurePosixPath(path_value)
    if str(path) != path_value or any(part in {"", ".", ".."} for part in path.parts):
        raise WorkerRequestContractError("worker_request_object_key_invalid")
    return key


def _local_target(value: str) -> str:
    target = _line(value)
    path = PurePosixPath(target)
    root = PurePosixPath("/work")
    if (
        not path.is_absolute()
        or str(path) != target
        or path == root
        or not path.is_relative_to(root)
        or ".." in path.parts
    ):
        raise WorkerRequestContractError("worker_request_local_target_invalid")
    return target


def _exact_object(item: ExactWorkerObject) -> None:
    _line(item.role)
    if item.bucket is not None:
        _line(item.bucket)
    _object_key(item.object_key)
    _local_target(item.local_target)
    if re.fullmatch(r"[0-9a-f]{64}", item.sha256) is None:
        raise WorkerRequestContractError("worker_request_sha256_invalid")


def _target(item: WorkerObjectTarget, prefix: str) -> None:
    _line(item.role)
    _object_key(item.object_key)
    _local_target(item.local_target)
    if not item.object_key.startswith(prefix):
        raise WorkerRequestContractError("worker_request_output_prefix_drift")


def _prepared_paths(fasta: str) -> dict[str, str]:
    return {
        "genomic_fasta": fasta,
        "fasta_fai": fasta + ".fai",
        "reference_dict": str(PurePosixPath(fasta).with_suffix(".dict")),
        **{
            f"bwa_{extension}": fasta + f".{extension}"
            for extension in ("amb", "ann", "bwt", "pac", "sa")
        },
    }


def _option_values(
    arguments: tuple[str, ...], option: str, count: int
) -> tuple[str, ...]:
    if arguments.count(option) != 1:
        raise WorkerRequestContractError("worker_request_parabricks_binding_invalid")
    index = arguments.index(option)
    values = arguments[index + 1 : index + 1 + count]
    if len(values) != count or any(value.startswith("--") for value in values):
        raise WorkerRequestContractError("worker_request_parabricks_binding_invalid")
    return values


def _preparation_source_checked(parameters: PrepareReferenceParameters):
    """A stored source needs its accepted plain FASTA; a download only its target."""
    source, plain = parameters.source, parameters.plain_fasta
    if isinstance(source, PublicReferenceDownload):
        if not isinstance(plain, PlainFastaTarget):
            raise WorkerRequestContractError("worker_request_plain_fasta_invalid")
        if not _line(source.url).startswith("https://"):
            raise WorkerRequestContractError("worker_request_source_url_invalid")
        _object_key(source.object_key)
        _local_target(source.local_target)
    else:
        if not isinstance(plain, AcceptedPlainFasta):
            raise WorkerRequestContractError("worker_request_plain_fasta_invalid")
        _exact_object(source)
        if source.role != "reference_source":
            raise WorkerRequestContractError("worker_request_role_inventory_invalid")
        if re.fullmatch(r"[0-9a-f]{64}", plain.sha256) is None:
            raise WorkerRequestContractError("worker_request_sha256_invalid")
    _local_target(plain.local_target)
    excluded = parameters.excluded_sequences
    if len(set(excluded)) != len(excluded) or any(
        not name or name.split() != [name] for name in excluded
    ):
        raise WorkerRequestContractError("worker_request_excluded_sequences_invalid")
    return source, plain


def _validate_parameters(parameters: WorkerRequestParameters) -> None:
    _line(parameters.request_id)
    if isinstance(parameters, RuntimeSmokeParameters):
        return
    _line(parameters.storage.bucket)
    if not _line(parameters.storage.endpoint_url).startswith("https://"):
        raise WorkerRequestContractError("worker_request_endpoint_invalid")
    _line(parameters.storage.region)
    prefix = _object_key(parameters.output_prefix, prefix=True)
    _target(parameters.receipt_output, prefix)
    if parameters.receipt_output.role != "worker_receipt":
        raise WorkerRequestContractError("worker_request_receipt_role_invalid")

    if isinstance(parameters, PrepareReferenceParameters):
        source, plain = _preparation_source_checked(parameters)
        if source.local_target == plain.local_target:
            raise WorkerRequestContractError("worker_request_local_target_collision")
        outputs = {item.role: item for item in parameters.prepared_outputs}
        if len(outputs) != len(parameters.prepared_outputs) or set(outputs) != set(
            PREPARED_ROLES
        ):
            raise WorkerRequestContractError("worker_request_role_inventory_invalid")
        expected_paths = _prepared_paths(plain.local_target)
        for role in PREPARED_ROLES:
            _target(outputs[role], prefix)
            if outputs[role].local_target != expected_paths[role]:
                raise WorkerRequestContractError("worker_request_local_target_drift")
        targets = [
            parameters.source.local_target,
            *(outputs[role].local_target for role in PREPARED_ROLES),
            parameters.receipt_output.local_target,
        ]
        keys = [
            *(outputs[role].object_key for role in PREPARED_ROLES),
            parameters.receipt_output.object_key,
        ]
        if isinstance(source, PublicReferenceDownload):
            keys.append(source.object_key)
    else:
        for item in parameters.inputs:
            _exact_object(item)
        inputs = {item.role: item for item in parameters.inputs}
        input_roles = set(GPU_INPUT_ROLES)
        if len(inputs) != len(parameters.inputs) or set(inputs) != input_roles:
            raise WorkerRequestContractError("worker_request_role_inventory_invalid")
        fasta = inputs["genomic_fasta"].local_target
        for role, target in _prepared_paths(fasta).items():
            if inputs[role].local_target != target:
                raise WorkerRequestContractError("worker_request_local_target_drift")
        outputs = {item.role: item for item in parameters.outputs}
        if len(outputs) != len(parameters.outputs) or set(outputs) != {
            "aligned_bam",
            "VCF",
        }:
            raise WorkerRequestContractError("worker_request_role_inventory_invalid")
        for role in ("aligned_bam", "VCF"):
            _target(outputs[role], prefix)
        arguments = tuple(_line(value) for value in parameters.parabricks_argv)
        if arguments[:2] != ("pbrun", "germline"):
            raise WorkerRequestContractError(
                "worker_request_parabricks_binding_invalid"
            )
        bindings = {
            "--ref": (fasta,),
            "--in-fq": (
                inputs["read_1"].local_target,
                inputs["read_2"].local_target,
            ),
            "--out-bam": (outputs["aligned_bam"].local_target,),
            "--out-variants": (outputs["VCF"].local_target,),
            "--num-gpus": (str(parameters.resources.gpu_count),),
        }
        if any(
            _option_values(arguments, option, len(values)) != values
            for option, values in bindings.items()
        ):
            raise WorkerRequestContractError(
                "worker_request_parabricks_binding_invalid"
            )
        targets = [
            *(item.local_target for item in parameters.inputs),
            *(outputs[role].local_target for role in ("aligned_bam", "VCF")),
            parameters.receipt_output.local_target,
        ]
        keys = [
            *(item.object_key for item in parameters.inputs),
            *(outputs[role].object_key for role in ("aligned_bam", "VCF")),
            parameters.receipt_output.object_key,
        ]
    if len(targets) != len(set(targets)) or len(keys) != len(set(keys)):
        raise WorkerRequestContractError("worker_request_identity_collision")


_Object = TypeVar("_Object", ExactWorkerObject, WorkerObjectTarget)


def _ordered_by_role(
    values: tuple[_Object, ...], roles: tuple[str, ...]
) -> list[_Object]:
    by_role = {item.role: item for item in values}
    if len(by_role) != len(values) or set(by_role) != set(roles):
        raise WorkerRequestContractError("worker_request_role_inventory_invalid")
    return [by_role[role] for role in roles]


def _preparation_source(parameters: PrepareReferenceParameters) -> dict[str, object]:
    """The wire names a stored source with its accepted plain FASTA, or a download."""
    lineage = {
        "source_role": "reference_source",
        "compression": "gzip",
        "excluded_sequences": list(parameters.excluded_sequences),
        "commands": list(PREPARATION_COMMANDS),
    }
    if isinstance(parameters.source, PublicReferenceDownload):
        return {
            "inputs": [],
            "download": parameters.source,
            "reference": PreparationLineage(
                **lineage,
                plain_fasta=WirePlainFastaTarget(
                    role="genomic_fasta",
                    local_target=parameters.plain_fasta.local_target,
                ),
            ),
        }
    return {
        "inputs": [parameters.source],
        "reference": PreparationLineage(
            **lineage,
            accepted_plain_fasta=WireAcceptedPlainFasta(
                role="genomic_fasta", **_model_json(parameters.plain_fasta)
            ),
        ),
    }


def _compile_document(parameters: WorkerRequestParameters) -> dict[str, object]:
    # Caller parameters group storage and accept unordered role collections.
    # The wire flattens storage, fixes operation declarations, and orders roles
    # before hashing. These conversions preserve the existing canonical bytes.
    if isinstance(parameters, RuntimeSmokeParameters):
        request = RuntimeSmokeRequest(
            schema_version=REQUEST_SCHEMA_VERSION,
            operation="runtime_smoke",
            request_id=parameters.request_id,
            workflow="nvidia-parabricks-germline",
        )
    elif isinstance(parameters, PrepareReferenceParameters):
        request = PrepareReferenceRequest(
            schema_version=REQUEST_SCHEMA_VERSION,
            operation="prepare_reference",
            request_id=parameters.request_id,
            **_model_json(parameters.storage),
            output_prefix=parameters.output_prefix,
            workflow="bwa-samtools-reference-preparation",
            **_preparation_source(parameters),
            prepared_outputs=_ordered_by_role(
                parameters.prepared_outputs, PREPARED_ROLES
            ),
            receipt_output=parameters.receipt_output,
        )
    else:
        request = ConsumePreparedReferenceRequest(
            schema_version=REQUEST_SCHEMA_VERSION,
            operation="consume_prepared_reference",
            request_id=parameters.request_id,
            **_model_json(parameters.storage),
            output_prefix=parameters.output_prefix,
            workflow="nvidia-parabricks-germline",
            resources=parameters.resources,
            inputs=_ordered_by_role(parameters.inputs, GPU_INPUT_ROLES),
            outputs=_ordered_by_role(parameters.outputs, ("aligned_bam", "VCF")),
            receipt_output=parameters.receipt_output,
            parabricks_argv=list(parameters.parabricks_argv),
        )
    return _model_json(request)


def compile_worker_request(
    parameters: WorkerRequestParameters,
) -> CompiledWorkerRequest:
    """Compile typed caller parameters into the baked worker's exact schema."""
    if not isinstance(
        parameters,
        (
            PrepareReferenceParameters,
            ConsumePreparedReferenceParameters,
            RuntimeSmokeParameters,
        ),
    ):
        raise WorkerRequestContractError("worker_request_parameters_invalid")
    try:
        _validate_parameters(parameters)
        payload = canonical_bytes(_compile_document(parameters))
    except ValueError as error:
        if isinstance(error, WorkerRequestContractError):
            raise
        raise WorkerRequestContractError("worker_request_invalid") from error
    if len(payload) > MAX_REQUEST_BYTES:
        raise WorkerRequestContractError("worker_request_oversized")
    operation: WorkerOperation
    if isinstance(parameters, PrepareReferenceParameters):
        operation = "prepare_reference"
    elif isinstance(parameters, ConsumePreparedReferenceParameters):
        operation = "consume_prepared_reference"
    else:
        operation = "runtime_smoke"
    identity = WorkerRequestIdentity(
        schema_version=REQUEST_SCHEMA_VERSION,
        operation=operation,
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    return CompiledWorkerRequest._create(payload, identity)


__all__ = [
    "AcceptedPlainFasta",
    "CompiledWorkerRequest",
    "ConsumePreparedReferenceParameters",
    "ExactWorkerObject",
    "MAX_REQUEST_BYTES",
    "PREPARED_ROLES",
    "PlainFastaTarget",
    "PrepareReferenceParameters",
    "PublicReferenceDownload",
    "REQUEST_SCHEMA_VERSION",
    "RuntimeSmokeParameters",
    "WorkerObjectStorage",
    "WorkerObjectTarget",
    "WorkerRequestContractError",
    "WorkerRequestIdentity",
    "WorkerResources",
    "compile_worker_request",
]
