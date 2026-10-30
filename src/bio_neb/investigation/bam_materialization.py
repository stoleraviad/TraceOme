"""Materialize the Run's verified BAM: one download in ranged parts (#363).

Each part is retried only for a transient fault, by the ``bam_materialization``
policy bound here; the streamed digest over every part, matched to the Execution
result's descriptor, is the acquisition proof. Nothing rereads the written file.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from bio_neb.application.artifact_materialization import (
    MaterializationFiles,
    MaterializedFile,
    ObjectMaterializationLineage,
    ObjectMaterializationRefused,
    ObjectReadPort,
    ObjectReadResult,
    PartObserver,
    execution_object_storage,
)
from bio_neb.contracts import FrozenContract
from bio_neb.failure import FailureSurface, operation_failure_policy, policy_for
from bio_neb.journey import (
    JourneyFile,
    JourneyValidationError,
    canonical_json,
    canonical_json_digest,
)
from bio_neb.parabricks.result import (
    ParabricksExecutionIncomplete,
    ParabricksExecutionOutcome,
    read_completed_parabricks_result,
)
from bio_neb.providers.nebius_common import ExecutionContactObserver
from bio_neb.providers.nebius_object_storage import (
    NebiusObjectStorageError,
    StorageMalformedResponse,
)
from bio_neb.storage_contracts import ObjectStorageCredentialReuseReceipt
from bio_neb.verified_artifacts import ArtifactIdentity, ArtifactProof, VerifiedObject

_BAM_NAME = "aligned.bam"
_RECEIPT_NAME = "current-bam-acquisition-receipt.json"


class MaterializeCurrentBamCommand(FrozenContract):
    """The Run whose verified ``aligned_bam`` to acquire, and where.

    The BAM's identity comes only from the Run's verified Execution result.
    """

    execution_journey_file: JourneyFile
    output_directory: StrictStr

    @field_validator("output_directory")
    @classmethod
    def absolute_canonical_path(cls, value: str) -> str:
        if not value or not os.path.isabs(value) or os.path.normpath(value) != value:
            raise ValueError("BAM materialization paths must be absolute and canonical")
        return value


class BamMaterializationRefused(ObjectMaterializationRefused):
    pass


class BamMaterializedFile(MaterializedFile):
    independently_full_read: StrictBool = True


class BamObjectReadResult(ObjectReadResult):
    pass


class BamObjectReadPort(ObjectReadPort, Protocol):
    def download(
        self,
        object_key: str,
        destination: BinaryIO,
        *,
        expected_byte_count: int,
        expected_sha256: str,
        observe_part: PartObserver | None = None,
    ) -> object: ...


class CurrentBamMaterializationLineage(ObjectMaterializationLineage):
    journey_file: BamMaterializedFile
    reference_configuration_id: StrictStr = Field(
        pattern=r"^reference-preparation-configuration:[0-9a-f]{64}$"
    )
    reference_asset_id: StrictStr
    reference_assembly_accession: StrictStr
    reference_fasta_byte_count: StrictInt = Field(gt=0)
    reference_fasta_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class CurrentBamAcquisitionReceipt(FrozenContract):
    """Verified execution-output bytes, without coverage or header assertions.

    The download's streamed digest over every part matched the descriptor; that
    is the ``proof``. The file was never reread (``independently_full_read`` is
    false); the part attempts are the grant's journal rows.
    """

    schema_version: Literal["bio-neb.current-bam-acquisition/2"] = (
        "bio-neb.current-bam-acquisition/2"
    )
    receipt_id: StrictStr = Field(
        pattern=r"^current-bam-acquisition-receipt:[0-9a-f]{64}$"
    )
    lineage: CurrentBamMaterializationLineage
    descriptor: VerifiedObject
    materialized_bam: BamMaterializedFile
    proof: ArtifactProof
    credential_receipt: ObjectStorageCredentialReuseReceipt
    part_bytes: StrictInt = Field(gt=0)
    part_count: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def exact_acquisition(self) -> "CurrentBamAcquisitionReceipt":
        selected = self.lineage.selected_context.topology
        identity = ArtifactIdentity(
            byte_count=self.descriptor.byte_count, sha256=self.descriptor.sha256
        )
        if (
            self.descriptor.role != "aligned_bam"
            or self.materialized_bam.byte_count != self.descriptor.byte_count
            or self.materialized_bam.sha256 != self.descriptor.sha256
            or self.materialized_bam.independently_full_read
            or not self.materialized_bam.read_only
            or self.proof
            != ArtifactProof(
                identity=identity,
                location=self.materialized_bam.path,
                producer="bam_object_get",
                expected_identity=identity,
                source_proof=self.lineage.h100_receipt_id,
            )
            or self.part_count != -(-self.descriptor.byte_count // self.part_bytes)
            or self.credential_receipt.service_account_id != selected.service_account_id
            or self.credential_receipt.access_key_resource_id
            != selected.access_key_resource_id
            or self.credential_receipt.access_key_id != selected.access_key_id
        ):
            raise ValueError("current BAM receipt changed exact acquisition")
        material = self.model_dump(mode="json", exclude={"receipt_id"})
        if (
            self.receipt_id
            != f"current-bam-acquisition-receipt:{canonical_json_digest(material)}"
        ):
            raise ValueError("current BAM acquisition receipt binding drifted")
        return self


class CurrentBamMaterializationResult(FrozenContract):
    receipt: CurrentBamAcquisitionReceipt
    receipt_file: BamMaterializedFile
    materialized_bam: BamMaterializedFile


_files = MaterializationFiles(
    BamMaterializedFile,
    lambda reason, *details: BamMaterializationRefused(
        f"bam_materialization_{reason}", *details
    ),
)


def _execution_lineage(
    outcome: ParabricksExecutionOutcome,
) -> tuple[CurrentBamMaterializationLineage, VerifiedObject]:
    journey_path = Path(outcome.journey_file)
    journey_before, _ = _files.stable_full_read(
        journey_path, require_read_only=False, capture=False
    )
    try:
        retained = read_completed_parabricks_result(outcome.journey_file)
        if retained != outcome:
            raise ValueError("execution result changed")
    except (
        OSError,
        UnicodeError,
        JourneyValidationError,
        ValidationError,
        ValueError,
    ) as error:
        raise BamMaterializationRefused(
            "bam_materialization_execution_invalid"
        ) from error
    journey_after, _ = _files.stable_full_read(
        journey_path, require_read_only=False, capture=False
    )
    if journey_before != journey_after:
        raise BamMaterializationRefused("bam_materialization_execution_unstable")
    descriptors = tuple(
        item for item in outcome.verified_objects if item.role == "aligned_bam"
    )
    if len(descriptors) != 1:
        raise BamMaterializationRefused("bam_materialization_descriptor_invalid")
    descriptor = descriptors[0]
    if descriptor.bucket != outcome.storage.bucket:
        raise BamMaterializationRefused("bam_materialization_descriptor_drift")
    reference = outcome.reference
    return (
        CurrentBamMaterializationLineage(
            journey_id=outcome.journey_id,
            journey_file=journey_after,
            execution_id=outcome.execution_id,
            composition_binding_id=outcome.composition_binding_id,
            authority_id=outcome.authority_id,
            proposal_id=outcome.proposal_id,
            h100_configuration_id=outcome.h100_configuration_id,
            h100_request_id=outcome.h100_request_id,
            h100_receipt_id=outcome.h100_receipt_id,
            h100_checkpoint_event_id=outcome.h100_checkpoint_event_id,
            provider_job_id=outcome.provider_job_id,
            selected_context=outcome.selected_context,
            runtime=outcome.runtime,
            reference_configuration_id=reference.configuration_id,
            reference_asset_id=reference.reference_asset_id,
            reference_assembly_accession=reference.assembly_accession,
            reference_fasta_byte_count=reference.plain_fasta_byte_count,
            reference_fasta_sha256=reference.plain_fasta_sha256,
        ),
        descriptor,
    )


def _publish_receipt(
    receipt: CurrentBamAcquisitionReceipt, path: Path
) -> BamMaterializedFile:
    payload = canonical_json(receipt.model_dump(mode="json")).encode("utf-8")
    with _files.exclusive_output(path) as stream:
        if stream.write(payload) != len(payload):
            raise BamMaterializationRefused("bam_materialization_receipt_write_failed")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o444)
    observed, reread = _files.stable_full_read(
        path, require_read_only=True, capture=True
    )
    if reread != payload:
        raise BamMaterializationRefused("bam_materialization_receipt_replaced")
    return observed


def _acquire_bam(
    packet: Any,
    output_directory: Path,
    descriptor: VerifiedObject,
    object_read_factory: Callable[[Any], BamObjectReadPort] | None,
    observe_part: PartObserver | None,
    observe_credentials: ExecutionContactObserver | None,
) -> tuple[BamMaterializedFile, BamObjectReadResult]:
    _files.fresh_directory(output_directory)
    output_path = output_directory / _BAM_NAME
    source = (
        object_read_factory(packet)
        if object_read_factory is not None
        else execution_object_storage(
            packet,
            BamMaterializationRefused("bam_materialization_context_missing"),
            observe_credentials,
        )
    )
    # Each part retries by the Run's recorded `bam_materialization` policy.
    policy = policy_for(FailureSurface(scope="stage", name="bam_materialization"))
    try:
        with operation_failure_policy(policy):
            raw_read = _files.read_object(output_path, source, descriptor, observe_part)
    except StorageMalformedResponse as error:
        raise BamMaterializationRefused(
            "bam_materialization_object_read_mismatch",
            error.safe_error_category or "malformed",
        ) from error
    except NebiusObjectStorageError as error:
        raise BamMaterializationRefused(
            "bam_materialization_object_read_failed",
            type(error).__name__,
            error.contacts[-1].classification if error.contacts else "failed",
        ) from error
    try:
        read = BamObjectReadResult.model_validate(raw_read, from_attributes=True)
    except ValidationError as error:
        _files.discard(output_path)
        raise BamMaterializationRefused(
            "bam_materialization_object_read_mismatch"
        ) from error
    if (read.byte_count, read.sha256) != (descriptor.byte_count, descriptor.sha256):
        _files.discard(output_path)
        raise BamMaterializationRefused("bam_materialization_object_read_mismatch")
    # The transfer proved the bytes; sealing admits the replica, with no reread.
    _files.seal_replica(output_path)
    materialized = BamMaterializedFile(
        path=str(output_path),
        byte_count=read.byte_count,
        sha256=read.sha256,
        independently_full_read=False,
        read_only=True,
    )
    return materialized, read


def materialize_current_bam(
    command: MaterializeCurrentBamCommand,
    *,
    object_read_factory: Callable[[Any], BamObjectReadPort] | None = None,
    observe_part: PartObserver | None = None,
    observe_credentials: ExecutionContactObserver | None = None,
) -> CurrentBamMaterializationResult:
    """Acquire the Run's verified BAM by its recorded identity; grants no coverage.

    ``observe_part`` sees the part limits, then each part attempt start before
    its request and end before any retry; ``observe_credentials`` sees the
    context's credential lookups.
    """
    selected = MaterializeCurrentBamCommand.model_validate(command, strict=True)
    output_directory = Path(selected.output_directory)
    if output_directory.exists() or output_directory.is_symlink():
        raise BamMaterializationRefused("bam_materialization_output_exists")
    try:
        packet = read_completed_parabricks_result(selected.execution_journey_file)
    except ParabricksExecutionIncomplete as error:
        raise BamMaterializationRefused(
            "bam_materialization_execution_incomplete"
        ) from error
    except (OSError, UnicodeError, JourneyValidationError, ValueError) as error:
        raise BamMaterializationRefused(
            "bam_materialization_execution_invalid"
        ) from error
    if not isinstance(packet, ParabricksExecutionOutcome):
        raise BamMaterializationRefused("bam_materialization_execution_incomplete")
    lineage, descriptor = _execution_lineage(packet)
    materialized, read = _acquire_bam(
        packet,
        output_directory,
        descriptor,
        object_read_factory,
        observe_part,
        observe_credentials,
    )
    identity = ArtifactIdentity(
        byte_count=descriptor.byte_count, sha256=descriptor.sha256
    )
    proof = ArtifactProof(
        identity=identity,
        location=materialized.path,
        producer="bam_object_get",
        expected_identity=identity,
        source_proof=lineage.h100_receipt_id,
    )
    material = {
        "schema_version": "bio-neb.current-bam-acquisition/2",
        "lineage": lineage.model_dump(mode="json"),
        "descriptor": descriptor.model_dump(mode="json"),
        "materialized_bam": materialized.model_dump(mode="json"),
        "proof": proof.model_dump(mode="json"),
        "credential_receipt": read.credential_receipt.model_dump(mode="json"),
        "part_bytes": read.part_bytes,
        "part_count": read.part_count,
    }
    receipt = CurrentBamAcquisitionReceipt.model_validate(
        {
            **material,
            "receipt_id": f"current-bam-acquisition-receipt:{canonical_json_digest(material)}",
        }
    )
    receipt_file = _publish_receipt(receipt, output_directory / _RECEIPT_NAME)
    return CurrentBamMaterializationResult(
        receipt=receipt,
        receipt_file=receipt_file,
        materialized_bam=materialized,
    )


__all__ = [
    "CurrentBamAcquisitionReceipt",
    "CurrentBamMaterializationLineage",
    "CurrentBamMaterializationResult",
    "MaterializeCurrentBamCommand",
    "materialize_current_bam",
    "BamMaterializationRefused",
    "BamMaterializedFile",
    "BamObjectReadPort",
    "BamObjectReadResult",
]
