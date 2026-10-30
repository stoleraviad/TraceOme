"""Acquire one verified current-Run VCF without granting observation authority."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
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
from bio_neb.application.local_artifact_proof import ArtifactVerificationCheck
from bio_neb.application.verification_policy import (
    ArtifactVerificationDefaults,
    ArtifactVerificationOverrides,
    ResolvedArtifactVerificationPolicy,
    resolve_artifact_verification_policy,
)
from bio_neb.contracts import FrozenContract
from bio_neb.failure import FailureSurface, operation_failure_policy, policy_for
from bio_neb.investigation.vcf_observation import AttestedPlainVcf
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

_VCF_NAME = "variants.vcf"
_RECEIPT_NAME = "vcf-materialization-receipt.json"


class MaterializeCurrentVcfCommand(FrozenContract):
    execution_journey_file: JourneyFile
    output_directory: StrictStr
    candidate_file: StrictStr | None = None
    candidate_receipt_file: StrictStr | None = None
    verification_defaults: ArtifactVerificationDefaults = Field(
        default_factory=ArtifactVerificationDefaults
    )
    verification_overrides: ArtifactVerificationOverrides = Field(
        default_factory=ArtifactVerificationOverrides
    )

    @field_validator(
        "output_directory",
        "candidate_file",
        "candidate_receipt_file",
    )
    @classmethod
    def absolute_canonical_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or not os.path.isabs(value) or os.path.normpath(value) != value:
            raise ValueError("VCF materialization paths must be absolute and canonical")
        return value


class VcfMaterializationRefused(ObjectMaterializationRefused):
    pass


class VcfObjectReadResult(ObjectReadResult):
    pass


class VcfObjectReadPort(ObjectReadPort, Protocol):
    def download(
        self,
        object_key: str,
        destination: BinaryIO,
        *,
        expected_byte_count: int,
        expected_sha256: str,
        observe_part: PartObserver | None = None,
    ) -> object: ...


class VcfMaterializationFile(MaterializedFile):
    independently_full_read: StrictBool = True


class VcfMaterializationLineage(ObjectMaterializationLineage):
    journey_file: VcfMaterializationFile


def _receipt_id(material: Mapping[str, Any]) -> str:
    digest = canonical_json_digest(dict(material))
    return f"vcf-materialization-receipt:{digest}"


class VcfMaterializationReceipt(FrozenContract):
    """One replica's identity, transfer and lineage, established once (0029).

    The streamed GET or copy establishes the replica's identity; no destination
    reread follows unless ``independent_full_read`` is selected, and then its one
    full read is the only retained check.
    """

    schema_version: Literal["bio-neb.vcf-materialization/2"] = (
        "bio-neb.vcf-materialization/2"
    )
    receipt_id: StrictStr = Field(pattern=r"^vcf-materialization-receipt:[0-9a-f]{64}$")
    lineage: VcfMaterializationLineage
    descriptor: VerifiedObject
    materialized_vcf: VcfMaterializationFile
    proof: ArtifactProof
    source_candidate: VcfMaterializationFile | None = None
    source_candidate_receipt: VcfMaterializationFile | None = None
    source_candidate_receipt_id: StrictStr | None = Field(
        default=None,
        pattern=r"^vcf-materialization-receipt:[0-9a-f]{64}$",
    )
    credential_receipt: ObjectStorageCredentialReuseReceipt | None = None
    verification_policy: ResolvedArtifactVerificationPolicy
    verification_checks: tuple[ArtifactVerificationCheck, ...] = ()

    @model_validator(mode="after")
    def exact_receipt(self) -> "VcfMaterializationReceipt":
        identity = ArtifactIdentity(
            byte_count=self.descriptor.byte_count, sha256=self.descriptor.sha256
        )
        copied = self.proof.producer == "vcf_local_copy"
        if (
            self.proof.identity != identity
            or self.proof.expected_identity != identity
            or self.proof.location != self.materialized_vcf.path
            or self.proof.producer not in {"vcf_object_get", "vcf_local_copy"}
            or self.proof.source_proof
            != (
                self.source_candidate_receipt_id
                if copied
                else self.lineage.h100_receipt_id
            )
        ):
            raise ValueError("VCF replica proof changed acquisition binding")
        audited = self.verification_policy.effective.mode == "independent_full_read"
        if any(
            check.policy != self.verification_policy
            or check.reason != "policy_full_hash"
            for check in self.verification_checks
        ) or (
            audited
            and (
                not self.verification_checks
                or self.verification_checks[-1].path != self.materialized_vcf.path
                or self.verification_checks[-1].bytes_hashed
                != self.materialized_vcf.byte_count
            )
        ):
            raise ValueError("VCF verification checks changed acquisition")
        candidate_evidence = (
            self.source_candidate,
            self.source_candidate_receipt,
            self.source_candidate_receipt_id,
        )
        selected = self.lineage.selected_context.topology
        credential_matches_context = self.credential_receipt is not None and (
            self.credential_receipt.service_account_id == selected.service_account_id
            and self.credential_receipt.access_key_resource_id
            == selected.access_key_resource_id
            and self.credential_receipt.access_key_id == selected.access_key_id
        )
        if (
            self.descriptor.role != "VCF"
            or self.materialized_vcf.byte_count != self.descriptor.byte_count
            or self.materialized_vcf.sha256 != self.descriptor.sha256
            or not self.materialized_vcf.read_only
            or self.materialized_vcf.independently_full_read != audited
            or len({item is None for item in candidate_evidence}) != 1
            or copied != (candidate_evidence[0] is not None)
            or (copied and self.credential_receipt is not None)
            or (not copied and not credential_matches_context)
        ):
            raise ValueError("VCF materialization receipt changed exact acquisition")
        material = self.model_dump(mode="json", exclude={"receipt_id"})
        if self.receipt_id != _receipt_id(material):
            raise ValueError("VCF materialization receipt binding drifted")
        return self

    @classmethod
    def bind(cls, **values: Any) -> "VcfMaterializationReceipt":
        material = {
            "schema_version": "bio-neb.vcf-materialization/2",
            "source_candidate": None,
            "source_candidate_receipt": None,
            "source_candidate_receipt_id": None,
            "credential_receipt": None,
            "verification_checks": [],
            **{
                key: (
                    value.model_dump(mode="json")
                    if isinstance(value, BaseModel)
                    else value
                )
                for key, value in values.items()
            },
        }
        return cls.model_validate({**material, "receipt_id": _receipt_id(material)})


class VcfMaterializationResult(FrozenContract):
    receipt: VcfMaterializationReceipt
    receipt_file: VcfMaterializationFile
    attested_vcf: AttestedPlainVcf

    @model_validator(mode="after")
    def exact_attestation(self) -> "VcfMaterializationResult":
        if (
            self.attested_vcf.local_path != self.receipt.materialized_vcf.path
            or self.attested_vcf.byte_count != self.receipt.materialized_vcf.byte_count
            or self.attested_vcf.sha256 != self.receipt.materialized_vcf.sha256
            or self.attested_vcf.acquisition_receipt_path != self.receipt_file.path
            or self.attested_vcf.acquisition_receipt_sha256 != self.receipt_file.sha256
            or not self.receipt_file.read_only
        ):
            raise ValueError("VCF materialization result changed attested binding")
        return self


_files = MaterializationFiles(
    VcfMaterializationFile,
    lambda reason, *details: VcfMaterializationRefused(
        f"vcf_materialization_{reason}", *details
    ),
)


def _canonical_receipt(
    path: Path,
) -> tuple[VcfMaterializationFile, VcfMaterializationReceipt]:
    observed, payload = _files.stable_full_read(
        path, require_read_only=True, capture=True
    )
    assert payload is not None

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError("duplicate JSON member")
            document[key] = value
        return document

    try:
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object)
        if canonical_json(document).encode("utf-8") != payload:
            raise ValueError("receipt is not canonical")
        receipt = VcfMaterializationReceipt.model_validate(document)
    except (UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as error:
        raise VcfMaterializationRefused(
            "vcf_materialization_candidate_receipt_invalid"
        ) from error
    return observed, receipt


def _lineage(
    outcome: ParabricksExecutionOutcome,
) -> tuple[VcfMaterializationLineage, VerifiedObject]:
    if not isinstance(outcome, ParabricksExecutionOutcome):
        raise VcfMaterializationRefused("vcf_materialization_execution_incomplete")
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
        raise VcfMaterializationRefused(
            "vcf_materialization_execution_invalid"
        ) from error
    journey_after, _ = _files.stable_full_read(
        journey_path, require_read_only=False, capture=False
    )
    if journey_before != journey_after:
        raise VcfMaterializationRefused("vcf_materialization_execution_unstable")
    descriptors = tuple(item for item in outcome.verified_objects if item.role == "VCF")
    if len(descriptors) != 1:
        raise VcfMaterializationRefused("vcf_materialization_descriptor_invalid")
    descriptor = descriptors[0]
    if descriptor.bucket != outcome.storage.bucket:
        raise VcfMaterializationRefused("vcf_materialization_descriptor_drift")
    return (
        VcfMaterializationLineage(
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
        ),
        descriptor,
    )


def _publish_receipt(
    receipt: VcfMaterializationReceipt, path: Path
) -> VcfMaterializationFile:
    payload = canonical_json(receipt.model_dump(mode="json")).encode("utf-8")
    with _files.exclusive_output(path) as stream:
        if stream.write(payload) != len(payload):
            raise VcfMaterializationRefused("vcf_materialization_receipt_write_failed")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o444)
    observed, reread = _files.stable_full_read(
        path, require_read_only=True, capture=True
    )
    if reread != payload:
        raise VcfMaterializationRefused("vcf_materialization_receipt_replaced")
    return observed


@dataclass(frozen=True)
class LocalVcfReplicaCopy:
    """Byte mechanics only: no acquisition, current-Run or scientific authority."""

    source_candidate: VcfMaterializationFile
    materialized_vcf: VcfMaterializationFile
    proof: ArtifactProof
    verification_checks: tuple[ArtifactVerificationCheck, ...]


def _seal_replica(
    path: Path,
    descriptor: VerifiedObject,
    *,
    transferred: VcfMaterializationFile,
    policy: ResolvedArtifactVerificationPolicy,
) -> tuple[VcfMaterializationFile, tuple[ArtifactVerificationCheck, ...]]:
    """Admit a fresh replica once; only an explicit selection rereads it."""
    _files.seal_replica(path)
    if (transferred.byte_count, transferred.sha256) != (
        descriptor.byte_count,
        descriptor.sha256,
    ):
        raise VcfMaterializationRefused("vcf_materialization_output_identity_mismatch")
    if policy.effective.mode != "independent_full_read":
        return transferred.model_copy(
            update={"independently_full_read": False, "read_only": True}
        ), ()
    started = time.monotonic()
    audited, _ = _files.stable_full_read(path, require_read_only=True)
    if (audited.byte_count, audited.sha256) != (
        descriptor.byte_count,
        descriptor.sha256,
    ):
        raise VcfMaterializationRefused("vcf_materialization_output_identity_mismatch")
    return audited, (
        ArtifactVerificationCheck(
            policy=policy,
            path=audited.path,
            reason="policy_full_hash",
            bytes_read=audited.byte_count,
            bytes_hashed=audited.byte_count,
            elapsed_seconds=time.monotonic() - started,
        ),
    )


def copy_verified_local_vcf(
    *,
    source_path: str,
    output_directory: str,
    descriptor: VerifiedObject,
    source_receipt_id: str,
    source_proof: ArtifactProof,
    verification_policy: ResolvedArtifactVerificationPolicy | None = None,
) -> LocalVcfReplicaCopy:
    """Copy one proved local VCF into a fresh replica.

    The caller owns descriptor eligibility and receipt provenance. This operation
    has no provider or Journey port and cannot establish current action authority.
    Proved bytes are copied as trusted (0029); an explicit ``independent_full_read``
    hashes the copy stream and rereads the replica.
    """
    candidate_path, destination = Path(source_path), Path(output_directory)
    if descriptor.role != "VCF":
        raise VcfMaterializationRefused("vcf_materialization_descriptor_invalid")
    for path in (candidate_path, destination):
        if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
            raise VcfMaterializationRefused(
                "vcf_materialization_file_invalid", str(path)
            )
    if destination.exists() or destination.is_symlink():
        raise VcfMaterializationRefused("vcf_materialization_output_exists")
    policy = verification_policy or resolve_artifact_verification_policy(
        ArtifactVerificationDefaults(),
        ArtifactVerificationOverrides(),
    )
    identity = ArtifactIdentity(
        byte_count=descriptor.byte_count, sha256=descriptor.sha256
    )
    if source_proof.identity != identity or source_proof.location != source_path:
        raise VcfMaterializationRefused("vcf_materialization_proof_drift")
    trusted = policy.effective.mode != "independent_full_read"
    output_path = destination / _VCF_NAME
    started = time.monotonic()
    if trusted:
        # Proved bytes: one open and one stream; no stat, hash or custody check
        # (0029). Its creation sealed the source read-only, as its receipt records.
        with _files.open_source(candidate_path) as source:
            _files.fresh_directory(destination)
            with _files.exclusive_output(output_path) as output:
                copied = _files.copy_stream(source, output)
                output.flush()
                os.fsync(output.fileno())
        candidate = VcfMaterializationFile(
            path=source_path,
            byte_count=copied,
            sha256=descriptor.sha256,
            independently_full_read=False,
            read_only=True,
        )
    else:
        # The selected audit reads the source in full, read-only and stable,
        # hashing the copy stream. A refused source leaves no partial replica.
        _files.fresh_directory(destination)
        try:
            with _files.exclusive_output(output_path) as output:
                candidate, _ = _files.stable_full_read(
                    candidate_path, require_read_only=True, destination=output
                )
                output.flush()
                os.fsync(output.fileno())
        except VcfMaterializationRefused:
            output_path.unlink(missing_ok=True)
            raise
    if (candidate.byte_count, candidate.sha256) != (
        descriptor.byte_count,
        descriptor.sha256,
    ):
        raise VcfMaterializationRefused(
            "vcf_materialization_candidate_identity_mismatch"
        )
    checks = (
        ()
        if trusted
        else (
            ArtifactVerificationCheck(
                policy=policy,
                path=source_path,
                reason="policy_full_hash",
                bytes_read=candidate.byte_count,
                bytes_hashed=candidate.byte_count,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
    )
    materialized, audit = _seal_replica(
        output_path,
        descriptor,
        transferred=candidate.model_copy(update={"path": str(output_path)}),
        policy=policy,
    )
    proof = ArtifactProof(
        identity=identity,
        location=materialized.path,
        producer="vcf_local_copy",
        expected_identity=identity,
        source_proof=source_receipt_id,
    )
    return LocalVcfReplicaCopy(candidate, materialized, proof, (*checks, *audit))


def _object_get(
    output_directory: Path,
    source: VcfObjectReadPort,
    descriptor: VerifiedObject,
    observe_part: PartObserver | None,
) -> VcfObjectReadResult:
    """One download in ranged parts, each retried by ``vcf_materialization``.

    A part is retried only for a transient fault; a failed download leaves no
    replica, so nothing partial is ever reused (#363).
    """
    output_path = output_directory / _VCF_NAME
    policy = policy_for(FailureSurface(scope="stage", name="vcf_materialization"))
    try:
        with operation_failure_policy(policy):
            raw_read = _files.read_object(output_path, source, descriptor, observe_part)
    except StorageMalformedResponse as error:
        raise VcfMaterializationRefused(
            "vcf_materialization_object_read_mismatch",
            error.safe_error_category or "malformed",
        ) from error
    except NebiusObjectStorageError as error:
        raise VcfMaterializationRefused(
            "vcf_materialization_object_read_failed",
            type(error).__name__,
            error.contacts[-1].classification if error.contacts else "failed",
        ) from error
    try:
        validated = VcfObjectReadResult.model_validate(raw_read, from_attributes=True)
    except ValidationError as error:
        _files.discard(output_path)
        raise VcfMaterializationRefused(
            "vcf_materialization_object_read_mismatch"
        ) from error
    if (validated.byte_count, validated.sha256) != (
        descriptor.byte_count,
        descriptor.sha256,
    ):
        _files.discard(output_path)
        raise VcfMaterializationRefused("vcf_materialization_object_read_mismatch")
    return validated


def materialize_current_vcf(
    command: MaterializeCurrentVcfCommand,
    *,
    object_read_factory: Callable[[Any], VcfObjectReadPort] | None = None,
    observe_part: PartObserver | None = None,
    observe_credentials: ExecutionContactObserver | None = None,
) -> VcfMaterializationResult:
    """Materialize only the exact VCF of one terminal execution result.

    The GET uses the Execution's own bucket and context unless a test supplies
    ``object_read_factory``. ``observe_part`` sees the part limits, then each
    part attempt start before its request and end before any retry;
    ``observe_credentials`` sees the context's credential lookups.
    """
    candidates = (command.candidate_file, command.candidate_receipt_file)
    if (candidates[0] is None) != (candidates[1] is None):
        raise VcfMaterializationRefused("vcf_materialization_candidate_unpaired")
    output_directory = Path(command.output_directory)
    if output_directory.exists() or output_directory.is_symlink():
        raise VcfMaterializationRefused("vcf_materialization_output_exists")
    try:
        packet = read_completed_parabricks_result(command.execution_journey_file)
    except ParabricksExecutionIncomplete as error:
        raise VcfMaterializationRefused(
            "vcf_materialization_execution_incomplete"
        ) from error
    except (OSError, UnicodeError, JourneyValidationError, ValueError) as error:
        raise VcfMaterializationRefused(
            "vcf_materialization_execution_invalid"
        ) from error
    lineage, descriptor = _lineage(packet)
    policy = resolve_artifact_verification_policy(
        command.verification_defaults, command.verification_overrides
    )
    identity = ArtifactIdentity(
        byte_count=descriptor.byte_count, sha256=descriptor.sha256
    )
    candidate_file = None
    candidate_receipt_file = None
    candidate_receipt_id = None
    credential_receipt = None
    if candidates[0] is not None and candidates[1] is not None:
        candidate_path = Path(candidates[0])
        receipt_path = Path(candidates[1])
        if candidate_path == receipt_path:
            raise VcfMaterializationRefused("vcf_materialization_candidate_collision")
        candidate_receipt_file, candidate_receipt = _canonical_receipt(receipt_path)
        candidate_receipt_id = candidate_receipt.receipt_id
        if (
            candidate_receipt.materialized_vcf.path != str(candidate_path)
            or candidate_receipt.descriptor != descriptor
        ):
            if not candidate_path.exists():
                raise VcfMaterializationRefused(
                    "vcf_materialization_file_unavailable", str(candidate_path)
                )
            raise VcfMaterializationRefused(
                "vcf_materialization_candidate_identity_mismatch"
            )
        copied = copy_verified_local_vcf(
            source_path=str(candidate_path),
            output_directory=str(output_directory),
            descriptor=descriptor,
            source_receipt_id=candidate_receipt_id,
            source_proof=candidate_receipt.proof,
            verification_policy=policy,
        )
        candidate_file, materialized, proof, checks = (
            copied.source_candidate,
            copied.materialized_vcf,
            copied.proof,
            copied.verification_checks,
        )
    else:
        _files.fresh_directory(output_directory)
        source = (
            object_read_factory(packet)
            if object_read_factory is not None
            else execution_object_storage(
                packet,
                VcfMaterializationRefused("vcf_materialization_context_missing"),
                observe_credentials,
            )
        )
        read = _object_get(
            output_directory,
            source,
            descriptor,
            observe_part,
        )
        credential_receipt = read.credential_receipt
        materialized, checks = _seal_replica(
            output_directory / _VCF_NAME,
            descriptor,
            transferred=VcfMaterializationFile(
                path=str(output_directory / _VCF_NAME),
                byte_count=read.byte_count,
                sha256=read.sha256,
                independently_full_read=False,
                read_only=True,
            ),
            policy=policy,
        )
        proof = ArtifactProof(
            identity=identity,
            location=materialized.path,
            producer="vcf_object_get",
            expected_identity=identity,
            source_proof=lineage.h100_receipt_id,
        )
    receipt = VcfMaterializationReceipt.bind(
        lineage=lineage,
        descriptor=descriptor,
        materialized_vcf=materialized,
        proof=proof,
        source_candidate=candidate_file,
        source_candidate_receipt=candidate_receipt_file,
        source_candidate_receipt_id=candidate_receipt_id,
        credential_receipt=credential_receipt,
        verification_policy=policy,
        verification_checks=tuple(check.model_dump(mode="json") for check in checks),
    )
    receipt_path = output_directory / _RECEIPT_NAME
    receipt_file = _publish_receipt(receipt, receipt_path)
    return VcfMaterializationResult(
        receipt=receipt,
        receipt_file=receipt_file,
        attested_vcf=AttestedPlainVcf(
            local_path=materialized.path,
            byte_count=materialized.byte_count,
            sha256=materialized.sha256,
            acquisition_receipt_path=receipt_file.path,
            acquisition_receipt_sha256=receipt_file.sha256,
        ),
    )


__all__ = [
    "copy_verified_local_vcf",
    "LocalVcfReplicaCopy",
    "MaterializedFile",
    "MaterializeCurrentVcfCommand",
    "ObjectMaterializationLineage",
    "ObjectMaterializationRefused",
    "ObjectReadPort",
    "ObjectReadResult",
    "VcfMaterializationFile",
    "VcfMaterializationLineage",
    "VcfMaterializationReceipt",
    "VcfMaterializationRefused",
    "VcfMaterializationResult",
    "VcfObjectReadPort",
    "materialize_current_vcf",
]
