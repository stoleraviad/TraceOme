"""Independently readable execution results; reading never constructs execution dependencies."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from bio_neb.content import ContentId
from bio_neb.contracts import FrozenContract
from bio_neb.failure import Ask
from bio_neb.journey import Journey, canonical_json_digest
from bio_neb.journey_path import (
    ArtifactPointer,
    ExecutionState,
    JourneyPath,
    Pending,
    StageFailure,
    artifact_location,
    has_execution_path,
)
from bio_neb.parabricks.execution_runtime import ExecutionCompositionRuntime
from bio_neb.parabricks.h100_recipe import configuration_id as h100_configuration_id
from bio_neb.parabricks.reference_recipe import (
    candidate_compatibility,
)
from bio_neb.parabricks.reference_recipe import (
    configuration_id as reference_configuration_id,
)
from bio_neb.providers.nebius_context import NebiusExecutionContextSelection
from bio_neb.providers.object_locations import object_address, object_location
from bio_neb.providers.runtime import RunObjectStorageTarget
from bio_neb.verified_artifacts import VerifiedObject


class ParabricksResultUnavailable(ValueError):
    """No execution has been bound yet; other reader refusals are not absence."""


class ParabricksExecutionIncomplete(ValueError):
    """A valid execution has not produced a completed result."""


class ParabricksReferenceResult(FrozenContract):
    configuration_id: StrictStr
    reference_asset_id: StrictStr
    assembly_accession: StrictStr
    plain_fasta_byte_count: StrictInt
    plain_fasta_sha256: StrictStr


class ParabricksExecutionOutcome(FrozenContract):
    """Compact catalog-ready result of one approved execution's Journey truth."""

    schema_version: Literal["bio-neb.parabricks-execution-outcome/1"] = (
        "bio-neb.parabricks-execution-outcome/1"
    )
    journey_id: StrictStr
    journey_file: StrictStr
    run_root: StrictStr
    execution_id: StrictStr
    execution_package_id: StrictStr
    composition_binding_id: StrictStr
    selected_context: NebiusExecutionContextSelection
    runtime: ExecutionCompositionRuntime
    storage: RunObjectStorageTarget
    reference: ParabricksReferenceResult
    h100_configuration_id: StrictStr
    h100_checkpoint_event_id: StrictStr
    h100_request_id: StrictStr
    provider_job_id: StrictStr
    execution_evidence_event_id: StrictStr
    execution_evidence_sha256: StrictStr
    # The pass this outcome closed: the first pass keeps the bound accounting
    # locators; a continuation pass names its own ``pass-<n>`` ones.
    execution_pass: StrictInt = Field(ge=1, default=1)
    accounting_journal_file: StrictStr
    accounting_ledger_file: StrictStr
    authority_id: StrictStr
    proposal_id: StrictStr
    disposition: Literal["succeeded"] = "succeeded"
    status: Literal["execution_artifacts_verified"] = "execution_artifacts_verified"
    cache_outcome: Literal["hit", "miss"]
    prepared_reference_source: Literal["cache_hit", "cpu_built"]
    prepared_reference_receipt_id: StrictStr
    h100_receipt_id: StrictStr
    verified_objects: tuple[VerifiedObject, ...] = ()
    output_proofs: tuple[ArtifactPointer, ...]
    input_proofs: tuple[ArtifactPointer, ...]
    reference_proofs: tuple[ArtifactPointer, ...]
    object_scope: StrictStr
    outcome_event_id: StrictStr

    @field_validator("journey_file", "run_root")
    @classmethod
    def absolute_canonical_paths(cls, value: str) -> str:
        if not value or not os.path.isabs(value) or os.path.normpath(value) != value:
            raise ValueError("outcome paths must be absolute and canonical")
        return value

    @model_validator(mode="after")
    def verified_inventory(self) -> ParabricksExecutionOutcome:
        roles = ("aligned_bam", "VCF", "worker_receipt", "execution_manifest")
        if (
            tuple(item.role for item in self.verified_objects) != roles
            or tuple(item.role for item in self.output_proofs) != roles
        ):
            raise ValueError("Parabricks result lacks its exact verified inventory")
        for item, proof in zip(self.verified_objects, self.output_proofs, strict=True):
            if (
                ContentId.parse(proof.content_id).digest != item.sha256
                or proof.bytes != item.byte_count
                or proof.producer.authority_id != self.authority_id
                or proof.grounding != "verified"
                or item.bucket != self.storage.bucket
            ):
                raise ValueError("Parabricks result changed its artifact proof")
        # Each producer's exact retained roles; the reference manifest is the
        # document whose identity the prepared-reference receipt names.
        reference_stage, reference_manifest = (
            ("admit_candidate", "prepared_reference_admission")
            if self.cache_outcome == "hit"
            else ("cpu_preparation", "prepared_reference_manifest")
        )
        # The source is placed by input storage, or fetched by the CPU job.
        if [
            proof.role
            for proof in (*self.input_proofs, *self.reference_proofs)
            if proof.role == "reference_source"
        ] != ["reference_source"]:
            raise ValueError("Parabricks result changed its input or reference proof")
        for stage, proofs, required in (
            ("input_storage", self.input_proofs, {"input_manifest"}),
            (reference_stage, self.reference_proofs, {reference_manifest}),
        ):
            roles = [proof.role for proof in proofs]
            if (
                len(roles) != len(set(roles))
                or not required <= set(roles)
                or any(
                    proof.producer.stage != stage
                    or proof.producer.authority_id != self.authority_id
                    or proof.grounding != "verified"
                    or not self._run_location(proof)
                    for proof in proofs
                )
            ):
                raise ValueError(
                    "Parabricks result changed its input or reference proof"
                )
        if not any(
            proof.role == reference_manifest
            and proof.content_id == self.prepared_reference_receipt_id
            for proof in self.reference_proofs
        ):
            raise ValueError("Parabricks result changed its input or reference proof")
        return self

    def _run_location(self, proof: ArtifactPointer) -> bool:
        """A proof names a retained bundle document or an object in the Run bucket."""
        if proof.location.startswith("artifacts/"):
            return proof.location == artifact_location(proof.content_id)
        try:
            bucket, _ = object_address(proof.location)
        except ValueError:
            return False
        return bucket == self.storage.bucket


class ParabricksNoncompletion(FrozenContract):
    """A retained stop or unfinished operation, never a verified output inventory."""

    journey_id: StrictStr
    journey_file: StrictStr
    execution_id: StrictStr
    authority_id: StrictStr
    proposal_id: StrictStr
    storage: RunObjectStorageTarget
    object_scope: StrictStr
    disposition: Literal["incomplete", "failed"]
    status: StrictStr
    execution_pass: StrictInt = Field(ge=1)
    outcome_event_id: StrictStr | None
    milestone: Literal["input_storage", "reference_preparation"] | None = None
    pending_review: Ask | Pending | None = None
    request_event_id: StrictStr | None = None
    failures: tuple[StageFailure, ...] = ()
    failure_event_ids: tuple[StrictStr, ...] = ()


ParabricksResult = ParabricksExecutionOutcome | ParabricksNoncompletion


def parabricks_result(journey: JourneyPath) -> ParabricksResult:
    """Interpret one checked Journey snapshot at the operation's public boundary."""
    state = journey.state
    if state.bound is None:
        raise ParabricksResultUnavailable("parabricks_result_unavailable")
    if state.disposition == "succeeded":
        return _completed_result(journey, state)
    bound = state.bound
    pending = (
        state.failure_review
        if isinstance(state.failure_review, Ask)
        else state.review
        if isinstance(state.review, Pending)
        else None
    )
    request_event_id = (
        state.latest_event_ids["FailureReviewRequested"]
        if isinstance(pending, Ask)
        else state.latest_event_ids["CpuPreparationReviewRequested"]
        if pending
        else None
    )
    failures = tuple(
        failure
        for event_id, failure in zip(
            state.failure_event_ids, state.failures, strict=True
        )
        if event_id in state.unresolved_failure_ids
    )
    return ParabricksNoncompletion(
        journey_id=bound.run_id,
        journey_file=str(journey.file),
        execution_id=bound.execution_id,
        authority_id=bound.authority_id,
        proposal_id=bound.proposal_id,
        storage=bound.approved_input["launch"]["object_storage"],
        object_scope=bound.approved_input["launch"]["object_scope"],
        disposition=state.disposition,
        status=state.status,
        execution_pass=state.pass_,
        outcome_event_id=state.latest_event_ids.get("ExecutionOutcome")
        if state.outcome
        else None,
        milestone=state.outcome.milestone if state.outcome else None,
        pending_review=pending,
        request_event_id=request_event_id,
        failures=failures,
        failure_event_ids=state.unresolved_failure_ids,
    )


def execution_result_context(plan, state) -> dict:
    """Freeze consumer metadata from the effective plan beside verified artifacts."""
    approved = state.bound.approved_input
    package = approved["proposal"]["execution_package"]
    reference = plan.reference
    # The FASTA the H100 job consumed: built by the CPU job or admitted with a
    # matching supplied index; both are acquired identities.
    (fasta,) = (p for p in state.reference.pointers if p.role == "genomic_fasta")
    compatibility = candidate_compatibility(plan)
    runtime = plan.runtimes.h100
    selection = approved["proposal"].get("runtime_selection")
    runtime_value = ExecutionCompositionRuntime(
        **{
            key: getattr(runtime, key)
            for key in (
                "provider_image_reference",
                "provider_image_index_digest",
                "worker_runtime_image",
                "worker_sha256",
                "python_path",
                "python_version",
                "pbrun_path",
                "pbrun_version",
            )
        },
        nebius_cli_version=plan.scope.cli_version,
        worker_request_schema=compatibility.worker_request_schema,
        bwa_path=compatibility.bwa_path,
        bwa_version=compatibility.bwa_version,
        samtools_path=compatibility.samtools_path,
        samtools_version=compatibility.samtools_version,
    )
    if selection is not None:
        runtime_value = runtime_value.model_copy(
            update={
                "receipt_sha256": None,
                "capability_snapshot_id": selection["capability"]["snapshot_id"],
            }
        )
    return dict(
        journey_id=plan.identity.journey_id,
        journey_file=approved["journey_file"],
        run_root=str(Path(approved["journey_file"]).parent.parent),
        execution_id=plan.identity.execution_id,
        execution_package_id=package["package_id"],
        authority_id=plan.identity.authority_id,
        proposal_id=plan.identity.proposal_id,
        composition_binding_id=state.latest_event_ids["ExecutionBound"],
        selected_context=plan.launch.context.model_dump(mode="json"),
        runtime=runtime_value.model_dump(mode="json"),
        storage=plan.storage.model_dump(mode="json"),
        reference=ParabricksReferenceResult(
            configuration_id=reference_configuration_id(plan),
            reference_asset_id=reference.reference_asset_id,
            assembly_accession=reference.assembly_accession,
            plain_fasta_byte_count=fasta.bytes,
            plain_fasta_sha256=ContentId.parse(fasta.content_id).digest,
        ).model_dump(mode="json"),
        h100_configuration_id=h100_configuration_id(plan),
    )


def _completed_result(
    journey: JourneyPath, state: ExecutionState
) -> ParabricksExecutionOutcome:
    """Project the successful state and its bound documents for actual consumers."""
    from bio_neb.journey_path import Hit

    if state.disposition != "succeeded":
        raise ParabricksExecutionIncomplete("parabricks_execution_incomplete")
    pointers = {p.role: p for p in state.h100.pointers}
    if len(pointers) != len(state.h100.pointers):
        raise ValueError("duplicate_parabricks_output_proof")
    manifest_pointer = pointers["execution_manifest"]
    manifest = json.loads(journey.read_artifact(manifest_pointer))
    context = json.loads(journey.read_artifact(pointers["execution_result_context"]))
    approved = state.bound.approved_input
    if (
        context["execution_id"] != state.bound.execution_id
        or context["authority_id"] != state.bound.authority_id
        or context["proposal_id"] != state.bound.proposal_id
        or not journey.recorded_in(context["journey_file"])
        or state.bound.execution_id
        != "parabricks-execution:" + canonical_json_digest(approved)
        or manifest["prepared_reference_receipt_id"] != state.reference.receipt_id
        or manifest["configuration_id"] != context["h100_configuration_id"]
        or state.h100.receipt_id != manifest_pointer.content_id
    ):
        raise ValueError("parabricks_result_identity_drift")
    objects = [
        VerifiedObject.model_validate(item)
        for item in (*manifest["outputs"], manifest["worker_receipt"])
    ]
    remote = pointers["execution_manifest_object"]
    bucket, key = object_address(remote.location)
    if (
        bucket != context["storage"]["bucket"]
        or remote.content_id != manifest_pointer.content_id
        or remote.bytes != manifest_pointer.bytes
    ):
        raise ValueError("parabricks_result_manifest_drift")
    objects.append(
        VerifiedObject(
            role="execution_manifest",
            bucket=bucket,
            object_key=key,
            byte_count=remote.bytes,
            sha256=ContentId.parse(remote.content_id).digest,
        )
    )
    for item in objects[:2]:
        pointer = pointers[item.role]
        if (
            pointer.location != object_location(item.bucket, item.object_key)
            or pointer.bytes != item.byte_count
            or ContentId.parse(pointer.content_id).digest != item.sha256
        ):
            raise ValueError("parabricks_result_descriptor_drift")
    worker = pointers["worker_receipt"]
    if (
        worker.bytes != objects[2].byte_count
        or ContentId.parse(worker.content_id).digest != objects[2].sha256
    ):
        raise ValueError("parabricks_result_receipt_drift")
    journey.read_artifact(worker)
    cache = "hit" if isinstance(state.reference, Hit) else "miss"
    source = "cache_hit" if cache == "hit" else "cpu_built"
    events = journey.read()
    # The reading Run's file; a Fork child keeps the parent's Execution identity.
    context = dict(
        context,
        journey_file=str(journey.file),
        run_root=str(Path(journey.file).parent.parent),
    )
    return ParabricksExecutionOutcome(
        **context,
        execution_pass=state.pass_,
        accounting_journal_file=state.outcome.contact_journal,
        accounting_ledger_file=str(
            Path(state.outcome.contact_journal).with_name(
                "predecessor-contact-ledger.json"
            )
        ),
        disposition="succeeded",
        status=state.status,
        cache_outcome=cache,
        prepared_reference_source=source,
        prepared_reference_receipt_id=state.reference.receipt_id,
        h100_receipt_id=state.h100.receipt_id,
        h100_checkpoint_event_id=state.latest_event_ids["H100ArtifactsVerified"],
        h100_request_id=manifest["request_id"],
        provider_job_id=manifest["provider_job_id"],
        verified_objects=tuple(objects),
        output_proofs=tuple(
            pointers[role]
            for role in ("aligned_bam", "VCF", "worker_receipt", "execution_manifest")
        ),
        input_proofs=state.inputs.pointers,
        reference_proofs=state.reference.pointers,
        object_scope=approved["launch"]["object_scope"],
        outcome_event_id=state.latest_event_ids["ExecutionOutcome"],
        execution_evidence_event_id=events[-1].event_id,
        execution_evidence_sha256=canonical_json_digest(
            [e.model_dump(mode="json", by_alias=True) for e in events]
        ),
    )


def refuse_legacy_execution(journey: Journey) -> None:
    """A Journey without execution-path facts may still carry the old format."""
    if any(
        event["event_type"] == "ParabricksExecutionBound" for event in journey.read()
    ):
        raise ValueError("legacy_execution_unsupported")


def read_parabricks_result(
    journey_file: str,
) -> ParabricksResult:
    """Read semantic execution truth; old execution formats require their source era."""
    journey_file = str(Path(journey_file).resolve())
    if has_execution_path(journey_file):
        journey = JourneyPath.open(Path(journey_file))
        return parabricks_result(journey)
    refuse_legacy_execution(Journey.open(Path(journey_file)))
    raise ParabricksResultUnavailable("parabricks_result_unavailable")


def read_completed_parabricks_result(journey_file: str) -> ParabricksExecutionOutcome:
    """Read the verified result required by artifact and scientific consumers."""
    journey_file = str(Path(journey_file).resolve())
    result = read_parabricks_result(journey_file)
    if not isinstance(result, ParabricksExecutionOutcome):
        raise ParabricksExecutionIncomplete("parabricks_execution_incomplete")
    return result
