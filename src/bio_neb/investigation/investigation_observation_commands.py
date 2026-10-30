"""Public application operation for one approved Investigation observation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from bio_neb.agentic_journey.refusal_disposition import RefusalDisposition
from bio_neb.application.input_journey import RunInputsRegistered, validate_fact
from bio_neb.application.queries import (
    InvestigationRead,
    read_investigation,
)
from bio_neb.application.verification_policy import (
    ResolvedArtifactVerificationPolicy,
)
from bio_neb.failure import FailureTuning, run_failure_tuning
from bio_neb.investigation.download_review import (
    InputDownloadRefused,
    ask_download_review,
    project_download_reviews,
)
from bio_neb.investigation.investigation import (
    InvestigationAuthority,
    InvestigationAuthorityGranted,
    InvestigationProjection,
    WorkItem,
    validate_investigation_fact,
)
from bio_neb.investigation.investigation_observation import (
    InvestigationCollectionEvidenceRecorded,
    InvestigationCollectionItemState,
    InvestigationEvidenceRecorded,
    InvestigationObservationCommand,
    InvestigationWorkItemStarted,
    RunProvenance,
    SampleObservation,
    assess_investigation_observation,
    derive_investigation_run_provenance,
    fold_investigation_collection,
    observation_failure_reason,
    observation_target,
)
from bio_neb.investigation.observation_defaults import RegionDefaults
from bio_neb.investigation.vcf_materialization import VcfMaterializationRefused
from bio_neb.investigation.vcf_observation import (
    AttestedPlainVcf,
    BcftoolsRuntimeBinding,
    DeterministicVcfObservationTool,
    LocalFileObservation,
    ObserveRegionInput,
    ObserveVariantInput,
    PreparedVcfObservationReceipt,
    VariantObservationRefused,
    VcfMaterializationBinding,
    VcfPreparationFailed,
    VcfProcessAdapter,
)
from bio_neb.journey import Journey
from bio_neb.recording import work_frontier
from bio_neb.verified_artifacts import VerifiedObject


class InvestigationObservationApplicationRefused(ValueError):
    def __init__(self, code: str, *fields: str) -> None:
        super().__init__(code)
        self.code = code
        self.fields = tuple(fields)
        self.disposition = RefusalDisposition(kind="caller_correction_and_reissue")


class MaterializationObservationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_id: StrictStr = Field(pattern=r"^vcf-materialization-receipt:[0-9a-f]{64}$")
    descriptor: VerifiedObject
    attested_vcf: AttestedPlainVcf
    receipt_file: LocalFileObservation

    @model_validator(mode="after")
    def immutable_receipt(self) -> "MaterializationObservationEvidence":
        if (
            self.receipt_file.is_writable
            or self.receipt_file.sha256 is None
            or self.receipt_file.sha256 != self.attested_vcf.acquisition_receipt_sha256
            or self.receipt_file.path != self.attested_vcf.acquisition_receipt_path
        ):
            raise ValueError("materialization receipt evidence is not immutable")
        return self


class RuntimeProbeObservationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: BcftoolsRuntimeBinding
    receipt_file: LocalFileObservation

    @model_validator(mode="after")
    def stable_bound_receipt(self) -> "RuntimeProbeObservationEvidence":
        if (
            self.receipt_file.sha256 is None
            or self.receipt_file.path != self.runtime.probe_receipt_path
            or self.receipt_file.sha256 != self.runtime.probe_receipt_sha256
        ):
            raise ValueError("runtime probe receipt evidence is not stably bound")
        return self


class InvestigationObservationEvidenceReader(Protocol):
    def materialization(
        self, *, receipt_file: str, vcf_file: str
    ) -> MaterializationObservationEvidence: ...

    def runtime(
        self, *, receipt_file: str, executable: str
    ) -> RuntimeProbeObservationEvidence: ...


def _absolute(value: str, field: str) -> str:
    if not os.path.isabs(value) or os.path.normpath(value) != value:
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_path_invalid", field
        )
    return value


@dataclass(frozen=True)
class _AuthorityBinding:
    continuation_id: str
    journey_id: str
    reference: Any
    sample: Any
    vcf: VerifiedObject


def _refused_preparation(
    *,
    authority: InvestigationAuthority,
    target: ObserveVariantInput,
    preparation: VcfPreparationFailed,
) -> VariantObservationRefused:
    failure_id = (
        "vcf-observation-preparation-failure:"
        + hashlib.sha256(
            json.dumps(
                preparation.model_dump(mode="json"),
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    return VariantObservationRefused(
        schema_version="bio-neb.variant-observation/2",
        code="tool_failure",
        violations=(f"preparation:{preparation.code}", *preparation.details),
        authority_kind="investigation_authority",
        continuation_id=authority.authority_id,
        preparation_receipt_id=failure_id,
        preparation_failure=preparation,
        target=target,
    )


@dataclass(frozen=True)
class _PreparedObservation:
    command: InvestigationObservationCommand
    authority: InvestigationAuthority
    materialization: MaterializationObservationEvidence
    binding: _AuthorityBinding
    preparation: PreparedVcfObservationReceipt | VcfPreparationFailed
    adapter: VcfProcessAdapter
    run_provenance: RunProvenance
    provenance_origin_event_id: str | None
    model_contact_count: int
    source_contact_count: int

    def observe_item(self, work_item: WorkItem) -> InvestigationEvidenceRecorded:
        """Build one approved member's observation, assessment and admission evidence."""
        authority = self.authority
        target = observation_target(work_item)
        result = (
            _refused_preparation(
                authority=authority, target=target, preparation=self.preparation
            )
            if isinstance(self.preparation, VcfPreparationFailed)
            # A region reads every record in it, its REFs checked.
            else DeterministicVcfObservationTool().observe_region(
                target,
                self.binding,
                self.preparation,
                self.adapter,
                record_limit=RegionDefaults().record_limit,
            )
            if isinstance(target, ObserveRegionInput)
            else DeterministicVcfObservationTool().observe(
                target,
                self.binding,
                self.preparation,
                self.adapter,
                authority_kind="investigation_authority",
            )
        )
        observation = SampleObservation(
            investigation_authority_id=authority.authority_id,
            proposal_id=authority.proposal_id,
            review_response_id=authority.review_response_id,
            work_item_id=work_item.work_item_id,
            sample_id=work_item.sample_id,
            assembly_accession=work_item.target.assembly_accession,
            assembly_name=work_item.target.assembly_name,
            vcf_descriptor=self.materialization.descriptor,
            materialization_receipt_id=self.materialization.receipt_id,
            materialization_receipt_sha256=self.materialization.receipt_file.sha256,
            preparation=self.preparation,
            result=result,
            run_provenance=self.run_provenance,
            provenance_origin_event_id=self.provenance_origin_event_id,
        )
        candidate, assessment, admission, state = assess_investigation_observation(
            authority=authority,
            work_item=work_item,
            observation=observation,
            inherited_model_contact_count=self.model_contact_count,
            inherited_source_contact_count=self.source_contact_count,
        )
        return InvestigationEvidenceRecorded(
            command=self.command,
            authority=authority,
            work_item=work_item,
            sample_observation=observation,
            candidate_evidence=candidate,
            assessment=assessment,
            admission=admission,
            evidence_state=state,
        )


class InvestigationObservationSession:
    """Own local preparation across item operations and selected approvals.

    The caller may retain this session across commands. A fresh owner may restore
    applicable local byte proof; every command independently establishes authority.
    """

    def __init__(
        self,
        *,
        adapter: VcfProcessAdapter | None = None,
        evidence_reader: InvestigationObservationEvidenceReader | None = None,
        adapter_factory: Callable[[], VcfProcessAdapter] | None = None,
        verification_policy: ResolvedArtifactVerificationPolicy | None = None,
    ) -> None:
        from bio_neb.application.local_file_observation import (
            ObservationFiles,
        )
        from bio_neb.investigation.investigation_observation_adapter import (
            DirectInvestigationObservationEvidence,
        )
        from bio_neb.investigation.local_bcftools import LocalBcftoolsAdapter

        if adapter is not None and adapter_factory is not None:
            raise InvestigationObservationApplicationRefused(
                "investigation_observation_adapter_ambiguous", "adapter"
            )
        self._adapter = adapter
        self._adapter_factory = adapter_factory
        self._files = (
            adapter.files
            if isinstance(adapter, LocalBcftoolsAdapter)
            else evidence_reader.files
            if isinstance(evidence_reader, DirectInvestigationObservationEvidence)
            else None
        ) or ObservationFiles(
            verification_policy=verification_policy,
        )
        if self._files.closed:
            self._files = ObservationFiles(
                verification_policy=verification_policy,
            )
        if (
            verification_policy is not None
            and self._files.verification_policy != verification_policy
        ):
            raise InvestigationObservationApplicationRefused(
                "investigation_observation_verification_policy_conflict", "session"
            )
        if isinstance(adapter, LocalBcftoolsAdapter):
            adapter.files = self._files
        self.evidence_reader = (
            evidence_reader
            if evidence_reader is not None
            else (DirectInvestigationObservationEvidence(files=self._files))
        )
        if isinstance(self.evidence_reader, DirectInvestigationObservationEvidence):
            self.evidence_reader.files = self._files
        self._materialization: VcfMaterializationBinding | None = None
        self._preparation: PreparedVcfObservationReceipt | None = None

    def __enter__(self) -> InvestigationObservationSession:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._files.close()
        self._preparation = None

    @property
    def adapter(self) -> VcfProcessAdapter:
        if self._adapter is None:
            from bio_neb.investigation.local_bcftools import (
                LocalBcftoolsAdapter,
            )

            self._adapter = (
                self._adapter_factory()
                if self._adapter_factory is not None
                else LocalBcftoolsAdapter(files=self._files)
            )
            if isinstance(self._adapter, LocalBcftoolsAdapter):
                # Receipt admission precedes factory effects. The newly constructed
                # local adapter inherits the session's already owned content proof.
                self._adapter.files = self._files
        return self._adapter

    def prepare(
        self,
        authority: _AuthorityBinding,
        binding: VcfMaterializationBinding,
    ) -> PreparedVcfObservationReceipt | VcfPreparationFailed:
        tool = DeterministicVcfObservationTool()
        owned = self._preparation
        if (
            owned is not None
            and self._materialization == binding
            and owned.journey_id == authority.journey_id
            and owned.vcf_descriptor == authority.vcf
            and owned.assembly == authority.reference.assembly_accession
            and owned.continuation_sample_id == authority.sample.sample_id
        ):
            return tool.rebind_preparation(owned, authority)
        result = tool.prepare(
            authority,
            binding,
            self.adapter,
            authority_kind="investigation_authority",
            read_original=(
                self._files.verification_policy.effective.mode
                == "independent_full_read"
            ),
        )
        if isinstance(result, PreparedVcfObservationReceipt):
            self._materialization = binding
            self._preparation = result
        return result


def _work_item_started(
    authority: InvestigationAuthority, work_item: WorkItem, ordinal: int
) -> InvestigationWorkItemStarted:
    return InvestigationWorkItemStarted(
        authority_id=authority.authority_id,
        work_item_id=work_item.work_item_id,
        selected_ordinal=ordinal,
        sample_id=work_item.sample_id,
        target=observation_target(work_item),
    )


def _prepare_observation(
    command: InvestigationObservationCommand,
    *,
    journey: Journey,
    timeline: list[dict[str, Any]],
    projection: InvestigationProjection,
    session: InvestigationObservationSession,
    acquire_inputs: Callable[[], object] | None = None,
) -> _PreparedObservation:
    authority = projection.authority
    if acquire_inputs is not None:
        acquire_inputs()
    investigation_input = projection.investigation_input
    assert authority is not None and investigation_input is not None
    run_provenance, provenance_origin_event_id = derive_investigation_run_provenance(
        timeline
    )
    selected_evidence = session.evidence_reader
    materialization = selected_evidence.materialization(
        receipt_file=command.materialization_receipt_file,
        vcf_file=command.vcf_file,
    )
    expected_vcf = investigation_input.vcf
    if materialization.descriptor != VerifiedObject(
        role="VCF",
        bucket=expected_vcf.bucket,
        object_key=expected_vcf.object_key,
        byte_count=expected_vcf.byte_count,
        sha256=expected_vcf.sha256,
    ):
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_vcf_identity_changed", "vcf_file"
        )
    runtime = selected_evidence.runtime(
        receipt_file=command.bcftools_probe_receipt_file,
        executable=command.bcftools_executable,
    )
    binding = _AuthorityBinding(
        continuation_id=authority.authority_id,
        journey_id=journey.journey_id,
        reference=SimpleNamespace(
            assembly_accession=investigation_input.assembly_accession
        ),
        sample=SimpleNamespace(sample_id=investigation_input.sample_id),
        vcf=materialization.descriptor,
    )
    materialization_binding = VcfMaterializationBinding(
        attested_vcf=materialization.attested_vcf,
        runtime=runtime.runtime,
        derivation_root=command.derivation_root,
        continuation_sample_id=investigation_input.sample_id,
        expected_vcf_sample_names=(investigation_input.sample_id,),
    )
    adapter = session.adapter
    try:
        preparation = session.prepare(binding, materialization_binding)
    except Exception as error:
        preparation = VcfPreparationFailed(
            code="tool_failure",
            details=(f"unexpected_exception:{type(error).__name__}",),
        )
    return _PreparedObservation(
        command=command,
        authority=authority,
        materialization=materialization,
        binding=binding,
        preparation=preparation,
        adapter=adapter,
        run_provenance=run_provenance,
        provenance_origin_event_id=provenance_origin_event_id,
        model_contact_count=projection.model_contact_count,
        source_contact_count=projection.source_contact_count,
    )


def _registration(timeline) -> RunInputsRegistered:
    """The Run's one registration; a Fork inherits it."""
    registrations = [
        validate_fact(event)
        for event in timeline
        if event.get("event_type") == "RunInputsRegistered"
    ]
    if len(registrations) != 1 or not isinstance(registrations[0], RunInputsRegistered):
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_registration_invalid", "journey_file"
        )
    return registrations[0]


def _registered_failure_tuning(timeline) -> FailureTuning:
    """The failure section the Run registered; a Fork inherits it."""
    return _registration(timeline).runtime_selections.failure


def advance_approved_observation(
    journey: Journey,
    *,
    inputs: InvestigationObservationCommand | None = None,
    adapter: VcfProcessAdapter | None = None,
    evidence_reader: InvestigationObservationEvidenceReader | None = None,
    object_read_factory: Callable[[Any], Any] | None = None,
) -> bool:
    """Hand one eligible exact grant to the existing observation operation.

    The Run's first grant supplies its own inputs (#107): the probed tools and the
    Run's verified VCF under its own directory, acquired by the preparation. A
    later grant reuses them with no transfer, hash or probe. A tool the Run cannot
    provide records ``RuntimeConfigurationFailed`` and holds the Run. Test inputs
    may still be supplied; this operation neither widens a grant nor schedules.
    """
    read = read_investigation(journey)
    downloads = project_download_reviews(read.timeline)
    if downloads.pending:
        # A refused download waits for its answer; no grant observes.
        return False
    eligible = set(read.work_policy.eligible_work_item_ids)
    authority = next(
        (
            grant
            for grant in read.investigation.authorities
            if {item.work_item_id for item in grant.authorized_work_items} <= eligible
        ),
        None,
    )
    if authority is None:
        return False
    supplied = None
    if inputs is None:
        inputs = next(
            (
                collection.evidence.command
                for collection in read.collections
                # An inherited collection names its parent Run's files.
                if collection.evidence.command.journey_file == str(journey.path)
            ),
            None,
        )
    if inputs is None:
        supplied = _supply_inputs(
            journey, read, authority, object_read_factory, downloads
        )
        inputs = supplied.command
    command = InvestigationObservationCommand(
        journey_file=str(journey.path),
        authority_id=authority.authority_id,
        **inputs.model_dump(
            exclude={"derivation_root", "journey_file", "authority_id"}
        ),
        derivation_root=str(
            Path(inputs.derivation_root) / authority.authority_id.split(":", 1)[1]
        ),
    )
    if supplied is None:
        investigation_observe(
            command, adapter=adapter, evidence_reader=evidence_reader, journey=journey
        )
        return True
    with InvestigationObservationSession(
        adapter=adapter,
        evidence_reader=evidence_reader,
        verification_policy=supplied.verification_policy,
    ) as session:
        _investigation_observe(
            command,
            session=session,
            journey=journey,
            acquire_inputs=supplied.acquire_inputs,
        )
    return True


def _supply_inputs(journey, read, authority, object_read_factory, downloads):
    """The Run's own first inputs; a configuration fault holds the Run."""
    from bio_neb.agentic_journey.runtime_config import RuntimeConfigError
    from bio_neb.application.run_failure import hold_for_configuration_fix
    from bio_neb.investigation.observation_inputs import supply_observation_inputs

    timeline = read.timeline
    vcf = read.investigation.investigation_input.vcf
    grant_event = next(
        event
        for event in timeline
        if event.get("event_type") == "InvestigationAuthorityGranted"
        and event["payload"]["authority"]["authority_id"] == authority.authority_id
    )
    try:
        return supply_observation_inputs(
            journey,
            authority_id=authority.authority_id,
            grant_event_id=grant_event["event_id"],
            descriptor=VerifiedObject(
                role="VCF",
                bucket=vcf.bucket,
                object_key=vcf.object_key,
                byte_count=vcf.byte_count,
                sha256=vcf.sha256,
            ),
            registration=_registration(timeline),
            downloads=downloads,
            object_read_factory=object_read_factory,
        )
    except RuntimeConfigError as error:
        hold_for_configuration_fix(journey, work_frontier(timeline)["event_id"], error)


def investigation_observe(
    command: InvestigationObservationCommand,
    *,
    adapter: VcfProcessAdapter | None = None,
    adapter_factory: Callable[[], VcfProcessAdapter] | None = None,
    evidence_reader: InvestigationObservationEvidenceReader | None = None,
    session: InvestigationObservationSession | None = None,
    journey: Journey | None = None,
    verification_policy: ResolvedArtifactVerificationPolicy | None = None,
) -> InvestigationCollectionEvidenceRecorded:
    if session is not None:
        if verification_policy is not None:
            raise InvestigationObservationApplicationRefused(
                "investigation_observation_session_ambiguous", "verification_policy"
            )
        return _investigation_observe(
            command,
            adapter=adapter,
            adapter_factory=adapter_factory,
            evidence_reader=evidence_reader,
            session=session,
            journey=journey,
        )
    with InvestigationObservationSession(
        adapter=adapter,
        adapter_factory=adapter_factory,
        evidence_reader=evidence_reader,
        verification_policy=verification_policy,
    ) as owned:
        return _investigation_observe(command, session=owned, journey=journey)


def _retained_observation(
    read: InvestigationRead, command: InvestigationObservationCommand
) -> InvestigationCollectionEvidenceRecorded | None:
    for collection in read.collections:
        if collection.evidence.command == command:
            return collection.evidence
    if any(
        collection.evidence.command.authority_id == command.authority_id
        for collection in read.collections
    ):
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_conflict", "authority_id"
        )
    return None


def _investigation_observe(  # noqa: C901 - one batch loop routes item and shared faults (R2, #105).
    command: InvestigationObservationCommand,
    *,
    adapter: VcfProcessAdapter | None = None,
    adapter_factory: Callable[[], VcfProcessAdapter] | None = None,
    evidence_reader: InvestigationObservationEvidenceReader | None = None,
    session: InvestigationObservationSession | None = None,
    journey: Journey | None = None,
    acquire_inputs: Callable[[], object] | None = None,
) -> InvestigationCollectionEvidenceRecorded | None:
    """Observe the approved Work items sequentially after one shared preparation.

    ``acquire_inputs`` downloads the Run's VCF and BAM first when they are not yet
    local; its failure is the preparation's, so the first item fails with it
    (#107). A download its stage policy gave up on instead asks its Review and
    returns ``None`` with no item started (#363).
    """

    trusted = InvestigationObservationCommand.model_validate(command, strict=True)
    if adapter is not None and adapter_factory is not None:
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_adapter_ambiguous", "adapter"
        )
    if session is not None and any(
        item is not None for item in (adapter, adapter_factory, evidence_reader)
    ):
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_session_ambiguous", "session"
        )
    for name in (
        "vcf_file",
        "materialization_receipt_file",
        "bcftools_executable",
        "bcftools_probe_receipt_file",
        "derivation_root",
    ):
        _absolute(getattr(trusted, name), name)
    if journey is None:
        journey = Journey.open(Path(trusted.journey_file))
    elif journey.path != Path(trusted.journey_file) or not journey.is_current():
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_context_stale", "journey_file"
        )
    read = read_investigation(journey)
    timeline = read.timeline
    projection = read.investigation
    previous_collection = read.evidence
    replay = _retained_observation(read, trusted)
    if replay is not None:
        return replay
    retained_items = tuple(item.evidence for item in read.recorded_members)
    if previous_collection is not None and previous_collection.failed:
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_conflict", "authority_id"
        )
    authority = next(
        (
            grant
            for grant in projection.authorities
            if grant.authority_id == trusted.authority_id
        ),
        None,
    )
    if (
        projection.status
        not in {"approved", "review_pending", "investigation_work_ready"}
        or authority is None
        or authority.authority_id != trusted.authority_id
        or not authority.authorized_work_items
        or projection.investigation_input is None
        or not timeline
    ):
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_authority_stale", "authority_id"
        )
    frontier = work_frontier(timeline)
    authority_fact = validate_investigation_fact(frontier)
    if not (
        isinstance(authority_fact, InvestigationAuthorityGranted)
        or frontier.get("event_type") == "InvestigationCollectionEvidenceRecorded"
        or frontier.get("event_type")
        in {
            "InvestigationAssertionRecorded",
            "InvestigationReviewRequested",
            "InvestigationPolicyAmended",
            "InvestigationDownloadReviewAnswered",
        }
    ):
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_head_changed", "journey_file"
        )
    investigation_input = projection.investigation_input
    if any(
        item.work_item.work_item_id
        in {member.work_item_id for member in authority.authorized_work_items}
        for item in retained_items
    ):
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_conflict", "authority_id"
        )
    # Preparation binds the exact batch, even when another pending batch was granted later.
    projection = projection.model_copy(update={"authority": authority})
    if (
        any(
            work_item.sample_id != investigation_input.sample_id
            or work_item.target.assembly_accession
            != investigation_input.assembly_accession
            for work_item in authority.authorized_work_items
        )
        or any(
            item not in projection.proposals[-1].work_items
            for item in authority.authorized_work_items
        )
        or (
            projection.revisions
            and projection.revisions[-1].supersedes_proposal_id
            == projection.proposals[-1].proposal_id
            and any(
                item not in projection.revisions[-1].carried_work_items
                for item in authority.authorized_work_items
            )
        )
    ):
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_work_item_changed", "authority_id"
        )
    prepared = None
    preparation_error = None
    try:
        if session is None:
            session = InvestigationObservationSession(
                adapter=adapter,
                adapter_factory=adapter_factory,
                evidence_reader=evidence_reader,
            )
        prepared = _prepare_observation(
            trusted,
            journey=journey,
            timeline=timeline,
            projection=projection,
            session=session,
            acquire_inputs=acquire_inputs,
        )
    except InvestigationObservationApplicationRefused:
        raise
    except InputDownloadRefused as refused:
        # The failure module asks the researcher; no item runs before the
        # answer.
        if not journey.is_current():
            raise InvestigationObservationApplicationRefused(
                "investigation_observation_head_changed", "journey_file"
            ) from refused
        ask_download_review(
            journey,
            parent_event_id=frontier["event_id"],
            authority_id=authority.authority_id,
            role=refused.role,
            object_key=refused.descriptor.object_key,
            code=refused.refusal.code,
            details=refused.refusal.details,
        )
        return None
    except Exception as error:
        preparation_error = error
    current = journey
    if not current.is_current():
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_head_changed", "journey_file"
        )
    parent_event_id = frontier["event_id"]
    tuning = _registered_failure_tuning(timeline)
    item_states: list[InvestigationCollectionItemState] = []
    for ordinal, work_item in enumerate(authority.authorized_work_items, start=1):
        started = _work_item_started(authority, work_item, ordinal)
        retained_started = current.record(
            started.event_type,
            started.model_dump(
                mode="json",
                exclude={"event_type"},
                exclude_computed_fields=True,
            ),
            [parent_event_id],
        )
        parent_event_id = retained_started["event_id"]
        try:
            if preparation_error is not None:
                raise preparation_error
            assert prepared is not None
            # The Run's recorded failure section tunes each item's query.
            with run_failure_tuning(tuning):
                fact = prepared.observe_item(work_item)
        except Exception as error:
            item_states.append(_item_fault_state(work_item, ordinal, error))
            if preparation_error is not None:
                # The shared preparation failed: no later item can run either.
                break
            # An item's own fault leaves its siblings to run (R2).
            continue
        state = fact.evidence_state
        retained = current.record(
            fact.event_type,
            fact.model_dump(
                mode="json",
                exclude={"event_type"},
                exclude_computed_fields=True,
            ),
            [parent_event_id],
        )
        parent_event_id = retained["event_id"]
        failure_reason = observation_failure_reason(fact.sample_observation)
        status = (
            "failed"
            if failure_reason
            else "completed"
            if state.work_item_disposition == "supported"
            else "negative"
            if state.work_item_disposition == "contradicted"
            else "refused"
            if state.work_item_disposition == "refused"
            else "unresolved"
        )
        item_states.append(
            InvestigationCollectionItemState(
                work_item_id=work_item.work_item_id,
                selected_ordinal=ordinal,
                status=status,
                evidence_state_id=state.evidence_state_id,
                outcome=state.outcome,
                reason=failure_reason,
            )
        )
        if failure_reason:
            break
    item_states.extend(
        InvestigationCollectionItemState(
            work_item_id=remaining.work_item_id,
            selected_ordinal=index,
            status="not_started",
        )
        for index, remaining in enumerate(
            authority.authorized_work_items[len(item_states) :],
            start=len(item_states) + 1,
        )
    )
    states_by_id = {
        item.work_item_id: item
        for item in (previous_collection.item_states if previous_collection else ())
    }
    states_by_id.update({item.work_item_id: item for item in item_states})
    collection_ids = tuple(
        item.work_item_id for item in projection.proposals[-1].work_items
    )
    approved_ids = {
        item.work_item_id
        for grant in projection.authorities
        for item in grant.authorized_work_items
    }
    collection_states = tuple(
        states_by_id[work_item_id].model_copy(update={"selected_ordinal": ordinal})
        if work_item_id in states_by_id
        and states_by_id[work_item_id].status
        not in {"approval_pending", "observation_pending"}
        else InvestigationCollectionItemState(
            work_item_id=work_item_id,
            selected_ordinal=ordinal,
            status="approval_pending"
            if work_item_id not in approved_ids
            else "observation_pending",
        )
        for ordinal, work_item_id in enumerate(collection_ids, start=1)
    )
    collection = fold_investigation_collection(
        command=trusted,
        authority=authority,
        item_states=collection_states,
        collection_work_item_ids=collection_ids,
        collection_proposal_id=projection.proposals[-1].proposal_id,
        previous_collection_state_id=previous_collection.collection_state_id
        if previous_collection
        else None,
        work_policy=read.work_policy.effective,
        work_policy_id=read.work_policy.policy_id,
    )
    recorded = current.record(
        collection.event_type,
        collection.model_dump(
            mode="json",
            exclude={"event_type"},
            exclude_computed_fields=True,
        ),
        [parent_event_id],
    )
    _hold_item_faults(
        recorded, item_states, preparation_failed=preparation_error is not None
    )
    return collection


def _item_fault_state(work_item, ordinal, error) -> InvestigationCollectionItemState:
    """An item whose lookup raised: failed, naming the exception's type."""
    return InvestigationCollectionItemState(
        work_item_id=work_item.work_item_id,
        selected_ordinal=ordinal,
        status="failed",
        reason=f"observation_exception:{type(error).__name__}"
        + (f":{error.code}" if isinstance(error, VcfMaterializationRefused) else ""),
    )


def _hold_item_faults(recorded, item_states, *, preparation_failed: bool) -> None:
    """Hold the Run for a failed lookup, after every item that could run ran.

    The collection is retained with each failed item accounted. An item's own
    unclassified fault is our bug; a shared preparation or observation fault
    (the VCF, the sample's identity) stops the batch at its item and holds too
    (R2, #105). `resume` continues past the accounted items; members a shared
    fault left unstarted hold completion until the researcher decides.
    """
    from bio_neb.failure import Invariant

    failed = [item for item in item_states if item.status == "failed"]
    if not failed:
        return
    shared = preparation_failed or any(
        (item.reason or "").startswith("shared_") for item in failed
    )
    raise Invariant(
        "investigation_shared_observation_failed"
        if shared
        else "investigation_item_failed",
        "an approved item's lookup failed",
        evidence={
            "collection_event_id": recorded["event_id"],
            "work_item_ids": [item.work_item_id for item in failed],
        },
    )


__all__ = [
    "InvestigationObservationApplicationRefused",
    "InvestigationObservationEvidenceReader",
    "InvestigationObservationSession",
    "MaterializationObservationEvidence",
    "RuntimeProbeObservationEvidence",
    "investigation_observe",
]
