"""Typed Journey facts and contact-free projection for reference companions."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Tuple

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from bio_neb.agentic_journey.source_answers import SourceAnswer
from bio_neb.contracts import FrozenContract
from bio_neb.journey import JourneyEvent, ReadOnlyJsonObject
from bio_neb.prebuilt_index.choice import PrebuiltIndexCandidate, stated_bwa_version
from bio_neb.reference_discovery.assembly_suggestion import SuggestedTarget
from bio_neb.reference_discovery.prebuilt_index_tool import (
    PrebuiltIndexesObserved,
    PrebuiltIndexGrounding,
    PrebuiltIndexReadObserved,
    ReadPrebuiltIndexInput,
    ReadPrebuiltIndexResult,
    SearchPrebuiltIndexesInput,
    SearchPrebuiltIndexesResult,
)
from bio_neb.reference_discovery.reference_companions_tool import (
    CompanionObserved,
    CompanionRequirementGrounding,
    CompanionRole,
    CompanionToolFailed,
    CompanionWorkItem,
    DescribeCompanionInput,
    DescribeCompanionResult,
    VerifiedReferenceBinding,
    companion_asset_disposition_id,
    derive_companion_work_items,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_PATTERN,
)

if TYPE_CHECKING:
    from bio_neb.application.planning_read import PrefixSnapshot


def _canonical_uuid(value: str, field_name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{field_name} must be canonical")
    return value


class _Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_id: StrictStr

    @model_validator(mode="after")
    def canonical_attempt(self) -> "_Fact":
        _canonical_uuid(self.attempt_id, "attempt_id")
        return self


class CompanionObservationRecorded(_Fact):
    event_type: Literal["CompanionObservationRecorded"] = "CompanionObservationRecorded"
    tool_input: DescribeCompanionInput
    result: DescribeCompanionResult
    # What each of the Tool's GETs' sources answered, as the model is shown it; each
    # GET's bytes are retained on its own `ReferenceSourceRetrieved`.
    source_answers: tuple[SourceAnswer, ...]

    @model_validator(mode="after")
    def exact_result(self) -> "CompanionObservationRecorded":
        if self.result.work_item != self.tool_input.work_item:
            raise ValueError("companion result changed its exact work item")
        if isinstance(self.result, CompanionObserved):
            if self.result.assembly_accession != self.tool_input.assembly_accession:
                raise ValueError("companion result changed its exact query")
            if self.tool_input.selected_asset_id is not None and any(
                candidate.asset_id != self.tool_input.selected_asset_id
                for candidate in self.result.candidates
            ):
                raise ValueError("companion result changed the selected asset")
        return self


def validate_reference_facts_fact(
    event: Mapping[str, Any],
) -> CompanionObservationRecorded | None:
    if event.get("event_type") != "CompanionObservationRecorded":
        return None
    return CompanionObservationRecorded.model_validate(
        {"event_type": event["event_type"], **event.get("payload", {})}
    )


class PrebuiltIndexesListed(_Fact):
    """One page of a Galaxy repository's BWA-MEM rows the model searched (#464)."""

    event_type: Literal["PrebuiltIndexesListed"] = "PrebuiltIndexesListed"
    tool_input: SearchPrebuiltIndexesInput
    result: SearchPrebuiltIndexesResult
    source_answers: tuple[SourceAnswer, ...]

    @model_validator(mode="after")
    def exact_result(self) -> "PrebuiltIndexesListed":
        if isinstance(self.result, PrebuiltIndexesObserved) and (
            self.result.repository,
            self.result.search_text,
            self.result.offset,
        ) != (
            self.tool_input.repository,
            self.tool_input.search_text,
            self.tool_input.offset,
        ):
            raise ValueError("an index listing changed its exact query")
        return self

    def grounding_of(self) -> PrebuiltIndexGrounding:
        return self.tool_input.grounding


class PrebuiltIndexRead(_Fact):
    """One listed candidate's files and contigs beside the Run's sequences (#464)."""

    event_type: Literal["PrebuiltIndexRead"] = "PrebuiltIndexRead"
    tool_input: ReadPrebuiltIndexInput
    result: ReadPrebuiltIndexResult
    source_answers: tuple[SourceAnswer, ...]

    @model_validator(mode="after")
    def exact_result(self) -> "PrebuiltIndexRead":
        if (
            isinstance(self.result, PrebuiltIndexReadObserved)
            and self.result.candidate != self.tool_input.candidate
        ):
            raise ValueError("an index read changed its listed candidate")
        return self

    def grounding_of(self) -> PrebuiltIndexGrounding:
        return self.tool_input.grounding

    def bound_candidate(self) -> PrebuiltIndexCandidate | None:
        """What a choice of this read binds; None when it holds not every requested
        sequence, or the read failed."""
        result = self.result
        if (
            not isinstance(result, PrebuiltIndexReadObserved)
            or not result.contigs.holds_every_requested_sequence
        ):
            return None
        return PrebuiltIndexCandidate(
            candidate=result.candidate,
            files=result.files,
            contigs=result.contigs,
            stated_bwa_version=stated_bwa_version(result.candidate),
        )


class PrebuiltIndexChosen(_Fact):
    """The model's index for the Run: a read that holds every requested sequence, or
    none (the CPU build), with its reason (#464; ADR 0049 membership)."""

    event_type: Literal["PrebuiltIndexChosen"] = "PrebuiltIndexChosen"
    grounding: PrebuiltIndexGrounding
    read_event_id: StrictStr | None
    candidate: PrebuiltIndexCandidate | None
    reason: StrictStr = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def exact_choice(self) -> "PrebuiltIndexChosen":
        if (self.read_event_id is None) != (self.candidate is None):
            raise ValueError("an index choice binds the candidate of the read it cites")
        if self.reason.strip() != self.reason:
            raise ValueError("an index choice states a concise reason")
        return self

    def grounding_of(self) -> PrebuiltIndexGrounding:
        return self.grounding


class SuggestAssemblyInput(FrozenContract):
    """The read the model suggests: another assembly's index that holds that
    assembly's main sequences, copied from the read (#459 D4)."""

    grounding: PrebuiltIndexGrounding
    read_event_id: StrictStr
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    candidate: PrebuiltIndexCandidate
    source: Literal["ncbi_dbsnp", "eva"]
    """The variant source the model named for the known targets' check (#469)."""
    reason: StrictStr = Field(min_length=1, max_length=2000)


class SuggestedAssembly(FrozenContract):
    """The suggested assembly and each known target as its lookup there answered it
    (#459 D3)."""

    status: Literal["observed"] = "observed"
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    targets: tuple[SuggestedTarget, ...]
    summary: StrictStr


class AssemblyWithIndexSuggested(_Fact):
    """The model's suggestion of another assembly of the Run's organism whose Galaxy
    index fits it, the known targets checked there, and the question it opens: use
    that assembly for the whole Run, or build the requested one's index on CPU
    (owner, #459 D3/D4, 2026-10-30)."""

    event_type: Literal["AssemblyWithIndexSuggested"] = "AssemblyWithIndexSuggested"
    tool_input: SuggestAssemblyInput
    result: SuggestedAssembly
    source_answers: tuple[SourceAnswer, ...]

    @model_validator(mode="after")
    def exact_result(self) -> "AssemblyWithIndexSuggested":
        if self.result.assembly_accession != self.tool_input.assembly_accession:
            raise ValueError("a suggestion changed its assembly")
        return self

    def grounding_of(self) -> PrebuiltIndexGrounding:
        return self.tool_input.grounding


PrebuiltIndexFact = (
    PrebuiltIndexesListed
    | PrebuiltIndexRead
    | PrebuiltIndexChosen
    | AssemblyWithIndexSuggested
)
_PREBUILT_FACTS = {
    owner.model_fields["event_type"].default: owner
    for owner in (
        PrebuiltIndexesListed,
        PrebuiltIndexRead,
        PrebuiltIndexChosen,
        AssemblyWithIndexSuggested,
    )
}


def validate_prebuilt_index_fact(
    event: Mapping[str, Any],
) -> PrebuiltIndexFact | None:
    owner = _PREBUILT_FACTS.get(event.get("event_type"))
    if owner is None:
        return None
    return owner.model_validate(
        {"event_type": event["event_type"], **event.get("payload", {})}
    )


class PrebuiltIndexChoiceView(BaseModel):
    """The Run's current index choice: its event, the read it cites, the candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_event_id: StrictStr
    read_event_id: StrictStr | None
    candidate: PrebuiltIndexCandidate | None


class PrebuiltIndexWork(BaseModel):
    """The verified assembly's index work: the description it answers, its listings
    and reads, and the current choice (None while it is open: before any
    choice, or after an Execution Review revise reopened it, owner #464 I2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grounding: PrebuiltIndexGrounding
    listing_event_ids: Tuple[StrictStr, ...]
    read_event_ids: Tuple[StrictStr, ...]
    choice: PrebuiltIndexChoiceView | None
    reopened_by_event_id: StrictStr | None
    suggestion_event_ids: Tuple[StrictStr, ...] = ()
    """Each suggestion of another assembly's index made on this work (#459)."""


class CompanionFactView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_event_id: StrictStr
    attempt_id: StrictStr
    work_item_id: StrictStr
    tool_input: ReadOnlyJsonObject
    result: ReadOnlyJsonObject


class CompanionAssetDisposition(BaseModel):
    """A researcher choice over retained candidates, never compatibility proof."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    work_item_id: StrictStr
    asset_id: StrictStr
    request_event_id: StrictStr
    response_event_id: StrictStr
    evidence_event_ids: Tuple[StrictStr, ...]
    request_evidence_event_ids: Tuple[StrictStr, ...]


class ReferenceFactsProjection(BaseModel):
    """Current companion readiness, derived only from immutable Journey evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    journey_id: StrictStr
    plan_event_id: StrictStr
    plan_id: StrictStr
    requirement_id: StrictStr
    subject_key: StrictStr
    verified_reference: VerifiedReferenceBinding
    role_selection_event_id: StrictStr | None
    required_roles: Tuple[CompanionRole, ...]
    work_items: Tuple[CompanionWorkItem, ...]
    pending_work_item_ids: Tuple[StrictStr, ...]
    completed_work_item_ids: Tuple[StrictStr, ...]
    conflicted_work_item_ids: Tuple[StrictStr, ...]
    no_progress_work_item_ids: Tuple[StrictStr, ...]
    asset_dispositions: Tuple[CompanionAssetDisposition, ...]
    facts: Tuple[CompanionFactView, ...]
    # The verified genomic FASTA's prebuilt-index work: the work item completes only
    # once the model's choice follows its verified observation and the latest Execution
    # Review revise; None until the FASTA is verified with its publisher file.
    prebuilt_index: PrebuiltIndexWork | None = None
    status: Literal[
        "companion_roles_pending",
        "planning_hitl_pending",
        "reference_facts_progress",
        "reference_facts_no_progress",
        "typed_failure",
        "reference_facts",
    ]


_ROLE_SETS: dict[str, tuple[CompanionRole, ...]] = {
    "companion-role-set:genomic_fasta": ("genomic_fasta",),
    "companion-role-set:annotation": ("annotation",),
    "companion-role-set:annotation+genomic_fasta": ("annotation", "genomic_fasta"),
}


def _requirement_evidence_ids(requirement) -> tuple[str, ...]:
    event_ids: list[str] = []
    for evidence in requirement.evidence:
        for name, value in evidence.model_dump(mode="json").items():
            if (
                name.endswith("event_id")
                and isinstance(value, str)
                and value not in event_ids
            ):
                event_ids.append(value)
    return tuple(event_ids)


def project_reference_facts(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
) -> ReferenceFactsProjection:
    """Project exact role work, completion, conflict, and replay without contact."""

    from bio_neb.application.planning_read import PrefixSnapshot

    return PrefixSnapshot(events).companions


def _project_reference_facts(  # noqa: C901, PLR0915 - Existing domain grammar retained; this slice changes dependency ownership.
    context: PrefixSnapshot,
) -> ReferenceFactsProjection:
    timeline = context.timeline
    by_id = context.by_id
    ancestors_by_id = context.ancestry
    plan_event, plan = context.require_plan()
    companion_requirements = [
        item for item in plan.plan.requirements if item.kind == "companion_asset"
    ]
    if not companion_requirements:
        raise ValueError("reference facts require a companion requirement")
    requirement = companion_requirements[0]
    reference = context.reference
    if reference.verified_reference is None:
        raise ValueError("reference facts require current verified reference identity")
    if reference.description_event_id is None:
        raise ValueError("verified reference lacks its source event")
    binding = VerifiedReferenceBinding(
        reference_event_id=reference.description_event_id,
        assembly_accession=reference.verified_reference.assembly_accession,
        reference_evidence_id=reference.verified_reference.evidence_id,
    )

    hitl = context.hitl.for_requirements({requirement.requirement_id})
    from bio_neb.parabricks.reference_requirements import reference_asset_roles

    receipt = None
    if not reference.choices:
        # Source-era role evidence keeps its original grammar. Current capabilities
        # always retain a choice and obtain requirements from the consumer.
        selected_receipts = [
            r
            for r in hitl.receipts
            if r.plan_id == plan.plan.plan_id
            and r.requirement_id == requirement.requirement_id
            and r.subject_key == requirement.subject_key
            and r.assertion.selected_candidate_id in _ROLE_SETS
        ]
        if len(selected_receipts) > 1:
            raise ValueError("companion requirement has conflicting role dispositions")
        receipt = selected_receipts[0] if selected_receipts else None
        if (
            receipt is not None
            and binding.reference_event_id not in receipt.assertion.evidence_event_ids
        ):
            raise ValueError(
                "companion role disposition lacks verified-reference evidence"
            )
    selected_roles = (
        reference_asset_roles()
        if reference.choices
        else (_ROLE_SETS[receipt.assertion.selected_candidate_id] if receipt else ())
    )
    source_ids = [
        plan_event["event_id"],
        *_requirement_evidence_ids(requirement),
        binding.reference_event_id,
    ]
    if receipt is not None:
        source_ids.append(receipt.response_event_id)
    grounding = CompanionRequirementGrounding(
        journey_id=plan_event["journey_id"],
        plan_id=plan.plan.plan_id,
        requirement_id=requirement.requirement_id,
        subject_key=requirement.subject_key,
        source_event_ids=tuple(source_ids),
    )
    work_items = (
        derive_companion_work_items(
            grounding=grounding,
            reference=binding,
            selected_roles=selected_roles,
            role_selection_event_ids=(
                receipt.response_event_id if receipt else plan_event["event_id"],
            ),
        )
        if selected_roles
        else ()
    )
    work_by_id = {item.work_item_id: item for item in work_items}

    fact_views: list[CompanionFactView] = []
    results_by_work: dict[str, list[CompanionObserved]] = {
        item.work_item_id: [] for item in work_items
    }
    retained_facts: list[
        tuple[dict[str, Any], CompanionObservationRecorded, CompanionWorkItem, set[str]]
    ] = []
    call_keys_by_work: dict[str, set[tuple[str, str | None, str | None]]] = {
        item.work_item_id: set() for item in work_items
    }
    unchanged_repeat_work: set[str] = set()
    for event_id, fact in context.companion_facts.items():
        event = by_id[event_id]
        event_id = event["event_id"]
        ancestors = ancestors_by_id[event_id]
        starts = [
            ancestor_id
            for ancestor_id in ancestors
            if by_id[ancestor_id].get("event_type") == "AttemptStarted"
            and by_id[ancestor_id].get("payload", {}).get("attempt_id")
            == fact.attempt_id
        ]
        if len(starts) != 1:
            raise ValueError("companion Tool fact lacks its exact owning Attempt")
        if (
            fact.tool_input.work_item.reference.reference_event_id
            != binding.reference_event_id
            and (
                reference.revision_event_id is not None
                or reference.chosen_suggestion is not None
            )
        ):
            # Old immutable observations keep their original checked reference; they
            # cannot satisfy the new selection's required assets.
            _ = context.through(event_id).companions
            continue
        expected = work_by_id.get(fact.tool_input.work_item.work_item_id)
        if expected is None or fact.tool_input.work_item != expected:
            raise ValueError("companion Tool fact changed its current exact work item")
        required_ancestry = set(expected.grounding.source_event_ids)
        if not required_ancestry <= ancestors:
            raise ValueError("companion Tool fact omitted immutable grounding ancestry")
        call_key = (
            fact.tool_input.assembly_accession,
            fact.tool_input.selected_asset_id,
            fact.tool_input.selection_event_id,
        )
        prior_keys = call_keys_by_work[expected.work_item_id]
        if call_key in prior_keys:
            unchanged_repeat_work.add(expected.work_item_id)
        prior_keys.add(call_key)
        if isinstance(fact.result, CompanionObserved):
            results_by_work[expected.work_item_id].append(fact.result)
        retained_facts.append((event, fact, expected, ancestors))
        fact_views.append(
            CompanionFactView(
                source_event_id=event_id,
                attempt_id=fact.attempt_id,
                work_item_id=expected.work_item_id,
                tool_input=fact.tool_input.model_dump(mode="json"),
                result=fact.result.model_dump(mode="json"),
            )
        )

    fact_by_event = {
        event["event_id"]: (fact, work) for event, fact, work, _ in retained_facts
    }
    candidate_by_id: dict[str, tuple[str, str, set[str]]] = {}
    for event, fact, work, _ in retained_facts:
        if not isinstance(fact.result, CompanionObserved):
            continue
        for candidate in fact.result.candidates:
            disposition_id = companion_asset_disposition_id(
                work.work_item_id, candidate.asset_id
            )
            retained = candidate_by_id.get(disposition_id)
            if retained is None:
                candidate_by_id[disposition_id] = (
                    work.work_item_id,
                    candidate.asset_id,
                    {event["event_id"]},
                )
            elif retained[:2] == (work.work_item_id, candidate.asset_id):
                retained[2].add(event["event_id"])
            else:
                raise ValueError("companion disposition identity collided")

    requests_by_id = {request.request_event_id: request for request in hitl.requests}
    asset_dispositions: list[CompanionAssetDisposition] = []
    for selected in hitl.receipts:
        candidate_id = selected.assertion.selected_candidate_id
        retained = candidate_by_id.get(candidate_id or "")
        if (
            retained is None
            or selected.plan_id != plan.plan.plan_id
            or selected.requirement_id != requirement.requirement_id
            or selected.subject_key != requirement.subject_key
        ):
            continue
        work_item_id, asset_id, evidence_events = retained
        request = requests_by_id[selected.request_event_id]
        candidate = next(
            (item for item in request.candidates if item.candidate_id == candidate_id),
            None,
        )
        if (
            candidate is None
            or candidate.asserted_value != asset_id
            or not set(candidate.evidence_event_ids) <= evidence_events
            or selected.assertion.asserted_value != asset_id
        ):
            raise ValueError("companion asset disposition changed its exact candidate")
        asset_dispositions.append(
            CompanionAssetDisposition(
                work_item_id=work_item_id,
                asset_id=asset_id,
                request_event_id=selected.request_event_id,
                response_event_id=selected.response_event_id,
                evidence_event_ids=candidate.evidence_event_ids,
                request_evidence_event_ids=request.evidence_event_ids,
            )
        )

    disposition_by_response = {
        item.response_event_id: item for item in asset_dispositions
    }
    for _event, fact, work, ancestors in retained_facts:
        selected_asset_id = fact.tool_input.selected_asset_id
        selection_event_id = fact.tool_input.selection_event_id
        if selected_asset_id is None:
            continue
        disposition = disposition_by_response.get(selection_event_id or "")
        if (
            disposition is None
            or disposition.work_item_id != work.work_item_id
            or disposition.asset_id != selected_asset_id
            or disposition.response_event_id not in ancestors
        ):
            raise ValueError(
                "companion Tool selection lacks exact HITL disposition ancestry"
            )

    latest_fact_by_work: dict[
        str, tuple[dict[str, Any], CompanionObservationRecorded]
    ] = {}
    for event, fact, work, _ in retained_facts:
        latest_fact_by_work[work.work_item_id] = (event, fact)
    latest_disposition_by_work: dict[str, CompanionAssetDisposition] = {}
    for disposition in asset_dispositions:
        latest_disposition_by_work[disposition.work_item_id] = disposition

    no_progress: list[str] = list(unchanged_repeat_work)
    for response in hitl.receipts:
        if (
            response.plan_id != plan.plan.plan_id
            or response.requirement_id != requirement.requirement_id
            or response.assertion.selected_candidate_id is not None
        ):
            continue
        request = requests_by_id[response.request_event_id]
        if request.candidates:
            continue
        targeted = {
            fact_by_event[event_id][1].work_item_id
            for event_id in request.evidence_event_ids
            if event_id in fact_by_event
        }
        if len(targeted) != 1:
            continue
        work_item_id = targeted.pop()
        latest = latest_fact_by_work.get(work_item_id)
        if (
            latest is not None
            and latest[0]["event_id"] in request.evidence_event_ids
            and timeline.index(by_id[response.response_event_id])
            > timeline.index(latest[0])
        ):
            no_progress.append(work_item_id)

    completed: list[str] = []
    conflicted: list[str] = []
    for work_item in work_items:
        observations = results_by_work[work_item.work_item_id]
        verified_asset_ids = {
            observation.candidates[0].asset_id
            for observation in observations
            if observation.outcome == "verified"
        }
        retained_latest = latest_fact_by_work.get(work_item.work_item_id)
        latest = (
            retained_latest[1].result
            if retained_latest is not None
            and isinstance(retained_latest[1].result, CompanionObserved)
            else None
        )
        disposition = latest_disposition_by_work.get(work_item.work_item_id)
        selected_closure = bool(
            disposition is not None
            and retained_latest is not None
            and retained_latest[1].tool_input.selected_asset_id == disposition.asset_id
            and retained_latest[1].tool_input.selection_event_id
            == disposition.response_event_id
            and latest is not None
            and latest.outcome == "verified"
            and latest.candidates[0].asset_id == disposition.asset_id
        )
        if selected_closure:
            completed.append(work_item.work_item_id)
            no_progress = [
                item for item in no_progress if item != work_item.work_item_id
            ]
        elif len(verified_asset_ids) > 1:
            conflicted.append(work_item.work_item_id)
        elif (
            latest is not None
            and latest.outcome == "verified"
            and len(verified_asset_ids) == 1
            and work_item.work_item_id not in unchanged_repeat_work
        ):
            completed.append(work_item.work_item_id)
    prebuilt_index = _prebuilt_index_work(context, binding)
    if prebuilt_index.choice is None:
        # The genomic FASTA's work completes only on the index choice.
        completed[:] = [
            work_id
            for work_id in completed
            if next(w for w in work_items if w.work_item_id == work_id).role
            != "genomic_fasta"
        ]
    pending = [
        item.work_item_id for item in work_items if item.work_item_id not in completed
    ]
    typed_failure = any(
        isinstance(fact.result, CompanionToolFailed) for _, fact, _, _ in retained_facts
    )
    if typed_failure:
        status = "typed_failure"
    elif (
        hitl.pending_request is not None
        and hitl.pending_request.requirement_id == requirement.requirement_id
    ):
        status = "planning_hitl_pending"
    elif not reference.choices and receipt is None:
        status = "companion_roles_pending"
    elif work_items and not pending:
        status = "reference_facts"
    elif pending and set(pending) <= set(no_progress):
        status = "reference_facts_no_progress"
    else:
        status = "reference_facts_progress"
    return ReferenceFactsProjection(
        journey_id=plan_event["journey_id"],
        plan_event_id=plan_event["event_id"],
        plan_id=plan.plan.plan_id,
        requirement_id=requirement.requirement_id,
        subject_key=requirement.subject_key,
        verified_reference=binding,
        role_selection_event_id=receipt.response_event_id if receipt else None,
        required_roles=selected_roles,
        work_items=work_items,
        pending_work_item_ids=tuple(pending),
        completed_work_item_ids=tuple(completed),
        conflicted_work_item_ids=tuple(conflicted),
        no_progress_work_item_ids=tuple(dict.fromkeys(no_progress)),
        asset_dispositions=tuple(asset_dispositions),
        facts=tuple(fact_views),
        prebuilt_index=prebuilt_index,
        status=status,
    )


def _prebuilt_index_work(context, binding):
    """The verified assembly's index work, read from its listings, reads and choices in
    order (#464; owner #459 D1: the index is the assembly's, so it opens with the
    verified reference, before any companion is described). A fact on another
    reference is history; a choice binds exactly its read's candidate."""
    grounding = PrebuiltIndexGrounding(
        reference_event_id=binding.reference_event_id,
        assembly_accession=binding.assembly_accession,
    )
    event = context.by_id[binding.reference_event_id]
    facts = context.prebuilt_index_facts
    position = {e["event_id"]: i for i, e in enumerate(context.timeline)}
    revisions = [
        e["event_id"]
        for e in context.timeline
        if e.get("event_type") == "ExecutionHitlResponded"
        and e.get("payload", {}).get("response", {}).get("action") == "revise"
        and position[e["event_id"]] > position[event["event_id"]]
    ]
    reopened = revisions[-1] if revisions else None
    opened_at = position[reopened or event["event_id"]]
    listings, reads, choice, suggestions = {}, {}, None, []
    for event_id, fact in facts.items():
        if fact.grounding_of() != grounding:
            earlier = context.by_id.get(fact.grounding_of().reference_event_id)
            if (
                earlier is None
                or earlier["event_type"] != "ReferenceDescriptionRecorded"
            ):
                raise ValueError("an index fact names no reference description")
            continue
        if isinstance(fact, PrebuiltIndexesListed):
            listings[event_id] = fact
        elif isinstance(fact, PrebuiltIndexRead):
            listing = listings.get(fact.tool_input.listing_event_id)
            if (
                listing is None
                or not isinstance(listing.result, PrebuiltIndexesObserved)
                or listing.result.candidate(fact.tool_input.value)
                != fact.tool_input.candidate
            ):
                raise ValueError("an index read names no row of an earlier listing")
            reads[event_id] = fact
        elif isinstance(fact, AssemblyWithIndexSuggested):
            read = reads.get(fact.tool_input.read_event_id)
            if (
                read is None
                or read.tool_input.other_assembly != fact.tool_input.assembly_accession
                or read.bound_candidate() != fact.tool_input.candidate
            ):
                raise ValueError(
                    "a suggestion cites no earlier read of its assembly's index"
                )
            suggestions.append(event_id)
        else:
            if fact.read_event_id is None:
                expected = None
            elif fact.read_event_id in reads:
                expected = reads[fact.read_event_id].bound_candidate()
                if expected is None:
                    raise ValueError(
                        "an index choice cites a read that lacks sequences"
                    )
            else:
                raise ValueError("an index choice cites no earlier read")
            if fact.candidate != expected:
                raise ValueError("an index choice changed its read's candidate")
            if position[event_id] > opened_at:
                choice = PrebuiltIndexChoiceView(
                    source_event_id=event_id,
                    read_event_id=fact.read_event_id,
                    candidate=fact.candidate,
                )
    return PrebuiltIndexWork(
        grounding=grounding,
        listing_event_ids=tuple(listings),
        read_event_ids=tuple(reads),
        choice=choice,
        reopened_by_event_id=reopened,
        suggestion_event_ids=tuple(suggestions),
    )


__all__ = [
    "CompanionFactView",
    "CompanionObservationRecorded",
    "AssemblyWithIndexSuggested",
    "PrebuiltIndexChoiceView",
    "PrebuiltIndexChosen",
    "PrebuiltIndexFact",
    "PrebuiltIndexRead",
    "PrebuiltIndexWork",
    "PrebuiltIndexesListed",
    "validate_prebuilt_index_fact",
    "ReferenceFactsProjection",
    "project_reference_facts",
    "validate_reference_facts_fact",
]
