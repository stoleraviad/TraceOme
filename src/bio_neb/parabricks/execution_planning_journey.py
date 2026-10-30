"""Recipe/resource facts and deterministic execution-planning readiness."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from bio_neb.application.input_journey import InputMetadataInspected, validate_fact
from bio_neb.identifier_rule import word_aligned_spans
from bio_neb.journey import JourneyEvent, ReadOnlyJsonObject
from bio_neb.parabricks.recipe_resources import (
    CapabilityGrounding,
    ListRecipesInput,
    LookupResourcesInput,
    RecipeCandidate,
    RecipeEligibilityAssertion,
    RecipeObservation,
    RecipeWorkItem,
    ResourceCandidate,
    ResourceObservation,
    ResourceWorkItem,
    candidate_disposition_id,
    make_recipe_work_item,
    make_resource_work_item,
    recipe_eligibility_assertion_id,
    resource_recipe_dependency_id,
)

if TYPE_CHECKING:
    from bio_neb.application.planning_read import PrefixSnapshot

# Execution always produces an aligned BAM and a VCF: the fixed product
# contract recipe work is planned for and the package emits.
EXECUTION_OUTPUT_FORMATS = ("BAM", "VCF")


class _Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    attempt_id: StrictStr

    @model_validator(mode="after")
    def canonical_attempt(self) -> _Fact:
        if str(uuid.UUID(self.attempt_id)) != self.attempt_id:
            raise ValueError("attempt_id must be canonical")
        return self


class RecipeObservationRecorded(_Fact):
    event_type: Literal["RecipeObservationRecorded"] = "RecipeObservationRecorded"
    tool_input: ListRecipesInput
    result: RecipeObservation

    @model_validator(mode="after")
    def exact_result(self) -> RecipeObservationRecorded:
        if self.result.work_item != self.tool_input.work_item:
            raise ValueError("recipe result changed its exact work item")
        return self


class ResourceObservationRecorded(_Fact):
    event_type: Literal["ResourceObservationRecorded"] = "ResourceObservationRecorded"
    tool_input: LookupResourcesInput
    result: ResourceObservation

    @model_validator(mode="after")
    def exact_result(self) -> ResourceObservationRecorded:
        if self.result.work_item != self.tool_input.work_item:
            raise ValueError("resource result changed its exact work item")
        return self


# The records the reference stage keeps of the Run's organism: the ploidy
# judgment cites one of them, never a code list of organisms.
ORGANISM_RECORD_EVENTS = frozenset(
    {
        "TaxonResolutionRecorded",
        "ReferenceAssembliesRecorded",
        "ReferenceDescriptionRecorded",
    }
)
PloidyJudgment = Literal["diploid", "likely_non_diploid", "unknown"]


class OrganismPloidyJudgment(BaseModel):
    """The model's judgment of the Run organism's ploidy, with the organism record
    it rests on and the words of that record it read (ADR 0049; #470 Q1 a)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    judgment: PloidyJudgment
    record_event_id: StrictStr
    quote: StrictStr = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def exact_quote(self) -> OrganismPloidyJudgment:
        if self.quote.strip() != self.quote:
            raise ValueError("organism quote must be exact text")
        return self


def quoted_in_record(record: Mapping[str, Any], quote: str) -> bool:
    """Whether the quote stands, whole words, in a string of the organism record's
    copied result: the only check code makes of the model's judgment."""

    def strings(value: object) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from strings(item)

    return any(
        word_aligned_spans(text, quote) for text in strings(record["payload"]["result"])
    )


class PlanningCandidateSelected(_Fact):
    """Agent or researcher disposition over already validated candidate evidence."""

    event_type: Literal["PlanningCandidateSelected"] = "PlanningCandidateSelected"
    kind: Literal["recipe", "resource"]
    work_item_id: StrictStr
    candidate_id: StrictStr
    basis_event_id: StrictStr
    authority: Literal["agent_ranking", "researcher_selection"]
    authority_event_id: StrictStr
    # A recipe selection carries the model's ploidy judgment; an older selection
    # carries none and reads as not judged.
    organism_ploidy: OrganismPloidyJudgment | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def judged_recipe(self) -> PlanningCandidateSelected:
        if self.kind == "resource" and self.organism_ploidy is not None:
            raise ValueError("a resource selection carries no ploidy judgment")
        return self

    def admitted_arguments(self) -> dict[str, str]:
        """The selection Tool's admitted arguments this fact was recorded from."""
        arguments = {
            "work_item_id": self.work_item_id,
            "candidate_id": self.candidate_id,
            "basis_event_id": self.basis_event_id,
        }
        if self.organism_ploidy is not None:
            arguments |= {
                "organism_ploidy": self.organism_ploidy.judgment,
                "organism_record_event_id": self.organism_ploidy.record_event_id,
                "organism_quote": self.organism_ploidy.quote,
            }
        return arguments


ExecutionPlanningFact = (
    RecipeObservationRecorded | ResourceObservationRecorded | PlanningCandidateSelected
)
_FACTS = {
    "RecipeObservationRecorded": RecipeObservationRecorded,
    "ResourceObservationRecorded": ResourceObservationRecorded,
    "PlanningCandidateSelected": PlanningCandidateSelected,
}


def validate_execution_planning_fact(
    event: Mapping[str, Any],
) -> ExecutionPlanningFact | None:
    owner = _FACTS.get(event.get("event_type"))
    if owner is None:
        return None
    return owner.model_validate(
        {"event_type": event["event_type"], **event.get("payload", {})}
    )


class PlanningFactView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_event_id: StrictStr
    attempt_id: StrictStr
    kind: Literal["recipe", "resource"]
    work_item_id: StrictStr
    tool_input: ReadOnlyJsonObject
    result: ReadOnlyJsonObject


class RequirementReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    requirement_id: StrictStr
    kind: StrictStr
    subject_key: StrictStr
    state: Literal[
        "missing",
        "needs_validation",
        "needs_selection",
        "conflicted",
        "typed_failure",
        "ready",
    ]
    evidence_event_ids: tuple[StrictStr, ...]
    candidate_ids: tuple[StrictStr, ...] = ()
    next_actions: tuple[StrictStr, ...] = ()


class SelectedRecipeBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    requirement_id: StrictStr
    subject_key: StrictStr
    work_item_id: StrictStr
    candidate: RecipeCandidate
    selection_event_id: StrictStr


class SelectedResourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    requirement_id: StrictStr
    subject_key: StrictStr
    work_item_id: StrictStr
    candidate: ResourceCandidate
    selection_event_id: StrictStr


class PlanningSelectionView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_event_id: StrictStr
    attempt_id: StrictStr
    kind: Literal["recipe", "resource"]
    work_item_id: StrictStr
    candidate_id: StrictStr
    basis_event_id: StrictStr
    authority: Literal["agent_ranking", "researcher_selection"]
    authority_event_id: StrictStr
    organism_ploidy: OrganismPloidyJudgment | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ExecutionRevisionContext(BaseModel):
    """Exact researcher direction used only to re-enter execution planning."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    revision_event_id: StrictStr
    request_event_id: StrictStr
    proposal_id: StrictStr
    superseded_proposal: ReadOnlyJsonObject
    instruction: StrictStr
    revalidation_scope: Literal["execution_planning", "reference"] = (
        "execution_planning"
    )
    authority: Literal["researcher_direction"] = "researcher_direction"
    scientifically_verified: Literal[False] = False
    authorizes_execution: Literal[False] = False

    @model_validator(mode="after")
    def exact_direction(self) -> ExecutionRevisionContext:
        for value, name in (
            (self.revision_event_id, "revision_event_id"),
            (self.request_event_id, "request_event_id"),
        ):
            if str(uuid.UUID(value)) != value:
                raise ValueError(f"{name} must be canonical")
        if self.superseded_proposal.get("proposal_id") != self.proposal_id:
            raise ValueError("revision context changed its superseded proposal")
        if not self.instruction or self.instruction.strip() != self.instruction:
            raise ValueError("revision instruction must be exact non-empty text")
        return self


class ExecutionPlanningProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    journey_id: StrictStr
    plan_event_id: StrictStr
    plan_id: StrictStr
    revision_context_event_id: StrictStr | None = None
    revision_context: ExecutionRevisionContext | None = None
    recipe_work_items: tuple[RecipeWorkItem, ...]
    recipe_tool_inputs: tuple[ListRecipesInput, ...]
    resource_work_items: tuple[ResourceWorkItem, ...]
    selected_recipe_candidates: tuple[RecipeCandidate, ...]
    selected_recipe_bindings: tuple[SelectedRecipeBinding, ...]
    selected_resource_candidates: tuple[ResourceCandidate, ...]
    selected_resource_bindings: tuple[SelectedResourceBinding, ...]
    active_recipe_work_item_id: StrictStr | None
    active_resource_work_item_id: StrictStr | None
    requirements: tuple[RequirementReadiness, ...]
    facts: tuple[PlanningFactView, ...]
    selections: tuple[PlanningSelectionView, ...]
    unresolved_execution_constraints: tuple[StrictStr, ...]
    status: Literal[
        "recipe_validation_pending",
        "recipe_selection_pending",
        "resource_validation_pending",
        "resource_selection_pending",
        "planning_hitl_pending",
        "typed_failure",
        "execution_planning_ready",
    ]

    @property
    def unresolved_requirement_ids(self) -> tuple[str, ...]:
        return tuple(
            item.requirement_id for item in self.requirements if item.state != "ready"
        )

    @property
    def recipe_work_item(self) -> RecipeWorkItem:
        selected = (
            self.active_recipe_work_item_id or self.recipe_work_items[0].work_item_id
        )
        return next(
            item for item in self.recipe_work_items if item.work_item_id == selected
        )

    @property
    def recipe_tool_input(self) -> ListRecipesInput:
        work_item_id = self.recipe_work_item.work_item_id
        return next(
            item
            for item in self.recipe_tool_inputs
            if item.work_item.work_item_id == work_item_id
        )

    @property
    def resource_work_item(self) -> ResourceWorkItem | None:
        if not self.resource_work_items:
            return None
        selected = (
            self.active_resource_work_item_id
            or self.resource_work_items[0].work_item_id
        )
        return next(
            item for item in self.resource_work_items if item.work_item_id == selected
        )

    @property
    def recipe_candidate(self) -> RecipeCandidate | None:
        work = self.resource_work_item
        return (
            work.recipe_candidate
            if work is not None
            else (
                self.selected_recipe_candidates[0]
                if self.selected_recipe_candidates
                else None
            )
        )

    @property
    def resource_candidate(self) -> ResourceCandidate | None:
        return (
            self.selected_resource_candidates[0]
            if self.selected_resource_candidates
            else None
        )


def _evidence_ids(requirement) -> tuple[str, ...]:
    ids: list[str] = []
    for evidence in requirement.evidence:
        for key, value in evidence.model_dump(mode="json").items():
            if key.endswith("event_id") and isinstance(value, str) and value not in ids:
                ids.append(value)
    return tuple(ids)


def _execution_revision_context(
    timeline: list[dict[str, Any]], by_id: Mapping[str, Mapping[str, Any]]
) -> ExecutionRevisionContext | None:
    """Return an exact researcher direction as planning context, never verified evidence."""

    responses = [
        event
        for event in timeline
        if event.get("event_type") == "ExecutionHitlResponded"
    ]
    if not responses:
        return None
    revised = [
        event
        for event in responses
        if isinstance((payload := event.get("payload")), Mapping)
        and isinstance((response := payload.get("response")), Mapping)
        and response.get("action") == "revise"
    ]
    if not revised:
        return None
    event = revised[-1]
    payload = event["payload"]
    response = payload["response"]
    if set(payload) != {
        "request_event_id",
        "response",
        "authority",
        "scientifically_verified",
        "authorizes_execution",
    } or set(response) != {
        "action",
        "proposal_id",
        "instruction",
        "revalidation_scope",
    }:
        raise ValueError("execution revision context has an invalid exact contract")
    request_id = payload.get("request_event_id")
    request = by_id.get(request_id)
    proposal = request.get("payload", {}).get("proposal") if request else None
    if (
        request is None
        or request.get("event_type") != "ExecutionProposalSubmitted"
        or event.get("parent_ids") != [request_id]
        or not isinstance(proposal, Mapping)
        or proposal.get("proposal_id") != response.get("proposal_id")
        or payload.get("authority") != "user_decision"
        or payload.get("scientifically_verified") is not False
        or payload.get("authorizes_execution") is not False
        or not isinstance(response.get("instruction"), str)
        or not response["instruction"]
        or response["instruction"].strip() != response["instruction"]
    ):
        raise ValueError("execution revision context changed its exact request binding")
    return ExecutionRevisionContext(
        revision_event_id=event["event_id"],
        request_event_id=request_id,
        proposal_id=response["proposal_id"],
        superseded_proposal=dict(proposal),
        instruction=response["instruction"],
        revalidation_scope=response["revalidation_scope"],
    )


def project_execution_planning(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
) -> ExecutionPlanningProjection:
    from bio_neb.application.planning_read import PrefixSnapshot

    return PrefixSnapshot(events).execution


def _project_execution_planning(  # noqa: C901, PLR0915 - Existing domain grammar retained; this slice changes dependency ownership.
    context: PrefixSnapshot,
) -> ExecutionPlanningProjection:
    timeline = context.timeline
    by_id = context.by_id
    ancestry = context.ancestry
    plan_event, plan = context.require_plan()
    recipe_requirements = tuple(
        item for item in plan.plan.requirements if item.kind == "analysis_recipe"
    )
    resource_requirements = tuple(
        item for item in plan.plan.requirements if item.kind == "resource_profile"
    )
    if not recipe_requirements or not resource_requirements:
        raise ValueError("execution planning lacks recipe/resource requirements")

    reference = context.reference
    companions = context.companions
    if (
        reference.verified_reference is None
        or not companions.required_roles
        or companions.pending_work_item_ids
        or companions.conflicted_work_item_ids
        or companions.no_progress_work_item_ids
        or len(companions.completed_work_item_ids) != len(companions.work_items)
    ):
        raise ValueError("execution planning requires exact verified upstream facts")
    inspections = []
    for event in timeline:
        fact = (
            validate_fact(event)
            if event.get("event_type") == "InputMetadataInspected"
            else None
        )
        if (
            isinstance(fact, InputMetadataInspected)
            and fact.result.status == "succeeded"
        ):
            inspections.append((event, fact))
    if not inspections:
        raise ValueError("execution planning requires successful input inspection")
    inspection_event, inspection = inspections[-1]
    bound = next(
        (
            e["payload"]["payload"]
            for e in timeline
            if e["event_type"] == "ExecutionPathRecorded"
            and e["payload"]["event_type"] == "ExecutionBound"
        ),
        None,
    )
    if bound is not None:
        # A correction metadata observation does not rewrite the approved recipe's
        # input grounding. Changed inputs require their own fresh authority.
        approved_inspection = bound["approved_input"]["proposal"]["inspection_event_id"]
        inspection_event, inspection = next(
            (e, f) for e, f in inspections if e["event_id"] == approved_inspection
        )
    revision_context = _execution_revision_context(timeline, by_id)
    revision_context_event_id = (
        revision_context.revision_event_id if revision_context is not None else None
    )
    common_sources = tuple(
        dict.fromkeys(
            [
                plan_event["event_id"],
                inspection_event["event_id"],
                reference.description_event_id,
                *[item.source_event_id for item in companions.facts],
                *([revision_context_event_id] if revision_context_event_id else []),
            ]
        )
    )
    recipe_works = tuple(
        make_recipe_work_item(
            grounding=CapabilityGrounding(
                journey_id=plan_event["journey_id"],
                plan_id=plan.plan.plan_id,
                requirement_id=requirement.requirement_id,
                subject_key=requirement.subject_key,
                source_event_ids=tuple(
                    dict.fromkeys([*common_sources, *_evidence_ids(requirement)])
                ),
            ),
            input_set_id=plan.plan.input_set_id,
            input_roles=tuple(item.role for item in inspection.result.inputs),
            assembly_accession=reference.verified_reference.assembly_accession,
            companion_roles=companions.required_roles,
            proposal_output_values=EXECUTION_OUTPUT_FORMATS,
        )
        for requirement in recipe_requirements
    )
    recipe_work_by_id = {item.work_item_id: item for item in recipe_works}

    retained: list[tuple[dict[str, Any], ExecutionPlanningFact, set[str]]] = []
    for event_id, fact in context.execution_facts.items():
        event = by_id[event_id]
        ancestors = ancestry[event["event_id"]]
        starts = [
            item
            for item in ancestors
            if by_id[item].get("event_type") == "AttemptStarted"
            and by_id[item].get("payload", {}).get("attempt_id") == fact.attempt_id
        ]
        if len(starts) != 1:
            raise ValueError("planning fact lacks its exact owning Attempt")
        if (
            not isinstance(fact, PlanningCandidateSelected)
            and not set(fact.tool_input.work_item.grounding.source_event_ids)
            <= ancestors
        ):
            raise ValueError("planning Tool fact omitted immutable grounding ancestry")
        if isinstance(fact, PlanningCandidateSelected) and fact.organism_ploidy:
            cited = by_id.get(fact.organism_ploidy.record_event_id)
            if (
                cited is None
                or cited["event_id"] not in ancestors
                or cited["event_type"] not in ORGANISM_RECORD_EVENTS
                or not quoted_in_record(cited, fact.organism_ploidy.quote)
            ):
                raise ValueError("ploidy judgment cites no organism record it read")
        retained.append((event, fact, ancestors))

    recipe_facts_by_work: dict[
        str, list[tuple[dict[str, Any], RecipeObservationRecorded, set[str]]]
    ] = {item.work_item_id: [] for item in recipe_works}
    for event, fact, ancestors in retained:
        if not isinstance(fact, RecipeObservationRecorded):
            continue
        expected = recipe_work_by_id.get(fact.tool_input.work_item.work_item_id)
        if expected is None and revision_context_event_id is not None:
            # A Planning-HITL response can advance the revision frontier and
            # replace the preceding revised work item. Retain that observation
            # as history without using it for the current snapshot.
            continue
        if expected is None or fact.tool_input.work_item != expected:
            raise ValueError("recipe Tool fact changed its exact subject")
        recipe_facts_by_work[expected.work_item_id].append((event, fact, ancestors))

    # The coordinator's target question bears on no recipe or resource, so only
    # questions on the plan's other requirements hold execution planning.
    hitl = context.hitl.for_requirements(
        {
            item.requirement_id
            for item in plan.plan.requirements
            if item.kind != "scientific_investigation"
        }
    )
    receipts_by_response = {item.response_event_id: item for item in hitl.receipts}
    eligibility_assertions: dict[str, RecipeEligibilityAssertion] = {}
    for requirement, work in zip(recipe_requirements, recipe_works, strict=False):
        receipt = next(
            (
                item
                for item in reversed(hitl.receipts)
                if item.plan_id == plan.plan.plan_id
                and item.requirement_id == requirement.requirement_id
                and (
                    revision_context_event_id is None
                    or (
                        revision_context is not None
                        and revision_context.revalidation_scope == "reference"
                    )
                    or revision_context_event_id in ancestry[item.response_event_id]
                )
                and item.assertion.selected_candidate_id is not None
                and item.assertion.selected_candidate_id.startswith(
                    "recipe-eligibility-assertion:"
                )
            ),
            None,
        )
        if receipt is None:
            continue
        request = next(
            item
            for item in hitl.requests
            if item.request_event_id == receipt.request_event_id
        )
        selected = next(
            (
                item
                for item in request.candidates
                if item.candidate_id == receipt.assertion.selected_candidate_id
            ),
            None,
        )
        if (
            selected is None
            or receipt.assertion.asserted_value != selected.asserted_value
        ):
            raise ValueError("recipe eligibility assertion changed its exact candidate")
        try:
            assertion_value = json.loads(selected.asserted_value)
            assertion = RecipeEligibilityAssertion(
                assay=assertion_value["assay"],
                read_layout=assertion_value["read_layout"],
                assertion_event_id=receipt.response_event_id,
            )
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("recipe eligibility assertion is invalid") from error
        if len(selected.evidence_event_ids) != 1:
            raise ValueError("recipe eligibility assertion lacks one observation")
        basis = next(
            (
                item
                for item in (
                    retained
                    if revision_context is not None
                    and revision_context.revalidation_scope == "reference"
                    else recipe_facts_by_work[work.work_item_id]
                )
                if isinstance(item[1], RecipeObservationRecorded)
                and item[1].tool_input.work_item.input_set_id == work.input_set_id
                and item[1].tool_input.work_item.grounding.requirement_id
                == work.grounding.requirement_id
                and item[0]["event_id"] == selected.evidence_event_ids[0]
            ),
            None,
        )
        candidate = (
            next(
                (
                    item
                    for item in basis[1].result.candidates
                    if item.compatibility == "ambiguous"
                    and recipe_eligibility_assertion_id(
                        basis[1].tool_input.work_item.work_item_id, item
                    )
                    == selected.candidate_id
                    and item.assay == assertion.assay
                    and item.read_layout == assertion.read_layout
                ),
                None,
            )
            if basis is not None
            else None
        )
        if candidate is None:
            raise ValueError(
                "recipe eligibility assertion lacks ambiguous Tool evidence"
            )
        eligibility_assertions[requirement.requirement_id] = assertion

    recipe_tool_inputs = tuple(
        ListRecipesInput(
            work_item=work,
            eligibility_assertion=eligibility_assertions.get(
                requirement.requirement_id
            ),
        )
        for requirement, work in zip(recipe_requirements, recipe_works, strict=False)
    )
    recipe_tool_input_by_work = {
        item.work_item.work_item_id: item for item in recipe_tool_inputs
    }
    for _, fact, ancestors in retained:
        if not isinstance(fact, RecipeObservationRecorded):
            continue
        if fact.tool_input.work_item.work_item_id not in recipe_work_by_id:
            continue
        assertion = fact.tool_input.eligibility_assertion
        if assertion is not None and (
            assertion not in eligibility_assertions.values()
            or assertion.assertion_event_id not in ancestors
        ):
            raise ValueError("recipe observation lacks exact eligibility assertion")
    selections = [
        (event, fact, ancestors)
        for event, fact, ancestors in retained
        if isinstance(fact, PlanningCandidateSelected)
    ]
    if revision_context_event_id is None and any(
        fact.kind == "recipe" and fact.work_item_id not in recipe_work_by_id
        for _, fact, _ in selections
    ):
        raise ValueError("recipe selection changed its exact work subject")

    def selection_history(kind: str, work, observations, owner):
        matching = [
            item
            for item in selections
            if item[1].kind == kind and item[1].work_item_id == work.work_item_id
        ]
        by_basis = {item[0]["event_id"]: item for item in observations}
        retained_history = []
        seen_bases: set[str] = set()
        for event, selected, ancestors in matching:
            basis = by_basis.get(selected.basis_event_id)
            if basis is None or selected.basis_event_id not in ancestors:
                raise ValueError(
                    f"{kind} selection lacks its exact validation ancestry"
                )
            if selected.basis_event_id in seen_bases:
                raise ValueError(
                    f"{kind} selection was repeated for one immutable evidence basis"
                )
            seen_bases.add(selected.basis_event_id)
            compatible = [
                item
                for item in basis[1].result.candidates
                if item.compatibility == "compatible"
            ]
            candidate = next(
                (
                    item
                    for item in compatible
                    if item.candidate_id == selected.candidate_id
                ),
                None,
            )
            if candidate is None:
                raise ValueError(f"{kind} selection lacks compatible Tool evidence")
            if selected.authority == "agent_ranking":
                authority = by_id.get(selected.authority_event_id)
                if (
                    authority is None
                    or authority["event_type"] != "ToolCallValidated"
                    or authority["payload"].get("attempt_id") != selected.attempt_id
                    or authority["payload"].get("tool_name")
                    != f"select_{kind}_candidate"
                    or authority["payload"].get("status") != "valid"
                    or authority["payload"].get("errors") not in ([], ())
                    or authority["payload"].get("arguments")
                    != selected.admitted_arguments()
                    or selected.authority_event_id not in ancestors
                ):
                    raise ValueError(
                        f"{kind} Agent selection lacks exact validated authority"
                    )
            else:
                receipt = receipts_by_response.get(selected.authority_event_id)
                expected_id = candidate_disposition_id(
                    kind, work.work_item_id, selected.candidate_id
                )
                if (
                    receipt is None
                    or receipt.requirement_id != owner.requirement_id
                    or receipt.assertion.selected_candidate_id != expected_id
                    or receipt.assertion.asserted_value != selected.candidate_id
                    or basis[0]["event_id"] not in receipt.assertion.evidence_event_ids
                    or selected.authority_event_id not in ancestors
                ):
                    raise ValueError(
                        f"{kind} researcher selection lacks exact HITL authority"
                    )
            retained_history.append(
                (candidate, event["event_id"], basis[0]["event_id"], basis)
            )
        return retained_history

    recipe_candidates: dict[str, RecipeCandidate] = {}
    recipe_selection_events: dict[str, str] = {}
    recipe_selection_history: dict[str, list[tuple[Any, ...]]] = {}
    recipe_current_basis: dict[
        str, tuple[dict[str, Any], RecipeObservationRecorded, set[str]]
    ] = {}
    recipe_states: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {}
    for requirement, work in zip(recipe_requirements, recipe_works, strict=False):
        current_facts = [
            item
            for item in recipe_facts_by_work[work.work_item_id]
            if item[1].tool_input == recipe_tool_input_by_work[work.work_item_id]
        ]
        latest = current_facts[-1] if current_facts else None
        history = selection_history(
            "recipe", work, recipe_facts_by_work[work.work_item_id], requirement
        )
        recipe_selection_history[requirement.requirement_id] = history
        current = next(
            (
                item
                for item in reversed(history)
                if latest is not None and item[2] == latest[0]["event_id"]
            ),
            None,
        )
        candidate = current[0] if current is not None else None
        selection_event_id = current[1] if current is not None else None
        if candidate is not None:
            recipe_candidates[requirement.requirement_id] = candidate
            recipe_selection_events[requirement.requirement_id] = selection_event_id
            recipe_current_basis[requirement.requirement_id] = latest
        if latest is None:
            recipe_states[requirement.requirement_id] = (
                "needs_validation",
                (),
                ("list_compatible_analysis_recipes",),
            )
        else:
            ids = tuple(item.candidate_id for item in latest[1].result.candidates)
            compatible = [
                item
                for item in latest[1].result.candidates
                if item.compatibility == "compatible"
            ]
            ambiguous = [
                item
                for item in latest[1].result.candidates
                if item.compatibility == "ambiguous"
            ]
            stale_candidates = {item[0].candidate_id for item in history}
            current_compatible_ids = {item.candidate_id for item in compatible}
            contradicted_selection = (
                bool(stale_candidates) and not current_compatible_ids
            )
            state = (
                "ready"
                if candidate is not None
                else "conflicted"
                if contradicted_selection
                else "needs_selection"
                if ambiguous
                else "needs_selection"
                if compatible
                else "typed_failure"
            )
            recipe_states[requirement.requirement_id] = (
                state,
                ids,
                ("request_recipe_eligibility_assertion",)
                if ambiguous
                else ("select_recipe_candidate", "request_recipe_selection")
                if state == "needs_selection"
                else ("list_compatible_analysis_recipes",)
                if state == "conflicted"
                else (),
            )

    selected_recipe_rows = [
        (
            requirement,
            recipe_candidates[requirement.requirement_id],
            recipe_selection_events[requirement.requirement_id],
        )
        for requirement in recipe_requirements
        if requirement.requirement_id in recipe_candidates
    ]
    resource_works: list[ResourceWorkItem] = []
    resource_work_by_requirement: dict[str, ResourceWorkItem] = {}
    dependency_pending: set[str] = set()

    def resource_work_for(
        requirement,
        recipe_requirement,
        candidate,
        selection_event_id,
        recipe_basis,
    ) -> ResourceWorkItem:
        grounding = CapabilityGrounding(
            journey_id=plan_event["journey_id"],
            plan_id=plan.plan.plan_id,
            requirement_id=requirement.requirement_id,
            subject_key=requirement.subject_key,
            source_event_ids=tuple(
                dict.fromkeys(
                    [
                        plan_event["event_id"],
                        *_evidence_ids(requirement),
                        recipe_basis[0]["event_id"],
                        selection_event_id,
                        *recipe_basis[
                            1
                        ].tool_input.work_item.grounding.source_event_ids,
                    ]
                )
            ),
        )
        return make_resource_work_item(
            grounding=grounding,
            recipe_candidate=candidate,
            recipe_event_id=selection_event_id,
        )

    for requirement in resource_requirements:
        paired = selected_recipe_rows[0] if len(selected_recipe_rows) == 1 else None
        if len(selected_recipe_rows) > 1:
            possible = {
                resource_recipe_dependency_id(
                    requirement.requirement_id,
                    recipe_requirement.requirement_id,
                    selection_event_id,
                ): (recipe_requirement, candidate, selection_event_id)
                for recipe_requirement, candidate, selection_event_id in selected_recipe_rows
            }
            receipt = next(
                (
                    item
                    for item in reversed(hitl.receipts)
                    if item.plan_id == plan.plan.plan_id
                    and item.requirement_id == requirement.requirement_id
                    and (
                        revision_context_event_id is None
                        or revision_context_event_id in ancestry[item.response_event_id]
                    )
                    and item.assertion.selected_candidate_id in possible
                ),
                None,
            )
            if receipt is not None:
                paired = possible[receipt.assertion.selected_candidate_id]
                request = next(
                    item
                    for item in hitl.requests
                    if item.request_event_id == receipt.request_event_id
                )
                expected_candidates = {
                    candidate_id: (row[0].requirement_id, row[2])
                    for candidate_id, row in possible.items()
                }
                retained_candidates = {
                    item.candidate_id: (item.asserted_value, item.evidence_event_ids[0])
                    for item in request.candidates
                    if len(item.evidence_event_ids) == 1
                }
                if (
                    request.plan_id != plan.plan.plan_id
                    or request.requirement_id != requirement.requirement_id
                    or request.subject_key != requirement.subject_key
                    or receipt.assertion.asserted_value != paired[0].requirement_id
                    or set(request.evidence_event_ids)
                    != {item[2] for item in selected_recipe_rows}
                    or retained_candidates != expected_candidates
                ):
                    raise ValueError(
                        "resource dependency HITL changed exact selected recipes"
                    )
        if paired is None:
            dependency_pending.add(requirement.requirement_id)
            continue
        recipe_requirement, candidate, selection_event_id = paired
        work = resource_work_for(
            requirement,
            recipe_requirement,
            candidate,
            selection_event_id,
            recipe_current_basis[recipe_requirement.requirement_id],
        )
        resource_works.append(work)
        resource_work_by_requirement[requirement.requirement_id] = work
    resource_work_by_id = {item.work_item_id: item for item in resource_works}
    historical_resource_work_by_id = dict(resource_work_by_id)
    for requirement in resource_requirements:
        for recipe_requirement in recipe_requirements:
            for candidate, selection_event_id, _, basis in recipe_selection_history[
                recipe_requirement.requirement_id
            ]:
                historical = resource_work_for(
                    requirement,
                    recipe_requirement,
                    candidate,
                    selection_event_id,
                    basis,
                )
                historical_resource_work_by_id[historical.work_item_id] = historical
    if revision_context_event_id is None and any(
        fact.kind == "resource"
        and fact.work_item_id not in historical_resource_work_by_id
        for _, fact, _ in selections
    ):
        raise ValueError("resource selection changed its exact work subject")
    resource_facts_by_work: dict[
        str, list[tuple[dict[str, Any], ResourceObservationRecorded, set[str]]]
    ] = {work_item_id: [] for work_item_id in historical_resource_work_by_id}
    for event, fact, ancestors in retained:
        if not isinstance(fact, ResourceObservationRecorded):
            continue
        expected = historical_resource_work_by_id.get(
            fact.tool_input.work_item.work_item_id
        )
        if expected is None and revision_context_event_id is not None:
            # Superseded revised snapshots remain replayable history but
            # cannot satisfy the current resource work item.
            continue
        if expected is None or fact.tool_input.work_item != expected:
            raise ValueError("resource Tool fact changed its exact recipe subject")
        resource_facts_by_work[expected.work_item_id].append((event, fact, ancestors))

    resource_requirement_by_id = {
        item.requirement_id: item for item in resource_requirements
    }
    resource_selection_history = {
        work_item_id: selection_history(
            "resource",
            work,
            resource_facts_by_work[work_item_id],
            resource_requirement_by_id[work.grounding.requirement_id],
        )
        for work_item_id, work in historical_resource_work_by_id.items()
    }
    resource_candidates: dict[str, ResourceCandidate] = {}
    resource_selection_events: dict[str, str] = {}
    resource_states: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {}
    for requirement in resource_requirements:
        work = resource_work_by_requirement.get(requirement.requirement_id)
        if work is None:
            resource_states[requirement.requirement_id] = (
                "needs_selection"
                if requirement.requirement_id in dependency_pending
                and len(selected_recipe_rows) > 1
                else "missing",
                tuple(item[1].candidate_id for item in selected_recipe_rows),
                ("request_resource_recipe_dependency",)
                if requirement.requirement_id in dependency_pending
                and len(selected_recipe_rows) > 1
                else (),
            )
            continue
        latest = (
            resource_facts_by_work[work.work_item_id][-1]
            if resource_facts_by_work[work.work_item_id]
            else None
        )
        history = resource_selection_history[work.work_item_id]
        current = next(
            (
                item
                for item in reversed(history)
                if latest is not None and item[2] == latest[0]["event_id"]
            ),
            None,
        )
        candidate = current[0] if current is not None else None
        if candidate is not None:
            resource_candidates[requirement.requirement_id] = candidate
            resource_selection_events[requirement.requirement_id] = current[1]
        if latest is None:
            resource_states[requirement.requirement_id] = (
                "needs_validation",
                (),
                ("lookup_compatible_resource_profiles",),
            )
        else:
            if latest[1].result.outcome == "stale_source":
                recipe_requirement_id = next(
                    item.requirement_id
                    for item, _, selection_event_id in selected_recipe_rows
                    if selection_event_id == work.recipe_event_id
                )
                recipe_states[recipe_requirement_id] = (
                    "conflicted",
                    (),
                    ("list_compatible_analysis_recipes",),
                )
                recipe_candidates.pop(recipe_requirement_id, None)
                recipe_selection_events.pop(recipe_requirement_id, None)
                resource_candidates.pop(requirement.requirement_id, None)
                resource_selection_events.pop(requirement.requirement_id, None)
                resource_states[requirement.requirement_id] = (
                    "conflicted",
                    (),
                    ("list_compatible_analysis_recipes",),
                )
                continue
            ids = tuple(item.candidate_id for item in latest[1].result.candidates)
            compatible = [
                item
                for item in latest[1].result.candidates
                if item.compatibility == "compatible"
            ]
            stale_candidates = {item[0].candidate_id for item in history}
            current_compatible_ids = {item.candidate_id for item in compatible}
            contradicted_selection = (
                bool(stale_candidates) and not current_compatible_ids
            )
            state = (
                "ready"
                if candidate is not None
                else "conflicted"
                if contradicted_selection
                else "needs_selection"
                if compatible
                else "typed_failure"
            )
            resource_states[requirement.requirement_id] = (
                state,
                ids,
                ("select_resource_candidate", "request_resource_selection")
                if state == "needs_selection"
                else ("lookup_compatible_resource_profiles",)
                if state == "conflicted"
                else (),
            )

    views = tuple(
        PlanningFactView(
            source_event_id=event["event_id"],
            attempt_id=fact.attempt_id,
            kind="recipe"
            if isinstance(fact, RecipeObservationRecorded)
            else "resource",
            work_item_id=fact.tool_input.work_item.work_item_id,
            tool_input=fact.tool_input.model_dump(mode="json"),
            result=fact.result.model_dump(mode="json"),
        )
        for event, fact, _ in retained
        if isinstance(fact, (RecipeObservationRecorded, ResourceObservationRecorded))
    )
    selection_views = tuple(
        PlanningSelectionView(
            source_event_id=event["event_id"],
            attempt_id=fact.attempt_id,
            kind=fact.kind,
            work_item_id=fact.work_item_id,
            candidate_id=fact.candidate_id,
            basis_event_id=fact.basis_event_id,
            authority=fact.authority,
            authority_event_id=fact.authority_event_id,
            organism_ploidy=fact.organism_ploidy,
        )
        for event, fact, _ in selections
    )
    readiness: list[RequirementReadiness] = []
    for requirement in plan.plan.requirements:
        evidence = [plan_event["event_id"], *_evidence_ids(requirement)]
        ids: tuple[str, ...] = ()
        actions: tuple[str, ...] = ()
        if requirement.kind in {
            "research_goal",
            "scientific_investigation",
            # Settled by the paper stage before the reference work.
            "cited_paper",
        }:
            state = "ready"
        elif requirement.kind == "registered_input":
            state = "ready"
            evidence.append(inspection_event["event_id"])
        elif requirement.kind in {
            "sample_identity",
            "taxon_identity",
            "reference_assembly_identity",
        }:
            if requirement.requirement_id in {
                item.requirement_id for item in reference.requirements
            }:
                state = "ready"
                evidence.extend(item.source_event_id for item in reference.facts)
            else:
                state = "missing"
        elif requirement.kind == "companion_asset":
            if requirement.requirement_id == companions.requirement_id:
                state = "ready"
                evidence.extend(item.source_event_id for item in companions.facts)
            else:
                state = "missing"
        elif requirement.kind == "analysis_recipe":
            state, ids, actions = recipe_states[requirement.requirement_id]
            work = next(
                item
                for item in recipe_works
                if item.grounding.requirement_id == requirement.requirement_id
            )
            evidence.extend(
                item.source_event_id
                for item in views
                if item.work_item_id == work.work_item_id
            )
            if requirement.requirement_id in eligibility_assertions:
                evidence.append(
                    eligibility_assertions[
                        requirement.requirement_id
                    ].assertion_event_id
                )
            if requirement.requirement_id in recipe_selection_events:
                evidence.append(recipe_selection_events[requirement.requirement_id])
        elif requirement.kind == "resource_profile":
            state, ids, actions = resource_states[requirement.requirement_id]
            work = resource_work_by_requirement.get(requirement.requirement_id)
            if work is not None:
                evidence.extend(
                    item.source_event_id
                    for item in views
                    if item.work_item_id == work.work_item_id
                )
            if requirement.requirement_id in resource_selection_events:
                evidence.append(resource_selection_events[requirement.requirement_id])
        else:
            state = "missing"
        readiness.append(
            RequirementReadiness(
                requirement_id=requirement.requirement_id,
                kind=requirement.kind,
                subject_key=requirement.subject_key,
                state=state,
                evidence_event_ids=tuple(dict.fromkeys(evidence)),
                candidate_ids=ids,
                next_actions=actions,
            )
        )
    recipe_pending = [
        item
        for item in readiness
        if item.kind == "analysis_recipe" and item.state != "ready"
    ]
    resource_pending = [
        item
        for item in readiness
        if item.kind == "resource_profile" and item.state != "ready"
    ]
    if hitl.pending_request is not None:
        status = "planning_hitl_pending"
    elif any(item.state == "typed_failure" for item in readiness):
        status = "typed_failure"
    elif recipe_pending:
        status = (
            "recipe_selection_pending"
            if recipe_pending[0].state == "needs_selection"
            else "recipe_validation_pending"
        )
    elif resource_pending:
        status = (
            "resource_selection_pending"
            if resource_pending[0].state == "needs_selection"
            else "resource_validation_pending"
        )
    elif any(item.state != "ready" for item in readiness):
        status = "typed_failure"
    else:
        status = "execution_planning_ready"
    active_recipe = next(
        (
            item
            for item in recipe_works
            if any(
                r.requirement_id == item.grounding.requirement_id
                for r in recipe_pending
            )
        ),
        None,
    )
    active_resource = next(
        (
            item
            for item in resource_works
            if any(
                r.requirement_id == item.grounding.requirement_id
                for r in resource_pending
            )
        ),
        None,
    )
    selected_recipes = tuple(
        recipe_candidates[req.requirement_id]
        for req in recipe_requirements
        if req.requirement_id in recipe_candidates
    )
    selected_recipe_bindings = tuple(
        SelectedRecipeBinding(
            requirement_id=req.requirement_id,
            subject_key=req.subject_key,
            work_item_id=next(
                item.work_item_id
                for item in recipe_works
                if item.grounding.requirement_id == req.requirement_id
            ),
            candidate=recipe_candidates[req.requirement_id],
            selection_event_id=recipe_selection_events[req.requirement_id],
        )
        for req in recipe_requirements
        if req.requirement_id in recipe_candidates
    )
    selected_resources = tuple(
        resource_candidates[req.requirement_id]
        for req in resource_requirements
        if req.requirement_id in resource_candidates
    )
    selected_resource_bindings = tuple(
        SelectedResourceBinding(
            requirement_id=req.requirement_id,
            subject_key=req.subject_key,
            work_item_id=resource_work_by_requirement[req.requirement_id].work_item_id,
            candidate=resource_candidates[req.requirement_id],
            selection_event_id=resource_selection_events[req.requirement_id],
        )
        for req in resource_requirements
        if req.requirement_id in resource_candidates
    )
    return ExecutionPlanningProjection(
        journey_id=plan_event["journey_id"],
        plan_event_id=plan_event["event_id"],
        plan_id=plan.plan.plan_id,
        revision_context_event_id=revision_context_event_id,
        revision_context=revision_context,
        recipe_work_items=recipe_works,
        recipe_tool_inputs=recipe_tool_inputs,
        resource_work_items=tuple(resource_works),
        selected_recipe_candidates=selected_recipes,
        selected_recipe_bindings=selected_recipe_bindings,
        selected_resource_candidates=selected_resources,
        selected_resource_bindings=selected_resource_bindings,
        active_recipe_work_item_id=active_recipe.work_item_id
        if active_recipe
        else None,
        active_resource_work_item_id=active_resource.work_item_id
        if active_resource
        else None,
        requirements=tuple(readiness),
        facts=views,
        selections=selection_views,
        status=status,
        unresolved_execution_constraints=tuple(
            dict.fromkeys(
                [
                    *[
                        value
                        for item in selected_recipes
                        for value in item.unresolved_constraints
                    ],
                    *[
                        value
                        for item in selected_resources
                        for value in item.unresolved_constraints
                    ],
                ]
            )
        ),
    )


__all__ = [
    "ExecutionPlanningProjection",
    "PlanningCandidateSelected",
    "PlanningFactView",
    "RecipeObservationRecorded",
    "RequirementReadiness",
    "ResourceObservationRecorded",
    "project_execution_planning",
    "validate_execution_planning_fact",
]
