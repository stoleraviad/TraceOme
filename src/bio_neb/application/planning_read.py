"""The Run prefix snapshot and its checked planning meaning, including open Attempts.

``PrefixSnapshot`` is the one public exact-prefix read: readers outside
``application`` read a Run prefix and the pieces derived from it through it.

The dependency order lives here: reference and companions, prerequisite
requirements, planning selections, then final requirement resolution. Domain
projectors retain their scientific and Review checks; no read grants authority.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any
from weakref import ref

from pydantic import Field, StrictStr

from bio_neb.contracts import FrozenContract
from bio_neb.journey import Journey, JourneyEvent, ReadOnlyDict, read_only_json
from bio_neb.paper_intake.journey import RunPapers, project_run_papers
from bio_neb.parabricks.execution_planning_journey import (
    ExecutionPlanningFact,
    ExecutionPlanningProjection,
    _project_execution_planning,
    validate_execution_planning_fact,
)
from bio_neb.parabricks.planning_hitl import (
    PlanningHitlProjection,
    project_planning_hitl,
)
from bio_neb.parabricks.proposal_journey import (
    ProposalRequirementsIdentified,
    TargetAnswerRead,
    validate_proposal_fact,
    validate_target_answer_read,
)
from bio_neb.parabricks.requirement_resolution import (
    RequirementResolutionProjection,
    _project_requirement_resolution,
)
from bio_neb.prebuilt_index.choice import PrebuiltIndexCandidate
from bio_neb.reference_discovery.reference_companions_tool import (
    CompanionCandidate,
    CompanionObserved,
    CompanionRole,
)
from bio_neb.reference_discovery.reference_facts_journey import (
    CompanionObservationRecorded,
    PrebuiltIndexFact,
    ReferenceFactsProjection,
    _project_reference_facts,
    validate_prebuilt_index_fact,
    validate_reference_facts_fact,
)
from bio_neb.reference_discovery.reference_journey import (
    ReferenceIdentityFact,
    ReferenceIdentityProjection,
    _project_reference_identity,
    validate_reference_fact,
)

if TYPE_CHECKING:
    from bio_neb.application.investigation_read import _InvestigationContext
    from bio_neb.paper_reading.journey import PaperReadings

# Facts retired with no compatibility reader (a clean cut). A Journey carrying one
# is historical evidence that the current schema cannot read.
RETIRED_JOURNEY_FACTS = frozenset(
    {
        "ExecutionAuthorityGranted",
        "AgentActionReviewRequested",
        "AgentActionReviewResponded",
        # The one Fork's RunForked origin replaced every earlier child shape.
        "RetainedExecutionContinuationEstablished",
        "CausalJourneyForkEstablished",
        "CausalEvidenceRetained",
        "CausalToolExecutionClaimed",
        "CausalToolResultPublished",
        "CausalJourneyForkCompleted",
        "CausalForkParentCompleted",
        "CausalInvestigationArtifactImported",
        "CausalInvestigationEvidenceImported",
        "CausalInvestigationCollectionImported",
        "CausalInvestigationContinuationAccepted",
        # The one-Attempt explanation's binding, replaced by per-request
        # members; its /1 artifact is refused by name too.
        "ReportExplanationBound",
    }
)


class VerifiedCompanionBinding(FrozenContract):
    work_item_id: StrictStr
    role: CompanionRole
    candidate: CompanionCandidate
    validation_event_id: StrictStr
    # The genomic FASTA's index choice: the event, and the candidate it binds (None
    # for the CPU build); absent for another role.
    prebuilt_choice_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    prebuilt_index: PrebuiltIndexCandidate | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


@dataclass(frozen=True)
class PlanningRead:
    """Correlated planning facts; absent facets have unmet prerequisites.

    Every reachable value is frozen or read-only, so readers share one read.
    """

    prefix_event_ids: tuple[str, ...]
    plan_event_id: str | None
    plan: ProposalRequirementsIdentified | None
    hitl: PlanningHitlProjection
    requirements: RequirementResolutionProjection | None
    reference: ReferenceIdentityProjection | None
    companions: ReferenceFactsProjection | None
    execution: ExecutionPlanningProjection | None
    verified_companion_bindings: tuple[VerifiedCompanionBinding, ...]
    # The Run's papers in this prefix: one fold shared by every paper Tool.
    papers: RunPapers
    # The readings of the Run's papers in this prefix: planning reads a cited
    # paper's reading for the values it plans to find there.
    paper_readings: PaperReadings


class PrefixSnapshot:
    """Exact-prefix evidence and its lazy family-specific interpretation."""

    def __init__(
        self,
        events: Iterable[JourneyEvent | Mapping[str, Any]],
        *,
        owner: Journey | None = None,
        check_model_transport: bool = True,
    ):
        self._check_model_transport = check_model_transport
        self._admitted_hitl = (
            owner.shared_checked_view("planning_review") if owner is not None else None
        )
        # An owner's evidence view and its checkpoints are already shared
        # read-only facts; caller-supplied events become private read-only
        # copies once, so prefixes and derived contexts share them.
        self.timeline = [
            event
            if isinstance(event, ReadOnlyDict)
            else read_only_json(
                event.model_dump(mode="json")
                if isinstance(event, JourneyEvent)
                else dict(event)
            )
            for event in events
        ]
        from bio_neb.agentic_journey.model_transport import (
            validate_model_transport_requests,
            validate_model_transport_responses,
        )

        if check_model_transport:
            validate_model_transport_requests(self.timeline, journey=owner)
            validate_model_transport_responses(self.timeline, journey=owner)
        self.by_id: dict[str, dict[str, Any]] = {}
        for event in self.timeline:
            if event.get("event_type") in RETIRED_JOURNEY_FACTS:
                raise ValueError(f"retired_journey_fact:{event['event_type']}")
            event_id = event.get("event_id")
            if str(uuid.UUID(event_id)) != event_id or event_id in self.by_id:
                raise ValueError("Journey has invalid or repeated event identity")
            parents = event.get("parent_ids")
            if not isinstance(parents, list) or any(
                p not in self.by_id for p in parents
            ):
                raise ValueError("Journey event has invalid backward parentage")
            self.by_id[event_id] = event
        from bio_neb.reference_discovery.reference_journey import (
            validate_source_lookup_gets,
        )

        # A planning source lookup shows exactly its own retained GETs, read over
        # the prefix whose backward parentage was just checked.
        validate_source_lookup_gets(self.by_id)
        if owner is not None:
            # What a source Tool showed is its retained bytes: an owner's
            # artifact-byte claim, which a detached event read does not make.
            from bio_neb.agentic_journey.source_answers import validate_shown_answers

            validate_shown_answers(self.timeline, journey=owner)
        self.prefix_event_ids = tuple(self.by_id)
        self._positions = {
            event_id: index for index, event_id in enumerate(self.prefix_event_ids)
        }
        # The owner whose admitted evidence this is; its prefixes share it.
        self._owner = ref(owner) if owner is not None else None
        self._prefixes: dict[str, PrefixSnapshot] = {}
        # Later readers' interpretations of this exact snapshot, keyed by reader.
        self._derived: dict[Any, Any] = {}

    @cached_property
    def ancestry(self) -> dict[str, set[str]]:
        """Each event's ancestors, built when a reader first asks."""
        ancestry: dict[str, set[str]] = {}
        for event in self.timeline:
            ancestors = set(event["parent_ids"])
            for parent in event["parent_ids"]:
                ancestors.update(ancestry[parent])
            ancestry[event["event_id"]] = ancestors
        return ancestry

    def through(self, event_id: str) -> PrefixSnapshot:
        """The snapshot ending at ``event_id``, one per event across the family.

        An owner's merged evidence only extends, so the prefix through an
        admitted event never changes: the owner holds it, and what readers
        derived from it, across appends until it refreshes.
        """
        if event_id not in self.by_id:
            raise ValueError("planning checkpoint is outside the owned prefix")
        if event_id == self.prefix_event_ids[-1]:
            return self
        owner = None if self._owner is None else self._owner()
        if owner is not None:
            return owner.derived_once(
                ("prefix_snapshot", event_id), lambda: self._prefix(event_id)
            )
        if event_id not in self._prefixes:
            self._prefixes[event_id] = self._prefix(event_id)
        return self._prefixes[event_id]

    @classmethod
    def held(
        cls, owner: Journey, events: list[Mapping[str, Any]], event_id: str
    ) -> PrefixSnapshot:
        """``owner``'s one snapshot of its merged evidence ``events`` through
        ``event_id``, the one ``through`` shares, interpreting only that prefix."""

        def build():
            position = next(
                index
                for index, event in enumerate(events)
                if event["event_id"] == event_id
            )
            prefix = cls(events[: position + 1])
            prefix._owner = ref(owner)
            return prefix

        return owner.derived_once(("prefix_snapshot", event_id), build)

    def _prefix(self, event_id: str) -> PrefixSnapshot:
        prefix = PrefixSnapshot(
            self.timeline[: self._positions[event_id] + 1],
            check_model_transport=self._check_model_transport,
        )
        prefix._owner = self._owner
        prefix._prefixes = self._prefixes
        return prefix

    def derived(self, reader):
        """``reader``'s interpretation of this exact snapshot, derived once."""
        if reader not in self._derived:
            self._derived[reader] = reader(self)
        return self._derived[reader]

    @property
    def investigation(self) -> _InvestigationContext:
        """This prefix's Investigation, Finalization and report facets, derived
        once for every reader of the snapshot."""
        from bio_neb.application.investigation_read import _InvestigationContext

        return self.derived(_InvestigationContext)

    def checked_once(self, key, check):
        """A check fixed by the owner's admitted evidence, run once per owner.

        A snapshot without an owner checks every time.
        """
        owner = None if self._owner is None else self._owner()
        return check() if owner is None else owner.derived_once(key, check)

    def before(self, event_id: str) -> PrefixSnapshot:
        if event_id not in self.by_id:
            raise ValueError("planning source is outside the owned prefix")
        index = self._positions[event_id]
        if index == 0:
            return PrefixSnapshot((), check_model_transport=self._check_model_transport)
        return self.through(self.prefix_event_ids[index - 1])

    @cached_property
    def plan_event(self) -> dict[str, Any] | None:
        events = [
            e
            for e in self.timeline
            if e.get("event_type") == "ProposalRequirementsIdentified"
        ]
        if len(events) > 1:
            raise ValueError("planning requires one immutable proposal plan")
        return events[0] if events else None

    @cached_property
    def plan(self) -> ProposalRequirementsIdentified | None:
        if self.plan_event is None:
            return None
        fact = validate_proposal_fact(self.plan_event)
        if not isinstance(fact, ProposalRequirementsIdentified):
            raise ValueError("planning proposal plan is invalid")
        return fact

    def require_plan(self) -> tuple[dict[str, Any], ProposalRequirementsIdentified]:
        if self.plan_event is None or self.plan is None:
            raise ValueError("planning requires one immutable proposal plan")
        return self.plan_event, self.plan

    @cached_property
    def hitl(self) -> PlanningHitlProjection:
        return (
            self._admitted_hitl
            if self._admitted_hitl is not None
            else project_planning_hitl(self.timeline)
        )

    @cached_property
    def target_answer_reads(self) -> dict[str, TargetAnswerRead]:
        """Planning's reading of each target answer, by the answer's event (#462)."""
        reads = {}
        for event in self.timeline:
            if (read := validate_target_answer_read(event)) is not None:
                if read.answer_event_id in reads:
                    raise ValueError("a target answer is read once")
                reads[read.answer_event_id] = read
        return reads

    @cached_property
    def reference_facts(self) -> dict[str, ReferenceIdentityFact]:
        return {
            event["event_id"]: fact
            for event in self.timeline
            if (fact := validate_reference_fact(event)) is not None
        }

    @cached_property
    def companion_facts(self) -> dict[str, CompanionObservationRecorded]:
        return {
            event["event_id"]: fact
            for event in self.timeline
            if (fact := validate_reference_facts_fact(event)) is not None
        }

    @cached_property
    def prebuilt_index_facts(self) -> dict[str, PrebuiltIndexFact]:
        """The genomic FASTA's index listings, reads and choices, in order (#464)."""
        return {
            event["event_id"]: fact
            for event in self.timeline
            if (fact := validate_prebuilt_index_fact(event)) is not None
        }

    @cached_property
    def execution_facts(self) -> dict[str, ExecutionPlanningFact]:
        return {
            event["event_id"]: fact
            for event in self.timeline
            if (fact := validate_execution_planning_fact(event)) is not None
        }

    @cached_property
    def registered_requests(self):
        """The researcher's registered request files, in order; planning places
        each (#367)."""
        from bio_neb.investigation.investigation_request import (
            project_captured_investigation_requests,
        )

        return project_captured_investigation_requests(self.timeline)

    @cached_property
    def reference(self) -> ReferenceIdentityProjection:
        return _project_reference_identity(self)

    @cached_property
    def companions(self) -> ReferenceFactsProjection:
        return _project_reference_facts(self)

    @cached_property
    def verified_companion_bindings(self) -> tuple[VerifiedCompanionBinding, ...]:
        bindings = []
        for work_id in self.companions.completed_work_item_ids:
            source_event_id, fact = next(
                (event_id, fact)
                for event_id, fact in reversed(self.companion_facts.items())
                if fact.tool_input.work_item.work_item_id == work_id
                and isinstance(fact.result, CompanionObserved)
                and fact.result.outcome == "verified"
            )
            if (
                not isinstance(fact.result, CompanionObserved)
                or len(fact.result.candidates) != 1
            ):
                raise ValueError("verified companion binding lacks exact Tool evidence")
            candidate = fact.result.candidates[0]
            if candidate.role != fact.tool_input.work_item.role:
                raise ValueError("verified companion binding changed its role")
            index = self.companions.prebuilt_index
            choice = (
                index.choice
                if index is not None and candidate.role == "genomic_fasta"
                else None
            )
            bindings.append(
                VerifiedCompanionBinding(
                    work_item_id=work_id,
                    role=candidate.role,
                    candidate=candidate,
                    validation_event_id=source_event_id,
                    prebuilt_choice_event_id=None
                    if choice is None
                    else choice.source_event_id,
                    prebuilt_index=None if choice is None else choice.candidate,
                )
            )
        return tuple(bindings)

    @cached_property
    def base_requirements(self) -> RequirementResolutionProjection:
        return _project_requirement_resolution(self, execution=None)

    @cached_property
    def execution(self) -> ExecutionPlanningProjection:
        return _project_execution_planning(self)

    @cached_property
    def reference_work_ready(self):
        return (
            self.plan is not None
            and self.reference.verified_reference is not None
            and self.companions.status == "reference_facts"
        )

    @cached_property
    def execution_planning_open(self) -> bool:
        """The one gate: reference work is ready and recipe work is planned."""
        return self.reference_work_ready and {
            "analysis_recipe",
            "resource_profile",
        }.issubset({item.kind for item in self.plan.plan.requirements})

    @cached_property
    def requirements(self) -> RequirementResolutionProjection:
        execution = (
            self.execution
            if self.execution_planning_open
            and any(
                e.get("event_type") == "PlanningCandidateSelected"
                for e in self.timeline
            )
            else None
        )
        return (
            self.base_requirements
            if execution is None
            else _project_requirement_resolution(self, execution=execution)
        )

    @cached_property
    def papers(self) -> RunPapers:
        return project_run_papers(self.timeline)

    @cached_property
    def paper_readings(self) -> PaperReadings:
        from bio_neb.paper_reading.journey import project_paper_readings

        return project_paper_readings(self.timeline)

    @cached_property
    def read(self) -> PlanningRead:
        # Recognized malformed facts must not disappear behind unavailable facets.
        reference_facts = self.reference_facts
        _ = self.companion_facts
        _ = self.prebuilt_index_facts
        execution_facts = self.execution_facts
        reference = (
            self.reference if self.plan is not None or any(reference_facts) else None
        )
        companions = (
            self.companions
            if (reference is not None and reference.verified_reference is not None)
            else None
        )
        requirements = self.requirements if self.plan is not None else None
        execution = self.execution if self.execution_planning_open else None
        if execution is None and self.reference_work_ready and any(execution_facts):
            # The gate never closes again once reference work is ready.
            raise ValueError("execution planning facts precede ready reference work")
        return PlanningRead(
            self.prefix_event_ids,
            self.plan_event["event_id"] if self.plan_event else None,
            self.plan,
            self.hitl,
            requirements,
            reference,
            companions,
            execution,
            self.verified_companion_bindings if companions is not None else (),
            self.papers,
            self.paper_readings,
        )


def read_planning_events(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
) -> PlanningRead:
    """Detached event adapter; it makes no Journey-owner artifact-byte claim."""
    return PrefixSnapshot(events).read


def _read_owned_planning_context(journey: Journey) -> PrefixSnapshot:
    from bio_neb.recording import evidence_view

    return PrefixSnapshot(evidence_view(journey), owner=journey)


def prefix_snapshot(journey: Journey) -> PrefixSnapshot:
    """One owner-private interpretation for the current semantic/journal prefix."""
    from bio_neb.recording import evidence_revision

    return journey._read_context(
        _read_owned_planning_context, revision=evidence_revision(journey)
    )


def read_planning(journey: Journey) -> PlanningRead:
    """Share this writer's checked current meaning; it refuses mutation."""
    return prefix_snapshot(journey).read


def current_planning(journey) -> PlanningRead:
    """Agent-side planning meaning for a live owner or a replayed exact prefix."""
    if isinstance(journey, Journey):
        return read_planning(journey)
    return read_planning_events(journey.read())


def planning_evidence(journey) -> list[dict[str, Any]]:
    """Checked semantic facts and physical contacts, shared read-only with Agent callers.

    A live owner returns a fresh list of its context's read-only events: one
    view per prefix, not a history copy per offered Tool. Mutating an event
    raises ``TypeError``; a caller that needs to change one copies it locally.

    A live owner supplies its transport-validated planning timeline; a
    semantic-only ``Journey.read()`` would drop the physical model contacts
    that transport evidence names. Historical replay passes detached
    dependencies whose ``read`` already is the exact merged evidence prefix;
    reopening current state would change it.
    """
    if isinstance(journey, Journey):
        return list(prefix_snapshot(journey).timeline)
    return list(journey.read())
