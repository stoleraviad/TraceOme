"""Journey-derived requirement resolution and scientific investigation handoff."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, StrictStr, model_validator

from bio_neb.contracts import FrozenContract
from bio_neb.investigation.investigation_request import (
    RSID_CANONICAL_PATTERN,
    RSID_RULE,
    CapturedInvestigationRequest,
    StatedAssembly,
    StatedChange,
    StatedPosition,
)
from bio_neb.journey import JourneyEvent, canonical_json, canonical_json_digest
from bio_neb.parabricks.proposal_information import (
    AcceptedPreParabricksEvidence,
    ContextEvidence,
    InspectedInputEvidence,
    ProposalInformationRequirement,
    RequestAssertionEvidence,
    requirement_identifier_rule,
)
from bio_neb.parabricks.slot_satisfaction import RequirementValueSource, slot_rows
from bio_neb.reference_discovery.reference_identity_tool import candidate_value
from bio_neb.reference_discovery.reference_journey import (
    ReferenceDescriptionRecorded,
    SampleAttestationRecorded,
    TaxonResolutionRecorded,
)

if TYPE_CHECKING:
    from bio_neb.application.planning_read import PrefixSnapshot
    from bio_neb.parabricks.execution_planning_journey import (
        ExecutionPlanningProjection,
    )
    from bio_neb.parabricks.proposal_journey import TargetAnswerRead


def _uuid(value: str, field: str) -> str:
    if str(uuid.UUID(value)) != value:
        raise ValueError(f"{field} must be a canonical UUID")
    return value


class RequirementUserAssertion(FrozenContract):
    """What the researcher said (a quote or an answer) and the typed value it states.

    The value is checked by its identifier rule's format only: the researcher can
    never give a verified value, so it grants no truth.
    """

    assertion_kind: Literal["research_request", "planning_hitl"]
    asserted_value: StrictStr
    identifier_rule: StrictStr | None
    value: StrictStr | None
    source_event_id: StrictStr
    evidence_event_ids: tuple[StrictStr, ...]
    scientifically_verified: Literal[False] = False

    @model_validator(mode="after")
    def canonical_events(self) -> "RequirementUserAssertion":
        _uuid(self.source_event_id, "source_event_id")
        for event_id in self.evidence_event_ids:
            _uuid(event_id, "evidence_event_id")
        return self


class RequirementCandidateEvidence(FrozenContract):
    source_event_id: StrictStr
    event_type: StrictStr
    value: StrictStr | None
    evidence_id: StrictStr | None = None
    disposition: Literal["admitted", "rejected", "unresolved"]


_RESEARCHER_SOURCES = frozenset({"research_request", "review_answer"})


class RequirementValue(FrozenContract):
    """The current settled value of one requirement, its source and its status.

    A value from the researcher (the request or a Review answer) stays asserted,
    even when a Tool binds it; a Tool result, an inspected input or an accepted
    pre-Parabricks fact is verified. It is not an Admitted fact.
    """

    value_id: StrictStr = Field(pattern=r"^requirement-value:[0-9a-f]{64}$")
    requirement_id: StrictStr
    kind: StrictStr
    subject_key: StrictStr
    value: StrictStr
    source: RequirementValueSource
    status: Literal["asserted", "verified"]
    lineage_event_ids: tuple[StrictStr, ...] = Field(min_length=1)

    @classmethod
    def create(
        cls,
        *,
        requirement_id: str,
        kind: str,
        subject_key: str,
        value: str,
        source: RequirementValueSource,
        lineage_event_ids: tuple[str, ...],
    ) -> "RequirementValue":
        material = dict(
            requirement_id=requirement_id,
            kind=kind,
            subject_key=subject_key,
            value=value,
            source=source,
            lineage_event_ids=lineage_event_ids,
            status="asserted" if source in _RESEARCHER_SOURCES else "verified",
        )
        return cls(
            value_id="requirement-value:" + canonical_json_digest(material), **material
        )

    @model_validator(mode="after")
    def exact_identity(self) -> "RequirementValue":
        if (self.status == "asserted") != (self.source in _RESEARCHER_SOURCES):
            raise ValueError("requirement value status disagrees with its source")
        material = self.model_dump(mode="json", exclude={"value_id"})
        if self.value_id != ("requirement-value:" + canonical_json_digest(material)):
            raise ValueError("requirement value identity changed its exact lineage")
        return self


class FactSlotStatus(FrozenContract):
    """A fact slot's requested and verified values, its status and source.

    ``mismatch`` flags a verified value that differs from the requested one; it
    blocks nothing by itself and is shown wherever the slot is.
    """

    requirement_id: StrictStr
    kind: StrictStr
    subject_key: StrictStr
    status: Literal["to_find", "asserted", "verified"]
    verification_source: StrictStr | None
    requested_value: StrictStr | None
    verified_value: StrictStr | None
    mismatch: bool
    summary: StrictStr

    @model_validator(mode="after")
    def coherent_status(self) -> "FactSlotStatus":
        expected = (
            "verified"
            if self.verified_value is not None
            else "asserted"
            if self.requested_value is not None
            else "to_find"
        )
        if self.status != expected or self.mismatch != (
            self.requested_value is not None
            and self.verified_value is not None
            and self.requested_value != self.verified_value
        ):
            raise ValueError("fact slot status disagrees with its values")
        if self.verified_value is not None and self.verification_source is None:
            raise ValueError("a fact slot without a source cannot be verified")
        return self


class RequirementContradiction(FrozenContract):
    requirement_id: StrictStr
    value_ids: tuple[StrictStr, ...] = Field(min_length=2)
    values: tuple[StrictStr, ...] = Field(min_length=2)


class ResolvedExecutionRequirement(FrozenContract):
    """One consequence-bearing requirement and its exact Requirement value."""

    requirement_id: StrictStr
    kind: StrictStr
    subject_key: StrictStr
    requirement_value: RequirementValue

    @model_validator(mode="after")
    def exact_requirement(self) -> "ResolvedExecutionRequirement":
        value = self.requirement_value
        if (
            self.requirement_id != value.requirement_id
            or self.kind != value.kind
            or self.subject_key != value.subject_key
        ):
            raise ValueError("execution requirement changed its requirement value")
        return self


class ExecutionRequirementResolution(FrozenContract):
    """Exact consequence-only requirement snapshot accepted by Execution."""

    resolution_id: StrictStr = Field(pattern=r"^execution-resolution:[0-9a-f]{64}$")
    journey_id: StrictStr
    plan_event_id: StrictStr
    plan_id: StrictStr
    input_set_id: StrictStr
    requirements: tuple[ResolvedExecutionRequirement, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_identity(self) -> "ExecutionRequirementResolution":
        _uuid(self.journey_id, "journey_id")
        _uuid(self.plan_event_id, "plan_event_id")
        requirement_ids = tuple(item.requirement_id for item in self.requirements)
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("execution resolution repeats a requirement")
        material = self.model_dump(mode="json", exclude={"resolution_id"})
        if self.resolution_id != (
            "execution-resolution:" + canonical_json_digest(material)
        ):
            raise ValueError("execution resolution identity changed its exact lineage")
        return self

    def one_value(self, kind: str) -> RequirementValue:
        selected = tuple(
            item.requirement_value for item in self.requirements if item.kind == kind
        )
        if len(selected) != 1:
            raise ValueError(f"execution resolution has ambiguous {kind}")
        return selected[0]


class RequirementResolutionView(FrozenContract):
    requirement_id: StrictStr
    kind: StrictStr
    subject_key: StrictStr
    meaning: StrictStr
    execution_consequence: bool
    state: Literal[
        "resolved",
        "needs_tool",
        "needs_review",
        "conflicted",
        "post_execution",
        "rejected",
    ]
    user_assertions: tuple[RequirementUserAssertion, ...]
    candidate_evidence: tuple[RequirementCandidateEvidence, ...]
    values: tuple[RequirementValue, ...]
    contradiction: RequirementContradiction | None = None
    fact_slot: FactSlotStatus | None = None


class InvestigationTarget(FrozenContract):
    """One target in the researcher's own words: the request's quote for it, or an
    rsID or a position a Review answer states, typed by what those words write: the
    rsID, or the one position and the version it is stated on (#462 F3). Text the
    planning model wrote is never one."""

    words: StrictStr
    rsid: StrictStr | None = Field(
        default=None,
        pattern=RSID_CANONICAL_PATTERN,
        exclude_if=lambda value: value is None,
    )
    # Left out when empty, so an rsID target's handoff digest is unchanged.
    position: StatedPosition | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    stated_assembly: StatedAssembly | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The gene change the model typed from the words.
    gene_change: StatedChange | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    source_event_id: StrictStr
    # Who asserts a target a cited paper's reading proposes: the paper; left out for
    # the researcher's own, so their handoff digest is unchanged.
    asserted_by: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def typed(self) -> bool:
        """Whether the words state something the Run can check: an rsID, a position
        or a typed gene change, which the Investigation places (#473 Q1 a)."""
        return (
            self.rsid is not None
            or self.position is not None
            or (self.gene_change is not None and self.gene_change.typed)
        )


class ResearchGoal(FrozenContract):
    """The research goal the request states, in the researcher's words."""

    requirement_id: StrictStr
    words: StrictStr
    source_event_id: StrictStr


class UnsettledTarget(FrozenContract):
    """A scientific investigation item still to settle before the Execution proposal:
    it names no target, or its words state no rsID or position the Run can check
    (#462 R3: a dead end is asked, never dropped in silence), or the researcher's
    answer waits for planning to read it (``answer_unread``; owner 2026-10-26: an
    answer goes back to planning's target reading). ``answer`` is the researcher's
    latest answer and ``answer_event_id`` its event, when there is one."""

    requirement_id: StrictStr
    reason: Literal["no_target", "untyped_target", "answer_unread"]
    answer: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    answer_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ScientificInvestigationItem(FrozenContract):
    requirement_id: StrictStr
    subject_key: StrictStr
    purpose: Literal[
        "known_variant",
        "literature_claim",
        "hypothesis",
        "open_question",
        "discovery_request",
        "other",
    ]
    question_or_assertion: StrictStr | None
    state: Literal["open", "grounded"]
    lineage_event_ids: tuple[StrictStr, ...] = Field(min_length=1)
    # The researcher's words for each target this item names; none while it names
    # none, or when it stands for the registered request file's targets.
    targets: tuple[InvestigationTarget, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    registered_request_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ScientificInvestigationHandoff(FrozenContract):
    """What planning hands the Investigation: the research goal and the items in
    the order the researcher wrote them (ADR 0010: planning owns the targets)."""

    handoff_id: StrictStr = Field(pattern=r"^scientific-investigation:[0-9a-f]{64}$")
    journey_id: StrictStr
    plan_event_id: StrictStr
    items: tuple[ScientificInvestigationItem, ...]
    research_goals: tuple[ResearchGoal, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def exact_identity(self) -> "ScientificInvestigationHandoff":
        material = self.model_dump(mode="json", exclude={"handoff_id"})
        if self.handoff_id != (
            "scientific-investigation:" + canonical_json_digest(material)
        ):
            raise ValueError("scientific investigation identity changed its lineage")
        return self


class RequirementResolutionProjection(FrozenContract):
    journey_id: StrictStr
    plan_event_id: StrictStr
    plan_id: StrictStr
    requirements: tuple[RequirementResolutionView, ...]
    values: tuple[RequirementValue, ...]
    contradictions: tuple[RequirementContradiction, ...]
    fact_slots: tuple[FactSlotStatus, ...]
    unresolved_execution_requirement_ids: tuple[StrictStr, ...]
    ready_for_execution_compilation: bool
    execution_resolution: ExecutionRequirementResolution | None
    scientific_investigation: ScientificInvestigationHandoff
    # Items the coordinator asks the researcher about before the Execution proposal.
    unsettled_targets: tuple[UnsettledTarget, ...] = ()

    @model_validator(mode="after")
    def coherent_readiness(self) -> "RequirementResolutionProjection":
        expected = _unresolved_execution_requirements(self.requirements)
        if expected != self.unresolved_execution_requirement_ids:
            raise ValueError("requirement readiness changed its projected requirements")
        if self.ready_for_execution_compilation != (not expected):
            raise ValueError("requirement readiness status is incoherent")
        if (
            self.execution_resolution is not None
        ) != self.ready_for_execution_compilation:
            raise ValueError(
                "execution resolution does not match requirement readiness"
            )
        if self.execution_resolution is not None and (
            self.execution_resolution.journey_id != self.journey_id
            or self.execution_resolution.plan_event_id != self.plan_event_id
            or self.execution_resolution.plan_id != self.plan_id
        ):
            raise ValueError("execution resolution changed its plan lineage")
        return self

    def one_value(self, kind: str) -> RequirementValue:
        selected = tuple(item for item in self.values if item.kind == kind)
        if len(selected) != 1:
            raise ValueError(f"requirement resolution has ambiguous {kind}")
        return selected[0]


# A cited paper is the paper stage's, before the reference work; no Execution
# input depends on it.
_NON_EXECUTION_KINDS = frozenset(
    {"research_goal", "scientific_investigation", "cited_paper"}
)


def _unresolved_execution_requirements(
    requirements: tuple[RequirementResolutionView, ...],
) -> tuple[str, ...]:
    """Consequence-bearing requirements not yet resolved; and every target once the
    researcher has rejected them all, since nothing is left to investigate."""
    targets = [item for item in requirements if item.kind == "scientific_investigation"]
    every_target_rejected = bool(targets) and all(
        item.state == "rejected" for item in targets
    )
    return tuple(
        item.requirement_id
        for item in requirements
        if (item.execution_consequence and item.state != "resolved")
        or (every_target_rejected and item.kind == "scientific_investigation")
    )


def _target_answer(requirement: ProposalInformationRequirement, planning):
    """The researcher's latest answer that names or keeps this item's targets."""
    return next(
        (
            receipt
            for receipt in reversed(planning.receipts)
            if receipt.requirement_id == requirement.requirement_id
            and receipt.response.action in {"clarify", "select", "confirm"}
        ),
        None,
    )


def _stated_reference(
    requirements: Sequence[ProposalInformationRequirement],
) -> StatedAssembly | None:
    """The Run's reference version as the request states it: the accession the
    researcher asserted, with the words that state it (#462 F3), or None."""
    for requirement in requirements:
        if (
            requirement.kind == "reference_assembly_identity"
            and requirement.knowledge_state == "asserted"
            and requirement.value
        ):
            quote = next(
                (
                    item
                    for item in requirement.evidence
                    if isinstance(item, RequestAssertionEvidence)
                ),
                None,
            )
            if quote is not None:
                return StatedAssembly(
                    name=requirement.value,
                    cited_line=" ".join(quote.exact_text.split()),
                )
    return None


def _target(
    words: str,
    rsid: str | None,
    site: StatedPosition | None,
    source_event_id: str,
    reference: StatedAssembly | None,
    version: str | None,
    gene_change: StatedChange | None = None,
) -> InvestigationTarget:
    """One target as the planning model typed it from its words (#472 Q1 a): its rsID
    and its site, which grounding found in the words; no code reads either out of
    them. A site is on ``version``, the version the planning model named for it (#462
    SP4: kept as named, unchecked; the placement lookup resolves it), else on the
    request's stated reference version, else on none (the owner's no-version rule)."""
    position = site
    stated = None
    if position is not None:
        stated = (
            StatedAssembly(name=version, cited_line=" ".join(words.split()))
            if version
            else reference
        )
    return InvestigationTarget(
        words=words,
        rsid=rsid,
        position=position,
        stated_assembly=stated,
        gene_change=gene_change,
        source_event_id=source_event_id,
    )


def _paper_targets(
    context: PrefixSnapshot, requirement: ProposalInformationRequirement
) -> tuple[tuple[InvestigationTarget, ...], bool] | None:
    """The targets the readings of the papers this item cites propose, each asserted
    by its paper (#383 A2), and whether a reading is still due: None when the item
    cites no paper the plan reads. The reading proposes them in the paper's own words
    and typed values (``paper-targets/1``); the Investigation resolves each one, as
    it resolves a prompt's (owner, 2026-10-26: one flow whatever the source)."""
    from bio_neb.paper_reading.planning import planned_readings, value_papers
    from bio_neb.target_request.paper_targets import PaperTargets

    cited = value_papers(context, requirement)
    planned = [r for r in planned_readings(context) if r.paper_event_id in cited]
    if not planned:
        return None
    if not all(reading.answered for reading in planned):
        return (), True
    targets = []
    for reading in planned:
        answer_event_id = reading.reading.answer_event_id
        proposed = context.by_id[answer_event_id]["payload"]["targets"]
        if proposed is None:
            continue
        paper = PaperTargets.model_validate(proposed)
        targets.extend(
            InvestigationTarget(
                words=target.request.request_text,
                rsid=target.request.rsid,
                position=target.request.position,
                stated_assembly=target.request.stated_assembly,
                gene_change=target.request.gene_change,
                source_event_id=answer_event_id,
                asserted_by=paper.asserted_by,
            )
            for target in paper.targets
        )
    return tuple(targets), False


def _investigation_targets(
    requirement: ProposalInformationRequirement,
    planning,
    registered: Sequence[CapturedInvestigationRequest],
    reference: StatedAssembly | None,
    reads: Mapping[str, TargetAnswerRead],
) -> tuple[tuple[InvestigationTarget, ...], str | None, Any]:
    """This item's targets in the researcher's words, or the registered request
    it stands for (one per item, #367), and the latest clarifying answer, if any.
    An answer's targets are the ones planning's model read in it (``reads``, by
    answer event; #462, owner 2026-10-26), none while it is unread; otherwise an
    asserted item's quote is the target. A to_find item's quote only says what to
    find, and an item's ``meaning`` is the model's text: neither is ever a target, so
    such an item names none until the researcher answers."""
    cited = next(
        (
            file.source_event_id
            for file in registered
            if any(
                isinstance(item, ContextEvidence)
                and item.source_event_id == file.source_event_id
                for item in requirement.evidence
            )
        ),
        None,
    )
    if cited is not None:
        return (), cited, None
    answer = _target_answer(requirement, planning)
    if answer is not None and answer.response.action == "clarify":
        read = reads.get(answer.response_event_id)
        return (
            ()
            if read is None
            else tuple(
                _target(
                    target.quote,
                    target.rsid,
                    target.site,
                    answer.response_event_id,
                    reference,
                    target.stated_version,
                    target.gene_change,
                )
                for target in read.targets
            ),
            None,
            answer,
        )
    quote = next(
        (
            item
            for item in requirement.evidence
            if isinstance(item, RequestAssertionEvidence)
        ),
        None,
    )
    if quote is None or requirement.knowledge_state != "asserted":
        return (), None, None
    return (
        (
            _target(
                quote.exact_text,
                (
                    requirement.value
                    if requirement.identifier_rule == RSID_RULE.name
                    else None
                ),
                requirement.site,
                quote.source_event_id,
                reference,
                requirement.stated_version,
                requirement.gene_change,
            ),
        ),
        None,
        None,
    )


def _unsettled_targets(
    items: list[ScientificInvestigationItem],
    views: list[RequirementResolutionView],
    answers: Mapping[str, Any],
    reads: Mapping[str, TargetAnswerRead],
    awaiting: frozenset[str] = frozenset(),
) -> tuple[UnsettledTarget, ...]:
    """Items still to settle before the Execution proposal: an item whose answer
    planning has not read yet, one that names no target, or one whose words state no
    rsID or position the Run can check (#462 R3). A rejected item, and one standing for
    a registered request file, is left out; a target the request names beside the
    file is the researcher's own ask and needs no confirmation (#462 F2, owner Q1 a).
    An item whose cited paper is still to read (``awaiting``) waits for its reading,
    which proposes its targets (#383 A2)."""
    states = {view.requirement_id: view.state for view in views}
    unsettled = []
    for item in items:
        if (
            states[item.requirement_id] == "rejected"
            or item.requirement_id in awaiting
            or item.registered_request_event_id
            or (item.targets and all(target.typed for target in item.targets))
        ):
            continue
        answer = answers.get(item.requirement_id)
        unsettled.append(
            UnsettledTarget(
                requirement_id=item.requirement_id,
                reason=(
                    "answer_unread"
                    if answer is not None and answer.response_event_id not in reads
                    else "untyped_target"
                    if item.targets
                    else "no_target"
                ),
                answer=None if answer is None else answer.response.instruction,
                answer_event_id=None if answer is None else answer.response_event_id,
            )
        )
    return tuple(unsettled)


def _lineage_from_requirement(
    requirement: ProposalInformationRequirement,
) -> tuple[str, ...]:
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


def _assertions(
    requirement: ProposalInformationRequirement, planning
) -> tuple[RequirementUserAssertion, ...]:
    values: list[RequirementUserAssertion] = []
    for evidence in requirement.evidence:
        if isinstance(evidence, RequestAssertionEvidence):
            values.append(
                RequirementUserAssertion(
                    assertion_kind="research_request",
                    asserted_value=evidence.exact_text,
                    identifier_rule=requirement.identifier_rule,
                    value=(
                        requirement.value
                        if requirement.knowledge_state == "asserted"
                        else None
                    ),
                    source_event_id=evidence.source_event_id,
                    evidence_event_ids=(evidence.source_event_id,),
                )
            )
    rule = requirement_identifier_rule(requirement.kind)
    for receipt in planning.receipts:
        if receipt.requirement_id != requirement.requirement_id:
            continue
        # A selected candidate carries its typed value in its id; an answer's words
        # are typed by the stage that uses them, never read here.
        selected = receipt.assertion.selected_candidate_id
        value = None if selected is None else candidate_value(selected)
        values.append(
            RequirementUserAssertion(
                assertion_kind="planning_hitl",
                asserted_value=receipt.assertion.asserted_value,
                identifier_rule=None if value is None or rule is None else rule.name,
                value=value,
                source_event_id=receipt.response_event_id,
                evidence_event_ids=receipt.assertion.evidence_event_ids,
            )
        )
    return tuple(values)


def _requirement_value(
    requirement: ProposalInformationRequirement,
    *,
    value: str,
    source: RequirementValueSource,
    lineage: tuple[str, ...],
) -> RequirementValue:
    return RequirementValue.create(
        requirement_id=requirement.requirement_id,
        kind=requirement.kind,
        subject_key=requirement.subject_key,
        value=value,
        source=source,
        lineage_event_ids=lineage,
    )


def _slot_rows(requirement: ProposalInformationRequirement):
    return slot_rows(
        requirement.kind,
        requirement.knowledge_state,
        requirement.validation.kind,
        requirement.acquisition_route,
    )


def _base_values(
    requirement: ProposalInformationRequirement, planning
) -> tuple[RequirementValue, ...]:
    lineage = _lineage_from_requirement(requirement)
    sources = {row.source for row in _slot_rows(requirement)}
    if (
        "observed_input" in sources
        and requirement.value is not None
        and any(
            isinstance(item, InspectedInputEvidence) for item in requirement.evidence
        )
    ):
        return (
            _requirement_value(
                requirement,
                value=requirement.value,
                source="observed_input",
                lineage=lineage,
            ),
        )
    if "research_request" in sources and requirement.value is not None:
        return (
            _requirement_value(
                requirement,
                value=requirement.value,
                source="research_request",
                lineage=lineage,
            ),
        )
    accepted_fact = next(
        (
            item
            for item in requirement.evidence
            if isinstance(item, AcceptedPreParabricksEvidence)
        ),
        None,
    )
    if accepted_fact is not None and requirement.value is not None:
        return (
            _requirement_value(
                requirement,
                value=requirement.value,
                source="accepted_pre_parabricks",
                lineage=lineage,
            ),
        )
    return ()


def _fact_slot(
    requirement: ProposalInformationRequirement,
    assertions: tuple[RequirementUserAssertion, ...],
    values: tuple[RequirementValue, ...],
) -> FactSlotStatus | None:
    """The slot's latest requested value against the value its source verified."""

    if requirement.verification is None:
        return None
    source = requirement.verification.source
    requested = next(
        (item.value for item in reversed(assertions) if item.value is not None), None
    )
    return fact_slot_status(
        requirement_id=requirement.requirement_id,
        kind=requirement.kind,
        subject_key=requirement.subject_key,
        source=source,
        requested=requested,
        values=values,
    )


def fact_slot_status(
    *,
    requirement_id: str,
    kind: str,
    subject_key: str,
    source: str | None,
    requested: str | None,
    values: tuple[RequirementValue, ...],
    source_outcome: str | None = None,
) -> FactSlotStatus:
    """Compare one asserted slot with its independently verified values.

    ``source_outcome`` is what the source answered when it verified nothing, such
    as an rsID dbSNP does not know (#360); the slot then says so.
    """
    verified = tuple(
        dict.fromkeys(item.value for item in values if item.status == "verified")
    )
    verified_value = verified[0] if len(verified) == 1 and source else None
    mismatch = requested is not None and verified_value not in (None, requested)
    if verified_value is not None:
        summary = (
            f"requested {requested}, but {source} verified {verified_value}"
            if mismatch
            else f"verified by {source}"
        )
    elif requested is not None:
        summary = (
            "asserted by researcher, unverified"
            if source is None
            else f"asserted by researcher; {source} checked it: {source_outcome}"
            if source_outcome is not None
            else f"asserted by researcher; not yet verified by {source}"
        )
    else:
        summary = "to find" + ("" if source is None else f"; verified by {source}")
    return FactSlotStatus(
        requirement_id=requirement_id,
        kind=kind,
        subject_key=subject_key,
        status=(
            "verified"
            if verified_value is not None
            else "asserted"
            if requested is not None
            else "to_find"
        ),
        verification_source=source,
        requested_value=requested,
        verified_value=verified_value,
        mismatch=mismatch,
        summary=summary,
    )


def _reference_candidates(
    context: PrefixSnapshot, requirement: ProposalInformationRequirement
) -> tuple[
    tuple[RequirementCandidateEvidence, ...],
    tuple[RequirementValue, ...],
]:
    candidates: list[RequirementCandidateEvidence] = []
    values: list[RequirementValue] = []
    for event_id, fact in context.reference_facts.items():
        event = context.by_id[event_id]
        if fact.tool_input.grounding.requirement_id != requirement.requirement_id:
            continue
        result = fact.result.model_dump(mode="json")
        value: str | None = None
        admitted = False
        source: RequirementValueSource = "tool_result"
        if isinstance(fact, SampleAttestationRecorded):
            # The attestation binds the researcher's sample name; no source verifies it.
            value = fact.result.sample_identifier
            admitted = fact.result.status == "verified"
            source = (
                "review_answer"
                if context.by_id[fact.tool_input.assertion_event_id]["event_type"]
                == "PlanningHitlResponded"
                else "research_request"
            )
        elif isinstance(fact, TaxonResolutionRecorded):
            raw = result.get("candidates", [])
            if (
                result.get("resolution") == "resolved"
                and isinstance(raw, list)
                and len(raw) == 1
            ):
                value = str(raw[0].get("ncbi_tax_id"))
                admitted = True
        elif isinstance(fact, ReferenceDescriptionRecorded):
            raw = result.get("references", [])
            if (
                result.get("resolution") == "resolved"
                and isinstance(raw, list)
                and len(raw) == 1
            ):
                value = raw[0].get("assembly_accession")
                admitted = isinstance(value, str)
        evidence_id = result.get("evidence_id")
        candidates.append(
            RequirementCandidateEvidence(
                source_event_id=event["event_id"],
                event_type=event["event_type"],
                value=value,
                evidence_id=evidence_id if isinstance(evidence_id, str) else None,
                disposition=(
                    "admitted"
                    if admitted
                    else "rejected"
                    if value is not None
                    else "unresolved"
                ),
            )
        )
        # Only the current resolution and description are the slot's value; earlier
        # ones stay as candidate evidence (the model may look a taxon up again).
        current_taxon = context.reference.resolved_taxon
        if (
            isinstance(fact, ReferenceDescriptionRecorded)
            and event_id != context.reference.description_event_id
        ) or (
            isinstance(fact, TaxonResolutionRecorded)
            and (current_taxon is None or event_id != current_taxon.source_event_id)
        ):
            admitted = False
        if admitted and value is not None:
            lineage = tuple(
                dict.fromkeys(
                    (*fact.tool_input.grounding.source_event_ids, event["event_id"])
                )
            )
            values.append(
                _requirement_value(
                    requirement, value=value, source=source, lineage=lineage
                )
            )
    return tuple(candidates), tuple(values)


def _downstream_values(
    context: PrefixSnapshot,
    requirements: tuple[ProposalInformationRequirement, ...],
    execution: ExecutionPlanningProjection | None,
) -> dict[str, tuple[RequirementValue, ...]]:
    by_id = {item.requirement_id: item for item in requirements}
    resolved: dict[str, tuple[RequirementValue, ...]] = {}
    if context.reference.verified_reference is not None and any(
        e.get("event_type") == "CompanionObservationRecorded" for e in context.timeline
    ):
        companions = context.companions
        requirement = by_id.get(companions.requirement_id)
        if requirement is not None and companions.status == "reference_facts":
            bindings = context.verified_companion_bindings
            resolved[requirement.requirement_id] = (
                _requirement_value(
                    requirement,
                    value=canonical_json(
                        sorted(item.candidate.asset_id for item in bindings)
                    ),
                    source="tool_result",
                    lineage=tuple(
                        dict.fromkeys(
                            (
                                companions.plan_event_id,
                                *(item.validation_event_id for item in bindings),
                            )
                        )
                    ),
                ),
            )
    if execution is not None:
        for binding in (
            *execution.selected_recipe_bindings,
            *execution.selected_resource_bindings,
        ):
            requirement = by_id.get(binding.requirement_id)
            if requirement is not None:
                resolved[requirement.requirement_id] = (
                    _requirement_value(
                        requirement,
                        value=binding.candidate.candidate_id,
                        source="tool_result",
                        lineage=(execution.plan_event_id, binding.selection_event_id),
                    ),
                )
    return resolved


def project_requirement_resolution(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
) -> RequirementResolutionProjection:
    """Replay current requirements through the owned planning dependency closure."""
    from bio_neb.application.planning_read import PrefixSnapshot

    return PrefixSnapshot(events).requirements


def planned_fact_slots(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
) -> tuple[FactSlotStatus, ...]:
    """The plan's fact slot statuses; a Journey without a plan has none."""
    from bio_neb.application.planning_read import PrefixSnapshot

    context = PrefixSnapshot(events)
    return () if context.plan_event is None else context.requirements.fact_slots


def _project_requirement_resolution(
    context: PrefixSnapshot,
    *,
    execution: ExecutionPlanningProjection | None,
) -> RequirementResolutionProjection:
    plan_event, identified = context.require_plan()
    journey_id = _uuid(plan_event["journey_id"], "journey_id")
    planning = context.hitl
    downstream_values = _downstream_values(
        context, identified.plan.requirements, execution
    )
    pending_ids = {item.requirement_id for item in planning.pending_requests}
    rejected_ids = {
        item.requirement_id
        for item in planning.receipts
        if item.response.action == "reject"
    }
    views: list[RequirementResolutionView] = []
    all_values: list[RequirementValue] = []
    fact_slots: list[FactSlotStatus] = []
    contradictions: list[RequirementContradiction] = []
    investigation_items: list[ScientificInvestigationItem] = []
    answers: dict[str, Any] = {}
    awaiting: set[str] = set()
    registered = context.registered_requests
    for requirement in identified.plan.requirements:
        assertions = _assertions(requirement, planning)
        candidate_evidence, tool_values = _reference_candidates(context, requirement)
        values_by_value = {
            item.value: item
            for item in (
                *_base_values(requirement, planning),
                *tool_values,
                *downstream_values.get(requirement.requirement_id, ()),
            )
        }
        values = tuple(values_by_value.values())
        distinct = tuple(dict.fromkeys(item.value for item in values))
        contradiction = None
        if requirement.requirement_id in rejected_ids:
            state = "rejected"
        elif len(distinct) > 1:
            contradiction = RequirementContradiction(
                requirement_id=requirement.requirement_id,
                value_ids=tuple(item.value_id for item in values),
                values=distinct,
            )
            state = "conflicted"
        elif requirement.kind == "scientific_investigation":
            state = "post_execution"
        elif values:
            state = "resolved"
        elif requirement.requirement_id in pending_ids:
            state = "needs_review"
        elif requirement.acquisition_route == "ask_planning_hitl" and _slot_rows(
            requirement
        ):
            state = "needs_review"
        else:
            # A Tool or downstream fact is still due, or no source can ever
            # supply a value recorded before admission refused its shape.
            state = "needs_tool"
        view = RequirementResolutionView(
            requirement_id=requirement.requirement_id,
            kind=requirement.kind,
            subject_key=requirement.subject_key,
            meaning=requirement.meaning,
            execution_consequence=requirement.kind not in _NON_EXECUTION_KINDS,
            state=state,
            user_assertions=assertions,
            candidate_evidence=candidate_evidence,
            values=values,
            contradiction=contradiction,
            fact_slot=_fact_slot(requirement, assertions, values),
        )
        views.append(view)
        all_values.extend(values)
        if view.fact_slot is not None:
            fact_slots.append(view.fact_slot)
        if contradiction is not None:
            contradictions.append(contradiction)
        if requirement.kind == "scientific_investigation":
            lineage = tuple(
                dict.fromkeys(
                    (plan_event["event_id"], *_lineage_from_requirement(requirement))
                )
            )
            targets, registered_request_event_id, answer = (
                ((), None, None)
                if state == "rejected"
                else _investigation_targets(
                    requirement,
                    planning,
                    registered,
                    _stated_reference(identified.plan.requirements),
                    context.target_answer_reads,
                )
            )
            # A registered request file's item stands for that file's targets
            # alone: a cited paper's targets are its own item's, never the file's,
            # though a value citing no paper is found in the plan's papers.
            paper = (
                None
                if state == "rejected" or registered_request_event_id is not None
                else _paper_targets(context, requirement)
            )
            if paper is not None and answer is None:
                # A researcher's answer to the item's question outranks the paper;
                # targets the researcher named stand beside the paper's.
                proposed, due = paper
                targets = (*targets, *proposed)
                if due:
                    awaiting.add(requirement.requirement_id)
            if answer is not None:
                answers[requirement.requirement_id] = answer
            investigation_items.append(
                ScientificInvestigationItem(
                    requirement_id=requirement.requirement_id,
                    subject_key=requirement.subject_key,
                    purpose=requirement.investigation_purpose,
                    question_or_assertion=(
                        assertions[-1].asserted_value
                        if assertions
                        else requirement.value
                    ),
                    state="grounded" if values else "open",
                    lineage_event_ids=lineage,
                    targets=targets,
                    registered_request_event_id=registered_request_event_id,
                )
            )
    unresolved = _unresolved_execution_requirements(tuple(views))
    research_goals = tuple(
        ResearchGoal(
            requirement_id=requirement.requirement_id,
            words=evidence.exact_text,
            source_event_id=evidence.source_event_id,
        )
        for requirement in identified.plan.requirements
        if requirement.kind == "research_goal"
        for evidence in requirement.evidence
        if isinstance(evidence, RequestAssertionEvidence)
    )
    handoff_material = {
        "journey_id": journey_id,
        "plan_event_id": plan_event["event_id"],
        "items": [item.model_dump(mode="json") for item in investigation_items],
        **(
            {
                "research_goals": [
                    goal.model_dump(mode="json") for goal in research_goals
                ]
            }
            if research_goals
            else {}
        ),
    }
    execution_resolution = None
    if not unresolved:
        resolved_requirements = tuple(
            ResolvedExecutionRequirement(
                requirement_id=view.requirement_id,
                kind=view.kind,
                subject_key=view.subject_key,
                requirement_value=view.values[0],
            )
            for view in views
            if view.execution_consequence
        )
        execution_material = {
            "journey_id": journey_id,
            "plan_event_id": plan_event["event_id"],
            "plan_id": identified.plan.plan_id,
            "input_set_id": identified.plan.input_set_id,
            "requirements": [
                item.model_dump(mode="json") for item in resolved_requirements
            ],
        }
        execution_resolution = ExecutionRequirementResolution(
            resolution_id=(
                "execution-resolution:" + canonical_json_digest(execution_material)
            ),
            **execution_material,
        )
    return RequirementResolutionProjection(
        journey_id=journey_id,
        plan_event_id=plan_event["event_id"],
        plan_id=identified.plan.plan_id,
        requirements=tuple(views),
        values=tuple(all_values),
        contradictions=tuple(contradictions),
        fact_slots=tuple(fact_slots),
        unresolved_execution_requirement_ids=unresolved,
        ready_for_execution_compilation=not unresolved,
        execution_resolution=execution_resolution,
        scientific_investigation=ScientificInvestigationHandoff(
            handoff_id=(
                "scientific-investigation:" + canonical_json_digest(handoff_material)
            ),
            **handoff_material,
        ),
        unsettled_targets=_unsettled_targets(
            investigation_items,
            views,
            answers,
            context.target_answer_reads,
            frozenset(awaiting),
        ),
    )


__all__ = [
    "ExecutionRequirementResolution",
    "FactSlotStatus",
    "InvestigationTarget",
    "RequirementResolutionProjection",
    "RequirementResolutionView",
    "RequirementUserAssertion",
    "RequirementValue",
    "ResearchGoal",
    "ResolvedExecutionRequirement",
    "ScientificInvestigationHandoff",
    "ScientificInvestigationItem",
    "UnsettledTarget",
    "fact_slot_status",
    "planned_fact_slots",
    "project_requirement_resolution",
]
