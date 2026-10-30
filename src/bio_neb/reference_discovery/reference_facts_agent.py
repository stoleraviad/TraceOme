"""Reference-companion Tools the planning assignment offers through its one Agent."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from bio_neb.agentic_journey.invocation import ToolContext
from bio_neb.agentic_journey.lifecycle import Activity
from bio_neb.agentic_journey.planning_preconditions import (
    COMPANION_WORK_COMPLETE,
    COMPANIONS_NOT_OPEN,
    CORRECTION_PENDING,
    QUESTION_PENDING,
)
from bio_neb.agentic_journey.tool_execution import (
    QUESTION_OPENED,
    ObservedToolSession,
    ToolExecutionAdapter,
    ToolPrecondition,
    asked_view,
    function_tool_contract,
)
from bio_neb.journey import Journey
from bio_neb.parabricks.planning_hitl import (
    PlanningHitlCandidate,
    PlanningHitlRequested,
)
from bio_neb.reference_discovery.reference_companions_tool import (
    DescribeCompanionTool,
    ReferenceCompanionAdapter,
    companion_asset_disposition_id,
)
from bio_neb.reference_discovery.reference_facts_journey import (
    ReferenceFactsProjection,
)


@dataclass
class ReferenceFactsPlannerDependencies:
    parent_activity: Activity
    attempt_id: str
    journey: Journey
    companion_adapter: ReferenceCompanionAdapter
    companion_tool: DescribeCompanionTool = field(default_factory=DescribeCompanionTool)
    observed_tool_session: ObservedToolSession | None = None
    # Galaxy's datacache, which the genomic FASTA's index Tools search and
    # read.
    galaxy_adapter: Any = None
    # The Investigation's NCBI and EVA sources, which check the known targets on a
    # suggested assembly; None is the public ones.
    variant_sources: Any = None

    def __post_init__(self) -> None:
        if str(uuid.UUID(self.attempt_id)) != self.attempt_id:
            raise ValueError("attempt_id must be canonical")


def _projection(deps: ReferenceFactsPlannerDependencies) -> ReferenceFactsProjection:
    from bio_neb.reference_discovery.reference_tools import (
        reference_facts_projection,
    )

    return reference_facts_projection(deps)


def _asset_candidates(
    projection: ReferenceFactsProjection, work_item_id: str
) -> tuple[PlanningHitlCandidate, ...]:
    evidence_by_asset: dict[str, list[str]] = {}
    for fact in projection.facts:
        if fact.work_item_id != work_item_id or fact.result.get("status") != "observed":
            continue
        include = fact.result.get("outcome") in {"multiple", "ambiguous"}
        if work_item_id in projection.conflicted_work_item_ids:
            include = fact.result.get("outcome") == "verified"
        if not include:
            continue
        for candidate in fact.result.get("candidates", []):
            if not isinstance(candidate, dict) or not isinstance(
                candidate.get("asset_id"), str
            ):
                continue
            evidence_by_asset.setdefault(candidate["asset_id"], []).append(
                fact.source_event_id
            )
    return tuple(
        PlanningHitlCandidate(
            candidate_id=companion_asset_disposition_id(work_item_id, asset_id),
            asserted_value=asset_id,
            display_label=asset_id[:300],
            evidence_event_ids=tuple(dict.fromkeys(evidence_ids)),
        )
        for asset_id, evidence_ids in evidence_by_asset.items()
    )


def _execute_planning_request(
    context: ToolContext[ReferenceFactsPlannerDependencies],
    make_fact: Callable[[], PlanningHitlRequested],
    *,
    stage: Literal[
        "request_companion_asset_selection",
        "request_companion_refinement",
    ],
) -> dict[str, Any]:
    request_event: dict[str, Any] | None = None

    def typed_operation() -> dict[str, Any]:
        nonlocal request_event
        deps = context.deps
        fact = make_fact()
        activity = deps.parent_activity.start_child(
            adapter="deterministic",
            stage=stage,
        )
        accepted = activity.record(fact)
        request_event = accepted
        activity.succeeded(accepted)
        return accepted

    return context.deps.observed_tool_session.execute(
        context.tool_call_id,
        typed_operation,
        ToolExecutionAdapter(
            identity="deterministic",
            stage_result_event=lambda _result: (
                request_event if request_event is not None else {}
            ),
            artifact_digests=lambda _result: (),
        ),
    )


def request_companion_asset_selection(
    context: ToolContext[ReferenceFactsPlannerDependencies],
    work_item_id: str,
    basis_event_id: str,
) -> dict[str, Any]:
    """The answer is a selection that describe_companion revalidates, not
    compatibility proof.
    """
    del basis_event_id
    deps = context.deps

    def make_fact() -> PlanningHitlRequested:
        projection = _projection(deps)
        candidates = _asset_candidates(projection, work_item_id)
        if len(candidates) < 2:
            raise ValueError(
                "companion asset selection lacks multiple exact candidates"
            )
        evidence_event_ids = tuple(
            dict.fromkeys(
                event_id
                for candidate in candidates
                for event_id in candidate.evidence_event_ids
            )
        )
        return PlanningHitlRequested(
            attempt_id=deps.attempt_id,
            plan_event_id=projection.plan_event_id,
            plan_id=projection.plan_id,
            requirement_id=projection.requirement_id,
            subject_key=projection.subject_key,
            issue="Multiple current companion asset identities require disposition.",
            question=(
                "Which exact source-owned asset identity should be revalidated for this "
                "assembly and role? Selection is not compatibility proof."
            ),
            evidence_event_ids=evidence_event_ids,
            candidates=candidates,
        )

    return _execute_planning_request(
        context, make_fact, stage="request_companion_asset_selection"
    )


def request_companion_refinement(
    context: ToolContext[ReferenceFactsPlannerDependencies],
    work_item_id: str,
    evidence_event_id: str,
) -> dict[str, Any]:
    """The source is not contacted again with an unchanged request."""
    deps = context.deps

    def make_fact() -> PlanningHitlRequested:
        projection = _projection(deps)
        work = next(
            item for item in projection.work_items if item.work_item_id == work_item_id
        )
        latest = next(
            fact
            for fact in reversed(projection.facts)
            if fact.work_item_id == work_item_id
        )
        return PlanningHitlRequested(
            attempt_id=deps.attempt_id,
            plan_event_id=projection.plan_event_id,
            plan_id=projection.plan_id,
            requirement_id=projection.requirement_id,
            subject_key=projection.subject_key,
            issue=(
                "Companion observation cannot progress from "
                f"outcome={latest.result.get('outcome')}."
            ),
            question=(
                f"Clarify the required {work.role} companion for exact "
                f"{work.reference.assembly_accession}; unchanged source contact will "
                "not be repeated."
            ),
            evidence_event_ids=(evidence_event_id,),
            candidates=(),
        )

    return _execute_planning_request(
        context, make_fact, stage="request_companion_refinement"
    )


# This domain's Tool rows (ADR 0040): handler, row, bound values and refusal
# facts sit together, and the loop assembles every domain's rows into the one
# inventory (`tool_execution.tool_row_modules`).
TOOL_ROWS = (
    function_tool_contract(
        "request_companion_asset_selection",
        "planning_agent",
        ("work_item_id: str", "basis_event_id: str"),
        "PlanningHitlRequested",
        "deterministic",
        "PlanningHitlRequested",
        None,
        None,
        asked_view(),
        "generic_mechanical_failure_only",
        stages=("companions",),
        examples=("{} (the server supplies the work item and its basis)",),
        opens_question=True,
        preconditions=(
            QUESTION_OPENED,
            CORRECTION_PENDING,
            COMPANIONS_NOT_OPEN,
            QUESTION_PENDING,
            COMPANION_WORK_COMPLETE,
            ToolPrecondition(
                "companion_selection_not_needed",
                "companion observation",
                "A selection is asked only when the latest observation found several or ambiguous assets and none was selected yet.",
            ),
        ),
    ),
    function_tool_contract(
        "request_companion_refinement",
        "planning_agent",
        ("work_item_id: str", "evidence_event_id: str"),
        "PlanningHitlRequested",
        "deterministic",
        "PlanningHitlRequested",
        None,
        None,
        asked_view(),
        "generic_mechanical_failure_only",
        stages=("companions",),
        examples=("{} (the server supplies the work item and its evidence)",),
        opens_question=True,
        preconditions=(
            QUESTION_OPENED,
            CORRECTION_PENDING,
            COMPANIONS_NOT_OPEN,
            QUESTION_PENDING,
            COMPANION_WORK_COMPLETE,
            ToolPrecondition(
                "companion_refinement_not_needed",
                "companion observation",
                "Refinement is asked only when the latest observation was partial, absent or incompatible.",
            ),
        ),
    ),
)


__all__ = [
    "TOOL_ROWS",
    "ReferenceFactsPlannerDependencies",
    "request_companion_asset_selection",
    "request_companion_refinement",
]
