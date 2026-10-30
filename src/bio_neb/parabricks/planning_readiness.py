"""Is execution planning complete? One judgement over the Journey's recorded facts.

Planning closure and the preparation compiler both read it, so a planning Attempt
closes exactly when the compiler has nothing left that the Agent can acquire; the
coordinator asks for preparation information itself (#301), and the compiler keeps
only its own integrity refusals (#217).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bio_neb.application.planning_read import PlanningRead

PlanningGapCode = Literal[
    "proposal_planning_not_ready",
    "proposal_required_evidence_unresolved",
    "investigation_targets_unsettled",
    "execution_configuration_required",
    "execution_configuration_incompatible",
]


# What the researcher supplies through the preparation Review; every other gap
# is a planning fact the Agent acquires or asks for in its own Review.
_PREPARATION_INFORMATION = frozenset(
    {
        "execution_configuration_required",
        "execution_configuration_incompatible",
    }
)


@dataclass(frozen=True)
class PlanningIncomplete:
    """The first piece of information planning still lacks."""

    code: PlanningGapCode
    details: tuple[str, ...] = ()
    status: Literal["missing_information"] = "missing_information"

    @property
    def preparation_information(self) -> bool:
        """Whether the researcher supplies it through the preparation Review."""
        return self.code in _PREPARATION_INFORMATION

    @property
    def asked_by_coordinator(self) -> bool:
        """Whether only the researcher supplies it, through a coordinator question:
        the targets, or the preparation information."""
        return (
            self.code == "investigation_targets_unsettled"
            or self.preparation_information
        )


def readiness_of(view: PlanningRead, events) -> PlanningIncomplete | None:
    """None when execution planning is complete; otherwise what it still lacks.

    The caller passes the checked planning view and evidence it already holds
    (ADR 0032); this judgement reads no Journey.

    A submitted Execution proposal was compiled from complete planning, so planning
    stays complete until the researcher's revise answer reopens it. Otherwise
    recipe and resource selection come first (the planning view), then every
    execution requirement's resolution (#207's slot table and resolver), then
    the targets the Investigation checks (#342: no GPU job runs before they
    exist), then the preparation information the researcher supplies.
    """
    from bio_neb.parabricks.execution_proposal import project_execution_hitl
    from bio_neb.parabricks.preparation_review import information_gap

    if any(e["event_type"] == "ExecutionProposalSubmitted" for e in events) and (
        project_execution_hitl(events).status != "execution_revision_pending"
    ):
        return None
    if view.execution is None or view.execution.status != "execution_planning_ready":
        return PlanningIncomplete("proposal_planning_not_ready")
    if not view.requirements.ready_for_execution_compilation:
        return PlanningIncomplete(
            "proposal_required_evidence_unresolved",
            view.requirements.unresolved_execution_requirement_ids,
        )
    if view.requirements.unsettled_targets:
        return PlanningIncomplete(
            "investigation_targets_unsettled",
            tuple(item.requirement_id for item in view.requirements.unsettled_targets),
        )
    gap = information_gap(events)
    return None if gap is None else PlanningIncomplete(*gap)


__all__ = [
    "PlanningGapCode",
    "PlanningIncomplete",
    "readiness_of",
]
