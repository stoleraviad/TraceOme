"""Deterministic Parabricks preparation from checked retained information.

The operation owns completeness, immutable package publication and proposal binding.
It performs no source acquisition and grants no execution authority.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from bio_neb.application.planning_read import PrefixSnapshot, prefix_snapshot
from bio_neb.journey import Journey
from bio_neb.parabricks.execution_package import (
    ExecutionPackageCompiled,
    compile_execution_package,
    validate_execution_package_fact,
)
from bio_neb.parabricks.execution_proposal import (
    ExecutionProposal,
    ProposalCompilationRefused,
    _compile_execution_proposal,
    project_execution_hitl,
)
from bio_neb.parabricks.planning_readiness import PlanningIncomplete, readiness_of


@dataclass(frozen=True)
class PreparationReady:
    proposal: ExecutionProposal
    status: Literal["ready"] = "ready"


@dataclass(frozen=True)
class PreparationRefused:
    code: str
    details: tuple[str, ...] = ()
    status: Literal["refused"] = "refused"


PreparationResult = PreparationReady | PlanningIncomplete | PreparationRefused


def prepare_execution(journey: Journey) -> PreparationResult:
    """Return one reviewable proposal, what planning still lacks, or a refusal.

    Completeness is planning readiness's judgement, which planning closure reads
    too (#217); once it holds, every compiler refusal is an integrity fault.
    Publication uses the shared writer only after the full candidate compiles.
    Later coordinator handoffs use this same operation.
    """
    try:
        context = prefix_snapshot(journey)
        review = project_execution_hitl(context.timeline)
        if review.pending_request is not None:
            return PreparationReady(review.pending_request.proposal)
        gap = readiness_of(context.read, context.timeline)
        if gap is not None:
            return gap
        return PreparationReady(_prepare(journey, context))
    except ProposalCompilationRefused as error:
        return PreparationRefused(error.code, error.details)


def _prepare(journey: Journey, context: PrefixSnapshot) -> ExecutionProposal:
    planned = _compile_execution_proposal(context, bind_execution_package=False)
    if planned.reference_source_admission is None:
        # The discovered genomic FASTA is the source to admit.
        from bio_neb.reference_discovery.commands import (
            admit_discovered_reference_source,
        )

        admit_discovered_reference_source(journey)
        context = prefix_snapshot(journey)
        planned = _compile_execution_proposal(context, bind_execution_package=False)
    compiled = [
        fact.package
        for event in context.timeline
        if isinstance(
            (fact := validate_execution_package_fact(event)), ExecutionPackageCompiled
        )
    ]
    if compiled and compiled[-1].source_plan.planned_proposal_id == planned.proposal_id:
        return _compile_execution_proposal(context, bind_execution_package=True)
    package = compile_execution_package(context.timeline, planned=planned)
    fact = ExecutionPackageCompiled(package=package)
    payload = fact.model_dump(mode="json", exclude={"event_type"})
    parents = list(
        dict.fromkeys(
            [
                journey.head_event_id,
                package.source_plan.checkpoint_event_id,
                package.reference.admission_event_id,
                *(
                    item.source_event_id
                    for item in package.source_plan.evidence_closure
                ),
            ]
        )
    )
    event_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL, f"{journey.journey_id}:{package.package_id}:compiled"
        )
    )
    preview = journey.preview(
        journey.project(), fact.event_type, payload, parents, event_id=event_id
    )
    proposal = _compile_execution_proposal(
        PrefixSnapshot([*context.timeline, preview.events[-1].model_dump(mode="json")]),
        bind_execution_package=True,
    )
    journey.record(fact.event_type, payload, parents, event_id=event_id)
    return proposal
