"""Execute approved Parabricks plans through its domain-owned runner."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import Field, StrictInt

from bio_neb.contracts import FrozenContract
from bio_neb.journey_path import ExecutionBound, ExecutionState
from bio_neb.observation import live_heartbeat
from bio_neb.parabricks.decisions import Done, RunStage, Wait
from bio_neb.parabricks.execution_proposal import current_execution_approval
from bio_neb.parabricks.plan import (
    ApprovedParabricksInput,
    ExecutionPlan,
    RetainedExecution,
    plan_execution,
)
from bio_neb.parabricks.result import ParabricksResult, parabricks_result

ExecutionContinuation = Literal[
    "continues", "succeeded", "failed", "review_pending", "stopped", "interrupted"
]


class ExecutionContinuationAssessment(FrozenContract):
    """What another operation call on a bound execution does, and why."""

    continuation: ExecutionContinuation
    status: str
    execution_pass: StrictInt = Field(ge=1)


def _execution_journal_file(journey_file: str, execution_id: str, pass_: int) -> Path:
    directory = (
        Path(journey_file).resolve().parent
        / "execution-contact"
        / execution_id.rsplit(":", 1)[1]
    )
    if pass_ > 1:
        directory /= f"pass-{pass_}"
    return directory / "journal.jsonl"


def _execution_interval_started(journey_file, state, pass_):
    from bio_neb.journey_path import JourneyPath
    from bio_neb.recording import read_passes

    path = JourneyPath.open(Path(journey_file))
    if path._owner is None:
        return _execution_journal_file(
            journey_file, state.bound.execution_id, pass_
        ).exists()
    return any(
        row.domain is not None
        and row.domain.operation_id == state.bound.execution_id
        and row.domain.iteration == pass_
        for interval in read_passes(path._owner)
        for row in interval.contacts
    )


def _path_continuation(
    journey_file: str, state: ExecutionState
) -> ExecutionContinuationAssessment:
    """Admit another call contact-free; an interrupted pass needs a fresh Run.

    Answered designed stops can open their unused next pass. A journal already
    present for that pass is retained evidence, never permission to replay it.
    """
    from bio_neb.failure import ReviewAnswer
    from bio_neb.journey_path import Answered

    status = state.status
    if state.disposition == "succeeded":
        continuation = "succeeded"
    elif state.disposition == "failed":
        continuation = "failed"
    elif state.outcome is not None:
        continuation = (
            "review_pending"
            if state.frontier
            in {"failure_review_pending", "cpu_preparation_review_pending"}
            else "stopped"
        )
    elif state.bound is None:
        continuation = "continues"
    elif state.held_failure_id is not None:
        # Held for a fix: the next pass re-enters once, never a started one.
        continuation = (
            "interrupted"
            if _execution_interval_started(journey_file, state, state.pass_ + 1)
            else "continues"
        )
    elif isinstance(
        state.failure_review or state.review, (ReviewAnswer, Answered)
    ) and not _execution_interval_started(journey_file, state, state.pass_):
        continuation = "continues"
    else:
        continuation = "interrupted"
        status = "execution_interrupted_requires_new_run"
    return ExecutionContinuationAssessment(
        continuation=continuation, status=status, execution_pass=state.pass_
    )


def next_action(state: ExecutionState, plan: ExecutionPlan):
    """Choose the next execution stage or a designed review stop."""
    from bio_neb.failure import Ask
    from bio_neb.journey_path import NotStarted
    from bio_neb.parabricks.input_storage import INPUT_STORAGE

    if isinstance(state.failure_review, Ask):
        return state.failure_review
    if state.outcome is not None:
        return Done(state)
    if state.unresolved_failure_ids:
        # A failure continues only through its Review or a held re-entry; reaching
        # here is our bug, held by the failure module, never an Execution end.
        from bio_neb.failure import Invariant

        raise Invariant(
            "execution_failure_unrouted",
            "unresolved stage failure has no Review or re-entry",
            evidence={"failure_event_ids": list(state.unresolved_failure_ids)},
        )
    if isinstance(state.inputs, NotStarted):
        return RunStage(INPUT_STORAGE)
    from bio_neb.parabricks.decisions import Continue, reference_decision
    from bio_neb.parabricks.reference_stages import ADMIT_CANDIDATE, CPU_PREPARATION

    action = reference_decision(state, plan)
    if isinstance(action, RunStage):
        return RunStage(
            {"admit_candidate": ADMIT_CANDIDATE, "cpu_preparation": CPU_PREPARATION}[
                action.stage
            ]
        )
    if isinstance(action, Continue):
        from bio_neb.journey_path import Verified
        from bio_neb.parabricks.h100_stage import H100

        return Done(state) if isinstance(state.h100, Verified) else RunStage(H100)
    return action


def _make_provider(plan, journal):
    from bio_neb.providers.nebius import NebiusProvider

    return NebiusProvider(
        selection=plan.launch.context,
        endpoint_url=plan.storage.endpoint_url,
        region=plan.storage.region,
        journal=journal,
    )


def execute_parabricks(
    approved: ApprovedParabricksInput,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    infrastructure_factory=None,
) -> ParabricksResult:
    """Execute within the coordinator's interval, owning one when called directly."""
    from bio_neb.journey import Journey
    from bio_neb.journey_path import JourneyPath
    from bio_neb.recording import current_processing_pass, processing_pass

    path = JourneyPath.open(Path(approved.journey_file))
    if path._owner is None:
        return _execute_parabricks(
            approved, progress=progress, infrastructure_factory=infrastructure_factory
        )
    owner = path._owner
    existing = current_processing_pass()
    writer = (
        nullcontext() if Journey.processing_owner(owner.path) else owner.processing()
    )
    with writer, processing_pass(owner) as interval:
        try:
            result = _execute_parabricks(
                approved,
                progress=progress,
                infrastructure_factory=infrastructure_factory,
            )
        except BaseException as error:
            if existing is None:
                try:
                    interval.close("interrupted")
                except (OSError, ValueError) as closure_error:
                    error.add_note(
                        f"Pass closure could not be retained: {type(closure_error).__name__}"
                    )
            raise
        if existing is None:
            interval.close(
                "waiting" if result.disposition == "incomplete" else "completed"
            )
        return result


def _registered_failure_tuning(events):
    """The failure section this Run recorded at registration; the Journey is its authority."""
    from bio_neb.application.input_journey import validate_fact

    registrations = [
        validate_fact(event)
        for event in events
        if event["event_type"] == "RunInputsRegistered"
    ]
    if len(registrations) != 1:
        raise ValueError("execution requires its Run's one registration")
    return registrations[0].runtime_selections.failure


def _execute_parabricks(
    approved: ApprovedParabricksInput,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    infrastructure_factory=None,
) -> ParabricksResult:
    """Run approved inputs through verified H100 artifacts in one stage loop.

    The pre-execution planning prefix remains in this same Journey. The domain
    runner applies shared policy, observation and recording mechanics; historical executions cannot resume.
    """
    from bio_neb.failure import Ask, run_failure_tuning
    from bio_neb.journey_path import (
        JourneyPath,
        Outcome,
        PathFact,
        Pending,
    )
    from bio_neb.parabricks.execution_contact_accounting import ExecutionContactJournal
    from bio_neb.parabricks.runner import record_outcome, run_stage

    journey = JourneyPath.open(Path(approved.journey_file))
    if any(
        event["event_type"] == "ParabricksExecutionBound"
        for event in journey.planning_events()
    ):
        raise ValueError("legacy_execution_requires_migration")
    state = journey.state
    plan = plan_execution(
        approved,
        RetainedExecution(
            state, *current_execution_approval(journey.planning_events())
        ),
    )
    assessment = _path_continuation(approved.journey_file, state)
    if assessment.continuation == "interrupted":
        raise ValueError("unfinished_execution_requires_new_run")
    if assessment.continuation != "continues":
        return parabricks_result(journey)
    from bio_neb.recording import current_processing_pass

    interval = current_processing_pass()
    if interval is None:
        raise ValueError("historical_execution_is_read_only")
    interval.start()
    journal_file = interval.journal.path
    if state.held_failure_id is not None:
        state = journey.reenter(state)
    if state.bound is None:
        _, state = journey.path(
            state,
            PathFact(
                event_type="ExecutionBound",
                kind="authority",
                grounding="human",
                **{"pass": 1},
                payload=ExecutionBound(
                    execution_id=plan.identity.execution_id,
                    authority_id=plan.identity.authority_id,
                    proposal_id=plan.identity.proposal_id,
                    run_id=plan.identity.journey_id,
                    approved_input=approved.model_dump(mode="json"),
                ),
            ),
        )
    from bio_neb.parabricks.execution_contact_accounting import (
        retained_contact_unit_count,
    )

    prior_units = retained_contact_unit_count(journey)
    journal = ExecutionContactJournal(
        journal_file,
        plan.identity.execution_id,
        execution_pass=state.pass_,
        unit_offset=prior_units,
        processing_interval=interval,
        owner_event_id=state.latest_event_ids["ExecutionBound"],
    )
    try:
        with (
            live_heartbeat(
                progress,
                plan.policies.heartbeat_interval_seconds,
                kind="execution_flow",
            ) as live,
            run_failure_tuning(_registered_failure_tuning(journey.planning_events())),
        ):
            provider = (infrastructure_factory or _make_provider)(plan, journal)
            while True:
                action = next_action(state, plan)
                match action:
                    case RunStage(stage):
                        state = run_stage(
                            state,
                            plan,
                            stage,
                            provider,
                            journey=journey,
                            journal=journal,
                            live=live,
                        )
                    case Pending():
                        from bio_neb.parabricks.review import request_review

                        state = request_review(
                            state,
                            plan,
                            action,
                            journey=journey,
                            contact_journal=str(journal_file),
                            journal=journal,
                        )
                        return parabricks_result(journey)
                    case Ask():
                        return parabricks_result(journey)
                    case Done(outcome):
                        if state.outcome is None:
                            failed = outcome == "failed"
                            dependencies = (
                                state.latest_event_ids[
                                    "CpuPreparationReviewAnswered"
                                    if failed
                                    else "H100ArtifactsVerified"
                                ],
                            )
                            state = record_outcome(
                                state,
                                journey,
                                PathFact(
                                    event_type="ExecutionOutcome",
                                    kind="outcome",
                                    grounding="agent_composed"
                                    if failed
                                    else "verified",
                                    **{"pass": state.pass_},
                                    rests_on=dependencies,
                                    payload=Outcome(
                                        disposition="failed" if failed else "succeeded",
                                        contact_journal=str(journal_file),
                                    ),
                                ),
                                journal=journal,
                            )
                        return parabricks_result(journey)
                    case Wait(milestone):
                        state = record_outcome(
                            state,
                            journey,
                            PathFact(
                                event_type="ExecutionOutcome",
                                kind="outcome",
                                grounding="verified",
                                **{"pass": state.pass_},
                                rests_on=(
                                    state.latest_event_ids[
                                        "ReferencePrepared"
                                        if "ReferencePrepared" in state.latest_event_ids
                                        else "ReferenceResolved"
                                    ],
                                ),
                                payload=Outcome(
                                    disposition="incomplete",
                                    milestone=milestone,
                                    contact_journal=str(journal_file),
                                ),
                            ),
                            journal=journal,
                        )
                        return parabricks_result(journey)
    finally:
        # An interrupted pass remains an inspectable prefix. Only a lawful
        # outcome requests closure; optional accounting never decides execution.
        if sys.exception() is None and state.outcome is not None:
            from bio_neb.parabricks.execution_contact_accounting import (
                retain_path_accounting,
            )

            retain_path_accounting(journey)
