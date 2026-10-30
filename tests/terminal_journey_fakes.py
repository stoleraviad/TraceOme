"""Current complete operation with supplied input identities and local output bytes.

No worker or network executes; the Journey establishes artifact/flow contracts,
not physical contact totals or scientific evidence. Executed Runs are built on
the coordinator path: the exact Execution approval is answered through
``advance``, which executes in that answer's Pass, as the product does.
"""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bio_neb.application.advancement import (
    AnswerReview,
    ExecutionReviewAnswer,
    ObserveRun,
    advance,
    observe,
    use_run_advancement_dependencies,
)
from bio_neb.application.fork import ForkRun, fork_run
from bio_neb.journey import Journey
from bio_neb.parabricks.operation import execute_parabricks
from tests.execution_approval import approve_execution
from tests.h100_stage_fakes import h100_provider
from tests.parabricks_operation_fakes import input_storage_provider
from tests.prerequisite_builds import measured_build
from tests.reference_stage_fakes import seed_cpu

# The request the evidence and Finalization prerequisites investigate. Their
# executed parent registers it, so its plan names it: a child's Investigation
# request is its parent plan's, never one supplied beside it.
EVIDENCE_REQUEST = {
    "schema_version": "bio-neb.investigation-request/2",
    "asserted_by": "researcher:owner",
    "request": "Investigate rs429358 in sample-001.",
    "target": {"rsid": "rs429358"},
}


class _StopBeforeInvestigation(BaseException):
    """A build's stop at the after-Execution child point; not a product fault."""


@contextmanager
def stopped_before_investigation():
    """Let ``advance`` verify Execution and close its child point, then stop.

    The Run's plan names its targets, so the coordinator would begin
    Investigation next. A build that wants the Run resting at its after-Execution
    child point stops it there: an interrupt that cancelled no Attempt writes
    nothing (``failure.coordinator_fault``), so the Journey ends with that closed
    Pass. A missing model credential would instead hold the Run (#107).
    """
    with patch(
        "bio_neb.application.advancement._begin_investigation",
        side_effect=_StopBeforeInvestigation,
    ):
        try:
            yield
        except _StopBeforeInvestigation:
            return
    raise AssertionError("the build never reached Investigation")


@measured_build("executed")
def execute_terminal(
    root: Path,
    *,
    output_payloads=None,
    investigation_request=None,
    scientific_intent=None,
    scientific_rsid=None,
    failed_turns=None,
    artifact_verification=None,
):
    """Approve a Run at root with terminal launch choices through ``advance``.

    The Run's plan names its targets (the scripted prompt's, or the registered
    ``investigation_request`` file's), so the coordinator would begin
    Investigation after the verified Execution; this build stops it first
    (``stopped_before_investigation``). The Run rests at its after-Execution child
    point; its Journey ends with that closed Pass.
    """
    from bio_neb.parabricks.configuration import ParabricksConfiguration
    from tests.planning_assignment_fakes import (
        execution_configuration,
        prepare_configured_run,
    )

    root.mkdir(parents=True, exist_ok=True)
    material = execution_configuration(root).model_dump(mode="json")
    # A prompt sentence naming one target gives the plan its own rsID slot.
    intent = (
        {}
        if scientific_intent is None
        else {
            "scientific_intent": scientific_intent,
            "scientific_rsid": scientific_rsid,
        }
    )
    registration, review, _ = prepare_configured_run(
        root,
        configuration=ParabricksConfiguration.model_validate(material),
        investigation_request=investigation_request,
        failed_turns=failed_turns,
        artifact_verification=artifact_verification,
        **intent,
    )
    journey_file = registration.journey_file
    executed = SimpleNamespace(journey_file=journey_file)

    def provider(plan, journal):
        return h100_provider(
            seed_cpu(input_storage_provider(plan, journal), plan, executed),
            plan,
            executed,
            output_payloads=output_payloads,
        )

    request = review.execution_hitl.pending_request
    with (
        use_run_advancement_dependencies(
            environ={}, parabricks_infrastructure_factory=provider
        ),
        stopped_before_investigation(),
    ):
        advance(
            AnswerReview(
                journey_file=journey_file,
                request_event_id=request.request_event_id,
                answer=ExecutionReviewAnswer(
                    kind="execution",
                    response=approve_execution(request.proposal),
                ),
            )
        )
    result = observe(ObserveRun(journey_file=journey_file))
    if getattr(result, "disposition", None) != "succeeded":
        raise AssertionError(result)
    return journey_file


def execute_admitted_terminal(approved, *, output_payloads=None):
    """Execute the caller's existing approval at the Execution operation seam."""

    def provider(plan, journal):
        return h100_provider(
            seed_cpu(input_storage_provider(plan, journal), plan, approved),
            plan,
            approved,
            output_payloads=output_payloads,
        )

    state = execute_parabricks(approved, infrastructure_factory=provider)
    if state.disposition != "succeeded":
        raise AssertionError(state)
    return approved.journey_file


def _shared_executed(investigation_request=None) -> Path:
    """The running module's read-only executed Run (`tests/run_prerequisites.py`),
    planned for ``investigation_request`` when given."""
    from tests.run_prerequisites import prerequisite

    choices = (
        {}
        if investigation_request is None
        else {"investigation_request": investigation_request}
    )
    return prerequisite("executed", **choices).journey_file


def published_continuation(root: Path, *, parent_file: Path | None = None) -> Path:
    """Publish the scientific continuation compiled from an executed parent's Execution."""
    from bio_neb.investigation import scientific_continuation

    if parent_file is None:
        parent_file = _shared_executed()
    root.mkdir(parents=True, exist_ok=True)
    output = root / "scientific-continuation.json"
    scientific_continuation.publish_scientific_continuation(
        scientific_continuation.PublishScientificContinuationCommand(
            execution_journey_file=str(parent_file), output=str(output)
        )
    )
    return output


@measured_build("forked-child")
def forked_child(
    root: Path, *, parent_file: Path | None = None, investigation_request=None
) -> Journey:
    """Fork an idle child at an executed parent's after-Execution point.

    Without parent_file the parent is the module's shared executed Run, planned
    for ``investigation_request`` (a registered request document) when given:
    the child investigates its parent plan's request (#342). The parent is only
    read.
    """
    if parent_file is None:
        parent_file = _shared_executed(investigation_request)
    elif investigation_request is not None:
        raise ValueError("a supplied parent already planned its request")
    receipt = fork_run(
        ForkRun(
            parent_journey_file=str(parent_file), child_run_root=str(root / "child")
        )
    )
    return Journey.open(Path(receipt.child_journey_file))
