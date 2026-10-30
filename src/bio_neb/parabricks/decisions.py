"""Pure actions for the Parabricks operation. Recipes are supplied by its runner."""

from dataclasses import dataclass
from typing import Any

from bio_neb.failure import Ask
from bio_neb.journey_path import Answered, Hit, Pending, Prepared, Unresolved


@dataclass(frozen=True)
class RunStage:
    stage: Any


@dataclass(frozen=True)
class Wait:
    milestone: str | None = None


@dataclass(frozen=True)
class Done:
    outcome: Any


@dataclass(frozen=True)
class Continue:
    pass


def reference_decision(state, plan):
    """The Run's reference: the Execution Review's bound prebuilt index, else one
    CPU preparation job (owner, 2026-10-29, #464). A miss asks only abort or CPU."""
    if isinstance(state.failure_review, Ask):
        return state.failure_review
    if isinstance(state.reference, (Hit, Prepared)):
        return Continue()
    if isinstance(state.review, Pending):
        return Wait()
    if isinstance(state.review, Answered):
        if state.review.action == "abort":
            return Done("failed")
        return RunStage("cpu_preparation")
    if not isinstance(state.reference, Unresolved):
        return Pending(
            request_id="reference-review:"
            + state.latest_event_ids.get("ReferenceResolved", str(state.pass_)),
            reason=state.reference.reason,
            legal_actions=("abort", "approve_cpu_preparation"),
            action_destinations={
                "abort": "stop",
                "approve_cpu_preparation": "parabricks",
            },
        )
    return RunStage(
        "admit_candidate"
        if plan.reference.prebuilt_index is not None
        else "cpu_preparation"
    )
