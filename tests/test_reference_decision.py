"""Plain-value proof of the G4 reference junction, without effect substitutes.

A Run's prepared reference comes only from the Execution Review's bound prebuilt
index or one CPU preparation job; a miss's Review offers abort or that CPU job and
amends nothing (owner, 2026-10-29, #464).
"""

import unittest
from types import SimpleNamespace

from bio_neb.journey_path import ExecutionState, Hit, Prepared, Unresolved
from tests.retained_execution_path import path_events


def _reference_pointers():
    """The end-of-dev-cleanup Run's verified reference-index pointers."""
    return path_events()[2]["payload"]["reference"]["pointers"]


def _plan(prebuilt_index=None):
    return SimpleNamespace(
        amendable=frozenset({"gpu_class"}),
        reference=SimpleNamespace(prebuilt_index=prebuilt_index),
    )


class ReferenceDecisionTests(unittest.TestCase):
    def test_forged_index_destinations_cannot_change_answer_consequences(self):
        """The Review declares each action's destination (owner, 2026-10-29, #464)."""
        from bio_neb.journey_path import Pending

        request = {
            "request_id": "unsuitable-index",
            "reason": "candidate_incompatible",
            "legal_actions": ("abort", "approve_cpu_preparation"),
            "action_destinations": {
                "abort": "stop",
                "approve_cpu_preparation": "parabricks",
            },
        }
        Pending.model_validate(request)
        for forged in (
            {
                **request,
                "action_destinations": {
                    "abort": "stop",
                    "approve_cpu_preparation": "stop",
                },
            },
            {**request, "action_destinations": {"abort": "stop"}},
        ):
            with (
                self.subTest(forged=forged),
                self.assertRaisesRegex(ValueError, "destination"),
            ):
                Pending.model_validate(forged)

    def test_the_cpu_review_offers_no_amendment(self):
        """Owner decision 2026-10-29 (#464): the CPU preparation Review offers
        exactly abort or one CPU job; the location and input-correction
        amendments are gone, so a request or answer that names one is refused."""
        from bio_neb.journey_path import Answered, Pending

        base = {
            "request_id": "unsuitable-index",
            "reason": "candidate_incompatible",
            "legal_actions": ("abort", "approve_cpu_preparation"),
        }
        for forged in (
            {**base, "legal_actions": ("abort", "approve_cpu_preparation", "amend")},
            {**base, "amendable": ("location",)},
            {**base, "amendment_destinations": {"input_correction": "agentic"}},
        ):
            with self.subTest(forged=forged), self.assertRaises(ValueError):
                Pending.model_validate(forged)
        for answer in (
            {"action": "amend", "field": "location", "value": {}},
            {"action": "approve_cpu_preparation", "field": "location", "value": {}},
        ):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                Answered.model_validate(
                    {"request_id": "unsuitable-index", "by": "owner", **answer}
                )

    def test_a_miss_asks_only_abort_or_cpu_preparation(self):
        from bio_neb.journey_path import Miss
        from bio_neb.parabricks.decisions import reference_decision

        request = reference_decision(
            ExecutionState(
                reference=Miss(reason="candidate_incompatible"),
                latest_event_ids={"ReferenceResolved": "unsuitable-index"},
            ),
            _plan(prebuilt_index=object()),
        )
        self.assertEqual(request.legal_actions, ("abort", "approve_cpu_preparation"))
        self.assertEqual(
            request.model_dump(),
            {
                "state": "pending",
                "stage": "cpu_preparation",
                "request_id": "reference-review:unsuitable-index",
                "reason": "candidate_incompatible",
                "legal_actions": ("abort", "approve_cpu_preparation"),
                "action_destinations": {
                    "abort": "stop",
                    "approve_cpu_preparation": "parabricks",
                },
            },
        )

    def test_operation_stops_unresolved_failure_before_any_input_work(self):
        from bio_neb.failure import Invariant
        from bio_neb.parabricks.operation import next_action

        state = ExecutionState(failure_event_ids=("failed-input-step",))
        # No input work and no failed outcome: our bug, held for a fix.
        with self.assertRaises(Invariant) as unrouted:
            next_action(state, _plan())
        self.assertEqual(unrouted.exception.invariant_id, "execution_failure_unrouted")

    def test_miss_drift_and_admission_failure_require_the_declared_human_choice(self):
        from bio_neb.journey_path import (
            AdmissionFailed,
            Answered,
            Drifted,
            Miss,
            Pending,
        )
        from bio_neb.parabricks.decisions import (
            Done,
            RunStage,
            Wait,
            reference_decision,
        )

        plan = _plan(prebuilt_index=object())
        evidence = _reference_pointers()
        for reference in (
            Miss(reason="missing"),
            Drifted(reason="changed", evidence=evidence),
            AdmissionFailed(reason="unreadable", evidence=evidence),
        ):
            state = ExecutionState(
                reference=reference, latest_event_ids={"ReferenceResolved": "reference"}
            )
            request = reference_decision(state, plan)
            self.assertIsInstance(request, Pending)
            self.assertEqual(
                request.legal_actions, ("abort", "approve_cpu_preparation")
            )
            self.assertIsInstance(
                reference_decision(state.model_copy(update={"review": request}), plan),
                Wait,
            )
            for action, expected in (
                ("approve_cpu_preparation", RunStage("cpu_preparation")),
                ("abort", Done("failed")),
            ):
                answer = Answered(
                    request_id=request.request_id, action=action, by="owner"
                )
                answered = state.model_copy(update={"review": answer})
                self.assertEqual(reference_decision(answered, plan), expected)
                if action == "approve_cpu_preparation":
                    self.assertEqual(answered.status, "cpu_preparation_authorized")

    def test_only_a_bound_prebuilt_index_is_admitted_and_a_verified_reference_continues(
        self,
    ):
        """Owner decision 2026-10-29 (#464): without the Execution Review's
        bound prebuilt index the Run prepares on CPU; with it, admission
        inspects that index."""
        from bio_neb.parabricks.decisions import Continue, RunStage, reference_decision

        self.assertEqual(
            reference_decision(ExecutionState(), _plan()), RunStage("cpu_preparation")
        )
        bound = _plan(prebuilt_index=object())
        self.assertEqual(
            reference_decision(ExecutionState(reference=Unresolved()), bound),
            RunStage("admit_candidate"),
        )
        pointers = _reference_pointers()
        for reference in (
            Hit(pointers=pointers, receipt_id="hit"),
            Prepared(
                pointers=pointers, receipt_id="prepared", reference_acquisition=None
            ),
        ):
            for plan in (_plan(), bound):
                self.assertIsInstance(
                    reference_decision(ExecutionState(reference=reference), plan),
                    Continue,
                )
