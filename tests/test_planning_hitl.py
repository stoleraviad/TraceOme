from __future__ import annotations

import unittest
import uuid

from pydantic import ValidationError

from bio_neb.parabricks.planning_hitl import (
    ClarifyPlanningHitlResponse,
    ConfirmPlanningHitlResponse,
    PlanningHitlCandidate,
    PlanningHitlCandidateRecommendation,
    PlanningHitlRequested,
    PlanningHitlResponseRefused,
    SelectPlanningHitlResponse,
    StopPlanningHitlResponse,
    evaluate_planning_hitl_response,
    make_planning_hitl_responded,
    planning_hitl_request_view,
    project_planning_hitl,
)
from tests.reference_fakes import payload

PLAN_ID = "plan_" + "a" * 64
REQUIREMENT_ID = "requirement_" + "b" * 32
SUBJECT_KEY = "reference_assembly"


def identity(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"planning-hitl-test:{label}"))


def event(
    label: str,
    event_type: str,
    payload: dict[str, object],
    parents: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "event_id": identity(label),
        "event_type": event_type,
        "parent_ids": list(parents),
        "payload": payload,
    }


def request_fact(
    attempt_id: str,
    evidence_event_id: str,
    plan_event_id: str,
) -> PlanningHitlRequested:
    candidate = PlanningHitlCandidate(
        candidate_id="ncbi-assembly:GCF_000001405.40",
        asserted_value="GCF_000001405.40",
        display_label="GRCh38.p14",
        evidence_event_ids=(evidence_event_id,),
    )
    return PlanningHitlRequested(
        attempt_id=attempt_id,
        plan_event_id=plan_event_id,
        plan_id=PLAN_ID,
        requirement_id=REQUIREMENT_ID,
        subject_key=SUBJECT_KEY,
        issue="More than one reference assembly is relevant.",
        question="Should planning continue with the recommended assembly?",
        evidence_event_ids=(evidence_event_id,),
        candidates=(candidate,),
        recommendation=PlanningHitlCandidateRecommendation(
            candidate_id=candidate.candidate_id,
            evidence_event_ids=candidate.evidence_event_ids,
        ),
    )


def pending_timeline() -> tuple[list[dict[str, object]], PlanningHitlRequested]:
    evidence = event("evidence-1", "ReferenceCandidatesObserved", {})
    plan = event(
        "plan",
        "ProposalRequirementsIdentified",
        {
            "plan": {
                "plan_id": PLAN_ID,
                "requirements": [
                    {
                        "requirement_id": REQUIREMENT_ID,
                        "subject_key": SUBJECT_KEY,
                    }
                ],
            }
        },
        (evidence["event_id"],),
    )
    attempt_id = identity("attempt-1")
    started = event(
        "attempt-start-1",
        "AttemptStarted",
        {"attempt_id": attempt_id},
        (plan["event_id"],),
    )
    request = request_fact(attempt_id, evidence["event_id"], plan["event_id"])
    requested = event(
        "request-1",
        request.event_type,
        payload(request),
        (started["event_id"],),
    )
    return [evidence, plan, started, requested], request


class PlanningHitlTests(unittest.TestCase):
    def test_independent_question_can_be_answered_while_first_remains_pending(self):
        timeline, first = pending_timeline()
        other_id = "requirement_" + "c" * 32
        timeline[1]["payload"]["plan"]["requirements"].append(
            {"requirement_id": other_id, "subject_key": "independent_output"}
        )
        started = event(
            "other-attempt",
            "AttemptStarted",
            {"attempt_id": identity("other-attempt")},
            (timeline[-1]["event_id"],),
        )
        second = first.model_copy(
            update={
                "attempt_id": identity("other-attempt"),
                "requirement_id": other_id,
                "subject_key": "independent_output",
            }
        )
        requested = event(
            "other-request", second.event_type, payload(second), (started["event_id"],)
        )
        timeline.extend((started, requested))
        response = ConfirmPlanningHitlResponse(action="confirm")
        projection = project_planning_hitl(timeline)
        self.assertEqual(len(projection.pending_requests), 2)
        self.assertEqual(
            evaluate_planning_hitl_response(
                projection, requested["event_id"], response
            ).status,
            "accept",
        )
        answered = make_planning_hitl_responded(requested["event_id"], response)
        timeline.append(
            event(
                "other-answer",
                answered.event_type,
                payload(answered),
                (requested["event_id"],),
            )
        )
        current = project_planning_hitl(timeline)
        self.assertEqual(
            [r.request_event_id for r in current.pending_requests],
            [identity("request-1")],
        )
        self.assertEqual(current.receipts[0].requirement_id, other_id)
        self.assertFalse(current.receipts[0].assertion.scientifically_verified)

    def test_request_derives_exact_action_schemas_and_contracts_are_strict(self):
        timeline, request = pending_timeline()
        view = planning_hitl_request_view(
            timeline[-1]["event_id"], request, pending=True
        )
        self.assertEqual(
            [action.action for action in view.legal_actions],
            ["confirm", "select", "clarify", "reject", "stop"],
        )
        self.assertEqual(view.plan_event_id, identity("plan"))
        self.assertEqual(view.plan_id, PLAN_ID)
        self.assertEqual(view.requirement_id, REQUIREMENT_ID)
        self.assertEqual(view.subject_key, SUBJECT_KEY)
        schemas = {
            action.action: action.response_schema for action in view.legal_actions
        }
        self.assertEqual(schemas["confirm"]["required"], ["action"])
        self.assertFalse(schemas["confirm"]["additionalProperties"])
        self.assertEqual(
            schemas["select"]["properties"]["candidate_id"]["enum"],
            ["ncbi-assembly:GCF_000001405.40"],
        )
        self.assertEqual(schemas["clarify"]["required"], ["action", "instruction"])
        self.assertEqual(schemas["stop"]["required"], ["action", "reason"])
        with self.assertRaises(ValidationError):
            ConfirmPlanningHitlResponse.model_validate(
                {"action": "confirm", "instruction": "not legal"}
            )
        with self.assertRaises(ValidationError):
            ClarifyPlanningHitlResponse(action="clarify", instruction=" ")
        with self.assertRaises(ValidationError):
            StopPlanningHitlResponse(action="stop", reason=" ")
        with self.assertRaises(ValidationError):
            SelectPlanningHitlResponse(
                action="select",
                candidate_id="ncbi-assembly:GCF_000001405.40",
                instruction=" surrounding ",
            )

    def test_projection_retains_selection_as_unverified_user_assertion(self):
        timeline, _ = pending_timeline()
        response = SelectPlanningHitlResponse(
            action="select",
            candidate_id="ncbi-assembly:GCF_000001405.40",
            instruction="Use this exact accession as my asserted preference.",
        )
        responded = make_planning_hitl_responded(timeline[-1]["event_id"], response)
        timeline.append(
            event(
                "response-1",
                responded.event_type,
                payload(responded),
                (timeline[-1]["event_id"],),
            )
        )
        projection = project_planning_hitl(timeline)
        self.assertIsNone(projection.pending_request)
        self.assertEqual(len(projection.receipts), 1)
        receipt = projection.receipts[0]
        self.assertEqual(receipt.assertion.authority, "user_assertion")
        self.assertFalse(receipt.assertion.scientifically_verified)
        self.assertFalse(receipt.assertion.authorizes_execution)
        self.assertEqual(
            receipt.assertion.selected_candidate_id,
            "ncbi-assembly:GCF_000001405.40",
        )
        self.assertEqual(receipt.assertion.asserted_value, "GCF_000001405.40")
        self.assertEqual(receipt.plan_event_id, identity("plan"))
        self.assertEqual(receipt.plan_id, PLAN_ID)
        self.assertEqual(receipt.requirement_id, REQUIREMENT_ID)
        self.assertEqual(receipt.subject_key, SUBJECT_KEY)
        self.assertEqual(receipt.assertion.plan_event_id, receipt.plan_event_id)
        self.assertEqual(receipt.assertion.plan_id, receipt.plan_id)
        self.assertEqual(receipt.assertion.requirement_id, receipt.requirement_id)
        self.assertEqual(receipt.assertion.subject_key, receipt.subject_key)

    def test_identical_is_idempotent_and_conflicting_or_stale_is_refused(self):
        timeline, _ = pending_timeline()
        response = ConfirmPlanningHitlResponse(action="confirm")
        responded = make_planning_hitl_responded(timeline[-1]["event_id"], response)
        timeline.append(
            event(
                "response-1",
                responded.event_type,
                payload(responded),
                (timeline[-1]["event_id"],),
            )
        )
        projection = project_planning_hitl(timeline)
        disposition = evaluate_planning_hitl_response(
            projection, responded.request_event_id, response
        )
        self.assertEqual(disposition.status, "identical")
        self.assertEqual(disposition.retained_receipt, projection.receipts[0])

        with self.assertRaises(PlanningHitlResponseRefused) as conflict:
            evaluate_planning_hitl_response(
                projection,
                responded.request_event_id,
                ClarifyPlanningHitlResponse(
                    action="clarify", instruction="Choose another assembly."
                ),
            )
        self.assertEqual(conflict.exception.code, "planning_hitl_response_conflict")

        with self.assertRaises(PlanningHitlResponseRefused) as stale:
            evaluate_planning_hitl_response(
                projection, identity("unknown-request"), response
            )
        self.assertEqual(stale.exception.code, "planning_hitl_response_stale")

    def test_sequential_requests_have_no_count_cap_but_only_one_may_be_pending(self):
        timeline, _ = pending_timeline()
        response = ClarifyPlanningHitlResponse(
            action="clarify", instruction="Refine the source query to the named strain."
        )
        responded = make_planning_hitl_responded(timeline[-1]["event_id"], response)
        timeline.append(
            event(
                "response-1",
                responded.event_type,
                payload(responded),
                (timeline[-1]["event_id"],),
            )
        )
        for index in range(2, 7):
            evidence = event(
                f"evidence-{index}",
                "ReferenceCandidatesObserved",
                {},
                (timeline[-1]["event_id"],),
            )
            attempt_id = identity(f"attempt-{index}")
            started = event(
                f"attempt-start-{index}",
                "AttemptStarted",
                {"attempt_id": attempt_id},
                (evidence["event_id"],),
            )
            request = request_fact(
                attempt_id,
                evidence["event_id"],
                identity("plan"),
            )
            requested = event(
                f"request-{index}",
                request.event_type,
                payload(request),
                (started["event_id"],),
            )
            timeline.extend((evidence, started, requested))
            if index < 6:
                answered = make_planning_hitl_responded(
                    requested["event_id"],
                    ConfirmPlanningHitlResponse(action="confirm"),
                )
                timeline.append(
                    event(
                        f"response-{index}",
                        answered.event_type,
                        payload(answered),
                        (requested["event_id"],),
                    )
                )
        projection = project_planning_hitl(timeline)
        self.assertEqual(len(projection.requests), 6)
        self.assertEqual(len(projection.receipts), 5)
        self.assertEqual(
            projection.pending_request.request_event_id, identity("request-6")
        )

        duplicate_pending = list(timeline)
        evidence = duplicate_pending[-3]
        duplicate_request = request_fact(
            identity("attempt-6"), evidence["event_id"], identity("plan")
        )
        duplicate_pending.append(
            event(
                "request-while-pending",
                duplicate_request.event_type,
                payload(duplicate_request),
                (duplicate_pending[-1]["event_id"],),
            )
        )
        with self.assertRaisesRegex(ValueError, "another request is pending"):
            project_planning_hitl(duplicate_pending)

    def test_request_rejects_plan_requirement_and_subject_identity_tampering(self):
        timeline, request = pending_timeline()
        mutations = (
            (
                {"plan_id": "plan_" + "c" * 64},
                "changed the immutable plan identity",
            ),
            (
                {"requirement_id": "requirement_" + "d" * 32},
                "lacks its exact plan requirement",
            ),
            ({"subject_key": "different_subject"}, "changed the requirement subject"),
        )
        for update, message in mutations:
            with self.subTest(update=update):
                changed = request.model_copy(update=update)
                changed_timeline = [
                    *timeline[:-1],
                    event(
                        "request-1",
                        changed.event_type,
                        payload(changed),
                        (timeline[-2]["event_id"],),
                    ),
                ]
                with self.assertRaisesRegex(ValueError, message):
                    project_planning_hitl(changed_timeline)

        foreign_plan = event(
            "foreign-plan",
            "ProposalRequirementsIdentified",
            timeline[1]["payload"],
        )
        changed = request.model_copy(update={"plan_event_id": foreign_plan["event_id"]})
        with self.assertRaisesRegex(ValueError, "plan event is outside its ancestry"):
            project_planning_hitl(
                [
                    foreign_plan,
                    *timeline[:-1],
                    event(
                        "request-1",
                        changed.event_type,
                        payload(changed),
                        (timeline[-2]["event_id"],),
                    ),
                ]
            )

        invalid = request.model_dump(mode="json")
        invalid["plan_event_id"] = "not-a-uuid"
        with self.assertRaises(ValidationError):
            PlanningHitlRequested.model_validate(invalid)

    def test_projection_rejects_nonancestral_candidate_evidence(self):
        timeline, request = pending_timeline()
        unrelated = event("unrelated", "ReferenceCandidatesObserved", {})
        changed = request_fact(
            request.attempt_id, unrelated["event_id"], request.plan_event_id
        )
        broken = [
            unrelated,
            *timeline[:-1],
            event(
                "request-1",
                changed.event_type,
                payload(changed),
                (timeline[-2]["event_id"],),
            ),
        ]
        with self.assertRaisesRegex(ValueError, "outside its ancestry"):
            project_planning_hitl(broken)

    def test_response_rejects_unpresented_candidate(self):
        timeline, _ = pending_timeline()
        projection = project_planning_hitl(timeline)
        with self.assertRaises(PlanningHitlResponseRefused) as illegal:
            evaluate_planning_hitl_response(
                projection,
                timeline[-1]["event_id"],
                SelectPlanningHitlResponse(
                    action="select", candidate_id="not-presented"
                ),
            )
        self.assertEqual(illegal.exception.code, "planning_hitl_response_illegal")


if __name__ == "__main__":
    unittest.main()
