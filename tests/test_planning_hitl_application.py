from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from bio_neb.agentic_journey.attempt_journey import AttemptStarted
from bio_neb.journey import Journey
from bio_neb.parabricks.planning_commands import (
    RespondPlanningHitlCommand,
    respond_planning_hitl,
)
from bio_neb.parabricks.planning_hitl import (
    ClarifyPlanningHitlResponse,
    ConfirmPlanningHitlResponse,
    PlanningHitlCandidate,
    PlanningHitlCandidateRecommendation,
    PlanningHitlRequested,
    PlanningHitlResponseRefused,
    StopPlanningHitlResponse,
)
from tests.reference_fakes import payload

PLAN_ID = "plan_" + "a" * 64
REQUIREMENT_ID = "requirement_" + "b" * 32
SUBJECT_KEY = "reference_assembly"


def identity(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"planning-hitl-test:{label}"))


def pending_journey(root: Path) -> tuple[Journey, dict[str, object]]:
    journey = Journey.create(root)
    evidence = journey.record("ReferenceCandidatesObserved", {})
    plan = journey.record(
        "ProposalRequirementsIdentified",
        {
            "plan": {
                "plan_id": PLAN_ID,
                "requirements": [
                    {
                        "requirement_id": REQUIREMENT_ID,
                        "kind": "reference_assembly_identity",
                        "subject_key": SUBJECT_KEY,
                    }
                ],
            }
        },
        [evidence["event_id"]],
    )
    attempt_id = identity(f"application-attempt:{journey.journey_id}")
    started = journey.record(
        "AttemptStarted",
        payload(
            AttemptStarted(attempt_id=attempt_id, input_set_id=identity("input-set"))
        ),
        [plan["event_id"]],
    )
    candidate = PlanningHitlCandidate(
        candidate_id="ncbi-assembly:GCF_000001405.40",
        asserted_value="GCF_000001405.40",
        display_label="GRCh38.p14",
        evidence_event_ids=(evidence["event_id"],),
    )
    request = PlanningHitlRequested(
        attempt_id=attempt_id,
        plan_event_id=plan["event_id"],
        plan_id=PLAN_ID,
        requirement_id=REQUIREMENT_ID,
        subject_key=SUBJECT_KEY,
        issue="More than one reference assembly is relevant.",
        question="Should planning continue with the recommended assembly?",
        evidence_event_ids=(evidence["event_id"],),
        candidates=(candidate,),
        recommendation=PlanningHitlCandidateRecommendation(
            candidate_id=candidate.candidate_id,
            evidence_event_ids=candidate.evidence_event_ids,
        ),
    )
    requested = journey.record(
        request.event_type, payload(request), [started["event_id"]]
    )
    return journey, requested


class PlanningHitlApplicationTests(unittest.TestCase):
    def test_writer_refuses_semantically_invalid_answer_before_bytes(self):
        with TemporaryDirectory() as temporary:
            journey, requested = pending_journey(Path(temporary))
            before = journey.path.read_bytes()
            bad = {
                "request_event_id": identity("unknown"),
                "response": {"action": "reject", "reason": "decline"},
            }
            for write in (journey.preview, journey.record_against):
                with self.subTest(write=write.__name__), self.assertRaises(ValueError):
                    write(
                        journey.project(),
                        "PlanningHitlResponded",
                        bad,
                        [requested["event_id"]],
                    )
                self.assertEqual(journey.path.read_bytes(), before)
            good = {**bad, "request_event_id": requested["event_id"]}
            journey.record_against(
                journey.project(),
                "PlanningHitlResponded",
                good,
                [requested["event_id"]],
            )
            before = journey.path.read_bytes()
            with self.assertRaises(ValueError):
                journey.record_against(
                    journey.project(),
                    "PlanningHitlResponded",
                    good,
                    [requested["event_id"]],
                )
            self.assertEqual(journey.path.read_bytes(), before)

    def test_reject_resolves_only_the_request_and_retains_its_history(self):
        with TemporaryDirectory() as temporary:
            journey, requested = pending_journey(Path(temporary))
            command = RespondPlanningHitlCommand.model_validate(
                {
                    "journey_file": str(journey.path),
                    "request_event_id": requested["event_id"],
                    "response": {
                        "action": "reject",
                        "reason": "Do not pursue this item.",
                    },
                }
            )
            receipt = respond_planning_hitl(command)
            retained = Journey.open(journey.path).read()
            self.assertEqual(retained[-1]["event_type"], "PlanningHitlResponded")
            self.assertFalse(any(e["event_type"] == "RunFailed" for e in retained))
            self.assertEqual(
                receipt.assertion.asserted_value, "Do not pursue this item."
            )
            self.assertFalse(receipt.assertion.scientifically_verified)
            self.assertIn(requested, retained)
            frozen = journey.path.read_bytes()
            self.assertEqual(respond_planning_hitl(command), receipt)
            self.assertEqual(journey.path.read_bytes(), frozen)

    def test_confirmed_inability_ends_with_required_evidence_unresolved(self):
        with TemporaryDirectory() as temporary:
            journey, requested = pending_journey(Path(temporary))
            command = RespondPlanningHitlCommand(
                journey_file=str(journey.path),
                request_event_id=requested["event_id"],
                response=StopPlanningHitlResponse(
                    action="stop",
                    reason="No trustworthy reference choice can be supplied.",
                ),
            )

            receipt = respond_planning_hitl(command)
            retained = Journey.open(journey.path).read()

            self.assertEqual(receipt.response.action, "stop")
            self.assertEqual(
                tuple(event["event_type"] for event in retained[-2:]),
                ("PlanningHitlResponded", "RunFailed"),
            )
            self.assertEqual(
                retained[-1]["payload"]["code"], "required_evidence_unresolved"
            )
            self.assertEqual(
                retained[-1]["payload"]["source_event_id"], receipt.response_event_id
            )
            self.assertFalse(
                any(
                    event["event_type"] == "ExecutionAuthorityGranted"
                    for event in retained
                )
            )

    def test_response_command_appends_once_and_is_byte_idempotent(self):
        with TemporaryDirectory() as temporary:
            journey, requested = pending_journey(Path(temporary))
            command = RespondPlanningHitlCommand(
                journey_file=str(journey.path),
                request_event_id=requested["event_id"],
                response=ConfirmPlanningHitlResponse(action="confirm"),
            )
            receipt = respond_planning_hitl(command)
            expected_response_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{journey.journey_id}:{requested['event_id']}:"
                    "planning-hitl-response",
                )
            )
            self.assertEqual(receipt.response_event_id, expected_response_id)
            retained = Journey.open(journey.path).read()
            self.assertEqual(retained[-1]["event_type"], "PlanningHitlResponded")
            self.assertEqual(retained[-1]["event_id"], expected_response_id)
            self.assertEqual(retained[-1]["parent_ids"], [requested["event_id"]])
            accepted_bytes = journey.path.read_bytes()

            identical = respond_planning_hitl(command)
            self.assertEqual(identical, receipt)
            self.assertEqual(journey.path.read_bytes(), accepted_bytes)

    def test_conflicting_response_refuses_without_append(self):
        with TemporaryDirectory() as temporary:
            journey, requested = pending_journey(Path(temporary))
            accepted = RespondPlanningHitlCommand(
                journey_file=str(journey.path),
                request_event_id=requested["event_id"],
                response=ConfirmPlanningHitlResponse(action="confirm"),
            )
            respond_planning_hitl(accepted)
            accepted_bytes = journey.path.read_bytes()
            command = accepted.model_copy(
                update={
                    "response": ClarifyPlanningHitlResponse(
                        action="clarify", instruction="Use another assembly."
                    )
                }
            )
            with self.assertRaises(PlanningHitlResponseRefused) as refused:
                respond_planning_hitl(command)
            self.assertEqual(refused.exception.code, "planning_hitl_response_conflict")
            self.assertEqual(journey.path.read_bytes(), accepted_bytes)

    def test_response_command_is_strict_before_journey_open(self):
        valid = {
            "journey_file": str((Path.cwd() / "journey.jsonl").resolve()),
            "request_event_id": identity("request-command-shape"),
            "response": {"action": "confirm"},
        }
        command = RespondPlanningHitlCommand.model_validate(valid)
        self.assertIsInstance(command.response, ConfirmPlanningHitlResponse)
        for update in (
            {"journey_file": "relative.jsonl"},
            {"request_event_id": "not-a-uuid"},
            {"extra": True},
        ):
            with self.subTest(update=update):
                with self.assertRaises(ValidationError):
                    RespondPlanningHitlCommand.model_validate({**valid, **update})


if __name__ == "__main__":
    unittest.main()
