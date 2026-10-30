"""The post-approval roles ask the researcher, and a stopped item can be dismissed
(owner, #447 Q8 and its follow-ups, 2026-10-25).

One shared Tool, ``ask_researcher(found_so_far, conflict, question)``, ends the
Attempt and opens the existing Agent work Review (cause ``researcher_question``);
the continue instruction is the answer. ``dismiss`` settles only the stopped item:
a table chunk's items fold as dismissed with the reason and the other chunks go on.
"""

import tempfile
import unittest
from pathlib import Path

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from bio_neb.agentic_journey.agent_work_review import (
    ContinueAgentWorkResponse,
    DismissAgentWorkResponse,
    project_agent_work_review,
    respond_agent_work_review,
)
from bio_neb.agentic_journey.tool_execution import validate_observed_function_tool_role
from bio_neb.investigation.item_evidence import validate_item_evidence_fact
from bio_neb.investigation.vcf_table_reading import advance_table_reading
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.sequential_investigation_support import observed_collection
from tests.vcf_lookup_fakes import TableReaderModel, chunk_rows

ENVIRON = {"LOCAL_MODEL_KEY": "secret-value-not-retained"}
_ASKED = {
    "found_so_far": "Row 1 has no record at this position.",
    "conflict": "The request names a variant the table does not show.",
    "question": "Which variant did you mean for this item?",
}


def _run(journey, model, adapter):
    while advance_table_reading(
        journey, environ=ENVIRON, model_factory=model, adapter=adapter
    ):
        journey = Journey.open(journey.path)
    return Journey.open(journey.path)


def _facts(journey, event_type):
    return [
        validate_item_evidence_fact(event)
        for event in read_evidence(journey)
        if event["event_type"] == event_type
    ]


class _AsksFirst:
    """Asks once on the first chunk it is sent, then reads like the scripted reader;
    ``silent`` chunks only ever end in text instead."""

    def __init__(self, *, ask=True, silent_first=False):
        self.reader = TableReaderModel()
        self.ask = ask
        self.silent_first = silent_first
        self.first_rows = None
        self.prompts: list[str] = []
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        rows = chunk_rows(messages)
        self.prompts.append(str(messages[0]))
        if self.first_rows is None:
            self.first_rows = rows
        if rows == self.first_rows and self.ask:
            self.ask = False
            return ModelResponse(
                parts=[ToolCallPart("ask_researcher", dict(_ASKED), tool_call_id="ask")]
            )
        if rows == self.first_rows and self.silent_first:
            return ModelResponse(parts=[TextPart("I will not read this chunk.")])
        return await self.reader.respond(messages, info)


class TableReaderQuestionTests(unittest.TestCase):
    def test_the_question_opens_the_review_and_the_answer_reenters_the_chunk(self):
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = observed_collection(
                Path(temporary).resolve(),
                investigation_table={"chunk_size": 1},
                investigation_concurrency={"planner_members": 1, "table_chunks": 1},
            )
            model = _AsksFirst()
            journey = _run(journey, model.model, adapter)
            request = project_agent_work_review(read_evidence(journey)).pending_request
            assert request is not None
            # Reopen rebuilds the question's exact return (ADR 0034; Spec r1 SP1).
            validate_observed_function_tool_role(
                read_evidence(journey), agent_role="table_reader"
            )
            self.assertEqual(
                (request.agent_role, request.cause),
                ("table_reader", "researcher_question"),
            )
            self.assertEqual(
                (request.found_so_far, request.conflict, request.question),
                tuple(_ASKED.values()),
            )
            self.assertEqual(
                [item.action for item in request.legal_actions],
                ["continue", "reject", "dismiss"],
            )
            respond_agent_work_review(
                journey,
                request.request_event_id,
                ContinueAgentWorkResponse(
                    action="continue", instruction="I meant the SNV at this row."
                ),
            )
            journey = _run(Journey.open(journey.path), model.model, adapter)
            bound = _facts(journey, "InvestigationTableChunkBound")
            (fold,) = _facts(journey, "InvestigationItemEvidenceCollected")
            validate_observed_function_tool_role(
                read_evidence(journey), agent_role="table_reader"
            )
        # The answer reached the same chunk's new Attempt, which then read it.
        rebound = [item for item in bound if item.rows == bound[0].rows]
        self.assertEqual(len(rebound), 2)
        self.assertEqual(rebound[-1].continued_attempt_id, rebound[0].attempt_id)
        self.assertTrue(any("I meant the SNV at this row." in p for p in model.prompts))
        self.assertEqual({item.reading.status for item in fold.items}, {"recorded"})

    def test_a_dismissed_chunk_folds_dismissed_and_the_other_chunks_go_on(self):
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = observed_collection(
                Path(temporary).resolve(),
                investigation_table={"chunk_size": 1},
                investigation_concurrency={"planner_members": 1, "table_chunks": 1},
            )
            model = _AsksFirst(ask=False, silent_first=True)
            journey = _run(journey, model.model, adapter)
            request = project_agent_work_review(read_evidence(journey)).pending_request
            assert request is not None
            self.assertEqual(request.cause, "failed_turns")
            self.assertEqual(request.stuck_on.surface, "a text ending")
            dismiss = next(
                item for item in request.legal_actions if item.action == "dismiss"
            )
            self.assertIn("other chunks continue", dismiss.effect)
            answer = respond_agent_work_review(
                journey,
                request.request_event_id,
                DismissAgentWorkResponse(
                    action="dismiss",
                    reason="rs9999 does not exist; drop this item.",
                    asserted_by="researcher",
                ),
            )
            journey = _run(Journey.open(journey.path), model.model, adapter)
            (fold,) = _facts(journey, "InvestigationItemEvidenceCollected")
        first, *rest = fold.items
        self.assertEqual(first.reading.status, "dismissed")
        self.assertEqual(
            (first.reading.dismissal_event_id, first.reading.dismissal_reason),
            (answer.response_event_id, "rs9999 does not exist; drop this item."),
        )
        self.assertTrue(rest)
        self.assertEqual({item.reading.status for item in rest}, {"recorded"})


if __name__ == "__main__":
    unittest.main()
