"""The Investigation model's reading of a researcher's words, at its boundary (#472).

The model types the target an untyped item's words, or an answer's, state with
`record_target_reading`; code checks only that the quote stands in the words and holds
what it typed (owner Q3 a, R1 b; ADR 0049). These tests send readings the words do not
hold, and readings moved to other words at reopen, through the real Tool and reader.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.investigation.investigation import (
    UserAssertion,
    project_investigation,
)
from bio_neb.investigation.investigation_request import InvestigationRequestItem
from bio_neb.recording import read_evidence
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.sequential_investigation_support import (
    SourceResponsesOnly,
    input_and_assertion,
    run_attempt,
)

WORDS = "Does HG001 carry rs429358 at chr19:44908684 T>C?"
SIBLING = "Is rs7412 present at chr19:44908822 C>T?"
VALID = {"quote": "rs429358", "rsid": "rs429358"}
_ASK = {
    "question": "Which target do you mean?",
    "missing_information": "The intended target.",
    "asks_about": "target",
}


def _of(events, event_type):
    return [e for e in events if e["event_type"] == event_type]


def _read(directory, readings, *items):
    """Run the collection's Attempts with a scripted model that sends ``readings`` in
    order while words are unread, then asks: a test's stand-in for the model."""
    investigation_input, _ = input_and_assertion()
    assertion = UserAssertion.create_collection(
        asserted_by="researcher",
        requests=items or (InvestigationRequestItem(request_text=WORDS),),
        predecessor_ids=(investigation_input.input_id,),
    )
    sent = []

    async def respond(messages, info):
        tool = next(
            (t for t in info.function_tools if t.name == "record_target_reading"),
            None,
        )
        unread = (
            tool.parameters_json_schema["properties"]["source_event_id"].get("enum")
            if tool is not None
            else None
        )
        sent.append(info)
        if unread and len(sent) <= len(readings):
            name, arguments = (
                "record_target_reading",
                {"source_event_id": unread[0], **readings[len(sent) - 1]},
            )
        else:
            name, arguments = "request_investigation_information", _ASK
        return ModelResponse(
            parts=[ToolCallPart(name, arguments, tool_call_id=f"reading-{len(sent)}")]
        )

    result, journey = run_attempt(
        Path(directory),
        SimpleNamespace(model=ScriptedModelEndpoint(respond)),
        SourceResponsesOnly(),
        investigation_input=investigation_input,
        assertion=assertion,
        invocation_overrides={"model_requests": 6},
    )
    return result, read_evidence(journey)


class TargetReadingAdmissionTests(unittest.TestCase):
    """#472 Q3 a, R1 b (ADR 0049): a reading is admitted only by membership in the
    words it reads. A refused reading records no target, and the model repairs it in
    the same Attempt."""

    def test_a_reading_the_words_do_not_hold_is_refused_and_records_no_target(self):
        for reading, code in (
            # The quote stands nowhere in the words.
            ({"quote": "rs7412", "rsid": "rs7412"}, "target_quote_ungrounded"),
            # The quote stands in them, but holds another rsID than the one typed.
            ({"quote": "rs429358", "rsid": "rs7412"}, "target_rsid_not_in_quote"),
            # The quote holds a site, but not the position typed.
            (
                {
                    "quote": "chr19:44908684 T>C",
                    "site": {
                        "contig": "chr19",
                        "position": 44908685,
                        "ref": "T",
                        "alt": "C",
                    },
                },
                "target_site_not_in_quote",
            ),
        ):
            with (
                self.subTest(code=code),
                tempfile.TemporaryDirectory() as directory,
            ):
                result, events = _read(directory, (reading, VALID))
                refused, admitted = [
                    v["payload"]
                    for v in _of(events, "ToolCallValidated")
                    if v["payload"]["tool_name"] == "record_target_reading"
                ]
                self.assertEqual(
                    (refused["status"], refused["errors"]), ("invalid", [code])
                )
                self.assertEqual(admitted["errors"], [])
                # Only the repaired reading is recorded, with what it typed.
                (read,) = [e["payload"] for e in _of(events, "InvestigationTargetRead")]
                self.assertEqual(
                    (read["quote"], read["target"]["rsid"]), ("rs429358", "rs429358")
                )
                self.assertEqual(result.pending_request.question.asks_about, "target")

    def test_a_gene_change_is_typed_from_the_words_and_checked_by_membership(self):
        """#473 Q1 a, Q2 a (ADR 0049): a reading types the gene and change the words
        write; one the quote does not hold, or one with no typed part (code reads no
        change text), is refused, and the model repairs it in the same Attempt."""
        words = "Does HG001 carry APOE c.4T>C?"
        typed = {
            "quote": "APOE c.4T>C",
            "gene_change": {
                "gene": "APOE",
                "change": "c.4T>C",
                "coding": {"cds_position": 4, "ref": "T", "alt": "C"},
            },
        }
        for reading, code in (
            (
                {
                    **typed,
                    "gene_change": {
                        **typed["gene_change"],
                        "coding": {"cds_position": 5, "ref": "T", "alt": "C"},
                    },
                },
                "target_gene_change_not_in_quote",
            ),
            (
                {
                    "quote": "APOE c.4T>C",
                    "gene_change": {"gene": "APOE", "change": "c.4T>C"},
                },
                "target_gene_change_untyped",
            ),
        ):
            with (
                self.subTest(code=code),
                tempfile.TemporaryDirectory() as directory,
            ):
                _result, events = _read(
                    directory,
                    (reading, typed),
                    InvestigationRequestItem(request_text=words),
                )
                refused, admitted = [
                    v["payload"]
                    for v in _of(events, "ToolCallValidated")
                    if v["payload"]["tool_name"] == "record_target_reading"
                ]
                self.assertEqual(
                    (refused["status"], refused["errors"]), ("invalid", [code])
                )
                self.assertEqual(admitted["errors"], [])
                (read,) = [e["payload"] for e in _of(events, "InvestigationTargetRead")]
                self.assertEqual(read["target"]["gene_change"], typed["gene_change"])

    def test_a_malformed_site_is_refused_as_an_argument_before_the_tool_runs(self):
        """#472 Q1 a, review r1 ST1/SP1 (ADR 0031: an invalid model action gets its
        bounded repair): a site with one allele breaks the stored position's rule, so
        the argument is refused and no reading is recorded for it."""
        site = {"contig": "chr19", "position": 44908684, "ref": "T"}
        with tempfile.TemporaryDirectory() as directory:
            result, events = _read(
                directory, ({"quote": "chr19:44908684 T>C", "site": site}, VALID)
            )
        refused, admitted = [
            v["payload"]
            for v in _of(events, "ToolCallValidated")
            if v["payload"]["tool_name"] == "record_target_reading"
        ]
        self.assertEqual(
            (refused["status"], refused["errors"]),
            (
                "invalid",
                [
                    "tool_arguments_invalid:site:value_error:Value error, a position "
                    "states both alleles or neither"
                ],
            ),
        )
        self.assertEqual(admitted["status"], "valid")
        (read,) = _of(events, "InvestigationTargetRead")
        self.assertNotIn("site", read["payload"]["target"])
        self.assertEqual(result.pending_request.question.asks_about, "target")

    def test_a_padded_version_is_refused_and_a_blank_one_is_absent(self):
        """#476 (owner, 2026-10-28, G3; sweep B): admission refuses a padded
        version with the schema's correction instead of the handler raising, and a
        blank one is the absent value, so the reading is admitted without it."""
        with tempfile.TemporaryDirectory() as directory:
            _result, events = _read(
                directory,
                (
                    {**VALID, "stated_version": "GRCh38 "},
                    {**VALID, "stated_version": ""},
                ),
            )
        refused, admitted = [
            v["payload"]
            for v in _of(events, "ToolCallValidated")
            if v["payload"]["tool_name"] == "record_target_reading"
        ]
        self.assertEqual(refused["status"], "invalid")
        self.assertTrue(
            refused["errors"][0].startswith(
                "tool_arguments_invalid:stated_version:string_pattern_mismatch"
            ),
            refused["errors"],
        )
        self.assertEqual(admitted["status"], "valid")
        self.assertNotIn("stated_version", admitted["arguments"])
        (read,) = _of(events, "InvestigationTargetRead")
        self.assertNotIn("stated_assembly", read["payload"]["target"])


class TargetReadingReopenTests(unittest.TestCase):
    """#472 Q3 a (ADR 0049): reopen checks that each reading reads its own member's
    words, unaltered; a reading moved to other words, another member or another event
    is refused before any projection uses it."""

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            _result, cls.events = _read(
                directory,
                (VALID,),
                InvestigationRequestItem(request_text=WORDS),
                # A typed sibling: it has no words to read, in an Attempt of its own.
                InvestigationRequestItem(request_text=SIBLING, rsid="rs7412"),
            )

    def test_each_reading_reopens_as_recorded(self):
        project_investigation(self.events)
        self.assertEqual(
            [
                e["payload"]["words"]
                for e in _of(self.events, "InvestigationTargetRead")
            ],
            [WORDS],
        )

    def test_a_reading_moved_to_other_words_is_refused(self):
        (reading,) = _of(self.events, "InvestigationTargetRead")
        bound = [
            e["payload"]
            for e in _of(self.events, "InvestigationAttemptBound")
            if e["payload"]["collection_member_id"]
            != reading["payload"]["collection_member_id"]
        ]

        def other_words(payload):
            # Still quoting rs429358, so only reopen's own check can refuse it.
            payload["words"] = WORDS + " Also check its neighbour."

        def other_member(payload):
            payload["collection_member_id"] = bound[0]["collection_member_id"]

        def other_event(payload):
            payload["source_event_id"] = _of(self.events, "InvestigationAttemptBound")[
                0
            ]["event_id"]

        def another_source(payload):
            payload["source"] = "answer"

        self.assertTrue(bound, "the sibling member is bound in its own Attempt")
        for change in (other_words, other_member, other_event, another_source):
            with self.subTest(change=change.__name__):
                hostile = copy.deepcopy(self.events)
                change(
                    next(e for e in hostile if e["event_id"] == reading["event_id"])[
                        "payload"
                    ]
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "Investigation target reading changed the words it reads",
                ):
                    project_investigation(hostile)


if __name__ == "__main__":
    unittest.main()
