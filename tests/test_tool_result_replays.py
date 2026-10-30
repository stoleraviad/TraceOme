"""Recorded replies and results read through #422's two shared mechanisms, offline.

Each fixture is copied from a frozen lane P spike Journey (fixture registrations in
``tests/testing_ownership.json``): a ``read_paper`` result that held raw tabs (#395 run
2), and an output call whose nullable ``reference`` object arrived as JSON text (#403
closing spike run 1), with the output Tools its request showed.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from bio_neb.agentic_journey.model_turn import read_messages_record, read_turn
from bio_neb.agentic_journey.tool_execution import model_visible

_FIXTURES = Path(__file__).parent / "fixtures" / "tool_result_cleanup"
_CONTROL = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")


def _event(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


class RecordedResultTests(unittest.TestCase):
    def test_a_recorded_paper_reading_with_tabs_is_shown_with_spaces(self):
        recorded = _event("issue395-run2-paper-reading-1c2e42ae.jsonl")["payload"][
            "result"
        ]
        self.assertIn("\t", recorded["text"])

        shown, guarded = model_visible(recorded)

        self.assertTrue(guarded)
        self.assertEqual([s for s in _strings(shown) if _CONTROL.search(s)], [])
        # One character for one: the offsets the result states still count.
        self.assertEqual(shown["text"], recorded["text"].replace("\t", " "))
        self.assertEqual(len(shown["text"]), recorded["end"] - recorded["start"])
        self.assertEqual(shown["text"].count("\n"), recorded["text"].count("\n"))


class RecordedReplyTests(unittest.TestCase):
    def test_a_recorded_reference_sent_as_json_text_is_read_as_its_object(self):
        response = _event("issue403-s3-run1-response-53c14084.jsonl")
        message = response["payload"]["compatible_adapter_evidence"]["response"][
            "message"
        ]
        shown = [
            SimpleNamespace(
                name=tool["name"], parameters_json_schema=tool["parameters_json_schema"]
            )
            for tool in json.loads(
                (
                    _FIXTURES / "issue403-s3-run1-request-d82e8e38-output-tools.json"
                ).read_text(encoding="utf-8")
            )
        ]
        (raw,) = [part for part in message["parts"] if part["part_kind"] == "tool-call"]
        sent = json.loads(raw["args"])
        self.assertIsInstance(sent["reference"], str)
        (reply,) = read_messages_record([message])

        (call,) = read_turn(reply, shown).calls
        (unread,) = read_turn(reply).calls

        self.assertEqual(call.tool_name, "propose_reading_areas")
        self.assertEqual(call.arguments.decoded, ("reference",))
        self.assertEqual(
            call.arguments.value,
            {**sent, "reference": json.loads(sent["reference"])},
        )
        self.assertEqual(call.arguments.value["reference"]["name"], "UCSC sacCer1")
        # The areas arrived as objects already and stay as sent; without the
        # Tools the request showed, the reply reads as it did before.
        self.assertEqual(call.arguments.value["areas"], sent["areas"])
        self.assertEqual((unread.arguments.value, unread.arguments.decoded), (sent, ()))


if __name__ == "__main__":
    unittest.main()
