"""Same-Run read access exposes provenance without granting effect authority."""

import tempfile
import unittest
import uuid
from pathlib import Path

from bio_neb.agentic_journey.assignment_context import (
    current_assignment_context,
    search_journey,
)
from bio_neb.journey import Journey
from tests.application_fakes import _registered


class AssignmentContextTests(unittest.TestCase):
    def test_history_is_prefix_bound_isolated_and_refuses_foreign_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _registered(Path(directory))
            journey = Journey.open(Path(receipt.journey_file))
            events = journey.read()
            before = journey.path.read_bytes()
            with self.assertRaisesRegex(ValueError, "outside this Run"):
                search_journey(journey, event_ids=(str(uuid.uuid4()),))
            selected = search_journey(
                journey,
                kinds=("ResearchRequestRegistered",),
                before_event_id=events[-1]["event_id"],
            )
            self.assertEqual(len(selected["events"]), 1)
            event = selected["events"][0]
            self.assertEqual(event["event_type"], "ResearchRequestRegistered")
            self.assertIn("Analyze rs1234", event["payload"]["prompt"])
            event["payload"]["prompt"] = "tampered by caller"
            retained = next(
                e for e in journey.read() if e["event_id"] == event["event_id"]
            )
            self.assertIn("Analyze rs1234", retained["payload"]["prompt"])
            self.assertEqual(journey.path.read_bytes(), before)

    def test_versioned_history_links_transcripts_and_replays_legacy_receipts(self):
        import copy
        import hashlib

        from bio_neb.agentic_journey.assignment_context import (
            search_journey_events,
            validate_search_receipt,
        )
        from bio_neb.journey import canonical_json

        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _registered(Path(directory))
            prefix = Journey.open(Path(receipt.journey_file)).read()
            transcript = copy.deepcopy(prefix[-1])
            transcript["event_id"] = str(uuid.uuid4())
            transcript["event_type"] = "ModelRequestRecorded"
            transcript["payload"] = {"messages": ["retained transcript " * 20000]}
            prefix.append(transcript)
            query = {
                "event_ids": [],
                "kinds": [],
                "limit": 100,
                "before_event_id": None,
            }
            current = search_journey_events(prefix, receipt.journey_id, **query)
            linked = current["events"][-1]
            self.assertNotIn("payload", linked)
            self.assertEqual(
                linked["artifact_references"], transcript["artifact_references"]
            )
            self.assertEqual(
                linked["payload_reference"]["event_id"], transcript["event_id"]
            )
            self.assertEqual(
                linked["payload_reference"]["sha256"],
                hashlib.sha256(
                    canonical_json(transcript["payload"]).encode()
                ).hexdigest(),
            )
            self.assertLess(len(canonical_json(current).encode()), 65536)
            legacy = {
                "journey_id": receipt.journey_id,
                "through_event_id": transcript["event_id"],
                "events": copy.deepcopy(prefix),
                "has_earlier_matches": False,
            }
            for result in (legacy, current):
                fact = {
                    "journey_id": receipt.journey_id,
                    "payload": {"query": query, "result": result},
                }
                validate_search_receipt(prefix, fact, query)
                forged = copy.deepcopy(fact)
                forged["payload"]["result"]["events"][-1]["event_id"] = "another-event"
                with self.assertRaisesRegex(ValueError, "exact Run prefix"):
                    validate_search_receipt(prefix, forged, query)
            with self.assertRaisesRegex(ValueError, "unknown history representation"):
                search_journey_events(
                    prefix, receipt.journey_id, representation="future"
                )
            self.assertEqual(prefix[-1]["payload"], transcript["payload"])

            # Many legitimate backward provenance edges force page compaction,
            # independently of payload size. The existing cursor retrieves older
            # matches without changing the selected prefix or original evidence.
            dense = []
            for number in range(100):
                event = copy.deepcopy(prefix[0])
                event["event_id"] = str(uuid.uuid4())
                event["parent_ids"] = [e["event_id"] for e in dense]
                event["payload"] = {"observation": number}
                dense.append(event)
            page = search_journey_events(dense, receipt.journey_id, limit=100)
            self.assertLessEqual(len(canonical_json(page).encode()), 65536)
            self.assertTrue(page["has_earlier_matches"])
            self.assertEqual(page["through_event_id"], dense[-1]["event_id"])
            first = page["events"][0]["event_id"]
            earlier = search_journey_events(
                dense, receipt.journey_id, limit=100, before_event_id=first
            )
            index = next(
                i for i, event in enumerate(dense) if event["event_id"] == first
            )
            self.assertEqual(earlier["through_event_id"], dense[index - 1]["event_id"])
            self.assertEqual(
                earlier["events"][-1]["event_id"], dense[index - 1]["event_id"]
            )
            self.assertTrue(
                {e["event_id"] for e in page["events"]}.isdisjoint(
                    e["event_id"] for e in earlier["events"]
                )
            )

    def test_payload_slices_recover_exact_unicode_bytes_by_event_identity(self):
        import copy

        from bio_neb.agentic_journey.assignment_context import (
            journey_search_query,
            search_action_errors,
            search_journey_events,
            validate_search_receipt,
        )
        from bio_neb.content import canonical_json, sha256_bytes

        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _registered(Path(directory))
            prefix = Journey.open(Path(receipt.journey_file)).read()
            finding = copy.deepcopy(prefix[-1])
            finding["event_id"] = str(uuid.uuid4())
            finding["event_type"] = "RefSnpToolResultRecorded"
            finding["payload"] = {"finding": "observed β🧬\n" * 4000}
            prefix.append(finding)
            original = copy.deepcopy(prefix)
            page = search_journey_events(
                prefix, receipt.journey_id, event_ids=[finding["event_id"]]
            )
            reference = page["events"][0]["payload_reference"]
            # The event ID is the identity; the page's digest is information only.
            selection = {"event_id": reference["event_id"], "length": 4096}
            chunks = []
            offset = 0
            while offset is not None:
                query = journey_search_query(
                    payload_slice={**selection, "offset": offset}
                )
                self.assertEqual(search_action_errors(prefix, query), ())
                result = search_journey_events(prefix, receipt.journey_id, **query)
                self.assertLess(len(canonical_json(result).encode()), 32768)
                chunk = result["payload_slice"]
                self.assertEqual(chunk["offset"], offset)
                chunks.append(chunk["text"])
                offset = chunk["next_offset"]
                fact = {
                    "journey_id": receipt.journey_id,
                    "payload": {"query": query, "result": result},
                }
                validate_search_receipt(prefix, fact, query)
                for key, value in (
                    ("text", "forged"),
                    ("offset", -1),
                    ("next_offset", 999999),
                    ("event_id", str(uuid.uuid4())),
                ):
                    forged = copy.deepcopy(fact)
                    forged["payload"]["result"]["payload_slice"][key] = value
                    with self.assertRaisesRegex(ValueError, "exact Run prefix"):
                        validate_search_receipt(prefix, forged, query)
            recovered = "".join(chunks)
            self.assertEqual(
                recovered.encode(), canonical_json(finding["payload"]).encode()
            )
            self.assertEqual(sha256_bytes(recovered.encode()), reference["sha256"])
            self.assertGreater(len(chunks), 1)
            # Each refusal names its own cause: only an event the Run does not
            # hold is outside it.
            for change, code in (
                ({"event_id": str(uuid.uuid4())}, "search_outside_run"),
                ({"sha256": reference["sha256"]}, "search_slice_invalid"),
                ({"offset": -1}, "search_slice_invalid"),
                ({"offset": len(recovered) + 1}, "search_slice_offset_outside_text"),
                ({"offset": True}, "search_slice_invalid"),
                ({"length": 4097}, "search_slice_invalid"),
                ({"length": 0}, "search_slice_invalid"),
            ):
                with self.subTest(change=change):
                    arguments = {"payload_slice": {**selection, **change}}
                    self.assertEqual(search_action_errors(prefix, arguments), (code,))
                    with self.assertRaises(ValueError):
                        search_journey_events(prefix, receipt.journey_id, **arguments)
            for arguments, code in (
                # A recorded live Run: a limit beside a slice of an event in the
                # Run.
                (
                    {"limit": 10, "payload_slice": selection},
                    "search_slice_with_page_filter",
                ),
                (
                    {"text": "β", "payload_slice": selection},
                    "search_slice_with_page_filter",
                ),
                ({"limit": 0}, "search_limit_invalid"),
                ({"text": ""}, "search_text_invalid"),
                ({"kinds": ["k"] * 21}, "search_filters_too_many"),
                ({"event_ids": [str(uuid.uuid4())]}, "search_outside_run"),
                (
                    {
                        "event_ids": [finding["event_id"]],
                        "before_event_id": prefix[0]["event_id"],
                    },
                    "search_outside_prefix",
                ),
            ):
                with self.subTest(arguments=arguments):
                    self.assertEqual(search_action_errors(prefix, arguments), (code,))
            with self.assertRaisesRegex(ValueError, "outside this Run prefix"):
                search_journey_events(
                    prefix,
                    receipt.journey_id,
                    payload_slice=selection,
                    before_event_id=finding["event_id"],
                )
            self.assertEqual(prefix, original)

    def test_context_supplies_original_intent_inputs_and_scope_without_history_dump(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _registered(
                Path(directory), prompt_text="Investigate my supplied sample"
            )
            journey = Journey.open(Path(receipt.journey_file))
            context = current_assignment_context(journey, "Prepare reviewed work")
            self.assertEqual(context["objective"], "Prepare reviewed work")
            self.assertEqual(
                context["original_intent"]["prompt"], "Investigate my supplied sample"
            )
            self.assertEqual(len(context["supplied_inputs"]["inputs"]), 2)
            self.assertEqual(context["open_questions"], [])
            self.assertFalse(context["authority"]["history_grants_effect_authority"])
            self.assertNotIn("events", context)

    def test_planning_context_speaks_requirement_state_without_call_shapes(self):
        import json

        from bio_neb.agentic_journey.planning_assignment import ROLE
        from bio_neb.agentic_journey.tool_execution import stage_tool_names
        from bio_neb.application import ObserveRun, observe
        from tests.application_fakes import _planning_ready

        def keys(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from keys(item)
            elif isinstance(value, list):
                for item in value:
                    yield from keys(item)

        with tempfile.TemporaryDirectory() as directory:
            receipt, composition = _planning_ready(Path(directory))
            # Each request's context is the canonical JSON after its stage's text:
            # every Tool its stage offers, callable now or waiting with the reason
            # its refusal gives (ADR 0044).
            contexts = [
                json.loads(text.rpartition("\n")[2])
                for text in composition.instructions_seen
            ]
            waited = []
            for context in contexts:
                waiting = {item["tool"]: item["reason"] for item in context["waiting"]}
                waited.extend(waiting.values())
                self.assertEqual(
                    sorted([*context["callable_now"], *waiting]),
                    sorted(
                        stage_tool_names(ROLE, context["stage_findings"]["stage"])
                        | {"search_journey"}
                    ),
                )
                self.assertNotIn("offered_tools", context)
            self.assertTrue(waited)
            self.assertTrue(all(reason.startswith("`") for reason in waited), waited)
            planned = [c for c in contexts if c["requirements"] is not None]
            # The last request saw recipe and companion facts, so absence is meaningful.
            last = contexts[-1]
            self.assertTrue(last["selections"]["facts"])
            self.assertTrue(last["companions"]["facts"])
            self.assertIn("result", last["selections"]["facts"][0])
            self.assertTrue(last["selections"]["recipe_work_items"])
            for context in contexts:
                self.assertFalse(
                    {"execution_ready", "tool_input", "recipe_tool_inputs"}
                    & set(keys(context))
                )
            self.assertTrue(planned)
            for context in planned:
                goal = next(
                    item
                    for item in context["requirements"]["requirements"]
                    if item["kind"] == "research_goal"
                )
                self.assertEqual(goal["state"], "resolved")
            observed = observe(ObserveRun(journey_file=receipt.journey_file))
            self.assertEqual(
                {
                    item.state
                    for item in observed.requirement_resolution.requirements
                    if item.kind == "research_goal"
                },
                {"resolved"},
            )

    def test_investigation_context_preserves_assertion_classification_and_input(self):
        # Every member's original intent is the research goal, in the
        # researcher's words; no member's own request stands in for it.
        from bio_neb.investigation.investigation import (
            InvestigationStarted,
            UserAssertion,
        )
        from bio_neb.investigation.investigation_request import (
            InvestigationRequestItem,
        )
        from tests.sequential_investigation_support import input_and_assertion

        goal = "Tell me whether HG001 carries rs429358 and rs7412."
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory))
            selected_input, _ = input_and_assertion()
            assertion = UserAssertion.create_collection(
                asserted_by="researcher",
                requests=(
                    InvestigationRequestItem(request_text="rs429358", rsid="rs429358"),
                    InvestigationRequestItem(request_text="rs7412", rsid="rs7412"),
                ),
                predecessor_ids=(selected_input.input_id,),
                research_goals=(goal,),
            )
            fact = InvestigationStarted(
                continuation_id="scientific-continuation:" + "b" * 64,
                investigation_input=selected_input,
                assertion=assertion,
            )
            started = journey.record(
                fact.event_type, fact.model_dump(mode="json", exclude={"event_type"})
            )
            context = current_assignment_context(
                journey, "Resolve the supplied scientific request"
            )
            self.assertEqual(
                context["original_intent"]["event_id"], started["event_id"]
            )
            self.assertEqual(context["original_intent"]["research_goal"], [goal])
            self.assertEqual(
                set(context["original_intent"]),
                {"event_id", "grounding", "research_goal"},
            )
            self.assertEqual(
                context["supplied_inputs"]["input_id"], selected_input.input_id
            )
            self.assertEqual(context["original_intent"]["grounding"], "user_assertion")
            self.assertFalse(context["authority"]["history_grants_effect_authority"])


class SearchRefusalFactTests(unittest.TestCase):
    """A refused search names what it sent wrongly, from the same read admission
    makes (#409, #372 map X-search-slice and X-search-outside-scope)."""

    EVENTS = [
        {
            "event_id": event_id,
            "journey_id": "journey",
            "event_type": "Recorded",
            "payload": payload,
        }
        for event_id, payload in (
            ("scope", {"evidence_event_ids": ["inside"]}),
            ("inside", {"text": "in scope"}),
            ("outside", {"text": "not in scope"}),
        )
    ]

    def _facts(self, arguments, scope_event_id=None):
        from bio_neb.agentic_journey.assignment_context import (
            search_action_errors,
            search_refusal_facts,
        )

        (code,) = search_action_errors(
            self.EVENTS, arguments, scope_event_id=scope_event_id
        )
        return code, search_refusal_facts(
            self.EVENTS, arguments, code, scope_event_id=scope_event_id
        )

    def test_a_slice_read_names_the_filters_sent_beside_it(self):
        self.assertEqual(
            self._facts(
                {
                    "text": "in",
                    "limit": 5,
                    "payload_slice": {"event_id": "inside", "offset": 0, "length": 9},
                }
            ),
            ("search_slice_with_page_filter", {"fields": ("text", "limit")}),
        )

    def test_a_scoped_search_names_the_events_outside_its_scope(self):
        self.assertEqual(
            self._facts({"event_ids": ["inside", "outside"]}, scope_event_id="scope"),
            ("search_outside_scope", {"outside": ("outside",)}),
        )


class SearchReceiptReopenTests(unittest.TestCase):
    """Every search the Run records reopens: its receipt re-reads the prefix the
    search read, whatever the query names (#457 C. elegans a2)."""

    EVENTS = [
        {
            "event_id": event_id,
            "journey_id": "journey",
            "event_type": "Recorded",
            "payload": payload,
        }
        for event_id, payload in (
            ("scope", {"evidence_event_ids": ["first", "second"]}),
            ("first", {"text": "rsid one"}),
            ("second", {"text": "rsid two"}),
            ("third", {"text": "rsid three"}),
        )
    ]

    def _reopens(self, arguments):
        from bio_neb.agentic_journey.assignment_context import (
            journey_search_query,
            search_journey_events,
            validate_search_receipt,
        )

        result = search_journey_events(self.EVENTS, "journey", **arguments)
        query = journey_search_query(**arguments)
        fact = {"journey_id": "journey", "payload": {"query": query, "result": result}}
        validate_search_receipt(
            self.EVENTS,
            fact,
            {key: value for key, value in arguments.items() if key != "scope_event_id"},
        )

    def test_each_search_shape_reopens_from_its_receipt(self):
        for arguments in (
            {"text": "rsid"},
            {"text": "rsid", "limit": 10, "before_event_id": "third"},
            {"kinds": ["Recorded"], "before_event_id": "second"},
            {"event_ids": ["first"], "before_event_id": "third"},
            {"payload_slice": {"event_id": "first", "offset": 0, "length": 4}},
            {"text": "rsid", "scope_event_id": "scope"},
            {"text": "rsid", "before_event_id": "second", "scope_event_id": "scope"},
        ):
            with self.subTest(arguments=arguments):
                self._reopens(arguments)
