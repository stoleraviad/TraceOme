"""Exact Tool mechanical evidence lives only in the shared Contact journal."""

import copy
import tempfile
import unittest
import uuid
from pathlib import Path

from bio_neb.agentic_journey.tool_execution import (
    ObservedToolExecutionStarted,
    ObservedToolExecutionSucceeded,
)
from bio_neb.journey import Journey
from bio_neb.recording import (
    evidence_revision,
    processing_pass,
    read_evidence,
    read_passes,
    record_contact_evidence,
)


def start_tool(journey, *, call="call-a"):
    validation = journey.record(
        "ToolCallValidated",
        {"attempt_id": "attempt-a", "tool_call_id": call, "tool_name": "read_history"},
        [journey.read()[-1]["event_id"]],
    )
    return record_contact_evidence(
        journey,
        ObservedToolExecutionStarted(
            attempt_id="attempt-a",
            response_event_id=str(uuid.uuid4()),
            validation_event_id=validation["event_id"],
            tool_call_id=call,
            tool_name="read_history",
            adapter_identity="local",
        ),
    )


def finish_tool(journey, started, result):
    return record_contact_evidence(
        journey,
        ObservedToolExecutionSucceeded(
            attempt_id="attempt-a",
            started_event_id=started["event_id"],
            stage_result_event_id=result["event_id"],
            model_visible_result_digest="a" * 64,
            artifact_digests=(),
            artifact_source_event_ids=(),
        ),
    )


class ContactToolRecordTests(unittest.TestCase):
    def test_tool_evidence_reopens_without_semantic_duplicates_or_authority_edges(self):
        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary))
            origin = journey.record("JourneyStarted", {})
            with processing_pass(journey) as interval:
                started = start_tool(journey)
                semantic_head = journey.read()[-1]
                result = journey.record(
                    "Result", {"value": 7}, [semantic_head["event_id"]]
                )
                finished = finish_tool(journey, started, result)
                before_close = read_evidence(journey)
                assert [e["event_id"] for e in before_close][-3:] == [
                    started["event_id"],
                    result["event_id"],
                    finished["event_id"],
                ]
                assert result["parent_ids"] == [semantic_head["event_id"]]
                assert started["parent_ids"] == finished["parent_ids"] == []
                interval.close("completed")
            reopened = Journey.open(journey.path)
            assert all(
                not e["event_type"].startswith("ObservedToolExecution")
                for e in reopened.read()
            )
            evidence = read_evidence(reopened)
            assert started in evidence and finished in evidence and origin in evidence
            assert read_passes(reopened)[0].journal_complete

    def test_journal_growth_invalidates_reader_cache_without_semantic_append(self):
        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary))
            journey.record("JourneyStarted", {})
            with processing_pass(journey):
                started = start_tool(journey)
                result = journey.record("Result", {}, [journey.read()[-1]["event_id"]])
                revision = evidence_revision(journey)
                finish_tool(journey, started, result)
                assert evidence_revision(journey) != revision

    def test_unfinished_tool_stays_incomplete_and_cannot_receive_wrong_call_terminal(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary))
            journey.record("JourneyStarted", {})
            with processing_pass(journey) as interval:
                started = start_tool(journey)
                wrong = copy.deepcopy(started)
                wrong["event_id"] = str(uuid.uuid4())
                with self.assertRaisesRegex(ValueError, "exact open start"):
                    finish_tool(journey, wrong, journey.read()[-1])
                interval.close("interrupted")
            view = read_passes(Journey.open(journey.path))[0]
            assert not view.journal_complete
            assert len(view.contacts) == 2  # start and final, never a claimed success

    def test_contact_free_reader_rejects_rebound_attempt_and_cross_pass_anchor(self):
        for mutation in ("attempt", "anchor"):
            with tempfile.TemporaryDirectory() as temporary:
                journey = Journey.create(Path(temporary))
                origin = journey.record("JourneyStarted", {})
                with processing_pass(journey) as interval:
                    start_tool(journey)
                    row = interval.journal.snapshot().splitlines()[0]
                    import json

                    from bio_neb.content import canonical_json

                    value = json.loads(row)
                    if mutation == "attempt":
                        value["scope"]["attempt_id"] = "other-attempt"
                    else:
                        value["evidence"]["after_event_id"] = origin["event_id"]
                    tampered = (canonical_json(value) + "\n").encode()
                    interval.journal.path.write_bytes(tampered)
                with self.assertRaises(ValueError):
                    read_evidence(Journey.open(journey.path))

    def test_source_physical_records_use_shared_journal_and_exact_scope_on_reopen(self):
        from bio_neb.agentic_journey.source_contacts import (
            SourceContactScopeFinished,
            SourceContactScopeStarted,
            SourcePhysicalContactFinished,
            SourcePhysicalContactStarted,
        )
        from bio_neb.recording import ContactScope, physical_contact

        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary))
            journey.record("JourneyStarted", {})
            with processing_pass(journey) as interval:
                scope = record_contact_evidence(
                    journey,
                    SourceContactScopeStarted(
                        attempt_id="attempt-a",
                        tool_call_id="call-a",
                        tool_name="fetch_refsnp",
                    ),
                )
                started = record_contact_evidence(
                    journey,
                    SourcePhysicalContactStarted(
                        scope_event_id=scope["event_id"],
                        request_identity="https://ncbi.example/rs1",
                    ),
                )
                with physical_contact(
                    ContactScope(
                        attempt_id="attempt-a",
                        request_id=scope["event_id"],
                        call_id="call-a",
                        evidence_event_id=started["event_id"],
                    ),
                    "source.get",
                ):
                    record_contact_evidence(
                        journey,
                        SourcePhysicalContactFinished(
                            started_event_id=started["event_id"],
                            disposition="succeeded",
                        ),
                    )
                record_contact_evidence(
                    journey,
                    SourceContactScopeFinished(
                        scope_event_id=scope["event_id"], transport_observed=True
                    ),
                )
                interval.close("completed")
            reopened = Journey.open(journey.path)
            assert not any(
                e["event_type"].startswith("Source") for e in reopened.read()
            )
            evidence = read_evidence(reopened)
            assert scope in evidence and started in evidence
            assert read_passes(reopened)[0].journal_complete

    def test_a_source_head_contact_needs_its_own_physical_evidence(self):
        from bio_neb.agentic_journey.source_contacts import (
            SourceContactScopeFinished,
            SourceContactScopeStarted,
            SourcePhysicalContactFinished,
            SourcePhysicalContactStarted,
        )
        from bio_neb.recording import ContactScope, physical_contact

        for evidence_contract, refused in (
            ("SourcePhysicalContactStarted", False),
            ("SourceContactScopeFinished", True),
        ):
            with (
                self.subTest(evidence=evidence_contract),
                tempfile.TemporaryDirectory() as temporary,
            ):
                journey = Journey.create(Path(temporary))
                journey.record("JourneyStarted", {})
                with processing_pass(journey) as interval:
                    scope = record_contact_evidence(
                        journey,
                        SourceContactScopeStarted(
                            attempt_id="attempt-a",
                            tool_call_id="call-a",
                            tool_name="describe_companion",
                        ),
                    )
                    evidence = record_contact_evidence(
                        journey,
                        SourcePhysicalContactStarted(
                            scope_event_id=scope["event_id"],
                            request_identity="https://ftp.ncbi.example/x_genomic.fna.gz",
                            method="HEAD",
                        )
                        if not refused
                        else SourceContactScopeFinished(
                            scope_event_id=scope["event_id"], transport_observed=True
                        ),
                    )
                    contact = physical_contact(
                        ContactScope(
                            attempt_id="attempt-a",
                            request_id=scope["event_id"],
                            call_id="call-a",
                            evidence_event_id=evidence["event_id"],
                        ),
                        "source.head",
                    )
                    if refused:
                        with self.assertRaisesRegex(ValueError, "wrong contract"):
                            with contact:
                                pass
                        continue
                    with contact:
                        record_contact_evidence(
                            journey,
                            SourcePhysicalContactFinished(
                                started_event_id=evidence["event_id"],
                                disposition="succeeded",
                            ),
                        )
                    interval.close("completed")
                (closed,) = read_passes(Journey.open(journey.path))
                self.assertTrue(closed.journal_complete)
                self.assertEqual(
                    [row.operation for row in closed.contacts if row.kind == "start"],
                    ["source.head"],
                )

    def test_model_contact_records_use_shared_journal_and_preserve_logical_request(
        self,
    ):
        from bio_neb.agentic_journey.attempt_journey import ModelPhysicalContactStarted
        from bio_neb.recording import ContactScope, physical_contact
        from tests.model_availability_fakes import ATTEMPT_ID, model_request_payload

        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary))
            journey.record("JourneyStarted", {})
            with processing_pass(journey) as interval:
                request = journey.record(
                    "ModelRequestRecorded",
                    model_request_payload(ATTEMPT_ID),
                    [journey.read()[-1]["event_id"]],
                )
                started = record_contact_evidence(
                    journey,
                    ModelPhysicalContactStarted(
                        attempt_id=ATTEMPT_ID,
                        request_number=1,
                        logical_request_event_id=request["event_id"],
                        contact_ordinal=1,
                        transport_evidence_expected=False,
                    ),
                )
                with physical_contact(
                    ContactScope(
                        attempt_id=ATTEMPT_ID,
                        request_id=request["event_id"],
                        evidence_event_id=started["event_id"],
                    ),
                    "model.request",
                ):
                    pass
                interval.close("completed")
            reopened = Journey.open(journey.path)
            assert started not in reopened.read()
            assert started in read_evidence(reopened)
            assert request in reopened.read()

    def test_provider_recovery_reader_checks_journal_identity_without_semantic_parent_edges(
        self,
    ):
        from bio_neb.agentic_journey.model_availability import (
            project_model_availability_review,
        )
        from tests.model_availability_fakes import _pending_review_timeline

        # Derived from the maintained offline provider-recovery grammar fixture.
        # Only the owning record/parent representation changes; retry policy and
        # response/call/request identities remain the fixture's original values.
        timeline, _ = _pending_review_timeline(two_contacts=True)
        by_id = {event["event_id"]: event for event in copy.deepcopy(timeline)}
        mechanical = {
            "ModelPhysicalContactStarted",
            "ModelPhysicalContactFailed",
            "ModelPhysicalContactCancelled",
            "ModelAvailabilityRetryScheduled",
            "ModelAvailabilityRetryDelayCompleted",
            "ModelAvailabilityRetryDelayCancelled",
        }
        for event in timeline:
            if event["event_type"] in mechanical:
                event["parent_ids"] = []
            else:
                parents = event.get("parent_ids", [])
                while (
                    len(parents) == 1 and by_id[parents[0]]["event_type"] in mechanical
                ):
                    parents = by_id[parents[0]].get("parent_ids", [])
                event["parent_ids"] = parents
        read = project_model_availability_review(timeline)
        assert len(read.pending_request.contacts) == 2
        assert len(read.pending_request.delays) == 1
        hostile = copy.deepcopy(timeline)
        next(
            event
            for event in hostile
            if event["event_type"] == "ModelPhysicalContactFailed"
        )["payload"]["contact_started_event_id"] = str(uuid.uuid4())
        with self.assertRaises(ValueError):
            project_model_availability_review(hostile)
