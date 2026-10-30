"""Shared interval evidence, with explicit synthetic requests and no adapters."""

import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from bio_neb.content import canonical_json
from bio_neb.journey import Journey
from bio_neb.recording import (
    ContactJournal,
    ContactScope,
    physical_contact,
    processing_pass,
    read_contacts,
    read_passes,
)


class RecordingTests(unittest.TestCase):
    def test_torn_active_contact_tail_preserves_prior_result_and_unknown_completion(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary))
            origin = journey.record("JourneyStarted", {})
            with journey.processing(), processing_pass(journey) as interval:
                attempt_id = str(uuid.uuid4())
                request = journey.record(
                    "Request", {"attempt_id": attempt_id}, [origin["event_id"]]
                )
                interval.journal.start(
                    ContactScope(attempt_id=attempt_id, request_id=request["event_id"]),
                    "inspect",
                )
                result = journey.record("Result", {"value": 7}, [request["event_id"]])
            with interval.journal.path.open("ab") as stream:
                stream.write(b'{"kind":"terminal"')
            reopened = Journey.open(journey.path)
            view = read_passes(reopened)[0]
            self.assertEqual(view.accounting_error, "truncated_prefix")
            self.assertFalse(view.journal_complete)
            self.assertEqual([r.kind for r in view.contacts], ["start"])
            self.assertIn(result, reopened.read())

    def test_model_physical_evidence_cannot_be_omitted_or_reused(self):
        from bio_neb.agentic_journey.attempt_journey import ModelPhysicalContactStarted
        from bio_neb.recording import record_contact_evidence
        from tests.model_availability_fakes import model_request_payload

        for omit in (True, False):
            with self.subTest(omit=omit), tempfile.TemporaryDirectory() as temporary:
                journey = Journey.create(Path(temporary))
                origin = journey.record("JourneyStarted", {})
                with journey.processing(), processing_pass(journey) as interval:
                    attempt_id = str(uuid.uuid4())
                    request = journey.record(
                        "ModelRequestRecorded",
                        model_request_payload(attempt_id),
                        [origin["event_id"]],
                    )
                    evidence = record_contact_evidence(
                        journey,
                        ModelPhysicalContactStarted(
                            attempt_id=attempt_id,
                            request_number=1,
                            logical_request_event_id=request["event_id"],
                            contact_ordinal=1,
                            transport_evidence_expected=False,
                        ),
                    )
                    scope = ContactScope(
                        attempt_id=attempt_id,
                        request_id=request["event_id"],
                        evidence_event_id=None if omit else evidence["event_id"],
                    )
                    for _ in range(1 if omit else 2):
                        contact = interval.journal.start(scope, "model.request")
                        interval.journal.finish(contact, "succeeded")
                    with self.assertRaises(ValueError):
                        read_passes(journey)

    def test_active_contact_scope_is_checked_against_retained_request(self):
        for changed in (
            {"item_id": "other"},
            {"revision_id": "other"},
            {"request_id": str(uuid.uuid4())},
            {"evidence_event_id": "origin"},
        ):
            with (
                self.subTest(changed=changed),
                tempfile.TemporaryDirectory() as temporary,
            ):
                journey = Journey.create(Path(temporary))
                origin = journey.record("JourneyStarted", {})
                with journey.processing(), processing_pass(journey) as interval:
                    attempt_id = str(uuid.uuid4())
                    request = journey.record(
                        "Request",
                        {"attempt_id": attempt_id, "item_id": "a", "revision_id": "r1"},
                        [origin["event_id"]],
                    )
                    altered = dict(changed)
                    if altered.get("evidence_event_id") == "origin":
                        altered["evidence_event_id"] = origin["event_id"]
                    scope = ContactScope(
                        **{
                            **dict(
                                attempt_id=attempt_id,
                                request_id=request["event_id"],
                                item_id="a",
                                revision_id="r1",
                            ),
                            **altered,
                        }
                    )
                    contact = interval.journal.start(scope, "inspect")
                    with self.assertRaises(ValueError):
                        read_passes(journey)
                    interval.journal.finish(contact, "succeeded")
                    interval.close("completed")
                with self.assertRaises(ValueError):
                    read_passes(Journey.open(journey.path))

    def test_contact_start_checks_only_its_row_and_refuses_like_the_full_check(self):
        from unittest.mock import patch

        import bio_neb.recording as recording
        from bio_neb.agentic_journey.attempt_journey import ModelPhysicalContactStarted
        from bio_neb.recording import (
            ContactRow,
            read_evidence,
            record_contact_evidence,
            validate_contact_ownership,
        )
        from tests.model_availability_fakes import model_request_payload

        def outcome(call, *args):
            try:
                call(*args)
            except ValueError as error:
                return "refused", str(error)
            return ("admitted",)

        def contact(scope):
            with physical_contact(scope, "model.request"):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary))
            origin = journey.record("JourneyStarted", {})
            attempt_id = str(uuid.uuid4())
            early = journey.record(
                "ModelRequestRecorded",
                model_request_payload(attempt_id),
                [origin["event_id"]],
            )
            with journey.processing(), processing_pass(journey) as interval:
                request = journey.record(
                    "ModelRequestRecorded",
                    model_request_payload(attempt_id),
                    [origin["event_id"]],
                )

                def evidence():
                    return record_contact_evidence(
                        journey,
                        ModelPhysicalContactStarted(
                            attempt_id=attempt_id,
                            request_number=1,
                            logical_request_event_id=request["event_id"],
                            contact_ordinal=1,
                            transport_evidence_expected=False,
                        ),
                    )["event_id"]

                used = evidence()
                contact(
                    ContactScope(
                        attempt_id=attempt_id,
                        request_id=request["event_id"],
                        evidence_event_id=used,
                    )
                )
                fresh = evidence()
                cases = {
                    "reused evidence": {"evidence_event_id": used},
                    "missing evidence": {"evidence_event_id": None},
                    "request before the Pass": {"request_id": early["event_id"]},
                    "unknown request": {"request_id": str(uuid.uuid4())},
                    "changed Attempt": {"attempt_id": str(uuid.uuid4())},
                    "changed Tool call": {"call_id": "other-call"},
                    "evidence not after request": {
                        "evidence_event_id": request["event_id"]
                    },
                    "lawful": {},
                }
                for name, changed in cases.items():
                    scope = ContactScope(
                        **{
                            "attempt_id": attempt_id,
                            "request_id": request["event_id"],
                            "evidence_event_id": fresh,
                            **changed,
                        }
                    )
                    before = interval.journal.snapshot()
                    full = outcome(
                        validate_contact_ownership,
                        (
                            *read_contacts(
                                before,
                                run_id=journey.journey_id,
                                pass_id=interval.pass_id,
                            ),
                            ContactRow(
                                sequence=1,
                                run_id=journey.journey_id,
                                pass_id=interval.pass_id,
                                kind="start",
                                contact_id=str(uuid.uuid4()),
                                scope=scope,
                                operation="model.request",
                            ),
                        ),
                        read_evidence(journey),
                        interval.started["event_id"],
                    )
                    history = AssertionError("a contact start reread history")
                    with (
                        self.subTest(case=name),
                        patch.object(recording, "read_contacts", side_effect=history),
                        patch.object(recording, "read_passes", side_effect=history),
                        patch.object(Journey, "read", side_effect=history),
                    ):
                        self.assertEqual(outcome(contact, scope), full)
                        if full[0] == "refused":
                            self.assertEqual(interval.journal.snapshot(), before)
                self.assertEqual(full, ("admitted",))
                interval.close("completed")
            self.assertTrue(read_passes(Journey.open(journey.path))[0].journal_complete)

    def test_closed_pass_is_read_once_per_owner_and_active_rows_once_each(self):
        from unittest.mock import patch

        import bio_neb.recording as recording

        parsed = []
        original = recording.ContactRow.model_validate_json

        def count(line, *args, **kwargs):
            parsed.append(json.loads(line)["pass_id"])
            return original(line, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary))
            origin = journey.record("JourneyStarted", {})
            attempt_id = str(uuid.uuid4())
            with journey.processing(), processing_pass(journey) as first:
                request = journey.record(
                    "Request", {"attempt_id": attempt_id}, [origin["event_id"]]
                )
                with physical_contact(
                    ContactScope(attempt_id=attempt_id, request_id=request["event_id"]),
                    "inspect",
                ):
                    pass
                first.close("completed")
            with (
                patch.object(recording.ContactRow, "model_validate_json", count),
                journey.processing(),
                processing_pass(journey) as second,
            ):
                request = journey.record(
                    "Request", {"attempt_id": attempt_id}, [request["event_id"]]
                )
                scope = ContactScope(
                    attempt_id=attempt_id, request_id=request["event_id"]
                )
                for _ in range(2):
                    with physical_contact(scope, "inspect"):
                        pass
                    self.assertEqual(len(read_passes(journey)), 2)
                    self.assertEqual(read_passes(journey), read_passes(journey))
                # Three closed rows once; the active rows once each as they grew.
                self.assertEqual(parsed.count(first.pass_id), 3)
                self.assertEqual(parsed.count(second.pass_id), 4)
                parsed.clear()
                second.close("completed")
                closed = read_passes(journey)
                self.assertEqual(read_passes(journey), closed)
                self.assertEqual(parsed.count(second.pass_id), 5)
            parsed.clear()
            with patch.object(recording.ContactRow, "model_validate_json", count):
                # An independent open is a trust boundary: it reads every row again.
                self.assertEqual(read_passes(Journey.open(journey.path)), closed)
            self.assertEqual(
                (parsed.count(first.pass_id), parsed.count(second.pass_id)), (3, 5)
            )

    def test_journal_initialization_failure_preserves_original_cause(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary))
            origin = journey.record("JourneyStarted", {})
            with journey.processing(), processing_pass(journey) as interval:
                with patch(
                    "bio_neb.recording.ContactJournal",
                    side_effect=OSError("cannot create journal"),
                ):
                    with self.assertRaisesRegex(OSError, "cannot create journal"):
                        journey.record("Result", {}, [origin["event_id"]])
                interval.close("interrupted")
            view = read_passes(Journey.open(journey.path))[0]
            self.assertEqual(view.accounting_error, "journal_unavailable")
            self.assertIsNone(view.closed_event_id)
            self.assertFalse(view.journal_complete)

    def test_early_result_survives_interleaved_contacts_and_one_final_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary))
            origin = journey.record("JourneyStarted", {})
            with journey.processing(), processing_pass(journey) as interval:
                attempt_a, attempt_b = str(uuid.uuid4()), str(uuid.uuid4())
                a = journey.record(
                    "Request",
                    {"attempt_id": attempt_a, "item_id": "a"},
                    [origin["event_id"]],
                )
                b = journey.record(
                    "Request",
                    {"attempt_id": attempt_b, "item_id": "b"},
                    [origin["event_id"]],
                )
                ca = interval.journal.start(
                    ContactScope(
                        attempt_id=attempt_a, request_id=a["event_id"], item_id="a"
                    ),
                    "inspect",
                )
                cb = interval.journal.start(
                    ContactScope(
                        attempt_id=attempt_b, request_id=b["event_id"], item_id="b"
                    ),
                    "inspect",
                )
                interval.journal.finish(cb, "succeeded")
                result = journey.record(
                    "Result", {"item_id": "b", "value": 7}, [b["event_id"]]
                )
                early = read_passes(journey)
                self.assertIsNone(early[0].closed_event_id)
                self.assertFalse(early[0].journal_complete)
                self.assertIn(result, journey.read())
                interval.journal.finish(ca, "failed", cause="TimeoutError")
                journey.record("Question", {"item_id": "a"}, [a["event_id"]])
                interval.close("waiting")
            reopened = Journey.open(journey.path)
            views = read_passes(reopened)
            self.assertEqual(len(views), 1)
            self.assertEqual(views[0].disposition, "waiting")
            self.assertTrue(views[0].journal_complete)
            self.assertEqual(
                [
                    (r.scope.item_id, r.disposition)
                    for r in views[0].contacts
                    if r.kind == "terminal"
                ],
                [("b", "succeeded"), ("a", "failed")],
            )
            self.assertIn(result, reopened.read())
            frozen = interval.journal.path.read_bytes()
            with self.assertRaisesRegex(ValueError, "not writable"):
                interval.journal.start(
                    ContactScope(attempt_id=attempt_a, request_id=a["event_id"]),
                    "inspect",
                )
            self.assertEqual(interval.journal.path.read_bytes(), frozen)

    def test_wrong_scope_terminal_is_refused_even_with_a_recomputed_final_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = ContactJournal(
                Path(temporary) / "journal.jsonl", str(uuid.uuid4()), str(uuid.uuid4())
            )
            scope = ContactScope(
                attempt_id=str(uuid.uuid4()), request_id=str(uuid.uuid4()), item_id="a"
            )
            contact = journal.start(scope, "inspect")
            journal.finish(contact, "succeeded")
            rows = [json.loads(line) for line in journal.close().splitlines()]
            rows[1]["scope"]["item_id"] = "b"
            prefix = b"".join(
                (canonical_json(row) + "\n").encode() for row in rows[:-1]
            )
            rows[-1]["prefix_sha256"] = hashlib.sha256(prefix).hexdigest()
            content = prefix + (canonical_json(rows[-1]) + "\n").encode()
            with self.assertRaisesRegex(ValueError, "exact start"):
                read_contacts(content, run_id=journal.run_id, pass_id=journal.pass_id)

    def test_cancelled_contact_and_unclosed_prefix_never_claim_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary))
            attempt_id = str(uuid.uuid4())
            origin = journey.record("JourneyStarted", {})
            with journey.processing(), processing_pass(journey) as interval:
                request = journey.record(
                    "Request", {"attempt_id": attempt_id}, [origin["event_id"]]
                )
                with self.assertRaises(KeyboardInterrupt):
                    with physical_contact(
                        ContactScope(
                            attempt_id=attempt_id, request_id=request["event_id"]
                        ),
                        "inspect",
                    ):
                        raise KeyboardInterrupt()
                prefix = read_passes(journey)[0]
                self.assertFalse(prefix.journal_complete)
                self.assertEqual(prefix.contacts[-1].disposition, "cancelled")
                interval.close("interrupted")
            view = read_passes(Journey.open(journey.path))[0]
            self.assertEqual(view.disposition, "interrupted")
            self.assertEqual(
                [r.disposition for r in view.contacts if r.kind == "terminal"],
                ["cancelled"],
            )

    def test_closure_accounting_damage_preserves_early_result(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary))
            origin = journey.record("JourneyStarted", {})
            with journey.processing(), processing_pass(journey) as interval:
                result = journey.record("Result", {"value": 7}, [origin["event_id"]])
                with patch.object(
                    interval.journal, "close", side_effect=OSError("disk unavailable")
                ):
                    interval.close("completed")
            reopened = Journey.open(journey.path)
            self.assertIn(result, reopened.read())
            view = read_passes(reopened)[0]
            self.assertFalse(view.journal_complete)
            self.assertEqual(view.accounting_error, "OSError")
            self.assertEqual(view.disposition, "completed")
