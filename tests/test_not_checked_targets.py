"""A target the Run cannot check by what it states (#367, readiness Q2 b).

A target that no placement step placed, and that states no rsID, is recorded not checked
with its reason by the coordinator before any model request. It asks
nothing (no ``investigation_target_missing`` refusal or researcher question), no rsID is
guessed for it, and it is its own accounted row through Finalization and the report.
An item with only words is no longer skipped (#472 R1 b): the model reads them.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bio_neb.investigation.collection_view import collection_view
from bio_neb.investigation.investigation import (
    UserAssertion,
    investigation_output_member_id,
)
from bio_neb.investigation.investigation_request import (
    InvestigationRequestItem,
    RequestPlacement,
)
from bio_neb.recording import read_evidence
from tests.sequential_investigation_support import (
    ASSEMBLY_BODY,
    SequentialAdaptiveModel,
    SequentialSourceResponses,
    input_and_assertion,
    run_attempt,
)


class NotCheckedTargetTests(unittest.TestCase):
    """#367 (readiness Q2 b): a target with no rsID that no step placed is recorded
    not checked, with its reason, before any model request; it asks
    nothing and is its own accounted row (no investigation_target_missing refusal)."""

    def _collection(self, *items):
        investigation_input, _ = input_and_assertion()
        assertion = UserAssertion.create_collection(
            asserted_by="researcher",
            requests=items,
            predecessor_ids=(investigation_input.input_id,),
        )
        return investigation_input, assertion

    def test_an_unplaced_target_with_no_rsid_is_recorded_not_checked(self):
        checked = InvestigationRequestItem(
            request_text="Investigate rs429358 in sample HG001.", rsid="rs429358"
        )
        unplaced = InvestigationRequestItem(
            request_text="a deletion upstream of the promoter (placement)",
            placement=RequestPlacement(
                step="none", reason="the paper states no rsID or position for it"
            ),
            asserted_by="paper:doi:10.1000/x",
        )
        investigation_input, assertion = self._collection(checked, unplaced)
        model = SequentialAdaptiveModel()
        with tempfile.TemporaryDirectory() as directory:
            result, journey = run_attempt(
                Path(directory),
                model,
                SequentialSourceResponses(assembly_body=ASSEMBLY_BODY),
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        recorded = [
            e["payload"]
            for e in events
            if e["event_type"] == "InvestigationRequestNotChecked"
        ]
        self.assertEqual(
            [
                (r["collection_member_id"], r["reason_code"], r["reason"])
                for r in recorded
            ],
            [
                (
                    investigation_output_member_id(unplaced, 1),
                    "not_placed",
                    "the paper states no rsID or position for it",
                ),
            ],
        )
        bound = [
            e["payload"]
            for e in events
            if e["event_type"] == "InvestigationAttemptBound"
        ]
        self.assertEqual(
            {b["collection_member_id"] for b in bound},
            {investigation_output_member_id(checked, 0)},
        )
        self.assertNotIn("investigation_target_missing", json.dumps(events))
        proposal = result.pending_request.proposal
        members = proposal.request_members
        self.assertEqual(
            [(m.not_checked.reason_code if m.not_checked else None) for m in members],
            [None, "not_placed"],
        )
        self.assertEqual(
            [(m.refsnp, m.work_item_ids) for m in members[1:]], [(None, ())]
        )
        self.assertEqual(proposal.unresolved_request_member_ids, ())
        # The collection view lists each with its reason and its own asserter.
        view = collection_view(proposal)
        self.assertEqual(
            [(snp.asserted_by, snp.no_work_note()) for snp in view.snps[1:]],
            [
                (
                    "paper:doi:10.1000/x",
                    "not checked: the paper states no rsID or position for it",
                ),
            ],
        )

    def test_words_that_name_no_target_are_read_and_asked_about_not_skipped(self):
        """#472 R1 b, refined (orchestrator msg_35431e3d4d18): an item with only words
        is read by the model, and a reading that finds no target opens the member's
        Review asking which target the words mean; it is not recorded not checked."""
        stated_nothing = InvestigationRequestItem(
            request_text="the variant the researcher saw last week"
        )
        investigation_input, assertion = self._collection(stated_nothing)
        with tempfile.TemporaryDirectory() as directory:
            result, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                SequentialSourceResponses(assembly_body=ASSEMBLY_BODY),
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        types = [e["event_type"] for e in events]
        self.assertNotIn("InvestigationRequestNotChecked", types)
        (reading,) = (
            e["payload"] for e in events if e["event_type"] == "InvestigationTargetRead"
        )
        self.assertEqual(
            (reading["source"], reading["words"], reading.get("target")),
            ("request", stated_nothing.request_text, None),
        )
        self.assertIn(stated_nothing.request_text, reading["reason"])
        (review,) = (
            e["payload"]
            for e in events
            if e["event_type"] == "InvestigationReviewRequested"
        )
        question = review["question"]
        self.assertEqual(
            (question["asks_about"], review["legal_actions"]),
            ("target", ["clarify", "reject"]),
        )
        self.assertIn(stated_nothing.request_text, question["question"])
        self.assertIsNone(result.pending_request.proposal)
