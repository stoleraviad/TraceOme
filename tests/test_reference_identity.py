from __future__ import annotations

import copy
import unittest

from pydantic import ValidationError

from bio_neb.agentic_journey.attempt_journey import SliceCheckpointReached
from bio_neb.agentic_journey.lifecycle import ActivityStarted
from bio_neb.reference_discovery.reference_journey import (
    ReferenceIdentityProjection,
    project_reference_identity,
)
from tests.reference_fakes import (
    INPUT_SET_ID,
    JOURNEY_ID,
    REFERENCE_REQUIREMENT_ID,
    SAMPLE_REQUIREMENT_ID,
    TAXON_REQUIREMENT_ID,
    build_timeline,
    identity,
)


class ReferenceJourneyTests(unittest.TestCase):
    def test_projection_advances_through_exact_tool_and_hitl_evidence(self) -> None:
        timeline, stages = build_timeline()
        expected = (
            ("plan", "reference_identity_ready"),
            ("sample", "sample_identity"),
            ("taxon", "taxon_identity"),
            ("discovery", "reference_candidates"),
            ("pending", "planning_hitl_pending"),
            ("responded", "reference_candidates"),
            ("final", "reference_identity"),
        )
        for stage, status in expected:
            with self.subTest(stage=stage):
                projected = project_reference_identity(timeline[: stages[stage]])
                self.assertEqual(projected.status, status)
        final = project_reference_identity(timeline)
        self.assertEqual(len(final.planning_hitl.requests), 6)
        self.assertEqual(len(final.planning_hitl.receipts), 6)
        self.assertIsNone(final.planning_hitl.pending_request)
        self.assertEqual(final.verified_reference.assembly_accession, "GCF_000009999.1")
        self.assertEqual(final.status, "reference_identity")
        locators = {item.kind: item for item in final.requirements}
        self.assertEqual(final.journey_id, JOURNEY_ID)
        self.assertEqual(
            locators["sample_identity"].requirement_id, SAMPLE_REQUIREMENT_ID
        )
        self.assertEqual(
            locators["taxon_identity"].requirement_id, TAXON_REQUIREMENT_ID
        )
        self.assertEqual(
            locators["reference_assembly_identity"].requirement_id,
            REFERENCE_REQUIREMENT_ID,
        )
        changed = final.model_dump(mode="json")
        changed["requirements"][1]["subject_key"] = changed["requirements"][0][
            "subject_key"
        ]
        with self.assertRaisesRegex(ValidationError, "exact requirement set"):
            ReferenceIdentityProjection.model_validate(changed)

    def test_projection_is_exact_replay(self) -> None:
        timeline, _ = build_timeline()
        first = project_reference_identity(timeline)
        second = project_reference_identity(copy.deepcopy(timeline))
        self.assertEqual(first.model_dump_json(), second.model_dump_json())

    def test_projection_rejects_foreign_requirement_grounding(self) -> None:
        timeline, _ = build_timeline()
        changed_requirement = copy.deepcopy(timeline)
        sample = next(
            item
            for item in changed_requirement
            if item["event_type"] == "SampleAttestationRecorded"
        )
        replacement = "requirement_" + "f" * 32
        sample["payload"]["tool_input"]["grounding"]["requirement_id"] = replacement
        sample["payload"]["result"]["grounding"]["requirement_id"] = replacement
        with self.assertRaisesRegex(ValueError, "non-reference requirement"):
            project_reference_identity(changed_requirement)

    def test_projection_rejects_nonancestral_reference_selection(self) -> None:
        timeline, _ = build_timeline()
        changed_selection = copy.deepcopy(timeline)
        description = next(
            item
            for item in changed_selection
            if item["event_type"] == "ReferenceDescriptionRecorded"
        )
        stale = identity("stale-selection-response")
        description["payload"]["tool_input"]["selected"]["selection_event_id"] = stale
        description["payload"]["result"]["selected"]["selection_event_id"] = stale
        description["payload"]["tool_input"]["grounding"]["source_event_ids"][-1] = (
            stale
        )
        description["payload"]["result"]["grounding"]["source_event_ids"][-1] = stale
        with self.assertRaisesRegex(ValueError, "non-ancestral evidence"):
            project_reference_identity(changed_selection)

    def test_reference_checkpoints_and_lifecycle_pairs_remain_strict(self) -> None:
        checkpoint = SliceCheckpointReached(
            attempt_id=identity("checkpoint-attempt"),
            checkpoint="reference_identity_progress",
            input_set_id=INPUT_SET_ID,
            proposal_plan_event_id=identity("plan"),
            reference_evidence_event_id=identity("description-result"),
        )
        self.assertEqual(checkpoint.checkpoint, "reference_identity_progress")
        with self.assertRaises(ValidationError):
            SliceCheckpointReached(
                attempt_id=identity("checkpoint-attempt"),
                checkpoint="reference_identity_progress",
                input_set_id=INPUT_SET_ID,
                proposal_plan_event_id=identity("plan"),
            )
        ActivityStarted(
            activity_id=identity("activity"),
            adapter="ncbi_datasets",
            stage="describe_reference",
            attempt=1,
        )
        with self.assertRaises(ValidationError):
            ActivityStarted(
                activity_id=identity("bad-activity"),
                adapter="deterministic",
                stage="describe_reference",
                attempt=1,
            )


if __name__ == "__main__":
    unittest.main()
