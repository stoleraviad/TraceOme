from __future__ import annotations

import copy
import unittest

from bio_neb.application.planning_read import read_planning_events
from bio_neb.parabricks.planning_hitl import (
    ClarifyPlanningHitlResponse,
    PlanningHitlCandidate,
    PlanningHitlRequested,
    SelectPlanningHitlResponse,
    make_planning_hitl_responded,
)
from bio_neb.reference_discovery.reference_companions_tool import (
    companion_asset_disposition_id,
)
from bio_neb.reference_discovery.reference_facts_journey import project_reference_facts
from tests.reference_fakes import (
    append,
    append_observation,
    attempt_start,
    companion_timeline,
    compatible_record,
    payload,
)


def request_asset_selection(timeline, work, evidence_by_asset, *, selected_asset_id):
    attempt_id = attempt_start(timeline, f"asset-hitl-{len(timeline)}")
    candidates = tuple(
        PlanningHitlCandidate(
            candidate_id=companion_asset_disposition_id(work.work_item_id, asset_id),
            asserted_value=asset_id,
            display_label=asset_id,
            evidence_event_ids=tuple(event_ids),
        )
        for asset_id, event_ids in evidence_by_asset.items()
    )
    request = PlanningHitlRequested(
        attempt_id=attempt_id,
        plan_event_id=work.grounding.source_event_ids[0],
        plan_id=work.grounding.plan_id,
        requirement_id=work.grounding.requirement_id,
        subject_key=work.grounding.subject_key,
        issue="Multiple current companion asset identities require disposition.",
        question="Which exact source-owned asset identity should be revalidated?",
        evidence_event_ids=tuple(
            dict.fromkeys(
                event_id for ids in evidence_by_asset.values() for event_id in ids
            )
        ),
        candidates=candidates,
    )
    requested = append(
        timeline, f"asset-request-{len(timeline)}", request.event_type, payload(request)
    )
    response = make_planning_hitl_responded(
        requested["event_id"],
        SelectPlanningHitlResponse(
            action="select",
            candidate_id=companion_asset_disposition_id(
                work.work_item_id, selected_asset_id
            ),
        ),
    )
    return append(
        timeline,
        f"asset-response-{len(timeline)}",
        response.event_type,
        payload(response),
    )


class ReferenceFactsProjectionTests(unittest.TestCase):
    def test_role_disposition_is_required_before_work_is_derived(self):
        pending, _, _ = companion_timeline(respond=False)
        projection = project_reference_facts(pending)
        self.assertEqual(projection.status, "planning_hitl_pending")
        self.assertEqual(projection.required_roles, ())
        self.assertEqual(projection.work_items, ())

        timeline, work, response = companion_timeline()
        projection = project_reference_facts(timeline)
        self.assertEqual(projection.status, "reference_facts_progress")
        self.assertEqual(projection.required_roles, ("genomic_fasta",))
        self.assertEqual(projection.work_items, (work,))
        self.assertEqual(projection.role_selection_event_id, response["event_id"])
        self.assertEqual(projection.pending_work_item_ids, (work.work_item_id,))

    def test_only_one_current_non_conflicted_verified_observation_satisfies_readiness(
        self,
    ):
        timeline, work, _ = companion_timeline()
        append_observation(
            timeline, work, label="verified", records=(compatible_record(work),)
        )
        ready = project_reference_facts(timeline)
        self.assertEqual(ready.status, "reference_facts")
        self.assertEqual(ready.completed_work_item_ids, (work.work_item_id,))
        self.assertEqual(ready.pending_work_item_ids, ())

    def test_unchanged_repeat_is_no_progress_and_hitl_revalidation_closes_conflict(
        self,
    ):
        timeline, work, _ = companion_timeline()
        append_observation(
            timeline,
            work,
            label="verified-a",
            records=(compatible_record(work, "opaque-a"),),
        )
        append_observation(
            timeline,
            work,
            label="verified-a-repeat",
            records=(compatible_record(work, "opaque-a"),),
        )
        repeated = project_reference_facts(timeline)
        self.assertEqual(repeated.status, "reference_facts_no_progress")
        self.assertEqual(repeated.no_progress_work_item_ids, (work.work_item_id,))
        self.assertEqual(repeated.conflicted_work_item_ids, ())

        append_observation(
            timeline,
            work,
            label="verified-b",
            records=(compatible_record(work, "opaque-b"),),
        )
        conflicted = project_reference_facts(timeline)
        self.assertEqual(conflicted.status, "reference_facts_no_progress")
        self.assertEqual(conflicted.conflicted_work_item_ids, (work.work_item_id,))
        self.assertEqual(conflicted.pending_work_item_ids, (work.work_item_id,))

        fact_events = [
            event
            for event in timeline
            if event["event_type"] == "CompanionObservationRecorded"
        ]
        response = request_asset_selection(
            timeline,
            work,
            {
                "opaque-a": (fact_events[0]["event_id"], fact_events[1]["event_id"]),
                "opaque-b": (fact_events[2]["event_id"],),
            },
            selected_asset_id="opaque-a",
        )
        append_observation(
            timeline,
            work,
            label="selected-a",
            records=(compatible_record(work, "opaque-a"),),
            selected_asset_id="opaque-a",
            selection_event_id=response["event_id"],
        )
        closed = project_reference_facts(timeline)
        self.assertEqual(closed.status, "reference_facts")
        self.assertEqual(closed.conflicted_work_item_ids, ())
        self.assertEqual(closed.no_progress_work_item_ids, ())

    def test_multiple_observation_requires_selection_and_fresh_tool_validation(self):
        timeline, work, _ = companion_timeline()
        observed = append_observation(
            timeline,
            work,
            label="multiple",
            records=(
                compatible_record(work, "opaque-a"),
                compatible_record(work, "opaque-b"),
            ),
        )
        response = request_asset_selection(
            timeline,
            work,
            {
                "opaque-a": (observed["event_id"],),
                "opaque-b": (observed["event_id"],),
            },
            selected_asset_id="opaque-b",
        )
        disposition = project_reference_facts(timeline)
        self.assertEqual(disposition.status, "reference_facts_progress")
        self.assertEqual(disposition.asset_dispositions[0].asset_id, "opaque-b")
        selected_prefix = copy.deepcopy(timeline)
        selected = read_planning_events(selected_prefix)
        self.assertEqual(
            selected.companions.pending_work_item_ids, (work.work_item_id,)
        )
        self.assertEqual(selected.verified_companion_bindings, ())
        verified = append_observation(
            timeline,
            work,
            label="selected-b",
            records=(compatible_record(work, "opaque-b"),),
            selected_asset_id="opaque-b",
            selection_event_id=response["event_id"],
        )
        self.assertEqual(project_reference_facts(timeline).status, "reference_facts")
        ready = read_planning_events(timeline)
        self.assertEqual(ready.companions.status, "reference_facts")
        self.assertEqual(len(ready.verified_companion_bindings), 1)
        binding = ready.verified_companion_bindings[0]
        self.assertEqual(binding.candidate.asset_id, "opaque-b")
        self.assertEqual(binding.work_item_id, work.work_item_id)
        self.assertEqual(binding.validation_event_id, verified["event_id"])
        self.assertEqual(
            read_planning_events(selected_prefix).verified_companion_bindings, ()
        )
        # The read is shared, not copied: a caller cannot change it.
        with self.assertRaisesRegex(TypeError, "read-only"):
            ready.companions.facts[-1].result.clear()
        self.assertEqual(
            read_planning_events(timeline).companions.status, "reference_facts"
        )

    def test_partial_clarification_pending_then_unchanged_response_is_no_progress(
        self,
    ):
        timeline, work, _ = companion_timeline()
        observed = append_observation(
            timeline,
            work,
            label="partial",
            records=(compatible_record(work, authority=False),),
        )
        attempt_id = attempt_start(timeline, "partial-hitl")
        request = PlanningHitlRequested(
            attempt_id=attempt_id,
            plan_event_id=work.grounding.source_event_ids[0],
            plan_id=work.grounding.plan_id,
            requirement_id=work.grounding.requirement_id,
            subject_key=work.grounding.subject_key,
            issue="Companion observation cannot progress from outcome=partial.",
            question="Clarify the exact required companion.",
            evidence_event_ids=(observed["event_id"],),
        )
        requested = append(
            timeline, "partial-request", request.event_type, payload(request)
        )
        self.assertEqual(
            project_reference_facts(timeline).status, "planning_hitl_pending"
        )
        response = make_planning_hitl_responded(
            requested["event_id"],
            ClarifyPlanningHitlResponse(
                action="clarify", instruction="No new identity."
            ),
        )
        append(timeline, "partial-response", response.event_type, payload(response))
        projected = project_reference_facts(timeline)
        self.assertEqual(projected.status, "reference_facts_no_progress")
        self.assertEqual(projected.no_progress_work_item_ids, (work.work_item_id,))

    def test_exact_replay_and_tampered_assembly_or_role_fail_closed(self):
        timeline, work, _ = companion_timeline()
        append_observation(
            timeline, work, label="verified", records=(compatible_record(work),)
        )
        first = project_reference_facts(timeline)
        second = project_reference_facts(copy.deepcopy(timeline))
        self.assertEqual(first.model_dump_json(), second.model_dump_json())

        changed = copy.deepcopy(timeline)
        fact = next(
            event
            for event in changed
            if event["event_type"] == "CompanionObservationRecorded"
        )
        fact["payload"]["tool_input"]["work_item"]["reference"][
            "assembly_accession"
        ] = "GCF_000001405.39"
        fact["payload"]["result"]["work_item"]["reference"]["assembly_accession"] = (
            "GCF_000001405.39"
        )
        with self.assertRaisesRegex(ValueError, "work identity|exact work item"):
            project_reference_facts(changed)

        changed_role = copy.deepcopy(timeline)
        fact = next(
            event
            for event in changed_role
            if event["event_type"] == "CompanionObservationRecorded"
        )
        fact["payload"]["tool_input"]["work_item"]["role"] = "annotation"
        fact["payload"]["result"]["work_item"]["role"] = "annotation"
        with self.assertRaisesRegex(ValueError, "work identity|exact work item"):
            project_reference_facts(changed_role)


if __name__ == "__main__":
    unittest.main()
