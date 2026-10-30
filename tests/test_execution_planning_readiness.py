from __future__ import annotations

import copy
import unittest

from bio_neb.agentic_journey.attempt_journey import ToolCallValidated
from bio_neb.parabricks.capability_policy import (
    CapabilitySnapshotAdapter,
)
from bio_neb.parabricks.execution_planning_journey import (
    PlanningCandidateSelected,
    RecipeObservationRecorded,
    ResourceObservationRecorded,
    project_execution_planning,
)
from bio_neb.parabricks.planning_hitl import (
    PlanningHitlCandidate,
    PlanningHitlRequested,
    SelectPlanningHitlResponse,
    make_planning_hitl_responded,
)
from bio_neb.parabricks.recipe_resources import (
    ListRecipesTool,
    LookupResourcesInput,
    LookupResourcesTool,
    RecipeAdapterObservation,
    candidate_disposition_id,
    resource_recipe_dependency_id,
)
from bio_neb.parabricks.requirement_resolution import project_requirement_resolution
from tests.planning_fakes import (
    add_goal_requirement,
    add_inspection,
    add_requirement,
    add_target_requirement,
    append_asserted_recipe_observation,
    policy,
)
from tests.recipe_resource_fakes import IncompatibleAdapter
from tests.reference_fakes import (
    append,
    append_observation,
    attempt_start,
    companion_timeline,
    compatible_record,
    payload,
)


class OperationalFailureAdapter(CapabilitySnapshotAdapter):
    def list_recipes(self, tool_input):
        self.contact_count += 1
        return RecipeAdapterObservation(
            status="operational_failure",
            candidates=(),
            source=self._source(tool_input.model_dump(mode="json"), []),
            failure_code="synthetic_transport_failure",
            failure_class="transient",
        )


def _planning_timeline():
    timeline, companion_work, _ = companion_timeline()
    add_requirement(timeline, "analysis_recipe", "analysis_recipe", "1")
    add_requirement(timeline, "resource_profile", "resource_profile", "2")
    add_target_requirement(timeline)
    add_inspection(timeline)
    append_observation(
        timeline,
        companion_work,
        label="verified-companion",
        records=(compatible_record(companion_work),),
    )
    return timeline


def _selected_recipe_timeline():
    timeline = _planning_timeline()
    initial = project_execution_planning(timeline)
    adapter = CapabilitySnapshotAdapter(policy())
    work, recipe_result, recipe_event = append_asserted_recipe_observation(
        timeline, adapter, "recipe"
    )
    observed = project_execution_planning(timeline)
    selection_attempt = attempt_start(timeline, "recipe-selection-attempt")
    candidate = recipe_result.candidates[0]
    validated = append(
        timeline,
        "recipe-selection-validated",
        "ToolCallValidated",
        payload(
            ToolCallValidated(
                attempt_id=selection_attempt,
                tool_name="select_recipe_candidate",
                tool_call_id="select-recipe",
                arguments={
                    "work_item_id": work.work_item_id,
                    "candidate_id": candidate.candidate_id,
                    "basis_event_id": recipe_event["event_id"],
                },
                status="valid",
                errors=(),
            )
        ),
    )
    append(
        timeline,
        "recipe-selected",
        "PlanningCandidateSelected",
        payload(
            PlanningCandidateSelected(
                attempt_id=selection_attempt,
                kind="recipe",
                work_item_id=work.work_item_id,
                candidate_id=candidate.candidate_id,
                basis_event_id=recipe_event["event_id"],
                authority="agent_ranking",
                authority_event_id=validated["event_id"],
            )
        ),
    )
    return {
        "timeline": timeline,
        "initial": initial,
        "observed": observed,
        "adapter": adapter,
        "recipe_ready": project_execution_planning(timeline),
    }


def _ready_planning_timeline():
    context = _selected_recipe_timeline()
    timeline = context["timeline"]
    adapter = context["adapter"]
    recipe_ready = context["recipe_ready"]
    resource_attempt = attempt_start(timeline, "resource-observation-attempt")
    resource_input = LookupResourcesInput(work_item=recipe_ready.resource_work_item)
    resource_result = LookupResourcesTool().lookup(
        resource_input,
        expected_work_item=recipe_ready.resource_work_item,
        adapter=adapter,
    )
    resource_event = append(
        timeline,
        "resource-observation",
        "ResourceObservationRecorded",
        payload(
            ResourceObservationRecorded(
                attempt_id=resource_attempt,
                tool_input=resource_input,
                result=resource_result,
            )
        ),
    )
    resource_observed = project_execution_planning(timeline)
    resource_selection_attempt = attempt_start(timeline, "resource-selection-attempt")
    resource = resource_result.candidates[0]
    validated = append(
        timeline,
        "resource-selection-validated",
        "ToolCallValidated",
        payload(
            ToolCallValidated(
                attempt_id=resource_selection_attempt,
                tool_name="select_resource_candidate",
                tool_call_id="select-resource",
                arguments={
                    "work_item_id": recipe_ready.resource_work_item.work_item_id,
                    "candidate_id": resource.candidate_id,
                    "basis_event_id": resource_event["event_id"],
                },
                status="valid",
                errors=(),
            )
        ),
    )
    append(
        timeline,
        "resource-selected",
        "PlanningCandidateSelected",
        payload(
            PlanningCandidateSelected(
                attempt_id=resource_selection_attempt,
                kind="resource",
                work_item_id=recipe_ready.resource_work_item.work_item_id,
                candidate_id=resource.candidate_id,
                basis_event_id=resource_event["event_id"],
                authority="agent_ranking",
                authority_event_id=validated["event_id"],
            )
        ),
    )
    return {
        **context,
        "resource_input": resource_input,
        "resource_observed": resource_observed,
        "ready": project_execution_planning(timeline),
    }


class ExecutionPlanningReadinessTests(unittest.TestCase):
    def test_the_fixed_bam_and_vcf_outputs_match_the_vcf_recipe_output(self):
        # Execution always produces BAM + VCF, so recipe listing matches a
        # recipe by that fixed contract, never by an output the request names.
        timeline, companion_work, _ = companion_timeline()
        add_requirement(timeline, "analysis_recipe", "analysis_recipe", "1")
        add_requirement(timeline, "resource_profile", "resource_profile", "2")
        add_target_requirement(timeline)
        add_inspection(timeline)
        append_observation(
            timeline,
            companion_work,
            label="germline-vcf-companion",
            records=(compatible_record(companion_work),),
        )

        projected = project_execution_planning(timeline)
        result = ListRecipesTool().list(
            projected.recipe_tool_input,
            expected_input=projected.recipe_tool_input,
            adapter=CapabilitySnapshotAdapter(policy()),
        )
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]

        self.assertEqual(candidate.compatibility, "ambiguous")
        self.assertEqual(
            candidate.incompatibility_reasons,
            ("input_eligibility_assertion_required",),
        )

    def test_ambiguous_recipe_needs_assertion(self):
        timeline, companion_work, _ = companion_timeline()
        add_requirement(timeline, "analysis_recipe", "analysis_recipe", "1")
        add_requirement(timeline, "resource_profile", "resource_profile", "2")
        add_target_requirement(timeline)
        add_inspection(timeline)
        append_observation(
            timeline,
            companion_work,
            label="eligibility-companion",
            records=(compatible_record(companion_work),),
        )
        initial = project_execution_planning(timeline)
        attempt_id = attempt_start(timeline, "ambiguous-recipe-attempt")
        result = ListRecipesTool().list(
            initial.recipe_tool_input,
            expected_input=initial.recipe_tool_input,
            adapter=CapabilitySnapshotAdapter(policy(recipes=2)),
        )
        append(
            timeline,
            "ambiguous-recipe",
            "RecipeObservationRecorded",
            payload(
                RecipeObservationRecorded(
                    attempt_id=attempt_id,
                    tool_input=initial.recipe_tool_input,
                    result=result,
                )
            ),
        )
        ambiguous = project_execution_planning(timeline)
        row = next(
            item for item in ambiguous.requirements if item.kind == "analysis_recipe"
        )
        self.assertEqual(row.state, "needs_selection")
        self.assertEqual(row.next_actions, ("request_recipe_eligibility_assertion",))
        self.assertEqual(ambiguous.status, "recipe_selection_pending")

    def test_zero_recipe_result_is_typed_failure(self):
        timeline = _planning_timeline()
        missing_initial = project_execution_planning(timeline)
        missing_attempt = attempt_start(timeline, "missing-recipe-attempt")
        missing = ListRecipesTool().list(
            missing_initial.recipe_tool_input,
            expected_input=missing_initial.recipe_tool_input,
            adapter=CapabilitySnapshotAdapter(policy(recipes=0)),
        )
        append(
            timeline,
            "missing-recipe",
            "RecipeObservationRecorded",
            payload(
                RecipeObservationRecorded(
                    attempt_id=missing_attempt,
                    tool_input=missing_initial.recipe_tool_input,
                    result=missing,
                )
            ),
        )
        failed = project_execution_planning(timeline)
        self.assertEqual(
            next(
                item for item in failed.requirements if item.kind == "analysis_recipe"
            ).state,
            "typed_failure",
        )
        self.assertEqual(failed.status, "typed_failure")

    def test_recipe_operational_failure_blocks_without_scientific_negative(self):
        timeline = _planning_timeline()
        projected = project_execution_planning(timeline)
        attempt_id = attempt_start(timeline, "operational-recipe-failure-attempt")
        result = ListRecipesTool().list(
            projected.recipe_tool_input,
            expected_input=projected.recipe_tool_input,
            adapter=OperationalFailureAdapter(policy()),
        )
        append(
            timeline,
            "operational-recipe-failure",
            "RecipeObservationRecorded",
            payload(
                RecipeObservationRecorded(
                    attempt_id=attempt_id,
                    tool_input=projected.recipe_tool_input,
                    result=result,
                )
            ),
        )
        blocked = project_execution_planning(timeline)
        row = next(
            item for item in blocked.requirements if item.kind == "analysis_recipe"
        )
        self.assertEqual(result.outcome, "operational_failure")
        self.assertNotEqual(result.outcome, "missing")
        self.assertFalse(result.candidates)
        self.assertEqual(row.state, "typed_failure")
        self.assertEqual(row.candidate_ids, ())
        self.assertEqual(blocked.status, "typed_failure")

    def test_changed_capability_snapshot_invalidates_selected_recipe(self):
        context = _selected_recipe_timeline()
        changed_snapshot = copy.deepcopy(context["timeline"])
        recipe_ready = context["recipe_ready"]
        stale_attempt = attempt_start(changed_snapshot, "stale-resource-policy-attempt")
        stale_input = LookupResourcesInput(work_item=recipe_ready.resource_work_item)
        stale_result = LookupResourcesTool().lookup(
            stale_input,
            expected_work_item=recipe_ready.resource_work_item,
            adapter=CapabilitySnapshotAdapter(policy(resources=2)),
        )
        append(
            changed_snapshot,
            "stale-resource-policy",
            "ResourceObservationRecorded",
            payload(
                ResourceObservationRecorded(
                    attempt_id=stale_attempt,
                    tool_input=stale_input,
                    result=stale_result,
                )
            ),
        )
        stale_projection = project_execution_planning(changed_snapshot)
        self.assertEqual(stale_result.outcome, "stale_source")
        self.assertEqual(
            next(
                item
                for item in stale_projection.requirements
                if item.kind == "analysis_recipe"
            ).state,
            "conflicted",
        )
        self.assertEqual(stale_projection.status, "recipe_validation_pending")
        self.assertFalse(stale_projection.selected_recipe_candidates)

    def test_planning_ready_remains_not_execution_ready(self):
        context = _ready_planning_timeline()
        self.assertEqual(
            context["resource_observed"].status, "resource_selection_pending"
        )
        ready = context["ready"]
        self.assertEqual(ready.status, "execution_planning_ready")
        self.assertIn(
            "prepared_reference_not_verified", ready.unresolved_execution_constraints
        )
        self.assertFalse(ready.unresolved_requirement_ids)
        self.assertEqual(len(ready.selected_resource_bindings), 1)
        recipe_row = next(
            item for item in ready.requirements if item.kind == "analysis_recipe"
        )
        resource_row = next(
            item for item in ready.requirements if item.kind == "resource_profile"
        )
        self.assertIn(
            ready.selected_recipe_bindings[0].selection_event_id,
            recipe_row.evidence_event_ids,
        )
        self.assertIn(
            ready.selected_resource_bindings[0].selection_event_id,
            resource_row.evidence_event_ids,
        )

    def test_resolved_research_goal_keeps_the_execution_resolution(self):
        timeline = _ready_planning_timeline()["timeline"]
        with_goal = copy.deepcopy(timeline)
        goal_id = add_goal_requirement(with_goal)

        before = project_requirement_resolution(timeline)
        after = project_requirement_resolution(with_goal)

        goal = next(
            item for item in after.requirements if item.requirement_id == goal_id
        )
        self.assertEqual(goal.state, "resolved")
        self.assertIsNotNone(before.execution_resolution)
        self.assertEqual(after.execution_resolution, before.execution_resolution)
        self.assertEqual(
            project_execution_planning(with_goal).status, "execution_planning_ready"
        )

    def test_unresolved_subjects_block_ready_and_exact_replay_is_stable(self):
        context = _ready_planning_timeline()
        timeline = context["timeline"]
        ready = context["ready"]
        incomplete_plan = copy.deepcopy(timeline)
        add_requirement(incomplete_plan, "sample_identity", "secondary_sample", "8")
        add_requirement(incomplete_plan, "companion_asset", "secondary_companion", "9")
        incomplete = project_execution_planning(incomplete_plan)
        self.assertNotEqual(incomplete.status, "execution_planning_ready")
        self.assertTrue(incomplete.unresolved_requirement_ids)
        self.assertEqual(
            ready.model_dump(mode="json"),
            project_execution_planning(
                list(reversed(list(reversed(timeline))))
            ).model_dump(mode="json"),
        )

    def test_newer_incompatible_resource_invalidates_selection(self):
        context = _ready_planning_timeline()
        timeline = context["timeline"]
        resource_input = context["resource_input"]
        recipe_ready = context["recipe_ready"]
        negative_resource = LookupResourcesTool().lookup(
            resource_input,
            expected_work_item=recipe_ready.resource_work_item,
            adapter=IncompatibleAdapter(policy()),
        )
        negative_resource_attempt = attempt_start(
            timeline, "newer-incompatible-resource-attempt"
        )
        append(
            timeline,
            "newer-incompatible-resource",
            "ResourceObservationRecorded",
            payload(
                ResourceObservationRecorded(
                    attempt_id=negative_resource_attempt,
                    tool_input=resource_input,
                    result=negative_resource,
                )
            ),
        )
        regressed = project_execution_planning(timeline)
        resource_row = next(
            item for item in regressed.requirements if item.kind == "resource_profile"
        )
        self.assertEqual(resource_row.state, "conflicted")
        self.assertEqual(regressed.status, "resource_validation_pending")
        self.assertFalse(regressed.selected_resource_candidates)

    def test_current_resource_after_negative_requires_reselection(self):
        context = _ready_planning_timeline()
        timeline = context["timeline"]
        resource_input = context["resource_input"]
        recipe_ready = context["recipe_ready"]
        negative = LookupResourcesTool().lookup(
            resource_input,
            expected_work_item=recipe_ready.resource_work_item,
            adapter=IncompatibleAdapter(policy()),
        )
        append(
            timeline,
            "newer-incompatible-resource",
            "ResourceObservationRecorded",
            payload(
                ResourceObservationRecorded(
                    attempt_id=attempt_start(
                        timeline, "newer-incompatible-resource-attempt"
                    ),
                    tool_input=resource_input,
                    result=negative,
                )
            ),
        )
        current_resource = LookupResourcesTool().lookup(
            resource_input,
            expected_work_item=recipe_ready.resource_work_item,
            adapter=context["adapter"],
        )
        current_resource_attempt = attempt_start(
            timeline, "newer-compatible-resource-attempt"
        )
        append(
            timeline,
            "newer-compatible-resource",
            "ResourceObservationRecorded",
            payload(
                ResourceObservationRecorded(
                    attempt_id=current_resource_attempt,
                    tool_input=resource_input,
                    result=current_resource,
                )
            ),
        )
        needs_reselection = project_execution_planning(timeline)
        resource_row = next(
            item
            for item in needs_reselection.requirements
            if item.kind == "resource_profile"
        )
        self.assertEqual(resource_row.state, "needs_selection")
        self.assertEqual(needs_reselection.status, "resource_selection_pending")
        self.assertFalse(needs_reselection.selected_resource_candidates)

    def test_newer_incompatible_recipe_invalidates_selection(self):
        context = _ready_planning_timeline()
        timeline = context["timeline"]
        ready = context["ready"]
        negative_recipe = ListRecipesTool().list(
            ready.recipe_tool_input,
            expected_input=ready.recipe_tool_input,
            adapter=IncompatibleAdapter(policy()),
        )
        negative_recipe_attempt = attempt_start(
            timeline, "newer-incompatible-recipe-attempt"
        )
        append(
            timeline,
            "newer-incompatible-recipe",
            "RecipeObservationRecorded",
            payload(
                RecipeObservationRecorded(
                    attempt_id=negative_recipe_attempt,
                    tool_input=ready.recipe_tool_input,
                    result=negative_recipe,
                )
            ),
        )
        recipe_regressed = project_execution_planning(timeline)
        recipe_row = next(
            item
            for item in recipe_regressed.requirements
            if item.kind == "analysis_recipe"
        )
        self.assertEqual(recipe_row.state, "conflicted")
        self.assertEqual(recipe_regressed.status, "recipe_validation_pending")
        self.assertFalse(recipe_regressed.selected_recipe_candidates)

    def test_stale_hitl_selection_rejects_changed_validation_ancestry(
        self,
    ):
        timeline, companion_work, _ = companion_timeline()
        add_requirement(timeline, "analysis_recipe", "analysis_recipe", "1")
        add_requirement(timeline, "resource_profile", "resource_profile", "2")
        add_target_requirement(timeline)
        add_inspection(timeline)
        append_observation(
            timeline,
            companion_work,
            label="hitl-companion",
            records=(compatible_record(companion_work),),
        )
        adapter = CapabilitySnapshotAdapter(policy(recipes=2))
        work, result, basis = append_asserted_recipe_observation(
            timeline, adapter, "hitl-recipe"
        )
        projected = project_execution_planning(timeline)
        request_attempt = attempt_start(timeline, "hitl-recipe-request")
        requirement = next(
            item for item in projected.requirements if item.kind == "analysis_recipe"
        )
        candidates = tuple(
            PlanningHitlCandidate(
                candidate_id=candidate_disposition_id(
                    "recipe", work.work_item_id, item.candidate_id
                ),
                asserted_value=item.candidate_id,
                display_label=item.recipe_id,
                evidence_event_ids=(basis["event_id"],),
            )
            for item in result.candidates
        )
        request = PlanningHitlRequested(
            attempt_id=request_attempt,
            plan_event_id=projected.plan_event_id,
            plan_id=projected.plan_id,
            requirement_id=requirement.requirement_id,
            subject_key=requirement.subject_key,
            issue="Multiple compatible recipe policies.",
            question="Select exact recipe.",
            evidence_event_ids=(basis["event_id"],),
            candidates=candidates,
        )
        requested = append(
            timeline, "hitl-recipe-requested", request.event_type, payload(request)
        )
        responded = make_planning_hitl_responded(
            requested["event_id"],
            SelectPlanningHitlResponse(
                action="select", candidate_id=candidates[1].candidate_id
            ),
        )
        response = append(
            timeline, "hitl-recipe-responded", responded.event_type, payload(responded)
        )
        selection_attempt = attempt_start(timeline, "stale-researcher-selection")
        append(
            timeline,
            "stale-researcher-selected",
            "PlanningCandidateSelected",
            payload(
                PlanningCandidateSelected(
                    attempt_id=selection_attempt,
                    kind="recipe",
                    work_item_id=work.work_item_id,
                    candidate_id=result.candidates[1].candidate_id,
                    basis_event_id=projected.plan_event_id,
                    authority="researcher_selection",
                    authority_event_id=response["event_id"],
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "validation ancestry"):
            project_execution_planning(timeline)

    def test_additional_same_kind_subject_remains_independently_unresolved(self):
        timeline, companion_work, _ = companion_timeline()
        add_requirement(timeline, "analysis_recipe", "analysis_recipe", "1")
        add_requirement(timeline, "analysis_recipe", "secondary_recipe", "3")
        add_requirement(timeline, "resource_profile", "resource_profile", "2")
        add_target_requirement(timeline)
        add_inspection(timeline)
        append_observation(
            timeline,
            companion_work,
            label="verified-companion-extra",
            records=(compatible_record(companion_work),),
        )
        projected = project_execution_planning(timeline)
        recipes = [
            item for item in projected.requirements if item.kind == "analysis_recipe"
        ]
        self.assertEqual(len(recipes), 2)
        self.assertEqual(
            [item.state for item in recipes], ["needs_validation", "needs_validation"]
        )
        self.assertNotEqual(projected.status, "execution_planning_ready")

    def test_extra_upstream_subject_cannot_inherit_aggregate_reference_or_companion_evidence(
        self,
    ):
        timeline, companion_work, _ = companion_timeline()
        add_requirement(timeline, "sample_identity", "secondary_sample", "6")
        add_requirement(timeline, "companion_asset", "secondary_companion", "7")
        add_requirement(timeline, "analysis_recipe", "analysis_recipe", "1")
        add_requirement(timeline, "resource_profile", "resource_profile", "2")
        add_target_requirement(timeline)
        add_inspection(timeline)
        append_observation(
            timeline,
            companion_work,
            label="extra-upstream-companion",
            records=(compatible_record(companion_work),),
        )
        projected = project_execution_planning(timeline)
        by_subject = {item.subject_key: item for item in projected.requirements}
        self.assertEqual(by_subject["secondary_sample"].state, "missing")
        self.assertEqual(by_subject["secondary_companion"].state, "missing")
        self.assertNotEqual(projected.status, "execution_planning_ready")

    def test_two_recipe_two_resource_subjects_progress_independently(self):
        timeline, companion_work, _ = companion_timeline()
        add_requirement(timeline, "analysis_recipe", "analysis_recipe_tumor", "1")
        add_requirement(timeline, "analysis_recipe", "analysis_recipe_normal", "3")
        add_requirement(timeline, "resource_profile", "resource_profile_tumor", "2")
        add_requirement(timeline, "resource_profile", "resource_profile_normal", "5")
        add_target_requirement(timeline)
        add_inspection(timeline)
        append_observation(
            timeline,
            companion_work,
            label="two-by-two-companion",
            records=(compatible_record(companion_work),),
        )
        adapter = CapabilitySnapshotAdapter(policy())

        def select(kind, work, result, basis, label):
            attempt = attempt_start(timeline, f"{label}-selection-attempt")
            candidate = result.candidates[0]
            validated = append(
                timeline,
                f"{label}-validated",
                "ToolCallValidated",
                payload(
                    ToolCallValidated(
                        attempt_id=attempt,
                        tool_name=f"select_{kind}_candidate",
                        tool_call_id=label,
                        arguments={
                            "work_item_id": work.work_item_id,
                            "candidate_id": candidate.candidate_id,
                            "basis_event_id": basis["event_id"],
                        },
                        status="valid",
                        errors=(),
                    )
                ),
            )
            append(
                timeline,
                f"{label}-selected",
                "PlanningCandidateSelected",
                payload(
                    PlanningCandidateSelected(
                        attempt_id=attempt,
                        kind=kind,
                        work_item_id=work.work_item_id,
                        candidate_id=candidate.candidate_id,
                        basis_event_id=basis["event_id"],
                        authority="agent_ranking",
                        authority_event_id=validated["event_id"],
                    )
                ),
            )

        for index in range(2):
            work, result, basis = append_asserted_recipe_observation(
                timeline, adapter, f"recipe-{index}"
            )
            select("recipe", work, result, basis, f"recipe-{index}")
        projected = project_execution_planning(timeline)
        resource_requirements = [
            item for item in projected.requirements if item.kind == "resource_profile"
        ]
        for index, requirement in enumerate(resource_requirements):
            request_attempt = attempt_start(timeline, f"dependency-{index}-attempt")
            candidates = tuple(
                PlanningHitlCandidate(
                    candidate_id=resource_recipe_dependency_id(
                        requirement.requirement_id,
                        binding.requirement_id,
                        binding.selection_event_id,
                    ),
                    asserted_value=binding.requirement_id,
                    display_label=binding.subject_key,
                    evidence_event_ids=(binding.selection_event_id,),
                )
                for binding in projected.selected_recipe_bindings
            )
            request = PlanningHitlRequested(
                attempt_id=request_attempt,
                plan_event_id=projected.plan_event_id,
                plan_id=projected.plan_id,
                requirement_id=requirement.requirement_id,
                subject_key=requirement.subject_key,
                issue="Explicit recipe dependency required.",
                question="Which recipe owns this resource?",
                evidence_event_ids=tuple(
                    binding.selection_event_id
                    for binding in projected.selected_recipe_bindings
                ),
                candidates=candidates,
            )
            requested = append(
                timeline,
                f"dependency-{index}-requested",
                request.event_type,
                payload(request),
            )
            responded = make_planning_hitl_responded(
                requested["event_id"],
                SelectPlanningHitlResponse(
                    action="select", candidate_id=candidates[index].candidate_id
                ),
            )
            append(
                timeline,
                f"dependency-{index}-responded",
                responded.event_type,
                payload(responded),
            )
            projected = project_execution_planning(timeline)
            if index == 0:
                self.assertNotEqual(projected.status, "execution_planning_ready")
        projected = project_execution_planning(timeline)
        self.assertEqual(len(projected.resource_work_items), 2)
        for index in range(2):
            projected = project_execution_planning(timeline)
            work = projected.resource_work_item
            attempt = attempt_start(timeline, f"resource-{index}-attempt")
            tool_input = LookupResourcesInput(work_item=work)
            result = LookupResourcesTool().lookup(
                tool_input, expected_work_item=work, adapter=adapter
            )
            basis = append(
                timeline,
                f"resource-{index}-observed",
                "ResourceObservationRecorded",
                payload(
                    ResourceObservationRecorded(
                        attempt_id=attempt, tool_input=tool_input, result=result
                    )
                ),
            )
            select("resource", work, result, basis, f"resource-{index}")
        ready = project_execution_planning(timeline)
        self.assertEqual(ready.status, "execution_planning_ready")
        self.assertEqual(len(ready.selected_recipe_candidates), 2)
        self.assertEqual(len(ready.selected_resource_candidates), 2)


if __name__ == "__main__":
    unittest.main()
