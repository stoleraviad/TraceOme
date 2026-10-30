from __future__ import annotations

import unittest
import uuid

from pydantic import ValidationError

from bio_neb.parabricks.capability_policy import (
    CapabilitySnapshotAdapter,
)
from bio_neb.parabricks.recipe_resources import (
    CapabilityGrounding,
    ListRecipesInput,
    ListRecipesTool,
    LookupResourcesInput,
    LookupResourcesTool,
    RecipeAdapterObservation,
    RecipeEligibilityAssertion,
    ResourceAdapterObservation,
    make_recipe_work_item,
    make_resource_work_item,
)
from tests.planning_fakes import ExceptionalAdapter, policy
from tests.reference_fakes import JOURNEY_ID, PLAN_ID


def recipe_work():
    grounding = CapabilityGrounding(
        journey_id=JOURNEY_ID,
        plan_id=PLAN_ID,
        requirement_id="requirement_" + "a" * 32,
        subject_key="analysis_recipe",
        source_event_ids=(JOURNEY_ID,),
    )
    return make_recipe_work_item(
        grounding=grounding,
        input_set_id=JOURNEY_ID,
        input_roles=("READ-1", "READ-2"),
        assembly_accession="GCF_000001405.40",
        companion_roles=("genomic_fasta",),
        proposal_output_values=("VCF",),
    )


def asserted_recipe_input():
    return ListRecipesInput(
        work_item=recipe_work(),
        eligibility_assertion=RecipeEligibilityAssertion(
            assay="gDNA", read_layout="paired", assertion_event_id=JOURNEY_ID
        ),
    )


def resource_work(policy_value=None):
    policy_value = policy_value or policy()
    adapter = CapabilitySnapshotAdapter(policy_value)
    tool_input = asserted_recipe_input()
    recipe = (
        ListRecipesTool()
        .list(tool_input, expected_input=tool_input, adapter=adapter)
        .candidates[0]
    )
    return make_resource_work_item(
        grounding=CapabilityGrounding(
            journey_id=JOURNEY_ID,
            plan_id=PLAN_ID,
            requirement_id="requirement_" + "b" * 32,
            subject_key="resource_profile",
            source_event_ids=(JOURNEY_ID,),
        ),
        recipe_candidate=recipe,
        recipe_event_id=JOURNEY_ID,
    )


class DuplicateAdapter(CapabilitySnapshotAdapter):
    def list_recipes(self, tool_input):
        observed = super().list_recipes(tool_input)
        candidate = observed.candidates[0]
        response = [
            candidate.model_dump(mode="json"),
            candidate.model_dump(mode="json"),
        ]
        return RecipeAdapterObservation(
            status="ok",
            candidates=(candidate, candidate),
            source=self._source(tool_input.model_dump(mode="json"), response),
        )

    def lookup_resources(self, work_item):
        observed = super().lookup_resources(work_item)
        candidate = observed.candidates[0]
        response = [
            candidate.model_dump(mode="json"),
            candidate.model_dump(mode="json"),
        ]
        return ResourceAdapterObservation(
            status="ok",
            candidates=(candidate, candidate),
            source=self._source(work_item.model_dump(mode="json"), response),
        )


class RecipeResourceToolTests(unittest.TestCase):
    def test_transient_capability_result_uses_shared_policy_before_one_tool_result(
        self,
    ):
        from unittest.mock import patch

        tool_input = asserted_recipe_input()
        adapter = ExceptionalAdapter(policy())
        failure = adapter.list_recipes(tool_input).model_copy(
            update={"failure_class": "transient"}
        )
        with (
            patch.object(adapter, "list_recipes", return_value=failure) as contact,
            patch("random.uniform", side_effect=lambda low, high: high),
            patch("time.sleep") as sleep,
        ):
            result = ListRecipesTool().list(
                tool_input, expected_input=tool_input, adapter=adapter
            )
        self.assertEqual(contact.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        self.assertEqual(result.outcome, "operational_failure")
        self.assertEqual(result.failure_class, "exhausted")
        self.assertEqual(result.original_failure_class, "transient")
        self.assertEqual(result.contact_attempt, 3)
        self.assertEqual(result.candidates, ())

    def test_operational_result_requires_an_explicit_failure_class(self):
        observed = ExceptionalAdapter(policy()).list_recipes(asserted_recipe_input())
        self.assertEqual(observed.failure_class, "rejected")
        missing_class = observed.model_dump()
        missing_class.pop("failure_class")
        with self.assertRaises(ValidationError):
            RecipeAdapterObservation.model_validate(missing_class)

    def test_multiple_candidates_without_input_assertion_are_ambiguous_not_compatible(
        self,
    ):
        work = recipe_work()
        adapter = CapabilitySnapshotAdapter(policy(recipes=2))
        result = ListRecipesTool().list(
            ListRecipesInput(work_item=work),
            expected_input=ListRecipesInput(work_item=work),
            adapter=adapter,
        )
        self.assertEqual(result.outcome, "observed")
        self.assertEqual(len(result.candidates), 2)
        self.assertTrue(
            all(item.compatibility == "ambiguous" for item in result.candidates)
        )
        self.assertFalse(hasattr(result, "selected_candidate_id"))

    def test_exact_input_assertion_can_validate_one_compatible_recipe(self):
        tool_input = asserted_recipe_input()
        result = ListRecipesTool().list(
            tool_input,
            expected_input=tool_input,
            adapter=CapabilitySnapshotAdapter(policy()),
        )
        self.assertEqual(result.outcome, "observed")
        self.assertEqual(result.candidates[0].compatibility, "compatible")

    def test_missing_recipe_is_typed(self):
        tool_input = asserted_recipe_input()
        missing = ListRecipesTool().list(
            tool_input,
            expected_input=tool_input,
            adapter=CapabilitySnapshotAdapter(policy(recipes=0)),
        )
        self.assertEqual(missing.outcome, "missing")

    def test_mixed_recipe_compatibility_keeps_canonical_candidate_order(self):
        tool_input = asserted_recipe_input()
        mixed = ListRecipesTool().list(
            tool_input,
            expected_input=tool_input,
            adapter=CapabilitySnapshotAdapter(
                policy(recipes=2, assays=("RNA", "gDNA"))
            ),
        )
        self.assertEqual(
            {item.compatibility for item in mixed.candidates},
            {"compatible", "incompatible"},
        )
        self.assertEqual(
            [item.candidate_id for item in mixed.candidates],
            sorted(item.candidate_id for item in mixed.candidates),
        )

    def test_all_incompatible_recipes_remain_scientific_negative_candidates(self):
        tool_input = asserted_recipe_input()
        incompatible = ListRecipesTool().list(
            tool_input,
            expected_input=tool_input,
            adapter=CapabilitySnapshotAdapter(policy(recipes=1, assays=("RNA",))),
        )
        self.assertTrue(
            all(
                item.compatibility == "incompatible" for item in incompatible.candidates
            )
        )

    def test_absent_required_companion_role_is_recipe_incompatibility(self):
        work = recipe_work()
        without_required_companion = make_recipe_work_item(
            grounding=work.grounding,
            input_set_id=work.input_set_id,
            input_roles=work.input_roles,
            assembly_accession=work.assembly_accession,
            companion_roles=("annotation",),
            proposal_output_values=work.proposal_output_values,
        )
        tool_input = ListRecipesInput(
            work_item=without_required_companion,
            eligibility_assertion=RecipeEligibilityAssertion(
                assay="gDNA",
                read_layout="paired",
                assertion_event_id=JOURNEY_ID,
            ),
        )
        result = ListRecipesTool().list(
            tool_input,
            expected_input=tool_input,
            adapter=CapabilitySnapshotAdapter(policy()),
        )
        self.assertEqual(result.candidates[0].compatibility, "incompatible")
        self.assertIn(
            "policy_required_companion_roles_absent",
            result.candidates[0].incompatibility_reasons,
        )

    def test_recipe_operational_failure_is_distinct_from_scientific_missing(self):
        tool_input = asserted_recipe_input()
        failed = ListRecipesTool().list(
            tool_input,
            expected_input=tool_input,
            adapter=ExceptionalAdapter(policy()),
        )
        self.assertEqual(failed.outcome, "operational_failure")
        self.assertNotEqual(failed.outcome, "missing")

    def test_recipe_candidate_identity_collisions_are_refused(self):
        tool_input = asserted_recipe_input()
        with self.assertRaisesRegex(ValueError, "identity collision"):
            ListRecipesTool().list(
                tool_input,
                expected_input=tool_input,
                adapter=DuplicateAdapter(policy()),
            )

    def test_changed_recipe_work_rejects_before_adapter_contact(self):
        work = recipe_work()
        adapter = CapabilitySnapshotAdapter(policy())
        changed = make_recipe_work_item(
            grounding=work.grounding,
            input_set_id=work.input_set_id,
            input_roles=work.input_roles,
            assembly_accession="GCF_000001405.41",
            companion_roles=work.companion_roles,
            proposal_output_values=work.proposal_output_values,
        )
        with self.assertRaisesRegex(ValueError, "ungrounded"):
            ListRecipesTool().list(
                ListRecipesInput(work_item=changed),
                expected_input=ListRecipesInput(work_item=work),
                adapter=adapter,
            )
        self.assertEqual(adapter.contact_count, 0)

    def test_fabricated_recipe_assertion_event_rejects_before_adapter_contact(self):
        work = recipe_work()
        adapter = CapabilitySnapshotAdapter(policy())
        fabricated = ListRecipesInput(
            work_item=work,
            eligibility_assertion=RecipeEligibilityAssertion(
                assay="gDNA",
                read_layout="paired",
                assertion_event_id=str(uuid.uuid4()),
            ),
        )
        with self.assertRaisesRegex(ValueError, "ungrounded"):
            ListRecipesTool().list(
                fabricated,
                expected_input=ListRecipesInput(work_item=work),
                adapter=adapter,
            )
        self.assertEqual(adapter.contact_count, 0)

    def test_resource_is_exactly_recipe_bound_and_snapshot_correlated(self):
        adapter = CapabilitySnapshotAdapter(policy())
        recipe = (
            ListRecipesTool()
            .list(
                asserted_recipe_input(),
                expected_input=asserted_recipe_input(),
                adapter=adapter,
            )
            .candidates[0]
        )
        resource_work = make_resource_work_item(
            grounding=CapabilityGrounding(
                journey_id=JOURNEY_ID,
                plan_id=PLAN_ID,
                requirement_id="requirement_" + "b" * 32,
                subject_key="resource_profile",
                source_event_ids=(JOURNEY_ID,),
            ),
            recipe_candidate=recipe,
            recipe_event_id=JOURNEY_ID,
        )
        result = LookupResourcesTool().lookup(
            LookupResourcesInput(work_item=resource_work),
            expected_work_item=resource_work,
            adapter=adapter,
        )
        self.assertEqual(result.outcome, "observed")
        self.assertEqual(result.candidates[0].recipe_candidate_id, recipe.candidate_id)
        self.assertEqual(result.source.authority, "compiled_capability_snapshot")
        self.assertEqual(result.source.claim_scope, "planning_policy_compatibility")
        self.assertEqual(result.candidates[0].availability, "not_assessed")
        self.assertIn(
            "provider_environment_availability_not_assessed",
            result.candidates[0].unresolved_constraints,
        )

    def test_missing_resource_is_typed(self):
        empty_policy = policy(resources=0)
        empty_work = resource_work(empty_policy)
        missing = LookupResourcesTool().lookup(
            LookupResourcesInput(work_item=empty_work),
            expected_work_item=empty_work,
            adapter=CapabilitySnapshotAdapter(empty_policy),
        )
        self.assertEqual(missing.outcome, "missing")

    def test_multiple_resources_keep_canonical_candidate_order(self):
        mixed_policy = policy(resources=2)
        mixed_work = resource_work(mixed_policy)
        mixed = LookupResourcesTool().lookup(
            LookupResourcesInput(work_item=mixed_work),
            expected_work_item=mixed_work,
            adapter=CapabilitySnapshotAdapter(mixed_policy),
        )
        self.assertEqual(
            {item.compatibility for item in mixed.candidates},
            {"compatible"},
        )
        self.assertEqual(
            [item.candidate_id for item in mixed.candidates],
            sorted(item.candidate_id for item in mixed.candidates),
        )

    def test_mixed_resources_use_exact_policy_recipe_mismatch(self):
        mixed_policy = policy(
            recipes=2,
            resources=2,
            resource_recipe_ids=("germline-0", "germline-1"),
        )
        adapter = CapabilitySnapshotAdapter(mixed_policy)
        recipe_input = asserted_recipe_input()
        recipe = next(
            item
            for item in ListRecipesTool()
            .list(recipe_input, expected_input=recipe_input, adapter=adapter)
            .candidates
            if item.recipe_id == "germline-0"
        )
        work = make_resource_work_item(
            grounding=CapabilityGrounding(
                journey_id=JOURNEY_ID,
                plan_id=PLAN_ID,
                requirement_id="requirement_" + "b" * 32,
                subject_key="resource_profile",
                source_event_ids=(JOURNEY_ID,),
            ),
            recipe_candidate=recipe,
            recipe_event_id=JOURNEY_ID,
        )
        result = LookupResourcesTool().lookup(
            LookupResourcesInput(work_item=work),
            expected_work_item=work,
            adapter=adapter,
        )
        self.assertEqual(
            {item.compatibility for item in result.candidates},
            {"compatible", "incompatible"},
        )
        incompatible = next(
            item for item in result.candidates if item.compatibility == "incompatible"
        )
        self.assertEqual(
            incompatible.incompatibility_reasons,
            ("policy_recipe_mismatch",),
        )

    def test_resource_operational_failure_is_distinct(self):
        mixed_policy = policy(resources=2)
        mixed_work = resource_work(mixed_policy)
        failed = LookupResourcesTool().lookup(
            LookupResourcesInput(work_item=mixed_work),
            expected_work_item=mixed_work,
            adapter=ExceptionalAdapter(mixed_policy),
        )
        self.assertEqual(failed.outcome, "operational_failure")

    def test_resource_candidate_identity_collisions_are_refused(self):
        mixed_policy = policy(resources=2)
        mixed_work = resource_work(mixed_policy)
        with self.assertRaisesRegex(ValueError, "identity collision"):
            LookupResourcesTool().lookup(
                LookupResourcesInput(work_item=mixed_work),
                expected_work_item=mixed_work,
                adapter=DuplicateAdapter(mixed_policy),
            )

    def test_resource_policy_snapshot_change_is_typed_stale_not_ready_evidence(self):
        original = policy()
        work = resource_work(original)
        changed = policy(resources=2)
        result = LookupResourcesTool().lookup(
            LookupResourcesInput(work_item=work),
            expected_work_item=work,
            adapter=CapabilitySnapshotAdapter(changed),
        )
        self.assertEqual(result.outcome, "stale_source")
        self.assertEqual(result.failure_code, "capability_snapshot_changed")
        self.assertFalse(result.candidates)

    def test_recipe_candidate_identity_covers_material_fields(self):
        recipe_candidate = (
            CapabilitySnapshotAdapter(policy())
            .list_recipes(asserted_recipe_input())
            .candidates[0]
        )
        with self.assertRaises(ValidationError):
            recipe_candidate.__class__(
                **{**recipe_candidate.model_dump(), "assay": "RNA"}
            )

    def test_resource_candidate_identity_covers_material_fields(self):
        snapshot = policy()
        work = resource_work(snapshot)
        resource_candidate = (
            CapabilitySnapshotAdapter(snapshot).lookup_resources(work).candidates[0]
        )
        with self.assertRaises(ValidationError):
            resource_candidate.__class__(
                **{**resource_candidate.model_dump(), "memory_gib": 1}
            )


if __name__ == "__main__":
    unittest.main()
