"""Independent worked identity contracts at public domain boundaries."""

import tempfile
import unittest
from pathlib import Path

from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationDefaults,
    AgentInvocationOverrides,
    ContinuationPolicy,
    resolve_agent_invocation_policy,
)
from bio_neb.application.queries import (
    ReviewJourneyQuery,
    review_journey,
)
from bio_neb.failure import FailureSurface, FailureTuning, policy_for
from bio_neb.journey import Journey
from bio_neb.parabricks.capability_policy import CapabilitySnapshotAdapter
from bio_neb.parabricks.execution_proposal import execution_authority_id
from bio_neb.parabricks.recipe_resources import (
    CapabilityGrounding,
    ListRecipesInput,
    ListRecipesTool,
    LookupResourcesInput,
    LookupResourcesTool,
    RecipeEligibilityAssertion,
    candidate_disposition_id,
    make_recipe_work_item,
    make_resource_work_item,
    recipe_eligibility_assertion_id,
    resource_recipe_dependency_id,
)
from tests.content_identity_vectors import expected_digest, fields
from tests.planning_assignment_fakes import prepare_configured_run
from tests.planning_fakes import policy

# Independent field membership for public JSON contracts, including optional proposal packaging.
_EXECUTION_VECTORS = {
    "execution-file": (
        "file_id",
        "role registered_path byte_count sha256 filesystem_device filesystem_inode modified_at_ns registration_event_id inspection_event_id",
    ),
    "execution-package": (
        "package_id",
        "journey_id source_plan input_set_id inputs reference workflow resource outputs",
    ),
    "execution-proposal": (
        "proposal_id",
        "journey_id plan_event_id requirement_resolution inspection_event_id inspected_inputs reference_event_id reference companions reference_source_admission recipes resources unresolved_execution_constraints evidence_closure supersedes_proposal_id revision_event_id revision_direction",
    ),
    "reference-source-admission": (
        "admission_id",
        "reference_asset_id assembly_accession authority_locator source_mode source_locator source_representation expected_size_bytes expected_checksum_algorithm expected_checksum checksum_authority_locator derivation_rule",
    ),
    "requirement-value": (
        "value_id",
        "requirement_id kind subject_key value source status lineage_event_ids",
    ),
    "execution-resolution": (
        "resolution_id",
        "journey_id plan_event_id plan_id input_set_id requirements",
    ),
    "scientific-investigation": ("handoff_id", "journey_id plan_event_id items"),
}


def assert_execution_vectors(test, value):
    seen = set()

    def visit(item):
        if isinstance(item, dict):
            for namespace, (key, names) in _EXECUTION_VECTORS.items():
                if key in item and str(item[key]).startswith(namespace + ":"):
                    # Nested pointers carrying only an identity are not complete contracts.
                    if not set(names.split()) <= item.keys():
                        continue
                    material = fields(item, names)
                    if namespace == "execution-proposal":
                        # Optional members join the identity only when present.
                        for optional in (
                            "execution_package",
                            "fact_slots",
                            "reference_basis",
                            "configuration",
                            "runtime_selection",
                            # The judged recipe selection's ploidy notice.
                            "ploidy",
                        ):
                            if optional in item:
                                material[optional] = item[optional]
                    if namespace == "scientific-investigation" and (
                        "research_goals" in item
                    ):
                        # The goal joins the handoff identity when the plan states one.
                        material["research_goals"] = item["research_goals"]
                    test.assertEqual(
                        item[key],
                        namespace + ":" + expected_digest(material),
                        namespace,
                    )
                    seen.add(namespace)
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return seen


class ContentIdentityTests(unittest.TestCase):
    def test_policy_identities_preserve_exact_resolved_material(self):
        from bio_neb.agentic_journey.model_settings import model_endpoint_policy

        selection = model_endpoint_policy("Qwen/Qwen3.5-397B-A17B").settings_selection
        # One failure rule; an untuned stage records no key.
        continuation = ContinuationPolicy(failed_turns=5)
        self.assertEqual(
            continuation.policy_id,
            "continuation-policy:" + expected_digest({"failed_turns": 5}),
        )
        staged = ContinuationPolicy(failed_turns=5, stage_failed_turns={"reference": 2})
        self.assertEqual(
            staged.policy_id,
            "continuation-policy:"
            + expected_digest(
                {"failed_turns": 5, "stage_failed_turns": {"reference": 2}}
            ),
        )
        effective = resolve_agent_invocation_policy(
            AgentInvocationDefaults(model_reference="qwen", failed_turns=5),
            AgentInvocationOverrides(),
            agent_role="planning_agent",
            model_settings={"qwen": selection},
            model_contact=policy_for(
                FailureSurface(scope="role", name="planning_agent"), FailureTuning()
            ),
        )
        expected = {
            "agent_role": "planning_agent",
            "model_reference": "qwen",
            # None means an omitted setting, so it is not identity material.
            "settings_selection": {
                "model": "Qwen/Qwen3.5-397B-A17B",
                "accepted_settings": [
                    "parallel_tool_calls",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "seed",
                ],
                "baseline": {"parallel_tool_calls": False, "max_tokens": 245760},
                "context_budget": {
                    "context_window": 262144,
                    "input_reserve": 16384,
                    "serialized_body_safety_margin": 8192,
                    "source": "preserved_product_configuration",
                },
            },
            # The per-call window.
            "model_contact": {
                "contacts": 2,
                "base_delay_seconds": 30.0,
                "max_delay_seconds": 60.0,
                "max_wait_seconds": 60.0,
                "stall_seconds": 180.0,
            },
            # The failure module's resolution records each value's layer.
            "model_contact_sources": {
                "contacts": "default",
                "base_delay_seconds": "default",
                "max_delay_seconds": "default",
                "max_wait_seconds": "default",
                "stall_seconds": "default",
            },
            "model_requests": 200,
            "loop_retry": 0,
            "tool_retry": 0,
            "continuation": {"failed_turns": 5},
            "exhaustion": {
                "failed_turns": "review",
                "model_availability": "review",
            },
            "model_settings": {"parallel_tool_calls": False, "max_tokens": 245760},
            # An unset setting the model leaves omitted has no source either.
            "settings_sources": {
                "parallel_tool_calls": "model_default",
                "max_tokens": "model_default",
            },
            "stage_settings": {},
        }
        self.assertEqual(
            effective.policy_id, "invocation-policy:" + expected_digest(expected)
        )

    def test_an_unset_new_optional_setting_leaves_the_invocation_policy_id(self):
        # A setting added later with no value is an omission, not new identity
        # material, so a Run's recorded policy_id survives the addition.
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationPolicy
        from bio_neb.agentic_journey.model_settings import (
            AgentModelSettings,
            model_endpoint_policy,
        )

        class WithNewSetting(AgentModelSettings):
            reasoning_effort: str | None = None

        class PolicyWithNewSetting(AgentInvocationPolicy):
            model_settings: WithNewSetting

        selection = model_endpoint_policy("Qwen/Qwen3.5-397B-A17B").settings_selection
        policy = resolve_agent_invocation_policy(
            AgentInvocationDefaults(model_reference="qwen", failed_turns=5),
            AgentInvocationOverrides(),
            agent_role="planning_agent",
            model_settings={"qwen": selection},
            model_contact=policy_for(
                FailureSurface(scope="role", name="planning_agent"), FailureTuning()
            ),
        )
        # Through the merge, an unset setting (as a new one would be) records no
        # value and no source, so nothing of it reaches the digest.
        for unset in ("temperature", "top_p", "top_k", "seed"):
            self.assertIsNone(getattr(policy.model_settings, unset))
            self.assertNotIn(unset, policy.settings_sources)
        extended = PolicyWithNewSetting.model_validate(policy.model_dump())
        self.assertIsNone(extended.model_dump()["model_settings"]["reasoning_effort"])
        self.assertEqual(extended.policy_id, policy.policy_id)

    def test_recipe_resource_public_identities_preserve_their_distinct_material(self):
        event = "11111111-1111-4111-8111-111111111111"
        grounding = CapabilityGrounding(
            journey_id=event,
            plan_id="plan_" + "a" * 64,
            requirement_id="requirement_" + "a" * 32,
            subject_key="analysis_recipe",
            source_event_ids=(event,),
        )
        ground = {
            "journey_id": event,
            "plan_id": "plan_" + "a" * 64,
            "requirement_id": "requirement_" + "a" * 32,
            "subject_key": "analysis_recipe",
            "source_event_ids": [event],
        }
        work = make_recipe_work_item(
            grounding=grounding,
            input_set_id=event,
            input_roles=("READ-1", "READ-2"),
            assembly_accession="GCF_000001405.40",
            companion_roles=("genomic_fasta",),
            proposal_output_values=("VCF",),
        )
        self.assertEqual(
            work.work_item_id,
            "recipe-work:"
            + expected_digest(
                [
                    ground,
                    event,
                    ["READ-1", "READ-2"],
                    "GCF_000001405.40",
                    ["genomic_fasta"],
                    ["VCF"],
                ]
            ),
        )
        snapshot = policy()
        snapshot_material = fields(
            snapshot.model_dump(mode="json"),
            "schema_version capability_version source_manifest recipes resources images tools",
        )
        self.assertEqual(snapshot.snapshot_id, expected_digest(snapshot_material))
        adapter = CapabilitySnapshotAdapter(snapshot)
        tool_input = ListRecipesInput(
            work_item=work,
            eligibility_assertion=RecipeEligibilityAssertion(
                assay="gDNA", read_layout="paired", assertion_event_id=event
            ),
        )
        result = ListRecipesTool().list(
            tool_input, expected_input=tool_input, adapter=adapter
        )
        recipe = result.candidates[0]
        candidate = recipe.model_dump(mode="json")
        recipe_material = [
            candidate[key]
            for key in "recipe_id assay read_layout output_kind requires_prepared_reference requires_companion_roles compatibility compatibility_scope incompatibility_reasons unresolved_constraints snapshot_id".split()
        ]
        self.assertEqual(
            recipe.candidate_id, "recipe-candidate:" + expected_digest(recipe_material)
        )
        self.assertEqual(
            recipe.source_evidence_id,
            "recipe-evidence:"
            + expected_digest(recipe_material + [recipe.candidate_id]),
        )
        self.assertEqual(
            result.evidence_id,
            "recipe-result:"
            + expected_digest(
                fields(
                    result.model_dump(mode="json"),
                    "work_item outcome candidates source failure_code",
                )
            ),
        )
        request = {
            "work_item": work.model_dump(mode="json"),
            "eligibility_assertion": {
                "assay": "gDNA",
                "read_layout": "paired",
                "assertion_event_id": event,
            },
        }
        self.assertEqual(result.source.request_identity, expected_digest(request))
        self.assertEqual(result.source.response_digest, expected_digest([candidate]))
        resource_work = make_resource_work_item(
            grounding=grounding, recipe_candidate=recipe, recipe_event_id=event
        )
        self.assertEqual(
            resource_work.work_item_id,
            "resource-work:" + expected_digest([ground, candidate, event]),
        )
        resource_input = LookupResourcesInput(work_item=resource_work)
        resource = LookupResourcesTool().lookup(
            resource_input, expected_work_item=resource_work, adapter=adapter
        )
        selected = resource.candidates[0]
        values = selected.model_dump(mode="json")
        resource_material = [
            values[key]
            for key in "profile_id recipe_candidate_id accelerator_count cpu_count memory_gib accelerator_memory_gib availability compatibility compatibility_scope incompatibility_reasons unresolved_constraints snapshot_id".split()
        ]
        self.assertEqual(
            selected.candidate_id,
            "resource-candidate:" + expected_digest(resource_material),
        )
        self.assertEqual(
            selected.source_evidence_id,
            "resource-evidence:"
            + expected_digest(resource_material + [selected.candidate_id]),
        )
        self.assertEqual(
            resource.evidence_id,
            "resource-result:"
            + expected_digest(
                fields(
                    resource.model_dump(mode="json"),
                    "work_item outcome candidates source failure_code",
                )
            ),
        )
        self.assertEqual(
            resource.source.request_identity,
            expected_digest(
                {
                    "work_item_id": resource_work.work_item_id,
                    "grounding": ground,
                    "recipe_candidate": candidate,
                    "recipe_event_id": event,
                }
            ),
        )
        self.assertEqual(resource.source.response_digest, expected_digest([values]))
        for kind, item in (("recipe", recipe), ("resource", selected)):
            self.assertEqual(
                candidate_disposition_id(kind, work.work_item_id, item.candidate_id),
                kind
                + "-candidate-disposition:"
                + expected_digest([work.work_item_id, item.candidate_id]),
            )
        self.assertEqual(
            recipe_eligibility_assertion_id(work.work_item_id, recipe),
            "recipe-eligibility-assertion:"
            + expected_digest(
                [
                    work.work_item_id,
                    recipe.assay,
                    recipe.read_layout,
                    snapshot.snapshot_id,
                ]
            ),
        )
        self.assertEqual(
            resource_recipe_dependency_id("resource", "recipe", event),
            "resource-recipe-dependency:"
            + expected_digest(["resource", "recipe", event]),
        )

    def test_registered_execution_public_identities_preserve_namespace_and_field_membership(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            receipt, review, _ = prepare_configured_run(Path(directory).resolve())
            request = review.execution_hitl.pending_request
            public = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            seen = assert_execution_vectors(
                self,
                [
                    public.model_dump(mode="json"),
                    request.model_dump(mode="json"),
                    Journey.open(Path(receipt.journey_file)).read(),
                ],
            )
            self.assertGreaterEqual(seen, set(_EXECUTION_VECTORS))
            proposal = request.proposal.proposal_id
            approval = "22222222-2222-4222-8222-222222222222"
            self.assertEqual(
                execution_authority_id(proposal, request.request_event_id, approval),
                "execution-authority:"
                + expected_digest(
                    {
                        "proposal_id": proposal,
                        "request_event_id": request.request_event_id,
                        "approval_event_id": approval,
                    }
                ),
            )


class ContentIdTests(unittest.TestCase):
    # Independent worked example: SHA-256 of b"abc" (FIPS 180-2, appendix B.1).
    ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    def test_content_id_names_exact_bytes_and_round_trips_its_text(self):
        from bio_neb.content import ContentId

        identity = ContentId.of(b"abc")

        self.assertEqual(identity.digest, self.ABC)
        self.assertEqual(str(identity), "sha256:" + self.ABC)
        self.assertEqual(ContentId.parse("sha256:" + self.ABC), identity)
        self.assertEqual(ContentId(self.ABC), identity)

    def test_content_id_refuses_other_algorithms_and_malformed_digests(self):
        from bio_neb.content import ContentId

        for value in (
            self.ABC,
            "SHA256:" + self.ABC,
            "sha512:" + self.ABC,
            "sha256:" + self.ABC.upper(),
            "sha256:" + self.ABC[:-1],
            "sha256:" + self.ABC + "0",
            "sha256:" + self.ABC[:-1] + "g",
            " sha256:" + self.ABC,
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ContentId.parse(value)
        for digest in ("sha256:" + self.ABC, self.ABC[:-1], self.ABC.upper()):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                ContentId(digest)
