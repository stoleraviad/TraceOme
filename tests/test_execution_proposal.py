from __future__ import annotations

import copy
import unittest
import uuid

from pydantic import ValidationError

from bio_neb.agentic_journey.attempt_journey import ToolCallValidated
from bio_neb.application.run_failure import RunFailed
from bio_neb.parabricks.capability_policy import (
    CapabilitySnapshotAdapter,
)
from bio_neb.parabricks.execution_planning_journey import (
    PlanningCandidateSelected,
    ResourceObservationRecorded,
    project_execution_planning,
)
from bio_neb.parabricks.execution_proposal import (
    ApproveExecutionResponse,
    ExecutionHitlResponded,
    ExecutionHitlResponseRefused,
    ExecutionProposal,
    ProposalCompilationRefused,
    RejectExecutionResponse,
    ReviseExecutionResponse,
    compile_execution_proposal,
    evaluate_execution_hitl_response,
    project_execution_hitl,
)
from bio_neb.parabricks.planning_hitl import (
    PlanningHitlCandidate,
    PlanningHitlRequested,
    SelectPlanningHitlResponse,
    make_planning_hitl_responded,
)
from bio_neb.parabricks.recipe_resources import (
    LookupResourcesInput,
    LookupResourcesTool,
)
from bio_neb.reference_discovery.reference_companions_tool import (
    CompanionRequirementGrounding,
    VerifiedReferenceBinding,
    derive_companion_work_items,
)
from bio_neb.reference_discovery.reference_facts_journey import project_reference_facts
from bio_neb.reference_discovery.reference_identity_tool import (
    AssemblyAdapterObservation,
    DescribeReferenceInput,
    DescribeReferenceTool,
    DiscoverReferenceAssembliesInput,
    DiscoverReferenceAssembliesTool,
    ResolveTaxonInput,
    ResolveTaxonTool,
    SampleAttestationInput,
    SelectedAssemblyEvidence,
    TaxonomyAdapterObservation,
    TaxonomySourceRecord,
    VerifySampleAttestationTool,
)
from bio_neb.reference_discovery.reference_journey import (
    ReferenceAssembliesRecorded,
    ReferenceDescriptionRecorded,
    SampleAttestationRecorded,
    TaxonResolutionRecorded,
    project_reference_identity,
)
from tests.execution_proposal_fakes import ready_timeline, submit
from tests.planning_fakes import (
    append_asserted_recipe_observation,
    policy,
)
from tests.reference_fakes import (
    INPUT_SET_ID,
    JOURNEY_ID,
    PLAN_ID,
    REFERENCE_REQUIREMENT_ID,
    SAMPLE_REQUIREMENT_ID,
    TAXON_REQUIREMENT_ID,
    FakeAdapter,
    append,
    append_index_rechoice,
    append_observation,
    assembly,
    attempt_start,
    compatible_record,
    grounding,
    payload,
)
from tests.reference_fakes import reference_source as source


def revalidate_reference_and_companion(timeline):
    plan_event = next(
        item
        for item in timeline
        if item["event_type"] == "ProposalRequirementsIdentified"
    )
    requirements = plan_event["payload"]["plan"]["requirements"]
    by_kind = {item["kind"]: item for item in requirements}
    sample_assertion = by_kind["sample_identity"]["evidence"][0]["source_event_id"]
    taxon_assertion = by_kind["taxon_identity"]["evidence"][0]["source_event_id"]
    context = by_kind["reference_assembly_identity"]["evidence"][0]["source_event_id"]

    sample_attempt = attempt_start(timeline, f"revised-sample-{len(timeline)}")
    sample_grounding = grounding(
        SAMPLE_REQUIREMENT_ID,
        "sample_identity",
        "sample_identity",
        plan_event["event_id"],
        sample_assertion,
    )
    sample_input = SampleAttestationInput(
        grounding=sample_grounding,
        input_set_id=INPUT_SET_ID,
        sample_identifier="sample-001",
        assertion_event_id=sample_assertion,
        quote="sample-001",
    )
    sample_event = append(
        timeline,
        f"revised-sample-result-{len(timeline)}",
        "SampleAttestationRecorded",
        payload(
            SampleAttestationRecorded(
                attempt_id=sample_attempt,
                tool_input=sample_input,
                result=VerifySampleAttestationTool().verify(
                    sample_input,
                    expected_grounding=sample_grounding,
                    expected_input_set_id=INPUT_SET_ID,
                    expected_sample_identifier="sample-001",
                ),
            )
        ),
    )
    taxonomy = TaxonomyAdapterObservation(
        records=(
            TaxonomySourceRecord(
                ncbi_tax_id=9606, scientific_name="Homo sapiens", rank="species"
            ),
        ),
        source=source("ncbi_taxonomy", "taxonomy/9606"),
    )
    assemblies = AssemblyAdapterObservation(
        records=(assembly("GCF_000001405.40"), assembly("GCF_000009999.1")),
        source=source("ncbi_datasets", "genome/taxon/9606"),
        total_count=2,
    )
    exact = AssemblyAdapterObservation(
        records=(assembly("GCF_000009999.1"),),
        source=source("ncbi_datasets", "genome/accession/GCF_000009999.1"),
        total_count=1,
    )
    adapter = FakeAdapter(taxonomy, assemblies, exact)
    taxon_attempt = attempt_start(timeline, f"revised-taxon-{len(timeline)}")
    taxon_grounding = grounding(
        TAXON_REQUIREMENT_ID,
        "taxon_identity",
        "taxon_identity",
        plan_event["event_id"],
        taxon_assertion,
        sample_event["event_id"],
    )
    taxon_input = ResolveTaxonInput(
        grounding=taxon_grounding,
        query="Homo sapiens",
        assertion_event_id=taxon_assertion,
    )
    taxon_result = ResolveTaxonTool().resolve(
        taxon_input,
        expected_grounding=taxon_grounding,
        expected_query="Homo sapiens",
        adapter=adapter,
    )
    taxon_event = append(
        timeline,
        f"revised-taxon-result-{len(timeline)}",
        "TaxonResolutionRecorded",
        payload(
            TaxonResolutionRecorded(
                attempt_id=taxon_attempt,
                tool_input=taxon_input,
                result=taxon_result,
                source_answers=(),
            )
        ),
    )
    resolved_taxon = project_reference_identity(timeline).resolved_taxon
    discovery_attempt = attempt_start(timeline, f"revised-discovery-{len(timeline)}")
    reference_grounding = grounding(
        REFERENCE_REQUIREMENT_ID,
        "reference_assembly_identity",
        "reference_assembly",
        plan_event["event_id"],
        context,
        taxon_event["event_id"],
    )
    discovery_input = DiscoverReferenceAssembliesInput(
        grounding=reference_grounding, taxon=resolved_taxon
    )
    discovery_result = DiscoverReferenceAssembliesTool().discover(
        discovery_input,
        expected_grounding=reference_grounding,
        expected_taxon=resolved_taxon,
        expected_strain_or_isolate=None,
        expected_attributed_assembly_accession=None,
        expected_refinement_event_id=None,
        adapter=adapter,
    )
    discovery_event = append(
        timeline,
        f"revised-discovery-result-{len(timeline)}",
        "ReferenceAssembliesRecorded",
        payload(
            ReferenceAssembliesRecorded(
                attempt_id=discovery_attempt,
                tool_input=discovery_input,
                result=discovery_result,
                source_answers=(),
            )
        ),
    )
    selection_attempt = attempt_start(
        timeline, f"revised-reference-hitl-{len(timeline)}"
    )
    candidates = tuple(
        PlanningHitlCandidate(
            candidate_id=item.candidate_id,
            asserted_value=item.assembly_accession,
            display_label=item.assembly_name,
            evidence_event_ids=(discovery_event["event_id"],),
        )
        for item in discovery_result.candidates
    )
    request = PlanningHitlRequested(
        attempt_id=selection_attempt,
        plan_event_id=plan_event["event_id"],
        plan_id=PLAN_ID,
        requirement_id=REFERENCE_REQUIREMENT_ID,
        subject_key="reference_assembly",
        issue="Fresh reference evidence requires exact selection.",
        question="Which exact assembly should be revalidated?",
        evidence_event_ids=(discovery_event["event_id"],),
        candidates=candidates,
    )
    requested = append(
        timeline,
        f"revised-reference-request-{len(timeline)}",
        request.event_type,
        payload(request),
    )
    response = make_planning_hitl_responded(
        requested["event_id"],
        SelectPlanningHitlResponse(
            action="select", candidate_id=candidates[1].candidate_id
        ),
    )
    response_event = append(
        timeline,
        f"revised-reference-response-{len(timeline)}",
        response.event_type,
        payload(response),
    )
    describe_attempt = attempt_start(timeline, f"revised-description-{len(timeline)}")
    selected_candidate = discovery_result.candidates[1]
    description_grounding = grounding(
        REFERENCE_REQUIREMENT_ID,
        "reference_assembly_identity",
        "reference_assembly",
        plan_event["event_id"],
        context,
        taxon_event["event_id"],
        discovery_event["event_id"],
        response_event["event_id"],
    )
    selected = SelectedAssemblyEvidence(
        candidate=selected_candidate,
        discovery_evidence_id=discovery_result.evidence_id,
        discovery_event_id=discovery_event["event_id"],
        selection_event_id=response_event["event_id"],
    )
    description_input = DescribeReferenceInput(
        grounding=description_grounding,
        selected=selected,
        assembly_accession=selected_candidate.assembly_accession,
    )
    description_result = DescribeReferenceTool().describe(
        description_input,
        expected_grounding=description_grounding,
        expected_selection=selected,
        adapter=adapter,
    )
    description_event = append(
        timeline,
        f"revised-description-result-{len(timeline)}",
        "ReferenceDescriptionRecorded",
        payload(
            ReferenceDescriptionRecorded(
                attempt_id=describe_attempt,
                tool_input=description_input,
                result=description_result,
                source_answers=(),
            )
        ),
    )
    companion_requirement = by_kind["companion_asset"]
    role_attempt = attempt_start(timeline, f"revised-companion-role-{len(timeline)}")
    role_candidate = PlanningHitlCandidate(
        candidate_id="companion-role-set:genomic_fasta",
        asserted_value="genomic_fasta",
        display_label="Assembly roles: genomic_fasta",
        evidence_event_ids=(description_event["event_id"],),
    )
    role_request = PlanningHitlRequested(
        attempt_id=role_attempt,
        plan_event_id=plan_event["event_id"],
        plan_id=PLAN_ID,
        requirement_id=companion_requirement["requirement_id"],
        subject_key=companion_requirement["subject_key"],
        issue="Fresh reference requires companion roles.",
        question="Which role is required?",
        evidence_event_ids=(description_event["event_id"],),
        candidates=(role_candidate,),
    )
    role_requested = append(
        timeline,
        f"revised-companion-request-{len(timeline)}",
        role_request.event_type,
        payload(role_request),
    )
    role_response = make_planning_hitl_responded(
        role_requested["event_id"],
        SelectPlanningHitlResponse(
            action="select", candidate_id=role_candidate.candidate_id
        ),
    )
    role_response_event = append(
        timeline,
        f"revised-companion-response-{len(timeline)}",
        role_response.event_type,
        payload(role_response),
    )
    reference = project_reference_identity(timeline)
    companion_grounding = CompanionRequirementGrounding(
        journey_id=JOURNEY_ID,
        plan_id=PLAN_ID,
        requirement_id=companion_requirement["requirement_id"],
        subject_key=companion_requirement["subject_key"],
        source_event_ids=(
            plan_event["event_id"],
            timeline[0]["event_id"],
            description_event["event_id"],
            role_response_event["event_id"],
        ),
    )
    binding = VerifiedReferenceBinding(
        reference_event_id=description_event["event_id"],
        assembly_accession=reference.verified_reference.assembly_accession,
        reference_evidence_id=reference.verified_reference.evidence_id,
    )
    work = derive_companion_work_items(
        grounding=companion_grounding,
        reference=binding,
        selected_roles=("genomic_fasta",),
        role_selection_event_ids=(role_response_event["event_id"],),
    )[0]
    append_observation(
        timeline,
        work,
        label=f"revised-companion-observation-{len(timeline)}",
        records=(compatible_record(work),),
    )
    assert project_reference_facts(timeline).status == "reference_facts"


class ExecutionProposalTests(unittest.TestCase):
    def test_submitted_proposal_requires_evidence_inside_request_ancestry(self):
        timeline = ready_timeline()
        submit(timeline)
        outside = copy.deepcopy(timeline)
        outside[-1]["parent_ids"] = [outside[0]["event_id"]]
        with self.assertRaisesRegex(ValueError, "outside request ancestry"):
            project_execution_hitl(outside)

    def test_execution_hitl_response_requires_direct_request_parent(self):
        timeline = ready_timeline()
        _, request = submit(timeline)
        proposal = compile_execution_proposal(timeline[:-1])
        intermediary = append(timeline, "review-viewed", "ReviewViewed", {})
        approve = ApproveExecutionResponse(
            action="approve", proposal_id=proposal.proposal_id
        )
        response = append(
            timeline,
            "indirect-approval",
            "ExecutionHitlResponded",
            payload(
                ExecutionHitlResponded(
                    request_event_id=request["event_id"], response=approve
                )
            ),
        )
        self.assertEqual(response["parent_ids"], [intermediary["event_id"]])
        with self.assertRaisesRegex(ValueError, "outside request ancestry"):
            project_execution_hitl(timeline)

    def test_compiles_complete_canonical_proposal_and_preserves_opaque_identity(self):
        opaque_ids = (
            "provider-a:GCF_000005845.2",
            "provider-b:GCF_000005845.2",
            "ncbi-datasets:GCF_000005845.2:genomic-fasta",
        )
        proposals = tuple(
            compile_execution_proposal(ready_timeline(asset_id=opaque_id))
            for opaque_id in opaque_ids
        )
        for opaque_id, proposal in zip(opaque_ids, proposals):  # noqa: B905 - Existing fixture pairing preserved during import relocation.
            self.assertEqual(proposal.companions[0].candidate.asset_id, opaque_id)
        self.assertEqual(len({item.proposal_id for item in proposals}), len(opaque_ids))
        proposal = proposals[0]

        self.assertEqual(
            proposal,
            compile_execution_proposal(
                copy.deepcopy(ready_timeline(asset_id=opaque_ids[0]))
            ),
        )
        evidence = {
            identity
            for item in proposal.evidence_closure
            for identity in item.evidence_identities
        }
        self.assertIn(proposal.reference.evidence_id, evidence)
        self.assertIn(proposal.companions[0].candidate.evidence_id, evidence)
        self.assertIn(proposal.recipes[0].candidate.source_evidence_id, evidence)
        self.assertIn(proposal.resources[0].candidate.source_evidence_id, evidence)

        changed = proposal.model_dump(mode="json")
        changed["companions"][0]["candidate"]["asset_id"] = "GCF_000005845.2"
        with self.assertRaisesRegex(ValidationError, "proposal identity omitted"):
            ExecutionProposal.model_validate(changed)
        missing = proposal.model_dump(mode="json")
        missing["companions"][0]["candidate"]["asset_id"] = ""
        with self.assertRaisesRegex(ValidationError, "non-empty"):
            ExecutionProposal.model_validate(missing)
        changed_basis = proposal.model_dump(mode="json")
        closure = next(
            item
            for item in changed_basis["evidence_closure"]
            if item["evidence_identities"]
        )
        closure["evidence_identities"][0] += ":changed-basis"
        with self.assertRaisesRegex(ValidationError, "proposal identity omitted"):
            ExecutionProposal.model_validate(changed_basis)

    def test_the_reviewed_proposal_shows_each_fact_slot_status_and_source(self):
        proposal = compile_execution_proposal(ready_timeline())

        self.assertEqual(
            sorted(
                (item.kind, item.status, item.verification_source, item.summary)
                for item in proposal.fact_slots
            ),
            [
                (
                    "reference_assembly_identity",
                    "verified",
                    "NCBI Datasets",
                    "verified by NCBI Datasets",
                ),
                (
                    "sample_identity",
                    "asserted",
                    None,
                    "asserted by researcher, unverified",
                ),
                (
                    "taxon_identity",
                    "verified",
                    "NCBI Taxonomy",
                    "verified by NCBI Taxonomy",
                ),
            ],
        )
        self.assertIn("fact_slots", proposal.model_dump(mode="json"))

    def test_execution_hitl_request_view_binds_exact_legal_schemas(self):
        timeline = ready_timeline()
        proposal, request = submit(timeline)
        pending = project_execution_hitl(timeline)
        self.assertEqual(pending.status, "execution_hitl_pending")
        self.assertEqual(
            pending.pending_request.review_policy.model_dump(),
            {"stage": "execution", "approval": "mandatory"},
        )
        for value in (
            {},
            {"stage": "execution"},
            {"stage": "investigation", "approval": "mandatory"},
            {"stage": "execution", "approval": "optional"},
        ):
            hostile = copy.deepcopy(timeline)
            next(e for e in hostile if e["event_id"] == request["event_id"])["payload"][
                "review_policy"
            ] = value
            with self.subTest(review_policy=value), self.assertRaises(ValueError):
                project_execution_hitl(hostile)
        self.assertEqual(
            [item.action for item in pending.pending_request.legal_actions],
            ["approve", "revise", "reject"],
        )
        for legal in pending.pending_request.legal_actions:
            self.assertEqual(
                legal.response_schema["properties"]["proposal_id"]["const"],
                proposal.proposal_id,
            )
        reject_schema = next(
            item.response_schema
            for item in pending.pending_request.legal_actions
            if item.action == "reject"
        )
        self.assertNotIn("reason", reject_schema["required"])
        without_reason = RejectExecutionResponse(
            action="reject", proposal_id=proposal.proposal_id
        )
        self.assertEqual(
            evaluate_execution_hitl_response(
                pending, request["event_id"], without_reason
            ),
            "accept",
        )

    def test_execution_hitl_response_evaluation_preserves_illegal_and_stale_codes(
        self,
    ):
        timeline = ready_timeline()
        proposal, request = submit(timeline)
        pending = project_execution_hitl(timeline)
        with self.assertRaisesRegex(
            ExecutionHitlResponseRefused, "execution_hitl_response_illegal"
        ) as illegal:
            evaluate_execution_hitl_response(
                pending,
                request["event_id"],
                ApproveExecutionResponse(
                    action="approve",
                    proposal_id="execution-proposal:" + "0" * 64,
                ),
            )
        self.assertEqual(
            illegal.exception.disposition.kind,
            "caller_correction_and_reissue",
        )
        with self.assertRaisesRegex(
            ExecutionHitlResponseRefused, "execution_hitl_response_stale"
        ) as stale:
            evaluate_execution_hitl_response(
                pending,
                str(uuid.uuid4()),
                ApproveExecutionResponse(
                    action="approve", proposal_id=proposal.proposal_id
                ),
            )
        self.assertEqual(
            stale.exception.disposition.kind,
            "caller_correction_and_reissue",
        )

    def test_retained_rejection_is_identical_and_conflicting_revision_refuses(self):
        timeline = ready_timeline()
        proposal, request = submit(timeline)
        response = RejectExecutionResponse(
            action="reject", proposal_id=proposal.proposal_id, reason="Do not run."
        )
        rejection_event = append(
            timeline,
            "reject",
            "ExecutionHitlResponded",
            payload(
                ExecutionHitlResponded(
                    request_event_id=request["event_id"], response=response
                )
            ),
        )
        append(
            timeline,
            "reject-terminal",
            "RunFailed",
            payload(
                RunFailed(
                    run_id=JOURNEY_ID,
                    code="rejected_by_user",
                    source_event_id=rejection_event["event_id"],
                )
            ),
        )
        rejected = project_execution_hitl(timeline)
        self.assertEqual(rejected.status, "execution_rejected")
        self.assertEqual(
            evaluate_execution_hitl_response(rejected, request["event_id"], response),
            "identical",
        )
        with self.assertRaisesRegex(
            ExecutionHitlResponseRefused, "execution_hitl_response_conflict"
        ) as conflict:
            evaluate_execution_hitl_response(
                rejected,
                request["event_id"],
                ReviseExecutionResponse(
                    action="revise",
                    proposal_id=proposal.proposal_id,
                    instruction="Use another policy.",
                    revalidation_scope="execution_planning",
                ),
            )
        self.assertEqual(
            conflict.exception.disposition.kind,
            "caller_correction_and_reissue",
        )

    def test_revision_direction_reenters_new_grounded_planning(self):
        timeline = ready_timeline()
        original_work_id = project_execution_planning(
            timeline
        ).recipe_work_item.work_item_id
        proposal, request = submit(timeline)
        revised = append(
            timeline,
            "revise",
            "ExecutionHitlResponded",
            payload(
                ExecutionHitlResponded(
                    request_event_id=request["event_id"],
                    response=ReviseExecutionResponse(
                        action="revise",
                        proposal_id=proposal.proposal_id,
                        instruction="Revalidate against the retained sources.",
                        revalidation_scope="execution_planning",
                    ),
                )
            ),
        )
        append_index_rechoice(timeline, label="revise-index")

        planning = project_execution_planning(timeline)
        self.assertEqual(planning.status, "recipe_validation_pending")
        self.assertEqual(planning.revision_context_event_id, revised["event_id"])
        self.assertNotEqual(planning.recipe_work_item.work_item_id, original_work_id)
        self.assertIn(
            revised["event_id"],
            planning.recipe_work_item.grounding.source_event_ids,
        )
        with self.assertRaises(ProposalCompilationRefused) as refused:
            compile_execution_proposal(timeline)
        self.assertEqual(
            refused.exception.code, "proposal_compiled_before_planning_ready"
        )

    def test_fresh_tool_closure_compiles_superseding_proposal_and_new_review(self):
        timeline = ready_timeline()
        original, request = submit(timeline)
        revision = append(
            timeline,
            "superseding-revision",
            "ExecutionHitlResponded",
            payload(
                ExecutionHitlResponded(
                    request_event_id=request["event_id"],
                    response=ReviseExecutionResponse(
                        action="revise",
                        proposal_id=original.proposal_id,
                        instruction="Revalidate every Tool-owned consequence.",
                        revalidation_scope="execution_planning",
                    ),
                )
            ),
        )
        append_index_rechoice(timeline, label="superseding-revision-index")
        adapter = CapabilitySnapshotAdapter(policy())
        work, recipe_result, recipe_event = append_asserted_recipe_observation(
            timeline, adapter, "superseding-recipe"
        )
        recipe_attempt = attempt_start(timeline, "superseding-recipe-selection")
        recipe = recipe_result.candidates[0]
        validated = append(
            timeline,
            "superseding-recipe-selection-validated",
            "ToolCallValidated",
            payload(
                ToolCallValidated(
                    attempt_id=recipe_attempt,
                    tool_name="select_recipe_candidate",
                    tool_call_id="select-recipe",
                    arguments={
                        "work_item_id": work.work_item_id,
                        "candidate_id": recipe.candidate_id,
                        "basis_event_id": recipe_event["event_id"],
                    },
                    status="valid",
                    errors=(),
                )
            ),
        )
        append(
            timeline,
            "superseding-recipe-selected",
            "PlanningCandidateSelected",
            payload(
                PlanningCandidateSelected(
                    attempt_id=recipe_attempt,
                    kind="recipe",
                    work_item_id=work.work_item_id,
                    candidate_id=recipe.candidate_id,
                    basis_event_id=recipe_event["event_id"],
                    authority="agent_ranking",
                    authority_event_id=validated["event_id"],
                )
            ),
        )
        recipe_ready = project_execution_planning(timeline)
        resource_input = LookupResourcesInput(work_item=recipe_ready.resource_work_item)
        resource_result = LookupResourcesTool().lookup(
            resource_input,
            expected_work_item=recipe_ready.resource_work_item,
            adapter=adapter,
        )
        resource_attempt = attempt_start(timeline, "superseding-resource-observation")
        resource_event = append(
            timeline,
            "superseding-resource-observed",
            "ResourceObservationRecorded",
            payload(
                ResourceObservationRecorded(
                    attempt_id=resource_attempt,
                    tool_input=resource_input,
                    result=resource_result,
                )
            ),
        )
        resource_attempt = attempt_start(timeline, "superseding-resource-selection")
        resource = resource_result.candidates[0]
        validated = append(
            timeline,
            "superseding-resource-selection-validated",
            "ToolCallValidated",
            payload(
                ToolCallValidated(
                    attempt_id=resource_attempt,
                    tool_name="select_resource_candidate",
                    tool_call_id="select-resource",
                    arguments={
                        "work_item_id": resource_input.work_item.work_item_id,
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
            "superseding-resource-selected",
            "PlanningCandidateSelected",
            payload(
                PlanningCandidateSelected(
                    attempt_id=resource_attempt,
                    kind="resource",
                    work_item_id=resource_input.work_item.work_item_id,
                    candidate_id=resource.candidate_id,
                    basis_event_id=resource_event["event_id"],
                    authority="agent_ranking",
                    authority_event_id=validated["event_id"],
                )
            ),
        )

        superseding = compile_execution_proposal(timeline)
        self.assertEqual(superseding.supersedes_proposal_id, original.proposal_id)
        self.assertEqual(superseding.revision_event_id, revision["event_id"])
        self.assertEqual(
            superseding.revision_direction.instruction,
            "Revalidate every Tool-owned consequence.",
        )
        self.assertNotEqual(superseding.proposal_id, original.proposal_id)
        changed_direction = superseding.model_dump(mode="json")
        changed_direction["revision_direction"]["instruction"] = "Different direction."
        with self.assertRaisesRegex(ValidationError, "proposal identity omitted"):
            ExecutionProposal.model_validate(changed_direction)
        _, next_request = submit(timeline)
        projected = project_execution_hitl(timeline)
        self.assertEqual(projected.status, "execution_hitl_pending")
        self.assertEqual(
            projected.pending_request.request_event_id, next_request["event_id"]
        )
        self.assertEqual(
            projected.pending_request.proposal.proposal_id, superseding.proposal_id
        )


if __name__ == "__main__":
    unittest.main()


class ExecutionPloidyTests(unittest.TestCase):
    """The Execution Review states that calling is diploid and asks before a likely
    non-diploid Run; the model judges the organism, citing its record, and no code
    list of organisms decides it (#470; owner RC2 3 (c), Q2 a, R1 a, R2 a)."""

    def _pending(self, ploidy):
        timeline = ready_timeline(ploidy=ploidy)
        proposal, request = submit(timeline)
        return timeline, proposal, request, project_execution_hitl(timeline)

    def test_every_judged_review_states_diploid_calling_with_the_models_judgment(self):
        timeline = ready_timeline(ploidy="diploid")
        taxon = next(
            e for e in timeline if e["event_type"] == "TaxonResolutionRecorded"
        )
        selection = next(
            e
            for e in timeline
            if e["event_type"] == "PlanningCandidateSelected"
            and e["payload"]["kind"] == "recipe"
        )

        ploidy = compile_execution_proposal(timeline).ploidy

        self.assertEqual(ploidy.calling_ploidy, 2)
        self.assertEqual(
            ploidy.statement,
            "Variant calling assumes a diploid sample: `pbrun germline` runs "
            "HaplotypeCaller at ploidy 2.",
        )
        self.assertEqual(
            (
                ploidy.organism.judgment,
                ploidy.organism.record_event_id,
                ploidy.organism.quote,
                ploidy.selection_event_id,
            ),
            ("diploid", taxon["event_id"], "Homo sapiens", selection["event_id"]),
        )
        self.assertIsNone(ploidy.question)

    def test_a_likely_non_diploid_or_unknown_organism_gets_a_question(self):
        for judgment, words in (
            ("likely_non_diploid", "likely not diploid"),
            ("unknown", "of unknown ploidy"),
        ):
            with self.subTest(judgment=judgment):
                ploidy = compile_execution_proposal(
                    ready_timeline(ploidy=judgment)
                ).ploidy

                self.assertEqual(
                    ploidy.question,
                    f"The planning model judged this organism {words}, citing "
                    "its record's words “Homo sapiens”. Calling is diploid only: "
                    "choose calling_ploidy 2 to call it diploid, or revise or "
                    "reject the Run.",
                )

    def test_a_selection_recorded_before_the_judgment_compiles_as_before(self):
        """Owner Q3 b and R2 a: a pre-#470 Journey keeps its proposal identity."""
        proposal = compile_execution_proposal(ready_timeline())

        self.assertIsNone(proposal.ploidy)
        self.assertNotIn("ploidy", proposal.model_dump(mode="json"))

    def test_approval_names_the_offered_calling_ploidy(self):
        """Owner R2 addition: the researcher answers the calling ploidy, and only the
        current diploid 2 is offered until Parabricks ploidy support is verified."""
        _, proposal, request, pending = self._pending("likely_non_diploid")
        approve = next(
            item.response_schema
            for item in pending.pending_request.legal_actions
            if item.action == "approve"
        )

        self.assertEqual(approve["properties"]["calling_ploidy"], {"enum": [2]})
        self.assertIn("calling_ploidy", approve["required"])
        self.assertEqual(
            evaluate_execution_hitl_response(
                pending,
                request["event_id"],
                ApproveExecutionResponse(
                    action="approve", proposal_id=proposal.proposal_id, calling_ploidy=2
                ),
            ),
            "accept",
        )
        with self.assertRaisesRegex(
            ExecutionHitlResponseRefused, "execution_hitl_response_illegal"
        ):
            evaluate_execution_hitl_response(
                pending,
                request["event_id"],
                ApproveExecutionResponse(
                    action="approve", proposal_id=proposal.proposal_id
                ),
            )

    def test_an_unjudged_proposal_offers_no_calling_ploidy(self):
        _, proposal, request, pending = self._pending(None)
        approve = next(
            item.response_schema
            for item in pending.pending_request.legal_actions
            if item.action == "approve"
        )

        self.assertNotIn("calling_ploidy", approve["properties"])
        with self.assertRaisesRegex(
            ExecutionHitlResponseRefused, "execution_hitl_response_illegal"
        ):
            evaluate_execution_hitl_response(
                pending,
                request["event_id"],
                ApproveExecutionResponse(
                    action="approve", proposal_id=proposal.proposal_id, calling_ploidy=2
                ),
            )
        self.assertNotIn(
            "calling_ploidy",
            ApproveExecutionResponse(
                action="approve", proposal_id=proposal.proposal_id
            ).model_dump(mode="json"),
        )

    def test_a_retained_judgment_is_reread_against_the_record_it_cites(self):
        """ADR 0049 membership on reopen: the cited record is an organism record the
        Run kept before the selection, and the quote stands in it."""
        timeline = ready_timeline(ploidy="diploid")
        plan = next(
            e for e in timeline if e["event_type"] == "ProposalRequirementsIdentified"
        )
        for name, changes in (
            ("words the record does not hold", {"quote": "Mus musculus"}),
            (
                "a record that is no organism record",
                {"record_event_id": plan["event_id"]},
            ),
        ):
            hostile = copy.deepcopy(timeline)
            selected = next(
                e
                for e in hostile
                if e["event_type"] == "PlanningCandidateSelected"
                and e["payload"]["kind"] == "recipe"
            )
            selected["payload"]["organism_ploidy"].update(changes)
            validated = next(
                e
                for e in hostile
                if e["event_id"] == selected["payload"]["authority_event_id"]
            )
            validated["payload"]["arguments"] = (
                PlanningCandidateSelected.model_validate(
                    selected["payload"]
                ).admitted_arguments()
            )
            with (
                self.subTest(name),
                self.assertRaisesRegex(
                    ValueError, "ploidy judgment cites no organism record it read"
                ),
            ):
                project_execution_planning(hostile)
