"""Reusable execution capability policy fixtures."""

from __future__ import annotations

import hashlib

from bio_neb.application.input_journey import InputMetadataInspected
from bio_neb.input_inspection.tool import (
    InputMetadataInspectionTool,
    InspectInputMetadataInput,
    RegisteredInputPath,
)
from bio_neb.journey import canonical_json
from bio_neb.parabricks.capability_policy import (
    CapabilityPolicyImage,
    CapabilityPolicyRecipe,
    CapabilityPolicyResource,
    CapabilityPolicyTool,
    CapabilitySnapshotAdapter,
    CapabilitySnapshotAuthoring,
    RetainedCapabilitySourceInput,
    compile_capability_snapshot_value,
)
from bio_neb.parabricks.execution_planning_journey import (
    RecipeObservationRecorded,
    project_execution_planning,
)
from bio_neb.parabricks.planning_hitl import (
    PlanningHitlCandidate,
    PlanningHitlRequested,
    SelectPlanningHitlResponse,
    make_planning_hitl_responded,
)
from bio_neb.parabricks.proposal_information import (
    ContextEvidence,
    NoRemainingValidation,
    ProposalInformationRequirement,
    RequestAssertionEvidence,
    ToolEvidenceValidation,
)
from bio_neb.parabricks.recipe_resources import (
    ListRecipesTool,
    RecipeAdapterObservation,
    ResourceAdapterObservation,
    recipe_eligibility_assertion_id,
)
from tests.input_metadata_fakes import ReturningAdapter, metadata, observation
from tests.reference_fakes import append, attempt_start, payload


def policy(*, recipes=1, resources=1, assays=None, resource_recipe_ids=None):
    if recipes == 0:
        resources = 0
    content_digest = hashlib.sha256(b"synthetic-policy-content").hexdigest()
    policy_recipes = tuple(
        CapabilityPolicyRecipe(
            recipe_id=f"germline-{index}",
            assay=(assays[index] if assays is not None else "gDNA"),
            read_layout="paired",
            output_kind="VCF",
            requires_prepared_reference=True,
            requires_companion_roles=("genomic_fasta",),
        )
        for index in range(recipes)
    )
    policy_resources = tuple(
        CapabilityPolicyResource(
            profile_id=f"profile-{index}",
            recipe_id=(
                resource_recipe_ids[index]
                if resource_recipe_ids is not None
                else "germline-0"
            ),
            image_id="parabricks-image",
            tool_ids=("pbrun",),
            accelerator_count=1,
            cpu_count=24,
            memory_gib=100,
            accelerator_memory_gib=18,
        )
        for index in range(resources)
    )
    return compile_capability_snapshot_value(
        CapabilitySnapshotAuthoring(
            schema_version="bio-neb.execution-capability-authoring/1",
            capability_version="synthetic-tests/1",
            sources=(
                RetainedCapabilitySourceInput(
                    kind="retained_nonlocal",
                    locator="https://tests.invalid/recipe-resource-policy",
                    content_digest=content_digest,
                    retrieved_at="2026-08-07T00:00:00Z",
                ),
            ),
            recipes=policy_recipes,
            resources=policy_resources,
            images=(
                CapabilityPolicyImage(
                    image_id="parabricks-image",
                    provider_image="registry.example.test/parabricks:t49",
                    image_index_digest="sha256:" + "1" * 64,
                    platform="linux/amd64",
                    platform_child_digest="sha256:" + "2" * 64,
                    worker_sha256="3" * 64,
                    worker_module=(
                        "bio_neb.parabricks.reusable_object_storage_parabricks"
                    ),
                    worker_request_schema="bio-neb.parabricks-worker-request/7",
                    worker_receipt_schema="bio-neb.parabricks-worker-receipt/3",
                    worker_entrypoint=(
                        "/opt/bio-neb/venv/bin/python",
                        "/opt/bio-neb/reusable_object_storage_parabricks.py",
                    ),
                ),
            ),
            tools=(
                CapabilityPolicyTool(
                    tool_id="pbrun",
                    execution_scope="worker_image",
                    absolute_path="/usr/local/parabricks/pbrun",
                    version="pbrun: 4.7.0-1",
                    image_id="parabricks-image",
                ),
            ),
        )
    )


def add_requirement(timeline, kind, subject, fill):
    plan = next(
        item
        for item in timeline
        if item["event_type"] == "ProposalRequirementsIdentified"
    )
    requirement = ProposalInformationRequirement(
        requirement_id="requirement_" + fill * 32,
        kind=kind,
        subject_key=subject,
        domain="operational" if kind == "resource_profile" else "scientific",
        meaning=f"Resolve {subject}.",
        knowledge_state="to_find",
        value=None,
        evidence=(ContextEvidence(source_event_id=timeline[0]["event_id"]),),
        acquisition_route="validate_with_tool",
        validation=ToolEvidenceValidation(requirement=f"Validate {subject}."),
    )
    plan["payload"]["plan"]["requirements"].append(requirement.model_dump(mode="json"))


def add_target_requirement(timeline, *, rsid="rs1234"):
    """Add one target the request states: an rsID, quoted as the researcher wrote it."""

    plan = next(
        item
        for item in timeline
        if item["event_type"] == "ProposalRequirementsIdentified"
    )
    requirement = ProposalInformationRequirement(
        requirement_id="requirement_" + "4" * 32,
        kind="scientific_investigation",
        subject_key="requested_variant",
        domain="scientific",
        meaning=f"Whether the sample carries {rsid}.",
        knowledge_state="asserted",
        value=rsid,
        evidence=(
            RequestAssertionEvidence(
                source_event_id=timeline[0]["event_id"],
                start_offset=0,
                end_offset=len(rsid),
                exact_text=rsid,
            ),
        ),
        acquisition_route="use_supplied_assertion",
        validation=ToolEvidenceValidation(requirement="Observe it after execution."),
        investigation_purpose="known_variant",
        identifier_rule="rsid",
    )
    plan["payload"]["plan"]["requirements"].append(requirement.model_dump(mode="json"))


def add_goal_requirement(timeline, *, validation=None, quote="Analyze sample-001"):
    """Add an asserted research goal: prose, so its quote is its value."""

    plan = next(
        item
        for item in timeline
        if item["event_type"] == "ProposalRequirementsIdentified"
    )
    requirement = ProposalInformationRequirement(
        requirement_id="requirement_" + "5" * 32,
        kind="research_goal",
        subject_key="research_goal",
        domain="scientific",
        meaning="The researcher's stated analysis goal.",
        knowledge_state="asserted",
        value=quote,
        evidence=(
            RequestAssertionEvidence(
                source_event_id=timeline[0]["event_id"],
                start_offset=0,
                end_offset=len(quote),
                exact_text=quote,
            ),
        ),
        acquisition_route="use_supplied_assertion",
        validation=validation or NoRemainingValidation(),
    )
    plan["payload"]["plan"]["requirements"].append(requirement.model_dump(mode="json"))
    return requirement.requirement_id


def add_inspection(timeline):
    plan = next(
        item
        for item in timeline
        if item["event_type"] == "ProposalRequirementsIdentified"
    )
    input_set_id = plan["payload"]["plan"]["input_set_id"]
    paths = [("READ-1", "/reads/r1.fastq"), ("READ-2", "/reads/r2.fastq")]

    # Synthetic current-contract registration; no retained fixture is rewritten.
    from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration
    from bio_neb.application.input_journey import (
        RegisteredFastqInput,
        RunInputsRegistered,
    )
    from tests.proposal_information_fakes import runtime_configuration

    registration = append(
        timeline,
        "planning-input-registration",
        "RunInputsRegistered",
        payload(
            RunInputsRegistered(
                input_set_id=input_set_id,
                run_root="/runs/planning",
                runtime_config_file="/runtime.json",
                runtime_selections=RuntimeConfiguration.model_validate(
                    runtime_configuration()
                ).runtime_selections(),
                inputs=tuple(
                    RegisteredFastqInput(role=role, registered_path=path)
                    for role, path in paths
                ),
            )
        ),
    )
    tool_input = InspectInputMetadataInput(
        input_set_id=input_set_id,
        inputs=tuple(
            RegisteredInputPath(role=role, registered_path=path) for role, path in paths
        ),
    )
    items = [
        metadata(role, path, inode=str(index + 1))
        for index, (role, path) in enumerate(paths)
    ]
    result = InputMetadataInspectionTool().inspect(
        tool_input, ReturningAdapter(observation(items))
    )
    append(
        timeline,
        "input-inspection",
        "InputMetadataInspected",
        payload(
            InputMetadataInspected(
                input_set_id=input_set_id,
                selection_event_id=registration["event_id"],
                result=result,
            )
        ),
    )


def append_asserted_recipe_observation(timeline, adapter, label):
    projected = project_execution_planning(timeline)
    work = projected.recipe_work_item
    initial_input = projected.recipe_tool_input
    initial_result = ListRecipesTool().list(
        initial_input, expected_input=initial_input, adapter=adapter
    )
    initial_attempt = attempt_start(
        timeline, f"{label}-eligibility-observation-attempt"
    )
    basis = append(
        timeline,
        f"{label}-eligibility-observation",
        "RecipeObservationRecorded",
        payload(
            RecipeObservationRecorded(
                attempt_id=initial_attempt,
                tool_input=initial_input,
                result=initial_result,
            )
        ),
    )
    requirement = next(
        item
        for item in projected.requirements
        if item.requirement_id == work.grounding.requirement_id
    )
    candidates = tuple(
        {
            recipe_eligibility_assertion_id(
                work.work_item_id, item
            ): PlanningHitlCandidate(
                candidate_id=recipe_eligibility_assertion_id(work.work_item_id, item),
                asserted_value=canonical_json(
                    {"assay": item.assay, "read_layout": item.read_layout}
                ),
                display_label=f"{item.assay} / {item.read_layout}",
                evidence_event_ids=(basis["event_id"],),
            )
            for item in initial_result.candidates
            if item.compatibility == "ambiguous"
        }.values()
    )
    request_attempt = attempt_start(timeline, f"{label}-eligibility-request-attempt")
    request = PlanningHitlRequested(
        attempt_id=request_attempt,
        plan_event_id=projected.plan_event_id,
        plan_id=projected.plan_id,
        requirement_id=requirement.requirement_id,
        subject_key=requirement.subject_key,
        issue="Input eligibility is unverified.",
        question="Select exact assay and layout.",
        evidence_event_ids=(basis["event_id"],),
        candidates=candidates,
    )
    requested = append(
        timeline,
        f"{label}-eligibility-requested",
        request.event_type,
        payload(request),
    )
    responded = make_planning_hitl_responded(
        requested["event_id"],
        SelectPlanningHitlResponse(
            action="select", candidate_id=candidates[0].candidate_id
        ),
    )
    append(
        timeline,
        f"{label}-eligibility-responded",
        responded.event_type,
        payload(responded),
    )
    asserted = project_execution_planning(timeline)
    asserted_input = asserted.recipe_tool_input
    result = ListRecipesTool().list(
        asserted_input, expected_input=asserted_input, adapter=adapter
    )
    attempt = attempt_start(timeline, f"{label}-asserted-observation-attempt")
    event = append(
        timeline,
        f"{label}-asserted-observation",
        "RecipeObservationRecorded",
        payload(
            RecipeObservationRecorded(
                attempt_id=attempt,
                tool_input=asserted_input,
                result=result,
            )
        ),
    )
    return work, result, event


class ExceptionalAdapter(CapabilitySnapshotAdapter):
    def list_recipes(self, tool_input):
        return RecipeAdapterObservation(
            status="operational_failure",
            candidates=(),
            source=self._source(tool_input.model_dump(mode="json"), []),
            failure_code="synthetic_transport_failure",
            failure_class="rejected",
        )

    def lookup_resources(self, work_item):
        return ResourceAdapterObservation(
            status="operational_failure",
            candidates=(),
            source=self._source(work_item.model_dump(mode="json"), []),
            failure_code="synthetic_transport_failure",
            failure_class="rejected",
        )
