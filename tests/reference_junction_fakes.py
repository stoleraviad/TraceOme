"""Runtime-profile variations of the end-of-dev-cleanup Run's approved input.

They grant no live or continuation authority.
"""

from tests.retained_execution_path import approved_input


def approved_with_runtimes(
    *,
    profile_ids=("alternate", "fallback"),
    cpu_disk="100Gi",
    gpu_disk="250Gi",
    fallback_cpu_disk=None,
    approved=None,
):
    from bio_neb.journey import canonical_json_digest
    from bio_neb.parabricks.capability_policy import (
        CapabilityRuntimeProfile,
        ExecutionRuntimeSelection,
    )
    from bio_neb.parabricks.execution_proposal import (
        ExecutionAuthority,
        ExecutionProposal,
        execution_authority_id,
    )
    from bio_neb.parabricks.execution_runtime import CpuReferenceRuntime, H100Runtime
    from tests.planning_fakes import policy

    approved = approved or approved_input()
    package = approved.proposal.execution_package
    from bio_neb.parabricks.capability_policy import (
        CapabilityPolicyTool,
        CapabilitySnapshot,
    )
    from bio_neb.parabricks.execution_runtime import ATTESTED_IMAGE_CHILD

    base = policy()
    profile = CapabilityRuntimeProfile(
        profile_id="alternate",
        cpu=CpuReferenceRuntime(disk_size=cpu_disk),
        h100=H100Runtime(
            recipe_id=package.workflow.workflow_id,
            resource_profile_id=package.resource.profile_id,
            platform="gpu-alternative",
            preset="alternative-preset",
            disk_size=gpu_disk,
        ),
    )
    fallback = profile.model_copy(
        update={
            "profile_id": "fallback",
            "cpu": CpuReferenceRuntime(disk_size=fallback_cpu_disk or cpu_disk),
            "h100": profile.h100.model_copy(update={"platform": "gpu-fallback"}),
        }
    )
    resource = base.resources[0].model_copy(
        update={
            "profile_id": package.resource.profile_id,
            "recipe_id": package.workflow.workflow_id,
            "accelerator_count": 1,
            "cpu_count": 16,
            "memory_gib": 200,
            "accelerator_memory_gib": 80,
            "tool_ids": ("bwa", "pbrun", "python", "samtools"),
        }
    )
    recipe = base.recipes[0].model_copy(
        update={"recipe_id": package.workflow.workflow_id}
    )
    image = base.images[0].model_copy(
        update={
            "provider_image": profile.h100.provider_image_reference,
            "image_index_digest": profile.h100.provider_image_index_digest,
            "platform_child_digest": ATTESTED_IMAGE_CHILD,
            "worker_sha256": profile.h100.worker_sha256,
        }
    )
    tools = tuple(
        CapabilityPolicyTool(
            tool_id=name,
            execution_scope="worker_image",
            absolute_path=path,
            version=version,
            image_id=image.image_id,
        )
        for name, path, version in (
            ("bwa", "/usr/bin/bwa", profile.cpu.bwa_version),
            ("pbrun", profile.h100.pbrun_path, profile.h100.pbrun_version),
            ("python", profile.h100.python_path, profile.h100.python_version),
            ("samtools", "/usr/bin/samtools", profile.cpu.samtools_version),
        )
    )
    claims = base.model_dump(mode="json", exclude={"snapshot_id"})
    claims.update(
        resources=[resource.model_dump(mode="json")],
        recipes=[recipe.model_dump(mode="json")],
        images=[image.model_dump(mode="json")],
        tools=[item.model_dump(mode="json") for item in tools],
        runtime_profiles=[
            profile.model_dump(mode="json"),
            fallback.model_dump(mode="json"),
        ],
    )
    snapshot = CapabilitySnapshot(snapshot_id=canonical_json_digest(claims), **claims)
    selection = ExecutionRuntimeSelection(capability=snapshot, profile_ids=profile_ids)
    material = approved.proposal.model_dump(mode="json", exclude={"proposal_id"})
    material["runtime_selection"] = selection.model_dump(mode="json")
    # The reviewed configuration carries the same selection.
    material["configuration"]["runtime_selection"] = material["runtime_selection"]
    proposal = ExecutionProposal(
        proposal_id="execution-proposal:" + canonical_json_digest(material), **material
    )
    authority_material = approved.authority.model_dump(exclude={"authority_id"})
    authority_material["proposal_id"] = proposal.proposal_id
    authority = ExecutionAuthority(
        authority_id=execution_authority_id(
            proposal.proposal_id,
            authority_material["request_event_id"],
            authority_material["approval_event_id"],
        ),
        **authority_material,
    )
    changed = approved.model_copy(update={"proposal": proposal, "authority": authority})
    return changed
