"""Scientific negative capability Adapter used across planning proofs."""

from __future__ import annotations

from bio_neb.parabricks.capability_policy import (
    CapabilitySnapshotAdapter,
)
from bio_neb.parabricks.recipe_resources import (
    RecipeAdapterObservation,
    ResourceAdapterObservation,
    make_recipe_candidate,
    make_resource_candidate,
)


class IncompatibleAdapter(CapabilitySnapshotAdapter):
    def list_recipes(self, tool_input):
        observed = super().list_recipes(tool_input)
        candidates = tuple(
            make_recipe_candidate(
                recipe_id=item.recipe_id,
                assay=item.assay,
                read_layout=item.read_layout,
                output_kind=item.output_kind,
                requires_prepared_reference=item.requires_prepared_reference,
                requires_companion_roles=item.requires_companion_roles,
                compatibility="incompatible",
                incompatibility_reasons=("synthetic_current_negative",),
                unresolved_constraints=item.unresolved_constraints,
                snapshot_id=item.snapshot_id,
            )
            for item in observed.candidates
        )
        response = [item.model_dump(mode="json") for item in candidates]
        return RecipeAdapterObservation(
            status="ok",
            candidates=candidates,
            source=self._source(tool_input.model_dump(mode="json"), response),
        )

    def lookup_resources(self, work_item):
        observed = super().lookup_resources(work_item)
        candidates = tuple(
            make_resource_candidate(
                profile_id=item.profile_id,
                recipe_candidate_id=item.recipe_candidate_id,
                accelerator_count=item.accelerator_count,
                cpu_count=item.cpu_count,
                memory_gib=item.memory_gib,
                accelerator_memory_gib=item.accelerator_memory_gib,
                availability=item.availability,
                compatibility="incompatible",
                incompatibility_reasons=("synthetic_current_negative",),
                unresolved_constraints=item.unresolved_constraints,
                snapshot_id=item.snapshot_id,
            )
            for item in observed.candidates
        )
        response = [item.model_dump(mode="json") for item in candidates]
        return ResourceAdapterObservation(
            status="ok",
            candidates=candidates,
            source=self._source(work_item.model_dump(mode="json"), response),
        )
