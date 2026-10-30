"""Exact recipe and resource capability Tool contracts."""

from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import Literal, Protocol, Tuple
from urllib.parse import parse_qsl, urlsplit

from pydantic import (
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from bio_neb.contracts import FrozenContract
from bio_neb.failure import (
    FailureSurface,
    ProviderError,
    call_with_failure_policy,
    policy_for,
)
from bio_neb.journey import canonical_json_digest
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_PATTERN,
)


def _credential_free_locator(value: str) -> str:
    parsed = urlsplit(value)
    sensitive = {"token", "key", "secret", "password", "credential", "signature"}
    if (
        parsed.username is not None
        or parsed.password is not None
        or any(
            any(marker in name.casefold() for marker in sensitive)
            for name, _ in parse_qsl(parsed.query, keep_blank_values=True)
        )
    ):
        raise ValueError("source locator must not retain credentials")
    return value


class CapabilitySourceManifestEntry(FrozenContract):
    locator: StrictStr
    content_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_at: StrictStr
    content_assurance: Literal[
        "application_hashed_local_bytes", "retained_exact_digest"
    ] = "retained_exact_digest"

    _safe_locator = field_validator("locator")(_credential_free_locator)


class CapabilityImageClaim(FrozenContract):
    image_id: StrictStr
    provider_image: StrictStr
    image_index_digest: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    platform: Literal["linux/amd64"]
    platform_child_digest: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    worker_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    worker_module: StrictStr
    worker_request_schema: StrictStr
    worker_receipt_schema: StrictStr
    worker_entrypoint: Tuple[StrictStr, StrictStr]

    @field_validator(
        "provider_image",
        "worker_module",
        "worker_request_schema",
        "worker_receipt_schema",
    )
    @classmethod
    def pinned_text(cls, value: str) -> str:
        if (
            not value
            or value != value.strip()
            or any(character.isspace() for character in value)
        ):
            raise ValueError("image identity fields must be exact non-whitespace pins")
        return value

    @field_validator("worker_entrypoint")
    @classmethod
    def canonical_entrypoint(cls, value: Tuple[str, str]) -> Tuple[str, str]:
        if any(
            not item
            or not item.startswith("/")
            or item != str(PurePosixPath(item))
            or any(character.isspace() for character in item)
            for item in value
        ):
            raise ValueError(
                "worker entrypoint must contain two canonical absolute paths"
            )
        return value


class CapabilityToolClaim(FrozenContract):
    tool_id: StrictStr
    execution_scope: Literal["controller", "worker_image"]
    absolute_path: StrictStr
    version: StrictStr
    image_id: StrictStr | None = None
    executable_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    executable_byte_count: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def exact_scope(self) -> "CapabilityToolClaim":
        if not self.absolute_path.startswith(
            "/"
        ) or self.absolute_path != os.path.normpath(self.absolute_path):
            raise ValueError("tool path must be one canonical absolute Linux path")
        if not self.version or self.version != self.version.strip():
            raise ValueError("tool version must be an exact non-empty pin")
        if self.execution_scope == "controller":
            if (
                self.image_id is not None
                or self.executable_sha256 is None
                or self.executable_byte_count is None
            ):
                raise ValueError(
                    "controller tool requires exact bytes/hash and no image"
                )
        elif (
            self.image_id is None
            or self.executable_sha256 is not None
            or self.executable_byte_count is not None
        ):
            raise ValueError(
                "worker tool identity must be bound only to its exact image"
            )
        return self


class CapabilityGrounding(FrozenContract):
    journey_id: StrictStr
    plan_id: StrictStr = Field(pattern=r"^plan_[0-9a-f]{64}$")
    requirement_id: StrictStr = Field(pattern=r"^requirement_[0-9a-f]{32}$")
    subject_key: StrictStr = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    source_event_ids: Tuple[StrictStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonical(self) -> "CapabilityGrounding":
        import uuid

        if str(uuid.UUID(self.journey_id)) != self.journey_id:
            raise ValueError("journey_id must be canonical")
        if len(self.source_event_ids) != len(set(self.source_event_ids)):
            raise ValueError("grounding evidence identities must be unique")
        return self


class RecipeWorkItem(FrozenContract):
    work_item_id: StrictStr = Field(pattern=r"^recipe-work:[0-9a-f]{64}$")
    grounding: CapabilityGrounding
    input_set_id: StrictStr
    input_roles: Tuple[StrictStr, ...] = Field(min_length=1)
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    companion_roles: Tuple[StrictStr, ...] = Field(min_length=1)
    proposal_output_values: Tuple[StrictStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_identity(self) -> "RecipeWorkItem":
        if tuple(sorted(set(self.input_roles))) != self.input_roles:
            raise ValueError("input roles must be canonical and unique")
        if tuple(sorted(set(self.companion_roles))) != self.companion_roles:
            raise ValueError("companion roles must be canonical and unique")
        expected = recipe_work_item_id(
            self.grounding,
            self.input_set_id,
            self.input_roles,
            self.assembly_accession,
            self.companion_roles,
            self.proposal_output_values,
        )
        if self.work_item_id != expected:
            raise ValueError("recipe work identity changed its exact subject")
        return self


def recipe_work_item_id(
    grounding: CapabilityGrounding,
    input_set_id: str,
    input_roles: Tuple[str, ...],
    assembly_accession: str,
    companion_roles: Tuple[str, ...],
    proposal_output_values: Tuple[str, ...],
) -> str:
    return "recipe-work:" + canonical_json_digest(
        [
            grounding.model_dump(mode="json"),
            input_set_id,
            input_roles,
            assembly_accession,
            companion_roles,
            proposal_output_values,
        ]
    )


def make_recipe_work_item(
    *,
    grounding: CapabilityGrounding,
    input_set_id: str,
    input_roles: Tuple[str, ...],
    assembly_accession: str,
    companion_roles: Tuple[str, ...],
    proposal_output_values: Tuple[str, ...],
) -> RecipeWorkItem:
    roles = tuple(sorted(set(input_roles)))
    companions = tuple(sorted(set(companion_roles)))
    outputs = tuple(sorted(set(proposal_output_values)))
    return RecipeWorkItem(
        work_item_id=recipe_work_item_id(
            grounding, input_set_id, roles, assembly_accession, companions, outputs
        ),
        grounding=grounding,
        input_set_id=input_set_id,
        input_roles=roles,
        assembly_accession=assembly_accession,
        companion_roles=companions,
        proposal_output_values=outputs,
    )


class RecipeCandidate(FrozenContract):
    candidate_id: StrictStr = Field(pattern=r"^recipe-candidate:[0-9a-f]{64}$")
    recipe_id: StrictStr
    assay: StrictStr
    read_layout: Literal["paired", "single"]
    output_kind: StrictStr
    requires_prepared_reference: bool
    requires_companion_roles: Tuple[StrictStr, ...]
    compatibility: Literal["compatible", "incompatible", "ambiguous"]
    compatibility_scope: Literal["documented_planning_policy"] = (
        "documented_planning_policy"
    )
    incompatibility_reasons: Tuple[StrictStr, ...] = ()
    unresolved_constraints: Tuple[StrictStr, ...] = ()
    snapshot_id: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    source_evidence_id: StrictStr = Field(pattern=r"^recipe-evidence:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def coherent(self) -> "RecipeCandidate":
        if self.compatibility == "compatible" and self.incompatibility_reasons:
            raise ValueError("recipe compatibility and reasons disagree")
        if self.compatibility != "compatible" and not self.incompatibility_reasons:
            raise ValueError("recipe compatibility and reasons disagree")
        material = [
            self.recipe_id,
            self.assay,
            self.read_layout,
            self.output_kind,
            self.requires_prepared_reference,
            self.requires_companion_roles,
            self.compatibility,
            self.compatibility_scope,
            self.incompatibility_reasons,
            self.unresolved_constraints,
            self.snapshot_id,
        ]
        if self.candidate_id != ("recipe-candidate:" + canonical_json_digest(material)):
            raise ValueError("recipe candidate identity omitted material fields")
        if self.source_evidence_id != (
            "recipe-evidence:" + canonical_json_digest([*material, self.candidate_id])
        ):
            raise ValueError("recipe evidence identity omitted material fields")
        return self


def make_recipe_candidate(**values) -> RecipeCandidate:
    material = [
        values["recipe_id"],
        values["assay"],
        values["read_layout"],
        values["output_kind"],
        values["requires_prepared_reference"],
        values["requires_companion_roles"],
        values["compatibility"],
        values.get("compatibility_scope", "documented_planning_policy"),
        values.get("incompatibility_reasons", ()),
        values.get("unresolved_constraints", ()),
        values["snapshot_id"],
    ]
    candidate_id = "recipe-candidate:" + canonical_json_digest(material)
    # Candidate identity includes both material claims and their exact source snapshot.
    return RecipeCandidate(
        **values,
        candidate_id=candidate_id,
        source_evidence_id=(
            "recipe-evidence:" + canonical_json_digest([*material, candidate_id])
        ),
    )


class ResourceCandidate(FrozenContract):
    candidate_id: StrictStr = Field(pattern=r"^resource-candidate:[0-9a-f]{64}$")
    profile_id: StrictStr
    recipe_candidate_id: StrictStr
    accelerator_count: StrictInt = Field(ge=1)
    cpu_count: StrictInt = Field(ge=1)
    memory_gib: StrictInt = Field(ge=1)
    accelerator_memory_gib: StrictInt = Field(ge=1)
    availability: Literal["not_assessed"] = "not_assessed"
    compatibility: Literal["compatible", "incompatible"]
    compatibility_scope: Literal["documented_planning_policy"] = (
        "documented_planning_policy"
    )
    incompatibility_reasons: Tuple[StrictStr, ...] = ()
    unresolved_constraints: Tuple[StrictStr, ...] = ()
    snapshot_id: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    source_evidence_id: StrictStr = Field(pattern=r"^resource-evidence:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def coherent(self) -> "ResourceCandidate":
        if (self.compatibility == "compatible") == bool(self.incompatibility_reasons):
            raise ValueError("resource compatibility and reasons disagree")
        material = [
            self.profile_id,
            self.recipe_candidate_id,
            self.accelerator_count,
            self.cpu_count,
            self.memory_gib,
            self.accelerator_memory_gib,
            self.availability,
            self.compatibility,
            self.compatibility_scope,
            self.incompatibility_reasons,
            self.unresolved_constraints,
            self.snapshot_id,
        ]
        if self.candidate_id != (
            "resource-candidate:" + canonical_json_digest(material)
        ):
            raise ValueError("resource candidate identity omitted material fields")
        if self.source_evidence_id != (
            "resource-evidence:" + canonical_json_digest([*material, self.candidate_id])
        ):
            raise ValueError("resource evidence identity omitted material fields")
        return self


def make_resource_candidate(**values) -> ResourceCandidate:
    material = [
        values["profile_id"],
        values["recipe_candidate_id"],
        values["accelerator_count"],
        values["cpu_count"],
        values["memory_gib"],
        values["accelerator_memory_gib"],
        values.get("availability", "not_assessed"),
        values["compatibility"],
        values.get("compatibility_scope", "documented_planning_policy"),
        values.get("incompatibility_reasons", ()),
        values.get("unresolved_constraints", ()),
        values["snapshot_id"],
    ]
    candidate_id = "resource-candidate:" + canonical_json_digest(material)
    return ResourceCandidate(
        **values,
        candidate_id=candidate_id,
        source_evidence_id=(
            "resource-evidence:" + canonical_json_digest([*material, candidate_id])
        ),
    )


class CapabilitySource(FrozenContract):
    authority: Literal["compiled_capability_snapshot"]
    capability_version: StrictStr
    snapshot_id: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: Tuple[CapabilitySourceManifestEntry, ...] = Field(min_length=1)
    images: Tuple[CapabilityImageClaim, ...] = Field(min_length=1)
    tools: Tuple[CapabilityToolClaim, ...] = Field(min_length=1)
    request_identity: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    response_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    claim_scope: Literal["planning_policy_compatibility"] = (
        "planning_policy_compatibility"
    )


class CapabilityOperationalResult(FrozenContract):
    """Operational classification is explicit; negative capability results have none."""

    failure_class: (
        Literal["transient", "exhausted", "rejected", "worker_failed", "semantic"]
        | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    original_failure_class: (
        Literal["transient", "exhausted", "rejected", "worker_failed", "semantic"]
        | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    contact_attempt: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def classified_operational_result(self):
        operational = (
            getattr(self, "status", getattr(self, "outcome", None))
            == "operational_failure"
        )
        if operational != (self.failure_class is not None):
            raise ValueError(
                "only operational capability results require a failure class"
            )
        if not operational and (
            self.original_failure_class is not None or self.contact_attempt is not None
        ):
            raise ValueError("capability success cannot retain operational attempts")
        return self


def _observe_capability(tool_name, operation):
    """Apply the Tool's policy before recording a capability Tool's one result."""
    from bio_neb.agentic_journey.tool_execution import record_tool_operation_failure

    observed = None
    policy = policy_for(FailureSurface(scope="tool", name=tool_name))

    def invoke():
        nonlocal observed
        observed = None
        observed = operation()
        if observed.status == "operational_failure":
            raise ProviderError(observed.failure_class, observed.failure_code)
        return observed

    try:
        return call_with_failure_policy(
            invoke,
            policy,
            lambda error, ordinal, delay: record_tool_operation_failure(
                error, ordinal, delay, policy
            ),
        )
    except ProviderError as error:
        if observed is None:
            raise
        return observed.model_copy(
            update={
                "failure_class": error.failure_class,
                "original_failure_class": error.evidence["original_failure_class"],
                "contact_attempt": error.evidence["contact_attempt"],
            }
        )


def _operational_fields(observation):
    return {
        name: getattr(observation, name)
        for name in CapabilityOperationalResult.model_fields
        if getattr(observation, name) is not None
    }


class RecipeAdapterObservation(CapabilityOperationalResult):
    status: Literal["ok", "operational_failure"]
    candidates: Tuple[RecipeCandidate, ...]
    source: CapabilitySource
    failure_code: StrictStr | None = None

    @model_validator(mode="after")
    def coherent(self) -> "RecipeAdapterObservation":
        if self.status == "operational_failure" and (
            self.candidates or not self.failure_code
        ):
            raise ValueError("operational recipe failure cannot contain candidates")
        if self.status == "ok" and self.failure_code is not None:
            raise ValueError("successful recipe observation cannot contain failure")
        return self


class ResourceAdapterObservation(CapabilityOperationalResult):
    status: Literal["ok", "operational_failure"]
    candidates: Tuple[ResourceCandidate, ...]
    source: CapabilitySource
    failure_code: StrictStr | None = None

    @model_validator(mode="after")
    def coherent(self) -> "ResourceAdapterObservation":
        if self.status == "operational_failure" and (
            self.candidates or not self.failure_code
        ):
            raise ValueError("operational resource failure cannot contain candidates")
        if self.status == "ok" and self.failure_code is not None:
            raise ValueError("successful resource observation cannot contain failure")
        return self


class RecipeResourceAdapter(Protocol):
    def list_recipes(
        self, tool_input: "ListRecipesInput"
    ) -> RecipeAdapterObservation: ...

    def lookup_resources(
        self, work_item: "ResourceWorkItem"
    ) -> ResourceAdapterObservation: ...


class RecipeEligibilityAssertion(FrozenContract):
    assay: StrictStr
    read_layout: Literal["paired", "single"]
    assertion_event_id: StrictStr


class ListRecipesInput(FrozenContract):
    work_item: RecipeWorkItem
    eligibility_assertion: RecipeEligibilityAssertion | None = None


class RecipeObservation(CapabilityOperationalResult):
    kind: Literal["recipe"] = "recipe"
    work_item: RecipeWorkItem
    outcome: Literal["observed", "missing", "operational_failure"]
    candidates: Tuple[RecipeCandidate, ...]
    source: CapabilitySource
    evidence_id: StrictStr
    failure_code: StrictStr | None = None

    @model_validator(mode="after")
    def outcome_matches(self) -> "RecipeObservation":
        if self.outcome == "missing" and self.candidates:
            raise ValueError("missing recipe cannot contain candidates")
        if self.outcome == "observed" and not self.candidates:
            raise ValueError("observed recipe result requires candidates")
        if self.outcome == "operational_failure" and (
            self.candidates or not self.failure_code
        ):
            raise ValueError("operational recipe failure is separate from missing")
        payload = self.model_dump(mode="json", exclude={"kind", "evidence_id"})
        if self.evidence_id != ("recipe-result:" + canonical_json_digest(payload)):
            raise ValueError("recipe evidence identity changed its exact result")
        return self


class RecipeToolCallRejected(ValueError):
    pass


class ListRecipesTool:
    name = "list_compatible_analysis_recipes"

    def list(
        self,
        tool_input: ListRecipesInput,
        *,
        expected_input: ListRecipesInput,
        adapter: RecipeResourceAdapter,
    ) -> RecipeObservation:
        if tool_input != expected_input:
            raise RecipeToolCallRejected("recipe_input_ungrounded")
        expected_work_item = expected_input.work_item
        observed = _observe_capability(
            self.name, lambda: adapter.list_recipes(tool_input)
        )
        if len({item.candidate_id for item in observed.candidates}) != len(
            observed.candidates
        ):
            raise ValueError("recipe candidate identity collision")
        candidates = tuple(
            sorted(observed.candidates, key=lambda item: item.candidate_id)
        )
        if any(item.snapshot_id != observed.source.snapshot_id for item in candidates):
            raise ValueError("recipe candidate changed its source snapshot")
        if observed.source.request_identity != canonical_json_digest(
            tool_input.model_dump(mode="json")
        ):
            raise ValueError("recipe source changed its exact request identity")
        if observed.source.response_digest != canonical_json_digest(
            [item.model_dump(mode="json") for item in observed.candidates]
        ):
            raise ValueError("recipe source digest does not cover exact candidates")
        outcome = (
            "operational_failure"
            if observed.status == "operational_failure"
            else "observed"
            if candidates
            else "missing"
        )
        payload = {
            "work_item": expected_work_item.model_dump(mode="json"),
            "outcome": outcome,
            "candidates": [item.model_dump(mode="json") for item in candidates],
            "source": observed.source.model_dump(mode="json"),
            "failure_code": observed.failure_code,
            **_operational_fields(observed),
        }
        return RecipeObservation(
            **payload, evidence_id=("recipe-result:" + canonical_json_digest(payload))
        )


class ResourceWorkItem(FrozenContract):
    work_item_id: StrictStr = Field(pattern=r"^resource-work:[0-9a-f]{64}$")
    grounding: CapabilityGrounding
    recipe_candidate: RecipeCandidate
    recipe_event_id: StrictStr

    @model_validator(mode="after")
    def exact_identity(self) -> "ResourceWorkItem":
        expected = resource_work_item_id(
            self.grounding, self.recipe_candidate, self.recipe_event_id
        )
        if self.work_item_id != expected:
            raise ValueError("resource work identity changed its exact recipe")
        if self.recipe_candidate.compatibility != "compatible":
            raise ValueError("resource work requires compatible recipe evidence")
        return self


def resource_work_item_id(
    grounding: CapabilityGrounding,
    recipe_candidate: RecipeCandidate,
    recipe_event_id: str,
) -> str:
    return "resource-work:" + canonical_json_digest(
        [
            grounding.model_dump(mode="json"),
            recipe_candidate.model_dump(mode="json"),
            recipe_event_id,
        ]
    )


def make_resource_work_item(
    *,
    grounding: CapabilityGrounding,
    recipe_candidate: RecipeCandidate,
    recipe_event_id: str,
) -> ResourceWorkItem:
    return ResourceWorkItem(
        work_item_id=resource_work_item_id(
            grounding, recipe_candidate, recipe_event_id
        ),
        grounding=grounding,
        recipe_candidate=recipe_candidate,
        recipe_event_id=recipe_event_id,
    )


class LookupResourcesInput(FrozenContract):
    work_item: ResourceWorkItem


class ResourceObservation(CapabilityOperationalResult):
    kind: Literal["resource"] = "resource"
    work_item: ResourceWorkItem
    outcome: Literal["observed", "missing", "operational_failure", "stale_source"]
    candidates: Tuple[ResourceCandidate, ...]
    source: CapabilitySource
    evidence_id: StrictStr
    failure_code: StrictStr | None = None

    @model_validator(mode="after")
    def outcome_matches(self) -> "ResourceObservation":
        if self.outcome == "missing" and self.candidates:
            raise ValueError("missing resource cannot contain candidates")
        if self.outcome == "observed" and not self.candidates:
            raise ValueError("observed resource result requires candidates")
        if self.outcome == "operational_failure" and (
            self.candidates or not self.failure_code
        ):
            raise ValueError("operational resource failure is separate from missing")
        if self.outcome == "stale_source" and (
            self.candidates or self.failure_code != "capability_snapshot_changed"
        ):
            raise ValueError("stale resource source requires exact typed invalidation")
        payload = self.model_dump(mode="json", exclude={"kind", "evidence_id"})
        if self.evidence_id != ("resource-result:" + canonical_json_digest(payload)):
            raise ValueError("resource evidence identity changed its exact result")
        return self


class LookupResourcesTool:
    name = "lookup_compatible_resource_profiles"

    def lookup(
        self,
        tool_input: LookupResourcesInput,
        *,
        expected_work_item: ResourceWorkItem,
        adapter: RecipeResourceAdapter,
    ) -> ResourceObservation:
        if tool_input.work_item != expected_work_item:
            raise RecipeToolCallRejected("resource_work_item_ungrounded")
        observed = _observe_capability(
            self.name, lambda: adapter.lookup_resources(expected_work_item)
        )
        if len({item.candidate_id for item in observed.candidates}) != len(
            observed.candidates
        ):
            raise ValueError("resource candidate identity collision")
        candidates = tuple(
            sorted(observed.candidates, key=lambda item: item.candidate_id)
        )
        if any(item.snapshot_id != observed.source.snapshot_id for item in candidates):
            raise ValueError("resource candidate changed its source snapshot")
        if observed.source.request_identity != canonical_json_digest(
            expected_work_item.model_dump(mode="json")
        ):
            raise ValueError("resource source changed its exact request identity")
        if observed.source.response_digest != canonical_json_digest(
            [item.model_dump(mode="json") for item in observed.candidates]
        ):
            raise ValueError("resource source digest does not cover exact candidates")
        if any(
            item.recipe_candidate_id != expected_work_item.recipe_candidate.candidate_id
            for item in candidates
        ):
            raise ValueError("resource candidate crossed its exact recipe subject")
        if (
            observed.source.snapshot_id
            != expected_work_item.recipe_candidate.snapshot_id
        ):
            payload = {
                "work_item": expected_work_item.model_dump(mode="json"),
                "outcome": "stale_source",
                "candidates": [],
                "source": observed.source.model_dump(mode="json"),
                "failure_code": "capability_snapshot_changed",
            }
            return ResourceObservation(
                **payload,
                evidence_id=("resource-result:" + canonical_json_digest(payload)),
            )
        outcome = (
            "operational_failure"
            if observed.status == "operational_failure"
            else "observed"
            if candidates
            else "missing"
        )
        payload = {
            "work_item": expected_work_item.model_dump(mode="json"),
            "outcome": outcome,
            "candidates": [item.model_dump(mode="json") for item in candidates],
            "source": observed.source.model_dump(mode="json"),
            "failure_code": observed.failure_code,
            **_operational_fields(observed),
        }
        return ResourceObservation(
            **payload, evidence_id=("resource-result:" + canonical_json_digest(payload))
        )


def candidate_disposition_id(
    kind: Literal["recipe", "resource"], work_item_id: str, candidate_id: str
) -> str:
    return (f"{kind}-candidate-disposition" + ":") + canonical_json_digest(
        [work_item_id, candidate_id]
    )


def recipe_eligibility_assertion_id(
    work_item_id: str, candidate: RecipeCandidate
) -> str:
    return "recipe-eligibility-assertion:" + canonical_json_digest(
        [
            work_item_id,
            candidate.assay,
            candidate.read_layout,
            candidate.snapshot_id,
        ]
    )


def resource_recipe_dependency_id(
    resource_requirement_id: str,
    recipe_requirement_id: str,
    recipe_selection_event_id: str,
) -> str:
    return "resource-recipe-dependency:" + canonical_json_digest(
        [resource_requirement_id, recipe_requirement_id, recipe_selection_event_id]
    )


# The capability Tools whose operations make a classified provider contact.
CAPABILITY_TOOLS = frozenset({ListRecipesTool.name, LookupResourcesTool.name})


__all__ = [name for name in globals() if not name.startswith("_")]
