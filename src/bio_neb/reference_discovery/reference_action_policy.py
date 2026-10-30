"""Contact-free reference projection, grounding and choice readers."""

from __future__ import annotations

from typing import Literal, Protocol

from bio_neb.journey import Journey
from bio_neb.parabricks.planning_hitl import PlanningHitlResponseReceipt
from bio_neb.parabricks.proposal_information import ProposalInformationRequirement
from bio_neb.parabricks.proposal_journey import (
    ProposalRequirementsIdentified,
    validate_proposal_fact,
)
from bio_neb.parabricks.requirement_resolution import (
    RequirementUserAssertion,
    project_requirement_resolution,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    DiscoverAssembliesObserved,
    RequirementGrounding,
    SelectedAssemblyEvidence,
    SuppliedAssemblyEvidence,
)
from bio_neb.reference_discovery.reference_journey import (
    ReferenceIdentityProjection,
    project_reference_identity,
)

RequirementKind = Literal[
    "sample_identity", "taxon_identity", "reference_assembly_identity"
]


class ReferenceActionPolicyDependencies(Protocol):
    journey: Journey


def reference_projection(
    dependencies: ReferenceActionPolicyDependencies,
) -> ReferenceIdentityProjection:
    if hasattr(dependencies.journey, "read_view"):
        from bio_neb.application.planning_read import read_planning

        projected = read_planning(dependencies.journey).reference
        if projected is None:
            raise ValueError("reference capability lacks its checked prerequisites")
        return projected
    return project_reference_identity(dependencies.journey.read())


def reference_plan(
    dependencies: ReferenceActionPolicyDependencies,
    projection: ReferenceIdentityProjection,
) -> ProposalRequirementsIdentified:
    from bio_neb.application.planning_read import planning_evidence

    event = next(
        (
            item
            for item in planning_evidence(dependencies.journey)
            if item["event_id"] == projection.plan_event_id
        ),
        None,
    )
    fact = validate_proposal_fact(event or {})
    if not isinstance(fact, ProposalRequirementsIdentified):
        raise ValueError("reference Agent lacks its immutable proposal plan")
    return fact


def reference_requirement(
    dependencies: ReferenceActionPolicyDependencies,
    projection: ReferenceIdentityProjection,
    kind: RequirementKind,
) -> ProposalInformationRequirement:
    matches = tuple(
        item
        for item in reference_plan(dependencies, projection).plan.requirements
        if item.kind == kind
    )
    if len(matches) != 1:
        raise ValueError(f"reference Agent requires exactly one {kind} requirement")
    return matches[0]


def requirement_evidence_event_ids(
    requirement: ProposalInformationRequirement,
) -> tuple[str, ...]:
    event_ids: list[str] = []
    for evidence in requirement.evidence:
        for name, value in evidence.model_dump(mode="json").items():
            if (
                name.endswith("event_id")
                and isinstance(value, str)
                and value not in event_ids
            ):
                event_ids.append(value)
    return tuple(event_ids)


def reference_grounding(
    dependencies: ReferenceActionPolicyDependencies,
    projection: ReferenceIdentityProjection,
    kind: RequirementKind,
    *extra_sources: str | None,
) -> RequirementGrounding:
    requirement = reference_requirement(dependencies, projection, kind)
    sources = [projection.plan_event_id, *requirement_evidence_event_ids(requirement)]
    if projection.revision_event_id is not None:
        sources.append(projection.revision_event_id)
    sources.extend(item for item in extra_sources if item is not None)
    return RequirementGrounding(
        journey_id=projection.journey_id,
        plan_id=projection.plan_id,
        requirement_id=requirement.requirement_id,
        kind=kind,
        subject_key=requirement.subject_key,
        source_event_ids=tuple(dict.fromkeys(sources)),
    )


def reference_assertion(
    dependencies: ReferenceActionPolicyDependencies,
    projection: ReferenceIdentityProjection,
    kind: Literal["sample_identity", "taxon_identity"],
) -> RequirementUserAssertion | None:
    """The latest User assertion for a reference requirement, if any: the statement
    the model types the sample name from, or the taxon it resolves (#472 Q2 a)."""

    if hasattr(dependencies.journey, "read_view"):
        from bio_neb.application.planning_read import read_planning

        resolution = read_planning(dependencies.journey).requirements
        if resolution is None:
            raise ValueError("reference assertion lacks its checked requirements")
    else:
        # Historical replay supplies its exact merged evidence prefix, without a
        # live Journey owner. Reopening current state would change that scope.
        resolution = project_requirement_resolution(dependencies.journey.read())
    matches = tuple(item for item in resolution.requirements if item.kind == kind)
    if len(matches) != 1 or not matches[0].user_assertions:
        return None
    return matches[0].user_assertions[-1]


def reference_discovery(
    projection: ReferenceIdentityProjection, event_id: str | None = None
) -> DiscoverAssembliesObserved:
    """The discovery page ``event_id`` names, the latest page by default."""
    event_id = event_id or projection.discovery_event_id
    if event_id is None or event_id not in projection.discovery_event_ids:
        raise ValueError("reference action requires current discovery evidence")
    views = tuple(item for item in projection.facts if item.source_event_id == event_id)
    if len(views) != 1 or views[0].event_type != "ReferenceAssembliesRecorded":
        raise ValueError("current discovery evidence is missing from the projection")
    return DiscoverAssembliesObserved.model_validate(views[0].result)


def applicable_reference_receipt(
    projection: ReferenceIdentityProjection,
) -> PlanningHitlResponseReceipt | None:
    """The researcher's latest answer to a question citing the latest discovery page.

    An answer holds for the page it was asked on: once the model reads another
    page, the reference question and choice open again on that page (#471 live
    a1 F3).
    """
    if projection.discovery_event_id is None:
        return None
    matches = tuple(
        receipt
        for receipt in projection.planning_hitl.receipts
        if projection.discovery_event_id in receipt.assertion.evidence_event_ids
    )
    return matches[-1] if matches else None


def answered_choice_receipt(
    projection: ReferenceIdentityProjection,
) -> PlanningHitlResponseReceipt | None:
    """The researcher's latest answer to a question citing the current choice."""
    choices = current_choices(projection)
    if not choices:
        return None
    matches = tuple(
        receipt
        for receipt in projection.planning_hitl.receipts
        if choices[-1].source_event_id in receipt.assertion.evidence_event_ids
    )
    return matches[-1] if matches else None


def selected_reference(
    projection: ReferenceIdentityProjection,
) -> SelectedAssemblyEvidence | None:
    if (
        current_choices(projection)
        and current_choices(projection)[-1].discovery_event_id is None
        and projection.verified_reference is None
        # A chosen suggestion replaces the supplied accession.
        and projection.chosen_suggestion is None
    ):
        choice = current_choices(projection)[-1]
        return SuppliedAssemblyEvidence(
            assembly_accession=choice.assembly_accession,
            assertion_event_id=choice.assertion_event_id,
            selection_event_id=choice.source_event_id,
            expected_taxon=projection.resolved_taxon,
        )
    if projection.status != "reference_candidates":
        return None
    choice = discovery_choice(projection)
    if choice is not None:
        if choice.assembly_accession is None:
            return None
        discovery = reference_discovery(projection, choice.discovery_event_id)
        row = discovery.row(choice.assembly_accession)
        if row is None:
            return None
        return SelectedAssemblyEvidence(
            row=row,
            discovery_evidence_id=discovery.evidence_id,
            discovery_event_id=choice.discovery_event_id,
            selection_event_id=choice.source_event_id,
        )
    # Code picks no row: only the model's choice or the researcher's
    # selection from the page their question cited selects one.
    receipt = applicable_reference_receipt(projection)
    if receipt is None or receipt.assertion.selected_candidate_id is None:
        return None
    for event_id in reversed(projection.discovery_event_ids):
        if event_id not in receipt.assertion.evidence_event_ids:
            continue
        discovery = reference_discovery(projection, event_id)
        accession = receipt.assertion.selected_candidate_id.removeprefix("assembly:")
        row = discovery.row(accession)
        if row is not None:
            return SelectedAssemblyEvidence(
                row=row,
                discovery_evidence_id=discovery.evidence_id,
                discovery_event_id=event_id,
                selection_event_id=receipt.response_event_id,
            )
    return None


def discovery_choice(projection):
    """The Agent's latest current choice over a current discovery page, if any.

    ``assembly_accession`` is None for explicit ambiguity. A new choice clears the
    projection's description, so ``description_event_id`` belongs to this choice.
    """

    choices = [
        c
        for c in current_choices(projection)
        if c.discovery_event_id in projection.discovery_event_ids
    ]
    return choices[-1] if choices else None


def question_choice(projection):
    """The Agent's latest discovery choice no researcher answer has cited yet.

    The reference question asks about it; once answered, a question asks about
    the latest page instead (#471 live a1 F3).
    """
    choice = discovery_choice(projection)
    if choice is None or any(
        choice.source_event_id in receipt.assertion.evidence_event_ids
        for receipt in projection.planning_hitl.receipts
    ):
        return None
    return choice


def question_page(projection) -> str | None:
    """The discovery page the reference question shows: the open choice's page,
    else the latest page."""
    choice = question_choice(projection)
    return choice.discovery_event_id if choice else projection.discovery_event_id


def current_choices(projection):
    return tuple(
        c
        for c in projection.choices
        if (
            projection.revision_event_id is None
            or projection.revision_event_id in c.grounding.source_event_ids
        )
        and (
            c.discovery_event_id is None
            or c.discovery_event_id in projection.discovery_event_ids
        )
    )
