"""Reference discovery application commands."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from bio_neb.application.planning_read import prefix_snapshot
from bio_neb.journey import Journey
from bio_neb.parabricks.execution_proposal import (
    ProposalCompilationRefused,
    _compile_execution_proposal,
    project_execution_hitl,
)
from bio_neb.reference_discovery.reference_source import (
    GZIP_REFERENCE_SEQUENCES,
    ReferenceSourceAdmission,
    ReferenceSourceAdmitted,
    reference_source_admission_id,
)

if TYPE_CHECKING:
    pass

from bio_neb.application.commands import ApprovedExecutionCommandRefused


def admit_discovered_reference_source(
    journey: Journey,
) -> ReferenceSourceAdmission:
    """Retain the discovered genomic FASTA's source identity for proposal/HITL review.

    The NCBI adapter observed the publisher file URL, its md5 and HEAD size; nobody
    types them. Acquisition later proves the bytes against this identity.
    """

    # Compilation needs the owner's semantic facts and physical-contact evidence.
    context = prefix_snapshot(journey)
    events = context.timeline
    proposal = _compile_execution_proposal(context, bind_execution_package=False)
    fasta = [
        item.candidate for item in proposal.companions if item.role == "genomic_fasta"
    ]
    if len(fasta) != 1:
        raise ApprovedExecutionCommandRefused("reference_source_admission_conflict")
    (candidate,) = fasta
    if (
        candidate.authority_locator is None
        or candidate.source_locator is None
        or candidate.checksum_authority_locator is None
        or candidate.checksum_algorithm is None
        or not candidate.byte_size
    ):
        raise ProposalCompilationRefused("reference_source_unobserved")
    material = {
        "reference_asset_id": candidate.asset_id,
        "assembly_accession": proposal.reference.assembly_accession,
        "authority_locator": candidate.authority_locator,
        "source_mode": "public_url",
        "source_locator": candidate.source_locator,
        "source_representation": "gzip_fasta",
        "expected_size_bytes": candidate.byte_size,
        "expected_checksum_algorithm": candidate.checksum_algorithm,
        "expected_checksum": candidate.checksum,
        "checksum_authority_locator": candidate.checksum_authority_locator,
        "derivation_rule": GZIP_REFERENCE_SEQUENCES,
    }
    admission = ReferenceSourceAdmission(
        admission_id=reference_source_admission_id(material), **material
    )
    existing = [
        ReferenceSourceAdmitted.model_validate(
            {"event_type": e["event_type"], **e["payload"]}
        )
        for e in events
        if e["event_type"] == "ReferenceSourceAdmitted"
    ]
    current = [
        e for e in existing if e.reference_event_id == proposal.reference_event_id
    ]
    if current:
        if len(current) != 1 or current[0].admission != admission:
            raise ApprovedExecutionCommandRefused("reference_source_admission_conflict")
        return current[0].admission
    if project_execution_hitl(events).status not in {
        "not_requested",
        "execution_revision_pending",
    }:
        raise ApprovedExecutionCommandRefused("reference_source_admission_too_late")
    # This is the reviewable source identity, not acquired-byte proof.
    # Necessary acquisition owns the physical comparison.
    fact = ReferenceSourceAdmitted(
        admission=admission,
        reference_event_id=proposal.reference_event_id,
        selection_event_id=proposal.reference_basis.choice.source_event_id
        if proposal.reference_basis is not None
        else None,
    )
    payload = fact.model_dump(mode="json")
    payload.pop("event_type")
    journey.record(
        fact.event_type,
        payload,
        [events[-1]["event_id"]],
        event_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{journey.journey_id}:{proposal.reference_event_id}:{admission.admission_id}",
            )
        ),
    )
    return admission
