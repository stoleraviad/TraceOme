"""Checked collection membership and exact historical Investigation meaning."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

from bio_neb.application.planning_read import PrefixSnapshot
from bio_neb.investigation.investigation import (
    InvestigationProjection,
    InvestigationProposal,
    UserAssertion,
    project_snapshot_investigation,
)
from bio_neb.investigation.investigation_observation import (
    InvestigationCollectionEvidenceRecorded,
    InvestigationCollectionItemState,
    InvestigationEvidenceRecorded,
    correlate_investigation_collection_evidence,
    derive_investigation_run_provenance,
    observation_target,
    validate_investigation_collection_evidence_fact,
    validate_investigation_evidence_fact,
    validate_investigation_work_item_started_fact,
)
from bio_neb.investigation.investigation_policy import (
    InvestigationPolicyDecision,
    project_investigation_policy,
)
from bio_neb.investigation.item_evidence import (
    InvestigationItemEvidenceCollected,
    ItemEvidence,
    checked_item_evidence_folds,
    fold_item_evidence,
)

if TYPE_CHECKING:
    from bio_neb.journey import Journey
    from bio_neb.reporting.finalization import FinalizationProjection, ReportBasis


@dataclass(frozen=True)
class RecordedInvestigationMember:
    """Local evidence identity and its checked original authority/contact frontier."""

    event: dict[str, Any]
    evidence: InvestigationEvidenceRecorded
    model_contact_count: int
    source_contact_count: int


@dataclass(frozen=True)
class InvestigationMember:
    state: InvestigationCollectionItemState
    recorded: RecordedInvestigationMember | None


@dataclass(frozen=True)
class InvestigationCollectionRead:
    event: dict[str, Any]
    evidence: InvestigationCollectionEvidenceRecorded
    proposal: InvestigationProposal
    question: UserAssertion
    active_members: tuple[InvestigationMember, ...]


@dataclass(frozen=True)
class _CorrelatedCollection:
    event: dict[str, Any]
    evidence: InvestigationCollectionEvidenceRecorded
    proposal: InvestigationProposal
    members: tuple[InvestigationEvidenceRecorded, ...]


@dataclass(frozen=True)
class InvestigationRead:
    """Checked collection history, separate from command eligibility."""

    prefix_event_ids: tuple[str, ...]
    timeline: list[dict[str, Any]]
    investigation: InvestigationProjection
    evidence: InvestigationCollectionEvidenceRecorded | None
    work_policy: InvestigationPolicyDecision
    collections: tuple[InvestigationCollectionRead, ...]
    recorded_members: tuple[RecordedInvestigationMember, ...]
    # The latest collection's item evidence: its fold, or the items so far (0047).
    item_evidence: tuple[ItemEvidence, ...] = ()

    @property
    def active_members(self) -> tuple[InvestigationMember, ...]:
        return self.collections[-1].active_members if self.collections else ()


@dataclass(frozen=True)
class TableReadingStage:
    """The open post-approval VCF table stage of the latest collection (ADR 0047).

    ``windows_pending`` are the items whose window the batch has not read;
    ``unread`` the items with lookup evidence that no chunk has bound, in
    selected order.
    """

    collection_event_id: str
    collection: InvestigationCollectionEvidenceRecorded
    proposal: InvestigationProposal
    windows_pending: tuple[str, ...]
    unread: tuple[str, ...]


@dataclass(frozen=True)
class FinalizationCandidate:
    source_head_event: dict[str, Any]
    collection: InvestigationCollectionRead | None
    investigation: InvestigationRead
    outcome: str = "investigation_evidence_ready"


@dataclass(frozen=True)
class ApprovedReportRead:
    basis: ReportBasis
    approval_event_id: str


@dataclass(frozen=True)
class CompletionRead:
    investigation: InvestigationRead
    finalization: FinalizationProjection
    candidate: FinalizationCandidate | None
    approved_report: ApprovedReportRead | None


class _InvestigationContext:
    """Lazy domain facets over the same isolated snapshot and exact prefix owner."""

    def __init__(self, snapshot: PrefixSnapshot):
        self.snapshot = snapshot
        self.timeline = snapshot.timeline

    def through(self, event_id: str) -> _InvestigationContext:
        if event_id not in self.snapshot.by_id:
            raise ValueError("Investigation source is outside the owned prefix")
        if event_id == self.snapshot.prefix_event_ids[-1]:
            return self
        # The prefix's own facets, shared by every reader of that prefix.
        return self.snapshot.through(event_id).derived(_InvestigationContext)

    @property
    def projection(self) -> InvestigationProjection:
        # The snapshot's one projection, shared with its Attempts' terminals.
        return self.snapshot.derived(project_snapshot_investigation)

    @cached_property
    def member_facts(
        self,
    ) -> tuple[tuple[dict[str, Any], InvestigationEvidenceRecorded], ...]:
        return tuple(
            (event, fact)
            for event in self.timeline
            if (fact := validate_investigation_evidence_fact(event)) is not None
        )

    @cached_property
    def collection_facts(
        self,
    ) -> tuple[tuple[dict[str, Any], InvestigationCollectionEvidenceRecorded], ...]:
        return tuple(
            (event, fact)
            for event in self.timeline
            if (fact := validate_investigation_collection_evidence_fact(event))
            is not None
        )

    @cached_property
    def item_evidence_folds(self) -> dict[str, InvestigationItemEvidenceCollected]:
        from bio_neb.investigation.vcf_table_reading import validate_table_reading

        # Each window and chunk against the facts it rests on, and each chunk's
        # readings against its accepted output, before any fold or item view
        # reads them (ADR 0034).
        validate_table_reading(
            self.timeline, self.projection, checked_once=self.snapshot.checked_once
        )
        return checked_item_evidence_folds(
            self.timeline,
            {event["event_id"]: fact for event, fact in self.collection_facts},
            checked_once=self.snapshot.checked_once,
        )

    @cached_property
    def read(self) -> InvestigationRead:
        collections, recorded = project_investigation_evidence(self)
        evidence = collections[-1].evidence if collections else None
        fold = (
            self.item_evidence_folds.get(collections[-1].event["event_id"])
            if collections
            else None
        )
        return InvestigationRead(
            self.snapshot.prefix_event_ids,
            self.timeline,
            self.projection,
            evidence,
            project_investigation_policy(self.timeline, self.projection, evidence),
            collections,
            recorded,
            fold.items
            if fold is not None
            else fold_item_evidence(
                collections[-1].event["event_id"], evidence, self.timeline
            )
            if evidence is not None
            else (),
        )

    @cached_property
    def table_reading_stage(self) -> TableReadingStage | None:
        """Open after the latest collection may finalize, until its item fold.

        Every settled item is accounted: its window is pending until the batch
        reads it, and its reading until a chunk binds it.
        """
        read = self.read
        if not read.collections:
            return None
        latest = read.collections[-1]
        collection_event_id = latest.event["event_id"]
        if collection_event_id in self.item_evidence_folds or not read.item_evidence:
            return None
        index = next(
            i
            for i, event in enumerate(self.timeline)
            if event["event_id"] == collection_event_id
        )
        later = self.timeline[index + 1 :]
        # A compiled basis or an ended Run closes the stage.
        if any(
            event["event_type"] in {"FinalizationBasisCompiled", "RunFailed"}
            for event in later
        ):
            return None
        anchor = [
            latest.event,
            *(e for e in later if e["event_type"] == "InvestigationPolicyAmended"),
        ][-1]
        if not self.through(anchor["event_id"]).read.work_policy.finalization_eligible:
            return None
        return TableReadingStage(
            collection_event_id,
            latest.evidence,
            latest.proposal,
            tuple(
                item.work_item_id
                for item in read.item_evidence
                if item.window.status == "pending"
            ),
            tuple(
                item.work_item_id
                for item in read.item_evidence
                if item.reading.status == "pending" and item.reading.attempt_id is None
            ),
        )

    @cached_property
    def candidate(self) -> FinalizationCandidate | None:
        from bio_neb.application.queries import ended_run_outcome
        from bio_neb.recording import review_head
        from bio_neb.reporting.explanation import without_explanation

        if not self.timeline:
            return None
        # A report's explanation is presentation after it: it ends no work.
        head = review_head(without_explanation(self.timeline))
        if head != self.timeline[-1]:
            return self.through(head["event_id"]).candidate
        if head["event_type"] in {
            "FinalizationBasisCompiled",
            "FinalizationReviewRecorded",
            "GroundedReportPublished",
        }:
            return None
        if self.table_reading_stage is not None:
            # The post-approval VCF table stage runs first; Finalization reads its fold.
            return None
        read = self.read
        if (
            read.collections
            and head["event_type"]
            in {
                "InvestigationCollectionEvidenceRecorded",
                "InvestigationPolicyAmended",
                "InvestigationItemEvidenceCollected",
            }
            and read.work_policy.finalization_eligible
        ):
            return FinalizationCandidate(head, read.collections[-1], read)
        outcome = ended_run_outcome(self.snapshot)
        return (
            FinalizationCandidate(
                head, read.collections[-1] if read.collections else None, read, outcome
            )
            if outcome is not None
            else None
        )

    @cached_property
    def completion(self) -> CompletionRead:
        from bio_neb.reporting.finalization import (
            held_finalization_fact,
            project_finalization,
            validate_finalization_review_fact,
        )

        finalization = project_finalization(self.timeline, snapshot=self.snapshot)
        approved = None
        if finalization.report_basis is not None:
            matches = [
                event
                for event in self.timeline
                if (
                    review := held_finalization_fact(
                        self.snapshot, validate_finalization_review_fact, event
                    )
                )
                is not None
                and review.report_basis == finalization.report_basis
            ]
            if len(matches) != 1:
                raise ValueError("Report basis lacks exact approval event")
            approved = ApprovedReportRead(
                finalization.report_basis, matches[0]["event_id"]
            )
        return CompletionRead(self.read, finalization, self.candidate, approved)


def current_investigation(journey: Journey) -> InvestigationProjection:
    """The owner's checked Investigation of its current evidence.

    One projection per prefix, held by the owner across its appends, so a member
    admission, an item question or a command step re-folds only a new prefix.
    The projection is frozen and shared; nothing is copied per reader.
    """
    from bio_neb.application.planning_read import prefix_snapshot

    return prefix_snapshot(journey).investigation.projection


def project_investigation_evidence(  # noqa: C901, PLR0915 - Retained structural checks keep stage-specific lineage explicit.
    context: _InvestigationContext,
) -> tuple[
    tuple[InvestigationCollectionRead, ...], tuple[RecordedInvestigationMember, ...]
]:
    from bio_neb.recording import work_frontier

    timeline = context.timeline
    investigation = context.projection
    matches = list(context.member_facts)
    collections = list(context.collection_facts)
    if not matches and not collections:
        return (), ()
    if investigation.authority is None or investigation.investigation_input is None:
        raise ValueError("Journey repeated or fabricated Investigation evidence")
    investigation_input = investigation.investigation_input
    grants = {grant.authority_id: grant for grant in investigation.authorities}
    grant_events = {
        event["payload"]["authority"]["authority_id"]: event
        for event in timeline
        if event.get("event_type") == "InvestigationAuthorityGranted"
    }
    started = [
        (event, fact)
        for event in timeline
        if (fact := validate_investigation_work_item_started_fact(event)) is not None
    ]
    if len({fact.work_item_id for _, fact in started}) != len(started):
        raise ValueError("Investigation collection repeated a Work-item start")
    if len({fact.work_item.work_item_id for _, fact in matches}) != len(matches):
        raise ValueError("Investigation collection repeated a Work item")
    checked_collections = []
    previous_collection = None
    consumed_starts: set[str] = set()
    consumed_evidence: set[str] = set()
    # Each member's inherited contacts, checked against its collection's prefix.
    inherited: dict[str, tuple[int, int]] = {}
    consumed_grants: set[str] = set()
    for collection_event, collection in collections:
        authority = grants.get(collection.authority.authority_id)
        grant_event = grant_events.get(collection.authority.authority_id)
        collection_index = timeline.index(collection_event)
        collection_projection = context.through(collection_event["event_id"]).projection
        proposal = next(
            (
                item
                for item in investigation.proposals
                if item.proposal_id
                == (
                    collection.collection_proposal_id
                    or collection.authority.proposal_id
                )
            ),
            None,
        )
        if (
            authority != collection.authority
            or grant_event is None
            or proposal is None
            or proposal != collection_projection.proposals[-1]
            or (
                collection.carried_revision_id is None
                and any(
                    item not in proposal.work_items
                    for item in authority.authorized_work_items
                )
            )
            or (
                collection.carried_revision_id is None
                and collection.authority.authority_id in consumed_grants
            )
            or timeline.index(grant_event) >= collection_index
            or (
                collection.collection_work_item_ids or collection.selected_work_item_ids
            )
            != tuple(item.work_item_id for item in proposal.work_items)
            or collection.previous_collection_state_id
            != (
                previous_collection.collection_state_id if previous_collection else None
            )
        ):
            raise ValueError("Investigation collection changed its approved authority")
        if collection.carried_revision_id is None:
            consumed_grants.add(authority.authority_id)
        prior_states = (
            {item.work_item_id: item for item in previous_collection.item_states}
            if previous_collection
            else {}
        )
        selected = (
            {}
            if collection.carried_revision_id
            else {item.work_item_id: item for item in authority.authorized_work_items}
        )
        states = {item.work_item_id: item for item in collection.item_states}
        granted_ids = {
            item.work_item_id
            for grant_id, grant in grants.items()
            if timeline.index(grant_events[grant_id]) < collection_index
            for item in grant.authorized_work_items
        }
        for work_item_id, state in states.items():
            if work_item_id in selected:
                if work_item_id in consumed_starts:
                    raise ValueError("Investigation collection repeated observed work")
                continue
            prior = prior_states.get(work_item_id)
            if prior is not None and prior.status not in {
                "approval_pending",
                "observation_pending",
            }:
                if state != prior.model_copy(
                    update={"selected_ordinal": state.selected_ordinal}
                ):
                    raise ValueError(
                        "Investigation collection changed retained sibling evidence"
                    )
            elif (
                state.status
                != (
                    "observation_pending"
                    if work_item_id in granted_ids
                    else "approval_pending"
                )
                or state.evidence_state_id is not None
            ):
                raise ValueError("Investigation collection changed pending membership")
        if collection.carried_revision_id is not None:
            revision = (
                collection_projection.revisions[-1]
                if collection_projection.revisions
                else None
            )
            if (
                previous_collection is None
                or revision is None
                or revision.revision_id != collection.carried_revision_id
                or revision.work_requests
                or collection.command != previous_collection.command
                or collection.authority != previous_collection.authority
                or timeline[collection_index - 1].get("event_type")
                != "InvestigationReviewRequested"
                or timeline[collection_index - 1].get("payload", {}).get("proposal_id")
                != proposal.proposal_id
                or collection_event.get("parent_ids")
                != [timeline[collection_index - 1]["event_id"]]
            ):
                raise ValueError(
                    "Investigation carried collection changed revision lineage"
                )
            correlated = correlate_investigation_collection_evidence(
                collection,
                tuple(
                    fact
                    for event, fact in matches
                    if timeline.index(event) < collection_index
                    and fact.evidence_state.evidence_state_id
                    in {item.evidence_state_id for item in collection.item_states}
                ),
            )
            checked_collections.append(
                _CorrelatedCollection(
                    collection_event, collection, proposal, correlated
                )
            )
            previous_collection = collection
            continue
        batch_starts = [
            (event, fact)
            for event, fact in started
            if fact.authority_id == authority.authority_id
        ]
        expected_started_ids = {
            item.work_item_id
            for item in collection.item_states
            if item.work_item_id in selected
            and item.status not in {"not_started", "stopped"}
        }
        if {fact.work_item_id for _, fact in batch_starts} != expected_started_ids:
            raise ValueError("Investigation collection start membership changed")
        first_index = (
            timeline.index(batch_starts[0][0]) if batch_starts else collection_index
        )
        if first_index <= timeline.index(grant_event):
            raise ValueError("Investigation evidence preceded its approval")
        # An observation resumed in a fresh Pass, or after a held configuration
        # fault, starts after that metadata, not after a domain change.
        previous_event = work_frontier(timeline[:first_index])
        if previous_event.get("event_type") not in {
            "InvestigationAuthorityGranted",
            "InvestigationCollectionEvidenceRecorded",
            "InvestigationAssertionRecorded",
            "InvestigationReviewRequested",
            "InvestigationPolicyAmended",
            "InvestigationDownloadReviewAnswered",
        }:
            raise ValueError("Investigation observation changed its starting frontier")
        for ordinal, work_item in enumerate(authority.authorized_work_items, start=1):
            work_item_id = work_item.work_item_id
            entry = next(
                (
                    (event, fact)
                    for event, fact in batch_starts
                    if fact.work_item_id == work_item_id
                ),
                None,
            )
            if entry is None:
                continue
            started_event, started_fact = entry
            target = work_item.target
            if (
                started_event.get("parent_ids") != [previous_event["event_id"]]
                or work_frontier(timeline[: timeline.index(started_event)])
                is not previous_event
                or started_fact.selected_ordinal != ordinal
                or started_fact.sample_id != work_item.sample_id
                or started_fact.target.assembly != target.assembly_accession
                or started_fact.target.contig != target.contig
                or (
                    # A region's start names the region it reads.
                    started_fact.target != observation_target(work_item)
                    if work_item.intended_action == "observe_region"
                    else started_fact.target.one_based_position
                    != target.one_based_position
                    or started_fact.target.ref != target.ref
                    or started_fact.target.alt != target.alt
                )
            ):
                raise ValueError(
                    "Investigation collection Work-item start changed lineage"
                )
            consumed_starts.add(work_item_id)
            previous_event = started_event
            retained = next(
                (
                    (event, fact)
                    for event, fact in matches
                    if fact.work_item.work_item_id == work_item_id
                ),
                None,
            )
            if retained is not None:
                evidence_event, fact = retained
                descriptor = fact.sample_observation.vcf_descriptor
                if (
                    evidence_event.get("parent_ids") != [previous_event["event_id"]]
                    or timeline.index(evidence_event)
                    != timeline.index(previous_event) + 1
                    or fact.authority != authority
                    or fact.command != collection.command
                    or fact.work_item != work_item
                    or fact.sample_observation.sample_id
                    != investigation_input.sample_id
                    or fact.sample_observation.assembly_accession
                    != investigation_input.assembly_accession
                    or descriptor.bucket != investigation_input.vcf.bucket
                    or descriptor.object_key != investigation_input.vcf.object_key
                    or descriptor.byte_count != investigation_input.vcf.byte_count
                    or descriptor.sha256 != investigation_input.vcf.sha256
                    # Inherited evidence keeps the provenance it was recorded under.
                    or (
                        fact.sample_observation.run_provenance,
                        fact.sample_observation.provenance_origin_event_id,
                    )
                    != derive_investigation_run_provenance(
                        timeline[: timeline.index(evidence_event)]
                    )
                    or fact.evidence_state.inherited_model_contact_count
                    != collection_projection.model_contact_count
                    or fact.evidence_state.inherited_source_contact_count
                    != collection_projection.source_contact_count
                ):
                    raise ValueError(
                        "Investigation collection evidence membership changed"
                    )
                previous_event = evidence_event
                consumed_evidence.add(work_item_id)
                inherited[evidence_event["event_id"]] = (
                    collection_projection.model_contact_count,
                    collection_projection.source_contact_count,
                )
        if (
            collection_event.get("parent_ids") != [previous_event["event_id"]]
            or collection_index != timeline.index(previous_event) + 1
        ):
            raise ValueError("Investigation collection terminal changed parent")
        correlated = correlate_investigation_collection_evidence(
            collection,
            tuple(
                fact
                for event, fact in matches
                if timeline.index(event) < collection_index
                and fact.evidence_state.evidence_state_id
                in {item.evidence_state_id for item in collection.item_states}
            ),
        )
        checked_collections.append(
            _CorrelatedCollection(collection_event, collection, proposal, correlated)
        )
        previous_collection = collection
    if collections:
        if consumed_starts != {
            fact.work_item_id for _, fact in started
        } or consumed_evidence != {fact.work_item.work_item_id for _, fact in matches}:
            raise ValueError("Investigation evidence has no terminal collection")
        return _retain_checked_members(context, tuple(checked_collections), inherited)
    raise ValueError("Investigation evidence has no terminal collection")


def _retain_checked_members(
    context: _InvestigationContext,
    collections: tuple[_CorrelatedCollection, ...],
    inherited: dict[str, tuple[int, int]],
) -> tuple[
    tuple[InvestigationCollectionRead, ...], tuple[RecordedInvestigationMember, ...]
]:
    """Keep the fold's checked associations and each member's original frontier.

    A member's evidence follows its start inside its collection's batch, where
    no Investigation fact is recorded, so its original frontier's contacts are
    the collection's, which the batch check already compared.
    """
    recorded = [
        RecordedInvestigationMember(event, fact, *inherited[event["event_id"]])
        for event, fact in context.member_facts
    ]
    by_evidence = {
        item.evidence.evidence_state.evidence_state_id: item for item in recorded
    }
    retained = []
    for checked in collections:
        collection = checked.evidence
        proposal = checked.proposal
        question = next(
            (
                item
                for item in context.projection.assertions
                if item.assertion_id == proposal.assertion_id
            ),
            None,
        )
        if question is None:
            raise ValueError("Investigation collection question is missing")
        selected = {
            item.evidence_state.evidence_state_id: by_evidence[
                item.evidence_state.evidence_state_id
            ]
            for item in checked.members
        }
        retained.append(
            InvestigationCollectionRead(
                checked.event,
                collection,
                proposal,
                question,
                tuple(
                    InvestigationMember(state, selected.get(state.evidence_state_id))
                    for state in collection.item_states
                ),
            )
        )
    return tuple(retained), tuple(recorded)
