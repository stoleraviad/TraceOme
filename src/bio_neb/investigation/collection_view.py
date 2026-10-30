"""The whole Investigation collection in one view, with its cross-SNP cases (#360 W).

Each member's planning reads only its own evidence, so the researcher reviews the
collection here: one row per requested SNP, in the researcher's order, with its
verified identity, its candidate Work items and, once looked up, each item's VCF
outcome. A deterministic check flags every case that is not simple across SNPs:
two requests resolving to one rsID, two rsIDs at one locus, sources that disagree,
several alternate alleles, a request with no Work item, and after the lookups an
overlapping record, mixed ALT outcomes or a match found by trimming. A case names
its members and the evidence it rests on. It reports only: no case changes an
admission, a claim scope or any authority.

Once looked up, each item's row also carries the VCF table's cells (#361): its
call quality, its fixed window of the prepared VCF and its coverage, and, after
the table reading, that item's own reading. ``vcf_table`` gives the rows the
table reader reads.

When the Run has ended, ``ended_collection_view`` adds each request's and each
item's real state (#339 ``report``): observed, approved but not observed,
failed, never approved, an unselected candidate, removed, deferred, rejected,
unresolved,
checked but never proposed, or never run. Every request of the question has a row, also when the Run ended
before its check. A state reports what happened; it admits nothing.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Literal

from pydantic import Field, StrictInt, StrictStr

from bio_neb.contracts import StrictFrozenContract
from bio_neb.investigation.case_origins import (
    CaseOrigin,
    ContradictorySources,
    MergedIdentity,
    MixedAltOutcomes,
    MultiAlt,
    NoWorkItem,
    NoWorkReason,
    OverlappingRecord,
    SharedIdentity,
    SharedLocus,
    TrimmedMatch,
    no_work_note,
)
from bio_neb.investigation.investigation_request import (
    PlacedPosition,
    StatedPosition,
    effective_requested_rsid,
)
from bio_neb.investigation.lookup_origins import LookupOrigin, ModelMatchedName
from bio_neb.investigation.qualification_origins import (
    ItemApprovedNotObserved,
    ItemFailed,
    ItemNotApproved,
    ItemSetAside,
    ItemUnselected,
    RequestIssueRejected,
    RequestNeverRun,
    RequestNotChecked,
    RequestNotProposed,
    RequestRejected,
    RequestSetAside,
    RequestUnresolved,
    StateOrigin,
)
from bio_neb.investigation.state_words import (
    ItemState,
    RequestState,
    item_words,
    request_words,
)
from bio_neb.investigation.vcf_observation import (
    ExactAlleleObserved,
    LocusOverlapped,
    ObservedVcfRecord,
    TargetAlleleNotCarried,
    TargetGenotypeUnresolved,
    VcfRegionObserved,
    VcfWindowObserved,
)

if TYPE_CHECKING:
    # The proposal and its observations are read, never imported at load: the
    # Investigation projection carries this view.
    from bio_neb.investigation.investigation import (
        InvestigationProjection,
        InvestigationProposal,
        InvestigationRequestMember,
        RefSnpToolResult,
        UserAssertion,
        WorkItem,
    )
    from bio_neb.investigation.investigation_observation import (
        InvestigationCollectionItemState,
        SampleObservation,
    )
    from bio_neb.investigation.item_evidence import ItemEvidence
    from bio_neb.investigation.vcf_observation import VcfWindowQueryResult

CollectionCaseCode = Literal[
    "merged_identity",
    "shared_identity",
    "shared_locus",
    "contradictory_sources",
    "multi_alt",
    "no_work_item",
    "overlapping_record",
    "mixed_alt_outcomes",
    "trimmed_match",
]
# The lookup outcomes one record answers, with its genotype and the overlapping
# records beside it.
_RECORD_ANSWERS = (
    ExactAlleleObserved,
    TargetAlleleNotCarried,
    TargetGenotypeUnresolved,
)


class CollectionCase(StrictFrozenContract):
    """One case across the collection that is not simple, and what shows it."""

    request_member_ids: tuple[StrictStr, ...] = Field(min_length=1)
    work_item_ids: tuple[StrictStr, ...] = ()
    # The retrieved sources and sample observations the case rests on.
    evidence_ids: tuple[StrictStr, ...] = Field(min_length=1)
    # What the case is, typed: its audit sentence and plain words.
    origin: CaseOrigin

    @property
    def code(self) -> CollectionCaseCode:
        return self.origin.code


def _absent(value) -> bool:
    return value is None or value == ()


def _noted(words: str | None, origin) -> str | None:
    note = None if origin is None else origin.note()
    return words if note is None else f"{words} ({note})"


class ItemWindowCells(StrictFrozenContract):
    """An item's fixed window as the table shows it: one line per record."""

    flank_bases: StrictInt = Field(ge=0)
    outcome: Literal["window_observed", "refused"]
    records: tuple[StrictStr, ...] = ()
    refusal: StrictStr | None = Field(default=None, exclude_if=_absent)


class RequestedSnpItem(StrictFrozenContract):
    """One candidate Work item of a requested SNP and, once looked up, its outcome.

    The VCF table's cells and the item's reading are present once they exist.
    """

    work_item_id: StrictStr
    target: StrictStr
    outcome: StrictStr | None = None
    genotype: StrictStr | None = None
    observation_id: StrictStr | None = None
    genotype_quality: StrictInt | None = Field(default=None, exclude_if=_absent)
    read_depth: StrictInt | None = Field(default=None, exclude_if=_absent)
    allele_depths: tuple[StrictInt | None, ...] | None = Field(
        default=None, exclude_if=_absent
    )
    filters: tuple[StrictStr, ...] | None = Field(default=None, exclude_if=_absent)
    qual: StrictStr | None = Field(default=None, exclude_if=_absent)
    window: ItemWindowCells | None = Field(default=None, exclude_if=_absent)
    coverage: Literal["not_measured", "measured"] | None = Field(
        default=None, exclude_if=_absent
    )
    reading: Literal["no_concern", "concern"] | None = Field(
        default=None, exclude_if=_absent
    )
    concerns: tuple[StrictStr, ...] = Field(default=(), exclude_if=_absent)
    reading_note: StrictStr | None = Field(default=None, exclude_if=_absent)
    # Present once the Run has ended: what happened to the item, and why, as its
    # typed origin; an observed item has none.
    state: ItemState | None = Field(default=None, exclude_if=_absent)
    state_origin: StateOrigin | None = Field(default=None, exclude_if=_absent)

    def state_words(self) -> str | None:
        """The ended state in the CLI's words, with its origin's audit note."""
        if self.state is None:
            return None
        return _noted(item_words(self.state).cli, self.state_origin)


class VcfTableRow(StrictFrozenContract):
    """One item's row of the VCF table (#361): what the table reader reads.

    ``table_version`` names this shape; a later cell is an optional field.
    """

    table_version: Literal["bio-neb.vcf-table/1"] = "bio-neb.vcf-table/1"
    work_item_id: StrictStr
    rsid: StrictStr | None
    target: StrictStr
    outcome: StrictStr | None
    genotype: StrictStr | None = Field(default=None, exclude_if=_absent)
    genotype_quality: StrictInt | None = Field(default=None, exclude_if=_absent)
    read_depth: StrictInt | None = Field(default=None, exclude_if=_absent)
    allele_depths: tuple[StrictInt | None, ...] | None = Field(
        default=None, exclude_if=_absent
    )
    filters: tuple[StrictStr, ...] | None = Field(default=None, exclude_if=_absent)
    qual: StrictStr | None = Field(default=None, exclude_if=_absent)
    window: ItemWindowCells | None = Field(default=None, exclude_if=_absent)
    coverage: Literal["not_measured", "measured"]
    cases: tuple[CollectionCaseCode, ...] = Field(default=(), exclude_if=_absent)


class RegionRecordRow(StrictFrozenContract):
    """One VCF record a region holds, as its own row (#367): a record the VCF holds
    there, never a finding."""

    position: StrictInt = Field(gt=0)
    ref: StrictStr
    alts: tuple[StrictStr, ...]
    genotype: StrictStr | None
    reference: Literal["matches", "differs", "outside"]


class LookupPlacement(StrictFrozenContract):
    """Where a lookup placed a target on the Run's assembly (#481 H2, owner Q1 a):
    the position and version as its source states them, when it states them (a
    placed gene and change states no position), and the placed position."""

    stated: StatedPosition | None
    stated_version: StrictStr | None
    placed: PlacedPosition


class RequestLookup(StrictFrozenContract):
    """The source answer a request's check read (#460), copied from its lookup
    result: who was asked, what, how it answered and how its finding read it."""

    authority: StrictStr
    operation: StrictStr
    asked: StrictStr
    status_code: StrictInt
    outcome: StrictStr
    verdict: StrictStr
    rationale: StrictStr
    source_id: StrictStr
    # Each sequence the model named for a contig that is not literally one of its
    # names, typed, shown wherever the request is shown.
    model_matched: tuple[ModelMatchedName, ...] = Field(default=(), exclude_if=_absent)
    # The ending's typed origin, when its lookup typed one: its plain words are what
    # the explainer's request row reads.
    origin: LookupOrigin | None = Field(default=None, exclude_if=_absent)
    # Where it placed a target stated on another version or as a gene change.
    placement: LookupPlacement | None = Field(default=None, exclude_if=_absent)


class RequestedSnp(StrictFrozenContract):
    """One requested SNP as the researcher asked it and as its sources read it."""

    request_member_id: StrictStr
    request_text: StrictStr
    requested_rsid: StrictStr | None
    verified_rsid: StrictStr | None
    source_status: StrictStr | None
    locus: StrictStr | None
    items: tuple[RequestedSnpItem, ...]
    # Who asserts this target when it is not the question's asserter.
    asserted_by: StrictStr | None = Field(default=None, exclude_if=_absent)
    # A region's records once read, each its own row, and how many more it held
    # past its record limit.
    records: tuple[RegionRecordRow, ...] = Field(default=(), exclude_if=_absent)
    records_not_reported: StrictInt | None = Field(default=None, exclude_if=_absent)
    # Why its check left it without a Work item, as its typed origin: also before
    # the Run ends, when it has no state yet.
    no_work: NoWorkReason | None = Field(default=None, exclude_if=_absent)
    cases: tuple[CollectionCaseCode, ...] = ()
    # Present once the Run has ended: what happened to the request, and why, as
    # its typed origin; a request with candidates has none.
    state: RequestState | None = Field(default=None, exclude_if=_absent)
    state_origin: StateOrigin | None = Field(default=None, exclude_if=_absent)
    # The position as its researcher stated it, before placement.
    stated_position: StatedPosition | None = Field(default=None, exclude_if=_absent)
    # The source answer its check read, when a lookup ran.
    lookup: RequestLookup | None = Field(default=None, exclude_if=_absent)

    def state_words(self) -> str | None:
        """The ended state in the CLI's words, with its origin's audit note."""
        if self.state is None:
            return None
        reason = (
            self.state_origin.reason_code
            if isinstance(self.state_origin, RequestNotChecked)
            else None
        )
        return _noted(request_words(self.state, reason).cli, self.state_origin)

    def no_work_note(self) -> str | None:
        """Why it has no Work item, in its origin's audit note."""
        return None if self.no_work is None else no_work_note(self.no_work)


class CollectionView(StrictFrozenContract):
    """Every requested SNP together, and the cases across them."""

    snps: tuple[RequestedSnp, ...] = Field(min_length=1)
    cases: tuple[CollectionCase, ...]


def collection_view(
    proposal: InvestigationProposal,
    observations: Iterable[SampleObservation] = (),
    windows: Mapping[str, VcfWindowQueryResult] | None = None,
    item_evidence: Iterable[ItemEvidence] = (),
) -> CollectionView | None:
    """The collection of ``proposal``, with each item's latest lookup if given,
    and its window and own record (coverage, reading) once they exist.

    A proposal that names no request members (an older single-request shape)
    has no per-SNP view.
    """
    members = proposal.request_members
    if not members:
        return None
    items = {item.work_item_id: item for item in proposal.work_items}
    observed = {item.work_item_id: item for item in observations}
    cells = _TableCells(windows or {}, {e.work_item_id: e for e in item_evidence})
    cases = [
        *_identity_cases(members),
        *_member_cases(members, items),
        *_lookup_cases(members, items, observed),
    ]
    cases.sort(key=lambda case: (case.code, case.request_member_ids))
    by_member: dict[str, list[CollectionCaseCode]] = defaultdict(list)
    for case in cases:
        for member_id in case.request_member_ids:
            if case.code not in by_member[member_id]:
                by_member[member_id].append(case.code)
    return CollectionView(
        snps=tuple(
            _snp(
                member,
                items,
                observed,
                tuple(by_member[member.request_member_id]),
                cells,
            )
            for member in members
        ),
        cases=tuple(cases),
    )


def vcf_table(
    proposal: InvestigationProposal,
    observations: Iterable[SampleObservation] = (),
    windows: Mapping[str, VcfWindowQueryResult] | None = None,
) -> dict[str, VcfTableRow]:
    """Every looked-up item's row, keyed by its Work item.

    A row names the cases of its item, or of its request when the case names no
    item. An item no request member names (a revision that carried Work items
    only) still gets its row, with its target's first alias as its rsID.
    """
    observations = tuple(observations)
    view = collection_view(proposal, observations, windows)
    rows: dict[str, VcfTableRow] = {}
    for snp in () if view is None else view.snps:
        for item in snp.items:
            if item.outcome is None:
                continue
            cases = tuple(
                dict.fromkeys(
                    case.code
                    for case in view.cases
                    if item.work_item_id in case.work_item_ids
                    or (
                        not case.work_item_ids
                        and snp.request_member_id in case.request_member_ids
                    )
                )
            )
            rows[item.work_item_id] = _row(
                item, snp.verified_rsid or snp.requested_rsid, cases
            )
    observed = {item.work_item_id: item for item in observations}
    cells = _TableCells(windows or {}, {})
    for work_item in proposal.work_items:
        if work_item.work_item_id in rows or work_item.work_item_id not in observed:
            continue
        aliases = (
            ()
            if work_item.intended_action == "observe_region"
            else work_item.target.aliases
        )
        rows[work_item.work_item_id] = _row(
            _item(work_item, observed[work_item.work_item_id], cells),
            aliases[0] if aliases else None,
            (),
        )
    return rows


def ended_collection_view(
    investigation: InvestigationProjection,
    proposal: InvestigationProposal | None,
    observations: Iterable[SampleObservation] = (),
    windows: Mapping[str, VcfWindowQueryResult] | None = None,
    item_evidence: Iterable[ItemEvidence] = (),
    item_states: Iterable[InvestigationCollectionItemState] = (),
    lookups: Mapping[str, RefSnpToolResult] | None = None,
) -> CollectionView | None:
    """The collection as Finalization freezes it: every request and item with its
    real state (#339 ``report``).

    ``lookups`` is each output member's latest lookup result
    (``investigation.latest_lookup_results``): a request no member planned keeps the
    source answer its check read (#460).

    ``proposal`` is the collection's, or the latest when no collection was
    observed. A member a revision removed or deferred keeps its row, and a request of the
    original question that no member planned gets one, so no request is dropped.
    Rows follow the question's order. A state admits nothing.
    """
    view = (
        None
        if proposal is None
        else collection_view(proposal, observations, windows, item_evidence)
    )
    states = _EndedStates(
        investigation,
        item_states,
        proposal,
        lookups or {},
        _output_ids(investigation),
    )
    snps: list[RequestedSnp] = []
    if view is not None:
        members = {item.request_member_id: item for item in proposal.request_members}
        snps.extend(
            states.request(snp, members[snp.request_member_id]) for snp in view.snps
        )
    rows = {snp.request_member_id: index for index, snp in enumerate(snps)}
    replaced = {
        member.previous_request_member_id
        for earlier in investigation.proposals
        for member in earlier.request_members
        if member.previous_request_member_id is not None
    }
    empty = _TableCells({}, {})
    for earlier in investigation.proposals:
        work_items = {item.work_item_id: item for item in earlier.work_items}
        for member in earlier.request_members:
            removed = [
                identity
                for identity in member.work_item_ids
                if identity in states.set_aside
            ]
            if not removed:
                continue
            if member.request_member_id in rows:
                # A revision kept the request but removed or deferred some
                # candidates: its row still lists them, each with that state.
                index = rows[member.request_member_id]
                row = snps[index]
                listed = {item.work_item_id for item in row.items}
                added = tuple(
                    states.item(
                        _item(work_items[identity], None, empty), member.work_item_ids
                    )
                    for identity in removed
                    if identity not in listed
                )
                snps[index] = row.model_copy(
                    update={
                        "items": tuple(
                            sorted((*row.items, *added), key=lambda item: item.target)
                        )
                    }
                )
            elif member.request_member_id not in replaced and set(
                member.work_item_ids
            ) <= set(removed):
                rows[member.request_member_id] = len(snps)
                snps.append(
                    states.request(_snp(member, work_items, {}, (), empty), member)
                )
    shown = rows.keys() | replaced
    question = investigation.assertions[0] if investigation.assertions else None
    order: dict[str, int] = {}
    for ordinal, request in enumerate(_question_requests(question)):
        member_id = _member_id(question, request, ordinal)
        order[member_id] = ordinal
        if member_id not in shown:
            snps.append(states.unplanned(question, request, ordinal))
    # A request's replacement keeps its place in the question, however often
    # it was amended or clarified.
    for earlier in investigation.proposals:
        for member in earlier.request_members:
            if member.previous_request_member_id in order:
                order.setdefault(
                    member.request_member_id, order[member.previous_request_member_id]
                )
    if not snps:
        return None
    snps.sort(key=lambda snp: order.get(snp.request_member_id, len(order)))
    return CollectionView(snps=tuple(snps), cases=() if view is None else view.cases)


def _output_ids(investigation) -> dict[str, str]:
    """Each request member's output member, the id its lookups are recorded under:
    the request of the assertion that introduced it, at its ordinal there, the
    question's or a revision's (Standards r1 ST1). A replacement's lookup is its own;
    its predecessor's lineage orders rows only."""
    from bio_neb.investigation.investigation import investigation_output_member_id

    return {
        _member_id(assertion, request, ordinal): investigation_output_member_id(
            request, ordinal
        )
        for assertion in investigation.assertions
        for ordinal, request in enumerate(_question_requests(assertion))
    }


def _question_requests(question: UserAssertion | None) -> tuple:
    from bio_neb.investigation.investigation_request import InvestigationRequestItem

    if question is None:
        return ()
    return question.requests or (
        InvestigationRequestItem(
            request_text=question.request_text, rsid=question.rsid
        ),
    )


def _member_id(question: UserAssertion, request, ordinal: int) -> str:
    from bio_neb.investigation.investigation import request_member_id

    return request_member_id(question.assertion_id, request, ordinal)


class _EndedStates:
    """What the ended Run did with each item and request, from its retained facts."""

    def __init__(self, investigation, item_states, proposal, lookups, outputs) -> None:
        self.investigation = investigation
        self.lookups = lookups
        self.outputs = outputs
        self.approved = {
            item.work_item_id
            for authority in investigation.authorities
            for item in authority.authorized_work_items
        }
        self.collection = {item.work_item_id: item for item in item_states}
        answers = {
            receipt.response_id: receipt.response
            for receipt in investigation.review_receipts
        }
        # A candidate a later revision proposed again is current, not set aside.
        current = (
            set()
            if proposal is None
            else {item.work_item_id for item in proposal.work_items}
        )
        # Each candidate a revision removed or deferred, and who answered it.
        self.set_aside = {
            work_item_id: (kind, answers[revision.response_id].asserted_by)
            for revision in investigation.revisions
            for kind, identities in (
                ("removed", revision.removed_work_item_ids),
                ("deferred", revision.deferred_work_item_ids),
            )
            for work_item_id in identities
            if work_item_id not in current
        }

    def item(self, item: RequestedSnpItem, siblings) -> RequestedSnpItem:
        identity = item.work_item_id
        held = self.collection.get(identity)
        origin: StateOrigin | None
        if item.outcome is not None:
            state, origin = "observed", None
        elif held is not None and held.status == "failed":
            state, origin = "failed", ItemFailed(reason=held.reason)
        elif identity in self.set_aside:
            state, who = self.set_aside[identity]
            origin = ItemSetAside(code=f"item_{state}", by=who)
        elif identity in self.approved:
            state = "approved_not_observed"
            origin = ItemApprovedNotObserved(
                status=None if held is None else held.status,
                reason=None if held is None else held.reason,
            )
        elif self.approved.intersection(siblings):
            state, origin = "unselected_candidate", ItemUnselected()
        else:
            state, origin = "not_approved", ItemNotApproved()
        return item.model_copy(update={"state": state, "state_origin": origin})

    def request(self, snp: RequestedSnp, member) -> RequestedSnp:
        items = tuple(self.item(item, member.work_item_ids) for item in snp.items)
        origin: StateOrigin | None
        lookup = snp.lookup
        if member.rejection is not None:
            state = "rejected"
            origin = _rejected(member.rejection)
            # Its accounting keeps no source, but the answer its check read stays
            # its row's.
            looked = self.lookups.get(self.outputs.get(member.request_member_id))
            lookup = None if looked is None else _lookup(looked)
        elif member.not_checked is not None:
            state = "not_checked"
            origin = _not_checked(member.not_checked)
        elif (
            items
            and items[0].state in {"removed", "deferred"}
            and len({(item.state, item.state_origin) for item in items}) == 1
        ):
            # One answer set the whole request aside; mixed dispositions keep
            # each item's own state instead.
            state = items[0].state
            origin = RequestSetAside(
                code=f"request_{state}", by=items[0].state_origin.by
            )
        elif items:
            state, origin = "candidates", None
        else:
            state, origin = "unresolved", _unresolved(member)
        return snp.model_copy(
            update={
                "items": items,
                "state": state,
                "state_origin": origin,
                "lookup": lookup,
            }
        )

    def unplanned(self, question, request, ordinal: int) -> RequestedSnp:
        """A request no member planned: its issue or rejection, else never run."""
        from bio_neb.investigation.investigation import (
            investigation_issue_items,
            investigation_output_member_id,
        )

        output_id = investigation_output_member_id(request, ordinal)
        issue = next(
            (
                item
                for item in investigation_issue_items(self.investigation)
                if item.collection_member_id == output_id
            ),
            None,
        )
        rejection = _question_rejection(self.investigation, output_id)
        not_checked = next(
            (
                view.outcome
                for view in self.investigation.not_checked
                if view.collection_member_id == output_id
            ),
            None,
        )
        origin: StateOrigin
        if not_checked is not None:
            state = "not_checked"
            origin = _not_checked(not_checked)
        elif issue is not None and issue.decision == "rejected":
            state = "rejected"
            origin = RequestIssueRejected(
                by=issue.decided_by,
                source=issue.issue.source,
                verdict=issue.issue.verdict,
            )
        elif rejection is not None:
            state = "rejected"
            origin = RequestRejected(by=rejection.asserted_by, reason=rejection.reason)
        elif issue is not None:
            state = "unresolved"
            origin = RequestUnresolved(
                source=issue.issue.source,
                verdict=issue.issue.verdict,
                decision=issue.decision,
            )
        elif any(
            lineage.collection_member_id == output_id
            and lineage.accepted_output is not None
            for lineage in self.investigation.collection_lineages
        ):
            state, origin = "not_proposed", RequestNotProposed()
        else:
            state, origin = "never_run", RequestNeverRun()
        return RequestedSnp(
            request_member_id=_member_id(question, request, ordinal),
            request_text=request.request_text,
            requested_rsid=effective_requested_rsid(request),
            verified_rsid=None,
            source_status=None,
            locus=None,
            items=(),
            asserted_by=request.asserted_by,
            state=state,
            state_origin=origin,
            stated_position=request.position,
            lookup=None
            if (looked := self.lookups.get(output_id)) is None
            else _lookup(looked),
        )


def _rejected(rejection) -> RequestRejected | RequestIssueRejected:
    """A member's rejection as its state origin: the issue it removed the item over,
    with its source and verdict (owner #479 Q1 A), else the researcher's reason."""
    if rejection.issue is not None:
        return RequestIssueRejected(
            by=rejection.asserted_by,
            source=rejection.issue.source,
            verdict=rejection.issue.verdict,
        )
    return RequestRejected(by=rejection.asserted_by, reason=rejection.reason)


def _question_rejection(investigation, output_id: str):
    """The researcher's rejection of one request's question, if any."""
    questions = {
        review.request_event_id: review.question
        for review in investigation.review_requests
        if review.question is not None and review.question.issue is None
    }
    return next(
        (
            receipt.response
            for receipt in investigation.review_receipts
            if receipt.response.action == "reject"
            and receipt.request_event_id in questions
            and questions[receipt.request_event_id].collection_member_id == output_id
        ),
        None,
    )


def _row(item: RequestedSnpItem, rsid: str | None, cases) -> VcfTableRow:
    return VcfTableRow(
        work_item_id=item.work_item_id,
        rsid=rsid,
        target=item.target,
        outcome=item.outcome,
        genotype=item.genotype,
        genotype_quality=item.genotype_quality,
        read_depth=item.read_depth,
        allele_depths=item.allele_depths,
        filters=item.filters,
        qual=item.qual,
        window=item.window,
        coverage=item.coverage or "not_measured",
        cases=cases,
    )


class _TableCells:
    """The windows and item records the view's rows carry, by Work item."""

    def __init__(self, windows, evidence) -> None:
        self.windows = windows
        self.evidence = evidence


def _snp(member, items, observed, cases, cells) -> RequestedSnp:
    refsnp = member.refsnp
    candidate = refsnp.candidates[0] if refsnp and refsnp.candidates else None
    return RequestedSnp(
        request_member_id=member.request_member_id,
        request_text=member.request.request_text,
        requested_rsid=_requested(member),
        verified_rsid=None if refsnp is None else refsnp.verified_rsid,
        source_status=None if refsnp is None else refsnp.status,
        locus=None
        if candidate is None
        else (
            f"{candidate.assembly_name} {candidate.contig}:"
            f"{candidate.one_based_position} {candidate.ref}"
        ),
        items=tuple(
            _item(items[item_id], observed.get(item_id), cells)
            for item_id in sorted(
                member.work_item_ids, key=lambda item_id: _target(items[item_id])
            )
        ),
        no_work=None if member.work_item_ids else _no_work_reason(member),
        cases=cases,
        # An answer's replacement target is its researcher's.
        asserted_by=member.target_asserted_by or member.request.asserted_by,
        stated_position=member.request.position,
        lookup=None if refsnp is None else _lookup(refsnp),
        **_region_rows(member, observed),
    )


def _lookup(refsnp) -> RequestLookup:
    from bio_neb.investigation.investigation import model_matched_names

    return RequestLookup(
        authority=refsnp.source.authority,
        operation=refsnp.source.operation,
        asked=refsnp.source.request_identity,
        status_code=refsnp.source.status_code,
        outcome=refsnp.status,
        verdict=refsnp.assessment.verdict,
        rationale=refsnp.assessment.rationale,
        source_id=refsnp.source.source_id,
        model_matched=model_matched_names(refsnp.location),
        origin=refsnp.origin,
        placement=_placement(refsnp.location),
    )


def _placement(location) -> LookupPlacement | None:
    """Where the lookup placed its target, when it moved one onto the Run's assembly."""
    if location == "not_applicable" or location.placed is None:
        return None
    return LookupPlacement(
        stated=location.stated,
        stated_version=None
        if location.stated_assembly is None
        else location.stated_assembly.name,
        placed=location.placed,
    )


def _region_rows(member, observed) -> dict:
    """A region member's records, one row each, once its region is read (#367)."""
    regions = [
        observed[item_id].result
        for item_id in member.work_item_ids
        if item_id in observed
        and isinstance(observed[item_id].result, VcfRegionObserved)
    ]
    if not regions:
        return {}
    [region] = regions
    return {
        "records": tuple(
            RegionRecordRow(
                position=row.record.one_based_position,
                ref=row.record.ref,
                alts=row.record.alts,
                genotype=row.record.genotypes[0].genotype
                if len(row.record.genotypes) == 1
                else None,
                reference=row.reference,
            )
            for row in region.records
        ),
        "records_not_reported": region.not_reported,
    }


def _item(
    item: WorkItem, observation: SampleObservation | None, cells: _TableCells
) -> RequestedSnpItem:
    result = None if observation is None else observation.result
    answered = isinstance(result, _RECORD_ANSWERS)
    call = (
        next(
            (
                genotype
                for genotype in result.record.genotypes
                if genotype.sample_name == result.genotype.sample_name
            ),
            None,
        )
        if answered
        else None
    )
    evidence = cells.evidence.get(item.work_item_id)
    window = cells.windows.get(item.work_item_id)
    reading = (
        None
        if evidence is None or evidence.reading.reading is None
        else evidence.reading.reading
    )
    return RequestedSnpItem(
        work_item_id=item.work_item_id,
        target=_target(item),
        outcome=None if result is None else result.outcome,
        genotype=result.genotype.raw_genotype if answered else None,
        observation_id=None if observation is None else observation.observation_id,
        genotype_quality=None if call is None else call.gq,
        read_depth=None if call is None else call.dp,
        allele_depths=None if call is None else call.ad,
        filters=(result.record.filters or None) if answered else None,
        qual=result.record.qual if answered else None,
        window=None if window is None else _window_cells(window),
        coverage=None if evidence is None else evidence.coverage.status,
        reading=None if reading is None else reading.reading,
        concerns=() if reading is None else reading.concerns,
        reading_note=None if reading is None else reading.note,
    )


def _window_cells(window: VcfWindowQueryResult) -> ItemWindowCells:
    if isinstance(window, VcfWindowObserved):
        return ItemWindowCells(
            flank_bases=window.flank_bases,
            outcome=window.outcome,
            records=tuple(_record_line(record) for record in window.records),
        )
    return ItemWindowCells(
        flank_bases=window.flank_bases,
        outcome=window.outcome,
        refusal=":".join((window.code, *window.violations)),
    )


def _record_line(record: ObservedVcfRecord) -> str:
    """One window record on one line, as written: locus, alleles, quality, calls."""
    parts = [
        f"{record.contig}:{record.one_based_position} "
        f"{record.ref}>{','.join(record.alts)}"
    ]
    if record.qual is not None:
        parts.append(f"QUAL {record.qual}")
    if record.filters:
        parts.append(f"FILTER {';'.join(record.filters)}")
    for call in record.genotypes:
        values = [f"GT {call.genotype}"]
        if call.gq is not None:
            values.append(f"GQ {call.gq}")
        if call.dp is not None:
            values.append(f"DP {call.dp}")
        if call.ad is not None:
            values.append(
                "AD "
                + ",".join("." if depth is None else str(depth) for depth in call.ad)
            )
        parts.append(" ".join(values))
    return " | ".join(parts)


def _target(item: WorkItem) -> str:
    return item.target.variant_text


def _no_work_reason(member: InvestigationRequestMember) -> NoWorkReason:
    """Why a member has no Work, as its state origin (#474)."""
    if member.rejection is not None:
        return _rejected(member.rejection)
    if member.not_checked is not None:
        return _not_checked(member.not_checked)
    return _unresolved(member)


def _not_checked(not_checked) -> RequestNotChecked:
    return RequestNotChecked(
        reason_code=not_checked.reason_code, reason=not_checked.reason
    )


def _unresolved(member: InvestigationRequestMember) -> RequestUnresolved:
    """The source and verdict that left a member without Work, and whether its
    issue was approved."""
    from bio_neb.investigation.investigation import ADMISSIBLE_LOOKUP_STATUSES

    refsnp, assembly = member.refsnp, member.assembly_report
    decision = "approved" if member.issue_approval is not None else None
    if refsnp is not None and refsnp.status not in ADMISSIBLE_LOOKUP_STATUSES:
        return RequestUnresolved(
            source="refsnp", verdict=refsnp.status, decision=decision
        )
    if assembly is not None:
        return RequestUnresolved(
            source="assembly_report", verdict=assembly.status, decision=decision
        )
    return RequestUnresolved(decision=decision)


def _sources(*members: InvestigationRequestMember) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                source.source.source_id
                for member in members
                for source in (member.refsnp, member.assembly_report)
                if source is not None
            }
        )
    )


def _resolved(members) -> list[InvestigationRequestMember]:
    from bio_neb.investigation.investigation import ADMISSIBLE_LOOKUP_STATUSES

    return [
        member
        for member in members
        if member.refsnp is not None
        and member.refsnp.status in ADMISSIBLE_LOOKUP_STATUSES
        and member.refsnp.candidates
    ]


def _record(member: InvestigationRequestMember) -> tuple:
    """A record's normalized targets: equal records present them alike."""
    return tuple(
        sorted(candidate.target_identity for candidate in member.refsnp.candidates)
    )


def _identity(member: InvestigationRequestMember) -> str:
    """A resolved member's identity: its verified rsID, else its normalized target on
    the Run, the Work item's own (`target_identity`), never the request's wording or
    allele presentation (Standards r6 S2, r7 S3)."""
    if member.refsnp.verified_rsid is not None:
        return member.refsnp.verified_rsid
    target = _normalized_target(member)
    return (
        f"{target.contig}:{target.one_based_position} "
        f"{target.ref or '-'}>{target.alt or '-'}"
    )


def _normalized_target(member: InvestigationRequestMember):
    """The member's first candidate as its Work item holds it (#368)."""
    return member.refsnp.candidates[0].target_identity


def _references_disagree(refs) -> bool:
    """Whether REFs that start at one position state different reference bases: each
    is a span from there, so one that begins another (an insertion's empty REF begins
    every one) agrees with it (Spec r8)."""
    spans = sorted(set(refs), key=len)
    return any(
        not longer.startswith(shorter)
        for index, shorter in enumerate(spans)
        for longer in spans[index + 1 :]
    )


def _identity_cases(members) -> list[CollectionCase]:
    cases = []
    for member in members:
        refsnp = member.refsnp
        if (
            refsnp is not None
            and refsnp.verified_rsid is not None
            and (requested := _requested(member)) is not None
            and refsnp.verified_rsid != requested
            and requested in refsnp.merged_rsids
        ):
            cases.append(
                CollectionCase(
                    request_member_ids=(member.request_member_id,),
                    evidence_ids=_sources(member),
                    origin=MergedIdentity(
                        requested=requested, verified=refsnp.verified_rsid
                    ),
                )
            )
    resolved = _resolved(members)
    by_rsid: dict[str, list[InvestigationRequestMember]] = defaultdict(list)
    by_locus: dict[tuple, list[InvestigationRequestMember]] = defaultdict(list)
    for member in resolved:
        candidate = member.refsnp.candidates[0]
        target = candidate.target_identity
        # A position no source lists resolves with no rsID: no identity to share.
        if member.refsnp.verified_rsid is not None:
            by_rsid[member.refsnp.verified_rsid].append(member)
        by_locus[
            (candidate.assembly_accession, target.contig, target.one_based_position)
        ].append(member)
    for rsid, group in by_rsid.items():
        if len(group) < 2:
            continue
        cases.append(
            CollectionCase(
                request_member_ids=_ids(group),
                evidence_ids=_sources(*group),
                origin=SharedIdentity(
                    requests=tuple(_name(member) for member in group), rsid=rsid
                ),
            )
        )
        if len({_record(member) for member in group}) > 1:
            cases.append(
                CollectionCase(
                    request_member_ids=_ids(group),
                    evidence_ids=_sources(*group),
                    origin=ContradictorySources(rsid=rsid),
                )
            )
    for (_assembly, contig, position), group in by_locus.items():
        identities = sorted({_identity(member) for member in group})
        if len(identities) < 2:
            continue
        cases.append(
            CollectionCase(
                request_member_ids=_ids(group),
                evidence_ids=_sources(*group),
                origin=SharedLocus(
                    identities=tuple(identities), contig=contig, position=position
                ),
            )
        )
        if _references_disagree(_normalized_target(member).ref for member in group):
            cases.append(
                CollectionCase(
                    request_member_ids=_ids(group),
                    evidence_ids=_sources(*group),
                    origin=ContradictorySources(contig=contig, position=position),
                )
            )
    return cases


def _member_cases(members, items) -> list[CollectionCase]:
    cases = []
    for member in members:
        if len(member.work_item_ids) > 1 and all(
            items[item].intended_action == "observe_variant"
            for item in member.work_item_ids
        ):
            cases.append(
                CollectionCase(
                    request_member_ids=(member.request_member_id,),
                    work_item_ids=tuple(sorted(member.work_item_ids)),
                    evidence_ids=_sources(member),
                    origin=MultiAlt(
                        request=_name(member),
                        alts=tuple(
                            sorted(
                                items[item].target.alt for item in member.work_item_ids
                            )
                        ),
                    ),
                )
            )
        elif not member.work_item_ids and (sources := _sources(member)):
            cases.append(
                CollectionCase(
                    request_member_ids=(member.request_member_id,),
                    evidence_ids=sources,
                    origin=NoWorkItem(
                        request=_name(member), reason=_no_work_reason(member)
                    ),
                )
            )
    return cases


def _lookup_cases(members, items, observed) -> list[CollectionCase]:
    cases = []
    for member in members:
        looked_up = [
            (item_id, observed[item_id])
            for item_id in sorted(member.work_item_ids)
            if item_id in observed
        ]
        name = _name(member)
        overlapped = [
            (item_id, observation)
            for item_id, observation in looked_up
            if isinstance(observation.result, LocusOverlapped)
            or (
                isinstance(observation.result, _RECORD_ANSWERS)
                and observation.result.other_records
            )
        ]
        if overlapped:
            cases.append(
                _lookup_case(OverlappingRecord(request=name), member, overlapped)
            )
        if len({observation.result.outcome for _, observation in looked_up}) > 1:
            cases.append(
                _lookup_case(MixedAltOutcomes(request=name), member, looked_up)
            )
        trimmed = [
            (item_id, observation)
            for item_id, observation in looked_up
            if isinstance(observation.result, _RECORD_ANSWERS)
            and items[item_id].intended_action == "observe_variant"
            and (
                observation.result.record.one_based_position,
                observation.result.record.ref,
            )
            != (
                items[item_id].target.one_based_position,
                items[item_id].target.ref,
            )
        ]
        if trimmed:
            cases.append(_lookup_case(TrimmedMatch(request=name), member, trimmed))
    return cases


def _lookup_case(origin, member, looked_up) -> CollectionCase:
    return CollectionCase(
        request_member_ids=(member.request_member_id,),
        work_item_ids=tuple(item_id for item_id, _ in looked_up),
        evidence_ids=tuple(
            sorted(
                {
                    *_sources(member),
                    *(observation.observation_id for _, observation in looked_up),
                }
            )
        ),
        origin=origin,
    )


def _requested(member: InvestigationRequestMember) -> str | None:
    """The rsID the request states: typed, or the one its lookup was asked for, which
    the model typed from words that state it only there (Spec r1; #472 R1 b)."""
    return effective_requested_rsid(member.request) or (
        None if member.refsnp is None else member.refsnp.requested_rsid
    )


def _name(member: InvestigationRequestMember) -> str:
    return _requested(member) or member.request.request_text


def _ids(group) -> tuple[str, ...]:
    return tuple(member.request_member_id for member in group)


__all__ = [
    "CollectionCase",
    "CollectionView",
    "ItemState",
    "ItemWindowCells",
    "LookupPlacement",
    "RequestState",
    "RequestedSnp",
    "RequestedSnpItem",
    "VcfTableRow",
    "collection_view",
    "ended_collection_view",
    "vcf_table",
]
