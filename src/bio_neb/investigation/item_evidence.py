"""Each approved Work item's evidence after its deterministic lookup (ADR 0047).

The lookup's ``InvestigationEvidenceRecorded`` alone decides an item's disposition
and admission. After the lookups the coordinator reads every item's fixed window
of the prepared VCF in one model-free batch (``InvestigationItemWindowRecorded``,
one fact per item). The VCF table, one row per item, is read by the table reader
in chunks: each chunk's Attempt binds its rows (``InvestigationTableChunkBound``)
and ends at one reading per item (``InvestigationTableChunkRead``) or its failure
(``InvestigationTableChunkFailed``). A reading reports; it never changes the
lookup. An item's ``coverage`` is filled only by its own
``InvestigationItemCoverageRecorded``, which the Run's measurement step
(``run_health``, #414) records for every item before the windows: its read depth
over its span, or why none was measured. The coordinator folds every item of a collection into one model-free
``InvestigationItemEvidenceCollected`` before Finalization, so each item keeps its
own record whatever its chunk did.

The four facts stay adjustable without another clean cut (owner, #361 Q1): each
names its ``schema_version``, a later field is optional, the concern codes are a
closed set that grows by adding a code, and the chunk size and window are the
Run's runtime selection (``investigation_table``), recorded on every fact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from bio_neb.agentic_journey.continuation_policy import ContinuationPolicy
from bio_neb.agentic_journey.runtime_config import RuntimeSelections
from bio_neb.contracts import FrozenContract, StrictFrozenContract
from bio_neb.investigation.collection_view import VcfTableRow
from bio_neb.investigation.investigation_observation import (
    InvestigationCollectionEvidenceRecorded,
    InvestigationEvidenceRecorded,
    WorkItemDisposition,
)
from bio_neb.investigation.observation_defaults import CoverageDefaults
from bio_neb.investigation.qualification_origins import CoverageNotMeasuredCode
from bio_neb.investigation.vcf_observation import (
    ObserveRegionInput,
    VcfWindowQueryResult,
)
from bio_neb.journey import canonical_json

_WORK_ITEM = r"^work-item:[0-9a-f]{64}$"
_AUTHORITY = r"^investigation-authority:[0-9a-f]{64}$"
_PREPARATION = r"^vcf-observation-preparation:[0-9a-f]{64}$"

TABLE_READER_ROLE = "table_reader"

# What a reading may flag. A closed set: a new concern is a new code, and every
# recorded reading stays readable.
ConcernCode = Literal[
    "low_genotype_quality",
    "low_read_depth",
    "allele_balance",
    "low_call_quality",
    "filter_not_pass",
    "no_call",
    "nearby_variant",
    "overlapping_record",
    "absence_not_established",
    "cross_snp_case",
    "other",
]
CONCERN_CODES: tuple[ConcernCode, ...] = ConcernCode.__args__  # type: ignore[attr-defined]


class InvestigationItemWindowRecorded(FrozenContract):
    """One item's fixed window of the prepared VCF, read with no model (#361)."""

    event_type: Literal["InvestigationItemWindowRecorded"] = (
        "InvestigationItemWindowRecorded"
    )
    schema_version: Literal["bio-neb.item-window/1"] = "bio-neb.item-window/1"
    authority_id: StrictStr = Field(pattern=_AUTHORITY)
    collection_event_id: StrictStr
    work_item_id: StrictStr = Field(pattern=_WORK_ITEM)
    evidence_event_id: StrictStr
    evidence_state_id: StrictStr
    preparation_receipt_id: StrictStr = Field(pattern=_PREPARATION)
    flank_bases: StrictInt = Field(ge=0)
    result: VcfWindowQueryResult

    @model_validator(mode="after")
    def bound_result(self) -> "InvestigationItemWindowRecorded":
        if (
            self.result.preparation_receipt_id != self.preparation_receipt_id
            or self.result.continuation_id != self.authority_id
            or self.result.flank_bases != self.flank_bases
        ):
            raise ValueError("item window changed its bound item or VCF")
        return self


class InvestigationTableChunkBound(FrozenContract):
    """One table reading Attempt's chunk: its items' rows, as the reader saw them."""

    event_type: Literal["InvestigationTableChunkBound"] = "InvestigationTableChunkBound"
    schema_version: Literal["bio-neb.table-chunk-bound/1"] = (
        "bio-neb.table-chunk-bound/1"
    )
    attempt_id: StrictStr
    continued_attempt_id: StrictStr | None = None
    authority_id: StrictStr = Field(pattern=_AUTHORITY)
    collection_event_id: StrictStr
    collection_state_id: StrictStr
    # The Run's selected chunk size; the last chunk may hold fewer rows.
    chunk_size: StrictInt = Field(gt=0)
    rows: tuple[VcfTableRow, ...] = Field(min_length=1)
    runtime_selections: RuntimeSelections
    continuation_policy: ContinuationPolicy
    continuation_policy_id: StrictStr

    @model_validator(mode="after")
    def bounded_chunk(self) -> "InvestigationTableChunkBound":
        ids = self.work_item_ids
        if len(ids) > self.chunk_size or len(set(ids)) != len(ids):
            raise ValueError("table chunk changed its size or repeated an item")
        return self

    @property
    def work_item_ids(self) -> tuple[str, ...]:
        return tuple(row.work_item_id for row in self.rows)


class ItemReading(StrictFrozenContract):
    """One item's reading: whether its row needs a look, why, and a short note."""

    work_item_id: StrictStr = Field(pattern=_WORK_ITEM)
    reading: Literal["no_concern", "concern"]
    concerns: tuple[ConcernCode, ...] = ()
    note: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def coherent(self) -> "ItemReading":
        if (self.reading == "concern") != bool(self.concerns) or len(
            set(self.concerns)
        ) != len(self.concerns):
            raise ValueError("item reading names its concerns only when it has one")
        return self


class InvestigationTableChunkRead(FrozenContract):
    """A chunk's accepted output: one reading per item of its rows, in order."""

    event_type: Literal["InvestigationTableChunkRead"] = "InvestigationTableChunkRead"
    schema_version: Literal["bio-neb.table-chunk-read/1"] = "bio-neb.table-chunk-read/1"
    attempt_id: StrictStr
    validation_event_id: StrictStr
    readings: tuple[ItemReading, ...] = Field(min_length=1)


class InvestigationTableChunkFailed(FrozenContract):
    event_type: Literal["InvestigationTableChunkFailed"] = (
        "InvestigationTableChunkFailed"
    )
    schema_version: Literal["bio-neb.table-chunk-failed/1"] = (
        "bio-neb.table-chunk-failed/1"
    )
    attempt_id: StrictStr
    code: Literal["model_contact_failed", "agent_run_failed"]
    source_event_id: StrictStr
    reason: StrictStr | None = None


class ItemWindowField(FrozenContract):
    """The item's window read; ``not_run`` says why it was never read.

    ``pending`` appears only in a view read before the window batch.
    """

    status: Literal["recorded", "not_run", "pending"]
    window_event_id: StrictStr | None = None
    flank_bases: StrictInt | None = Field(default=None, ge=0)
    outcome: Literal["window_observed", "refused"] | None = None
    # ``region_read``: a region item's own read is its whole region.
    reason: Literal["no_lookup_evidence", "no_prepared_vcf", "region_read"] | None = (
        None
    )

    @model_validator(mode="after")
    def coherent(self) -> "ItemWindowField":
        recorded = self.status == "recorded"
        if (
            recorded != (self.window_event_id is not None)
            or recorded != (self.outcome is not None)
            or recorded != (self.flank_bases is not None)
            or (self.status == "not_run") != (self.reason is not None)
        ):
            raise ValueError("item window field is incoherent")
        return self


class ItemReadingField(FrozenContract):
    """What the item's chunk reading did; ``not_run`` says why it never started.

    ``pending`` (an item not yet read, or a chunk Attempt with no terminal)
    appears only in a view read while the stage is open; a fold never holds it.
    """

    status: Literal["recorded", "stopped", "failed", "not_run", "pending", "dismissed"]
    attempt_id: StrictStr | None = None
    chunk_event_id: StrictStr | None = None
    reading: ItemReading | None = None
    # ``region_read``: a region's records are its rows; no model reads them.
    reason: Literal["no_lookup_evidence", "stage_stopped", "region_read"] | None = None
    # A chunk the researcher dismissed from its stop's Review: the answer and
    # its reason; the other chunks went on.
    dismissal_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    dismissal_reason: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def coherent(self) -> "ItemReadingField":
        recorded = self.status == "recorded"
        dismissed = self.status == "dismissed"
        if (
            recorded != (self.chunk_event_id is not None)
            or recorded != (self.reading is not None)
            or (self.status == "not_run") != (self.reason is not None)
            or (self.status == "not_run" and self.attempt_id is not None)
            or (
                self.status in {"recorded", "stopped", "failed", "dismissed"}
                and self.attempt_id is None
            )
            or dismissed != (self.dismissal_event_id is not None)
            or dismissed != (self.dismissal_reason is not None)
        ):
            raise ValueError("item reading field is incoherent")
        return self


class CoverageMeasurement(FrozenContract):
    """An item's read depth over its span, as the Run's samtools measured it."""

    parameters: CoverageDefaults
    depth: StrictInt = Field(ge=0)
    mean_depth: StrictFloat = Field(ge=0)
    meets_minimum_depth: StrictBool

    @model_validator(mode="after")
    def coherent(self) -> "CoverageMeasurement":
        if (
            self.meets_minimum_depth != (self.depth >= self.parameters.minimum_depth)
            or self.mean_depth < self.depth
        ):
            raise ValueError("coverage measurement is incoherent")
        return self


class CoverageNotMeasured(FrozenContract):
    """Why the Run measured no depth for an item, as its coverage fact records it
    (the report's origin of the same name words it, ``qualification_origins``)."""

    code: CoverageNotMeasuredCode
    detail: StrictStr = Field(min_length=1)


# Why an item's measurement is unsure: below the minimum depth while its span has
# reads (no read at all is a plain absence, not a doubt).
BatchFlag = Literal["low_depth"]


class ItemCoverageField(FrozenContract):
    """Measured only from the item's own coverage fact; otherwise not measured,
    with the fact's reason when one was recorded."""

    status: Literal["not_measured", "measured"] = "not_measured"
    coverage_event_id: StrictStr | None = None
    measurement: CoverageMeasurement | None = None
    not_measured: CoverageNotMeasured | None = None
    batch_flags: tuple[BatchFlag, ...] = ()

    @model_validator(mode="after")
    def coherent(self) -> "ItemCoverageField":
        measured = self.status == "measured"
        if (
            measured != (self.measurement is not None)
            or (measured and self.not_measured is not None)
            or (self.coverage_event_id is None)
            != (self.measurement is None and self.not_measured is None)
            or (self.batch_flags and not measured)
        ):
            raise ValueError("coverage field is incoherent")
        return self


def coverage_span(target) -> tuple[int, int]:
    """The one-based positions an item's coverage is measured over: a region's
    own stretch (#367); a substitution or deletion spans its REF; an insertion
    sits after its anchor (#368)."""
    if isinstance(target, ObserveRegionInput):
        return target.start, target.end
    if not target.ref:
        anchor = max(1, target.one_based_position - 1)
        return anchor, anchor
    return target.one_based_position, target.one_based_position + len(target.ref) - 1


class InvestigationItemCoverageRecorded(FrozenContract):
    """One item's read depth over its span, or why none was measured (#414)."""

    event_type: Literal["InvestigationItemCoverageRecorded"] = (
        "InvestigationItemCoverageRecorded"
    )
    schema_version: Literal["bio-neb.item-coverage/2"] = "bio-neb.item-coverage/2"
    authority_id: StrictStr = Field(pattern=_AUTHORITY)
    collection_event_id: StrictStr
    work_item_id: StrictStr = Field(pattern=_WORK_ITEM)
    contig: StrictStr = Field(min_length=1)
    span_start: StrictInt = Field(gt=0)
    span_end: StrictInt = Field(gt=0)
    measurement: CoverageMeasurement | None = None
    not_measured: CoverageNotMeasured | None = None
    batch_flags: tuple[BatchFlag, ...] = ()

    @model_validator(mode="after")
    def coherent(self) -> "InvestigationItemCoverageRecorded":
        if (
            (self.measurement is None) == (self.not_measured is None)
            or (self.batch_flags and self.measurement is None)
            or len(set(self.batch_flags)) != len(self.batch_flags)
            or self.span_end < self.span_start
        ):
            raise ValueError("item coverage fact is incoherent")
        return self


class ItemEvidence(FrozenContract):
    """One item's own record: its lookup (copied, never derived here), its
    window, its reading and its coverage."""

    work_item_id: StrictStr = Field(pattern=_WORK_ITEM)
    selected_ordinal: StrictInt = Field(gt=0)
    lookup_evidence_state_id: StrictStr | None = None
    lookup_outcome: StrictStr | None = None
    lookup_disposition: WorkItemDisposition | None = None
    window: ItemWindowField
    reading: ItemReadingField
    coverage: ItemCoverageField = ItemCoverageField()


class InvestigationItemEvidenceCollected(FrozenContract):
    """The coordinator's model-free fold of one collection's item evidence."""

    event_type: Literal["InvestigationItemEvidenceCollected"] = (
        "InvestigationItemEvidenceCollected"
    )
    collection_event_id: StrictStr
    collection_state_id: StrictStr
    items: tuple[ItemEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def closed_stage(self) -> "InvestigationItemEvidenceCollected":
        if any(
            item.window.status == "pending" or item.reading.status == "pending"
            for item in self.items
        ):
            raise ValueError("item evidence fold holds an item still pending")
        return self


_FACTS = {
    owner.model_fields["event_type"].default: owner
    for owner in (
        InvestigationItemWindowRecorded,
        InvestigationTableChunkBound,
        InvestigationTableChunkRead,
        InvestigationTableChunkFailed,
        InvestigationItemCoverageRecorded,
        InvestigationItemEvidenceCollected,
    )
}


def validate_item_evidence_fact(event: Mapping[str, Any]) -> FrozenContract | None:
    owner = _FACTS.get(event.get("event_type"))
    if owner is None:
        return None
    return owner.model_validate_json(
        canonical_json({"event_type": event["event_type"], **event.get("payload", {})}),
        strict=True,
    )


_SETTLED = {
    "completed",
    "negative",
    "unresolved",
    "refused",
    "failed",
    "stopped",
    "not_started",
}


# A shared Attempt terminal of a chunk reading that recorded no reading.
_SHARED_TERMINALS: dict[str, Literal["stopped", "failed"]] = {
    "AttemptStopped": "stopped",
    "AttemptCancelled": "stopped",
    "AttemptFailed": "failed",
}
_RUN_ARGUMENTS = {
    "continuation_availability_review_event_id": "availability_review_response_event_id",
    "continuation_effort_review_event_id": "effort_review_response_event_id",
    "continuation_work_review_event_id": "work_review_response_event_id",
    "continuation_operational_review_event_id": "operational_review_response_event_id",
}


@dataclass(frozen=True)
class TableReadingReviews:
    """The table reader's Agent Reviews: pending, stopped, or an unused answer.

    ``answer`` names the Attempt argument that consumes it.
    """

    pending: bool = False
    stopped: bool = False
    answer: Mapping[str, str] | None = None
    # Each dismissed chunk Attempt: its answer's event id and reason.
    dismissed: Mapping[str, tuple[str, str]] = field(default_factory=dict)


def table_reading_reviews(events: Sequence[Mapping[str, Any]]) -> TableReadingReviews:
    """Read the role's Reviews from their typed continuations (no status table)."""
    from bio_neb.agentic_journey.continuation import REVIEW_CONTINUATIONS

    roles = {
        event["payload"]["attempt_id"]: event["payload"]["agent_role"]
        for event in events
        if event["event_type"] == "AttemptStarted"
    }
    consumed = {
        event["payload"].get(kind.field)
        for event in events
        if event["event_type"] == "AttemptStarted"
        for kind in REVIEW_CONTINUATIONS
    }
    pending = stopped = False
    answer = None
    dismissed: dict[str, tuple[str, str]] = {}
    for kind in REVIEW_CONTINUATIONS:
        answers = {
            event["payload"]["request_event_id"]: event
            for event in events
            if event["event_type"] == kind.response_kind
        }
        for request in events:
            if (
                request["event_type"] != kind.request_kind
                or roles.get(request["payload"]["attempt_id"]) != TABLE_READER_ROLE
            ):
                continue
            response = answers.get(request["event_id"])
            if response is None:
                pending = True
            elif response["payload"]["response"]["action"] == "dismiss":
                # Settled alone: the chunk's items fold as dismissed and the
                # other chunks go on.
                dismissed[request["payload"]["attempt_id"]] = (
                    response["event_id"],
                    response["payload"]["response"]["reason"],
                )
            elif response["payload"]["response"]["action"] not in kind.actions:
                stopped = True
            elif response["event_id"] not in consumed:
                answer = {_RUN_ARGUMENTS[kind.field]: response["event_id"]}
    return TableReadingReviews(pending, stopped, answer, dismissed)


def region_lookup(evidence: InvestigationEvidenceRecorded) -> bool:
    """A region item's own read (#367): its records are its rows, unread by a model."""
    return evidence.work_item.intended_action == "observe_region"


def queryable_lookup(evidence: InvestigationEvidenceRecorded) -> bool:
    """A lookup whose item has a prepared VCF its window can read."""
    from bio_neb.investigation.vcf_observation import PreparedVcfObservationReceipt

    return isinstance(
        evidence.sample_observation.preparation, PreparedVcfObservationReceipt
    )


def fold_item_evidence(
    collection_event_id: str,
    collection: InvestigationCollectionEvidenceRecorded,
    timeline: Sequence[Mapping[str, Any]],
) -> tuple[ItemEvidence, ...]:
    """Every settled item of one collection, in its order, with its fields; its
    coverage keyed by the collection's event, the one key its facts name."""
    from bio_neb.investigation.investigation_observation import (
        validate_investigation_evidence_fact,
    )

    lookups: dict[str, InvestigationEvidenceRecorded] = {}
    windows: dict[str, tuple[str, InvestigationItemWindowRecorded]] = {}
    bound: dict[str, InvestigationTableChunkBound] = {}
    read: dict[str, tuple[str, InvestigationTableChunkRead]] = {}
    ended: dict[str, Literal["stopped", "failed"]] = {}
    coverage: dict[str, tuple[str, InvestigationItemCoverageRecorded]] = {}
    for event in timeline:
        if (lookup := validate_investigation_evidence_fact(event)) is not None:
            lookups[lookup.evidence_state.evidence_state_id] = lookup
        fact = validate_item_evidence_fact(event)
        if isinstance(fact, InvestigationItemWindowRecorded):
            windows[fact.work_item_id] = (event["event_id"], fact)
        elif isinstance(fact, InvestigationTableChunkBound):
            # A continued chunk binds the same items again; the latest holds them.
            for work_item_id in fact.work_item_ids:
                bound[work_item_id] = fact
        elif isinstance(fact, InvestigationTableChunkRead):
            read[fact.attempt_id] = (event["event_id"], fact)
        elif isinstance(fact, InvestigationTableChunkFailed):
            ended[fact.attempt_id] = "failed"
        elif (
            isinstance(fact, InvestigationItemCoverageRecorded)
            and fact.collection_event_id == collection_event_id
        ):
            coverage[fact.work_item_id] = (event["event_id"], fact)
        elif event.get("event_type") in _SHARED_TERMINALS:
            ended[event["payload"]["attempt_id"]] = _SHARED_TERMINALS[
                event["event_type"]
            ]
    reviews = table_reading_reviews(timeline)
    stopped = reviews.stopped
    items: list[ItemEvidence] = []
    for state in collection.item_states:
        if state.status not in _SETTLED:
            continue
        lookup = (
            lookups.get(state.evidence_state_id) if state.evidence_state_id else None
        )
        items.append(
            ItemEvidence(
                work_item_id=state.work_item_id,
                selected_ordinal=state.selected_ordinal,
                lookup_evidence_state_id=state.evidence_state_id,
                lookup_outcome=state.outcome,
                lookup_disposition=None
                if lookup is None
                else lookup.evidence_state.work_item_disposition,
                window=_window_field(lookup, windows.get(state.work_item_id)),
                reading=_reading_field(
                    state.work_item_id,
                    lookup,
                    bound.get(state.work_item_id),
                    read,
                    ended,
                    stopped,
                    reviews.dismissed,
                ),
                coverage=_coverage_field(coverage.get(state.work_item_id)),
            )
        )
    return tuple(items)


def _coverage_field(recorded) -> ItemCoverageField:
    if recorded is None:
        return ItemCoverageField()
    event_id, fact = recorded
    return ItemCoverageField(
        status="not_measured" if fact.measurement is None else "measured",
        coverage_event_id=event_id,
        measurement=fact.measurement,
        not_measured=fact.not_measured,
        batch_flags=fact.batch_flags,
    )


def _window_field(lookup, window) -> ItemWindowField:
    if window is not None:
        event_id, fact = window
        return ItemWindowField(
            status="recorded",
            window_event_id=event_id,
            flank_bases=fact.flank_bases,
            outcome=fact.result.outcome,
        )
    if lookup is None:
        return ItemWindowField(status="not_run", reason="no_lookup_evidence")
    if region_lookup(lookup):
        return ItemWindowField(status="not_run", reason="region_read")
    if not queryable_lookup(lookup):
        return ItemWindowField(status="not_run", reason="no_prepared_vcf")
    return ItemWindowField(status="pending")


def _reading_field(
    work_item_id, lookup, binding, read, ended, stopped, dismissed=None
) -> ItemReadingField:
    if binding is not None:
        attempt_id = binding.attempt_id
        if attempt_id in read:
            # Recorded only once its chunk's Attempt ended at its reading.
            event_id, chunk = read[attempt_id]
            return ItemReadingField(
                status="recorded",
                attempt_id=attempt_id,
                chunk_event_id=event_id,
                reading=next(
                    item for item in chunk.readings if item.work_item_id == work_item_id
                ),
            )
        if attempt_id in (dismissed or {}):
            event_id, reason = dismissed[attempt_id]
            return ItemReadingField(
                status="dismissed",
                attempt_id=attempt_id,
                dismissal_event_id=event_id,
                dismissal_reason=reason,
            )
        # A chunk Attempt with no terminal yet is still running.
        return ItemReadingField(
            status=ended.get(attempt_id, "pending"), attempt_id=attempt_id
        )
    if lookup is None:
        return ItemReadingField(status="not_run", reason="no_lookup_evidence")
    if region_lookup(lookup):
        return ItemReadingField(status="not_run", reason="region_read")
    if stopped:
        # The role's stop or reject answer holds the items not yet read.
        return ItemReadingField(status="not_run", reason="stage_stopped")
    return ItemReadingField(status="pending")


def checked_item_evidence_folds(
    timeline: Sequence[Mapping[str, Any]],
    collections: Mapping[str, InvestigationCollectionEvidenceRecorded],
    *,
    checked_once=None,
) -> dict[str, InvestigationItemEvidenceCollected]:
    """Each collection's one fold, checked against the items it accounts.

    A fold's check reads only the prefix before it, which its admission fixes:
    ``checked_once``, the reading owner's, keeps it across the owner's appends
    (#437, read (b)).
    """
    folds: dict[str, InvestigationItemEvidenceCollected] = {}
    for index, event in enumerate(timeline):
        fact = validate_item_evidence_fact(event)
        if not isinstance(fact, InvestigationItemEvidenceCollected):
            continue
        collection = collections.get(fact.collection_event_id)
        if (
            collection is None
            or fact.collection_event_id in folds
            or collection.collection_state_id != fact.collection_state_id
        ):
            raise ValueError("item evidence fold names no exact collection")

        def check(index=index, fact=fact, collection=collection) -> None:
            prefix = timeline[:index]
            reviews = table_reading_reviews(prefix)
            # The stage closed: no Review of the role pending or answered unused,
            # and every item accounted (the fact refuses a pending item).
            if reviews.pending or reviews.answer is not None:
                raise ValueError("item evidence fold precedes its stage's Review")
            if fact.items != fold_item_evidence(
                fact.collection_event_id, collection, prefix
            ):
                raise ValueError("item evidence fold changed its items")

        if checked_once is None:
            check()
        else:
            checked_once(("item_evidence_fold", event["event_id"]), check)
        folds[fact.collection_event_id] = fact
    return folds


__all__ = [
    "CONCERN_CODES",
    "TABLE_READER_ROLE",
    "ConcernCode",
    "BatchFlag",
    "CoverageMeasurement",
    "CoverageNotMeasured",
    "InvestigationItemCoverageRecorded",
    "InvestigationItemEvidenceCollected",
    "InvestigationItemWindowRecorded",
    "InvestigationTableChunkBound",
    "InvestigationTableChunkFailed",
    "InvestigationTableChunkRead",
    "ItemCoverageField",
    "ItemEvidence",
    "ItemReading",
    "ItemReadingField",
    "ItemWindowField",
    "TableReadingReviews",
    "checked_item_evidence_folds",
    "coverage_span",
    "fold_item_evidence",
    "queryable_lookup",
    "table_reading_reviews",
    "validate_item_evidence_fact",
]
