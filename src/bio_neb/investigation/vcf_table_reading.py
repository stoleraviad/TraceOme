"""The post-approval VCF table stage: one window batch, then chunked readings (ADR 0047).

Each approved item's deterministic lookup has already decided its disposition
and admission. The coordinator then reads every item's fixed window of the Run's
prepared VCF in one model-free batch, which completes the VCF table: one row per
item with its call, call quality, window and coverage, built on the collection
view. The table reader reads the table in chunks of the Run's selected size: one
Attempt per chunk binds its rows, offers no VCF Tool, and ends at one typed
reading per row. A reading never changes a lookup. When no item is left to read,
the coordinator records the model-free fold of the collection's item evidence;
Finalization compiles from it.

Chunks run one at a time in selected order; a stopped chunk holds the remaining
items, which then fold as not run (owner, #344 F3).
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, StrictInt, StrictStr, model_validator
from pydantic_core import PydanticCustomError

from bio_neb.agentic_journey.adapters import (
    NebiusModelEndpointFactory,
    create_nebius_model_endpoint,
)
from bio_neb.agentic_journey.agent_roles import (
    AttemptAdmission,
    AttemptAssignment,
    AttemptBinding,
    AttemptContext,
    AttemptReopen,
    AttemptTerminalReading,
    BoundInvocation,
    DomainBoundReopen,
    Reentry,
)
from bio_neb.agentic_journey.agent_work_review import AgentResultMissing
from bio_neb.agentic_journey.continuation import review_direction
from bio_neb.agentic_journey.continuation_policy import validate_continuation
from bio_neb.agentic_journey.failure_cause import cause_identity
from bio_neb.agentic_journey.invocation import (
    ActionAdmissionSpec,
    AssignmentOutput,
    OutputRepair,
    ToolContext,
    assemble_agent,
)
from bio_neb.agentic_journey.researcher_question import researcher_question_row
from bio_neb.agentic_journey.role_declaration import role
from bio_neb.agentic_journey.runtime_config import RuntimeSelections
from bio_neb.agentic_journey.structured_output import (
    StructuredOutputSession,
    project_structured_output_lineages,
)
from bio_neb.application.input_journey import RunInputsRegistered, validate_fact
from bio_neb.contracts import StrictFrozenContract
from bio_neb.investigation.collection_view import VcfTableRow, vcf_table
from bio_neb.investigation.investigation_observation import (
    InvestigationEvidenceRecorded,
    observation_target,
    validate_investigation_collection_evidence_fact,
    validate_investigation_evidence_fact,
    validate_investigation_work_item_started_fact,
)
from bio_neb.investigation.item_evidence import (
    TABLE_READER_ROLE,
    ConcernCode,
    InvestigationItemCoverageRecorded,
    InvestigationItemEvidenceCollected,
    InvestigationItemWindowRecorded,
    InvestigationTableChunkBound,
    InvestigationTableChunkFailed,
    InvestigationTableChunkRead,
    ItemReading,
    coverage_span,
    fold_item_evidence,
    queryable_lookup,
    table_reading_reviews,
    validate_item_evidence_fact,
)
from bio_neb.investigation.local_bcftools import LocalBcftoolsAdapter
from bio_neb.investigation.observation_defaults import (
    VcfTableDefaults,
    selected_concurrency,
)
from bio_neb.investigation.vcf_observation import (
    DeterministicVcfObservationTool,
    PreparedVcfObservationReceipt,
    VcfProcessAdapter,
    VcfWindowQuery,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.recording import read_evidence

if TYPE_CHECKING:
    from bio_neb.investigation.investigation import InvestigationProjection

ROLE = TABLE_READER_ROLE
STAGE = "vcf_table_reading"
OUTPUT_TOOL = "record_item_readings"
_SHARED_TERMINALS = {
    "AttemptStopped": "stopped",
    "AttemptFailed": "failed",
    "AttemptCancelled": "cancelled",
}


class ItemReadingDraft(StrictFrozenContract):
    """One row's reading as the model writes it; the server binds its item."""

    row: StrictInt = Field(ge=1, description="The row number this reading is for.")
    reading: Literal["no_concern", "concern"]
    concerns: tuple[ConcernCode, ...] = Field(
        default=(),
        description="Each reason the row needs a look; empty when no_concern.",
    )
    note: StrictStr = Field(
        min_length=1,
        max_length=400,
        description="One or two sentences on what the row shows.",
    )

    @model_validator(mode="after")
    def coherent(self) -> "ItemReadingDraft":
        """The recorded reading's own rules, each refused with its own type, so
        an output the reading would refuse is repaired, never accepted."""
        if (self.reading == "concern") != bool(self.concerns):
            raise PydanticCustomError(
                "reading_concerns_mismatch",
                "a concern names at least one concern code; no_concern names none",
            )
        if len(set(self.concerns)) != len(self.concerns):
            raise PydanticCustomError(
                "reading_concerns_repeated", "each concern code is named once"
            )
        return self


class TableReadingDraft(StrictFrozenContract):
    """One reading per row of the chunk; a missing row is refused by its code."""

    readings: tuple[ItemReadingDraft, ...]


@dataclass(frozen=True)
class ChunkWork:
    """One chunk's table rows, its collection and the Run's runtime selections."""

    authority_id: str
    collection_event_id: str
    collection_state_id: str
    chunk_size: int
    rows: tuple[VcfTableRow, ...]
    runtime_selections: RuntimeSelections
    continued_attempt_id: str | None = None
    researcher_instruction: str | None = None

    @property
    def member_id(self) -> str:
        """The chunk's output member: its first item, unique across chunks."""
        return "vcf-table-chunk:" + self.rows[0].work_item_id


@dataclass
class ChunkReadingDependencies:
    """One chunk's state; nothing is shared with another chunk's Attempt."""

    journey: Journey
    attempt_id: str
    work: ChunkWork
    outputs: StructuredOutputSession
    observed_tool_session: Any
    validation_event_id: str | None = None
    readings: tuple[ItemReading, ...] = field(default=())


def chunk_readings(
    rows: tuple[VcfTableRow, ...], draft: TableReadingDraft
) -> tuple[ItemReading, ...]:
    """Each row's reading bound to its item, in row order; every row exactly once."""
    numbers = [item.row for item in draft.readings]
    expected = range(1, len(rows) + 1)
    missing = [str(row) for row in expected if row not in numbers]
    repeated = sorted({str(row) for row in numbers if numbers.count(row) > 1})
    unknown = sorted({str(row) for row in numbers if row > len(rows)})
    errors = tuple(
        f"{code}:{','.join(values)}"
        for code, values in (
            ("chunk_rows_missing", missing),
            ("chunk_rows_repeated", repeated),
            ("chunk_rows_unknown", unknown),
        )
        if values
    )
    if errors:
        raise ChunkReadingRefused(errors)
    by_row = {item.row: item for item in draft.readings}
    return tuple(
        ItemReading(
            work_item_id=row.work_item_id,
            reading=by_row[number].reading,
            concerns=by_row[number].concerns,
            note=by_row[number].note,
        )
        for number, row in enumerate(rows, start=1)
    )


class ChunkReadingRefused(ValueError):
    """A chunk output that misses, repeats or invents a row; each code names them."""

    def __init__(self, errors: tuple[str, ...]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _validate_readings(
    context: ToolContext[ChunkReadingDependencies], output: TableReadingDraft
) -> TableReadingDraft:
    deps = context.deps
    try:
        readings = chunk_readings(deps.work.rows, output)
    except ChunkReadingRefused as error:
        # The row range lets an unknown row be told where the chunk ends.
        message = (
            f"This chunk has rows 1 to {len(deps.work.rows)}; record exactly one "
            "reading for each row: " + "; ".join(error.errors)
        )
        deps.outputs.validate(output.model_dump(mode="json"), error.errors, message)
        raise OutputRepair(message) from error
    valid = deps.outputs.validate(output.model_dump(mode="json"), ())
    deps.validation_event_id = valid["event_id"]
    deps.readings = readings
    return output


TABLE_READING_OUTPUT = AssignmentOutput(
    output_type=TableReadingDraft,
    name=OUTPUT_TOOL,
    description=(
        "Record one reading for every row of this chunk of the VCF table: "
        "no_concern or concern, its concern codes and a short note."
    ),
    validator=_validate_readings,
)


def _retained_registration(events) -> RunInputsRegistered:
    registrations = [
        validate_fact(event)
        for event in events
        if event.get("event_type") == "RunInputsRegistered"
    ]
    if len(registrations) != 1 or not isinstance(registrations[0], RunInputsRegistered):
        raise ValueError("the VCF table requires the Run's one registration")
    return registrations[0]


def _stage_preparation(events, stage) -> PreparedVcfObservationReceipt | None:
    """The collection's shared VCF preparation, as its lookups recorded it; None
    when it failed."""
    return next(
        (
            preparation
            for _, fact in _lookups(events, stage).values()
            if isinstance(
                preparation := fact.sample_observation.preparation,
                PreparedVcfObservationReceipt,
            )
        ),
        None,
    )


def _lookups(events, stage) -> dict[str, tuple[dict, InvestigationEvidenceRecorded]]:
    """Each settled item's lookup event and fact, by Work item, from retained facts."""
    states = {
        state.evidence_state_id: state.work_item_id
        for state in stage.collection.item_states
        if state.evidence_state_id is not None
    }
    return {
        states[fact.evidence_state.evidence_state_id]: (event, fact)
        for event in events
        if (fact := validate_investigation_evidence_fact(event)) is not None
        and fact.evidence_state.evidence_state_id in states
    }


def _windows(events, collection_event_id: str) -> dict[str, Any]:
    return {
        fact.work_item_id: fact.result
        for event in events
        if isinstance(
            fact := validate_item_evidence_fact(event), InvestigationItemWindowRecorded
        )
        and fact.collection_event_id == collection_event_id
    }


def table_rows(events, stage) -> dict[str, VcfTableRow]:
    """The collection's VCF table from its proposal, lookups and windows."""
    return vcf_table(
        stage.proposal,
        tuple(
            fact.sample_observation for _event, fact in _lookups(events, stage).values()
        ),
        _windows(events, stage.collection_event_id),
    )


def run_table_settings(events) -> VcfTableDefaults:
    """The Run's chunk size and window flank, fixed once used.

    A Run whose runtime names ``investigation_table`` uses it. Otherwise the
    named defaults apply, and the first window and the first chunk the Run
    records freeze the values they used, so a later change of a default never
    changes or refuses the Run's retained facts (#361 Spec r1).
    """
    selected = _retained_registration(events).runtime_selections.investigation_table
    if selected is not None:
        return selected
    default = VcfTableDefaults()
    facts = [validate_item_evidence_fact(event) for event in events]
    flank = next(
        (
            f.flank_bases
            for f in facts
            if isinstance(f, InvestigationItemWindowRecorded)
        ),
        default.window_flank_bases,
    )
    size = next(
        (f.chunk_size for f in facts if isinstance(f, InvestigationTableChunkBound)),
        default.chunk_size,
    )
    return VcfTableDefaults(chunk_size=size, window_flank_bases=flank)


def record_item_windows(
    journey: Journey, stage, adapter: VcfProcessAdapter | None = None
) -> None:
    """Read every pending item's window in one model-free batch, one fact each."""
    events = read_evidence(journey)
    flank = run_table_settings(events).window_flank_bases
    lookups = _lookups(events, stage)
    items = {item.work_item_id: item for item in stage.proposal.work_items}
    tool = DeterministicVcfObservationTool()
    adapter = adapter or LocalBcftoolsAdapter()
    for work_item_id in stage.windows_pending:
        event, evidence = lookups[work_item_id]
        observation = evidence.sample_observation
        preparation = observation.preparation
        assert isinstance(preparation, PreparedVcfObservationReceipt)
        result = tool.query_window(
            observation_target(items[work_item_id]),
            VcfWindowQuery(flank_bases=flank),
            SimpleNamespace(
                continuation_id=observation.investigation_authority_id,
                journey_id=journey.journey_id,
                reference=SimpleNamespace(
                    assembly_accession=observation.assembly_accession
                ),
                sample=SimpleNamespace(sample_id=observation.sample_id),
                vcf=observation.vcf_descriptor,
            ),
            preparation,
            adapter,
        )
        fact = InvestigationItemWindowRecorded(
            authority_id=observation.investigation_authority_id,
            collection_event_id=stage.collection_event_id,
            work_item_id=work_item_id,
            evidence_event_id=event["event_id"],
            evidence_state_id=evidence.evidence_state.evidence_state_id,
            preparation_receipt_id=preparation.receipt_id,
            flank_bases=flank,
            result=result,
        )
        journey.record(
            fact.event_type,
            fact.model_dump(mode="json", exclude={"event_type"}),
            [journey.head_event_id],
        )


@dataclass(frozen=True)
class AdmittedChunk:
    work: ChunkWork


def _chunk_work(
    journey, work_item_ids, *, continued_attempt_id=None, researcher_instruction=None
) -> ChunkWork:
    from bio_neb.application.queries import read_table_reading_stage

    stage = read_table_reading_stage(journey)
    if stage is None:
        raise ValueError("VCF table stage is not open")
    events = read_evidence(journey)
    rows = table_rows(events, stage)
    if not work_item_ids or any(item not in rows for item in work_item_ids):
        raise ValueError("table chunk is not eligible")
    selections = _retained_registration(events).runtime_selections
    return ChunkWork(
        authority_id=stage.collection.authority.authority_id,
        collection_event_id=stage.collection_event_id,
        collection_state_id=stage.collection.collection_state_id,
        chunk_size=run_table_settings(events).chunk_size,
        rows=tuple(rows[item] for item in work_item_ids),
        runtime_selections=selections,
        continued_attempt_id=continued_attempt_id,
        researcher_instruction=researcher_instruction,
    )


def next_chunk(journey: Journey) -> tuple[str, ...]:
    """The next unread items, in selected order, up to the Run's chunk size."""
    return next_chunks(journey, 1)[0]


def next_chunks(journey: Journey, count: int) -> tuple[tuple[str, ...], ...]:
    """Up to ``count`` disjoint next chunks of unread items, in selected order.

    A batch reserves its chunks here, before any of them binds, so concurrent
    chunk Attempts never read the same row (#105).
    """
    from bio_neb.application.queries import read_table_reading_stage

    stage = read_table_reading_stage(journey)
    if stage is None or not stage.unread or count < 1:
        raise ValueError("table chunk is not eligible")
    size = run_table_settings(read_evidence(journey)).chunk_size
    return tuple(
        stage.unread[start : start + size]
        for start in range(0, len(stage.unread), size)
    )[:count]


def _bound_chunk(events, attempt_id: str) -> InvestigationTableChunkBound:
    bound = [
        fact
        for event in events
        if isinstance(
            fact := validate_item_evidence_fact(event), InvestigationTableChunkBound
        )
        and fact.attempt_id == attempt_id
    ]
    if len(bound) != 1:
        raise ValueError("table reading Attempt lacks its exact chunk")
    return bound[0]


def _admit(
    journey: Journey, reentry: Reentry, work_item_ids: tuple[str, ...] | None = None
) -> AttemptAdmission[AdmittedChunk]:
    from bio_neb.agentic_journey.agent_roles import RuntimeBinding

    events = read_evidence(journey)
    direction = review_direction(journey, reentry, agent_role=ROLE)
    if direction.previous_attempt_id is None:
        work = _chunk_work(journey, work_item_ids or next_chunk(journey))
    else:
        previous = _bound_chunk(events, direction.previous_attempt_id)
        work = _chunk_work(
            journey,
            previous.work_item_ids,
            continued_attempt_id=direction.previous_attempt_id,
            researcher_instruction=direction.instruction,
        )
    registration = _retained_registration(events)
    return AttemptAdmission(
        AdmittedChunk(work),
        RuntimeBinding(registration.runtime_selections, registration.input_set_id),
        direction.causal_head,
        direction.availability_frontier,
        # Each chunk's stops hold its own rows' work, so answering one chunk's
        # Review re-enters it beside a sibling's pending Review.
        chunk_scope(tuple(row.work_item_id for row in work.rows)),
    )


def chunk_scope(work_item_ids: tuple[str, ...]) -> str:
    """The stable hold scope of a chunk: its first row's Work item."""
    return f"table-chunk:{work_item_ids[0]}"


def _prompt_rows(rows: tuple[VcfTableRow, ...]) -> list[dict[str, Any]]:
    """The rows as the model reads them: numbered, with no identity to restate."""
    return [
        {
            "row": number,
            **row.model_dump(mode="json", exclude={"table_version", "work_item_id"}),
        }
        for number, row in enumerate(rows, start=1)
    ]


async def _drive(context, invocation: BoundInvocation, invoke_member, message_history):
    deps = invocation.deps
    deps.outputs.begin_member(deps.work.member_id)
    await invoke_member(
        invocation.agent,
        None if message_history is not None else invocation.prompt(),
        deps=deps,
        member_id=deps.work.member_id,
        message_history=message_history,
    )
    if deps.validation_event_id is None:
        raise AgentResultMissing("model ended without the chunk's readings")
    return deps


def _complete(context: AttemptContext[AdmittedChunk], deps) -> None:
    if deps is None or deps.validation_event_id is None:
        raise RuntimeError("table reading Attempt completed without its readings")
    terminal = context.activity.record(
        InvestigationTableChunkRead(
            attempt_id=context.attempt_id,
            validation_event_id=deps.validation_event_id,
            readings=deps.readings,
        )
    )
    context.activity.succeeded(terminal)


def _fail(context: AttemptContext[AdmittedChunk], invocation, error) -> bool:
    """Retain the chunk's failure; the lookups and the other chunks are untouched."""
    source = context.failure_source
    model_failed = source.event_type == "ModelRequestFailed"
    source_event_id = (
        source.event_id
        if model_failed
        else next(
            event["event_id"]
            for event in reversed(read_evidence(context.journey))
            if event.get("payload", {}).get("attempt_id") == context.attempt_id
            and event["event_type"]
            in {
                "ModelResponseRecorded",
                "InvestigationTableChunkBound",
                "AttemptStarted",
            }
        )
    )
    failed = context.activity.record(
        InvestigationTableChunkFailed(
            attempt_id=context.attempt_id,
            code="model_contact_failed" if model_failed else "agent_run_failed",
            source_event_id=source_event_id,
            reason=None if model_failed else cause_identity(error),
        )
    )
    context.activity.failed(failed)
    return True


def _window_attempt_id(selected) -> str:
    return next(
        event["payload"]["attempt_id"]
        for event in selected
        if event["event_type"] == "AttemptStarted"
    )


def _attempt_binding(timeline, selected) -> AttemptBinding:
    bound = _bound_chunk(selected, _window_attempt_id(selected))
    return AttemptBinding(
        _retained_registration(timeline).input_set_id, bound.runtime_selections
    )


def _attempt_terminal(snapshot, selected) -> AttemptTerminalReading:
    attempt_id = _window_attempt_id(selected)
    terminals = [
        event
        for event in selected
        if event["payload"].get("attempt_id") == attempt_id
        and event["event_type"]
        in {
            *_SHARED_TERMINALS,
            "InvestigationTableChunkRead",
            "InvestigationTableChunkFailed",
        }
    ]
    if len(terminals) != 1:
        raise ValueError("table reading does not bind this Attempt's exact terminal")
    terminal = terminals[0]
    outcome = {
        **_SHARED_TERMINALS,
        "InvestigationTableChunkRead": "succeeded",
        "InvestigationTableChunkFailed": "failed",
    }[terminal["event_type"]]
    return AttemptTerminalReading(outcome, terminal["event_id"])


def validate_table_chunk_facts(timeline, selected) -> None:
    """A chunk's readings are its accepted output, bound to its own rows."""
    attempt_id = _window_attempt_id(selected)
    bound = _bound_chunk(selected, attempt_id)
    (started,) = [e for e in selected if e["event_type"] == "AttemptStarted"]
    if started["payload"].get("member_id") not in {
        None,
        chunk_scope(tuple(row.work_item_id for row in bound.rows)),
    }:
        # A chunk's hold scope is its own rows'; an older start names none.
        raise ValueError("table reading Attempt changed its hold scope")
    reads = [
        (event, fact)
        for event in selected
        if isinstance(
            fact := validate_item_evidence_fact(event), InvestigationTableChunkRead
        )
    ]
    if len(reads) > 1:
        raise ValueError("table reading Attempt recorded more than one reading")
    lineages = project_structured_output_lineages(
        timeline,
        attempt_id=attempt_id,
        agent_role=ROLE,
        tool_name=OUTPUT_TOOL,
        max_retries=bound.continuation_policy.most_failed_turns,
    )
    member = "vcf-table-chunk:" + bound.rows[0].work_item_id
    if any(lineage.collection_member_id != member for lineage in lineages):
        raise ValueError("table reading output names another chunk")
    for _event, read in reads:
        accepted = [
            attempt
            for lineage in lineages
            for attempt in lineage.validation_attempts
            if attempt.status == "valid"
        ]
        if (
            len(accepted) != 1
            or accepted[0].validation_event_id != read.validation_event_id
            or read.readings
            != chunk_readings(
                bound.rows,
                TableReadingDraft.model_validate_json(
                    canonical_json(accepted[0].output), strict=True
                ),
            )
        ):
            raise ValueError("table reading changed its accepted output")


def validate_table_reading(
    timeline, investigation: InvestigationProjection, *, checked_once=None
) -> None:
    """Check every window and chunk against the retained facts it rests on.

    A window reads its own item's lookup and the Run's prepared VCF with the
    Run's selected flank, once per collection item; a chunk's rows are the VCF
    table those facts give at its binding, within the selected chunk size; each
    chunk Attempt's readings are its accepted output. The checked Investigation
    read runs this with the ``investigation`` it holds for ``timeline``, so
    Finalization reads no item evidence its lineage refuses. A window's check and
    a chunk's rows are fixed by their admitted events: ``checked_once``, the
    reading owner's, keeps each across the owner's appends.
    """

    def once(key, check):
        return check() if checked_once is None else checked_once(key, check)

    windows: set[tuple[str, str]] = set()
    chunks: list[tuple[int, InvestigationTableChunkBound]] = []
    flank = None
    retained: set[str] | None = None
    _validate_measurements(timeline)
    for index, event in enumerate(timeline):
        fact = validate_item_evidence_fact(event)
        if isinstance(fact, InvestigationItemWindowRecorded):
            key = (fact.collection_event_id, fact.work_item_id)
            if key in windows:
                raise ValueError("item window repeated its item")
            windows.add(key)
            if flank is None:
                flank = run_table_settings(timeline).window_flank_bases
                retained = {item["event_id"] for item in timeline}
            if fact.evidence_event_id in retained:
                # Fixed by the window, the lookup it binds and the Run's flank.
                once(
                    ("table_window", event["event_id"], flank),
                    lambda fact=fact: _validate_window(timeline, fact),
                )
            else:
                # A prefix without the lookup refuses afresh, whatever was held.
                _validate_window(timeline, fact)
        elif isinstance(fact, InvestigationTableChunkBound):
            chunks.append((index, fact))
    if not chunks:
        return
    settings = run_table_settings(timeline)
    proposals = {proposal.proposal_id: proposal for proposal in investigation.proposals}
    for index, chunk in chunks:
        if chunk.chunk_size != settings.chunk_size:
            raise ValueError("table chunk changed the Run's chunk size")
        prefix = timeline[:index]
        stage = _historical_stage(prefix, chunk, proposals)
        if chunk.authority_id != stage.collection.authority.authority_id:
            raise ValueError("table chunk changed its collection authority")
        rows = once(
            ("table_rows", timeline[index]["event_id"]),
            lambda prefix=prefix, stage=stage: table_rows(prefix, stage),
        )
        if chunk.rows != tuple(rows.get(row.work_item_id) for row in chunk.rows):
            raise ValueError("table chunk changed its VCF table rows")
    validate_table_reading_attempts(timeline)


def _historical_stage(prefix, chunk, proposals):
    from bio_neb.investigation.investigation_observation import (
        validate_investigation_collection_evidence_fact,
    )

    collection = next(
        (
            fact
            for event in prefix
            if event["event_id"] == chunk.collection_event_id
            and (fact := validate_investigation_collection_evidence_fact(event))
            is not None
        ),
        None,
    )
    if collection is None or collection.collection_state_id != (
        chunk.collection_state_id
    ):
        raise ValueError("table chunk names no exact collection")
    proposal = proposals.get(
        collection.collection_proposal_id or collection.authority.proposal_id
    )
    if proposal is None:
        raise ValueError("table chunk names no collection proposal")
    return SimpleNamespace(
        collection=collection,
        collection_event_id=chunk.collection_event_id,
        proposal=proposal,
    )


def _validate_measurements(timeline) -> None:
    """Each measurement fact rests on its own collection, before its fold.

    The Run's health is measured once (``project_run_health`` checks its Review
    lineage), at a recorded collection. A coverage fact's contig and span are its
    item's target's (#368), its authority the collection's, and an item is
    covered once per collection.
    """
    from bio_neb.investigation.run_health import (
        RunHealthMeasured,
        project_run_health,
        validate_run_health_fact,
    )

    project_run_health(timeline)
    collections: dict[str, Any] = {}
    targets: dict[str, Any] = {}
    folded: set[str] = set()
    covered: set[tuple[str, str]] = set()
    for event in timeline:
        health = validate_run_health_fact(event)
        if isinstance(health, RunHealthMeasured) and (
            health.collection_event_id not in collections
            or health.collection_event_id in folded
        ):
            raise ValueError("run health names no open collection")
        if collection := validate_investigation_collection_evidence_fact(event):
            collections[event["event_id"]] = collection
        elif started := validate_investigation_work_item_started_fact(event):
            targets[started.work_item_id] = started.target
        fact = validate_item_evidence_fact(event)
        if isinstance(fact, InvestigationItemEvidenceCollected):
            folded.add(fact.collection_event_id)
        if not isinstance(fact, InvestigationItemCoverageRecorded):
            continue
        collection = collections.get(fact.collection_event_id)
        if collection is None:
            raise ValueError("item coverage names no exact collection")
        if fact.collection_event_id in folded:
            raise ValueError("item coverage follows its fold")
        if fact.authority_id != collection.authority.authority_id:
            raise ValueError("item coverage changed its collection authority")
        target = targets.get(fact.work_item_id)
        if target is None or fact.work_item_id not in {
            item.work_item_id for item in collection.item_states
        }:
            raise ValueError("item coverage names no collection item")
        if (fact.contig, (fact.span_start, fact.span_end)) != (
            target.contig,
            coverage_span(target),
        ):
            raise ValueError("item coverage changed its item's span")
        key = (fact.collection_event_id, fact.work_item_id)
        if key in covered:
            raise ValueError("item coverage repeated its item")
        covered.add(key)


def _validate_window(timeline, window: InvestigationItemWindowRecorded) -> None:
    lookup_event = next(
        (event for event in timeline if event["event_id"] == window.evidence_event_id),
        None,
    )
    evidence = (
        None
        if lookup_event is None
        else validate_investigation_evidence_fact(lookup_event)
    )
    if evidence is None:
        raise ValueError("item window binds no retained lookup")
    observation = evidence.sample_observation
    preparation = observation.preparation
    flank = run_table_settings(timeline).window_flank_bases
    if (
        evidence.evidence_state.evidence_state_id != window.evidence_state_id
        or observation.work_item_id != window.work_item_id
        or observation.investigation_authority_id != window.authority_id
        or not isinstance(preparation, PreparedVcfObservationReceipt)
        or preparation.receipt_id != window.preparation_receipt_id
        or window.flank_bases != flank
        or window.result.target != observation.result.target
    ):
        raise ValueError("item window changed its lookup")
    invocation = window.result.query_invocation
    if invocation is not None and (
        invocation.argv[0] != preparation.runtime.executable_path
        or invocation.argv[-1] != preparation.derived_bgzf.path
    ):
        raise ValueError("item window changed its prepared VCF")


def validate_table_reading_attempts(timeline) -> None:
    """Check every table reading Attempt's readings against its accepted output.

    Chunks the coordinator runs together interleave; each reads its own events.
    """
    from bio_neb.agentic_journey.attempt_journey import attempt_events

    for event in timeline:
        if (
            event["event_type"] != "AttemptStarted"
            or event["payload"]["agent_role"] != ROLE
        ):
            continue
        try:
            selected = attempt_events(timeline, event)
        except ValueError:
            # An open Attempt has no terminal yet; its facts so far are its own.
            selected = [
                item
                for item in timeline[timeline.index(event) :]
                if item.get("payload", {}).get("attempt_id")
                == event["payload"]["attempt_id"]
            ]
        if any(
            item["event_type"] == "InvestigationTableChunkBound" for item in selected
        ):
            validate_table_chunk_facts(timeline, selected)


def _read_result(journey: Journey):
    from bio_neb.application.queries import read_table_reading_stage

    return read_table_reading_stage(journey)


def table_reading_assignment(
    *,
    model_factory: NebiusModelEndpointFactory | None = None,
    work_item_ids: tuple[str, ...] | None = None,
) -> AttemptAssignment[AdmittedChunk, BoundInvocation]:
    """The table reader with its model bound; ``work_item_ids`` reserves a chunk."""
    from bio_neb.agentic_journey.role_instructions import stage_text

    @contextmanager
    def open_chunk(context: AttemptContext[AdmittedChunk], observe):
        work = context.admitted.work
        continuation = context.effective_policy.continuation
        context.activity.record(
            InvestigationTableChunkBound(
                attempt_id=context.attempt_id,
                continued_attempt_id=work.continued_attempt_id,
                authority_id=work.authority_id,
                collection_event_id=work.collection_event_id,
                collection_state_id=work.collection_state_id,
                chunk_size=work.chunk_size,
                rows=work.rows,
                runtime_selections=work.runtime_selections,
                continuation_policy=continuation,
                continuation_policy_id=continuation.policy_id,
            )
        )
        outputs = context.output_session(
            tool_name=OUTPUT_TOOL,
            output_type=TableReadingDraft,
        )
        agent, deps = assemble_agent(
            observe(structured_output_session=outputs),
            assignment=ROLE,
            instructions=lambda: stage_text(context.instructions.text, STAGE),
            tools={},
            stage=lambda _deps: STAGE,
            bound_values=lambda _deps, _name: SimpleNamespace(constants={}, enums={}),
            admission=ActionAdmissionSpec(validator=lambda _deps, _action: ()),
            output=TABLE_READING_OUTPUT,
            policy=context.effective_policy,
            model_policy=context.model,
            role_text=context.instructions,
            journey=context.journey,
            attempt_id=context.attempt_id,
            parent_activity=context.activity,
            failure_source=context.failure_source,
            dependencies=lambda _admission, observed: ChunkReadingDependencies(
                journey=context.journey,
                attempt_id=context.attempt_id,
                work=work,
                outputs=outputs,
                observed_tool_session=observed,
            ),
        )
        prompt = canonical_json(
            {
                # The role's job as its model reads it.
                "task": context.instructions.message(role(ROLE).job),
                "rows": _prompt_rows(work.rows),
                "researcher_instruction": work.researcher_instruction,
            }
        )
        yield BoundInvocation(agent, deps, lambda: prompt)

    return AttemptAssignment(
        agent_role=ROLE,
        activity_stage="investigation_table_reading_attempt",
        admit=lambda journey, reentry: _admit(journey, reentry, work_item_ids),
        validate_continuation=validate_continuation,
        open=open_chunk,
        drive=_drive,
        complete=_complete,
        fail=_fail,
        read_result=_read_result,
        model_factory=model_factory or create_nebius_model_endpoint,
        reopen=AttemptReopen(
            admission=ActionAdmissionSpec(validator=lambda _deps, _action: ()),
            validate_facts=lambda snapshot, selected: validate_table_chunk_facts(
                snapshot.timeline, selected
            ),
            structured_output=True,
            domain_bound=DomainBoundReopen(
                binding=_attempt_binding, terminal=_attempt_terminal
            ),
        ),
    )


def advance_table_reading(
    journey: Journey,
    *,
    environ: Mapping[str, str],
    model_factory: NebiusModelEndpointFactory | None = None,
    adapter: VcfProcessAdapter | None = None,
) -> bool:
    """Take one step of the open VCF table stage; True when it wrote.

    Pending windows are read first, all in one batch. An unused answer to this
    role's Review re-enters its stopped chunk; a pending Review holds the stage;
    a stop or reject answer holds the remaining items, which then fold as not
    run (#344 F3). Otherwise the next chunks get their Attempts, at most the
    Run's ``table_chunks`` bound at once, collected before the next step (#105);
    with none left the collection's item evidence is folded.

    First of all, the Run measures its health and each item's coverage
    (``run_measurement``, #414); its pending Review holds the stage.
    """
    from bio_neb.application.queries import read_table_reading_stage
    from bio_neb.investigation.run_measurement import advance_run_measurement

    stage = read_table_reading_stage(journey)
    if stage is None:
        return False
    events = read_evidence(journey)
    # First, whatever else the stage holds: the Run's health and every item's
    # coverage, a region's or an unprepared VCF's items too.
    measured = advance_run_measurement(
        journey,
        stage,
        events,
        _stage_preparation(events, stage),
        adapter or LocalBcftoolsAdapter(),
    )
    if measured is not None:
        return measured
    if stage.windows_pending:
        record_item_windows(journey, stage, adapter)
        return True
    reviews = table_reading_reviews(events)
    assignment = table_reading_assignment(model_factory=model_factory)
    if reviews.answer is not None:
        # An answered chunk re-enters even while a sibling's Review pends.
        assignment.run(journey, environ=environ, **reviews.answer)
        return True
    if reviews.pending:
        return False
    if stage.unread and not reviews.stopped:
        from bio_neb.agentic_journey.attempt_runner import AttemptUnit, run_attempts

        bound = selected_concurrency(_retained_registration(events).runtime_selections)
        chunks = next_chunks(journey, bound.table_chunks)
        run_attempts(
            tuple(
                AttemptUnit(
                    table_reading_assignment(
                        model_factory=model_factory, work_item_ids=chunk
                    )
                )
                for chunk in chunks
            ),
            journey,
            environ=environ,
            concurrency=bound.table_chunks,
        )
        return True
    fold = InvestigationItemEvidenceCollected(
        collection_event_id=stage.collection_event_id,
        collection_state_id=stage.collection.collection_state_id,
        items=fold_item_evidence(stage.collection_event_id, stage.collection, events),
    )
    journey.record(
        fold.event_type,
        fold.model_dump(mode="json", exclude={"event_type"}),
        [journey.head_event_id],
    )
    return True


# This domain's Tool rows (ADR 0040): the shared researcher question; its one
# handler is the Agent assembly's, as the Journey search's is.
TOOL_ROWS = (researcher_question_row("table_reader", ("vcf_table_reading",)),)


__all__ = [
    "TOOL_ROWS",
    "OUTPUT_TOOL",
    "STAGE",
    "AdmittedChunk",
    "ChunkReadingDependencies",
    "ChunkReadingRefused",
    "ChunkWork",
    "ItemReadingDraft",
    "TableReadingDraft",
    "advance_table_reading",
    "chunk_readings",
    "next_chunk",
    "next_chunks",
    "queryable_lookup",
    "record_item_windows",
    "run_table_settings",
    "table_reading_assignment",
    "table_rows",
    "validate_table_chunk_facts",
    "validate_table_reading",
    "validate_table_reading_attempts",
]
