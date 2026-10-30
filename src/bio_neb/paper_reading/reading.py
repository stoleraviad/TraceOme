"""A researcher's question about one Run paper, read in reading items (ADR 0046).

The ``paper_reader`` role reads one paper the Run stored (``PaperRecorded``, #420) with four
reading Tools (``tools``). Each reading item is one Attempt of the shared Attempt lifecycle,
assembled by the one Agent assembly, and one member of the reading's collection (ADR
0039): a stopped item holds only itself, through the shared hold rule, and its siblings go
on; an item's failure is retained by its true class and routed by the failure module.

The paper item reads the outline and proposes the areas the question needs
(``propose_reading_areas``), with the genome version and the organism as the paper writes
them; each area, and each id the question lists that code did not find, is its own item,
answered by the values the model reads (``answer_reading_item``). The ids the question
lists are looked up in code first (``lookup``), so the model reads only the misses. Code
keeps each value only where the general parser reads it in its span (``answer``).

Every decision is one fact, recorded when code makes it (``journey``): each finding as
code judges it, the paper item's plan as code keeps it, each item's completion naming
them. A stopped item keeps every finding it recorded; its Review's continue answer
re-enters it with those findings shown as done, and the model is asked only for what is
still refused or unread.

``answer_question`` is the entry a Run calls: it reads one stored paper for the question
the caller passes, as written, one item after another under the coordinator, and returns
the reading with its ``PaperAnswerRecorded``. ``advance_reading`` is its one step: it
settles as not read an item its Review's terminal answer ended (a reject, a dismiss or
a stop of any kind), re-enters a stopped item with its continue answer, runs the next
open item, and, once every item is settled, records the answer, the fold of the
reading's facts.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from bio_neb.agentic_journey.adapters import create_nebius_model_endpoint
from bio_neb.agentic_journey.agent_roles import (
    AttemptAdmission,
    AttemptAssignment,
    AttemptBinding,
    AttemptReopen,
    AttemptTerminalReading,
    DomainBoundReopen,
    RuntimeBinding,
)
from bio_neb.agentic_journey.continuation import review_direction
from bio_neb.agentic_journey.continuation_policy import (
    current_turn,
    validate_continuation,
)
from bio_neb.agentic_journey.invocation import (
    AssignmentOutput,
    OutputRepair,
    assemble_agent,
)
from bio_neb.agentic_journey.role_declaration import role
from bio_neb.content import canonical_json, shown_to_model
from bio_neb.contracts import FrozenContract
from bio_neb.paper_reading.answer import (
    Judged,
    contig_name,
    findings_repair,
    judge_all,
    kept_finding,
    organism_problems,
    reference_problems,
    repair_text,
    stated_organism,
    stated_reference,
    stated_text,
)
from bio_neb.paper_reading.contracts import (
    MAX_AREAS,
    FindingPointer,
    IdLookup,
    ItemAnswer,
    PaperAnswer,
    ReadDocument,
    ReadingItem,
    ReadingPlan,
    ValuePointer,
)
from bio_neb.paper_reading.journey import (
    TERMINAL_ANSWERS,
    ItemState,
    PaperAnswerRecorded,
    PaperReading,
    PaperReadingStarted,
    ReadingFindingJudged,
    ReadingItemCompleted,
    ReadingItemNotRead,
    ReadingPlanKept,
    item_member,
    member_item,
    paper_item,
    reading_members,
    settled_answer,
)
from bio_neb.paper_reading.question import model_question, requested_ids
from bio_neb.paper_reading.reader import Paper, lookup
from bio_neb.paper_reading.tools import (
    READING_ADMISSION,
    READING_TOOLS,
    ROLE,
    STAGE,
    ReadingDependencies,
    reading_bound_values,
)
from bio_neb.recording import evidence_view, read_evidence
from bio_neb.target_request.paper_targets import SourceLine
from bio_neb.target_request.paper_text import (
    contig_before,
    is_contig,
    pair_after,
)

PLAN_TOOL = "propose_reading_areas"
ANSWER_TOOL = "answer_reading_item"


def plan_problems(plan: ReadingPlan, paper: Paper, *, ids_listed: bool) -> list[str]:
    """Why code cannot take the paper item's plan, as ``code:detail``."""

    problems = reference_problems(plan.reference, paper)
    problems += organism_problems(plan.organism, paper)
    views = {*paper.sections, *paper.tables, *paper.documents}
    problems += [
        f"unknown_view:{view}"
        for area in plan.areas
        for view in area.views
        # A span id is a view too: the area item reads it with read_paper.
        if view not in views and paper.unit(view) is None
    ]
    if len(plan.areas) > MAX_AREAS:
        problems.append(f"too_many_areas:{len(plan.areas)}")
    if not plan.not_answered and not plan.areas and not ids_listed:
        problems.append("nothing_proposed")
    return list(dict.fromkeys(problems))


def item_problems(answer: ItemAnswer, *, kept: int = 0) -> list[str]:
    """Why code cannot take an area or id item's output at all, as ``code:detail``: it
    gives no finding and no reason, and no earlier call of the item gave a finding.
    Each finding is decided on its own (``answer.judge``)."""

    if not answer.findings and not answer.not_answered and not kept:
        return ["item_unanswered"]
    return []


@dataclass(frozen=True)
class ItemOutputTool:
    """One item kind's output Tool as the model sees it."""

    name: str
    output_type: type[FrozenContract]
    description: str


PLAN = ItemOutputTool(
    PLAN_TOOL,
    ReadingPlan,
    "Propose the areas the question needs, each read as its own item, the genome "
    "version the paper's positions are on and the organism its samples come from, each "
    "as written, with the span id of its line.",
)
ANSWER = ItemOutputTool(
    ANSWER_TOOL,
    ItemAnswer,
    "Answer this item: the findings the question asks for, each with every value the "
    "paper writes for it, as written and by the span id that writes it, and its reason.",
)


@dataclass(frozen=True)
class RecordedJudgement:
    """One finding as code judged it, with the ``ReadingFindingJudged`` that records it."""

    judged: Judged
    event_id: str


@dataclass
class ItemFindings:
    """An item's findings as code decided them across its calls and Attempts (#404 d2),
    each recorded when judged: the accepted and unresolved ones it keeps, the refused
    ones a repair asks for (``pending``, numbered as the repair numbers them) and the
    refused ones it records with no repair asked (``closed``). A finding naming a
    pending one (``corrects``, the same values sent again, or the same written values by
    slot and text, as defined in ``_named``) answers it: a kept correction clears it, a
    refused one is asked again. A pending finding the repair call leaves out stays
    recorded as refused, never withdrawn. A re-entered item starts from what its
    Journey records: its kept findings held, its refused ones pending."""

    held: list[RecordedJudgement] = field(default_factory=list)
    pending: list[RecordedJudgement] = field(default_factory=list)
    closed: list[RecordedJudgement] = field(default_factory=list)

    @classmethod
    def recorded(cls, reading: PaperReading, item_id: str) -> ItemFindings:
        """The item's standing findings as its Journey records them."""

        state = cls()
        for recorded in reading.standing(item_id):
            kept = RecordedJudgement(recorded.fact.judged(), recorded.event_id)
            (state.pending if kept.judged.outcome == "refused" else state.held).append(
                kept
            )
        return state

    @property
    def count(self) -> int:
        return len(self.held) + len(self.pending) + len(self.closed)

    @property
    def accepted(self) -> int:
        return sum(k.judged.outcome == "accepted" for k in self.held)

    @property
    def unresolved(self) -> int:
        return sum(k.judged.outcome == "unresolved" for k in self.held)

    def _named(self, one: Judged) -> int | None:
        """The pending finding ``one`` answers, by number, if any."""

        pending = [p.judged for p in self.pending]
        number = one.stated.corrects
        if number is None:
            # The same finding sent again, value for value, is that finding.
            number = next(
                (
                    i
                    for i, p in enumerate(pending, start=1)
                    if (p.stated.values, p.stated.spans)
                    == (one.stated.values, one.stated.spans)
                ),
                None,
            )
        if number is None and one.stated.values:
            # Or it restates that one pending finding's written values, by slot and text,
            # wherever it now cites them: all of its own among the pending one's (a
            # restatement that drops a refused change), or all of the pending one's among
            # its own (one that moves a value's cell or names each value's column). A
            # rewritten text matches only by ``corrects``.
            written = {(v.slot, v.written) for v in one.stated.values}
            restated = [
                i
                for i, p in enumerate(pending, start=1)
                if (theirs := {(v.slot, v.written) for v in p.stated.values})
                and (written <= theirs or theirs <= written)
            ]
            number = restated[0] if len(restated) == 1 else None
        return number if number is not None and 1 <= number <= len(pending) else None

    def answered(self, judged: Sequence[Judged]) -> list[RecordedJudgement | None]:
        """The pending finding each of one call's ``judged`` answers, if any; each
        pending finding is answered once, by the first that names it."""

        taken: set[int] = set()
        answers: list[RecordedJudgement | None] = []
        for one in judged:
            number = self._named(one)
            if number is None or number in taken:
                answers.append(None)
            else:
                taken.add(number)
                answers.append(self.pending[number - 1])
        return answers

    def keeps(self, one: Judged) -> bool:
        """Code already keeps this accepted or unresolved finding: it is sent again."""

        return one.outcome != "refused" and one in {k.judged for k in self.held}

    def keep(
        self,
        recorded: Sequence[RecordedJudgement],
        answered: Sequence[RecordedJudgement | None],
    ) -> None:
        """Once one call's findings are recorded: the kept ones held, the pending ones
        it left unanswered closed as refused, its refused ones pending."""

        named = {a.event_id for a in answered if a is not None}
        self.held = [
            *self.held,
            *(r for r in recorded if r.judged.outcome != "refused"),
        ]
        self.closed = [
            *self.closed,
            *(p for p in self.pending if p.event_id not in named),
        ]
        self.pending = [r for r in recorded if r.judged.outcome == "refused"]

    def shown(self) -> dict[str, Any] | None:
        """What a re-entered item's message shows of its recorded findings: each kept
        finding as done, and each refused one, numbered for the correction that names
        it, with its reasons. None when it recorded none."""

        if not (self.held or self.pending):
            return None
        return {
            "kept": [
                f"{k.judged.outcome}: {stated_text(k.judged.stated)}" for k in self.held
            ],
            "refused": findings_repair(
                [p.judged for p in self.pending],
                kept=self.accepted,
                unclear=self.unresolved,
            )
            if self.pending
            else None,
        }


_ID_POSITION = re.compile(
    r"(?:chr)?(?P<contig>[A-Za-z0-9_.]+?):(?P<number>[\d,]+)", re.I
)
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+|\d+")


def found_in_code(found: IdLookup, paper: Paper) -> FindingPointer:
    """A finding for an id code found: the values the id itself states, each as its line
    writes it, by span (an rsID; or a position, its contig as the line spells it and the
    allele pair written right after it), and every other line holding it. Code names no
    table cell REF or ALT by its column's words (owner, #486 Q2 a)."""

    first = found.hits[0]
    text = paper.documents[first.document_id].text
    values: list[ValuePointer] = []
    if found.kind == "rsid":
        written = text[first.offset : first.offset + len(found.id.strip())]
        values.append(ValuePointer(slot="rsid", written=written, span=first.span))
    elif found.kind == "position":
        stated = _ID_POSITION.fullmatch(found.id.strip())
        written = _NUMBER.match(text, first.offset).group()
        unit = paper.unit(first.span)
        # In a table row each value is its cell's; in text, the line's.
        cells = [(c, cell) for c in unit.columns or () if (cell := unit.cell(c))]
        holder, column = next(
            ((cell, c) for c, cell in cells if cell.start <= first.offset < cell.end),
            (unit.line, None),
        )
        before = text[holder.start : first.offset]
        after = text[first.offset + len(written) : holder.end]
        # The contig as written: before the number (chr4:54,108,297), else the row's
        # cell naming the id's contig.
        contig = contig_before(before)
        contig_column = column if contig else None
        if contig is None:
            wanted = contig_name(stated.group("contig"))
            contig_column, contig = next(
                (
                    (c, cell.text.strip())
                    for c, cell in cells
                    if is_contig(cell.text.strip(), bare=True)
                    and contig_name(cell.text.strip()) == wanted
                ),
                (None, None),
            )
        if contig is not None:
            values.append(
                ValuePointer(
                    slot="contig", written=contig, span=first.span, column=contig_column
                )
            )
        values.append(
            ValuePointer(
                slot="position",
                written=written,
                span=first.span,
                column=column,
                number=int(written.replace(",", "")),
            )
        )
        if pair := pair_after(after):
            values.append(
                ValuePointer(
                    slot="alleles", written=pair, span=first.span, column=column
                )
            )
    return FindingPointer(
        values=tuple(values),
        spans=tuple(
            dict.fromkeys(hit.span for hit in found.hits if hit.span != first.span)
        ),
        reason=(
            f"Code found {found.id} written on {found.lines} line(s) of the paper "
            f"before any model read it."
        ),
    )


def _prompt(
    question: str,
    paper: Paper,
    item: ReadingItem,
    requested: Sequence[IdLookup],
    job: str,
    instruction: str | None = None,
    recorded: Mapping[str, Any] | None = None,
) -> str:
    """The item's message: the role's job, which item it is and what it reads for, its
    paper text as every reading's result is shown (``content.shown_to_model``), and, for
    a re-entered item, the researcher's instruction and the findings its Journey already
    records (``ItemFindings.shown``): the kept ones as done, the refused ones to
    correct."""

    if item.kind == "paper":
        task: dict[str, Any] = {
            "task": job,
            "item": "paper",
            "question": question,
            "paper": paper.identity.model_dump(mode="json", exclude_none=True)
            if paper.identity
            else None,
            # The model reads only the misses; the rest are already answered.
            "ids_found_in_code": sum(found.found for found in requested),
            "ids_missed": [missed.id for missed in requested if not missed.found],
        }
    elif item.kind == "area":
        task = {
            "task": job,
            "item": "area",
            "question": question,
            "area": item.subject,
            "views": list(item.views),
            "reason": item.reason,
        }
    else:
        task = {
            "task": job,
            "item": "id",
            "question": question,
            "id": item.subject,
            "near": [hit.model_dump(mode="json") for hit in item.near],
        }
    if instruction is not None:
        task["researcher_instruction"] = instruction
    if recorded is not None:
        task["findings_recorded"] = {
            "note": (
                "Code already records these findings of this item: do not send a kept "
                "finding again. Correct a refused finding by its number (corrects), or "
                "leave it out; then give only the findings not yet recorded."
            ),
            **recorded,
        }
    return canonical_json(shown_to_model(task))


# ---------------------------------------------------------------------------
# One reading item as one Attempt of the shared lifecycle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemWork:
    """What one item's Attempt reads: its reading, its item, and, for a fresh paper
    item, the opening fact its Attempt records first."""

    reading_id: str
    item: ReadingItem
    paper_event_id: str
    question: str
    requested: tuple[IdLookup, ...]
    opening: PaperReadingStarted | None = None
    continued_attempt_id: str | None = None
    instruction: str | None = None


def _reading(journey, reading_id: str) -> PaperReading | None:
    """The reading as the Journey's checked view holds it (ADR 0032)."""
    try:
        readings = journey.checked_view("paper_readings")
    except KeyError:  # no reading fact yet: the family holds no view
        return None
    return readings.reading(reading_id)


def _item_work(journey, reentry, fresh: ItemWork | None) -> ItemWork:
    """The item a fresh Attempt reads, or the stopped item a Review's answer re-enters."""

    direction = review_direction(journey, reentry, agent_role=ROLE)
    if direction.previous_attempt_id is None:
        if fresh is None:
            raise ValueError("a reading item's Attempt names its reading and item")
        return fresh
    (started,) = [
        event
        for event in read_evidence(journey)
        if event["event_type"] == "AttemptStarted"
        and event["payload"]["attempt_id"] == direction.previous_attempt_id
    ]
    reading_id, item_id = member_item(started["payload"].get("member_id"))
    reading = _reading(journey, reading_id)
    if reading is None or reading.item(item_id) is None:
        raise ValueError("a re-entered reading item names its started reading")
    return ItemWork(
        reading_id=reading_id,
        item=reading.item(item_id),
        paper_event_id=reading.paper_event_id,
        question=reading.question,
        requested=reading.requested,
        continued_attempt_id=direction.previous_attempt_id,
        instruction=direction.instruction,
    )


def reading_item_assignment(
    *,
    runtime: RuntimeBinding | None = None,
    work: ItemWork | None = None,
    model_factory: Any = None,
) -> AttemptAssignment:
    """One reading item as one Attempt of the shared lifecycle, its member the item.

    ``work`` names a fresh item; a Review's answer names the stopped item it re-enters.
    ``runtime`` is the caller's binding; without one, the Run's registered binding."""

    def admit(journey, reentry) -> AttemptAdmission:
        admitted = _item_work(journey, reentry, work)
        direction = review_direction(journey, reentry, agent_role=ROLE)
        return AttemptAdmission(
            admitted,
            runtime or _registered_runtime(journey),
            direction.causal_head,
            direction.availability_frontier,
            member_id=item_member(admitted.reading_id, admitted.item.item_id),
        )

    @contextmanager
    def open_item(context, observe):
        admitted: ItemWork = context.admitted
        paper = Paper.stored(context.journey, admitted.paper_event_id)
        item = admitted.item
        record = _judgements(context, admitted, paper)
        if admitted.opening is not None:
            context.activity.record(
                admitted.opening.model_copy(update={"attempt_id": context.attempt_id})
            )
            # Code judges the ids it found before any model reads (ADR 0046).
            found = [f for f in admitted.requested if f.found]
            record(judge_all([found_in_code(f, paper) for f in found], paper), (), True)
        reading = _reading(context.journey, admitted.reading_id)
        # A re-entered item starts from what its Journey records.
        state = ItemFindings.recorded(reading, item.item_id)
        output = PLAN if item.kind == "paper" else ANSWER
        # The validator and normalizer are the output type's (``StructuredOutputSession``).
        outputs = context.output_session(
            tool_name=output.name, output_type=output.output_type
        )
        member = item_member(admitted.reading_id, item.item_id)
        outputs.begin_member(member)
        filled: dict[str, Any] = {}
        continuation = context.effective_policy.continuation

        def asked_again() -> bool:
            # Asked again only while this refusal's failed turn stops nothing.
            count, limit = current_turn(
                evidence_view(context.journey),
                attempt_id=context.attempt_id,
                policy=continuation,
            )
            return count < limit

        def problems(draft) -> list[str]:
            if item.kind == "paper":
                return plan_problems(draft, paper, ids_listed=bool(admitted.requested))
            return item_problems(draft, kept=state.count)

        def validate(_context, draft):
            dumped = draft.model_dump(mode="json")
            if refusals := tuple(problems(draft)):
                repair = repair_text(refusals)
                outputs.validate(dumped, refusals, repair=repair)
                raise OutputRepair(repair)
            if item.kind == "paper":
                # The plan is recorded as soon as code keeps it.
                filled["plan_event_id"] = _keep_plan(context, admitted, paper, draft)
                outputs.validate(dumped, ())
                return draft
            judged = judge_all(draft.findings, paper)
            answered = state.answered(judged)
            # Each finding is recorded as code judges it, the refused ones with their
            # reasons; one code already keeps, sent again, is not recorded twice.
            fresh = [
                (one, answer)
                for one, answer in zip(judged, answered, strict=True)
                if answer is not None or not state.keeps(one)
            ]
            recorded = record(
                [one for one, _ in fresh], [answer for _, answer in fresh], False
            )
            state.keep(recorded, answered)
            # A refused finding is asked again while failed turns remain; on the
            # item's last allowed call it stays recorded as refused and its siblings
            # stay kept.
            if state.pending and asked_again():
                errors = tuple(
                    f"{reason}@refused {number}"
                    for number, one in enumerate(state.pending, start=1)
                    for reason in one.judged.reasons
                )
                repair = findings_repair(
                    [p.judged for p in state.pending],
                    kept=state.accepted,
                    unclear=state.unresolved,
                )
                outputs.validate(dumped, errors, repair=repair)
                raise OutputRepair(repair)
            outputs.validate(dumped, ())
            filled["not_answered"] = draft.not_answered
            return draft

        agent, deps = assemble_agent(
            observe(structured_output_session=outputs),
            assignment=ROLE,
            instructions=context.instructions.text,
            tools=READING_TOOLS,
            stage=lambda _deps: STAGE,
            bound_values=reading_bound_values,
            admission=READING_ADMISSION,
            policy=context.effective_policy,
            model_policy=context.model,
            role_text=context.instructions,
            journey=context.journey,
            attempt_id=context.attempt_id,
            parent_activity=context.activity,
            failure_source=context.failure_source,
            dependencies=lambda _admission, observed: ReadingDependencies(
                paper=paper,
                attempt_id=context.attempt_id,
                parent_activity=context.activity,
                observed_tool_session=observed,
            ),
            output=AssignmentOutput(
                output_type=output.output_type,
                name=output.name,
                description=output.description,
                validator=validate,
            ),
        )
        yield SimpleNamespace(
            agent=agent,
            deps=deps,
            filled=filled,
            paper=paper,
            member=member,
            # A re-entered area or id item is shown what it already recorded.
            recorded=state.shown()
            if admitted.continued_attempt_id and item.kind != "paper"
            else None,
        )

    async def drive(context, bound, invoke_member, message_history) -> dict[str, Any]:
        admitted: ItemWork = context.admitted
        # The item's message states the role's job as its model reads it.
        job = context.instructions.message(role(ROLE).job)
        asked = model_question(
            admitted.question, {f.id for f in admitted.requested if f.found}
        )
        prompt = _prompt(
            asked,
            bound.paper,
            admitted.item,
            admitted.requested,
            job,
            admitted.instruction,
            bound.recorded,
        )
        await invoke_member(
            bound.agent,
            None if message_history is not None else prompt,
            deps=bound.deps,
            member_id=bound.member,
            message_history=message_history,
        )
        if not bound.filled:
            from bio_neb.agentic_journey.agent_work_review import AgentResultMissing

            raise AgentResultMissing("model ended without the item's output")
        return bound.filled

    def complete(context, filled: dict[str, Any]) -> None:
        # The item's terminal names the findings and plan it recorded, copying none.
        admitted: ItemWork = context.admitted
        reading = _reading(context.journey, admitted.reading_id)
        terminal = context.activity.record(
            ReadingItemCompleted(
                attempt_id=context.attempt_id,
                reading_id=admitted.reading_id,
                item=admitted.item,
                finding_event_ids=tuple(
                    f.event_id for f in reading.item_findings(admitted.item.item_id)
                ),
                plan_event_id=filled.get("plan_event_id"),
                not_answered=filled.get("not_answered"),
            )
        )
        context.activity.succeeded(terminal)

    def fail(context, _bound, error) -> bool:
        from bio_neb.agentic_journey.attempt_journey import record_attempt_failure
        from bio_neb.agentic_journey.failure_cause import cause_identity
        from bio_neb.failure import Invariant

        failed = record_attempt_failure(context, error)
        raise Invariant(
            "paper_reading_attempt_failed",
            "paper reading item Attempt failed",
            cause=cause_identity(error),
            evidence={
                "attempt_id": context.attempt_id,
                "attempt_failed_event_id": failed["event_id"],
                "code": failed["payload"]["code"],
            },
        ) from error

    return AttemptAssignment(
        agent_role=ROLE,
        activity_stage="paper_reading_attempt",
        admit=admit,
        validate_continuation=validate_continuation,
        open=open_item,
        drive=drive,
        complete=complete,
        fail=fail,
        read_result=lambda journey: None,
        model_factory=model_factory or create_nebius_model_endpoint,
        reopen=AttemptReopen(
            admission=READING_ADMISSION,
            structured_output=True,
            domain_bound=DomainBoundReopen(
                binding=_attempt_binding, terminal=_attempt_terminal
            ),
        ),
    )


def _cited_documents(
    paper: Paper, lines: Sequence[SourceLine]
) -> tuple[ReadDocument, ...]:
    """The documents a fact's lines cite, by their stored text's digest."""

    cited = {line.document_id for line in lines}
    return tuple(d for d in paper.listing if d.document_id in cited)


def _judgements(context, admitted: ItemWork, paper: Paper):
    """The item's recorder of findings: each judged finding one ``ReadingFindingJudged``
    through the Attempt's activity, when code judges it, naming the refused finding it
    answers; the recorded findings, in order."""

    def record(
        judged: Sequence[Judged],
        answers: Sequence[RecordedJudgement | None],
        found_in_code: bool,
    ) -> list[RecordedJudgement]:
        recorded = []
        for number, one in enumerate(judged):
            answer = answers[number] if answers else None
            quote = kept_finding(one).passages[0] if one.outcome == "accepted" else None
            event = context.activity.record(
                ReadingFindingJudged(
                    attempt_id=context.attempt_id,
                    reading_id=admitted.reading_id,
                    item_id=admitted.item.item_id,
                    outcome=one.outcome,
                    stated=one.stated,
                    reasons=one.reasons,
                    values=one.values,
                    lines=one.lines,
                    quote=quote,
                    documents=_cited_documents(
                        paper,
                        [*one.lines, *(v.source for v in one.values)],
                    ),
                    answers=None if answer is None else answer.event_id,
                    found_in_code=found_in_code,
                )
            )
            recorded.append(RecordedJudgement(one, event["event_id"]))
        return recorded

    return record


def _keep_plan(context, admitted: ItemWork, paper: Paper, plan: ReadingPlan) -> str:
    """Record the paper item's plan as code keeps it: the version as written with its
    aliases and line, the organism as written, and the areas; its event id."""

    reference = stated_reference(plan.reference, paper)
    organism = stated_organism(plan.organism, paper)
    lines = [
        *(reference.candidate.sources if reference else ()),
        *((organism.source, organism.passage) if organism else ()),
    ]
    return context.activity.record(
        ReadingPlanKept(
            attempt_id=context.attempt_id,
            reading_id=admitted.reading_id,
            reference=reference,
            organism=organism,
            areas=plan.areas,
            not_answered=plan.not_answered,
            documents=_cited_documents(paper, lines),
        )
    )["event_id"]


def _window_attempt_id(selected) -> str:
    return next(
        event["payload"]["attempt_id"]
        for event in selected
        if event["event_type"] == "AttemptStarted"
    )


def _registration(timeline):
    from bio_neb.application.input_journey import RunInputsRegistered, validate_fact

    registrations = [
        fact
        for event in timeline
        if event.get("event_type") == "RunInputsRegistered"
        and isinstance(fact := validate_fact(event), RunInputsRegistered)
    ]
    if len(registrations) != 1:
        raise ValueError("a reading item in a Run is bound to its one registration")
    return registrations[0]


def _registered_runtime(journey) -> RuntimeBinding:
    """The Run's registered runtime, which binds a reading item a Review re-enters."""
    registration = _registration(read_evidence(journey))
    return RuntimeBinding(registration.runtime_selections, registration.input_set_id)


def _attempt_binding(timeline, selected) -> AttemptBinding:
    """A reading item in a Run is bound to the Run's registered runtime."""
    registration = _registration(timeline)
    return AttemptBinding(registration.input_set_id, registration.runtime_selections)


_TERMINALS = {
    "ReadingItemCompleted": "succeeded",
    "AttemptStopped": "stopped",
    "AttemptFailed": "failed",
    "AttemptCancelled": "cancelled",
}


def _attempt_terminal(_snapshot, selected) -> AttemptTerminalReading:
    """The item's one terminal: its completion, or the lifecycle's stop, failure or
    cancellation."""

    attempt_id = _window_attempt_id(selected)
    owned = [
        event
        for event in selected
        if event["payload"].get("attempt_id") == attempt_id
        and event["event_type"] in _TERMINALS
    ]
    if len(owned) != 1:
        raise ValueError("a paper reading does not bind this Attempt's exact terminal")
    (terminal,) = owned
    return AttemptTerminalReading(
        _TERMINALS[terminal["event_type"]], terminal["event_id"]
    )


# ---------------------------------------------------------------------------
# The reading in a Run: one stored paper, one question, under the coordinator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemOutcome:
    item: ReadingItem
    attempt_id: str | None  # the item's latest Attempt; None before it ran
    # The Review that ended the item's latest Attempt without its output, if any.
    stopped: str | None = None
    state: ItemState = "open"  # as the reading's fold settles it


@dataclass(frozen=True)
class QuestionOutcome:
    """One reading as the Run's Journey records it: its answer once every item has
    settled (``answer``, from ``PaperAnswerRecorded``), and each item's latest Attempt."""

    reading_id: str | None
    answer: PaperAnswer | None
    recorded: PaperAnswerRecorded | None
    items: tuple[ItemOutcome, ...]
    journey_file: str
    model: str | None  # the model the runtime configuration selected
    requests: RequestCounts = field(default_factory=lambda: RequestCounts())

    @property
    def stopped(self) -> tuple[str, ...]:
        """Each item whose latest Attempt ended without its output, with its Review."""

        return tuple(
            f"{o.item.kind} item {o.item.subject!r}: {o.stopped}"
            for o in self.items
            if o.stopped
        )


@dataclass(frozen=True)
class RequestCounts:
    """What the reading sent, counted from its Journey's evidence."""

    model_requests: int = 0  # logical requests, each Attempt's ceiling counts these
    physical_contacts: int = 0  # sends to the endpoint, availability retries included
    failed_contacts: int = 0
    tool_calls: tuple[str, ...] = ()  # each executed Tool call, in order
    refused_answers: int = 0  # outputs code refused and sent back for repair


def _counts(evidence: Sequence[Mapping[str, Any]], attempts: set[str]) -> RequestCounts:
    owned = [e for e in evidence if e.get("payload", {}).get("attempt_id") in attempts]

    def count(event_type: str) -> int:
        return sum(e["event_type"] == event_type for e in owned)

    return RequestCounts(
        model_requests=count("ModelRequestRecorded"),
        physical_contacts=count("ModelPhysicalContactStarted"),
        failed_contacts=count("ModelPhysicalContactFailed"),
        tool_calls=tuple(
            e["payload"]["tool_name"]
            for e in owned
            if e["event_type"] == "ObservedToolExecutionStarted"
        ),
        refused_answers=sum(
            e["event_type"] == "AgentOutputValidated"
            and e["payload"]["status"] == "invalid"
            for e in owned
        ),
    )


def _item_attempts(evidence, reading_id: str) -> dict[str, list[Mapping[str, Any]]]:
    """Each item's Attempts, in order, by the member they name."""

    attempts: dict[str, list[Mapping[str, Any]]] = {}
    prefix = reading_members(reading_id)
    for event in evidence:
        member = event.get("payload", {}).get("member_id")
        if (
            event["event_type"] == "AttemptStarted"
            and event["payload"]["agent_role"] == ROLE
            and isinstance(member, str)
            and member.startswith(prefix)
        ):
            attempts.setdefault(member.removeprefix(prefix), []).append(event)
    return attempts


def _stop_of(evidence, attempt_id: str) -> str | None:
    """What stopped an Attempt: its ``AttemptStopped`` code, if it stopped."""

    return next(
        (
            event["payload"]["code"]
            for event in evidence
            if event["event_type"] == "AttemptStopped"
            and event["payload"]["attempt_id"] == attempt_id
        ),
        None,
    )


def _reentry(evidence, attempt_ids: set[str]) -> dict[str, str] | None:
    """The one unused continuing answer to a stopped item's Review, as the runner's
    re-entry argument, if any (each continuation kind's own answers)."""
    from bio_neb.agentic_journey.continuation import REVIEW_CONTINUATIONS

    consumed = {
        event["payload"].get(kind.field)
        for event in evidence
        if event["event_type"] == "AttemptStarted"
        for kind in REVIEW_CONTINUATIONS
    }
    for kind in REVIEW_CONTINUATIONS:
        answers = {
            event["payload"]["request_event_id"]: event
            for event in evidence
            if event["event_type"] == kind.response_kind
        }
        for request in evidence:
            if (
                request["event_type"] != kind.request_kind
                or request["payload"].get("attempt_id") not in attempt_ids
            ):
                continue
            answer = answers.get(request["event_id"])
            if (
                answer is not None
                and answer["payload"]["response"]["action"] in kind.actions
                and answer["event_id"] not in consumed
            ):
                return {_RUN_ARGUMENTS[kind.field]: answer["event_id"]}
    return None


_RUN_ARGUMENTS = {
    "continuation_availability_review_event_id": "availability_review_response_event_id",
    "continuation_effort_review_event_id": "effort_review_response_event_id",
    "continuation_work_review_event_id": "work_review_response_event_id",
    "continuation_operational_review_event_id": "operational_review_response_event_id",
}


def _stop_reviews(journey) -> dict[str, Any]:
    """Each stop kind's checked Review (ADR 0031), by its answer's kind: Agent work,
    effort, model availability and operational."""
    from bio_neb.agentic_journey.agent_work_review import project_agent_work_review
    from bio_neb.agentic_journey.effort import project_effort_review
    from bio_neb.agentic_journey.model_availability import (
        project_model_availability_review,
    )
    from bio_neb.agentic_journey.operational_review import project_operational_review
    from bio_neb.recording import evidence_view

    events = evidence_view(journey)
    once = journey.derived_once
    return {
        "AgentWorkReviewResponded": project_agent_work_review(
            events, checked_once=once
        ),
        "AgentEffortReviewResponded": project_effort_review(events, checked_once=once),
        "ModelAvailabilityReviewResponded": project_model_availability_review(
            events, checked_once=once
        ),
        "AgentFailureReviewResponded": project_operational_review(
            events, checked_once=once
        ),
    }


def _owned(attempts: Mapping[str, list[Mapping[str, Any]]], item_id: str) -> set[str]:
    return {e["payload"]["attempt_id"] for e in attempts.get(item_id, ())}


def _held(
    journey,
    reviews: Mapping[str, Any],
    reading_id: str,
    item_id: str,
    owned: set[str],
) -> bool:
    """The item's own stop holds it: the shared hold rule, in its member scope (#105),
    and its own pending Review of any kind. The shared rule holds a member by its effort
    and Agent work stops; a model availability or operational Review holds only
    planning's role there, so the item's own pending one holds it here."""
    from bio_neb.agentic_journey.agent_roles import Reentry
    from bio_neb.agentic_journey.attempt_runner import hold_stopped_role

    if any(
        request.pending and request.attempt_id in owned
        for review in reviews.values()
        for request in review.requests
    ):
        return True
    try:
        hold_stopped_role(
            journey,
            Reentry(None, None),
            agent_role=ROLE,
            member_id=item_member(reading_id, item_id),
        )
    except ValueError:
        return True
    return False


def _not_read(
    reviews: Mapping[str, Any],
    reading: PaperReading,
    attempts: Mapping[str, list[Mapping[str, Any]]],
) -> ReadingItemNotRead | None:
    """The first open item a terminal answer to its own Review settled, as the fact
    that records it as not read (ADR 0039), if any: its Agent work Review's reject or
    dismiss, or a stop of its effort, model availability or operational Review. The
    role settles a member for a stop of any kind (``member_scope``, ADR 0031); the
    answer grants no new model work."""

    for item in reading.items:
        if reading.state(item.item_id) != "open":
            continue
        owned = _owned(attempts, item.item_id)
        for kind, review in reviews.items():
            _, outcomes = TERMINAL_ANSWERS[kind]
            asked = {
                r.request_event_id for r in review.requests if r.attempt_id in owned
            }
            for receipt in review.receipts:
                if (
                    receipt.request_event_id in asked
                    and receipt.response.action in outcomes
                ):
                    return ReadingItemNotRead(
                        reading_id=reading.reading_id,
                        item_id=item.item_id,
                        outcome=outcomes[receipt.response.action],
                        review_response_event_id=receipt.response_event_id,
                        reason=receipt.response.reason,
                    )
    return None


def asked_only_by_its_items(journey, reading_id: str) -> bool:
    """Whether every pending Review of a stop in the Run was asked by an item of this
    reading: each holds only its own item, so the reading's other items go on (ADR
    0039; owner, #383 Q23). A pending Review of any other work holds the reading."""

    owned = {
        event["payload"]["attempt_id"]
        for started in _item_attempts(read_evidence(journey), reading_id).values()
        for event in started
    }
    return all(
        request.attempt_id in owned
        for review in _stop_reviews(journey).values()
        for request in review.requests
        if request.pending
    )


def _record_settlement(journey, fact: ReadingItemNotRead | PaperAnswerRecorded):
    """The coordinator's one write: a reading's settling fact, appended through the
    Journey's one semantic writer, whose fold checks it first (ADR 0032)."""

    payload = fact.model_dump(mode="json")
    payload.pop("event_type")
    return journey.record(fact.event_type, payload, [journey.head_event_id])


def advance_reading(
    journey,
    reading_id: str,
    *,
    runtime: RuntimeBinding,
    environ: Mapping[str, str],
    model_factory: Any = None,
) -> bool:
    """Take one step of a started reading; True when it took one.

    In order: an open item its Review's terminal answer settled (a reject or dismiss,
    or a stop of any kind) is recorded as not read; an unused continuing answer to a stopped item's Review re-enters that item; the next
    open item no stop holds runs (a stopped item holds only itself, so its siblings go
    on; ADR 0039); and once every item is settled, the coordinator records the
    reading's answer, the fold of its facts (``settled_answer``). With the answer
    recorded, or an item still held by its Review, nothing runs."""

    reading = _reading(journey, reading_id)
    if reading is None or reading.answer_event_id is not None:
        return False
    evidence = read_evidence(journey)
    attempts = _item_attempts(evidence, reading_id)
    reviews = _stop_reviews(journey)
    if (settled := _not_read(reviews, reading, attempts)) is not None:
        _record_settlement(journey, settled)
        return True
    reentry = _reentry(
        evidence,
        {
            event["payload"]["attempt_id"]
            for item_id, started in attempts.items()
            if reading.state(item_id) == "open"
            for event in started
        },
    )
    assignment = reading_item_assignment(runtime=runtime, model_factory=model_factory)
    if reentry is not None:
        assignment.run(journey, environ=environ, **reentry)
        return True
    for item in reading.items:
        if reading.state(item.item_id) != "open" or _held(
            journey, reviews, reading_id, item.item_id, _owned(attempts, item.item_id)
        ):
            continue
        reading_item_assignment(
            runtime=runtime,
            model_factory=model_factory,
            work=ItemWork(
                reading_id=reading_id,
                item=item,
                paper_event_id=reading.paper_event_id,
                question=reading.question,
                requested=reading.requested,
            ),
        ).run(journey, environ=environ)
        return True
    if reading.all_settled:
        _record_settlement(journey, settled_answer(reading))
        return True
    return False


def start_reading(
    journey,
    *,
    paper_event_id: str,
    question: str,
    runtime: RuntimeBinding,
    environ: Mapping[str, str],
    model_factory: Any = None,
) -> str | None:
    """Open a reading of one stored paper for the caller's question, kept as written:
    its paper item's Attempt records ``PaperReadingStarted`` first, then each finding
    code judged for an id the question lists. The reading's id, or None when the
    Attempt never opened it."""

    paper = Paper.stored(journey, paper_event_id)
    (recorded,) = [
        event for event in read_evidence(journey) if event["event_id"] == paper_event_id
    ]
    listed = requested_ids(question)
    reading_id = str(uuid.uuid4())
    opening = PaperReadingStarted(
        attempt_id="",  # the paper item's Attempt records it with its own id
        reading_id=reading_id,
        paper_id=recorded["payload"]["paper_id"],
        paper_event_id=paper_event_id,
        question=question,
        documents=paper.listing,
        requested=lookup(paper, listed) if listed else (),
    )
    reading_item_assignment(
        runtime=runtime,
        model_factory=model_factory,
        work=ItemWork(
            reading_id=reading_id,
            item=paper_item(question, opening.requested),
            paper_event_id=paper_event_id,
            question=question,
            requested=opening.requested,
            opening=opening,
        ),
    ).run(journey, environ=environ)
    return reading_id if _reading(journey, reading_id) is not None else None


def answer_question(
    journey,
    *,
    paper_event_id: str,
    question: str,
    runtime: RuntimeBinding,
    environ: Mapping[str, str],
    model_factory: Any = None,
) -> QuestionOutcome:
    """Read one stored Run paper for the caller's question, in reading items, and
    record the answer once every item has settled (``PaperAnswerRecorded``).

    The question is the caller's words, as written (in a Run, the plan's research goal
    as the researcher wrote it): code neither narrows nor widens it, and whether it asks
    for one finding or every minor one is the paper item's to read and the areas it
    proposes. The items run one after another, each its own Attempt under the role's
    declared budgets (``ROLES["paper_reader"]`` over the global role default). A
    stopped item holds only itself: the others run, and the reading's answer waits for
    the stopped item's Review (its continue answer re-enters it, a reject or dismiss
    settles it as not read, through ``advance_reading``)."""

    reading_id = start_reading(
        journey,
        paper_event_id=paper_event_id,
        question=question,
        runtime=runtime,
        environ=environ,
        model_factory=model_factory,
    )
    if reading_id is not None:
        while advance_reading(
            journey,
            reading_id,
            runtime=runtime,
            environ=environ,
            model_factory=model_factory,
        ):
            pass
    return reading_outcome(journey, reading_id)


def reading_outcome(journey, reading_id: str | None) -> QuestionOutcome:
    """A reading as its Journey records it: its answer, and each item's latest Attempt."""

    evidence = read_evidence(journey)
    reading = None if reading_id is None else _reading(journey, reading_id)
    attempts = {} if reading is None else _item_attempts(evidence, reading_id)
    recorded = None
    if reading is not None and reading.answer_event_id is not None:
        (event,) = [e for e in evidence if e["event_id"] == reading.answer_event_id]
        recorded = PaperAnswerRecorded.model_validate(event["payload"])
    items = []
    for item in () if reading is None else reading.items:
        latest = attempts.get(item.item_id, [None])[-1]
        attempt_id = None if latest is None else latest["payload"]["attempt_id"]
        state = reading.state(item.item_id)
        items.append(
            ItemOutcome(
                item=item,
                attempt_id=attempt_id,
                stopped=None
                if attempt_id is None or state != "open"
                else _stop_of(evidence, attempt_id),
                state=state,
            )
        )
    owned = {
        event["payload"]["attempt_id"]
        for started in attempts.values()
        for event in started
    }
    return QuestionOutcome(
        reading_id=reading_id,
        answer=None if recorded is None else recorded.answer,
        recorded=recorded,
        items=tuple(items),
        journey_file=str(journey.path),
        model=next(
            (
                e["payload"]["nebius_model_endpoint"]["model"]
                for e in evidence
                if e["event_type"] == "ModelRequestRecorded"
                and e["payload"]["attempt_id"] in owned
            ),
            None,
        ),
        requests=_counts(evidence, owned),
    )


def paper_reading_assignment(*, model_factory: Any = None) -> AttemptAssignment:
    """The reader's assignment as the coordinator re-enters or reopens it: the item a
    Review's answer names, bound to the Run's registered runtime."""

    return reading_item_assignment(model_factory=model_factory)


__all__ = [
    "ANSWER",
    "PLAN",
    "ItemFindings",
    "ItemOutcome",
    "ItemWork",
    "QuestionOutcome",
    "RequestCounts",
    "advance_reading",
    "answer_question",
    "found_in_code",
    "item_member",
    "item_problems",
    "model_question",
    "paper_reading_assignment",
    "plan_problems",
    "reading_item_assignment",
    "reading_outcome",
    "requested_ids",
    "start_reading",
]
