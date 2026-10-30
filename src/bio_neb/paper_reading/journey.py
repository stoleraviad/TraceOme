"""A paper reading in the Run's Journey: its facts in one semantic family, each decision
one fact, recorded when it is made and never changed (#437, #479; owner, #383 A1b).

A reading reads one Run paper (``PaperRecorded``, #420) for one question in reading items
(ADR 0046), each its own Attempt. Every fact an Attempt makes is written through the one
semantic writer, its ``activity.record`` (ADR 0032):

- ``PaperReadingStarted`` opens the reading, in its paper item's Attempt: the paper, the
  question as the caller passed it, each document by its ``PaperDocumentRead`` and text
  digest, and each id the question lists, looked up in code;
- ``PaperReadingRecorded`` is one reading Tool call and its result;
- ``ReadingFindingJudged`` is one finding as code judged it, when it judged it: a finding
  code stated for a listed id it found, or one the model stated in a reading item's
  output, accepted, unresolved or refused with its reasons (#404 d2);
- ``ReadingPlanKept`` is the paper item's plan as code kept it: the genome version as
  written with its aliases, the organism as written, and the areas, each its own item;
- ``ReadingItemCompleted`` closes an item by naming its finding facts (and the paper
  item's plan), copying none of them.

The coordinator's entry records the reading's two settling facts:

- ``ReadingItemNotRead``: a stopped item its Review's reject or dismiss answer settled
  as not read, with the researcher's reason (ADR 0039);
- ``PaperAnswerRecorded``: the answer, once every item is settled, with its proposed
  targets. It is the fold of the facts above (``settled_answer``), and the fold refuses
  any other answer, on append and on reopen.

Each finding keeps its pointer, the ``document_id`` (whose text digest its ``documents``
name) and ``start:end``, and the chunk code copied from the stored text (its quote, and
its paragraph or table row), with the model's reason (owner, #383 Q2). A Fork child
inherits these facts in its prefix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from bio_neb.contracts import FrozenContract
from bio_neb.paper_reading.answer import (
    Judged,
    answer_of,
    finding_outcome,
    kept_finding,
    merged_findings,
)
from bio_neb.paper_reading.contracts import (
    FindingOutcome,
    FindingPointer,
    IdLookup,
    KeptFinding,
    KeptValue,
    PaperAnswer,
    ReadDocument,
    ReadingArea,
    ReadingItem,
    StatedOrganism,
)
from bio_neb.paper_reading.question import model_question
from bio_neb.paper_reading.targets import paper_targets
from bio_neb.target_request.paper_targets import (
    PaperIdentity,
    PaperTargets,
    SourceLine,
    StatedReference,
)

_MEMBER = "paper-reading"


def item_member(reading_id: str, item_id: str) -> str:
    """The collection member an item's Attempt names: its stops hold only it (#105)."""

    return f"{_MEMBER}:{reading_id}:{item_id}"


def member_item(member_id: str | None) -> tuple[str, str]:
    """The reading and item a reading item's member names."""

    kind, _, rest = (member_id or "").partition(":")
    reading_id, _, item_id = rest.rpartition(":")
    if kind != _MEMBER or not reading_id or not item_id:
        raise ValueError("a reading item's Attempt names its reading and item")
    return reading_id, item_id


def reading_members(reading_id: str) -> str:
    """The prefix every member of one reading's items starts with."""

    return item_member(reading_id, "")


class _Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PaperReadingStarted(_Fact):
    """A reading's opening fact: the paper and question it reads, and each document by
    the ``PaperDocumentRead`` whose text its spans point into."""

    event_type: Literal["PaperReadingStarted"] = "PaperReadingStarted"
    attempt_id: StrictStr
    reading_id: StrictStr
    paper_id: StrictStr = Field(pattern=r"^paper-[0-9a-f]{64}$")
    paper_event_id: StrictStr
    question: StrictStr = Field(min_length=1)
    documents: tuple[ReadDocument, ...] = Field(min_length=1)
    requested: tuple[IdLookup, ...] = ()


class PaperReadingRecorded(_Fact):
    """The one tool-owned fact of a reading Tool's call: what it read and its result."""

    event_type: Literal["PaperReadingRecorded"] = "PaperReadingRecorded"
    attempt_id: StrictStr
    tool_call_id: StrictStr
    tool_name: StrictStr
    arguments: dict[str, Any]
    outcome: Literal["read", "nothing_found"]
    result: dict[str, Any]


class ReadingFindingJudged(_Fact):
    """One finding as code judged it, recorded when judged (#404 d2): ``accepted``, with
    the quote code chose from its kept values; ``unresolved`` (the paper itself is
    unclear) or ``refused`` (a check failed), with its reasons.

    It keeps what the model stated (``stated``: each value as written by span, and its
    reason), each value code kept with its slot, span and place, and the paragraph or
    row of each span it cites (``lines``), all copied from the stored texts ``documents``
    name by digest. ``answers`` names the refused finding of its item that this one
    answers: a correction (``corrects``) or the same finding sent again; the answered
    one stands no more. ``found_in_code`` marks a finding code stated for an id the
    question lists, before any model read."""

    event_type: Literal["ReadingFindingJudged"] = "ReadingFindingJudged"
    attempt_id: StrictStr
    reading_id: StrictStr
    item_id: StrictStr
    outcome: Literal["accepted", "unresolved", "refused"]
    stated: FindingPointer
    reasons: tuple[StrictStr, ...] = ()
    values: tuple[KeptValue, ...] = ()
    lines: tuple[SourceLine, ...] = ()
    quote: SourceLine | None = None
    documents: tuple[ReadDocument, ...] = ()
    answers: StrictStr | None = None
    found_in_code: bool = False

    @model_validator(mode="after")
    def _outcome_shape(self) -> ReadingFindingJudged:
        accepted = self.outcome == "accepted"
        if accepted != (self.quote is not None) or accepted == bool(self.reasons):
            raise ValueError(
                "an accepted finding keeps its quote and no reason; another, its reasons"
            )
        return self

    def judged(self) -> Judged:
        return Judged(
            stated=self.stated,
            outcome=self.outcome,
            reasons=self.reasons,
            values=self.values,
            lines=self.lines,
        )


class ReadingPlanKept(_Fact):
    """The paper item's plan as code kept it, recorded when kept: the genome version as
    the paper writes it, with its aliases and line; the organism as written; and the
    areas the question needs, each its own item; or why there are none."""

    event_type: Literal["ReadingPlanKept"] = "ReadingPlanKept"
    attempt_id: StrictStr
    reading_id: StrictStr
    reference: StatedReference | None = None
    organism: StatedOrganism | None = None
    areas: tuple[ReadingArea, ...] = ()
    not_answered: StrictStr | None = None
    documents: tuple[ReadDocument, ...] = ()


class ReadingItemCompleted(_Fact):
    """A reading item's terminal: it names every finding its Attempts recorded, in
    order, and, for the paper item, its plan; an area or id item's ``not_answered`` is
    the model's reason when it gives no finding."""

    event_type: Literal["ReadingItemCompleted"] = "ReadingItemCompleted"
    attempt_id: StrictStr
    reading_id: StrictStr
    item: ReadingItem
    finding_event_ids: tuple[StrictStr, ...] = ()
    plan_event_id: StrictStr | None = None
    not_answered: StrictStr | None = None


class ReadingItemNotRead(_Fact):
    """A stopped item its own Review's terminal answer settled as not read (ADR 0039):
    its Agent work Review's reject or dismiss, or its effort, model availability or
    operational Review's stop (``stopped``), the role settling a member for a stop of
    any kind (ADR 0031). Its findings stay recorded; the answer accounts it as not
    answered with the researcher's reason, where the answer gives one."""

    event_type: Literal["ReadingItemNotRead"] = "ReadingItemNotRead"
    reading_id: StrictStr
    item_id: StrictStr
    outcome: Literal["rejected", "dismissed", "stopped"]
    review_response_event_id: StrictStr
    # An operational stop may give no reason; every other terminal answer gives one.
    reason: StrictStr | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )


class PaperAnswerRecorded(_Fact):
    """The reading's answer once every item is settled: the fold of its facts
    (``settled_answer``), with the settling fact of each item (``item_event_ids``), the
    finding facts it rests on (``finding_event_ids``) and its proposed targets
    (``paper-targets/1``) when its paper names an id to assert them by."""

    event_type: Literal["PaperAnswerRecorded"] = "PaperAnswerRecorded"
    reading_id: StrictStr
    answer: PaperAnswer
    targets: PaperTargets | None = None
    item_event_ids: tuple[StrictStr, ...] = Field(min_length=1)
    finding_event_ids: tuple[StrictStr, ...] = ()


PAYLOADS = {
    fact.model_fields["event_type"].default: fact
    for fact in (
        PaperReadingStarted,
        PaperReadingRecorded,
        ReadingFindingJudged,
        ReadingPlanKept,
        ReadingItemCompleted,
        ReadingItemNotRead,
        PaperAnswerRecorded,
    )
}


# ---------------------------------------------------------------------------
# The checked view: a projection over the reading's facts
# ---------------------------------------------------------------------------

ItemState = Literal["open", "completed", "rejected", "dismissed", "stopped"]


class ItemSettlement(FrozenContract):
    """How an item settled: completed (``reason`` its ``not_answered``), or rejected,
    dismissed or stopped by its Review (``reason`` the researcher's)."""

    state: Literal["completed", "rejected", "dismissed", "stopped"]
    event_id: str
    reason: str | None = None


class RecordedFinding(FrozenContract):
    """One ``ReadingFindingJudged`` of an item; ``answered`` once a later finding of its
    item answers it, so it stands no more."""

    event_id: str
    fact: ReadingFindingJudged
    answered: bool = False


class PaperReading(FrozenContract):
    """One reading as its facts state it: its items in order (the paper item first, then
    its areas and each id code missed), each item's findings and settlement, the paper
    item's plan, and the answer."""

    reading_id: str
    paper_id: str
    paper_event_id: str
    question: str
    started_event_id: str
    documents: tuple[ReadDocument, ...]
    requested: tuple[IdLookup, ...] = ()
    paper: PaperIdentity
    items: tuple[ReadingItem, ...]
    plan_event_id: str | None = None
    plan: ReadingPlanKept | None = None
    findings: tuple[RecordedFinding, ...] = ()
    settled: dict[str, ItemSettlement] = {}
    answer_event_id: str | None = None

    def item(self, item_id: str) -> ReadingItem | None:
        return next((i for i in self.items if i.item_id == item_id), None)

    def state(self, item_id: str) -> ItemState:
        settlement = self.settled.get(item_id)
        return "open" if settlement is None else settlement.state

    def item_findings(self, item_id: str) -> tuple[RecordedFinding, ...]:
        """Every finding the item's Attempts recorded, in order."""
        return tuple(f for f in self.findings if f.fact.item_id == item_id)

    def standing(self, item_id: str) -> tuple[RecordedFinding, ...]:
        """The item's findings no later finding answered: what code holds for it."""
        return tuple(f for f in self.item_findings(item_id) if not f.answered)

    @property
    def all_settled(self) -> bool:
        return all(self.state(i.item_id) != "open" for i in self.items)


class PaperReadings(FrozenContract):
    readings: tuple[PaperReading, ...] = ()

    def reading(self, reading_id: str) -> PaperReading | None:
        return next((r for r in self.readings if r.reading_id == reading_id), None)

    def with_reading(self, reading: PaperReading) -> PaperReadings:
        others = tuple(r for r in self.readings if r.reading_id != reading.reading_id)
        if len(others) == len(self.readings):
            return PaperReadings(readings=(*self.readings, reading))
        return PaperReadings(
            readings=tuple(
                reading if r.reading_id == reading.reading_id else r
                for r in self.readings
            )
        )


def paper_item(question: str, requested: Sequence[IdLookup] = ()) -> ReadingItem:
    """The reading's first item: the question as the model reads it, each id code
    found replaced by their count."""
    return ReadingItem(
        item_id="paper",
        kind="paper",
        subject=model_question(
            question, {found.id for found in requested if found.found}
        ),
    )


def later_items(
    areas: Sequence[ReadingArea], requested: Sequence[IdLookup]
) -> tuple[ReadingItem, ...]:
    """The items the paper item's plan opens: each area, then each id code missed."""

    return (
        *(
            ReadingItem(
                item_id=f"area-{number}",
                kind="area",
                subject=area.area,
                views=area.views,
                reason=area.reason,
            )
            for number, area in enumerate(areas, start=1)
        ),
        *(
            ReadingItem(
                item_id=f"id-{number}", kind="id", subject=missed.id, near=missed.near
            )
            for number, missed in enumerate(
                (lookup for lookup in requested if not lookup.found), start=1
            )
        ),
    )


def _asserted(paper: PaperIdentity) -> bool:
    return any((paper.doi, paper.pmid, paper.pmcid, paper.paper_id))


def settled_answer(reading: PaperReading) -> PaperAnswerRecorded:
    """The answer the reading's facts fold to once every item is settled.

    Each item's standing findings in item order (the ids code found first, with the
    paper item), one variant one finding (``merged_findings``); the reference and
    organism the paper item's plan kept; and each item with no accepted finding
    accounted in ``not_answered`` by its reason, a rejected, dismissed or stopped item
    as not read with the researcher's (ADR 0039). An item settled as not read
    contributes none of the model's findings; the ids code found stay. Nothing here
    picks or caps a finding: every standing finding is in the answer."""

    if reading.answer_event_id is not None or not reading.all_settled:
        raise ValueError("a reading's answer follows every item's settlement")
    plan = reading.plan
    findings: list[KeptFinding] = []
    unresolved: list[FindingOutcome] = []
    refused: list[FindingOutcome] = []
    cited: list[str] = []
    reasons = [plan.not_answered] if plan is not None and plan.not_answered else []
    for item in reading.items:
        settlement = reading.settled[item.item_id]
        read = settlement.state == "completed"
        own = [
            f for f in reading.standing(item.item_id) if read or f.fact.found_in_code
        ]
        for recorded in own:
            judged = recorded.fact.judged()
            if judged.outcome == "accepted":
                findings.append(kept_finding(judged))
            else:
                (unresolved if judged.outcome == "unresolved" else refused).append(
                    finding_outcome(judged)
                )
            cited.append(recorded.event_id)
        label = "the paper item" if item.kind == "paper" else item.subject
        if not read:
            reasons.append(
                f"{label}: not read, {settlement.state} by the researcher"
                + (f": {settlement.reason}" if settlement.reason else "")
            )
        elif (
            item.kind != "paper"
            and settlement.reason
            and not any(f.fact.outcome == "accepted" for f in own)
        ):
            reasons.append(f"{label}: {settlement.reason}")
    answer = answer_of(
        reading.question,
        reading.paper,
        reading.documents,
        None if plan is None else plan.reference,
        merged_findings(findings),
        "; ".join(reasons) or None,
        unresolved,
        refused,
        organism=None if plan is None else plan.organism,
    )
    return PaperAnswerRecorded(
        reading_id=reading.reading_id,
        answer=answer,
        targets=paper_targets(answer) if _asserted(reading.paper) else None,
        item_event_ids=tuple(
            reading.settled[i.item_id].event_id for i in reading.items
        ),
        finding_event_ids=tuple(cited),
    )


# ---------------------------------------------------------------------------
# The fold: each reading fact checked against what it grounds on, then folded
# ---------------------------------------------------------------------------

# Each stop kind's answer, by the Review it answers, with the terminal answers that
# settle a member as not read (the outcome each records); every other answer is a
# continuation or an extension, the item's re-entry (ADR 0031, 0039).
TERMINAL_ANSWERS = {
    "AgentWorkReviewResponded": (
        "AgentWorkReviewRequested",
        {"reject": "rejected", "dismiss": "dismissed"},
    ),
    "AgentEffortReviewResponded": ("AgentEffortReviewRequested", {"stop": "stopped"}),
    "ModelAvailabilityReviewResponded": (
        "ModelAvailabilityReviewRequested",
        {"stop": "stopped"},
    ),
    "AgentFailureReviewResponded": (
        "AgentFailureReviewRequested",
        {"stop": "stopped"},
    ),
}

# The facts outside this family a reading fact grounds on, by event id.
_GROUNDS = frozenset(
    {
        "PaperRecorded",
        "PaperDocumentRead",
        *TERMINAL_ANSWERS,
        *(request for request, _ in TERMINAL_ANSWERS.values()),
    }
)


class _Grounds:
    """What a reading fact grounds on in its prefix: the Run paper and its read texts,
    each stop kind's Review and its answer, and each reader Attempt's member."""

    def event(self, event_id: str) -> Mapping[str, Any] | None:
        raise NotImplementedError

    def started(self, attempt_id: str) -> Mapping[str, Any] | None:
        raise NotImplementedError

    def member(self, attempt_id: str) -> str | None:
        from bio_neb.agentic_journey.attempt_journey import AttemptStarted

        started = self.started(attempt_id)
        if started is None:
            return None
        attempt = AttemptStarted.model_validate(started["payload"])
        return attempt.member_id if attempt.agent_role == "paper_reader" else None


class _Index(_Grounds):
    """The grounds a full fold gathers as it goes."""

    def __init__(self) -> None:
        self.events: dict[str, Mapping[str, Any]] = {}
        self.attempts: dict[str, Mapping[str, Any]] = {}

    def add(self, event: Mapping[str, Any]) -> None:
        kind = event["event_type"]
        if kind in _GROUNDS:
            self.events[event["event_id"]] = event
        elif kind == "AttemptStarted":
            self.attempts[event["payload"]["attempt_id"]] = event

    def event(self, event_id):
        return self.events.get(event_id)

    def started(self, attempt_id):
        return self.attempts.get(attempt_id)


class _Prefix(_Grounds):
    """The grounds a step finds in its admitted prefix, newest first."""

    def __init__(self, prefix: Sequence[Mapping[str, Any]]) -> None:
        self.prefix = prefix

    def event(self, event_id):
        return next(
            (
                e
                for e in reversed(self.prefix)
                if e["event_id"] == event_id and e["event_type"] in _GROUNDS
            ),
            None,
        )

    def started(self, attempt_id):
        return next(
            (
                e
                for e in reversed(self.prefix)
                if e["event_type"] == "AttemptStarted"
                and e["payload"]["attempt_id"] == attempt_id
            ),
            None,
        )


def _fact(event: Mapping[str, Any]):
    return PAYLOADS[event["event_type"]].model_validate(
        {"event_type": event["event_type"], **event["payload"]}
    )


def _open(state: PaperReadings, reading_id: str) -> PaperReading:
    reading = state.reading(reading_id)
    if reading is None or reading.answer_event_id is not None:
        raise ValueError("a reading fact names a started, unanswered reading")
    return reading


def _open_item(reading: PaperReading, item_id: str) -> ReadingItem:
    item = reading.item(item_id)
    if item is None or reading.state(item_id) != "open":
        raise ValueError("a reading fact names an open item of its reading")
    return item


def _own_attempt(grounds: _Grounds, attempt_id: str, reading_id: str, item_id: str):
    if grounds.member(attempt_id) != item_member(reading_id, item_id):
        raise ValueError("a reading fact is recorded by its own item's Attempt")


def _cites(
    reading: PaperReading,
    documents: Sequence[ReadDocument],
    lines: Sequence[SourceLine],
) -> None:
    if not set(documents) <= set(reading.documents) or not {
        line.document_id for line in lines
    } <= {d.document_id for d in documents}:
        raise ValueError("a reading fact cites the reading's documents by digest")


def _started(state, fact: PaperReadingStarted, event, grounds) -> PaperReading:
    from bio_neb.paper_intake.journey import PaperRecorded
    from bio_neb.paper_reading.reader import paper_identity

    paper = grounds.event(fact.paper_event_id)
    if (
        state.reading(fact.reading_id) is not None
        or paper is None
        or paper["event_type"] != "PaperRecorded"
        or paper["payload"]["paper_id"] != fact.paper_id
    ):
        raise ValueError("a paper reading names a recorded Run paper once")
    # Each document is the paper's, by the text its PaperDocumentRead retained.
    expected = tuple(
        (d["document_id"], d["read_event_id"], d["text_sha256"])
        for d in paper["payload"]["documents"]
    )
    named = tuple(
        (d.document_id, d.read_event_id, d.text_sha256) for d in fact.documents
    )
    if named != expected or any(
        (read := grounds.event(d.read_event_id)) is None
        or d.characters != read["payload"]["characters"]
        for d in fact.documents
    ):
        raise ValueError("a paper reading names the paper's stored documents")
    _own_attempt(grounds, fact.attempt_id, fact.reading_id, "paper")
    return PaperReading(
        reading_id=fact.reading_id,
        paper_id=fact.paper_id,
        paper_event_id=fact.paper_event_id,
        question=fact.question,
        started_event_id=event["event_id"],
        documents=fact.documents,
        requested=fact.requested,
        paper=paper_identity(PaperRecorded.model_validate(paper["payload"])),
        items=(paper_item(fact.question, fact.requested),),
    )


def _judged(state, fact: ReadingFindingJudged, event, grounds) -> PaperReading:
    reading = _open(state, fact.reading_id)
    _open_item(reading, fact.item_id)
    _own_attempt(grounds, fact.attempt_id, fact.reading_id, fact.item_id)
    if fact.found_in_code and fact.item_id != "paper":
        raise ValueError("a finding code found for a listed id is the paper item's")
    _cites(
        reading,
        fact.documents,
        (*fact.lines, *(v.source for v in fact.values), *filter(None, [fact.quote])),
    )
    if fact.quote is not None and fact.quote != kept_finding(fact.judged()).passages[0]:
        raise ValueError("an accepted finding's quote is the one code chooses")
    findings = list(reading.findings)
    if fact.answers is not None:
        number = next(
            (
                n
                for n, f in enumerate(findings)
                if f.event_id == fact.answers
                and f.fact.item_id == fact.item_id
                and f.fact.outcome == "refused"
                and not f.answered
            ),
            None,
        )
        if number is None:
            raise ValueError("a finding answers a standing refused finding of its item")
        findings[number] = findings[number].model_copy(update={"answered": True})
    findings.append(RecordedFinding(event_id=event["event_id"], fact=fact))
    return reading.model_copy(update={"findings": tuple(findings)})


def _planned(state, fact: ReadingPlanKept, event, grounds) -> PaperReading:
    reading = _open(state, fact.reading_id)
    _open_item(reading, "paper")
    if reading.plan is not None:
        raise ValueError("a reading's paper item keeps one plan")
    _own_attempt(grounds, fact.attempt_id, fact.reading_id, "paper")
    lines = [
        *(fact.reference.candidate.sources if fact.reference else ()),
        *((fact.organism.source, fact.organism.passage) if fact.organism else ()),
    ]
    _cites(reading, fact.documents, lines)
    return reading.model_copy(
        update={
            "plan": fact,
            "plan_event_id": event["event_id"],
            "items": (*reading.items, *later_items(fact.areas, reading.requested)),
        }
    )


def _completed(state, fact: ReadingItemCompleted, event, grounds) -> PaperReading:
    reading = _open(state, fact.reading_id)
    item = _open_item(reading, fact.item.item_id)
    if item != fact.item:
        raise ValueError("a reading item completes as its reading named it")
    _own_attempt(grounds, fact.attempt_id, fact.reading_id, item.item_id)
    if fact.finding_event_ids != tuple(
        f.event_id for f in reading.item_findings(item.item_id)
    ):
        raise ValueError("a reading item's completion names every finding it recorded")
    if item.kind == "paper":
        planned = (
            fact.plan_event_id is not None
            and fact.plan_event_id == reading.plan_event_id
            and fact.not_answered is None
        )
    else:
        planned = fact.plan_event_id is None
    if not planned:
        raise ValueError("the paper item completes with its kept plan, an area without")
    return reading.model_copy(
        update={
            "settled": {
                **reading.settled,
                item.item_id: ItemSettlement(
                    state="completed",
                    event_id=event["event_id"],
                    reason=fact.not_answered,
                ),
            }
        }
    )


def _not_read(state, fact: ReadingItemNotRead, event, grounds) -> PaperReading:
    reading = _open(state, fact.reading_id)
    _open_item(reading, fact.item_id)
    response = grounds.event(fact.review_response_event_id)
    asked, outcomes = (
        (None, {})
        if response is None
        else TERMINAL_ANSWERS.get(response["event_type"], (None, {}))
    )
    request = (
        None
        if asked is None
        else grounds.event(response["payload"]["request_event_id"])
    )
    if (
        request is None
        or request["event_type"] != asked
        or outcomes.get(response["payload"]["response"]["action"]) != fact.outcome
        or response["payload"]["response"]["reason"] != fact.reason
        or grounds.member(request["payload"]["attempt_id"])
        != item_member(fact.reading_id, fact.item_id)
    ):
        raise ValueError("an item is settled as not read by its own Review's answer")
    return reading.model_copy(
        update={
            "settled": {
                **reading.settled,
                fact.item_id: ItemSettlement(
                    state=fact.outcome, event_id=event["event_id"], reason=fact.reason
                ),
            }
        }
    )


def _answered(state, fact: PaperAnswerRecorded, event, grounds) -> PaperReading:
    reading = _open(state, fact.reading_id)
    if fact != settled_answer(reading):
        raise ValueError("a reading's answer is the fold of its settled items")
    return reading.model_copy(update={"answer_event_id": event["event_id"]})


_STEPS = {
    "PaperReadingStarted": _started,
    "ReadingFindingJudged": _judged,
    "ReadingPlanKept": _planned,
    "ReadingItemCompleted": _completed,
    "ReadingItemNotRead": _not_read,
    "PaperAnswerRecorded": _answered,
}


def _apply(
    state: PaperReadings, event: Mapping[str, Any], grounds: _Grounds
) -> PaperReadings:
    step = _STEPS.get(event["event_type"])
    if step is None:  # a reading Tool's fact changes no reading
        return state
    return state.with_reading(step(state, _fact(event), event, grounds))


def project_paper_readings(events: Sequence[Mapping[str, Any]]) -> PaperReadings:
    """Each reading of a Run paper: its items, findings, settlements and answer."""

    grounds = _Index()
    state = PaperReadings()
    for event in events:
        grounds.add(event)
        if event["event_type"] in PAYLOADS:
            state = _apply(state, event, grounds)
    return state


def step_paper_readings(
    state: PaperReadings, event: Mapping[str, Any], prefix: Sequence[Mapping[str, Any]]
) -> PaperReadings:
    """One appended fact folded into its prefix's view, checked against what it
    grounds on in that prefix; equal to ``project_paper_readings([*prefix, event])``."""

    if event["event_type"] not in _STEPS:
        return state
    return _apply(state, event, _Prefix(prefix))


__all__ = [
    "PAYLOADS",
    "ItemSettlement",
    "PaperAnswerRecorded",
    "PaperReading",
    "PaperReadingRecorded",
    "PaperReadingStarted",
    "PaperReadings",
    "ReadingFindingJudged",
    "ReadingItemCompleted",
    "ReadingItemNotRead",
    "ReadingPlanKept",
    "RecordedFinding",
    "TERMINAL_ANSWERS",
    "item_member",
    "member_item",
    "reading_members",
    "later_items",
    "paper_item",
    "project_paper_readings",
    "settled_answer",
    "step_paper_readings",
]
