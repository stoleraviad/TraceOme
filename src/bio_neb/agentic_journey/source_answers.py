"""What a source Tool shows the model of each answer its source sent (#458).

A source Tool returns its typed result and, beside it, every GET's answer as the source
sent it: its operation, request, HTTP status, size and digest, and its body. A body is
shown whole when it fits the return's share; otherwise it is cut, and every cut says
how much it left out, so the model reads the source's own words and knows what it did
not see. The same bytes are always shown the same way. The whole body stays retained
under its digest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue, StrictInt, StrictStr, TypeAdapter

from bio_neb.content import canonical_json
from bio_neb.contracts import StrictFrozenContract

SOURCE_ANSWER_BYTES = 16 * 1024
"""The body bytes one Tool return shows, shared by its answers (owner, #458 Q1): the UTF-8
of each shown body's canonical JSON, or of its text."""

# Each step keeps fewer items of every list and object and fewer characters of every
# string, at every depth, until the body fits its share.
_CUTS = (
    (256, 4096),
    (128, 2048),
    (64, 1024),
    (32, 512),
    (16, 256),
    (8, 128),
    (4, 64),
    (2, 32),
    (1, 16),
    (0, 16),
)


_NOT_AN_ANSWER = frozenset({401, 403, 407, 429})


def source_answered(status_code: int) -> bool:
    """Whether a status is the source's answer about the request, for its reader.

    A success, a redirect or a client status is (a 404's negative, a 410's retired
    record, a 400's refusal of the key, a 301's body naming where the record went):
    the source said something about the request and its body is the evidence. A
    redirect is read, never followed (#458 Spec r1 SP4). An access fault, a rate limit
    or a server fault is not an answer.
    """
    return 200 <= status_code < 500 and status_code not in _NOT_AN_ANSWER


@dataclass(frozen=True)
class AnsweredGet:
    """One GET's answer as received: what was asked and the bytes that came back."""

    operation: str
    request_identity: str
    status_code: int
    body: bytes
    # The earlier retained GET this answer is, when a Tool read it in place of a
    # contact of its own (a cited prior source).
    cited_event_id: str | None = None


class SourceAnswer(StrictFrozenContract):
    """One GET's answer as the model is shown it."""

    operation: StrictStr
    request_identity: StrictStr
    status_code: StrictInt = Field(ge=100, le=599)
    byte_count: StrictInt = Field(ge=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    # ``json``: the body parsed; ``text``: the body decoded, as it is not JSON.
    media: Literal["json", "text"]
    shown: Literal["whole", "excerpt"]
    body: JsonValue
    # A cited prior source: the earlier retained GET whose bytes the Tool read in
    # place of its own contact; left out for the Tool's own GET.
    cited_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def shown_answers(
    answers: Sequence[AnsweredGet], budget: int = SOURCE_ANSWER_BYTES
) -> tuple[SourceAnswer, ...]:
    """Each answer in call order, its body whole or cut to its share of ``budget`` bytes.

    Answers that fit an even share stay whole and leave theirs to the rest; the larger
    ones share what is left evenly.
    """
    return _rendered(
        answers, [hashlib.sha256(answer.body).hexdigest() for answer in answers], budget
    )


def _rendered(
    answers: Sequence[AnsweredGet], digests: Sequence[str], budget: int
) -> tuple[SourceAnswer, ...]:
    read = [_read(answer.body) for answer in answers]
    sizes = [_size(media, body) for media, body in read]
    # ``None``: shown whole; a number: the bytes its excerpt may take.
    shares: dict[int, int | None] = {}
    order = sorted(range(len(answers)), key=lambda i: (sizes[i], i))
    left = budget
    for position, index in enumerate(order):
        share = left // (len(order) - position)
        if sizes[index] > share:
            shares.update(dict.fromkeys(order[position:], share))
            break
        shares[index] = None
        left -= sizes[index]
    return tuple(
        _shown(answer, digests[index], media, body, shares[index])
        for index, (answer, (media, body)) in enumerate(zip(answers, read, strict=True))
    )


_SHOWN = TypeAdapter(tuple[SourceAnswer, ...])


def validate_shown_answers(
    timeline: Sequence[Mapping[str, object]], *, journey
) -> None:
    """Refuse any fact whose ``source_answers`` are not exactly how its retained bytes
    show (#458 Standards r1 ST1).

    Each answer's sha256 names retained bytes; re-rendering every answer from them
    (`shown_answers`) must give the recorded answers back, body, media and cut
    included. The bytes are read from ``journey``, the owner whose artifacts these
    are, once per fact for that owner (`Journey.derived_once`).
    """
    retained = {
        reference["artifact_id"]: reference
        for event in timeline
        for reference in event["artifact_references"]
    }
    for event in timeline:
        if "source_answers" not in event["payload"]:
            continue

        def check(event=event) -> None:
            recorded = _SHOWN.validate_python(event["payload"]["source_answers"])
            gets = []
            for answer in recorded:
                reference = retained.get(f"sha256:{answer.sha256}")
                if reference is None:
                    raise ValueError("a source answer names bytes no event retains")
                gets.append(
                    AnsweredGet(
                        answer.operation,
                        answer.request_identity,
                        answer.status_code,
                        journey.read_artifact(reference),
                        answer.cited_event_id,
                    )
                )
            # The read verified each body against its content address, so its digest
            # is the answer's, not hashed again.
            digests = [answer.sha256 for answer in recorded]
            if _rendered(gets, digests, SOURCE_ANSWER_BYTES) != recorded:
                raise ValueError("a source answer is not how its retained bytes show")

        journey.derived_once(("source_answers", event["event_id"]), check)


_collected: ContextVar[tuple[str, list[AnsweredGet]] | None] = ContextVar(
    "collected_source_answers", default=None
)


@contextmanager
def collected_answers(operation: str) -> Iterator[list[AnsweredGet]]:
    """Every answer a transport reports while ``operation`` runs, in order: how a
    source Tool whose adapter hides its GETs (planning's NCBI Datasets reads) keeps
    them."""
    answers: list[AnsweredGet] = []
    token = _collected.set((operation, answers))
    try:
        yield answers
    finally:
        _collected.reset(token)


def answer_received(request_identity: str, status_code: int, body: bytes) -> None:
    """Report one GET's answer to the collection running now; outside one, nothing."""
    collecting = _collected.get()
    if collecting is not None and source_answered(status_code):
        operation, answers = collecting
        answers.append(AnsweredGet(operation, request_identity, status_code, body))


def _shown(
    answer: AnsweredGet, sha256: str, media: str, body: JsonValue, share: int | None
) -> SourceAnswer:
    if share is None:
        shown, kept = "whole", body
    else:
        shown, kept = "excerpt", _cut(media, body, share)
    return SourceAnswer(
        operation=answer.operation,
        request_identity=answer.request_identity,
        status_code=answer.status_code,
        byte_count=len(answer.body),
        sha256=sha256,
        media=media,
        shown=shown,
        body=kept,
        cited_event_id=answer.cited_event_id,
    )


def _read(body: bytes) -> tuple[str, JsonValue]:
    """The body as JSON a Journey can hold (no NaN or infinity, an overflowing number
    such as ``1e999`` included), or else as its text."""
    try:
        value = json.loads(body)
        canonical_json(value)
    except (ValueError, UnicodeDecodeError):
        return "text", body.decode("utf-8", errors="replace")
    return "json", value


def _size(media: str, body: JsonValue) -> int:
    """A shown body's bytes: its canonical JSON's UTF-8, or its text's."""
    return len((canonical_json(body) if media == "json" else body).encode("utf-8"))


def _cut(media: str, body: JsonValue, share: int) -> JsonValue:
    if media == "text":
        return _cut_text(body, share)
    for items, characters in _CUTS:
        kept = _cut_value(body, items, characters)
        if _size(media, kept) <= share:
            return kept
    return _cut_value(body, 0, 0)


def _cut_text(text: str, share: int) -> str:
    """The longest head of ``text`` that fits ``share`` bytes with its marker; a cut
    falls between characters."""

    def cut(keep: int) -> str:
        return f"{text[:keep]}… {len(text) - keep} more characters"

    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if len(cut(middle).encode("utf-8")) <= share:
            low = middle
        else:
            high = middle - 1
    return cut(low)


def _cut_value(value: JsonValue, items: int, characters: int) -> JsonValue:
    if isinstance(value, str):
        if len(value) <= characters:
            return value
        return f"{value[:characters]}… {len(value) - characters} more characters"
    if isinstance(value, list):
        kept = [_cut_value(item, items, characters) for item in value[:items]]
        if len(value) > items:
            kept.append(f"… {len(value) - items} more items")
        return kept
    if isinstance(value, dict):
        names = list(value)
        kept = {
            name: _cut_value(value[name], items, characters) for name in names[:items]
        }
        if len(names) > items:
            kept["…"] = f"{len(names) - items} more keys"
        return kept
    return value


__all__ = [
    "SOURCE_ANSWER_BYTES",
    "AnsweredGet",
    "SourceAnswer",
    "answer_received",
    "collected_answers",
    "shown_answers",
    "source_answered",
    "validate_shown_answers",
]
