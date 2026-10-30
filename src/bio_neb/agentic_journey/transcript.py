"""The transcript composer: what the model reads next (ADR 0044).

One owner decides, from the latest reply and everything waiting for the next
request, the kept reply and the ordered parts of that request. The framework
hooks only apply its answer, and reopen asks it the same question instead of
re-deriving the turn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bio_neb.agentic_journey.model_turn import (
    empty_reply,
    prompt_part,
    read_turn_record,
    retry_part,
)


def kept_reply(
    record: Mapping[str, Any], run: Sequence[str] | None = None
) -> dict[str, Any]:
    """A retained reply as the next request shows it: never a stub, never nothing.

    ``record`` is the raw reply as the Journey retains it. ``run`` names the
    calls that run, in run order; ``None`` keeps the reply as it came. A call
    stub's fragment is not kept, so the model never builds on it (#387
    celegans-a1). A kept reply with no call and no text is padded with empty
    text, as a reasoning-only reply was (#421): the model sees that it replied,
    and two user messages never follow each other.
    """
    kept = dict(record) if run is None else _only(record, run)
    if run is None and read_turn_record(record).unusable == "call_stub":
        kept["parts"] = [part for part in kept["parts"] if part["part_kind"] != "text"]
    if not any(part["part_kind"] in {"text", "tool-call"} for part in kept["parts"]):
        kept["parts"] = [*kept["parts"], empty_reply(None)["parts"][0]]
    return kept


def _only(record: Mapping[str, Any], run: Sequence[str]) -> dict[str, Any]:
    """Only the kept calls, in run order (ADR 0044).

    Each kept call keeps the parts that belong to it (a part naming the call in
    its provider details, such as its reasoning), just before it. The reply's
    other parts stay ahead of the calls, unless it was cut off: then only its
    kept calls remain, and they all ended, so the kept reply no longer ends on
    its ceiling. The provider's details describe the raw reply, so the kept
    reply carries none.
    """
    cut_off = record.get("finish_reason") == "length"
    calls = {}
    own: dict[str, list[Any]] = {}
    others = []
    for part in record["parts"]:
        if part["part_kind"] == "tool-call":
            calls[part["tool_call_id"]] = part
        elif (
            owner := (part.get("provider_details") or {}).get("tool_call_id")
        ) is not None:
            own.setdefault(owner, []).append(part)
        elif not cut_off:
            others.append(part)
    parts = [*others]
    for call_id in run:
        parts.extend((*own.get(call_id, ()), calls[call_id]))
    return {
        **record,
        "parts": parts,
        "finish_reason": "tool_call"
        if cut_off and run
        else record.get("finish_reason"),
        "provider_details": None,
    }


def corrected_outputs(
    events: Sequence[Mapping[str, Any]], response_event_id: str
) -> str | None:
    """The correction a reply's several output calls drew, or None (#373 E4)."""
    validations = {
        event["event_id"]: event["payload"]
        for event in events
        if event["event_type"] == "AgentOutputValidated"
    }
    return next(
        (
            validation.get("feedback")
            for event in events
            if event["event_type"] == "StructuredOutputValidationLinked"
            and event["payload"]["response_event_id"] == response_event_id
            and (validation := validations.get(event["payload"]["validation_event_id"]))
            and "multiple_output_calls" in validation["errors"]
            and validation.get("feedback") is not None
        ),
        None,
    )


def kept_message(
    events: Sequence[Mapping[str, Any]],
    response: Mapping[str, Any],
    answered: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """A retained reply as the loop kept it, read back from its turn's records.

    The same answer ``kept_reply`` gave live: a routed reply keeps its run order;
    a refused single call did not run, so its reply was kept with no call,
    unless the next request answers that call in its own slot (``answered``,
    #373); several corrected outputs keep none (#373 E4); any other reply,
    one refused for holding no call included, was kept as it came.
    """
    message = response["payload"]["compatible_adapter_evidence"]["response"]["message"]
    turns = [
        event
        for event in events
        if event["event_type"]
        in {"AgentTurnRouted", "AgentActionTurnDispositionRecorded"}
        and event["payload"]["response_event_id"] == response["event_id"]
    ]
    if corrected_outputs(events, response["event_id"]) is not None:
        return kept_reply(message, ())
    if any(
        event["event_type"] == "AgentActionTurnDispositionRecorded"
        and event["payload"]["disposition"] in {"text_refused", "reasoning_refused"}
        for event in turns
    ):
        # A reply with no call is refused before routing changes it, so it is kept
        # as it came even when routed for a resend.
        return kept_reply(message)
    for event in turns:
        if event["event_type"] == "AgentTurnRouted":
            return kept_reply(message, event["payload"]["run_order"])
    if any(event["payload"]["disposition"] == "singleton_refused" for event in turns):
        return kept_reply(
            message,
            [
                part["tool_call_id"]
                for part in message["parts"]
                if part["part_kind"] == "tool-call" and part["tool_call_id"] in answered
            ],
        )
    return kept_reply(message)


def reply_retry(correction: str | None, note: str | None) -> str:
    """The one retry part of a reply none of whose calls run.

    The reply's correction (several outputs refused, #373 E4), then the routed
    turn's note, which is the reason alone when there is no correction. A reply
    with no call is corrected alone: its turn owes no note (#444).
    """
    text = "\n\n".join(item for item in (correction, note) if item is not None)
    if not text:
        raise ValueError("a retry needs a correction or a note")
    return text


@dataclass(frozen=True)
class Pending:
    """What waits for the next request, as retained part records.

    ``parts`` are the request's parts as the framework built them: the Tool
    results of the calls that ran, or the reply's own retry part (its correction,
    or its note when nothing ran). ``answer`` is a refused single call's
    correction in that call's own slot (#373); ``note`` the routed turn's note,
    owed after its results; ``notices`` the injected messages the Attempt's
    policy would send now (ADR 0048).
    """

    parts: tuple[Mapping[str, Any], ...] = ()
    answer: Mapping[str, Any] | None = None
    note: str | None = None
    notices: tuple[str, ...] = ()


def request_parts(pending: Pending) -> list[dict[str, Any]]:
    """The next request's parts, in the order the model reads them.

    The results come first, then a refused single call's answer in its own
    slot, which replaces the retry part the framework wrote for it; the turn's
    note comes last. Each fault is one message (#373; every model, #445): a
    request that carries a correction (a retry part, a refused call's result)
    gets no notice, and a pattern the failure module writes into its correction
    never speaks (``FOLDED``); a notice joins after the results.
    """
    from bio_neb.agentic_journey.injection import injected_pattern
    from bio_neb.agentic_journey.injection_patterns import FOLDED
    from bio_neb.failure import Invariant

    parts = [dict(part) for part in pending.parts]
    if pending.answer is not None:
        if [
            (part["part_kind"], part["content"], part["tool_name"]) for part in parts
        ] != [("retry-prompt", pending.answer["content"], None)]:
            raise Invariant(
                "refused_call_slot_lost",
                "a refused call's correction did not follow its reply",
            )
        parts = [dict(pending.answer)]
    if pending.note is not None:
        parts.append(retry_part(pending.note))
    corrected = any(
        part["part_kind"] == "retry-prompt"
        or (part["part_kind"] == "tool-return" and part["outcome"] == "failed")
        for part in parts
    )
    if corrected:
        return parts
    return [
        *parts,
        *(
            prompt_part(text)
            for text in pending.notices
            if injected_pattern(text) not in FOLDED
        ),
    ]


def reply_between(
    previous: Mapping[str, Any], pending: Mapping[str, Any]
) -> dict[str, Any] | None:
    """The empty reply that goes between a request that got none and the retry
    that follows it, so the two are never sent as one turn of two user messages.

    A request whose reply never arrived (a stall, #443) is followed by one that
    carries only its note: the model reads that it replied nothing, then the
    note, as after any unusable reply (owner, #444 Q5). The empty reply is dated
    when its request was, so reopen rebuilds it exactly. A request that brings
    results for the one before it (a resumed call's) is the same turn, as the
    framework sends it; any other pair needs nothing between.
    """
    if (
        previous["kind"] == "request"
        and pending["kind"] == "request"
        and pending["parts"]
        and all(_loop_retry(part) for part in pending["parts"])
    ):
        return empty_reply(previous["timestamp"])
    return None


def check_request(
    request: Mapping[str, Any],
    kept: Mapping[str, Any],
    *,
    note: str | None,
) -> None:
    """Reopen asks the composer: is ``request`` its answer after ``kept``?

    ``request`` is the request recorded right after a reply, in the same
    transcript; ``kept`` that reply as the composer keeps it (``kept_message``);
    ``note`` the turn's note owed after its results, or None when none is owed.
    No retry follows the request it answers without the empty reply between,
    the history keeps the reply as kept, and the last request is the composer's
    order for its own parts: the note once and last, each injected message where
    the slot rule puts it. Raises ValueError otherwise.
    """
    messages = request["payload"]["compatible_adapter_evidence"]["request"]["messages"]
    if any(
        reply_between(previous, message) is not None
        for previous, message in zip(messages, messages[1:], strict=False)
    ):
        raise ValueError("request holds a retry right after a request it answers")
    if len(messages) < 2 or _material(messages[-2]) != _material(kept):
        raise ValueError("request changed the reply it keeps")
    _check_order(messages[-1]["parts"], note=note)


def check_stall_notes(events, wording) -> None:
    """Each request after a stall or a refusal is the composer's answer for it
    (#443, #444; #477 G1.5).

    It names the stall or refusal of its own Attempt's previous request, holds
    that request's transcript, then exactly the empty reply ``reply_between``
    puts after a request that got none, then its own request: the note (the
    stall note, or the provider's own words), the failure package's retry part
    re-rendered with the Attempt's own wording, once and last, in the composer's
    order. Raises ValueError otherwise.
    """
    from bio_neb.failure import ProviderRejected, ReplyStalled, render_correction

    notes = {
        "after_stall_event_id": (
            "ModelRequestStalled",
            "stall",
            lambda fact: ReplyStalled(seconds=fact["stall_seconds"]),
        ),
        "after_rejection_event_id": (
            "ModelRequestRejected",
            "refusal",
            lambda fact: ProviderRejected(
                status=fact["http_status_code"], statement=fact["provider_statement"]
            ),
        ),
    }
    by_id = {event["event_id"]: event for event in events}
    requests = {
        (event["payload"]["attempt_id"], event["payload"]["request_number"]): event
        for event in events
        if event["event_type"] == "ModelRequestRecorded"
    }
    for event in events:
        if event["event_type"] != "ModelRequestRecorded":
            continue
        payload = event["payload"]
        for field, (event_type, kind, correction) in notes.items():
            ended_id = payload.get(field)
            if ended_id is None:
                continue
            ended = by_id.get(ended_id)
            asked_event = requests.get(
                (payload["attempt_id"], payload["request_number"] - 1)
            )
            if (
                ended is None
                or ended["event_type"] != event_type
                or ended["payload"]["attempt_id"] != payload["attempt_id"]
                or ended["payload"]["request_number"] != payload["request_number"] - 1
                or asked_event is None
            ):
                raise ValueError(f"a request after a {kind} names another {kind}")
            note = render_correction(correction(ended["payload"]), wording)
            asked = asked_event["payload"]["compatible_adapter_evidence"]["request"][
                "messages"
            ]
            messages = payload["compatible_adapter_evidence"]["request"]["messages"]
            pending = messages[-1] if messages else None
            if (
                pending is None
                or pending["kind"] != "request"
                or not pending["parts"]
                or not _loop_retry(pending["parts"][-1])
                or pending["parts"][-1]["content"] != note
                or sum(part["content"] == note for part in pending["parts"]) != 1
            ):
                raise ValueError(f"a request after a {kind} lost its exact {kind} note")
            if (
                len(messages) != len(asked) + 2
                or [_material(m) for m in messages[: len(asked)]]
                != [_material(m) for m in asked]
                or _material(messages[-2])
                != _material(empty_reply(asked[-1]["timestamp"]))
            ):
                raise ValueError(
                    f"a request after a {kind} lost the empty reply before its note"
                )
            _check_order(pending["parts"], note=None)


def _check_order(pending: Sequence[Mapping[str, Any]], *, note: str | None) -> None:
    """The request's own parts are the composer's order for them."""
    from bio_neb.agentic_journey.injection import injected_pattern

    notices = tuple(
        part["content"]
        for part in pending
        if part["part_kind"] == "user-prompt" and injected_pattern(part["content"])
    )
    base = [
        part
        for part in pending
        if not (
            part["part_kind"] == "user-prompt" and injected_pattern(part["content"])
        )
    ]
    if note is not None:
        last = base[-1] if base else None
        if (
            last is None
            or not _loop_retry(last)
            or last["content"] != note
            or any(
                isinstance(part["content"], str) and note in part["content"]
                for part in base[:-1]
            )
        ):
            raise ValueError("routed turn lost its exact note")
        base = base[:-1]
    expected = request_parts(Pending(tuple(base), note=note, notices=notices))
    if _shapes(expected) != _shapes(pending):
        raise ValueError("request changed the order the composer gives its parts")


def _loop_retry(part: Mapping[str, Any]) -> bool:
    """A retry part the loop wrote: it answers no Tool call."""
    return part["part_kind"] == "retry-prompt" and part["tool_name"] is None


def _material(message: Mapping[str, Any]) -> dict[str, Any]:
    """A message without the framework's run bookkeeping, never sent to a model."""
    return {
        key: value
        for key, value in message.items()
        if key not in {"run_id", "conversation_id"}
    }


def _shapes(parts: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    """Each part as the model reads it; a retry part the loop wrote answers no
    call, so its framework id is not part of it."""
    shapes = []
    for part in parts:
        tool = (
            part["tool_name"]
            if part["part_kind"] in {"retry-prompt", "tool-return"}
            else None
        )
        shapes.append(
            (
                part["part_kind"],
                part["content"],
                tool,
                None if tool is None else part["tool_call_id"],
            )
        )
    return shapes
