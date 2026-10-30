"""The correction renderer: every correction a model reads is written here,
from the fault and the selected model's wording (Model endpoint policy, #222)."""

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
)
from pydantic_core import ErrorDetails

from bio_neb.content import shown_to_model

# The retry route's rendering: every correction a model reads is written here,
# from the fault and the selected model's wording (Model endpoint policy).


class FaultTuning(BaseModel):
    """One (fault kind, Tool surface) row: a line its correction adds, and its cap.

    The cap is the count that stops the Attempt; None keeps the Attempt's
    retained allowance. A cap of its own needs the retained continuation policy
    to carry it, so reopen checks the count the Attempt ran under (#387).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    line: str = ""
    cap: None = None


# The refusals of a value the request does not state as sent (literal_guard).
_LITERAL_GUARD_CODES = frozenset(
    {"request_assertion_ungrounded", "asserted_value_not_in_quote"}
)
# literal_guard's line on those refusals.
_EXACT_VALUES = (
    "Exact values: copy each value and quote from the request character for "
    "character, keeping its spelling, case, spaces and punctuation."
)
_SERVER_BOUND = (
    "- Field `{name}`: you sent {sent}.\n"
    "  Allowed form: omit it; the server supplies it.\n"
)


class CorrectionWording(BaseModel):
    """A model's correction sentences, as format templates.

    The Model endpoint policy carries one per model. The defaults are the one
    text every model reads (owner, #445 Q2 (c); Nemotron's text, #242, #373): each
    names the field, the value sent and the allowed form, one line each, with a
    literal example where the fault has one. An entry states only what differs.
    A routed reply's note says only what routing did, never whether a kept call
    ran (ADR 0044, "The note").
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    refused: str = "Refused before execution ({codes}).\n"
    tool_unknown: str = (
        "Tool `{tool}`: no such tool.\n{sent}Allowed: one of {callable}.{waiting}"
    )
    tool_unoffered: str = (
        "Tool `{tool}`: not offered now.{reason}\n"
        "{sent}Allowed now: one of {callable}.{waiting}"
    )
    not_run: str = (
        "Tool `{tool}` was not run{fix}\n"
        "{details}Allowed now: one of {callable}.{waiting}{guidance}"
    )
    # Appended as written: it has no fields to format.
    arguments_not_object: str = (
        "- Arguments: not one JSON object.\n  Allowed form: one JSON object of "
        'argument names to values, for example {"name": "value"}.\n'
    )
    argument_invalid: str = (
        "- Field `{location}`: you sent {sent}.\n  Allowed form: {reason}.\n"
    )
    argument_server_bound: str = _SERVER_BOUND
    # An argument refusal's code in the ``refused`` line: ``{error}`` is the whole
    # code, its validation message included; ``{kind}`` is its kind alone. One fact
    # per line, codes only in the ``refused`` line.
    argument_code: str = "{kind}"
    # A refusal reason's own sentence, from the role's registry or the invocation.
    explanation: str = "- {sentence}\n"
    # A server-bound value's refusal when the invocation explains it; it may use
    # ``{name}`` and ``{sent}`` as ``argument_server_bound`` does, and keeps the
    # server-bound form over the invocation's sentence.
    bound_explained: str = _SERVER_BOUND
    # The refused Tool's own repair text, after the offer.
    guidance: str = "\n{guidance}"
    # The offered Tools whose step or state is not current, after the offer, and
    # the sentence a refused Tool among them reads instead of ``fix``. Templates
    # may use ``{offered}`` (every offered Tool) or ``{callable}`` (the offered
    # Tools that do not wait); ADR 0040 keeps the others offered.
    waiting: str = "\nWaiting on their step, not callable now: {tools}."
    # No "never resend it unchanged": after a precondition refusal such as
    # `sample_unattested` the same call is right once the state changes; a Tool
    # that waits on its step is not called again now.
    fix: str = ". Fix each item, then call it again."
    fix_waiting: str = ": it waits on its step. Call an allowed Tool now."
    # Only a cut-off reply that held no call reads this, inside
    # ``cut_off_no_call``.
    truncation: str = (
        "Your reply was cut at {output_tokens} tokens before any tool call.\n"
        "{quirks}Allowed form: a short reply that calls one of {offered}."
    )
    # The action_commit notice's step, folded in; no researcher line, since a role
    # without a question Tool cannot ask.
    text_ending: str = (
        "Your reply ended in text without a tool call; text is not a result, so "
        "nothing happened.\n"
        "A step happens only through a tool call: make the call that does it.\n"
        "Allowed form: a tool call to one of {offered}."
    )
    # A reply that held only reasoning: no call arrived and no text.
    reasoning_only: str = (
        "Your reply held only reasoning; no tool call arrived, so nothing happened.\n"
        "Allowed form: a tool call to one of {offered}."
    )
    # A reply with no call, no text and no reasoning, and one the provider ended
    # on a tool call that brought only a fragment of text (Q5): each counts as
    # reasoning-only and names what came back.
    empty_reply: str = (
        "Your last reply came back empty: no tool call and no text arrived, so "
        "nothing happened.\n"
        "Allowed form: a tool call to one of {offered}."
    )
    call_stub: str = (
        "Your last reply ended as a tool call, but no call arrived: only a "
        "fragment of text, so nothing happened. Do not repeat the fragment.\n"
        "Allowed form: a whole tool call to one of {offered}."
    )
    output_schema: str = (
        "Your output has {count} error{plural}. In each, `loc` is the field, "
        "`input` is what you sent and `msg` is the allowed form:\n"
        "```json\n{errors}\n```"
    )
    output_refused: str = "Output refused: {reason}"
    # A role's refusal given as facts (ADR 0043): its codes once, then each
    # fault's own sentence as ``explanation`` writes it, then the output's fields.
    output_faults: str = "Output refused ({codes}).\n"
    output_fields: str = (
        "Allowed form: `{tool}` again with its fields {fields}, each fault above fixed."
    )
    # The turn's one note: one item per reply fault, each line one fact. A
    # model-contact count (a resend, a stall) names its kind; every other fault is
    # one more failed turn in a row.
    fault: str = "Count: {kind} fault {count} of {limit}."
    failed_turn: str = (
        "Count: failed turn {count} of {limit} in a row. At {limit} the Attempt "
        "stops and the researcher reviews it."
    )
    copied_calls: str = (
        "Copied calls: you sent {copies}.\n"
        "Kept: the first of each; the copies were dropped.\n"
        "Allowed form: each distinct call once per reply.\n{fault}"
    )
    unoffered_calls: str = (
        "Unoffered calls: this stage does not offer these Tools; each was "
        "refused before execution.\n{refusals}\n{fault}"
    )
    output_not_alone: str = (
        "Output {outputs}: sent with other calls in one reply; not accepted.\n"
        "Allowed form: {outputs} alone in its own reply, after you read this "
        "turn's results.\n{fault}"
    )
    cut_off_partial: str = (
        "Your reply was cut at {output_tokens} tokens; it held {calls}.\n"
        "{quirks}Last call `{tool}`: it reached the ceiling and was dropped.\n"
        "Allowed form: call `{tool}` again in a later reply if you still need "
        "it.\n{fault}"
    )
    cut_off_no_call: str = "{correction}\n{fault}"
    model_resend: str = (
        "No reply arrived: the provider timed out or dropped the connection.\n"
        "Sent: this request {sent} times.\n{fault}"
    )
    # A request whose reply did not arrive within the stall limit: the next
    # request carries this, so the model does not resend it.
    reply_stalled: str = (
        "No reply arrived within {seconds} s: it was cut off, and nothing of it "
        "was received.\n"
        "Allowed form: a shorter reply that makes one Tool call; do not send the "
        "same reply again."
    )
    # The endpoint refused the last request: what happened, in the provider's
    # words, then where to go.
    provider_rejected: str = (
        "The AI model service refused the last request (HTTP {status}). "
        "Its own words: {statement}\n"
        "Change what you send so the service accepts it; the same refusal again "
        "stops for the researcher."
    )
    # A Tool that ran and could not produce its fact: where to go next, for a
    # declared refusal and for an undeclared failure shown once.
    tool_refused_next: str = (
        "Call it again only with arguments that can produce its result, or call "
        "another Tool callable now."
    )
    tool_failed_next: str = (
        "Call another Tool callable now; the same failure again holds the Run for "
        "the researcher."
    )
    # The repeat_call notice's next step, folded in. It restates the earlier
    # call's status and next step from that call's model view.
    repeated_call: str = (
        "Refused before execution (repeated_call).\n"
        "Tool `{tool}` was not run: your earlier `{tool}` call with the same "
        "arguments ended `{status}`, and nothing has changed since. Its next step: "
        "{next}\n"
        "This call sends the same arguments, so it adds no evidence.\n"
        "Allowed now: one of {callable}.{waiting}"
    )
    # The fault notices' facts, folded into the one correction (option B):
    # refused_call_context's trace for a dropped call (``{arguments}``),
    # placeholder_argument's (``{location}``, ``{value}``) and empty_arguments'
    # when a refusal comes with them, the cut-off notices' (``{run}``), and
    # hint_as_user's as the firm line at a fault's last correction (``{kind}``),
    # which is the stop rule.
    sent_call: str = "You sent it with {arguments}.\n"
    placeholder: str = (
        "- Field `{location}`: you sent the placeholder {value}.\n"
        "  Allowed form: the value another call returns. Call that tool first, "
        "read its result, then send the real value in a later reply.\n"
    )
    no_arguments: str = (
        "- Arguments: you sent none.\n"
        "  Allowed form: each value in its own argument field.\n"
    )
    cut_off_text_call: str = (
        "It began a tool call as text in the message. Send each call in the "
        "tool-call field, with no text before it."
    )
    cut_off_blank_run: str = (
        "It held a run of {run} blank characters. End each reply right after its "
        "last tool call."
    )
    last_correction: str = (
        "Last correction: one more failed turn stops the Attempt for review."
    )
    last_contact: str = (
        "Last correction: repeating this will not work; one more {kind} fault "
        "stops the Attempt for review."
    )
    # Several output calls in one reply, a counted output repair.
    output_calls: str = (
        "Outputs: you sent `{tool}` {sent} times in one reply; none was accepted.\n"
        "Allowed form: one `{tool}` call alone in its own reply."
    )
    # The per-surface tuning seam: fault kind, then Tool name or "*";
    # literal_guard's lines for the refusals of a value the request does not
    # state as sent.
    surfaces: dict[str, dict[str, FaultTuning]] = {
        code: {"*": FaultTuning(line=_EXACT_VALUES)} for code in _LITERAL_GUARD_CODES
    }


# Each fault kind as the firm line names it; the tuning table keys a fault by
# this kind, or a refused call by its refusal code.
_KIND_WORDS = {
    "same_call": "same-refusal",
    "repeated_call": "repeat",
    "text_ending": "text-ending",
    "reasoning_only": "reasoning-only",
    "cut_off": "cut-off",
    "copied_calls": "copied-call",
    "unoffered_calls": "unoffered-call",
    "output_not_alone": "output-not-alone",
    "output_calls": "output",
    "output_refused": "output",
    "model_resend": "resend",
    "reply_stalled": "stall",
    "provider_rejected": "refused-request",
}


def names_waiting(wording: CorrectionWording) -> bool:
    """Whether this wording renders the offer's waiting Tools, so a refusal
    computes them only for a model that reads them."""
    return any(
        "{callable}" in text or "{waiting}" in text
        for text in (
            wording.not_run,
            wording.tool_unknown,
            wording.tool_unoffered,
            wording.repeated_call,
        )
    )


@dataclass(frozen=True)
class ActionRefused:
    """One refused Tool call: its reasons and the role's sentence per reason.

    ``explanations`` holds the role's registry sentence for each reason it
    explains; ``guidance`` is the refused Tool's own repair text, shown only while
    that Tool is offered.
    """

    tool_name: str
    errors: tuple[str, ...]
    arguments: Mapping[str, Any]
    offered: frozenset[str]
    explanations: Mapping[str, str] = field(default_factory=dict)
    guidance: str | None = None
    # The offered Tools whose precondition does not hold now (ADR 0040 keeps them
    # offered; the writer names them apart from the callable ones).
    waiting: frozenset[str] = frozenset()
    # The call's key count, for the firm line at its last correction.
    fault: "FaultCount | None" = None
    # Routing dropped the call, so the history the model reads next omits it.
    dropped: bool = False


@dataclass(frozen=True)
class ReplyCutOff:
    """A reply cut off on its output ceiling, with the calls the Model turn read.

    Alone it is the note for a reply that held no call; a routed reply's note
    names its partial call (``TurnFaults``). ``text_call`` and ``blank_run`` are
    what the model's adapter read in it: a call begun as text, and its longest
    run of blank characters.
    """

    output_tokens: int
    offered: frozenset[str]
    calls: int = 0
    text_call: bool = False
    blank_run: int = 0


@dataclass(frozen=True)
class ReplyStalled:
    """A request whose reply did not arrive within the stall limit (#443).

    Its one note is the first stall in a row of two: the next stall asks the
    availability Review, so the note is never sent twice in a row.
    """

    seconds: float
    fault: "FaultCount" = field(default_factory=lambda: FaultCount(1, 2))


@dataclass(frozen=True)
class ProviderRejected:
    """A request the model endpoint refused, in the provider's own words (owner,
    #476 alignment G1.5 B; #477).

    Its one note is the first refusal in a row of two: the same refusal of the
    request that carries it asks the Run-level Review, so it is never sent twice.
    """

    status: int
    statement: str
    fault: "FaultCount" = field(default_factory=lambda: FaultCount(1, 2))


@dataclass(frozen=True)
class TextEnding:
    """A reply that ended in text with no Tool call."""

    offered: frozenset[str]
    fault: "FaultCount | None" = None


@dataclass(frozen=True)
class ReasoningOnly:
    """A reply with no Tool call and nothing to act on, by what it held.

    ``reply`` is the Model turn's reading: only reasoning, nothing at all, or a
    call stub; all three count under the reasoning-only key (owner, #444 Q4, Q5)
    and each has its own sentence.
    """

    offered: frozenset[str]
    fault: "FaultCount | None" = None
    reply: Literal["reasoning_only", "empty_reply", "call_stub"] = "reasoning_only"


@dataclass(frozen=True)
class OutputFaults:
    """A role's own refusal of an output, as facts the rendering words (#448):
    each fault's code and sentence, and the fields of the output's type."""

    codes: tuple[str, ...]
    sentences: tuple[str, ...]
    fields: tuple[str, ...]


@dataclass(frozen=True)
class OutputRefused:
    """A refused output: its type's schema errors, the domain validator's reason,
    or the role's faults.

    ``fault`` is the member's output repair count, for the firm line at its last
    correction (#373 E3).
    """

    schema_errors: tuple[Mapping[str, Any], ...] | None = None
    reason: str | None = None
    fault: "FaultCount | None" = None
    # The output Tool refused, for its per-surface row and inspection.
    tool_name: str | None = None
    faults: OutputFaults | None = None


@dataclass(frozen=True)
class RepeatedCall:
    """A call stopped before contact: it repeats an earlier call with nothing changed.

    ``earlier_status`` and ``earlier_next`` are the earlier call's own model view:
    the status it ended with and its next step (#478 G1.4).
    """

    tool_name: str
    count: int
    cap: int
    offered: frozenset[str]
    earlier_status: str
    earlier_next: str
    waiting: frozenset[str] = frozenset()


@dataclass(frozen=True)
class FaultCount:
    """One fault key's count in the Attempt and the count that stops the Attempt."""

    count: int
    limit: int

    @property
    def last(self) -> bool:
        """This is the key's last correction: its next fault stops the Attempt."""
        return self.count == self.limit - 1


@dataclass(frozen=True)
class OutputCalls:
    """A reply that sent its output Tool more than once (#373 E4)."""

    tool_name: str
    sent: int
    fault: FaultCount


@dataclass(frozen=True)
class TurnFaults:
    """A routed reply's faults, named in the one note after the turn's results.

    ``copies`` holds each copied call's Tool and how many times the reply sent
    it; ``unoffered`` the refusal of each distinct unoffered call; ``outputs``
    the output Tools refused because the reply sent them beside calls that run.
    A cut-off reply names its tokens, the calls it held and its partial last
    call; ``resent`` how many times the request was sent after timing out.
    """

    copies: tuple[tuple[str, int], ...] = ()
    copies_fault: FaultCount | None = None
    unoffered: tuple[ActionRefused, ...] = ()
    unoffered_fault: FaultCount | None = None
    outputs: tuple[str, ...] = ()
    outputs_fault: FaultCount | None = None
    cut_off: ReplyCutOff | None = None
    dropped_tool: str | None = None
    cut_off_fault: FaultCount | None = None
    sent: int = 0
    resend_fault: FaultCount | None = None


Correction = (
    ActionRefused
    | ReplyCutOff
    | TextEnding
    | ReasoningOnly
    | OutputRefused
    | RepeatedCall
    | TurnFaults
    | OutputCalls
    | ReplyStalled
    | ProviderRejected
)

_SENT_LIMIT = 200
_ERROR_DETAILS = TypeAdapter(list[ErrorDetails])


def model_value_text(value: Any) -> str:
    """A value as the model reads it in a correction: JSON, cut when long.

    A long value is cut at ``_SENT_LIMIT`` characters; the model still has its call.
    """
    text = json.dumps(value, default=str)
    return text if len(text) <= _SENT_LIMIT else text[:_SENT_LIMIT] + "…"


def tool_list_text(tool_names: Iterable[str]) -> str:
    """Tool names as a correction lists them."""
    return ", ".join(f"`{name}`" for name in sorted(tool_names)) or "no Tool"


@dataclass(frozen=True)
class ExpectedValue:
    """A refused output's typed fact: a field and the value its source holds,
    and what the model sent when that differs (#372 D2)."""

    field: str
    expected: Any
    observed: Any = None
    sent: bool = False


@dataclass(frozen=True)
class OmitFields:
    """A refused output's typed fact: the fields it must leave out."""

    fields: tuple[str, ...]


RefusalDetail = str | ExpectedValue | OmitFields


def refusal_detail_text(detail: RefusalDetail) -> str:
    """One refusal detail as every model reads it and the record keeps it."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, OmitFields):
        return "omit: " + ", ".join(detail.fields)
    expected = json.dumps(_jsonable(detail.expected))
    if detail.sent:
        return (
            f"{detail.field}: observed={json.dumps(_jsonable(detail.observed))}; "
            f"expected={expected}"
        )
    return f"{detail.field}: expected={expected}"


def _jsonable(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _fact_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, tuple | list):
        return ", ".join(f"`{item}`" for item in value) or "none"
    return model_value_text(value)


def sentence_with_facts(sentence: str, facts: Mapping[str, Any]) -> str:
    """A reason sentence with its refusal's typed facts filled in (#372 D2).

    The facts are computed once, in the code that refuses; every model reads the
    same values. A sequence reads as a list of names in its own order, empty as
    "none"; every field the sentence names must be given. Only ``{name}`` fields
    are filled, so a sentence may quote a JSON example.
    """
    return _FACT_FIELD.sub(lambda match: _fact_text(facts[match[1]]), sentence)


_FACT_FIELD = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


def _offer(
    offered: frozenset[str], waiting: frozenset[str], wording: "CorrectionWording"
) -> dict[str, str]:
    """The offer's template fields: every offered Tool, the callable ones, and
    the line naming those that wait on their step."""
    waits = waiting & offered
    return {
        "offered": tool_list_text(offered),
        "callable": tool_list_text(offered - waits),
        "waiting": wording.waiting.format(tools=tool_list_text(waits)) if waits else "",
    }


_PLACEHOLDER = re.compile(r"<[^<>\s][^<>]*>")


def placeholders(
    value: Any, location: str = "", *, names_id: bool = False
) -> Iterator[tuple[str, str]]:
    """Each string argument left unfilled where a value another call returns
    belongs, with its dotted location: a placeholder such as "<EVENT_ID>" (#357
    N2), or a blank id (I-25), an id being a field named ``id``, ``*_id`` or an
    item of ``*_ids``."""
    if isinstance(value, str):
        if _PLACEHOLDER.fullmatch(value) or (names_id and not value.strip()):
            yield location, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from placeholders(
                item,
                f"{location}.{key}" if location else str(key),
                names_id=_names_id(str(key), item),
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from placeholders(
                item,
                f"{location}.{index}" if location else str(index),
                names_id=names_id,
            )


def _names_id(key: str, value: Any) -> bool:
    return (
        key == "id"
        or key.endswith("_id")
        or (key.endswith("_ids") and isinstance(value, list))
    )


def _tuned(wording: "CorrectionWording", kind: str, tool: str | None) -> str:
    """The line the (fault kind, Tool surface) row adds, on a line of its own."""
    rows = wording.surfaces.get(kind, {})
    row = (rows.get(tool) if tool is not None else None) or rows.get("*")
    return f"\n{row.line}" if row is not None and row.line else ""


# Model-contact counts: the contact window's, not failed turns.
_CONTACT_KINDS = frozenset({"model_resend", "reply_stalled"})
# Counts kept apart from the run of failed turns: the contact window's, and the
# endpoint's refusals of one request (two in a row ask the Run-level Review).
_COUNTED_APART = _CONTACT_KINDS | {"provider_rejected"}


def _firm(wording: "CorrectionWording", kind: str, counted: "FaultCount | None") -> str:
    """The firm line at a fault's last correction, on a line of its own."""
    line = wording.last_contact if kind in _CONTACT_KINDS else wording.last_correction
    if counted is None or not counted.last or not line:
        return ""
    return "\n" + line.format(kind=_KIND_WORDS.get(kind, kind))


def _sent(arguments: Any, location: str) -> str:
    """The value sent at a dotted argument location, or ``nothing``."""
    value = arguments
    for part in location.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return "nothing"
    return model_value_text(value)


def _render_refusal(refusal: ActionRefused, wording: CorrectionWording) -> str:
    errors = refusal.errors
    tool = refusal.tool_name
    # The refusal's own tuned lines; its count and firm line follow.
    after = "".join(
        dict.fromkeys(
            _tuned(wording, error.rpartition(":")[2], tool) for error in errors
        )
    )
    return _render_refused_call(refusal, wording) + after


def _render_refused_call(refusal: ActionRefused, wording: CorrectionWording) -> str:
    errors = refusal.errors
    tool = refusal.tool_name
    offer = _offer(refusal.offered, refusal.waiting, wording)
    codes = (
        wording.argument_code.format(error=error, kind=error.partition(":")[0])
        if error.startswith("tool_arguments_invalid:")
        else error
        for error in errors
    )
    refused = wording.refused.format(codes=", ".join(dict.fromkeys(codes)))
    # A dropped call is not in the history the model reads next, so its
    # correction says what it sent.
    trace = (
        wording.sent_call.format(
            tool=tool, arguments=model_value_text(refusal.arguments)
        )
        if refusal.dropped
        else ""
    )
    # A placeholder sent where another call's value belongs is that field's fact.
    held = (
        dict(placeholders(dict(refusal.arguments)))
        if wording.placeholder and isinstance(refusal.arguments, Mapping)
        else {}
    )

    def explained(code: str) -> str:
        sentence = refusal.explanations.get(code)
        return f" {sentence}" if sentence else ""

    if "tool_name_ineligible" in errors:
        return refused + wording.tool_unknown.format(tool=tool, sent=trace, **offer)
    if "tool_name_unavailable" in errors:
        return refused + wording.tool_unoffered.format(
            tool=tool, reason=explained("tool_name_unavailable"), sent=trace, **offer
        )
    details: list[str] = []
    for error in errors:
        name, _, code = error.partition(":")
        if error == "tool_arguments_invalid_json":
            details.append(wording.arguments_not_object)
        elif name == "tool_arguments_invalid":
            location, _, rest = code.partition(":")
            if not location and isinstance(refusal.arguments, Mapping):
                # A model-level error is about the call's one argument.
                (location,) = tuple(refusal.arguments)[:1] or ("arguments",)
            detail = (
                wording.placeholder.format(
                    location=location, value=json.dumps(held[location])
                )
                if location in held
                else wording.argument_invalid.format(
                    location=location,
                    sent=_sent(refusal.arguments, location),
                    reason=_field_reason(rest),
                )
            )
            if detail not in details:
                details.append(detail)
        elif code == "bound_value_mismatch":
            sentence = explained(error)
            sent = json.dumps(refusal.arguments.get(name))
            details.append(
                wording.bound_explained.format(
                    sentence=sentence.strip(), name=name, sent=sent
                )
                if sentence
                else wording.argument_server_bound.format(name=name, sent=sent)
            )
        elif sentence := explained(error):
            details.append(wording.explanation.format(sentence=sentence.strip()))
    for location, value in held.items():
        detail = wording.placeholder.format(location=location, value=json.dumps(value))
        if detail not in details:
            details.append(detail)
    if (
        refusal.arguments == {}
        and wording.no_arguments
        and any(error.startswith("tool_arguments_invalid:") for error in errors)
    ):
        details.append(wording.no_arguments)
    guidance = refusal.guidance if tool in refusal.offered else None
    return refused + wording.not_run.format(
        tool=tool,
        sent=trace,
        details="".join(details),
        fix=wording.fix_waiting if tool in refusal.waiting else wording.fix,
        guidance=wording.guidance.format(guidance=guidance) if guidance else "",
        **offer,
    )


def _field_reason(error: str) -> str:
    """A field error's message as the model reads it: a domain ValueError's own
    words, without the "Value error, " pydantic puts before them (#409)."""
    kind, _, message = error.partition(":")
    return message.removeprefix("Value error, ") if kind == "value_error" else message


def _calls_text(calls: int) -> str:
    return f"{calls} tool call{'s' if calls != 1 else ''}"


def _cut_off_facts(cut_off: ReplyCutOff, wording: CorrectionWording) -> str:
    """What the adapter read in a cut-off reply, one line each (#373)."""
    lines = []
    if cut_off.text_call and wording.cut_off_text_call:
        lines.append(wording.cut_off_text_call)
    if cut_off.blank_run and wording.cut_off_blank_run:
        lines.append(wording.cut_off_blank_run.format(run=cut_off.blank_run))
    return "".join(f"{line}\n" for line in lines)


def _copies_surface(faults: TurnFaults) -> str | None:
    """The Tool whose row a copied-call item reads: one Tool, or the group's "*"."""
    return faults.copies[0][0] if len(faults.copies) == 1 else None


def _outputs_surface(faults: TurnFaults) -> str | None:
    """The output Tool whose row an output-not-alone item reads, likewise."""
    return faults.outputs[0] if len(set(faults.outputs)) == 1 else None


def _render_turn_faults(faults: TurnFaults, wording: CorrectionWording) -> str:
    def fault(kind: str, counted: FaultCount) -> str:
        return count_line(wording, kind, counted)

    def item(kind: str, text: str, counted: FaultCount, tool: str | None) -> str:
        return text + _tuned(wording, kind, tool) + _firm(wording, kind, counted)

    lines = []
    if faults.copies:
        lines.append(
            item(
                "copied_calls",
                wording.copied_calls.format(
                    copies=", ".join(
                        f"`{tool}` {sent} times" for tool, sent in faults.copies
                    ),
                    fault=fault("copied_calls", faults.copies_fault),
                ),
                faults.copies_fault,
                _copies_surface(faults),
            )
        )
    if faults.unoffered:
        lines.append(
            item(
                "unoffered_calls",
                wording.unoffered_calls.format(
                    fault=fault("unoffered_calls", faults.unoffered_fault),
                    refusals="\n".join(
                        _render_refusal(refusal, wording)
                        for refusal in faults.unoffered
                    ),
                ),
                faults.unoffered_fault,
                None,
            )
        )
    if faults.outputs:
        lines.append(
            item(
                "output_not_alone",
                wording.output_not_alone.format(
                    outputs=tool_list_text(set(faults.outputs)),
                    fault=fault("output_not_alone", faults.outputs_fault),
                ),
                faults.outputs_fault,
                _outputs_surface(faults),
            )
        )
    if faults.cut_off is not None:
        counted = fault("cut_off", faults.cut_off_fault)
        if faults.dropped_tool is not None:
            text = wording.cut_off_partial.format(
                output_tokens=faults.cut_off.output_tokens,
                calls=_calls_text(faults.cut_off.calls),
                tool=faults.dropped_tool,
                quirks=_cut_off_facts(faults.cut_off, wording),
                fault=counted,
            )
        else:
            text = wording.cut_off_no_call.format(
                correction=render_correction(faults.cut_off, wording),
                fault=counted,
            )
        lines.append(item("cut_off", text, faults.cut_off_fault, faults.dropped_tool))
    if faults.sent:
        lines.append(
            wording.model_resend.format(
                sent=faults.sent, fault=fault("model_resend", faults.resend_fault)
            )
            + _tuned(wording, "model_resend", None)
        )
    return "\n".join(lines)


def _row(wording: CorrectionWording, kind: str, tool: str | None) -> str | None:
    rows = wording.surfaces.get(kind, {})
    if tool is not None and tool in rows and rows[tool].line:
        return f"{kind} / {tool}"
    return f"{kind} / *" if "*" in rows and rows["*"].line else None


def _refusal_templates(refusal: ActionRefused, wording: CorrectionWording) -> list[str]:
    if "tool_name_ineligible" in refusal.errors:
        names = ["refused", "tool_unknown"]
    elif "tool_name_unavailable" in refusal.errors:
        names = ["refused", "tool_unoffered"]
    else:
        names = ["refused", "not_run"]
        if isinstance(refusal.arguments, Mapping) and wording.placeholder:
            names += ["placeholder"] * any(placeholders(dict(refusal.arguments)))
        if refusal.arguments == {} and any(
            error.startswith("tool_arguments_invalid:") for error in refusal.errors
        ):
            names.append("no_arguments")
    if refusal.dropped:
        names.append("sent_call")
    return names


# The injection patterns folded into the correction (option B), by the
# template that now writes each one's line; literal_guard's are surface rows.
_FOLDED_TEMPLATES = {
    "text_ending": "action_commit",
    "repeated_call": "repeat_call",
    "sent_call": "refused_call_context",
    "last_correction": "hint_as_user",
    "cut_off_text_call": "text_form_tool_call_cut_off",
    "cut_off_blank_run": "whitespace_runaway",
}


def fault_trace(correction: Correction, wording: CorrectionWording) -> dict[str, Any]:
    """What a correction answers, for development inspection only (#373).

    The fault kinds it names, every Tool they affect, the wording templates and the
    per-surface rows that wrote it (a group of Tools reads the "*" row, as the
    renderer does), and the kinds at their last correction; never stored.
    """
    items: list[tuple[str, str | None, FaultCount | None]] = []
    rows: list[str | None] = []
    templates: list[str] = []
    if isinstance(correction, ActionRefused):
        items.append(("same_call", correction.tool_name, correction.fault))
        rows.extend(
            _row(wording, error.rpartition(":")[2], correction.tool_name)
            for error in correction.errors
        )
        templates += _refusal_templates(correction, wording)
    elif isinstance(correction, RepeatedCall):
        counted = FaultCount(correction.count, correction.cap)
        items.append(("repeated_call", correction.tool_name, counted))
        rows.append(_row(wording, "repeated_call", correction.tool_name))
        templates.append("repeated_call")
    elif isinstance(correction, TextEnding):
        items.append(("text_ending", None, correction.fault))
        rows.append(_row(wording, "text_ending", None))
        templates.append("text_ending")
    elif isinstance(correction, ReasoningOnly):
        items.append(("reasoning_only", None, correction.fault))
        rows.append(_row(wording, "reasoning_only", None))
        templates.append(correction.reply)
    elif isinstance(correction, ReplyCutOff):
        items.append(("cut_off", None, None))
        templates.append("truncation")
    elif isinstance(correction, ProviderRejected):
        items.append(("provider_rejected", None, correction.fault))
        rows.append(_row(wording, "provider_rejected", None))
        templates.append("provider_rejected")
    elif isinstance(correction, ReplyStalled):
        items.append(("reply_stalled", None, correction.fault))
        rows.append(_row(wording, "reply_stalled", None))
        templates.append("reply_stalled")
    elif isinstance(correction, OutputCalls):
        items.append(("output_calls", correction.tool_name, correction.fault))
        rows.append(_row(wording, "output_calls", correction.tool_name))
        templates.append("output_calls")
    elif isinstance(correction, OutputRefused):
        items.append(("output_refused", correction.tool_name, correction.fault))
        rows.append(_row(wording, "output_refused", correction.tool_name))
        templates += (
            ["output_faults", "explanation", "output_fields"]
            if correction.faults is not None
            else ["output_schema"]
            if correction.schema_errors is not None
            else ["output_refused"]
        )
    else:
        items.extend(
            ("copied_calls", tool, correction.copies_fault)
            for tool, _ in correction.copies
        )
        if correction.copies:
            templates.append("copied_calls")
            rows.append(_row(wording, "copied_calls", _copies_surface(correction)))
        items.extend(
            ("unoffered_calls", refusal.tool_name, correction.unoffered_fault)
            for refusal in correction.unoffered
        )
        if correction.unoffered:
            templates.append("unoffered_calls")
            rows.append(_row(wording, "unoffered_calls", None))
            for refusal in correction.unoffered:
                templates += _refusal_templates(refusal, wording)
                rows.extend(
                    _row(wording, error.rpartition(":")[2], refusal.tool_name)
                    for error in refusal.errors
                )
        items.extend(
            ("output_not_alone", tool, correction.outputs_fault)
            for tool in correction.outputs
        )
        if correction.outputs:
            templates.append("output_not_alone")
            rows.append(_row(wording, "output_not_alone", _outputs_surface(correction)))
        if correction.cut_off is not None:
            items.append(("cut_off", correction.dropped_tool, correction.cut_off_fault))
            rows.append(_row(wording, "cut_off", correction.dropped_tool))
            templates += (
                ["cut_off_partial"]
                if correction.dropped_tool is not None
                else ["cut_off_no_call", "truncation"]
            )
        if correction.sent:
            items.append(("model_resend", None, correction.resend_fault))
            rows.append(_row(wording, "model_resend", None))
            templates.append("model_resend")
        templates.append("fault")
    if (counted := _counted(correction)) is not None:
        templates.append("fault" if counted[0] in _COUNTED_APART else "failed_turn")
    cut_off = (
        correction
        if isinstance(correction, ReplyCutOff)
        else getattr(correction, "cut_off", None)
    )
    if cut_off is not None:
        templates += ["cut_off_text_call"] * cut_off.text_call
        templates += ["cut_off_blank_run"] * bool(cut_off.blank_run)
    last = sorted({kind for kind, _, counted in items if counted and counted.last})
    if last and wording.last_correction:
        templates.append("last_correction")
    surface_rows = sorted(set(filter(None, rows)))
    # The rows are those the renderer read; the Tools are every Tool affected.
    return {
        "fault_kinds": sorted({kind for kind, _, _ in items}),
        "tools": sorted({tool for _, tool, _ in items if tool is not None}),
        "templates": sorted(set(templates)),
        "surface_rows": surface_rows,
        "last_correction": last,
        # When a folded pattern triggered: the line it once sent is in this
        # correction.
        "folded": sorted(
            {_FOLDED_TEMPLATES[name] for name in templates if name in _FOLDED_TEMPLATES}
            | {
                "literal_guard"
                for row in surface_rows
                if row.partition(" / ")[0] in _LITERAL_GUARD_CODES
            }
        ),
    }


def render_correction(correction: Correction, wording: CorrectionWording) -> str:
    """The one writer of every correction the model reads (#222).

    Each reads in one shape (#478 G1.4): what happened, in the source's or the
    provider's own words when there are any, then where to go next, then the
    reply's place in its count (``count_line``, F8), then the firm line at its last
    correction. The loop sends it within the fault's allowance; ``wording`` is the
    selected model's, from its Model endpoint policy. A correction can quote what
    the model sent; it is shown as every Tool result is (``shown_to_model``, #422).
    """
    text = _rendered(correction, wording)
    counted = _counted(correction)
    if counted is not None:
        kind, count = counted
        text += "\n" + count_line(wording, kind, count)
        # The provider's refusal states what the same refusal again does.
        if kind != "provider_rejected":
            text += _firm(wording, kind, count)
    return shown_to_model(text)


def execution_refused_view(
    reason: str, statement: str, fault: FaultCount, wording: CorrectionWording
) -> dict[str, str]:
    """A refused execution's result as the model is shown it, in the one message
    shape (#477 C8; #478 G1.4): what happened in the Tool's own words, where to go
    next, and the reply's place in the failed turns in a row."""
    from bio_neb.failure._faults import TOOL_FAILED

    return {
        "status": "refused",
        "reason": reason,
        "statement": statement,
        "next": wording.tool_failed_next
        if reason == TOOL_FAILED
        else wording.tool_refused_next,
        "count": count_line(wording, "same_call", fault),
    }


def count_line(wording: CorrectionWording, kind: str, counted: FaultCount) -> str:
    """The one line naming a correction's place in its count: a failed turn's in
    the Attempt's run of failed turns, a model-contact fault's in its own."""
    line = wording.fault if kind in _COUNTED_APART else wording.failed_turn
    return line.format(
        kind=_KIND_WORDS.get(kind, kind), count=counted.count, limit=counted.limit
    )


def _counted(correction: Correction) -> tuple[str, FaultCount] | None:
    """The fault kind and count a single correction reads its place in; a turn's
    note names each item's own."""
    if isinstance(correction, ActionRefused):
        counted = correction.fault
        kind = "same_call"
    elif isinstance(correction, RepeatedCall):
        return "repeated_call", FaultCount(correction.count, correction.cap)
    elif isinstance(correction, TextEnding):
        counted, kind = correction.fault, "text_ending"
    elif isinstance(correction, ReasoningOnly):
        counted, kind = correction.fault, "reasoning_only"
    elif isinstance(correction, OutputRefused):
        counted, kind = correction.fault, "output_refused"
    elif isinstance(correction, OutputCalls):
        counted, kind = correction.fault, "output_calls"
    elif isinstance(correction, ReplyStalled):
        counted, kind = correction.fault, "reply_stalled"
    elif isinstance(correction, ProviderRejected):
        counted, kind = correction.fault, "provider_rejected"
    else:
        return None
    return None if counted is None else (kind, counted)


def _rendered(correction: Correction, wording: CorrectionWording) -> str:
    if isinstance(correction, ActionRefused):
        return _render_refusal(correction, wording)
    if isinstance(correction, ReplyCutOff):
        return wording.truncation.format(
            output_tokens=correction.output_tokens,
            offered=tool_list_text(correction.offered),
            quirks=_cut_off_facts(correction, wording),
        )
    if isinstance(correction, RepeatedCall):
        return wording.repeated_call.format(
            tool=correction.tool_name,
            status=correction.earlier_status,
            next=correction.earlier_next,
            count=correction.count,
            cap=correction.cap,
            **_offer(correction.offered, correction.waiting, wording),
        ) + _tuned(wording, "repeated_call", correction.tool_name)
    if isinstance(correction, TurnFaults):
        return _render_turn_faults(correction, wording)
    if isinstance(correction, ReplyStalled):
        return wording.reply_stalled.format(seconds=f"{correction.seconds:g}") + _tuned(
            wording, "reply_stalled", None
        )
    if isinstance(correction, ProviderRejected):
        # The template states what the same refusal again does.
        return wording.provider_rejected.format(
            status=correction.status, statement=correction.statement
        ) + _tuned(wording, "provider_rejected", None)
    if isinstance(correction, TextEnding):
        return wording.text_ending.format(
            offered=tool_list_text(correction.offered)
        ) + _tuned(wording, "text_ending", None)
    if isinstance(correction, ReasoningOnly):
        sentence = {
            "reasoning_only": wording.reasoning_only,
            "empty_reply": wording.empty_reply,
            "call_stub": wording.call_stub,
        }[correction.reply]
        return sentence.format(offered=tool_list_text(correction.offered)) + _tuned(
            wording, "reasoning_only", None
        )
    if isinstance(correction, OutputCalls):
        return wording.output_calls.format(
            tool=correction.tool_name, sent=correction.sent
        ) + _tuned(wording, "output_calls", correction.tool_name)
    after = _tuned(wording, "output_refused", correction.tool_name)
    if correction.faults is not None:
        faults = correction.faults
        return (
            wording.output_faults.format(codes=", ".join(faults.codes))
            + "".join(
                wording.explanation.format(sentence=sentence)
                for sentence in faults.sentences
            )
            + wording.output_fields.format(
                tool=correction.tool_name, fields=", ".join(faults.fields)
            )
            + after
        )
    if correction.schema_errors is not None:
        errors = list(correction.schema_errors)
        return (
            wording.output_schema.format(
                count=len(errors),
                plural="" if len(errors) == 1 else "s",
                errors=_ERROR_DETAILS.dump_json(
                    errors, exclude={"__all__": {"ctx"}}, indent=2
                ).decode(),
            )
            + after
        )
    return wording.output_refused.format(reason=correction.reason) + after
