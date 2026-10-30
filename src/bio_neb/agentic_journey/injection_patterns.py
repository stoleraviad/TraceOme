"""The registered injection patterns (ADR 0048).

Each pattern reads only ``InjectionSignals`` and writes one notice: guidance for
the model's next reply, never a copy of a correction or note the pending request
carries, and never a stop. The tag the model reads is in the notice's content
(``<loop_notice>``; owner, #279 review); the slot's ``[injection:<name>]`` line names
the pattern in the record. Thresholds start loose and are tuned per scope by the
eval's review (#243; owner R3: "why wait till it did 3 time?").

The seven loop patterns come from the Deep Agents Nemotron profile, NVIDIA's
loop notes and cards C6, C11-C13; the adapter quirk triggers read the reply's
recorded quirks (#241, #242); the gap patterns answer live findings (#336 F1/F3,
lane P #334, #357 N2). A model switches them on in its endpoint policy.
This module imports no ``pydantic_ai``.
"""

import json
import re
from collections.abc import Mapping
from typing import Annotated

from pydantic import Field

from bio_neb.agentic_journey.injection import (
    PATTERNS,
    InjectionPattern,
    InjectionSignals,
    PatternTuning,
)
from bio_neb.failure import model_value_text, placeholders

_TAG = "loop_notice"


def _notice(*lines: str) -> str:
    return f"<{_TAG}>\n" + "\n".join(lines) + f"\n</{_TAG}>"


# Configuration arrives as JSON, so a list of names is accepted for a tuple.
_Names = Annotated[tuple[str, ...], Field(strict=False)]


class _Capped(PatternTuning):
    """A pattern that speaks at most ``max_sends`` times in an Attempt."""

    max_sends: int = Field(default=2, ge=0)


def _capped(name, speak):
    """The pattern's message while its sends in the Attempt stay under the cap."""

    def message(signals: InjectionSignals, tuning: _Capped) -> str | None:
        if signals.sent.get(name, 0) >= tuning.max_sends:
            return None
        return speak(signals, tuning)

    return message


def _register(name, tuning, speak) -> None:
    PATTERNS[name] = InjectionPattern(name, tuning, _capped(name, speak))


# action_commit: a reply that explains instead of acting (Deep Agents before_model).
def _action_commit(signals, _tuning):
    if signals.reply is None or not signals.reply.text_ended:
        return None
    # Folded into the text-ending correction, so it never speaks; it names no
    # ask Tool, since a role without one cannot ask.
    return _notice(
        "Your last reply described the next step in text, so nothing happened.",
        "A step happens only through a tool call: make the call that does it.",
    )


# plan_review: review the plan before its first draft (from NVIDIA's agent
# guidance): keep stated values exact, find what is unstated, and order dependent
# calls.
class _PlanReview(_Capped):
    stages: _Names = ("proposal_requirements", "source_reading")


def _plan_review(signals, tuning):
    if signals.stage not in tuning.stages or signals.stage_request_number != 1:
        return None
    return _notice(
        "Plan review, before you call a tool in this stage:",
        "- Keep each value the request states exactly as written.",
        "- A fact the request does not state is one to find: never fill it with a "
        "guess or a placeholder.",
        "- Order the calls: a call that needs another call's result comes after "
        "it, in a later reply.",
    )


# literal_guard: a refused value the request does not state as sent (Deep Agents
# after_agent, card C6).
class _LiteralGuard(_Capped):
    codes: _Names = (
        "request_assertion_ungrounded",
        "asserted_value_not_in_quote",
    )


def _literal_guard(signals, tuning):
    if signals.reply is None or not any(
        error.rpartition(":")[2] in tuning.codes
        for refusal in signals.reply.refusals
        for error in refusal.codes
    ):
        return None
    return _notice(
        "Exact values: copy each value and quote from the request character for "
        "character.",
        "Keep its spelling, case, spaces and punctuation; do not shorten, expand "
        "or reword it.",
    )


# result_suffix: a Tool result that is one page or slice of a longer one (from
# NVIDIA's agent guidance). The notice follows the result; the result's own bytes
# stay as recorded (ADR 0034).
class _ResultSuffix(_Capped):
    max_sends: int = Field(default=3, ge=0)


def _result_suffix(signals, _tuning):
    for result in signals.results:
        content = result.content
        if not isinstance(content, Mapping):
            continue
        tool = result.tool_name
        if content.get("has_earlier_matches") and content.get("events"):
            oldest = content["events"][0]["event_id"]
            return _notice(
                f"The `{tool}` result above is one page: earlier matches exist.",
                f'To read them, call `{tool}` again with before_event_id "{oldest}"; '
                "skip it if this page answers your question.",
            )
        piece = content.get("payload_slice")
        if isinstance(piece, Mapping) and piece.get("next_offset") is not None:
            cursor = json.dumps(
                {"event_id": piece["event_id"], "offset": piece["next_offset"]}
            )
            return _notice(
                f"The `{tool}` result above is one slice of a longer payload: "
                f"{piece['next_offset']} of {piece['total_characters']} characters.",
                f"To read on, call `{tool}` with payload_slice {cursor}; skip it if "
                "this slice answers your question.",
            )
    return None


# repeat_call: after the loop guard's refusal of a repeat, the next step; the
# refusal itself already says what repeated and its count.
class _RepeatCall(_Capped):
    max_sends: int = Field(default=3, ge=0)


_REPEATED_CALL = "repeated_call"


def _repeat_call(signals, _tuning):
    repeated = next(
        (
            refusal
            for refusal in (signals.reply.refusals if signals.reply else ())
            if refusal.codes == (_REPEATED_CALL,)
        ),
        None,
    )
    if repeated is None:
        return None
    return _notice(
        f"`{repeated.tool_name}` already ran with these arguments; its result is in "
        "the history above.",
        "Next: use that result. Change the arguments only to get different "
        "evidence, or call another tool.",
    )


# progress_notice: the Deep Agents progress caps (16 turns, 48 Tool results) as a
# notice, first after request 4, then every few requests; the request budget
# still ends the Attempt (ADR 0048).
class _ProgressNotice(_Capped):
    max_sends: int = Field(default=25, ge=0)
    after_requests: int = Field(default=4, ge=0)
    after_tool_results: int = Field(default=24, ge=0)
    every: int = Field(default=8, ge=1)


def _progress_notice(signals, tuning):
    request = signals.request_number
    if request > tuning.after_requests:
        due = (request - tuning.after_requests - 1) % tuning.every == 0
    else:
        due = (
            signals.tool_results > tuning.after_tool_results
            and signals.sent.get("progress_notice", 0) == 0
        )
    if not due:
        return None
    from bio_neb.agentic_journey.tool_execution import question_tool_names

    # Only a role whose Tools open a researcher question is told to ask (I-63).
    asks = bool(question_tool_names(signals.agent_role))
    return _notice(
        f"Progress: this is request {request} of this Attempt, with "
        f"{signals.tool_results} tool results so far.",
        "If you have the facts this stage needs, make the call that completes it.",
        *(
            (
                "If a fact cannot be found, ask the researcher with the question "
                "tool instead of searching again.",
            )
            if asks
            else ()
        ),
    )


# hint_as_user: the same refusal again, as a user message (R4 coded_user); it
# points at the refusal's own fix instead of repeating it.
class _HintAsUser(_Capped):
    max_sends: int = Field(default=3, ge=0)
    after_refusals: int = Field(default=2, ge=1)


def _codes_text(codes: tuple[str, ...]) -> str:
    # An argument refusal's code carries its whole validation message; name its kind.
    named = (
        "tool_arguments_invalid" if code.startswith("tool_arguments_invalid:") else code
        for code in codes
    )
    return ", ".join(dict.fromkeys(named))


def _hint_as_user(signals, tuning):
    # The most repeated refusal of the reply is the one to name.
    refusal = max(
        (
            refusal
            for refusal in (signals.reply.refusals if signals.reply else ())
            if refusal.codes != (_REPEATED_CALL,)
            and refusal.count >= tuning.after_refusals
        ),
        key=lambda refusal: refusal.count,
        default=None,
    )
    if refusal is None:
        return None
    tool = refusal.tool_name
    stops = "" if refusal.limit is None else f"; at {refusal.limit} the Attempt stops"
    return _notice(
        f"The same refusal again: `{tool}` ({_codes_text(refusal.codes)}), "
        f"{refusal.count} times in this Attempt{stops}.",
        "The fix is in that call's refusal above: change what it names, then call "
        f"`{tool}` again.",
    )


# The adapter quirk triggers: each reads the latest raw reply's recorded
# quirk. A cut-off reply's note names the cut; its notice adds only how to
# send the next call.
_QUIRK_NOTICES = {
    "text_form_tool_call": (
        "Your last reply wrote its tool call as text in the message; it was read "
        "this time.",
        "Send each tool call in the tool-call field, with no call text in the message.",
    ),
    "text_form_tool_call_cut_off": (
        "Your last reply wrote a tool call as text and was cut off.",
        "Send the next call in the tool-call field, with no text before it.",
    ),
    "think_tags": (
        "Your last reply put think tags in the message text.",
        "Keep the message to the tool call, with no think tags.",
    ),
    "call_reasoning_missing": (
        "A call in your last reply had no `reasoning` argument.",
        "Every tool call carries `reasoning`: one sentence on why you chose these "
        "values.",
    ),
    "argument_name": (
        "An argument name in your last reply was spelled differently from the "
        "tool's schema; it was fixed this time.",
        "Use each argument name exactly as the tool's schema spells it.",
    ),
}


def _quirk(quirk, lines):
    def speak(signals, _tuning):
        if signals.reply is None or quirk not in signals.reply.quirks:
            return None
        return _notice(*lines)

    return speak


# whitespace_runaway: a reply that ran on in blank space.
class _WhitespaceRunaway(_Capped):
    min_run: int = Field(default=1000, ge=1)


_BLANK = re.compile(r"\s+")


def _whitespace_runaway(signals, tuning):
    if signals.reply is None:
        return None
    run = max((len(match) for match in _BLANK.findall(signals.reply.text)), default=0)
    if run < tuning.min_run:
        return None
    return _notice(
        f"Your last reply held a run of {run} blank characters.",
        "End each reply right after its last tool call.",
    )


# refused_call_context: a refused single call is kept with no call, so the next
# request shows no assistant turn for it; the notice shows it.
class _RefusedCallContext(_Capped):
    max_sends: int = Field(default=3, ge=0)


def _refused_call_context(signals, _tuning):
    refusal = next(
        (
            refusal
            for refusal in (signals.reply.refusals if signals.reply else ())
            if refusal.whole_reply
        ),
        None,
    )
    if refusal is None:
        return None
    return _notice(
        "Your last call is not in the history above: it was refused before it ran.",
        f"You sent `{refusal.tool_name}` with "
        f"{model_value_text(dict(refusal.arguments))}.",
    )


# empty_arguments: empty arguments with the answer in the call reasoning (seen
# where every field is optional).
class _EmptyArguments(_Capped):
    min_reasoning: int = Field(default=80, ge=0)


def _empty_arguments(signals, tuning):
    call = next(
        (
            call
            for call in (signals.reply.calls if signals.reply else ())
            if call.takes_arguments
            and not call.arguments
            and len(call.reasoning or "") >= tuning.min_reasoning
        ),
        None,
    )
    if call is None:
        return None
    return _notice(
        f"Your call to `{call.tool_name}` sent no arguments and put its values in "
        "`reasoning`.",
        "Put each value in its own argument field; `reasoning` only says why you "
        "chose them.",
    )


# placeholder_argument: a placeholder such as "<DISCOVERY_EVENT_ID>" where another
# call's result belongs, with the step order.
def _placeholder_argument(signals, _tuning):
    for call in signals.reply.calls if signals.reply else ():
        for location, value in placeholders(dict(call.arguments)):
            return _notice(
                f"Your call to `{call.tool_name}` sent the placeholder "
                f"{json.dumps(value)} in `{location}`.",
                "A value another call returns exists only after that call's "
                "result: call it first, read its result, then send the real "
                "value in a later reply.",
            )
    return None


_register("action_commit", _Capped, _action_commit)
_register("plan_review", _PlanReview, _plan_review)
_register("literal_guard", _LiteralGuard, _literal_guard)
_register("result_suffix", _ResultSuffix, _result_suffix)
_register("repeat_call", _RepeatCall, _repeat_call)
_register("progress_notice", _ProgressNotice, _progress_notice)
_register("hint_as_user", _HintAsUser, _hint_as_user)
for _name, _lines in _QUIRK_NOTICES.items():
    _register(_name, _Capped, _quirk(_name, _lines))
_register("whitespace_runaway", _WhitespaceRunaway, _whitespace_runaway)
_register("refused_call_context", _RefusedCallContext, _refused_call_context)
_register("empty_arguments", _EmptyArguments, _empty_arguments)
_register("placeholder_argument", _Capped, _placeholder_argument)

NAMES: tuple[str, ...] = tuple(PATTERNS)
"""Every registered pattern, in registration order."""

FOLDED: frozenset[str] = frozenset(
    {
        "action_commit",
        "literal_guard",
        "repeat_call",
        "hint_as_user",
        "text_form_tool_call_cut_off",
        "whitespace_runaway",
        "refused_call_context",
    }
)
"""The patterns that answered a fault: the failure module now writes their lines
into the fault's one correction (#373, owner option B), so the slot never sends
them. They stay registered, so retained Journeys that switched them on reopen."""

__all__ = ["FOLDED", "NAMES"]
