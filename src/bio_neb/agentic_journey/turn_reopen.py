"""Reopen of the loop's model turns: each reply's recorded admission and transcript.

Read from the Journey, never re-decided (ADR 0044): each reply's routing, its
calls' validations and dispositions in the order they ran, and the request that
followed it as the transcript composer would have written it (``transcript``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from bio_neb.agentic_journey.action_admission import (
    refused_validation_ids,
    request_bound_arguments,
    validated_model_call,
)
from bio_neb.agentic_journey.action_policy import (
    ProposedAgentAction,
    action_admission_errors,
)
from bio_neb.agentic_journey.attempt_journey import (
    ToolExecutionRefused,
    validate_attempt_fact,
)
from bio_neb.agentic_journey.continuation_policy import OUTPUT_ALONE_KEY
from bio_neb.agentic_journey.model_availability import prior_model_attempt_ids
from bio_neb.agentic_journey.model_turn import read_turn_record
from bio_neb.agentic_journey.operational_review import project_operational_review
from bio_neb.agentic_journey.role_declaration import AgentRole
from bio_neb.agentic_journey.transcript import corrected_outputs


def _repeat_identity(reopen, action):
    from bio_neb.agentic_journey.assignment_context import assignment_repeat_identity

    if reopen.admission is None or reopen.admission.repeat_error is None:
        return None
    return assignment_repeat_identity(action)


def _prior_action_fingerprints(timeline, turns, reopen):
    fingerprints = set()
    if reopen.admission is not None:
        by_id = {event["event_id"]: event for event in timeline}
        # A call whose Tool refused it did nothing.
        refused = refused_validation_ids(timeline)
        for turn in turns:
            if (
                turn["disposition"] in {"singleton_admitted", "multi_call_admitted"}
                and turn["validation_event_ids"][0] not in refused
            ):
                validation = by_id[turn["validation_event_ids"][0]]["payload"]
                key = _repeat_identity(
                    reopen,
                    ProposedAgentAction(
                        tool_name=validation["tool_name"],
                        tool_call_id=validation["tool_call_id"],
                        arguments=validation["arguments"],
                    ),
                )
                if key is not None:
                    fingerprints.add(key)
    return fingerprints


def _cited_event_ids(value: Any) -> Iterable[str]:
    """Yield every Journey event id an argument names by ``event_id``/``*_event_id(s)``."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if (key == "event_id" or key.endswith("_event_id")) and isinstance(
                item, str
            ):
                yield item
            elif key.endswith("_event_ids") and isinstance(item, list):
                yield from (entry for entry in item if isinstance(entry, str))
            else:
                yield from _cited_event_ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from _cited_event_ids(item)


def _validate_admitted_action(
    action: ProposedAgentAction,
    *,
    role: AgentRole,
    reopen: Any,
    exposed_tool_names: set[str],
    prior_event_ids: set[str],
    admitted_action_fingerprints: set[str],
    bound_arguments: Mapping[str, Any],
) -> None:
    """Reopen an admitted call by its recorded relationships, never by re-admission.

    The shared generic gates run against the recorded request and earlier admitted
    calls; no domain validator runs, so the role's verdict is not
    re-decided.
    """

    from bio_neb.agentic_journey.tool_execution import function_tool_names

    errors = action_admission_errors(
        action,
        eligible_tool_names=function_tool_names(role),
        exposed_tool_names=frozenset(exposed_tool_names),
        action_validator=lambda _recorded: (),
        action_fingerprint=lambda recorded: _repeat_identity(reopen, recorded),
        admitted_action_fingerprints=frozenset(admitted_action_fingerprints),
        admitted_repeat_error=None
        if reopen.admission is None
        else reopen.admission.repeat_error,
        bound_arguments=bound_arguments,
    )
    if errors:
        raise ValueError(
            f"admitted action fails its recorded admission gates: {', '.join(errors)}"
        )
    if not set(_cited_event_ids(dict(action.arguments))) <= prior_event_ids:
        raise ValueError("admitted action cites an event not retained before it")


def _validate_ordered_calls(  # noqa: C901, PLR0912, PLR0913 - one response's recorded relationships
    response: Mapping[str, Any],
    calls: list[Mapping[str, Any]],
    validations: list[Mapping[str, Any]],
    dispositions: list[Mapping[str, Any]],
    *,
    role: AgentRole,
    reopen: Any,
    exposed_tool_names: set[str],
    bound_arguments: Mapping[str, Mapping[str, Any]],
    timeline: list[Mapping[str, Any]],
    selected: list[Mapping[str, Any]],
    admitted_action_fingerprints: set[str],
    next_request: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    """Reopen a routed or multi-call reply: each kept call's own decision, in run order.

    ``calls`` are the kept calls in the order they ran. Each call has exactly its
    validation and disposition, and was admitted just before it ran: its
    validation, disposition, execution start and terminal precede the next call's
    validation; an admitted call passes the shared gates against the calls
    admitted before it; no call is admitted after a question-opening call; a
    refused call's errors are not replayed, and its correction is its own return
    in the next request. The calls stop short of the reply only at a refusal
    beyond its allowance or a failed execution.
    """

    from bio_neb.agentic_journey.tool_execution import function_tool_contracts

    if len(validations) != len(dispositions) or not 1 <= len(validations) <= len(calls):
        raise ValueError("ordered calls lack their exact validations and dispositions")
    for call, validation, disposition in zip(  # noqa: B905 - lengths checked above
        calls, validations, dispositions
    ):
        recorded = validation["payload"]
        turn = disposition["payload"]
        if validated_model_call(call, recorded, bound_arguments) is None:
            raise ValueError("action validation changed exact model call order")
        if (
            turn["attempt_id"] != response["payload"]["attempt_id"]
            or turn["response_event_id"] != response["event_id"]
            or turn["validation_event_ids"] != [validation["event_id"]]
            or (turn["disposition"] == "multi_call_admitted")
            != (recorded["status"] == "valid")
            or (recorded["status"] == "valid") == bool(recorded["errors"])
        ):
            raise ValueError("ordered call disposition changed its exact lineage")
    parts = (
        []
        if next_request is None
        else [
            part
            for message in compatible_model_request(next_request).get("messages", [])
            for part in message.get("parts", [])
        ]
    )
    position = {event["event_id"]: index for index, event in enumerate(selected)}
    starts = {
        event["payload"]["validation_event_id"]: event
        for event in selected
        if event["event_type"] == "ObservedToolExecutionStarted"
    }
    terminals = {
        event["payload"]["started_event_id"]: event
        for event in selected
        if event["event_type"]
        in {"ObservedToolExecutionSucceeded", "ObservedToolExecutionFailed"}
    }
    # A failed execution the model was shown as its call's result.
    refusals = {
        event["payload"]["failed_event_id"]: event["payload"]
        for event in selected
        if event["event_type"] == "ToolExecutionRefused"
    }
    returns = {
        part.get("tool_call_id"): part
        for part in parts
        if part.get("part_kind") == "tool-return"
    }
    opens_question = {
        item.tool_name for item in function_tool_contracts(role) if item.opens_question
    }
    question_opened = False
    after = position[response["event_id"]]
    last_failed = False
    for call, validation, disposition in zip(  # noqa: B905 - lengths checked above
        calls, validations, dispositions
    ):
        recorded = validation["payload"]
        turn = disposition["payload"]
        if (
            not after
            < position[validation["event_id"]]
            < position[disposition["event_id"]]
        ):
            raise ValueError("ordered call was not admitted in the order sent")
        after = position[disposition["event_id"]]
        if turn["disposition"] == "multi_call_refused":
            # A refused call has no effect or authority; its errors are not replayed.
            returned = returns.get(recorded["tool_call_id"])
            if next_request is not None and (
                returned is None or returned.get("content") != turn["feedback"]
            ):
                raise ValueError("refused call lost its exact correction return")
            continue
        if question_opened:
            raise ValueError("a call after an opened researcher question was admitted")
        action = validated_model_call(call, recorded, bound_arguments)
        _validate_admitted_action(
            action,
            role=role,
            reopen=reopen,
            exposed_tool_names=exposed_tool_names,
            prior_event_ids={
                event["event_id"] for event in timeline[: timeline.index(validation)]
            },
            admitted_action_fingerprints=admitted_action_fingerprints,
            bound_arguments=bound_arguments.get(action.tool_name, {}),
        )
        fingerprint = _repeat_identity(reopen, action)
        if fingerprint is not None:
            admitted_action_fingerprints.add(fingerprint)
        question_opened = action.tool_name in opens_question
        # The admitted call ran next, and finished before the next call's turn.
        start = starts.get(validation["event_id"])
        terminal = None if start is None else terminals.get(start["event_id"])
        if start is None or terminal is None or not after < position[start["event_id"]]:
            raise ValueError("admitted ordered call did not run in its turn")
        after = position[terminal["event_id"]]
        refusal = refusals.get(terminal["event_id"])
        last_failed = (
            terminal["event_type"] == "ObservedToolExecutionFailed" and refusal is None
        )
        if refusal is not None:
            # A refused call did nothing: calling it again is no repeat.
            if fingerprint is not None:
                admitted_action_fingerprints.discard(fingerprint)
            returned = returns.get(recorded["tool_call_id"])
            if next_request is not None and (
                returned is None
                or returned.get("content")
                != ToolExecutionRefused.model_validate(refusal).model_view()
            ):
                raise ValueError("refused execution lost its exact result")
    if len(validations) < len(calls):
        last = dispositions[-1]["payload"]
        stopped = last["disposition"] == "multi_call_refused" and any(
            event["event_type"] == "AgentWorkReviewRequested"
            and event["payload"]["source_event_id"] == validations[-1]["event_id"]
            for event in timeline
        )
        if next_request is not None or not (stopped or last_failed):
            raise ValueError("ordered calls stopped short without their exact stop")
    return dispositions


def _validate_routed_reply(
    response, routed, tool_calls, offered, output_tools, validations, shown_tools
):
    """The kept calls of a routed reply in run order, and its unoffered validations.

    Routing never re-decides: the raw reply's calls are exactly the kept calls,
    each copy of an earlier identical call, each refused unoffered call, each
    output refused beside the calls that run and a cut-off reply's partial last
    call. An output is kept only alone, and refused only beside a kept call.
    """
    from bio_neb.agentic_journey.action_policy import call_identity
    from bio_neb.agentic_journey.model_turn import object_arguments, read_arguments

    payload = routed["payload"]
    message = compatible_model_response(response)["message"]
    by_id = {part["tool_call_id"]: part for part in tool_calls}
    cut_off = message.get("finish_reason") == "length"
    dropped = tool_calls[-1]["tool_call_id"] if cut_off and tool_calls else None
    complete = [part for part in tool_calls if part["tool_call_id"] != dropped]
    position = {part["tool_call_id"]: index for index, part in enumerate(complete)}
    unoffered = {
        event["payload"]["tool_call_id"]: event["event_id"]
        for event in validations
        if event["event_id"] in payload["unoffered_validation_event_ids"]
    }

    def identity(call_id):
        # Routing compared the calls as the Model turn read them, against the
        # Tools the request showed: an object sent as JSON text is that object.
        part = by_id[call_id]
        objects = next(
            (
                object_arguments(tool["parameters_json_schema"])
                for tool in shown_tools
                if tool["name"] == part["tool_name"]
            ),
            (),
        )
        return call_identity(
            ProposedAgentAction(
                tool_name=part["tool_name"],
                tool_call_id=call_id,
                arguments=read_arguments(part.get("args"), objects).value,
            )
        )

    # Written only when an output was refused; the typed fact reads it.
    refused_outputs = validate_attempt_fact(routed).refused_output_call_ids
    kept_kinds = {
        by_id[call_id]["tool_name"] in output_tools for call_id in payload["run_order"]
    }
    if (
        payload["attempt_id"] != response["payload"]["attempt_id"]
        or payload["dropped_call_id"] != dropped
        or len(unoffered) != len(payload["unoffered_validation_event_ids"])
        or sorted(
            [*payload["run_order"], *payload["copies"], *unoffered, *refused_outputs],
            key=lambda call_id: position.get(call_id, -1),
        )
        != [part["tool_call_id"] for part in complete]
        or kept_kinds == {True, False}
        or (bool(refused_outputs) and kept_kinds != {False})
        or any(
            by_id[call_id]["tool_name"] not in output_tools
            for call_id in refused_outputs
        )
        or bool(refused_outputs)
        != any(fault["key"] == OUTPUT_ALONE_KEY for fault in payload["faults"])
        or any(
            target in payload["copies"]
            or position[target] > position[copy]
            or identity(copy) != identity(target)
            for copy, target in payload["copies"].items()
        )
        or any(
            by_id[event["payload"]["tool_call_id"]]["tool_name"] in offered
            or event["payload"]["status"] != "invalid"
            for event in validations
            if event["event_id"] in unoffered.values()
        )
    ):
        raise ValueError("routed reply changed its exact raw reply")
    return [by_id[call_id] for call_id in payload["run_order"]], set(unoffered.values())


def _stopped_at_routing(routed, selected) -> bool:
    """Whether a reply fault stopped the Attempt as its reply was routed.

    The stop's source is the routing itself, or the reply's own truncation.
    """
    response_id = routed["payload"]["response_event_id"]
    sources = {routed["event_id"]} | {
        event["event_id"]
        for event in selected
        if event["event_type"] == "ModelCompletionTruncated"
        and event["payload"]["response_event_id"] == response_id
    }
    return any(
        event["event_type"] == "AgentWorkReviewRequested"
        and event["payload"]["source_event_id"] in sources
        for event in selected
    )


def _check_next_request(
    timeline,
    response,
    member_id,
    next_request,
    *,
    routed=None,
    kept_calls=(),
    refused=None,
    outputs=None,
) -> None:
    """The request after a reply is the transcript composer's answer for it.

    ``kept_calls`` are the calls of the reply that ran; a routed turn's note is
    owed after their results. When nothing ran, the request's own retry part is
    the reply's correction (``refused``, a reply with no call), its several
    outputs' correction followed by the note (``outputs``, #373 E4), or the note
    alone. A request of another member starts its own transcript.
    """
    from bio_neb.agentic_journey.transcript import (
        check_request,
        kept_message,
        reply_retry,
    )

    if next_request is None or next_request["payload"]["member_id"] != member_id:
        return
    messages = compatible_model_request(next_request).get("messages", [])
    pending = messages[-1].get("parts", []) if messages else []
    answered = frozenset(
        part["tool_call_id"]
        for part in pending
        if part.get("part_kind") == "tool-return"
    )
    note = None if routed is None else routed["payload"]["feedback"]
    ran = bool(kept_calls) and outputs is None
    check_request(
        next_request,
        kept_message(timeline, response, answered),
        note=note if ran else None,
    )
    if ran:
        return
    if refused is None and outputs is None and note is None:
        return
    retry = refused if refused is not None else reply_retry(outputs, note)
    own = [
        part
        for part in pending
        if not (
            part.get("part_kind") == "user-prompt"
            and str(part.get("content", "")).startswith("[injection:")
        )
    ]
    if (
        not own
        or own[-1].get("part_kind") != "retry-prompt"
        or own[-1].get("tool_name") is not None
        or own[-1].get("content") != retry
    ):
        raise ValueError("request lost its reply's exact correction")


def _check_refused_results(selected: list[dict[str, Any]]) -> None:
    """A refused execution's result is the next request's return for its call,
    exactly as the model was shown it, singleton or ordered (#477 C8; Standards
    r1 ST2)."""
    by_id = {event["event_id"]: event for event in selected}
    for index, event in enumerate(selected):
        if event["event_type"] != "ToolExecutionRefused":
            continue
        refusal = event["payload"]
        failed = by_id.get(refusal["failed_event_id"])
        started = by_id.get((failed or {}).get("payload", {}).get("started_event_id"))
        if started is None:
            raise ValueError("refused execution lost its exact call")
        call_id = started["payload"]["tool_call_id"]
        later = next(
            (
                request
                for request in selected[index + 1 :]
                if request["event_type"] == "ModelRequestRecorded"
                and request["payload"]["attempt_id"] == refusal["attempt_id"]
            ),
            None,
        )
        if later is None:
            continue
        returned = [
            part
            for message in compatible_model_request(later).get("messages", [])
            for part in message.get("parts", [])
            if part.get("part_kind") == "tool-return"
            and part.get("tool_call_id") == call_id
        ]
        if (
            not returned
            or returned[-1].get("tool_name") != started["payload"]["tool_name"]
            or returned[-1].get("content")
            != ToolExecutionRefused.model_validate(refusal).model_view()
            # The refusal's own id rides beside the view, never sent.
            or returned[-1].get("metadata") != {"event_id": event["event_id"]}
        ):
            raise ValueError("refused execution lost its exact result")


def validate_action_turns(  # noqa: C901, PLR0915 - Existing orchestration complexity; contract-base migration preserves this flow.
    selected: list[dict[str, Any]],
    *,
    role: AgentRole,
    assignment: Any,
    timeline: list[dict[str, Any]],
) -> Callable[[], None]:
    """Reconstruct whole-response admission relationships; never re-decide admission.

    Returns the composer's question of each reply's next request, which the
    caller asks once its domain checks pass, so a changed admission or domain
    fact keeps its own refusal.
    """

    requests = [
        event for event in selected if event["event_type"] == "ModelRequestRecorded"
    ]
    responses = [
        event for event in selected if event["event_type"] == "ModelResponseRecorded"
    ]
    validations = [
        event for event in selected if event["event_type"] == "ToolCallValidated"
    ]
    dispositions = [
        event
        for event in selected
        if event["event_type"] == "AgentActionTurnDispositionRecorded"
    ]
    contacts = {
        event["event_id"]: event
        for event in timeline
        if event["event_type"] == "ModelPhysicalContactStarted"
    }
    _check_refused_results(selected)
    correlated: list[dict[str, Any]] = []
    # Each reply's next request, asked of the composer once admission reads clean.
    transcripts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    correlated_validation_ids: set[str] = set()
    attempt_id = next(
        event["payload"]["attempt_id"]
        for event in selected
        if event["event_type"] == "AttemptStarted"
    )
    prior_ids = set(prior_model_attempt_ids(timeline, attempt_id))
    prior_turns = [
        event["payload"]
        for event in timeline
        if event["event_type"] == "AgentActionTurnDispositionRecorded"
        and event["payload"]["attempt_id"] in prior_ids
    ]
    routings = {
        event["payload"]["response_event_id"]: event
        for event in selected
        if event["event_type"] == "AgentTurnRouted"
    }
    admitted_action_fingerprints = _prior_action_fingerprints(
        timeline, prior_turns, assignment.reopen
    )
    refused_calls = refused_validation_ids(timeline)
    for validation in validations:
        request_id = validation["payload"].get("continuation_request_event_id")
        if request_id is None:
            continue
        projection = project_operational_review(timeline)
        request = next(
            (r for r in projection.requests if r.request_event_id == request_id), None
        )
        start = next(e for e in selected if e["event_type"] == "AttemptStarted")
        receipt = next(
            (
                r
                for r in projection.receipts
                if r.response_event_id
                == start["payload"].get("continuation_operational_review_event_id")
            ),
            None,
        )
        turns = [
            d
            for d in dispositions
            if d["payload"]["validation_event_ids"] == [validation["event_id"]]
        ]
        if (
            request is None
            or receipt is None
            or receipt.request_event_id != request_id
            or receipt.response.action != "retry"
            or len(turns) != 1
            or any(
                validation["payload"][k] != getattr(request, k)
                for k in ("arguments", "tool_name", "tool_call_id")
            )
            or turns[0]["payload"]["response_event_id"] != request.response_event_id
        ):
            raise ValueError("Tool continuation changed its exact pending call")
        correlated.append(turns[0])
        correlated_validation_ids.add(validation["event_id"])
        # The stopped reply's routed note follows the replayed call's result.
        routed = next(
            (
                event
                for event in timeline
                if event["event_type"] == "AgentTurnRouted"
                and event["payload"]["response_event_id"] == request.response_event_id
            ),
            None,
        )
        next_request = next(
            (
                item
                for item in requests
                if selected.index(item) > selected.index(validation)
            ),
            None,
        )
        if routed is not None:
            response = next(
                event
                for event in timeline
                if event["event_id"] == request.response_event_id
            )
            transcripts.append(
                (
                    (response, request.member_id, next_request),
                    {"routed": routed, "kept_calls": routed["payload"]["run_order"]},
                )
            )

    for response in responses:
        response_index = selected.index(response)
        next_request_index = next(
            (
                selected.index(request)
                for request in requests
                if selected.index(request) > response_index
            ),
            len(selected),
        )
        request = next(
            (
                item
                for item in requests
                if item["payload"]["request_number"]
                == response["payload"]["request_number"]
            ),
            None,
        )
        if request is None:
            raise ValueError("action response lacks its exact retained request")
        # A typed-output call is correlated by its structured-output session.
        output_tool_names = {
            item["name"]
            for item in compatible_model_request(request).get("output_tools") or ()
        }
        tool_calls = [
            part
            for part in compatible_model_response(response)["message"]["parts"]
            if part.get("part_kind") == "tool-call"
        ]
        exposed_tool_names = {
            item["name"] for item in compatible_model_request(request)["function_tools"]
        }
        bound_arguments = request_bound_arguments(
            compatible_model_request(request)["function_tools"]
        )
        turn_validations = [
            event
            for event in validations
            if response_index < selected.index(event) < next_request_index
        ]
        turn_dispositions = [
            event
            for event in dispositions
            if response_index < selected.index(event) < next_request_index
        ]
        next_request = (
            selected[next_request_index] if next_request_index < len(selected) else None
        )
        turn = read_turn_record(compatible_model_response(response)["message"])
        # No call and no cut-off: the reply ended in text, or held only reasoning.
        # A reply whose contact kept no response body is an older record, from
        # when every reply without a call was read as a text ending.
        text_ending = not tool_calls and not turn.cut_off
        contact = validate_attempt_fact(
            contacts[response["payload"]["physical_contact_started_event_id"]]
        )
        without_call = (
            "reasoning_refused"
            if turn.unusable is not None and contact.response_body_expected
            else "text_refused"
        )
        routed = routings.get(response["event_id"])
        if routed is not None:
            # The kept calls, in run order: the raw reply minus what routing names.
            tool_calls, unoffered = _validate_routed_reply(
                response,
                routed,
                tool_calls,
                exposed_tool_names,
                output_tool_names,
                turn_validations,
                (
                    *compatible_model_request(request)["function_tools"],
                    *compatible_model_request(request)["output_tools"],
                ),
            )
            correlated_validation_ids.update(unoffered)
            turn_validations = [
                event
                for event in turn_validations
                if event["event_id"] not in unoffered
            ]
        refused = next(
            (
                event["payload"]["feedback"]
                for event in turn_dispositions
                if event["payload"]["disposition"]
                in {"text_refused", "reasoning_refused"}
            ),
            None,
        )
        transcripts.append(
            (
                (response, request["payload"]["member_id"], next_request),
                {
                    "routed": routed,
                    "kept_calls": tool_calls if routed is not None else (),
                    "refused": refused,
                    "outputs": corrected_outputs(timeline, response["event_id"]),
                },
            )
        )
        calls = [
            part
            for part in tool_calls
            if part.get("tool_name") not in output_tool_names
        ]
        if routed is not None and (
            (not tool_calls and not text_ending)
            or _stopped_at_routing(routed, selected)
        ):
            # Nothing of the reply ran: its note is the next request's reason, or
            # a fault beyond its allowance stopped the Attempt before any call.
            if turn_validations or turn_dispositions:
                raise ValueError("action admission invented a call for a response")
            continue
        if not calls and tool_calls:
            if turn_validations or turn_dispositions:
                raise ValueError("action admission invented a call for typed output")
            continue
        if not calls:
            # A text ending or a reasoning-only reply is refused once, under its
            # own disposition, with its correction and no validation.
            if (
                turn_validations
                or len(turn_dispositions) != 1
                or turn_dispositions[0]["payload"]["disposition"] != without_call
                or turn_dispositions[0]["payload"]["response_event_id"]
                != response["event_id"]
            ):
                raise ValueError("text ending lacks its exact refused turn")
            correlated.append(turn_dispositions[0])
            continue
        if (
            (len(calls) > 1 or routed is not None)
            and turn_dispositions
            and all(
                event["payload"]["disposition"].startswith("multi_call_")
                for event in turn_dispositions
            )
        ):
            correlated_validation_ids.update(
                event["event_id"] for event in turn_validations
            )
            correlated.extend(
                _validate_ordered_calls(
                    response,
                    calls,
                    turn_validations,
                    turn_dispositions,
                    role=role,
                    reopen=assignment.reopen,
                    exposed_tool_names=exposed_tool_names,
                    bound_arguments=bound_arguments,
                    timeline=timeline,
                    selected=selected,
                    admitted_action_fingerprints=admitted_action_fingerprints,
                    next_request=next_request,
                )
            )
            continue
        if len(calls) != len(turn_validations) or len(turn_dispositions) != 1:
            raise ValueError("action turn lacks exact validation and disposition")
        correlated_validation_ids.update(
            event["event_id"] for event in turn_validations
        )
        prior_event_ids = {
            event["event_id"] for event in timeline[: timeline.index(response)]
        }
        for call, validation in zip(calls, turn_validations):  # noqa: B905 - Preserve existing zip truncation; contract-base migration adds no options.
            recorded = validation["payload"]
            action = validated_model_call(call, recorded, bound_arguments)
            if action is None:
                raise ValueError("action validation changed exact model call order")
            if (recorded["status"] == "valid") == bool(recorded["errors"]):
                raise ValueError("action validation verdict is incoherent")
            # A refused call has no effect or authority; its errors are not replayed.
            if recorded["status"] == "valid":
                _validate_admitted_action(
                    action,
                    role=role,
                    reopen=assignment.reopen,
                    exposed_tool_names=exposed_tool_names,
                    prior_event_ids=prior_event_ids,
                    admitted_action_fingerprints=admitted_action_fingerprints,
                    bound_arguments=bound_arguments.get(action.tool_name, {}),
                )
        turn = turn_dispositions[0]["payload"]
        if (
            turn["attempt_id"] != response["payload"]["attempt_id"]
            or turn["response_event_id"] != response["event_id"]
            or turn["validation_event_ids"]
            != [event["event_id"] for event in turn_validations]
        ):
            raise ValueError("action disposition changed its exact response lineage")
        kind = turn["disposition"]
        if len(calls) != 1:
            raise ValueError("multi-action response has a singleton disposition")
        expected = (
            "singleton_admitted"
            if turn_validations[0]["payload"]["status"] == "valid"
            else "singleton_refused"
        )
        if kind != expected:
            raise ValueError("singleton action disposition is incoherent")
        if (
            kind == "singleton_admitted"
            and turn_validations[0]["event_id"] not in refused_calls
        ):
            # The checked call's arguments are the recorded admitted arguments;
            # a call its Tool refused did nothing.
            fingerprint = _repeat_identity(assignment.reopen, action)
            if fingerprint is not None:
                admitted_action_fingerprints.add(fingerprint)
        correlated.append(turn_dispositions[0])

    # The continuation check bounds each fault key's corrections.
    if len(dispositions) != len(correlated):
        raise ValueError("action disposition is outside a retained model response")
    if correlated_validation_ids != {event["event_id"] for event in validations}:
        raise ValueError("action validation is outside a retained model response")

    def ask_composer() -> None:
        for arguments, facts in transcripts:
            _check_next_request(timeline, *arguments, **facts)

    return ask_composer


def compatible_model_request(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the normalized compatible request from its sole evidence seam."""

    return event["payload"]["compatible_adapter_evidence"]["request"]


def compatible_model_response(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the normalized compatible response from its sole evidence seam."""

    return event["payload"]["compatible_adapter_evidence"]["response"]
