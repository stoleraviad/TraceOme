"""Sibling Journey correlation for model structured-output calls."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from bio_neb.agentic_journey.attempt_journey import (
    AgentOutputValidated,
    AttemptFailureSource,
)
from bio_neb.agentic_journey.invocation_observation import inspection_event
from bio_neb.agentic_journey.lifecycle import Activity
from bio_neb.agentic_journey.model_turn import (
    as_received,
    output_schema_errors,
    read_arguments,
    read_turn,
)
from bio_neb.content import canonical_json
from bio_neb.contracts import StrictFrozenContract
from bio_neb.failure import (
    CorrectionWording,
    FaultCount,
    Invariant,
    OutputCalls,
    OutputFaults,
    OutputRefused,
    fault_trace,
    render_correction,
)


class SeveralOutputs(Exception):
    """A reply sent its output Tool more than once: the model reads ``correction``
    and the reply keeps none of its output calls (#373 E4)."""

    def __init__(self, correction: str) -> None:
        super().__init__(correction)
        self.correction = correction


class StructuredOutputCallRecorded(StrictFrozenContract):
    """One model output call, observed but never represented as Tool execution."""

    event_type: Literal["StructuredOutputCallRecorded"] = "StructuredOutputCallRecorded"
    attempt_id: StrictStr
    agent_role: StrictStr
    collection_member_id: StrictStr
    response_event_id: StrictStr
    tool_name: StrictStr
    tool_call_id: StrictStr
    arguments: dict[str, JsonValue]
    schema_normalized_arguments: dict[str, JsonValue] | None = None
    # The model wrote the arguments inside a Markdown fence, which the Model turn
    # stripped; written only when true.
    arguments_fenced: StrictBool = Field(
        default=False, exclude_if=lambda value: not value
    )
    # The top-level arguments the Model turn read as the object their JSON text
    # held, in the order sent; written only when some were. A retained payload
    # holds the names as a JSON list.
    arguments_decoded: tuple[StrictStr, ...] = Field(
        default=(), strict=False, exclude_if=lambda value: not value
    )


class StructuredOutputValidationLinked(StrictFrozenContract):
    """Link one response's exact output-call set to one validation attempt."""

    event_type: Literal["StructuredOutputValidationLinked"] = (
        "StructuredOutputValidationLinked"
    )
    attempt_id: StrictStr
    agent_role: StrictStr
    collection_member_id: StrictStr
    tool_name: StrictStr
    validation_attempt: StrictInt = Field(gt=0)
    response_event_id: StrictStr
    output_call_event_ids: tuple[StrictStr, ...] = Field(min_length=1)
    validation_event_id: StrictStr


class StructuredOutputValidationAttempt(StrictFrozenContract):
    validation_attempt: StrictInt = Field(gt=0)
    response_event_id: StrictStr
    output_call_event_ids: tuple[StrictStr, ...] = Field(min_length=1)
    output_call_ids: tuple[StrictStr, ...] = Field(min_length=1)
    validation_event_id: StrictStr
    validation_link_event_id: StrictStr
    status: Literal["valid", "invalid"]
    output: dict[str, JsonValue]
    errors: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def status_matches_errors(self) -> "StructuredOutputValidationAttempt":
        if (self.status == "valid") == bool(self.errors):
            raise ValueError("structured-output validation status changed its errors")
        return self


class StructuredOutputLineage(StrictFrozenContract):
    collection_member_id: StrictStr
    tool_name: StrictStr
    # The failure rule's largest number of failed turns in a row for the Attempt,
    # as recorded. Refused outputs are failed turns like any other fault, so the
    # continuation check, not this lineage, bounds them.
    max_retries: StrictInt = Field(ge=0)
    validation_attempts: tuple[StructuredOutputValidationAttempt, ...] = Field(
        min_length=1
    )
    accepted_output: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def exact_validation_sequence(self) -> "StructuredOutputLineage":
        if tuple(item.validation_attempt for item in self.validation_attempts) != tuple(
            range(1, len(self.validation_attempts) + 1)
        ):
            raise ValueError("structured-output validation attempts are not contiguous")
        valid = [item for item in self.validation_attempts if item.status == "valid"]
        if valid and valid != [self.validation_attempts[-1]]:
            raise ValueError("structured-output validation continued after acceptance")
        # Several output calls in one reply are an invalid attempt: terminal unless
        # corrected, which the continuation check reads from its feedback.
        expected_output = valid[0].output if valid else None
        if self.accepted_output != expected_output:
            raise ValueError("structured-output accepted output changed validation")
        return self


def output_errors(
    output_type: type[BaseModel], arguments: Mapping[str, Any]
) -> tuple[str, ...]:
    """The output type's refusals of an output call's arguments, in the one form every
    role records (``AgentOutputValidated.errors``) and its repair names: each error's
    location, then its own type (``findings.0.reason:missing``), or its type alone for
    a rule of the whole output. The arguments are read as JSON, strictly."""
    try:
        output_type.model_validate_json(canonical_json(dict(arguments)), strict=True)
    except ValidationError as error:
        return tuple(
            ":".join(
                part
                for part in (
                    ".".join(str(item) for item in detail["loc"]),
                    detail["type"],
                )
                if part
            )
            for detail in error.errors(include_url=False, include_input=False)
        )
    return ()


def normalized_output(
    output_type: type[BaseModel], arguments: Mapping[str, Any]
) -> dict[str, JsonValue]:
    """The output type's own JSON of arguments it admits: what the record keeps."""
    return output_type.model_validate_json(
        canonical_json(dict(arguments)), strict=True
    ).model_dump(mode="json")


class StructuredOutputSession:
    """Record output-call validation and repair without executing a Function Tool."""

    def __init__(
        self,
        *,
        parent_activity: Activity,
        attempt_id: str,
        agent_role: str,
        tool_name: str,
        max_retries: int,
        failure_source: AttemptFailureSource,
        output_type: type[BaseModel],
        wording: CorrectionWording,
        turn_count: Callable[[], FaultCount],
        arguments_refusal: Callable[[dict[str, Any]], OutputFaults | None]
        | None = None,
    ) -> None:
        self._parent_activity = parent_activity
        self._attempt_id = attempt_id
        self._agent_role = agent_role
        self._tool_name = tool_name
        self._max_retries = max_retries
        self._failure_source = failure_source
        self._output_type = output_type
        # The role's one correction for refused arguments, which then replaces
        # the schema's: it covers the shape and the role's own checks.
        self._arguments_refusal = arguments_refusal
        self._wording = wording
        # The reply's place in the failed turns in a row and the limit.
        self._turn_count = turn_count
        self._collection_member_id: str | None = None
        self._pending: tuple[str, tuple[dict[str, Any], ...]] | None = None
        self._attempts: dict[str, list[StructuredOutputValidationAttempt]] = (
            defaultdict(list)
        )

    def restore(self, lineages: tuple[StructuredOutputLineage, ...]) -> None:
        """Restore checked validation evidence; completed members cannot be reopened."""
        if self._attempts or self._pending is not None:
            raise Invariant(
                "structured_output_session_reused",
                "structured-output restore requires a fresh session",
            )
        for lineage in lineages:
            if (
                lineage.tool_name != self._tool_name
                or lineage.max_retries != self._max_retries
            ):
                raise ValueError("structured-output restore changed its policy")
            self._attempts[lineage.collection_member_id] = list(
                lineage.validation_attempts
            )

    def begin_member(self, collection_member_id: str) -> None:
        if self._pending is not None:
            raise Invariant(
                "structured_output_member_changed",
                "structured-output member changed before validation",
            )
        if (
            collection_member_id in self._attempts
            and self._attempts[collection_member_id][-1].status == "valid"
        ):
            raise Invariant(
                "structured_output_member_repeated",
                "structured-output collection member repeated",
            )
        self._collection_member_id = collection_member_id

    def observe_response(
        self, response_event_id: str, response: Any, output_tools: Sequence[Any]
    ) -> None:
        """Record the output calls of a reply as admission routed it (ADR 0044),
        read against the output Tools its request showed."""
        parts = [
            call
            for call in read_turn(response, output_tools).calls
            if call.tool_name == self._tool_name
        ]
        if not parts:
            return
        member_id = self._require_member()
        calls: list[dict[str, Any]] = []
        for part in parts:
            read = part.arguments
            errors = (
                ("output_arguments_invalid_json",)
                if read.malformed
                else output_errors(self._output_type, read.value)
            )
            normalized_arguments = (
                None if errors else normalized_output(self._output_type, read.value)
            )
            call = self._parent_activity.record(
                StructuredOutputCallRecorded(
                    attempt_id=self._attempt_id,
                    agent_role=self._agent_role,
                    collection_member_id=member_id,
                    response_event_id=response_event_id,
                    tool_name=self._tool_name,
                    tool_call_id=part.tool_call_id,
                    arguments=read.value,
                    schema_normalized_arguments=normalized_arguments,
                    arguments_fenced=read.fenced,
                    arguments_decoded=read.decoded,
                )
            )
            calls.append({**call, "errors": errors})
        if len(calls) > 1:
            # Several output calls in one reply are the member's next output
            # repair; past the allowance the continuation check stops the
            # Attempt before another request.
            fault = OutputCalls(self._tool_name, len(calls), self._next_repair())
            correction = render_correction(fault, self._wording)
            self._inspect(fault, correction)
            self._record_validation(
                response_event_id=response_event_id,
                calls=tuple(calls),
                output={},
                errors=("multiple_output_calls",),
                feedback=correction,
            )
            raise SeveralOutputs(correction)
        call = calls[0]
        errors = call["errors"]
        if errors:
            # The framework refuses this output only on its schema; the repair the
            # model then reads is the schema's, or the role's one correction.
            schema_errors = output_schema_errors(self._output_type, parts[0].arguments)
            faults = None if read.malformed else self.arguments_refusal(read.value)
            self._record_validation(
                response_event_id=response_event_id,
                calls=(call,),
                output=call["payload"]["arguments"],
                # The type's errors, then each further code of the role's own
                # correction, so the record names every fault the model read.
                errors=tuple(
                    dict.fromkeys((*errors, *(() if faults is None else faults.codes)))
                ),
                feedback=None
                if schema_errors is None
                else self._repair(
                    OutputRefused(
                        schema_errors=None
                        if faults is not None
                        else tuple(schema_errors),
                        faults=faults,
                        fault=self._next_repair(),
                        tool_name=self._tool_name,
                    )
                ),
            )
            return
        self._pending = (response_event_id, (call,))

    def validate(
        self,
        output: Mapping[str, JsonValue],
        errors: tuple[str, ...],
        repair: str | None = None,
        faults: OutputFaults | None = None,
    ) -> dict[str, Any]:
        """Record the validator's verdict; ``repair`` is its message to the model,
        or ``faults`` the facts the failure package words as one."""
        if self._pending is None:
            raise Invariant(
                "structured_output_call_unobserved",
                "structured-output validation lacks one observed call",
            )
        response_event_id, calls = self._pending
        return self._record_validation(
            response_event_id=response_event_id,
            calls=calls,
            output=dict(output),
            errors=errors,
            feedback=None
            if repair is None and faults is None
            else self._repair(
                OutputRefused(
                    reason=repair,
                    faults=faults,
                    fault=self._next_repair(),
                    tool_name=self._tool_name,
                )
            ),
        )

    def arguments_refusal(self, arguments: Any) -> OutputFaults | None:
        """The role's faults for these schema-refused arguments, which the
        failure package words as one correction, or None to leave the schema's
        (#448 D1)."""
        if self._arguments_refusal is None or not isinstance(arguments, dict):
            return None
        return self._arguments_refusal(arguments)

    def _next_repair(self) -> FaultCount:
        """This refusal's reply's place in the failed turns in a row, and the
        place that stops the Attempt (the one failure rule, #447 B)."""
        self._require_member()
        return self._turn_count()

    def recorded_repair(self) -> FaultCount | None:
        """The current member's repair count after its latest refusal was recorded,
        so a correction sent after the record reads the same count (#373 E3)."""
        if self._collection_member_id is None:
            return None
        attempts = self._attempts.get(self._collection_member_id)
        if not attempts or attempts[-1].status == "valid":
            return None
        return self._turn_count()

    def _repair(self, correction: OutputRefused) -> str:
        """The repair as the model receives it, written by the retry rendering."""
        text = as_received(
            render_correction(correction, self._wording), self._tool_name
        )
        self._inspect(correction, text)
        return text

    def _inspect(self, correction, text: str) -> None:
        """Show an output correction to development inspection (#373)."""
        inspection_event(
            "Fault message",
            attempt_id=self._attempt_id,
            slot="retry" if isinstance(correction, OutputCalls) else "output_slot",
            text=text,
            **fault_trace(correction, self._wording),
        )

    def latest_validation(
        self, collection_member_id: str
    ) -> StructuredOutputValidationAttempt | None:
        """The member's latest recorded validation, or ``None`` before its first."""
        attempts = self._attempts.get(collection_member_id)
        return attempts[-1] if attempts else None

    def lineage(self, collection_member_id: str) -> StructuredOutputLineage:
        attempts = tuple(self._attempts.get(collection_member_id, ()))
        if not attempts:
            raise Invariant(
                "structured_output_member_unvalidated",
                "structured-output member lacks validation",
            )
        return StructuredOutputLineage(
            collection_member_id=collection_member_id,
            tool_name=self._tool_name,
            max_retries=self._max_retries,
            validation_attempts=attempts,
            accepted_output=(
                attempts[-1].output if attempts[-1].status == "valid" else None
            ),
        )

    def _require_member(self) -> str:
        if self._collection_member_id is None:
            raise Invariant(
                "structured_output_member_absent",
                "structured-output response lacks a collection member",
            )
        return self._collection_member_id

    def _record_validation(
        self,
        *,
        response_event_id: str,
        calls: tuple[dict[str, Any], ...],
        output: dict[str, JsonValue],
        errors: tuple[str, ...],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        member_id = self._require_member()
        validation = self._parent_activity.record(
            AgentOutputValidated(
                attempt_id=self._attempt_id,
                agent_role=self._agent_role,
                status="invalid" if errors else "valid",
                output=output,
                errors=errors,
                feedback=feedback,
            )
        )
        link = self._parent_activity.record(
            StructuredOutputValidationLinked(
                attempt_id=self._attempt_id,
                agent_role=self._agent_role,
                collection_member_id=member_id,
                tool_name=self._tool_name,
                validation_attempt=len(self._attempts[member_id]) + 1,
                response_event_id=response_event_id,
                output_call_event_ids=tuple(call["event_id"] for call in calls),
                validation_event_id=validation["event_id"],
            )
        )
        attempt = StructuredOutputValidationAttempt(
            validation_attempt=len(self._attempts[member_id]) + 1,
            response_event_id=response_event_id,
            output_call_event_ids=tuple(call["event_id"] for call in calls),
            output_call_ids=tuple(call["payload"]["tool_call_id"] for call in calls),
            validation_event_id=validation["event_id"],
            validation_link_event_id=link["event_id"],
            status=validation["payload"]["status"],
            output=validation["payload"]["output"],
            errors=tuple(validation["payload"]["errors"]),
        )
        self._attempts[member_id].append(attempt)
        self._pending = None
        if errors:
            self._failure_source.capture(validation)
        return validation


def _routed_reply_closure(
    routed: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
    response_event_id: str,
) -> Mapping[str, Any] | None:
    """The reply closure behind a routed turn, or ``None`` when its chain changed.

    Routing records, in order after the reply's closure, only the reply's
    truncation and its unoffered calls' own refusals, then the routed turn.
    """

    def parent(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
        parents = event.get("parent_ids", [])
        return by_id.get(parents[0]) if len(parents) == 1 else None

    event = parent(routed)
    for validation_id in reversed(
        routed.get("payload", {}).get("unoffered_validation_event_ids", [])
    ):
        if (
            event is None
            or event.get("event_id") != validation_id
            or event.get("event_type") != "ToolCallValidated"
        ):
            return None
        event = parent(event)
    if (
        event is not None
        and event.get("event_type") == "ModelCompletionTruncated"
        and event.get("payload", {}).get("response_event_id") == response_event_id
    ):
        event = parent(event)
    return event


def project_structured_output_lineages(  # noqa: C901 - Existing orchestration complexity; contract-base migration preserves this flow.
    events: Iterable[Mapping[str, Any]],
    *,
    attempt_id: str,
    agent_role: str,
    tool_name: str,
    max_retries: int,
    prior_attempt_ids: tuple[str, ...] = (),
    retained_lineages: tuple[StructuredOutputLineage, ...] = (),
) -> tuple[StructuredOutputLineage, ...]:
    """Recompute exact response/call/validation lineage from retained Journey facts."""

    timeline = [dict(event) for event in events]
    owned_attempts = {*prior_attempt_ids, attempt_id}
    by_id = {event.get("event_id"): event for event in timeline}
    calls_by_id: dict[str, tuple[Mapping[str, Any], StructuredOutputCallRecorded]] = {}
    call_ids_by_response: dict[str, list[str]] = defaultdict(list)
    last_call_by_response: dict[str, str] = {}
    # A routed reply's output is recorded only when its routed turn kept it.
    kept_by_response = {
        event["payload"]["response_event_id"]: event["payload"]["run_order"]
        for event in timeline
        if event.get("event_type") == "AgentTurnRouted"
    }
    for event in timeline:
        if event.get("event_type") != "StructuredOutputCallRecorded":
            continue
        fact = StructuredOutputCallRecorded.model_validate(
            {"event_type": event["event_type"], **event.get("payload", {})}
        )
        if (
            fact.attempt_id not in owned_attempts
            or fact.agent_role != agent_role
            or fact.tool_name != tool_name
        ):
            continue
        if fact.response_event_id in kept_by_response and (
            fact.tool_call_id not in kept_by_response[fact.response_event_id]
        ):
            raise ValueError("structured-output call was refused by its routed turn")
        response = by_id.get(fact.response_event_id)
        call_parents = event.get("parent_ids", [])
        previous_call_id = last_call_by_response.get(fact.response_event_id)
        response_terminal = None
        if previous_call_id is None:
            response_terminal = (
                by_id.get(call_parents[0])
                if isinstance(call_parents, list) and len(call_parents) == 1
                else None
            )
        # A reply's first output call follows the reply, or its routed turn when
        # admission routed it (ADR 0044).
        if (
            response_terminal is not None
            and response_terminal.get("event_type") == "AgentTurnRouted"
            and response_terminal.get("payload", {}).get("response_event_id")
            == fact.response_event_id
        ):
            response_terminal = _routed_reply_closure(
                response_terminal, by_id, fact.response_event_id
            )
        if (
            response is None
            or response.get("event_type") != "ModelResponseRecorded"
            or response.get("payload", {}).get("attempt_id") != fact.attempt_id
            or (
                previous_call_id is None
                and (
                    response_terminal is None
                    or response_terminal.get("event_type") != "activity_succeeded"
                    or response_terminal.get("payload", {}).get("result_event_id")
                    != fact.response_event_id
                )
            )
            or (previous_call_id is not None and call_parents != [previous_call_id])
        ):
            raise ValueError("structured-output call changed its model response")
        response_parts = (
            response.get("payload", {})
            .get("compatible_adapter_evidence", {})
            .get("response", {})
            .get("message", {})
            .get("parts", [])
        )
        matching = [
            part
            for part in response_parts
            if part.get("part_kind") == "tool-call"
            and part.get("tool_name") == fact.tool_name
            and part.get("tool_call_id") == fact.tool_call_id
        ]
        if len(matching) != 1:
            raise ValueError("structured-output call identity changed its response")
        read = read_arguments(matching[0].get("args"), fact.arguments_decoded)
        if (
            read.value != fact.arguments
            or read.fenced != fact.arguments_fenced
            or read.decoded != fact.arguments_decoded
        ):
            raise ValueError("structured-output call changed its arguments")
        calls_by_id[event["event_id"]] = (event, fact)
        call_ids_by_response[fact.response_event_id].append(event["event_id"])
        last_call_by_response[fact.response_event_id] = event["event_id"]

    attempts_by_member: dict[str, list[StructuredOutputValidationAttempt]] = (
        defaultdict(list)
    )
    consumed_call_event_ids: set[str] = set()
    member_order: list[str] = []
    for lineage in retained_lineages:
        if (
            lineage.tool_name != tool_name
            or lineage.max_retries != max_retries
            or lineage.collection_member_id in attempts_by_member
        ):
            raise ValueError("retained structured-output lineage changed its scope")
        member_order.append(lineage.collection_member_id)
        attempts_by_member[lineage.collection_member_id].extend(
            lineage.validation_attempts
        )
    for event in timeline:
        if event.get("event_type") != "StructuredOutputValidationLinked":
            continue
        fact = StructuredOutputValidationLinked.model_validate_json(
            json.dumps({"event_type": event["event_type"], **event.get("payload", {})}),
            strict=True,
        )
        if (
            fact.attempt_id not in owned_attempts
            or fact.agent_role != agent_role
            or fact.tool_name != tool_name
        ):
            continue
        validation = by_id.get(fact.validation_event_id)
        expected_call_event_ids = tuple(
            call_ids_by_response.get(fact.response_event_id, ())
        )
        if fact.output_call_event_ids != expected_call_event_ids:
            raise ValueError("structured-output validation changed its full call set")
        if any(item in consumed_call_event_ids for item in expected_call_event_ids):
            raise ValueError("structured-output call was not consumed exactly once")
        call_entries = [calls_by_id.get(item) for item in fact.output_call_event_ids]
        if validation is not None and validation.get("parent_ids") != [
            expected_call_event_ids[-1]
        ]:
            raise ValueError("structured-output validation changed its exact call")
        if (
            validation is None
            or validation.get("event_type") != "AgentOutputValidated"
            or validation.get("payload", {}).get("attempt_id") != fact.attempt_id
            or validation.get("payload", {}).get("agent_role") != agent_role
            or event.get("parent_ids") != [fact.validation_event_id]
            or any(item is None for item in call_entries)
            or any(
                item[1].attempt_id != fact.attempt_id
                or item[1].collection_member_id != fact.collection_member_id
                or item[1].response_event_id != fact.response_event_id
                for item in call_entries
                if item is not None
            )
        ):
            raise ValueError("structured-output validation changed its exact lineage")
        validation_payload = validation.get("payload", {})
        call_facts = [item[1] for item in call_entries if item is not None]
        if len(call_facts) == 1:
            expected_output = (
                call_facts[0].schema_normalized_arguments
                if call_facts[0].schema_normalized_arguments is not None
                else call_facts[0].arguments
            )
            if validation_payload.get("output") != expected_output:
                raise ValueError("structured-output validation changed its exact call")
        elif (
            validation_payload.get("status") != "invalid"
            or validation_payload.get("output") != {}
            or validation_payload.get("errors") != ["multiple_output_calls"]
        ):
            raise ValueError(
                "structured-output duplicate call changed its invalid validation"
            )
        consumed_call_event_ids.update(expected_call_event_ids)
        if fact.collection_member_id not in attempts_by_member:
            member_order.append(fact.collection_member_id)
        attempt = StructuredOutputValidationAttempt(
            validation_attempt=fact.validation_attempt,
            response_event_id=fact.response_event_id,
            output_call_event_ids=fact.output_call_event_ids,
            output_call_ids=tuple(
                item[1].tool_call_id for item in call_entries if item is not None
            ),
            validation_event_id=fact.validation_event_id,
            validation_link_event_id=event["event_id"],
            status=validation["payload"]["status"],
            output=validation["payload"]["output"],
            errors=tuple(validation["payload"]["errors"]),
        )
        attempts_by_member[fact.collection_member_id].append(attempt)
    if set(calls_by_id) != consumed_call_event_ids:
        raise ValueError("structured-output call was not consumed exactly once")
    return tuple(
        StructuredOutputLineage(
            collection_member_id=member_id,
            tool_name=tool_name,
            max_retries=max_retries,
            validation_attempts=tuple(attempts_by_member[member_id]),
            accepted_output=(
                attempts_by_member[member_id][-1].output
                if attempts_by_member[member_id][-1].status == "valid"
                else None
            ),
        )
        for member_id in member_order
    )


__all__ = [
    "normalized_output",
    "output_errors",
    "StructuredOutputCallRecorded",
    "StructuredOutputLineage",
    "StructuredOutputSession",
    "StructuredOutputValidationAttempt",
    "StructuredOutputValidationLinked",
    "SeveralOutputs",
    "project_structured_output_lineages",
]
