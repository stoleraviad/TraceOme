"""What one Investigation member's first request tells the model (#444).

The member context is a recorded fact of pointers into the immutable Journey
(owner, #444 Q1–Q3): the Investigation and the bound work it belongs to, the
member's own latest source results, the researcher's answers and instruction,
and the one thing decided as the member starts, the Tools it may and may not
use. The model reads it rendered, byte for byte as before the fact existed;
reopen renders it again from the same events and checks the bytes, once per
fact. Readers take the member's request from it instead of parsing prompts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import Field, StrictStr

from bio_neb.agentic_journey.role_declaration import role
from bio_neb.agentic_journey.role_instructions import RoleInstructions
from bio_neb.contracts import StrictFrozenContract
from bio_neb.journey import canonical_json

# Parses one admitted event's Investigation fact; a reading owner parses each once.
Parse = Callable[[Mapping[str, Any]], Any]

# The source results a member's assignment context shows: its latest of each.
SOURCE_RESULTS = ("RefSnpToolResultRecorded", "AssemblyReportToolResultRecorded")


class InvestigationMemberContextRecorded(StrictFrozenContract):
    """One member's context, recorded once before the member's first request."""

    event_type: Literal["InvestigationMemberContextRecorded"] = (
        "InvestigationMemberContextRecorded"
    )
    attempt_id: StrictStr
    collection_member_id: StrictStr
    # The Investigation: its input and the research goal the member serves.
    started_event_id: StrictStr
    # The Attempt's bound work: the member's own request.
    bound_event_id: StrictStr
    # The member's latest source results, one of each kind at most.
    source_result_event_ids: tuple[StrictStr, ...] = ()
    # The researcher's answers to this member's questions, in order.
    answer_event_ids: tuple[StrictStr, ...] = ()
    # The Review response whose instruction the member reads, if any.
    instruction_event_id: StrictStr | None = None
    eligible_source_operations: tuple[StrictStr, ...] = Field(min_length=1)
    forbidden_operations: tuple[StrictStr, ...] = ()
    # The item's last lookup for a target a researcher's answer or a new revision
    # replaced, and the event that replaced it (the record kept): shown marked
    # replaced, so the model does not drift back. Left out otherwise.
    replaced_result_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    replaced_by_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The paper reading's answer that proposed the member's target, when a paper
    # asserts it: the member receives the paper and what planning's reading kept
    # for it. Left out otherwise.
    paper_answer_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def latest_source_results(
    events, collection_member_id: str
) -> dict[str, Mapping[str, Any]]:
    """The member's latest source result of each kind, as its assignment shows."""
    latest: dict[str, Mapping[str, Any]] = {}
    for event in events:
        if (
            event["event_type"] in SOURCE_RESULTS
            and event["payload"]["collection_member_id"] == collection_member_id
        ):
            latest[event["event_type"]] = event
    return latest


def member_context_fact(
    request: Mapping[str, Any], timeline, parse: Parse
) -> InvestigationMemberContextRecorded:
    """The member context a model request's transcript holds: its member's latest
    context recorded before it, by the request's own Attempt or one that Attempt
    exactly continues (a resumed transcript keeps the context its first Attempt
    recorded)."""
    from bio_neb.agentic_journey.model_availability import prior_model_attempt_ids

    attempt_id = request["payload"]["attempt_id"]
    owners = {attempt_id, *prior_model_attempt_ids(timeline, attempt_id)}
    member_id = request["payload"]["member_id"]
    found = None
    for event in timeline:
        if event["event_id"] == request["event_id"]:
            break
        if (
            event["event_type"] == "InvestigationMemberContextRecorded"
            and event["payload"]["collection_member_id"] == member_id
            and event["payload"]["attempt_id"] in owners
        ):
            found = event
    if found is None:
        raise ValueError("model request has no recorded member context")
    return parse(found)


def member_request(
    fact: InvestigationMemberContextRecorded,
    by_id: Mapping[str, Mapping[str, Any]],
    parse: Parse,
):
    """The member's own request, from the Attempt's bound work."""
    from bio_neb.investigation.investigation import (
        InvestigationAttemptBound,
        investigation_output_member_id,
    )

    bound = by_id.get(fact.bound_event_id)
    work = None if bound is None else parse(bound)
    if not isinstance(work, InvestigationAttemptBound):
        raise ValueError("member context names no bound Investigation work")
    for ordinal, item in enumerate(work.work_requests):
        if investigation_output_member_id(item, ordinal) == fact.collection_member_id:
            return item
    raise ValueError("member context names a member its bound work does not hold")


def member_job(instructions: RoleInstructions, *, positioned: bool) -> str:
    """The objective a member's message states. Nemotron acts on its user message,
    so its member reads its own job there: stating the job stopped its genotype
    questions (#398 arm A1; #401). Each form's words are its own (map #23 D5)."""
    return instructions.message(
        role("investigation_planner").job,
        lookup=instructions.message(
            "job_lookup_position" if positioned else "job_lookup_rsid"
        ),
    )


def render_member_context(
    fact: InvestigationMemberContextRecorded,
    by_id: Mapping[str, Mapping[str, Any]],
    instructions: RoleInstructions,
    parse: Parse,
) -> dict[str, Any]:
    """The member context the model reads, rendered from the fact's events."""
    from bio_neb.agentic_journey.assignment_context import member_assignment
    from bio_neb.investigation.investigation import InvestigationStarted
    from bio_neb.investigation.investigation_request import untyped

    started = by_id.get(fact.started_event_id)
    investigation = None if started is None else parse(started)
    if not isinstance(investigation, InvestigationStarted):
        raise ValueError("member context names no started Investigation")
    request = member_request(fact, by_id, parse)
    answers = [_answer(by_id, event_id, parse) for event_id in fact.answer_event_ids]
    # Positioned as its recorded operations say: a typed or read position, or a region,
    # is located by locate_variant.
    positioned = "locate_variant" in fact.eligible_source_operations
    results = {}
    for event_id in fact.source_result_event_ids:
        result = by_id.get(event_id)
        if (
            result is None
            or result["event_type"] not in SOURCE_RESULTS
            or result["payload"]["collection_member_id"] != fact.collection_member_id
            or result["event_type"] in results
        ):
            raise ValueError("member context names another member's source result")
        results[result["event_type"]] = result
    context = {
        "collection_member_id": fact.collection_member_id,
        "assignment": member_assignment(
            started, member_job(instructions, positioned=positioned), results
        ),
        "investigation_input": investigation.investigation_input.model_dump(
            mode="json"
        ),
        "user_assertion": request.model_dump(mode="json"),
        "eligible_source_operations": fact.eligible_source_operations,
        "forbidden_operations": fact.forbidden_operations,
        "researcher_answers": answers,
    }
    if untyped(request):
        # The event an untyped item's own words are read by.
        context["request_words_event_id"] = fact.started_event_id
    if answers:
        # After an answer, the product's next step: the wording that let
        # members proceed (11 times of 14 in a measured run) instead of asking
        # again.
        context["next_step"] = instructions.message(
            "next_step",
            assembly=investigation.investigation_input.assembly_accession,
            # The member's own lookups: locate_variant, or both rsID lookup Tools,
            # the source the model's to choose.
            lookup=" or ".join(
                name
                for name in fact.eligible_source_operations
                if name in ("locate_variant", "fetch_refsnp", "fetch_eva_variant")
            ),
        )
    if fact.replaced_result_event_id is not None:
        context["replaced_target"] = _replaced(fact, by_id, instructions)
    if fact.paper_answer_event_id is not None:
        from bio_neb.paper_reading.investigation import paper_context

        context["paper"] = paper_context(by_id, fact.paper_answer_event_id, request)
    if fact.instruction_event_id is not None:
        # A continue answer's text: a User assertion, not verified fact.
        context["researcher_instruction"] = _instruction(
            by_id.get(fact.instruction_event_id)
        )
    return context


def _replaced(
    fact: InvestigationMemberContextRecorded,
    by_id: Mapping[str, Mapping[str, Any]],
    instructions: RoleInstructions,
) -> str:
    """The replaced target's last lookup call and result, marked replaced by the
    event that replaced it (owner #479 Q2 A)."""
    result = by_id.get(fact.replaced_result_event_id)
    replaced_by = by_id.get(fact.replaced_by_event_id or "")
    # The item's earlier revision may hold it under its own request's member id;
    # reopen checks it is exactly the item history's (`investigation`).
    if (
        result is None
        or result["event_type"] != "RefSnpToolResultRecorded"
        or replaced_by is None
    ):
        raise ValueError("member context names no replaced lookup of its item")
    call = next(
        (
            event["payload"]
            for event in by_id.values()
            if event["event_type"] == "ToolCallValidated"
            and event["payload"]["attempt_id"] == result["payload"]["attempt_id"]
            and event["payload"]["tool_call_id"] == result["payload"]["tool_call_id"]
        ),
        None,
    )
    if call is None:
        raise ValueError("member context's replaced lookup names no call")
    return instructions.message(
        "replaced_target",
        by=fact.replaced_by_event_id,
        call=f"{call['tool_name']}({canonical_json(call['arguments'])})",
        result=fact.replaced_result_event_id,
        status=result["payload"]["result"]["status"],
    )


def rendered_text(
    fact: InvestigationMemberContextRecorded,
    by_id: Mapping[str, Mapping[str, Any]],
    instructions: RoleInstructions,
    parse: Parse,
) -> str:
    """The member context as its request's user part carries it."""
    return canonical_json(render_member_context(fact, by_id, instructions, parse))


def _answer(
    by_id: Mapping[str, Mapping[str, Any]], event_id: str, parse: Parse
) -> dict[str, Any]:
    """One researcher answer with the member's own question it answers."""
    from bio_neb.investigation.investigation import (
        InvestigationReviewRequested,
        InvestigationReviewResponded,
    )

    response = by_id.get(event_id)
    answered = None if response is None else parse(response)
    if not isinstance(answered, InvestigationReviewResponded):
        raise ValueError("member context names no researcher answer")
    request = parse(by_id[answered.request_event_id])
    if not isinstance(request, InvestigationReviewRequested):
        raise ValueError("member context names an answer to no Review")
    question = None if request.question is None else request.question.question
    return {
        "question": question,
        **answered.response.model_dump(mode="json"),
        # The event a target reading of this answer names.
        "response_event_id": event_id,
    }


# Where each Review response keeps the instruction a member reads.
_INSTRUCTIONS: Mapping[str, Callable[[Mapping[str, Any]], Any]] = {
    "AgentWorkReviewResponded": lambda response: response["instruction"],
    "ModelAvailabilityReviewResponded": lambda response: response["instruction"],
    "AgentFailureReviewResponded": lambda response: response["value"],
}


def _instruction(event: Mapping[str, Any] | None) -> str:
    reader = None if event is None else _INSTRUCTIONS.get(event["event_type"])
    text = None if reader is None else reader(event["payload"]["response"])
    if not isinstance(text, str):
        raise ValueError("member context names no researcher instruction")
    return text
