"""The report's explanation Agent (#109, #408, ADR 0038).

A researcher requests an explanation of one published Grounded report
(``report_commands.request_report_explanation``); the coordinator then takes one
step at a time (``advance_report_explanation``). Each request of the report is
explained by its own member Attempt over that request's own rows, the members
running together under the Run's ``planner_members`` bound; an answered Review
re-enters only its own member, and a rejected one settles that request as not
explained. Once every member is settled, one combination Attempt writes the
combined answer over them, published as its own immutable artifact.

Every Attempt runs through the shared lifecycle and assembly: it offers no
domain Tool, its Journey search reads only its bound evidence scope, and it
ends at one typed output whose claim fields are checked against its rows. The
report is never written again. A one-Attempt (``/1``) request of an earlier
Run still reads back, but no new one-Attempt work starts.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

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
    RuntimeBinding,
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
from bio_neb.agentic_journey.structured_output import StructuredOutputSession
from bio_neb.application.input_journey import RunInputsRegistered, validate_fact
from bio_neb.journey import Journey, canonical_json
from bio_neb.recording import evidence_view, read_evidence
from bio_neb.reporting.explanation import (
    EXPLANATION_MEDIA_TYPE,
    EXPLANATION_OUTPUT_TOOL,
    EXPLANATION_ROLE,
    ReportExplanationCombinationBound,
    ReportExplanationFailed,
    ReportExplanationMemberBound,
    ReportExplanationPublished,
    ReportExplanationRequested,
    ReportExplanationView,
    admitted_members,
    bound_fact,
    bound_member,
    derive_explanation_by_request,
    derive_member_admitted,
    draft_binding,
    member_binding,
    project_report_explanations,
    settled_sections,
    validate_explanation_attempt_facts,
)
from bio_neb.reporting.explanation_draft import (
    CombinationDraft,
    MemberDraft,
    Refused,
    check,
)
from bio_neb.reporting.explanation_members import (
    COMBINATION_OUTPUT_TOOL,
    CombinationSection,
    combination_qualifications,
    row_claim,
)
from bio_neb.reporting.explanation_rows import (
    ExplanationRow,
    ReportQualification,
    explanation_qualifications,
    explanation_rows,
    explanation_scope,
)

ROLE = EXPLANATION_ROLE
MEMBER_STAGE = "report_explanation"
COMBINATION_STAGE = "report_combination"
OUTPUT_TOOL = EXPLANATION_OUTPUT_TOOL
_SHARED_TERMINALS = {
    "AttemptStopped": "stopped",
    "AttemptFailed": "failed",
    "AttemptCancelled": "cancelled",
}


@dataclass(frozen=True)
class ExplanationWork:
    """One step of one request, as its Attempt sees it: a request's member over
    that request's targets (``member_id``), or the combination over every
    request's settled section (``member_id`` None)."""

    request_event_id: str
    request: ReportExplanationRequested
    member_id: str | None
    question: str | None
    outcome: str
    targets: tuple[ExplanationRow, ...]
    qualifications: tuple[ReportQualification, ...]
    evidence: tuple[dict[str, str], ...]
    runtime_selections: Any
    research_goals: tuple[str, ...] = ()
    # A member's request in the researcher's words and its positions' nested
    # observations with their own events; sent, never bound.
    request_text: str | None = None
    position_evidence: tuple[dict[str, Any], ...] = ()
    # The combination's sections, the qualifications it owns and each
    # request's section as the model reads it.
    sections: tuple[CombinationSection, ...] = ()
    combination_qualifications: tuple[int, ...] = ()
    members: tuple[dict[str, Any], ...] = ()
    continued_attempt_id: str | None = None
    researcher_instruction: str | None = None
    # The continued Attempt's last refused draft and what refused it, so it
    # corrects that draft instead of starting over (live a3); sent, never bound.
    previous_attempt: dict[str, Any] | None = None

    @property
    def evidence_event_ids(self) -> tuple[str, ...]:
        return tuple(sorted(item["event_id"] for item in self.evidence))

    @property
    def stage(self) -> str:
        return COMBINATION_STAGE if self.member_id is None else MEMBER_STAGE


@dataclass(frozen=True)
class AdmittedExplanation:
    work: ExplanationWork


@dataclass
class ExplanationDependencies:
    """One Attempt's state; nothing is shared with another step's Attempt."""

    journey: Journey
    attempt_id: str
    work: ExplanationWork
    outputs: StructuredOutputSession
    observed_tool_session: Any
    bound: ReportExplanationMemberBound | ReportExplanationCombinationBound
    validation_event_id: str | None = None
    draft: MemberDraft | CombinationDraft | None = None


def _retained_registration(events) -> RunInputsRegistered:
    registrations = [
        validate_fact(event)
        for event in events
        if event.get("event_type") == "RunInputsRegistered"
    ]
    if len(registrations) != 1 or not isinstance(registrations[0], RunInputsRegistered):
        raise ValueError("the explanation requires the Run's one registration")
    return registrations[0]


def _last_refusal(events, attempt_id: str | None) -> dict[str, Any] | None:
    """An Attempt's last refused output and its correction, or None."""
    refused = [
        event["payload"]
        for event in events
        if event["event_type"] == "AgentOutputValidated"
        and event["payload"]["attempt_id"] == attempt_id
        and event["payload"]["status"] == "invalid"
    ]
    if not refused:
        return None
    last = refused[-1]
    return {
        "draft": last["output"],
        "refusal": last.get("feedback") or "; ".join(last["errors"]),
    }


def _work(
    journey: Journey,
    request_event_id: str,
    member_id: str | None,
    *,
    continued_attempt_id: str | None = None,
    researcher_instruction: str | None = None,
) -> ExplanationWork:
    from bio_neb.reporting.explanation import _report_before, _RequestPrefixes

    # The owner's shared read-only evidence: a member reads it, never copies it.
    events = evidence_view(journey)
    index = next(
        position
        for position, event in enumerate(events)
        if event["event_id"] == request_event_id
    )
    request = ReportExplanationRequested.model_validate(
        {"event_type": "ReportExplanationRequested", **events[index]["payload"]}
    )
    # One request's report is fixed by the events before it: every member of a
    # batch and every other reader of the owner's evidence reads the one
    # derivation (ADR 0032).
    report = _report_before(
        _RequestPrefixes(events, journey), request_event_id, request.report_digest
    ).report
    question = None if report.question is None else report.question.value
    rows = explanation_rows(report.collection_view)
    common = {
        "request_event_id": request_event_id,
        "request": request,
        "member_id": member_id,
        "question": None if question is None else question.request_text,
        "outcome": report.outcome,
        "runtime_selections": _retained_registration(events).runtime_selections,
        "research_goals": () if question is None else question.research_goals,
        "continued_attempt_id": continued_attempt_id,
        "researcher_instruction": researcher_instruction,
        "previous_attempt": _last_refusal(events, continued_attempt_id)
        if continued_attempt_id is not None
        else None,
    }
    if member_id is not None:
        own, qualifications, member_ids = member_binding(
            events[:index], report, request.publication_event_id, member_id
        )
        allowed = set(member_ids)
        return ExplanationWork(
            **common,
            targets=own,
            qualifications=qualifications,
            evidence=_index(events, allowed),
            request_text=next(
                snp.request_text
                for snp in report.collection_view.snps
                if snp.request_member_id == member_id
            ),
            position_evidence=_position_evidence(
                report.collection_view, own, events, allowed
            ),
        )
    qualifications = explanation_qualifications(report)
    sections = settled_sections(events, request_event_id, rows)
    if sections is None:
        raise ValueError("the combined answer waits until every member is settled")
    admitted = admitted_members(events)
    return ExplanationWork(
        **common,
        targets=rows,
        qualifications=qualifications,
        evidence=_index(
            events, set(explanation_scope(report, request.publication_event_id))
        ),
        sections=sections,
        combination_qualifications=tuple(
            item.qualification
            for item in combination_qualifications(rows, qualifications)
        ),
        members=tuple(_member_entry(section, admitted) for section in sections),
    )


def _index(events, allowed) -> tuple[dict[str, str], ...]:
    return tuple(
        {"event_id": event["event_id"], "event_type": event["event_type"]}
        for event in events
        if event["event_id"] in allowed
    )


def _member_entry(section: CombinationSection, admitted) -> dict[str, Any]:
    """One request as the combination reads it: its member's claim per target and
    why, or the reason it was not explained."""
    if section.status == "not_explained":
        return {
            "request": section.request,
            "subject": section.subject,
            "status": section.status,
            "reason": section.reason,
        }
    bound, fact, _instructions = admitted[section.attempt_id]
    subjects = {row.number: row.subject for row in bound.targets}
    return {
        "request": section.request,
        "subject": section.subject,
        "status": section.status,
        "claims": [
            {"subject": subjects[claim.target], **claim.model_dump(mode="json")}
            for claim in fact.claims
        ],
        "claim_words": list(section.claims),
        "why": fact.why.text,
    }


def _item_events(events, scope) -> dict[str, list[str]]:
    """Each Work item's in-scope events, in Journey order: its lookup's evidence,
    its window and the table reading that read it."""
    held: dict[str, list[str]] = {}
    for event in events:
        if event["event_id"] not in scope:
            continue
        payload = event["payload"]
        if event["event_type"] == "InvestigationEvidenceRecorded":
            owners = [(payload.get("sample_observation") or {}).get("work_item_id")]
        elif event["event_type"] == "InvestigationItemWindowRecorded":
            owners = [payload["work_item_id"]]
        elif event["event_type"] == "InvestigationTableChunkRead":
            owners = [reading["work_item_id"] for reading in payload["readings"]]
        else:
            continue
        for owner in owners:
            if owner is not None:
                held.setdefault(owner, []).append(event["event_id"])
    return held


def _position_evidence(view, rows, events, scope) -> tuple[dict[str, Any], ...]:
    """Each searched position's lookup outcome apart from its nested observation
    (owner D4): the window, coverage and reading the approved basis recorded, and
    the events that hold them, which the scoped search may read. A position never
    searched has no outcome and no events (Spec r1 F1, F2)."""
    if view is None:
        return ()
    items = {item.work_item_id: item for snp in view.snps for item in snp.items}
    held = _item_events(events, scope)
    evidence = []
    for row in rows:
        item = items.get(row.work_item_id) if row.kind == "position" else None
        if item is None:
            continue
        evidence.append(
            {
                "target": row.number,
                "lookup": item.outcome,
                "window": None
                if item.window is None
                else item.window.model_dump(mode="json"),
                "coverage": item.coverage or "not_measured",
                "reading": item.reading,
                "concerns": list(item.concerns),
                "reading_note": item.reading_note,
                "events": held.get(item.work_item_id, []),
            }
        )
    return tuple(evidence)


def _explanations(journey: Journey) -> ReportExplanationView:
    """The Run's explanation requests on the owner's current evidence, derived
    once for every Attempt a batch admits there (ADR 0032).

    The view reads the active Pass journal too, so it is the current snapshot's
    facet, not a value fixed by the head event.
    """
    from bio_neb.application.planning_read import prefix_snapshot

    return prefix_snapshot(journey).derived(_snapshot_explanations)


def _snapshot_explanations(snapshot) -> ReportExplanationView:
    return project_report_explanations(snapshot.timeline, snapshot=snapshot)


def _admit(
    journey: Journey, reentry: Reentry, member_id: str | None
) -> AttemptAdmission[AdmittedExplanation]:
    events = evidence_view(journey)
    direction = review_direction(journey, reentry, agent_role=ROLE)
    if direction.previous_attempt_id is None:
        current = _explanations(journey).current
        step = None
        if current is not None and current.combination is not None:
            step = (
                current.combination
                if member_id is None
                else next(
                    (item for item in current.members if item.member_id == member_id),
                    None,
                )
            )
        if step is None or step.status != "waiting":
            raise ValueError("no explanation step is waiting for its Attempt")
        work = _work(journey, current.request_event_id, member_id)
    else:
        previous = bound_fact(events, direction.previous_attempt_id)
        if not isinstance(
            previous, ReportExplanationMemberBound | ReportExplanationCombinationBound
        ):
            # Its single Attempt is retired; a reject closes its request.
            raise ValueError("a one-Attempt explanation does not continue")
        work = _work(
            journey,
            previous.request_event_id,
            bound_member(previous),
            continued_attempt_id=direction.previous_attempt_id,
            researcher_instruction=direction.instruction,
        )
    registration = _retained_registration(events)
    return AttemptAdmission(
        AdmittedExplanation(work),
        RuntimeBinding(registration.runtime_selections, registration.input_set_id),
        direction.causal_head,
        direction.availability_frontier,
        member_id=work.member_id,
        holds_from_event_id=work.request_event_id,
    )


def _refusal(bound, arguments: dict[str, Any]) -> Refused | None:
    """The one correction a draft's raw arguments meet, or None when admitted."""
    result = check(draft_binding(bound), arguments)
    return result if isinstance(result, Refused) else None


def current_checks(bound, draft: dict[str, Any]) -> dict[str, list[str]]:
    """What the current checks refuse in an earlier draft over this Attempt's
    bound rows (live a4 L20), as facts: prospective guidance beside the refusal
    the draft met when it was recorded, which stays history. Empty when the
    current checks admit it."""
    refused = _refusal(bound, draft)
    return {
        "codes": [] if refused is None else list(refused.faults.codes),
        "faults": [] if refused is None else list(refused.faults.sentences),
    }


def _validate_draft(context: ToolContext[ExplanationDependencies], output):
    deps = context.deps
    arguments = output.model_dump(mode="json")
    refused = _refusal(deps.bound, arguments)
    if refused is not None:
        deps.outputs.validate(arguments, refused.codes, faults=refused.faults)
        raise OutputRepair("; ".join(refused.codes), refused.faults)
    valid = deps.outputs.validate(arguments, ())
    deps.validation_event_id = valid["event_id"]
    deps.draft = output
    return output


MEMBER_OUTPUT = AssignmentOutput(
    output_type=MemberDraft,
    name=OUTPUT_TOOL,
    description=(
        "Record this request's explanation: "
        + ", ".join(MemberDraft.model_fields)
        + "."
    ),
    validator=_validate_draft,
)
COMBINATION_OUTPUT = AssignmentOutput(
    output_type=CombinationDraft,
    name=COMBINATION_OUTPUT_TOOL,
    description=(
        "Record the report's conclusion: "
        + ", ".join(CombinationDraft.model_fields)
        + "."
    ),
    validator=_validate_draft,
)


def _prompt(work: ExplanationWork, job: str) -> str:
    """The step's task: its stage's job, then a member's request and rows, or every
    request's section."""
    task: dict[str, Any] = {
        "question": work.question,
        "research_goals": list(work.research_goals),
        "outcome": work.outcome,
    }
    if work.member_id is not None:
        task.update(
            task=job,
            request=work.request_text,
            # Each target with the claim its recorded state reads as; a
            # position's Work item id stays out of what the model reads.
            targets=[
                {
                    **row.model_dump(mode="json", exclude={"work_item_id"}),
                    "claim": row_claim(row),
                }
                for row in work.targets
            ],
            qualifications=[
                item.model_dump(mode="json") for item in work.qualifications
            ],
            position_evidence=list(work.position_evidence),
        )
    else:
        owned = set(work.combination_qualifications)
        task.update(
            task=job,
            requests=list(work.members),
            qualifications=[
                item.model_dump(mode="json")
                for item in work.qualifications
                if item.qualification in owned
            ],
        )
    task.update(
        evidence=list(work.evidence),
        researcher_instruction=work.researcher_instruction,
    )
    if work.previous_attempt is not None:
        task["task"] += (
            " previous_attempt holds your last refused draft, the refusal it met "
            "then, and current_checks: what the current checks refuse in it now. "
            "Correct that draft for current_checks, keeping what it got right, "
            "rather than starting over."
        )
        task["previous_attempt"] = work.previous_attempt
    return canonical_json(task)


def _bound(context: AttemptContext[AdmittedExplanation], work: ExplanationWork):
    continuation = context.effective_policy.continuation
    common = {
        "attempt_id": context.attempt_id,
        "request_event_id": work.request_event_id,
        "continued_attempt_id": work.continued_attempt_id,
        "report_digest": work.request.report_digest,
        "report_basis_digest": work.request.report_basis_digest,
        "publication_event_id": work.request.publication_event_id,
        "evidence_event_ids": work.evidence_event_ids,
        "runtime_selections": work.runtime_selections,
        "continuation_policy": continuation,
        "continuation_policy_id": continuation.policy_id,
        "researcher_instruction": work.researcher_instruction,
    }
    if work.member_id is not None:
        return ReportExplanationMemberBound(
            **common,
            request_member_id=work.member_id,
            targets=work.targets,
            qualifications=work.qualifications,
        )
    return ReportExplanationCombinationBound(
        **common,
        targets=work.targets,
        qualifications=work.qualifications,
        combination_qualifications=work.combination_qualifications,
        sections=work.sections,
    )


async def _drive(context, invocation: BoundInvocation, invoke_member, message_history):
    deps = invocation.deps
    member = deps.work.member_id or deps.work.request_event_id
    deps.outputs.begin_member(member)
    await invoke_member(
        invocation.agent,
        None if message_history is not None else invocation.prompt(),
        deps=deps,
        member_id=member,
        message_history=message_history,
    )
    if deps.validation_event_id is None:
        raise AgentResultMissing("model ended without the explanation")
    return deps


def _complete(context: AttemptContext[AdmittedExplanation], deps) -> None:
    if deps is None or deps.validation_event_id is None or deps.draft is None:
        raise RuntimeError("explanation Attempt completed without its explanation")
    events = read_evidence(context.journey)
    bound = bound_fact(events, context.attempt_id)
    if isinstance(bound, ReportExplanationMemberBound):
        terminal = context.activity.record(
            derive_member_admitted(
                bound, deps.draft, validation_event_id=deps.validation_event_id
            )
        )
        context.activity.succeeded(terminal)
        return
    (started,) = [
        event
        for event in events
        if event["event_type"] == "AttemptStarted"
        and event["payload"]["attempt_id"] == context.attempt_id
    ]
    explanation = derive_explanation_by_request(
        bound,
        admitted_members(events),
        deps.draft,
        journey_id=str(context.journey.journey_id),
        supersedes_explanation_id=deps.work.request.supersedes_explanation_id,
        # The instructions this Attempt recorded at its start.
        instruction_identity=started["payload"]["instruction_identity"],
        validation_event_id=deps.validation_event_id,
    )
    fact = ReportExplanationPublished(
        attempt_id=context.attempt_id,
        validation_event_id=deps.validation_event_id,
        explanation=explanation,
    )
    terminal = context.activity.record_with_artifact(
        fact,
        explanation.canonical_bytes(),
        media_type=EXPLANATION_MEDIA_TYPE,
    )
    context.activity.succeeded(terminal)


_BOUND_TYPES = frozenset(
    {
        "ReportExplanationMemberBound",
        "ReportExplanationCombinationBound",
    }
)


def _fail(context: AttemptContext[AdmittedExplanation], invocation, error) -> bool:
    """Retain the explanation's failure; the report is untouched."""
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
            in {"ModelResponseRecorded", "AttemptStarted", *_BOUND_TYPES}
        )
    )
    failed = context.activity.record(
        ReportExplanationFailed(
            attempt_id=context.attempt_id,
            code="model_contact_failed" if model_failed else "agent_run_failed",
            source_event_id=source_event_id,
            reason=None if model_failed else cause_identity(error),
        )
    )
    context.activity.failed(failed)
    return True


def _attempt_id(selected) -> str:
    return next(
        event["payload"]["attempt_id"]
        for event in selected
        if event["event_type"] == "AttemptStarted"
    )


def _attempt_binding(timeline, selected) -> AttemptBinding:
    bound = bound_fact(selected, _attempt_id(selected))
    return AttemptBinding(
        _retained_registration(timeline).input_set_id, bound.runtime_selections
    )


def _attempt_terminal(snapshot, selected) -> AttemptTerminalReading:
    attempt_id = _attempt_id(selected)
    outcomes = {
        **_SHARED_TERMINALS,
        "ReportExplanationMemberAdmitted": "succeeded",
        "ReportExplanationPublished": "succeeded",
        "ReportExplanationFailed": "failed",
    }
    terminals = [
        event
        for event in selected
        if event["payload"].get("attempt_id") == attempt_id
        and event["event_type"] in outcomes
    ]
    if len(terminals) != 1:
        raise ValueError("explanation does not bind this Attempt's exact terminal")
    terminal = terminals[0]
    return AttemptTerminalReading(
        outcomes[terminal["event_type"]], terminal["event_id"]
    )


def _read_result(journey: Journey) -> ReportExplanationView:
    return _explanations(journey)


def explanation_assignment(
    *,
    model_factory: NebiusModelEndpointFactory | None = None,
    member_id: str | None = None,
) -> AttemptAssignment[AdmittedExplanation, BoundInvocation]:
    """The explanation Agent with its model bound: one request's member, or
    (``member_id`` None) the combination. A re-entry reads its step from the
    stopped Attempt's own binding."""
    from bio_neb.agentic_journey.role_instructions import stage_text

    @contextmanager
    def open_explanation(context: AttemptContext[AdmittedExplanation], observe):
        work = context.admitted.work
        bound_value = _bound(context, work)
        bound = context.activity.record(bound_value)
        output_type = draft_binding(bound_value).output_type
        member = work.member_id is not None
        outputs = context.output_session(
            tool_name=OUTPUT_TOOL if member else COMBINATION_OUTPUT_TOOL,
            output_type=output_type,
            arguments_refusal=lambda arguments: (
                refused.faults
                if (refused := _refusal(bound_value, arguments)) is not None
                else None
            ),
        )
        agent, deps = assemble_agent(
            observe(structured_output_session=outputs),
            assignment=ROLE,
            instructions=lambda: stage_text(context.instructions.text, work.stage),
            tools={},
            stage=lambda _deps: work.stage,
            bound_values=lambda _deps, _name: SimpleNamespace(constants={}, enums={}),
            admission=ActionAdmissionSpec(validator=lambda _deps, _action: ()),
            output=MEMBER_OUTPUT if member else COMBINATION_OUTPUT,
            policy=context.effective_policy,
            model_policy=context.model,
            role_text=context.instructions,
            journey=context.journey,
            attempt_id=context.attempt_id,
            parent_activity=context.activity,
            failure_source=context.failure_source,
            dependencies=lambda _admission, observed: ExplanationDependencies(
                journey=context.journey,
                attempt_id=context.attempt_id,
                work=work,
                outputs=outputs,
                observed_tool_session=observed,
                bound=bound_value,
            ),
            search_scope_event_id=bound["event_id"],
        )
        if work.previous_attempt is not None:
            work = replace(
                work,
                previous_attempt={
                    **work.previous_attempt,
                    "current_checks": current_checks(
                        bound_value, work.previous_attempt["draft"]
                    ),
                },
            )
        # The stage's job as its model reads it.
        prompt = _prompt(
            work, context.instructions.message(role(ROLE).jobs[work.stage])
        )
        yield BoundInvocation(agent, deps, lambda: prompt)

    return AttemptAssignment(
        agent_role=ROLE,
        activity_stage="report_explanation_attempt",
        admit=lambda journey, reentry: _admit(journey, reentry, member_id),
        validate_continuation=validate_continuation,
        open=open_explanation,
        drive=_drive,
        complete=_complete,
        fail=_fail,
        read_result=_read_result,
        model_factory=model_factory or create_nebius_model_endpoint,
        reopen=AttemptReopen(
            admission=ActionAdmissionSpec(validator=lambda _deps, _action: ()),
            validate_facts=validate_explanation_attempt_facts,
            structured_output=True,
            domain_bound=DomainBoundReopen(
                binding=_attempt_binding, terminal=_attempt_terminal
            ),
        ),
    )


def _reentry(answer: Mapping[str, str] | None) -> Reentry:
    return Reentry(
        **{
            "availability_review_response_event_id": None,
            "effort_review_response_event_id": None,
            **(answer or {}),
        }
    )


def request_run_default_explanation(journey: Journey, report_digest: str) -> bool:
    """Every published report is explained by default (#460 Q5 b, ADR 0038):
    record the Run's own request when no request names this report and none is
    open; True when it wrote. It reads the owner's shared explanation view."""
    from bio_neb.reporting.report_commands import (
        RUN_DEFAULT_REQUESTER,
        RequestReportExplanationCommand,
        request_report_explanation,
    )

    view = _explanations(journey)
    current = view.current
    if any(item.request.report_digest == report_digest for item in view.requests) or (
        current is not None
        and current.status in {"requested", "review_pending", "answered"}
    ):
        return False
    request_report_explanation(
        RequestReportExplanationCommand(
            journey_file=str(journey.path),
            report_digest=report_digest,
            requested_by=RUN_DEFAULT_REQUESTER,
        ),
        journey=journey,
    )
    return True


def advance_report_explanation(
    journey: Journey,
    *,
    environ: Mapping[str, str],
    model_factory: NebiusModelEndpointFactory | None = None,
) -> bool:
    """Take one step of the Run's open explanation request; True when it wrote.

    Every member waiting for its Attempt, or whose Review answer is unused, runs
    in one batch, at most the Run's ``planner_members`` at once; once every
    member is settled, the combination runs, or re-enters its answered Review. A
    one-Attempt request of an earlier Run starts no work.
    """
    from bio_neb.agentic_journey.attempt_runner import AttemptUnit, run_attempts
    from bio_neb.investigation.observation_defaults import selected_concurrency

    current = _explanations(journey).current
    if (
        current is None
        or current.combination is None
        or current.status not in {"requested", "answered"}
    ):
        return False
    units = tuple(
        AttemptUnit(
            explanation_assignment(
                model_factory=model_factory, member_id=member.member_id
            ),
            reentry=_reentry(member.answer),
        )
        for member in current.members
        if member.status in {"waiting", "answered"}
    )
    if units:
        run_attempts(
            units,
            journey,
            environ=environ,
            concurrency=selected_concurrency(
                _retained_registration(read_evidence(journey)).runtime_selections
            ).planner_members,
        )
        return True
    combination = current.combination
    if combination.status == "answered" or (
        combination.status == "waiting"
        and all(member.section is not None for member in current.members)
    ):
        explanation_assignment(model_factory=model_factory).run(
            journey, environ=environ, **(combination.answer or {})
        )
        return True
    return False


# This domain's Tool rows (ADR 0040): the shared researcher question; its one
# handler is the Agent assembly's, as the Journey search's is.
TOOL_ROWS = (
    researcher_question_row(
        "report_explainer", ("report_explanation", "report_combination")
    ),
)


__all__ = [
    "TOOL_ROWS",
    "COMBINATION_OUTPUT",
    "COMBINATION_STAGE",
    "MEMBER_OUTPUT",
    "MEMBER_STAGE",
    "OUTPUT_TOOL",
    "advance_report_explanation",
    "explanation_assignment",
    "request_run_default_explanation",
    "validate_explanation_attempt_facts",
]
