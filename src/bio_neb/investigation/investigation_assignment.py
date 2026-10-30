"""Investigation planning as an assignment of the shared Attempt lifecycle.

Investigation owns member, source and proposal meaning and its collection
closure. ``attempt_runner`` owns policy, contact, effort, availability and
cancellation. Each collection member is planned in its own Attempt, with its own
request allowance; the coordinator runs members together under the Run's bound
and closes the proposal once every member settles (#105).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
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
    DomainBoundReopen,
    EvidenceSnapshot,
    Reentry,
    RuntimeBinding,
)
from bio_neb.agentic_journey.agent_work_review import (
    AgentFailedTurns,
    pending_agent_work,
)
from bio_neb.agentic_journey.attempt_runner import (
    AttemptConfiguration,
    resolve_attempt_configuration,
)
from bio_neb.agentic_journey.continuation import pending_continuation
from bio_neb.agentic_journey.continuation_policy import (
    OUTPUT_KEY,
    AgentInvocationOverrides,
    failed_turns_spent,
    validate_continuation,
)
from bio_neb.agentic_journey.effort import (
    pending_effort_work,
)
from bio_neb.agentic_journey.failure_cause import cause_identity
from bio_neb.agentic_journey.model_availability import (
    model_availability_direction,
)
from bio_neb.agentic_journey.model_observation import JourneyObservedModel
from bio_neb.agentic_journey.runtime_config import RuntimeSelections
from bio_neb.agentic_journey.structured_output import StructuredOutputSession
from bio_neb.application.input_journey import RunInputsRegistered, validate_fact
from bio_neb.application.investigation_read import current_investigation
from bio_neb.failure import Invariant
from bio_neb.investigation.investigation import (
    AssemblySearchPage,
    InvestigationAttemptBound,
    InvestigationAttemptCompleted,
    InvestigationAttemptFailed,
    InvestigationCollectionRevision,
    InvestigationInput,
    InvestigationPlanDraft,
    InvestigationProjection,
    InvestigationProposal,
    InvestigationRequestItem,
    InvestigationReviewRequested,
    UserAssertion,
    asking_member_by_attempt,
    investigation_output_member_id,
    project_snapshot_investigation,
    target_reads,
    validate_investigation_fact,
    without_unfinished_repair,
)
from bio_neb.investigation.investigation_agent import (
    InvestigationPlannerDependencies,
    ask_about_finding_issue,
    ask_about_retrieved_target,
    assemble_investigation_planner,
)
from bio_neb.investigation.investigation_commands import (
    InvestigationApplicationRefused,
)
from bio_neb.investigation.investigation_request import (
    effective_gene_change,
    effective_location,
    researcher_answers,
    typed_answers,
    unread_target_words,
)
from bio_neb.investigation.item_history import item_history
from bio_neb.investigation.member_context import (
    InvestigationMemberContextRecorded,
    latest_source_results,
    rendered_text,
)
from bio_neb.investigation.run_reference import carried_operation
from bio_neb.journey import Journey
from bio_neb.paper_reading.investigation import target_paper
from bio_neb.paper_reading.reader import Paper
from bio_neb.paper_reading.tools import READING_TOOLS
from bio_neb.providers.eva_investigation import PublicInvestigationSources
from bio_neb.providers.ncbi_investigation import InvestigationSourceAdapter
from bio_neb.recording import evidence_view, read_evidence

ROLE = "investigation_planner"
OUTPUT_TOOL = "compile_proposal"


def _answered_questions(projection) -> tuple[tuple[Any, str | None], ...]:
    """Each researcher answer with the member's own question it answers."""
    asked = {
        review.request_event_id: review.question.question
        for review in projection.review_requests
        if review.question is not None
    }
    return tuple(
        (receipt, asked.get(receipt.request_event_id))
        for receipt in researcher_answers(projection.review_receipts)
    )


@dataclass(frozen=True)
class InvestigationWork:
    """What one Attempt plans, and what it inherits from the Attempt it continues."""

    runtime_selections: RuntimeSelections
    investigation_input: InvestigationInput
    assertion: UserAssertion
    revision_number: int
    supersedes_proposal_id: str | None
    revision: InvestigationCollectionRevision | None = None
    previous_proposal: InvestigationProposal | None = None
    prior_lineages: tuple[Any, ...] = ()
    researcher_instruction: str | None = None
    # The Review response the instruction is read from.
    researcher_instruction_event_id: str | None = None
    continued_attempt_id: str | None = None
    eligible_member_ids: tuple[str, ...] | None = None
    # Each researcher answer beside the question it answers, as (receipt, question).
    question_answers: tuple[tuple[Any, str | None], ...] = ()
    rejections: dict = field(default_factory=dict)
    issue_approvals: dict = field(default_factory=dict)
    member_id: str | None = None

    @property
    def requests(self) -> tuple[InvestigationRequestItem, ...]:
        assertion = self.assertion
        return assertion.requests or (
            InvestigationRequestItem(
                request_text=assertion.request_text, rsid=assertion.rsid
            ),
        )


@dataclass(frozen=True)
class AdmittedInvestigation:
    """The admitted work and the source adapter bound before the Attempt writes.

    A source adapter that cannot be set up refuses before ``AttemptStarted``, so
    the coordinator's hold leaves no Attempt without its terminal.
    """

    work: InvestigationWork
    source_adapter: InvestigationSourceAdapter


@dataclass
class OpenInvestigation:
    """The bound Attempt's source adapter, observed model and member results."""

    bound_event: dict[str, Any]
    adapter: InvestigationSourceAdapter
    model: JourneyObservedModel
    outputs: StructuredOutputSession
    members: list[InvestigationPlannerDependencies] = field(default_factory=list)
    held_review_ids: list[str] = field(default_factory=list)


def _retained_runtime_binding(journey: Journey) -> RuntimeBinding:
    """The Run's one registration binds Investigation's runtime; a Fork inherits it."""
    registrations = [
        validate_fact(event)
        for event in read_evidence(journey)
        if event.get("event_type") == "RunInputsRegistered"
    ]
    if len(registrations) != 1 or not isinstance(registrations[0], RunInputsRegistered):
        raise InvestigationApplicationRefused("investigation_runtime_binding_invalid")
    bound = registrations[0]
    return RuntimeBinding(bound.runtime_selections, bound.input_set_id)


def set_up_investigation_sources(
    source_adapter_factory: Callable[[], InvestigationSourceAdapter] | None,
) -> InvestigationSourceAdapter:
    """Set up the source adapter; a failure raises before the caller writes."""
    return (source_adapter_factory or PublicInvestigationSources)()


def resolve_investigation_configuration(
    runtime: RuntimeBinding,
    environ,
    *,
    overrides: AgentInvocationOverrides,
) -> AttemptConfiguration:
    """Resolve Investigation's configuration as its Attempt will, raising as it would.

    ``begin_investigation`` writes the Investigation's start before the Attempt
    runs, so it resolves here first to refuse before that write.
    """
    return resolve_attempt_configuration(
        runtime,
        environ,
        agent_role=ROLE,
        overrides=overrides,
    )


def _admit(
    journey: Journey,
    reentry: Reentry,
    source_adapter_factory: Callable[[], InvestigationSourceAdapter] | None,
    member_id: str | None = None,
) -> AttemptAdmission[AdmittedInvestigation]:
    events = read_evidence(journey)
    projection = current_investigation(journey)
    runtime = _retained_runtime_binding(journey)
    if projection.investigation_input is None:
        raise ValueError("Investigation Attempt requires a started Investigation")
    response_id = (
        reentry.availability_review_response_event_id
        or reentry.operational_review_response_event_id
        or reentry.effort_review_response_event_id
        or reentry.work_review_response_event_id
    )
    causal_head = (
        pending_continuation(events, response_id, agent_role=ROLE).causal_head
        if response_id is not None
        else journey.prefix_event_ids[-1]
    )
    availability_frontier = causal_head
    if reentry.work_review_response_event_id is not None:
        request, instruction = pending_agent_work(
            journey, reentry.work_review_response_event_id, agent_role=ROLE
        )
        work = _restarted_work(
            events, projection, request.attempt_id, runtime.runtime_selections
        )
        work = replace(
            work,
            researcher_instruction=instruction,
            researcher_instruction_event_id=None
            if instruction is None
            else reentry.work_review_response_event_id,
        )
    elif (
        reentry.availability_review_response_event_id is not None
        or reentry.operational_review_response_event_id is not None
    ):
        if reentry.availability_review_response_event_id is not None:
            request, receipt = model_availability_direction(
                journey, reentry.availability_review_response_event_id, agent_role=ROLE
            )
            instruction = (
                receipt.response.instruction
                if receipt.response.action == "amend"
                else None
            )
        else:
            from bio_neb.agentic_journey.operational_review import (
                pending_operational_work,
            )

            request, receipt = pending_operational_work(
                journey, reentry.operational_review_response_event_id, agent_role=ROLE
            )
            instruction = (
                receipt.response.value if receipt.response.action == "amend" else None
            )
        work = _scoped_recovery_work(
            events,
            projection,
            request.attempt_id,
            request.member_id,
            runtime.runtime_selections,
        )
        if receipt.response.action == "amend":
            work = replace(
                work,
                researcher_instruction=instruction,
                researcher_instruction_event_id=None
                if instruction is None
                else (
                    reentry.availability_review_response_event_id
                    or reentry.operational_review_response_event_id
                ),
                prior_lineages=tuple(
                    lineage
                    for lineage in work.prior_lineages
                    if lineage.collection_member_id != request.member_id
                    or lineage.accepted_output is not None
                ),
            )
        availability_frontier = (
            request.retry_frontier_event_id
            if receipt.response.action == "retry"
            else causal_head
        )
    elif (
        reentry.effort_review_response_event_id is None
        and reentry.availability_review_response_event_id is None
    ):
        work = _fresh_member_work(
            events, projection, runtime.runtime_selections, member_id
        )
    else:
        if reentry.effort_review_response_event_id is not None:
            request, _, _ = pending_effort_work(
                journey, reentry.effort_review_response_event_id, agent_role=ROLE
            )
            previous_id = request.attempt_id
        work = _continued_work(
            events, projection, previous_id, runtime.runtime_selections
        )
    if (work.revision is None) != (work.previous_proposal is None):
        raise ValueError("Investigation revision changed its previous proposal")
    return AttemptAdmission(
        AdmittedInvestigation(
            work, set_up_investigation_sources(source_adapter_factory)
        ),
        runtime,
        causal_head,
        availability_frontier,
        work.member_id,
    )


def collection_revision(projection: InvestigationProjection):
    """The revision members are planned in now, and its requests.

    A revision that supersedes the latest proposal is planned afresh; otherwise
    the collection's first revision.
    """
    if (
        projection.revisions
        and projection.proposals
        and projection.revisions[-1].supersedes_proposal_id
        == projection.proposals[-1].proposal_id
    ):
        return projection.revisions[-1], projection.assertions[-1]
    return None, projection.assertions[0]


def dispatchable_members(projection: InvestigationProjection) -> tuple[str, ...]:
    """The members the coordinator may plan now, each in its own Attempt (#105)."""
    if not projection.attempts and projection.assertions:
        # A started Investigation: every original member gets its first Attempt.
        assertion = projection.assertions[0]
        requests = assertion.requests or (
            InvestigationRequestItem(
                request_text=assertion.request_text, rsid=assertion.rsid
            ),
        )
        # A member recorded not checked asks no model.
        settled = {view.collection_member_id for view in projection.not_checked}
        return tuple(
            member
            for ordinal, request in enumerate(requests)
            if (member := investigation_output_member_id(request, ordinal))
            not in settled
        )
    if projection.status != "investigation_work_ready":
        return ()
    if projection.eligible_member_ids:
        return projection.eligible_member_ids
    revision, assertion = collection_revision(projection)
    if revision is None or any(
        attempt.revision_id == revision.revision_id for attempt in projection.attempts
    ):
        return ()
    requests = assertion.requests or (
        InvestigationRequestItem(
            request_text=assertion.request_text, rsid=assertion.rsid
        ),
    )
    return tuple(
        investigation_output_member_id(request, ordinal)
        for ordinal, request in enumerate(requests)
    )


def _fresh_member_work(
    events,
    projection: InvestigationProjection,
    runtime_selections: RuntimeSelections,
    member_id: str | None,
) -> InvestigationWork:
    """One member's work: its own continuation, or its first Attempt (#105)."""
    assert projection.investigation_input is not None
    if member_id is None or member_id not in dispatchable_members(projection):
        raise ValueError("Investigation Attempt is not eligible")
    revision, assertion = collection_revision(projection)
    revision_id = None if revision is None else revision.revision_id
    previous = next(
        (
            attempt
            for attempt in reversed(projection.attempts)
            if attempt.revision_id == revision_id
            and attempt.collection_member_id == member_id
        ),
        None,
    )
    if previous is not None and previous.outcome != "failed":
        # An interrupted Attempt keeps its completed work, not its unfinished repair.
        # A failed one held the Run; after `resume` its member starts afresh (R2).
        continued = (
            _restarted_work if previous.outcome == "cancelled" else _continued_work
        )
        work = continued(events, projection, previous.attempt_id, runtime_selections)
        asker = asking_member_by_attempt(events).get(previous.attempt_id)
        if asker is not None:
            work = replace(
                work,
                prior_lineages=without_unfinished_repair(work.prior_lineages, asker),
            )
        return replace(
            work,
            continued_attempt_id=previous.attempt_id,
            eligible_member_ids=(member_id,),
            question_answers=_answered_questions(projection),
        )
    return InvestigationWork(
        runtime_selections=runtime_selections,
        investigation_input=projection.investigation_input,
        assertion=assertion,
        revision_number=1 if revision is None else revision.revision_number,
        supersedes_proposal_id=None
        if revision is None
        else revision.supersedes_proposal_id,
        revision=revision,
        previous_proposal=None if revision is None else projection.proposals[-1],
        eligible_member_ids=(member_id,),
        question_answers=_answered_questions(projection),
        member_id=member_id,
    )


def _restarted_work(
    events: list[dict[str, Any]],
    projection: InvestigationProjection,
    stopped_id: str,
    runtime_selections: RuntimeSelections,
) -> InvestigationWork:
    """Reset unfinished output repair while retaining completed work and sources."""
    work = _continued_work(events, projection, stopped_id, runtime_selections)
    return replace(
        work,
        prior_lineages=tuple(
            lineage
            for lineage in work.prior_lineages
            if lineage.accepted_output is not None
        ),
    )


def _scoped_recovery_work(events, projection, origin_id, member_id, runtime_selections):
    """Resume one held member using the latest compatible collection state."""
    from bio_neb.investigation.investigation_questions import latest_collection_frontier

    latest = latest_collection_frontier(events, origin_id)
    work = _continued_work(events, projection, latest["attempt_id"], runtime_selections)
    if member_id not in work.eligible_member_ids:
        raise ValueError("operational member re-entry is no longer eligible")
    return replace(
        work,
        continued_attempt_id=latest["attempt_id"],
        eligible_member_ids=(member_id,),
    )


def _continued_work(
    events: list[dict[str, Any]],
    projection: InvestigationProjection,
    previous_id: str,
    runtime_selections: RuntimeSelections,
) -> InvestigationWork:
    """Rebind the stopped Attempt's work with its accepted members and sources."""
    assert projection.investigation_input is not None
    bound = next(
        validate_investigation_fact(e)
        for e in events
        if e["event_type"] == "InvestigationAttemptBound"
        and e["payload"]["attempt_id"] == previous_id
    )
    from bio_neb.investigation.investigation_questions import (
        eligible_members,
        issue_approvals,
        request_rejections,
        without_amended_findings,
    )

    lineages = next(
        a.structured_output_lineages
        for a in projection.attempts
        if a.attempt_id == previous_id
    )

    eligible = eligible_members(
        bound.work_requests,
        bound.assertion_id,
        bound.revision_number,
        lineages,
        projection.review_requests,
        projection.review_receipts,
    )
    if bound.collection_member_id is not None:
        # A member's Attempt continues only that member's work.
        eligible = tuple(m for m in eligible if m == bound.collection_member_id)
    return InvestigationWork(
        member_id=bound.collection_member_id,
        eligible_member_ids=eligible,
        rejections=request_rejections(
            projection.review_requests,
            projection.review_receipts,
            bound.assertion_id,
            bound.revision_number,
        ),
        issue_approvals=issue_approvals(
            projection.review_requests,
            projection.review_receipts,
            bound.assertion_id,
            bound.revision_number,
        ),
        question_answers=_answered_questions(projection),
        runtime_selections=runtime_selections,
        investigation_input=projection.investigation_input,
        assertion=next(
            a for a in projection.assertions if a.assertion_id == bound.assertion_id
        ),
        revision_number=bound.revision_number,
        supersedes_proposal_id=bound.supersedes_proposal_id,
        revision=next(
            (r for r in projection.revisions if r.revision_id == bound.revision_id),
            None,
        ),
        previous_proposal=next(
            (
                p
                for p in projection.proposals
                if p.proposal_id == bound.supersedes_proposal_id
            ),
            None,
        ),
        continued_attempt_id=bound.continued_attempt_id,
        prior_lineages=without_amended_findings(
            lineages, projection.review_requests, projection.review_receipts
        ),
    )


async def _drive(
    context: AttemptContext[AdmittedInvestigation],
    investigation: OpenInvestigation,
    invoke_member,
    message_history,
) -> OpenInvestigation:
    work = context.admitted.work
    first_member = True
    for ordinal, request in enumerate(work.requests):
        member_id = investigation_output_member_id(request, ordinal)
        if any(
            lineage.collection_member_id == member_id
            and lineage.accepted_output is not None
            for lineage in work.prior_lineages
        ) or (
            work.eligible_member_ids is not None
            and member_id not in work.eligible_member_ids
        ):
            continue
        # Each member has its own Attempt and allowance: no member starts on
        # another's spent requests (Errors 18935528).
        investigation.outputs.begin_member(member_id)
        agent, dependencies = assemble_investigation_planner(
            investigation.model,
            instructions=context.instructions,
            policy=context.effective_policy,
            model_policy=context.model,
            journey=context.journey,
            attempt_id=context.attempt_id,
            parent_activity=context.activity,
            failure_source=context.failure_source,
            dependencies=lambda observed, request=request, member_id=member_id: (
                InvestigationPlannerDependencies(
                    journey=context.journey,
                    parent_activity=context.activity,
                    attempt_id=context.attempt_id,
                    investigation_input=work.investigation_input,
                    assertion=work.assertion,
                    request=request,
                    collection_member_id=member_id,
                    revision_number=work.revision_number,
                    supersedes_proposal_id=work.supersedes_proposal_id,
                    adapter=investigation.adapter,
                    failure_source=context.failure_source,
                    observed_tool_session=observed,
                    structured_output_session=investigation.outputs,
                )
            ),
        )
        # The item's one history: every fact its admission reads, from its earlier
        # Attempts and revisions by the one rule `item_history` states.
        timeline = read_evidence(context.journey)
        history = item_history(
            timeline,
            attempt_id=context.attempt_id,
            member_id=member_id,
            revision_number=work.revision_number,
            read_artifact=lambda event: context.journey.read_artifact(
                event["artifact_references"][0]
            ),
        )
        dependencies.source_events.extend(history.source_results)
        dependencies.assembly_searches.extend(
            AssemblySearchPage.model_validate_json(
                json.dumps(event["payload"]["result"])
            )
            for event in history.assembly_searches
        )
        # Planning's GETs of the Run's assembly the start carries: read in place of
        # a contact, and the Run assembly's sequence names searchable now.
        started = next(
            e["payload"] for e in timeline if e["event_type"] == "InvestigationStarted"
        )
        dependencies.run_taxid = started.get("run_taxid")
        # The paper that proposed this member's target, read with the reading Tools
        # beside what planning's reading kept for it.
        paper = target_paper(timeline, request)
        if paper is not None:
            dependencies.paper = Paper.stored(context.journey, paper.paper_event_id)
        run_accession = work.investigation_input.assembly_accession
        for cited_id in started.get("run_reference_sources", ()):
            cited = next(e for e in timeline if e["event_id"] == cited_id)
            (reference,) = cited["artifact_references"]
            body = context.journey.read_artifact(reference)
            operation = carried_operation(cited["payload"], run_accession)
            dependencies.carried_sources[(operation, run_accession)] = (
                cited,
                body,
                reference["media_type"],
            )
            if operation == "ncbi_sequence_names":
                dependencies.sequence_reports[run_accession] = (cited_id, body)
        dependencies.sequence_reports.update(history.sequence_reports)
        dependencies.decisions.extend(history.decisions)
        dependencies.replaced_result_ids = history.replaced_result_ids
        investigation.members.append(dependencies)
        answered = [
            (receipt, question)
            for receipt, question in work.question_answers
            if receipt.response.collection_member_id == member_id
            and receipt.response.revision_number == work.revision_number
        ]
        dependencies.raw_answers = tuple(
            {
                "question": question,
                **receipt.response.model_dump(mode="json"),
                "response_event_id": receipt.response_event_id,
            }
            for receipt, question in answered
        )
        # The member's words the model has read, in this revision and its earlier
        # Attempts (R1 b), and where an untyped item's own words start.
        dependencies.started_event_id = next(
            e["event_id"] for e in timeline if e["event_type"] == "InvestigationStarted"
        )
        dependencies.target_reads = list(
            target_reads(timeline, member_id, work.revision_number)
        )
        # This member's gene-change placements and picks for its current
        # target.
        dependencies.gene_change_placements = list(history.gene_change_placements)
        dependencies.gene_change_picks = [
            event["payload"] for event in history.gene_change_picks
        ]
        dependencies.researcher_answers = typed_answers(
            request,
            dependencies.raw_answers,
            dependencies.target_reads,
            dependencies.gene_change_picks,
        )
        answers = dependencies.researcher_answers
        prompt = None
        if message_history is None or not first_member:
            # The member's context, recorded once as pointers and sent
            # rendered.
            location = effective_location(request, answers)
            # A region is located by locate_variant like a position.
            positioned = location.position is not None or location.region is not None
            evidence = evidence_view(context.journey)
            fact = InvestigationMemberContextRecorded(
                attempt_id=context.attempt_id,
                collection_member_id=member_id,
                started_event_id=next(
                    e["event_id"]
                    for e in evidence
                    if e["event_type"] == "InvestigationStarted"
                ),
                bound_event_id=next(
                    e["event_id"]
                    for e in evidence
                    if e["event_type"] == "InvestigationAttemptBound"
                    and e["payload"]["attempt_id"] == context.attempt_id
                ),
                source_result_event_ids=tuple(
                    e["event_id"]
                    for e in latest_source_results(evidence, member_id).values()
                ),
                answer_event_ids=tuple(
                    receipt.response_event_id for receipt, _ in answered
                ),
                instruction_event_id=work.researcher_instruction_event_id,
                # The replaced target's last lookup, marked replaced.
                replaced_result_event_id=None
                if history.replaced_result is None
                else history.replaced_result["event_id"],
                replaced_by_event_id=None
                if history.replaced_result is None
                else history.target_event_id,
                paper_answer_event_id=None if paper is None else paper.answer_event_id,
                eligible_source_operations=(
                    # A stated position is placed first; its lookup reads the rsID
                    # too. A region is located by the same Tool. An rsID is looked
                    # up in the source the model chooses.
                    *(
                        ("locate_variant",)
                        if positioned
                        # A gene change is placed, then a candidate picked.
                        else ("place_gene_change", "pick_gene_change_candidate")
                        if effective_gene_change(request, answers) is not None
                        else ("fetch_refsnp", "fetch_eva_variant")
                    ),
                    "fetch_assembly_report",
                    # The member's paper, read where it writes the target.
                    *(() if paper is None else tuple(READING_TOOLS)),
                    "search_journey",
                    "request_investigation_information",
                    # Words the model has yet to type a target from.
                    *(
                        ("record_target_reading",)
                        if unread_target_words(
                            request,
                            dependencies.raw_answers,
                            dependencies.target_reads,
                            started_event_id=dependencies.started_event_id,
                        )
                        else ()
                    ),
                ),
                forbidden_operations=(
                    "vcf_observation",
                    "provider_contact",
                    "storage_contact",
                ),
            )
            try:
                prompt = rendered_text(
                    fact,
                    {e["event_id"]: e for e in evidence},
                    context.instructions,
                    # The owner's one parse of each admitted fact, which its
                    # reopen shares.
                    lambda event: context.journey.derived_once(
                        ("investigation_fact", event["event_id"]),
                        lambda: validate_investigation_fact(event),
                    ),
                )
            except ValueError as error:
                # Pointers this Attempt just read that do not resolve: our bug,
                # held under its own name before any model contact.
                raise Invariant(
                    "member_context_unrenderable",
                    str(error),
                    evidence=fact.model_dump(mode="json", exclude={"event_type"}),
                ) from error
            context.activity.record(fact)

        def completed_work(dependencies=dependencies):
            if dependencies.question_event is not None:
                dependencies.observed_tool_session.close_response("researcher_question")
                return SimpleNamespace(checkpoint="researcher_question")
            return None

        from bio_neb.agentic_journey.model_availability import (
            ModelAvailabilityReviewPending,
            record_model_availability_review_pending,
        )
        from bio_neb.agentic_journey.operational_review import (
            OperationalStop,
            record_operational_stop,
        )

        try:
            await invoke_member(
                agent,
                prompt,
                deps=dependencies,
                member_id=member_id,
                message_history=message_history if first_member else None,
                completed_work=completed_work,
            )
        except AgentFailedTurns:
            # A failed turn at its limit on a retrieved target that still differs
            # from the request is this member's researcher question. A stop on
            # another fault (a text ending, a refused call) keeps the shared
            # Agent-work Review.
            if not failed_turns_spent(
                read_evidence(context.journey),
                attempt_id=context.attempt_id,
                policy=context.effective_policy.continuation,
                key=OUTPUT_KEY,
            ) or not ask_about_retrieved_target(dependencies):
                raise
        except OperationalStop as stop:
            request = record_operational_stop(
                context, stop, agent_role=ROLE, close_attempt=False
            )
            investigation.held_review_ids.append(request["event_id"])
        except ModelAvailabilityReviewPending as pending:
            request_id = record_model_availability_review_pending(
                context.activity,
                pending,
                attempt_id=context.attempt_id,
                agent_role=ROLE,
                retry_frontier_event_id=context.availability_retry_frontier,
                close_attempt=False,
            )
            investigation.held_review_ids.append(request_id)
        first_member = False
        if dependencies.question_event is not None or ask_about_finding_issue(
            dependencies
        ):
            return investigation
    return investigation


def _complete(
    context: AttemptContext[AdmittedInvestigation], investigation: OpenInvestigation
) -> None:
    """Close this member's Attempt; the coordinator closes the collection's proposal.

    A member ends with its accepted output, its own question or issue Review, or
    its held operational Reviews. Once every member settles, the collection's
    proposal is recorded once (``investigation_questions.close_collection``).
    """
    if not investigation.members:
        raise RuntimeError("Investigation Attempt planned no member")
    member = investigation.members[-1]
    if (
        not investigation.held_review_ids
        and member.question_event is None
        and member.validation_event_id is None
    ):
        raise RuntimeError("Investigation Planner completed without typed validation")
    terminal = context.activity.record(
        InvestigationAttemptCompleted(
            attempt_id=context.attempt_id,
            question_request_event_id=None
            if member.question_event is None
            else member.question_event["event_id"],
            held_review_event_ids=tuple(investigation.held_review_ids),
        )
    )
    context.activity.succeeded(terminal)


def _fail(
    context: AttemptContext[AdmittedInvestigation],
    investigation: OpenInvestigation | None,
    error: Exception,
) -> bool:
    """Retain an item-scoped failure; Investigation never fails the Run.

    Refused actions and outputs beyond their allowances and a cut-off completion
    are the lifecycle's reviewable stop, not a failure code of this hook.
    """
    if investigation is None or not investigation.members:
        return False
    journey = context.journey
    attempt_id = context.attempt_id
    failure_source = context.failure_source
    code = (
        "source_contact_failed"
        if failure_source.event_type == "InvestigationSourceContactFailed"
        else "model_contact_failed"
        if failure_source.event_type in {"ModelRequestFailed"}
        else "agent_run_failed"
    )
    if code == "agent_run_failed":
        source_event_id = next(
            (
                event["event_id"]
                for event in reversed(read_evidence(journey))
                if event.get("payload", {}).get("attempt_id") == attempt_id
                and event.get("event_type")
                in {
                    "ObservedToolExecutionFailed",
                    "ModelResponseRecorded",
                    "ToolCallValidated",
                    "InvestigationMemberContextRecorded",
                    "InvestigationAttemptBound",
                }
                and (
                    event["event_type"] != "ToolCallValidated"
                    or event["payload"].get("status") == "invalid"
                )
            ),
            investigation.bound_event["event_id"],
        )
    else:
        source_event_id = failure_source.event_id
    failed = context.activity.record(
        InvestigationAttemptFailed(
            attempt_id=attempt_id,
            code=code,
            source_event_id=source_event_id,
            reason=cause_identity(error) if code == "agent_run_failed" else None,
        )
    )
    context.activity.failed(failed)
    return True


_SHARED_OUTCOMES = {
    "question_pending": "succeeded",
    "work_completed": "succeeded",
    "review_pending": "succeeded",
    "operational_review_pending": "succeeded",
    "failed": "failed",
    "model_availability_review_pending": "failed",
    "effort_review_pending": "stopped",
    "agent_work_review_pending": "stopped",
    "cancelled": "cancelled",
}
_SHARED_TERMINALS = {"AttemptCancelled", "AttemptStopped", "AttemptFailed"}


def _window_attempt_id(selected: list[dict[str, Any]]) -> str:
    return next(
        event["payload"]["attempt_id"]
        for event in selected
        if event["event_type"] == "AttemptStarted"
    )


def _attempt_binding(
    timeline: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> AttemptBinding:
    """Read the runtime the Investigation start bound for this Attempt."""
    attempt_id = _window_attempt_id(selected)
    bound = [
        fact
        for event in selected
        if isinstance(
            fact := validate_investigation_fact(event), InvestigationAttemptBound
        )
        and fact.attempt_id == attempt_id
    ]
    if len(bound) != 1:
        raise ValueError("Investigation Attempt lacks its exact binding")
    registrations = [
        validate_fact(event)
        for event in timeline
        if event["event_type"] == "RunInputsRegistered"
    ]
    if len(registrations) != 1:
        raise ValueError("Investigation Attempt lacks its exact input set")
    return AttemptBinding(registrations[0].input_set_id, bound[0].runtime_selections)


def _attempt_terminal(
    snapshot: EvidenceSnapshot, selected: list[dict[str, Any]]
) -> AttemptTerminalReading:
    """Read the terminal the Investigation projector checked for this Attempt.

    Every Attempt of one rebuild reads the snapshot's one projection (ADR 0032).
    """
    attempt_id = _window_attempt_id(selected)
    views = [
        view
        for view in snapshot.derived(project_snapshot_investigation).attempts
        if view.attempt_id == attempt_id
    ]
    terminals = [
        event
        for event in selected
        if event["payload"].get("attempt_id") == attempt_id
        and (
            event["event_type"] in _SHARED_TERMINALS
            or event["event_type"]
            in {"InvestigationAttemptCompleted", "InvestigationAttemptFailed"}
            or (
                event["event_type"] == "InvestigationReviewRequested"
                and event["payload"].get("question") is None
            )
        )
    ]
    if len(views) != 1 or len(terminals) != 1:
        raise ValueError("Investigation does not bind this Attempt's exact terminal")
    # A question the domain asked itself, past the intent check's allowance or for
    # an issue finding, is no Tool call's fact; the projector checked it.
    domain_questions = tuple(
        event["event_id"]
        for event in selected
        if isinstance(
            fact := validate_investigation_fact(event), InvestigationReviewRequested
        )
        and fact.attempt_id == attempt_id
        and fact.question is not None
        and (
            fact.question.refused_output_event_id is not None
            or fact.question.issue is not None
        )
    )
    return AttemptTerminalReading(
        _SHARED_OUTCOMES[views[0].outcome],
        terminals[0]["event_id"],
        domain_questions,
    )


def _read_result(journey: Journey) -> InvestigationProjection:
    return current_investigation(journey)


def _settle(journey: Journey) -> None:
    """A member's Attempt that settles the last member closes the proposal."""
    from bio_neb.investigation.investigation_questions import close_collection

    close_collection(journey, _read_result(journey))


def investigation_assignment(
    *,
    model_factory: NebiusModelEndpointFactory | None = None,
    source_adapter_factory: Callable[[], InvestigationSourceAdapter] | None = None,
    member_id: str | None = None,
) -> AttemptAssignment[AdmittedInvestigation, OpenInvestigation]:
    """Investigation planning with its adapters bound; ``member_id`` names a fresh
    Attempt's one member (a review answer names its own)."""

    @contextmanager
    def open_investigation(context, observe):
        work = context.admitted.work
        continuation = context.effective_policy.continuation
        bound_event = context.activity.record(
            InvestigationAttemptBound(
                attempt_id=context.attempt_id,
                continued_attempt_id=work.continued_attempt_id,
                investigation_input_id=work.investigation_input.input_id,
                assertion_id=work.assertion.assertion_id,
                revision_number=work.revision_number,
                supersedes_proposal_id=work.supersedes_proposal_id,
                revision_id=None
                if work.revision is None
                else work.revision.revision_id,
                work_requests=work.requests,
                runtime_selections=work.runtime_selections,
                continuation_policy=continuation,
                continuation_policy_id=continuation.policy_id,
                collection_member_id=work.member_id,
            )
        )
        outputs = context.output_session(
            tool_name=OUTPUT_TOOL,
            output_type=InvestigationPlanDraft,
        )
        outputs.restore(work.prior_lineages)
        yield OpenInvestigation(
            bound_event,
            context.admitted.source_adapter,
            observe(structured_output_session=outputs),
            outputs,
        )

    return AttemptAssignment(
        agent_role=ROLE,
        activity_stage="investigation_planning_attempt",
        admit=lambda journey, reentry: _admit(
            journey, reentry, source_adapter_factory, member_id
        ),
        validate_continuation=validate_continuation,
        open=open_investigation,
        drive=_drive,
        complete=_complete,
        fail=_fail,
        read_result=_read_result,
        model_factory=model_factory or create_nebius_model_endpoint,
        reopen=AttemptReopen(
            structured_output=True,
            domain_bound=DomainBoundReopen(
                binding=_attempt_binding, terminal=_attempt_terminal
            ),
        ),
        settle=_settle,
    )


__all__ = [
    "InvestigationWork",
    "collection_revision",
    "dispatchable_members",
    "investigation_assignment",
    "resolve_investigation_configuration",
    "set_up_investigation_sources",
]
