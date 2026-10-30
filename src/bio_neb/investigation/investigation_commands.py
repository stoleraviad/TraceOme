"""Public application operations for one source-resolved Investigation proposal."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationError,
)

from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationOverrides,
    InvestigationReviewPolicy,
)
from bio_neb.agentic_journey.refusal_disposition import RefusalDisposition
from bio_neb.application.input_journey import (
    RunInputsRegistered,
    validate_fact,
)
from bio_neb.application.investigation_read import current_investigation
from bio_neb.investigation.investigation import (
    ApproveInvestigationResponse,
    InvestigationAssertionRecorded,
    InvestigationAuthority,
    InvestigationAuthorityGranted,
    InvestigationCollectionRevisionRecorded,
    InvestigationInput,
    InvestigationProjection,
    InvestigationProposalCarriedForward,
    InvestigationReviewRequested,
    InvestigationReviewResponded,
    InvestigationReviewResponse,
    InvestigationRevisionRefused,
    InvestigationStarted,
    InvestigationStopped,
    InvestigationVcfIdentity,
    ReviseInvestigationResponse,
    StopInvestigationResponse,
    UserAssertion,
    compile_contact_free_investigation_revision_proposal,
    compile_investigation_collection_revision,
)
from bio_neb.investigation.investigation_request import (
    InvestigationRequest,
    InvestigationRequestRefused,
    combined_request,
    current_investigation_request,
    read_investigation_request,
)
from bio_neb.journey import Journey, JourneyFile
from bio_neb.recording import current_processing_pass, read_evidence

if TYPE_CHECKING:
    from bio_neb.agentic_journey.adapters import NebiusModelEndpointFactory
    from bio_neb.investigation.scientific_continuation import (
        ScientificContinuationPacket,
    )
    from bio_neb.providers.ncbi_investigation import (
        InvestigationSourceAdapter,
    )


class _Command(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BeginInvestigationCommand(_Command):
    invocation_overrides: AgentInvocationOverrides = Field(
        default_factory=AgentInvocationOverrides
    )
    journey_file: JourneyFile
    # Omitted, the continuation is compiled from this Run's verified Execution.
    scientific_continuation_file: StrictStr | None = None
    # Omitted, the Run's captured request is used; supplied, the files together
    # must match it.
    request_files: tuple[StrictStr, ...] = ()


class ContinueInvestigationCommand(_Command):
    invocation_overrides: AgentInvocationOverrides = Field(
        default_factory=AgentInvocationOverrides
    )
    journey_file: JourneyFile


class RespondInvestigationReviewCommand(_Command):
    journey_file: JourneyFile
    request_event_id: StrictStr
    response: InvestigationReviewResponse


class InvestigationApplicationRefused(ValueError):
    def __init__(
        self,
        code: str,
        *fields: str,
        request_event_id: str | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.fields = tuple(fields)
        self.request_event_id = request_event_id
        self.disposition = RefusalDisposition(kind="caller_correction_and_reissue")


def _absolute_file(value: str, name: str) -> Path:
    if not os.path.isabs(value) or os.path.normpath(value) != value:
        raise InvestigationApplicationRefused("investigation_input_path_invalid", name)
    return Path(value)


def _read_continuation(path: str) -> ScientificContinuationPacket:
    from bio_neb.investigation.scientific_continuation import (
        ScientificContinuationPacket,
    )

    try:
        document = _absolute_file(path, "scientific_continuation_file").read_text(
            encoding="utf-8"
        )
        value = json.loads(document)
        if not isinstance(value, dict):
            raise ValueError("continuation must be an object")
        if value.get("schema_version") != "bio-neb.scientific-continuation/1":
            raise ValueError("continuation schema is not supported")
        return ScientificContinuationPacket.model_validate_json(document, strict=True)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ) as error:
        raise InvestigationApplicationRefused(
            "investigation_continuation_invalid", "scientific_continuation_file"
        ) from error


def _registration(journey: Journey) -> RunInputsRegistered:
    matches = [
        validate_fact(event)
        for event in read_evidence(journey)
        if event.get("event_type") == "RunInputsRegistered"
    ]
    if len(matches) != 1 or not isinstance(matches[0], RunInputsRegistered):
        raise InvestigationApplicationRefused("investigation_registration_invalid")
    return matches[0]


def _validated_continuation_timeline(
    continuation: ScientificContinuationPacket, journey: Journey
) -> tuple[dict[str, Any], ...]:
    """Recompile the supplied packet from this Run's verified Execution."""

    timeline = tuple(read_evidence(journey))
    if _same_run_continuation(journey) != continuation:
        raise InvestigationApplicationRefused(
            "investigation_continuation_stale", "scientific_continuation_file"
        )
    return timeline


def _same_run_continuation(journey: Journey) -> ScientificContinuationPacket:
    """Compile the deterministic handoff from this Run's verified Execution."""

    from bio_neb.investigation.scientific_continuation import (
        compile_scientific_continuation,
    )
    from bio_neb.journey_path import JourneyPath
    from bio_neb.parabricks.result import read_completed_parabricks_result

    try:
        outcome = read_completed_parabricks_result(str(journey.path))
        return compile_scientific_continuation(outcome, JourneyPath.open(journey.path))
    except (TypeError, ValueError) as error:
        raise InvestigationApplicationRefused(
            "investigation_continuation_stale", "scientific_continuation_file"
        ) from error


def resolve_investigation_request(
    journey: Journey, request_files: tuple[str, ...]
) -> InvestigationRequest:
    """One request per Run: the captured request, or the files supplied in its
    absence, combined in order with each target's asserter (#367). A Fork child
    keeps its parent's request and takes no new one (#367 readiness Q4)."""

    captured = current_investigation_request(journey)
    if not request_files:
        if captured is None:
            raise InvestigationApplicationRefused(
                "investigation_request_needed", "request_file"
            )
        return captured.request
    try:
        supplied = combined_request(
            [read_investigation_request(path) for path in request_files]
        )
    except InvestigationRequestRefused as error:
        raise InvestigationApplicationRefused(error.code, *error.fields) from error
    if captured is not None and supplied != captured.request:
        raise InvestigationApplicationRefused(
            "investigation_request_conflict", "request_file"
        )
    return supplied


def _validate_begin_replay(journey, command, projection) -> None:
    """A repeated entry uses its original continuation, without work."""
    start = next(
        event["payload"]
        for event in read_evidence(journey)
        if event["event_type"] == "InvestigationStarted"
    )
    continuation = (
        _same_run_continuation(journey)
        if command.scientific_continuation_file is None
        else _read_continuation(command.scientific_continuation_file)
    )
    # Packet validation establishes its immutable identity. The retained start
    # already checked the original terminal; it need not still be the current head.
    if continuation.continuation_id != start["continuation_id"]:
        raise InvestigationApplicationRefused(
            "investigation_continuation_stale", "scientific_continuation_file"
        )


def begin_investigation(
    command: BeginInvestigationCommand,
    *,
    environ: Mapping[str, str],
    model_factory: NebiusModelEndpointFactory | None = None,
    source_adapter_factory: Callable[[], InvestigationSourceAdapter] | None = None,
) -> InvestigationProjection:
    """Begin inside the coordinator's processing Pass; ``advance`` is the caller."""
    journey = Journey.open(Path(command.journey_file))
    interval = current_processing_pass()
    if interval is None or interval.journey is not journey:
        raise InvestigationApplicationRefused("investigation_begin_outside_pass")
    projection = current_investigation(journey)
    request = resolve_investigation_request(journey, command.request_files)
    if projection.status != "not_started":
        _validate_begin_replay(journey, command, projection)
        return projection
    if command.scientific_continuation_file is None:
        continuation = _same_run_continuation(journey)
        timeline = tuple(read_evidence(journey))
    else:
        continuation = _read_continuation(command.scientific_continuation_file)
        timeline = _validated_continuation_timeline(continuation, journey)
    investigation_input = InvestigationInput.create(
        vcf=InvestigationVcfIdentity.create(
            bucket=continuation.vcf.bucket,
            object_key=continuation.vcf.object_key,
            byte_count=continuation.vcf.byte_count,
            sha256=continuation.vcf.sha256,
        ),
        sample_id=continuation.sample.sample_id,
        assembly_accession=continuation.reference.assembly_accession,
    )
    request_items = request.items
    origin = current_investigation_request(journey)
    predecessors = tuple(
        dict.fromkeys(
            (
                investigation_input.input_id,
                continuation.continuation_id,
                *((origin.source_event_id, *origin.source_event_ids) if origin else ()),
            )
        )
    )
    research_goals = () if origin is None else origin.research_goals
    # Every new Investigation is a collection, a collection of one included (ADR
    # 0039); readers still read retained singleton assertions.
    assertion = UserAssertion.create_collection(
        asserted_by=request.asserted_by,
        requests=request_items,
        predecessor_ids=predecessors,
        research_goals=research_goals,
    )
    from bio_neb.investigation.run_reference import (
        carried_reference_sources,
        run_species_taxid,
    )

    start = InvestigationStarted(
        continuation_id=continuation.continuation_id,
        investigation_input=investigation_input,
        assertion=assertion,
        # Planning's reference facts for the Run's assembly, carried once.
        run_reference_sources=carried_reference_sources(
            timeline, investigation_input.assembly_accession
        ),
        run_taxid=run_species_taxid(timeline, investigation_input.assembly_accession),
    )
    from bio_neb.agentic_journey.agent_roles import RuntimeBinding
    from bio_neb.agentic_journey.runtime_config import RuntimeConfigError
    from bio_neb.application.run_failure import hold_for_configuration_fix
    from bio_neb.investigation.investigation_assignment import (
        resolve_investigation_configuration,
        set_up_investigation_sources,
    )

    # A Fork child reads its parent's registration in its inherited prefix.
    registration = _registration(journey)
    runtime = RuntimeBinding(registration.runtime_selections, registration.input_set_id)
    try:
        resolve_investigation_configuration(
            runtime, environ, overrides=command.invocation_overrides
        )
    except RuntimeConfigError as error:
        # Before the start is written: the fault is retained at the frontier
        # and holds the Run until the fix and ``resume``.
        hold_for_configuration_fix(journey, timeline[-1]["event_id"], error)
    # Set up before the start is written, as configuration is: a start with no
    # Attempt reads as a failed Investigation, never as the setup fault's hold.
    sources = set_up_investigation_sources(source_adapter_factory)
    payload = start.model_dump(mode="json", exclude={"event_type"})
    journey.record(start.event_type, payload, [timeline[-1]["event_id"]])
    return plan_collection_members(
        journey,
        environ=environ,
        model_factory=model_factory,
        source_adapter_factory=lambda: sources,
        invocation_overrides=lambda _member: command.invocation_overrides,
    )


def _record_not_checked(journey: Journey, projection: InvestigationProjection):
    """Before any Attempt, record each original member the Run cannot check by what
    it states, with its reason, once (#367, readiness Q2 b): it asks nothing."""
    from bio_neb.investigation.investigation import (
        InvestigationRequestItem,
        InvestigationRequestNotChecked,
        investigation_output_member_id,
    )
    from bio_neb.investigation.investigation_request import not_checked_reason

    if projection.attempts or not projection.assertions or projection.not_checked:
        return projection
    events = read_evidence(journey)
    start = next(e for e in events if e["event_type"] == "InvestigationStarted")
    assertion = projection.assertions[0]
    requests = assertion.requests or (
        InvestigationRequestItem(
            request_text=assertion.request_text, rsid=assertion.rsid
        ),
    )
    recorded = False
    for ordinal, request in enumerate(requests):
        reason = not_checked_reason(request)
        if reason is None:
            continue
        fact = InvestigationRequestNotChecked(
            collection_member_id=investigation_output_member_id(request, ordinal),
            reason_code=reason[0],
            reason=reason[1],
        )
        journey.record(
            fact.event_type,
            fact.model_dump(mode="json", exclude={"event_type"}),
            [start["event_id"]],
        )
        recorded = True
    if not recorded:
        return projection
    return current_investigation(journey)


def plan_collection_members(
    journey: Journey,
    *,
    environ: Mapping[str, str],
    model_factory: NebiusModelEndpointFactory | None,
    source_adapter_factory: Callable[[], InvestigationSourceAdapter] | None,
    invocation_overrides,
) -> InvestigationProjection:
    """Plan every dispatchable member in its own Attempt, then close the proposal.

    Members run together, at most the Run's ``planner_members`` bound at once,
    and are collected; a member's question or stop holds only that member. Once
    every member settles the collection's proposal is recorded (#105).
    """
    from bio_neb.agentic_journey.attempt_runner import AttemptUnit, run_attempts
    from bio_neb.investigation.investigation_assignment import (
        dispatchable_members,
        investigation_assignment,
    )
    from bio_neb.investigation.investigation_questions import close_collection
    from bio_neb.investigation.observation_defaults import selected_concurrency

    projection = _record_not_checked(
        journey,
        current_investigation(journey),
    )
    members = dispatchable_members(projection)
    if members:
        bound = selected_concurrency(_registration(journey).runtime_selections)
        try:
            run_attempts(
                tuple(
                    AttemptUnit(
                        investigation_assignment(
                            model_factory=model_factory,
                            source_adapter_factory=source_adapter_factory,
                            member_id=member,
                        ),
                        invocation_overrides=invocation_overrides(member),
                    )
                    for member in members
                ),
                journey,
                environ=environ,
                concurrency=bound.planner_members,
            )
        except Exception as fault:
            # Work every member settled before the fault still closes; the
            # fault then reaches the coordinator as it was raised.
            try:
                close_collection(
                    journey,
                    current_investigation(journey),
                )
            except (ValueError, OSError) as closure_error:
                fault.add_note(
                    "settled collection could not close: "
                    + type(closure_error).__name__
                )
            raise
        projection = current_investigation(journey)
    return close_collection(journey, projection)


def continue_investigation(
    command: ContinueInvestigationCommand,
    *,
    environ: Mapping[str, str],
    model_factory: NebiusModelEndpointFactory | None = None,
    source_adapter_factory: Callable[[], InvestigationSourceAdapter] | None = None,
) -> InvestigationProjection:
    journey = Journey.open(Path(command.journey_file))
    projection = current_investigation(journey)
    if (
        projection.status != "investigation_work_ready"
        or projection.investigation_input is None
        or not projection.assertions
    ):
        raise InvestigationApplicationRefused("investigation_continue_ineligible")
    revision_before = journey.revision

    def overrides(member):
        # A member continues under the overrides its own last Attempt used.
        previous = next(
            (
                attempt
                for attempt in reversed(projection.attempts)
                if attempt.collection_member_id == member
            ),
            projection.attempts[-1] if projection.attempts else None,
        )
        if projection.eligible_member_ids and previous is not None:
            return AgentInvocationOverrides.model_validate(
                previous.invocation_overrides
            )
        return command.invocation_overrides

    planned = plan_collection_members(
        journey,
        environ=environ,
        model_factory=model_factory,
        source_adapter_factory=source_adapter_factory,
        invocation_overrides=overrides,
    )
    if Journey.open(journey.path).revision == revision_before:
        # No member to plan and no settled collection to close.
        raise InvestigationApplicationRefused("investigation_continue_ineligible")
    return planned


def _respond_member_question(journey, read, command):
    """Answer one member's question or issue Review; only that member is affected."""
    from bio_neb.investigation.investigation_questions import (
        close_collection,
        question_answer,
    )

    projection = read.investigation
    request, retained_answer = question_answer(
        projection.review_requests,
        projection.review_receipts,
        command.request_event_id,
        command.response,
    )
    if retained_answer is not None:
        return projection
    responded = InvestigationReviewResponded.create(
        request_event_id=command.request_event_id,
        response=command.response,
    )
    response_event = journey.record(
        responded.event_type,
        responded.model_dump(mode="json", exclude={"event_type"}),
        [command.request_event_id],
    )
    # Rejecting an issue removes only its item; other work continues.
    return close_collection(
        journey,
        current_investigation(journey),
        response_event["event_id"],
    )


def respond_investigation_review(  # noqa: C901 - Existing review semantics preserved during import relocation.
    command: RespondInvestigationReviewCommand,
) -> InvestigationProjection:
    from bio_neb.application.queries import read_investigation

    journey = Journey.open(Path(command.journey_file))
    read = read_investigation(journey)
    projection = read.investigation
    if command.response.action in {"clarify", "reject", "amend"} or any(
        request.request_event_id == command.request_event_id
        and request.question is not None
        for request in projection.review_requests
    ):
        return _respond_member_question(journey, read, command)
    from bio_neb.junction import evaluate_scoped_answer

    def refusal(reason, event_id):
        return InvestigationApplicationRefused(
            f"investigation_review_{reason}", request_event_id=event_id
        )

    def validate(request, response):
        if response.action not in request.legal_actions:
            raise refusal("illegal", request.request_event_id)
        if response.proposal_id != request.proposal.proposal_id:
            raise refusal("stale", request.request_event_id)

    requests = projection.review_requests
    if isinstance(command.response, ReviseInvestigationResponse) and requests:
        # A fully selected collection has no approval hold, but its fresh final
        # Review still declares a scoped revision opportunity.
        requests = (*requests[:-1], requests[-1].model_copy(update={"pending": True}))
    pending, retained = evaluate_scoped_answer(
        requests,
        projection.review_receipts,
        command.request_event_id,
        command.response,
        validate=validate,
        refusal=refusal,
    )
    if retained is not None:
        return projection
    if isinstance(command.response, ReviseInvestigationResponse):
        evidence = read.evidence
        if projection.status in {"failed", "stopped"} or (
            evidence is not None and evidence.failed
        ):
            raise InvestigationApplicationRefused("investigation_rework_ineligible")
        if pending is not None and pending.proposal != projection.proposals[-1]:
            raise InvestigationApplicationRefused(
                "investigation_review_stale", "proposal_id"
            )
    if pending is None or pending.request_event_id != command.request_event_id:
        raise InvestigationApplicationRefused(
            "investigation_review_stale",
            "request_event_id",
            request_event_id=command.request_event_id,
        )
    if command.response.proposal_id != pending.proposal.proposal_id:
        raise InvestigationApplicationRefused(
            "investigation_review_stale",
            "proposal_id",
            request_event_id=command.request_event_id,
        )
    selected_items = ()
    if isinstance(command.response, ApproveInvestigationResponse):
        try:
            selected_items = command.response.selected_items(pending.proposal)
        except ValueError as error:
            raise InvestigationApplicationRefused(
                "investigation_review_selection_invalid", "work_item_ids"
            ) from error
        if pending.remaining_work_item_ids is not None and not {
            item.work_item_id for item in selected_items
        } <= set(pending.remaining_work_item_ids):
            raise InvestigationApplicationRefused(
                "investigation_review_conflict", "work_item_ids"
            )
        policy = read.work_policy
        if set(policy.held_work_item_ids).intersection(
            item.work_item_id for item in selected_items
        ):
            raise InvestigationApplicationRefused(
                "investigation_work_item_held", "work_item_ids"
            )
        if any(
            item.work_item_id not in projection.pending_work_item_ids
            for item in selected_items
        ):
            raise InvestigationApplicationRefused(
                "investigation_review_conflict", "work_item_ids"
            )
    if (
        isinstance(command.response, ApproveInvestigationResponse)
        and not pending.proposal.work_items
    ):
        raise InvestigationApplicationRefused(
            "investigation_cardinality_invalid",
            "proposal_id",
            request_event_id=command.request_event_id,
        )
    responded = InvestigationReviewResponded.create(
        request_event_id=command.request_event_id,
        response=command.response,
    )
    revision = None
    contact_free_proposal = None
    new_assertion = None
    if isinstance(command.response, ReviseInvestigationResponse):
        try:
            revision = compile_investigation_collection_revision(
                proposal=pending.proposal,
                response_id=responded.response_id,
                response=command.response,
            )
        except InvestigationRevisionRefused as error:
            raise InvestigationApplicationRefused(
                f"investigation_review_{error.code}",
                "operations",
                request_event_id=command.request_event_id,
            ) from error
        if revision.work_requests:
            new_assertion = UserAssertion.create_collection(
                asserted_by=command.response.asserted_by,
                requests=revision.work_requests,
                predecessor_ids=(
                    pending.proposal.proposal_id,
                    command.request_event_id,
                    revision.revision_id,
                ),
            )
        else:
            assert projection.investigation_input is not None
            contact_free_proposal = (
                compile_contact_free_investigation_revision_proposal(
                    investigation_input=projection.investigation_input,
                    previous_proposal=pending.proposal,
                    revision=revision,
                )
            )
    response_parents = [command.request_event_id]
    if isinstance(command.response, ReviseInvestigationResponse):
        from bio_neb.reporting.finalization import (
            validate_finalization_review_fact,
        )

        head = journey.latest_event()
        final_review = validate_finalization_review_fact(head)
        if (
            final_review is not None
            and final_review.return_to_investigation is not None
            and final_review.receipt.response.action == "correct"
            and final_review.receipt.response.correction.investigation_revision
            == command.response
        ):
            responded = responded.model_copy(
                update={"finalization_response_event_id": head["event_id"]}
            )
            response_parents.append(head["event_id"])
    response_event = journey.record(
        responded.event_type,
        responded.model_dump(mode="json", exclude={"event_type"}),
        response_parents,
    )
    if isinstance(command.response, ApproveInvestigationResponse):
        authority = InvestigationAuthority.create(
            proposal_id=pending.proposal.proposal_id,
            review_response_id=responded.response_id,
            authorized_work_items=selected_items,
            predecessor_ids=(
                pending.proposal.proposal_id,
                responded.response_id,
            ),
        )
        fact: BaseModel = InvestigationAuthorityGranted(
            request_event_id=command.request_event_id,
            response_event_id=response_event["event_id"],
            authority=authority,
        )
    elif isinstance(command.response, StopInvestigationResponse):
        fact = InvestigationStopped(
            request_event_id=command.request_event_id,
            response_event_id=response_event["event_id"],
            proposal_id=pending.proposal.proposal_id,
            reason=command.response.reason,
        )
    else:
        assert revision is not None
        revision_event = journey.record(
            "InvestigationCollectionRevisionRecorded",
            InvestigationCollectionRevisionRecorded(
                response_event_id=response_event["event_id"], revision=revision
            ).model_dump(mode="json", exclude={"event_type"}),
            [response_event["event_id"]],
        )
        if new_assertion is not None:
            fact = InvestigationAssertionRecorded(
                revision_event_id=revision_event["event_id"],
                assertion=new_assertion,
            )
            journey.record(
                fact.event_type,
                fact.model_dump(mode="json", exclude={"event_type"}),
                [revision_event["event_id"]],
            )
            return current_investigation(journey)
        assert contact_free_proposal is not None
        proposal_event = journey.record(
            "InvestigationProposalCarriedForward",
            InvestigationProposalCarriedForward(
                revision_event_id=revision_event["event_id"],
                proposal=contact_free_proposal,
            ).model_dump(mode="json", exclude={"event_type"}),
            [revision_event["event_id"]],
        )
        request = InvestigationReviewRequested(
            review_policy=InvestigationReviewPolicy(),
            attempt_id=None,
            proposal_event_id=proposal_event["event_id"],
            proposal_id=contact_free_proposal.proposal_id,
        )
        request_event = journey.record(
            request.event_type,
            request.model_dump(mode="json", exclude={"event_type"}),
            [proposal_event["event_id"]],
        )
        if evidence is not None:
            from bio_neb.investigation.investigation_observation import (
                InvestigationCollectionItemState,
                fold_investigation_collection,
            )

            prior_states = {item.work_item_id: item for item in evidence.item_states}
            approved_ids = {
                item.work_item_id
                for grant in projection.authorities
                for item in grant.authorized_work_items
            }
            carried_states = tuple(
                prior_states[item.work_item_id].model_copy(
                    update={"selected_ordinal": ordinal}
                )
                if item.work_item_id in prior_states
                and prior_states[item.work_item_id].status
                not in {"approval_pending", "observation_pending"}
                else InvestigationCollectionItemState(
                    work_item_id=item.work_item_id,
                    selected_ordinal=ordinal,
                    status="observation_pending"
                    if item.work_item_id in approved_ids
                    else "approval_pending",
                )
                for ordinal, item in enumerate(
                    contact_free_proposal.work_items, start=1
                )
            )
            carried = fold_investigation_collection(
                command=evidence.command,
                authority=evidence.authority,
                item_states=carried_states,
                collection_work_item_ids=tuple(
                    item.work_item_id for item in contact_free_proposal.work_items
                ),
                collection_proposal_id=contact_free_proposal.proposal_id,
                carried_revision_id=revision.revision_id,
                previous_collection_state_id=evidence.collection_state_id,
                work_policy=read.work_policy.effective,
                work_policy_id=read.work_policy.policy_id,
            )
            journey.record(
                carried.event_type,
                carried.model_dump(
                    mode="json", exclude={"event_type"}, exclude_computed_fields=True
                ),
                [request_event["event_id"]],
            )
        return current_investigation(journey)
    journey.record(
        fact.event_type,
        fact.model_dump(mode="json", exclude={"event_type"}),
        [response_event["event_id"]],
    )
    if isinstance(command.response, ApproveInvestigationResponse):
        remaining = tuple(
            sorted(
                set(projection.pending_work_item_ids)
                - {item.work_item_id for item in selected_items}
            )
        )
        proposal_event = next(
            e
            for e in read_evidence(journey)
            if e["event_type"]
            in {"InvestigationProposalRecorded", "InvestigationProposalCarriedForward"}
            and e["payload"]["proposal"]["proposal_id"] == pending.proposal.proposal_id
        )
        following = InvestigationReviewRequested(
            review_policy=pending.review_policy,
            proposal_event_id=proposal_event["event_id"],
            proposal_id=pending.proposal.proposal_id,
            follows_response_event_id=response_event["event_id"],
            remaining_work_item_ids=remaining,
            legal_actions=("approve", "revise", "stop")
            if remaining
            else ("revise", "stop"),
        )
        journey.record(
            following.event_type,
            following.model_dump(mode="json", exclude={"event_type"}),
            [proposal_event["event_id"], response_event["event_id"]],
        )
    return current_investigation(journey)


__all__ = [
    "BeginInvestigationCommand",
    "ContinueInvestigationCommand",
    "InvestigationApplicationRefused",
    "RespondInvestigationReviewCommand",
    "begin_investigation",
    "continue_investigation",
    "respond_investigation_review",
]
