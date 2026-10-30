"""Shared outer lifecycle for every Attempt assignment."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.usage import RunUsage

from bio_neb.agentic_journey.adapters import NebiusModelEndpointFactory
from bio_neb.agentic_journey.agent_roles import (
    AttemptAssignment,
    AttemptContext,
    Reentry,
    RuntimeBinding,
)
from bio_neb.agentic_journey.agent_work_review import (
    AgentFailedTurns,
    AgentResultMissing,
    AgentWorkStop,
    granted_failed_turns,
    pending_agent_work,
    record_agent_work_stop,
    refuse_fresh_agent_work,
)
from bio_neb.agentic_journey.attempt_journey import (
    AttemptFailureSource,
    AttemptStarted,
)
from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationOverrides,
    AgentInvocationPolicy,
    model_output_faults,
)
from bio_neb.agentic_journey.effort import (
    AgentEffortExhausted,
    attempt_requests_used,
    pending_effort_work,
    record_effort_stop,
    refuse_fresh_effort_allowance,
)
from bio_neb.agentic_journey.instruction_evidence import INSTRUCTION_MEDIA_TYPE
from bio_neb.agentic_journey.invocation import (
    INTERRUPTED_TOOL_FAILURES,
    invoke_agent,
)
from bio_neb.agentic_journey.lifecycle import ActivityLifecycle
from bio_neb.agentic_journey.model_availability import (
    ModelAvailabilityReviewPending,
    model_availability_direction,
    pending_model_work,
    record_model_attempt_cancelled,
    record_model_availability_review_pending,
)
from bio_neb.agentic_journey.model_observation import JourneyObservedModel
from bio_neb.agentic_journey.model_settings import ModelEndpointPolicy
from bio_neb.agentic_journey.operational_review import (
    OperationalStop,
    record_operational_stop,
)
from bio_neb.agentic_journey.role_instructions import (
    InstructionsUnrenderable,
    RoleInstructions,
    instructions_for,
)
from bio_neb.agentic_journey.runtime_config import (
    LoadedRuntimeConfig,
    NebiusModelEndpointSelection,
    RuntimeConfigError,
    model_endpoint_identity,
    resolve_runtime_config,
)
from bio_neb.agentic_journey.tool_execution import function_tool_names
from bio_neb.application.run_failure import hold_for_configuration_fix
from bio_neb.failure import run_failure_tuning
from bio_neb.journey import Journey
from bio_neb.recording import evidence_view, invocation_pass, read_evidence


@dataclass(frozen=True)
class AttemptConfiguration:
    """Validated runtime, policy and the role's instructions in its model's form;
    resolving them acquires no model client."""

    loaded: LoadedRuntimeConfig
    effective_policy: AgentInvocationPolicy
    model_selection: NebiusModelEndpointSelection
    instructions: RoleInstructions

    @property
    def model(self) -> ModelEndpointPolicy:
        """The Attempt's bound model policy, resolved once with its configuration."""
        return self.loaded.model


def resolve_attempt_configuration(
    runtime: RuntimeBinding,
    environ: Mapping[str, str],
    *,
    agent_role: str,
    overrides: AgentInvocationOverrides,
) -> AttemptConfiguration:
    """Resolve the bound runtime for one Agent role, or raise ``RuntimeConfigError``.

    The lifecycle resolves it before ``AttemptStarted``; a caller that writes
    before the Attempt may resolve it first to refuse before its own write. The
    role's instructions render here, so ones that cannot render are refused
    before any model is acquired (``instructions_unrenderable``, I-65).
    """
    loaded = resolve_runtime_config(
        runtime.runtime_selections,
        environ,
        agent_role=agent_role,
        overrides=overrides,
    )
    effective_policy = loaded.invocation_policy
    selected = loaded.selections.model_for_policy(effective_policy)
    try:
        instructions = instructions_for(agent_role, loaded.model)
        instructions.require_tool_purposes(function_tool_names(agent_role))
    except InstructionsUnrenderable as error:
        raise RuntimeConfigError(error.code, error.violations) from error
    return AttemptConfiguration(loaded, effective_policy, selected, instructions)


@dataclass
class _ModelLifetime:
    """Release acquired resources while preserving an already handled work failure."""

    model: Any
    primary_error: BaseException | None = None

    async def __aenter__(self):
        try:
            await self.model.__aenter__()
        except BaseException as error:
            await self.__aexit__(type(error), error, error.__traceback__)
            raise
        return self

    async def __aexit__(self, exc_type, error, traceback):
        primary = error if error is not None else self.primary_error
        try:
            await self.model.__aexit__(
                type(primary) if primary is not None else None,
                primary,
                primary.__traceback__ if primary is not None else None,
            )
        except BaseException as cleanup_error:
            # Domain facts are already retained. Never replay work or run its
            # failure hook merely because model release failed.
            if primary is not None:
                primary.add_note(
                    "Model client cleanup also failed: " + type(cleanup_error).__name__
                )
                raise primary from cleanup_error
            raise


def hold_stopped_role(
    journey: Journey,
    reentry: Reentry,
    *,
    agent_role: str,
    member_id: str | None,
    holds_from_event_id: str | None = None,
) -> None:
    """A stopped role's review holds a fresh Attempt until its own answer reopens it.

    Applied after role eligibility and before any write. A member-scoped stop
    holds only that member's work (#105); work that answers a later researcher
    request is held only by stops since that request (#408).
    """
    # Shared read-only evidence: the hold checks read it, never copy it.
    events = evidence_view(journey)
    if reentry.effort_review_response_event_id is None:
        refuse_fresh_effort_allowance(
            events,
            agent_role=agent_role,
            member_id=member_id,
            holds_from_event_id=holds_from_event_id,
            checked_once=journey.derived_once,
        )
    if reentry.work_review_response_event_id is None:
        refuse_fresh_agent_work(
            events,
            agent_role=agent_role,
            member_id=member_id,
            holds_from_event_id=holds_from_event_id,
            checked_once=journey.derived_once,
        )
    from bio_neb.agentic_journey.model_availability import (
        project_model_availability_review,
    )
    from bio_neb.agentic_journey.operational_review import project_operational_review

    holds = (
        *project_model_availability_review(
            events, checked_once=journey.derived_once
        ).requests,
        *project_operational_review(events, checked_once=journey.derived_once).requests,
    )
    if (
        agent_role == "planning_agent"
        and reentry.availability_review_response_event_id is None
        and reentry.operational_review_response_event_id is None
    ):
        if any(
            request.agent_role == agent_role and request.pending for request in holds
        ):
            raise ValueError(
                "operational Review holds this Agent role until its exact answer"
            )
        stopped = [
            e
            for e in events
            if e["event_type"]
            in {"ModelAvailabilityReviewResponded", "AgentFailureReviewResponded"}
            and e["payload"]["response"]["action"] == "stop"
        ]
        if any(
            any(
                r.request_event_id == e["payload"]["request_event_id"]
                and r.agent_role == agent_role
                for r in holds
            )
            for e in stopped
        ):
            raise ValueError("operational stop cannot start a fresh Agent Attempt")


def run_attempt(
    assignment: AttemptAssignment,
    journey: Journey,
    *,
    environ: Mapping[str, str],
    model_factory: NebiusModelEndpointFactory,
    availability_review_response_event_id: str | None = None,
    effort_review_response_event_id: str | None = None,
    work_review_response_event_id: str | None = None,
    operational_review_response_event_id: str | None = None,
    invocation_overrides: AgentInvocationOverrides | None = None,
) -> Any:
    """Run one Attempt and return the assignment's read of its closed Pass."""

    def invoke(*args, **kwargs):
        result = asyncio.run(_run_attempt(*args, **kwargs))
        if assignment.settle is None:
            return result
        assignment.settle(journey)
        return assignment.read_result(journey)

    return invocation_pass(invoke, read_result=assignment.read_result)(
        assignment,
        journey,
        environ=environ,
        model_factory=model_factory,
        reentry=Reentry(
            availability_review_response_event_id,
            effort_review_response_event_id,
            work_review_response_event_id,
            operational_review_response_event_id,
        ),
        invocation_overrides=invocation_overrides,
    )


def _model_reentry_work(
    journey, response_event_id, *, agent_role, invocation_overrides
):
    """Keep policy fixed; only retry inherits the failed model transcript."""
    request, receipt = model_availability_direction(
        journey, response_event_id, agent_role=agent_role
    )
    prior_start = next(
        e
        for e in read_evidence(journey)
        if e["event_type"] == "AttemptStarted"
        and e["payload"]["attempt_id"] == request.attempt_id
    )
    overrides = AgentInvocationOverrides.model_validate(
        prior_start["payload"]["invocation_overrides"]
    )
    if (
        invocation_overrides is not None
        and invocation_overrides.model_fields_set
        and invocation_overrides != overrides
    ):
        raise ValueError("model availability direction cannot amend invocation policy")
    # An item that joined the open Review before sending has no pending request
    # to resend: its retry starts fresh from its frontier, as an amendment does.
    if receipt.response.action == "retry" and not request.joined_before_sending:
        return pending_model_work(journey, response_event_id, agent_role=agent_role)
    return None, None, overrides


def _operational_reentry_work(journey, response_event_id, *, agent_role, overrides):
    from bio_neb.agentic_journey.operational_review import (
        pending_operational_work,
        resumed_transcript,
    )

    request, receipt = pending_operational_work(
        journey, response_event_id, agent_role=agent_role
    )
    prior_start = next(
        e
        for e in read_evidence(journey)
        if e["event_type"] == "AttemptStarted"
        and e["payload"]["attempt_id"] == request.attempt_id
    )
    retained_overrides = AgentInvocationOverrides.model_validate(
        prior_start["payload"]["invocation_overrides"]
    )
    if overrides.model_fields_set and overrides != retained_overrides:
        raise ValueError("operational direction cannot amend invocation policy")
    if receipt.response.action != "retry":
        return None, None, retained_overrides
    return request, resumed_transcript(request), retained_overrides


def _restore_member_tool(request, member_id, deps):
    if request is None or member_id != request.member_id:
        return None
    deps.observed_tool_session.restore_pending(request)
    return request.tool_call_id


def _validate_reentry(reentry: Reentry) -> None:
    directions = (
        reentry.availability_review_response_event_id,
        reentry.effort_review_response_event_id,
        reentry.work_review_response_event_id,
        reentry.operational_review_response_event_id,
    )
    if sum(value is not None for value in directions) > 1:
        raise ValueError("Attempt cannot consume two review directions")


@dataclass(frozen=True)
class AttemptUnit:
    """One independent Attempt of a batch: its assignment and the answer it consumes."""

    assignment: AttemptAssignment
    reentry: Reentry = Reentry(None, None)
    invocation_overrides: AgentInvocationOverrides | None = None


def run_attempts(
    units: tuple[AttemptUnit, ...],
    journey: Journey,
    *,
    environ: Mapping[str, str],
    concurrency: int,
) -> None:
    """Run independent Attempts, at most ``concurrency`` at once, and collect them.

    Every unit is admitted before any of them writes, so siblings share the one
    causal head their batch began at and never read each other's open work. Each
    unit is its own task with its own allowance and terminal; a sibling's fault
    never stops another, and the first fault reaches the caller once every
    sibling has closed. An interrupt cancels every unit still in flight (#105).
    """
    if concurrency < 1:
        raise ValueError("an Attempt batch needs a positive concurrency bound")

    def invoke(_units, journey):
        prepared = [
            _prepare_attempt(
                unit.assignment,
                journey,
                environ=environ,
                model_factory=unit.assignment.model_factory,
                reentry=unit.reentry,
                invocation_overrides=unit.invocation_overrides,
            )
            for unit in _units
        ]
        asyncio.run(_execute_bounded(prepared, journey, concurrency))

    invocation_pass(invoke)(tuple(units), journey)


async def _execute_bounded(prepared, journey: Journey, concurrency: int) -> None:
    gate = asyncio.Semaphore(concurrency)

    async def one(item):
        async with gate:
            await _execute_attempt(item, journey, read=False)

    outcomes = await asyncio.gather(
        *(one(item) for item in prepared), return_exceptions=True
    )
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome


@dataclass
class _PreparedAttempt:
    """What an Attempt decided before it writes: admission, policy and model client."""

    assignment: AttemptAssignment
    reentry: Reentry
    admission: Any
    overrides: AgentInvocationOverrides
    pending_request: Any
    message_history: Any
    operational_request: Any
    configuration: AttemptConfiguration
    model_factory: NebiusModelEndpointFactory


async def _run_attempt(
    assignment: AttemptAssignment,
    journey: Journey,
    *,
    environ: Mapping[str, str],
    model_factory: NebiusModelEndpointFactory,
    reentry: Reentry,
    invocation_overrides: AgentInvocationOverrides | None,
) -> Any:
    prepared = _prepare_attempt(
        assignment,
        journey,
        environ=environ,
        model_factory=model_factory,
        reentry=reentry,
        invocation_overrides=invocation_overrides,
    )
    return await _execute_attempt(prepared, journey)


def _prepare_attempt(
    assignment: AttemptAssignment,
    journey: Journey,
    *,
    environ: Mapping[str, str],
    model_factory: NebiusModelEndpointFactory,
    reentry: Reentry,
    invocation_overrides: AgentInvocationOverrides | None,
) -> _PreparedAttempt:
    effort_review_response_event_id = reentry.effort_review_response_event_id
    availability_review_response_event_id = (
        reentry.availability_review_response_event_id
    )
    _validate_reentry(reentry)
    effort_request = None
    if effort_review_response_event_id is not None:
        effort_request, effort_messages, effort_overrides = pending_effort_work(
            journey, effort_review_response_event_id, agent_role=assignment.agent_role
        )
    granted_turns = None
    if reentry.work_review_response_event_id is not None:
        # A new Attempt with a fresh count: no transcript or override carries
        # over, apart from a number of failed turns the researcher's answer grants.
        pending_agent_work(
            journey,
            reentry.work_review_response_event_id,
            agent_role=assignment.agent_role,
        )
        granted_turns = granted_failed_turns(
            journey, reentry.work_review_response_event_id
        )
    admission = assignment.admit(journey, reentry)
    hold_stopped_role(
        journey,
        reentry,
        agent_role=assignment.agent_role,
        member_id=admission.member_id,
        holds_from_event_id=admission.holds_from_event_id,
    )
    runtime = admission.runtime
    pending_request = None
    message_history = None
    overrides = invocation_overrides or AgentInvocationOverrides()
    if granted_turns is not None:
        if overrides.model_fields_set and overrides.failed_turns != granted_turns:
            raise ValueError("a work review grant cannot be amended by the caller")
        overrides = AgentInvocationOverrides(failed_turns=granted_turns)
    if effort_request is not None:
        if (
            invocation_overrides is not None
            and invocation_overrides.model_fields_set
            and invocation_overrides != effort_overrides
        ):
            raise ValueError(
                "effort extension cannot amend unrelated invocation policy"
            )
        message_history, overrides = effort_messages, effort_overrides
    if availability_review_response_event_id is not None:
        pending_request, message_history, overrides = _model_reentry_work(
            journey,
            availability_review_response_event_id,
            agent_role=assignment.agent_role,
            invocation_overrides=invocation_overrides,
        )
    operational_request = None
    if reentry.operational_review_response_event_id is not None:
        operational_request, message_history, overrides = _operational_reentry_work(
            journey,
            reentry.operational_review_response_event_id,
            agent_role=assignment.agent_role,
            overrides=overrides,
        )
    try:
        configuration = resolve_attempt_configuration(
            runtime,
            environ,
            agent_role=assignment.agent_role,
            overrides=overrides,
        )
    except RuntimeConfigError as error:
        hold_for_configuration_fix(journey, admission.causal_head, error)
    return _PreparedAttempt(
        assignment,
        reentry,
        admission,
        overrides,
        pending_request,
        message_history,
        operational_request,
        configuration,
        model_factory,
    )


async def _execute_attempt(
    prepared: _PreparedAttempt, journey: Journey, *, read: bool = True
) -> Any:
    assignment, reentry, admission = (
        prepared.assignment,
        prepared.reentry,
        prepared.admission,
    )
    causal_head, runtime = admission.causal_head, admission.runtime
    overrides, pending_request = prepared.overrides, prepared.pending_request
    message_history = prepared.message_history
    operational_request = prepared.operational_request
    configuration = prepared.configuration
    try:
        # The client is made inside the Attempt's event loop, as its transport is.
        selected_model = prepared.model_factory(
            configuration.loaded, configuration.model_selection
        )
    except RuntimeConfigError as error:
        hold_for_configuration_fix(journey, admission.causal_head, error)
    effort_review_response_event_id = reentry.effort_review_response_event_id
    availability_review_response_event_id = (
        reentry.availability_review_response_event_id
    )

    def result():
        # A batch unit returns nothing: its siblings may still be open.
        return assignment.read_result(journey) if read else None

    effective_policy = configuration.effective_policy
    lifetime = _ModelLifetime(selected_model)
    # The Run's recorded failure section tunes this Attempt's Tool contacts.
    with run_failure_tuning(runtime.runtime_selections.failure):
        async with lifetime:
            try:
                selected_endpoint = model_endpoint_identity(
                    selected_model, configuration.model_selection
                )
            except RuntimeConfigError as error:
                lifetime.primary_error = error
                hold_for_configuration_fix(journey, admission.causal_head, error)

            instructions = configuration.instructions
            attempt_id = str(uuid.uuid4())
            activity = ActivityLifecycle(journey, causal_head=causal_head).start(
                adapter="pydantic_ai",
                stage=assignment.activity_stage,
            )
            activity.record_with_artifact(
                AttemptStarted(
                    attempt_id=attempt_id,
                    instruction_identity=instructions.identity,
                    continuation_policy=effective_policy.continuation,
                    continuation_policy_id=effective_policy.continuation.policy_id,
                    effective_invocation_policy=effective_policy,
                    invocation_policy_id=effective_policy.policy_id,
                    invocation_overrides=overrides.explicit_settings(),
                    continuation_effort_review_event_id=effort_review_response_event_id,
                    continuation_availability_review_event_id=availability_review_response_event_id,
                    continuation_work_review_event_id=reentry.work_review_response_event_id,
                    continuation_operational_review_event_id=reentry.operational_review_response_event_id,
                    continuation_request_event_id=None
                    if pending_request is None
                    else pending_request["event_id"],
                    agent_role=assignment.agent_role,
                    input_set_id=runtime.input_set_id,
                    member_id=admission.member_id,
                    holds_from_event_id=admission.holds_from_event_id,
                ),
                instructions.content,
                media_type=INSTRUCTION_MEDIA_TYPE,
            )
            context = AttemptContext(
                admission.admitted,
                journey,
                causal_head,
                admission.availability_retry_frontier,
                attempt_id,
                runtime.input_set_id,
                activity,
                AttemptFailureSource(),
                effective_policy,
                configuration.model,
                instructions,
            )

            current_member_id = None

            def observe(*, structured_output_session=None) -> JourneyObservedModel:
                return JourneyObservedModel(
                    selected_model,
                    member_id=lambda: current_member_id,
                    parent_activity=activity,
                    attempt_id=attempt_id,
                    failure_source=context.failure_source,
                    nebius_model_endpoint=selected_endpoint,
                    settings_selection=effective_policy.settings_selection,
                    structured_output_session=structured_output_session,
                    contact_policy=effective_policy.model_contact,
                    continuation_request=pending_request,
                    before_request=lambda: assignment.validate_continuation(
                        evidence_view(journey),
                        attempt_id=attempt_id,
                        policy=effective_policy.continuation,
                        next_request=True,
                        member_id=current_member_id,
                    ),
                    agent_role=assignment.agent_role,
                    # An item an availability answer released re-sends once,
                    # past the hold of its still-open Review.
                    released=availability_review_response_event_id is not None,
                    run_evidence=lambda: evidence_view(journey),
                )

            # A Tool failing after an interrupt hands its failure to invoke_agent
            # through this list; this Attempt's task owns the binding.
            INTERRUPTED_TOOL_FAILURES.set([])
            # One allowance per Attempt: every member invocation draws on this usage.
            usage = RunUsage(
                requests=attempt_requests_used(evidence_view(journey), attempt_id)
            )

            async def invoke_member(
                agent,
                prompt,
                *,
                deps,
                member_id,
                message_history=None,
                completed_work=None,
            ):
                nonlocal current_member_id
                current_member_id = member_id
                deferred_call_id = _restore_member_tool(
                    operational_request, member_id, deps
                )
                try:
                    return await invoke_agent(
                        agent,
                        prompt,
                        deps=deps,
                        policy=effective_policy,
                        journey_id=journey.journey_id,
                        attempt_id=attempt_id,
                        member_id=member_id,
                        message_history=message_history,
                        usage=usage,
                        completed_work=completed_work,
                        deferred_call_id=deferred_call_id,
                    )
                except ModelAvailabilityReviewPending as pending:
                    pending.member_id = member_id
                    raise
                except AgentEffortExhausted as exhausted:
                    # The reply that reached its limit of failed turns stops for
                    # the Agent work Review even when it also spent the request
                    # budget: the framework's limit fires before the check.
                    faults = model_output_faults(
                        read_evidence(journey), attempt_id=attempt_id
                    )
                    if faults and not faults[-1].corrected(
                        effective_policy.continuation
                    ):
                        raise AgentFailedTurns(
                            "the model's replies failed their limit of turns in a row",
                            faults[-1].event_id,
                        ) from exhausted
                    raise
                except UnexpectedModelBehavior as error:
                    if context.failure_source.event_id is None:
                        # Any other model behavior the framework refused is the
                        # Agent work Review's, not a held Run; a recorded model
                        # or Tool failure keeps its own record.
                        raise AgentResultMissing(
                            f"the framework refused the model's reply: {error}"
                        ) from error
                    raise

            bound = None
            try:
                with assignment.open(context, observe) as bound:
                    outcome = await assignment.drive(
                        context, bound, invoke_member, message_history
                    )
            except OperationalStop as stop:
                record_operational_stop(context, stop, agent_role=assignment.agent_role)
                return result()
            except AgentEffortExhausted as exhausted:
                record_effort_stop(
                    activity,
                    exhausted,
                    attempt_id=attempt_id,
                    policy=effective_policy,
                    retry_frontier_event_id=causal_head,
                )
                return result()
            except AgentWorkStop as stop:
                record_agent_work_stop(
                    activity,
                    stop,
                    journey=journey,
                    attempt_id=attempt_id,
                    agent_role=assignment.agent_role,
                )
                return result()
            except ModelAvailabilityReviewPending as pending:
                record_model_availability_review_pending(
                    activity,
                    pending,
                    attempt_id=attempt_id,
                    agent_role=assignment.agent_role,
                    retry_frontier_event_id=admission.availability_retry_frontier,
                )
                return result()
            except asyncio.CancelledError:
                record_model_attempt_cancelled(activity, journey, attempt_id=attempt_id)
                raise
            except Exception as error:
                lifetime.primary_error = error
                assignment.fail(context, bound, error)
                raise
            assignment.complete(context, outcome)
            return result()
