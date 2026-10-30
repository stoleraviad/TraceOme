"""Shared Attempt assignment contract and its bound Attempt context.

The shared lifecycle in ``attempt_runner`` owns policy, contact, effort and
availability mechanics. An assignment supplies what its work means through the
hooks below: what it admits, how it drives its members and which terminals it
records. The runner never interprets an assignment's admitted state.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic_ai import Agent

from bio_neb.agentic_journey.adapters import (
    NebiusModelEndpointFactory,
    create_nebius_model_endpoint,
)
from bio_neb.agentic_journey.attempt_journey import AttemptFailureSource
from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationOverrides,
    AgentInvocationPolicy,
    ContinuationPolicy,
)
from bio_neb.agentic_journey.lifecycle import Activity
from bio_neb.agentic_journey.model_observation import JourneyObservedModel
from bio_neb.agentic_journey.model_settings import ModelEndpointPolicy
from bio_neb.agentic_journey.role_instructions import (
    RoleInstructions,
)
from bio_neb.agentic_journey.role_instructions import (
    role_instructions as role_instructions,
)
from bio_neb.agentic_journey.runtime_config import (
    RuntimeSelections,
)
from bio_neb.journey import Journey

Admitted = TypeVar("Admitted")
Bound = TypeVar("Bound")
Dependencies = TypeVar("Dependencies")


@dataclass(frozen=True)
class Reentry:
    availability_review_response_event_id: str | None
    effort_review_response_event_id: str | None
    work_review_response_event_id: str | None = None
    operational_review_response_event_id: str | None = None


@dataclass(frozen=True)
class RuntimeBinding:
    """The exact secret-free runtime selection an Attempt resolves before writing."""

    runtime_selections: RuntimeSelections
    input_set_id: str


@dataclass(frozen=True)
class AttemptAdmission(Generic[Admitted]):
    """What an assignment binds before the shared lifecycle writes anything.

    An effort stop retries from ``causal_head``; the effort review requires it.
    A provider-availability stop retries from ``availability_retry_frontier``:
    the causal head, or the frontier of the availability review being answered.
    """

    admitted: Admitted
    runtime: RuntimeBinding
    causal_head: str
    availability_retry_frontier: str
    # The one collection member the Attempt works on; its stops hold only it.
    member_id: str | None = None
    # The researcher's request this work answers: stops of Attempts that started
    # before it do not hold it, since a new request opens fresh work.
    holds_from_event_id: str | None = None


@dataclass(frozen=True)
class AttemptContext(Generic[Admitted]):
    """One running Attempt; ``availability_retry_frontier`` is its admission's."""

    admitted: Admitted
    journey: Journey
    causal_head: str
    availability_retry_frontier: str
    attempt_id: str
    input_set_id: str
    activity: Activity
    failure_source: AttemptFailureSource
    effective_policy: AgentInvocationPolicy
    # The Attempt's bound model policy: roles, assembly and the adapter read it
    # and never look a model name up.
    model: ModelEndpointPolicy
    # The role's text in its model's form, as the Attempt's start recorded it.
    instructions: RoleInstructions

    def output_session(
        self,
        *,
        tool_name: str,
        output_type: type,
        arguments_refusal: Callable[[dict[str, Any]], str | None] | None = None,
    ):
        """The Attempt's output Tool session: the bound failure rule and its model's
        correction wording; the role supplies only what its output means (#446): its
        output type, from which the session derives its one validation (#383 Q6).
        A refused output names its reply's place in the failed turns in a row."""
        from bio_neb.agentic_journey.continuation_policy import current_turn
        from bio_neb.agentic_journey.structured_output import StructuredOutputSession
        from bio_neb.failure import FaultCount
        from bio_neb.recording import evidence_view

        continuation = self.effective_policy.continuation

        def turn_count() -> FaultCount:
            return FaultCount(
                *current_turn(
                    evidence_view(self.journey),
                    attempt_id=self.attempt_id,
                    policy=continuation,
                )
            )

        return StructuredOutputSession(
            parent_activity=self.activity,
            attempt_id=self.attempt_id,
            agent_role=self.effective_policy.agent_role,
            tool_name=tool_name,
            max_retries=continuation.most_failed_turns,
            turn_count=turn_count,
            failure_source=self.failure_source,
            output_type=output_type,
            wording=self.model.correction_wording,
            arguments_refusal=arguments_refusal,
        )


@dataclass(frozen=True)
class BoundInvocation(Generic[Dependencies]):
    agent: Agent[Dependencies, Any]
    deps: Dependencies
    prompt: Callable[[], str]


class ObserveModel(Protocol):
    """Wrap the Attempt's model; a typed outcome supplies its output options."""

    def __call__(
        self, *, structured_output_session: Any | None = None
    ) -> JourneyObservedModel: ...


class ValidateContinuation(Protocol):
    def __call__(
        self,
        events: Iterable[Mapping[str, Any]],
        *,
        attempt_id: str,
        policy: ContinuationPolicy,
        next_request: bool = False,
    ) -> None: ...


class InvokeMember(Protocol):
    """Run one member Agent inside the Attempt, against its one request allowance."""

    def __call__(
        self,
        agent: Agent[Any, Any],
        prompt: str | None,
        *,
        deps: Any,
        member_id: str,
        message_history: Sequence[Any] | None = None,
        completed_work: Callable[[], Any | None] | None = None,
    ) -> Awaitable[Any]: ...


@dataclass(frozen=True)
class AttemptBinding:
    """The input set and runtime selections one Attempt's own domain start bound."""

    input_set_id: str
    runtime_selections: RuntimeSelections


AttemptOutcome = Literal["succeeded", "failed", "cancelled", "stopped"]


@dataclass(frozen=True)
class AttemptTerminalReading:
    """An Attempt's checked terminal, read back by the domain that recorded it.

    ``domain_event_ids`` are the other facts the domain itself recorded in the
    Attempt with a Tool's fact kind, such as a question no Tool call asked.
    """

    outcome: AttemptOutcome
    terminal_event_id: str
    domain_event_ids: tuple[str, ...] = ()


class EvidenceSnapshot(Protocol):
    """One exact evidence prefix and the interpretations derived from it."""

    timeline: list[dict[str, Any]]

    def derived(self, reader: Callable[[EvidenceSnapshot], Any]) -> Any: ...

    def before(self, event_id: str) -> EvidenceSnapshot: ...


@dataclass(frozen=True)
class DomainBoundReopen:
    """How an Attempt that its domain's own start binds reads back.

    Such an Attempt does not compose in the Run's registered pre-execution
    slice: ``binding`` reads the runtime its domain start bound and ``terminal``
    the terminal its domain checked, both from retained evidence only. The
    terminal reads the snapshot every Attempt of the rebuild shares, so a
    domain projection is derived once for all of them (ADR 0032).
    """

    binding: Callable[[list[dict[str, Any]], list[dict[str, Any]]], AttemptBinding]
    terminal: Callable[[EvidenceSnapshot, list[dict[str, Any]]], AttemptTerminalReading]


@dataclass(frozen=True)
class AttemptReopen:
    """How the one Attempt view reopens this assignment's retained Attempts.

    ``admission`` is the assignment's admission spec, whose repeat identity and
    error the recorded turns replay; ``validate_facts`` reopens its domain facts
    against their admitted calls, reading the rebuild's shared snapshot; ``compiled_checkpoint`` says a successful
    Attempt closes at the checkpoint the terminal projector compiles;
    ``structured_output`` says the Attempt may retain model-authored typed
    output; ``domain_bound`` reads an Attempt its domain's own start binds,
    and is ``None`` for an Attempt in the Run's registered slice.
    """

    admission: Any = None
    validate_facts: Callable[[EvidenceSnapshot, list[dict[str, Any]]], None] | None = (
        None
    )
    compiled_checkpoint: bool = False
    structured_output: bool = False
    domain_bound: DomainBoundReopen | None = None


@dataclass(frozen=True)
class AttemptAssignment(Generic[Admitted, Bound]):
    """One kind of Agent work run by the shared Attempt lifecycle.

    ``admit`` binds the assignment's state, runtime, causal head and retry
    frontiers without writing. A configuration fault is not the assignment's:
    the shared lifecycle records it at the causal head and holds the Run.
    ``drive`` may invoke several members; they share the Attempt's request
    allowance. ``complete`` and ``fail`` record the assignment's own terminals.
    ``fail`` also receives what ``open`` yielded (``None`` if it never did) and
    returns ``False`` when the error is not one it can retain.
    ``reopen`` is what the Attempt view needs to read this work back.
    """

    agent_role: str
    activity_stage: str
    admit: Callable[[Journey, Reentry], AttemptAdmission[Admitted]]
    validate_continuation: ValidateContinuation
    open: Callable[
        [AttemptContext[Admitted], ObserveModel], AbstractContextManager[Bound]
    ]
    drive: Callable[
        [AttemptContext[Admitted], Bound, InvokeMember, Sequence[Any] | None],
        Awaitable[Any],
    ]
    complete: Callable[[AttemptContext[Admitted], Any], None]
    # Retains the Attempt's failure (False: nothing to retain); a role whose
    # failure holds the Run raises the failure module's Invariant instead.
    fail: Callable[[AttemptContext[Admitted], Bound | None, Exception], bool]
    read_result: Callable[[Journey], Any]
    model_factory: NebiusModelEndpointFactory = create_nebius_model_endpoint
    reopen: AttemptReopen = AttemptReopen()
    # After one Attempt run alone closes, the domain settles what it completed
    # (a collection whose last member settled closes its proposal). A batch's
    # caller settles once every unit has closed.
    settle: Callable[[Journey], None] | None = None

    def run(
        self,
        journey: Journey,
        *,
        environ: Mapping[str, str],
        availability_review_response_event_id: str | None = None,
        effort_review_response_event_id: str | None = None,
        work_review_response_event_id: str | None = None,
        operational_review_response_event_id: str | None = None,
        invocation_overrides: AgentInvocationOverrides | None = None,
    ) -> Any:
        """Run one eligible Attempt with this assignment's bound dependencies."""
        from bio_neb.agentic_journey.attempt_runner import run_attempt

        return run_attempt(
            self,
            journey,
            environ=environ,
            model_factory=self.model_factory,
            availability_review_response_event_id=availability_review_response_event_id,
            effort_review_response_event_id=effort_review_response_event_id,
            work_review_response_event_id=work_review_response_event_id,
            operational_review_response_event_id=operational_review_response_event_id,
            invocation_overrides=invocation_overrides,
        )
