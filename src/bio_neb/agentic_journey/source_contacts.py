"""Stage-bound source transport evidence, independent of scientific results."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import cache
from types import MappingProxyType
from typing import Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
)

from bio_neb.agentic_journey.lifecycle import Activity
from bio_neb.failure import (
    ContactLimitOverrides,
    ContactLimits,
    FailureClass,
    FailureSurface,
    ProviderError,
    ResolvedFailurePolicy,
    ResolvedLimits,
    call_with_failure_policy,
    limits_for,
    operation_retry_decision,
    policy_for,
)
from bio_neb.journey import _writers_lock, canonical_json
from bio_neb.providers.source_operations import SOURCE_OPERATIONS, SourceSurface
from bio_neb.recording import (
    ContactScope,
    current_processing_pass,
    physical_contact,
    read_evidence,
)


@cache
def _source_surfaces() -> Mapping[str, SourceSurface]:
    """Every source Tool's declared surface: each Agent Tool's on its registry row,
    and the paper download's in its own module (#447). The inventory is read on
    first use: the domains declaring rows import this module."""
    from bio_neb.agentic_journey.tool_execution import function_tool_contracts
    from bio_neb.providers.paper_sources import PAPER_DOWNLOAD_SURFACES

    surfaces = {
        row.tool_name: row.source_surface
        for row in function_tool_contracts()
        if row.source_surface is not None
    }
    surfaces |= PAPER_DOWNLOAD_SURFACES
    if not set().union(*(s.operations for s in surfaces.values())) <= set(
        SOURCE_OPERATIONS
    ):
        raise RuntimeError("a source Tool names an operation no definition registers")
    return MappingProxyType(surfaces)


def source_tools() -> frozenset[str]:
    """Every source failure surface: the source Tools' and each registered operation's."""
    return frozenset(_source_surfaces()) | frozenset(SOURCE_OPERATIONS)


def contact_surfaces(tool_name: str) -> frozenset[str]:
    """The failure surfaces a source Tool's contacts may run on: its operations'
    own surfaces, or else its own one surface."""
    surface = _source_surfaces().get(tool_name)
    if surface is not None and surface.operations:
        return surface.operations
    return frozenset({tool_name})


def declared_limits(surface_name: str) -> ContactLimitOverrides:
    """A source surface's own limits over the failure package's defaults: an
    operation's registration, else its Tool's declaration, else none."""
    for surface in (
        SOURCE_OPERATIONS.get(surface_name),
        _source_surfaces().get(surface_name),
    ):
        if surface is not None and surface.limits is not None:
            return surface.limits
    return ContactLimitOverrides()


T = TypeVar("T")


class _Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceContactScopeStarted(_Fact):
    event_type: Literal["SourceContactScopeStarted"] = "SourceContactScopeStarted"
    attempt_id: StrictStr
    tool_call_id: StrictStr
    tool_name: StrictStr

    @field_validator("tool_name")
    @classmethod
    def registered_tool(cls, value: str) -> str:
        if value not in source_tools():
            raise ValueError("source scope names no registered source Tool")
        return value


class SourceContactScopeFinished(_Fact):
    event_type: Literal["SourceContactScopeFinished"] = "SourceContactScopeFinished"
    scope_event_id: StrictStr
    transport_observed: StrictBool


class SourcePhysicalContactStarted(_Fact):
    event_type: Literal["SourcePhysicalContactStarted"] = "SourcePhysicalContactStarted"
    scope_event_id: StrictStr
    request_identity: StrictStr
    method: Literal["GET", "HEAD"] = "GET"


class SourcePhysicalContactFinished(_Fact):
    event_type: Literal["SourcePhysicalContactFinished"] = (
        "SourcePhysicalContactFinished"
    )
    started_event_id: StrictStr
    disposition: Literal["succeeded", "failed"]
    error_type: StrictStr | None = None


class SourceOperationFailureRecorded(_Fact):
    event_type: Literal["SourceOperationFailureRecorded"] = (
        "SourceOperationFailureRecorded"
    )
    scope_event_id: StrictStr
    policy: ResolvedFailurePolicy
    contact_attempt: int = Field(ge=1)
    failure_class: FailureClass
    original_failure_class: FailureClass
    detail: StrictStr
    retry_delay_seconds: float | None = Field(default=None, ge=0)


class SourceContactLimitReached(_Fact):
    """A source limit refused a contact before it was sent, or an answer as read."""

    event_type: Literal["SourceContactLimitReached"] = "SourceContactLimitReached"
    scope_event_id: StrictStr
    limit: Literal["contacts", "run_contacts", "response_bytes"]
    bound: int = Field(ge=1)
    observed: int = Field(ge=0)
    limits: ResolvedLimits


def call_source(tool_name: str, operation):
    """Apply the Tool's transient policy inside its admitted source operation.

    The surface's limits stay bound while it runs, so its transport reads the
    byte bound (`response_byte_bound`) without a parameter.
    """
    scope = _scope.get()
    if scope is not None and tool_name not in contact_surfaces(scope.tool_name):
        raise RuntimeError("source contact names a Tool other than its scope's")
    surface = FailureSurface(scope="tool", name=tool_name)
    policy = policy_for(surface)

    def observe(error, ordinal, delay):
        if scope is not None:
            scope.activity.record_contact(
                SourceOperationFailureRecorded(
                    scope_event_id=scope.event_id,
                    policy=policy,
                    contact_attempt=ordinal,
                    failure_class=error.failure_class,
                    original_failure_class=error.evidence["original_failure_class"],
                    detail=error.detail,
                    retry_delay_seconds=delay,
                )
            )

    token = _surface_limits.set(limits_for(surface))
    try:
        return call_with_failure_policy(operation, policy, observe)
    except ProviderError as error:
        # The surface whose policy routed it: a Tool's or one of its operations'.
        error.evidence.setdefault("surface", tool_name)
        raise
    finally:
        _surface_limits.reset(token)


_surface_limits: ContextVar[ResolvedLimits | None] = ContextVar(
    "source_surface_limits", default=None
)


def response_byte_bound() -> int:
    """The byte bound of the surface whose contact runs now; the default outside one."""
    limits = _surface_limits.get()
    return (limits.effective if limits else ContactLimits()).response_bytes


def response_over_bound(size: int) -> bool:
    """Whether an answer of ``size`` bytes breaks its bound; its scope records a breach."""
    limits = _surface_limits.get()
    bound = (limits.effective if limits else ContactLimits()).response_bytes
    if size <= bound:
        return False
    scope = _scope.get()
    if scope is not None and limits is not None:
        scope.activity.record_contact(
            SourceContactLimitReached(
                scope_event_id=scope.event_id,
                limit="response_bytes",
                bound=bound,
                observed=size,
                limits=limits,
            )
        )
    return True


_FACTS = {
    owner.model_fields["event_type"].default: owner
    for owner in (
        SourceContactScopeStarted,
        SourceContactScopeFinished,
        SourcePhysicalContactStarted,
        SourcePhysicalContactFinished,
        SourceOperationFailureRecorded,
        SourceContactLimitReached,
    )
}


def validate_source_contact_fact(event: Mapping[str, Any]) -> BaseModel | None:
    owner = _FACTS.get(event.get("event_type"))
    return None if owner is None else owner.model_validate(event.get("payload", {}))


@dataclass
class _Scope:
    activity: Activity
    event_id: str
    attempt_id: str
    tool_call_id: str
    tool_name: str
    transport_observed: bool = False
    # The call's recorded physical contacts; a child stage shares the list.
    contacts: list[str] = field(default_factory=list)


_scope: ContextVar[_Scope | None] = ContextVar("source_contact_scope", default=None)


def observed_source_operation(
    activity: Activity,
    attempt_id: str,
    tool_call_id: str,
    tool_name: str,
    operation: Callable[[], T],
) -> Callable[[], T]:
    """Let the stage bind transport evidence to its admitted Tool operation."""

    def execute() -> T:
        event = activity.record_contact(
            SourceContactScopeStarted(
                attempt_id=attempt_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
        )
        scope = _Scope(activity, event["event_id"], attempt_id, tool_call_id, tool_name)
        token = _scope.set(scope)
        try:
            return operation()
        finally:
            _scope.reset(token)
            activity.record_contact(
                SourceContactScopeFinished(
                    scope_event_id=scope.event_id,
                    transport_observed=scope.transport_observed,
                )
            )

    return execute


@contextmanager
def source_contact_stage(activity: Activity) -> Iterator[None]:
    """Bind physical observations to an active child without opening a new Tool scope."""
    scope = _scope.get()
    if scope is None:
        raise RuntimeError("source stage requires an open source scope")
    scope.activity.require_active_child(activity)
    child_scope = replace(scope, activity=activity)
    token = _scope.set(child_scope)
    try:
        yield
    finally:
        scope.transport_observed |= child_scope.transport_observed
        _scope.reset(token)


def scope_contact_event_ids() -> tuple[str, ...]:
    """The open source scope's physical contacts so far, by their journal evidence
    event ids, in the order they were sent: what a Tool's result names as the
    contacts it made (#420)."""
    scope = _scope.get()
    if scope is None:
        raise RuntimeError("source contacts are read inside an open source scope")
    return tuple(scope.contacts)


def observe_source_transport() -> None:
    """Mark instrumentation before the transport's contact-free validation."""
    scope = _scope.get()
    if scope is not None:
        scope.transport_observed = True


@contextmanager
def source_physical_attempt(
    request_identity: str, *, method: Literal["GET", "HEAD"]
) -> Iterator[None]:
    """Observe exactly one invoked transport attempt; never initiate or retry one.

    The Journey fact and the Pass-journal operation name the method actually sent.
    """
    scope = _scope.get()
    if scope is None:
        yield
        return
    # Count and record under the writer lock, so concurrent calls share a total.
    with _writers_lock:
        _refuse_past_limits(scope)
        event = scope.activity.record_contact(
            SourcePhysicalContactStarted(
                scope_event_id=scope.event_id,
                request_identity=request_identity,
                method=method,
            )
        )
        scope.contacts.append(event["event_id"])
    with physical_contact(
        ContactScope(
            attempt_id=scope.attempt_id,
            request_id=scope.event_id,
            call_id=scope.tool_call_id,
            evidence_event_id=event["event_id"],
        ),
        f"source.{method.lower()}",
    ):
        try:
            yield
        except BaseException as error:
            scope.activity.record_contact(
                SourcePhysicalContactFinished(
                    started_event_id=event["event_id"],
                    disposition="failed",
                    error_type=type(error).__name__,
                )
            )
            raise
        else:
            scope.activity.record_contact(
                SourcePhysicalContactFinished(
                    started_event_id=event["event_id"],
                    disposition="succeeded",
                )
            )


def _refuse_past_limits(scope: _Scope) -> None:
    """Refuse the next contact of a Tool call, or of the Run, past its limit.

    The breach is a failure (ADR 0031): the call's policy routes it to the
    surface's Review. Counts are recorded facts, so no Pass resets the Run's.
    """
    limits = limits_for(FailureSurface(scope="tool", name=scope.tool_name))
    reached = [("contacts", limits.effective.contacts, len(scope.contacts))]
    if limits.run_contacts is not None:
        reached.append(("run_contacts", limits.run_contacts, _run_contacts()))
    for limit, bound, observed in reached:
        if observed >= bound:
            scope.activity.record_contact(
                SourceContactLimitReached(
                    scope_event_id=scope.event_id,
                    limit=limit,
                    bound=bound,
                    observed=observed,
                    limits=limits,
                )
            )
            raise ProviderError(
                "exhausted",
                f"The source contact was not sent: its {limit} limit of {bound}"
                " is reached.",
                evidence={"limit": limit, "bound": bound},
            )


def _run_contacts() -> int:
    """The Run's own physical source contacts; a Fork's inherited ones keep their Run."""
    journey = current_processing_pass().journey
    return sum(
        event["event_type"] == "SourcePhysicalContactStarted"
        and event["journey_id"] == journey.journey_id
        for event in read_evidence(journey)
    )


def _bind_source_tool_terminal(
    event: Mapping[str, Any],
    *,
    calls: Mapping[tuple[str, str], tuple[str, str]],
    contracts,
    terminal_calls: set[str],
    bound_results: set[str],
    by_id: Mapping[str, Mapping[str, Any]],
    indexed_events: Mapping[str, tuple[int, Mapping[str, Any]]],
) -> None:
    """Bind a source Tool's one terminal to its exact existing domain result."""
    from bio_neb.agentic_journey.tool_execution import (
        ObservedToolExecutionFailed,
        ObservedToolExecutionSucceeded,
    )

    terminal_owner = {
        "ObservedToolExecutionSucceeded": ObservedToolExecutionSucceeded,
        "ObservedToolExecutionFailed": ObservedToolExecutionFailed,
    }.get(event.get("event_type"))
    if terminal_owner is not None:
        terminal = terminal_owner.model_validate_json(
            canonical_json(event.get("payload", {})), strict=True
        )
        matching = [
            (key, name)
            for key, (start_id, name) in calls.items()
            if start_id == terminal.started_event_id
        ]
        if matching:
            key, name = matching[0]
            if (
                terminal.started_event_id in terminal_calls
                or terminal.attempt_id != key[0]
            ):
                raise ValueError("source Tool terminal changed its unique call")
            terminal_calls.add(terminal.started_event_id)
            result_id = (
                terminal.stage_result_event_id
                if isinstance(terminal, ObservedToolExecutionSucceeded)
                else terminal.stage_failure_event_id
            )
            if result_id is not None:
                result = by_id.get(result_id)
                expected = (
                    contracts[name].stage_result_event_type
                    if isinstance(terminal, ObservedToolExecutionSucceeded)
                    else contracts[name].stage_failure_event_type
                )
                if (
                    result is None
                    or result.get("event_type") != expected
                    or result.get("payload", {}).get("attempt_id") != key[0]
                    or result_id in bound_results
                    or indexed_events[result_id][0]
                    <= indexed_events[terminal.started_event_id][0]
                ):
                    raise ValueError("source Tool terminal changed its stage result")
                bound_results.add(result_id)


def _check_operation_failure(
    fact: SourceOperationFailureRecorded,
    scopes: Mapping[str, SourceContactScopeStarted],
    finished_scopes: set[str],
    operation_failures: dict[str, list[list[SourceOperationFailureRecorded]]],
) -> None:
    """A source failure continues its contact's attempts under one scoped policy."""
    contacts_failed = operation_failures.setdefault(fact.scope_event_id, [])
    prior = contacts_failed[-1] if contacts_failed and fact.contact_attempt != 1 else []
    routed, ceiling = operation_retry_decision(
        fact.original_failure_class, fact.contact_attempt, fact.policy
    )
    if (
        fact.scope_event_id not in scopes
        or fact.scope_event_id in finished_scopes
        or fact.policy.surface.scope != "tool"
        or fact.policy.surface.name
        not in contact_surfaces(scopes[fact.scope_event_id].tool_name)
        or fact.contact_attempt != len(prior) + 1
        # A contact after one the policy gave up on cannot start.
        or (
            not prior
            and contacts_failed
            and contacts_failed[-1][-1].retry_delay_seconds is None
        )
        or fact.original_failure_class == "invariant"
        or (
            prior
            and (
                prior[-1].retry_delay_seconds is None or prior[-1].policy != fact.policy
            )
        )
        or routed != fact.failure_class
        or (ceiling is None) != (fact.retry_delay_seconds is None)
        or (ceiling is not None and not 0 <= fact.retry_delay_seconds <= ceiling)
    ):
        raise ValueError("source failure changed its scoped policy or attempt")
    if prior:
        prior.append(fact)
    else:
        contacts_failed.append([fact])


def _check_limit_breach(
    fact: SourceContactLimitReached,
    scopes: Mapping[str, SourceContactScopeStarted],
    finished_scopes: set[str],
) -> None:
    """A breach names its open scope's surface and its recorded limit's bound.

    A count is the called Tool's; a byte bound the surface its contact ran on.
    """
    bound = {
        "contacts": fact.limits.effective.contacts,
        "run_contacts": fact.limits.run_contacts,
        "response_bytes": fact.limits.effective.response_bytes,
    }[fact.limit]
    scope = scopes.get(fact.scope_event_id)
    surfaces = (
        set()
        if scope is None
        else contact_surfaces(scope.tool_name)
        if fact.limit == "response_bytes"
        else {scope.tool_name}
    )
    if (
        fact.scope_event_id in finished_scopes
        or fact.limits.surface.name not in surfaces
        or fact.bound != bound
        # A count reached its bound; an answer exceeded its bound.
        or fact.observed < bound + (fact.limit == "response_bytes")
    ):
        raise ValueError("source limit breach lacks its exact open scope")


def source_contact_units(
    events: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], bool]:
    """Validate stage/call/transport correlation and retain every attempted unit."""
    from bio_neb.agentic_journey.tool_execution import (
        ObservedToolExecutionStarted,
        function_tool_contracts,
        validate_observed_tool_admission,
    )

    contracts = {
        item.tool_name: item
        for item in function_tool_contracts()
        if item.tool_name in source_tools()
    }
    by_id: dict[str, Mapping[str, Any]] = {}
    indexed_events: dict[str, tuple[int, Mapping[str, Any]]] = {}
    terminal_calls: set[str] = set()
    bound_results: set[str] = set()

    calls: dict[tuple[str, str], tuple[str, str]] = {}
    scopes: dict[str, SourceContactScopeStarted] = {}
    scoped_calls: set[tuple[str, str]] = set()
    finished_scopes: set[str] = set()
    starts: dict[str, SourcePhysicalContactStarted] = {}
    finished: set[str] = set()
    # Each scope's contacts' failures in order: a first attempt opens one contact's
    # list, a later attempt continues it (a Tool may make several contacts).
    operation_failures: dict[str, list[list[SourceOperationFailureRecorded]]] = {}
    observed = True
    seen: set[str] = set()
    for index, event in enumerate(events):
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or event_id in seen:
            raise ValueError(
                "source accounting requires unique Journey event identities"
            )
        seen.add(event_id)
        by_id[event_id] = event
        indexed_events[event_id] = (index, event)
        generic = (
            ObservedToolExecutionStarted.model_validate(event.get("payload", {}))
            if event.get("event_type") == "ObservedToolExecutionStarted"
            else None
        )
        if generic is not None and generic.event_type == "ObservedToolExecutionStarted":
            if generic.tool_name in source_tools():
                key = (generic.attempt_id, generic.tool_call_id)
                if key in calls:
                    raise ValueError("source Tool execution repeated")
                if (
                    generic.adapter_identity
                    != contracts[generic.tool_name].adapter_identity
                ):
                    raise ValueError(
                        "source Tool changed its declared Adapter identity"
                    )
                validate_observed_tool_admission(index, generic, indexed_events)
                calls[key] = (event_id, generic.tool_name)
        _bind_source_tool_terminal(
            event,
            calls=calls,
            contracts=contracts,
            terminal_calls=terminal_calls,
            bound_results=bound_results,
            by_id=by_id,
            indexed_events=indexed_events,
        )
        fact = validate_source_contact_fact(event)
        if isinstance(fact, SourceContactScopeStarted):
            key = (fact.attempt_id, fact.tool_call_id)
            if (
                key not in calls
                or calls[key][1] != fact.tool_name
                or key in scoped_calls
            ):
                raise ValueError("source scope lacks its exact unique Tool execution")
            # Exact Attempt/call identity binds the journal execution. Semantic
            # parentage remains between Journey facts, never invented from a
            # mechanical record's position.
            if calls[key][0] in terminal_calls:
                raise ValueError("source scope follows its Tool terminal")
            scopes[event_id] = fact
            scoped_calls.add(key)
        elif isinstance(fact, SourcePhysicalContactStarted):
            if (
                fact.scope_event_id not in scopes
                or fact.scope_event_id in finished_scopes
            ):
                raise ValueError("source attempt is outside its open scope")
            starts[event_id] = fact
        elif isinstance(fact, SourcePhysicalContactFinished):
            start = starts.get(fact.started_event_id)
            if (
                start is None
                or fact.started_event_id in finished
                or start.scope_event_id in finished_scopes
            ):
                raise ValueError("source terminal lacks its exact open attempt")
            finished.add(fact.started_event_id)
        elif isinstance(fact, SourceOperationFailureRecorded):
            _check_operation_failure(fact, scopes, finished_scopes, operation_failures)
        elif isinstance(fact, SourceContactLimitReached):
            _check_limit_breach(fact, scopes, finished_scopes)
        elif isinstance(fact, SourceContactScopeFinished):
            if (
                fact.scope_event_id not in scopes
                or fact.scope_event_id in finished_scopes
            ):
                raise ValueError("source scope terminal lacks its exact open scope")
            finished_scopes.add(fact.scope_event_id)
            observed = observed and fact.transport_observed
    result_types = {item.stage_result_event_type for item in contracts.values()}
    result_types.update(
        item.stage_failure_event_type
        for item in contracts.values()
        if item.stage_failure_event_type
    )
    unbound_results = {
        event["event_id"] for event in events if event.get("event_type") in result_types
    } - bound_results
    if (
        scopes
        and unbound_results
        and terminal_calls == {item[0] for item in calls.values()}
    ):
        raise ValueError(
            "source accounting contains unsupported unbound stage evidence"
        )
    complete = (
        bool(scopes)
        and observed
        and scoped_calls == set(calls)
        and finished_scopes == set(scopes)
        and finished == set(starts)
        and terminal_calls == {item[0] for item in calls.values()}
        and not unbound_results
    )
    return tuple(f"journey-source:{event_id}" for event_id in starts), complete


def source_semantic_parent(
    event: Mapping[str, Any],
    events_by_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Follow only transport evidence between a stage contact and its result."""
    parents = event.get("parent_ids", ())
    seen: set[str] = set()
    while len(parents) == 1:
        event_id = parents[0]
        if event_id in seen:
            raise ValueError("source contact ancestry repeats")
        seen.add(event_id)
        parent = events_by_id.get(event_id)
        if parent is None:
            return None
        fact = validate_source_contact_fact(parent)
        if not isinstance(
            fact, (SourcePhysicalContactStarted, SourcePhysicalContactFinished)
        ):
            return parent
        parents = parent.get("parent_ids", ())
    return None
