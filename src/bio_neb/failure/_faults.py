"""Faults and policy: the failure classes, routes, held faults and each surface's
resolved policy and limits. This part records nothing in a Journey."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Annotated, Any, Literal, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from bio_neb.contracts import FrozenContract

FailureClass = Literal[
    "transient", "exhausted", "rejected", "worker_failed", "semantic", "invariant"
]
# What a held Pass records: an invariant, or the operator's interrupt that
# cancelled an Agent Attempt. The interrupt is no classified fault, so no failure
# policy routes it.
HeldClass = FailureClass | Literal["interrupted"]


class ProviderError(Exception):
    def __init__(
        self,
        failure_class: FailureClass,
        detail: str,
        evidence: dict[str, JsonValue] | None = None,
    ):
        super().__init__(detail)
        if failure_class not in {
            "transient",
            "exhausted",
            "rejected",
            "worker_failed",
            "semantic",
        }:
            raise ValueError(
                "provider error requires a provider class; bugs use Invariant"
            )
        self.failure_class = failure_class
        self.detail = detail
        self.evidence = evidence or {}


def operation_retry_decision(failure_class, ordinal, policy):
    """One shared decision for executing and validating classified operation faults."""
    if ordinal < 1 or ordinal > policy.effective.retry_attempts:
        raise ValueError("operation attempt exceeds its retained failure policy")
    if failure_class == "transient" and policy.effective.transient == "retry":
        if ordinal < policy.effective.retry_attempts:
            return failure_class, min(
                policy.effective.backoff_seconds * 2 ** (ordinal - 1),
                policy.effective.max_backoff_seconds,
            )
        return "exhausted", None
    return failure_class, None


def call_with_failure_policy(operation, policy, observe_failure):
    """Retry only classified transient contact faults, before domain effects.

    The caller owns the operation boundary and evidence sink. Unknown exceptions
    propagate unchanged; neither biological results nor normalization run here.
    """
    import random
    import time

    for ordinal in range(1, policy.effective.retry_attempts + 1):
        try:
            return operation()
        except ProviderError as error:
            original_class = error.failure_class
            error.failure_class, delay = operation_retry_decision(
                original_class, ordinal, policy
            )
            if delay is not None:
                delay = random.uniform(0, delay)
            error.evidence.update(
                original_failure_class=original_class, contact_attempt=ordinal
            )
            observe_failure(error, ordinal, delay)
            if delay is None:
                raise
            time.sleep(delay)
    raise AssertionError("failure policy returned without an outcome")


class ModelResponseMalformed(Exception):
    """A received model payload failed the provider adapter's response contract."""


class ModelRequestRefusedShown(Exception):
    """A request the model endpoint refused, ended so the next carries the
    provider's words to the model (owner, #476 alignment G1.5 B; #477)."""

    def __init__(self, status: int, statement: str):
        super().__init__(f"model endpoint refused the request: HTTP {status}")
        self.status = status
        self.statement = statement


class ModelContactStalled(Exception):
    """A send the contact loop cut off at the window's stall limit, with no reply.

    Raised by the contact loop itself: the provider may still be generating, and
    nothing of the reply was received (owner, #443 readiness).
    """

    def __init__(self, stall_seconds: float):
        super().__init__(f"no model reply within {stall_seconds:g} s")
        self.stall_seconds = stall_seconds


class ModelContextExhausted(Exception):
    """The next model request leaves no completion room in its context window.

    Raised at the adapter edge before any request bytes leave the process, so
    nothing was sent and no provider completion is unknown.
    """

    def __init__(self, *, input_estimate: int, safety_margin: int, context_window: int):
        super().__init__(
            f"model context exhausted: {input_estimate} estimated input tokens "
            f"plus {safety_margin} safety tokens leave no completion room in the "
            f"{context_window}-token context window"
        )
        self.input_estimate = input_estimate
        self.safety_margin = safety_margin
        self.context_window = context_window


class Invariant(Exception):
    """Our bug: retain its identity; the failure module holds the Run for a fix.

    A site that already recorded its failure names that failure's own ``cause``
    and ``evidence`` for the hold.
    """

    def __init__(
        self,
        invariant_id: str,
        detail: str,
        *,
        cause: str | None = None,
        evidence: dict[str, JsonValue] | None = None,
    ):
        super().__init__(detail)
        self.invariant_id = invariant_id
        self.cause = cause
        self.evidence = evidence or {}


class RecordedFactRefused(Invariant):
    """A fact whose own reopen check refused it before it was appended (owner,
    #476 C8): nothing was written, and the hold names the fact and the check."""

    def __init__(self, event_type: str, check: str, reason: str, payload: dict):
        super().__init__(
            "recorded_fact_refused",
            f"{event_type} refused by {check}: {reason}",
            evidence={
                "event_type": event_type,
                "check": check,
                "reason": reason,
                **{
                    name: payload[name]
                    for name in ("attempt_id", "tool_call_id")
                    if isinstance(payload.get(name), str)
                },
            },
        )
        self.event_type = event_type
        self.check = check
        self.reason = reason


class ToolRefused(Exception):
    """A Tool's declared refusal (owner, #477 C8): its arguments cannot produce its
    fact. The model is shown ``reason`` and ``statement`` as the call's result,
    the Attempt continues, and the refusal is one more failed turn."""

    def __init__(self, reason: str, statement: str):
        super().__init__(statement)
        self.reason = reason
        self.statement = statement


TOOL_FAILED = "tool_failed"
"""The reason a Tool's undeclared failure is shown to the model with, once."""


def tool_fault_key(tool_name: str, error: BaseException) -> str:
    """One Tool fault's key (owner, #477 C8 Q3): its Tool and its cause, a declared
    refusal's reason or else the exception's type; a recorded fact's refused
    check names that check."""
    if isinstance(error, ToolRefused):
        cause = error.reason
    elif isinstance(error, RecordedFactRefused):
        cause = f"{error.event_type}: {error.check}"
    else:
        cause = type(error).__name__
    return f"{tool_name}: {cause}"


def tool_failure_holds_at_once(error: BaseException) -> bool:
    """Whether a Tool's failure holds the Run before the model sees it (owner,
    #477 C8): our own fault on recorded data, a refused check or a failed local
    write, which continuing would build on."""
    from bio_neb.journey import JourneyValidationError

    return isinstance(error, (Invariant, JourneyValidationError, OSError))


FailureRoute = Literal["retry", "ask", "hold"]


def holds_run(failure_class: HeldClass) -> bool:
    """An invariant holds the Run for a fix and `resume`; no policy tunes it.

    That is our bug, or a runtime configuration fault, which keeps the class by
    owner decision D5 (#216; ADR 0031 Classification). An operator's interrupt
    that cancelled an Agent Attempt holds it for `resume` too (owner, #327).
    """
    return failure_class in {"invariant", "interrupted"}


class FailurePolicy(FrozenContract):
    retry_attempts: int = Field(default=3, ge=1)
    backoff_seconds: float = Field(default=1, ge=0, allow_inf_nan=False)
    max_backoff_seconds: float = Field(default=30, ge=0, allow_inf_nan=False)
    transient: Literal["retry", "ask"] = "retry"
    # No route ends a Run: each stored word says what happens, and no field
    # offers "terminal". An invariant holds the Run until a fix and `resume`.
    exhausted: Literal["ask"] = "ask"
    rejected: Literal["ask"] = "ask"
    worker_failed: Literal["ask"] = "ask"
    semantic: Literal["ask"] = "ask"
    invariant: Literal["hold"] = "hold"

    def route(self, failure_class: FailureClass) -> FailureRoute:
        if holds_run(failure_class):
            return "hold"
        return getattr(self, failure_class)


DEFAULT_FAILURE_POLICY = FailurePolicy()


class HeldFailure(FrozenContract):
    """The failure a held Pass records: a fault or an interrupt, typed id and evidence."""

    failure_class: HeldClass
    invariant_id: str | None = Field(default=None, min_length=1)
    cause: str = Field(min_length=1)
    evidence: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def typed_invariant(self):
        if (self.failure_class == "invariant") != (self.invariant_id is not None):
            raise ValueError("an invariant, and only an invariant, names its id")
        return self


class CommandInputInvalid(ValueError):
    """A caller's input broke its contract: the command refuses, the Run unchanged.

    Raised where the coordinator validates caller-supplied input, so the failure
    module tells it apart from the coordinator's own model validation (#214).
    """

    code = "command_invalid"

    def __init__(self, location: str, error: ValidationError):
        # Each violation's reason, never its input: a value may carry credentials.
        violations = error.errors(include_url=False, include_input=False)
        super().__init__(
            f"{location} breaks its input contract: "
            + "; ".join(item["msg"] for item in violations)
        )
        self.fields = tuple(
            sorted(
                {
                    ".".join((location, *(str(part) for part in item["loc"])))
                    for item in violations
                }
            )
        )


def validate_command_input(contract, location: str, value):
    """Validate caller-supplied ``value``; a violation refuses the command."""
    try:
        return contract.model_validate(value)
    except ValidationError as error:
        raise CommandInputInvalid(location, error) from error


def coordinator_fault(
    error: BaseException,
    *,
    cancelled_attempt: Mapping[str, Any] | None = None,
    cancelled_attempts: tuple[Mapping[str, Any], ...] = (),
) -> HeldFailure | None:
    """Classify a fault escaping the Run coordinator; None leaves it to its owner.

    A typed coordinator cause carries its own record, and an ``Invariant`` (a
    typed integrity refusal among them) keeps its own id. A command refusal (a
    typed ``code``, including a caller's input contract violation) refuses the
    command, and an invalid Journey refuses its reader; both leave the Run as it
    was. Anything else escaping the coordinator, a model validation inside it
    among them, is our bug.
    """
    from bio_neb.agentic_journey.failure_cause import cause_identity
    from bio_neb.journey import JourneyValidationError

    if not isinstance(error, Exception):
        # An interrupt that cancelled an Agent Attempt (``cancelled_attempt``, its
        # ``AttemptCancelled``) holds the Run; any other cancellation or
        # interpreter exit interrupts the Pass.
        if cancelled_attempt is None:
            return None
        evidence: dict[str, JsonValue] = {
            "attempt_id": cancelled_attempt["payload"]["attempt_id"],
            "attempt_cancelled_event_id": cancelled_attempt["event_id"],
        }
        if len(cancelled_attempts) > 1:
            # Concurrent Attempts the one interrupt cancelled, each named.
            evidence["cancelled_attempt_ids"] = [
                item["payload"]["attempt_id"] for item in cancelled_attempts
            ]
        return HeldFailure(
            failure_class="interrupted",
            cause=type(error).__name__,
            evidence=evidence,
        )
    failure = getattr(error, "failure", None)
    if isinstance(failure, HeldFailure):
        return failure
    if isinstance(error, Invariant):
        return HeldFailure(
            failure_class="invariant",
            invariant_id=error.invariant_id,
            cause=error.cause or cause_identity(error),
            evidence=error.evidence,
        )
    if isinstance(getattr(error, "code", None), str) or isinstance(
        error, JourneyValidationError
    ):
        return None
    return HeldFailure(
        failure_class="invariant",
        invariant_id="coordinator_unhandled_error",
        cause=cause_identity(error),
    )


ModelContactSetting = Literal[
    "contacts",
    "base_delay_seconds",
    "max_delay_seconds",
    "max_wait_seconds",
    "stall_seconds",
]


class ModelContactPolicy(FrozenContract):
    """One logical model request's physical window; independent of Agent effort.

    Budgets are per call (owner, #443 readiness 2026-10-24). A send with no reply
    after ``stall_seconds`` is cut off: a stall, which the next request notes. Any
    other short failure is sent once more after a jittered wait of up to
    ``base_delay_seconds``, within ``max_wait_seconds`` of waits, then asks the
    availability Review. ``stall_seconds`` stays under the transport's own
    1,800 s timeout and the gateway's 600 s.
    """

    contacts: int = Field(default=2, ge=1, strict=True)
    base_delay_seconds: float = Field(default=30, ge=0, allow_inf_nan=False)
    max_delay_seconds: float = Field(default=60, ge=0, allow_inf_nan=False)
    max_wait_seconds: float = Field(default=60, ge=0, allow_inf_nan=False)
    stall_seconds: float = Field(default=180, gt=0, le=1800, allow_inf_nan=False)


def _explicit_only(model: BaseModel, values: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value for name, value in values.items() if name in model.model_fields_set
    }


class FailureOverrides(FailurePolicy):
    """Sparse operation settings using the same constraints as the effective policy.

    Only supplied fields participate in resolution or serialization. Consumers
    execute with the resolved FailurePolicy, never these inherited defaults.
    """

    @model_serializer(mode="wrap")
    def explicit_settings(self, handler):
        return _explicit_only(self, handler(self))


class ModelContactOverrides(ModelContactPolicy):
    """Sparse model-contact settings, constrained as the effective window."""

    @model_serializer(mode="wrap")
    def explicit_settings(self, handler):
        return _explicit_only(self, handler(self))


# The failure section owns its surface names. Execution's are its StageDef
# names. Investigation's VCF query calls `vcf_observation`; the other
# Investigation names are accepted before their callers (R3).
EXECUTION_STAGES = ("input_storage", "admit_candidate", "cpu_preparation", "h100")
INVESTIGATION_STAGES = (
    "vcf_materialization",
    "bam_materialization",
    "vcf_observation",
    # The Run's own measurement: its processes' limits and Review.
    "run_health",
)
FailureScope = Literal["stage", "tool", "role"]
_UNBUILT = "this scope cannot set it until its seam is built"


def _stage_surface(name: str) -> str:
    if name not in EXECUTION_STAGES + INVESTIGATION_STAGES:
        raise ValueError("not a failure stage surface")
    return name


def _tool_surface(name: str) -> str:
    # The Tool registries import this module, so look them up late.
    from bio_neb.agentic_journey.source_contacts import source_tools
    from bio_neb.parabricks.recipe_resources import CAPABILITY_TOOLS

    if name not in source_tools() | CAPABILITY_TOOLS:
        raise ValueError("not a Tool that makes an operation contact")
    return name


def _role_surface(name: str) -> str:
    from bio_neb.agentic_journey.role_declaration import AgentRole

    if name not in get_args(AgentRole):
        raise ValueError("not an Agent role")
    return name


_SURFACE_NAMES = {"stage": _stage_surface, "tool": _tool_surface, "role": _role_surface}


class _SparseEntry(FrozenContract):
    """An omitted key inherits the wider value; an explicit null is refused."""

    @field_validator("*", mode="before")
    @classmethod
    def explicit_fields_are_values(cls, value):
        if value is None:
            raise ValueError("explicit failure settings cannot be null")
        return value

    @model_serializer(mode="wrap")
    def explicit_settings(self, handler):
        # Registration keeps omissions as omissions at every level.
        return _explicit_only(self, handler(self))


class OperationEntry(_SparseEntry):
    """A stage's or Tool's operation settings."""

    operation: FailureOverrides | None = None


class RoleEntry(_SparseEntry):
    """A role's model contact. No contact site knows its Attempt's role (U7)."""

    model_contact: ModelContactOverrides | None = None
    operation: FailureOverrides | None = None

    @field_validator("operation")
    @classmethod
    def unbuilt(cls, value):
        raise ValueError(_UNBUILT)


class FailureScopes(_SparseEntry):
    """Entries narrower than the global values, keyed by scope and then name."""

    role: dict[Annotated[str, AfterValidator(_role_surface)], RoleEntry] = Field(
        default_factory=dict
    )
    stage: dict[Annotated[str, AfterValidator(_stage_surface)], OperationEntry] = Field(
        default_factory=dict
    )
    tool: dict[Annotated[str, AfterValidator(_tool_surface)], OperationEntry] = Field(
        default_factory=dict
    )


class ContactLimits(FrozenContract):
    """A source Tool's limits, set in code (owner, #416).

    These are the module's defaults; a Tool's own declaration on its registry row,
    or an operation's on its registration, overrides them
    (`source_contacts.declared_limits`, #447). One Tool call makes at most
    ``contacts`` physical contacts, retries included; the next is refused before
    it is sent.
    An answer over ``response_bytes`` is refused as it is read.
    """

    contacts: int = Field(default=30, ge=1, strict=True)
    response_bytes: int = Field(default=4 * 1024 * 1024, ge=1, strict=True)


class ContactLimitOverrides(ContactLimits):
    """A Tool's own limits: only its supplied fields override the defaults."""

    @model_serializer(mode="wrap")
    def explicit_settings(self, handler):
        return _explicit_only(self, handler(self))


class StageLimits(FrozenContract):
    """A stage's limits, set in code (owners, #418 and #414).

    Each attempt of one object part (a ranged GET or an upload part of
    ``MULTIPART_PART_BYTES``) has ``part_seconds`` in total: it ends within them
    plus at most one bounded socket operation, and never succeeds after them
    (owner, #418 a). Each local process a stage runs ends within
    ``process_seconds`` or fails as a timeout (#414 Q4). These are the module's
    defaults; a stage's own declaration overrides them (`STAGE_LIMITS`). No
    runtime entry tunes them.
    """

    part_seconds: int = Field(default=300, ge=1, strict=True)
    process_seconds: int = Field(default=600, ge=1, strict=True)


class StageLimitOverrides(StageLimits):
    """A stage's own limits: only its supplied fields override the defaults."""

    @model_serializer(mode="wrap")
    def explicit_settings(self, handler):
        return _explicit_only(self, handler(self))


# Each stage's own limits declaration, in code. The Run's measurement reads its
# whole BAM in one process (a human Run's 3.66 GB).
STAGE_LIMITS: dict[str, StageLimitOverrides] = {
    "run_health": StageLimitOverrides(process_seconds=3600),
}


class RunLimits(_SparseEntry):
    """The runtime's limits: a development Run's total of source contacts.

    Absent by default. It counts the Run's own physical source contacts, so a
    Fork child does not count its parent's.
    """

    run_contacts: int | None = Field(default=None, ge=1, strict=True)


class FailureTuning(_SparseEntry):
    """The runtime's `failure` section: this module's tuning parameters.

    Global values plus scoped entries, the shape of `agent_invocation` (#220).
    Only `policy_for` reads it.
    """

    operation: FailureOverrides | None = None
    model_contact: ModelContactOverrides | None = None
    limits: RunLimits | None = None
    scopes: FailureScopes = Field(default_factory=FailureScopes)


class FailureSurface(FrozenContract):
    """Where a policy applies: a stage or Tool operation, or a role's model contact."""

    scope: FailureScope
    name: str = Field(min_length=1)

    @model_validator(mode="after")
    def known_surface(self):
        _SURFACE_NAMES[self.scope](self.name)
        return self


class _Resolution(FrozenContract):
    """Self-contained evidence: the surface, its layers and their result."""

    surface: FailureSurface

    @field_validator("defaults", "effective", mode="before", check_fields=False)
    @classmethod
    def complete_policy(cls, value, info):
        fields = cls.model_fields[info.field_name].annotation.model_fields
        if isinstance(value, dict) and set(value) != set(fields):
            raise ValueError("recorded failure policy must contain every setting")
        return value

    @model_validator(mode="after")
    def recorded_resolution(self):
        expected = self.defaults.model_dump()
        expected.update(self.global_overrides.model_dump())
        expected.update(self.surface_overrides.model_dump())
        if self.effective.model_dump() != expected:
            raise ValueError("recorded failure policy disagrees with its overrides")
        return self

    def sources(self) -> dict[str, str]:
        """Each setting's layer: ``default``, ``global`` or the surface's scope."""
        return {
            name: self.surface.scope
            if name in self.surface_overrides.model_fields_set
            else "global"
            if name in self.global_overrides.model_fields_set
            else "default"
            for name in type(self.effective).model_fields
        }


class ResolvedFailurePolicy(_Resolution):
    """A stage's or Tool's resolved operation policy, recorded with each failure."""

    defaults: FailurePolicy
    global_overrides: FailureOverrides
    surface_overrides: FailureOverrides
    effective: FailurePolicy

    @model_validator(mode="after")
    def operation_surface(self):
        if self.surface.scope == "role":
            raise ValueError("a role resolves model contact, not an operation")
        return self


class ResolvedModelContactPolicy(_Resolution):
    """A role's resolved model-contact window."""

    defaults: ModelContactPolicy
    global_overrides: ModelContactOverrides
    surface_overrides: ModelContactOverrides
    effective: ModelContactPolicy

    @model_validator(mode="after")
    def role_surface(self):
        if self.surface.scope != "role":
            raise ValueError("model contact resolves per role")
        return self


class ResolvedLimits(FrozenContract):
    """A source Tool's resolved limits, recorded with each breach.

    Self-contained like a resolved policy: it validates against its recorded
    defaults and the Tool's recorded declaration, never current code.
    """

    surface: FailureSurface
    defaults: ContactLimits
    tool_overrides: ContactLimitOverrides
    effective: ContactLimits
    run_contacts: int | None = Field(ge=1, strict=True)

    @field_validator("defaults", "effective", mode="before")
    @classmethod
    def complete_limits(cls, value):
        if isinstance(value, dict) and set(value) != set(ContactLimits.model_fields):
            raise ValueError("recorded limits must contain every setting")
        return value

    @model_validator(mode="after")
    def recorded_resolution(self):
        _source_tool_surface(self.surface)
        expected = self.defaults.model_dump()
        expected.update(self.tool_overrides.model_dump())
        if self.effective.model_dump() != expected:
            raise ValueError("recorded limits disagree with the Tool's declaration")
        return self

    def sources(self) -> dict[str, str]:
        """Each limit's layer: ``default`` or the Tool's own ``tool``."""
        return {
            name: "tool" if name in self.tool_overrides.model_fields_set else "default"
            for name in ContactLimits.model_fields
        }


class ResolvedStageLimits(FrozenContract):
    """A stage's resolved part limits, recorded where its parts are journaled.

    Self-contained like `ResolvedLimits`: it validates against its recorded
    defaults and the stage's recorded declaration, never current code.
    """

    surface: FailureSurface
    defaults: StageLimits
    stage_overrides: StageLimitOverrides
    effective: StageLimits

    @field_validator("defaults", "effective", mode="before")
    @classmethod
    def complete_limits(cls, value):
        if isinstance(value, dict) and set(value) != set(StageLimits.model_fields):
            raise ValueError("recorded limits must contain every setting")
        return value

    @model_validator(mode="after")
    def recorded_resolution(self):
        if self.surface.scope != "stage":
            raise ValueError("part limits resolve per stage")
        expected = self.defaults.model_dump()
        expected.update(self.stage_overrides.model_dump())
        if self.effective.model_dump() != expected:
            raise ValueError("recorded limits disagree with the stage's declaration")
        return self

    def sources(self) -> dict[str, str]:
        """Each limit's layer: ``default`` or the stage's own ``stage``."""
        return {
            name: "stage"
            if name in self.stage_overrides.model_fields_set
            else "default"
            for name in StageLimits.model_fields
        }


def _source_tool_surface(surface: FailureSurface) -> None:
    from bio_neb.agentic_journey.source_contacts import source_tools

    if surface.scope != "tool" or surface.name not in source_tools():
        raise ValueError("limits resolve per source Tool")


_bound_tuning: ContextVar[FailureTuning | None] = ContextVar(
    "run_failure_tuning", default=None
)


@contextmanager
def run_failure_tuning(tuning: FailureTuning) -> Iterator[None]:
    """Bind the Run's recorded failure section for the contact sites of its work."""
    token = _bound_tuning.set(tuning)
    try:
        yield
    finally:
        _bound_tuning.reset(token)


class UnboundOperationPolicy(FrozenContract):
    """Outside a bound operation (a DEV diagnostic, a seam test): the defaults."""

    effective: FailurePolicy = DEFAULT_FAILURE_POLICY


_bound_operation: ContextVar[ResolvedFailurePolicy | None] = ContextVar(
    "operation_failure_policy", default=None
)


@contextmanager
def operation_failure_policy(policy: ResolvedFailurePolicy) -> Iterator[None]:
    """Bind the resolved policy of the stage whose contacts run inside.

    A part transfer (#363) retries each part by it, as its stage would.
    """
    token = _bound_operation.set(policy)
    try:
        yield
    finally:
        _bound_operation.reset(token)


def bound_operation_policy() -> ResolvedFailurePolicy | UnboundOperationPolicy:
    return _bound_operation.get() or UnboundOperationPolicy()


class UnboundStageLimits(FrozenContract):
    """Outside a bound stage (a DEV diagnostic, a seam test): the defaults."""

    effective: StageLimits = StageLimits()


def bound_part_limits() -> ResolvedStageLimits | UnboundStageLimits:
    """The part limits of the stage whose bound policy runs its part transfer."""
    policy = _bound_operation.get()
    if policy is None:
        return UnboundStageLimits()
    if policy.surface.scope != "stage":
        raise ValueError("a part transfer runs under its stage's policy")
    return limits_for(policy.surface)


def policy_for(
    surface: FailureSurface, tuning: FailureTuning | None = None
) -> ResolvedFailurePolicy | ResolvedModelContactPolicy:
    """The one resolution: defaults, the global value, then the surface's entry.

    The narrowest layer wins. A role resolves its model contact; a stage or Tool
    its operation policy. Without ``tuning``, the bound Run's section applies;
    outside a Run's work (a seam test or devtools harness) no entry applies.
    """
    if tuning is None:
        tuning = _bound_tuning.get() or FailureTuning()
    entry = getattr(tuning.scopes, surface.scope).get(surface.name)
    if surface.scope == "role":
        owner, defaults, field = (
            ResolvedModelContactPolicy,
            ModelContactPolicy(),
            "model_contact",
        )
        sparse = ModelContactOverrides
    else:
        owner, defaults, field = (
            ResolvedFailurePolicy,
            DEFAULT_FAILURE_POLICY,
            "operation",
        )
        sparse = FailureOverrides
    layers = [getattr(tuning, field), None if entry is None else getattr(entry, field)]
    global_overrides, surface_overrides = (layer or sparse() for layer in layers)
    values = defaults.model_dump()
    values.update(global_overrides.model_dump())
    values.update(surface_overrides.model_dump())
    return owner(
        surface=surface,
        defaults=defaults,
        global_overrides=global_overrides,
        surface_overrides=surface_overrides,
        effective=type(defaults).model_validate(values),
    )


def limits_for(
    surface: FailureSurface, tuning: FailureTuning | None = None
) -> ResolvedLimits | ResolvedStageLimits:
    """A surface's limits: the defaults, then the surface's own declaration.

    Both layers are code (owners, #416 and #418). A source Tool's runtime adds
    only the Run total of ``failure.limits``; a stage's part limits have no
    runtime layer. Without ``tuning``, the bound Run's section applies.
    """
    from bio_neb.agentic_journey.source_contacts import declared_limits

    if surface.scope == "stage":
        stage_defaults = StageLimits()
        declared = STAGE_LIMITS.get(surface.name, StageLimitOverrides())
        return ResolvedStageLimits(
            surface=surface,
            defaults=stage_defaults,
            stage_overrides=declared,
            effective=StageLimits.model_validate(
                stage_defaults.model_dump() | declared.model_dump()
            ),
        )
    if surface.scope != "tool":
        raise ValueError("limits resolve per source Tool or stage")
    _source_tool_surface(surface)
    if tuning is None:
        tuning = _bound_tuning.get() or FailureTuning()
    defaults = ContactLimits()
    overrides = declared_limits(surface.name)
    values = defaults.model_dump()
    values.update(overrides.model_dump())
    return ResolvedLimits(
        surface=surface,
        defaults=defaults,
        tool_overrides=overrides,
        effective=ContactLimits.model_validate(values),
        run_contacts=None if tuning.limits is None else tuning.limits.run_contacts,
    )
