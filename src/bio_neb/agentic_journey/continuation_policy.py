"""Configured correction allowances, separate from ordinary permitted work."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from bio_neb.agentic_journey.injection import (
    InjectionPolicy,
    InjectionTunings,
    resolve_injection,
)
from bio_neb.agentic_journey.model_settings import (
    AgentModelSettings,
    ModelSettingsSelection,
    RequestSettingName,
)
from bio_neb.agentic_journey.role_declaration import (
    ROLE_DEFAULT,
    ROLES,
    AgentRole,
    output_tool_names,
)
from bio_neb.agentic_journey.role_declaration import role as declared_role
from bio_neb.failure import (
    ModelContactPolicy,
    ModelContactSetting,
    ResolvedModelContactPolicy,
)
from bio_neb.journey import canonical_json_digest


class ContinuationPolicy(BaseModel):
    """One Attempt's failure rule, retained with its start (owner, #447 B).

    A failed turn is a reply in which nothing worked: every call refused or
    repeated, a text or reasoning-only ending, a cut-off or a refused output. A
    reply in which a Tool ran or an output was valid resets the count. The
    ``failed_turns``-th failed turn in a row stops the Attempt for the Agent work
    Review; a stage a scope tunes counts its own requests' replies against its
    own number. One rule for every role.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    failed_turns: StrictInt = Field(ge=1)
    # Empty unless a stage scope tunes one: the recorded policy then has no key.
    stage_failed_turns: dict[str, Annotated[StrictInt, Field(ge=1)]] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )

    def limit(self, stage: str | None) -> int:
        """The failed turn in a row that stops a reply to a request in ``stage``."""
        return self.stage_failed_turns.get(stage, self.failed_turns)

    @property
    def most_failed_turns(self) -> int:
        """The largest limit any of the Attempt's stages has."""
        return max((self.failed_turns, *self.stage_failed_turns.values()))

    @property
    def policy_id(self) -> str:
        return "continuation-policy:" + canonical_json_digest(
            self.model_dump(mode="json")
        )


InvestigationCompletion = Literal[
    "allow_partial_with_limitations", "require_terminal_members"
]


class InvestigationWorkPolicy(BaseModel):
    """Separate existing item, sibling and completion meanings; no new recovery.

    An operational fault's route is the failure module's (``failure.policy_for``
    on its Investigation stage surface): it retries, asks or holds the Run (R2,
    #105). Completion waits for every member unless the researcher amends it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    item_disposition: Literal["retain_classified_outcome"] = "retain_classified_outcome"
    other_work: Literal["continue_eligible"] = "continue_eligible"
    unresolved_identity: Literal["block_finalization"] = "block_finalization"
    completion: InvestigationCompletion = "require_terminal_members"


def resolve_investigation_work_policy(
    defaults: InvestigationWorkPolicy, completion: InvestigationCompletion | None = None
) -> InvestigationWorkPolicy:
    return InvestigationWorkPolicy(
        **{
            **defaults.model_dump(),
            **({"completion": completion} if completion else {}),
        }
    )


class AgentExhaustionPolicy(BaseModel):
    """Existing, distinct outcomes; configuration grants no new eligibility."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    failed_turns: Literal["review"] = "review"
    model_availability: Literal["review"] = "review"


class RegisteredExhaustionPolicy(BaseModel):
    """The registered baseline is fixed; role applicability grants no new eligibility."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    singleton_action: Literal["review"] = "review"
    model_availability: Literal["review"] = "review"


ModelReference = Annotated[
    str, StringConstraints(strict=True, pattern=r"^[a-z][a-z0-9_-]*$")
]


def _registered_stage(name: str) -> str:
    # The role registry owns stages; it imports this module, so look it up late.
    if not any(name in item.stages for item in ROLES.values()):
        raise ValueError("not a registered Agent stage")
    return name


def _registered_tool(name: str) -> str:
    from bio_neb.agentic_journey.assignment_context import SEARCH_TOOL_NAME
    from bio_neb.agentic_journey.tool_execution import function_tool_names

    if name not in function_tool_names() | output_tool_names() | {SEARCH_TOOL_NAME}:
        raise ValueError("not a registered Tool")
    return name


StageName = Annotated[str, AfterValidator(_registered_stage)]
ToolName = Annotated[str, AfterValidator(_registered_tool)]
ToolChoice = Literal["required", "auto"] | ToolName
"""Force a call, leave it to the model, or force one named Tool. A forced value
needs its model's recorded grammar support (``model_settings.forces_a_call``)."""


class ScopedInvocation(BaseModel):
    """One scope's model overrides; an omitted field keeps the wider scope's value."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model_reference: ModelReference | None = None
    model_settings: AgentModelSettings | None = None
    model_requests: StrictInt | None = Field(default=None, gt=0)
    # The failure rule's number for this scope's requests.
    failed_turns: StrictInt | None = Field(default=None, ge=1)
    tool_choice: ToolChoice | None = None
    injection: InjectionTunings | None = None

    @model_validator(mode="after")
    def explicit_fields_are_values(self) -> "ScopedInvocation":
        if any(getattr(self, name) is None for name in self.model_fields_set):
            raise ValueError("explicit scoped settings cannot be null")
        return self

    @model_serializer
    def explicit_settings(self) -> dict[str, Any]:
        # Registration keeps omissions as omissions, including nested settings.
        return {
            name: value.model_dump(mode="json", exclude_unset=True)
            if isinstance(value := getattr(self, name), BaseModel)
            else value
            for name in self.model_fields_set
        }


_UNBUILT = "this scope cannot set it until its seam is built"


class StageInvocation(ScopedInvocation):
    """A stage's requests take its settings and its failure rule's number; a
    per-stage model or budget is unbuilt."""

    @field_validator("model_reference", "model_requests")
    @classmethod
    def unbuilt(cls, value: Any) -> Any:
        raise ValueError(_UNBUILT)


class ToolInvocation(ScopedInvocation):
    """A request does not know which Tool the model will call: nothing is built."""

    @field_validator(
        "model_reference",
        "model_settings",
        "model_requests",
        "failed_turns",
        "tool_choice",
        "injection",
    )
    @classmethod
    def unbuilt(cls, value: Any) -> Any:
        raise ValueError(_UNBUILT)


class InvocationScopes(BaseModel):
    """Overrides narrower than the global ones, keyed by scope and then name."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    role: dict[AgentRole, ScopedInvocation] = Field(default_factory=dict)
    stage: dict[StageName, StageInvocation] = Field(default_factory=dict)
    tool: dict[ToolName, ToolInvocation] = Field(default_factory=dict)


SettingSource = Literal["model_default", "global", "role", "stage"]
ScopedSettingName = RequestSettingName | Literal["tool_choice"]


class ResolvedModelSettings(BaseModel):
    """Request settings, the stage's Tool choice and the scope each value came from.

    An unset ``tool_choice`` keeps the default rule; it is resolved per stage only,
    because whether a named Tool is offered depends on the stage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model_settings: AgentModelSettings
    tool_choice: ToolChoice | None = None
    settings_sources: dict[ScopedSettingName, SettingSource]


def resolve_model_settings(
    selection: ModelSettingsSelection,
    layers: Iterable[
        tuple[SettingSource, AgentModelSettings | None, ToolChoice | None]
    ],
) -> ResolvedModelSettings:
    """The model's defaults, then each layer's explicit values; the narrowest wins.

    A setting the model leaves unset and no scope names has no source, so a setting
    added later adds no identity material until someone sets it.
    """
    values = selection.baseline.model_dump()
    sources: dict[str, SettingSource] = {
        name: "model_default" for name, value in values.items() if value is not None
    }
    tool_choice = None
    for source, settings, choice in layers:
        if choice is not None:
            tool_choice = choice
            sources["tool_choice"] = source
        if settings is None:
            continue
        for name, value in settings.model_dump(exclude_unset=True).items():
            values[name] = value
            sources[name] = source
    return ResolvedModelSettings(
        model_settings=AgentModelSettings(**values),
        tool_choice=tool_choice,
        settings_sources=sources,
    )


class _AgentInvocationControls(BaseModel):
    """Registration defaults alongside the existing per-role correction owner.

    Nonzero whole-loop or physical Tool retries and disabled mandatory gates are
    unsupported. Model contact is the failure module's tuning (#223).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    loop_retry: StrictInt = Field(default=0, ge=0, le=0)
    tool_retry: StrictInt = Field(default=0, ge=0, le=0)
    exhaustion: RegisteredExhaustionPolicy = Field(
        default_factory=RegisteredExhaustionPolicy
    )
    model_settings: AgentModelSettings = Field(default_factory=AgentModelSettings)
    model_requests: StrictInt = Field(default=ROLE_DEFAULT.model_requests, gt=0)


class AgentInvocationDefaults(_AgentInvocationControls):
    """One global selection with scoped overrides, merged once per Attempt.

    One failure rule serves every role (owner, #447 B): its number is the global
    role default's ``failed_turns`` (``roles.ROLE_DEFAULT``), then the selected
    model entry's own if it states one, then this file's, then the role scope's;
    a stage scope tunes its own requests. ``model_requests`` is the role default's
    per-Attempt budget; a role scope overrides it.
    """

    # The Run's own number; unset, the model entry's or the role default's.
    failed_turns: StrictInt | None = Field(default=None, ge=1)
    model_reference: ModelReference
    tool_choice: ToolChoice | None = None
    # Sparse pattern entries for every role; unset, a registration records no key.
    injection: InjectionTunings | None = None
    scopes: InvocationScopes = Field(default_factory=InvocationScopes)

    @field_serializer("model_settings")
    def explicit_model_settings(self, value: AgentModelSettings) -> dict[str, Any]:
        # Registration retains omissions as omissions, so a roundtrip still inherits
        # the selected model baseline rather than manufacturing explicit None values.
        return value.model_dump(mode="json", exclude_unset=True)

    @model_serializer(mode="wrap")
    def explicit_tool_choice(self, handler) -> dict[str, Any]:
        # An unset Tool choice keeps the default rule and is registered as omitted;
        # so is an unset injection entry.
        values = handler(self)
        for name in ("tool_choice", "injection", "failed_turns"):
            if getattr(self, name) is None:
                values.pop(name, None)
        return values

    def role_reference(self, agent_role: str) -> str:
        entry = self.scopes.role.get(agent_role)
        if entry is not None and entry.model_reference is not None:
            return entry.model_reference
        return self.model_reference

    def resolved_settings(
        self, selection: ModelSettingsSelection, *, agent_role: str
    ) -> tuple[ResolvedModelSettings, dict[str, ResolvedModelSettings]]:
        """The role's settings and, for each of its stages that any scope tunes, theirs.

        A configured ``tool_choice`` at any scope tunes every stage it reaches.
        """
        role = self.scopes.role.get(agent_role)
        layers: list[
            tuple[SettingSource, AgentModelSettings | None, ToolChoice | None]
        ] = [
            ("global", self.model_settings, self.tool_choice),
            (
                "role",
                None if role is None else role.model_settings,
                None if role is None else role.tool_choice,
            ),
        ]
        wider_choice = any(choice is not None for _, _, choice in layers)
        stages = {}
        for name in declared_role(agent_role).stages:
            entry = self.scopes.stage.get(name) or StageInvocation()
            if (
                entry.model_settings is None
                and entry.tool_choice is None
                and not wider_choice
            ):
                continue
            stages[name] = resolve_model_settings(
                selection,
                [*layers, ("stage", entry.model_settings, entry.tool_choice)],
            )
        role_settings = resolve_model_settings(
            selection, [(source, settings, None) for source, settings, _ in layers]
        )
        return role_settings, stages

    def resolved_injection(
        self, selection: ModelSettingsSelection, *, agent_role: str
    ) -> InjectionPolicy | None:
        """The role's switched-on patterns and, for each stage a scope tunes, theirs."""
        role = self.scopes.role.get(agent_role)
        return resolve_injection(
            selection.injection_patterns,
            self.injection,
            None if role is None else role.injection,
            {
                name: entry.injection
                for name in declared_role(agent_role).stages
                if (entry := self.scopes.stage.get(name)) is not None
            },
        )


class AgentInvocationPolicy(_AgentInvocationControls):
    """Secret-free, resolved policy for exactly one role's Attempt.

    ``model_settings`` serve every request whose stage has no settings of its
    own; ``stage_settings`` serve the rest. Each records its values' sources.
    ``model_contact`` is the failure module's resolution for the role.
    """

    agent_role: AgentRole
    model_reference: ModelReference
    settings_selection: ModelSettingsSelection
    settings_sources: dict[RequestSettingName, SettingSource]
    model_contact: ModelContactPolicy
    model_contact_sources: dict[
        ModelContactSetting, Literal["default", "global", "role"]
    ]
    stage_settings: dict[StageName, ResolvedModelSettings]
    continuation: ContinuationPolicy
    exhaustion: AgentExhaustionPolicy
    # None when no request switches a pattern on: the recorded policy and its
    # identity then carry no key (ADR 0048).
    injection: InjectionPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def preserve_role_eligibility(self) -> "AgentInvocationPolicy":
        for settings in (
            self.model_settings,
            *(item.model_settings for item in self.stage_settings.values()),
        ):
            self.settings_selection.validate_settings(settings.request_settings())
        expected_exhaustion = AgentExhaustionPolicy()
        if self.exhaustion != expected_exhaustion:
            raise ValueError("role exhaustion eligibility cannot change")
        return self

    def stage_model_settings(self, stage: str | None) -> AgentModelSettings:
        """The settings a request in this stage carries."""
        entry = self.stage_settings.get(stage) if stage is not None else None
        return self.model_settings if entry is None else entry.model_settings

    def stage_tool_choice(self, stage: str | None) -> ToolChoice | None:
        """The stage's configured Tool choice; None keeps the default rule."""
        entry = self.stage_settings.get(stage) if stage is not None else None
        return None if entry is None else entry.tool_choice

    @property
    def policy_id(self) -> str:
        # None means an omitted setting, so a setting added later with no value
        # leaves every recorded identity unchanged.
        return "invocation-policy:" + canonical_json_digest(
            self.model_dump(mode="json", exclude_none=True)
        )

    @property
    def framework_retry_allowance(self) -> int:
        # Pydantic AI charges every correction to its global output counter. Each
        # follows a counted response, so this never binds before the request
        # budget or the domain allowances.
        return self.model_requests


class AgentInvocationOverrides(BaseModel):
    """An Attempt's recorded Review amendments; model choices come only from config."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    # A work Review's "continue" may grant the next Attempt another number.
    failed_turns: StrictInt | None = Field(default=None, ge=1)
    loop_retry: StrictInt | None = Field(default=None, ge=0, le=0)
    tool_retry: StrictInt | None = Field(default=None, ge=0, le=0)
    exhaustion: AgentExhaustionPolicy | None = None
    # The narrowest layer, after every config scope: an effort Review's extension.
    model_requests: StrictInt | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def explicit_fields_are_values(self) -> "AgentInvocationOverrides":
        if any(getattr(self, name) is None for name in self.model_fields_set):
            raise ValueError("explicit invocation settings cannot be null")
        return self

    @model_serializer
    def explicit_settings(self) -> dict[str, Any]:
        # A command roundtrip must preserve omission, including nested settings.
        return {
            name: value.model_dump(mode="json", exclude_unset=True)
            if isinstance(value := getattr(self, name), BaseModel)
            else value
            for name in self.model_fields_set
        }


def resolve_agent_invocation_policy(
    defaults: AgentInvocationDefaults,
    overrides: AgentInvocationOverrides,
    *,
    agent_role: str,
    model_settings: Mapping[str, ModelSettingsSelection],
    model_contact: ResolvedModelContactPolicy,
) -> AgentInvocationPolicy:
    """The one merge: model default, global, role, stage, then the Attempt's amendments.

    The failure rule's number is the role default's, then the Run's (which a fresh
    configuration fills from the model entry when it states one), then the role
    scope's; each stage scope that names one tunes its own requests, and a work
    Review's grant replaces them all for the next Attempt. Model contact is
    resolved by the failure module and recorded here as given.
    """
    values = defaults.model_dump(
        mode="json", exclude={"scopes", "tool_choice", "injection", "failed_turns"}
    )
    role = defaults.scopes.role.get(agent_role)
    values["model_reference"] = defaults.role_reference(agent_role)
    try:
        selected = model_settings[values["model_reference"]]
    except KeyError as error:
        raise ValueError(
            "invocation references an unknown model roster entry"
        ) from error
    failed_turns = ROLE_DEFAULT.failed_turns
    for layer in (defaults.failed_turns, None if role is None else role.failed_turns):
        if layer is not None:
            failed_turns = layer
    correction = {
        "failed_turns": failed_turns,
        "stage_failed_turns": {
            name: entry.failed_turns
            for name in declared_role(agent_role).stages
            if (entry := defaults.scopes.stage.get(name)) is not None
            and entry.failed_turns is not None
        },
    }
    if overrides.failed_turns is not None:
        correction = {"failed_turns": overrides.failed_turns}
    if role is not None and role.model_requests is not None:
        values["model_requests"] = role.model_requests
    resolved, stages = defaults.resolved_settings(selected, agent_role=agent_role)
    values["settings_selection"] = selected
    values["model_settings"] = resolved.model_settings
    values["settings_sources"] = resolved.settings_sources
    values["stage_settings"] = stages
    if (
        injection := defaults.resolved_injection(selected, agent_role=agent_role)
    ) is not None:
        values["injection"] = injection
    values["exhaustion"] = AgentExhaustionPolicy().model_dump(mode="json")
    if model_contact.surface.name != agent_role:
        raise ValueError("model contact was resolved for another role")
    values["model_contact"] = model_contact.effective
    values["model_contact_sources"] = model_contact.sources()
    for name, value in overrides.explicit_settings().items():
        if name == "failed_turns":
            continue
        if isinstance(value, dict):
            values[name].update(value)
        else:
            values[name] = value
    return AgentInvocationPolicy(
        **values, continuation=correction, agent_role=agent_role
    )


class MandatoryReviewPolicy(BaseModel):
    """Stage selection requires exact authority independently of Agent exhaustion."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    approval: Literal["mandatory"] = "mandatory"


class ExecutionReviewPolicy(MandatoryReviewPolicy):
    stage: Literal["execution"] = "execution"


class InvestigationReviewPolicy(MandatoryReviewPolicy):
    stage: Literal["investigation"] = "investigation"


class FinalizationReviewPolicy(MandatoryReviewPolicy):
    stage: Literal["finalization"] = "finalization"


def explicit_review_policy(value: Any) -> Any:
    """Construction may use typed defaults; retained data must state every field."""
    if isinstance(value, Mapping) and set(value) != {"stage", "approval"}:
        raise ValueError("retained mandatory Review policy must be complete")
    return value


TEXT_ENDING_KEY = "text_ending"
REASONING_ONLY_KEY = "reasoning_only"
# A reply with no call keys on its own kind: text, or reasoning alone.
_CALLLESS_KEYS = {
    "text_refused": TEXT_ENDING_KEY,
    "reasoning_refused": REASONING_ONLY_KEY,
}
TRUNCATION_KEY = "output_truncated"
# A routed reply's faults (ADR 0044): copies collapsed, unoffered calls refused,
# an output refused beside calls that run, and resends after the provider timed out.
COPIES_KEY = "copied_calls"
UNOFFERED_KEY = "unoffered_calls"
OUTPUT_ALONE_KEY = "output_not_alone"
RESEND_KEY = "model_resend"
# A Tool that could not produce its fact: its fault key follows.
TOOL_REFUSED_KEY = "tool_refused: "
# A request the model endpoint refused, shown to the model: the failure
# module's key for its status and body follows.
PROVIDER_REJECTED_KEY = "provider_rejected: "
# Each fact that shows the model a fault: its key's prefix and the words shown.
_SHOWN_FAULTS = {
    "ModelRequestRejected": (PROVIDER_REJECTED_KEY, "provider_statement"),
    "ToolExecutionRefused": (TOOL_REFUSED_KEY, "statement"),
}
# A refused output: one more failed turn when nothing else of its reply worked.
OUTPUT_KEY = "output_refused"


def correction_key(
    tool_name: str, errors: Iterable[str], arguments: Mapping[str, Any]
) -> str:
    """The same failing call: the same Tool with the same effective arguments,
    refused for the same recorded reasons.

    A different call refused for the same reasons has its own key, so it never
    spends another call's allowance.
    """
    call = canonical_json_digest({"tool_name": tool_name, "arguments": arguments})
    return f"{tool_name}: {', '.join(errors)} @ {call[:16]}"


class ModelOutputFault(NamedTuple):
    """One model-output fault and its reply's place in the failed turns in a row.

    ``event_id`` is the fact that records it: the refused or text-ending turn's
    disposition, the truncation, the routed reply naming its faults or the refused
    output. A refused call also names its validation. Every fault of one reply has
    that reply's ``count``: one more than the failed turns in a row before it.
    ``failed`` says the reply was a failed turn (nothing of it worked); ``stage``
    is its request's.
    """

    event_id: str
    validation_event_id: str | None
    key: str
    count: int
    feedback: str | None
    stage: str | None = None
    failed: bool = False

    def corrected(self, policy: ContinuationPolicy) -> bool:
        """Whether the Attempt continues after this fault's reply."""
        if self.key == RESEND_KEY:
            return True
        return not (self.failed and self.count >= policy.limit(self.stage))


# A refused call (alone or one of several in its response), a text ending or a
# reasoning-only reply.
_REFUSED_TURNS = frozenset({"singleton_refused", "multi_call_refused", *_CALLLESS_KEYS})


class _Reply:
    """One reply's outcome so far: its request's stage, a fault, and progress."""

    def __init__(self, stage: str | None, count: int) -> None:
        self.stage = stage
        self.count = count
        self.faults: list[ModelOutputFault] = []
        self.faulted = False
        self.output_valid = False
        # Admitted calls, and those of them whose Tool refused.
        self.admitted = 0
        self.refused = 0

    @property
    def progress(self) -> bool:
        return self.output_valid or self.admitted > self.refused

    @property
    def failed(self) -> bool:
        return self.faulted and not self.progress


def _refused_turn_key(payload: Mapping[str, Any]) -> tuple[str | None, str]:
    """A refused turn's validation and fault key, as admission retained them;
    reopen never re-decides the key."""
    key = payload.get("correction_key")
    callless = _CALLLESS_KEYS.get(payload["disposition"])
    if key != callless if callless is not None else key in _CALLLESS_KEYS.values():
        raise ValueError("model-output fault changed its correction key")
    if callless is not None:
        return None, key
    (validation_id,) = payload["validation_event_ids"]
    return validation_id, key


def _replies(events: Iterable[Mapping[str, Any]], *, attempt_id: str) -> list[_Reply]:
    """The Attempt's replies in order, across its provider continuations and
    members, each with the place it takes in the failed turns in a row.

    A reply is everything after one request until the next. An admitted call
    (it runs; a failed run is the operational Review's) or a valid output is
    progress and resets the count; a reply with a fault and no progress is a
    failed turn. A resend is the provider's, not a failed turn. Only Journey facts
    decide, so the Review's own check reads the same count as the live loop.
    """
    from bio_neb.agentic_journey.model_availability import prior_model_attempt_ids

    events = list(events)
    owned = {*prior_model_attempt_ids(events, attempt_id), attempt_id}
    replies: list[_Reply] = []
    in_a_row = 0

    def fault(event, validation_id, key, count, feedback):
        reply = replies[-1]
        if key != RESEND_KEY:
            reply.faulted = True
            if count is not None and count != reply.count:
                raise ValueError("model-output fault changed its correction count")
        reply.faults.append(
            ModelOutputFault(
                event["event_id"],
                validation_id,
                key,
                # A refused output records no count of its own: it takes its reply's.
                reply.count if count is None else count,
                feedback,
                reply.stage,
            )
        )

    for event in events:
        payload = event.get("payload", {})
        if payload.get("attempt_id") not in owned:
            continue
        kind = event["event_type"]
        if kind == "ModelRequestRecorded":
            if replies:
                in_a_row = (
                    0
                    if replies[-1].progress
                    else in_a_row + 1
                    if replies[-1].failed
                    else in_a_row
                )
            replies.append(_Reply(payload.get("stage"), in_a_row + 1))
        elif not replies:
            continue
        elif kind == "ModelCompletionTruncated":
            fault(
                event,
                None,
                TRUNCATION_KEY,
                payload.get("correction_count"),
                payload.get("feedback"),
            )
        elif kind == "AgentTurnRouted":
            # The cut-off counts on its own truncation fact.
            for item in payload["faults"]:
                if item["key"] != TRUNCATION_KEY:
                    fault(event, None, item["key"], item["count"], payload["feedback"])
        elif kind == "AgentActionTurnDispositionRecorded" and payload[
            "disposition"
        ] in {"singleton_admitted", "multi_call_admitted"}:
            replies[-1].admitted += 1
        elif kind in _SHOWN_FAULTS:
            # The endpoint refused the request, or a Tool could not produce its
            # fact: the model was shown why (C8).
            prefix, words = _SHOWN_FAULTS[kind]
            replies[-1].refused += kind == "ToolExecutionRefused"
            fault(event, None, prefix + payload["fault_key"], None, payload[words])
        elif (
            kind == "AgentActionTurnDispositionRecorded"
            and payload["disposition"] in _REFUSED_TURNS
        ):
            fault(
                event,
                *_refused_turn_key(payload),
                payload.get("correction_count"),
                payload.get("feedback"),
            )
        elif kind == "AgentOutputValidated":
            if payload["status"] == "valid":
                replies[-1].output_valid = True
            else:
                fault(event, None, OUTPUT_KEY, None, payload.get("feedback"))
    return replies


def model_output_faults(
    events: Iterable[Mapping[str, Any]], *, attempt_id: str
) -> tuple[ModelOutputFault, ...]:
    """The Attempt's model-output faults in order, each with its reply's place in
    the failed turns in a row and whether that reply failed.

    One rule for every role (owner, #447 B): a refused call keys on its Tool,
    arguments and exact errors (``correction_key``, recorded at admission), a
    repeat on its call identity, a text ending, a reasoning-only reply, a cut-off
    completion and a refused output each have their own key, and a routed reply
    names its copies and unoffered calls once each. The key names the fault for
    its correction; the count is the reply's, and each recorded count must be the
    one its reply takes. A resend counts the sends of its own request.
    """
    return tuple(
        fault._replace(failed=reply.failed)
        for reply in _replies(events, attempt_id=attempt_id)
        for fault in reply.faults
    )


def failed_turns_spent(
    events: Iterable[Mapping[str, Any]],
    *,
    attempt_id: str,
    policy: ContinuationPolicy,
    key: str | None = None,
) -> bool:
    """Whether the Attempt's last fault is a failed turn at its limit; with
    ``key``, a fault of that kind."""
    faults = model_output_faults(events, attempt_id=attempt_id)
    return (
        bool(faults)
        and not faults[-1].corrected(policy)
        and key in {None, faults[-1].key}
    )


def current_turn(
    events: Iterable[Mapping[str, Any]], *, attempt_id: str, policy: ContinuationPolicy
) -> tuple[int, int]:
    """The current reply's place in the failed turns in a row, if it fails, and the
    place that stops the Attempt in its request's stage. A correction written now
    names both."""
    replies = _replies(events, attempt_id=attempt_id)
    if not replies:
        return 1, policy.failed_turns
    return replies[-1].count, policy.limit(replies[-1].stage)


class _ResponsePermission:
    """Whether a multi-call response permits the next request after a call's decision.

    Its calls are admitted one by one, each just before it runs: an admitted call
    waits for its execution, a corrected refusal leaves the earlier progress
    standing, and a refused call beyond its allowance stops the Attempt.
    """

    def __init__(self) -> None:
        self._response_event_id: str | None = None
        self._exhausted = False

    def decide(
        self, payload: Mapping[str, Any], corrected: bool | None, permitted: bool
    ) -> bool:
        first = payload["response_event_id"] != self._response_event_id
        if first:
            self._response_event_id = payload["response_event_id"]
            self._exhausted = False
        if payload["disposition"] == "multi_call_admitted":
            return False
        if not corrected:
            self._exhausted = True
        return not self._exhausted and (first or permitted)


def _routed_permission(payload, corrected, permitted):
    """A routed reply's faults beyond their allowance stop the Attempt; when
    nothing of the reply runs, its note is the next request's reason."""
    if not corrected:
        return False
    return True if not payload["run_order"] else permitted


def _action_permission(payload, corrected, permitted, response):
    disposition = payload["disposition"]
    if disposition.startswith("multi_call_"):
        return response.decide(payload, corrected, permitted)
    if disposition == "singleton_admitted":
        return False
    if disposition in {"singleton_refused", *_CALLLESS_KEYS}:
        return corrected
    return permitted


def validate_continuation(  # noqa: C901 - one ordered reader of every continuation decision
    events: Iterable[Mapping[str, Any]],
    *,
    attempt_id: str,
    policy: ContinuationPolicy,
    next_request: bool = False,
    member_id: str | None = None,
) -> None:
    """Check mechanical progress and correction allowances for every assignment.

    Both runtime and reopen use the Attempt's binding and completed observed Tool
    receipts. Domain event kinds are neither progress nor repeat entitlements.
    Their scientific meaning and exact call/result relationships are checked by
    the domain and shared Tool readers, independently of this allowance decision.
    One failure rule decides every fault (owner, #447 B): the reply that reaches
    its stage's ``failed_turns`` in a row stops the Attempt before the next
    request (``AgentFailedTurns``); provider/effort continuations keep the count.
    """
    from bio_neb.agentic_journey.model_availability import prior_model_attempt_ids

    events = list(events)
    owned = {*prior_model_attempt_ids(events, attempt_id), attempt_id}
    corrected: dict[str, bool] = {}
    for fault in model_output_faults(events, attempt_id=attempt_id):
        # A routed reply names several faults; it continues only if each does.
        corrected[fault.event_id] = corrected.get(
            fault.event_id, True
        ) and fault.corrected(policy)
    # A cut-off counts on its truncation fact; the routing of that reply, which
    # follows it, continues only if the truncation does.
    cut_off = {
        event["payload"]["response_event_id"]: corrected[event["event_id"]]
        for event in events
        if event.get("event_type") == "ModelCompletionTruncated"
        and event["event_id"] in corrected
    }
    permitted = True
    held_member = None
    # The fault whose failed turn reached its limit, while nothing came after it.
    spent_at = None
    # Several outputs in one reply without their correction stop at once.
    uncorrected_outputs = False
    response = _ResponsePermission()
    bound_attempts = set()
    # The stall the next request must name, and the refusal.
    stall = None
    rejection = None
    for event in events:
        payload = event.get("payload", {})
        owner = payload.get("attempt_id")
        if owner not in owned:
            continue
        kind = event.get("event_type")
        if kind == "AttemptStarted":
            if (
                owner in bound_attempts
                or payload.get("continuation_policy") != policy.model_dump(mode="json")
                or payload.get("continuation_policy_id") != policy.policy_id
            ):
                raise ValueError("continuation changed its Attempt policy binding")
            bound_attempts.add(owner)
            permitted = True
        elif kind == "ModelRequestRecorded":
            if held_member is not None and payload.get("member_id") not in {
                None,
                held_member,
            }:
                permitted = True
                held_member = None
            if owner not in bound_attempts or not permitted:
                raise ValueError(
                    "continuation lacks useful progress or eligible correction"
                )
            if payload.get("after_stall_event_id") != stall:
                raise ValueError("a request after a stall names exactly that stall")
            if payload.get("after_rejection_event_id") != (
                None if rejection is None else rejection["event_id"]
            ) or payload.get("after_rejection_fault_key") != (
                None if rejection is None else rejection["payload"]["fault_key"]
            ):
                raise ValueError("a request after a refusal names exactly that refusal")
            permitted, stall, rejection = False, None, None
        elif kind == "ModelRequestStalled":
            # No reply arrived: the next request carries the stall note.
            permitted, stall = True, event["event_id"]
        elif kind == "ModelRequestRejected":
            # The endpoint refused: the next request carries its words, within
            # the failed turns.
            permitted, rejection = corrected[event["event_id"]], event
        elif kind == "ModelCompletionTruncated":
            permitted = corrected[event["event_id"]]
        elif kind == "AgentTurnRouted":
            permitted = _routed_permission(
                payload,
                corrected.get(event["event_id"], True)
                and cut_off.get(payload["response_event_id"], True),
                permitted,
            )
        elif kind == "ObservedToolExecutionSucceeded":
            permitted = True
        elif kind == "ToolExecutionRefused":
            # Shown to the model: the Attempt goes on within its failed turns.
            permitted = corrected[event["event_id"]]
        elif kind == "AgentOutputValidated":
            if payload["status"] == "valid":
                permitted = True
            else:
                # Several outputs in one reply stop at once unless they carry
                # their correction; then they are one more failed turn.
                uncorrected_outputs = (
                    "multiple_output_calls" in payload["errors"]
                    and payload.get("feedback") is None
                )
                permitted = corrected[event["event_id"]] and not uncorrected_outputs
        elif kind == "AgentActionTurnDispositionRecorded":
            permitted = _action_permission(
                payload, corrected.get(event["event_id"]), permitted, response
            )
        elif (
            kind in {"AgentFailureReviewRequested", "ModelAvailabilityReviewRequested"}
            and payload.get("agent_role") == "investigation_planner"
        ):
            held_member = payload.get("member_id")
        elif kind in {
            "ModelRequestFailed",
            "ToolExecutionFailed",
            "ObservedToolExecutionFailed",
            "AttemptSucceeded",
            "AttemptFailed",
            "AttemptStopped",
            "AttemptCancelled",
        }:
            permitted = False
        if event["event_id"] in corrected and not corrected[event["event_id"]]:
            spent_at = event["event_id"]
        elif permitted:
            spent_at = None
            uncorrected_outputs = False
    if next_request:
        _refuse_next_request(
            bound=attempt_id in bound_attempts,
            permitted=permitted
            or (held_member is not None and member_id not in {None, held_member}),
            spent_at=spent_at,
            uncorrected_outputs=uncorrected_outputs,
        )


def _refuse_next_request(*, bound, permitted, spent_at, uncorrected_outputs=False):
    """Refuse a request the retained decisions do not permit.

    A failed turn at its limit is the typed stop that ends the Attempt for
    review, and so are several outputs in one reply without their correction;
    anything else lacks useful progress.
    """
    if bound and uncorrected_outputs and not permitted and spent_at is None:
        from bio_neb.agentic_journey.agent_work_review import AgentResultMissing

        raise AgentResultMissing("several outputs in one reply without a correction")
    if bound and spent_at is not None and not permitted:
        from bio_neb.agentic_journey.agent_work_review import AgentFailedTurns

        raise AgentFailedTurns(
            "the model's replies failed their limit of turns in a row", spent_at
        )
    if not bound or not permitted:
        from bio_neb.failure import Invariant

        raise Invariant(
            "continuation_lacks_progress",
            "continuation lacks useful progress or eligible correction",
        )
