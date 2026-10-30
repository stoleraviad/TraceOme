"""The Model endpoint policy: everything model-specific behind one seam.

Each model has one adapter that owns its request defaults, its accepted settings,
the translation of each setting to its wire fields, its Tool schema projection and
its correction wording. A model family's sizes share one profile and state only
their differences (the Nemotron family, #240). These are product
configuration assumptions, not provider capability discovery. A Run registers the
adapter's secret-free settings snapshot, which survives later adapter edits
without reinterpreting the Run. Roles, Tools, stages, budgets and guards stay in
the shared Agent module; override scopes resolve in ``continuation_policy``.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Literal, Mapping, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_serializer,
    model_validator,
)

from bio_neb.agentic_journey import injection_patterns
from bio_neb.agentic_journey.injection import PatternName
from bio_neb.failure import CorrectionWording

ReasoningMode = Literal["off", "medium", "on"]


class AgentModelSettings(BaseModel):
    """Typed request values; optional None means omit the provider parameter."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    parallel_tool_calls: bool = False
    max_tokens: StrictInt | None = Field(default=None, gt=0)
    temperature: StrictFloat | None = Field(
        default=None, ge=0, le=2, allow_inf_nan=False
    )
    top_p: StrictFloat | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    top_k: StrictInt | None = Field(default=None, gt=0)
    seed: StrictInt | None = None
    # Chat-template controls; only a model whose table has their rows accepts them.
    reasoning: ReasoningMode | None = None
    force_nonempty_content: bool | None = None

    @model_serializer(mode="wrap")
    def unset_template_controls_left_out(self, handler) -> dict[str, Any]:
        # Left out only while unset, so selections recorded before them still read as
        # complete and an explicit None (omit the parameter) is kept.
        values = handler(self)
        for name in _TEMPLATE_CONTROLS - self.model_fields_set:
            values.pop(name, None)
        return values

    def request_settings(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


_TEMPLATE_CONTROLS = frozenset({"reasoning", "force_nonempty_content"})


RequestSettingName = Literal[
    "parallel_tool_calls",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "reasoning",
    "force_nonempty_content",
]

DEFAULT_TOOL_CHOICE = "auto"
"""The one Tool choice of every model (owner, #476 alignment G1.2 B).

A forced choice puts the request under the provider's grammar, so only a model
entry with recorded ``grammar_evidence`` may state another. A Run file records the
effective choice once, at its global level.
"""


COMPATIBLE_SCHEMA_KEYWORDS = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "anyOf",
        "const",
        "default",
        "description",
        "enum",
        "exclusiveMinimum",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "oneOf",
        "pattern",
        "properties",
        "required",
        "type",
    }
)
"""The JSON Schema keywords offered Tools use on an endpoint that applies no
grammar to them (#476 alignment G1.2 B; the #457 sweep's Pattern C list). A new
keyword is declared here, or in one model's entry, before any Tool sends it."""


def forces_a_call(choice: object) -> bool:
    """Whether a Tool choice forces a call: any value other than unset or auto."""
    return choice not in (None, DEFAULT_TOOL_CHOICE)


WirePath = tuple[str, ...]


@dataclass(frozen=True)
class ExpandedWireRow:
    """A setting whose value is written as several wire fields, chosen by value."""

    fields: Mapping[Any, Mapping[WirePath, Any]]


WireRow = WirePath | ExpandedWireRow

COMPATIBLE_WIRE_SETTINGS: Mapping[RequestSettingName, WireRow] = {
    "parallel_tool_calls": ("parallel_tool_calls",),
    "max_tokens": ("max_completion_tokens",),
    "temperature": ("temperature",),
    "top_p": ("top_p",),
    "seed": ("seed",),
    # The compatible wire has no named field for it.
    "top_k": ("extra_body", "top_k"),
}
"""Where each typed setting sits in a compatible chat-completions request body."""


SERIALIZED_BYTES_PER_TOKEN = 2
"""Request body bytes per estimated input token; no tokenizer (#282 decision 11).

Below every request the provider has counted: over the 13 retained live Runs to
2026-10-14 (Qwen and Nemotron), body bytes per reported ``prompt_tokens`` ran from
2.41 to 4.78, median about 3. Dividing by less errs high, so the estimate never
falls below the provider's count on that evidence.
"""


class ModelContextBudget(BaseModel):
    """Configured budget, separate from a measured tokenizer or endpoint limit."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    context_window: StrictInt = Field(gt=0)
    input_reserve: StrictInt = Field(gt=0)
    serialized_body_safety_margin: StrictInt = Field(ge=0)
    source: Literal["preserved_product_configuration"] = (
        "preserved_product_configuration"
    )

    @model_validator(mode="after")
    def completion_room_exists(self) -> "ModelContextBudget":
        if self.input_reserve >= self.context_window:
            raise ValueError("configured input reserve leaves no output room")
        if self.serialized_body_safety_margin >= self.context_window:
            raise ValueError("configured safety margin leaves no output room")
        return self

    @property
    def output_ceiling(self) -> int:
        return self.context_window - self.input_reserve

    def input_estimate(self, body_byte_count: int) -> int:
        """Input tokens a serialized request body is assumed to use."""
        return -(-body_byte_count // SERIALIZED_BYTES_PER_TOKEN)

    def completion_allowance(self, requested: int, body_byte_count: int) -> int:
        """The requested ceiling, cut to the room the body's estimate leaves."""
        return min(
            requested,
            self.context_window
            - self.input_estimate(body_byte_count)
            - self.serialized_body_safety_margin,
        )


class ModelSettingsSelection(BaseModel):
    """One model's retained baseline and accepted request-setting vocabulary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model: str
    accepted_settings: tuple[RequestSettingName, ...] = Field(min_length=1)
    baseline: AgentModelSettings
    context_budget: ModelContextBudget
    # The injection patterns this model switches on, retained with the baseline so
    # a retained Attempt resolves as it ran; none records no key (ADR 0048).
    injection_patterns: tuple[PatternName, ...] = Field(
        default=(), exclude_if=lambda names: not names
    )

    @field_validator("accepted_settings", "injection_patterns", mode="before")
    @classmethod
    def accept_json_array(cls, value: object) -> object:
        # An enclosing before-validator materializes JSON arrays as Python lists.
        # Normalize only that container; strict Literal validation still owns names.
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def baseline_is_valid(self) -> "ModelSettingsSelection":
        if len(set(self.accepted_settings)) != len(self.accepted_settings):
            raise ValueError("accepted settings must be unique")
        self.validate_settings(self.baseline.request_settings())
        return self

    def refused_settings(self, settings: AgentModelSettings) -> tuple[str, ...]:
        """Typed values this model refuses: an unaccepted setting or one above its ceiling."""
        values = settings.request_settings()
        refused = set(values) - set(self.accepted_settings)
        if (
            settings.max_tokens is not None
            and settings.max_tokens > self.context_budget.output_ceiling
        ):
            refused.add("max_tokens")
        return tuple(sorted(refused))

    def validate_settings(self, values: Mapping[str, Any]) -> AgentModelSettings:
        unsupported = set(values) - set(self.accepted_settings)
        if unsupported:
            # Names outside the typed set never reach the typed check below.
            raise ValueError(
                "unsupported model request settings: " + ", ".join(sorted(unsupported))
            )
        settings = AgentModelSettings.model_validate(dict(values))
        if refused := self.refused_settings(settings):
            raise ValueError(
                f"{self.model} refuses request settings: {', '.join(refused)} "
                f"(output ceiling {self.context_budget.output_ceiling})"
            )
        return settings


_Definition = TypeVar("_Definition")

CallReasoningPlacement = Literal["first", "last", "none"]
"""Where a family's projection puts the call-reasoning argument; `none` adds none."""
CALL_REASONING_ARGUMENT = "reasoning"
"""The required top-level string a family's projection adds to every Tool (#241)."""

_CALL_REASONING_DESCRIPTIONS: Mapping[str, str] = {
    # The wording spike S1 sent; the field came last without an order line.
    "last": "After the arguments: why you chose these values.",
    "first": "Before the arguments: why you choose these values.",
}


ReplyQuirk = Literal[
    "text_form_tool_call",
    "text_form_tool_call_cut_off",
    "think_tags",
    "call_reasoning_missing",
    "argument_name",
    "reasoning_only",
    "empty_reply",
    "call_stub",
]
"""A wire quirk the model's adapter read in a reply (#241); an injection trigger.

``argument_name``: a call's argument name was fixed to the name its Tool showed (#242).
``reasoning_only``, ``empty_reply``, ``call_stub``: the Model turn read the reply as
unusable (#421, #444), marked as it arrived; the transcript composer keeps it padded.
"""

REPLY_READING = "reply_reading"
"""The reply's ``provider_details`` key holding the adapter's reading.

``{"quirks": [...], "raw": {...}}``, present only when the reading changed the
reply: ``raw`` keeps each wire field it changed (``content``, ``tool_calls``) as
received (owner, #241 readiness; interim shape).
"""


def reply_quirks(provider_details: Mapping[str, Any] | None) -> tuple[ReplyQuirk, ...]:
    """The quirks a recorded reply's adapter read, for an injection trigger to match."""
    reading = (provider_details or {}).get(REPLY_READING)
    return tuple(reading["quirks"]) if reading is not None else ()


def reply_content(provider_details: Mapping[str, Any] | None) -> str | None:
    """A recorded reply's wire content, when the adapter's reading changed it."""
    reading = (provider_details or {}).get(REPLY_READING)
    return None if reading is None else reading["raw"].get("content")


CANONICAL_TEXT_WORDS = "Not blank, with no leading or trailing whitespace."
"""The canonical-text rule as the model reads it (#336 F2)."""


def _canonical_text_in_words(schema: Any) -> Any:
    """A Tool schema with each canonical-text pattern stated in words instead.

    Nemotron on Nebius never answered a request whose Tool schemas carried that
    pattern (#336 F2: no reply in 300 s with it, 5-7 s without, same body), so
    the rule reaches every model in its argument's description. Admission's
    validator still checks the pattern.
    """
    from bio_neb.agentic_journey.action_policy import CANONICAL_TEXT_PATTERN

    if isinstance(schema, list):
        return [_canonical_text_in_words(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    projected = {key: _canonical_text_in_words(value) for key, value in schema.items()}
    if projected.get("pattern") == CANONICAL_TEXT_PATTERN:
        del projected["pattern"]
        described = projected.get("description")
        projected["description"] = (
            f"{described} {CANONICAL_TEXT_WORDS}" if described else CANONICAL_TEXT_WORDS
        )
    return projected


def _with_call_reasoning(
    definition: _Definition, placement: CallReasoningPlacement
) -> _Definition:
    schema = dict(definition.parameters_json_schema)
    properties = dict(schema.get("properties") or {})
    if CALL_REASONING_ARGUMENT in properties:
        # The adapter removes this argument from every call before admission.
        raise RuntimeError(
            f"{definition.name} already has a {CALL_REASONING_ARGUMENT} argument"
        )
    added = {
        CALL_REASONING_ARGUMENT: {
            "type": "string",
            "description": _CALL_REASONING_DESCRIPTIONS[placement],
        }
    }
    required = list(schema.get("required") or ())
    if placement == "first":
        properties = {**added, **properties}
        required = [CALL_REASONING_ARGUMENT, *required]
    else:
        properties = {**properties, **added}
        required = [*required, CALL_REASONING_ARGUMENT]
    schema.setdefault("type", "object")
    return replace(
        definition,
        parameters_json_schema={
            **schema,
            "properties": properties,
            "required": required,
        },
    )


NEBIUS_CREDENTIAL_ENV = "NEBIUS_API_KEY"
"""The credential variable's name every entry reads by default (owner, #440).

A name only: the key's value is a per-account value from the operator's
``model.env``, never part of the build.
"""


@dataclass(frozen=True)
class ModelEndpointPolicy:
    """One model's adapter behind the Model endpoint policy seam.

    It is also the model's registry entry: where the model is served, the
    credential variable it reads and its own Run defaults.
    """

    settings_selection: ModelSettingsSelection
    # The service address, not an account value: the API key carries the account.
    endpoint: str
    # The instruction form this model reads (only role_instructions reads it), named
    # after the model; every entry states its own.
    instruction_form: str
    credential_env: str = NEBIUS_CREDENTIAL_ENV
    correction_wording: CorrectionWording = field(default_factory=CorrectionWording)
    # The translation table: a typed setting without a row cannot be sent.
    wire_settings: Mapping[RequestSettingName, WireRow] = field(
        default_factory=lambda: dict(COMPATIBLE_WIRE_SETTINGS)
    )
    # Where the projection puts the call-reasoning argument, which the adapter
    # moves out of each call before admission.
    call_reasoning: CallReasoningPlacement = "none"
    # The adapter reads text-form Tool calls and think tags (ReplyQuirk).
    reads_reply_quirks: bool = False
    # The adapter fixes an argument name that spells a shown one differently.
    fixes_argument_names: bool = False
    # The model's own Run defaults: a fresh Run that omits one takes it, so
    # registration records it; a runtime file still overrides each. Unset, the
    # choice is DEFAULT_TOOL_CHOICE.
    tool_choice: str | None = None
    # A model's own number of failed turns in a row; unset, the global role
    # default's applies.
    failed_turns: int | None = None
    # Where it was recorded that this endpoint takes a provider grammar (a strict
    # Tool, a forced choice, a response format) and still answers. Unset, no
    # request to it may carry one.
    grammar_evidence: str | None = None
    # The JSON Schema keywords an offered Tool's parameters may use on this
    # endpoint; tests/test_offered_wire_schemas.py fails on any other.
    schema_keywords: frozenset[str] = COMPATIBLE_SCHEMA_KEYWORDS

    def __post_init__(self) -> None:
        if forces_a_call(self.tool_choice) and self.grammar_evidence is None:
            raise ValueError(
                f"{self.model} forces tool_choice {self.tool_choice!r} without "
                "recorded grammar evidence"
            )

    @property
    def model(self) -> str:
        return self.settings_selection.model

    def run_defaults(self) -> dict[str, Any]:
        """The ``agent_invocation`` values a fresh Run file omitting them takes."""
        values: dict[str, Any] = {}
        if self.failed_turns is not None:
            values["failed_turns"] = self.failed_turns
        values["tool_choice"] = self.tool_choice or DEFAULT_TOOL_CHOICE
        return values

    def wire_fields(
        self, name: RequestSettingName, value: Any
    ) -> Mapping[WirePath, Any]:
        """Where one valued setting sits in the compatible request, and as what."""
        row = self.wire_settings[name]
        if isinstance(row, ExpandedWireRow):
            return row.fields[value]
        return {row: value}

    def project_tool_schema(self, definition: _Definition) -> _Definition:
        """The Tool definition this model is shown: canonical-text rules in words,
        then the family's call reasoning, if any."""
        schema = _canonical_text_in_words(definition.parameters_json_schema)
        if schema != definition.parameters_json_schema:
            definition = replace(definition, parameters_json_schema=schema)
        if self.call_reasoning == "none":
            return definition
        return _with_call_reasoning(definition, self.call_reasoning)

    def untranslatable_settings(self, settings: AgentModelSettings) -> tuple[str, ...]:
        """Valued settings this model's wire cannot carry."""
        return tuple(sorted(set(settings.request_settings()) - set(self.wire_settings)))


_PRESERVED_CONTEXT_BUDGET = ModelContextBudget(
    context_window=262144,
    input_reserve=16384,
    serialized_body_safety_margin=8192,
)


_US_CENTRAL1 = "https://api.tokenfactory.us-central1.nebius.com/v1"
_US_NORTH1 = "https://api.tokenfactory.us-north1.nebius.com/v1"


def _plain_endpoint_policy(
    model: str, endpoint: str, *, instruction_form: str
) -> ModelEndpointPolicy:
    budget = _PRESERVED_CONTEXT_BUDGET
    return ModelEndpointPolicy(
        ModelSettingsSelection(
            model=model,
            accepted_settings=(
                "parallel_tool_calls",
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "seed",
            ),
            baseline=AgentModelSettings(max_tokens=budget.output_ceiling),
            context_budget=budget,
        ),
        endpoint=endpoint,
        instruction_form=instruction_form,
    )


@dataclass(frozen=True)
class ModelFamilyProfile:
    """A family's request defaults and wire; a size states only its differences."""

    accepted_settings: tuple[RequestSettingName, ...]
    defaults: AgentModelSettings
    context_budget: ModelContextBudget
    wire_settings: Mapping[RequestSettingName, WireRow]
    call_reasoning: CallReasoningPlacement = "none"
    reads_reply_quirks: bool = False
    fixes_argument_names: bool = False
    correction_wording: CorrectionWording = field(default_factory=CorrectionWording)
    injection_patterns: tuple[str, ...] = ()

    def model(
        self,
        model: str,
        *,
        endpoint: str,
        instruction_form: str,
        tool_choice: str | None = None,
        **overrides: Any,
    ) -> ModelEndpointPolicy:
        """A size entry: the family's defaults with this size's typed overrides.

        Each size names its own instruction form, so a Run records which model's
        words it read (owner, #441 R4)."""
        baseline = AgentModelSettings.model_validate(
            {**self.defaults.model_dump(exclude_none=True), **overrides}
        )
        return ModelEndpointPolicy(
            ModelSettingsSelection(
                model=model,
                accepted_settings=self.accepted_settings,
                baseline=baseline,
                context_budget=self.context_budget,
                injection_patterns=self.injection_patterns,
            ),
            endpoint=endpoint,
            correction_wording=self.correction_wording,
            wire_settings=self.wire_settings,
            call_reasoning=self.call_reasoning,
            reads_reply_quirks=self.reads_reply_quirks,
            fixes_argument_names=self.fixes_argument_names,
            instruction_form=instruction_form,
            tool_choice=tool_choice,
        )


_CHAT_TEMPLATE: WirePath = ("extra_body", "chat_template_kwargs")

# Nemotron 3 Ultra's own window: Nebius reports a context length of
# 1,048,576. Reserve and margin are the preserved budget's.
_NEMOTRON_CONTEXT_BUDGET = ModelContextBudget(
    context_window=1048576,
    input_reserve=_PRESERVED_CONTEXT_BUDGET.input_reserve,
    serialized_body_safety_margin=_PRESERVED_CONTEXT_BUDGET.serialized_body_safety_margin,
)
# The reply allowance is decided apart from the window and stays at today's
# maximum, not the window's 1,032,192 output ceiling.
_NEMOTRON_REPLY_ALLOWANCE = _PRESERVED_CONTEXT_BUDGET.output_ceiling


def _nemotron_family() -> ModelFamilyProfile:
    return ModelFamilyProfile(
        accepted_settings=(
            "parallel_tool_calls",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "reasoning",
            "force_nonempty_content",
        ),
        # Every known pattern on (S2); NVIDIA's sampling default (R1).
        defaults=AgentModelSettings(
            max_tokens=_NEMOTRON_REPLY_ALLOWANCE,
            temperature=1.0,
            top_p=0.95,
            reasoning="medium",
            force_nonempty_content=True,
        ),
        context_budget=_NEMOTRON_CONTEXT_BUDGET,
        wire_settings={
            **COMPATIBLE_WIRE_SETTINGS,
            # The template control that works on Nebius; both keys are always
            # written, so a value never inherits the server's default.
            "reasoning": ExpandedWireRow(
                {
                    value: {
                        (*_CHAT_TEMPLATE, "enable_thinking"): thinking,
                        (*_CHAT_TEMPLATE, "medium_effort"): medium,
                    }
                    for value, thinking, medium in (
                        ("off", False, False),
                        ("medium", True, True),
                        ("on", True, False),
                    )
                }
            ),
            "force_nonempty_content": (*_CHAT_TEMPLATE, "force_nonempty_content"),
        },
        # No per-call reasoning argument: every recorded runaway reply had one
        # set. The model's own reasoning_content stays.
        call_reasoning="none",
        # Watch-only quirks, handled defensively and recorded (run gate Q9).
        reads_reply_quirks=True,
        # The argument-name shim runs; the record keeps both forms (Q9).
        fixes_argument_names=True,
        # Every registered injection pattern and quirk trigger, loose.
        injection_patterns=injection_patterns.NAMES,
    )


NEMOTRON_FAMILY = _nemotron_family()

# Qwen reads its own words (its form text); both Nemotron sizes read the block
# source as written, each under its own form name.
QWEN_ENDPOINT_POLICY = _plain_endpoint_policy(
    "Qwen/Qwen3.5-397B-A17B", _US_CENTRAL1, instruction_form="qwen_3_5"
)
NEMOTRON_ULTRA_ENDPOINT_POLICY = NEMOTRON_FAMILY.model(
    "nvidia/Nemotron-3-Ultra-550b-a55b",
    endpoint=_US_CENTRAL1,
    instruction_form="nemotron_3_ultra",
)
# Served on Nebius us-north1: no sampling or template controls, so the provider
# decides them. The reply limit stays the family's: without one, Nebius cuts
# replies at 8,192 tokens. Its window is not published, so it keeps the family's.
# Its passing live Runs used "auto", now every model's default; its failure rule
# is the global role default's.
NEMOTRON_SUPER_VL_ENDPOINT_POLICY = NEMOTRON_FAMILY.model(
    "nvidia/Nemotron-3.5-Super-VL",
    endpoint=_US_NORTH1,
    instruction_form="nemotron_3_5_super",
    temperature=None,
    top_p=None,
    reasoning=None,
    force_nonempty_content=None,
)


class ModelUnlisted(ValueError):
    """A model the registry does not list; no adapter stands in for it (#440)."""

    def __init__(self, model: str):
        super().__init__(f"{model} is not listed in the model registry")
        self.model = model


@dataclass(frozen=True)
class ModelRegistry:
    """The build's listed models and its one default entry.

    A Run that names no model takes ``default`` under ``default_reference``, the
    roster key #387's Super VL Runs registered it as (owner, #440 A2, Q6).
    """

    entries: Mapping[str, ModelEndpointPolicy]
    default: ModelEndpointPolicy
    default_reference: str = "nemotron"

    def __post_init__(self) -> None:
        if any(model != entry.model for model, entry in self.entries.items()):
            raise ValueError("registry entries are keyed by their own model")
        if self.entries.get(self.default.model) != self.default:
            raise ValueError("the default entry must be listed")

    @classmethod
    def of(
        cls, *entries: ModelEndpointPolicy, default: ModelEndpointPolicy
    ) -> "ModelRegistry":
        return cls({entry.model: entry for entry in entries}, default)

    def policy(self, model: str) -> ModelEndpointPolicy:
        try:
            return self.entries[model]
        except KeyError:
            raise ModelUnlisted(model) from None

    def lists(self, model: object) -> bool:
        return isinstance(model, str) and model in self.entries

    def replacing(self, *entries: ModelEndpointPolicy) -> "ModelRegistry":
        """The same registry with these entries in place of their models' own."""
        replaced = {**self.entries, **{entry.model: entry for entry in entries}}
        return ModelRegistry(
            replaced, replaced[self.default.model], self.default_reference
        )


BUILD_MODEL_REGISTRY = ModelRegistry.of(
    QWEN_ENDPOINT_POLICY,
    NEMOTRON_ULTRA_ENDPOINT_POLICY,
    NEMOTRON_SUPER_VL_ENDPOINT_POLICY,
    default=NEMOTRON_SUPER_VL_ENDPOINT_POLICY,
)

_REGISTRY: ContextVar[ModelRegistry] = ContextVar(
    "model_registry", default=BUILD_MODEL_REGISTRY
)


def model_registry() -> ModelRegistry:
    """The registry configuration reads: the build's, unless a caller bound one."""
    return _REGISTRY.get()


@contextmanager
def use_model_registry(registry: ModelRegistry) -> Iterator[ModelRegistry]:
    """Bind another registry, such as a test's entry, for configuration and Attempts."""
    token = _REGISTRY.set(registry)
    try:
        yield registry
    finally:
        _REGISTRY.reset(token)


def model_endpoint_policy(model: str) -> ModelEndpointPolicy:
    """The model's registry entry, without contacting or claiming its availability.

    Configuration and an Attempt's resolution ask here; roles, assembly and the
    adapter read the Attempt's bound policy instead. An unlisted model raises
    ``ModelUnlisted``.
    """
    return model_registry().policy(model)
