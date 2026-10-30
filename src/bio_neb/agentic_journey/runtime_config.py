"""Strict, secret-free operational configuration."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Mapping, Tuple, get_args
from urllib.parse import urlparse, urlunparse

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationDefaults,
    AgentInvocationOverrides,
    AgentInvocationPolicy,
    InvestigationCompletion,
    InvestigationWorkPolicy,
    ModelReference,
    resolve_agent_invocation_policy,
    resolve_investigation_work_policy,
)
from bio_neb.agentic_journey.model_settings import (
    AgentModelSettings,
    ModelEndpointPolicy,
    ModelRegistry,
    ModelSettingsSelection,
    ModelUnlisted,
    forces_a_call,
    model_endpoint_policy,
    model_registry,
)
from bio_neb.agentic_journey.role_declaration import ROLE_DEFAULT, AgentRole
from bio_neb.agentic_journey.role_declaration import role as declared_role
from bio_neb.application.verification_policy import (
    ArtifactVerificationDefaults,
    ArtifactVerificationOverrides,
    ResolvedArtifactVerificationPolicy,
    resolve_artifact_verification_policy,
)
from bio_neb.failure import FailureSurface, FailureTuning, policy_for
from bio_neb.investigation.observation_defaults import (
    InvestigationConcurrency,
    VcfTableDefaults,
)
from bio_neb.parabricks.configuration import ParabricksConfiguration

_CREDENTIAL_ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HTTP_ENDPOINT = TypeAdapter(AnyHttpUrl)


def _canonical_runtime_string(value: str) -> str:
    if not value or value.strip() != value:
        raise ValueError("runtime selection must be a non-empty canonical string")
    return value


def _environment_variable_identifier(value: str) -> str:
    if not _CREDENTIAL_ENV.fullmatch(value):
        raise ValueError("credential_env must name an environment variable")
    return value


def _canonical_nebius_model_endpoint(value: str) -> str:
    malformed = any(character.isspace() or ord(character) < 32 for character in value)
    try:
        original_path = urlparse(value).path
        normalized = _HTTP_ENDPOINT.validate_python(value, strict=True)
        parsed = urlparse(str(normalized))
        hostname = parsed.hostname
        _ = parsed.port
    except (ValidationError, ValueError):
        malformed = True
        parsed = None
        hostname = None
    if (
        malformed
        or parsed is None
        or parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        raise ValueError("endpoint must be an HTTP endpoint without authorization data")
    if original_path not in {"", "/v1"}:
        raise ValueError("Nebius model endpoint path must be empty or exactly /v1")
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    default_port = (parsed.scheme == "https" and parsed.port == 443) or (
        parsed.scheme == "http" and parsed.port == 80
    )
    netloc = host if parsed.port is None or default_port else f"{host}:{parsed.port}"
    return urlunparse((parsed.scheme, netloc, "/v1", "", "", ""))


RuntimeEndpoint = Annotated[StrictStr, AfterValidator(_canonical_nebius_model_endpoint)]


def _contacted_nebius_model_endpoint(value: str) -> str:
    canonical = _canonical_nebius_model_endpoint(value)
    if value != canonical:
        raise ValueError(
            "contacted Nebius model endpoint must be canonical and end in /v1"
        )
    return canonical


ContactedRuntimeEndpoint = Annotated[
    StrictStr, AfterValidator(_contacted_nebius_model_endpoint)
]
RuntimeModelName = Annotated[StrictStr, AfterValidator(_canonical_runtime_string)]
CredentialEnvironmentName = Annotated[
    StrictStr, AfterValidator(_environment_variable_identifier)
]
HeartbeatInterval = Annotated[StrictFloat, Field(gt=0, allow_inf_nan=False)]


class RuntimeConfigError(ValueError):
    def __init__(self, code: str, violations: Tuple[str, ...]):
        super().__init__(code)
        self.code = code
        self.violations = violations


class _RuntimeConfigurationPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NebiusModelEndpointSelection(_RuntimeConfigurationPart):
    """One secret-free Nebius model endpoint selection."""

    provider: Literal["nebius"]
    endpoint: RuntimeEndpoint
    model: RuntimeModelName
    credential_env: CredentialEnvironmentName
    settings_selection: ModelSettingsSelection

    @model_validator(mode="after")
    def settings_belong_to_model(self) -> "NebiusModelEndpointSelection":
        if self.settings_selection.model != self.model:
            raise ValueError("settings selection belongs to another model")
        return self


class NebiusModelEndpointIdentity(_RuntimeConfigurationPart):
    """Secret-free product evidence for one contacted endpoint."""

    provider: Literal["nebius"]
    model: RuntimeModelName
    contacted_url: ContactedRuntimeEndpoint

    @model_validator(mode="before")
    @classmethod
    def contacted_url_is_already_canonical(cls, value: object) -> object:
        if isinstance(value, Mapping):
            contacted_url = value.get("contacted_url")
            if isinstance(contacted_url, str) and urlparse(contacted_url).path != "/v1":
                raise ValueError("contacted_url must already end in canonical /v1")
        return value


class ObservationTool(_RuntimeConfigurationPart):
    """One observation tool, provided by a package of the Run's locked environment."""

    package: Literal["pysam"]
    package_version: StrictStr = Field(min_length=1)


class InvestigationTools(_RuntimeConfigurationPart):
    """The observation tools a Run names (owner, #107).

    The Run resolves each one in its own environment and probes it; the section
    names no path, so a runtime file serves any checkout.
    """

    bcftools: ObservationTool
    samtools: ObservationTool


# The locked package's pin (`pyproject.toml`); a Run naming none takes it.
_LOCKED_PYSAM = ObservationTool(package="pysam", package_version="0.24.1")


def build_investigation_tools() -> InvestigationTools:
    return InvestigationTools(bcftools=_LOCKED_PYSAM, samtools=_LOCKED_PYSAM)


class RuntimeSelections(_RuntimeConfigurationPart):
    """Credential-free selections retained in Journey registration evidence."""

    model_roster: dict[ModelReference, NebiusModelEndpointSelection] = Field(
        min_length=1
    )
    agent_invocation: AgentInvocationDefaults
    # Recorded only: retained Journeys carry it and read back exactly, but nothing
    # paces by it since the Agent heartbeat retired. The Parabricks launch
    # heartbeat is a separate setting.
    heartbeat_interval_seconds: HeartbeatInterval = 30.0
    failure: FailureTuning = Field(default_factory=FailureTuning)
    # Additive and optional: a Run naming no tools records no key, so earlier
    # registrations read back exactly.
    investigation_tools: InvestigationTools | None = Field(
        default=None, exclude_if=lambda tools: tools is None
    )
    # The VCF table's window and chunk size; a Run naming none records no key,
    # and its first window and chunk freeze the named defaults they used
    # (``vcf_table_reading.run_table_settings``).
    investigation_table: VcfTableDefaults | None = Field(
        default=None, exclude_if=lambda table: table is None
    )
    # How many planner members and table chunks run at once; a Run naming none
    # records no key and runs the named defaults.
    investigation_concurrency: InvestigationConcurrency | None = Field(
        default=None, exclude_if=lambda bound: bound is None
    )
    # The artifact verification selection: every Run configured since records it,
    # so no later stage reads a file; one registered before records no key and
    # verifies by the product default.
    artifact_verification: ArtifactVerificationDefaults | None = Field(
        default=None, exclude_if=lambda selection: selection is None
    )

    @model_validator(mode="after")
    def configured_scopes_fit_their_models(self) -> "RuntimeSelections":
        """Refuse, by its path, a reference or setting no resolved Attempt could use.

        Every role and its stages resolve here exactly as their Attempts will,
        and each value is checked against its scope's model endpoint policy.
        """
        invocation = self.agent_invocation
        refused: dict[tuple[str | int, ...], InitErrorDetails] = {}
        references = {
            ("agent_invocation", "model_reference"): invocation.model_reference
        }
        references.update(
            {
                (
                    "agent_invocation",
                    "scopes",
                    "role",
                    role,
                    "model_reference",
                ): entry.model_reference
                for role, entry in invocation.scopes.role.items()
                if entry.model_reference is not None
            }
        )
        for location, reference in references.items():
            if reference not in self.model_roster:
                refused[location] = InitErrorDetails(
                    type=PydanticCustomError(
                        "model_reference_unknown",
                        "invocation references an unknown model roster entry",
                    ),
                    loc=location,
                    input=reference,
                )
        if not refused:
            for role in get_args(AgentRole):
                selection = self.model_roster[
                    invocation.role_reference(role)
                ].settings_selection
                resolved, stages = invocation.resolved_settings(
                    selection, agent_role=role
                )
                # Fresh configuration refuses an unlisted model before this; a
                # retained one still reads, and its Attempts are refused.
                registry = model_registry()
                endpoint = (
                    registry.policy(selection.model)
                    if registry.lists(selection.model)
                    else None
                )
                for stage, settings in ((None, resolved), *stages.items()):
                    for name in selection.refused_settings(settings.model_settings):
                        location = _setting_location(
                            settings.settings_sources[name], role, stage, name
                        )
                        refused[location] = InitErrorDetails(
                            type=PydanticCustomError(
                                "model_setting_refused",
                                "{model} refuses this request setting value",
                                {"model": selection.model},
                            ),
                            loc=location,
                            input=getattr(settings.model_settings, name),
                        )
                    for name in (
                        ()
                        if endpoint is None
                        else endpoint.untranslatable_settings(settings.model_settings)
                    ):
                        location = _setting_location(
                            settings.settings_sources[name], role, stage, name
                        )
                        refused[location] = InitErrorDetails(
                            type=PydanticCustomError(
                                "model_setting_untranslatable",
                                "{model} has no wire field for this request setting",
                                {"model": selection.model},
                            ),
                            loc=location,
                            input=getattr(settings.model_settings, name),
                        )
                    if (
                        endpoint is not None
                        and endpoint.grammar_evidence is None
                        and forces_a_call(settings.tool_choice)
                    ):
                        location = _setting_location(
                            settings.settings_sources["tool_choice"],
                            role,
                            stage,
                            "tool_choice",
                        )
                        refused[location] = InitErrorDetails(
                            type=PydanticCustomError(
                                "tool_choice_needs_grammar",
                                "{model} declares no grammar support for a "
                                "forced Tool choice",
                                {"model": selection.model},
                            ),
                            loc=location,
                            input=settings.tool_choice,
                        )
                    if stage is not None and not _choice_is_offered(
                        settings.tool_choice, role, stage
                    ):
                        location = _setting_location(
                            settings.settings_sources["tool_choice"],
                            role,
                            stage,
                            "tool_choice",
                        )
                        refused[location] = InitErrorDetails(
                            type=PydanticCustomError(
                                "tool_choice_not_offered",
                                "stage {stage} does not offer this Tool",
                                {"stage": stage},
                            ),
                            loc=location,
                            input=settings.tool_choice,
                        )
        if refused:
            raise ValidationError.from_exception_data(
                type(self).__name__, list(refused.values())
            )
        return self

    def selected_credential_envs(self) -> dict[str, str]:
        """Each roster reference a role selects, with its credential variable's name."""
        invocation = self.agent_invocation
        references = {invocation.model_reference} | {
            entry.model_reference
            for entry in invocation.scopes.role.values()
            if entry.model_reference is not None
        }
        return {
            reference: self.model_roster[reference].credential_env
            for reference in sorted(references)
        }

    @property
    def verification_defaults(self) -> ArtifactVerificationDefaults:
        """The Run's artifact verification selection, recorded or the product's."""
        return self.artifact_verification or ArtifactVerificationDefaults()

    def artifact_verification_policy(
        self, overrides: ArtifactVerificationOverrides
    ) -> ResolvedArtifactVerificationPolicy:
        return resolve_artifact_verification_policy(
            self.verification_defaults, overrides
        )

    def model_for_policy(
        self, policy: AgentInvocationPolicy
    ) -> NebiusModelEndpointSelection:
        """Look up an already resolved policy; never select another default."""
        try:
            return self.model_roster[policy.model_reference]
        except KeyError as error:
            raise RuntimeConfigError(
                "runtime_config_invalid", ("model_reference",)
            ) from error

    def agent_invocation_policy(
        self, agent_role: str, overrides: AgentInvocationOverrides
    ) -> AgentInvocationPolicy:
        policy = resolve_agent_invocation_policy(
            self.agent_invocation,
            overrides,
            agent_role=agent_role,
            model_settings={
                name: item.settings_selection
                for name, item in self.model_roster.items()
            },
            model_contact=policy_for(
                FailureSurface(scope="role", name=agent_role), self.failure
            ),
        )
        self.model_for_policy(policy)
        return policy


def _setting_location(
    source: str, role: str, stage: str | None, name: str
) -> tuple[str, ...]:
    """Where a resolved setting's value was configured."""
    scope = {
        "global": (),
        "role": ("scopes", "role", role),
        "stage": ("scopes", "stage", stage),
    }[source]
    if name == "tool_choice":
        return ("agent_invocation", *scope, name)
    return ("agent_invocation", *scope, "model_settings", name)


def _choice_is_offered(choice: str | None, role: str, stage: str) -> bool:
    """A mode, or a Tool the stage shows: its registered Tools, the role's output
    Tool and the search."""
    from bio_neb.agentic_journey.assignment_context import SEARCH_TOOL_NAME
    from bio_neb.agentic_journey.tool_execution import stage_tool_names

    return choice in (None, "required", "auto") or choice in (
        stage_tool_names(role, stage)
        | declared_role(role).output_tools
        | {SEARCH_TOOL_NAME}
    )


def untranslatable_setting_paths(
    endpoint: ModelEndpointPolicy, policy: AgentInvocationPolicy
) -> tuple[str, ...]:
    """Configured paths of the policy's valued settings the endpoint cannot send."""
    entries = ((None, policy.model_settings, policy.settings_sources),) + tuple(
        (stage, entry.model_settings, entry.settings_sources)
        for stage, entry in policy.stage_settings.items()
    )
    return tuple(
        sorted(
            {
                ".".join(
                    _setting_location(sources[name], policy.agent_role, stage, name)
                )
                for stage, settings, sources in entries
                for name in endpoint.untranslatable_settings(settings)
            }
        )
    )


def _listed_entry(model: object) -> ModelEndpointPolicy | None:
    """A model's registry entry, or None for a value the validators refuse."""
    registry = model_registry()
    return registry.policy(model) if registry.lists(model) else None


def _selected_model(document: Mapping) -> str | None:
    """The model a runtime document's global ``model_reference`` selects."""
    invocation = document.get("agent_invocation")
    roster = document.get("model_roster")
    reference = (
        invocation.get("model_reference") if isinstance(invocation, Mapping) else None
    )
    if not isinstance(reference, str):
        reference = model_registry().default_reference
    selected = roster.get(reference) if isinstance(roster, Mapping) else None
    model = selected.get("model") if isinstance(selected, Mapping) else None
    return model if isinstance(model, str) else None


def _model_run_defaults(document: Mapping) -> dict[str, object]:
    """The Run's model's own ``agent_invocation`` values (#387).

    A reference that selects no listed model is refused by its own path; the
    default entry's values keep that refusal the only one.
    """
    entry = _listed_entry(_selected_model(document)) or model_registry().default
    return entry.run_defaults()


def _default_roster(registry: ModelRegistry) -> dict[str, dict[str, object]]:
    return {registry.default_reference: {"model": registry.default.model}}


def _over(default: Mapping, named: Mapping) -> dict[str, object]:
    """A file's object over the build's: what it names wins, the rest stays (#440)."""
    return {
        **default,
        **{
            key: _over(default[key], value)
            if isinstance(default.get(key), Mapping) and isinstance(value, Mapping)
            else value
            for key, value in named.items()
        },
    }


def _entry_selection(value: Mapping, entry: ModelEndpointPolicy) -> dict[str, object]:
    """A roster entry naming a listed model, completed from its registry entry."""
    return {
        "provider": "nebius",
        "endpoint": entry.endpoint,
        "credential_env": entry.credential_env,
        "settings_selection": entry.settings_selection,
        **value,
    }


def _fresh_roster(roster: Mapping, registry: ModelRegistry) -> dict[str, object]:
    """The default roster with a file's entries over it, each entry completed.

    An unlisted model is refused by its path; no adapter stands in for it. Any
    other malformed entry reaches the typed validators as the file states it.
    """
    merged = _over(_default_roster(registry), roster)
    named = {
        key: value["model"]
        for key, value in merged.items()
        if isinstance(value, Mapping) and isinstance(value.get("model"), str)
    }
    unlisted = [
        InitErrorDetails(
            type=PydanticCustomError(
                "model_unlisted", "the model registry does not list this model"
            ),
            loc=("model_roster", key, "model"),
            input=model,
        )
        for key, model in named.items()
        if not registry.lists(model)
    ]
    if unlisted:
        raise ValidationError.from_exception_data("RuntimeConfiguration", unlisted)
    return {
        key: _entry_selection(value, registry.policy(named[key]))
        if key in named
        else value
        for key, value in merged.items()
    }


def retained_runtime_selections(value: object) -> RuntimeSelections:
    """Require complete recorded selections before interpreting their policy defaults."""
    selections = RuntimeSelections.model_validate(value)
    if isinstance(value, Mapping) and selections.model_dump(mode="json") != value:
        raise ValueError("retained runtime selections must be complete")
    return selections


RetainedRuntimeSelections = Annotated[
    RuntimeSelections, BeforeValidator(retained_runtime_selections)
]


def _without_retired_work_keys(value):
    """A registered runtime file may still name the retired operational route.

    #105 moved ``operational_failure`` into the failure module (owner Q1 b: old
    Runs stay forkable). A file naming its one former value reads as today's
    policy; any other value is refused as before. Journey facts that recorded
    it are not read this way: they keep refusing.
    """
    if isinstance(value, Mapping) and value.get("operational_failure") == (
        "stop_shared_or_unknown"
    ):
        return {
            key: item for key, item in value.items() if key != "operational_failure"
        }
    return value


class RuntimeConfiguration(RuntimeSelections):
    """The public, credential-free configuration contract.

    Every field a file omits takes the build's typed default (owner, #440 A2):
    the registry's default entry, its Run defaults and the product's sections. A
    file overrides only what it names; its ``model_roster`` merges by key.
    """

    contract_version: Literal["1.0"]
    parabricks: ParabricksConfiguration | None = None
    investigation_work: Annotated[
        InvestigationWorkPolicy, BeforeValidator(_without_retired_work_keys)
    ] = Field(default_factory=InvestigationWorkPolicy)
    investigation_tools: InvestigationTools | None = Field(
        default_factory=build_investigation_tools,
        exclude_if=lambda tools: tools is None,
    )
    # Command policies retain their own defaults and interpretation in the new
    # artifact receipt; historical Agent registration selections stay unchanged.
    artifact_verification: ArtifactVerificationDefaults = Field(
        default_factory=ArtifactVerificationDefaults
    )

    @model_validator(mode="before")
    @classmethod
    def fresh_configuration_takes_build_defaults(cls, value: object) -> object:
        # A fresh configuration that omits a value its model names takes the
        # model's, so registration records it and no later read consults the
        # model entry. The allowances are the Run's model's, and a role
        # selecting another model takes that model's in its role scope. The
        # Tool choice is recorded once, globally; a role carries one only
        # where its own model states another, unless the file states a global
        # one. Retained RuntimeSelections are never filled.
        if not isinstance(value, Mapping):
            return value
        registry = model_registry()
        tools = value.get("investigation_tools")
        if isinstance(tools, Mapping):
            value = {
                **value,
                "investigation_tools": _over(
                    build_investigation_tools().model_dump(mode="json"), tools
                ),
            }
        roster = value.get("model_roster", {})
        if isinstance(roster, Mapping):
            value = {**value, "model_roster": _fresh_roster(roster, registry)}
            roster = value["model_roster"]
        invocation = value.get("agent_invocation", {})
        if not isinstance(invocation, Mapping) or not isinstance(roster, Mapping):
            return value
        filled = {
            "model_reference": registry.default_reference,
            **_model_run_defaults(value),
            **invocation,
        }
        # Only valid object shapes are read or extended; any other value reaches
        # the typed validators exactly as the file states it.
        scopes = invocation.get("scopes", {})
        roles = scopes.get("role", {}) if isinstance(scopes, Mapping) else None
        if "tool_choice" not in invocation and isinstance(roles, Mapping):
            roles = dict(roles)
            for role, entry in roles.items():
                if not isinstance(entry, Mapping) or "tool_choice" in entry:
                    continue
                reference = entry.get("model_reference", filled["model_reference"])
                # A malformed reference is the validators' to refuse, not ours.
                selected = roster.get(reference) if isinstance(reference, str) else None
                model = selected.get("model") if isinstance(selected, Mapping) else None
                listed = _listed_entry(model)
                choice = (
                    None if listed is None else listed.run_defaults()["tool_choice"]
                )
                if choice not in (None, filled["tool_choice"]):
                    roles[role] = {**entry, "tool_choice": choice}
        if isinstance(roles, Mapping):
            roles = dict(roles)
            for role, entry in roles.items():
                reference = (
                    entry.get("model_reference") if isinstance(entry, Mapping) else None
                )
                if (
                    not isinstance(reference, str)
                    or reference == filled["model_reference"]
                ):
                    continue
                selected = roster.get(reference)
                model = selected.get("model") if isinstance(selected, Mapping) else None
                listed = _listed_entry(model)
                if listed is not None and listed.failed_turns is not None:
                    roles[role] = {"failed_turns": listed.failed_turns, **entry}
            if roles != scopes.get("role", {}):
                filled["scopes"] = {**scopes, "role": roles}
        return {**value, "agent_invocation": filled}

    @model_validator(mode="after")
    def fresh_settings_are_model_owned(self) -> "RuntimeConfiguration":
        # Fresh configurations may repeat the current snapshot, but cannot define
        # another baseline. Retained RuntimeSelections skip only this baseline
        # comparison, so a Run keeps the model settings it recorded. Every other
        # recorded selection must still validate under current rules, with no
        # compatibility for retired settings (ADR 0010): a retained Journey whose
        # recorded selections current rules refuse no longer reads.
        for selected in self.model_roster.values():
            if (
                selected.settings_selection
                != model_endpoint_policy(selected.model).settings_selection
            ):
                raise ValueError(
                    "fresh model settings differ from the model-owned baseline"
                )
        return self

    def runtime_selections(self) -> RuntimeSelections:
        return RuntimeSelections(
            model_roster=self.model_roster,
            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
            agent_invocation=self.agent_invocation,
            failure=self.failure,
            investigation_tools=self.investigation_tools,
            investigation_table=self.investigation_table,
            investigation_concurrency=self.investigation_concurrency,
            artifact_verification=self.artifact_verification,
        )

    def investigation_work_policy(
        self, completion: InvestigationCompletion | None = None
    ) -> InvestigationWorkPolicy:
        return resolve_investigation_work_policy(self.investigation_work, completion)


@dataclass(frozen=True)
class LoadedRuntimeConfig:
    """One role's resolution of the Run's registered selections.

    ``model`` is the selected model's endpoint policy, bound once here so the
    Attempt, its roles and its adapter read it and never look a model up.
    """

    selections: RuntimeSelections
    credential: str
    invocation_policy: AgentInvocationPolicy
    model: ModelEndpointPolicy


def retained_invocation_policy(
    payload: Mapping, selections: RuntimeSelections, *, agent_role: str
) -> AgentInvocationPolicy:
    """Reconstruct invocation provenance from retained registration and explicit inputs."""
    policy = AgentInvocationPolicy.model_validate(
        payload.get("effective_invocation_policy")
    )
    overrides = AgentInvocationOverrides.model_validate(
        payload.get("invocation_overrides")
    )
    expected = resolve_agent_invocation_policy(
        selections.agent_invocation,
        overrides,
        agent_role=agent_role,
        model_settings={
            name: item.settings_selection
            for name, item in selections.model_roster.items()
        },
        model_contact=policy_for(
            FailureSurface(scope="role", name=agent_role), selections.failure
        ),
    )
    selections.model_for_policy(expected)
    if (
        policy != expected
        or policy.model_dump(mode="json") != payload.get("effective_invocation_policy")
        or policy.policy_id != payload.get("invocation_policy_id")
        or policy.continuation.model_dump(mode="json")
        != payload.get("continuation_policy")
        or policy.continuation.policy_id != payload.get("continuation_policy_id")
        or payload.get("agent_role") != agent_role
        or overrides.explicit_settings() != payload.get("invocation_overrides")
    ):
        raise ValueError("Attempt invocation policy binding is invalid")
    return policy


def retained_endpoint_policy(
    policy: AgentInvocationPolicy, selections: RuntimeSelections
):
    """The endpoint policy a retained Attempt's model ran under, for reopen's
    checks: its correction wording."""
    return model_endpoint_policy(selections.model_for_policy(policy).model)


def validate_retained_model_request(
    payload: Mapping, policy: AgentInvocationPolicy
) -> None:
    """Bind normalized request values to their stage's settings in their Attempt."""
    if payload.get("contact_policy") != policy.model_contact.model_dump(mode="json"):
        raise ValueError("model request differs from retained contact policy")
    recorded = payload.get("settings_selection")
    settings = dict(payload["compatible_adapter_evidence"]["request"]["model_settings"])
    stage = payload.get("stage")
    choice = settings.pop("tool_choice", None)
    # An unconfigured stage sends no choice.
    if (
        settings != policy.stage_model_settings(stage).request_settings()
        or choice != policy.stage_tool_choice(stage)
        or (
            recorded is not None
            and recorded != policy.settings_selection.model_dump(mode="json")
        )
    ):
        raise ValueError("model request differs from retained invocation settings")


def _validation_violations(error: ValidationError) -> Tuple[str, ...]:
    fields = (
        ".".join(str(part) for part in item["loc"]) or "$"
        for item in error.errors(include_url=False, include_input=False)
    )
    return tuple(sorted(set(fields)))


def read_runtime_config(path: Path | None) -> RuntimeConfiguration:
    """Read and validate public selections without consulting the environment.

    No file is the build's defaults; a file states ``contract_version`` and
    overrides only what it names.
    """

    if path is None:
        return RuntimeConfiguration.model_validate({"contract_version": "1.0"})
    try:
        raw = Path(path).read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeConfigError("runtime_config_invalid", ("$",)) from error

    try:
        return RuntimeConfiguration.model_validate(value)
    except ValidationError as error:
        raise RuntimeConfigError(
            "runtime_config_invalid", _validation_violations(error)
        ) from error


class RuntimeFieldRefusal(_RuntimeConfigurationPart):
    path: StrictStr
    reason: StrictStr


class RuntimeFieldDifference(_RuntimeConfigurationPart):
    """A still-valid value that differs from what the product selects when omitted."""

    path: StrictStr
    value: JsonValue
    default: JsonValue
    default_source: Literal["product_default", "model_default", "absent"]


class RuntimeConfigDrift(_RuntimeConfigurationPart):
    """How one runtime file differs from current product defaults; a read, never a write.

    With no file the Run takes the defaults, so nothing differs.
    """

    schema_version: Literal["bio-neb.runtime-drift/1"] = "bio-neb.runtime-drift/1"
    runtime_config_file: StrictStr | None
    byte_count: StrictInt | None
    sha256: StrictStr | None
    valid: bool
    refused: tuple[RuntimeFieldRefusal, ...]
    stale_but_valid: tuple[RuntimeFieldDifference, ...]
    per_run: tuple[StrictStr, ...]
    # The labelled per-Run values from the same bytes, for in-process readers such
    # as readiness; the printed report names the paths only.
    per_run_values: dict[StrictStr, JsonValue] = Field(default={}, exclude=True)


def _leaves(value: object, prefix: tuple[str, ...] = ()):
    if isinstance(value, Mapping) and value:
        for key, item in value.items():
            yield from _leaves(item, (*prefix, str(key)))
    else:
        yield prefix, value


def _lookup(document: object, parts: tuple[str, ...]) -> tuple[bool, object]:
    for part in parts:
        if not isinstance(document, Mapping) or part not in document:
            return False, None
        document = document[part]
    return True, document


def _without(document: object, paths: set[str], prefix: str = "") -> object:
    if not isinstance(document, Mapping):
        return document
    return {
        key: _without(value, paths, f"{prefix}{key}.")
        for key, value in document.items()
        if f"{prefix}{key}" not in paths
    }


# Within `parabricks`, only these launch selections are chosen afresh for each Run.
PER_RUN_LAUNCH_SELECTIONS = (
    "parabricks.launch.object_scope",
    "parabricks.launch.object_storage.bucket",
    "parabricks.launch.work_directory",
)


def _per_run_values(document: Mapping, sections: tuple[str, ...]) -> dict[str, object]:
    """Labelled per-Run values: whole sections, except `parabricks` by its leaves."""
    values: dict[str, object] = {}
    for name in sections:
        for dotted in PER_RUN_LAUNCH_SELECTIONS if name == "parabricks" else (name,):
            found, value = _lookup(document, tuple(dotted.split(".")))
            if found:
                values[dotted] = value
    return values


def _product_defaults() -> tuple[dict[str, object], tuple[str, ...]]:
    """The document the build selects for every field a file may omit.

    A section with no build default is chosen per Run and is not compared.
    """
    defaults = read_runtime_config(None).model_dump(mode="json")
    per_run = tuple(
        sorted(
            name
            for name in RuntimeConfiguration.model_fields
            if defaults.get(name) is None
        )
    )
    return {
        name: value for name, value in defaults.items() if value is not None
    }, per_run


class RuntimeDefaults(_RuntimeConfigurationPart):
    """The selections a Run begun with no runtime file registers; a read only.

    Its Parabricks section, when it has one, is composed at `begin` from the
    operator home's account file and the Run root (#454), so it is not shown here.
    """

    schema_version: Literal["bio-neb.runtime-defaults/1"] = "bio-neb.runtime-defaults/1"
    configuration: JsonValue


def describe_runtime_defaults() -> RuntimeDefaults:
    return RuntimeDefaults(
        configuration=read_runtime_config(None).model_dump(mode="json")
    )


def _model_baseline(document: Mapping) -> dict[str, object]:
    """The settings the selected model's own baseline sets; class defaults excluded."""
    model = _selected_model(document)
    if model is None:
        return {}
    return model_endpoint_policy(model).settings_selection.baseline.model_dump(
        mode="json", exclude_unset=True
    )


def describe_runtime_drift(path: Path | None) -> RuntimeConfigDrift:
    """Report refused fields and still-valid values that differ from current defaults.

    Sections with no product default are not compared. Their per-Run selections are
    labelled: a whole section, or for `parabricks` its `PER_RUN_LAUNCH_SELECTIONS`
    leaves. Fields a file omits take their defaults and are not drift. Nothing is
    written.
    """
    if path is None:
        return RuntimeConfigDrift(
            runtime_config_file=None,
            byte_count=None,
            sha256=None,
            valid=True,
            refused=(),
            stale_but_valid=(),
            per_run=(),
        )
    try:
        content = Path(path).read_bytes()
    except OSError:
        content = None
    try:
        document = json.loads(content.decode("utf-8")) if content is not None else None
    except (UnicodeError, json.JSONDecodeError):
        document = None
    if not isinstance(document, dict):
        return RuntimeConfigDrift(
            runtime_config_file=str(path),
            byte_count=None if content is None else len(content),
            sha256=None if content is None else hashlib.sha256(content).hexdigest(),
            valid=False,
            refused=(RuntimeFieldRefusal(path="$", reason="unreadable"),),
            stale_but_valid=(),
            per_run=(),
        )
    try:
        RuntimeConfiguration.model_validate(document)
        refused: tuple[RuntimeFieldRefusal, ...] = ()
    except ValidationError as error:
        refused = tuple(
            sorted(
                {
                    RuntimeFieldRefusal(
                        path=".".join(str(part) for part in item["loc"]) or "$",
                        reason=item["type"],
                    )
                    for item in error.errors(include_url=False, include_input=False)
                },
                key=lambda item: (item.path, item.reason),
            )
        )
    defaults, per_run = _product_defaults()
    per_run_values = _per_run_values(document, per_run)
    baseline = _model_baseline(document)
    run_defaults = _model_run_defaults(document)
    refused_paths = {item.path for item in refused}
    # Compare what the product makes of each present value (an empty section takes
    # its defaults); fall back to the raw value when the remainder cannot validate.
    try:
        effective: object = RuntimeConfiguration.model_validate(
            _without(document, refused_paths)
        ).model_dump(mode="json")
    except ValidationError:
        effective = document
    differences = []
    for parts, value in _leaves(document):
        dotted = ".".join(parts)
        if (
            not parts
            or parts[0] in per_run
            or any(
                dotted == refused_path or dotted.startswith(refused_path + ".")
                for refused_path in refused_paths
            )
        ):
            continue
        if parts[:2] == ("agent_invocation", "model_settings") and len(parts) == 3:
            setting = AgentModelSettings.model_fields.get(parts[2])
            if parts[2] in baseline:
                default, source = baseline[parts[2]], "model_default"
            elif setting is not None:
                default, source = setting.default, "product_default"
            else:
                default, source = None, "absent"
        elif (
            parts[0] == "agent_invocation"
            and len(parts) == 2
            and parts[1] in run_defaults
        ):
            default, source = run_defaults[parts[1]], "model_default"
        elif parts == ("agent_invocation", "failed_turns"):
            # Unset, the global role default's number applies.
            default, source = ROLE_DEFAULT.failed_turns, "product_default"
        else:
            found, default = _lookup(defaults, parts)
            source = "product_default" if found else "absent"
        found, typed = _lookup(effective, parts)
        if source == "absent" or (typed if found else value) != default:
            differences.append(
                RuntimeFieldDifference(
                    path=dotted, value=value, default=default, default_source=source
                )
            )
    return RuntimeConfigDrift(
        runtime_config_file=str(path),
        byte_count=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        valid=not refused,
        refused=refused,
        stale_but_valid=tuple(differences),
        per_run=tuple(per_run_values),
        per_run_values=per_run_values,
    )


def resolve_runtime_config(
    selections: RuntimeSelections,
    environ: Mapping[str, str],
    *,
    agent_role: str,
    overrides: AgentInvocationOverrides | None = None,
    required_endpoint: str | None = None,
) -> LoadedRuntimeConfig:
    """Resolve one role from the Run's selections; no file is read.

    Registration records the resolved selections, so they are the only authority
    for every later Attempt (#440 Q9).
    """
    policy = selections.agent_invocation_policy(
        agent_role, overrides or AgentInvocationOverrides()
    )
    selected = selections.model_for_policy(policy)
    if required_endpoint is not None and selected.endpoint != required_endpoint:
        raise RuntimeConfigError(
            "runtime_config_invalid",
            ("model_roster", policy.model_reference, "endpoint"),
        )
    try:
        model = model_endpoint_policy(selected.model)
    except ModelUnlisted as error:
        raise RuntimeConfigError(
            "runtime_config_invalid",
            (f"model_roster.{policy.model_reference}.model",),
        ) from error
    credential = environ.get(selected.credential_env, "")
    if not credential:
        raise RuntimeConfigError(
            "model_credential_unavailable",
            (f"model_roster.{policy.model_reference}.credential_env",),
        )
    return LoadedRuntimeConfig(
        selections=selections,
        credential=credential,
        invocation_policy=policy,
        model=model,
    )


def model_endpoint_identity(
    model: object, selected: NebiusModelEndpointSelection
) -> NebiusModelEndpointIdentity:
    """Refuse an adapter that would contact outside the selected Run identity."""

    expected = NebiusModelEndpointIdentity(
        provider=selected.provider,
        model=selected.model,
        contacted_url=selected.endpoint,
    )
    advertised = getattr(model, "nebius_model_endpoint", expected)
    try:
        observed = NebiusModelEndpointIdentity.model_validate(advertised)
    except ValidationError as error:
        raise RuntimeConfigError(
            "runtime_config_invalid", ("nebius_model_endpoint",)
        ) from error
    if observed != expected:
        raise RuntimeConfigError("runtime_config_changed", ("nebius_model_endpoint",))
    return expected
