"""Reusable scientific reference-query Journey builders."""

from __future__ import annotations

import uuid
from pathlib import Path

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.agentic_journey.attempt_journey import (
    AttemptStarted,
    ModelPhysicalContactStarted,
    ModelRequestRecorded,
    ModelResponseRecorded,
)
from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationDefaults,
    AgentInvocationOverrides,
    resolve_agent_invocation_policy,
)
from bio_neb.agentic_journey.lifecycle import (
    ActivityStarted,
    ActivitySucceeded,
)
from bio_neb.agentic_journey.runtime_config import RuntimeSelections
from bio_neb.application.planning_read import read_planning
from bio_neb.failure import FailureSurface, policy_for
from bio_neb.input_inspection.commands import (
    InspectRegisteredInputsCommand,
    inspect_registered_inputs,
)
from bio_neb.journey import Journey
from bio_neb.parabricks.planning_commands import (
    RespondPlanningHitlCommand,
    respond_planning_hitl,
)
from bio_neb.parabricks.planning_hitl import SelectPlanningHitlResponse
from bio_neb.recording import read_evidence
from bio_neb.reference_discovery.reference_identity_tool import (
    AssemblyAdapterObservation,
)
from tests.application_fakes import _registered
from tests.model_endpoint_fakes import corrections
from tests.planning_assignment_fakes import PlanningComposition, eligible_tools
from tests.reference_fakes import (
    CompanionAdapter,
    ReferenceAdapter,
    payload,
    reference_const_arguments,
    reference_source,
)


def identity(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"reference-query:{label}"))


def append(timeline: list[dict[str, object]], label: str, fact) -> dict[str, object]:
    event = {
        "journey_id": timeline[0]["journey_id"],
        "event_id": identity(label),
        "event_type": fact.event_type,
        "parent_ids": [timeline[-1]["event_id"]],
        "payload": payload(fact),
    }
    timeline.append(event)
    return event


def registered_attempt_start(timeline, *, attempt_id, agent_role, input_set_id):
    registration = next(
        event for event in timeline if event["event_type"] == "RunInputsRegistered"
    )
    selections = RuntimeSelections.model_validate(
        registration["payload"]["runtime_selections"]
    )
    effective = resolve_agent_invocation_policy(
        AgentInvocationDefaults.model_validate(
            registration["payload"]["runtime_selections"]["agent_invocation"]
        ),
        AgentInvocationOverrides(),
        agent_role=agent_role,
        model_settings={
            name: selected.settings_selection
            for name, selected in selections.model_roster.items()
        },
        model_contact=policy_for(
            FailureSurface(scope="role", name=agent_role), selections.failure
        ),
    )
    return AttemptStarted(
        attempt_id=attempt_id,
        agent_role=agent_role,
        input_set_id=input_set_id,
        continuation_policy=effective.continuation,
        continuation_policy_id=effective.continuation.policy_id,
        effective_invocation_policy=effective,
        invocation_policy_id=effective.policy_id if effective else None,
        invocation_overrides={} if effective else None,
    )


def add_model_contact(
    timeline: list[dict[str, object]],
    label: str,
    *,
    attempt_id: str,
    request_number: int,
    tool_name: str,
    tool_call_id: str,
    arguments: dict[str, object],
    messages: list[dict[str, object]] | None = None,
    nebius_model_endpoint: dict[str, str] | None = None,
) -> dict[str, object]:
    activity_id = identity(f"{label}-activity")
    invocation_policy = next(
        (
            event["payload"].get("effective_invocation_policy")
            for event in timeline
            if event["event_type"] == "AttemptStarted"
            and event["payload"]["attempt_id"] == attempt_id
        ),
        None,
    )
    model_settings = (
        {
            key: value
            for key, value in invocation_policy["model_settings"].items()
            if value is not None
        }
        if invocation_policy
        else {}
    )
    registered = next(
        (event for event in timeline if event["event_type"] == "RunInputsRegistered"),
        None,
    )
    if registered is not None:
        selected = registered["payload"]["runtime_selections"]["model_roster"]["qwen"]
        endpoint_evidence = {
            "provider": selected["provider"],
            "model": selected["model"],
            "contacted_url": selected["endpoint"],
        }
    elif nebius_model_endpoint is not None:
        endpoint_evidence = nebius_model_endpoint
    else:
        raise ValueError("synthetic model contact requires explicit Nebius identity")
    append(
        timeline,
        f"{label}-start",
        ActivityStarted(
            activity_id=activity_id,
            adapter="pydantic_ai_model",
            stage="model_contact",
            attempt=1,
        ),
    )
    request = append(
        timeline,
        f"{label}-request",
        ModelRequestRecorded(
            attempt_id=attempt_id,
            request_number=request_number,
            nebius_model_endpoint=endpoint_evidence,
            compatible_adapter_evidence={
                "model_name": endpoint_evidence["model"],
                "model_system": "function",
                "request": {
                    "messages": messages or [],
                    "model_settings": model_settings,
                    "function_tools": [
                        {
                            "name": tool_name,
                            "parameters_json_schema": {"type": "object"},
                        }
                    ],
                    "output_tools": [],
                    "allow_text_output": False,
                    "output_mode": "tool",
                    "instruction_parts": None,
                },
            },
        ),
    )
    physical = append(
        timeline,
        f"{label}-physical-1",
        ModelPhysicalContactStarted(
            attempt_id=attempt_id,
            request_number=request_number,
            logical_request_event_id=request["event_id"],
            contact_ordinal=1,
        ),
    )
    response = append(
        timeline,
        f"{label}-response",
        ModelResponseRecorded(
            attempt_id=attempt_id,
            request_number=request_number,
            physical_contact_ordinal=1,
            physical_contact_started_event_id=physical["event_id"],
            completion_certainty="response_received",
            policy_disposition="response_admitted",
            nebius_model_endpoint=endpoint_evidence,
            compatible_adapter_evidence={
                "model_name": endpoint_evidence["model"],
                "model_system": "function",
                "response": {
                    "message": {
                        "parts": [
                            {
                                "part_kind": "tool-call",
                                "tool_name": tool_name,
                                "tool_call_id": tool_call_id,
                                "args": arguments,
                            }
                        ]
                    }
                },
            },
        ),
    )
    append(
        timeline,
        f"{label}-succeeded",
        ActivitySucceeded(
            activity_id=activity_id,
            adapter="pydantic_ai_model",
            stage="model_contact",
            attempt=1,
            elapsed_ms=1,
            result_event_id=response["event_id"],
        ),
    )
    return response


class RefiningPlanning(PlanningComposition):
    """Refine an ambiguous discovery by a retained isolate before choosing."""

    async def respond(self, messages, info):
        view = read_planning(Journey.open(Path(self.journey.path))).reference
        tools = eligible_tools(self.journey, info.function_tools)
        discoveries = [
            f
            for f in (() if view is None else view.facts)
            if f.event_type == "ReferenceAssembliesRecorded"
        ]
        if (
            "discover_assemblies" in tools
            and len(discoveries) == 1
            and len(discoveries[0].result["rows"]) > 1
        ):
            self.calls.append("discover_assemblies")
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "discover_assemblies",
                        {
                            "taxid": view.resolved_taxon.ncbi_tax_id,
                            "search_text": ["isolate-B"],
                        },
                        tool_call_id=f"assignment-{len(self.calls)}",
                    )
                ]
            )
        return await super().respond(messages, info)


class ChoosingPlanning(PlanningComposition):
    """Choose from the first discovery, then ask the reference question when offered.

    Live planning models asked the reference question after their own concrete
    choice whenever it was offered (#153 research note G4); this one asks it
    whenever its preconditions hold. ``candidate`` is
    ``"first"`` or ``"none"`` (explicit ambiguity); ``broken_inspections`` sends that
    many exploratory inspections of an accession it did not choose, which cannot verify its basis.
    """

    candidate = "first"
    reason = "Neither isolate is documented for the stated sample."
    broken_inspections = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chose = False
        # The admissible Tool names of each request after the Agent's choice.
        self.offers_after_choice = []
        # The admission feedback each request carries, newest last.
        self.feedback = []

    async def respond(self, messages, info):
        self.feedback.extend(corrections(messages[-1]))
        tools = eligible_tools(self.journey, info.function_tools)
        if self.chose:
            self.offers_after_choice.append(frozenset(tools))
            if "request_planning_hitl" in tools:
                name = "request_planning_hitl"
                self.calls.append(name)
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            name,
                            reference_const_arguments(tools[name]),
                            tool_call_id=f"assignment-{len(self.calls)}",
                        )
                    ]
                )
        response = await super().respond(messages, info)
        arguments = response.parts[0].args
        if self.calls[-1] == "record_reference_choice":
            self.chose = True
            if self.candidate == "none":
                arguments.update(assembly_accession="none", reason=self.reason)
                arguments.pop("quote", None)
        elif self.calls[-1] == "inspect_reference" and self.broken_inspections:
            self.broken_inspections -= 1
            arguments["accession"] = "GCF_000000001.1"
        return response


class UnverifiedReferenceAdapter(ReferenceAdapter):
    """Discovery returns two isolates; exact inspection finds no assembly record."""

    def __init__(self) -> None:
        super().__init__(ambiguous=True)

    def describe_assembly(self, assembly_accession: str) -> AssemblyAdapterObservation:
        self.contact_count += 1
        return AssemblyAdapterObservation(
            records=(),
            source=reference_source(
                "ncbi_datasets", f"genome/accession/{assembly_accession}"
            ),
            total_count=0,
        )


def planning_run(root: Path, composition_type=RefiningPlanning):
    """Register and inspect one Run whose reference source offers two isolates."""
    receipt, paths = _registered(root)
    journey = Journey.open(Path(receipt.journey_file))
    inspect_registered_inputs(
        InspectRegisteredInputsCommand(journey_file=str(journey.path)), journey=journey
    )
    composition = composition_type(receipt, paths)
    composition.reference_adapter = ReferenceAdapter(ambiguous=True)
    composition.companion_adapter = CompanionAdapter("GCF_000009999.1")
    return receipt.journey_file, composition


def run_planning_attempt(journey_file: str, composition):
    journey = Journey.open(Path(journey_file))
    return composition.planning_assignment(journey).run(
        journey, environ=composition.environ
    )


def select_first_candidate(journey_file: str, request):
    return respond_planning_hitl(
        RespondPlanningHitlCommand(
            journey_file=journey_file,
            request_event_id=request.request_event_id,
            response=SelectPlanningHitlResponse(
                action="select", candidate_id=request.candidates[0].candidate_id
            ),
        )
    )


def build_query_timeline(root: Path):
    """Retain two real planning Attempts around one answered scoped question.

    The timeline is the merged evidence a reader replays; stages are its prefixes.
    """
    journey_file, composition = planning_run(root)
    pending = run_planning_attempt(journey_file, composition)
    stages = {"pending": len(read_evidence(Journey.open(Path(journey_file))))}
    select_first_candidate(journey_file, pending.planning_hitl.pending_request)
    stages["responded"] = len(read_evidence(Journey.open(Path(journey_file))))
    run_planning_attempt(journey_file, composition)
    timeline = read_evidence(Journey.open(Path(journey_file)))
    stages["final"] = len(timeline)
    return journey_file, timeline, stages
