"""Specific preparation information through the shared exact-answer junction."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictStr

from bio_neb.contracts import FrozenContract
from bio_neb.junction import evaluate_scoped_answer
from bio_neb.parabricks.configuration import (
    ParabricksConfiguration,
    current_configuration,
)


class SupplyPreparationInformation(FrozenContract):
    action: Literal["supply"] = "supply"
    configuration: ParabricksConfiguration


class StopPreparation(FrozenContract):
    action: Literal["stop"] = "stop"
    reason: StrictStr = Field(min_length=1)


PreparationResponse = Annotated[
    SupplyPreparationInformation | StopPreparation, Field(discriminator="action")
]


class PreparationInformationRequested(FrozenContract):
    event_type: Literal["PreparationInformationRequested"] = (
        "PreparationInformationRequested"
    )
    attempt_id: StrictStr
    code: StrictStr
    missing_fields: tuple[StrictStr, ...]
    evidence_event_ids: tuple[StrictStr, ...]
    context: dict[str, JsonValue]


class PreparationInformationAnswered(FrozenContract):
    event_type: Literal["PreparationInformationAnswered"] = (
        "PreparationInformationAnswered"
    )
    request_event_id: StrictStr
    response: PreparationResponse


class PreparationLegalAction(FrozenContract):
    action: str
    response_schema: dict


class PreparationRequest(PreparationInformationRequested):
    request_event_id: StrictStr
    pending: bool
    legal_actions: tuple[PreparationLegalAction, ...] = ()


class PreparationReceipt(FrozenContract):
    request_event_id: StrictStr
    response_event_id: StrictStr
    response: PreparationResponse


class PreparationReview(FrozenContract):
    requests: tuple[PreparationRequest, ...] = ()
    receipts: tuple[PreparationReceipt, ...] = ()
    stopped: bool = False


class PreparationAnswerRefused(ValueError):
    def __init__(self, reason, request_event_id):
        self.code = "preparation_answer_" + reason
        self.request_event_id = request_event_id
        from bio_neb.agentic_journey.refusal_disposition import RefusalDisposition

        self.disposition = RefusalDisposition(kind="caller_correction_and_reissue")
        super().__init__(self.code)


def information_gap(events):
    """Pure configuration completeness; domain scientific readiness stays separate."""
    _, configuration = current_configuration(events)
    if configuration is None:
        return "execution_configuration_required", ("launch", "runtime_selection")
    from bio_neb.parabricks.execution_proposal import ProposalCompilationRefused
    from bio_neb.parabricks.plan import ExecutionCompositionRefused

    try:
        validate_information(events, configuration)
    except ProposalCompilationRefused:
        # Planning readiness checks planning completeness first; what remains is
        # the compiler's integrity refusal, which holds the Run when it compiles.
        return None
    except ExecutionCompositionRefused as error:
        return "execution_configuration_incompatible", error.details
    return None


def validate_information(events, configuration):
    """Validate supplied consequences without writing or contacting a source."""
    from bio_neb.application.input_journey import (
        paired_fastq_roles,
        project_input_selection,
    )
    from bio_neb.parabricks.execution_proposal import (
        _compile_agent_planned_execution_proposal,
    )
    from bio_neb.parabricks.plan import ExecutionCompositionRefused

    selected_inputs = project_input_selection(events)
    planned = _compile_agent_planned_execution_proposal(events)
    if not paired_fastq_roles(selected_inputs.fastq_roles):
        raise ExecutionCompositionRefused("fastq_roles", "paired_inputs")
    runtime = configuration.runtime_selection.resolve().h100
    if len(planned.recipes) != 1 or len(planned.resources) != 1:
        raise ExecutionCompositionRefused("lineage", "runtime_capability")
    recipe, resource = planned.recipes[0].candidate, planned.resources[0].candidate
    if (
        runtime.recipe_id,
        runtime.resource_profile_id,
        runtime.gpu_count,
        runtime.cpu_count,
        runtime.memory_gib,
        runtime.accelerator_memory_gib,
    ) != (
        recipe.recipe_id,
        resource.profile_id,
        resource.accelerator_count,
        resource.cpu_count,
        resource.memory_gib,
        resource.accelerator_memory_gib,
    ):
        raise ExecutionCompositionRefused("lineage", "runtime_capability")
    _validate_runtime_extension(
        events, configuration.runtime_selection.capability, resource
    )


def _validate_runtime_extension(events, capability, resource):
    """Adding runtime profiles preserves the exact earlier capability content.

    The original snapshot may lack profiles and the Python entrypoint claim.
    Remove only those explicit additions and reproduce its retained digest; no
    recipe, resource, source, existing tool or image claim may change.
    """
    from bio_neb.content import canonical_json_digest
    from bio_neb.parabricks.plan import ExecutionCompositionRefused
    from bio_neb.parabricks.recipe_resources import CapabilitySource

    if capability.snapshot_id == resource.snapshot_id:
        return
    observed = next(
        e["payload"]["result"]["source"]
        for e in reversed(events)
        if e["event_type"] == "ResourceObservationRecorded"
        and any(
            c["candidate_id"] == resource.candidate_id
            for c in e["payload"]["result"]["candidates"]
        )
    )
    source = CapabilitySource.model_validate(observed)
    retained_tools = {tool.tool_id: tool for tool in source.tools}
    supplied_tools = {tool.tool_id: tool for tool in capability.tools}
    images = {image.image_id: image for image in source.images}
    extra = [tool for tool in capability.tools if tool.tool_id not in retained_tools]
    if (
        capability.images != source.images
        or any(
            supplied_tools.get(name) != claim for name, claim in retained_tools.items()
        )
        or any(
            tool.tool_id != "python"
            or tool.execution_scope != "worker_image"
            or tool.image_id not in images
            or tool.absolute_path != images[tool.image_id].worker_entrypoint[0]
            for tool in extra
        )
    ):
        raise ExecutionCompositionRefused(
            "lineage", "runtime_capability_evidence_changed"
        )
    original = capability.model_dump(
        mode="json", exclude={"snapshot_id", "runtime_profiles"}
    )
    original["tools"] = [tool.model_dump(mode="json") for tool in source.tools]
    for claim in original["resources"]:
        claim["tool_ids"] = [
            name for name in claim["tool_ids"] if name in retained_tools
        ]
    if canonical_json_digest(original) != resource.snapshot_id:
        raise ExecutionCompositionRefused(
            "lineage", "runtime_capability_evidence_changed"
        )


def preparation_context(events) -> dict[str, JsonValue]:
    """Expose exact selected information the caller must match when supplying settings."""
    from bio_neb.parabricks.execution_proposal import (
        _compile_agent_planned_execution_proposal,
    )

    planned = _compile_agent_planned_execution_proposal(events)
    return {
        "reference": planned.reference.model_dump(mode="json"),
        "companions": [c.model_dump(mode="json") for c in planned.companions],
        "recipes": [r.model_dump(mode="json") for r in planned.recipes],
        "resources": [r.model_dump(mode="json") for r in planned.resources],
        "capability_sources": list(
            {
                e["payload"]["result"]["source"]["snapshot_id"]: e["payload"]["result"][
                    "source"
                ]
                for e in events
                if e["event_type"]
                in {"RecipeObservationRecorded", "ResourceObservationRecorded"}
                and e["payload"]["result"]["source"]["snapshot_id"]
                in {r.candidate.snapshot_id for r in planned.resources}
            }.values()
        ),
    }


def open_preparation_review(journey, gap) -> None:
    """The coordinator asks for the preparation information readiness lacks (#301).

    Planning closed with only this gap left, so no model turn shapes the question:
    it carries the gap's code and fields, the plan as evidence and the exact
    selected information. It names the planning Attempt whose closure it follows.
    """
    from bio_neb.application.planning_read import current_planning, planning_evidence

    if not gap.preparation_information:
        raise ValueError("only preparation information is asked through this Review")
    events = planning_evidence(journey)
    closed = next(
        e
        for e in reversed(events)
        if e["event_type"] == "SliceCheckpointReached"
        and e["payload"]["checkpoint"] == "execution_planning_ready"
    )
    fact = PreparationInformationRequested(
        attempt_id=closed["payload"]["attempt_id"],
        code=gap.code,
        missing_fields=gap.details,
        evidence_event_ids=(current_planning(journey).plan_event_id,),
        context=preparation_context(events),
    )
    journey.record(
        fact.event_type,
        fact.model_dump(mode="json", exclude={"event_type"}),
        [journey.head_event_id],
    )


def _check_coordinator_lineage(fact, event, prefix):
    """The coordinator asks right after the planning Attempt that closed ready (#301).

    The request names that Attempt, cites the current plan and descends from the
    causal head: the last fact before it, a lazily started Pass aside.
    """
    # An Attempt that only read a researcher's target answer is set aside: it
    # reopens no execution planning.
    reads = {
        e["payload"]["attempt_id"]
        for e in prefix
        if e["event_type"] == "SliceCheckpointReached"
        and e["payload"]["checkpoint"] == "target_answer_read"
    }
    started = next(
        (
            e
            for e in reversed(prefix)
            if e["event_type"] == "AttemptStarted"
            and e["payload"]["attempt_id"] not in reads
        ),
        None,
    )
    if (
        started is None
        or started["payload"]["attempt_id"] != fact.attempt_id
        or not any(
            e["event_type"] == "SliceCheckpointReached"
            and e["payload"]["attempt_id"] == fact.attempt_id
            and e["payload"]["checkpoint"] == "execution_planning_ready"
            for e in prefix
        )
    ):
        raise ValueError(
            "preparation request does not follow its closing planning Attempt"
        )
    plan = next(
        (
            e["event_id"]
            for e in reversed(prefix)
            if e["event_type"] == "ProposalRequirementsIdentified"
        ),
        None,
    )
    if plan is None or fact.evidence_event_ids != (plan,):
        raise ValueError("preparation request does not cite the current plan")
    head = next(
        (e["event_id"] for e in reversed(prefix) if e["event_type"] != "PassStarted"),
        None,
    )
    if event["parent_ids"] != [head]:
        raise ValueError("preparation request does not descend from its causal head")


def project_preparation_review(events):
    timeline = list(events)
    requests, receipts = [], []
    by_id = {}
    for index, event in enumerate(timeline):
        kind = event["event_type"]
        if kind == "ParabricksConfigurationRegistered":
            from bio_neb.parabricks.configuration import (
                ParabricksConfigurationRegistered,
            )

            fact = ParabricksConfigurationRegistered.model_validate(event["payload"])
            if any(e["event_type"] == kind for e in timeline[:index]) or not any(
                by_id.get(p, {}).get("event_type") == "RunInputsRegistered"
                for p in event["parent_ids"]
            ):
                raise ValueError("execution configuration lacks its exact registration")
        elif kind == "PreparationInformationRequested":
            fact = PreparationInformationRequested.model_validate(event["payload"])
            if (
                information_gap(timeline[:index]) != (fact.code, fact.missing_fields)
                or fact.context != preparation_context(timeline[:index])
                or any(r.pending for r in requests)
            ):
                raise ValueError(
                    "preparation request changed its current information gap"
                )
            _check_coordinator_lineage(fact, event, timeline[:index])
            actions = tuple(
                PreparationLegalAction(action=a, response_schema=c.model_json_schema())
                for a, c in (
                    ("supply", SupplyPreparationInformation),
                    ("stop", StopPreparation),
                )
            )
            requests.append(
                PreparationRequest(
                    **fact.model_dump(),
                    request_event_id=event["event_id"],
                    pending=True,
                    legal_actions=actions,
                )
            )
        elif kind == "PreparationInformationAnswered":
            fact = PreparationInformationAnswered.model_validate(event["payload"])
            request, replay = evaluate_scoped_answer(
                requests,
                receipts,
                fact.request_event_id,
                fact.response,
                validate=lambda r, a, index=index: validate_information(
                    timeline[:index], a.configuration
                )
                if isinstance(a, SupplyPreparationInformation)
                else None,
                refusal=PreparationAnswerRefused,
            )
            if replay is not None or fact.request_event_id not in event["parent_ids"]:
                raise ValueError("preparation answer lacks unique request ancestry")
            requests = [
                r.model_copy(update={"pending": False})
                if r.request_event_id == request.request_event_id
                else r
                for r in requests
            ]
            receipts.append(
                PreparationReceipt(
                    request_event_id=fact.request_event_id,
                    response_event_id=event["event_id"],
                    response=fact.response,
                )
            )
        by_id[event["event_id"]] = event
    return PreparationReview(
        requests=tuple(requests),
        receipts=tuple(receipts),
        stopped=bool(receipts and isinstance(receipts[-1].response, StopPreparation)),
    )


def answer_preparation(journey_file, request_event_id, response):
    from bio_neb.journey import Journey

    journey = Journey.open(Path(journey_file))
    events = journey.read()
    view = project_preparation_review(events)
    _, replay = evaluate_scoped_answer(
        view.requests,
        view.receipts,
        request_event_id,
        response,
        validate=lambda r, a: validate_information(events, a.configuration)
        if isinstance(a, SupplyPreparationInformation)
        else None,
        refusal=PreparationAnswerRefused,
    )
    if replay is not None:
        return replay
    fact = PreparationInformationAnswered(
        request_event_id=request_event_id, response=response
    )
    journey.record(
        fact.event_type,
        fact.model_dump(mode="json", exclude={"event_type"}),
        [request_event_id, events[-1]["event_id"]],
    )
    return project_preparation_review(journey.read()).receipts[-1]
