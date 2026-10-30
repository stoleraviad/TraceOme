"""Composition of migrated semantic definitions; persistence knows no domain names."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class SemanticFamily:
    name: str
    payloads: Mapping[str, type[BaseModel]]
    fold: Callable[[Sequence[dict]], Any]
    contact_identity: Callable[[dict], str] | None = None
    contact_validation: Callable[[Sequence], None] | None = None
    # ``step(state, event, prefix)`` folds one appended fact into the checked
    # state of its admitted prefix; it must equal ``fold([*prefix, event])``.
    step: Callable[[Any, Mapping, Sequence[Mapping]], Any] | None = None

    def admit(self, events: Sequence[dict]):
        for event in events:
            if contract := self.payloads.get(event["event_type"]):
                contract.model_validate(event["payload"])
        return self.fold(events)

    def admit_next(self, state: Any, event: Mapping, prefix: Sequence[Mapping]):
        """Admit one appended fact; a family without a step refolds its history."""
        if self.step is None:
            return self.admit([*prefix, event])
        if contract := self.payloads.get(event["event_type"]):
            contract.model_validate(event["payload"])
        return self.step(state, event, prefix)


@cache
def semantic_families(event_types: frozenset[str]) -> tuple[SemanticFamily, ...]:
    # Domain owners supply typed payloads and the same fold used by their readers.
    # Lazy composition avoids importing application/domain implementations into
    # the generic writer and keeps unknown legacy facts without invented meaning.
    from bio_neb.application.run_failure import RunFailed
    from bio_neb.investigation.investigation_request import (
        InvestigationRequestRegistered,
        project_captured_investigation_requests,
    )
    from bio_neb.journey import FORK_ORIGIN_EVENT_TYPE
    from bio_neb.journey_path import (
        PathEvent,
        execution_contact_identity,
        recorded_execution,
        step_recorded_execution,
        validate_execution_contacts,
    )
    from bio_neb.parabricks.planning_hitl import (
        PlanningHitlRequested,
        PlanningHitlResponded,
        project_planning_hitl,
    )
    from bio_neb.recording import (
        PassClosed,
        PassStarted,
        step_pass_facts,
        validate_pass_facts,
    )

    families = ()
    if FORK_ORIGIN_EVENT_TYPE in event_types:
        from bio_neb.application.fork import RunForked, fork_lineage

        families += (
            SemanticFamily(
                "fork_origin", {FORK_ORIGIN_EVENT_TYPE: RunForked}, fork_lineage
            ),
        )
    if event_types & {
        "RunInputsRegistered",
        "InputMetadataInspected",
        "InputInspectionErrored",
        "InputInspectionReviewRequested",
        "InputInspectionReviewAnswered",
        "ExecutionPathRecorded",
    }:
        from bio_neb.application.input_journey import (
            InputInspectionErrored,
            InputInspectionReviewAnswered,
            InputInspectionReviewRequested,
            InputMetadataInspected,
            RunInputsRegistered,
            project_input_selection,
        )

        families += (
            SemanticFamily(
                "input_selection",
                {
                    "RunInputsRegistered": RunInputsRegistered,
                    "InputMetadataInspected": InputMetadataInspected,
                    "InputInspectionErrored": InputInspectionErrored,
                    "InputInspectionReviewRequested": InputInspectionReviewRequested,
                    "InputInspectionReviewAnswered": InputInspectionReviewAnswered,
                    "ExecutionPathRecorded": PathEvent,
                },
                project_input_selection,
            ),
        )
    if event_types & {
        "InvestigationDownloadFailed",
        "InvestigationDownloadReviewRequested",
        "InvestigationDownloadReviewAnswered",
    }:
        from bio_neb.investigation.download_review import (
            InvestigationDownloadFailed,
            InvestigationDownloadReviewAnswered,
            InvestigationDownloadReviewRequested,
            project_download_reviews,
        )

        families += (
            SemanticFamily(
                "investigation_download",
                {
                    "InvestigationDownloadFailed": InvestigationDownloadFailed,
                    "InvestigationDownloadReviewRequested": (
                        InvestigationDownloadReviewRequested
                    ),
                    "InvestigationDownloadReviewAnswered": (
                        InvestigationDownloadReviewAnswered
                    ),
                },
                project_download_reviews,
            ),
        )
    from bio_neb.investigation.run_health import FACTS as RUN_HEALTH_FACTS

    if event_types & RUN_HEALTH_FACTS.keys():
        from bio_neb.investigation.run_health import project_run_health

        families += (
            SemanticFamily("run_health", RUN_HEALTH_FACTS, project_run_health),
        )
    from bio_neb.paper_intake.journey import PAYLOADS as PAPER_PAYLOADS

    if event_types & PAPER_PAYLOADS.keys():
        from bio_neb.paper_intake.journey import project_run_papers

        families += (SemanticFamily("run_papers", PAPER_PAYLOADS, project_run_papers),)
    from bio_neb.paper_reading.journey import PAYLOADS as READING_PAYLOADS

    if event_types & READING_PAYLOADS.keys():
        from bio_neb.paper_reading.journey import (
            project_paper_readings,
            step_paper_readings,
        )

        families += (
            SemanticFamily(
                "paper_readings",
                READING_PAYLOADS,
                project_paper_readings,
                step=step_paper_readings,
            ),
        )
    if "InvestigationAuthorityGranted" in event_types:
        from bio_neb.investigation.observation_inputs import (
            GrantContactOwner,
            fold_grant_contact_owners,
            grant_contact_identity,
            validate_investigation_contacts,
        )

        families += (
            SemanticFamily(
                "investigation_contacts",
                {"InvestigationAuthorityGranted": GrantContactOwner},
                fold_grant_contact_owners,
                grant_contact_identity,
                validate_investigation_contacts,
                step=lambda _state, _event, _prefix: None,
            ),
        )
    if event_types & {
        "AgentEffortReviewRequested",
        "AgentEffortReviewResponded",
        "AttemptStarted",
        "ModelRequestRecorded",
    }:
        families += (_agent_effort_family(),)
    if event_types & {
        "AgentWorkReviewRequested",
        "AgentWorkReviewResponded",
        "AttemptStarted",
    }:
        families += (_agent_work_family(),)
    if event_types & {
        "ParabricksConfigurationRegistered",
        "PreparationInformationRequested",
        "PreparationInformationAnswered",
    }:
        from bio_neb.parabricks.configuration import ParabricksConfigurationRegistered
        from bio_neb.parabricks.preparation_review import (
            PreparationInformationAnswered,
            PreparationInformationRequested,
            project_preparation_review,
        )

        families += (
            SemanticFamily(
                "preparation_review",
                {
                    c.model_fields["event_type"].default: c
                    for c in (
                        ParabricksConfigurationRegistered,
                        PreparationInformationRequested,
                        PreparationInformationAnswered,
                    )
                },
                project_preparation_review,
            ),
        )
    return families + (
        SemanticFamily(
            "execution",
            {"ExecutionPathRecorded": PathEvent},
            recorded_execution,
            execution_contact_identity,
            validate_execution_contacts,
            step_recorded_execution,
        ),
        SemanticFamily(
            "planning_review",
            {
                "PlanningHitlRequested": PlanningHitlRequested,
                "PlanningHitlResponded": PlanningHitlResponded,
                "RunFailed": RunFailed,
            },
            project_planning_hitl,
        ),
        SemanticFamily(
            "investigation_request",
            {"InvestigationRequestRegistered": InvestigationRequestRegistered},
            project_captured_investigation_requests,
        ),
        SemanticFamily(
            "processing_pass",
            {"PassStarted": PassStarted, "PassClosed": PassClosed},
            validate_pass_facts,
            step=step_pass_facts,
        ),
    )


@cache
def _agent_effort_family():
    from bio_neb.agentic_journey.attempt_journey import (
        AttemptStarted,
        ModelRequestRecorded,
    )
    from bio_neb.agentic_journey.effort import (
        AgentEffortReviewRequested,
        AgentEffortReviewResponded,
        project_effort_review,
    )

    return SemanticFamily(
        "agent_effort",
        {
            "AgentEffortReviewRequested": AgentEffortReviewRequested,
            "AgentEffortReviewResponded": AgentEffortReviewResponded,
            "AttemptStarted": AttemptStarted,
            "ModelRequestRecorded": ModelRequestRecorded,
        },
        project_effort_review,
    )


@cache
def _agent_work_family():
    from bio_neb.agentic_journey.agent_work_review import (
        AgentWorkReviewRequested,
        AgentWorkReviewResponded,
        project_agent_work_review,
    )
    from bio_neb.agentic_journey.attempt_journey import AttemptStarted

    return SemanticFamily(
        "agent_work",
        {
            "AgentWorkReviewRequested": AgentWorkReviewRequested,
            "AgentWorkReviewResponded": AgentWorkReviewResponded,
            "AttemptStarted": AttemptStarted,
        },
        project_agent_work_review,
    )


@dataclass(frozen=True)
class AppendCheck:
    """A fact's own reopen check, run on the prefix plus the fact before the fact
    is appended (owner, #476 C8; ADR 0032: no unchecked append). ``check`` reads the
    owner's prefix, the candidate and the owner (whose once-only derivations its
    reader shares) and raises ``ValueError`` with its reason."""

    name: str
    check: Callable[[Sequence[Mapping], Mapping, Any], None]


@cache
def append_checks(event_type: str) -> tuple[AppendCheck, ...]:
    # The checks are their readers' own, composed lazily like the families.
    if event_type in {"RefSnpToolResultRecorded", "AssemblyReportToolResultRecorded"}:
        from bio_neb.investigation.investigation import check_source_observation

        return (AppendCheck("source_observation_reopens", check_source_observation),)
    if event_type == "InvestigationLookupDecided":
        from bio_neb.investigation.investigation import check_lookup_decision

        return (AppendCheck("lookup_decision_reopens", check_lookup_decision),)
    if event_type == "JourneySearchRead":
        from bio_neb.agentic_journey.assignment_context import reread_search_receipt

        return (
            AppendCheck("search_receipt_rereads_its_prefix", reread_search_receipt),
        )
    return ()
