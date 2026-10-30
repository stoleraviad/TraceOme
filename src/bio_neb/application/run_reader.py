"""Tolerant, contact-free reads of one retained Run for every outside consumer.

The terminal, the GUI, run health and eval scoring read a Run here instead of re-reading raw
events: the Journey feed, one plain-language item per retained event, and the Run artifacts, a
completed Execution's verified objects with their recorded proofs. Neither read writes,
decides or contacts anything; Run status, pending Reviews and authority stay with the checked
`observe`.

The feed is a presentation of retained events, not the checked view: it reads the complete
lines of one `journey.jsonl` and leaves a half-written last line for the next read.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import Field, JsonValue, StrictInt, StrictStr, ValidationError

from bio_neb.content import ContentId, canonical_json
from bio_neb.contracts import FrozenContract
from bio_neb.journey import (
    FORK_ORIGIN_EVENT_TYPE,
    JourneyEvent,
    JourneyValidationError,
)
from bio_neb.journey_path import ArtifactPointer
from bio_neb.parabricks.result import (
    ParabricksExecutionIncomplete,
    ParabricksResultUnavailable,
    read_completed_parabricks_result,
)
from bio_neb.providers.object_locations import object_location

# The product path's stages (PROJECT.md), in Run order.
FeedStage = Literal[
    "request",
    "planning",
    "execution_review",
    "execution",
    "investigation",
    "finalization",
    "report",
]
# The events that begin a stage. Every other event belongs to the stage it occurs in,
# so Agent and Pass mechanics read as part of the work they serve.
_STAGE_STARTS: dict[str, FeedStage] = {
    "JourneyStarted": "request",
    FORK_ORIGIN_EVENT_TYPE: "request",
    "ResearchRequestRegistered": "request",
    "ProposalRequirementsIdentified": "planning",
    "ExecutionProposalSubmitted": "execution_review",
    "ExecutionPathRecorded": "execution",
    "InvestigationStarted": "investigation",
    "FinalizationBasisCompiled": "finalization",
    "GroundedReportPublished": "report",
}


# How much of the story a view shows: key milestones and decisions, concrete steps,
# quiet bookkeeping, noisy model and activity mechanics. The default view shows key
# and step items. This table is data to tune; an event type it does not name is
# quiet.
FeedLevel = Literal["key", "step", "quiet", "noise"]
_DEFAULT_LEVELS: frozenset[FeedLevel] = frozenset({"key", "step"})
_LEVELS: dict[FeedLevel, frozenset[str]] = {
    "key": frozenset(
        {
            "JourneyStarted",
            FORK_ORIGIN_EVENT_TYPE,
            "ResearchRequestRegistered",
            "InvestigationRequestRegistered",
            "ProposalRequirementsIdentified",
            "PlanningHitlRequested",
            "PlanningHitlResponded",
            "PreparationInformationRequested",
            "PreparationInformationAnswered",
            "InputInspectionReviewRequested",
            "InputInspectionReviewAnswered",
            "PaperRecorded",
            "PaperIntakeRefused",
            "PaperReviewRequested",
            "PaperReviewAnswered",
            "PaperAnswerRecorded",
            "ExecutionProposalSubmitted",
            "ExecutionHitlResponded",
            "ExecutionPathRecorded",
            "InvestigationStarted",
            "RefSnpToolResultRecorded",
            "InvestigationProposalRecorded",
            "InvestigationReviewRequested",
            "InvestigationReviewResponded",
            "InvestigationAuthorityGranted",
            "InvestigationDownloadFailed",
            "InvestigationDownloadReviewRequested",
            "InvestigationDownloadReviewAnswered",
            "InvestigationStopped",
            "RunHealthMeasured",
            "RunHealthMeasurementFailed",
            "RunHealthReviewRequested",
            "RunHealthReviewAnswered",
            "FinalizationBasisCompiled",
            "FinalizationReviewRecorded",
            "GroundedReportPublished",
            "ReportExplanationRequested",
            "ReportExplanationPublished",
            "AgentFailureReviewRequested",
            "AgentFailureReviewResponded",
            "AgentFailureRetryPreset",
            "AgentEffortReviewRequested",
            "AgentEffortReviewResponded",
            "AgentWorkReviewRequested",
            "AgentWorkReviewResponded",
            "ModelAvailabilityReviewRequested",
            "ModelAvailabilityReviewResponded",
            "ModelAvailabilityStopped",
            "RunFailed",
            "RuntimeConfigurationFailed",
            "AttemptFailed",
            "InvestigationAttemptFailed",
            "InvestigationTableChunkFailed",
            "InputInspectionErrored",
            "OperationalFailureRecorded",
            "SourceOperationFailureRecorded",
            "SourceContactLimitReached",
            "ToolOperationFailureRecorded",
            "ToolExecutionFailed",
            "ObservedToolExecutionFailed",
            "ToolExecutionRefused",
            "ModelRequestFailed",
            "ModelRequestRejected",
            "ModelPhysicalContactFailed",
            "InvestigationSourceContactFailed",
        }
    ),
    "step": frozenset(
        {
            "RunInputsRegistered",
            "PaperFileRegistered",
            "ParabricksConfigurationRegistered",
            "InputMetadataInspected",
            "SampleAttestationRecorded",
            "TaxonResolutionRecorded",
            "ReferenceAssembliesRecorded",
            "ReferenceChoiceRecorded",
            "ReferenceDescriptionRecorded",
            "PaperSearchRecorded",
            "PaperRetrievalRecorded",
            "PaperReadingStarted",
            "ReadingPlanKept",
            "ReadingFindingJudged",
            "ReadingItemCompleted",
            "ReadingItemNotRead",
            "CompanionObservationRecorded",
            "PrebuiltIndexesListed",
            "PrebuiltIndexRead",
            "PrebuiltIndexChosen",
            "AssemblyWithIndexSuggested",
            "RecipeObservationRecorded",
            "PlanningCandidateSelected",
            "PlanningChoiceReconsidered",
            "ResourceObservationRecorded",
            "ReferenceSourceAdmitted",
            "ExecutionPackageCompiled",
            "InvestigationSourceRetrieved",
            "AssemblyReportToolResultRecorded",
            "InvestigationWorkItemStarted",
            "InvestigationEvidenceRecorded",
            "InvestigationCollectionEvidenceRecorded",
            "InvestigationItemWindowRecorded",
            "InvestigationItemCoverageRecorded",
            "InvestigationTableChunkRead",
            "ReportExplanationFailed",
            "ReportExplanationMemberAdmitted",
            "ReportExplanationCombinationBound",
            "InvestigationItemEvidenceCollected",
            "InvestigationCollectionRevisionRecorded",
            "InvestigationProposalCarriedForward",
            "InvestigationAssertionRecorded",
            "InvestigationPolicyAmended",
            "ArtifactEnrichmentRetained",
        }
    ),
    "noise": frozenset(
        {
            "InvestigationMemberContextRecorded",
            "ModelRequestRecorded",
            "ModelResponseRecorded",
            "ModelTransportRequestRecorded",
            "ModelTransportResponseRecorded",
            "ModelPhysicalContactStarted",
            "ModelPhysicalContactCancelled",
            "activity_started",
            "activity_progress",
            "activity_heartbeat",
            "activity_succeeded",
            "activity_failed",
            "activity_cancelled",
        }
    ),
}


def _level(event: JourneyEvent) -> FeedLevel:
    # An Agent turn is noise unless it was refused: then it presents a correction.
    if event.event_type == "AgentActionTurnDispositionRecorded":
        refused = _string(event.payload["disposition"]).endswith("_refused")
        return "quiet" if refused else "noise"
    # A routed reply is noise unless it had faults: then it presents its note.
    if event.event_type == "AgentTurnRouted":
        return "quiet" if event.payload["faults"] else "noise"
    return next(
        (level for level, types in _LEVELS.items() if event.event_type in types),
        "quiet",
    )


# What an item says. Each event type has a status and one line, fixed or read by a detail
# from fields its current payload contract requires. An event type the table does not
# name is told by its name; a known type's payload outside its current contract is
# refused (retained Runs are evidence, not compatibility constraints).
FeedStatus = Literal["done", "question", "answer", "verified", "failed"]
_Detail = Callable[[Mapping[str, Any]], str]


class _Wording(NamedTuple):
    status: FeedStatus
    text: str | _Detail
    # The status a payload decides, for a type that records success and failure.
    status_of: Callable[[Mapping[str, Any]], FeedStatus] | None = None


def _string(value: Any) -> str:
    # A text field holding another type is outside its contract (refused in _told).
    if not isinstance(value, str):
        raise TypeError(f"expected a string, not {type(value).__name__}")
    return value


def _clip(text: str, limit: int = 70) -> str:
    """Free text on one line, without its trailing period, at most `limit` long."""
    line = " ".join(_string(text).split()).rstrip(".")
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def _count(number: int, one: str, many: str) -> str:
    return f"{number} {one if number == 1 else many}"


def _step(name: str) -> str:
    return _string(name).replace("_", " ").capitalize()


def _variant(target: Mapping[str, Any]) -> str:
    from bio_neb.investigation.vcf_observation import allele_text

    return _clip(
        f"{target['contig']}:{target['one_based_position']} "
        f"{allele_text(target['ref'], target['alt'])}",
        40,
    )


def _paper_name(payload) -> str:
    """A recorded paper's title as its source copied it, else its citation or file."""
    title = payload["ids"]["title"] if payload["ids"] is not None else None
    return title or payload["reference"] or payload["paper_file"]


# Why the Run's health was not measured, in a feed line's words, for every code
# `run_health.HealthNotMeasuredCode` declares.
HEALTH_NOT_MEASURED_WORDS = {
    "bam_unavailable": "the run holds no local BAM",
    "measurement_failed": "its measurement failed",
}
_STEP_ANSWER = {
    "retry": "You chose to retry the failed step",
    "abort": "You chose to abandon the failed step",
    "approve_cpu_preparation": "You approved preparing the reference on CPU",
    "amend": "You changed a setting for the failed step",
    "stop": "You chose to stop after the failed step",
    "continue": "You chose to go on without it",
}
_DOWNLOAD_FILE = {"VCF": "variant calls (VCF)", "BAM": "aligned reads (BAM)"}
# What not retrying means for each file.
_DOWNLOAD_QUESTION = {
    "VCF": "variant calls (VCF)? Without it, the approved variant checks fail.",
    "BAM": "aligned reads (BAM), or go on without it?",
}
_DOWNLOAD_ANSWER = {
    "retry": "You chose to download the file again",
    "abort": "You chose not to download the file again",
}
_CPU_REVIEW_ANSWER = {
    "abort": "You chose to stop instead of preparing the reference",
    "approve_cpu_preparation": "You approved preparing the reference on CPU",
    "amend": "You corrected the reference setup",
}
_PLANNING_ANSWER = {
    "confirm": "You confirmed the suggested answer",
    "select": "You chose one of the offered options",
    "clarify": "You gave the assistant more guidance",
    "stop": "You stopped the Run",
    "reject": "You rejected the proposed answer",
}
_EXECUTION_ANSWER = {
    "approve": "You approved the execution plan",
    "revise": "You asked for changes to the execution plan",
    "reject": "You rejected the execution plan",
}
_INVESTIGATION_ANSWER = {
    "approve": "You approved the investigation plan",
    "revise": "You revised the investigation plan",
    "stop": "You stopped the investigation",
    "clarify": "You clarified a variant question",
    "reject": "You rejected a variant in the plan",
}
# An answer to one variant's issue finding.
_ISSUE_ANSWER = {
    "approve": "You approved a variant's finding: checked, not verified",
    "amend": "You amended a variant's request; it is checked again",
    "reject": "You removed a variant whose check ended with an issue",
}
_EFFORT_ANSWER = {
    "extend": "You allowed the assistant more AI requests",
    "stop": "You stopped the assistant",
}
_WORK_ANSWER = {
    "continue": "You let the assistant continue",
    "reject": "You rejected the assistant's work",
    "dismiss": "You dismissed this item; the rest go on",
}
_AVAILABILITY_ANSWER = {
    "retry": "You chose to retry the AI model",
    "amend": "You adjusted how the AI model is contacted",
    "stop": "You stopped waiting for the AI model",
}
_WORK_CAUSE = {
    "failed_turns": "nothing the assistant tried worked several times in a row",
    "result_missing": "the assistant ended without a result",
    "context_exhausted": "the assistant ran out of working memory",
    "researcher_question": "the assistant has a question for you",
}
_AVAILABILITY_CAUSE = {
    "request_rejected": "the AI model service refused our request",
    "malformed_response": "the AI model sent an unreadable reply",
    "automatic_retry_exhausted": "the AI model stayed unavailable after retries",
    "retry_after_over_cap": "the AI model asked to wait too long",
    "wait_budget_exhausted": "the wait for the AI model ran out",
    "contact_stalled": "the AI model's reply stalled twice in a row",
}
_ATTEMPT_FAILURE = {
    "agent_run_failed": "The assistant's step failed",
    "agent_output_invalid": "The assistant's answer could not be used",
    "invariant": "The assistant's step failed an internal check",
    "model_availability_exhausted": "The assistant's step failed: AI model unavailable",
}
_INVESTIGATION_ATTEMPT_FAILURE = {
    "source_contact_failed": "Investigation step failed: a variant source could not be reached",
    "model_contact_failed": "Investigation step failed: AI model could not be reached",
    "agent_run_failed": "Investigation step failed",
}
_RUN_FAILURE = {
    "rejected_by_user": "The Run ended because you rejected it",
    "required_evidence_unresolved": "The Run ended: required information stayed unsettled",
}
_RUNTIME_FAILURE = {
    "runtime_config_invalid": "The Run's registered settings are invalid",
    "runtime_config_changed": "The AI model adapter does not match the Run's registered settings",
    "model_credential_unavailable": "The AI model credentials are unavailable",
    "instructions_unrenderable": "The AI model's instructions could not be prepared",
}
_OPERATIONAL_FAMILY = {"provider": "A cloud step failed"}
_MODEL_CONTACT_FAILURE = {
    "automatic_retry_scheduled": "AI model contact failed; retrying automatically",
    "availability_review_requested": "AI model contact failed; your decision is needed",
    "terminal_failure": "AI model contact failed",
}
_PASS_DISPOSITION = {
    "waiting": "Work paused, waiting for the next step",
    "completed": "Work finished for now",
    "interrupted": "Work was interrupted",
    "held": "Work held after a problem",
}
_SOURCE_CONTACT = {
    "succeeded": "The source answered",
    "failed": "Source contact failed",
}
_INVESTIGATION_LOOKUP = {
    "fetch_refsnp": "NCBI dbSNP lookup",
    "fetch_assembly_report": "NCBI assembly report",
    "eva_clustered_variant": "EVA variant lookup",
    "eva_contig_alias": "EVA contig names",
    "eva_submitted_variants": "EVA submitted alleles",
    "eva_clustered_variants_at": "EVA variants at a position",
    "ncbi_sequence_names": "NCBI sequence names",
    "ncbi_reference_span": "NCBI reference bases",
    "ncbi_spdi_rsids": "NCBI rsIDs at an allele",
}
# Which source a Tool or operation surface reads; every other is NCBI's.
_EVA_SURFACES = frozenset(
    {
        "fetch_eva_variant",
        "eva_clustered_variant",
        "eva_contig_alias",
        "eva_submitted_variants",
        "eva_clustered_variants_at",
    }
)


def _source_of(surface: str) -> str:
    return "EVA" if surface in _EVA_SURFACES else "NCBI"


_LIMIT = {
    "contacts": "contact limit",
    "run_contacts": "Run's contact limit",
    "response_bytes": "byte limit",
}


def _contact_source(request_identity: str) -> str:
    return "EVA" if "//www.ebi.ac.uk/" in request_identity else "NCBI"


_LOOKUP_STATUS = {
    "resolved": "found in {source}",
    "unlisted": "{source} lists no variant at this position",
    "merged": "{source} merged this rsID into a current one",
    "ambiguous": "{source} places this rsID at more than one position",
    "conflicted": "{source} records disagree",
    "incompatible": "not placed on your reference genome",
    "not_found": "not found in {source}",
    "malformed": "the {source} reply could not be read",
    "placement_unnamed": "{source} places this rsID at more than one position; the "
    "assistant names the one meant",
    "source_answered": "{source} answered with something other than a record; the "
    "assistant reads its reply",
}
_LOOKUP_SOURCE = {"ncbi_dbsnp": "dbSNP", "eva": "EVA", "ncbi": "NCBI"}


def _lookup_status(result: dict) -> str:
    return _LOOKUP_STATUS[result["status"]].format(
        source=_LOOKUP_SOURCE[result["lookup_source"]]
    )


_ASSEMBLY_STATUS = {
    "compatible": "matches your reference genome",
    "conflicted": "assembly details conflict",
    "incompatible": "does not match your reference genome",
    "not_found": "assembly report not found",
    "malformed": "assembly report could not be read",
}
_COMBINE_STATUS = {
    "complete": "Evidence collected for every selected variant",
    "partial": "Evidence collected for some selected variants",
    "stopped_before_start": "Evidence collection stopped before it began",
    "stopped_after_partial": "Evidence collection stopped partway",
    "failed_to_combine": "Collected evidence could not be combined",
}
_EVIDENCE_DISPOSITION = {
    "supported": "the evidence supports it",
    "contradicted": "the evidence contradicts it",
    "unresolved": "the evidence is inconclusive",
    "refused": "it could not be checked",
    "partial": "partly answered",
}
_REFERENCE_CHOICE = {
    "sample_identity": "Sample identity settled",
    "taxon_identity": "Organism settled",
    "reference_assembly_identity": "Reference genome chosen",
}
_CANDIDATE_KIND = {"recipe": "Analysis recipe", "resource": "Compute option"}
_CANDIDATE_AUTHORITY = {
    "agent_ranking": "by the assistant",
    "researcher_selection": "by you",
}
_CHECKED = {"valid": "passed checks", "invalid": "was refused"}
_TURN_DISPOSITION = {
    "singleton_admitted": "The assistant's next action was accepted",
    "multi_call_admitted": "The assistant's next actions were accepted",
    "singleton_refused": "The assistant's next action was refused",
    "multi_call_refused": "The assistant's next actions were refused",
    "text_refused": "The assistant replied without acting; asked again",
    "reasoning_refused": "The assistant reasoned but sent no action; asked again",
}
_EXECUTION_OUTCOME = {
    "succeeded": "The analysis run finished successfully",
    "failed": "The analysis run failed",
    "incomplete": "The analysis run paused before finishing",
}

_W = _Wording
_WORDING: dict[str, _Wording] = {
    # The request and its inputs.
    "JourneyStarted": _W("done", "Run started"),
    FORK_ORIGIN_EVENT_TYPE: _W("done", "This Run continues from an earlier Run"),
    "ResearchRequestRegistered": _W("done", "Your request was registered"),
    "RunInputsRegistered": _W(
        "done",
        lambda p: f"{_count(len(p['inputs']), 'read file', 'read files')} registered",
    ),
    "ParabricksConfigurationRegistered": _W("done", "Compute settings loaded"),
    "InvestigationRequestRegistered": _W(
        "done",
        lambda p: _count(
            len(p["request"]["items"]), "variant question", "variant questions"
        )
        + " registered",
    ),
    "PaperFileRegistered": _W("done", "Your paper file was registered"),
    "PaperRecorded": _W(
        "done",
        lambda p: "Paper "
        + _clip(_paper_name(p))
        + f" stored ({_count(len(p['documents']), 'document', 'documents')})",
    ),
    "PaperIntakeRefused": _W(
        "failed", lambda p: f"Paper not taken: {_clip(p['detail'])}"
    ),
    "PaperReviewRequested": _W(
        "question",
        lambda p: f"Question for you: {_clip(p['request']['reason'])}",
    ),
    "PaperReviewAnswered": _W(
        "answer", lambda p: _STEP_ANSWER[p["response"]["action"]]
    ),
    "PaperReadingStarted": _W(
        "done",
        lambda p: "The paper reader started reading the paper for: "
        + _clip(p["question"]),
    ),
    "ReadingPlanKept": _W(
        "done",
        lambda p: "The paper reader proposed "
        + _count(len(p["areas"]), "area", "areas")
        + " to read",
    ),
    "ReadingFindingJudged": _W(
        "done",
        lambda p: f"Reading item {p['item_id']}: a finding {p['outcome']}"
        + (f": {_clip(p['quote']['text'])}" if p["quote"] else ""),
    ),
    "ReadingItemCompleted": _W(
        "done",
        lambda p: f"Reading item {p['item']['item_id']} read: "
        + _count(len(p["finding_event_ids"]), "finding judged", "findings judged"),
    ),
    "ReadingItemNotRead": _W(
        "done",
        lambda p: f"Reading item {p['item_id']} {p['outcome']}: {_clip(p['reason'])}",
    ),
    "PaperAnswerRecorded": _W(
        "done",
        lambda p: "The paper was read: "
        + _count(len(p["answer"]["findings"]), "finding", "findings")
        + " from its own words",
    ),
    "InputMetadataInspected": _W("done", "Your read files were inspected"),
    "InputInspectionErrored": _W("failed", "Your read files could not be inspected"),
    "InputInspectionReviewRequested": _W(
        "question",
        lambda p: f"Question for you: {_clip(p['request']['reason'])}",
    ),
    "InputInspectionReviewAnswered": _W(
        "answer",
        lambda p: _STEP_ANSWER[p["response"]["action"]],
    ),
    "RuntimeConfigurationFailed": _W(
        "failed",
        lambda p: _RUNTIME_FAILURE[p["code"]],
    ),
    "RunFailed": _W("failed", lambda p: _RUN_FAILURE[p["code"]]),
    # Planning and the Execution Review.
    "ProposalRequirementsIdentified": _W(
        "done",
        lambda p: "The plan needs "
        + _count(len(p["plan"]["requirements"]), "item", "items")
        + " settled",
    ),
    "SampleAttestationRecorded": _W(
        "done", "Sample identity recorded as your statement"
    ),
    "TaxonResolutionRecorded": _W(
        "done",
        lambda p: f"Organism looked up at NCBI: {_clip(p['tool_input']['query'], 48)}",
    ),
    "ReferenceAssembliesRecorded": _W(
        "done", "NCBI searched for candidate reference genomes"
    ),
    "PaperSearchRecorded": _W(
        "done",
        lambda p: "Europe PMC searched for your paper: "
        + _clip(p["tool_input"]["query"], 48),
    ),
    "PaperRetrievalRecorded": _W(
        "done",
        lambda p: "Your paper was downloaded"
        if p["status"] == "recorded"
        else "Your paper could not be downloaded",
        lambda p: "done" if p["status"] == "recorded" else "failed",
    ),
    "ReferenceDescriptionRecorded": _W(
        "done",
        lambda p: f"Reference {_clip(p['tool_input']['assembly_accession'], 40)}"
        " checked at NCBI",
    ),
    "ReferenceChoiceRecorded": _W(
        "done",
        lambda p: _REFERENCE_CHOICE[p["grounding"]["kind"]],
    ),
    "CompanionObservationRecorded": _W(
        "done", "Reference genome files located at NCBI"
    ),
    "ReferenceSourceRetrieved": _W(
        "done",
        lambda p: "Source answer kept: "
        + _clip(p["request_identity"].rsplit("/", 1)[-1], 60),
    ),
    "PrebuiltIndexesListed": _W(
        "done",
        lambda p: f"Galaxy's {p['tool_input']['repository']} index catalogue searched",
    ),
    "PrebuiltIndexRead": _W(
        "done",
        lambda p: f"Galaxy index {_clip(p['tool_input']['value'], 40)} read",
    ),
    "PrebuiltIndexChosen": _W(
        "done",
        lambda p: "Galaxy index chosen for the Run"
        if p["read_event_id"]
        else "No Galaxy index fits: the Run builds its index on CPU",
    ),
    "AssemblyWithIndexSuggested": _W(
        "done",
        lambda p: f"{p['tool_input']['assembly_accession']}'s Galaxy index suggested; "
        "the known targets checked there",
    ),
    "ReferenceSourceAdmitted": _W("done", "Reference download source accepted"),
    "RecipeObservationRecorded": _W(
        "done",
        lambda p: _count(
            len(p["result"]["candidates"]), "analysis recipe", "analysis recipes"
        )
        + " found",
    ),
    "ResourceObservationRecorded": _W(
        "done",
        lambda p: _count(
            len(p["result"]["candidates"]), "compute option", "compute options"
        )
        + " found",
    ),
    "PlanningCandidateSelected": _W(
        "done",
        lambda p: f"{_CANDIDATE_KIND[p['kind']]} selected"
        f" {_CANDIDATE_AUTHORITY[p['authority']]}",
    ),
    "PlanningChoiceReconsidered": _W(
        "done", "The assistant reconsidered an earlier choice"
    ),
    "PlanningHitlRequested": _W(
        "question",
        lambda p: f"Question for you: {_clip(p['question'])}",
    ),
    "PlanningHitlResponded": _W(
        "answer",
        lambda p: _PLANNING_ANSWER[p["response"]["action"]],
    ),
    "PreparationInformationRequested": _W(
        "question",
        lambda p: "Question for you: "
        + _count(len(p["missing_fields"]), "setting is", "settings are")
        + " needed to prepare the run",
    ),
    "PreparationInformationAnswered": _W(
        "answer", "You answered the preparation question"
    ),
    "ExecutionPackageCompiled": _W("done", "The analysis job was assembled"),
    "ExecutionProposalSubmitted": _W(
        "question",
        lambda p: (
            "The analysis plan is ready; it asks you to confirm diploid calling"
            if p["proposal"].get("ploidy", {}).get("question")
            else "The analysis plan is ready for your approval"
        ),
    ),
    "ExecutionHitlResponded": _W(
        "answer",
        lambda p: _EXECUTION_ANSWER[p["response"]["action"]],
    ),
    "SliceCheckpointReached": _W("done", "The assistant reached a checkpoint"),
    # Passes, Agent Attempts and their model and Tool mechanics.
    "PassStarted": _W("done", "Work started on your Run"),
    "PassClosed": _W(
        "done",
        lambda p: "Work held after an interrupt; resume continues it"
        if p["disposition"] == "held" and p["failure"]["failure_class"] == "interrupted"
        else _PASS_DISPOSITION[p["disposition"]],
    ),
    "AttemptStarted": _W("done", "An AI assistant started a work step"),
    "AttemptSucceeded": _W("done", "The assistant finished its step"),
    "AttemptStopped": _W("done", "The assistant stopped its step"),
    "AttemptCancelled": _W("done", "The assistant's step was cancelled"),
    "AttemptFailed": _W(
        "failed",
        lambda p: _ATTEMPT_FAILURE[p["code"]],
    ),
    "JourneySearchRead": _W("done", "The assistant reread earlier steps of this Run"),
    "ToolCallValidated": _W(
        "done",
        lambda p: f"The assistant's tool request {_CHECKED[p['status']]}",
    ),
    "AgentActionTurnDispositionRecorded": _W(
        "done",
        lambda p: _TURN_DISPOSITION[p["disposition"]],
    ),
    "AgentTurnRouted": _W(
        "done",
        lambda p: "The assistant's reply was trimmed to its distinct offered calls"
        if p["faults"]
        else "The assistant's calls were put in the order they need",
    ),
    "AgentOutputValidated": _W(
        "done",
        lambda p: f"The assistant's answer {_CHECKED[p['status']]}",
    ),
    "StructuredOutputCallRecorded": _W(
        "done", "The assistant wrote a structured answer"
    ),
    "StructuredOutputValidationLinked": _W("done", "The structured answer was checked"),
    "ToolExecutionFailed": _W("failed", "A tool step failed"),
    "ObservedToolExecutionStarted": _W("done", "A tool step started"),
    "ObservedToolExecutionSucceeded": _W("done", "A tool step finished"),
    "ObservedToolExecutionFailed": _W("failed", "A tool step failed"),
    "ToolExecutionRefused": _W(
        "failed",
        lambda p: f"A tool step could not run, and the assistant was told: "
        f"{_clip(p['statement'], 66)}",
    ),
    "ToolOperationFailureRecorded": _W(
        "failed",
        lambda p: f"A tool step failed: {_clip(p['detail'], 66)}",
    ),
    "ModelRequestRecorded": _W("done", "The assistant asked the AI model"),
    "ModelTransportRequestRecorded": _W("done", "Request sent to the AI model"),
    "ModelTransportResponseRecorded": _W("done", "Reply received from the AI model"),
    "ModelPhysicalContactStarted": _W("done", "Contacting the AI model"),
    "ModelPhysicalContactCancelled": _W(
        "done", "Contact with the AI model was cancelled"
    ),
    "ModelPhysicalContactFailed": _W(
        "failed",
        lambda p: _MODEL_CONTACT_FAILURE[p["policy_disposition"]],
    ),
    "ModelResponseRecorded": _W("done", "The AI model replied"),
    "ModelCompletionTruncated": _W("done", "The AI model's reply was cut off"),
    "ModelRequestFailed": _W(
        "failed",
        lambda p: "The AI model request failed"
        + (
            f": {_clip(p['provider_statement'], 120)}"
            if p.get("provider_statement")
            else ""
        ),
    ),
    "ModelAvailabilityRetryScheduled": _W(
        "done",
        lambda p: f"AI model busy; retrying in {p['delay_seconds']:.0f} s",
    ),
    "ModelAvailabilityRetryDelayCompleted": _W("done", "Retrying the AI model now"),
    "ModelAvailabilityRetryDelayCancelled": _W(
        "done", "The AI model retry was cancelled"
    ),
    "ModelTurnCancelled": _W(
        "done", "The assistant's step was interrupted between turns"
    ),
    "ModelAvailabilityReviewRequested": _W(
        "question",
        lambda p: "Question for you: "
        f"{_AVAILABILITY_CAUSE[p['exhaustion_reason']]}"
        + (
            f", in its words: {_clip(p['provider_statement'], 160)}"
            if p.get("provider_statement")
            else ""
        )
        + (
            "; one answer serves the items that hit the same cause"
            if p.get("joins_event_id")
            else ""
        )
        + "; retry?",
    ),
    "ModelRequestRejected": _W(
        "failed",
        # What happened, in the provider's words, then where it goes.
        lambda p: f"The AI model service refused a request (HTTP "
        f"{p['http_status_code']}), in its words: "
        f"{_clip(p['provider_statement'], 120)}; the assistant reads them and "
        "changes what it sends",
    ),
    "ModelAvailabilityReviewResponded": _W(
        "answer",
        lambda p: _AVAILABILITY_ANSWER[p["response"]["action"]],
    ),
    "ModelAvailabilityStopped": _W(
        "done", "Work stopped while the AI model was unavailable"
    ),
    "AgentFailureReviewRequested": _W(
        "question",
        lambda p: f"Question for you: {_clip(p['request']['reason'])}",
    ),
    "AgentFailureReviewResponded": _W(
        "answer",
        lambda p: "Your retry preset retried the failed step"
        if "preset_event_id" in p
        else _STEP_ANSWER[p["response"]["action"]],
    ),
    "AgentFailureRetryPreset": _W(
        "answer",
        lambda p: f"You preset up to {p['count']} retries of failed steps"
        if p["count"]
        else "You revoked your retry preset",
    ),
    "AgentEffortReviewRequested": _W(
        "question",
        lambda p: f"Question for you: the assistant used {p['requests_used']} of"
        f" {p['request_limit']} AI requests; allow more?",
    ),
    "AgentEffortReviewResponded": _W(
        "answer",
        lambda p: _EFFORT_ANSWER[p["response"]["action"]],
    ),
    "AgentWorkReviewRequested": _W(
        "question",
        lambda p: f"Question for you: {p['question']}"
        if p["cause"] == "researcher_question"
        else f"Question for you: {_WORK_CAUSE[p['cause']]}; continue?",
    ),
    "AgentWorkReviewResponded": _W(
        "answer",
        lambda p: _WORK_ANSWER[p["response"]["action"]],
    ),
    "SourceContactScopeStarted": _W(
        "done",
        lambda p: f"{_source_of(p['tool_name'])} lookup started:"
        f" {_step(p['tool_name']).lower()}",
    ),
    "SourceContactScopeFinished": _W("done", "Source lookup finished"),
    "SourcePhysicalContactStarted": _W(
        "done", lambda p: f"Contacting {_contact_source(p['request_identity'])}"
    ),
    "SourcePhysicalContactFinished": _W(
        "done",
        lambda p: _SOURCE_CONTACT[p["disposition"]],
        lambda p: "failed" if p["disposition"] == "failed" else "done",
    ),
    "SourceOperationFailureRecorded": _W(
        "failed",
        lambda p: f"An {_source_of(p['policy']['surface']['name'])} lookup failed:"
        f" {_clip(p['detail'], 64)}",
    ),
    "SourceContactLimitReached": _W(
        "failed",
        lambda p: f"An {_source_of(p['limits']['surface']['name'])} lookup reached"
        f" its {_LIMIT[p['limit']]} of {p['bound']}",
    ),
    "OperationalFailureRecorded": _W(
        "failed",
        lambda p: (
            f"An {_source_of(p['policy']['surface']['name'])} lookup failed"
            if p["family"] == "source"
            else _OPERATIONAL_FAMILY[p["family"]]
        )
        + f": {_clip(p['detail'], 64)}",
    ),
    "activity_started": _W("done", lambda p: f"{_step(p['stage'])} started"),
    "activity_progress": _W("done", "A work step progressed"),
    "activity_heartbeat": _W("done", "A work step is still running"),
    "activity_succeeded": _W(
        "done",
        lambda p: f"{_step(p['stage'])} done in {p['elapsed_ms'] / 1000:.1f} s",
    ),
    "activity_failed": _W("failed", "A work step failed"),
    "activity_cancelled": _W("done", "A work step was cancelled"),
    # Investigation.
    "InvestigationStarted": _W("done", "Variant investigation started"),
    "InvestigationAttemptBound": _W("done", "The investigation assistant was assigned"),
    "InvestigationAttemptCompleted": _W(
        "done", "The investigation assistant finished its step"
    ),
    "InvestigationAttemptFailed": _W(
        "failed",
        lambda p: _INVESTIGATION_ATTEMPT_FAILURE[p["code"]],
    ),
    "InvestigationSourceContactStarted": _W(
        "done",
        lambda p: f"{_INVESTIGATION_LOOKUP[p['operation']]} for"
        f" {_clip(p['request_key'], 40)} started",
    ),
    "InvestigationSourceContactFailed": _W(
        "failed",
        lambda p: f"{_INVESTIGATION_LOOKUP[p['operation']]} for"
        f" {_clip(p['request_key'], 40)} failed",
    ),
    "InvestigationSourceRetrieved": _W(
        "done",
        lambda p: f"{_INVESTIGATION_LOOKUP[p['source']['operation']]} received",
    ),
    "InvestigationMemberContextRecorded": _W(
        "done", "The assistant set out one target's task for the AI model"
    ),
    "RefSnpToolResultRecorded": _W(
        "done",
        lambda p: f"{_clip(p['query'], 30)}: {_lookup_status(p['result'])}",
    ),
    "AssemblyReportToolResultRecorded": _W(
        "done",
        lambda p: f"Assembly {_clip(p['query'], 30)}:"
        f" {_ASSEMBLY_STATUS[p['result']['status']]}",
    ),
    "InvestigationProposalRecorded": _W(
        "done",
        lambda p: "Investigation plan drafted: "
        + _count(len(p["proposal"]["work_items"]), "variant", "variants")
        + " to check",
    ),
    "InvestigationReviewRequested": _W(
        "question",
        lambda p: "Question for you: the check of "
        f"{_clip(p['question']['request']['request_text'], 40)} ended with an issue"
        if p["question"] is not None and "issue" in p["question"]
        else "Question for you: review the variant investigation plan",
    ),
    "InvestigationReviewResponded": _W(
        "answer",
        lambda p: _INVESTIGATION_ANSWER[p["response"]["action"]],
    ),
    "InvestigationCollectionRevisionRecorded": _W(
        "done", "Your changes to the investigation plan were recorded"
    ),
    "InvestigationProposalCarriedForward": _W(
        "done", "The revised investigation plan was kept"
    ),
    "InvestigationAssertionRecorded": _W(
        "done", "Your revised variant question was recorded"
    ),
    "InvestigationAuthorityGranted": _W(
        "done",
        lambda p: "Approved: "
        + _count(len(p["authority"]["authorized_work_items"]), "variant", "variants")
        + " will be checked in your data",
    ),
    "InvestigationDownloadFailed": _W(
        "failed",
        lambda p: f"Your run's {_DOWNLOAD_FILE[p['role']]} could not be downloaded",
    ),
    "InvestigationDownloadReviewRequested": _W(
        "question",
        lambda p: "Question for you: retry downloading your run's "
        + _DOWNLOAD_QUESTION[p["role"]],
    ),
    "InvestigationDownloadReviewAnswered": _W(
        "answer",
        lambda p: _DOWNLOAD_ANSWER[p["response"]["action"]],
    ),
    "InvestigationStopped": _W("done", "The variant investigation was stopped"),
    "InvestigationPolicyAmended": _W("done", "Investigation settings changed"),
    "InvestigationWorkItemStarted": _W(
        "done",
        lambda p: f"Checking your data at {_variant(p['target'])}",
    ),
    "InvestigationEvidenceRecorded": _W(
        "done",
        lambda p: f"Variant {_variant(p['work_item']['target'])}:"
        f" {_EVIDENCE_DISPOSITION[p['evidence_state']['work_item_disposition']]}",
    ),
    "InvestigationCollectionEvidenceRecorded": _W(
        "done",
        lambda p: _COMBINE_STATUS[p["combine_status"]],
    ),
    "InvestigationItemWindowRecorded": _W(
        "done",
        lambda p: f"Variant {_variant(p['result']['target'])}: "
        + (
            "read "
            + _count(len(p["result"].get("records", ())), "VCF record", "VCF records")
            + f" within {p['flank_bases']} bases"
            if p["result"]["outcome"] == "window_observed"
            else "its window could not be read"
        ),
    ),
    # The Run's own measurement: its first Investigation step, once the BAM
    # and VCF are local; it reports and never gates.
    "RunHealthMeasured": _W(
        "verified",
        lambda p: "Health of your run: "
        + (
            "not measured: " + HEALTH_NOT_MEASURED_WORDS[p["not_measured"]["code"]]
            if p["measurement"] is None
            else f"{p['measurement']['reads']:,} reads,"
            f" {p['measurement']['mapped_percent']}% mapped, mean depth"
            f" {p['measurement']['mean_depth']:.1f},"
            + (
                f" Ti/Tv {p['measurement']['ti_tv']},"
                if p["measurement"]["ti_tv"] is not None
                else ""
            )
            + f" {p['measurement']['covered_contigs']} sequences with reads"
        ),
    ),
    "RunHealthMeasurementFailed": _W(
        "failed",
        lambda p: f"Your run's health could not be measured: {_clip(p['detail'])}",
    ),
    "RunHealthReviewRequested": _W(
        "question", "Question for you: measure your run's health again, or go on?"
    ),
    "RunHealthReviewAnswered": _W(
        "answer",
        lambda p: "Measuring your run's health again"
        if p["response"]["action"] == "retry"
        else "Going on without your run's health",
    ),
    "InvestigationItemCoverageRecorded": _W(
        "done",
        lambda p: f"Reads at {p['contig']}:{p['span_start']}"
        + ("" if p["span_end"] == p["span_start"] else f"-{p['span_end']}")
        + (
            f": {p['measurement']['depth']} at its least-covered base"
            if p["measurement"] is not None
            else f" not measured: {p['not_measured']['code']}"
        ),
    ),
    "InvestigationTableChunkBound": _W(
        "done",
        lambda p: "The model started reading "
        + _count(len(p["rows"]), "variant", "variants")
        + " of the results table",
    ),
    "InvestigationTableChunkRead": _W(
        "done",
        lambda p: "The model read "
        + _count(len(p["readings"]), "variant", "variants")
        + ": "
        + _count(
            sum(item["reading"] == "concern" for item in p["readings"]),
            "concern",
            "concerns",
        ),
    ),
    "InvestigationTableChunkFailed": _W(
        "failed", "The model's reading of part of the results table failed"
    ),
    "InvestigationItemEvidenceCollected": _W(
        "done", "Every approved variant's own record is complete"
    ),
    "ArtifactEnrichmentRetained": _W("done", "A supporting evidence file was saved"),
    # Finalization and the Grounded report.
    "FinalizationBasisCompiled": _W(
        "question", "Results are ready for your review before the report"
    ),
    "FinalizationReviewRecorded": _W("answer", "You reviewed the results"),
    "GroundedReportPublished": _W("done", "Your report was published"),
    # The report's optional Agent explanation.
    "ReportExplanationRequested": _W(
        "answer", "You asked for an explanation of your report"
    ),
    # Each request's member, then the combined answer.
    "ReportExplanationMemberBound": _W(
        "done",
        lambda p: f"The model started explaining request {p['targets'][0]['subject']}: "
        + _count(len(p["targets"]), "target", "targets"),
    ),
    "ReportExplanationMemberAdmitted": _W(
        "verified",
        lambda p: "One request's explanation matched its targets: "
        + _count(len(p["claims"]), "claim", "claims"),
    ),
    "ReportExplanationCombinationBound": _W(
        "done",
        lambda p: "The model started the combined answer over "
        + _count(len(p["sections"]), "request", "requests"),
    ),
    "ReportExplanationPublished": _W(
        "done", "An explanation of your report was published (Agent-written)"
    ),
    "ReportExplanationFailed": _W(
        "failed", "The explanation of your report failed; the report is unchanged"
    ),
}
_PATH_WORDING: dict[str, _Wording] = {
    "ExecutionBound": _W("done", "The approved analysis was set up to run"),
    "InputsVerified": _W(
        "verified", "Your read files were copied to cloud storage and verified"
    ),
    "ReferenceResolved": _W("done", "Ready-made reference index lookup finished"),
    "RuntimeSubstituted": _W(
        "done",
        lambda p: f"No capacity on {_clip(p['from_profile'], 28)};"
        f" switched to {_clip(p['to_profile'], 28)}",
    ),
    "CpuPreparationReviewRequested": _W(
        "question",
        "Question for you: no ready-made reference index; prepare one on CPU?",
    ),
    "CpuPreparationReviewAnswered": _W(
        "answer",
        lambda p: _CPU_REVIEW_ANSWER[p["action"]],
    ),
    "ReferencePrepared": _W("verified", "Reference genome index prepared and verified"),
    "H100ArtifactsVerified": _W("verified", "GPU analysis results verified"),
    "StageFailed": _W(
        "failed",
        lambda p: f"Analysis step failed: {_clip(p['detail'], 66)}",
    ),
    "FailureReviewRequested": _W(
        "question",
        lambda p: f"Question for you: {_clip(p['reason'])}",
    ),
    "FailureReviewAnswered": _W(
        "answer",
        lambda p: _STEP_ANSWER[p["action"]],
    ),
    "ExecutionOutcome": _W(
        "done",
        lambda p: _EXECUTION_OUTCOME[p["disposition"]],
        lambda p: "failed" if p["disposition"] == "failed" else "done",
    ),
}


def _asks_about_an_issue(event: JourneyEvent, sequence: int) -> bool:
    """Whether the event is an issue finding's own Review (#359)."""
    if event.event_type != "InvestigationReviewRequested":
        return False
    try:
        question = event.payload["question"]
        return question is not None and "issue" in question
    except (KeyError, TypeError) as error:
        raise JourneyValidationError(
            f"Journey event on line {sequence} does not match the current "
            "InvestigationReviewRequested contract"
        ) from error


def _told(
    event: JourneyEvent, sequence: int, issue_requests: frozenset[str] = frozenset()
) -> tuple[FeedLevel, FeedStatus, str]:
    """The item's level, status and line, read from the event's current payload contract.

    ``issue_requests`` are the issue Reviews read so far: an answer to one is told by
    its own consequence (#359), never as an ordinary member's answer.
    """
    name = event.event_type
    try:
        level = _level(event)
        payload, wordings = event.payload, _WORDING
        if name == "ExecutionPathRecorded":
            # An Execution path fact is told by its own type and payload.
            name, payload, wordings = (
                payload["event_type"],
                payload["payload"],
                _PATH_WORDING,
            )
        wording = wordings.get(name)
        if wording is None:
            return level, "done", name
        status = (
            wording.status if wording.status_of is None else wording.status_of(payload)
        )
        if (
            name == "InvestigationReviewResponded"
            and payload["request_event_id"] in issue_requests
        ):
            # An issue answer is told by its own consequence, before (and instead
            # of) an ordinary member's wording, which has no amend.
            text = _ISSUE_ANSWER[payload["response"]["action"]]
        elif isinstance(wording.text, str):
            text = wording.text
        else:
            text = wording.text(payload)
    except (KeyError, TypeError, ValueError) as error:
        # A missing field, another type or an unformattable value.
        raise JourneyValidationError(
            f"Journey event on line {sequence} does not match the current {name} contract"
        ) from error
    return level, status, text


class FeedItem(FrozenContract):
    """One retained event; `sequence` is its 1-based line in `journey.jsonl`."""

    sequence: StrictInt = Field(ge=1)
    event_id: StrictStr
    event_type: StrictStr
    occurred_at: StrictStr
    stage: FeedStage
    level: FeedLevel
    status: FeedStatus
    text: StrictStr
    # The Pass this event belongs to, counted in this file; 0 before the first.
    pass_number: StrictInt = Field(ge=0)
    payload_bytes: StrictInt = Field(ge=0)
    payload: JsonValue = Field(default=None, exclude_if=lambda value: value is None)


class JourneyFeed(FrozenContract):
    schema_version: Literal["bio-neb.journey-feed/1"] = "bio-neb.journey-feed/1"
    journey_file: StrictStr
    journey_id: StrictStr | None
    # Every complete event read and the last one, whatever the view shows.
    event_count: StrictInt = Field(ge=0)
    head_event_id: StrictStr | None
    items: tuple[FeedItem, ...]


def journey_feed(
    events: Sequence[JourneyEvent], *, payloads: bool = False
) -> tuple[FeedItem, ...]:
    """Project retained events in Journey order; payloads only when asked for."""
    items = []
    stage: FeedStage = "request"
    pass_number = 0
    issue_requests: set[str] = set()
    for sequence, event in enumerate(events, start=1):
        pass_number += event.event_type == "PassStarted"
        if _asks_about_an_issue(event, sequence):
            issue_requests.add(event.event_id)
        # The first Agent Attempt begins planning before its first planning fact.
        if event.event_type == "AttemptStarted" and stage == "request":
            stage = "planning"
        stage = _STAGE_STARTS.get(event.event_type, stage)
        level, status, text = _told(event, sequence, frozenset(issue_requests))
        items.append(
            FeedItem(
                sequence=sequence,
                event_id=event.event_id,
                event_type=event.event_type,
                occurred_at=event.occurred_at,
                stage=stage,
                level=level,
                status=status,
                text=text,
                pass_number=pass_number,
                payload_bytes=len(canonical_json(event.payload).encode()),
                payload=event.payload if payloads else None,
            )
        )
    return tuple(items)


def read_journey_feed(
    journey_file: str, *, everything: bool = False, payloads: bool = False
) -> JourneyFeed:
    """Read the complete lines of one Journey file without contact or checks of meaning.

    The default view keeps key and step items; `everything` keeps every event.
    """
    try:
        content = Path(journey_file).read_bytes()
    except FileNotFoundError as error:
        raise JourneyValidationError("Journey JSONL does not exist") from error
    complete = content[: content.rfind(b"\n") + 1]
    events = []
    for number, line in enumerate(complete.splitlines(), start=1):
        try:
            events.append(JourneyEvent.model_validate_json(line))
        except ValidationError as error:
            raise JourneyValidationError(
                f"invalid Journey event on line {number}"
            ) from error
    return JourneyFeed(
        journey_file=journey_file,
        journey_id=events[0].journey_id if events else None,
        event_count=len(events),
        head_event_id=events[-1].event_id if events else None,
        items=tuple(
            item
            for item in journey_feed(events, payloads=payloads)
            if everything or item.level in _DEFAULT_LEVELS
        ),
    )


class RunArtifactsUnavailable(ValueError):
    """The Run has no completed Execution, so no verified artifacts yet."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RunArtifact(FrozenContract):
    """One verified file: where it is, its identity and the proof recorded for it."""

    role: StrictStr
    # An object in the Run bucket (`s3://…`) or a Journey-bundle document
    # (`artifacts/sha256/…`, relative to the Journey's directory).
    location: StrictStr
    byte_count: StrictInt = Field(ge=0)
    sha256: StrictStr
    grounding: StrictStr
    producer_stage: StrictStr
    proof_ref: StrictStr


class LocalRunArtifact(FrozenContract):
    """One file the Run materialized under its own directory, as its Journey records it.

    ``receipt`` names what established the file: the materialization receipt's path
    for the VCF, and the preparation receipt's identity for its index (#107).
    """

    role: Literal["VCF", "VCF_CSI"]
    path: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr
    receipt: StrictStr


class RunArtifacts(FrozenContract):
    schema_version: Literal["bio-neb.run-artifacts/1"] = "bio-neb.run-artifacts/1"
    journey_id: StrictStr
    journey_file: StrictStr
    run_root: StrictStr
    execution_id: StrictStr
    authority_id: StrictStr
    outcome_event_id: StrictStr
    verified_event_id: StrictStr
    bucket: StrictStr
    object_scope: StrictStr
    outputs: tuple[RunArtifact, ...]
    inputs: tuple[RunArtifact, ...]
    reference: tuple[RunArtifact, ...]
    # Additive and optional: a Run with no observation of its own lists none, and
    # its document keeps its earlier bytes.
    local: tuple[LocalRunArtifact, ...] = Field(
        default=(), exclude_if=lambda entries: not entries
    )


def _proven(pointer: ArtifactPointer, location: str | None = None) -> RunArtifact:
    return RunArtifact(
        role=pointer.role,
        location=pointer.location if location is None else location,
        byte_count=pointer.bytes,
        sha256=ContentId.parse(pointer.content_id).digest,
        grounding=pointer.grounding,
        producer_stage=pointer.producer.stage,
        proof_ref=pointer.proof_ref,
    )


def _local_artifacts(journey_file: str) -> tuple[LocalRunArtifact, ...]:
    """The Run's own materialized VCF and its index, from its latest own observation.

    Every value is the Journey's record: the collection's command names the files
    and receipt, the observation the VCF's identity (the materialization receipt's
    ``materialized_vcf``, which the receipt binds to it) and the preparation the
    index's. No file is read. A collection a Fork inherited names its parent's
    files and is not this Run's.
    """
    from bio_neb.application.queries import read_investigation
    from bio_neb.investigation.vcf_observation import PreparedVcfObservationReceipt
    from bio_neb.journey import Journey

    read = read_investigation(Journey.open(journey_file))
    for collection in reversed(read.collections):
        command = collection.evidence.command
        if command.journey_file != journey_file:
            continue
        observation = next(
            (
                member.evidence.sample_observation
                for member in reversed(read.recorded_members)
                if member.evidence.command == command
            ),
            None,
        )
        if observation is None:
            continue
        local = [
            LocalRunArtifact(
                role="VCF",
                path=command.vcf_file,
                byte_count=observation.vcf_descriptor.byte_count,
                sha256=observation.vcf_descriptor.sha256,
                receipt=command.materialization_receipt_file,
            )
        ]
        preparation = observation.preparation
        if isinstance(preparation, PreparedVcfObservationReceipt):
            index = preparation.derived_csi
            local.append(
                LocalRunArtifact(
                    role="VCF_CSI",
                    path=index.path,
                    byte_count=index.byte_count,
                    sha256=index.sha256,
                    receipt=preparation.receipt_id,
                )
            )
        return tuple(local)
    return ()


def read_run_artifacts(journey_file: str) -> RunArtifacts:
    """List a completed Run's verified objects from its one checked Execution result,
    and the local files its own observation materialized."""
    try:
        outcome = read_completed_parabricks_result(journey_file)
    except (ParabricksResultUnavailable, ParabricksExecutionIncomplete) as error:
        raise RunArtifactsUnavailable(str(error)) from error
    return RunArtifacts(
        journey_id=outcome.journey_id,
        journey_file=outcome.journey_file,
        run_root=outcome.run_root,
        execution_id=outcome.execution_id,
        authority_id=outcome.authority_id,
        outcome_event_id=outcome.outcome_event_id,
        verified_event_id=outcome.h100_checkpoint_event_id,
        bucket=outcome.storage.bucket,
        object_scope=outcome.object_scope,
        # Each output is the verified object in the Run bucket, with the proof the
        # H100 stage recorded for those bytes.
        outputs=tuple(
            _proven(proof, object_location(item.bucket, item.object_key))
            for item, proof in zip(
                outcome.verified_objects, outcome.output_proofs, strict=True
            )
        ),
        inputs=tuple(_proven(pointer) for pointer in outcome.input_proofs),
        reference=tuple(_proven(pointer) for pointer in outcome.reference_proofs),
        local=_local_artifacts(outcome.journey_file),
    )
