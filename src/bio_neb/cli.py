"""Observer-driven CLI for Run advancement and contact-free Journey review."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    Field,
    StrictStr,
    TypeAdapter,
    ValidationError,
)

from bio_neb.agentic_journey.failure_cause import cause_identity
from bio_neb.agentic_journey.model_availability import (
    ModelAvailabilityReviewResponseRefused,
)
from bio_neb.agentic_journey.refusal_disposition import RefusalDisposition
from bio_neb.agentic_journey.runtime_config import (
    describe_runtime_defaults,
    describe_runtime_drift,
    read_runtime_config,
)
from bio_neb.application.advancement import (
    BeginInvestigation,
    BeginRun,
    ResumeRun,
    ReviewAnswerRefused,
    RunAdvancementRefused,
    advance,
    answer_from_json,
    explain_report,
    review_response_from_json,
    use_run_advancement_dependencies,
)
from bio_neb.application.commands import (
    ApprovedExecutionCommandRefused,
    RecordRetryPresetCommand,
    RegisterFastqInput,
    RegistrationError,
    RetryPresetRefused,
    record_retry_preset,
)
from bio_neb.application.fork import (
    ForkRefused,
    ForkRun,
    ParentJourneyUnreadable,
    fork_run,
    read_parent_journey_identity,
)
from bio_neb.application.queries import (
    JourneyObservation,
    ObserveRun,
    observe,
    read_journey_feed,
    read_run_artifacts,
)
from bio_neb.application.run_interaction import (
    RunStatus,
    run_interaction,
)
from bio_neb.application.verification_policy import (
    ArtifactVerificationDefaults,
    ArtifactVerificationOverrides,
)
from bio_neb.contracts import FrozenContract
from bio_neb.failure import HeldFailure
from bio_neb.investigation.bam_materialization import (
    BamMaterializationRefused,
    MaterializeCurrentBamCommand,
    materialize_current_bam,
)
from bio_neb.investigation.investigation_commands import (
    ContinueInvestigationCommand,
    InvestigationApplicationRefused,
    continue_investigation,
)
from bio_neb.investigation.investigation_observation_commands import (
    InvestigationObservationApplicationRefused,
)
from bio_neb.investigation.investigation_policy import InvestigationPolicyAmendment
from bio_neb.investigation.investigation_policy_commands import (
    AmendInvestigationPolicyCommand,
    InvestigationPolicyApplicationRefused,
    amend_investigation_policy,
    inspect_investigation_policy,
)
from bio_neb.investigation.scientific_continuation import (
    PublishScientificContinuationCommand,
    ScientificContinuationPublicationRefused,
    ScientificContinuationRefused,
    publish_scientific_continuation,
)
from bio_neb.investigation.vcf_materialization import (
    MaterializeCurrentVcfCommand,
    VcfMaterializationRefused,
    materialize_current_vcf,
)
from bio_neb.journey import (
    JourneyValidationError,
    canonical_json,
    resolve_journey_file,
)
from bio_neb.parabricks.execution_contact_accounting import (
    EXECUTION_CONTACT_SCOPE,
    review_execution_contacts,
)
from bio_neb.parabricks.execution_proposal import (
    ExecutionHitlResponseRefused,
)
from bio_neb.parabricks.nebius_execution_composition import (
    make_nebius_object_storage_tracer_adapters,
    make_nebius_storage_readiness_adapter,
)
from bio_neb.parabricks.object_storage_tracer import (
    ObjectStorageTracerCommand,
    ObjectStorageTracerRefused,
    review_object_storage_tracer,
    run_object_storage_tracer,
)
from bio_neb.parabricks.plan import ExecutionCompositionRefused
from bio_neb.parabricks.planning_hitl import (
    PlanningHitlResponseRefused,
)
from bio_neb.parabricks.preparation_review import PreparationAnswerRefused
from bio_neb.parabricks.result import ParabricksExecutionOutcome
from bio_neb.parabricks.storage_readiness import (
    DirectStorageReadinessCommand,
    DirectStorageReadinessRefused,
    review_direct_storage_readiness,
    run_direct_storage_readiness,
)
from bio_neb.providers.nebius_common import (
    NebiusAuthenticationRequired,
    NebiusCliError,
    S3ContactError,
)
from bio_neb.providers.nebius_context import (
    NebiusExecutionContextSelection,
)
from bio_neb.providers.nebius_controller_runtime import (
    NativeControllerPreflightRefused,
    make_accepted_nebius_cli_runner,
    run_native_controller_preflight,
)
from bio_neb.reporting.finalization_commands import (
    ApproveFinalizationCommand,
    CompileFinalizationCommand,
    CorrectFinalizationCommand,
    FinalizationApplicationRefused,
    InspectFinalizationCommand,
    StopFinalizationCommand,
    approve_finalization,
    compile_finalization,
    correct_finalization,
    inspect_finalization,
    stop_finalization,
)
from bio_neb.reporting.grounded_report import GroundedReport
from bio_neb.reporting.report_commands import (
    GroundedReportApplicationRefused,
    PublishGroundedReportCommand,
    ReadReadableReportCommand,
    ReopenGroundedReportCommand,
    ReopenReportExplanationCommand,
    ReportExplanationRefused,
    RequestReportExplanationCommand,
    publish_grounded_report,
    read_readable_report,
    reopen_grounded_report,
    reopen_report_explanation,
    written_readable_report,
)


class CliInputError(ValueError):
    def __init__(self, message: str, fields: tuple[str, ...] = ("inputs",)) -> None:
        super().__init__(message)
        self.fields = fields


def _configure_utf8_streams() -> None:
    """Use one portable encoding for retained scientific/model text on real streams."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="strict")


def _parser() -> argparse.ArgumentParser:  # noqa: PLR0915 - Existing orchestration complexity; contract-base migration preserves this flow.
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    begin = commands.add_parser(
        "begin", help="begin and automatically advance one research Run"
    )
    begin.add_argument("--run-root", required=True)
    # Not passed, `--prompt-file` and `--fastq` are asked for at a terminal.
    begin.add_argument("--prompt-file")
    # Omitted, the Run takes the build's defaults; a file overrides what it names.
    begin.add_argument("--runtime-config")
    begin.add_argument(
        "--request-file",
        action="append",
        default=[],
        help="a researcher SNP request document, retained with the Run; repeat for "
        "several, kept in order",
    )
    begin.add_argument(
        "--paper-file",
        action="append",
        default=[],
        help="a paper the researcher supplies (JATS/XML, HTML, text, .csv, .docx, "
        ".xlsx, PDF or a zip of these), retained with the Run; repeat for several",
    )
    begin.add_argument(
        "--fastq",
        action="append",
        metavar="ROLE=ABSOLUTE_PATH|s3://BUCKET/KEY",
        help="repeat once for each ordered FASTQ input (a local path). Eval/demo "
        "option only, not the product input path: an s3:// object staged for evals "
        "is copied in the provider rather than uploaded",
    )
    begin.add_argument(
        "--interactive",
        action="store_true",
        help="present and answer one genuine Review, then show the resumed frontier",
    )
    resume = commands.add_parser(
        "resume", help="automatically advance one retained research Run"
    )
    resume.add_argument("--journey-file", required=True)
    resume.add_argument(
        "--interactive",
        action="store_true",
        help="present and answer one genuine Review, then show the resumed frontier",
    )
    answer = commands.add_parser(
        "answer", help="answer one exact pending Review and automatically resume"
    )
    answer.add_argument("--journey-file", required=True)
    answer.add_argument("--request-event-id", required=True)
    answer_response = answer.add_mutually_exclusive_group(required=True)
    answer_response.add_argument(
        "--response-json",
        help="exact legal response object presented by the pending Review",
    )
    answer_response.add_argument(
        "--response-file",
        help="absolute canonical UTF-8 file containing the exact legal response object",
    )
    retry_preset = commands.add_parser(
        "retry-preset",
        help="preset a retry for one Agent role's failure Reviews, or revoke it",
    )
    retry_preset.add_argument("--journey-file", required=True)
    retry_preset.add_argument(
        "--preset-file",
        required=True,
        help="JSON preset: agent_role, failure_classes, tool_names, count, by",
    )
    observe_run = commands.add_parser(
        "observe", help="observe retained Run truth without contact or mutation"
    )
    observe_run.add_argument("--journey-file", required=True)
    observe_run.add_argument(
        "--rich",
        action="store_true",
        help="render the complete existing Journey review instead of concise progress",
    )
    journey_feed = commands.add_parser(
        "journey-feed",
        help="print the Run's plain-language feed, one item per event, without contact",
    )
    journey_feed.add_argument("--journey-file", required=True)
    journey_feed.add_argument(
        "--all",
        action="store_true",
        help="show every event instead of the key and step items",
    )
    journey_feed.add_argument(
        "--payloads", action="store_true", help="include each event's payload"
    )
    run_artifacts = commands.add_parser(
        "run-artifacts",
        help="print a completed Run's verified files and their proofs, without contact",
    )
    run_artifacts.add_argument("--journey-file", required=True)
    parent_identity = commands.add_parser(
        "parent-identity",
        help="print a retained Journey's whole-file parent identity without contact",
    )
    parent_identity.add_argument("--journey-file", required=True)
    runtime_drift = commands.add_parser(
        "runtime-drift",
        help="report how a runtime file differs from current defaults, or with no "
        "file the defaults themselves; writes nothing",
    )
    runtime_drift.add_argument("--runtime-config")
    investigation_begin = commands.add_parser(
        "investigation-begin",
        help="begin one source-resolved post-run Investigation proposal",
    )
    investigation_begin.add_argument("--journey-file", required=True)
    investigation_begin.add_argument(
        "--scientific-continuation-file",
        help="continuation packet; omitted, it is compiled from this Run's Execution",
    )
    investigation_begin.add_argument(
        "--request-file",
        action="append",
        default=[],
        help="researcher SNP request, repeatable; omitted, the Run's captured "
        "request is used",
    )
    fork = commands.add_parser(
        "fork",
        help="create an idle child Run from a parent's closed point; writes no parent",
    )
    fork.add_argument("--parent-journey-file", required=True)
    fork.add_argument("--child-run-root", required=True)
    fork.add_argument(
        "--point-event-id",
        help="a closed Pass of the parent; omitted, the parent's head",
    )
    investigation_continue = commands.add_parser(
        "investigation-continue",
        help="continue source work after an Investigation amendment or clarification",
    )
    investigation_continue.add_argument("--journey-file", required=True)
    policy_inspect = commands.add_parser(
        "investigation-policy",
        help="inspect separate item, continuation and completion decisions",
    )
    policy_inspect.add_argument("--journey-file", required=True)
    policy_amend = commands.add_parser(
        "investigation-policy-amend", help="amend only named future Investigation work"
    )
    policy_amend.add_argument("--journey-file", required=True)
    policy_amend.add_argument("--amendment-file", required=True)
    policy_amend.add_argument("--runtime-config")
    finalization_compile = commands.add_parser(
        "finalization-compile",
        help="compile the exact ended Run outcomes into Finalization review",
    )
    finalization_compile.add_argument("--journey-file", required=True)
    finalization_inspect = commands.add_parser(
        "finalization-inspect",
        help="inspect one exact retained Finalization basis without mutation",
    )
    finalization_inspect.add_argument("--journey-file", required=True)
    finalization_inspect.add_argument("--basis-digest", required=True)
    finalization_approve = commands.add_parser(
        "finalization-approve",
        help="approve one exact pending Finalization basis",
    )
    finalization_approve.add_argument("--journey-file", required=True)
    finalization_approve.add_argument("--request-event-id", required=True)
    finalization_approve.add_argument("--basis-digest", required=True)
    finalization_approve.add_argument("--head-event-id", required=True)
    finalization_correct = commands.add_parser(
        "finalization-correct",
        help="correct exact Report membership or return to Investigation",
    )
    finalization_correct.add_argument("--journey-file", required=True)
    finalization_correct.add_argument("--request-event-id", required=True)
    finalization_correct.add_argument("--basis-digest", required=True)
    finalization_correct.add_argument("--head-event-id", required=True)
    finalization_correct.add_argument("--correction-file", required=True)
    finalization_stop = commands.add_parser(
        "finalization-stop",
        help="stop Finalization without creating a Report basis",
    )
    finalization_stop.add_argument("--journey-file", required=True)
    finalization_stop.add_argument("--request-event-id", required=True)
    finalization_stop.add_argument("--basis-digest", required=True)
    finalization_stop.add_argument("--head-event-id", required=True)
    finalization_stop.add_argument("--reason-file", required=True)
    grounded_report_publish = commands.add_parser(
        "grounded-report-publish",
        help="publish one exact approved Report basis without contact",
    )
    grounded_report_publish.add_argument("--journey-file", required=True)
    grounded_report_publish.add_argument("--report-basis-digest", required=True)
    grounded_report_publish.add_argument("--head-event-id", required=True)
    grounded_report_reopen = commands.add_parser(
        "grounded-report-reopen",
        help="reopen one immutable Grounded report without contact",
    )
    grounded_report_reopen.add_argument("--journey-file", required=True)
    grounded_report_reopen.add_argument("--report-digest", required=True)
    report_explain = commands.add_parser(
        "report-explain",
        help="request an Agent explanation of one published report and resume",
    )
    report_explain.add_argument("--journey-file", required=True)
    report_explain.add_argument("--report-digest", required=True)
    report_explain.add_argument("--requested-by", required=True)
    report_explain.add_argument(
        "--supersedes",
        help="the published explanation of the same report this one rewrites",
    )
    report_readable = commands.add_parser(
        "report-readable",
        help="print the researcher's readable Markdown report without contact",
    )
    report_readable.add_argument("--journey-file", required=True)
    report_readable.add_argument("--report-digest", required=True)
    report_explanation_reopen = commands.add_parser(
        "report-explanation-reopen",
        help="reopen one published report explanation without contact",
    )
    report_explanation_reopen.add_argument("--journey-file", required=True)
    report_explanation_reopen.add_argument("--explanation-id", required=True)
    execution_contact_review = commands.add_parser(
        "execution-contact-review",
        help=(
            f"recompile the bound {EXECUTION_CONTACT_SCOPE} evidence without contact"
        ),
        description=(
            f"Recompile the bound {EXECUTION_CONTACT_SCOPE} evidence without contact."
        ),
    )
    execution_contact_review.add_argument("--execution-journey-file", required=True)
    scientific_continuation = commands.add_parser(
        "scientific-continuation",
        help="compile and publish one terminal scientific continuation",
    )
    scientific_continuation.add_argument("--execution-journey-file", required=True)
    scientific_continuation.add_argument("--output", required=True)
    vcf_materialize = commands.add_parser(
        "vcf-materialize",
        help="materialize the exact verified current-Run VCF without observation",
    )
    vcf_materialize.add_argument("--execution-journey-file", required=True)
    vcf_materialize.add_argument("--candidate-file")
    vcf_materialize.add_argument("--candidate-receipt-file")
    vcf_materialize.add_argument("--output-directory", required=True)
    _verification_arguments(vcf_materialize)
    current_bam = commands.add_parser(
        "bam-materialize-current",
        help="acquire one exact current-Run BAM without granting coverage authority",
    )
    current_bam.add_argument("--execution-journey-file", required=True)
    current_bam.add_argument("--output-directory", required=True)
    native_preflight = commands.add_parser(
        "nebius-controller-preflight",
        help="verify the pinned native Ubuntu CLI and external Run filesystem",
    )
    native_preflight.add_argument("--nebius-executable", required=True)
    native_preflight.add_argument("--run-root", required=True)
    native_preflight.add_argument("--receipt-file", required=True)
    profiles = commands.add_parser(
        "nebius-profiles",
        help="list locally configured Nebius CLI profile names",
    )
    profiles.add_argument("--nebius-executable", required=True)
    storage_readiness = commands.add_parser(
        "storage-readiness",
        help="create and verify one fresh empty bucket through the production Adapter",
    )
    storage_readiness.add_argument("--selection-file", required=True)
    storage_readiness.add_argument("--bucket-name", required=True)
    storage_readiness.add_argument("--receipt-file", required=True)
    storage_readiness.add_argument("--region", default="eu-north1")
    storage_readiness_review = commands.add_parser(
        "storage-readiness-review",
        help="reopen one immutable storage-readiness receipt without provider contact",
    )
    storage_readiness_review.add_argument("--receipt-file", required=True)
    object_storage_tracer = commands.add_parser(
        "object-storage-tracer",
        help="run one bounded shared-mechanics Object Storage tracer",
    )
    object_storage_tracer.add_argument("--selection-file", required=True)
    object_storage_tracer.add_argument("--bucket-name", required=True)
    object_storage_tracer.add_argument("--receipt-file", required=True)
    object_storage_tracer.add_argument("--region", default="eu-north1")
    object_storage_tracer_review = commands.add_parser(
        "object-storage-tracer-review",
        help="reopen one immutable Object Storage tracer receipt without contact",
    )
    object_storage_tracer_review.add_argument("--receipt-file", required=True)
    return parser


def _verification_arguments(parser: Any) -> None:
    parser.add_argument("--runtime-config")
    parser.add_argument(
        "--artifact-verification",
        choices=("reuse_creation_proof", "independent_full_read"),
    )
    parser.add_argument("--artifact-verification-reason")


def _verification_settings(
    arguments: Any,
) -> tuple[ArtifactVerificationDefaults, ArtifactVerificationOverrides]:
    defaults = (
        read_runtime_config(Path(arguments.runtime_config)).artifact_verification
        if arguments.runtime_config
        else ArtifactVerificationDefaults()
    )
    overrides = ArtifactVerificationOverrides(
        **(
            {
                "mode": arguments.artifact_verification,
                "reason": arguments.artifact_verification_reason,
            }
            if arguments.artifact_verification is not None
            or arguments.artifact_verification_reason is not None
            else {}
        )
    )
    return defaults, overrides


_JOURNEY_PATH_ARGUMENTS = (
    "journey_file",
    "parent_journey_file",
    "execution_journey_file",
)


def _resolve_journey_paths(arguments: argparse.Namespace) -> None:
    """Resolve each Journey-path argument; the application sees only absolute paths."""
    for name in _JOURNEY_PATH_ARGUMENTS:
        value = getattr(arguments, name, None)
        if value is None:
            continue
        try:
            resolved = resolve_journey_file(value)
        except ValidationError as error:
            raise CliInputError(
                f"--{name.replace('_', '-')} is not a Journey file",
                (name, *(f"reason:{item['type']}" for item in error.errors())),
            ) from error
        setattr(arguments, name, resolved)


def _fastq_argument(value: str) -> RegisterFastqInput:
    role, separator, path = value.partition("=")
    if not separator:
        raise CliInputError(
            "--fastq must use ROLE=ABSOLUTE_PATH or ROLE=s3://BUCKET/KEY"
        )
    return RegisterFastqInput(role=role, registered_path=path)


def _failure(code: str, fields: list[str]) -> None:
    print(
        canonical_json({"status": "failed", "code": code, "fields": fields}),
        file=sys.stderr,
    )


def _refusal_fields(error: Exception) -> list[str]:
    for name in ("fields", "details", "violations"):
        value = getattr(error, name, None)
        if isinstance(value, tuple) and value:
            return [str(item) for item in value]
    return []


def _supplied_files(arguments: argparse.Namespace) -> list[str]:
    return [
        name
        for name, value in vars(arguments).items()
        if value is not None
        and name.endswith(("_file", "_root", "_directory", "_config", "_executable"))
    ]


_OBSERVER_DRIVEN_COMMANDS = frozenset({"begin", "resume", "answer", "observe"})

_RESPONSE_REFUSALS = (
    PreparationAnswerRefused,
    ExecutionHitlResponseRefused,
    ModelAvailabilityReviewResponseRefused,
    PlanningHitlResponseRefused,
)


class _PublicRefusalView(FrozenContract):
    schema_version: Literal["bio-neb.agentic-cli/1"] = "bio-neb.agentic-cli/1"
    frontier: Literal["refused"] = "refused"
    code: StrictStr
    disposition: RefusalDisposition
    fields: tuple[StrictStr, ...]
    status: RunStatus | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    request_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    journey_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    journey_file: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    failure: HeldFailure | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def _interactive_answer(
    review: JourneyObservation,
) -> JourneyObservation | None:
    """Present and answer at most one Review through the same typed application seam."""

    from bio_neb.parabricks.result import (
        ParabricksNoncompletion,
    )

    _emit_run_view(review)
    if isinstance(review, ParabricksExecutionOutcome):
        return None
    pending = (
        review.pending_review
        if isinstance(review, ParabricksNoncompletion)
        else review.current_review
    )
    if pending is None:
        return None
    print(
        "Enter one exact legal Review response JSON: ",
        file=sys.stderr,
        end="",
        flush=True,
    )
    try:
        document = sys.stdin.readline()
    except KeyboardInterrupt:
        print(file=sys.stderr)  # End the prompt's line before the refusal.
        raise
    if not document:
        raise CliInputError("interactive Review response is required", ("response",))
    try:
        response = json.loads(document)
    except json.JSONDecodeError as error:
        raise CliInputError(
            "interactive Review response must be exact JSON", ("response",)
        ) from error
    request_event_id = (
        review.request_event_id
        if isinstance(review, ParabricksNoncompletion)
        else pending.request_event_id
    )
    return advance(
        review_response_from_json(review, request_event_id, json.dumps(response))
    )


def _emit_run_view(review: JourneyObservation, *, rich: bool = False) -> None:
    from bio_neb.parabricks.result import (
        ParabricksNoncompletion,
    )

    if isinstance(review, (ParabricksExecutionOutcome, ParabricksNoncompletion)):
        print(canonical_json(review.model_dump(mode="json")), flush=True)
        return
    document = (
        review.model_dump(mode="json")
        if rich
        else run_interaction(review).model_dump(mode="json")
    )
    print(canonical_json(document), flush=True)


def _emit_readable_report(review: JourneyObservation) -> None:
    """A finished Run's own readable report, after its run view (#476 C13)."""
    if review.status != "grounded_report_published":
        return
    page = written_readable_report(review.journey_file)
    if page is not None:
        print(page, end="", file=sys.stderr, flush=True)


def _question_lines(question: Any, membership: Any) -> list[str]:
    if question is None:
        return ["", "Question", "- no scientific question retained"]
    return [
        "",
        "Question",
        f"- id: {question.membership.member_id}",
        f"  membership: {membership(question)}",
        f"  asserted by: {question.value.asserted_by}",
        f"  request: {question.value.request_text}",
        f"  rsID: {question.value.rsid}",
        # Every request of a collection, in the researcher's order.
        *(
            f"  request {ordinal}: {item.request_text}"
            for ordinal, item in enumerate(question.value.requests, start=1)
        ),
    ]


def _table_cell_lines(item: Any) -> list[str]:
    """The item's VCF table cells and its own reading, once they exist (#361)."""
    quality = ", ".join(
        f"{name} {value}"
        for name, value in (
            ("GQ", item.genotype_quality),
            ("DP", item.read_depth),
            (
                "AD",
                None
                if item.allele_depths is None
                else ",".join(
                    "." if depth is None else str(depth) for depth in item.allele_depths
                ),
            ),
            ("QUAL", item.qual),
            ("FILTER", None if item.filters is None else ";".join(item.filters)),
        )
        if value is not None
    )
    lines = [f"    call quality: {quality}"] if quality else []
    if item.window is not None:
        window = item.window
        lines.append(
            f"    window ±{window.flank_bases}: "
            + (
                f"refused ({window.refusal})"
                if window.outcome == "refused"
                else f"{len(window.records)} record(s)"
            )
        )
        lines.extend(f"      {record}" for record in window.records)
    if item.coverage is not None:
        lines.append(f"    coverage: {item.coverage.replace('_', ' ')}")
    if item.reading is not None:
        concerns = f" ({', '.join(item.concerns)})" if item.concerns else ""
        lines.append(
            f"    model reading: {item.reading.replace('_', ' ')}{concerns}: "
            f"{item.reading_note}"
        )
    return lines


def _collection_view_lines(view: Any) -> list[str]:
    """Every requested SNP, then each case across them (#360 W)."""
    lines = ["", "Requested SNPs"]
    for snp in view.snps:
        identity = snp.requested_rsid or "no rsID stated"
        if snp.verified_rsid is None:
            identity += " (not verified)"
        elif snp.verified_rsid != snp.requested_rsid:
            identity += f" (verified {snp.verified_rsid})"
        lines.append(f"- {identity}: {snp.request_text}")
        if snp.asserted_by is not None:
            lines.append(f"  asserted by: {snp.asserted_by}")
        if snp.locus is not None:
            lines.append(f"  locus: {snp.locus}")
        # A request without Work says why on its "no Work item" line.
        if snp.state not in {None, "candidates"} and snp.no_work is None:
            lines.append(f"  state: {snp.state_words()}")
        for item in snp.items:
            outcome = (
                item.state_words()
                if item.outcome is None and item.state is not None
                else item.outcome or "not looked up"
            )
            genotype = "" if item.genotype is None else f" (GT {item.genotype})"
            lines.append(f"  {item.target}: {outcome}{genotype}")
            lines.extend(_table_cell_lines(item))
        for record in snp.records:
            genotype = "" if record.genotype is None else f" (GT {record.genotype})"
            lines.append(
                f"  record {record.position} {record.ref}>{','.join(record.alts)}"
                f"{genotype}: REF {record.reference}"
            )
        if snp.records_not_reported:
            lines.append(f"  records not reported: {snp.records_not_reported}")
        if snp.no_work is not None:
            lines.append(f"  no Work item: {snp.no_work_note()}")
        if snp.cases:
            lines.append(f"  cases: {', '.join(snp.cases)}")
    lines.extend(("", "Cross-SNP cases"))
    for case in view.cases:
        lines.extend(
            (
                f"- {case.code}: {case.origin.audit()}",
                f"  evidence: {', '.join(case.evidence_ids)}",
            )
        )
    if not view.cases:
        lines.append("- none found")
    return lines


def _emit_explanation(explanation: Any) -> None:
    """One published explanation, visibly Agent-written, with its references:
    the conclusion, then each request's own section (#408, #460)."""

    def cited(text: Any) -> str:
        return f"{text.text} [{', '.join(text.references)}]"

    rows = {row.number: row for row in explanation.targets}
    lines = [
        "Explanation (Agent-written, not a finding)",
        f"explanation id: {explanation.explanation_id}",
        f"report digest: {explanation.report_digest}",
        f"Report-basis digest: {explanation.report_basis_digest}",
        f"supersedes: {explanation.supersedes_explanation_id or 'none'}",
        f"instructions: {explanation.instruction_identity}",
        "",
        "Conclusion",
        *(() if explanation.title is None else (f"  Title: {explanation.title}",)),
        f"  {cited(explanation.conclusion)}",
        f"  What would change it: {explanation.would_change}",
        "",
    ]
    lines.append("Requests (each explained by its own member)")
    for request in explanation.requests:
        heading = f"  request {request.subject} (target {request.request})"
        if request.status == "not_explained":
            lines.append(f"{heading}: not explained: {request.reason}")
            continue
        lines.append(f"{heading}: explained")
        for claim in request.claims:
            row = rows[claim.target]
            genotype = "" if claim.genotype is None else f", genotype {claim.genotype}"
            lines.append(
                f"    target {claim.target} {row.kind} {row.subject}: "
                f"{claim.claim}{genotype}"
            )
        lines.append(f"    why: {cited(request.why)}")
        lines.extend(
            f"    Q{', Q'.join(map(str, item.qualifications))}: {cited(item)}"
            for item in request.limitations
        )
    lines.extend(("", "Report qualifications (in plain words)"))
    lines.extend(
        f"  Q{item.qualification} {item.kind}: {item.words}"
        for item in explanation.qualifications
    )
    if not explanation.qualifications:
        lines.append("  none recorded")
    lines.extend(("", "Limitations"))
    lines.extend(
        f"  Q{', Q'.join(map(str, item.qualifications))}: {cited(item)}"
        for item in explanation.limitations
    )
    lines.extend(("", "Not done", f"  {explanation.not_done}"))
    print("\n".join(lines), flush=True)


def _emit_grounded_report(report: GroundedReport) -> None:
    def membership(item: Any) -> str:
        return "included" if item.membership.included else "excluded"

    lines = [
        "Grounded report",
        f"schema: {report.schema_version}",
        f"renderer: {report.renderer_version}",
        f"report digest: {report.report_digest}",
        f"Report-basis digest: {report.report_basis_digest}",
        f"Journey/head: {report.journey_id} / {report.source_head_event_id}",
        f"run provenance: {report.run_provenance}",
        f"provenance origin event: {report.provenance_origin_event_id or 'unestablished'}",
        f"report external contact count: {report.report_external_contact_count}",
        f"outcome: {report.outcome}",
    ]
    lines.extend(_question_lines(report.question, membership))
    lines.extend(("", "Requested and verified identities"))
    for slot in report.fact_slots:
        lines.extend(
            (
                f"- kind: {slot.kind}",
                f"  requirement: {slot.requirement_id}",
                f"  subject: {slot.subject_key}",
                f"  requested: {slot.requested_value or 'not stated'}",
                f"  verified: {slot.verified_value or 'unverified'}",
                f"  status: {slot.status}",
                f"  verification source: {slot.verification_source or 'none retained'}",
                f"  mismatch: {str(slot.mismatch).lower()}",
                f"  assessment: {slot.summary}",
                # Planning asked for the same rsID: one subject, one entry.
                *(
                    f"  planning: {earlier.requirement_id} ({earlier.kind}), "
                    f"{earlier.status}: {earlier.summary}"
                    for earlier in slot.planning_slots
                ),
            )
        )
    if not report.fact_slots:
        lines.append("- none retained")
    if report.issue_items:
        lines.extend(("", "Issue items (checked, not verified)"))
    for issue_item in report.issue_items:
        issue = issue_item.issue
        decision = (
            issue_item.decision
            if issue_item.decided_by is None
            else f"{issue_item.decision} by {issue_item.decided_by}"
        )
        lines.extend(
            (
                f"- request: {issue_item.request.request_text}",
                f"  rsID: {issue_item.request.rsid or 'not stated'}",
                f"  issue: {issue.key}",
                f"  verdict: {issue.verdict} ({issue.source})",
                *(f"  conflict: {conflict}" for conflict in issue.conflicts),
                f"  evidence: {', '.join(issue.evidence_event_ids)}",
                f"  finding: {issue.finding_event_id}",
                f"  researcher decision: {decision}",
                f"  review: {issue_item.request_event_id}",
            )
        )
    if report.collection_view is not None:
        lines.extend(_collection_view_lines(report.collection_view))
    lines.extend(("", "Sample and normalized target"))
    for item in report.work_items:
        target = item.value.target
        lines.extend(
            (
                f"- id: {item.membership.member_id}",
                f"  membership: {membership(item)}",
                f"  sample: {item.value.sample_id}",
                "  target: "
                f"{target.assembly_accession} / {target.assembly_name} / "
                f"{target.variant_text}",
                # A region has no rsID aliases.
                *(
                    ()
                    if item.value.intended_action == "observe_region"
                    else (f"  aliases: {', '.join(target.aliases)}",)
                ),
            )
        )
    lines.extend(("", "Observations"))
    for item in report.sample_observations:
        result = item.value.result
        lines.extend(
            (
                f"- id: {item.membership.member_id}",
                f"  membership: {membership(item)}",
                f"  sample: {item.value.sample_id}",
                f"  outcome: {result.outcome}",
                f"  claim scope: {result.claim_scope}",
            )
        )
    lines.extend(("", "Candidate evidence"))
    for item in report.candidate_evidence:
        lines.extend(
            (
                f"- id: {item.membership.member_id}",
                f"  membership: {membership(item)}",
                f"  admitted authority: {str(item.membership.admitted).lower()}",
                f"  outcome: {item.value.outcome}",
                f"  claim scope: {item.value.claim_scope}",
            )
        )
    lines.extend(("", "Assessments and dispositions"))
    for item in report.observation_assessments:
        lines.extend(
            (
                f"- id: {item.membership.member_id}",
                f"  membership: {membership(item)}",
                f"  verdict: {item.value.verdict}",
                f"  disposition: {item.value.work_item_disposition}",
                f"  claim scope: {item.value.claim_scope}",
            )
        )
        for contradiction in item.value.contradictions:
            lines.append(f"  conflict: {contradiction}")
    lines.extend(("", "Admissions and non-admitted evidence"))
    for item in report.admissions:
        lines.extend(
            (
                f"- id: {item.membership.member_id}",
                f"  membership: {membership(item)}",
                f"  admission: {item.value.status}",
                f"  claim scope: {item.value.claim_scope}",
            )
        )
        reason = getattr(item.value, "reason", None)
        if reason is not None:
            lines.append(f"  reason: {reason}")
    lines.extend(("", "Retrieved sources"))
    for item in report.retrieved_sources:
        lines.extend(
            (
                f"- id: {item.membership.member_id}",
                f"  membership: {membership(item)}",
                f"  source: {item.value.authority}:{item.value.operation}",
                f"  request identity: {item.value.request_identity}",
                f"  artifact: {item.value.raw_artifact_id}",
            )
        )
    lines.extend(("", "Source assessments"))
    for item in report.source_assessments:
        lines.extend(
            (
                f"- id: {item.membership.member_id}",
                f"  membership: {membership(item)}",
                f"  verdict: {item.value.verdict}",
                f"  rationale: {item.value.rationale}",
            )
        )
    for label, values in (
        ("Conflicts", report.conflicts),
        ("Limitations", report.limitations),
        ("Gaps", report.gaps),
    ):
        lines.extend(("", label))
        lines.extend(f"- {value.text}" for value in values)
        if not values:
            lines.append("- none retained")
    lines.extend(("", "Review authority"))
    lines.extend(
        (
            f"- response: {report.review_authority.response_id}",
            f"- action: {report.review_authority.response.action}",
            f"- basis: {report.review_authority.basis_digest}",
            f"- source head: {report.review_authority.source_head_event_id}",
        )
    )
    # The retained Run record closes the report, after the researcher's
    # sections: its accounting, then its evidence pointers.
    lines.extend(("", "Run accounting"))
    for item in report.accounting:
        lines.extend(
            (
                f"- event: {item.value.event_type} / {item.value.source_event_id}",
                f"  membership: {membership(item)}",
                "  admitted authority: false",
                f"  retained: {canonical_json(item.value.payload)}",
            )
        )
    lines.extend(("", "Evidence pointers"))
    for pointer in report.evidence_pointers:
        lines.extend(
            (
                f"- id: {pointer.pointer_id}",
                f"  role: {pointer.role}",
                f"  locator: {pointer.locator}",
                f"  content id: {pointer.content_id or 'none'}",
                f"  sha256: {pointer.sha256 or 'none'}",
                f"  source event: {pointer.source_event_id}",
            )
        )
    print("\n".join(lines), flush=True)


def _emit_public_refusal(error: Any, fields: list[str]) -> None:
    document = _PublicRefusalView(
        code=error.code,
        disposition=error.disposition,
        fields=tuple(fields),
        status=getattr(error, "status", None),
        request_event_id=getattr(error, "request_event_id", None),
        journey_id=getattr(error, "journey_id", None),
        journey_file=getattr(error, "journey_file", None),
        failure=getattr(error, "failure", None),
    )
    print(
        canonical_json(document.model_dump(mode="json")),
        file=sys.stderr,
        flush=True,
    )


def _hitl_response(arguments: argparse.Namespace) -> Any:
    field = "response_json"
    document = arguments.response_json
    if arguments.response_file is not None:
        field = "response_file"
        value = arguments.response_file
        if not os.path.isabs(value) or os.path.normpath(value) != value:
            raise CliInputError(
                "--response-file must be absolute and canonical",
                (field,),
            )
        try:
            document = Path(value).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise CliInputError(
                "--response-file must be a readable UTF-8 file",
                (field,),
            ) from error
    try:
        return json.loads(document)
    except json.JSONDecodeError as error:
        raise CliInputError(
            f"--{field.replace('_', '-')} must be exact JSON",
            (field,),
        ) from error


def _profile_names(stdout: str) -> tuple[str, ...]:
    """Normalize profile-list output without inferring tenant or project scope."""

    names: list[str] = []
    for line in stdout.splitlines():
        if not line:
            continue
        match = re.fullmatch(r"([^\s\[\]]+)(?: \[default\])?", line)
        if match is None:
            raise CliInputError("Nebius profile inventory is invalid")
        names.append(match.group(1))
    return tuple(dict.fromkeys(names))


def _missing_begin_inputs(arguments: argparse.Namespace) -> list[str]:
    if arguments.command != "begin":
        return []
    return [
        flag
        for flag, value in (
            ("--prompt-file", arguments.prompt_file),
            ("--fastq", arguments.fastq),
        )
        if not value
    ]


def _ask_missing_begin_inputs(arguments: argparse.Namespace) -> None:
    """Ask the terminal for each begin input not passed, filling its flag's value.

    Only a begin missing a required input asks; flags passed skip their prompt (#453).
    """
    if not _missing_begin_inputs(arguments):
        return
    if arguments.prompt_file is None:
        arguments.prompt_file = _ask("prompt file path: ", "prompt_file")
    if not arguments.request_file:
        while request := _ask("request file path (empty to finish): "):
            arguments.request_file.append(request)
    if not arguments.paper_file:
        while paper := _ask("paper file path (empty to finish): "):
            arguments.paper_file.append(paper)
    if arguments.runtime_config is None:
        arguments.runtime_config = (
            _ask("runtime file path (empty for the build's defaults): ") or None
        )
    if not arguments.fastq:
        arguments.fastq = [
            f"{role}={_ask(f'{role} FASTQ path: ', 'inputs')}"
            for role in ("read_1", "read_2")
        ]


def _ask(prompt: str, required: str | None = None) -> str:
    """One answer typed at the terminal; a required one may not be empty."""
    print(prompt, file=sys.stderr, end="", flush=True)
    try:
        answer = sys.stdin.readline().strip()
    except KeyboardInterrupt:
        print(file=sys.stderr)  # End the prompt's line before the refusal.
        raise
    if required is not None and not answer:
        raise CliInputError(f"{prompt.rstrip(': ')} is required", (required,))
    return answer


def main(  # noqa: C901, PLR0915 - Existing orchestration complexity; contract-base migration preserves this flow.
    argv: list[str] | None = None,
    *,
    parabricks_infrastructure_factory: Any | None = None,
    nebius_profile_runner: Any | None = None,
    storage_readiness_adapter_factory: Any | None = None,
    object_storage_tracer_adapter_factory: Any | None = None,
    nebius_runner_factory: Any | None = None,
    vcf_object_read_factory: Any | None = None,
    bam_object_read_factory: Any | None = None,
    investigation_model_factory: Any | None = None,
    investigation_source_adapter_factory: Any | None = None,
) -> int:
    _configure_utf8_streams()
    parser = _parser()
    arguments = parser.parse_args(argv)
    missing = _missing_begin_inputs(arguments)
    if missing and not sys.stdin.isatty():
        # Without a terminal to ask, they stay required, as they always were.
        parser.error("the following arguments are required: " + ", ".join(missing))

    def nebius_runner(executable: str) -> Any:
        factory = nebius_runner_factory or make_accepted_nebius_cli_runner
        runner = factory(executable)
        runner.preflight()
        return runner

    def display_execution(record):
        print(canonical_json(record), file=sys.stderr, flush=True)

    def display_agent_turn(progress):
        print(
            f"repeat {progress.repeat}/{progress.cap} {progress.tool_name}",
            file=sys.stderr,
            flush=True,
        )

    try:
        _resolve_journey_paths(arguments)
        if arguments.command in {"begin", "resume", "answer"}:
            with use_run_advancement_dependencies(
                investigation_model_factory=investigation_model_factory,
                investigation_source_adapter_factory=investigation_source_adapter_factory,
                parabricks_infrastructure_factory=parabricks_infrastructure_factory,
                execution_progress=display_execution,
                agent_turn_progress=display_agent_turn,
            ):
                if arguments.command == "begin":
                    _ask_missing_begin_inputs(arguments)
                    review = advance(
                        BeginRun(
                            run_root=arguments.run_root,
                            prompt_file=arguments.prompt_file,
                            runtime_config_file=arguments.runtime_config,
                            request_files=tuple(arguments.request_file),
                            paper_files=tuple(arguments.paper_file),
                            inputs=tuple(
                                _fastq_argument(value) for value in arguments.fastq
                            ),
                        )
                    )
                    if arguments.interactive:
                        resumed = _interactive_answer(review)
                        if resumed is not None:
                            _emit_run_view(resumed)
                            _emit_readable_report(resumed)
                        return 0
                    _emit_run_view(review)
                    _emit_readable_report(review)
                    return 0
                if arguments.command == "resume":
                    review = advance(ResumeRun(journey_file=arguments.journey_file))
                    if arguments.interactive:
                        resumed = _interactive_answer(review)
                        if resumed is not None:
                            _emit_run_view(resumed)
                            _emit_readable_report(resumed)
                        return 0
                    _emit_run_view(review)
                    _emit_readable_report(review)
                    return 0
                if arguments.command == "answer":
                    review = answer_from_json(
                        arguments.journey_file,
                        arguments.request_event_id,
                        json.dumps(_hitl_response(arguments)),
                    )
                    _emit_run_view(review)
                    _emit_readable_report(review)
                    return 0
        if arguments.command == "report-explain":
            with use_run_advancement_dependencies(
                investigation_model_factory=investigation_model_factory,
                investigation_source_adapter_factory=investigation_source_adapter_factory,
                agent_turn_progress=display_agent_turn,
            ):
                review, current = explain_report(
                    RequestReportExplanationCommand(
                        journey_file=arguments.journey_file,
                        report_digest=arguments.report_digest,
                        requested_by=arguments.requested_by,
                        supersedes_explanation_id=arguments.supersedes,
                    )
                )
            if current.explanation is not None:
                _emit_explanation(current.explanation.explanation)
            else:
                print(f"Explanation request: {current.status}", flush=True)
                _emit_run_view(review)
            return 0
        if arguments.command == "report-readable":
            print(
                read_readable_report(
                    ReadReadableReportCommand(
                        journey_file=arguments.journey_file,
                        report_digest=arguments.report_digest,
                    )
                ),
                end="",
                flush=True,
            )
            return 0
        if arguments.command == "report-explanation-reopen":
            published = reopen_report_explanation(
                ReopenReportExplanationCommand(
                    journey_file=arguments.journey_file,
                    explanation_id=arguments.explanation_id,
                )
            )
            _emit_explanation(published.explanation)
            return 0
        if arguments.command == "observe":
            review = observe(ObserveRun(journey_file=arguments.journey_file))
            _emit_run_view(review, rich=arguments.rich)
            return 0
        if arguments.command == "journey-feed":
            feed = read_journey_feed(
                arguments.journey_file,
                everything=arguments.all,
                payloads=arguments.payloads,
            )
            print(canonical_json(feed.model_dump(mode="json")), flush=True)
            return 0
        if arguments.command == "run-artifacts":
            artifacts = read_run_artifacts(arguments.journey_file)
            print(canonical_json(artifacts.model_dump(mode="json")), flush=True)
            return 0
        if arguments.command == "parent-identity":
            identity = read_parent_journey_identity(Path(arguments.journey_file))
            print(canonical_json(identity.model_dump(mode="json")))
            return 0
        if arguments.command == "runtime-drift":
            report = (
                describe_runtime_defaults()
                if arguments.runtime_config is None
                else describe_runtime_drift(Path(arguments.runtime_config))
            )
            print(canonical_json(report.model_dump(mode="json")))
            return 0
        if arguments.command == "investigation-begin":
            with use_run_advancement_dependencies(
                investigation_model_factory=investigation_model_factory,
                investigation_source_adapter_factory=investigation_source_adapter_factory,
                agent_turn_progress=display_agent_turn,
            ):
                review = advance(
                    BeginInvestigation(
                        journey_file=arguments.journey_file,
                        scientific_continuation_file=(
                            arguments.scientific_continuation_file
                        ),
                        request_files=tuple(arguments.request_file),
                    )
                )
            _emit_run_view(review)
            return 0
        if arguments.command == "fork":
            receipt = fork_run(
                ForkRun(
                    parent_journey_file=arguments.parent_journey_file,
                    child_run_root=arguments.child_run_root,
                    point_event_id=arguments.point_event_id,
                )
            )
            print(canonical_json(receipt.model_dump(mode="json")))
            return 0
        if arguments.command == "investigation-continue":
            continue_investigation(
                ContinueInvestigationCommand(journey_file=arguments.journey_file),
                environ=os.environ,
                model_factory=investigation_model_factory,
                source_adapter_factory=investigation_source_adapter_factory,
            )
            _emit_run_view(observe(ObserveRun(journey_file=arguments.journey_file)))
            return 0
        if arguments.command == "investigation-policy":
            print(
                canonical_json(
                    inspect_investigation_policy(arguments.journey_file).model_dump(
                        mode="json"
                    )
                )
            )
            return 0
        if arguments.command == "retry-preset":
            preset = record_retry_preset(
                RecordRetryPresetCommand(
                    journey_file=arguments.journey_file,
                    preset=json.loads(Path(arguments.preset_file).read_bytes()),
                )
            )
            print(canonical_json(preset.model_dump(mode="json")))
            return 0
        if arguments.command == "investigation-policy-amend":
            amendment = InvestigationPolicyAmendment.model_validate_json(
                Path(arguments.amendment_file).read_bytes()
            )
            defaults = (
                read_runtime_config(
                    Path(arguments.runtime_config)
                ).investigation_work_policy()
                if arguments.runtime_config
                else None
            )
            result = amend_investigation_policy(
                AmendInvestigationPolicyCommand(
                    journey_file=arguments.journey_file, amendment=amendment
                ),
                defaults=defaults,
            )
            print(canonical_json(result.model_dump(mode="json")))
            return 0
        if arguments.command == "finalization-compile":
            compile_finalization(
                CompileFinalizationCommand(journey_file=arguments.journey_file)
            )
            _emit_run_view(
                observe(ObserveRun(journey_file=arguments.journey_file)), rich=True
            )
            return 0
        if arguments.command == "finalization-inspect":
            inspect_finalization(
                InspectFinalizationCommand(
                    journey_file=arguments.journey_file,
                    basis_digest=arguments.basis_digest,
                )
            )
            _emit_run_view(
                observe(ObserveRun(journey_file=arguments.journey_file)), rich=True
            )
            return 0
        if arguments.command == "finalization-approve":
            approve_finalization(
                ApproveFinalizationCommand(
                    journey_file=arguments.journey_file,
                    request_event_id=arguments.request_event_id,
                    basis_digest=arguments.basis_digest,
                    head_event_id=arguments.head_event_id,
                )
            )
            _emit_run_view(
                observe(ObserveRun(journey_file=arguments.journey_file)), rich=True
            )
            return 0
        if arguments.command == "finalization-correct":
            correct_finalization(
                CorrectFinalizationCommand(
                    journey_file=arguments.journey_file,
                    request_event_id=arguments.request_event_id,
                    basis_digest=arguments.basis_digest,
                    head_event_id=arguments.head_event_id,
                    correction_file=arguments.correction_file,
                )
            )
            _emit_run_view(
                observe(ObserveRun(journey_file=arguments.journey_file)), rich=True
            )
            return 0
        if arguments.command == "finalization-stop":
            stop_finalization(
                StopFinalizationCommand(
                    journey_file=arguments.journey_file,
                    request_event_id=arguments.request_event_id,
                    basis_digest=arguments.basis_digest,
                    head_event_id=arguments.head_event_id,
                    reason_file=arguments.reason_file,
                )
            )
            _emit_run_view(
                observe(ObserveRun(journey_file=arguments.journey_file)), rich=True
            )
            return 0
        if arguments.command == "grounded-report-publish":
            projection = publish_grounded_report(
                PublishGroundedReportCommand(
                    journey_file=arguments.journey_file,
                    report_basis_digest=arguments.report_basis_digest,
                    head_event_id=arguments.head_event_id,
                )
            )
            assert projection.report is not None
            _emit_grounded_report(projection.report)
            return 0
        if arguments.command == "grounded-report-reopen":
            projection = reopen_grounded_report(
                ReopenGroundedReportCommand(
                    journey_file=arguments.journey_file,
                    report_digest=arguments.report_digest,
                )
            )
            assert projection.report is not None
            _emit_grounded_report(projection.report)
            return 0
        if arguments.command == "execution-contact-review":
            review = review_execution_contacts(arguments.execution_journey_file)
            print(canonical_json(review.model_dump(mode="json")))
            return 0 if review.status == "complete" else 2
        if arguments.command == "scientific-continuation":
            receipt = publish_scientific_continuation(
                PublishScientificContinuationCommand(
                    execution_journey_file=arguments.execution_journey_file,
                    output=arguments.output,
                )
            )
            print(canonical_json(receipt.model_dump(mode="json")))
            return 0
        if arguments.command == "vcf-materialize":
            verification_defaults, verification_overrides = _verification_settings(
                arguments
            )
            result = materialize_current_vcf(
                MaterializeCurrentVcfCommand(
                    execution_journey_file=arguments.execution_journey_file,
                    candidate_file=arguments.candidate_file,
                    candidate_receipt_file=arguments.candidate_receipt_file,
                    output_directory=arguments.output_directory,
                    verification_defaults=verification_defaults,
                    verification_overrides=verification_overrides,
                ),
                object_read_factory=vcf_object_read_factory,
            )
            print(canonical_json(result.model_dump(mode="json")))
            return 0
        if arguments.command == "bam-materialize-current":
            result = materialize_current_bam(
                MaterializeCurrentBamCommand(
                    execution_journey_file=arguments.execution_journey_file,
                    output_directory=arguments.output_directory,
                ),
                object_read_factory=bam_object_read_factory,
            )
            print(canonical_json(result.model_dump(mode="json")))
            return 0
        if arguments.command == "nebius-controller-preflight":
            receipt = run_native_controller_preflight(
                executable=arguments.nebius_executable,
                run_root=arguments.run_root,
                receipt_file=arguments.receipt_file,
                runner_factory=(
                    nebius_runner_factory or make_accepted_nebius_cli_runner
                ),
            )
            print(canonical_json(receipt.model_dump(mode="json")))
            return 0
        if arguments.command == "nebius-profiles":
            accepted_runner = nebius_runner(arguments.nebius_executable)
            runner = nebius_profile_runner or accepted_runner
            profiles = _profile_names(
                runner(
                    "profile",
                    "list",
                    "--no-check-update",
                    "--no-progress",
                    "--retries",
                    "1",
                ).stdout
            )
            print(
                canonical_json(
                    {
                        "profiles": profiles,
                        "tenant_project_selection": "explicit_manual_entry_required",
                    }
                )
            )
            return 0
        if arguments.command == "storage-readiness":
            try:
                selection = TypeAdapter(NebiusExecutionContextSelection).validate_json(
                    Path(arguments.selection_file).read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, ValidationError) as error:
                raise DirectStorageReadinessRefused(
                    "direct_storage_readiness_selection_invalid", "selection_file"
                ) from error
            accepted_runner = nebius_runner(selection.nebius_executable)
            identity = accepted_runner.preflight()
            if (
                selection.resolved_nebius_executable != identity.resolved_executable
                or selection.nebius_cli_version != identity.version
                or selection.nebius_executable_byte_count != identity.byte_count
                or selection.nebius_executable_sha256 != identity.sha256
            ):
                raise NebiusCliError("provider_cli_identity_mismatch")
            factory = storage_readiness_adapter_factory
            if factory is None:

                def factory(config: Any, selected: Any) -> Any:
                    return make_nebius_storage_readiness_adapter(
                        config,
                        selected,
                        credential_cli_runner=accepted_runner,
                    )

            receipt = run_direct_storage_readiness(
                DirectStorageReadinessCommand(
                    selection_file=arguments.selection_file,
                    bucket_name=arguments.bucket_name,
                    receipt_file=arguments.receipt_file,
                    region=arguments.region,
                ),
                adapter_factory=factory,
                emit=lambda payload: print(canonical_json(payload), flush=True),
            )
            return 0 if receipt.status == "execution_storage_ready" else 2
        if arguments.command == "storage-readiness-review":
            receipt = review_direct_storage_readiness(arguments.receipt_file)
            print(canonical_json(receipt.model_dump(mode="json")), flush=True)
            return 0 if receipt.status == "execution_storage_ready" else 2
        if arguments.command == "object-storage-tracer":
            try:
                selection = TypeAdapter(NebiusExecutionContextSelection).validate_json(
                    Path(arguments.selection_file).read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, ValidationError) as error:
                raise ObjectStorageTracerRefused(
                    "object_storage_tracer_selection_invalid", "selection_file"
                ) from error
            accepted_runner = nebius_runner(selection.nebius_executable)
            identity = accepted_runner.preflight()
            if (
                selection.resolved_nebius_executable != identity.resolved_executable
                or selection.nebius_cli_version != identity.version
                or selection.nebius_executable_byte_count != identity.byte_count
                or selection.nebius_executable_sha256 != identity.sha256
            ):
                raise NebiusCliError("provider_cli_identity_mismatch")
            factory = object_storage_tracer_adapter_factory
            if factory is None:

                def factory(config, selected, observe):
                    return make_nebius_object_storage_tracer_adapters(
                        config,
                        selected,
                        observe,
                        credential_cli_runner=accepted_runner,
                    )

            receipt = run_object_storage_tracer(
                ObjectStorageTracerCommand(
                    selection_file=arguments.selection_file,
                    bucket_name=arguments.bucket_name,
                    receipt_file=arguments.receipt_file,
                    region=arguments.region,
                ),
                adapter_factory=factory,
                emit=lambda payload: print(canonical_json(payload), flush=True),
            )
            return 0 if receipt.status == "object_storage_tracer_succeeded" else 2
        if arguments.command == "object-storage-tracer-review":
            review = review_object_storage_tracer(arguments.receipt_file)
            print(canonical_json(review.model_dump(mode="json")), flush=True)
            return 0 if review.status == "object_storage_tracer_succeeded" else 2
    except KeyboardInterrupt:
        # Ctrl-C or the operator's interrupt: one line, the shell's SIGINT code. A
        # Run it interrupted mid-work is held; at a prompt nothing is sent.
        _failure("interrupted", [arguments.command])
        return 130
    except (RegistrationError, RetryPresetRefused) as error:
        _failure(error.code, list(error.fields))
    except CliInputError as error:
        _failure("command_invalid", list(error.fields))
    except (RunAdvancementRefused, ReviewAnswerRefused) as error:
        _emit_public_refusal(error, [arguments.command])
    except InvestigationApplicationRefused as error:
        _emit_public_refusal(error, list(error.fields) or [arguments.command])
    except InvestigationObservationApplicationRefused as error:
        _emit_public_refusal(error, list(error.fields) or [arguments.command])
    except ForkRefused as error:
        _emit_public_refusal(error, [arguments.command, *filter(None, [error.detail])])
    except FinalizationApplicationRefused as error:
        _emit_public_refusal(error, list(error.fields) or [arguments.command])
    except GroundedReportApplicationRefused as error:
        _emit_public_refusal(error, list(error.fields) or [arguments.command])
    except ReportExplanationRefused as error:
        _emit_public_refusal(error, list(error.fields) or [arguments.command])
    except _RESPONSE_REFUSALS as error:
        if arguments.command in _OBSERVER_DRIVEN_COMMANDS:
            _emit_public_refusal(error, ["request_event_id", "response"])
        else:
            _failure(error.code, ["request_event_id", "response"])
    except ApprovedExecutionCommandRefused as error:
        _failure(error.code, ["reference-source"])
    except ScientificContinuationRefused as error:
        _failure(error.code, list(error.details) or [arguments.command])
    except ScientificContinuationPublicationRefused as error:
        _failure(error.code, list(error.details) or [arguments.command])
    except ParentJourneyUnreadable as error:
        _failure(error.code, list(error.fields))
    except VcfMaterializationRefused as error:
        _failure(error.code, list(error.details) or [arguments.command])
    except BamMaterializationRefused as error:
        _failure(error.code, list(error.details) or [arguments.command])
    except ExecutionCompositionRefused as error:
        _failure(error.code, list(error.details) or [arguments.command])
    except NativeControllerPreflightRefused as error:
        _failure(error.code, list(error.fields) or [arguments.command])
    except DirectStorageReadinessRefused as error:
        _failure(error.code, list(error.fields) or [arguments.command])
    except ObjectStorageTracerRefused as error:
        _failure(error.code, list(error.fields) or [arguments.command])
    except NebiusAuthenticationRequired as error:
        # Its own halt, ahead of the general CLI boundary below.  A reader
        # of this line needs one fact the exit code alone never carried -- the only
        # action that resolves it is a person completing the CLI's own browser login.
        fields = ["human_authentication_required", arguments.command]
        if error.failure_evidence is not None:
            fields.extend(error.failure_evidence.public_fields())
        _failure(error.code, fields)
    except NebiusCliError as error:
        fields = [arguments.command]
        if error.failure_evidence is not None:
            fields.extend(error.failure_evidence.public_fields())
        _failure(error.code, fields)
    except S3ContactError as error:
        _failure(str(error), [arguments.command])
    except ValidationError as error:
        fields = sorted(
            {
                ".".join(str(part) for part in item["loc"]) or "$"
                for item in error.errors(include_url=False, include_input=False)
            }
        )
        _failure("command_invalid", fields)
    except InvestigationPolicyApplicationRefused as error:
        _failure(error.code, list(error.fields) or [arguments.command])
    except UnicodeError:
        raise
    except (JourneyValidationError, ValueError) as error:
        # The cause is code identity, so it is safe beside the field
        # names.  A typed refusal keeps its own code; an untyped one is a Journey
        # failure only when the Journey is the command's sole supplied file.
        code = getattr(error, "code", None)
        if isinstance(code, str) and not isinstance(error, JourneyValidationError):
            _failure(code, _refusal_fields(error) or [arguments.command])
        elif isinstance(error, JourneyValidationError) or _supplied_files(
            arguments
        ) == ["journey_file"]:
            _failure(
                "journey_review_invalid",
                ["journey_file", f"cause:{cause_identity(error)}"],
            )
        else:
            _failure(
                "command_failed",
                [arguments.command, f"cause:{cause_identity(error)}"],
            )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
