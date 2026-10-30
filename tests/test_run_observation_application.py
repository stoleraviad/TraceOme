import asyncio
import hashlib
import io
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import unittest
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.agentic_journey.model_availability import (
    RetryModelAvailabilityResponse,
)
from bio_neb.agentic_journey.planning_assignment import (
    planning_assignment,
)
from bio_neb.application import (
    AnswerReview,
    BeginRun,
    ExecutionReviewAnswer,
    ModelAvailabilityReviewAnswer,
    ObserveRun,
    PlanningReviewAnswer,
    RefusalDisposition,
    ResumeRun,
    ReviewAnswerRefused,
    RunAdvancementRefused,
    advance,
    observe,
)
from bio_neb.application.advancement import (
    answer_review_from_json,
    use_run_advancement_composition,
)
from bio_neb.application.commands import (
    RegisterFastqInput,
    RegisterRunCommand,
    register_run,
)
from bio_neb.application.planning_read import read_planning
from bio_neb.application.queries import project_journey_events
from bio_neb.application.run_interaction import run_interaction
from bio_neb.cli import main
from bio_neb.journey import Journey
from bio_neb.parabricks import accepted_pre_execution
from bio_neb.parabricks.accepted_pre_execution import (
    AcceptedPreExecutionBindingError,
)
from bio_neb.parabricks.execution_proposal import (
    ApproveExecutionResponse,
    ExecutionHitlResponseRefused,
    RejectExecutionResponse,
    ReviseExecutionResponse,
    compile_execution_proposal,
)
from bio_neb.parabricks.planning_commands import (
    RespondPlanningHitlCommand,
    respond_planning_hitl,
)
from bio_neb.parabricks.planning_hitl import (
    ClarifyPlanningHitlResponse,
    PlanningHitlRequested,
    PlanningHitlResponseRefused,
    SelectPlanningHitlResponse,
    StopPlanningHitlResponse,
)
from bio_neb.parabricks.proposal_information import (
    RegisteredInputRequirementDraft,
    ToFindKnowledgeDraft,
    ToolEvidenceValidation,
)
from bio_neb.providers.nebius_model_endpoint import NebiusModelEndpointAdapter
from bio_neb.recording import read_evidence
from bio_neb.reference_discovery.reference_identity_tool import (
    TaxonomyAdapterObservation,
)
from tests.application_fakes import (
    _grounded_draft,
    _registered,
    operator_sigint,
)
from tests.execution_approval import approve_execution
from tests.explainer_fakes import explained_by_default
from tests.input_correction_fakes import prepare_input_correction
from tests.model_endpoint_fakes import ScriptedModelEndpoint, corrections
from tests.planning_assignment_fakes import PlanningComposition, eligible_tools
from tests.proposal_information_fakes import InspectionModel
from tests.reference_fakes import (
    PREBUILT_INDEX_TOOLS,
    CompanionAdapter,
    ReferenceAdapter,
    reference_call,
    reference_const_arguments,
    reference_source,
)
from tests.terminal_journey_fakes import EVIDENCE_REQUEST


@contextmanager
def _recorded_as_formerly_admitted(rewrite):
    """Record the plan in a shape an earlier admission accepted (before #207).

    Admission now refuses these shapes, but the plan is create-once, so a Journey
    recorded earlier still carries one. ``rewrite`` maps the admitted draft's
    requirements to the recorded ones; grounding and the writer stay real.
    """
    from bio_neb.agentic_journey import planning_assignment as owner

    ground = owner._ground_draft

    def recorded(events, draft):
        return ground(
            events,
            draft.model_copy(update={"requirements": rewrite(draft.requirements)}),
        )

    with patch.object(owner, "_ground_draft", recorded):
        yield


def _unfindable_input(requirements):
    """Add a further FASTQ input no Tool can find (needs_tool once recorded)."""
    # The same request context the admitted taxon cites as unstated.
    unstated = next(
        item.knowledge for item in requirements if item.kind == "taxon_identity"
    )
    return (
        *requirements,
        RegisteredInputRequirementDraft.model_construct(
            kind="registered_input",
            subject_key="additional_input",
            domain="operational",
            meaning="A further FASTQ input nobody registered.",
            knowledge=ToFindKnowledgeDraft(
                evidence_event_ids=unstated.evidence_event_ids
            ),
            acquisition_route="validate_with_tool",
            validation=ToolEvidenceValidation(
                requirement="Find the further FASTQ input."
            ),
        ),
    )


def _planning_readiness(journey):
    """Planning readiness as an Agent-side reader takes it: the checked view and
    evidence it already holds (ADR 0032)."""
    from bio_neb.application.planning_read import current_planning, planning_evidence
    from bio_neb.parabricks.planning_readiness import readiness_of

    return readiness_of(current_planning(journey), planning_evidence(journey))


def _file_snapshot(root: Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _reference_facts_resolved_journey(root: Path, *, capability=None):
    receipt, paths = _registered(root)
    composition = PlanningComposition(receipt, paths)
    if capability is not None:
        composition.capability = capability
    composition.companion_adapter = CompanionAdapter(
        "GCF_000001405.40", asset_ids=("all_genomic_fasta",)
    )
    with use_run_advancement_composition(composition):
        pending = advance(
            ResumeRun(journey_file=receipt.journey_file)
        ).planning_hitl.pending_request
    assert pending is not None
    respond_planning_hitl(
        RespondPlanningHitlCommand(
            journey_file=receipt.journey_file,
            request_event_id=pending.request_event_id,
            response=SelectPlanningHitlResponse(
                action="select", candidate_id=pending.candidates[0].candidate_id
            ),
        )
    )
    resolved = observe(ObserveRun(journey_file=receipt.journey_file))
    assert resolved.reference_facts.status == "reference_facts"
    return receipt, paths


def _answer_selectable_planning_reviews(journey_file: str, current):
    for _ in range(8):
        planning = current.planning_hitl.pending_request
        if planning is not None:
            current = advance(
                AnswerReview(
                    journey_file=journey_file,
                    request_event_id=planning.request_event_id,
                    answer=PlanningReviewAnswer(
                        kind="planning",
                        response=SelectPlanningHitlResponse(
                            action="select",
                            candidate_id=planning.candidates[0].candidate_id,
                        ),
                    ),
                )
            )
            continue
        return current
    raise AssertionError("selectable planning Reviews did not reach a stable stop")


class _UnknownTaxonAdapter(ReferenceAdapter):
    """Taxonomy finds nothing, so the Agent's own question is the taxon."""

    def resolve_taxon(self, query):
        self.contact_count += 1
        return TaxonomyAdapterObservation(
            records=(),
            source=reference_source("ncbi_taxonomy", "taxonomy/not-found"),
        )


class _InterruptingTaxonomy(ReferenceAdapter):
    """The operator's SIGINT arrives while the first taxonomy lookup runs."""

    interrupted = False

    def resolve_taxon(self, query):
        if not self.interrupted:
            self.interrupted = True
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.5)  # the coordinator is cancelled while this Tool runs
        return super().resolve_taxon(query)


def _insert_copy(events, fact, *, before, **payload):
    """Retain a re-ided copy of ``fact`` in ``before``'s causal place."""
    forged = deepcopy(fact)
    forged["event_id"] = str(uuid.uuid4())
    forged["payload"].update(payload)
    forged["parent_ids"] = before["parent_ids"]
    before["parent_ids"] = [forged["event_id"]]
    events.insert(events.index(before), forged)


class _OfflineAdvancementComposition(PlanningComposition):
    environ = {"LOCAL_MODEL_KEY": "offline-only"}

    def __init__(
        self,
        receipt,
        paths,
        *,
        open_target: bool = False,
        tool_found_target: bool = False,
        independent_targets: bool = False,
        open_sibling_target: bool = False,
        ask_reference: bool = False,
        open_target_quote: str | None = None,
        untyped_sibling_quote: str | None = None,
    ):
        super().__init__(receipt, paths)
        if ask_reference:
            self.reference_adapter = _UnknownTaxonAdapter()
        self.ask_reference = ask_reference
        # The request names no target in words of its own: the coordinator asks.
        self.open_target = open_target
        # The open target's optional to_find quote: what to find, not a value.
        self.open_target_quote = open_target_quote
        # The same open target, recorded to be found by a Tool after Execution.
        self.tool_found_target = tool_found_target
        self.independent_targets = independent_targets
        # Beside the stated target, a second one still open for the researcher.
        self.open_sibling_target = open_sibling_target
        # Beside it, a second target in request words that state no rsID or position.
        self.untyped_sibling_quote = untyped_sibling_quote

    def input_inspection_attempt(self, observation):
        return planning_assignment(
            model_factory=lambda _, _selected: (
                InspectionModel(observation.registered_input_set.input_set_id).model(
                    _, _selected
                )
            )
        )

    async def respond(self, messages, info):
        tools = set(eligible_tools(self.journey, info.function_tools))
        if self.ask_reference and "request_planning_hitl" in tools:
            tool = next(
                t for t in info.function_tools if t.name == "request_planning_hitl"
            )
            self.calls.append(tool.name)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool.name,
                        reference_const_arguments(tool),
                        tool_call_id=f"assignment-{len(self.calls)}",
                    )
                ]
            )
        if (
            not self.calls
            or "inspect_input_metadata" in tools
            or "record_proposal_requirements" not in tools
        ):
            return await super().respond(messages, info)
        # Only the recorded requirements vary; every other call is the base script.
        draft = _grounded_draft(
            self.registered,
            self.paths,
            registered_requests=Journey.open(Path(self.journey.path)).checked_view(
                "investigation_request"
            ),
        )
        target = next(
            item
            for item in draft["requirements"]
            if item["kind"] == "scientific_investigation"
        )
        if self.open_target or self.tool_found_target:
            _open(
                target,
                self.registered.request.source_event_id,
                quote=self.open_target_quote,
            )
            if self.tool_found_target:
                target["acquisition_route"] = "validate_with_tool"
                target["validation"] = {
                    "kind": "tool_evidence",
                    "requirement": "Find the target after Execution.",
                }
        if self.independent_targets:
            sibling = deepcopy(target)
            sibling["subject_key"] = "additional_target"
            draft["requirements"].append(sibling)
        if self.open_sibling_target:
            sibling = deepcopy(target)
            sibling["subject_key"] = "additional_target"
            _open(sibling, self.registered.request.source_event_id)
            draft["requirements"].append(sibling)
        if self.untyped_sibling_quote is not None:
            sibling = deepcopy(target)
            sibling["subject_key"] = "additional_target"
            sibling["knowledge"] = {
                "state": "asserted",
                "quote": self.untyped_sibling_quote,
            }
            draft["requirements"].append(sibling)
        self.calls.append("record_proposal_requirements")
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "record_proposal_requirements",
                    draft,
                    tool_call_id=f"assignment-{len(self.calls)}",
                )
            ]
        )


def _open(target, request_event_id, *, quote=None):
    """The model names the target in no words of the request: it is open."""
    target["knowledge"] = {"state": "to_find", "evidence_event_ids": [request_event_id]}
    if quote is not None:
        target["knowledge"]["quote"] = quote
    target["acquisition_route"] = "ask_planning_hitl"
    target["validation"] = {
        "kind": "researcher_decision",
        "requirement": "The researcher names the targets.",
    }


# A recorded human Run: the two scientific_investigation items planning recorded,
# copied verbatim. Only their cited event ids are mapped onto the offline Journey:
# the request event f9b6a98d and the registered request file bbae30e4.
_RECORDED_REQUEST_FILE_ITEMS = (
    {
        "acquisition_route": "ask_planning_hitl",
        "domain": "scientific",
        "investigation_purpose": "known_variant",
        "kind": "scientific_investigation",
        "knowledge": {"evidence_event_ids": ["<file>"], "state": "to_find"},
        "meaning": "The variants listed in the researcher's request file to check in HG001",
        "subject_key": "targets_from_request_file",
        "validation": {
            "kind": "researcher_decision",
            "requirement": "the targets to investigate",
        },
    },
    {
        "acquisition_route": "ask_planning_hitl",
        "domain": "scientific",
        "investigation_purpose": "open_question",
        "kind": "scientific_investigation",
        "knowledge": {"evidence_event_ids": ["<request>"], "state": "to_find"},
        "meaning": "The targets the researcher wants investigated in HG001",
        "subject_key": "targets_from_request",
        "validation": {
            "kind": "researcher_decision",
            "requirement": "the targets to investigate",
        },
    },
)
# The recorded request file (snp-request.json): three untyped items.
_RECORDED_REQUEST_FILE = {
    "schema_version": "bio-neb.investigation-request/3",
    "asserted_by": "researcher:owner",
    "requests": [
        {"request": "Chromosome 19 position 44908822 on GRCh38."},
        {"request": "The APOE site at chromosome 19 position 45411941 on GRCh37."},
        {"request": "Also check rs429358."},
    ],
}


class _RecordedRequestFileComposition(_OfflineAdvancementComposition):
    """Planning records #480's two investigation items beside the request file; told
    the second is refused, it records the plan again without it."""

    recorded = False

    async def respond(self, messages, info):
        response = await super().respond(messages, info)
        part = response.parts[0]
        if part.tool_name != "record_proposal_requirements":
            return response
        ids = {
            "<request>": self.registered.request.source_event_id,
            "<file>": Journey.open(Path(self.journey.path))
            .checked_view("investigation_request")[0]
            .source_event_id,
        }
        items = []
        for recorded in _RECORDED_REQUEST_FILE_ITEMS:
            item = deepcopy(recorded)
            item["knowledge"]["evidence_event_ids"] = [
                ids[event_id] for event_id in item["knowledge"]["evidence_event_ids"]
            ]
            items.append(item)
        if self.recorded:
            items = items[:1]
        self.recorded = True
        part.args["requirements"] = [
            item
            for item in part.args["requirements"]
            if item["kind"] != "scientific_investigation"
        ] + items
        return response


class _SecondRequirementComposition(_OfflineAdvancementComposition):
    """The model records a second requirement of one kind; admission allows it."""

    def __init__(self, *args, second_kind, **kwargs):
        super().__init__(*args, **kwargs)
        self.second_kind = second_kind

    async def respond(self, messages, info):
        response = await super().respond(messages, info)
        part = response.parts[0]
        if part.tool_name == "record_proposal_requirements":
            requirements = part.args["requirements"]
            second = deepcopy(
                next(item for item in requirements if item["kind"] == self.second_kind)
            )
            second["subject_key"] = self.second_kind + "_second"
            requirements.append(second)
        return response


class _SearchingComposition(_OfflineAdvancementComposition):
    """With no Tool left that applies, the model keeps searching the Journey."""

    async def respond(self, messages, info):
        # The genomic FASTA's index Tools stay callable after its choice.
        tools = eligible_tools(self.journey, info.function_tools)
        if (
            self.calls
            and set(tools) - set(PREBUILT_INDEX_TOOLS) <= {"search_journey"}
            and reference_call(Journey.open(Path(self.journey.path)), tools) is None
        ):
            self.calls.append("search_journey")
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "search_journey",
                        {"kinds": ["ProposalRequirementsIdentified"]},
                        tool_call_id=f"assignment-{len(self.calls)}",
                    )
                ]
            )
        return await super().respond(messages, info)


class RunObservationApplicationTests(unittest.TestCase):
    def test_abort_and_scoped_reject_do_not_restart_reference_work_behind_target_question(
        self,
    ):
        for action in ("stop", "reject"):
            with (
                self.subTest(action=action),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                receipt, paths = _registered(root)
                composition = _OfflineAdvancementComposition(
                    receipt, paths, open_target=True, ask_reference=True
                )
                with use_run_advancement_composition(composition):
                    current = advance(ResumeRun(journey_file=receipt.journey_file))
                    target = next(
                        r
                        for r in current.planning_hitl.pending_requests
                        if r.owner == "agent_attempt"
                    )
                    sibling = next(
                        r
                        for r in current.planning_hitl.pending_requests
                        if r.owner == "requirement_resolution"
                    )
                    attempts = tuple(a.attempt_id for a in current.attempts)
                    response = answer_review_from_json(
                        current,
                        target.request_event_id,
                        json.dumps(
                            {"action": action, "reason": "Do not continue this work."}
                        ),
                    )
                    answered = advance(response)
                    self.assertEqual(
                        tuple(a.attempt_id for a in answered.attempts), attempts
                    )
                    path = Path(receipt.journey_file)
                    frozen = _file_snapshot(path.parent)
                    self.assertEqual(advance(response), answered)
                    self.assertEqual(
                        advance(ResumeRun(journey_file=receipt.journey_file)), answered
                    )
                    self.assertEqual(_file_snapshot(path.parent), frozen)
                    self.assertEqual(
                        observe(ObserveRun(journey_file=receipt.journey_file)), answered
                    )
                    sibling_answer = answer_review_from_json(
                        answered,
                        sibling.request_event_id,
                        json.dumps({"action": "clarify", "instruction": "rs1234"}),
                    )
                    if action == "stop":
                        self.assertEqual(answered.status, "finalization_review_pending")
                        with self.assertRaises(ValueError):
                            advance(sibling_answer)
                        self.assertEqual(_file_snapshot(path.parent), frozen)
                    else:
                        resolved = advance(sibling_answer)
                        self.assertEqual(
                            tuple(a.attempt_id for a in resolved.attempts), attempts
                        )
                        self.assertEqual(resolved.status, "finalization_review_pending")
                        self.assertFalse(
                            any(
                                e["event_type"] == "RunFailed"
                                for e in Journey.open(path).read()
                            )
                        )

    def test_public_answers_preserve_an_independent_pending_question_and_rejection(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            composition = _OfflineAdvancementComposition(
                receipt, paths, open_target=True, independent_targets=True
            )
            with use_run_advancement_composition(composition):
                current = advance(ResumeRun(journey_file=receipt.journey_file))
                for _ in range(8):
                    pending = next(
                        (
                            r
                            for r in current.planning_hitl.pending_requests
                            if r.owner != "requirement_resolution"
                        ),
                        None,
                    )
                    if pending is None:
                        break
                    current = advance(
                        answer_review_from_json(
                            current,
                            pending.request_event_id,
                            json.dumps(
                                {
                                    "action": "select",
                                    "candidate_id": pending.candidates[0].candidate_id,
                                }
                            ),
                        )
                    )
                self.assertEqual(len(current.pending_reviews), 2)
                first, second = current.pending_reviews
                self.assertEqual(current.reference_facts.status, "reference_facts")
                self.assertTrue(
                    all(p.closed_event_id is not None for p in current.passes)
                )
                completed_attempts = tuple(a.attempt_id for a in current.attempts)
                answer = answer_review_from_json(
                    current,
                    second.request_event_id,
                    json.dumps({"action": "clarify", "instruction": "rs1234"}),
                )
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(
                        main(
                            [
                                "answer",
                                "--journey-file",
                                receipt.journey_file,
                                "--request-event-id",
                                second.request_event_id,
                                "--response-json",
                                json.dumps(
                                    {"action": "clarify", "instruction": "rs1234"}
                                ),
                            ]
                        ),
                        0,
                    )
                self.assertEqual(
                    [
                        r["request_event_id"]
                        for r in json.loads(output.getvalue())["pending_reviews"]
                    ],
                    [first.request_event_id],
                )
                current = observe(ObserveRun(journey_file=receipt.journey_file))
                self.assertEqual(
                    [r.request_event_id for r in current.pending_reviews],
                    [first.request_event_id],
                )
                named = {
                    item.requirement_id: [(t.words, t.rsid) for t in item.targets]
                    for item in current.requirement_resolution.scientific_investigation.items
                }
                # The answer goes back to planning: one Attempt reads it; every
                # earlier Attempt is unchanged.
                attempts = tuple(a.attempt_id for a in current.attempts)
                self.assertEqual(
                    attempts[: len(completed_attempts)], completed_attempts
                )
                self.assertEqual(len(attempts), len(completed_attempts) + 1)
                # The answered item names its target; the other is still open.
                self.assertEqual(
                    named[second.request.requirement_id], [("rs1234", "rs1234")]
                )
                self.assertEqual(
                    [
                        u.requirement_id
                        for u in current.requirement_resolution.unsettled_targets
                    ],
                    [first.request.requirement_id],
                )
                path = Path(current.journey_file)
                frozen = _file_snapshot(path.parent)
                self.assertEqual(advance(answer), current)
                self.assertEqual(_file_snapshot(path.parent), frozen)
                rejected = advance(
                    answer_review_from_json(
                        current,
                        first.request_event_id,
                        json.dumps(
                            {"action": "reject", "reason": "Do not pursue this output."}
                        ),
                    )
                )
                # The rejected target is left out: the Run goes on with the
                # named one to its next stop, with no target question left.
                self.assertEqual(rejected.requirement_resolution.unsettled_targets, ())
                self.assertEqual(
                    (rejected.status, [r.kind for r in rejected.pending_reviews]),
                    ("execution_planning_ready", ["preparation"]),
                )
                self.assertFalse(
                    any(
                        e["event_type"] == "RunFailed"
                        for e in Journey.open(path).read()
                    )
                )
                self.assertEqual(
                    next(
                        r
                        for r in rejected.planning_hitl.receipts
                        if r.request_event_id == second.request_event_id
                    ).response.action,
                    "clarify",
                )
                self.assertEqual(observe(ObserveRun(journey_file=str(path))), rejected)

    def test_default_resume_holds_on_a_changed_worker_contract_before_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _reference_facts_resolved_journey(root)
            setup_events = len(Journey.open(Path(receipt.journey_file)).read())
            contract = root / "changed-worker-exchange.py"
            contract.write_text("# source differs from the executable dependency pin\n")
            with (
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "offline-only"}),
                patch.object(
                    accepted_pre_execution.accepted_worker.worker_exchange,
                    "__file__",
                    str(contract),
                ),
                patch.object(
                    socket.socket,
                    "connect",
                    side_effect=AssertionError("network contact"),
                ),
                patch.object(
                    subprocess, "Popen", side_effect=AssertionError("process contact")
                ),
                self.assertRaises(RunAdvancementRefused) as refused,
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(
                (
                    refused.exception.code,
                    refused.exception.failure.cause,
                    refused.exception.disposition.kind,
                ),
                (
                    "accepted_worker_binding_refused",
                    "accepted_worker_source_unavailable",
                    "hold_for_fix",
                ),
            )
            self.assertIsInstance(
                refused.exception.__cause__, AcceptedPreExecutionBindingError
            )
            self._assert_held_before_any_attempt(receipt, setup_events)

            # Fix and resume: with the worker restored, the same Run continues in
            # a fresh Pass and planning starts its next Attempt.
            composition = PlanningComposition(receipt, paths)
            composition.companion_adapter = CompanionAdapter(
                "GCF_000001405.40", asset_ids=("all_genomic_fasta",)
            )
            with use_run_advancement_composition(composition):
                resumed = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertNotEqual(resumed.status, "run_held")
            fresh = Journey.open(Path(receipt.journey_file)).read()[setup_events + 2 :]
            self.assertEqual(fresh[0]["event_type"], "PassStarted")
            self.assertIn("AttemptStarted", [event["event_type"] for event in fresh])

    def test_default_resume_holds_on_a_changed_worker_digest_before_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _reference_facts_resolved_journey(Path(directory).resolve())
            setup_events = len(Journey.open(Path(receipt.journey_file)).read())
            with (
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "offline-only"}),
                patch.object(
                    accepted_pre_execution, "ATTESTED_WORKER_SHA256", "0" * 64
                ),
                patch.object(
                    socket.socket,
                    "connect",
                    side_effect=AssertionError("network contact"),
                ),
                patch.object(
                    subprocess, "Popen", side_effect=AssertionError("process contact")
                ),
                self.assertRaises(RunAdvancementRefused) as refused,
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(
                (
                    refused.exception.code,
                    refused.exception.failure.cause,
                    refused.exception.disposition.kind,
                ),
                (
                    "accepted_worker_binding_refused",
                    "accepted_worker_sha256_mismatch",
                    "hold_for_fix",
                ),
            )
            self.assertIsInstance(
                refused.exception.__cause__, AcceptedPreExecutionBindingError
            )
            self._assert_held_before_any_attempt(receipt, setup_events)

    def _assert_held_before_any_attempt(self, receipt, setup_events):
        # Only the hold is recorded: no Attempt starts after the drift.
        resumed = Journey.open(Path(receipt.journey_file)).read()[setup_events:]
        self.assertEqual(
            [event["event_type"] for event in resumed], ["PassStarted", "PassClosed"]
        )
        self.assertEqual(resumed[-1]["payload"]["disposition"], "held")
        observed = observe(ObserveRun(journey_file=receipt.journey_file))
        self.assertEqual(observed.status, "run_held")
        self.assertEqual(observed.reference_facts.status, "reference_facts")

    def test_default_resume_binds_accepted_execution_capability_policy(self):
        # Offline accepted-source fixture; deployed image attestation remains
        # unchanged until the separately authorized worker publication proof.
        with (
            patch.object(
                accepted_pre_execution,
                "ATTESTED_WORKER_SHA256",
                hashlib.sha256(
                    accepted_pre_execution.accepted_worker.worker_source_bytes()
                ).hexdigest(),
            ),
            tempfile.TemporaryDirectory() as directory,
        ):
            # The setup answers the recipe question under the same accepted policy;
            # the resumed Attempt then binds it by default (no configuration exists).
            accepted = accepted_pre_execution._accepted_capability_snapshot(
                SimpleNamespace(read=lambda: [])
            )
            receipt, _ = _reference_facts_resolved_journey(
                Path(directory).resolve(), capability=accepted
            )
            setup_events = len(Journey.open(Path(receipt.journey_file)).read())
            planning = PlanningComposition(receipt, paths=None)
            contact_ledger = []

            def refuse_contact(kind):
                def refuse(*_args, **_kwargs):
                    contact_ledger.append(kind)
                    raise AssertionError(f"default execution planning attempted {kind}")

                return refuse

            with (
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "offline-only"}),
                # The default composition constructs the real Adapter; only its
                # HTTP transport is scripted.
                patch(
                    "bio_neb.providers.nebius_model_endpoint."
                    "NebiusModelEndpointAdapter",
                    side_effect=ScriptedModelEndpoint(
                        planning.respond, adapter=NebiusModelEndpointAdapter
                    ),
                ),
                patch.object(
                    socket.socket, "connect", side_effect=refuse_contact("network")
                ),
                patch.object(
                    subprocess, "Popen", side_effect=refuse_contact("process")
                ),
            ):
                review = advance(ResumeRun(journey_file=receipt.journey_file))
            resumed = Journey.open(Path(receipt.journey_file)).read()[setup_events:]
            sources = [
                event["payload"]["result"]["source"]
                for event in resumed
                if event["event_type"]
                in {"RecipeObservationRecorded", "ResourceObservationRecorded"}
            ]
            self.assertEqual(
                {
                    "status": review.status,
                    "last_agent_role": review.attempts[-1].agent_role,
                    "observations": len(sources),
                    "capability_versions": {
                        source["capability_version"] for source in sources
                    },
                    "snapshot_ids": {source["snapshot_id"] for source in sources},
                    "policy_authorities": {source["authority"] for source in sources},
                    "policy_claims_retained": all(
                        source[name]
                        for source in sources
                        for name in ("source_manifest", "images", "tools")
                    ),
                    "contact": tuple(contact_ledger),
                },
                {
                    "status": "execution_planning_ready",
                    "last_agent_role": "planning_agent",
                    "observations": 2,
                    "capability_versions": {"bio-neb-parabricks-germline-t125j/1"},
                    "snapshot_ids": {accepted.snapshot_id},
                    "policy_authorities": {"compiled_capability_snapshot"},
                    "policy_claims_retained": True,
                    "contact": (),
                },
            )

    def test_default_resume_admits_discovered_reference_source_for_execution_review(
        self,
    ):
        from tests.planning_assignment_fakes import configured_planning_run
        from tests.reference_fakes import SOURCE_GZIP, SOURCE_MD5

        with tempfile.TemporaryDirectory() as directory:
            receipt, composition, _ = configured_planning_run(Path(directory).resolve())
            # Planning opens its first question before the guarded region, so the
            # model client's process-local platform probe is not counted as contact.
            with use_run_advancement_composition(composition):
                first = advance(ResumeRun(journey_file=receipt.journey_file))
            contact_ledger = []

            def refuse_contact(kind):
                def refuse(*_args, **_kwargs):
                    contact_ledger.append(kind)
                    raise AssertionError(
                        f"default proposal preparation attempted {kind}"
                    )

                return refuse

            with (
                patch.object(
                    socket.socket, "connect", side_effect=refuse_contact("network")
                ),
                patch.object(
                    subprocess, "Popen", side_effect=refuse_contact("process")
                ),
                use_run_advancement_composition(composition),
            ):
                review = _answer_selectable_planning_reviews(
                    receipt.journey_file, first
                )

            pending = review.execution_hitl.pending_request
            assert pending is not None
            proposal_admission = pending.proposal.reference_source_admission
            assert proposal_admission is not None
            source_events = tuple(
                event
                for event in Journey.open(Path(receipt.journey_file)).read()
                if event["event_type"] == "ReferenceSourceAdmitted"
            )
            self.assertEqual(len(source_events), 1)
            admission = source_events[0]["payload"]["admission"]
            self.assertEqual(
                {
                    "status": review.status,
                    "reference_asset_id": admission["reference_asset_id"],
                    "assembly_accession": admission["assembly_accession"],
                    "source_locator": admission["source_locator"],
                    "expected_size_bytes": admission["expected_size_bytes"],
                    "expected_checksum_algorithm": admission[
                        "expected_checksum_algorithm"
                    ],
                    "expected_checksum": admission["expected_checksum"],
                    "proposal_admission": proposal_admission.model_dump(mode="json"),
                    "contact": tuple(contact_ledger),
                },
                {
                    # The source the discovery fixture observed; nobody typed it.
                    "status": "execution_hitl_pending",
                    "reference_asset_id": "source-owned-opaque-id",
                    "assembly_accession": "GCF_000001405.40",
                    "source_locator": "https://ftp.ncbi.nlm.nih.gov/genomes/all/offline/"
                    "GCF_000001405.40_genomic.fna.gz",
                    "expected_size_bytes": len(SOURCE_GZIP),
                    "expected_checksum_algorithm": "md5",
                    "expected_checksum": SOURCE_MD5,
                    "proposal_admission": admission,
                    "contact": (),
                },
            )

    def test_a_dispatch_that_writes_nothing_holds_the_run_until_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            journey_path = Path(receipt.journey_file)
            setup_events = len(Journey.open(journey_path).read())

            # The bug: the coordinator dispatches inspection, which records nothing.
            with (
                patch(
                    "bio_neb.input_inspection.commands.inspect_registered_inputs",
                    lambda command, journey: None,
                ),
                self.assertRaises(RunAdvancementRefused) as raised,
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))

            error = raised.exception
            self.assertEqual(
                (error.code, error.status, error.disposition.kind),
                ("coordinator_dispatch_wrote_nothing", "run_held", "hold_for_fix"),
            )
            self.assertEqual(
                error.failure.model_dump(mode="json"),
                {
                    "failure_class": "invariant",
                    "invariant_id": "coordinator_dispatch_wrote_nothing",
                    "cause": "inputs_registered",
                    "evidence": {"status": "inputs_registered"},
                },
            )
            # The failure module recorded the hold in the Journey: one Pass,
            # closed held with the failure, and nothing else.
            held_events = Journey.open(journey_path).read()[setup_events:]
            self.assertEqual(
                [event["event_type"] for event in held_events],
                ["PassStarted", "PassClosed"],
            )
            closed = held_events[-1]["payload"]
            self.assertEqual(closed["disposition"], "held")
            self.assertEqual(closed["failure"], error.failure.model_dump(mode="json"))
            held = observe(ObserveRun(journey_file=receipt.journey_file))
            self.assertEqual(held.status, "run_held")
            self.assertEqual(held.passes[-1].failure, error.failure)
            self.assertEqual(run_interaction(held).frontier, "held")

            # After the fix, resume opens a fresh Pass on the same Run and it
            # progresses to its next lawful stop.
            with use_run_advancement_composition(PlanningComposition(receipt, paths)):
                resumed = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(resumed.journey_id, held.journey_id)
            self.assertEqual(resumed.status, "planning_hitl_pending")
            fresh = Journey.open(journey_path).read()[setup_events + 2 :]
            self.assertEqual(fresh[0]["event_type"], "PassStarted")
            self.assertEqual(
                resumed.passes[-2].closed_event_id, held_events[-1]["event_id"]
            )

    def _assert_held(self, raised, code, status=None, cause=None):
        error = raised.exception
        self.assertEqual(
            (
                error.code,
                error.status,
                error.disposition.kind,
                error.failure.failure_class,
                error.failure.invariant_id,
                error.failure.evidence.get("status"),
            ),
            (code, "run_held", "hold_for_fix", "invariant", code, status),
        )
        if cause is not None:
            self.assertEqual(error.failure.cause, cause)
        closed = Journey.open(Path(error.journey_file)).latest_event(("PassClosed",))
        self.assertEqual(closed["payload"]["disposition"], "held")
        self.assertEqual(
            closed["payload"]["failure"], error.failure.model_dump(mode="json")
        )
        return error

    def test_a_target_question_without_its_plan_holds_the_run(self):
        from bio_neb.application import advancement

        def every_pass_asks(observation, journey_file):
            # The coordinator asks for targets while the observation has no plan.
            advancement._open_target_questions(
                Journey.open(Path(journey_file)), observation
            )
            return observation

        with tempfile.TemporaryDirectory() as directory:
            receipt, _paths = _registered(Path(directory).resolve())
            with (
                patch.object(advancement, "_admit_target_questions", every_pass_asks),
                self.assertRaises(RunAdvancementRefused) as raised,
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            self._assert_held(
                raised, "target_question_basis_missing", "inputs_registered"
            )

    def test_a_refused_proposal_compilation_holds_the_run(self):
        from bio_neb.parabricks.preparation import PreparationRefused
        from tests.planning_assignment_fakes import (
            configured_planning_run,
            execution_configuration,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, base, _ = configured_planning_run(root)
            composition = _OfflineAdvancementComposition(receipt, base.paths)
            composition.capability = execution_configuration(
                root
            ).runtime_selection.capability
            refused = PreparationRefused(
                "execution_package_lineage_changed", ("execution_planning",)
            )
            with (
                patch(
                    "bio_neb.parabricks.preparation.prepare_execution",
                    lambda journey: refused,
                ),
                use_run_advancement_composition(composition),
                self.assertRaises(RunAdvancementRefused) as raised,
            ):
                current = advance(ResumeRun(journey_file=receipt.journey_file))
                _answer_selectable_planning_reviews(receipt.journey_file, current)
            error = self._assert_held(
                raised,
                "proposal_compilation_refused",
                "execution_planning_ready",
                "execution_package_lineage_changed",
            )
            self.assertEqual(error.failure.evidence["details"], ["execution_planning"])

    def test_an_integrity_refusal_escaping_the_compiler_holds_the_run(self):
        # Q5: outside C5 an integrity refusal escapes `advance` raw; it is still
        # our bug, held under its family's id with its own code as cause.
        from bio_neb.application.commands import ApprovedExecutionCommandRefused
        from bio_neb.parabricks.execution_proposal import ProposalCompilationRefused
        from tests.planning_assignment_fakes import (
            configured_planning_run,
            execution_configuration,
        )

        faults = {
            # Submission recompiles the proposal the compiler just prepared.
            "proposal_compilation_refused": (
                "bio_neb.application.advancement.submit_execution_proposal",
                ProposalCompilationRefused(
                    "execution_package_evidence_changed", ("execution_planning",)
                ),
                {"details": ["execution_planning"]},
            ),
            # Preparation admits the discovered reference source.
            "approved_execution_command_refused": (
                "bio_neb.reference_discovery.commands.admit_discovered_reference_source",
                ApprovedExecutionCommandRefused("reference_source_admission_conflict"),
                {},
            ),
        }
        for code, (site, fault, evidence) in faults.items():
            with (
                self.subTest(code=code),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                receipt, base, _ = configured_planning_run(root)
                composition = _OfflineAdvancementComposition(receipt, base.paths)
                composition.capability = execution_configuration(
                    root
                ).runtime_selection.capability

                def failing(*_args, fault=fault, **_kwargs):
                    raise fault

                with (
                    patch(site, failing),
                    use_run_advancement_composition(composition),
                    self.assertRaises(RunAdvancementRefused) as raised,
                ):
                    current = advance(ResumeRun(journey_file=receipt.journey_file))
                    _answer_selectable_planning_reviews(receipt.journey_file, current)
                error = self._assert_held(raised, code, cause=fault.code)
                self.assertIs(error.__cause__, fault)
                self.assertEqual(error.failure.evidence, evidence)

    def test_an_unshapeable_model_request_holds_the_run_before_it_is_recorded(self):
        # Configuration read refuses an unoffered named Tool, so one reaching a
        # request is our bug: shaping raises an Invariant before the request is
        # recorded or sent, and the failure module holds the Run.
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationPolicy

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory).resolve())
            composition = _OfflineAdvancementComposition(receipt, paths)
            with (
                patch.object(
                    AgentInvocationPolicy,
                    "stage_tool_choice",
                    lambda _policy, _stage: "fetch_refsnp",
                ),
                use_run_advancement_composition(composition),
                self.assertRaises(RunAdvancementRefused) as raised,
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            error = self._assert_held(raised, "planning_attempt_failed")
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
        self.assertTrue(
            error.failure.cause.startswith("Invariant@nebius_model_endpoint")
        )
        (failed,) = [e for e in events if e["event_type"] == "AttemptFailed"]
        self.assertEqual(
            (failed["payload"]["code"], failed["payload"]["invariant_id"]),
            ("invariant", "model_request_invalid"),
        )
        self.assertEqual(
            error.failure.evidence["attempt_failed_event_id"], failed["event_id"]
        )
        # Nothing was recorded as a model request and nothing was sent.
        self.assertEqual(
            [e["event_type"] for e in events if e["event_type"].startswith("Model")],
            [],
        )
        self.assertEqual(composition.model_endpoint.bodies, [])

    def test_a_status_without_a_dispatch_branch_holds_the_run(self):
        from bio_neb.application import advancement

        project = advancement.observe

        def disagreeing(intent, **kwargs):
            # A malformed projection: a ready status with no execution planning.
            observation = project(intent, **kwargs)
            if observation.status != "inputs_registered":
                return observation
            return observation.model_copy(update={"status": "execution_planning_ready"})

        with tempfile.TemporaryDirectory() as directory:
            receipt, _paths = _registered(Path(directory).resolve())
            with (
                patch.object(advancement, "observe", disagreeing),
                self.assertRaises(RunAdvancementRefused) as raised,
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            self._assert_held(
                raised, "coordinator_status_unhandled", "execution_planning_ready"
            )

    def test_an_untyped_error_escaping_the_coordinator_holds_the_run(self):
        from pydantic import ValidationError

        from bio_neb.failure import Invariant

        # A coordinator-internal model validation is our bug, not the caller's
        # input contract: it holds the Run rather than refusing the command.
        internal_validation = ValidationError.from_exception_data(
            "InspectionResult", [{"type": "missing", "loc": ("digest",), "input": {}}]
        )
        faults = (
            (
                "coordinator_unhandled_error",
                ValueError("terminal source belongs to another Attempt"),
            ),
            ("coordinator_unhandled_error", internal_validation),
            ("inspection_bug", Invariant("inspection_bug", "our bug")),
        )
        for code, fault in faults:
            with (
                self.subTest(code=code, fault=type(fault).__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, _paths = _registered(Path(directory).resolve())

                def failing(command, journey, fault=fault):
                    raise fault

                with (
                    patch(
                        "bio_neb.input_inspection.commands.inspect_registered_inputs",
                        failing,
                    ),
                    self.assertRaises(RunAdvancementRefused) as raised,
                ):
                    advance(ResumeRun(journey_file=receipt.journey_file))
                error = self._assert_held(raised, code)
                self.assertIs(error.__cause__, fault)
                # The cause names the fault's class and raise site, not its text.
                self.assertTrue(
                    error.failure.cause.startswith(f"{type(fault).__name__}@")
                )
                self.assertEqual(error.failure.evidence, {})

    def test_every_run_status_is_in_exactly_one_projection_status_set(self):
        # Producer test: the status projection owns every RunStatus value through
        # its named sets; a value no set produces, or one two sets claim, fails
        # here before any coordinator list can miss it.
        from collections import Counter
        from typing import get_args

        from bio_neb.application.queries import RUN_STATUS_SETS, RunStatus

        statuses = get_args(RunStatus)
        claims = Counter(
            status for named in RUN_STATUS_SETS.values() for status in named
        )
        self.assertEqual(
            sorted(status for status in statuses if claims[status] != 1), []
        )
        self.assertEqual(set(claims), set(statuses))

    def test_every_run_status_has_one_coordinator_route(self):
        from typing import get_args

        from bio_neb.application import advancement
        from bio_neb.application import run_interaction as interaction
        from bio_neb.application.agent_reentry import _ANSWERED_REVIEWS
        from bio_neb.application.queries import (
            ENDED_RUN_STATUSES,
            REENTRY_STATUSES,
            RUN_STATUS_SETS,
            RunStatus,
        )

        statuses = set(get_args(RunStatus))
        routes = advancement.COORDINATOR_ROUTES
        self.assertEqual(set(routes), statuses)
        # Each route is a union of whole projection sets, and no set has two
        # routes: the table is derived, not a second copy.
        used = [
            status_set
            for status_sets in advancement._ROUTE_STATUS_SETS.values()
            for status_set in status_sets
        ]
        self.assertEqual(
            sorted(map(sorted, used)), sorted(map(sorted, RUN_STATUS_SETS.values()))
        )
        # Spot checks of the coordinator's meaning.
        for status, route in {
            "run_held": "stop",
            "failed": "stop",
            "investigation_evidence_ready": "stop",
            "finalization_returned_to_investigation": "stop",
            "finalization_stopped": "stop",
            "grounded_report_published": "stop",
            "execution_hitl_pending": "review",
            "investigation_review_pending": "review",
            "agent_work_review_pending": "review",
            "agent_work_review_direction_recorded": "dispatch",
            "finalization_approved": "dispatch",
            "execution_planning_ready": "dispatch",
            "execution_authorized": "dispatch",
            "execution_artifacts_verified": "execution_result",
        }.items():
            self.assertEqual(routes[status], route, status)
        for retired in (
            "investigation_not_started",
            "execution_in_progress",
            "execution_storage_ready",
        ):
            self.assertNotIn(retired, statuses)
        # The coordinator's lists are the projection's: the stop list, each
        # dispatch branch's set and the re-entry resolver's statuses.
        self.assertEqual(
            advancement._TERMINAL_STATUSES,
            {status for status in statuses if routes[status] == "stop"},
        )
        self.assertLessEqual(
            advancement._PLANNING_ASSIGNMENT_STATUSES,
            {status for status in statuses if routes[status] == "dispatch"},
        )
        self.assertNotIn(
            "execution_planning_ready", advancement._PLANNING_ASSIGNMENT_STATUSES
        )
        self.assertEqual(set(_ANSWERED_REVIEWS), REENTRY_STATUSES)
        # A held Run is not ended; every ended status stops the coordinator.
        self.assertNotIn("run_held", ENDED_RUN_STATUSES)
        self.assertLessEqual(ENDED_RUN_STATUSES, advancement._TERMINAL_STATUSES)
        # The run interaction gives every stop a frontier other than progress.
        self.assertLessEqual(
            advancement._TERMINAL_STATUSES,
            interaction._TERMINAL_RUN_STATUSES
            | interaction._CHECKPOINT_RUN_STATUSES
            | {"run_held", "grounded_report_published"},
        )
        self.assertNotIn("run_held", interaction._TERMINAL_RUN_STATUSES)

    def test_evidence_ready_with_nothing_to_do_is_a_quiet_wait(self):
        from bio_neb.application import advancement

        project = advancement.observe

        def evidence_ready(intent, **kwargs):
            # Investigation evidence is ready and no member is eligible.
            observation = project(intent, **kwargs)
            return observation.model_copy(
                update={"status": "investigation_evidence_ready"}
            )

        with tempfile.TemporaryDirectory() as directory:
            receipt, _paths = _registered(Path(directory).resolve())
            before = Path(receipt.journey_file).read_bytes()
            with patch.object(advancement, "observe", evidence_ready):
                waiting = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(waiting.status, "investigation_evidence_ready")
            self.assertEqual(Path(receipt.journey_file).read_bytes(), before)

    def test_a_closed_pass_records_a_failure_exactly_when_held(self):
        from bio_neb.failure import HeldFailure
        from bio_neb.recording import PassClosed

        closure = {
            "pass_id": str(uuid.uuid4()),
            "started_event_id": str(uuid.uuid4()),
            "journal_sha256": "0" * 64,
            "journal_bytes": 0,
        }
        failure = HeldFailure(
            failure_class="invariant", invariant_id="bug", cause="inputs_registered"
        )
        held = PassClosed(**closure, disposition="held", failure=failure)
        self.assertEqual(held.model_dump()["failure"]["invariant_id"], "bug")
        # Other closures keep their recorded shape, with no failure key.
        self.assertNotIn(
            "failure", PassClosed(**closure, disposition="completed").model_dump()
        )
        with self.assertRaises(ValidationError):
            PassClosed(**closure, disposition="held")
        with self.assertRaises(ValidationError):
            PassClosed(**closure, disposition="interrupted", failure=failure)

    def test_refusal_dispositions_name_only_their_closed_kinds(self):
        for kind in (
            "hold_for_fix",
            "caller_correction_and_reissue",
            "researcher_review_required",
        ):
            self.assertEqual(RefusalDisposition(kind=kind).kind, kind)
        for retired in ("terminal_for_run", "bounded_automatic_retry", "unknown"):
            with self.assertRaises(ValidationError):
                RefusalDisposition(kind=retired)
        with self.assertRaises(ValidationError):
            RefusalDisposition()

    def test_a_resume_that_writes_nothing_still_lifts_the_hold_durably(self):
        from bio_neb.application import advancement

        with tempfile.TemporaryDirectory() as directory:
            receipt, _paths = _registered(Path(directory).resolve())
            journey_path = Path(receipt.journey_file)
            with (
                patch(
                    "bio_neb.input_inspection.commands.inspect_registered_inputs",
                    lambda command, journey: None,
                ),
                self.assertRaises(RunAdvancementRefused) as wrote_nothing,
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(
                wrote_nothing.exception.code, "coordinator_dispatch_wrote_nothing"
            )
            held_events = len(Journey.open(journey_path).read())

            project = advancement.observe

            def quiet(intent, **kwargs):
                # After the fix the coordinator finds nothing to write.
                observation = project(intent, **kwargs)
                return observation.model_copy(
                    update={"status": "investigation_evidence_ready"}
                )

            with patch.object(advancement, "observe", quiet):
                advance(ResumeRun(journey_file=receipt.journey_file))
            # The resume's fresh Pass is recorded, so the Journey agrees: not held.
            fresh = Journey.open(journey_path).read()[held_events:]
            self.assertEqual(
                [(e["event_type"], e["payload"].get("disposition")) for e in fresh],
                [("PassStarted", None), ("PassClosed", "completed")],
            )
            self.assertEqual(
                observe(ObserveRun(journey_file=receipt.journey_file)).status,
                "inputs_registered",
            )

    def test_an_operator_interrupt_holds_the_run_until_resume(self):
        # The operator's SIGINT cancels the planning Attempt on
        # the exact source it met: between turns after a running Tool finishes, or
        # inside a model request. Either way the Run is held, still reads, and
        # `resume` continues it with one fresh Attempt.
        class InterruptingRequest(_OfflineAdvancementComposition):
            interrupted = False

            async def respond(self, messages, info):
                if (
                    not self.interrupted
                    and "record_proposal_requirements" in self.calls
                ):
                    self.interrupted = True
                    os.kill(os.getpid(), signal.SIGINT)
                    await asyncio.sleep(30)  # cancelled with the request in flight
                return await super().respond(messages, info)

        def tool_runs(receipt, paths):
            composition = _OfflineAdvancementComposition(receipt, paths)
            composition.reference_adapter = _InterruptingTaxonomy()
            return composition, lambda: composition.reference_adapter.interrupted

        def request_in_flight(receipt, paths):
            composition = InterruptingRequest(receipt, paths)
            return composition, lambda: composition.interrupted

        cases = (
            ("while a Tool runs", tool_runs, "ModelTurnCancelled"),
            ("in a model request", request_in_flight, "ModelPhysicalContactCancelled"),
        )
        for where, interrupting, source_type in cases:
            with (
                self.subTest(where=where),
                tempfile.TemporaryDirectory() as directory,
                operator_sigint(),
            ):
                receipt, paths = _registered(Path(directory).resolve())
                composition, interrupted = interrupting(receipt, paths)
                with (
                    use_run_advancement_composition(composition),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    advance(ResumeRun(journey_file=receipt.journey_file))
                journey = Journey.open(Path(receipt.journey_file))
                events = read_evidence(journey)
                closed = journey.latest_event(("PassClosed",))
                held = observe(ObserveRun(journey_file=receipt.journey_file))
                with use_run_advancement_composition(
                    _OfflineAdvancementComposition(receipt, paths)
                ):
                    resumed = advance(ResumeRun(journey_file=receipt.journey_file))

                self.assertTrue(interrupted())
                (cancelled,) = [
                    e for e in events if e["event_type"] == "AttemptCancelled"
                ]
                attempt_id = cancelled["payload"]["attempt_id"]
                owned = [
                    e["event_type"]
                    for e in events
                    if e["payload"].get("attempt_id") == attempt_id
                ]
                self.assertEqual(owned[-2:], [source_type, "AttemptCancelled"])
                self.assertEqual(
                    next(
                        e
                        for e in events
                        if e["event_id"] == cancelled["payload"]["source_event_id"]
                    )["event_type"],
                    source_type,
                )
                if source_type == "ModelTurnCancelled":
                    # The interrupted Tool finished and was receipted first.
                    self.assertEqual(owned[-3], "ObservedToolExecutionSucceeded")
                self.assertNotIn("AttemptFailed", [e["event_type"] for e in events])
                self.assertEqual(closed["payload"]["disposition"], "held")
                self.assertEqual(
                    closed["payload"]["failure"],
                    {
                        "failure_class": "interrupted",
                        "invariant_id": None,
                        "cause": "KeyboardInterrupt",
                        "evidence": {
                            "attempt_id": attempt_id,
                            "attempt_cancelled_event_id": cancelled["event_id"],
                        },
                    },
                )
                self.assertEqual(held.status, "run_held")
                self.assertEqual(held.attempts[-1].outcome, "cancelled")
                self.assertEqual(
                    held.attempts[-1].model_events[-1].event_type, source_type
                )
                self.assertEqual(resumed.status, "planning_hitl_pending")
                planning = [
                    a.outcome
                    for a in resumed.attempts
                    if a.agent_role == "planning_agent"
                ]
                self.assertEqual(planning[-2:], ["cancelled", "succeeded"])

    def test_a_tool_failing_after_an_interrupt_keeps_its_operational_review(self):
        # The interrupted Tool's own failure outranks the cancellation. Its
        # operational Review is recorded and holds the work as it does without
        # an interrupt; the interrupt still ends the command.
        from bio_neb.reference_discovery.reference_identity_tool import (
            ReferenceAdapterError,
        )

        class InterruptedRejection(ReferenceAdapter):
            def resolve_taxon(self, query):
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(0.5)  # the coordinator is cancelled while this runs
                raise ReferenceAdapterError(
                    "http_error", "offline source rejection", retryable=False
                )

        with tempfile.TemporaryDirectory() as directory, operator_sigint():
            receipt, paths = _registered(Path(directory).resolve())
            composition = _OfflineAdvancementComposition(receipt, paths)
            composition.reference_adapter = InterruptedRejection()
            with (
                use_run_advancement_composition(composition),
                self.assertRaises(KeyboardInterrupt),
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            journey = Journey.open(Path(receipt.journey_file))
            closed = journey.latest_event(("PassClosed",))
            interrupted = observe(ObserveRun(journey_file=receipt.journey_file))
            with use_run_advancement_composition(
                _OfflineAdvancementComposition(receipt, paths)
            ):
                resumed = advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))

        kinds = [e["event_type"] for e in events]
        (fault,) = [
            e for e in events if e["event_type"] == "OperationalFailureRecorded"
        ]
        self.assertEqual(
            (fault["payload"]["failure_class"], fault["payload"]["family"]),
            ("rejected", "source"),
        )
        self.assertEqual(kinds.count("AgentFailureReviewRequested"), 1)
        self.assertNotIn("AttemptCancelled", kinds)
        # The interrupt cancelled no Attempt, so it holds nothing; the Review does.
        self.assertEqual(closed["payload"]["disposition"], "interrupted")
        self.assertEqual(
            (interrupted.status, resumed.status),
            ("operational_review_pending", "operational_review_pending"),
        )
        self.assertEqual(
            [a.outcome for a in resumed.attempts if a.agent_role == "planning_agent"][
                -1
            ],
            "stopped",
        )
        self.assertEqual(kinds.count("AttemptStarted"), len(interrupted.attempts))

    def test_a_tool_crashing_after_an_interrupt_fails_its_attempt_as_without_one(
        self,
    ):
        # A mechanical Tool fault outranks the interrupt too; the Attempt fails on
        # it and the fault holds the Run.
        class InterruptedCrash(ReferenceAdapter):
            def resolve_taxon(self, query):
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(0.5)  # the coordinator is cancelled while this runs
                raise RuntimeError("offline adapter crash")

        with tempfile.TemporaryDirectory() as directory, operator_sigint():
            receipt, paths = _registered(Path(directory).resolve())
            composition = _OfflineAdvancementComposition(receipt, paths)
            composition.reference_adapter = InterruptedCrash()
            with (
                use_run_advancement_composition(composition),
                self.assertRaises(RunAdvancementRefused) as raised,
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            error = self._assert_held(raised, "planning_attempt_failed")

        (failed,) = [e for e in events if e["event_type"] == "AttemptFailed"]
        self.assertEqual(failed["payload"]["code"], "agent_run_failed")
        self.assertEqual(
            error.failure.evidence["attempt_failed_event_id"], failed["event_id"]
        )
        self.assertNotIn("AttemptCancelled", [e["event_type"] for e in events])

    def test_a_turn_cancellation_reads_only_as_its_exact_boundary(self):
        # A between-turns cancellation counts every model request, alone, and only
        # in a cancelled Attempt; a Journey the earlier writer left with no
        # Attempt terminal still refuses with its reason.
        with tempfile.TemporaryDirectory() as directory, operator_sigint():
            receipt, paths = _registered(Path(directory).resolve())
            composition = _OfflineAdvancementComposition(receipt, paths)
            composition.reference_adapter = _InterruptingTaxonomy()
            with (
                use_run_advancement_composition(composition),
                self.assertRaises(KeyboardInterrupt),
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            with use_run_advancement_composition(
                _OfflineAdvancementComposition(receipt, paths)
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))

            unclosed_root = Path(directory) / "earlier-writer"
            unclosed_root.mkdir()
            unclosed, unclosed_paths = _registered(unclosed_root)
            earlier = _OfflineAdvancementComposition(unclosed, unclosed_paths)
            earlier.reference_adapter = _InterruptingTaxonomy()
            with (
                use_run_advancement_composition(earlier),
                patch(
                    "bio_neb.agentic_journey.attempt_runner.record_model_attempt_cancelled",
                    side_effect=RuntimeError(
                        "model cancellation lacks its exact lifecycle source"
                    ),
                ),
                self.assertRaises(RunAdvancementRefused) as unhandled,
            ):
                advance(ResumeRun(journey_file=unclosed.journey_file))
            with self.assertRaisesRegex(
                ValueError, "observed Tool execution lost its next model request"
            ):
                observe(ObserveRun(journey_file=unclosed.journey_file))
        self.assertEqual(unhandled.exception.code, "coordinator_unhandled_error")

        (turn,) = [e for e in events if e["event_type"] == "ModelTurnCancelled"]
        succeeded = [e for e in events if e["event_type"] == "AttemptSucceeded"][-1]

        def miscounted(forgery):
            target = next(e for e in forgery if e["event_id"] == turn["event_id"])
            target["payload"]["model_request_count"] += 1

        def repeated(forgery):
            target = next(e for e in forgery if e["event_id"] == turn["event_id"])
            after = next(e for e in forgery if e["parent_ids"] == [target["event_id"]])
            _insert_copy(forgery, target, before=after)

        def outside_a_cancellation(forgery):
            target = next(e for e in forgery if e["event_id"] == succeeded["event_id"])
            _insert_copy(
                forgery,
                turn,
                before=target,
                attempt_id=target["payload"]["attempt_id"],
            )

        forgeries = (
            (miscounted, "Attempt cancellation lacks its exact model source"),
            (repeated, "Attempt cancellation lacks its exact model source"),
            (
                outside_a_cancellation,
                "a turn cancellation belongs only to a cancelled Attempt",
            ),
        )
        for forge, reason in forgeries:
            with self.subTest(forgery=forge.__name__):
                forgery = deepcopy(events)
                forge(forgery)
                with self.assertRaisesRegex(ValueError, reason):
                    project_journey_events(forgery, journey_file=receipt.journey_file)

    def test_ctrl_c_ends_the_cli_with_one_line_and_exit_130(self):
        # An interrupt prints one refusal line and exits 130 with no traceback.
        # During the command the Run is held; at the Review prompt nothing is
        # appended.
        class InterruptedPrompt(io.StringIO):
            def readline(self, *args):
                raise KeyboardInterrupt

        refusal = {"code": "interrupted", "fields": ["resume"], "status": "failed"}
        with tempfile.TemporaryDirectory() as directory, operator_sigint():
            receipt, paths = _registered(Path(directory).resolve())
            command = ["resume", "--journey-file", receipt.journey_file]
            composition = _OfflineAdvancementComposition(receipt, paths)
            composition.reference_adapter = _InterruptingTaxonomy()
            during = io.StringIO()
            with (
                use_run_advancement_composition(composition),
                redirect_stdout(io.StringIO()),
                redirect_stderr(during),
            ):
                during_exit = main(command)
            held = observe(ObserveRun(journey_file=receipt.journey_file)).status
            with use_run_advancement_composition(
                _OfflineAdvancementComposition(receipt, paths)
            ):
                pending = advance(ResumeRun(journey_file=receipt.journey_file))
            before = Path(receipt.journey_file).read_bytes()
            at_prompt, shown = io.StringIO(), io.StringIO()
            with (
                use_run_advancement_composition(
                    _OfflineAdvancementComposition(receipt, paths)
                ),
                patch("sys.stdin", InterruptedPrompt()),
                redirect_stdout(shown),
                redirect_stderr(at_prompt),
            ):
                prompt_exit = main([*command, "--interactive"])
            after = Path(receipt.journey_file).read_bytes()

        self.assertEqual(
            (
                during_exit,
                [json.loads(line) for line in during.getvalue().splitlines()],
            ),
            (130, [refusal]),
        )
        self.assertEqual(held, "run_held")
        self.assertEqual(pending.status, "planning_hitl_pending")
        prompt, line = at_prompt.getvalue().splitlines()
        self.assertEqual(
            (prompt_exit, prompt, json.loads(line)),
            (130, "Enter one exact legal Review response JSON: ", refusal),
        )
        self.assertEqual(json.loads(shown.getvalue())["frontier"], "review_required")
        self.assertEqual(after, before)

    def test_a_begin_that_holds_retains_the_registered_run_locator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _receipt, paths = _registered(root)
            intent = BeginRun(
                run_root=str((root / "begun-refusal").resolve()),
                prompt_file=str((root / "prompt.txt").resolve()),
                runtime_config_file=str((root / "runtime.json").resolve()),
                inputs=tuple(
                    RegisterFastqInput(
                        role=role,
                        registered_path=str(path.resolve()),
                    )
                    for role, path in paths.items()
                ),
            )

            # Begin with the product's model refuses before any Journey unless the
            # selected credential variable is set; no model is reached.
            with (
                patch.dict("os.environ", {"LOCAL_MODEL_KEY": "offline-test-value"}),
                patch(
                    "bio_neb.input_inspection.commands.inspect_registered_inputs",
                    lambda command, journey: None,
                ),
                self.assertRaises(RunAdvancementRefused) as raised,
            ):
                advance(intent)

            error = raised.exception
            self.assertEqual(error.code, "coordinator_dispatch_wrote_nothing")
            self.assertIsNotNone(error.journey_id)
            self.assertIsNotNone(error.journey_file)
            retained = observe(ObserveRun(journey_file=error.journey_file))
            self.assertEqual(error.journey_id, retained.journey_id)
            self.assertEqual(error.journey_file, retained.journey_file)
            self.assertEqual((error.status, retained.status), ("run_held", "run_held"))

    def test_answer_model_availability_retry_reenters_same_role_then_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            journey = Journey.open(Path(receipt.journey_file))

            async def unavailable(messages, info):
                del messages, info
                raise ModelHTTPError(429, "offline-availability", body={}, headers={})

            # No real waits: the window's delays are its cap times a zero jitter.
            with patch(
                "bio_neb.agentic_journey.model_observation.random.random",
                return_value=0,
            ):
                pending_review = planning_assignment(
                    model_factory=ScriptedModelEndpoint(unavailable)
                ).run(
                    prepare_input_correction(journey),
                    environ={"LOCAL_MODEL_KEY": "offline-only"},
                )
            pending = pending_review.model_availability_review.pending_request
            assert pending is not None
            answer = AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=pending.request_event_id,
                answer=ModelAvailabilityReviewAnswer(
                    kind="model_availability",
                    response=RetryModelAvailabilityResponse(action="retry"),
                ),
            )

            with use_run_advancement_composition(
                _OfflineAdvancementComposition(receipt, paths)
            ):
                continued = advance(answer)
                retained_count = len(journey.read())
                replayed = advance(
                    answer_review_from_json(
                        continued,
                        pending.request_event_id,
                        answer.answer.response.model_dump_json(),
                    )
                )

            self.assertEqual(continued.status, "planning_hitl_pending")
            self.assertEqual(
                continued.model_availability_review.receipts[-1].request_event_id,
                pending.request_event_id,
            )
            self.assertEqual(
                tuple(item.agent_role for item in continued.attempts[:2]),
                ("planning_agent", "planning_agent"),
            )
            self.assertEqual(replayed, continued)
            self.assertEqual(len(journey.read()), retained_count)

    def test_planning_retry_beside_target_question_resends_exact_pending_request(
        self,
    ):
        class Composition(_OfflineAdvancementComposition):
            unavailable = True

            async def respond(self, messages, info):
                if self.unavailable and "record_proposal_requirements" in self.calls:
                    raise ModelHTTPError(
                        503,
                        "offline-unavailable",
                        body={},
                        headers={"retry-after": "999"},
                    )
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory).resolve())
            composition = Composition(receipt, paths, open_target=True)
            path = Path(receipt.journey_file)
            with use_run_advancement_composition(composition):
                current = advance(ResumeRun(journey_file=receipt.journey_file))
                self.assertEqual(
                    {r.kind for r in current.pending_reviews},
                    {"planning", "model_availability"},
                )
                self.assertEqual(current.current_review.kind, "model_availability")
                output = current.planning_hitl.pending_request
                self.assertEqual(output.owner, "requirement_resolution")
                pending = current.model_availability_review.pending_request
                answer = answer_review_from_json(
                    current, pending.request_event_id, json.dumps({"action": "retry"})
                )
                composition.unavailable = False
                continued = advance(answer)

                events = Journey.open(path).read()
                requests = [
                    e for e in events if e["event_type"] == "ModelRequestRecorded"
                ]
                resent = next(
                    e
                    for e in requests
                    if e["payload"]["continuation_request_event_id"] is not None
                )
                original = next(
                    e
                    for e in requests
                    if e["event_id"]
                    == resent["payload"]["continuation_request_event_id"]
                )
                ids = [e["event_id"] for e in events]
                # The target question opened after the request the retry re-sends.
                self.assertLess(
                    ids.index(original["event_id"]), ids.index(output.request_event_id)
                )
                self.assertEqual(
                    resent["payload"]["compatible_adapter_evidence"]["request"],
                    original["payload"]["compatible_adapter_evidence"]["request"],
                )
                self.assertEqual(
                    continued.model_availability_review.receipts[-1].request_event_id,
                    pending.request_event_id,
                )
                self.assertEqual(
                    [a.agent_role for a in continued.attempts].count("planning_agent"),
                    2,
                )
                self.assertIn(
                    output.request_event_id,
                    [r.request_event_id for r in continued.pending_reviews],
                )
                frozen = _file_snapshot(path.parent)
                self.assertEqual(advance(answer), continued)
                self.assertEqual(
                    observe(ObserveRun(journey_file=receipt.journey_file)), continued
                )
                self.assertEqual(_file_snapshot(path.parent), frozen)

    def test_target_answer_before_provider_retry_preserves_current_tool_ancestry(
        self,
    ):
        class Composition(_OfflineAdvancementComposition):
            unavailable = True

            async def respond(self, messages, info):
                if self.unavailable and "record_proposal_requirements" in self.calls:
                    raise ModelHTTPError(
                        503,
                        "offline-unavailable",
                        body={},
                        headers={"retry-after": "999"},
                    )
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory).resolve())
            composition = Composition(receipt, paths, open_target=True)
            path = Path(receipt.journey_file)
            with use_run_advancement_composition(composition):
                current = advance(ResumeRun(journey_file=receipt.journey_file))
                self.assertEqual(
                    {r.kind for r in current.pending_reviews},
                    {"planning", "model_availability"},
                )
                self.assertEqual(current.current_review.kind, "model_availability")
                output = current.planning_hitl.pending_request
                self.assertEqual(output.owner, "requirement_resolution")
                pending = current.model_availability_review.pending_request
                current = advance(
                    answer_review_from_json(
                        current,
                        output.request_event_id,
                        json.dumps({"action": "clarify", "instruction": "rs1234"}),
                    )
                )
                output_answer_id = current.planning_hitl.receipts[-1].response_event_id
                answer = answer_review_from_json(
                    current, pending.request_event_id, json.dumps({"action": "retry"})
                )
                composition.unavailable = False
                continued = advance(answer)

                events = Journey.open(path).read()
                requests = [
                    e for e in events if e["event_type"] == "ModelRequestRecorded"
                ]
                resent = next(
                    e
                    for e in requests
                    if e["payload"]["continuation_request_event_id"] is not None
                )
                original = next(
                    e
                    for e in requests
                    if e["event_id"]
                    == resent["payload"]["continuation_request_event_id"]
                )
                ids = [e["event_id"] for e in events]
                # The target question opened after the request the retry re-sends.
                self.assertLess(
                    ids.index(original["event_id"]), ids.index(output.request_event_id)
                )
                self.assertEqual(
                    resent["payload"]["compatible_adapter_evidence"]["request"],
                    original["payload"]["compatible_adapter_evidence"]["request"],
                )
                self.assertEqual(
                    continued.model_availability_review.receipts[-1].request_event_id,
                    pending.request_event_id,
                )
                # The retry finishes its own work. The target answer is read by a fresh
                # Attempt once the planning model's own question is answered (Spec r3
                # SP5, owner "a"): here that question is open.
                self.assertEqual(
                    [a.agent_role for a in continued.attempts].count("planning_agent"),
                    2,
                )
                self.assertTrue(
                    any(
                        r.owner == "agent_attempt"
                        for r in continued.planning_hitl.pending_requests
                    )
                )
                self.assertEqual(
                    [
                        u.reason
                        for u in continued.requirement_resolution.unsettled_targets
                    ],
                    ["answer_unread"],
                )
                self.assertNotIn(
                    output.request_event_id,
                    [r.request_event_id for r in continued.pending_reviews],
                )
                ancestors = {}
                for event in events:
                    ancestors[event["event_id"]] = set(event["parent_ids"])
                    for parent in event["parent_ids"]:
                        ancestors[event["event_id"]].update(ancestors[parent])
                later_facts = [
                    e
                    for e in events
                    if e["event_type"]
                    in {
                        "ReferenceDescriptionRecorded",
                        "RecipeObservationRecorded",
                    }
                    and ids.index(e["event_id"]) > ids.index(resent["event_id"])
                ]
                self.assertTrue(later_facts)
                for fact in later_facts:
                    self.assertIn(output_answer_id, ancestors[fact["event_id"]])
                # Removing both bindings must not disguise this retry as fresh work.
                # Exercise the shared availability reader over actual caller evidence.
                from bio_neb.agentic_journey.model_availability import (
                    project_model_availability_review,
                )

                hostile = deepcopy(read_evidence(Journey.open(path)))
                resumed_start = next(
                    e
                    for e in hostile
                    if e["event_type"] == "AttemptStarted"
                    and e["payload"].get("continuation_availability_review_event_id")
                    is not None
                )
                resumed_start["payload"][
                    "continuation_availability_review_event_id"
                ] = None
                resumed_start["payload"]["continuation_request_event_id"] = None
                with self.assertRaisesRegex(
                    ValueError, "lost its exact direction binding"
                ):
                    project_model_availability_review(hostile)
                frozen = _file_snapshot(path.parent)
                self.assertEqual(advance(answer), continued)
                self.assertEqual(
                    observe(ObserveRun(journey_file=receipt.journey_file)), continued
                )
                self.assertEqual(_file_snapshot(path.parent), frozen)

    def test_target_answer_after_effort_extension_preserves_answers_across_passes(self):
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
        from bio_neb.agentic_journey.effort import (
            ExtendAgentEffortResponse,
            respond_effort_review,
        )
        from bio_neb.input_inspection.commands import (
            InspectRegisteredInputsCommand,
            inspect_registered_inputs,
        )

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory).resolve())
            composition = _OfflineAdvancementComposition(
                receipt, paths, open_target=True
            )
            journey = Journey.open(Path(receipt.journey_file))
            inspect_registered_inputs(
                InspectRegisteredInputsCommand(journey_file=receipt.journey_file),
                journey=journey,
            )
            stopped = composition.planning_assignment(journey).run(
                journey,
                environ=composition.environ,
                invocation_overrides=AgentInvocationOverrides(model_requests=2),
            )
            self.assertEqual(stopped.status, "effort_review_pending")
            with use_run_advancement_composition(composition):
                waiting = advance(ResumeRun(journey_file=receipt.journey_file))
                output = next(
                    r
                    for r in waiting.planning_hitl.pending_requests
                    if r.owner == "requirement_resolution"
                )
                request = waiting.agent_effort_review.pending_request
                extension = respond_effort_review(
                    Journey.open(Path(receipt.journey_file)),
                    request.request_event_id,
                    ExtendAgentEffortResponse(action="extend", additional_requests=20),
                )
                continued = advance(
                    answer_review_from_json(
                        waiting,
                        output.request_event_id,
                        json.dumps({"action": "clarify", "instruction": "rs1234"}),
                    )
                )
                self.assertIsNone(continued.agent_effort_review.pending_request)
                events = read_evidence(Journey.open(Path(receipt.journey_file)))
                answer = next(
                    e
                    for e in events
                    if e["event_type"] == "PlanningHitlResponded"
                    and e["payload"]["request_event_id"] == output.request_event_id
                )
                resumed = next(
                    e
                    for e in events
                    if e["event_type"] == "AttemptStarted"
                    and e["payload"].get("continuation_effort_review_event_id")
                    == extension.response_event_id
                )
                by_id = {e["event_id"]: e for e in events}
                lifecycle = by_id[resumed["parent_ids"][0]]
                self.assertEqual(lifecycle["parent_ids"], [answer["event_id"]])
                self.assertIn(extension.response_event_id, answer["parent_ids"])
                self.assertTrue(
                    any(
                        e["event_type"] == "PassClosed"
                        for e in events[
                            events.index(
                                next(
                                    e
                                    for e in events
                                    if e["event_type"] == "AgentEffortReviewRequested"
                                )
                            ) : events.index(resumed)
                        ]
                    )
                )
                self.assertEqual(composition.calls.count("search_journey"), 1)
                self.assertEqual(
                    composition.calls.count("record_proposal_requirements"), 1
                )
                self.assertEqual(
                    observe(ObserveRun(journey_file=receipt.journey_file)), continued
                )
                frozen = _file_snapshot(Path(receipt.journey_file).parent)
                with self.assertRaisesRegex(ValueError, "unused"):
                    composition.planning_assignment(
                        Journey.open(Path(receipt.journey_file))
                    ).run(
                        Journey.open(Path(receipt.journey_file)),
                        environ=composition.environ,
                        effort_review_response_event_id=extension.response_event_id,
                    )
                self.assertEqual(
                    _file_snapshot(Path(receipt.journey_file).parent), frozen
                )

    def test_an_open_target_reaches_fresh_execution_approval_authority(self):
        # The researcher names the targets before the Execution proposal; the
        # approval runs Execution once and Investigation begins from them.
        from tests.planning_assignment_fakes import configured_planning_run
        from tests.sequential_investigation_support import (
            ASSEMBLY_BODY,
            SequentialSourceResponses,
            _AdaptiveInvestigationModel,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            # The registration writes the files BeginRun registers again, with the
            # current Execution configuration and its capability.
            receipt, configured, configuration = configured_planning_run(root)
            paths = configured.paths
            constructed = []
            intent = BeginRun(
                run_root=str((root / "advanced-runs").resolve()),
                prompt_file=str((root / "prompt.txt").resolve()),
                runtime_config_file=str((root / "runtime.json").resolve()),
                inputs=tuple(
                    RegisterFastqInput(role=role, registered_path=str(path.resolve()))
                    for role, path in paths.items()
                ),
            )
            composition = _OfflineAdvancementComposition(
                receipt, paths, open_target=True
            )
            composition.capability = configuration.runtime_selection.capability
            target_request_id = None
            with use_run_advancement_composition(composition):
                current = advance(intent)
                for _ in range(10):
                    if current.status == "execution_hitl_pending":
                        break
                    pending = current.planning_hitl.pending_request
                    assert pending is not None
                    if pending.owner == "requirement_resolution":
                        # The target question: no Execution proposal exists yet.
                        self.assertIsNone(current.execution_hitl)
                        target_request_id = pending.request_event_id
                        response = ClarifyPlanningHitlResponse(
                            action="clarify", instruction="Check rs429358."
                        )
                    else:
                        response = SelectPlanningHitlResponse(
                            action="select",
                            candidate_id=pending.candidates[0].candidate_id,
                        )
                    current = advance(
                        AnswerReview(
                            journey_file=current.journey_file,
                            request_event_id=pending.request_event_id,
                            answer=PlanningReviewAnswer(
                                kind="planning", response=response
                            ),
                        )
                    )
                self.assertEqual(current.status, "execution_hitl_pending")
                self.assertIsNotNone(target_request_id)
                execution_request = current.execution_hitl.pending_request
                original_proposal_id = execution_request.proposal.proposal_id
                current = advance(
                    AnswerReview(
                        journey_file=current.journey_file,
                        request_event_id=execution_request.request_event_id,
                        answer=ExecutionReviewAnswer(
                            kind="execution",
                            response=ReviseExecutionResponse(
                                action="revise",
                                proposal_id=original_proposal_id,
                                instruction="Revalidate the complete execution plan.",
                                revalidation_scope="execution_planning",
                            ),
                        ),
                    )
                )
                revision_review = current.planning_hitl.pending_request
                assert revision_review is not None
                self.assertEqual(
                    (
                        current.status,
                        revision_review.owner,
                        revision_review.attempt_id,
                        current.execution_hitl.status,
                    ),
                    (
                        "planning_hitl_pending",
                        "agent_attempt",
                        current.attempt.attempt_id,
                        "execution_revision_pending",
                    ),
                )
                for _ in range(10):
                    if current.status == "execution_hitl_pending":
                        break
                    pending = current.planning_hitl.pending_request
                    assert pending is not None and pending.candidates
                    current = advance(
                        AnswerReview(
                            journey_file=current.journey_file,
                            request_event_id=pending.request_event_id,
                            answer=PlanningReviewAnswer(
                                kind="planning",
                                response=SelectPlanningHitlResponse(
                                    action="select",
                                    candidate_id=pending.candidates[0].candidate_id,
                                ),
                            ),
                        )
                    )
                self.assertEqual(current.status, "execution_hitl_pending")
                execution_request = current.execution_hitl.pending_request
                approval = AnswerReview(
                    journey_file=current.journey_file,
                    request_event_id=execution_request.request_event_id,
                    answer=ExecutionReviewAnswer(
                        kind="execution",
                        response=approve_execution(execution_request.proposal),
                    ),
                )
                approved_review = current
                before_illegal = len(Journey.open(Path(current.journey_file)).read())
                with self.assertRaises(ExecutionHitlResponseRefused) as illegal:
                    advance(
                        AnswerReview(
                            journey_file=current.journey_file,
                            request_event_id=execution_request.request_event_id,
                            answer=ExecutionReviewAnswer(
                                kind="execution",
                                response=ApproveExecutionResponse(
                                    action="approve",
                                    proposal_id="execution-proposal:" + "0" * 64,
                                ),
                            ),
                        )
                    )
                self.assertEqual(
                    illegal.exception.code, "execution_hitl_response_illegal"
                )
                self.assertEqual(
                    illegal.exception.disposition.kind,
                    "caller_correction_and_reissue",
                )
                self.assertEqual(
                    len(Journey.open(Path(current.journey_file)).read()),
                    before_illegal,
                )
                from bio_neb.application.advancement import (
                    use_run_advancement_dependencies,
                )
                from tests.planning_assignment_fakes import configured_provider

                # The approval's call continues into Execution with offline effects,
                # then into Investigation from the targets the researcher named.
                with use_run_advancement_dependencies(
                    environ={"LOCAL_MODEL_KEY": "not-retained"},
                    investigation_model_factory=_AdaptiveInvestigationModel().model,
                    investigation_source_adapter_factory=lambda: (
                        SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)
                    ),
                    parabricks_infrastructure_factory=configured_provider(
                        current.journey_file, constructed
                    ),
                ):
                    authority = advance(approval)
                retained_count = len(Journey.open(Path(authority.journey_file)).read())
                replayed_authority = advance(approval)
                with self.assertRaises(ExecutionHitlResponseRefused) as conflict:
                    advance(
                        AnswerReview(
                            journey_file=current.journey_file,
                            request_event_id=execution_request.request_event_id,
                            answer=ExecutionReviewAnswer(
                                kind="execution",
                                response=RejectExecutionResponse(
                                    action="reject",
                                    proposal_id=execution_request.proposal.proposal_id,
                                    reason="Conflicting replay.",
                                ),
                            ),
                        )
                    )

            from bio_neb.parabricks.execution_proposal import project_execution_hitl

            # The targets come from the approved review's handoff, the authority
            # from the retained facts.
            (target,) = (
                approved_review.requirement_resolution.scientific_investigation.items
            )
            target_receipt = next(
                item
                for item in approved_review.planning_hitl.receipts
                if item.request_event_id == target_request_id
            )
            assertion = next(
                e
                for e in Journey.open(Path(authority.journey_file)).read()
                if e["event_type"] == "InvestigationStarted"
            )["payload"]["assertion"]
            reviewed = SimpleNamespace(
                execution_hitl=project_execution_hitl(
                    read_evidence(Journey.open(Path(authority.journey_file)))
                )
            )
            retained = Journey.open(Path(authority.journey_file)).read()
            self.assertEqual(
                {
                    "targets": [
                        (item.words, item.rsid, item.source_event_id)
                        for item in target.targets
                    ],
                    # One target is a collection of one.
                    "investigated": [
                        (member["request_text"], member["rsid"])
                        for member in assertion["requests"]
                    ],
                    "execution_status": authority.status,
                    "fresh_proposal": execution_request.proposal.proposal_id
                    != original_proposal_id,
                    "supersedes": execution_request.proposal.supersedes_proposal_id,
                    "authority_count": len(reviewed.execution_hitl.authorities),
                    "authority_replay": replayed_authority == authority,
                    "authority_replay_appended": len(retained) - retained_count,
                    "authority_conflict": conflict.exception.code,
                    "proposal_id": reviewed.execution_hitl.authorities[0].proposal_id,
                    "reference_source_admission_count": sum(
                        event["event_type"] == "ReferenceSourceAdmitted"
                        for event in retained
                    ),
                    # One provider construction and one binding: approval's call ran
                    # Execution once, and its replay repeated no effect.
                    "execution_effects": (
                        len(constructed),
                        any(
                            event["event_type"] == "ExecutionPathRecorded"
                            for event in retained
                        ),
                    ),
                },
                {
                    "targets": [
                        ("rs429358", "rs429358", target_receipt.response_event_id)
                    ],
                    "investigated": [("rs429358", "rs429358")],
                    "execution_status": "investigation_review_pending",
                    "fresh_proposal": True,
                    "supersedes": original_proposal_id,
                    "authority_count": 1,
                    "authority_replay": True,
                    "authority_replay_appended": 0,
                    "authority_conflict": "execution_hitl_response_conflict",
                    "proposal_id": execution_request.proposal.proposal_id,
                    "reference_source_admission_count": 1,
                    "execution_effects": (1, True),
                },
            )

    def test_a_target_with_no_source_returns_to_the_agent_and_never_ends_the_run(
        self,
    ):
        # A recorded dead end, on the target: admission refuses a route no source
        # can satisfy with guidance naming ask_planning_hitl; the Agent repairs it
        # within its allowance and the coordinator's target question follows.
        class Repairing(_OfflineAdvancementComposition):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.feedback = []

            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                if self.calls[-1] != "record_proposal_requirements":
                    return response
                retries = [
                    text for message in messages for text in corrections(message)
                ]
                target = next(
                    item
                    for item in response.parts[0].args["requirements"]
                    if item["kind"] == "scientific_investigation"
                )
                if retries and "ask_planning_hitl" in retries[-1]:
                    self.feedback.append(retries[-1])
                else:
                    # The request's own words cannot supply a target it lacks.
                    target["acquisition_route"] = "use_supplied_assertion"
                return response

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            composition = Repairing(receipt, paths, open_target=True)
            with use_run_advancement_composition(composition):
                current = advance(ResumeRun(journey_file=receipt.journey_file))

            self.assertEqual(composition.calls.count("record_proposal_requirements"), 2)
            (feedback,) = composition.feedback
            self.assertIn(
                "no Tool, fact or Review can ever supply scientific_investigation",
                feedback,
            )
            self.assertIn("acquisition_route='ask_planning_hitl'", feedback)
            self.assertEqual(current.status, "planning_hitl_pending")
            self.assertIn(
                "requirement_resolution",
                {request.owner for request in current.planning_hitl.pending_requests},
            )

    def test_a_requirement_no_tool_can_satisfy_never_reruns_planning_unbounded(self):
        # A non-output requirement stuck in needs_tool leaves planning
        # incomplete, so the Attempt does not close as ready; the Agent's
        # unchanged repeated search stops it for the researcher's review (before
        # its request allowance). Nothing reruns it.
        from bio_neb.parabricks.planning_readiness import PlanningIncomplete
        from tests.planning_assignment_fakes import configured_planning_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, base, _ = configured_planning_run(
                root,
                agent_invocation={
                    "scopes": {"role": {"planning_agent": {"model_requests": 16}}}
                },
            )
            composition = _SearchingComposition(receipt, base.paths)
            runs = []
            assign = composition.planning_assignment

            def bounded(journey):
                runs.append(journey.head_event_id)
                if len(runs) > 12:
                    raise AssertionError("planning reran without bound")
                return assign(journey)

            composition.planning_assignment = bounded
            with (
                _recorded_as_formerly_admitted(_unfindable_input),
                use_run_advancement_composition(composition),
            ):
                current = advance(ResumeRun(journey_file=receipt.journey_file))
                stuck = _answer_selectable_planning_reviews(
                    receipt.journey_file, current
                )

            (unfindable,) = (
                item
                for item in stuck.requirement_resolution.requirements
                if item.subject_key == "additional_input"
            )
            self.assertEqual(unfindable.state, "needs_tool")
            # Planning readiness names the unresolved requirement ...
            self.assertEqual(
                _planning_readiness(Journey.open(Path(receipt.journey_file))),
                PlanningIncomplete(
                    "proposal_required_evidence_unresolved",
                    (unfindable.requirement_id,),
                ),
            )
            # ... so no Attempt closed as ready; the last one repeated its search
            # until its failed turns in a row stopped it, and its Review asks the
            # researcher, nothing held or rerun.
            self.assertFalse(
                any(
                    event.payload["checkpoint"] == "execution_planning_ready"
                    for attempt in stuck.attempts
                    for event in attempt.checkpoint_events
                )
            )
            self.assertEqual(
                (
                    stuck.status,
                    stuck.current_review.kind,
                    stuck.current_review.request.cause,
                ),
                ("agent_work_review_pending", "agent_work", "failed_turns"),
            )
            self.assertEqual(
                stuck.current_review.request.attempt_id, stuck.attempts[-1].attempt_id
            )
            self.assertEqual(len(runs), len(stuck.attempts))

    def test_a_preparation_question_waits_for_unresolved_planning_facts(self):
        # The Agent asks for preparation information only when planning
        # readiness lacks exactly that; while a requirement is unresolved no
        # configuration answer can complete planning, so none is asked.
        from tests.planning_assignment_fakes import configured_planning_run

        with tempfile.TemporaryDirectory() as directory:
            receipt, base, _ = configured_planning_run(
                Path(directory).resolve(),
                supply_configuration=False,
                agent_invocation={
                    "scopes": {"role": {"planning_agent": {"model_requests": 16}}}
                },
            )
            composition = _SearchingComposition(receipt, base.paths)
            composition.capability = base.capability
            with (
                _recorded_as_formerly_admitted(_unfindable_input),
                use_run_advancement_composition(composition),
            ):
                current = advance(ResumeRun(journey_file=receipt.journey_file))
                stopped = _answer_selectable_planning_reviews(
                    receipt.journey_file, current
                )

            # The repeated search stops the Attempt first: each repeat is a
            # failed turn, and the last in a row stops it.
            self.assertEqual(
                (
                    stopped.status,
                    stopped.current_review.kind,
                    stopped.current_review.request.cause,
                ),
                ("agent_work_review_pending", "agent_work", "failed_turns"),
            )
            self.assertNotIn("request_preparation_information", composition.calls)
            journey = Journey.open(Path(receipt.journey_file))
            self.assertFalse(
                any(
                    event["event_type"] == "PreparationInformationRequested"
                    for event in journey.read()
                )
            )
            self.assertEqual(
                _planning_readiness(journey).code,
                "proposal_required_evidence_unresolved",
            )

    def test_a_planning_closure_the_compiler_contradicts_holds_the_run(self):
        # C4, proved by forcing the disagreement planning readiness rules out:
        # closure says complete, the compiler says incomplete. A rerun would
        # close again with no new fact, so the Run is held instead.
        from bio_neb.parabricks.planning_readiness import PlanningIncomplete
        from tests.planning_assignment_fakes import (
            configured_planning_run,
            execution_configuration,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, base, _ = configured_planning_run(root)
            composition = _OfflineAdvancementComposition(receipt, base.paths)
            composition.capability = execution_configuration(
                root
            ).runtime_selection.capability
            runs = []
            assign = composition.planning_assignment

            def counted(journey):
                runs.append(journey.head_event_id)
                return assign(journey)

            composition.planning_assignment = counted
            contradicting = PlanningIncomplete(
                "proposal_required_evidence_unresolved", ("forced",)
            )
            with (
                patch(
                    "bio_neb.parabricks.preparation.prepare_execution",
                    lambda journey: contradicting,
                ),
                use_run_advancement_composition(composition),
                self.assertRaises(RunAdvancementRefused) as raised,
            ):
                current = advance(ResumeRun(journey_file=receipt.journey_file))
                _answer_selectable_planning_reviews(receipt.journey_file, current)
            error = self._assert_held(
                raised,
                "planning_closed_preparation_incomplete",
                "execution_planning_ready",
                "proposal_required_evidence_unresolved",
            )
            self.assertEqual(error.failure.evidence["details"], ["forced"])
            # No planning Attempt started after the one that made planning ready.
            held = observe(ObserveRun(journey_file=receipt.journey_file))
            self.assertEqual(
                held.attempts[-1].checkpoint_events[-1].payload["checkpoint"],
                "execution_planning_ready",
            )
            self.assertEqual(len(runs), len(held.attempts))

    def test_submitting_before_planning_is_complete_stops_at_planning(self):
        # A planning gap is not a compilation fault. Submission before planning
        # is complete writes nothing and returns the Run at its planning stop; no
        # Invariant escapes to hold the Run.
        from bio_neb.parabricks.planning_commands import (
            SubmitExecutionProposalCommand,
            submit_execution_proposal,
        )
        from bio_neb.parabricks.planning_readiness import PlanningIncomplete
        from bio_neb.parabricks.preparation import prepare_execution
        from tests.planning_assignment_fakes import configured_planning_run

        with tempfile.TemporaryDirectory() as directory:
            receipt, _, _ = configured_planning_run(Path(directory).resolve())
            journey = Journey.open(Path(receipt.journey_file))
            before = journey.read()
            self.assertIsInstance(prepare_execution(journey), PlanningIncomplete)

            review = submit_execution_proposal(
                SubmitExecutionProposalCommand(journey_file=receipt.journey_file)
            )

            self.assertEqual(Journey.open(Path(receipt.journey_file)).read(), before)
            self.assertEqual(review.status, "inputs_registered")
            self.assertIsNone(review.execution_hitl)
            self.assertEqual(
                observe(ObserveRun(journey_file=receipt.journey_file)).status,
                "inputs_registered",
            )

    def test_a_second_recipe_or_resource_asks_the_preparation_review(self):
        # Q4: admission allows a second recipe or resource requirement, and
        # planning selects one for each. The H100 runtime runs one recipe on one
        # resource, so planning readiness reports the selection incompatible and
        # the Agent asks the researcher through the existing preparation Review.
        from bio_neb.parabricks.planning_readiness import PlanningIncomplete
        from tests.planning_assignment_fakes import configured_planning_run

        for kind in ("analysis_recipe", "resource_profile"):
            with (
                self.subTest(kind=kind),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, base, _ = configured_planning_run(Path(directory).resolve())
                composition = _SecondRequirementComposition(
                    receipt, base.paths, second_kind=kind
                )
                composition.capability = base.capability
                with use_run_advancement_composition(composition):
                    current = advance(ResumeRun(journey_file=receipt.journey_file))
                    current = _answer_selectable_planning_reviews(
                        receipt.journey_file, current
                    )

                execution = current.execution_planning
                self.assertEqual(
                    (
                        execution.status,
                        len(execution.selected_recipe_bindings)
                        + len(execution.selected_resource_bindings),
                    ),
                    ("execution_planning_ready", 3),
                )
                self.assertEqual(
                    _planning_readiness(Journey.open(Path(receipt.journey_file))),
                    PlanningIncomplete(
                        "execution_configuration_incompatible", ("runtime_capability",)
                    ),
                )
                request = current.current_review.request
                self.assertEqual(
                    (
                        current.current_review.kind,
                        request.code,
                        request.missing_fields,
                    ),
                    (
                        "preparation",
                        "execution_configuration_incompatible",
                        ("runtime_capability",),
                    ),
                )

    def test_planning_closure_and_the_compiler_never_disagree(self):
        # D2: wherever the coordinator compiles, and at the stop each Run reaches,
        # a planning Attempt would close exactly when the compiler finds nothing
        # missing that the Agent can acquire on the same Journey: nothing, or only
        # the preparation information the coordinator asks for.
        from bio_neb.agentic_journey.planning_assignment import (
            planning_closes_before_contact,
        )
        from bio_neb.parabricks import preparation
        from tests.planning_assignment_fakes import configured_planning_run

        allowance = {"scopes": {"role": {"planning_agent": {"model_requests": 16}}}}
        recorded = lambda requirements: requirements  # noqa: E731
        scenarios = {
            # A recorded shape: scientific questions beside the Execution facts;
            # the proposal stands at its Review, so planning reads complete.
            "configured": (
                {"scientific_intent": "Is rs429358 present?"},
                recorded,
                _OfflineAdvancementComposition,
                {},
                (True, "ready", None),
            ),
            "unconfigured": (
                {"supply_configuration": False},
                recorded,
                _OfflineAdvancementComposition,
                {},
                (True, "missing_information", "execution_configuration_required"),
            ),
            "required_evidence_unresolved": (
                {"agent_invocation": allowance},
                _unfindable_input,
                _SearchingComposition,
                {},
                (False, "missing_information", "proposal_required_evidence_unresolved"),
            ),
            "two_recipes": (
                {},
                recorded,
                _SecondRequirementComposition,
                {"second_kind": "analysis_recipe"},
                (True, "missing_information", "execution_configuration_incompatible"),
            ),
        }
        researcher_supplied = {
            "execution_configuration_required",
            "execution_configuration_incompatible",
        }
        compile_ = preparation.prepare_execution
        for name, (setup, rewrite, model, options, at_stop) in scenarios.items():
            with (
                self.subTest(scenario=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, base, _ = configured_planning_run(
                    Path(directory).resolve(), **setup
                )
                composition = model(receipt, base.paths, **options)
                composition.capability = base.capability
                composition.scientific_intent = base.scientific_intent
                verdicts = []

                def judged(journey, verdicts=verdicts):
                    closed = planning_closes_before_contact(journey)
                    prepared = compile_(journey)
                    verdicts.append(
                        (closed, prepared.status, getattr(prepared, "code", None))
                    )
                    return prepared

                with (
                    _recorded_as_formerly_admitted(rewrite),
                    patch.object(preparation, "prepare_execution", judged),
                    use_run_advancement_composition(composition),
                ):
                    current = advance(ResumeRun(journey_file=receipt.journey_file))
                    _answer_selectable_planning_reviews(receipt.journey_file, current)
                # And at the stop it reached (a pending Review compiles nothing new).
                judged(Journey.open(Path(receipt.journey_file)))

                self.assertEqual(verdicts[-1], at_stop)
                self.assertEqual(
                    [
                        v
                        for v in verdicts
                        if v[0] != (v[1] == "ready" or v[2] in researcher_supplied)
                    ],
                    [],
                    verdicts,
                )

    def _until_target_question(self, current):
        """Answer the Agent's own planning questions until the target question."""
        for _ in range(8):
            request = current.planning_hitl.pending_request
            assert request is not None
            if request.owner == "requirement_resolution":
                return current, request
            self.assertTrue(request.candidates)
            current = advance(
                AnswerReview(
                    journey_file=current.journey_file,
                    request_event_id=request.request_event_id,
                    answer=PlanningReviewAnswer(
                        kind="planning",
                        response=SelectPlanningHitlResponse(
                            action="select",
                            candidate_id=request.candidates[0].candidate_id,
                        ),
                    ),
                )
            )
        self.fail("the target question did not open")

    def test_a_target_found_after_execution_is_still_asked_before_it(self):
        # A target recorded for a Tool to find after Execution names no target
        # in the researcher's words, so the coordinator asks before the
        # Execution proposal; no GPU job runs before the targets exist.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            intent = BeginRun(
                run_root=str((root / "advanced-runs").resolve()),
                prompt_file=str((root / "prompt.txt").resolve()),
                runtime_config_file=str((root / "runtime.json").resolve()),
                inputs=tuple(
                    RegisterFastqInput(role=role, registered_path=str(path.resolve()))
                    for role, path in paths.items()
                ),
            )
            composition = _OfflineAdvancementComposition(
                receipt, paths, tool_found_target=True
            )
            with use_run_advancement_composition(composition):
                current, request = self._until_target_question(advance(intent))
                plan = current.proposal_plan.plan
                recorded = next(
                    item
                    for item in plan.requirements
                    if item.kind == "scientific_investigation"
                )
                self.assertEqual(
                    (recorded.acquisition_route, recorded.validation.kind),
                    ("validate_with_tool", "tool_evidence"),
                )
                self.assertEqual(
                    (
                        current.status,
                        request.owner,
                        request.requirement_id,
                        request.candidates,
                        current.execution_hitl,
                    ),
                    (
                        "planning_hitl_pending",
                        "requirement_resolution",
                        recorded.requirement_id,
                        (),
                        None,
                    ),
                )
                answered = _answer_selectable_planning_reviews(
                    current.journey_file,
                    advance(
                        answer_review_from_json(
                            current,
                            request.request_event_id,
                            json.dumps(
                                {"action": "clarify", "instruction": "rs429358"}
                            ),
                        )
                    ),
                )

            (item,) = answered.requirement_resolution.scientific_investigation.items
            self.assertEqual(
                [(target.words, target.rsid) for target in item.targets],
                [("rs429358", "rs429358")],
            )
            self.assertEqual(answered.requirement_resolution.unsettled_targets, ())
            self.assertEqual(
                answered.execution_planning.status, "execution_planning_ready"
            )

    def test_a_to_find_quote_only_says_what_to_find_so_the_target_is_asked(self):
        # A to_find item's quote describes what to find; it is never the value,
        # even when it reads like an rsID. The coordinator still asks before the
        # Execution proposal, and no target reaches the handoff unasked.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            intent = BeginRun(
                run_root=str((root / "quoted-open-target-runs").resolve()),
                prompt_file=str((root / "prompt.txt").resolve()),
                runtime_config_file=str((root / "runtime.json").resolve()),
                inputs=tuple(
                    RegisterFastqInput(role=role, registered_path=str(path.resolve()))
                    for role, path in paths.items()
                ),
            )
            composition = _OfflineAdvancementComposition(
                receipt, paths, open_target=True, open_target_quote="rs1234"
            )

            with use_run_advancement_composition(composition):
                current, request = self._until_target_question(advance(intent))

        (item,) = current.requirement_resolution.scientific_investigation.items
        self.assertEqual(
            (
                item.targets,
                tuple(
                    (unsettled.requirement_id, unsettled.reason)
                    for unsettled in current.requirement_resolution.unsettled_targets
                ),
                current.execution_hitl,
            ),
            ((), ((request.requirement_id, "no_target"),), None),
        )

    def test_a_request_naming_no_target_opens_the_target_question(self):
        # The question offers no candidate: the researcher names the targets, or
        # leaves the item out (reject), or stops the Run.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            intent = BeginRun(
                run_root=str((root / "no-target-runs").resolve()),
                prompt_file=str((root / "prompt.txt").resolve()),
                runtime_config_file=str((root / "runtime.json").resolve()),
                inputs=tuple(
                    RegisterFastqInput(role=role, registered_path=str(path.resolve()))
                    for role, path in paths.items()
                ),
            )
            composition = _OfflineAdvancementComposition(
                receipt, paths, open_target=True
            )

            with use_run_advancement_composition(composition):
                current, request = self._until_target_question(advance(intent))
                target = next(
                    item
                    for item in current.requirement_resolution.requirements
                    if item.requirement_id == request.requirement_id
                )
                self.assertEqual(
                    {
                        "status": current.status,
                        "owner": request.owner,
                        "kind": target.kind,
                        "unsettled": tuple(
                            (item.requirement_id, item.reason)
                            for item in current.requirement_resolution.unsettled_targets
                        ),
                        "candidates": request.candidates,
                        "legal_actions": tuple(
                            item.action for item in request.legal_actions
                        ),
                        "execution_proposal": current.execution_hitl,
                    },
                    {
                        "status": "planning_hitl_pending",
                        "owner": "requirement_resolution",
                        "kind": "scientific_investigation",
                        "unsettled": ((request.requirement_id, "no_target"),),
                        "candidates": (),
                        "legal_actions": ("clarify", "reject", "stop"),
                        "execution_proposal": None,
                    },
                )

                continued = advance(
                    AnswerReview(
                        journey_file=current.journey_file,
                        request_event_id=request.request_event_id,
                        answer=PlanningReviewAnswer(
                            kind="planning",
                            response=ClarifyPlanningHitlResponse(
                                action="clarify",
                                instruction="Check RS429358 and rs7412.",
                            ),
                        ),
                    )
                )

            (item,) = continued.requirement_resolution.scientific_investigation.items
            self.assertEqual(
                [(target.words, target.rsid) for target in item.targets],
                [("RS429358", "rs429358"), ("rs7412", "rs7412")],
            )
            self.assertEqual(continued.requirement_resolution.unsettled_targets, ())

    def test_a_request_file_s_targets_open_no_target_question_beside_it(self):
        """#482 reproduction of #480 R2-1: the request file lists the targets, so a
        second, target-less item beside it is refused at admission (owner, #482 Q2 a;
        ADR 0049: code checks citation, never words) and no "names no target" question
        opens; the file's item carries the targets to the Investigation."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            request_file = root / "snp-request.json"
            request_file.write_text(
                json.dumps(_RECORDED_REQUEST_FILE), encoding="utf-8"
            )
            intent = BeginRun(
                run_root=str((root / "request-file-runs").resolve()),
                prompt_file=str((root / "prompt.txt").resolve()),
                runtime_config_file=str((root / "runtime.json").resolve()),
                request_files=(str(request_file),),
                inputs=tuple(
                    RegisterFastqInput(role=role, registered_path=str(path.resolve()))
                    for role, path in paths.items()
                ),
            )
            composition = _RecordedRequestFileComposition(receipt, paths)
            with use_run_advancement_composition(composition):
                current = advance(intent)
                asked = []
                for _ in range(8):
                    request = current.planning_hitl.pending_request
                    if request is None or request.owner == "requirement_resolution":
                        break
                    asked.append(request.subject_key)
                    current = advance(
                        AnswerReview(
                            journey_file=current.journey_file,
                            request_event_id=request.request_event_id,
                            answer=PlanningReviewAnswer(
                                kind="planning",
                                response=SelectPlanningHitlResponse(
                                    action="select",
                                    candidate_id=request.candidates[0].candidate_id,
                                ),
                            ),
                        )
                    )
            events = read_evidence(Journey.open(Path(current.journey_file)))

        (file_event_id,) = (
            e["event_id"]
            for e in events
            if e["event_type"] == "InvestigationRequestRegistered"
        )
        recorded = [
            (e["payload"]["status"], tuple(e["payload"]["errors"]))
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_name"] == "record_proposal_requirements"
        ]
        resolution = current.requirement_resolution
        self.maxDiff = None
        self.assertEqual(
            {
                "recorded": recorded,
                "items": [
                    (item.subject_key, item.registered_request_event_id)
                    for item in resolution.scientific_investigation.items
                ],
                "unsettled": resolution.unsettled_targets,
                "target_questions": [
                    request.issue
                    for e in events
                    if e["event_type"] == "PlanningHitlRequested"
                    and (
                        request := PlanningHitlRequested.model_validate(e["payload"])
                    ).owner
                    == "requirement_resolution"
                ],
                "asked": asked,
                "status": current.status,
            },
            {
                "recorded": [
                    (
                        "invalid",
                        (
                            "targets_from_request:"
                            "target_question_beside_registered_request",
                        ),
                    ),
                    ("valid", ()),
                ],
                "items": [("targets_from_request_file", file_event_id)],
                "unsettled": (),
                "target_questions": [],
                # Planning ends at its own recipe question, its last before Execution.
                "asked": ["analysis_recipe"],
                "status": "execution_planning_ready",
            },
        )

    def test_resume_opens_the_target_question_without_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            composition = _OfflineAdvancementComposition(
                receipt, paths, open_target=True
            )
            with use_run_advancement_composition(composition):
                before = advance(ResumeRun(journey_file=receipt.journey_file))
                before, _ = self._until_target_question(before)
            self.assertEqual(before.reference_facts.status, "reference_facts")
            target_requirement = next(
                item
                for item in before.requirement_resolution.requirements
                if item.kind == "scientific_investigation"
            )
            contact_ledger: list[str] = []

            def refuse_contact(kind: str):
                def refuse(*_args, **_kwargs):
                    contact_ledger.append(kind)
                    raise AssertionError(f"target question attempted {kind} contact")

                return refuse

            with (
                patch.object(socket, "socket", side_effect=refuse_contact("network")),
                patch.object(
                    subprocess, "Popen", side_effect=refuse_contact("process")
                ),
            ):
                pending = advance(ResumeRun(journey_file=receipt.journey_file))

            request = pending.planning_hitl.pending_request
            assert request is not None
            self.assertEqual(
                {
                    "status": pending.status,
                    "owner": request.owner,
                    "requirement": request.requirement_id,
                    "subject": request.subject_key,
                    "requirement_kind": next(
                        item.kind
                        for item in pending.requirement_resolution.requirements
                        if item.requirement_id == request.requirement_id
                    ),
                    "candidate_values": tuple(
                        item.asserted_value for item in request.candidates
                    ),
                    "legal_actions": tuple(
                        item.action for item in request.legal_actions
                    ),
                    "contact": tuple(contact_ledger),
                },
                {
                    "status": "planning_hitl_pending",
                    "owner": "requirement_resolution",
                    "requirement": target_requirement.requirement_id,
                    "subject": "requested_variant_rs1234",
                    "requirement_kind": "scientific_investigation",
                    "candidate_values": (),
                    "legal_actions": ("clarify", "reject", "stop"),
                    "contact": (),
                },
            )

            clarification = AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=request.request_event_id,
                answer=PlanningReviewAnswer(
                    kind="planning",
                    response=ClarifyPlanningHitlResponse(
                        action="clarify", instruction="Please check rs429358."
                    ),
                ),
            )
            with use_run_advancement_composition(composition):
                continued = advance(clarification)
                retained_after_first = Journey.open(Path(receipt.journey_file)).read()
                replayed = advance(clarification)
                retained_after_replay = Journey.open(Path(receipt.journey_file)).read()
            target_receipt = next(
                item
                for item in continued.planning_hitl.receipts
                if item.request_event_id == request.request_event_id
            )
            (item,) = continued.requirement_resolution.scientific_investigation.items
            self.assertEqual(
                {
                    "targets": [
                        (target.words, target.rsid, target.source_event_id)
                        for target in item.targets
                    ],
                    "canonical_replay": replayed == continued,
                    "replay_appended": len(retained_after_replay)
                    - len(retained_after_first),
                },
                {
                    "targets": [
                        ("rs429358", "rs429358", target_receipt.response_event_id)
                    ],
                    "canonical_replay": True,
                    "replay_appended": 0,
                },
            )

    def test_answer_planning_stop_is_terminal_idempotent_and_conflict_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            with use_run_advancement_composition(
                _OfflineAdvancementComposition(receipt, paths)
            ):
                pending = advance(
                    ResumeRun(journey_file=receipt.journey_file)
                ).planning_hitl.pending_request
            assert pending is not None
            journey = Journey.open(Path(receipt.journey_file))
            answer = AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=pending.request_event_id,
                answer=PlanningReviewAnswer(
                    kind="planning",
                    response=StopPlanningHitlResponse(
                        action="stop", reason="Required evidence is unavailable."
                    ),
                ),
            )

            before_refusals = len(journey.read())
            with self.assertRaises(ReviewAnswerRefused) as wrong_kind:
                advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=pending.request_event_id,
                        answer=ModelAvailabilityReviewAnswer(
                            kind="model_availability",
                            response=RetryModelAvailabilityResponse(action="retry"),
                        ),
                    )
                )
            with self.assertRaises(PlanningHitlResponseRefused) as illegal:
                advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=pending.request_event_id,
                        answer=PlanningReviewAnswer(
                            kind="planning",
                            response=SelectPlanningHitlResponse(
                                action="select", candidate_id="not-a-retained-candidate"
                            ),
                        ),
                    )
                )
            with self.assertRaises(PlanningHitlResponseRefused) as stale:
                advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=str(uuid.uuid4()),
                        answer=PlanningReviewAnswer(
                            kind="planning",
                            response=StopPlanningHitlResponse(
                                action="stop", reason="Unknown request."
                            ),
                        ),
                    )
                )
            self.assertEqual(len(journey.read()), before_refusals)
            self.assertEqual(
                (
                    wrong_kind.exception.code,
                    wrong_kind.exception.disposition.kind,
                ),
                (
                    "review_answer_kind_mismatch",
                    "caller_correction_and_reissue",
                ),
            )
            self.assertEqual(
                (
                    illegal.exception.code,
                    illegal.exception.disposition.kind,
                ),
                (
                    "planning_hitl_response_illegal",
                    "caller_correction_and_reissue",
                ),
            )
            self.assertEqual(stale.exception.code, "planning_hitl_response_stale")

            terminal = advance(answer)
            retained_after_first = Journey.open(Path(receipt.journey_file)).read()
            replayed = advance(answer)
            retained_after_replay = Journey.open(Path(receipt.journey_file)).read()
            conflict = AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=pending.request_event_id,
                answer=PlanningReviewAnswer(
                    kind="planning",
                    response=SelectPlanningHitlResponse(
                        action="select", candidate_id=pending.candidates[0].candidate_id
                    ),
                ),
            )
            with self.assertRaises(PlanningHitlResponseRefused) as raised:
                advance(conflict)

            self.assertEqual(
                {
                    "terminal": (terminal.status, terminal.failure_code),
                    "canonical_replay": replayed
                    == observe(ObserveRun(journey_file=receipt.journey_file)),
                    "replay_appended": len(retained_after_replay)
                    - len(retained_after_first),
                    "conflict_code": raised.exception.code,
                    "conflict_appended": len(
                        Journey.open(Path(receipt.journey_file)).read()
                    )
                    - len(retained_after_replay),
                },
                {
                    "terminal": (
                        "finalization_review_pending",
                        "required_evidence_unresolved",
                    ),
                    "canonical_replay": True,
                    "replay_appended": 0,
                    "conflict_code": "planning_hitl_response_conflict",
                    "conflict_appended": 0,
                },
            )

    def test_begin_registers_and_automatically_advances_to_planning_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            begin_root = root / "begun-runs"
            request_file = root / "snp-request.json"
            request_file.write_text(
                json.dumps(
                    {
                        "asserted_by": "researcher:owner",
                        "request": "Investigate rs4149056 in HG001.",
                        "target": {"rsid": "rs4149056"},
                    }
                ),
                encoding="utf-8",
            )

            intent = BeginRun(
                run_root=str(begin_root.resolve()),
                prompt_file=str((root / "prompt.txt").resolve()),
                runtime_config_file=str((root / "runtime.json").resolve()),
                request_files=(str(request_file),),
                inputs=tuple(
                    RegisterFastqInput(role=role, registered_path=str(path.resolve()))
                    for role, path in paths.items()
                ),
            )

            with use_run_advancement_composition(
                _OfflineAdvancementComposition(receipt, paths)
            ):
                advanced = advance(intent)

            canonical = observe(ObserveRun(journey_file=advanced.journey_file))
            events = Journey.open(Path(advanced.journey_file)).read()
            closures = [
                event for event in events if event["event_type"] == "PassClosed"
            ]
            self.assertEqual(len(closures), 1)
            self.assertEqual(closures[0]["payload"]["disposition"], "waiting")
            self.assertEqual(events[-1]["event_id"], closures[0]["event_id"])
            owner = Journey.open(Path(advanced.journey_file))
            journal = owner.read_artifact(closures[0]["artifact_references"][0])
            rows = [json.loads(line) for line in journal.splitlines()]
            model_starts = [
                row
                for row in rows
                if row["kind"] == "start" and row["operation"] == "model.request"
            ]
            self.assertEqual(
                {row["scope"]["attempt_id"] for row in model_starts},
                {attempt.attempt_id for attempt in advanced.attempts},
            )
            self.assertTrue(rows[-1]["complete"])
            self.assertEqual(
                {
                    "intent_fields": tuple(BeginRun.model_fields),
                    "canonical": advanced == canonical,
                    "status": advanced.status,
                    "pending_kind": advanced.planning_hitl.pending_request.subject_key,
                    "attempt_roles": tuple(
                        item.agent_role for item in advanced.attempts
                    ),
                    "execution_authority": advanced.execution_hitl,
                    "captured_request": [
                        (item.rsid, item.request_text)
                        for item in advanced.investigation_request.request.items
                    ],
                    "captured_by": advanced.investigation_request.request.asserted_by,
                    "concise_request": run_interaction(advanced).investigation_request,
                },
                {
                    "intent_fields": (
                        "run_root",
                        "prompt_file",
                        "runtime_config_file",
                        "inputs",
                        "request_files",
                        "paper_files",
                    ),
                    "canonical": True,
                    "status": "planning_hitl_pending",
                    "pending_kind": "analysis_recipe",
                    "attempt_roles": ("planning_agent",),
                    "execution_authority": None,
                    "captured_request": [
                        ("rs4149056", "Investigate rs4149056 in HG001.")
                    ],
                    "captured_by": "researcher:owner",
                    "concise_request": canonical.investigation_request,
                },
            )

    def test_observe_reconstructs_registered_run_without_contact_or_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prompt = root / "prompt.txt"
            runtime = root / "runtime.json"
            read_1 = root / "read-1.fastq.gz"
            read_2 = root / "read-2.fastq.gz"
            prompt.write_text(
                "Analyze the registered paired FASTQs and produce a VCF.",
                encoding="utf-8",
            )
            runtime.write_text(
                json.dumps(
                    {
                        "contract_version": "1.0",
                        "heartbeat_interval_seconds": 30,
                        "model_roster": {
                            "qwen": {
                                "provider": "nebius",
                                "endpoint": "https://model.invalid/v1",
                                "model": "Qwen/Qwen3.5-397B-A17B",
                                "credential_env": "NEBIUS_TOKEN",
                            }
                        },
                        "agent_invocation": {"model_reference": "qwen"},
                    }
                ),
                encoding="utf-8",
            )
            read_1.write_bytes(b"registered-read-1")
            read_2.write_bytes(b"registered-read-2")
            receipt = register_run(
                RegisterRunCommand(
                    run_root=str(root / "runs"),
                    prompt_file=str(prompt),
                    runtime_config_file=str(runtime),
                    inputs=(
                        RegisterFastqInput(role="read_1", registered_path=str(read_1)),
                        RegisterFastqInput(role="read_2", registered_path=str(read_2)),
                    ),
                ),
                repository_root=Path.cwd(),
            )
            intent = ObserveRun(journey_file=receipt.journey_file)
            journey_path = Path(receipt.journey_file)
            journey_before = journey_path.read_bytes()
            before = _file_snapshot(root)
            contact_ledger: list[str] = []

            def refuse_contact(kind: str):
                def refuse(*_args, **_kwargs):
                    contact_ledger.append(kind)
                    raise AssertionError(f"observation attempted {kind} contact")

                return refuse

            with (
                patch.object(socket, "socket", side_effect=refuse_contact("network")),
                patch.object(
                    subprocess, "Popen", side_effect=refuse_contact("process")
                ),
            ):
                observation = observe(intent)

            after = _file_snapshot(root)
            journey_bytes_unchanged = journey_path.read_bytes() == journey_before
            self.assertEqual(
                {
                    "intent": intent.model_dump(mode="json"),
                    "journey_id": observation.journey_id,
                    "journey_file": observation.journey_file,
                    "status": observation.status,
                    "prompt": observation.request.prompt,
                    "input_roles": tuple(
                        item.role for item in observation.registered_input_set.inputs
                    ),
                    "lineage": tuple(item.event_type for item in observation.lineage),
                    "journey_bytes_unchanged": journey_bytes_unchanged,
                    "files_unchanged": after == before,
                    "contact_ledger": tuple(contact_ledger),
                },
                {
                    "intent": {"journey_file": receipt.journey_file},
                    "journey_id": receipt.journey_id,
                    "journey_file": receipt.journey_file,
                    "status": "inputs_registered",
                    "prompt": "Analyze the registered paired FASTQs and produce a VCF.",
                    "input_roles": ("read_1", "read_2"),
                    "lineage": (
                        "JourneyStarted",
                        "ResearchRequestRegistered",
                        "RunInputsRegistered",
                    ),
                    "journey_bytes_unchanged": True,
                    "files_unchanged": True,
                    "contact_ledger": (),
                },
            )

    def test_resume_at_pending_planning_review_returns_canonical_observation_without_contact_or_mutation(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            with use_run_advancement_composition(
                _OfflineAdvancementComposition(receipt, paths)
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            canonical = observe(ObserveRun(journey_file=receipt.journey_file))
            pending = canonical.planning_hitl.pending_request
            assert pending is not None
            intent = ResumeRun(journey_file=receipt.journey_file)
            journey_path = Path(receipt.journey_file)
            journey_before = journey_path.read_bytes()
            tree_before = _file_snapshot(root)
            model_evidence_before = sum(
                len(attempt.model_events) for attempt in canonical.attempts
            )
            tool_evidence_before = sum(
                len(attempt.tool_events) for attempt in canonical.attempts
            )
            contact_ledger: list[str] = []

            def refuse_contact(kind: str):
                def refuse(*_args, **_kwargs):
                    contact_ledger.append(kind)
                    raise AssertionError(f"resume attempted {kind} contact")

                return refuse

            with (
                patch.object(socket, "socket", side_effect=refuse_contact("network")),
                patch.object(
                    subprocess, "Popen", side_effect=refuse_contact("process")
                ),
            ):
                resumed = advance(intent)

            model_evidence_after = sum(
                len(attempt.model_events) for attempt in resumed.attempts
            )
            tool_evidence_after = sum(
                len(attempt.tool_events) for attempt in resumed.attempts
            )
            self.assertEqual(
                {
                    "intent": intent.model_dump(mode="json"),
                    "canonical_observation": resumed == canonical,
                    "status": resumed.status,
                    "request_unchanged": resumed.planning_hitl.pending_request
                    == pending,
                    "request_event_id": pending.request_event_id,
                    "legal_actions": tuple(
                        action.action for action in pending.legal_actions
                    ),
                    "journey_bytes_unchanged": journey_path.read_bytes()
                    == journey_before,
                    "tree_bytes_unchanged": _file_snapshot(root) == tree_before,
                    "model_contact_delta": model_evidence_after - model_evidence_before,
                    "tool_contact_delta": tool_evidence_after - tool_evidence_before,
                    "contact_ledger": tuple(contact_ledger),
                },
                {
                    "intent": {"journey_file": receipt.journey_file},
                    "canonical_observation": True,
                    "status": "planning_hitl_pending",
                    "request_unchanged": True,
                    "request_event_id": pending.request_event_id,
                    "legal_actions": ("select", "clarify", "reject", "stop"),
                    "journey_bytes_unchanged": True,
                    "tree_bytes_unchanged": True,
                    "model_contact_delta": 0,
                    "tool_contact_delta": 0,
                    "contact_ledger": (),
                },
            )

    def test_public_dependencies_bind_planning_adapters_and_explicit_environment(self):
        from bio_neb.application.advancement import use_run_advancement_dependencies
        from bio_neb.parabricks.capability_policy import CapabilitySnapshotAdapter

        class RecordedRecipes(CapabilitySnapshotAdapter):
            def __init__(self, snapshot):
                super().__init__(snapshot)
                self.requests = []

            def list_recipes(self, tool_input):
                self.requests.append(tool_input)
                return super().list_recipes(tool_input)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            controlled = PlanningComposition(receipt, paths)
            recipes = RecordedRecipes(controlled.capability)
            credentials = []

            async def respond(messages, info):
                controlled.journey = Journey.open(Path(receipt.journey_file))
                return await controlled.respond(messages, info)

            def model_factory(loaded, _selected):
                credentials.append(loaded.credential)
                return ScriptedModelEndpoint(respond)(loaded, _selected)

            unbound = {k: v for k, v in os.environ.items() if k != "LOCAL_MODEL_KEY"}
            # The inner binding supplies only the credential; the outer adapters remain.
            with (
                patch.dict(os.environ, unbound, clear=True),
                use_run_advancement_dependencies(
                    planning_model_factory=model_factory,
                    reference_adapter=controlled.reference_adapter,
                    companion_adapter=controlled.companion_adapter,
                    galaxy_adapter=controlled.galaxy_adapter,
                    recipe_adapter=recipes,
                ),
                use_run_advancement_dependencies(
                    environ={"LOCAL_MODEL_KEY": "bound-only"}
                ),
            ):
                result = advance(ResumeRun(journey_file=receipt.journey_file))

        self.assertEqual(result.status, "planning_hitl_pending")
        self.assertEqual(set(credentials), {"bound-only"})
        self.assertGreater(controlled.reference_adapter.contact_count, 0)
        self.assertGreater(controlled.companion_adapter.contact_count, 0)
        self.assertTrue(recipes.requests)


def _recipe_observations(journey_file):
    return [
        e
        for e in Journey.open(Path(journey_file)).read()
        if e["event_type"] == "RecipeObservationRecorded"
    ]


def _assert_read_back_with_one_recipe_work_item(test, composition, current):
    """Resume and observe read the Journey back unchanged, and the recipe work item
    recorded first is the only one: no later Planning answer changed it."""
    path = Path(current.journey_file)
    with use_run_advancement_composition(composition):
        test.assertEqual(advance(ResumeRun(journey_file=str(path))), current)
    test.assertEqual(observe(ObserveRun(journey_file=str(path))), current)
    events = Journey.open(path).read()
    recipes = [e for e in events if e["event_type"] == "RecipeObservationRecorded"]
    first = recipes[0]["payload"]["tool_input"]["work_item"]["work_item_id"]
    # A Planning answer follows the first recipe fact, so the claim is exercised.
    test.assertIn(
        "PlanningHitlResponded",
        [e["event_type"] for e in events[events.index(recipes[0]) :]],
    )
    test.assertEqual(
        [e["payload"]["tool_input"]["work_item"]["work_item_id"] for e in recipes],
        [first] * len(recipes),
    )
    test.assertEqual(
        [
            item.work_item_id
            for item in read_planning(Journey.open(path)).execution.recipe_work_items
        ],
        [first],
    )


class _EarlyRecipeCall(_OfflineAdvancementComposition):
    """Once armed, calls the recipe listing before every output is settled."""

    armed = False

    async def respond(self, messages, info):
        if self.armed and "describe_companion" in {
            tool.name for tool in info.function_tools
        }:
            self.armed = False
            self.calls.append("list_compatible_analysis_recipes")
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "list_compatible_analysis_recipes",
                        {},
                        tool_call_id=f"early-{len(self.calls)}",
                    )
                ]
            )
        return await super().respond(messages, info)


class SettledTargetGateTests(unittest.TestCase):
    """The Execution proposal waits until every target is settled (#342).

    Recipe work does not wait: Execution always produces BAM + VCF. The offline
    composition supplies no execution configuration, so a Run stops at the
    preparation question after `execution_planning_ready`; the Execution proposal
    is compiled from that Journey.
    """

    def _advance_answering(self, root, paths, composition, target_response):
        """Begin a Run, answering each planning Review; the target question gets
        ``target_response``.

        Returns the stop and, per target question, the recipe facts recorded while
        it was pending.
        """
        intent = BeginRun(
            run_root=str((root / "advanced-runs").resolve()),
            prompt_file=str((root / "prompt.txt").resolve()),
            runtime_config_file=str((root / "runtime.json").resolve()),
            inputs=tuple(
                RegisterFastqInput(role=role, registered_path=str(path.resolve()))
                for role, path in paths.items()
            ),
        )
        recipes_at_target_question = []
        with use_run_advancement_composition(composition):
            current = advance(intent)
            for _ in range(12):
                pending = next(iter(current.planning_hitl.pending_requests), None)
                if pending is None:
                    return current, recipes_at_target_question
                if pending.owner == "requirement_resolution":
                    recipes_at_target_question.append(
                        _recipe_observations(current.journey_file)
                    )
                    response = target_response(pending)
                else:
                    response = {
                        "action": "select",
                        "candidate_id": pending.candidates[0].candidate_id,
                    }
                current = advance(
                    answer_review_from_json(
                        current, pending.request_event_id, json.dumps(response)
                    )
                )
        raise AssertionError("the Run did not reach a stable stop")

    def test_an_open_target_beside_a_stated_one_holds_only_the_execution_proposal(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            composition = _OfflineAdvancementComposition(
                receipt, paths, open_sibling_target=True
            )

            def name_it(pending):
                self.assertIsNone(
                    observe(
                        ObserveRun(journey_file=receipt.journey_file)
                    ).execution_hitl
                )
                return {"action": "clarify", "instruction": "rs7412"}

            current, recipes_at_question = self._advance_answering(
                root, paths, composition, name_it
            )

            self.assertEqual(len(recipes_at_question), 1)
            self.assertEqual(current.status, "execution_planning_ready")
            recipes = _recipe_observations(current.journey_file)
            self.assertTrue(recipes)
            # Recipe listing asks for the fixed product outputs.
            self.assertEqual(
                {
                    tuple(
                        e["payload"]["tool_input"]["work_item"][
                            "proposal_output_values"
                        ]
                    )
                    for e in recipes
                },
                {("BAM", "VCF")},
            )
            self.assertEqual(
                [
                    [(t.words, t.rsid) for t in item.targets]
                    for item in current.requirement_resolution.scientific_investigation.items
                ],
                [[("rs1234", "rs1234")], [("rs7412", "rs7412")]],
            )
            _assert_read_back_with_one_recipe_work_item(self, composition, current)

    def test_a_rejected_target_is_left_out_and_the_stated_one_goes_on(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            composition = _OfflineAdvancementComposition(
                receipt, paths, open_sibling_target=True
            )
            rejected_ids = []

            def reject(pending):
                rejected_ids.append(pending.requirement_id)
                return {"action": "reject", "reason": "Do not pursue this target."}

            current, _ = self._advance_answering(root, paths, composition, reject)

            self.assertEqual(current.status, "execution_planning_ready")
            states = {
                item.requirement_id: item.state
                for item in current.requirement_resolution.requirements
                if item.kind == "scientific_investigation"
            }
            self.assertEqual(states[rejected_ids[0]], "rejected")
            self.assertEqual(
                [
                    [(t.words, t.rsid) for t in item.targets]
                    for item in current.requirement_resolution.scientific_investigation.items
                ],
                [[("rs1234", "rs1234")], []],
            )
            # Planning is complete: the proposal compiles without the left-out target.
            proposal = compile_execution_proposal(
                read_evidence(Journey.open(Path(current.journey_file)))
            )
            self.assertNotIn(
                rejected_ids[0],
                {
                    item.requirement_id
                    for item in proposal.requirement_resolution.requirements
                },
            )
            _assert_read_back_with_one_recipe_work_item(self, composition, current)

    def test_mixed_planning_and_reference_history_survives_scientific_finalization(
        self,
    ):
        from bio_neb.parabricks.plan import approved_execution_input
        from bio_neb.parabricks.planning_commands import (
            RespondExecutionHitlCommand,
            respond_execution_hitl,
        )
        from bio_neb.reporting.finalization_commands import ApproveFinalizationCommand
        from tests.finalization_fakes import observe_existing_run
        from tests.planning_assignment_fakes import (
            configured_planning_run,
            execution_configuration,
        )
        from tests.terminal_journey_fakes import execute_admitted_terminal

        class Exploring(_OfflineAdvancementComposition):
            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                if (
                    self.calls[-1] in {"inspect_reference", "describe_companion"}
                    and self.calls.count(self.calls[-1]) == 1
                ):
                    response.parts[0].args["accession"] = "GCF_000000001.1"
                return response

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            configuration = execution_configuration(root)
            # The Run registers the request the evidence helper investigates;
            # planning cites it.
            receipt, base, _ = configured_planning_run(
                root,
                configuration=configuration,
                investigation_request=EVIDENCE_REQUEST,
                scientific_intent="Also check the APOE site.",
            )
            # Beside the cited file, the request's own words name a target with no
            # rsID or position, so it is asked, and the researcher leaves it out
            # (a target-less item beside a file is refused).
            composition = Exploring(
                receipt, base.paths, untyped_sibling_quote="the APOE site"
            )
            composition.capability = configuration.runtime_selection.capability
            with use_run_advancement_composition(composition):
                current = advance(ResumeRun(journey_file=receipt.journey_file))
                while current.current_review.kind == "planning":
                    request = current.current_review.request
                    response = (
                        {"action": "reject", "reason": "Do not pursue this target."}
                        if request.owner == "requirement_resolution"
                        else {
                            "action": "select",
                            "candidate_id": request.candidates[0].candidate_id,
                        }
                    )
                    current = advance(
                        answer_review_from_json(
                            current, request.request_event_id, json.dumps(response)
                        )
                    )
            self.assertEqual(current.current_review.kind, "execution")
            execution_request = current.current_review.request
            self.assertEqual(
                [
                    item.state
                    for item in current.requirement_resolution.requirements
                    if item.kind == "scientific_investigation"
                ],
                ["rejected", "post_execution"],
            )
            respond_execution_hitl(
                RespondExecutionHitlCommand(
                    journey_file=receipt.journey_file,
                    request_event_id=execution_request.request_event_id,
                    response=approve_execution(execution_request.proposal),
                )
            )
            approved = approved_execution_input(
                Journey.open(Path(receipt.journey_file))
            )
            execute_admitted_terminal(approved)
            journey = observe_existing_run(
                root, Journey.open(Path(receipt.journey_file))
            )
            ended = advance(ResumeRun(journey_file=str(journey.path)))
            self.assertEqual(ended.status, "finalization_review_pending")
            pending = ended.finalization.pending_request
            basis = pending.basis
            self.assertEqual(
                basis.work_items[0].target.assembly_accession, "GCF_000001405.40"
            )
            reference = [
                item
                for item in basis.accounting
                if item.event_type == "ReferenceDescriptionRecorded"
            ]
            companion = [
                item
                for item in basis.accounting
                if item.event_type == "CompanionObservationRecorded"
            ]
            self.assertEqual(
                {item.payload["result"]["resolution"] for item in reference},
                {"source_disagreement", "resolved"},
            )
            self.assertEqual(
                {item.payload["result"]["outcome"] for item in companion},
                {"incompatible", "verified"},
            )
            original = {event["event_id"]: event for event in read_evidence(journey)}
            for item in (*reference, *companion):
                self.assertEqual(
                    item.payload, original[item.source_event_id]["payload"]
                )
            self.assertIn(
                "Do not pursue this target.",
                json.dumps([item.model_dump(mode="json") for item in basis.accounting]),
            )
            with explained_by_default():
                published = advance(
                    ApproveFinalizationCommand(
                        journey_file=str(journey.path),
                        request_event_id=pending.request_event_id,
                        basis_digest=basis.basis_digest,
                        head_event_id=pending.request_event_id,
                    )
                )
            self.assertEqual(published.status, "grounded_report_published")
            report = published.grounded_report.report
            self.assertEqual(
                tuple(item.value for item in report.accounting), basis.accounting
            )
            self.assertTrue(
                all(not item.membership.admitted for item in report.accounting)
            )
            self.assertEqual(report.work_items[0].value, basis.work_items[0])
            self.assertEqual(
                observe(ObserveRun(journey_file=str(journey.path))), published
            )

    def test_a_target_hold_reentry_closes_without_new_facts_or_model_contact(self):
        # Only the researcher can settle the target, so once planning is ready the
        # coordinator dispatches no planning Attempt there: resuming writes no
        # fact and contacts no model.
        from bio_neb.agentic_journey.planning_assignment import (
            planning_closes_before_contact,
        )

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory).resolve())
            composition = _OfflineAdvancementComposition(
                receipt, paths, open_sibling_target=True
            )
            with use_run_advancement_composition(composition):
                held = advance(ResumeRun(journey_file=receipt.journey_file))
                for _ in range(8):
                    agent = [
                        r
                        for r in held.planning_hitl.pending_requests
                        if r.owner == "agent_attempt"
                    ]
                    if not agent:
                        break
                    held = advance(
                        answer_review_from_json(
                            held,
                            agent[0].request_event_id,
                            json.dumps(
                                {
                                    "action": "select",
                                    "candidate_id": agent[0].candidates[0].candidate_id,
                                }
                            ),
                        )
                    )
                journey = Journey.open(Path(receipt.journey_file))
                self.assertEqual(
                    [r.owner for r in held.planning_hitl.pending_requests],
                    ["requirement_resolution"],
                )
                self.assertEqual(
                    _planning_readiness(journey).code, "investigation_targets_unsettled"
                )
                self.assertTrue(planning_closes_before_contact(journey))
                before = Path(receipt.journey_file).read_bytes()
                calls = tuple(composition.calls)

                async def no_contact(messages, info):
                    raise AssertionError(
                        "a target-only hold must close before model contact"
                    )

                composition.respond = no_contact
                resumed = advance(ResumeRun(journey_file=receipt.journey_file))

            self.assertEqual(tuple(composition.calls), calls)
            self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
            self.assertEqual(
                resumed.planning_hitl.pending_requests,
                held.planning_hitl.pending_requests,
            )

    def test_a_premature_recipe_call_is_refused_while_companion_work_remains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            composition = _EarlyRecipeCall(receipt, paths)
            # Ask too early while companion work still remains.
            composition.armed = True
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            recipes = _recipe_observations(receipt.journey_file)

        validations = {
            e["event_id"]: e["payload"]["errors"]
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_name"] == "list_compatible_analysis_recipes"
            and e["payload"]["errors"]
        }
        self.assertTrue(validations)
        self.assertEqual(
            {tuple(errors) for errors in validations.values()},
            {("tool_name_unavailable",)},
        )
        # An unoffered call is refused while its reply is routed; its refusal is
        # named in the turn's note.
        feedback = next(
            e["payload"]["feedback"]
            for e in events
            if e["event_type"] == "AgentTurnRouted"
            and set(e["payload"]["unoffered_validation_event_ids"]) & set(validations)
        )
        self.assertIn("the current stage is `companions`", feedback)
        self.assertIn("reference and its companions are verified", feedback)
        # Recipe work runs only once the companion work is done.
        ids = [e["event_id"] for e in events]
        last_companion = max(
            ids.index(e["event_id"])
            for e in events
            if e["event_type"] == "CompanionObservationRecorded"
        )
        self.assertTrue(all(ids.index(e["event_id"]) > last_companion for e in recipes))

    def test_every_target_rejected_reaches_finalization_before_execution(self):
        # With no target left there is nothing to investigate, so no
        # Execution proposal is made; the Run ends and is accounted.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            composition = _OfflineAdvancementComposition(
                receipt, paths, open_target=True
            )

            current, _ = self._advance_answering(
                root,
                paths,
                composition,
                lambda _: {"action": "reject", "reason": "No target is wanted."},
            )

            self.assertEqual(current.status, "finalization_review_pending")
            self.assertFalse(
                any(
                    e["event_type"] == "ExecutionProposalSubmitted"
                    for e in read_evidence(Journey.open(Path(current.journey_file)))
                )
            )
            basis = current.finalization.pending_request.basis
            self.assertEqual(basis.outcome, "incomplete")
            self.assertEqual(basis.work_items, ())
            self.assertEqual(basis.admissions, ())
            self.assertIsNone(basis.question)
            self.assertTrue(
                any(
                    entry.event_type == "ResearchRequestRegistered"
                    for entry in basis.accounting
                )
            )
            self.assertIn(
                "No target is wanted.",
                json.dumps(
                    [entry.model_dump(mode="json") for entry in basis.accounting]
                ),
            )
            self.assertEqual(
                observe(ObserveRun(journey_file=current.journey_file)).finalization,
                current.finalization,
            )

            from bio_neb.reporting.finalization_commands import (
                ApproveFinalizationCommand,
                CorrectFinalizationCommand,
            )

            correction_file = root / "correction.json"
            excluded = basis.accounting[0].accounting_id
            correction_file.write_text(
                json.dumps(
                    {
                        "asserted_by": "researcher",
                        "reason": "Exclude this accounting entry.",
                        "requested_changes": [
                            {
                                "action": "set_report_membership",
                                "member_id": excluded,
                                "included": False,
                            }
                        ],
                    }
                )
            )
            pending = current.finalization.pending_request
            corrected = advance(
                CorrectFinalizationCommand(
                    journey_file=current.journey_file,
                    request_event_id=pending.request_event_id,
                    basis_digest=basis.basis_digest,
                    head_event_id=pending.request_event_id,
                    correction_file=str(correction_file),
                )
            )
            self.assertEqual(corrected.status, "finalization_review_pending")
            self.assertIsNone(corrected.finalization.report_basis)
            pending = corrected.finalization.pending_request
            with explained_by_default():
                published = advance(
                    ApproveFinalizationCommand(
                        journey_file=current.journey_file,
                        request_event_id=pending.request_event_id,
                        basis_digest=pending.basis.basis_digest,
                        head_event_id=pending.request_event_id,
                    )
                )
            self.assertEqual(published.status, "grounded_report_published")
            report = published.grounded_report.report
            self.assertEqual(report.work_items, ())
            self.assertEqual(report.admissions, ())
            self.assertIsNone(report.question)
            self.assertTrue(
                all(not member.membership.admitted for member in report.accounting)
            )
            self.assertFalse(
                next(
                    member.membership.included
                    for member in report.accounting
                    if member.membership.member_id == excluded
                )
            )
            self.assertEqual(
                observe(ObserveRun(journey_file=current.journey_file)), published
            )
            self.assertEqual(
                advance(ResumeRun(journey_file=current.journey_file)), published
            )
            rendered, error = io.StringIO(), io.StringIO()
            with redirect_stdout(rendered), redirect_stderr(error):
                exit_code = main(
                    [
                        "grounded-report-reopen",
                        "--journey-file",
                        current.journey_file,
                        "--report-digest",
                        report.report_digest,
                    ]
                )
            self.assertEqual((exit_code, error.getvalue()), (0, ""))
            self.assertIn("incomplete", rendered.getvalue())
            self.assertIn("No target is wanted.", rendered.getvalue())

    def test_a_tool_found_target_beside_a_stated_one_is_asked_before_the_proposal(
        self,
    ):
        # A sibling target recorded for a Tool to find after Execution still names
        # no target in the researcher's words: the coordinator asks for it, and the
        # Run reaches its preparation stop once it is named.
        class ToolFoundSibling(_OfflineAdvancementComposition):
            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                part = response.parts[0]
                if part.tool_name == "record_proposal_requirements":
                    sibling = next(
                        item
                        for item in part.args["requirements"]
                        if item["subject_key"] == "additional_target"
                    )
                    sibling["acquisition_route"] = "validate_with_tool"
                    sibling["validation"] = {
                        "kind": "tool_evidence",
                        "requirement": "Find the target after Execution.",
                    }
                return response

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            composition = ToolFoundSibling(receipt, paths, open_sibling_target=True)

            current, asked = self._advance_answering(
                root,
                paths,
                composition,
                lambda _: {"action": "clarify", "instruction": "rs7412"},
            )

            sibling = next(
                item
                for item in current.proposal_plan.plan.requirements
                if item.subject_key == "additional_target"
            )
            self.assertEqual(sibling.acquisition_route, "validate_with_tool")
            self.assertEqual(len(asked), 1)
            self.assertEqual(current.status, "execution_planning_ready")
            self.assertEqual(current.requirement_resolution.unsettled_targets, ())
            self.assertTrue(_recipe_observations(current.journey_file))


if __name__ == "__main__":
    unittest.main()
