import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError

from bio_neb.agentic_journey.agent_roles import role_instructions
from bio_neb.agentic_journey.continuation_policy import ExecutionReviewPolicy
from bio_neb.agentic_journey.instruction_evidence import INSTRUCTION_MEDIA_TYPE
from bio_neb.agentic_journey.model_availability import (
    ModelAvailabilityReviewRequestView,
)
from bio_neb.application.advancement import use_run_advancement_dependencies
from bio_neb.application.queries import (
    AttemptEvidenceReview,
    JourneyEvidenceItem,
    ObserveRun,
    observe,
)
from bio_neb.application.reviews import current_review
from bio_neb.application.run_interaction import (
    RunInteraction,
    run_interaction,
)
from bio_neb.cli import (
    _parser,
    _PublicRefusalView,
    main,
)
from bio_neb.investigation.investigation_commands import (
    BeginInvestigationCommand,
)
from bio_neb.investigation.scientific_continuation import (
    ScientificContinuationPacket,
    ScientificContinuationRefused,
)
from bio_neb.journey import ArtifactReference, Journey, canonical_json
from bio_neb.parabricks.execution_proposal import (
    ExecutionAuthority,
    ExecutionHitlRequestView,
    ExecutionProposal,
    compile_execution_proposal,
    execution_authority_id,
)
from bio_neb.parabricks.planning_hitl import PlanningHitlRequestView
from bio_neb.parabricks.requirement_resolution import (
    ScientificInvestigationHandoff,
)
from bio_neb.providers.nebius_common import NebiusCliResponse
from bio_neb.providers.nebius_controller_runtime import (
    ACCEPTED_NEBIUS_EXECUTABLE,
    ACCEPTED_NEBIUS_EXECUTABLE_BYTE_COUNT,
    ACCEPTED_NEBIUS_EXECUTABLE_SHA256,
)
from tests.execution_proposal_fakes import ready_timeline
from tests.observation_input_fakes import RUN_VCF, TABLE_READING_ENVIRON, RunOutputStore
from tests.run_prerequisites import prerequisite
from tests.sequential_investigation_support import (
    ASSEMBLY_BODY,
    SequentialAdaptiveModel,
    SequentialSourceResponses,
    begin_in_pass,
    forked_child,
)
from tests.vcf_lookup_fakes import TableReaderModel


def _cli_view(document=None, **attributes):
    payload = document if document is not None else dict(attributes)
    # A Run view's repeats; none unless a test supplies them.
    attributes.setdefault("repeats", ())
    return SimpleNamespace(
        **attributes,
        model_dump=lambda **_kwargs: payload,
    )


def _execution_proposal() -> ExecutionProposal:
    return compile_execution_proposal(ready_timeline())


def _execution_request(
    proposal: ExecutionProposal,
    *,
    request_event_id: str,
    pending: bool = True,
) -> ExecutionHitlRequestView:
    return ExecutionHitlRequestView.model_validate(
        {
            "request_event_id": request_event_id,
            "proposal": proposal,
            "review_policy": ExecutionReviewPolicy(),
            "legal_actions": (
                {"action": "approve", "response_schema": {}},
                {"action": "revise", "response_schema": {}},
                {"action": "reject", "response_schema": {}},
            ),
            "pending": pending,
        }
    )


def _execution_authority(
    proposal: ExecutionProposal,
    *,
    request_event_id: str,
    approval_event_id: str,
) -> ExecutionAuthority:
    return ExecutionAuthority(
        authority_id=execution_authority_id(
            proposal.proposal_id, request_event_id, approval_event_id
        ),
        proposal_id=proposal.proposal_id,
        request_event_id=request_event_id,
        approval_event_id=approval_event_id,
    )


def _scientific_investigation_handoff(
    *, journey_id: str, plan_event_id: str, items: list[dict[str, object]]
) -> ScientificInvestigationHandoff:
    material = {
        "journey_id": journey_id,
        "plan_event_id": plan_event_id,
        "items": items,
    }
    return ScientificInvestigationHandoff.model_validate(
        {
            "handoff_id": "scientific-investigation:"
            + hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest(),
            **material,
        }
    )


def _model_availability_request(
    request_event_id: str,
) -> ModelAvailabilityReviewRequestView:
    return ModelAvailabilityReviewRequestView.model_validate(
        {
            "request_event_id": request_event_id,
            "attempt_id": "00000000-0000-0000-0000-000000000021",
            "agent_role": "planning_agent",
            "retry_frontier_event_id": "00000000-0000-0000-0000-000000000022",
            "logical_request_event_id": "00000000-0000-0000-0000-000000000023",
            "logical_failure_event_id": "00000000-0000-0000-0000-000000000024",
            "request_number": 1,
            "nebius_model_endpoint": {
                "provider": "nebius",
                "model": "Qwen/Qwen3-235B-A22B",
                "contacted_url": "https://api.tokenfactory.nebius.com/v1",
            },
            "contacts": (
                {
                    "ordinal": 1,
                    "status_code": 503,
                    "failure_reason": "service_unavailable",
                    "completion_certainty": "not_started",
                    "policy_disposition": "review_required",
                },
            ),
            "delays": (),
            "exhaustion_reason": "automatic_retry_exhausted",
            "pending": True,
            "fault_key": "model endpoint: automatic_retry_exhausted",
        }
    )


def _planning_request(
    request_event_id: str,
    *,
    actions: tuple[str, ...],
    question: str,
) -> PlanningHitlRequestView:
    return PlanningHitlRequestView.model_validate(
        {
            "plan_event_id": "00000000-0000-0000-0000-000000000025",
            "plan_id": "plan_" + "a" * 64,
            "requirement_id": "requirement_" + "b" * 32,
            "subject_key": "analysis_output",
            "request_event_id": request_event_id,
            "attempt_id": "00000000-0000-0000-0000-000000000026",
            "issue": "A researcher decision is required.",
            "question": question,
            "evidence_event_ids": (),
            "candidates": (),
            "recommendation": None,
            "legal_actions": tuple(
                {"action": action, "response_schema": {}} for action in actions
            ),
            "pending": True,
        }
    )


def _cli_observation(*, status="inputs_registered", pending=None, execution=None):
    empty = _cli_view(requests=(), receipts=(), pending_request=None)
    planning = empty
    availability = empty
    if pending is not None:
        kind, request = pending
        projection = _cli_view(
            requests=(request,), receipts=(), pending_request=request
        )
        if kind == "planning":
            planning = projection
        elif kind == "model_availability":
            availability = projection
    execution_hitl = execution
    document = {
        "journey_id": "00000000-0000-0000-0000-000000000001",
        "journey_file": "/runs/run-1/journey.jsonl",
        "status": status,
        "rich": "retained Journey evidence",
    }
    observation = _cli_view(
        document,
        journey_id=document["journey_id"],
        journey_file=document["journey_file"],
        status=status,
        request=_cli_view(
            prompt="Investigate non-rs variants and the supporting literature.",
            prompt_file="/inputs/request.txt",
            source_event_id="00000000-0000-0000-0000-000000000010",
        ),
        attempts=(),
        pending_reviews=(),
        passes=(),
        requirement_resolution=None,
        planning_hitl=planning,
        model_availability_review=availability,
        agent_effort_review=empty,
        agent_work_review=empty,
        operational_review=empty,
        execution_hitl=execution_hitl,
        investigation=None,
        investigation_download_review=empty,
        finalization=_cli_view(
            bases=(), requests=(), receipts=(), status="not_started"
        ),
    )
    observation.current_review = current_review(observation)
    return observation


class CliTests(unittest.TestCase):
    def test_investigation_json_answers_authorize_only_explicit_selected_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            for transport in ("file", "inline", "interactive"):
                with self.subTest(transport=transport):
                    root = Path(temporary).resolve() / transport
                    request = {
                        "asserted_by": "researcher:owner",
                        "requests": [
                            {
                                "request": f"Investigate {rsid} in HG001.",
                                "target": {"rsid": rsid},
                            }
                            for rsid in SequentialAdaptiveModel.rsids
                        ],
                    }
                    journey = forked_child(
                        root,
                        parent_file=prerequisite(
                            "executed",
                            investigation_request=request,
                            output_payloads={"VCF": RUN_VCF},
                        ).journey_file,
                    )
                    request_file = root / "request.json"
                    request_file.write_text(json.dumps(request), encoding="utf-8")
                    pending = begin_in_pass(
                        BeginInvestigationCommand(
                            journey_file=str(journey.path),
                            request_files=(str(request_file),),
                        ),
                        environ={"LOCAL_MODEL_KEY": "offline-test-value"},
                        model_factory=lambda _, _selected: (
                            SequentialAdaptiveModel().model(_, _selected)
                        ),
                        source_adapter_factory=lambda: SequentialSourceResponses(
                            assembly_body=ASSEMBLY_BODY
                        ),
                    )
                    request = pending.pending_request
                    assert request is not None
                    current = observe(ObserveRun(journey_file=str(journey.path)))
                    self.assertEqual(current.current_review.kind, "investigation")
                    self.assertEqual(
                        current.current_review.request_event_id,
                        request.request_event_id,
                    )
                    self.assertEqual(
                        current.current_review.request.legal_actions,
                        ("approve", "revise", "stop"),
                    )
                    self.assertEqual(
                        current.current_review.request.review_policy.model_dump(),
                        {"stage": "investigation", "approval": "mandatory"},
                    )
                    selected = [
                        item.work_item_id for item in request.proposal.work_items[:2]
                    ][::-1]
                    response = {
                        "action": "approve",
                        "proposal_id": request.proposal.proposal_id,
                        "work_item_ids": selected,
                    }
                    before = journey.path.read_bytes()
                    for selection in (selected[0], [1], [True], selected):
                        with self.subTest(selection=selection):
                            document = json.dumps(
                                {**response, "work_item_ids": selection}
                            )
                            arguments = [
                                "answer",
                                "--journey-file",
                                str(journey.path),
                                "--request-event-id",
                                request.request_event_id,
                            ]
                            if transport == "file":
                                response_file = root / "response.json"
                                response_file.write_text(document, encoding="utf-8")
                                arguments += ["--response-file", str(response_file)]
                            elif transport == "inline":
                                arguments += ["--response-json", document]
                            else:
                                arguments = [
                                    "resume",
                                    "--journey-file",
                                    str(journey.path),
                                    "--interactive",
                                ]
                            output, errors = io.StringIO(), io.StringIO()
                            store = RunOutputStore()
                            with (
                                use_run_advancement_dependencies(
                                    environ=TABLE_READING_ENVIRON,
                                    investigation_model_factory=TableReaderModel().model,
                                    investigation_vcf_source_factory=store,
                                ),
                                patch("sys.stdin", io.StringIO(document + "\n")),
                                redirect_stdout(output),
                                redirect_stderr(errors),
                            ):
                                exit_code = main(arguments)
                            if selection != selected:
                                self.assertEqual(exit_code, 2)
                                self.assertEqual(journey.path.read_bytes(), before)
                                continue
                            # The approval is durable and continues into
                            # observation: the Run supplies its own verified VCF
                            # and BAM, one download each.
                            self.assertEqual(exit_code, 0, errors.getvalue())
                            self.assertEqual(len(store.gets), 2)
                            reviewed = observe(
                                ObserveRun(journey_file=str(journey.path))
                            )
                            authority = reviewed.investigation.authority
                            assert authority is not None
                            self.assertEqual(
                                {
                                    item.work_item_id
                                    for item in authority.authorized_work_items
                                },
                                set(selected),
                            )
                            self.assertEqual(len(request.proposal.work_items), 3)
                            # Completion waits for every member by default: no
                            # Finalization while one awaits approval.
                            self.assertEqual(
                                reviewed.status, "investigation_evidence_ready"
                            )
                            self.assertEqual(
                                reviewed.finalization.status, "not_started"
                            )
                            # The approval closed its Review; the unselected
                            # member waits on the Review it opened.
                            self.assertNotEqual(
                                reviewed.current_review.request_event_id,
                                request.request_event_id,
                            )
                            self.assertEqual(
                                len(reviewed.investigation.pending_work_item_ids), 1
                            )
                            retained = journey.path.read_bytes()
                            with (
                                patch("sys.stdin", io.StringIO(document + "\n")),
                                redirect_stdout(io.StringIO()),
                                redirect_stderr(io.StringIO()),
                            ):
                                replayed = main(arguments)
                            # An identical answer returns its retained receipt; an
                            # interactive resume offers it to the now-current
                            # remaining-scope Review, which refuses it unchanged.
                            self.assertEqual(
                                replayed, 2 if transport == "interactive" else 0
                            )
                            self.assertEqual(journey.path.read_bytes(), retained)

    def test_scoped_operational_stop_is_a_public_terminal_frontier(self):
        observation = _cli_observation(status="operational_stopped")
        observation.failure_code = "reference_facts_action_refused"
        observation.failure_source_event_id = "00000000-0000-0000-0000-000000000077"
        output = io.StringIO()
        with (
            patch("bio_neb.cli.advance", return_value=observation),
            redirect_stdout(output),
        ):
            exit_code = main(["resume", "--journey-file", observation.journey_file])

        rendered = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(rendered["status"], "operational_stopped")
        self.assertEqual(rendered["failure_code"], "reference_facts_action_refused")
        self.assertEqual(
            rendered["failure_source_event_id"], observation.failure_source_event_id
        )
        self.assertEqual(rendered["frontier"], "terminal")

    def test_observer_driven_begin_uses_only_registered_run_inputs(self):
        request = _planning_request(
            "00000000-0000-0000-0000-000000000002",
            actions=("clarify", "stop"),
            question="Which supported output should this Run produce?",
        )
        observation = _cli_observation(
            status="planning_hitl_pending", pending=("planning", request)
        )
        output = io.StringIO()
        with (
            patch(
                "bio_neb.cli.advance",
                return_value=observation,
            ) as advance,
            redirect_stdout(output),
        ):
            exit_code = main(
                [
                    "begin",
                    "--run-root",
                    "/runs",
                    "--prompt-file",
                    "/inputs/request.txt",
                    "--runtime-config",
                    "/inputs/runtime.json",
                    "--fastq",
                    "read_1=/inputs/read-1.fastq.gz",
                    "--fastq",
                    "read_2=/inputs/read-2.fastq.gz",
                ]
            )

        intent = advance.call_args.args[0]
        rendered = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            intent.model_dump(mode="json"),
            {
                "run_root": "/runs",
                "prompt_file": "/inputs/request.txt",
                "runtime_config_file": "/inputs/runtime.json",
                "inputs": [
                    {"role": "read_1", "registered_path": "/inputs/read-1.fastq.gz"},
                    {"role": "read_2", "registered_path": "/inputs/read-2.fastq.gz"},
                ],
                "request_files": [],
            },
        )
        self.assertEqual(rendered["frontier"], "review_required")
        self.assertEqual(rendered["review"]["kind"], "planning")
        self.assertNotIn("stage", type(intent).model_fields)
        self.assertNotIn("agent_role", type(intent).model_fields)
        self.assertNotIn("tool", type(intent).model_fields)

    def test_begin_asks_a_terminal_for_each_input_and_records_the_flag_form(self):
        class _Terminal(io.StringIO):
            def isatty(self):
                return True

        typed = _Terminal(
            "/inputs/request.txt\n"
            "/inputs/snp-request.json\n"
            "\n"
            "/inputs/paper.pdf\n"
            "\n"
            "/inputs/runtime.json\n"
            "/inputs/read-1.fastq.gz\n"
            "/inputs/read-2.fastq.gz\n"
        )
        observation = _cli_observation(status="inputs_registered")
        prompts = io.StringIO()
        with (
            patch("bio_neb.cli.advance", return_value=observation) as advance,
            patch("sys.stdin", typed),
            redirect_stdout(io.StringIO()),
            redirect_stderr(prompts),
        ):
            exit_code = main(["begin", "--run-root", "/runs"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            advance.call_args.args[0].model_dump(mode="json"),
            {
                "run_root": "/runs",
                "prompt_file": "/inputs/request.txt",
                "runtime_config_file": "/inputs/runtime.json",
                "inputs": [
                    {"role": "read_1", "registered_path": "/inputs/read-1.fastq.gz"},
                    {"role": "read_2", "registered_path": "/inputs/read-2.fastq.gz"},
                ],
                "request_files": ["/inputs/snp-request.json"],
                "paper_files": ["/inputs/paper.pdf"],
            },
        )
        self.assertEqual(
            prompts.getvalue(),
            "prompt file path: "
            "request file path (empty to finish): "
            "request file path (empty to finish): "
            "paper file path (empty to finish): "
            "paper file path (empty to finish): "
            "runtime file path (empty for the build's defaults): "
            "read_1 FASTQ path: "
            "read_2 FASTQ path: ",
        )

    def test_begin_given_its_required_inputs_asks_nothing_even_at_a_terminal(self):
        class _Terminal(io.StringIO):
            def isatty(self):
                return True

        typed = _Terminal("must stay unread\n")
        prompts = io.StringIO()
        with (
            patch(
                "bio_neb.cli.advance",
                return_value=_cli_observation(status="inputs_registered"),
            ) as advance,
            patch("sys.stdin", typed),
            redirect_stdout(io.StringIO()),
            redirect_stderr(prompts),
        ):
            main(
                [
                    "begin",
                    "--run-root",
                    "/runs",
                    "--prompt-file",
                    "/inputs/request.txt",
                    "--fastq",
                    "read_1=/inputs/read-1.fastq.gz",
                    "--fastq",
                    "read_2=/inputs/read-2.fastq.gz",
                ]
            )

        intent = advance.call_args.args[0]
        self.assertEqual((intent.runtime_config_file, intent.request_files), (None, ()))
        self.assertEqual((prompts.getvalue(), typed.tell()), ("", 0))

    def test_begin_without_a_terminal_refuses_missing_inputs_as_before(self):
        errors = io.StringIO()
        with (
            patch("bio_neb.cli.advance") as advance,
            patch("sys.stdin", io.StringIO("/inputs/request.txt\n")),
            redirect_stdout(io.StringIO()),
            redirect_stderr(errors),
            self.assertRaises(SystemExit) as refused,
        ):
            main(["begin", "--run-root", "/runs", "--prompt-file", "/inputs/p.txt"])

        self.assertEqual(refused.exception.code, 2)
        self.assertIn(
            "the following arguments are required: --fastq", errors.getvalue()
        )
        advance.assert_not_called()

    def test_begin_refuses_an_empty_answer_for_a_required_input(self):
        class _Terminal(io.StringIO):
            def isatty(self):
                return True

        errors = io.StringIO()
        with (
            patch("bio_neb.cli.advance") as advance,
            patch("sys.stdin", _Terminal("\n")),
            redirect_stdout(io.StringIO()),
            redirect_stderr(errors),
        ):
            exit_code = main(["begin", "--run-root", "/runs"])

        self.assertEqual(exit_code, 2)
        self.assertEqual(
            json.loads(errors.getvalue().split("prompt file path: ", 1)[1]),
            {"status": "failed", "code": "command_invalid", "fields": ["prompt_file"]},
        )
        advance.assert_not_called()

    def test_observer_driven_answer_derives_review_kind_and_resumes(self):
        request_id = "00000000-0000-0000-0000-000000000002"
        request = _planning_request(
            request_id,
            actions=("select",),
            question="Choose one candidate.",
        )
        current = _cli_observation(
            status="planning_hitl_pending", pending=("planning", request)
        )
        advanced = _cli_observation(status="execution_rejected")
        output = io.StringIO()
        owner = object()
        with (
            patch("bio_neb.application.advancement.Journey.open", return_value=owner),
            patch(
                "bio_neb.application.advancement.observe", return_value=current
            ) as observe,
            patch(
                "bio_neb.application.advancement.advance", return_value=advanced
            ) as advance,
            redirect_stdout(output),
        ):
            exit_code = main(
                [
                    "answer",
                    "--journey-file",
                    current.journey_file,
                    "--request-event-id",
                    request_id,
                    "--response-json",
                    '{"action":"select","candidate_id":"candidate-1"}',
                ]
            )

        intent = advance.call_args.args[0]
        self.assertEqual(exit_code, 0)
        self.assertEqual(intent.answer.kind, "planning")
        self.assertEqual(intent.answer.response.action, "select")
        # The Review is read and answered on one owner.
        self.assertIs(observe.call_args.kwargs["journey"], owner)
        self.assertIs(advance.call_args.kwargs["journey"], owner)
        self.assertEqual(json.loads(output.getvalue())["frontier"], "terminal")

    def test_interactive_begin_answers_one_review_and_shows_resumed_frontier(self):
        request_id = "00000000-0000-0000-0000-000000000002"
        request = _planning_request(
            request_id,
            actions=("select",),
            question="Choose one candidate.",
        )
        pending = _cli_observation(
            status="planning_hitl_pending", pending=("planning", request)
        )
        resumed = _cli_observation(status="execution_rejected")
        stdout = io.StringIO()
        stderr = io.StringIO()
        stdin = io.StringIO('{"action":"select","candidate_id":"candidate-1"}\n')
        with (
            patch(
                "bio_neb.cli.advance",
                side_effect=(pending, resumed),
            ) as advance,
            patch("sys.stdin", stdin),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main(
                [
                    "begin",
                    "--run-root",
                    "/runs",
                    "--prompt-file",
                    "/inputs/request.txt",
                    "--runtime-config",
                    "/inputs/runtime.json",
                    "--fastq",
                    "read_1=/inputs/read-1.fastq.gz",
                    "--interactive",
                ]
            )

        rendered = [json.loads(line) for line in stdout.getvalue().splitlines()]
        answered = advance.call_args_list[1].args[0]
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            tuple(item["frontier"] for item in rendered),
            ("review_required", "terminal"),
        )
        self.assertEqual(answered.request_event_id, request_id)
        self.assertEqual(answered.answer.kind, "planning")
        self.assertEqual(answered.answer.response.action, "select")
        self.assertIn("exact legal Review response JSON", stderr.getvalue())

    def test_observe_offers_concise_progress_and_existing_rich_view(self):
        observation = _cli_observation(status="inputs_registered")
        concise = io.StringIO()
        rich = io.StringIO()
        with patch("bio_neb.cli.observe", return_value=observation) as observe:
            with redirect_stdout(concise):
                self.assertEqual(
                    main(["observe", "--journey-file", observation.journey_file]), 0
                )
            with redirect_stdout(rich):
                self.assertEqual(
                    main(
                        [
                            "observe",
                            "--journey-file",
                            observation.journey_file,
                            "--rich",
                        ]
                    ),
                    0,
                )

        self.assertEqual(observe.call_count, 2)
        rendered = json.loads(concise.getvalue())
        self.assertEqual(rendered["view"], "concise")
        contract = RunInteraction.model_validate(rendered)
        self.assertEqual(contract.model_dump(mode="json"), rendered)
        with self.assertRaises(ValidationError):
            RunInteraction.model_validate({**rendered, "unexpected": True})
        with self.assertRaises(ValidationError):
            RunInteraction.model_validate({**rendered, "frontier": "unknown"})
        with self.assertRaises(ValidationError):
            RunInteraction.model_validate({**rendered, "status": "unknown"})
        with self.assertRaises(ValidationError):
            contract.frontier = "progress"
        self.assertEqual(
            json.loads(rich.getvalue()), observation.model_dump(mode="json")
        )

    def test_all_review_kinds_present_their_legal_researcher_choices(self):
        request_id = "00000000-0000-0000-0000-000000000002"
        cases = (
            (
                "planning",
                "planning_hitl_pending",
                _planning_request(
                    request_id,
                    actions=("confirm", "select", "clarify", "stop"),
                    question="Choose one supported planning response.",
                ),
            ),
            (
                "model_availability",
                "model_availability_review_pending",
                _model_availability_request(request_id),
            ),
        )
        for kind, status, request in cases:
            with self.subTest(kind=kind):
                observation = _cli_observation(status=status, pending=(kind, request))
                output = io.StringIO()
                with (
                    patch(
                        "bio_neb.cli.observe",
                        return_value=observation,
                    ),
                    redirect_stdout(output),
                ):
                    self.assertEqual(
                        main(
                            [
                                "observe",
                                "--journey-file",
                                observation.journey_file,
                            ]
                        ),
                        0,
                    )
                rendered = json.loads(output.getvalue())["review"]
                self.assertEqual(
                    rendered, observation.current_review.model_dump(mode="json")
                )
                self.assertEqual(
                    RunInteraction.model_validate_json(output.getvalue()).review,
                    observation.current_review,
                )
                self.assertEqual(rendered["kind"], kind)
                self.assertTrue(rendered["request"]["legal_actions"])
                if kind == "model_availability":
                    legal = rendered["request"]["legal_actions"]
                    # Retry, amend the instruction, or stop with a reason.
                    self.assertEqual(
                        tuple(item["action"] for item in legal),
                        ("retry", "amend", "stop"),
                    )
                    self.assertIn(
                        "instruction", legal[1]["response_schema"]["required"]
                    )
                    self.assertIn("reason", legal[2]["response_schema"]["required"])

        execution_request = _execution_request(
            _execution_proposal(), request_event_id=request_id
        )
        execution = _cli_view(
            requests=(execution_request,),
            receipts=(),
            pending_request=execution_request,
            authorities=(),
        )
        observation = _cli_observation(
            status="execution_hitl_pending", execution=execution
        )
        output = io.StringIO()
        with (
            patch("bio_neb.cli.observe", return_value=observation),
            redirect_stdout(output),
        ):
            self.assertEqual(
                main(["observe", "--journey-file", observation.journey_file]), 0
            )
        rendered = json.loads(output.getvalue())["review"]
        self.assertEqual(rendered["kind"], "execution")
        self.assertEqual(
            tuple(item["action"] for item in rendered["request"]["legal_actions"]),
            ("approve", "revise", "reject"),
        )

    def test_concise_progress_renders_the_attempt_evidence_summary(self):
        # A renderer test: the observation is hand-built and observe is patched,
        # so the projection-to-render handoff is not claimed here.
        model_event = JourneyEvidenceItem(
            source_event_id="00000000-0000-0000-0000-000000000011",
            event_type="ModelResponseRecorded",
            parent_event_ids=(),
            payload={"model": "selected/model"},
        )
        tool_event = JourneyEvidenceItem(
            source_event_id="00000000-0000-0000-0000-000000000012",
            event_type="ToolCallSucceeded",
            parent_event_ids=(model_event.source_event_id,),
            payload={
                "tool_name": "lookup_reference_metadata",
                "tool_call_id": "reference-1",
            },
        )
        lifecycle_event = JourneyEvidenceItem(
            source_event_id="00000000-0000-0000-0000-000000000013",
            event_type="ActivitySucceeded",
            parent_event_ids=(tool_event.source_event_id,),
            payload={"stage": "planning_attempt"},
        )
        instructions = role_instructions("planning_agent")
        attempt = AttemptEvidenceReview(
            attempt_id="00000000-0000-0000-0000-000000000014",
            agent_role="planning_agent",
            instruction_identity=instructions.identity,
            instruction_artifact=ArtifactReference(
                artifact_id=f"sha256:{hashlib.sha256(instructions.content).hexdigest()}",
                byte_count=len(instructions.content),
                journey_id="00000000-0000-0000-0000-000000000010",
                media_type=INSTRUCTION_MEDIA_TYPE,
                source_event_id="00000000-0000-0000-0000-000000000015",
            ),
            outcome="succeeded",
            events=(model_event, tool_event, lifecycle_event),
            model_events=(model_event,),
            tool_events=(tool_event,),
            lifecycle_events=(lifecycle_event,),
            validation_events=(),
            checkpoint_events=(),
        )
        observation = _cli_observation(status="reference_identity")
        observation.attempts = (attempt,)
        output = io.StringIO()
        with (
            patch("bio_neb.cli.observe", return_value=observation),
            redirect_stdout(output),
        ):
            self.assertEqual(
                main(["observe", "--journey-file", observation.journey_file]), 0
            )

        progress = json.loads(output.getvalue())["progress"]
        self.assertEqual(
            progress["activities"],
            [
                {
                    "attempt_id": attempt.attempt_id,
                    "agent_role": "planning_agent",
                    "outcome": "succeeded",
                    "model_evidence_count": 1,
                    "tool_evidence_count": 1,
                    "lifecycle_evidence_count": 1,
                }
            ],
        )
        self.assertEqual(
            progress["model_evidence"],
            [
                {
                    "source_event_id": model_event.source_event_id,
                    "event_type": "ModelResponseRecorded",
                }
            ],
        )
        self.assertEqual(
            progress["tool_evidence"],
            [
                {
                    "source_event_id": tool_event.source_event_id,
                    "event_type": "ToolCallSucceeded",
                    "tool_name": "lookup_reference_metadata",
                    "tool_call_id": "reference-1",
                }
            ],
        )
        self.assertEqual(
            progress["lifecycle_evidence"],
            [
                {
                    "source_event_id": lifecycle_event.source_event_id,
                    "event_type": "ActivitySucceeded",
                    "stage": "planning_attempt",
                    "activity_id": None,
                }
            ],
        )

    def test_concise_view_types_execution_authority_as_its_own_frontier(self):
        proposal = _execution_proposal()
        request_event_id = "00000000-0000-0000-0000-000000000015"
        request = _execution_request(
            proposal, request_event_id=request_event_id, pending=False
        )
        authority = _execution_authority(
            proposal,
            request_event_id=request_event_id,
            approval_event_id="00000000-0000-0000-0000-000000000016",
        )
        observation = _cli_observation(
            status="execution_authorized",
            execution=_cli_view(
                requests=(request,),
                receipts=(),
                pending_request=None,
                authorities=(authority,),
            ),
        )
        output = io.StringIO()
        with (
            patch("bio_neb.cli.observe", return_value=observation),
            redirect_stdout(output),
        ):
            self.assertEqual(
                main(["observe", "--journey-file", observation.journey_file]), 0
            )

        rendered = json.loads(output.getvalue())
        contract = run_interaction(observation)
        self.assertIsInstance(contract.execution_proposal, ExecutionProposal)
        self.assertIsInstance(contract.execution_authority[0], ExecutionAuthority)
        self.assertEqual(rendered["frontier"], "execution_authority")
        self.assertEqual(
            rendered["execution_proposal"]["proposal_id"], proposal.proposal_id
        )
        self.assertEqual(
            rendered["execution_authority"][0]["proposal_id"], proposal.proposal_id
        )

    def test_execution_review_presents_separate_proposal_and_scientific_handoff(self):
        request_id = "00000000-0000-0000-0000-000000000003"
        proposal = _execution_proposal()
        request = _execution_request(proposal, request_event_id=request_id)
        execution = _cli_view(
            requests=(request,),
            receipts=(),
            pending_request=request,
            authorities=(),
        )
        observation = _cli_observation(
            status="execution_hitl_pending", execution=execution
        )
        handoff = {
            "journey_id": observation.journey_id,
            "plan_event_id": "00000000-0000-0000-0000-000000000004",
            "items": [
                {
                    "requirement_id": "known-variant:chr7",
                    "subject_key": "chr7_140453136_a_t",
                    "purpose": "known_variant",
                    "question_or_assertion": "chr7:140453136 A>T was supplied.",
                    "state": "open",
                    "lineage_event_ids": ["00000000-0000-0000-0000-000000000005"],
                },
                {
                    "requirement_id": "paper:pmid-123",
                    "subject_key": "paper_pmid_123",
                    "purpose": "literature_claim",
                    "question_or_assertion": "PMID:123 reports a relevant finding.",
                    "state": "grounded",
                    "lineage_event_ids": ["00000000-0000-0000-0000-000000000005"],
                },
                {
                    "requirement_id": "hypothesis:response",
                    "subject_key": "treatment_response_hypothesis",
                    "purpose": "hypothesis",
                    "question_or_assertion": (
                        "The variant may affect treatment response."
                    ),
                    "state": "open",
                    "lineage_event_ids": ["00000000-0000-0000-0000-000000000005"],
                },
                {
                    "requirement_id": "question:novel-variants",
                    "subject_key": "novel_variant_significance",
                    "purpose": "open_question",
                    "question_or_assertion": (
                        "Which discovered variants are significant?"
                    ),
                    "state": "open",
                    "lineage_event_ids": ["00000000-0000-0000-0000-000000000005"],
                },
                {
                    "requirement_id": "discovery:variants",
                    "subject_key": "supplied_or_discovered_variants",
                    "purpose": "discovery_request",
                    "question_or_assertion": (
                        "Investigate supplied and discovered variants."
                    ),
                    "state": "open",
                    "lineage_event_ids": ["00000000-0000-0000-0000-000000000005"],
                },
                {
                    "requirement_id": "finding:limits",
                    "subject_key": "findings_and_limitations",
                    "purpose": "other",
                    "question_or_assertion": (
                        "Retain grounded findings and assay limitations."
                    ),
                    "state": "grounded",
                    "lineage_event_ids": ["00000000-0000-0000-0000-000000000005"],
                },
            ],
        }
        handoff_model = _scientific_investigation_handoff(
            journey_id=handoff["journey_id"],
            plan_event_id=handoff["plan_event_id"],
            items=handoff["items"],
        )
        observation.requirement_resolution = _cli_view(
            requirements=(),
            scientific_investigation=handoff_model,
        )
        output = io.StringIO()
        with (
            patch("bio_neb.cli.observe", return_value=observation),
            redirect_stdout(output),
        ):
            self.assertEqual(
                main(["observe", "--journey-file", observation.journey_file]), 0
            )

        rendered = json.loads(output.getvalue())
        contract = run_interaction(observation)
        self.assertIsInstance(
            contract.scientific_investigation_handoff.handoff,
            ScientificInvestigationHandoff,
        )
        self.assertEqual(
            rendered["execution_proposal"], proposal.model_dump(mode="json")
        )
        presentation = rendered["scientific_investigation_handoff"]
        self.assertEqual(presentation["handoff"], handoff_model.model_dump(mode="json"))
        self.assertEqual(
            presentation["research_intent"],
            {
                "prompt": observation.request.prompt,
                "prompt_file": observation.request.prompt_file,
                "source_event_id": observation.request.source_event_id,
            },
        )
        self.assertEqual(
            {item["purpose"] for item in presentation["handoff"]["items"]},
            {
                "known_variant",
                "literature_claim",
                "hypothesis",
                "open_question",
                "discovery_request",
                "other",
            },
        )
        self.assertIn(
            "chr7:140453136 A>T",
            presentation["handoff"]["items"][0]["question_or_assertion"],
        )
        self.assertNotEqual(
            rendered["execution_proposal"],
            rendered["scientific_investigation_handoff"],
        )

    def test_observer_driven_refusal_renders_its_typed_hold(self):
        from bio_neb.application import RunAdvancementRefused
        from bio_neb.failure import HeldFailure

        failure = HeldFailure(
            failure_class="invariant",
            invariant_id="coordinator_status_unhandled",
            cause="reference_facts",
            evidence={"status": "reference_facts"},
        )
        observation = _cli_observation(status="run_held")
        error = RunAdvancementRefused(failure, observation=observation)
        output = io.StringIO()
        with (
            patch("bio_neb.cli.advance", side_effect=error),
            redirect_stderr(output),
        ):
            exit_code = main(["resume", "--journey-file", "/runs/run-1/journey.jsonl"])

        rendered = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(rendered["frontier"], "refused")
        self.assertEqual(rendered["code"], "coordinator_status_unhandled")
        self.assertEqual(rendered["disposition"]["kind"], "hold_for_fix")
        self.assertEqual(rendered["status"], "run_held")
        self.assertEqual(
            rendered["failure"],
            {
                "failure_class": "invariant",
                "invariant_id": "coordinator_status_unhandled",
                "cause": "reference_facts",
                "evidence": {"status": "reference_facts"},
            },
        )
        self.assertEqual(rendered["journey_id"], observation.journey_id)
        self.assertEqual(rendered["journey_file"], observation.journey_file)
        contract = _PublicRefusalView.model_validate(rendered)
        self.assertEqual(contract.model_dump(mode="json"), rendered)
        with self.assertRaises(ValidationError):
            _PublicRefusalView.model_validate({**rendered, "unexpected": True})
        with self.assertRaises(ValidationError):
            _PublicRefusalView.model_validate({**rendered, "frontier": "progress"})
        with self.assertRaises(ValidationError):
            _PublicRefusalView.model_validate({**rendered, "status": "unknown"})
        with self.assertRaises(ValidationError):
            contract.frontier = "progress"

        unavailable = RunAdvancementRefused(failure)
        output = io.StringIO()
        with (
            patch("bio_neb.cli.advance", side_effect=unavailable),
            redirect_stderr(output),
        ):
            self.assertEqual(
                main(["resume", "--journey-file", observation.journey_file]), 2
            )
        rendered_without_locator = json.loads(output.getvalue())
        self.assertNotIn("journey_id", rendered_without_locator)
        self.assertNotIn("journey_file", rendered_without_locator)
        self.assertNotIn("status", rendered_without_locator)

    def test_an_untyped_fault_under_resume_reports_its_hold_not_an_invalid_journey(
        self,
    ):
        from tests.application_fakes import _registered

        with tempfile.TemporaryDirectory() as directory:
            receipt, _paths = _registered(Path(directory).resolve())

            def failing(command, journey):
                raise ValueError("terminal source belongs to another Attempt")

            output = io.StringIO()
            with (
                patch(
                    "bio_neb.input_inspection.commands.inspect_registered_inputs",
                    failing,
                ),
                redirect_stderr(output),
            ):
                exit_code = main(["resume", "--journey-file", receipt.journey_file])
            rendered = json.loads(output.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertEqual(
                (
                    rendered["frontier"],
                    rendered["code"],
                    rendered["disposition"]["kind"],
                    rendered["status"],
                ),
                ("refused", "coordinator_unhandled_error", "hold_for_fix", "run_held"),
            )
            self.assertTrue(rendered["failure"]["cause"].startswith("ValueError@"))

    def test_a_missing_model_credential_holds_the_run_on_each_resume(self):
        # Each planning dispatch records one configuration failure and holds;
        # observe reads the held Run, never a failed one.
        import os

        from tests.application_fakes import _registered

        with tempfile.TemporaryDirectory() as directory:
            receipt, _paths = _registered(Path(directory).resolve())
            journey_file = receipt.journey_file
            refusals = []
            with patch.dict(os.environ):
                os.environ.pop("LOCAL_MODEL_KEY", None)
                for _ in range(2):
                    output = io.StringIO()
                    with redirect_stderr(output):
                        exit_code = main(["resume", "--journey-file", journey_file])
                    refusals.append((exit_code, json.loads(output.getvalue())))
                observed = io.StringIO()
                with redirect_stdout(observed):
                    observe_code = main(["observe", "--journey-file", journey_file])
            events = Journey.open(Path(journey_file)).read()
        self.assertEqual(
            [
                (
                    code,
                    rendered["code"],
                    rendered["disposition"]["kind"],
                    rendered["status"],
                    rendered["failure"]["invariant_id"],
                )
                for code, rendered in refusals
            ],
            [
                (
                    2,
                    "model_credential_unavailable",
                    "hold_for_fix",
                    "run_held",
                    "model_credential_unavailable",
                )
            ]
            * 2,
        )
        self.assertEqual(
            (observe_code, json.loads(observed.getvalue())["status"]), (0, "run_held")
        )
        types = [e["event_type"] for e in events]
        self.assertEqual(types.count("RuntimeConfigurationFailed"), 2)
        self.assertNotIn("RunFailed", types)

    def test_public_commands_accept_no_stage_selector_arguments(self):
        # The exact command and export sets are
        # test_repository_contract's test_public_commands_and_application_exports_are_exactly_current.
        parser = _parser()
        cases = (
            [
                "begin",
                "--run-root",
                "/runs",
                "--prompt-file",
                "/request.txt",
                "--runtime-config",
                "/runtime.json",
                "--fastq",
                "read_1=/read-1.fastq.gz",
            ],
            ["resume", "--journey-file", "/runs/run/journey.jsonl"],
            [
                "answer",
                "--journey-file",
                "/runs/run/journey.jsonl",
                "--request-event-id",
                "00000000-0000-0000-0000-000000000002",
                "--response-json",
                '{"action":"retry"}',
            ],
            ["observe", "--journey-file", "/runs/run/journey.jsonl"],
        )
        for argv in cases:
            with self.subTest(command=argv[0]):
                arguments = vars(parser.parse_args(argv))
                self.assertTrue(
                    {"stage", "agent_role", "tool", "capability_snapshot"}.isdisjoint(
                        arguments
                    )
                )

    def test_public_scientific_continuation_runs_real_contact_free_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            command_file = str(prerequisite("executed").journey_file)
            journey_file = Path(command_file)
            target = root / "scientific-continuation.json"
            journey_before = journey_file.read_bytes()
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "scientific-continuation",
                        "--execution-journey-file",
                        command_file,
                        "--output",
                        str(target),
                    ]
                )
            self.assertEqual(exit_code, 0)
            packet = ScientificContinuationPacket.model_validate_json(
                target.read_bytes(), strict=True
            )
            receipt = json.loads(stdout.getvalue())
            self.assertEqual(receipt["continuation_id"], packet.continuation_id)
            self.assertEqual(receipt["byte_count"], len(target.read_bytes()))
            self.assertFalse(target.read_bytes().endswith(b"\n"))
            self.assertEqual(journey_file.read_bytes(), journey_before)

    def test_public_scientific_continuation_maps_typed_compiler_refusal(self):
        stderr = io.StringIO()
        with (
            patch(
                "bio_neb.cli.publish_scientific_continuation",
                side_effect=ScientificContinuationRefused("frontier_not_terminal"),
            ),
            redirect_stderr(stderr),
        ):
            exit_code = main(
                [
                    "scientific-continuation",
                    "--execution-journey-file",
                    "/runs/execution-command.json",
                    "--output",
                    "/runs/scientific-continuation.json",
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertEqual(json.loads(stderr.getvalue())["code"], "frontier_not_terminal")

    def test_answer_accepts_typed_pretty_utf8_response_file(self):
        request_id = "e59c9ebf-2f60-4a61-9f72-bd1478e22d83"
        request = _planning_request(
            request_id,
            actions=("clarify", "stop"),
            question="Which exact supported output should Planning use?",
        )
        observation = _cli_observation(
            status="planning_hitl_pending", pending=("planning", request)
        )
        with tempfile.TemporaryDirectory() as temporary:
            response_file = Path(temporary).resolve() / "réponse.json"
            response_file.write_text(
                json.dumps(
                    {
                        "action": "clarify",
                        "instruction": "Préférer un VCF exact — données conservées.",
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with (
                patch(
                    "bio_neb.application.advancement.Journey.open",
                    return_value=object(),
                ),
                patch(
                    "bio_neb.application.advancement.observe",
                    return_value=observation,
                ),
                patch(
                    "bio_neb.application.advancement.advance",
                    return_value=observation,
                ) as application,
                redirect_stdout(output),
            ):
                exit_code = main(
                    [
                        "answer",
                        "--journey-file",
                        observation.journey_file,
                        "--request-event-id",
                        request_id,
                        "--response-file",
                        str(response_file),
                    ]
                )

        self.assertEqual(exit_code, 0)
        intent = application.call_args.args[0]
        self.assertEqual(intent.answer.kind, "planning")
        self.assertEqual(intent.answer.response.action, "clarify")
        self.assertIn("Préférer", intent.answer.response.instruction)
        self.assertEqual(json.loads(output.getvalue())["frontier"], "review_required")

    def test_an_unreadable_journey_keeps_each_answering_commands_refusal(self):
        # Answer and report-explain read and act on one owner; a file that
        # opens no Journey still refuses as each command did before.
        with tempfile.TemporaryDirectory() as temporary:
            journey = Path(temporary).resolve() / "journey.jsonl"
            journey.write_text("not-json\n", encoding="utf-8")
            cases = {
                "answer": (
                    ["answer", "--request-event-id", str(uuid.uuid4())]
                    + ["--response-json", '{"action":"approve"}'],
                    "journey_review_invalid",
                    ["journey_file", "cause:JSONDecodeError@"],
                ),
                "report-explain": (
                    ["report-explain", "--report-digest", "grounded-report:" + "0" * 64]
                    + ["--requested-by", "researcher:owner"],
                    "report_explanation_unverifiable",
                    ["journey_file"],
                ),
            }
            for command, (argv, code, fields) in cases.items():
                with (
                    self.subTest(command=command),
                    redirect_stdout(io.StringIO()) as output,
                    redirect_stderr(io.StringIO()) as error,
                ):
                    exit_code = main([*argv, "--journey-file", str(journey)])
                refusal = json.loads(error.getvalue().strip().splitlines()[-1])
                self.assertEqual(exit_code, 2)
                self.assertEqual(output.getvalue(), "")
                self.assertEqual(refusal["code"], code)
                self.assertEqual(len(refusal["fields"]), len(fields))
                for field, expected in zip(refusal["fields"], fields, strict=True):
                    self.assertTrue(field.startswith(expected), (field, expected))
                self.assertEqual(journey.read_text(encoding="utf-8"), "not-json\n")

    def test_runner_import_keeps_nebius_model_endpoint_lazy_in_fresh_process(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import bio_neb.cli; "
                    "assert 'bio_neb.providers.nebius_model_endpoint' "
                    "not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_observer_driven_nouns_and_invalid_framing_cross_process(self):
        help_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "bio_neb.cli",
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        for noun in ("begin", "resume", "answer", "observe"):
            with self.subTest(noun=noun):
                self.assertIn(noun, help_result.stdout)

        invalid = subprocess.run(
            [
                sys.executable,
                "-m",
                "bio_neb.cli",
                "observe",
                "--journey-file",
                "",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(invalid.stdout, "")
        self.assertEqual(
            json.loads(invalid.stderr),
            {
                "status": "failed",
                "code": "command_invalid",
                "fields": ["journey_file", "reason:journey_file_empty"],
            },
        )

    def test_relative_journey_paths_resolve_at_the_cli_boundary(self):
        """The application only ever receives the canonical absolute Journey path."""

        class Received(Exception):
            pass

        cases = {
            "observe": ("bio_neb.cli.observe", ["observe", "--journey-file"]),
            "resume": ("bio_neb.cli.advance", ["resume", "--journey-file"]),
            "answer": (
                "bio_neb.cli.answer_from_json",
                ["answer", "--request-event-id", "request", "--response-json"]
                + ["{}", "--journey-file"],
            ),
            "parent-identity": (
                "bio_neb.cli.read_parent_journey_identity",
                ["parent-identity", "--journey-file"],
            ),
            "execution-contact-review": (
                "bio_neb.cli.review_execution_contacts",
                ["execution-contact-review", "--execution-journey-file"],
            ),
            "fork": (
                "bio_neb.cli.fork_run",
                ["fork", "--child-run-root", "/runs/child", "--parent-journey-file"],
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            expected = str(root / "runs" / "journey.jsonl")
            for command, (target, argv) in cases.items():
                with (
                    self.subTest(command=command),
                    patch("os.getcwd", return_value=str(root / "work")),
                    patch(target, side_effect=Received) as entry,
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(Received):
                        main([*argv, "../runs/./journey.jsonl"])
                    received = entry.call_args.args[0]
                    self.assertEqual(
                        str(
                            getattr(
                                received,
                                "journey_file",
                                getattr(received, "parent_journey_file", received),
                            )
                        ),
                        expected,
                    )

    def test_native_controller_preflight_retains_exact_pin_and_filesystem_proof(self):
        identity = SimpleNamespace(
            executable=ACCEPTED_NEBIUS_EXECUTABLE,
            resolved_executable=ACCEPTED_NEBIUS_EXECUTABLE,
            version="0.12.263",
            byte_count=ACCEPTED_NEBIUS_EXECUTABLE_BYTE_COUNT,
            sha256=ACCEPTED_NEBIUS_EXECUTABLE_SHA256,
        )
        preflight = Mock(return_value=identity)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt = root / "native-controller-preflight.json"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "nebius-controller-preflight",
                            "--nebius-executable",
                            ACCEPTED_NEBIUS_EXECUTABLE,
                            "--run-root",
                            str(root),
                            "--receipt-file",
                            str(receipt),
                        ],
                        nebius_runner_factory=lambda _: SimpleNamespace(
                            preflight=preflight
                        ),
                    ),
                    0,
                )
            rendered = json.loads(output.getvalue())
            self.assertEqual(rendered["controller_runtime"], "native_ubuntu")
            self.assertEqual(
                rendered["executable_sha256"], ACCEPTED_NEBIUS_EXECUTABLE_SHA256
            )
            self.assertTrue(rendered["hard_link_no_overwrite_verified"])
            self.assertTrue(rendered["file_fsync_verified"])
            self.assertTrue(rendered["directory_fsync_verified"])
            self.assertEqual(rendered["provider_contact_count"], 0)
            self.assertEqual(
                receipt.read_text(encoding="utf-8"), output.getvalue().strip()
            )
            self.assertEqual(list(root.glob(".bio-neb-native-filesystem-*")), [])
            preflight.assert_called_once_with()

            stderr = io.StringIO()
            later = Mock()
            with redirect_stderr(stderr):
                self.assertEqual(
                    main(
                        [
                            "nebius-controller-preflight",
                            "--nebius-executable",
                            ACCEPTED_NEBIUS_EXECUTABLE,
                            "--run-root",
                            str(root),
                            "--receipt-file",
                            str(receipt),
                        ],
                        nebius_runner_factory=later,
                    ),
                    2,
                )
            later.assert_not_called()
            self.assertEqual(
                json.loads(stderr.getvalue())["code"],
                "native_controller_preflight_paths_invalid",
            )

    def test_nebius_profile_listing_parses_plain_local_lines_without_scope_defaults(
        self,
    ):
        runner = Mock(
            return_value=NebiusCliResponse(stdout="research\nproduction [default]\n")
        )
        output = io.StringIO()
        with redirect_stdout(output):
            identity = SimpleNamespace(
                executable=ACCEPTED_NEBIUS_EXECUTABLE,
                resolved_executable=ACCEPTED_NEBIUS_EXECUTABLE,
                version="0.12.263",
                byte_count=ACCEPTED_NEBIUS_EXECUTABLE_BYTE_COUNT,
                sha256=ACCEPTED_NEBIUS_EXECUTABLE_SHA256,
            )
            self.assertEqual(
                main(
                    [
                        "nebius-profiles",
                        "--nebius-executable",
                        ACCEPTED_NEBIUS_EXECUTABLE,
                    ],
                    nebius_profile_runner=runner,
                    nebius_runner_factory=lambda _: SimpleNamespace(
                        preflight=lambda: identity
                    ),
                ),
                0,
            )
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "profiles": ["research", "production"],
                "tenant_project_selection": "explicit_manual_entry_required",
            },
        )
        runner.assert_called_once_with(
            "profile",
            "list",
            "--no-check-update",
            "--no-progress",
            "--retries",
            "1",
        )

    def test_answer_requires_exactly_one_response_source(self):
        base = [
            "answer",
            "--journey-file",
            "/runs/journey.jsonl",
            "--request-event-id",
            "e59c9ebf-2f60-4a61-9f72-bd1478e22d83",
        ]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(base)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(
                [
                    *base,
                    "--response-json",
                    '{"action":"confirm"}',
                    "--response-file",
                    "/runs/response.json",
                ]
            )

    def test_answer_response_file_failures_precede_advancement(self):
        request_id = "e59c9ebf-2f60-4a61-9f72-bd1478e22d83"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            invalid_json = root / "invalid.json"
            invalid_json.write_text("{not-json", encoding="utf-8")
            invalid_utf8 = root / "invalid-utf8.json"
            invalid_utf8.write_bytes(b"\xff")
            cases = {
                "invalid_json": str(invalid_json),
                "invalid_utf8": str(invalid_utf8),
                "unreadable": str(root / "missing.json"),
                "relative": "response.json",
                "noncanonical": str(root / "nested" / ".." / "response.json"),
            }
            observation = _cli_observation(
                status="planning_hitl_pending",
                pending=(
                    "planning",
                    _planning_request(
                        request_id,
                        actions=("confirm", "stop"),
                        question="Confirm the exact execution output?",
                    ),
                ),
            )
            for case, response_file in cases.items():
                with self.subTest(case=case):
                    errors = io.StringIO()
                    with (
                        patch(
                            "bio_neb.cli.observe",
                            return_value=observation,
                        ),
                        patch("bio_neb.cli.advance") as application,
                        redirect_stderr(errors),
                    ):
                        exit_code = main(
                            [
                                "answer",
                                "--journey-file",
                                observation.journey_file,
                                "--request-event-id",
                                request_id,
                                "--response-file",
                                response_file,
                            ]
                        )
                    self.assertEqual(exit_code, 2)
                    self.assertEqual(
                        json.loads(errors.getvalue())["code"], "command_invalid"
                    )
                    application.assert_not_called()

    def test_real_narrow_console_stream_is_reconfigured_for_unicode_observation(self):
        observation = _cli_observation()
        rich = observation.model_dump()
        observation.model_dump = lambda **_: {
            **rich,
            "request": {"prompt": "inspect 🧬"},
        }
        output_bytes = io.BytesIO()
        error_bytes = io.BytesIO()
        narrow_stdout = io.TextIOWrapper(output_bytes, encoding="cp1252")
        narrow_stderr = io.TextIOWrapper(error_bytes, encoding="cp1252")
        with (
            patch.object(sys, "stdout", narrow_stdout),
            patch.object(sys, "stderr", narrow_stderr),
            patch(
                "bio_neb.cli.observe",
                return_value=observation,
            ),
        ):
            self.assertEqual(
                main(
                    [
                        "observe",
                        "--journey-file",
                        observation.journey_file,
                        "--rich",
                    ]
                ),
                0,
            )
            narrow_stdout.flush()
        self.assertEqual(narrow_stdout.encoding, "utf-8")
        self.assertIn("🧬", output_bytes.getvalue().decode("utf-8"))

    def test_removed_identity_and_prepared_reference_flags_are_unknown(self):
        # Continuations name files; the application reads their identities.
        cases = {
            "--prepared-reference": [
                "begin",
                "--run-root",
                "/runs",
                "--prompt-file",
                "/inputs/request.txt",
                "--runtime-config",
                "/inputs/runtime.json",
                "--fastq",
                "read_1=/inputs/read-1.fastq.gz",
            ],
        }
        # The Fork names files and an optional point; it reads every identity.
        fork = [
            "fork",
            "--parent-journey-file",
            "/runs/parent/journey.jsonl",
            "--child-run-root",
            "/runs/child",
        ]
        for flag in (
            "--expected-parent-journey-sha256",
            "--expected-parent-head-event-id",
            "--output",
            "--scientific-continuation-file",
            "--committed-candidate-sha",
        ):
            cases[flag] = fork
        for flag, argv in cases.items():
            with self.subTest(flag=flag):
                error = io.StringIO()
                with (
                    patch("bio_neb.cli.advance") as advance,
                    redirect_stderr(error),
                    self.assertRaises(SystemExit) as exited,
                ):
                    main([*argv, flag, "value"])
                self.assertEqual(exited.exception.code, 2)
                self.assertIn(f"unrecognized arguments: {flag} value", error.getvalue())
                advance.assert_not_called()

    def _policy_amend(self, root: Path, *extra: str) -> tuple[int, dict]:
        amendment = root / "amendment.json"
        amendment.write_text(
            json.dumps(
                {
                    "proposal_id": "investigation-proposal:selected",
                    "expected_head_event_id": "head",
                    "previous_policy_id": None,
                    "asserted_by": "researcher:owner",
                    "reason": "hold one member",
                    "hold_work_item_ids": ["work-item"],
                }
            ),
            encoding="utf-8",
        )
        error = io.StringIO()
        with redirect_stderr(error):
            exit_code = main(
                [
                    "investigation-policy-amend",
                    "--journey-file",
                    str(root / "journey.jsonl"),
                    "--amendment-file",
                    str(amendment),
                    *extra,
                ]
            )
        return exit_code, json.loads(error.getvalue())

    def test_typed_refusal_keeps_its_code_and_names_no_journey_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            exit_code, failure = self._policy_amend(
                root, "--runtime-config", str(root / "absent-runtime.json")
            )
        self.assertEqual(exit_code, 2)
        self.assertEqual(failure["code"], "runtime_config_invalid")
        self.assertNotIn("journey_file", failure["fields"])

    def test_untyped_failure_of_a_multi_file_command_names_the_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch(
                "bio_neb.cli.amend_investigation_policy",
                side_effect=ValueError("unclassified"),
            ):
                exit_code, failure = self._policy_amend(root)
        self.assertEqual(exit_code, 2)
        self.assertEqual(failure["code"], "command_failed")
        self.assertEqual(failure["fields"][0], "investigation-policy-amend")
        self.assertTrue(failure["fields"][1].startswith("cause:ValueError@"))
        self.assertNotIn("journey_file", failure["fields"])

    def test_invalid_begin_returns_stable_machine_readable_failure(self):
        output = io.StringIO()
        with redirect_stderr(output):
            exit_code = main(
                [
                    "begin",
                    "--run-root",
                    "relative",
                    "--prompt-file",
                    "relative",
                    "--runtime-config",
                    "relative",
                    "--fastq",
                    "READ=relative",
                ]
            )
        failure = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["code"], "command_invalid")

    def test_begin_passes_the_researcher_snp_request_file_to_begin_run(self):
        observation = object()
        with (
            patch("bio_neb.cli.advance", return_value=observation) as advance,
            patch("bio_neb.cli._emit_run_view") as emitted,
            patch("bio_neb.cli._emit_readable_report"),
        ):
            exit_code = main(
                [
                    "begin",
                    "--run-root",
                    "/runs",
                    "--prompt-file",
                    "/inputs/request.txt",
                    "--runtime-config",
                    "/inputs/runtime.json",
                    "--request-file",
                    "/inputs/snp-request.json",
                    "--fastq",
                    "read_1=/inputs/read-1.fastq.gz",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            advance.call_args.args[0].request_files, ("/inputs/snp-request.json",)
        )
        emitted.assert_called_once_with(observation)

    def test_begin_with_an_invalid_snp_request_creates_no_journey(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name, content in (
                ("prompt.txt", b"Analyze rs429358."),
                ("r1.fastq", b"@r\nACGT\n+\n!!!!\n"),
                ("snp-request.json", b'{"asserted_by": "researcher:owner"}'),
            ):
                (root / name).write_bytes(content)
            (root / "runtime.json").write_text(
                json.dumps(
                    {
                        "contract_version": "1.0",
                        "heartbeat_interval_seconds": 30,
                        "model_roster": {
                            "qwen": {
                                "provider": "nebius",
                                "endpoint": "https://api.example.test",
                                "model": "Qwen/Qwen3.5-397B-A17B",
                                "credential_env": "NEBIUS_TOKEN",
                            }
                        },
                        "agent_invocation": {"model_reference": "qwen"},
                    }
                ),
                encoding="utf-8",
            )
            errors = io.StringIO()
            # Begin checks the credential first; it is set so the request
            # itself is what refuses.
            with (
                patch.dict(os.environ, {"NEBIUS_TOKEN": "offline-test-value"}),
                redirect_stderr(errors),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = main(
                    [
                        "begin",
                        "--run-root",
                        str(root / "runs"),
                        "--prompt-file",
                        str(root / "prompt.txt"),
                        "--runtime-config",
                        str(root / "runtime.json"),
                        "--request-file",
                        str(root / "snp-request.json"),
                        "--fastq",
                        "read_1=" + str(root / "r1.fastq"),
                    ],
                    parabricks_infrastructure_factory=lambda *_: self.fail(
                        "invalid registration constructed execution"
                    ),
                )
            run_root_exists = (root / "runs").exists()

        self.assertEqual(exit_code, 2)
        self.assertEqual(
            json.loads(errors.getvalue()),
            {
                "status": "failed",
                "code": "investigation_request_invalid",
                "fields": ["request_file"],
            },
        )
        self.assertFalse(run_root_exists)

    def test_begin_refuses_an_unsupported_fastq_role_set_before_any_journey(self):
        # Execution runs one forward/reverse pair, so begin refuses any other role
        # set before a Journey or model exists (single-end, other names, two lanes).
        cases = {
            "single_end": ("reads",),
            "lane_named_pair": ("lane1_r1", "lane1_r2"),
            "two_lanes": ("read_1", "read_2", "lane2_read_1", "lane2_read_2"),
            "half_pair": ("read_1",),
            "mixed_pair": ("read_1", "reverse_read"),
        }
        for name, roles in cases.items():
            with (
                self.subTest(case=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                (root / "prompt.txt").write_bytes(b"Analyze rs429358.")
                (root / "runtime.json").write_text(
                    json.dumps(
                        {
                            "contract_version": "1.0",
                            "heartbeat_interval_seconds": 30,
                            "model_roster": {
                                "qwen": {
                                    "provider": "nebius",
                                    "endpoint": "https://api.example.test",
                                    "model": "Qwen/Qwen3.5-397B-A17B",
                                    "credential_env": "NEBIUS_TOKEN",
                                }
                            },
                            "agent_invocation": {"model_reference": "qwen"},
                        }
                    ),
                    encoding="utf-8",
                )
                fastqs = []
                for role in roles:
                    (root / f"{role}.fastq").write_bytes(b"@r\nACGT\n+\n!!!!\n")
                    fastqs += ["--fastq", f"{role}={root / f'{role}.fastq'}"]
                errors = io.StringIO()
                with (
                    patch.dict(os.environ, {"NEBIUS_TOKEN": "offline-test-value"}),
                    redirect_stderr(errors),
                    redirect_stdout(io.StringIO()),
                ):
                    exit_code = main(
                        [
                            "begin",
                            "--run-root",
                            str(root / "runs"),
                            "--prompt-file",
                            str(root / "prompt.txt"),
                            "--runtime-config",
                            str(root / "runtime.json"),
                            *fastqs,
                        ],
                        parabricks_infrastructure_factory=lambda *_: self.fail(
                            "refused registration constructed execution"
                        ),
                    )

                self.assertEqual(exit_code, 2)
                self.assertEqual(
                    json.loads(errors.getvalue()),
                    {
                        "status": "failed",
                        "code": "fastq_roles_unsupported",
                        "fields": ["inputs"],
                    },
                )
                self.assertFalse((root / "runs").exists())


class ReadableReportPrintingTests(unittest.TestCase):
    def test_every_entry_command_prints_a_finished_runs_report(self):
        """#476 (owner, 2026-10-28, G5/C13; Spec r1 SP2): begin, interactive begin,
        resume and answer print the finished Run's own readable page after the run
        view."""
        final = _cli_observation(status="grounded_report_published")
        begin = [
            "begin",
            "--run-root",
            "/runs",
            "--prompt-file",
            "/inputs/request.txt",
            "--runtime-config",
            "/inputs/runtime.json",
            "--fastq",
            "read_1=/inputs/r1.fastq.gz",
        ]
        answer = [
            "answer",
            "--journey-file",
            "/runs/run-1/journey.jsonl",
            "--request-event-id",
            "request-1",
            "--response-json",
            "{}",
        ]
        for argv in (
            ["resume", "--journey-file", "/runs/run-1/journey.jsonl"],
            begin,
            [*begin, "--interactive"],
            answer,
        ):
            with self.subTest(command=argv[0], interactive="--interactive" in argv):
                stdout, stderr = io.StringIO(), io.StringIO()
                with (
                    patch("bio_neb.cli.advance", return_value=final),
                    patch("bio_neb.cli.answer_from_json", return_value=final),
                    patch("bio_neb.cli._hitl_response", return_value={}),
                    patch("bio_neb.cli._interactive_answer", return_value=final),
                    patch(
                        "bio_neb.cli.written_readable_report",
                        return_value="# Published page\n",
                    ),
                    redirect_stdout(stdout),
                    redirect_stderr(stderr),
                ):
                    code = main(argv)
                self.assertEqual(code, 0, stderr.getvalue())
                self.assertIn("# Published page\n", stderr.getvalue())


class ConfigurationFaultTests(unittest.TestCase):
    """A setting the selected model refuses is a configuration fault (#220, C1)."""

    _CEILING_PATH = "agent_invocation.model_settings.max_tokens"
    # An unknown setting and a typed value the model refuses take one path.
    _FAULTS = (
        ({"max_tokens": 300000}, _CEILING_PATH),
        (
            {"reasoning_effort": "high"},
            "agent_invocation.model_settings.reasoning_effort",
        ),
    )

    def _inputs(self, root: Path, *, model_settings=None) -> Path:
        (root / "prompt.txt").write_bytes(b"Analyze rs429358.")
        (root / "r1.fastq").write_bytes(b"@r\nACGT\n+\n!!!!\n")
        (root / "r2.fastq").write_bytes(b"@r\nTGCA\n+\n!!!!\n")
        runtime = {
            "contract_version": "1.0",
            "heartbeat_interval_seconds": 30,
            "model_roster": {
                "qwen": {
                    "provider": "nebius",
                    "endpoint": "https://api.tokenfactory.us-central1.nebius.com/v1",
                    "model": "Qwen/Qwen3.5-397B-A17B",
                    "credential_env": "BIO_NEB_TEST_ABSENT_KEY",
                }
            },
            "agent_invocation": {"model_reference": "qwen"},
        }
        if model_settings is not None:
            runtime["agent_invocation"]["model_settings"] = model_settings
        path = root / "runtime.json"
        path.write_text(json.dumps(runtime), encoding="utf-8")
        return path

    def _run(self, argv, *, credential: str | None = None):
        errors = io.StringIO()
        with (
            patch.dict("os.environ"),
            redirect_stderr(errors),
            redirect_stdout(io.StringIO()),
        ):
            os.environ.pop("BIO_NEB_TEST_ABSENT_KEY", None)
            if credential is not None:
                os.environ["BIO_NEB_TEST_ABSENT_KEY"] = credential
            exit_code = main(
                argv,
                parabricks_infrastructure_factory=lambda *_: self.fail(
                    "a configuration fault constructed execution"
                ),
            )
        return exit_code, json.loads(errors.getvalue())

    def _begin(self, root: Path):
        return self._run(
            [
                "begin",
                "--run-root",
                str(root / "runs"),
                "--prompt-file",
                str(root / "prompt.txt"),
                "--runtime-config",
                str(root / "runtime.json"),
                "--fastq",
                "read_1=" + str(root / "r1.fastq"),
                "--fastq",
                "read_2=" + str(root / "r2.fastq"),
            ]
        )

    def test_begin_refuses_a_refused_setting_before_any_journey(self):
        for settings, path in self._FAULTS:
            with (
                self.subTest(settings=settings),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                self._inputs(root, model_settings=settings)
                exit_code, failure = self._begin(root)
                self.assertEqual(exit_code, 2)
                self.assertEqual(
                    failure,
                    {
                        "status": "failed",
                        "code": "runtime_selections_invalid",
                        "fields": [path],
                    },
                )
                self.assertFalse((root / "runs").exists())

    def test_a_run_keeps_its_registered_selections_when_its_file_gains_a_fault(self):
        # The file is not re-read: an Attempt resolves the registered selections,
        # so a refused setting written into the file later, or the file's removal,
        # never reaches the Run. Resume holds where the registered selections do:
        # at the absent credential.
        for settings, _path in (*self._FAULTS, (None, None)):
            with (
                self.subTest(settings=settings),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                journey_file = self._registered(root)
                if settings is None:
                    (root / "runtime.json").unlink()
                else:
                    self._inputs(root, model_settings=settings)
                exit_code, failure = self._run(
                    ["resume", "--journey-file", str(journey_file)]
                )
                events = Journey.open(journey_file).read()
                self.assertEqual(
                    (exit_code, failure["code"], failure["status"]),
                    (2, "model_credential_unavailable", "run_held"),
                )
                recorded = [
                    (e["payload"]["code"], e["payload"]["violations"])
                    for e in events
                    if e["event_type"] == "RuntimeConfigurationFailed"
                ]
                self.assertEqual(
                    recorded,
                    [
                        (
                            "model_credential_unavailable",
                            ["model_roster.qwen.credential_env"],
                        )
                    ],
                )

    def test_a_setting_the_model_cannot_translate_is_a_configuration_fault(self):
        # The endpoint policy's translation table is data, checked at
        # configuration read: a typed setting with no wire field is
        # refused before any Journey and, on an existing Run, holds it.
        from dataclasses import replace

        from bio_neb.agentic_journey.model_settings import (
            QWEN_ENDPOINT_POLICY,
            model_registry,
            use_model_registry,
        )

        policy = QWEN_ENDPOINT_POLICY
        untranslating = replace(
            policy,
            wire_settings={
                name: field
                for name, field in policy.wire_settings.items()
                if name != "top_k"
            },
        )
        path = "agent_invocation.model_settings.top_k"
        with use_model_registry(model_registry().replacing(untranslating)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                self._inputs(root, model_settings={"top_k": 5})
                self.assertEqual(
                    self._begin(root),
                    (
                        2,
                        {
                            "status": "failed",
                            "code": "runtime_selections_invalid",
                            "fields": [path],
                        },
                    ),
                )
                self.assertFalse((root / "runs").exists())
        # A Run registered while the entry translated the setting no longer reads
        # once it cannot: retained selections validate under current rules (ADR
        # 0010). The hold this test made by editing the file mid-Run is retired
        # with the file re-read.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            journey_file = self._registered(root, model_settings={"top_k": 5})
            before = Journey.open(journey_file).read()
            with use_model_registry(model_registry().replacing(untranslating)):
                refused = self._run(
                    ["resume", "--journey-file", str(journey_file)],
                    credential="offline-test-value",
                )
            after = Journey.open(journey_file).read()
        self.assertEqual(
            refused,
            (
                2,
                {
                    "status": "failed",
                    "code": "command_invalid",
                    "fields": [f"runtime_selections.{path}"],
                },
            ),
        )
        self.assertEqual(after, before)

    def _registered(self, root: Path, *, model_settings=None) -> Path:
        # CLI begin refuses before any Journey without the credential, so the
        # Run is registered directly, as a bound model factory's would be.
        from bio_neb.application.commands import (
            RegisterFastqInput,
            RegisterRunCommand,
            register_run,
        )

        self._inputs(root, model_settings=model_settings)
        receipt = register_run(
            RegisterRunCommand(
                run_root=str(root / "runs"),
                prompt_file=str(root / "prompt.txt"),
                runtime_config_file=str(root / "runtime.json"),
                inputs=(
                    RegisterFastqInput(
                        role="read_1", registered_path=str(root / "r1.fastq")
                    ),
                    RegisterFastqInput(
                        role="read_2", registered_path=str(root / "r2.fastq")
                    ),
                ),
            )
        )
        return Path(receipt.journey_file)


def _sequential_request(rsids=SequentialAdaptiveModel.rsids):
    return {
        "asserted_by": "researcher:owner",
        "requests": [
            {
                "request": f"Investigate {rsid} in sample HG001.",
                "target": {"rsid": rsid},
            }
            for rsid in rsids
        ],
    }


class CommandValidationBoundaryTests(unittest.TestCase):
    """A caller's invalid input refuses the command; the coordinator's own
    validation fault holds the Run (#214, #273)."""

    def _cli(self, argv):
        errors = io.StringIO()
        with redirect_stderr(errors), redirect_stdout(io.StringIO()):
            exit_code = main(argv)
        return exit_code, json.loads(errors.getvalue())

    def test_an_invalid_review_answer_refuses_and_leaves_the_run_waiting(self):
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.reference_discovery.reference_identity_tool import (
            ReferenceAdapterError,
        )
        from tests.application_fakes import _registered
        from tests.planning_assignment_fakes import PlanningComposition
        from tests.reference_fakes import ReferenceAdapter

        class Unavailable(ReferenceAdapter):
            def resolve_taxon(self, query):
                raise ReferenceAdapterError(
                    "http_error", "Service rejected request.", retryable=False
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, paths = _registered(root)
            composition = PlanningComposition(receipt, paths)
            composition.reference_adapter = Unavailable()
            response = root / "answer.json"
            with use_run_advancement_composition(composition):
                stopped = advance(ResumeRun(journey_file=receipt.journey_file))
                request = stopped.current_review
                # The answer breaks its own contract: amend names no field.
                response.write_text(
                    json.dumps(
                        {
                            "request_id": request.request.request.request_id,
                            "action": "amend",
                            "by": "researcher",
                        }
                    ),
                    encoding="utf-8",
                )
                before = Journey.open(Path(receipt.journey_file)).read()
                exit_code, failure = self._cli(
                    [
                        "answer",
                        "--journey-file",
                        receipt.journey_file,
                        "--request-event-id",
                        request.request_event_id,
                        "--response-file",
                        str(response),
                    ]
                )
            self.assertEqual(exit_code, 2)
            self.assertEqual(failure["code"], "command_invalid")
            self.assertIn("answer.response", failure["fields"])
            # The refusal records no fault and leaves the Journey as it was.
            self.assertEqual(Journey.open(Path(receipt.journey_file)).read(), before)
            self.assertEqual(
                observe(ObserveRun(journey_file=receipt.journey_file)).status,
                stopped.status,
            )

    def test_a_coordinator_validation_fault_holds_the_run(self):
        from tests.application_fakes import _registered

        internal = ValidationError.from_exception_data(
            "InspectionResult", [{"type": "missing", "loc": ("digest",), "input": {}}]
        )

        def failing(command, journey):
            raise internal

        with tempfile.TemporaryDirectory() as directory:
            receipt, _paths = _registered(Path(directory).resolve())
            with patch(
                "bio_neb.input_inspection.commands.inspect_registered_inputs",
                failing,
            ):
                exit_code, failure = self._cli(
                    ["resume", "--journey-file", receipt.journey_file]
                )
            self.assertEqual(exit_code, 2)
            self.assertEqual(failure["code"], "coordinator_unhandled_error")
            closed = Journey.open(Path(receipt.journey_file)).latest_event(
                ("PassClosed",)
            )
            self.assertEqual(closed["payload"]["disposition"], "held")
            self.assertTrue(
                closed["payload"]["failure"]["cause"].startswith("ValidationError@")
            )
            status_code, status = self._status(receipt.journey_file)
            self.assertEqual((status_code, status), (0, "run_held"))

    def _status(self, journey_file):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            exit_code = main(["observe", "--journey-file", journey_file])
        return exit_code, json.loads(output.getvalue())["status"]


class CapturedInvestigationRequestCliTests(unittest.TestCase):
    """Investigation begin reads the Run's captured request before any other work."""

    def _registered(self, root, investigation_request):
        from bio_neb.application.commands import (
            RegisterFastqInput,
            RegisterRunCommand,
            register_run,
        )
        from tests.proposal_information_fakes import runtime_configuration

        (root / "prompt.txt").write_text("Analyze HG001.", encoding="utf-8")
        (root / "runtime.json").write_text(
            json.dumps(runtime_configuration()), encoding="utf-8"
        )
        (root / "r1.fastq").write_bytes(b"@r\nACGT\n+\n!!!!\n")
        (root / "r2.fastq").write_bytes(b"@r\nTGCA\n+\n!!!!\n")
        request_file = None
        if investigation_request is not None:
            request_file = root / "snp-request.json"
            request_file.write_text(json.dumps(investigation_request), encoding="utf-8")
        receipt = register_run(
            RegisterRunCommand(
                run_root=str(root / "runs"),
                prompt_file=str(root / "prompt.txt"),
                runtime_config_file=str(root / "runtime.json"),
                request_files=() if request_file is None else (str(request_file),),
                inputs=(
                    RegisterFastqInput(
                        role="read_1", registered_path=str(root / "r1.fastq")
                    ),
                    RegisterFastqInput(
                        role="read_2", registered_path=str(root / "r2.fastq")
                    ),
                ),
            )
        )
        return Journey.open(Path(receipt.journey_file))

    def _supplied(self, root, investigation_request):
        path = root / "supplied-request.json"
        path.write_text(json.dumps(investigation_request), encoding="utf-8")
        return str(path)

    def test_captured_request_is_the_runs_request_unless_a_different_one_is_supplied(
        self,
    ):
        from bio_neb.investigation.investigation_commands import (
            InvestigationApplicationRefused,
            resolve_investigation_request,
        )
        from bio_neb.investigation.investigation_request import (
            InvestigationRequest,
            InvestigationRequestItem,
        )

        captured = _sequential_request(SequentialAdaptiveModel.rsids[:2])
        expected = InvestigationRequest(
            asserted_by="researcher:owner",
            items=tuple(
                InvestigationRequestItem(
                    request_text=f"Investigate {rsid} in sample HG001.", rsid=rsid
                )
                for rsid in SequentialAdaptiveModel.rsids[:2]
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            journey = self._registered(root, captured)
            omitted = resolve_investigation_request(journey, ())
            identical = resolve_investigation_request(
                journey, (self._supplied(root, captured),)
            )
            with self.assertRaises(InvestigationApplicationRefused) as conflict:
                resolve_investigation_request(
                    journey,
                    (
                        self._supplied(
                            root, _sequential_request(SequentialAdaptiveModel.rsids[:1])
                        ),
                    ),
                )

        self.assertEqual(omitted, expected)
        self.assertEqual(identical, expected)
        self.assertEqual(conflict.exception.code, "investigation_request_conflict")
        self.assertEqual(conflict.exception.fields, ("request_file",))

    def test_begin_takes_several_request_files_in_order(self):
        # --request-file repeats on both entries; the files keep their order.
        from bio_neb.cli import _parser

        parser = _parser()
        begin = parser.parse_args(
            [
                "begin",
                "--run-root",
                "/runs",
                "--prompt-file",
                "/prompt.txt",
                "--runtime-config",
                "/runtime.json",
                "--fastq",
                "read_1=/r1.fastq",
                "--request-file",
                "/list.json",
                "--request-file",
                "/paper.json",
            ]
        )
        investigation = parser.parse_args(
            [
                "investigation-begin",
                "--journey-file",
                "/journey.jsonl",
                "--request-file",
                "/list.json",
                "--request-file",
                "/paper.json",
            ]
        )

        self.assertEqual(begin.request_file, ["/list.json", "/paper.json"])
        self.assertEqual(investigation.request_file, ["/list.json", "/paper.json"])

    def test_without_a_captured_request_only_a_supplied_one_is_used(self):
        from bio_neb.investigation.investigation_commands import (
            InvestigationApplicationRefused,
            resolve_investigation_request,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            journey = self._registered(root, None)
            supplied = resolve_investigation_request(
                journey,
                (
                    self._supplied(
                        root, _sequential_request(SequentialAdaptiveModel.rsids[2:])
                    ),
                ),
            )
            with self.assertRaises(InvestigationApplicationRefused) as needed:
                resolve_investigation_request(journey, ())

        self.assertEqual(
            [(item.rsid, item.request_text) for item in supplied.items],
            [("rs4149056", "Investigate rs4149056 in sample HG001.")],
        )
        self.assertEqual(needed.exception.code, "investigation_request_needed")
        self.assertEqual(needed.exception.fields, ("request_file",))

    def test_public_begin_refuses_conflicting_or_missing_requests_before_any_work(
        self,
    ):
        cases = {
            "conflict": (
                _sequential_request(),
                _sequential_request(SequentialAdaptiveModel.rsids[:1]),
                "investigation_request_conflict",
            ),
            "needed": (None, None, "investigation_request_needed"),
        }
        for name, (captured, supplied, code) in cases.items():
            with (
                self.subTest(case=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                journey = self._registered(root, captured)
                before = journey.path.read_bytes()
                extra = (
                    ()
                    if supplied is None
                    else ("--request-file", self._supplied(root, supplied))
                )
                errors = io.StringIO()
                with redirect_stderr(errors), redirect_stdout(io.StringIO()):
                    exit_code = main(
                        [
                            "investigation-begin",
                            "--journey-file",
                            str(journey.path),
                            "--scientific-continuation-file",
                            str(root / "never-read-continuation.json"),
                            *extra,
                        ],
                        investigation_model_factory=lambda *_: self.fail(
                            "model constructed"
                        ),
                        investigation_source_adapter_factory=lambda: self.fail(
                            "source adapter constructed"
                        ),
                    )
                unchanged = journey.path.read_bytes() == before

                self.assertEqual(exit_code, 2)
                refusal = json.loads(errors.getvalue())
                self.assertEqual(refusal["code"], code)
                self.assertEqual(refusal["fields"], ["request_file"])
                self.assertTrue(unchanged)


if __name__ == "__main__":
    unittest.main()
