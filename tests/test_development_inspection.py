"""Native Logfire capture at public Attempt and Journey-query seams."""

import asyncio
import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic_ai.exceptions import ModelHTTPError

from bio_neb import cli
from bio_neb.agentic_journey.invocation_observation import inspection_event
from bio_neb.agentic_journey.planning_assignment import (
    planning_assignment,
)
from bio_neb.application.queries import (
    ReviewJourneyQuery,
    review_journey,
)
from bio_neb.devtools import agent_inspection
from bio_neb.devtools.agent_inspection import LocalInspection
from bio_neb.investigation.investigation import (
    InvestigationRequestItem,
    UserAssertion,
    investigation_output_member_id,
    project_investigation,
)
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.application_fakes import _registered
from tests.input_correction_fakes import prepare_input_correction
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.parabricks_operation_fakes import approved_input
from tests.proposal_information_fakes import InspectionModel
from tests.sequential_investigation_support import (
    ASSEMBLY_BODY,
    SequentialAdaptiveModel,
    SequentialSourceResponses,
    _IndependentCorrectionsModel,
    _MalformedOutputModel,
    _RepeatedActionModel,
)
from tests.sequential_investigation_support import (
    SourceResponsesOnly as _SourceResponsesOnly,
)
from tests.sequential_investigation_support import (
    input_and_assertion as _input_and_assertion,
)
from tests.sequential_investigation_support import (
    run_attempt as _run_attempt,
)
from tests.terminal_journey_fakes import execute_admitted_terminal, forked_child


def captured_example(path, case):
    spans = [json.loads(line) for line in path.read_text().splitlines()]
    print(json.dumps({"case": case, "captured_spans": spans}))
    return spans


def observed_events(spans):
    return [
        json.loads(span["attributes"]["event"])
        for span in spans
        if span["name"] in {"Journey observation", "Contact observation"}
    ]


class DevelopmentInspectionTests(unittest.TestCase):
    def test_investigation_cancellation_retains_incomplete_prefix_without_authority(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            adapter = _SourceResponsesOnly()
            calls = 0

            async def cancel(messages, info):
                nonlocal calls
                del messages, info
                calls += 1
                raise asyncio.CancelledError()

            path = root / "cancelled.jsonl"
            with (
                LocalInspection(path) as capture,
                self.assertRaises(asyncio.CancelledError),
            ):
                _run_attempt(
                    root / "journey",
                    SimpleNamespace(model=ScriptedModelEndpoint(cancel)),
                    adapter,
                )
            self.assertEqual((calls, adapter.contact_count), (1, 0))
            self.assertEqual(capture.errors, [])
            spans = captured_example(path, "investigation-cancelled-prefix")
            events = observed_events(spans)
            self.assertEqual(
                sum(
                    event["event_type"] == "ModelPhysicalContactCancelled"
                    for event in events
                ),
                1,
            )
            self.assertFalse(
                any(
                    event["event_type"]
                    in {
                        "ModelResponseRecorded",
                        "InvestigationAttemptFailed",
                        "InvestigationProposalRecorded",
                        "InvestigationReviewRequested",
                    }
                    for event in events
                )
            )
            journey = Journey.open(next((root / "journey").glob("*/journey.jsonl")))
            # A cancelled Attempt does not end the Investigation: its work stays
            # ready for a fresh Attempt.
            self.assertEqual(
                project_investigation(read_evidence(journey)).status,
                "investigation_work_ready",
            )
            retained = journey.path.read_bytes()
            path.unlink()
            self.assertEqual(
                project_investigation(read_evidence(Journey.open(journey.path))).status,
                "investigation_work_ready",
            )
            self.assertEqual(journey.path.read_bytes(), retained)

    def test_native_payload_and_early_semantic_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, _ = _registered(root)
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            capture_path = root / "spans.jsonl"
            with LocalInspection(capture_path) as capture:
                result = planning_assignment(
                    model_factory=lambda _, _selected: (
                        InspectionModel(input_set_id).model(_, _selected)
                    )
                ).run(
                    prepare_input_correction(Journey.open(Path(receipt.journey_file))),
                    environ={"LOCAL_MODEL_KEY": "secret"},
                )
            self.assertEqual(result.status, "inspection_succeeded")
            self.assertEqual(capture.errors, [])
            spans = captured_example(capture_path, "early-semantic-completion")
            native = [
                span
                for span in spans
                if span["instrumentation_scope"]["name"].startswith("pydantic-ai")
            ]
            self.assertTrue(native)
            emitted = json.dumps(native)
            self.assertIn("inspect_input_metadata", emitted)
            self.assertIn("inspect-call", emitted)
            self.assertIn(input_set_id, emitted)
            self.assertNotIn("LOCAL_MODEL_KEY", emitted)
            markers = [
                span["attributes"]
                for span in spans
                if span["name"] == "Journey semantic completion"
            ]
            self.assertEqual(len(markers), 1)
            self.assertFalse(markers[0]["framework_end"])
            self.assertEqual(markers[0]["checkpoint"], "registered_inputs_inspected")
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).status,
                result.status,
            )

    def test_logical_request_contains_two_physical_contacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, _ = _registered(root)
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            scripted = InspectionModel(input_set_id)
            contacts = 0

            async def respond(messages, info):
                nonlocal contacts
                contacts += 1
                if contacts == 1:
                    raise ModelHTTPError(
                        503, "synthetic", body={"message": "synthetic-unavailable"}
                    )
                return await scripted.respond(messages, info)

            path = root / "retry.jsonl"
            with LocalInspection(path) as capture:
                result = planning_assignment(
                    model_factory=ScriptedModelEndpoint(respond)
                ).run(
                    prepare_input_correction(Journey.open(Path(receipt.journey_file))),
                    environ={"LOCAL_MODEL_KEY": "secret"},
                )
            self.assertEqual((result.status, contacts), ("inspection_succeeded", 2))
            self.assertEqual(capture.errors, [])
            spans = captured_example(path, "physical-availability-retry")
            events = observed_events(spans)
            logical = [
                event
                for event in events
                if event["event_type"] == "ModelRequestRecorded"
            ]
            physical = [
                event
                for event in events
                if event["event_type"] == "ModelPhysicalContactStarted"
            ]
            self.assertEqual(len(logical), 1)
            self.assertEqual(
                [event["payload"]["contact_ordinal"] for event in physical], [1, 2]
            )
            self.assertEqual(
                {event["payload"]["logical_request_event_id"] for event in physical},
                {logical[0]["event_id"]},
            )
            native_model = [
                span
                for span in spans
                if span["attributes"].get("gen_ai.operation.name") == "chat"
            ]
            self.assertEqual(len(native_model), 1)
            self.assertEqual(
                native_model[0]["attributes"]["attempt_id"],
                logical[0]["payload"]["attempt_id"],
            )
            self.assertEqual(
                native_model[0]["attributes"]["journey_id"], logical[0]["journey_id"]
            )
            self.assertEqual(
                native_model[0]["attributes"]["agent_role"],
                "planning_agent",
            )
            self.assertNotEqual(
                native_model[0]["start_time"], native_model[0]["end_time"]
            )

    def test_investigation_repairs_refusals_and_exhaustion_are_inspectable(self):
        cases = (
            (
                "repairs",
                _IndependentCorrectionsModel(),
                {"failed_turns": 3},
                "review_pending",
                7,
                2,
            ),
            (
                "output-exhaustion",
                _MalformedOutputModel(),
                {"failed_turns": 3},
                "agent_work_review_pending",
                3,
                0,
            ),
            (
                "repeated-action",
                _RepeatedActionModel(),
                {"failed_turns": 2},
                "agent_work_review_pending",
                3,
                1,
            ),
        )
        for case, model, policy, status, requests, contacts in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                path = root / "spans.jsonl"
                adapter = _SourceResponsesOnly()
                _, assertion = _input_and_assertion()
                expected_members = {
                    investigation_output_member_id(
                        InvestigationRequestItem(
                            request_text=assertion.request_text, rsid=assertion.rsid
                        )
                    )
                }
                with LocalInspection(path) as capture:
                    result, journey = _run_attempt(
                        root / "journey",
                        model,
                        adapter,
                        continuation_policy=policy,
                        assertion=assertion,
                    )
                self.assertEqual(
                    (result.status, model.calls, adapter.contact_count),
                    (status, requests, contacts),
                )
                self.assertEqual(capture.errors, [])
                spans = captured_example(path, case)
                events = observed_events(spans)
                self.assertEqual(
                    project_investigation(read_evidence(Journey.open(journey.path))),
                    result,
                )
                native = [
                    span
                    for span in spans
                    if span["instrumentation_scope"]["name"].startswith("pydantic-ai")
                ]
                self.assertTrue(native)
                self.assertEqual(
                    {span["attributes"]["member_id"] for span in native},
                    expected_members,
                )
                self.assertEqual(
                    {span["attributes"]["attempt_id"] for span in native},
                    {result.attempts[0].attempt_id},
                )
                self.assertEqual(
                    {span["attributes"]["journey_id"] for span in native},
                    {str(journey.journey_id)},
                )
                self.assertTrue(
                    all(
                        span["attributes"]["agent_role"] == "investigation_planner"
                        for span in native
                    )
                )
                if case == "repairs":
                    validations = [
                        event
                        for event in events
                        if event["event_type"] == "StructuredOutputValidationLinked"
                    ]
                    self.assertEqual(
                        [
                            event["payload"]["validation_attempt"]
                            for event in validations
                        ],
                        [1, 2, 3],
                    )
                    self.assertIn("wrong-output-1", json.dumps(native))
                    self.assertIn("wrong-action-1", json.dumps(native))
                    run = next(
                        span
                        for span in native
                        if span["attributes"].get("gen_ai.operation.name")
                        == "invoke_agent"
                    )
                    messages = json.loads(run["attributes"]["pydantic_ai.all_messages"])
                    repairs = [
                        part
                        for message in messages
                        for part in message["parts"]
                        if part.get("type") == "tool_call_response"
                        and part.get("id") in {"wrong-output-1", "wrong-output-2"}
                    ]
                    self.assertEqual(
                        [part["id"] for part in repairs],
                        ["wrong-output-1", "wrong-output-2"],
                    )
                    # A re-typed source fact is a field the draft does not take:
                    # the schema's repair, not a copy refusal.
                    for part in repairs:
                        self.assertEqual(part["name"], "compile_proposal")
                        self.assertIn('"type": "extra_forbidden"', part["result"])
                        self.assertIn('"one_based_position"', part["result"])
                        self.assertNotIn(
                            "model_output_changed_source_facts", part["result"]
                        )
                elif case == "output-exhaustion":
                    # Each refused output is a failed turn; the 3rd stops.
                    self.assertEqual(result.attempts[0].stop_reason, "failed_turns")
                    self.assertEqual(
                        len(
                            result.attempts[0]
                            .structured_output_lineages[0]
                            .validation_attempts
                        ),
                        3,
                    )
                else:
                    # The refused call is corrected once; the 2nd failed turn in a
                    # row stops.
                    refused = [
                        event["payload"]
                        for event in events
                        if event["event_type"] == "ToolCallValidated"
                        and event["payload"]["status"] == "invalid"
                    ]
                    self.assertEqual(
                        [
                            (payload["tool_call_id"], payload["errors"])
                            for payload in refused
                        ],
                        [
                            # A lookup exists, so it names the member's lookup.
                            (
                                f"repeated-{call}",
                                ["fetch_assembly_report_refsnp_event_unknown"],
                            )
                            for call in (2, 3)
                        ],
                    )
                    self.assertEqual(result.attempts[0].stop_reason, "failed_turns")
                    self.assertEqual(
                        len(
                            [
                                event
                                for event in events
                                if event["event_type"]
                                == "InvestigationSourceContactStarted"
                            ]
                        ),
                        1,
                    )

    def test_native_spans_follow_each_actual_collection_member(self):
        class LoopBoundModel(SequentialAdaptiveModel):
            loop = None

            async def respond(self, messages, info):
                loop = asyncio.get_running_loop()
                if self.loop is None:
                    self.loop = loop
                elif self.loop is not loop:
                    raise RuntimeError("model transport reused across event loops")
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, original = _input_and_assertion()
            assertion = UserAssertion.create_collection(
                asserted_by=original.asserted_by,
                requests=tuple(
                    InvestigationRequestItem(
                        request_text=f"Investigate {rsid}.", rsid=rsid
                    )
                    for rsid in ("rs429358", "rs7412")
                ),
                predecessor_ids=original.predecessor_ids,
            )
            path = root / "members.jsonl"
            expected_members = [
                investigation_output_member_id(request, ordinal)
                for ordinal, request in enumerate(assertion.requests)
            ]
            with LocalInspection(path) as capture:
                result, journey = _run_attempt(
                    root / "journey",
                    LoopBoundModel(),
                    SequentialSourceResponses(),
                    assertion=assertion,
                )
            self.assertEqual(result.status, "review_pending")
            self.assertEqual(capture.errors, [])
            spans = captured_example(path, "two-members")
            agents = [
                span
                for span in spans
                if span["attributes"].get("gen_ai.operation.name") == "invoke_agent"
            ]
            self.assertEqual(
                [span["attributes"]["member_id"] for span in agents],
                expected_members,
            )
            self.assertEqual(
                len({span["attributes"]["member_id"] for span in agents}), 2
            )
            # Each member is its own Attempt.
            self.assertEqual(
                len({span["attributes"]["attempt_id"] for span in agents}), 2
            )
            self.assertEqual(len({span["context"]["span_id"] for span in agents}), 2)
            self.assertEqual(project_investigation(read_evidence(journey)), result)

    def test_cancelled_attempt_preserves_incomplete_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, _ = _registered(root)
            path = root / "cancelled.jsonl"
            incomplete = []

            async def cancel(messages, info):
                del messages, info
                pending = observed_events(captured_example(path, "incomplete-prefix"))
                self.assertEqual(
                    len(
                        [
                            event
                            for event in pending
                            if event["event_type"] == "ModelPhysicalContactStarted"
                        ]
                    ),
                    1,
                )
                self.assertFalse(
                    any(
                        event["event_type"]
                        in {"AttemptSucceeded", "AttemptFailed", "AttemptCancelled"}
                        for event in pending
                    )
                )
                with self.assertRaises(ValueError):
                    review_journey(
                        ReviewJourneyQuery(journey_file=receipt.journey_file)
                    )
                incomplete.append("incomplete")
                raise asyncio.CancelledError()

            with (
                LocalInspection(path) as capture,
                self.assertRaises(asyncio.CancelledError),
            ):
                planning_assignment(model_factory=ScriptedModelEndpoint(cancel)).run(
                    prepare_input_correction(Journey.open(Path(receipt.journey_file))),
                    environ={"LOCAL_MODEL_KEY": "secret"},
                )
            self.assertEqual(incomplete, ["incomplete"])
            self.assertEqual(capture.errors, [])
            spans = captured_example(path, "cancelled-attempt")
            events = observed_events(spans)
            cancelled = next(
                event
                for event in events
                if event["event_type"] == "ModelPhysicalContactCancelled"
            )
            terminal = next(
                event for event in events if event["event_type"] == "AttemptCancelled"
            )
            self.assertEqual(
                cancelled["payload"]["attempt_id"], terminal["payload"]["attempt_id"]
            )
            self.assertFalse(
                any(event["event_type"] == "ModelResponseRecorded" for event in events)
            )
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).attempt.outcome,
                "cancelled",
            )

    def test_cloud_capture_admits_only_the_token_and_keeps_the_local_copy(self):
        import logfire

        real_configure = logfire.configure
        cases = (
            ("local-with-token", False, {"LOGFIRE_TOKEN": "write-token"}),
            ("cloud-without-token", True, {}),
            (
                "cloud-with-endpoint",
                True,
                {
                    "LOGFIRE_TOKEN": "write-token",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318",
                },
            ),
            (
                "cloud-with-base-url",
                True,
                {
                    "LOGFIRE_TOKEN": "write-token",
                    "LOGFIRE_BASE_URL": "https://logfire-eu.pydantic.dev",
                },
            ),
            ("cloud", True, {"LOGFIRE_TOKEN": "write-token"}),
        )
        for case, cloud, environment in cases:
            with (
                self.subTest(case=case),
                tempfile.TemporaryDirectory() as directory,
                ExitStack() as patches,
            ):
                root = Path(directory).resolve()
                receipt, _ = _registered(root)
                input_set_id = review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).registered_input_set.input_set_id
                path = root / "spans.jsonl"
                ambient = {
                    name: value
                    for name, value in os.environ.items()
                    if not name.startswith(("LOGFIRE_", "OTEL_"))
                }
                patches.enter_context(
                    patch.dict(os.environ, {**ambient, **environment}, clear=True)
                )
                network = patches.enter_context(
                    patch(
                        "socket.socket.connect",
                        side_effect=AssertionError("network contact forbidden"),
                    )
                )
                configured = []

                def offline_configure(configured=configured, **options):
                    # Record the cloud request, then configure only the local copy.
                    configured.append(options)
                    return real_configure(
                        **{**options, "send_to_logfire": False, "token": None}
                    )

                patches.enter_context(
                    patch("logfire.configure", side_effect=offline_configure)
                )
                with LocalInspection(path, cloud=cloud) as capture:
                    result = planning_assignment(
                        model_factory=lambda _, _selected, input_set_id=input_set_id: (
                            InspectionModel(input_set_id).model(_, _selected)
                        )
                    ).run(
                        prepare_input_correction(
                            Journey.open(Path(receipt.journey_file))
                        ),
                        environ={"LOCAL_MODEL_KEY": "secret"},
                    )
                self.assertEqual(result.status, "inspection_succeeded")
                network.assert_not_called()
                if case != "cloud":
                    # Token presence never selects cloud, and any other telemetry
                    # name disables capture instead of adding a destination.
                    self.assertEqual(configured, [])
                    self.assertTrue(capture.errors)
                    self.assertFalse(path.exists())
                    continue
                self.assertEqual(capture.errors, [])
                (options,) = configured
                self.assertTrue(options["send_to_logfire"])
                self.assertEqual(options["token"], "write-token")
                self.assertEqual(
                    options["advanced"].base_url, "https://logfire-us.pydantic.dev"
                )
                self.assertFalse(options["scrubbing"])
                spans = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertIn(
                    "inspect_input_metadata",
                    json.dumps(
                        [
                            span
                            for span in spans
                            if span["instrumentation_scope"]["name"].startswith(
                                "pydantic-ai"
                            )
                        ]
                    ),
                )
                self.assertNotIn("write-token", path.read_text())

    def test_capture_entry_wraps_one_native_command_and_keeps_its_exit_status(self):
        def native_command(argv):
            forwarded.append(argv)
            inspection_event("Entry probe", command=argv[0])
            return 7

        ambient = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("LOGFIRE_", "OTEL_"))
        }
        commands = (
            ["resume", "--journey-file", "/runs/r/journey.jsonl", "--interactive"],
            [
                "investigation-begin",
                "--journey-file",
                "/runs/with spaces/journey.jsonl",
                "--request-file",
                "/runs/researcher request.json",
            ],
            [
                "report-explain",
                "--journey-file",
                "/runs/r/journey.jsonl",
                "--report-digest",
                "grounded-report:synthetic",
                "--requested-by",
                "researcher:owner",
            ],
        )
        for command in commands:
            for case, environment in (
                ("captured", {}),
                (
                    "capture-unavailable",
                    {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://x:4318"},
                ),
            ):
                forwarded = []
                with (
                    self.subTest(case=case),
                    tempfile.TemporaryDirectory() as directory,
                    patch.dict(os.environ, {**ambient, **environment}, clear=True),
                    patch("bio_neb.cli.main", side_effect=native_command),
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    path = Path(directory).resolve() / "spans.jsonl"
                    status = agent_inspection.main(["--capture", str(path), *command])
                    self.assertEqual(status, 7)
                    self.assertEqual(forwarded, [command])
                    if case == "captured":
                        self.assertIn("Entry probe", path.read_text())
                        self.assertEqual(stderr.getvalue(), "")
                    else:
                        # Unavailable capture is reported and leaves the command alone.
                        self.assertFalse(path.exists())
                        self.assertIn("capture_errors", stderr.getvalue())

    def test_capture_error_report_failure_preserves_native_exit_status(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("logfire.configure", side_effect=OSError("capture unavailable")),
            patch("bio_neb.cli.main", return_value=7) as native,
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
            patch.object(
                stderr, "write", side_effect=OSError("diagnostic pipe closed")
            ),
        ):
            command = ["investigation-begin", "--journey-file", "/runs/r/journey.jsonl"]
            status = agent_inspection.main(
                ["--capture", str(Path(directory).resolve() / "spans.jsonl"), *command]
            )
            self.assertEqual(status, 7)
            native.assert_called_once_with(command)

    def test_capture_failures_preserve_primary_native_error(self):
        import logfire

        ambient = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("LOGFIRE_", "OTEL_"))
        }
        for failure in ("configure", "force_flush", "shutdown"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
                patch.dict(os.environ, ambient, clear=True),
                ExitStack() as patches,
            ):
                target = logfire if failure == "configure" else logfire.Logfire
                patches.enter_context(
                    patch.object(
                        target, failure, side_effect=OSError("diagnostic failure")
                    )
                )
                primary = RuntimeError("native command failed")
                native = patches.enter_context(
                    patch("bio_neb.cli.main", side_effect=primary)
                )
                command = [
                    "investigation-begin",
                    "--journey-file",
                    "/runs/r/journey.jsonl",
                ]
                with self.assertRaises(RuntimeError) as raised:
                    agent_inspection.main(
                        [
                            "--capture",
                            str(Path(directory).resolve() / "spans.jsonl"),
                            *command,
                        ]
                    )
                self.assertIs(raised.exception, primary)
                native.assert_called_once_with(command)

    def test_native_investigation_capture_preserves_review_and_journey(self):
        import logfire

        native_main = cli.main
        ambient = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("LOGFIRE_", "OTEL_"))
        }
        with tempfile.TemporaryDirectory() as parent_directory:
            # Generate one immutable synthetic Execution parent for this invocation.
            # Every capture mode admits its own fresh child and mutable adapters.
            # The parent's plan names the request every child investigates.
            investigation_request = {
                "schema_version": "bio-neb.investigation-request/2",
                "asserted_by": "researcher:owner",
                "requests": [
                    {
                        "request": "Investigate rs429358 in HG001.",
                        "target": {"rsid": "rs429358"},
                    }
                ],
            }
            approved = approved_input(
                Path(parent_directory).resolve() / "parent",
                investigation_request=investigation_request,
            )
            parent_file = Path(execute_admitted_terminal(approved))
            parent_bytes = parent_file.read_bytes()
            for failure in (
                "disabled",
                "captured",
                "ambient",
                "configure",
                "force_flush",
                "shutdown",
            ):
                with (
                    self.subTest(capture=failure),
                    tempfile.TemporaryDirectory() as directory,
                    ExitStack() as patches,
                ):
                    root = Path(directory).resolve()
                    # Synthetic Execution setup through the current public operation;
                    # no live fixture is rewritten and no provider or model contacts run.
                    journey = forked_child(root, parent_file=parent_file)
                    request = root / "request.json"
                    request.write_text(json.dumps(investigation_request))
                    model = SequentialAdaptiveModel()
                    sources = SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)
                    command = [
                        "investigation-begin",
                        "--journey-file",
                        str(journey.path),
                        "--request-file",
                        str(request),
                    ]
                    patches.enter_context(
                        patch.dict(
                            os.environ,
                            {**ambient, "LOCAL_MODEL_KEY": "synthetic-offline-key"},
                            clear=True,
                        )
                    )
                    network = patches.enter_context(
                        patch(
                            "socket.socket.connect",
                            side_effect=AssertionError("network forbidden"),
                        )
                    )
                    stdout = patches.enter_context(
                        patch("sys.stdout", new_callable=io.StringIO)
                    )
                    stderr = patches.enter_context(
                        patch("sys.stderr", new_callable=io.StringIO)
                    )
                    native = patches.enter_context(
                        patch(
                            "bio_neb.cli.main",
                            side_effect=lambda argv, model=model, sources=sources: (
                                native_main(
                                    argv,
                                    investigation_model_factory=lambda _loaded,
                                    _selected: (model.model(_loaded, _selected)),
                                    investigation_source_adapter_factory=lambda: (
                                        sources
                                    ),
                                )
                            ),
                        )
                    )
                    if failure == "ambient":
                        patches.enter_context(
                            patch.dict(
                                os.environ,
                                {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://x:4318"},
                            )
                        )
                    elif failure in {"disabled", "configure"}:
                        configure = patches.enter_context(
                            patch(
                                "logfire.configure",
                                side_effect=OSError("synthetic configure failure"),
                            )
                        )
                    elif failure in {"force_flush", "shutdown"}:
                        patches.enter_context(
                            patch.object(
                                logfire.Logfire,
                                failure,
                                side_effect=OSError("synthetic finalization failure"),
                            )
                        )
                    path = root / "spans.jsonl"
                    status = (
                        cli.main(command)
                        if failure == "disabled"
                        else agent_inspection.main(["--capture", str(path), *command])
                    )
                    self.assertEqual(status, 0)
                    native.assert_called_once_with(command)
                    view = json.loads(stdout.getvalue())
                    self.assertEqual(view["status"], "investigation_review_pending")
                    self.assertEqual(
                        review_journey(
                            ReviewJourneyQuery(journey_file=str(journey.path))
                        ).status,
                        "investigation_review_pending",
                    )
                    event_types = [
                        event["event_type"]
                        for event in Journey.open(journey.path).read()
                    ]
                    self.assertEqual(
                        event_types.count("InvestigationReviewRequested"), 1
                    )
                    self.assertNotIn("InvestigationAuthorityGranted", event_types)
                    self.assertEqual(parent_file.read_bytes(), parent_bytes)
                    before = journey.path.read_bytes()
                    inspection_event("Outside native capture")
                    self.assertEqual(journey.path.read_bytes(), before)
                    if failure == "disabled":
                        configure.assert_not_called()
                        self.assertFalse(path.exists())
                    elif failure == "captured":
                        self.assertIn("Journey observation", path.read_text())
                        self.assertNotIn("Outside native capture", path.read_text())
                        self.assertEqual(stderr.getvalue(), "")
                    else:
                        self.assertIn("capture_errors", stderr.getvalue())
                    network.assert_not_called()

    def test_capture_entry_refuses_other_commands_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            absolute = str(Path(directory).resolve() / "spans.jsonl")
            for argv in (
                ["--capture", absolute, "observe-run", "--journey-file", "/j"],
                ["--capture", "spans.jsonl", "begin", "--run-root", "/r"],
            ):
                with (
                    self.subTest(argv=argv),
                    patch("bio_neb.cli.main") as native_command,
                    patch("sys.stderr", new_callable=io.StringIO),
                ):
                    with self.assertRaises(SystemExit) as refused:
                        agent_inspection.main(argv)
                    self.assertEqual(refused.exception.code, 2)
                    native_command.assert_not_called()

    def test_disabled_and_unavailable_capture_preserve_application_outcomes(self):
        import logfire

        @contextmanager
        def broken_span(*_, **__):
            yield
            raise OSError("synthetic span close failure")

        for failure in (
            "disabled",
            "ambient",
            "configure",
            "info",
            "span",
            "span-close",
            "force_flush",
            "shutdown",
            "write",
            "close",
        ):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
                ExitStack() as patches,
            ):
                root = Path(directory).resolve()
                receipt, _ = _registered(root)
                input_set_id = review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).registered_input_set.input_set_id
                path = root / "optional.jsonl"
                network = patches.enter_context(
                    patch(
                        "socket.socket.connect",
                        side_effect=AssertionError("network contact forbidden"),
                    )
                )
                if failure == "ambient":
                    patches.enter_context(
                        patch.dict(
                            os.environ,
                            {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318"},
                        )
                    )
                elif failure == "disabled":
                    configure = patches.enter_context(
                        patch(
                            "logfire.configure",
                            side_effect=AssertionError("implicit initialization"),
                        )
                    )
                elif failure == "configure":
                    patches.enter_context(
                        patch(
                            "logfire.configure",
                            side_effect=OSError("synthetic unavailable configuration"),
                        )
                    )
                elif failure == "span-close":
                    patches.enter_context(
                        patch.object(logfire.Logfire, "span", broken_span)
                    )
                elif failure in {"info", "span", "force_flush", "shutdown"}:
                    patches.enter_context(
                        patch.object(
                            logfire.Logfire,
                            failure,
                            side_effect=OSError("synthetic unavailable SDK"),
                        )
                    )
                elif failure in {"write", "close"}:
                    original_open = Path.open

                    class FailedStream:
                        def __init__(self, stream, failure=failure):
                            self.stream = stream
                            self.failure = failure

                        def write(self, content):
                            if self.failure == "write":
                                raise OSError("synthetic disk failure")
                            return self.stream.write(content)

                        def flush(self):
                            return self.stream.flush()

                        def close(self):
                            self.stream.close()
                            if self.failure == "close":
                                raise OSError("synthetic close failure")

                    def open_path(
                        selected,
                        *args,
                        opener=original_open,
                        target=path,
                        stream_class=FailedStream,
                        **kwargs,
                    ):
                        stream = opener(selected, *args, **kwargs)
                        return stream_class(stream) if selected == target else stream

                    patches.enter_context(patch.object(Path, "open", open_path))
                capture = (
                    None
                    if failure == "disabled"
                    else patches.enter_context(LocalInspection(path))
                )
                result = planning_assignment(
                    model_factory=lambda _, _selected, input_set_id=input_set_id: (
                        InspectionModel(input_set_id).model(_, _selected)
                    )
                ).run(
                    prepare_input_correction(Journey.open(Path(receipt.journey_file))),
                    environ={"LOCAL_MODEL_KEY": "secret"},
                )
                self.assertEqual(result.status, "inspection_succeeded")
                before = Path(receipt.journey_file).read_bytes()
                patches.close()
                self.assertEqual(
                    review_journey(
                        ReviewJourneyQuery(journey_file=receipt.journey_file)
                    ).status,
                    result.status,
                )
                self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
                network.assert_not_called()
                if capture is None:
                    configure.assert_not_called()
                    self.assertFalse(path.exists())
                else:
                    self.assertTrue(capture.errors)
                    print(
                        json.dumps(
                            {
                                "case": "unavailable-" + failure,
                                "diagnostic_errors": capture.errors,
                                "application_status": result.status,
                            }
                        )
                    )
