"""Native contact-free read commands and the DEV live-readiness check (#171)."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from bio_neb import cli as runner
from bio_neb.devtools import live_readiness
from tests.clock_check_fakes import steady_clock, write_clock_record
from tests.planning_assignment_fakes import (
    execution_configuration,
    parabricks_account,
    prepare_configured_run,
)
from tests.proposal_information_fakes import runtime_configuration
from tests.run_prerequisites import RunPrerequisites, prerequisite


def _invoke(main, argv):
    stdout, stderr = io.StringIO(), io.StringIO()
    with (
        patch("socket.socket", side_effect=AssertionError("network contact")),
        patch("subprocess.Popen", side_effect=AssertionError("process contact")),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _tree(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _current_runtime(root: Path) -> dict:
    """A #154-shaped runtime file in the current contract (no retired fields).

    Its roster names the build's default model under its default key (#440), so
    the model selection is not drift.
    """
    configuration = execution_configuration(root)
    runtime = runtime_configuration()
    runtime.update(
        {
            "model_roster": {"nemotron": {"model": "nvidia/Nemotron-3.5-Super-VL"}},
            "agent_invocation": {
                "exhaustion": {
                    "model_availability": "review",
                },
                "loop_retry": 0,
                "model_requests": 200,
                # No failure rule stated: the role default's four failed turns
                # apply (the retired output repair was here).
                "scopes": {"role": {}, "stage": {}, "tool": {}},
                "tool_retry": 0,
            },
            "artifact_verification": {"mode": "reuse_creation_proof", "reason": None},
            "heartbeat_interval_seconds": 30.0,
            "investigation_work": {
                "completion": "require_terminal_members",
                "item_disposition": "retain_classified_outcome",
                "other_work": "continue_eligible",
                "unresolved_identity": "block_finalization",
            },
            "parabricks": configuration.model_dump(mode="json"),
        }
    )
    return runtime


def _a8_runtime(root: Path) -> dict:
    """The retained A8 runtime's agent sections over a current per-Run base."""
    runtime = _current_runtime(root)
    runtime["agent_continuation"] = {
        "execution_planning_agent": {"action_correction": 1, "output_repair": 0},
        "planning_agent": {"action_correction": 3, "output_repair": 0},
        "proposal_information_planner": {"action_correction": 0, "output_repair": 3},
        "reference_facts_planner": {"action_correction": 0, "output_repair": 0},
        "reference_identity_planner": {"action_correction": 1, "output_repair": 0},
    }
    runtime["agent_invocation"]["exhaustion"]["singleton_action"] = "terminal"
    runtime["agent_invocation"]["model_settings"] = {"max_tokens": 32768}
    runtime["investigation_continuation"] = {"action_correction": 1, "output_repair": 1}
    return runtime


class ParentIdentityCommandTests(unittest.TestCase):
    def test_pass_closed_journey_prints_its_whole_file_identity(self):
        parent = prerequisite("executed").journey_file
        retained = _tree(parent.parent)
        content = parent.read_bytes()
        lines = content.splitlines()
        head = json.loads(lines[-1])

        code, stdout, stderr = _invoke(
            runner.main, ["parent-identity", "--journey-file", str(parent)]
        )

        self.assertEqual((code, stderr), (0, ""))
        shown = json.loads(stdout)
        self.assertEqual(
            (shown["schema_version"], shown["journey_file"]),
            ("bio-neb.parent-journey-identity/2", str(parent)),
        )
        # Independent oracle: the whole file's bytes, including Pass mechanics.
        self.assertEqual(
            shown["identity"]["journey"],
            {
                "journey_id": head["journey_id"],
                "event_count": len(lines),
                "byte_count": len(content),
                "journey_sha256": hashlib.sha256(content).hexdigest(),
                "head_event_id": head["event_id"],
                "head_event_type": "PassClosed",
                "head_sha256": hashlib.sha256(lines[-1]).hexdigest(),
            },
        )
        self.assertEqual(_tree(parent.parent), retained)

    def test_unreadable_journey_is_a_typed_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "journey.jsonl"
            parent.write_text('{"not": "a journey"}\n', encoding="utf-8")

            code, stdout, stderr = _invoke(
                runner.main, ["parent-identity", "--journey-file", str(parent)]
            )

            self.assertEqual((code, stdout), (2, ""))
            refusal = json.loads(stderr)
            self.assertEqual(refusal["code"], "parent_journey_unreadable")
            self.assertEqual(refusal["fields"][0], "journey_file")


class RuntimeDriftCommandTests(unittest.TestCase):
    def _drift(self, document: dict) -> tuple[dict, Path, dict]:
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        runtime = root / "runtime.json"
        runtime.write_text(json.dumps(document, indent=2), encoding="utf-8")
        before = _tree(root)
        code, stdout, stderr = _invoke(
            runner.main, ["runtime-drift", "--runtime-config", str(runtime)]
        )
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(_tree(root), before, "the drift check wrote a file")
        return json.loads(stdout), runtime, before

    def test_a8_shaped_file_reports_refused_fields_and_its_output_ceiling(self):
        with tempfile.TemporaryDirectory() as temporary:
            document = _a8_runtime(Path(temporary))
        drift, runtime, before = self._drift(document)

        self.assertFalse(drift["valid"])
        self.assertEqual(drift["sha256"], hashlib.sha256(before[runtime]).hexdigest())
        self.assertEqual(
            {item["path"] for item in drift["refused"]},
            {
                # Retired whole: output repair is `agent_invocation`'s.
                "agent_continuation",
                "agent_invocation.exhaustion.singleton_action",
                "investigation_continuation",
            },
        )
        self.assertEqual(
            drift["stale_but_valid"],
            [
                {
                    "path": "agent_invocation.model_settings.max_tokens",
                    "value": 32768,
                    "default": 245760,
                    "default_source": "model_default",
                },
            ],
        )

    def test_a_setting_the_model_baseline_leaves_unset_is_a_product_default(self):
        # A recorded runtime file set `parallel_tool_calls: true`. The default
        # model's family baseline states it, so `false` is a model default;
        # `top_k` it leaves unset, so the settings class default applies.
        with tempfile.TemporaryDirectory() as temporary:
            document = _current_runtime(Path(temporary))
        document["agent_invocation"]["model_settings"] = {
            "parallel_tool_calls": True,
            "top_k": 40,
        }
        drift, _, _ = self._drift(document)

        self.assertTrue(drift["valid"])
        self.assertEqual(
            drift["stale_but_valid"],
            [
                {
                    "path": "agent_invocation.model_settings.parallel_tool_calls",
                    "value": True,
                    "default": False,
                    "default_source": "model_default",
                },
                {
                    "path": "agent_invocation.model_settings.top_k",
                    "value": 40,
                    "default": None,
                    "default_source": "product_default",
                },
            ],
        )

    def test_current_154_shaped_file_differs_only_in_labelled_per_run_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            document = _current_runtime(Path(temporary))
        drift, _, _ = self._drift(document)

        self.assertTrue(drift["valid"])
        self.assertEqual(drift["refused"], [])
        # The rest of `parabricks` is still not drift, but only its three
        # per-Run launch selections are labelled, by their dotted leaf paths.
        # The roster, heartbeat and tools have build defaults and are compared
        # like any other section.
        self.assertEqual(drift["stale_but_valid"], [])
        self.assertEqual(
            drift["per_run"],
            [
                "parabricks.launch.object_scope",
                "parabricks.launch.object_storage.bucket",
                "parabricks.launch.work_directory",
            ],
        )
        self.assertNotIn("per_run_values", drift)

    def test_malformed_sections_are_typed_refusals_not_crashes(self):
        for document in (
            {},
            {
                **runtime_configuration(),
                "agent_invocation": {"model_reference": ["qwen"]},
            },
        ):
            with self.subTest(document=document):
                drift, _, _ = self._drift(document)
                self.assertFalse(drift["valid"])
                self.assertTrue(drift["refused"])

    def test_an_empty_section_takes_its_defaults_and_is_not_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            document = _current_runtime(Path(temporary))
        document["agent_invocation"] = {}
        drift, _, _ = self._drift(document)

        self.assertTrue(drift["valid"])
        self.assertEqual(drift["stale_but_valid"], [])

    def test_unparseable_file_is_reported_refused_as_a_whole(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        runtime = root / "runtime.json"
        runtime.write_text("{not json", encoding="utf-8")

        code, stdout, _ = _invoke(
            runner.main, ["runtime-drift", "--runtime-config", str(runtime)]
        )

        self.assertEqual(code, 0)
        drift = json.loads(stdout)
        self.assertFalse(drift["valid"])
        self.assertEqual(drift["refused"], [{"path": "$", "reason": "unreadable"}])


class LiveReadinessTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(steady_clock())

    def test_unreadable_parent_is_refused_before_any_command_is_printed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            parent = root / "journey.jsonl"
            parent.write_text('{"not": "a journey"}\n', encoding="utf-8")

            code, stdout, stderr = _invoke(
                live_readiness.main,
                [
                    "--run-id",
                    "171-ready-a1",
                    "--parent-journey-file",
                    str(parent),
                ],
            )

            self.assertEqual((code, stdout), (2, ""))
            self.assertEqual(json.loads(stderr)["code"], "parent_journey_unreadable")

    def test_relative_parent_journey_file_is_read_at_its_resolved_absolute_path(self):
        class Received(Exception):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with (
                patch("os.getcwd", return_value=str(root / "work")),
                patch.object(
                    live_readiness, "read_parent_journey_identity", side_effect=Received
                ) as read,
                self.assertRaises(Received),
            ):
                _invoke(
                    live_readiness.main,
                    [
                        "--run-id",
                        "208-ready-a1",
                        "--parent-journey-file",
                        "../runs/./journey.jsonl",
                    ],
                )
        self.assertEqual(read.call_args.args[0], root / "runs" / "journey.jsonl")

    def test_closed_point_prints_the_native_contact_free_fork_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            parent = prerequisite("executed").journey_file
            retained = _tree(parent.parent)

            write_clock_record(root / "home")
            with patch.dict(
                "os.environ", {"BIO_NEB_CONTROLLER_HOME": str(root / "home")}
            ):
                code, stdout, stderr = _invoke(
                    live_readiness.main,
                    [
                        "--run-id",
                        "171-ready-a1",
                        "--parent-journey-file",
                        str(parent),
                    ],
                )

            self.assertEqual((code, stderr), (0, ""))
            readiness = json.loads(stdout)
            self.assertEqual(_tree(parent.parent), retained)
            self.assertTrue(readiness["runtime"]["valid"])
            self.assertEqual(
                {
                    name: item["present"]
                    for name, item in readiness["credentials"].items()
                },
                {"model_env": False, "controller_env": False},
            )
            steps = readiness["next_commands"]
            self.assertEqual([step["command"] for step in steps], ["fork", "observe"])
            self.assertEqual({step["contact"] for step in steps}, {"none"})
            self.assertEqual(readiness["entry"], "fork_point")
            head = json.loads(parent.read_bytes().splitlines()[-1])
            self.assertEqual(
                readiness["fork_point"]["point_event_id"], head["event_id"]
            )
            # Per-Run selection checks apply to a fresh begin only.
            self.assertEqual(readiness["researcher_decisions"], [])
            self.assertIn("investigation-begin", readiness["stop"])
            # An executed parent's view carries no registration; the tools are the
            # ones its prefix registered, which the child inherits.
            self.assertEqual(
                (
                    readiness["investigation_tools"]["section"],
                    {
                        tool["package"]
                        for tool in readiness["investigation_tools"]["tools"]
                    },
                ),
                ("named", {"pysam"}),
            )

            # The capsule's `start` reserves the fresh Run root the outputs go under.
            # Every printed command parses and runs as printed; only the child
            # Journey path, unknown until creation, comes from the prior receipt.
            Path(readiness["run_root"]).mkdir(parents=True)
            outputs = []
            for step in steps:
                argv = step["argv"]
                argv = argv[argv.index("bio_neb.cli") + 1 :]
                if step["command"] == "observe":
                    argv[-1] = outputs[-1]["child_journey_file"]
                code, stdout, stderr = _invoke(runner.main, argv)
                self.assertEqual((code, stderr), (0, ""), step["command"])
                outputs.append(json.loads(stdout))
            self.assertEqual(outputs[0]["status"], "fork_created")
            self.assertEqual(
                outputs[0]["origin"]["parent"],
                readiness["parent"]["identity"]["identity"]["journey"],
            )
            self.assertEqual(_tree(parent.parent), retained)

    def test_a_closed_point_without_a_clock_record_gets_no_fork_command(self):
        # The Fork is contact-free, but its child's next command is model-backed:
        # readiness withholds the chain until the clock check vouches for the clock.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            parent = prerequisite("executed").journey_file
            with patch.dict(
                "os.environ", {"BIO_NEB_CONTROLLER_HOME": str(root / "home")}
            ):
                code, stdout, stderr = _invoke(
                    live_readiness.main,
                    [
                        "--run-id",
                        "392-ready-a1",
                        "--parent-journey-file",
                        str(parent),
                    ],
                )

            self.assertEqual((code, stderr), (0, ""))
            readiness = json.loads(stdout)
            self.assertEqual(readiness["entry"], "fork_point")
            self.assertEqual(readiness["blockers"], ["clock_check_missing"])
            self.assertEqual(readiness["next_commands"], [])
            self.assertFalse(readiness["clock"]["present"])

    def test_an_earlier_point_is_read_from_its_prefix_when_later_facts_refuse(self):
        # A parent whose later facts no longer read on current code stays forkable
        # at an earlier closed point; readiness reads only the point's prefix, as
        # the child will.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            parent = prerequisite("executed").journey_file
            events = [json.loads(line) for line in parent.read_bytes().splitlines()]
            closed = [e["event_id"] for e in events if e["event_type"] == "PassClosed"]
            point, head = closed[-2], events[-1]["event_id"]
            read = runner.observe

            def observe_refusing_past_point(intent, *, journey=None):
                # The double stands for a later fact current code refuses.
                if journey is None or journey.latest_event()["event_id"] != point:
                    raise ValueError("a later fact no longer reads")
                return read(intent, journey=journey)

            write_clock_record(root / "home")
            with (
                patch.dict(
                    "os.environ", {"BIO_NEB_CONTROLLER_HOME": str(root / "home")}
                ),
                patch.object(
                    live_readiness, "observe", side_effect=observe_refusing_past_point
                ),
            ):
                code, stdout, stderr = _invoke(
                    live_readiness.main,
                    [
                        "--run-id",
                        "105-ready-a1",
                        "--parent-journey-file",
                        str(parent),
                        "--point-event-id",
                        point,
                    ],
                )
                refused = _invoke(
                    live_readiness.main,
                    [
                        "--run-id",
                        "105-ready-a2",
                        "--parent-journey-file",
                        str(parent),
                        "--point-event-id",
                        head,
                    ],
                )

            self.assertEqual((code, stderr), (0, ""))
            readiness = json.loads(stdout)
            self.assertEqual(readiness["entry"], "fork_point")
            self.assertEqual(readiness["fork_point"]["point_event_id"], point)
            self.assertEqual(
                [step["command"] for step in readiness["next_commands"]],
                ["fork", "observe"],
            )
            self.assertIsNotNone(readiness["parent"]["observed"]["view"])
            # A point past the refused fact still refuses as before.
            self.assertEqual(refused[0], 2)
            self.assertEqual(
                json.loads(refused[2])["code"], "parent_journey_unreadable"
            )

    def test_a_fork_given_a_runtime_file_is_refused_before_any_read(self):
        # The child runs on its parent's registered selections, so no file can
        # refuse it; readiness refuses one named for a Fork instead (formerly
        # a refused file blocked the fork chain).
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            parent = prerequisite("executed").journey_file
            runtime = root / "runtime.json"
            runtime.write_text(json.dumps(_current_runtime(root)), encoding="utf-8")
            write_clock_record(root / "home")
            with (
                patch.dict(
                    "os.environ", {"BIO_NEB_CONTROLLER_HOME": str(root / "home")}
                ),
                patch.object(
                    live_readiness,
                    "read_parent_journey_identity",
                    side_effect=AssertionError("parent read"),
                ),
                redirect_stderr(io.StringIO()) as errors,
                redirect_stdout(io.StringIO()) as output,
                self.assertRaises(SystemExit) as refused,
            ):
                live_readiness.main(
                    [
                        "--run-id",
                        "105-ready-b1",
                        "--runtime-config",
                        str(runtime),
                        "--parent-journey-file",
                        str(parent),
                    ]
                )
        self.assertEqual((refused.exception.code, output.getvalue()), (2, ""))
        self.assertIn("omit --runtime-config", errors.getvalue())

    def test_open_pass_head_gets_no_fork_command_and_a_review_wait_does(self):
        # A Run waiting at its Execution Review ends with a closed Pass: the
        # before-approval child point. The same prefix cut inside that Pass,
        # as a crash leaves it, has no closed point to fork.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            registration, review, _ = prepare_configured_run(
                source, configuration=execution_configuration(source)
            )
            self.assertIsNotNone(review.execution_hitl.pending_request)
            waiting = Path(registration.journey_file)
            cut = root / "cut" / waiting.parent.name / "journey.jsonl"
            shutil.copytree(waiting.parent, cut.parent)
            lines = waiting.read_bytes().splitlines(keepends=True)
            self.assertEqual(json.loads(lines[-1])["event_type"], "PassClosed")
            # Cut inside the last Pass, like a recorded prefix: its open journal
            # holds rows anchored after the cut, so the whole-Run read refuses it.
            cut.write_bytes(b"".join(lines[:-40]))
            self.assertNotIn(
                "Pass", json.loads(cut.read_bytes().splitlines()[-1])["event_type"]
            )
            with self.assertRaisesRegex(ValueError, "exact semantic prefix"):
                runner.observe(runner.ObserveRun(journey_file=str(cut)))
            retained = _tree(root / "cut"), _tree(waiting.parent)

            readiness = {}
            for name, parent in (("waiting", waiting), ("cut", cut)):
                write_clock_record(root / "home")
                with patch.dict(
                    "os.environ", {"BIO_NEB_CONTROLLER_HOME": str(root / "home")}
                ):
                    code, stdout, stderr = _invoke(
                        live_readiness.main,
                        [
                            "--run-id",
                            "211-ready-a1",
                            "--parent-journey-file",
                            str(parent),
                        ],
                    )
                self.assertEqual((code, stderr), (0, ""), name)
                readiness[name] = json.loads(stdout)

            self.assertEqual((_tree(root / "cut"), _tree(waiting.parent)), retained)
            self.assertEqual(readiness["waiting"]["entry"], "fork_point")
            self.assertEqual(
                readiness["waiting"]["fork_point"]["point_disposition"], "waiting"
            )
            self.assertEqual(
                [step["command"] for step in readiness["waiting"]["next_commands"]],
                ["fork", "observe"],
            )
            refused = readiness["cut"]
            self.assertEqual(refused["entry"], "parent_ineligible")
            self.assertFalse(refused["ready"])
            self.assertEqual(refused["blockers"], ["fork_point_not_closed"])
            self.assertIsNone(refused["parent"]["observed"]["view"])
            self.assertTrue(
                refused["parent"]["observed"]["cause"].startswith("ValueError@")
            )
            self.assertEqual(refused["next_commands"], [])
            self.assertEqual(
                refused["parent"]["identity"]["identity"]["journey"]["journey_sha256"],
                hashlib.sha256(cut.read_bytes()).hexdigest(),
            )

    def test_fresh_begin_refused_runtime_prints_no_model_backed_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            readiness, _ = _fresh_begin_readiness(
                root, "171-ready-a2", _a8_runtime(root)
            )

            self.assertFalse(readiness["ready"])
            self.assertIn("runtime_config_refused", readiness["blockers"])
            self.assertEqual(readiness["next_commands"], [])
            self.assertTrue(readiness["credentials"]["model_env"]["present"])

    def test_fresh_begin_with_an_unsupported_fastq_role_set_prints_no_command(self):
        # Begin refuses any role set but one forward/reverse pair before a
        # Journey exists; readiness names that blocker without running begin.
        for name, roles in (
            ("single_end", ("reads",)),
            ("two_lanes", ("read_1", "read_2", "lane2_read_1", "lane2_read_2")),
        ):
            with (
                self.subTest(case=name),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary).resolve()
                readiness, _ = _fresh_begin_readiness(
                    root, "342-roles-a1", _current_runtime(root), roles=roles
                )

                self.assertFalse(readiness["ready"])
                self.assertEqual(readiness["blockers"], ["fastq_roles_unsupported"])
                self.assertEqual(readiness["next_commands"], [])

    def test_fresh_begin_reports_the_tool_packages_the_locked_environment_provides(
        self,
    ):
        # The Run probes its named tools at its first grant; readiness reads only
        # package metadata (no tool runs: `_invoke` refuses any process).
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            readiness, _ = _fresh_begin_readiness(
                root, "107-tools-a1", _current_runtime(root)
            )

            self.assertTrue(readiness["ready"])
            self.assertEqual(
                readiness["investigation_tools"],
                {
                    "section": "named",
                    "tools": [
                        {
                            "name": name,
                            "package": "pysam",
                            "package_version": "0.24.1",
                            "installed_version": "0.24.1",
                            "provided": True,
                        }
                        for name in ("bcftools", "samtools")
                    ],
                },
            )

    def test_fresh_begin_without_its_tool_section_or_package_prints_no_command(
        self,
    ):
        # A section stated empty or a package version this checkout lacks would hold
        # the Run at its first Investigation grant; readiness blocks the begin. An
        # omitted section takes the build's tools.
        unpinned = {"package": "pysam", "package_version": "0.23.0"}
        for name, change, blocker in (
            (
                "stated_none",
                lambda runtime: runtime.update(investigation_tools=None),
                None,
            ),
            (
                "unavailable",
                lambda runtime: runtime.update(
                    investigation_tools={"bcftools": unpinned, "samtools": unpinned}
                ),
                "investigation_tools_unavailable",
            ),
        ):
            with (
                self.subTest(case=name),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary).resolve()
                runtime = _current_runtime(root)
                change(runtime)
                readiness, _ = _fresh_begin_readiness(root, "107-tools-a2", runtime)

                self.assertFalse(readiness["ready"])
                self.assertEqual(
                    readiness["blockers"], [blocker or "investigation_tools_missing"]
                )
                self.assertEqual(readiness["next_commands"], [])
                tools = readiness["investigation_tools"]
                if blocker is None:
                    self.assertEqual(tools, {"section": "absent", "tools": []})
                    continue
                self.assertEqual(
                    [
                        (tool["installed_version"], tool["provided"])
                        for tool in tools["tools"]
                    ],
                    [("0.24.1", False), ("0.24.1", False)],
                )

    def test_fresh_begin_prints_begin_with_the_model_credential_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            readiness, home = _fresh_begin_readiness(
                root, "171-ready-a3", _current_runtime(root)
            )

            self.assertTrue(readiness["ready"])
            [begin] = readiness["next_commands"]
            self.assertEqual((begin["command"], begin["contact"]), ("begin", "model"))
            self.assertEqual(begin["credential_file"], str(home / "model.env"))
            argv = begin["argv"]
            self.assertIn("--interactive", argv)
            self.assertEqual(
                argv[argv.index("--run-root") + 1],
                str(home / "live-runs" / "171-ready-a3"),
            )
            self.assertFalse((home / "live-runs" / "171-ready-a3").exists())

    def test_a_missing_stale_or_stepped_clock_record_blocks_the_begin(self):
        # Readiness never contacts the time source; it reads the clock check's record
        # and blocks a live step on one that cannot vouch for the clock.
        for name, clock, blocker in (
            ("missing", {"absent": True}, "clock_check_missing"),
            ("stale", {"age_seconds": 11 * 60}, "clock_check_stale"),
            ("from_the_future", {"age_seconds": -60}, "clock_check_stale"),
            ("behind", {"offset_seconds": 103.0}, "clock_offset_beyond_bound"),
            ("ahead", {"offset_seconds": -5.5}, "clock_offset_beyond_bound"),
            # The check's watch saw the wall clock step.
            ("stepping", {"largest_step_seconds": 111.0}, "clock_offset_beyond_bound"),
            # The wall clock moved against the boot clock after the check.
            (
                "stepped_since",
                {"stepped_since_seconds": -110.6},
                "clock_stepped_since_check",
            ),
        ):
            with (
                self.subTest(case=name),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary).resolve()
                readiness, home = _fresh_begin_readiness(
                    root, "392-clock-a1", _current_runtime(root), clock=clock
                )

                self.assertFalse(readiness["ready"])
                self.assertEqual(readiness["blockers"], [blocker])
                self.assertEqual(readiness["next_commands"], [])
                self.assertEqual(
                    readiness["clock"]["record"], str(home / "clock-check.json")
                )
                self.assertEqual(
                    readiness["clock"]["command"][-2:],
                    ["-m", "bio_neb.devtools.clock_check"],
                )

    def test_a_clock_record_in_the_warn_band_reads_warn_and_does_not_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            readiness, _ = _fresh_begin_readiness(
                root,
                "392-clock-a2",
                _current_runtime(root),
                clock={"offset_seconds": 3.0, "age_seconds": 60},
            )

            self.assertTrue(readiness["ready"], readiness["blockers"])
            self.assertEqual(readiness["clock"]["verdict"], "warn")
            self.assertEqual(readiness["clock"]["offset_seconds"], 3.0)
            self.assertEqual(readiness["clock"]["age_seconds"], 60)
            self.assertEqual(
                [step["command"] for step in readiness["next_commands"]], ["begin"]
            )


def _fresh_begin_readiness(
    root: Path,
    run_id: str,
    runtime: dict | None,
    *,
    request_files: tuple[Path, ...] = (),
    paper_files: tuple[Path, ...] = (),
    roles: tuple[str, ...] = ("read_1", "read_2"),
    clock: dict | None = None,
    credential_names: tuple[str, ...] = ("LOCAL_MODEL_KEY", "NEBIUS_API_KEY"),
) -> tuple[dict, Path]:
    """Readiness for a fresh begin with present inputs and a model credential file.

    The credential file assigns `credential_names` (names only, #440). A `None`
    runtime passes no `--runtime-config`. The controller home holds a fresh, in-bound
    clock record unless `clock` names the record's `offset_seconds` and
    `age_seconds`, or is `{"absent": True}`.
    """
    path = root / "runtime.json"
    if runtime is not None:
        path.write_text(json.dumps(runtime), encoding="utf-8")
    prompt = root / "prompt.txt"
    prompt.write_text("Analyze HG001.", encoding="utf-8")
    fastqs = []
    for role in roles:
        reads = root / f"{role}.fastq"
        reads.write_bytes(b"@r\nACGT\n+\n!!!!\n")
        fastqs += ["--fastq", f"{role}={reads}"]
    home = root / "home"
    home.mkdir(exist_ok=True)
    (home / "model.env").write_text(
        "".join(f"{name}=\n" for name in credential_names), encoding="utf-8"
    )
    clock = clock or {}
    if not clock.get("absent"):
        write_clock_record(home, **clock)
    with patch.dict("os.environ", {"BIO_NEB_CONTROLLER_HOME": str(home)}):
        code, stdout, stderr = _invoke(
            live_readiness.main,
            [
                "--run-id",
                run_id,
                *(() if runtime is None else ("--runtime-config", str(path))),
                "--prompt-file",
                str(prompt),
                *fastqs,
                *(
                    part
                    for request in request_files
                    for part in ("--request-file", str(request))
                ),
                *(
                    part
                    for paper in paper_files
                    for part in ("--paper-file", str(paper))
                ),
            ],
        )
    assert (code, stderr) == (0, ""), (code, stderr)
    return json.loads(stdout), home


class ReadinessResearcherDecisionTests(unittest.TestCase):
    """#209: a fresh begin names per-Run launch selections that belong to another Run."""

    RUN_ID = "209-check-20261009T120000Z-b08c4f34-a1"

    def setUp(self):
        self.enterContext(steady_clock())
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        # Readiness puts Run roots under <controller home>/live-runs.
        self.live_runs = self.root / "home" / "live-runs"

    def _readiness(
        self,
        run_id: str,
        *,
        bucket: str,
        scope: str,
        work_directory: Path | str,
    ) -> dict:
        runtime = _current_runtime(self.root)
        launch = runtime["parabricks"]["launch"]
        launch["object_storage"]["bucket"] = bucket
        launch["object_scope"] = scope
        launch["work_directory"] = str(work_directory)
        readiness, home = _fresh_begin_readiness(self.root, run_id, runtime)
        self.assertEqual(home / "live-runs", self.live_runs)
        return readiness

    def _own(self) -> dict[str, object]:
        """Synthetic selections prepared for RUN_ID, in the #205 a3 naming."""
        return {
            "bucket": "bio-neb-s209-20261009-a1",
            "scope": "issue209/20261009-a1",
            "work_directory": self.live_runs
            / self.RUN_ID
            / "runs"
            / "execution"
            / "work",
        }

    def test_copied_runtime_with_another_runs_selections_names_each_as_a_decision(
        self,
    ):
        # Synthetic values in a recorded shape: a donor runtime copied unchanged.
        foreign = (
            self.live_runs
            / "205-plan-20261009T0825Z-136e28bb-a2"
            / "runs"
            / "execution"
            / "work"
        )
        readiness = self._readiness(
            self.RUN_ID,
            bucket="bio-neb-s205-20261009-a2",
            scope="issue205/20261009-a2",
            work_directory=foreign,
        )

        self.assertEqual(
            readiness["researcher_decisions"],
            [
                {
                    "path": "parabricks.launch.object_scope",
                    "value": "issue205/20261009-a2",
                    "reason": "object_scope_lacks_run_token",
                },
                {
                    "path": "parabricks.launch.object_storage.bucket",
                    "value": "bio-neb-s205-20261009-a2",
                    "reason": "bucket_lacks_run_token",
                },
                {
                    "path": "parabricks.launch.work_directory",
                    "value": str(foreign),
                    "reason": "work_directory_outside_run_root",
                },
            ],
        )
        # A researcher decision, not a blocker: readiness and its command stand.
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["blockers"], [])
        self.assertEqual(
            [step["command"] for step in readiness["next_commands"]], ["begin"]
        )
        self.assertFalse((self.live_runs / self.RUN_ID).exists())

    def test_runtime_prepared_for_this_run_raises_no_decision(self):
        readiness = self._readiness(self.RUN_ID, **self._own())

        self.assertEqual(readiness["researcher_decisions"], [])
        self.assertTrue(readiness["ready"])

    def test_token_counts_only_as_a_whole_ticket_number_and_attempt_segment(self):
        bucket = ("parabricks.launch.object_storage.bucket", "bucket_lacks_run_token")
        scope = ("parabricks.launch.object_scope", "object_scope_lacks_run_token")
        work = (
            "parabricks.launch.work_directory",
            "work_directory_outside_run_root",
        )
        for run_id, changed, expected in (
            (self.RUN_ID, {"bucket": "bio-neb-s209-20261009-a10"}, bucket),
            (self.RUN_ID, {"bucket": "bio-neb-s2090-20261009-a1"}, bucket),
            (self.RUN_ID, {"scope": "issue209/20261009"}, scope),
            (self.RUN_ID, {"scope": "issue205/20261009-a1"}, scope),
            (
                self.RUN_ID,
                {"work_directory": f"{self.live_runs}/{self.RUN_ID}/../other"},
                work,
            ),
            # The attempt's own digits are not the ticket number.
            (
                "1-check-20261009T120000Z-b08c4f34-a1",
                {
                    "bucket": "bio-neb-x-a1",
                    "scope": "issue1/20261009-a1",
                    "work_directory": self.live_runs
                    / "1-check-20261009T120000Z-b08c4f34-a1",
                },
                bucket,
            ),
        ):
            with self.subTest(run_id=run_id, changed=changed):
                readiness = self._readiness(run_id, **{**self._own(), **changed})
                self.assertEqual(
                    [
                        (item["path"], item["reason"])
                        for item in readiness["researcher_decisions"]
                    ],
                    [expected],
                )

    def test_unparsed_run_id_is_itself_a_decision(self):
        readiness = self._readiness(
            "209-check-a1",
            bucket="bio-neb-s209-20261009-a1",
            scope="issue209/20261009-a1",
            work_directory="/elsewhere/execution/work",
        )

        # Without a parsed token, bucket and scope cannot be checked; the work
        # directory still can.
        self.assertEqual(
            readiness["researcher_decisions"],
            [
                {
                    "path": "run_id",
                    "value": "209-check-a1",
                    "reason": "run_id_unparsed",
                },
                {
                    "path": "parabricks.launch.work_directory",
                    "value": "/elsewhere/execution/work",
                    "reason": "work_directory_outside_run_root",
                },
            ],
        )
        self.assertTrue(readiness["ready"])


class ReadinessParabricksAccountTests(unittest.TestCase):
    """#454: a fresh begin with no Parabricks section reports the account file."""

    RUN_ID = "454-live-20261024T112855Z-ac17e721-a1"

    def setUp(self):
        self.enterContext(steady_clock())
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.home = self.root / "home"
        self.home.mkdir()
        self.account_file = self.home / "parabricks-account.json"
        self.account = parabricks_account(self.root)

    def test_a_valid_account_file_names_the_runs_own_leaves(self):
        content = json.dumps(self.account).encode()
        self.account_file.write_bytes(content)

        readiness, _ = _fresh_begin_readiness(self.root, self.RUN_ID, None)

        token = "454-live-20261024t112855z-ac17e721-a1"
        self.assertEqual(
            readiness["parabricks_account"],
            {
                "path": str(self.account_file),
                "present": True,
                "byte_count": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "valid": True,
                "code": None,
                "refused": [],
                "per_run": {
                    "parabricks.launch.object_scope": token,
                    "parabricks.launch.object_storage.bucket": f"bio-neb-{token}",
                    "parabricks.launch.work_directory": str(
                        self.home
                        / "live-runs"
                        / self.RUN_ID
                        / "runs"
                        / "execution"
                        / "work"
                    ),
                },
            },
        )
        self.assertEqual(readiness["researcher_decisions"], [])
        self.assertTrue(readiness["ready"])

    def test_a_missing_or_invalid_account_file_is_a_decision_not_a_blocker(self):
        extra = {**self.account, "unexpected": 1}
        for name, content, code, refused in (
            ("absent", None, "parabricks_account_missing", ["$"]),
            ("unreadable", b"{", "parabricks_account_unreadable", ["$"]),
            (
                "invalid",
                json.dumps(extra).encode(),
                "parabricks_account_invalid",
                ["unexpected"],
            ),
        ):
            with self.subTest(name):
                if content is None:
                    self.account_file.unlink(missing_ok=True)
                else:
                    self.account_file.write_bytes(content)

                readiness, _ = _fresh_begin_readiness(self.root, self.RUN_ID, None)

                report = readiness["parabricks_account"]
                self.assertEqual(
                    (report["present"], report["valid"], report["code"]),
                    (content is not None, False, code),
                )
                self.assertEqual(report["refused"], refused)
                self.assertEqual(report["per_run"], {})
                self.assertEqual(
                    readiness["researcher_decisions"],
                    [
                        {
                            "path": "parabricks_account",
                            "value": str(self.account_file),
                            "reason": code,
                        }
                    ],
                )
                self.assertTrue(readiness["ready"])

    def test_an_account_runtime_that_is_not_this_build_blocks_readiness(self):
        """#464 live a2 F4: an account whose runtime image states another worker (here
        request schema /6) would start a paid job the build's request cannot use;
        readiness blocks before `begin`, not after the job starts."""
        from bio_neb.content import canonical_json_digest

        text = json.dumps(self.account).replace(
            "bio-neb.parabricks-worker-request/7", "bio-neb.parabricks-worker-request/6"
        )
        capability = json.loads(text)["runtime_selection"]["capability"]
        text = text.replace(
            capability["snapshot_id"],
            canonical_json_digest(
                {k: v for k, v in capability.items() if k != "snapshot_id"}
            ),
        )
        self.account_file.write_text(text, encoding="utf-8")

        readiness, _ = _fresh_begin_readiness(self.root, self.RUN_ID, None)

        self.assertEqual(
            readiness["parabricks_account"]["code"], "runtime_not_this_build"
        )
        self.assertIn("runtime_not_this_build", readiness["blockers"])
        self.assertFalse(readiness["ready"])

    def test_a_long_run_id_raises_no_decision_for_the_leaves_begin_derives(self):
        # The shortened bucket loses the `a<N>` tail; begin derived it from this
        # Run, so it is reported, not suspected of being another Run's.
        run_id = "454-" + "long-purpose-" * 4 + "20261024T112855Z-ac17e721-a1"
        self.account_file.write_text(json.dumps(self.account), encoding="utf-8")

        readiness, _ = _fresh_begin_readiness(self.root, run_id, None)

        bucket = readiness["parabricks_account"]["per_run"][
            "parabricks.launch.object_storage.bucket"
        ]
        self.assertLessEqual(len(bucket), 63)
        self.assertNotIn("a1", bucket.split("-"))
        self.assertEqual(readiness["researcher_decisions"], [])
        self.assertTrue(readiness["ready"])

    def test_a_runtime_files_section_leaves_the_account_file_unread(self):
        self.account_file.write_text(json.dumps(self.account), encoding="utf-8")
        touched = []
        read_bytes, is_file = Path.read_bytes, Path.is_file

        def watched(method):
            def call(path, *args, **kwargs):
                if path == self.account_file:
                    touched.append(method.__name__)
                return method(path, *args, **kwargs)

            return call

        with (
            patch.object(Path, "read_bytes", watched(read_bytes)),
            patch.object(Path, "is_file", watched(is_file)),
        ):
            readiness, _ = _fresh_begin_readiness(
                self.root, self.RUN_ID, _current_runtime(self.root)
            )

        self.assertIsNone(readiness["parabricks_account"])
        self.assertEqual(touched, [])


class ReadinessArgvParsesTests(RunPrerequisites, unittest.TestCase):
    """Every command line readiness prints parses with `bio_neb.cli`'s own parser.

    Rule owner: docs/agents/coding.md#migration-and-proof (native CLI consumers agree).
    """

    def setUp(self):
        super().setUp()
        self.enterContext(steady_clock())

    def _assert_parses(self, step: dict) -> None:
        argv = step["argv"][step["argv"].index("bio_neb.cli") + 1 :]
        parser = runner._parser()
        # The CLI accepts unambiguous abbreviations; a printed flag must be exact.
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for command in action.choices.values():
                    command.allow_abbrev = False
        stderr = io.StringIO()
        try:
            with redirect_stderr(stderr):
                parsed = parser.parse_args(argv)
        except SystemExit:
            self.fail(f"{step['command']} argv does not parse: {stderr.getvalue()}")
        self.assertEqual(parsed.command, step["command"])

    def test_fork_chain_and_fresh_begin_argv_parse_as_printed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            parent = self.prerequisite(
                "executed"
            ).journey_file  # readiness only reads it
            head = json.loads(parent.read_bytes().splitlines()[-1])["event_id"]
            write_clock_record(root / "home")
            with patch.dict(
                "os.environ", {"BIO_NEB_CONTROLLER_HOME": str(root / "home")}
            ):
                code, stdout, stderr = _invoke(
                    live_readiness.main,
                    [
                        "--run-id",
                        "236-argv-a1",
                        "--parent-journey-file",
                        str(parent),
                        "--point-event-id",
                        head,
                    ],
                )
            self.assertEqual((code, stderr), (0, ""))
            fork_steps = json.loads(stdout)["next_commands"]

            # Several request files, each printed in order.
            requests = (root / "list.json", root / "paper.json")
            for request in requests:
                request.write_text("{}", encoding="utf-8")
            # A supplied paper file is passed as given.
            paper = root / "paper.pdf"
            paper.write_bytes(b"%PDF-1.4")
            begin = _fresh_begin_readiness(
                root,
                "236-argv-a2",
                _current_runtime(root),
                request_files=requests,
                paper_files=(paper,),
            )[0]["next_commands"]
            # With no runtime file the begin takes the build's defaults.
            defaults = _fresh_begin_readiness(root, "236-argv-a3", None)[0][
                "next_commands"
            ]

        steps = [*fork_steps, *begin, *defaults]
        self.assertEqual(
            [step["command"] for step in steps], ["fork", "observe", "begin", "begin"]
        )
        self.assertIn("--runtime-config", begin[0]["argv"])
        self.assertNotIn("--runtime-config", defaults[0]["argv"])
        self.assertIn("--point-event-id", fork_steps[0]["argv"])
        argv = begin[0]["argv"]
        self.assertEqual(
            [argv[i + 1] for i, part in enumerate(argv) if part == "--request-file"],
            [str(request) for request in requests],
        )
        self.assertEqual(
            [argv[i + 1] for i, part in enumerate(argv) if part == "--paper-file"],
            [str(paper)],
        )
        for step in steps:
            with self.subTest(command=step["command"]):
                self._assert_parses(step)
        # The capsule starts a credentialed command only in its own form, so the
        # printed begin is startable as printed (F1).
        from bio_neb.devtools import native_ubuntu_operator as operator

        for step in (*begin, *defaults):
            with self.subTest(capsule=step["argv"]):
                operator._validate_credential_argv(
                    step["argv"], Path(step["credential_file"])
                )


if __name__ == "__main__":
    unittest.main()
