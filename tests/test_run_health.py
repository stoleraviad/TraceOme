"""The run-health runner runs one table entry through one port and records what ran.

An entry is held either to its pinned package or to the Run's tool probe (main's
`ObservationToolsProbe`); the record keeps output digests and only what the probe lacks.
"""

import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bio_neb.agentic_journey.runtime_config import InvestigationTools, ObservationTool
from bio_neb.investigation.observation_defaults import CoverageDefaults
from bio_neb.investigation.observation_tools import (
    RECEIPT_NAME,
    ObservationToolsReceipt,
    probe_observation_tools,
)
from bio_neb.providers.nebius_object_storage import StorageTransferError
from tests.proposal_information_fakes import runtime_configuration
from tests.reference_fakes import SOURCE_PLAIN
from tests.run_health_fakes import credential_receipt, kept_probe
from tests.run_prerequisites import prerequisite
from tools.run_health import (
    PINS,
    TOOLS,
    DockerHostRunner,
    Invocation,
    LocalProcessRunner,
    Mount,
    OutputFile,
    Pin,
    ProbedProcessRunner,
    RunRefused,
    RunResult,
    ToolEntry,
    coverage_entry,
    pinned_binding,
    probe_binding,
    run_tool,
)
from tools.run_health.__main__ import main
from tools.run_health.cache import cache_run_files
from tools.run_health.table import indexcov_entry

PINNED = "sha256:" + "a" * 64


def _pin(**pinned):
    return Pin(package="samtools", version="1.24", **{"image_digest": PINNED} | pinned)


def _stats_entry(**pinned):
    return ToolEntry(
        pin=_pin(**pinned),
        argv=("samtools", "stats", "{bam}"),
        inputs=("bam",),
        stdout="stats.txt",
        outputs=("stats.txt",),
    )


class FakeRunner:
    """Runner port fake: records invocations, writes scripted outputs, reports what ran."""

    name = "fake"
    identity_fields = ("image_digest",)

    def __init__(self, *, ran=PINNED, exit_code=0, stdout=b"", stderr=b"", files=None):
        self.ran = ran
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.files = files or {}
        self.invocations = []

    def bind(self, entry):
        return pinned_binding(entry, runner=self.name, fields=self.identity_fields)

    def run(self, invocation):
        self.invocations.append(invocation)
        if invocation.stdout_path is not None:
            invocation.stdout_path.write_bytes(self.stdout)
        (outputs,) = [mount for mount in invocation.mounts if not mount.read_only]
        for name, payload in self.files.items():
            (outputs.host_path / name).write_bytes(payload)
        return RunResult(
            exit_code=self.exit_code,
            identity={"image_digest": self.ran},
            executed_argv=("fake-run", *invocation.argv),
            stderr=self.stderr,
        )


class FakeProbedRunner:
    """Runner port fake for probed entries: reports the probe's facts, or `ran` instead."""

    name = "fake-probed"

    def __init__(self, probe, *, ran=None, stdout=b"", files=None):
        self.probe = probe
        self.ran = ran
        self.stdout = stdout
        self.files = files or {}
        self.invocations = []

    def bind(self, entry):
        return probe_binding(entry, self.probe, runner=self.name)

    def run(self, invocation):
        self.invocations.append(invocation)
        if invocation.stdout_path is not None:
            invocation.stdout_path.write_bytes(self.stdout)
        (outputs,) = [mount for mount in invocation.mounts if not mount.read_only]
        for name, payload in self.files.items():
            (outputs.host_path / name).write_bytes(payload)
        tool = self.probe.receipt.tool(invocation.probed)
        return RunResult(
            exit_code=0,
            identity={
                "executable_sha256": self.ran or tool.executable_sha256,
                "tool_version": tool.tool_version,
            },
            executed_argv=(tool.executable, *invocation.argv[1:]),
            stderr=b"",
        )


class RunToolTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bam = self.root / "run" / "aligned.bam"
        self.bam.parent.mkdir()
        self.bam.write_bytes(b"BAM\x01")

    def test_a_run_records_argv_image_digest_and_each_output_digest(self):
        stats = b"SN\traw total sequences:\t2\n"
        runner = FakeRunner(stdout=stats, files={"extra.txt": b"x"})

        record = run_tool(
            "samtools_stats",
            {"bam": self.bam},
            self.root / "health",
            runner=runner,
            tools={"samtools_stats": _stats_entry()},
        )

        self.assertEqual(record.tool, "samtools_stats")
        self.assertEqual(record.runner, "fake")
        self.assertEqual(record.argv, ("samtools", "stats", "/inputs/aligned.bam"))
        self.assertEqual(
            record.executed_argv,
            ("fake-run", "samtools", "stats", "/inputs/aligned.bam"),
        )
        self.assertEqual(record.image_reference, "quay.io/biocontainers/samtools:1.24")
        self.assertEqual(record.identity, {"image_digest": PINNED})
        self.assertEqual(record.inputs, {"bam": str(self.bam)})
        self.assertEqual(
            record.outputs,
            {
                "stats.txt": OutputFile(
                    sha256=hashlib.sha256(stats).hexdigest(), byte_count=len(stats)
                ),
                "extra.txt": OutputFile(
                    sha256=hashlib.sha256(b"x").hexdigest(), byte_count=1
                ),
            },
        )
        (invocation,) = runner.invocations
        self.assertEqual(invocation.pin, _pin())
        self.assertEqual(
            {(m.host_path, m.container_path, m.read_only) for m in invocation.mounts},
            {
                (self.bam, "/inputs/aligned.bam", True),
                (self.root / "health", "/outputs", False),
            },
        )

    def test_a_digest_other_than_the_pin_is_refused_with_both_digests(self):
        ran = "sha256:" + "b" * 64

        with self.assertRaises(RunRefused) as refused:
            run_tool(
                "samtools_stats",
                {"bam": self.bam},
                self.root / "health",
                runner=FakeRunner(ran=ran, stdout=b"stats"),
                tools={"samtools_stats": _stats_entry()},
            )

        self.assertEqual(refused.exception.code, "identity_mismatch")
        self.assertIn(
            f"image_digest: pinned {PINNED}, ran {ran}", refused.exception.detail
        )

    def test_a_pin_the_runner_needs_but_lacks_is_refused_before_it_is_called(self):
        runner = FakeRunner()

        with self.assertRaises(RunRefused) as refused:
            run_tool(
                "samtools_stats",
                {"bam": self.bam},
                self.root / "health",
                runner=runner,
                tools={"samtools_stats": _stats_entry(image_digest=None)},
            )

        self.assertEqual(refused.exception.code, "not_pinned")
        self.assertIn("samtools 1.24 has no image_digest", refused.exception.detail)
        self.assertEqual(runner.invocations, [])
        self.assertFalse((self.root / "health").exists())

    def test_a_tool_that_exits_nonzero_is_refused_with_its_exit_code_and_stderr(self):
        runner = FakeRunner(exit_code=1, stderr=b"[E::hts_open] fail to open file")

        with self.assertRaises(RunRefused) as refused:
            run_tool(
                "samtools_stats",
                {"bam": self.bam},
                self.root / "health",
                runner=runner,
                tools={"samtools_stats": _stats_entry()},
            )

        self.assertEqual(refused.exception.code, "tool_failed")
        self.assertIn("exit 1", refused.exception.detail)
        self.assertIn("[E::hts_open] fail to open file", refused.exception.detail)

    def test_a_run_whose_image_never_ran_is_refused_as_the_failure_it_reports(self):
        runner = FakeRunner(
            ran=None, exit_code=125, stderr=b"pull access denied for samtools"
        )

        with self.assertRaises(RunRefused) as refused:
            run_tool(
                "samtools_stats",
                {"bam": self.bam},
                self.root / "health",
                runner=runner,
                tools={"samtools_stats": _stats_entry()},
            )

        self.assertEqual(refused.exception.code, "tool_failed")
        self.assertIn("exit 125", refused.exception.detail)
        self.assertIn("pull access denied", refused.exception.detail)

    def test_a_required_output_the_tool_did_not_write_is_refused(self):
        entry = _stats_entry().model_copy(
            update={"stdout": None, "outputs": ("stats.txt", "summary.txt")}
        )

        with self.assertRaises(RunRefused) as refused:
            run_tool(
                "samtools_stats",
                {"bam": self.bam},
                self.root / "health",
                runner=FakeRunner(files={"stats.txt": b"stats"}),
                tools={"samtools_stats": entry},
            )

        self.assertEqual(refused.exception.code, "output_missing")
        self.assertIn("summary.txt", refused.exception.detail)
        self.assertNotIn("stats.txt", refused.exception.detail)

    def test_inputs_other_than_the_entry_names_are_refused_before_contact(self):
        for inputs in ({}, {"bam": self.bam, "vcf": self.bam}):
            with self.subTest(inputs=sorted(inputs)):
                runner = FakeRunner()

                with self.assertRaises(RunRefused) as refused:
                    run_tool(
                        "samtools_stats",
                        inputs,
                        self.root / "health",
                        runner=runner,
                        tools={"samtools_stats": _stats_entry()},
                    )

                self.assertEqual(refused.exception.code, "inputs_mismatch")
                self.assertIn("['bam']", refused.exception.detail)
                self.assertEqual(runner.invocations, [])
                self.assertFalse((self.root / "health").exists())

    def test_an_existing_output_directory_is_refused_before_contact(self):
        (self.root / "health").mkdir()
        (self.root / "health" / "stats.txt").write_bytes(b"earlier")
        runner = FakeRunner()

        with self.assertRaises(RunRefused) as refused:
            run_tool(
                "samtools_stats",
                {"bam": self.bam},
                self.root / "health",
                runner=runner,
                tools={"samtools_stats": _stats_entry()},
            )

        self.assertEqual(refused.exception.code, "output_dir_exists")
        self.assertEqual(runner.invocations, [])
        self.assertEqual((self.root / "health" / "stats.txt").read_bytes(), b"earlier")

    def test_missing_parents_of_the_output_directory_are_created(self):
        record = run_tool(
            "samtools_stats",
            {"bam": self.bam},
            self.root / "runs" / "a1" / "health",
            runner=FakeRunner(stdout=b"stats"),
            tools={"samtools_stats": _stats_entry()},
        )

        self.assertEqual(set(record.outputs), {"stats.txt"})
        self.assertTrue((self.root / "runs" / "a1" / "health" / "stats.txt").is_file())

    def test_relative_paths_are_mounted_and_recorded_as_absolute_paths(self):
        cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, cwd)
        runner = FakeRunner(stdout=b"stats")

        record = run_tool(
            "samtools_stats",
            {"bam": Path("run/aligned.bam")},
            Path("health"),
            runner=runner,
            tools={"samtools_stats": _stats_entry()},
        )

        self.assertEqual(record.inputs, {"bam": str(self.bam)})
        (invocation,) = runner.invocations
        self.assertEqual(
            {m.host_path for m in invocation.mounts}, {self.bam, self.root / "health"}
        )
        self.assertEqual(invocation.stdout_path, self.root / "health" / "stats.txt")


class ProbedEntryTests(unittest.TestCase):
    """The health entries run the Run's probed samtools and bcftools, never a second pin."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bam = self.root / "cache" / "bam" / "aligned.bam"
        self.index = self.root / "index" / "aligned.bam.csi"
        self.vcf = self.root / "cache" / "vcf" / "variants.vcf"
        for path, payload in (
            (self.bam, b"BAM"),
            (self.index, b"CSI"),
            (self.vcf, b"##"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        self.probe = kept_probe(self.root / "tools")

    def test_each_health_entry_runs_its_argv_and_records_the_probe(self):
        summary = b"SN\tsummary\n"
        cases = {
            "samtools_index": (
                {"bam": self.bam},
                "samtools",
                [
                    "samtools",
                    "index",
                    "-c",
                    "-o",
                    "/outputs/aligned.bam.csi",
                    "/inputs/aligned.bam",
                ],
                "aligned.bam.csi",
            ),
            "samtools_index_bai": (
                {"bam": self.bam},
                "samtools",
                [
                    "samtools",
                    "index",
                    "-b",
                    "-o",
                    "/outputs/aligned.bam.bai",
                    "/inputs/aligned.bam",
                ],
                "aligned.bam.bai",
            ),
            "samtools_stats": (
                {"bam": self.bam},
                "samtools",
                ["samtools", "stats", "/inputs/aligned.bam"],
                "samtools-stats.txt",
            ),
            "samtools_flagstat": (
                {"bam": self.bam},
                "samtools",
                ["samtools", "flagstat", "/inputs/aligned.bam"],
                "samtools-flagstat.txt",
            ),
            "samtools_idxstats": (
                {"bam": self.bam, "bam_index": self.index},
                "samtools",
                ["samtools", "idxstats", "/inputs/aligned.bam"],
                "samtools-idxstats.tsv",
            ),
            "bcftools_stats": (
                {"vcf": self.vcf},
                "bcftools",
                ["bcftools", "stats", "-s", "-", "/inputs/variants.vcf"],
                "bcftools-stats.txt",
            ),
        }
        # The region entries take a region and run for real in
        # tests/test_run_health_region.py; the window entries run for real in
        # tests/test_run_health_items.py.
        self.assertEqual(
            sorted(cases),
            sorted(
                name
                for name, entry in TOOLS.items()
                if entry.probed
                and not entry.parameters
                and not name.endswith("_windows")
            ),
        )
        for tool, (inputs, probed, argv, output) in cases.items():
            with self.subTest(tool=tool):
                runner = FakeProbedRunner(
                    self.probe,
                    stdout=summary,
                    files={"aligned.bam.csi": summary, "aligned.bam.bai": summary},
                )

                record = run_tool(tool, inputs, self.root / tool, runner=runner)

                self.assertEqual(record.argv, tuple(argv))
                self.assertEqual(
                    record.executed_argv,
                    (f"/run/.venv/bin/bio-neb-{probed}", *argv[1:]),
                )
                self.assertIsNone(record.image_reference)
                self.assertEqual(
                    record.identity,
                    {
                        "probe_receipt": str(self.root / "tools" / RECEIPT_NAME),
                        "probe_receipt_sha256": self.probe.receipt_sha256,
                    },
                )
                self.assertEqual(
                    record.inputs, {name: str(path) for name, path in inputs.items()}
                )
                self.assertEqual(
                    record.outputs[output],
                    OutputFile(
                        sha256=hashlib.sha256(summary).hexdigest(),
                        byte_count=len(summary),
                    ),
                )
                (invocation,) = runner.invocations
                self.assertEqual(invocation.probed, probed)
                self.assertIsNone(invocation.pin)

    def test_an_executable_other_than_the_probed_one_is_refused_with_both_digests(self):
        ran = "d" * 64

        with self.assertRaises(RunRefused) as refused:
            run_tool(
                "samtools_stats",
                {"bam": self.bam},
                self.root / "health",
                runner=FakeProbedRunner(self.probe, ran=ran, stdout=b"stats"),
            )

        self.assertEqual(refused.exception.code, "identity_mismatch")
        self.assertIn(
            f"executable_sha256: probed {'b' * 64}, ran {ran}", refused.exception.detail
        )
        self.assertNotIn("tool_version", refused.exception.detail)

    def test_a_pinned_runner_refuses_a_probed_entry_before_contact(self):
        runner = FakeRunner()

        with self.assertRaises(RunRefused) as refused:
            run_tool(
                "samtools_stats", {"bam": self.bam}, self.root / "health", runner=runner
            )

        self.assertEqual(refused.exception.code, "not_pinned")
        self.assertIn("samtools is the Run's probed tool", refused.exception.detail)
        self.assertEqual(runner.invocations, [])
        self.assertFalse((self.root / "health").exists())

    def test_a_probed_runner_refuses_a_pinned_entry_before_contact(self):
        runner = FakeProbedRunner(self.probe)

        with self.assertRaises(RunRefused) as refused:
            run_tool(
                "mosdepth",
                {"bam": self.bam, "bam_index": self.index},
                self.root / "health",
                runner=runner,
            )

        self.assertEqual(refused.exception.code, "not_probed")
        self.assertIn("mosdepth is a pinned package", refused.exception.detail)
        self.assertEqual(runner.invocations, [])
        self.assertFalse((self.root / "health").exists())


class FakeInstalledRunner:
    """Runner port fake for pinned entries run locally: reports the pin's local facts."""

    name = "fake-installed"
    # The executable digest this install has; the pin holds only the version line.
    executable_sha256 = "sha256:" + "9" * 64

    def __init__(self, *, files=()):
        self.files = files
        self.invocations = []

    def bind(self, entry):
        return pinned_binding(entry, runner=self.name, fields=("version_line",))

    def run(self, invocation):
        self.invocations.append(invocation)
        (outputs,) = [mount for mount in invocation.mounts if not mount.read_only]
        for name in self.files:
            (outputs.host_path / name).parent.mkdir(parents=True, exist_ok=True)
            (outputs.host_path / name).write_bytes(name.encode())
        return RunResult(
            exit_code=0,
            identity={
                "executable_sha256": self.executable_sha256,
                "version_line": invocation.pin.version_line,
            },
            executed_argv=invocation.argv,
            stderr=b"",
        )


MOSDEPTH_OUTPUTS = (
    "coverage.mosdepth.global.dist.txt",
    "coverage.mosdepth.region.dist.txt",
    "coverage.mosdepth.summary.txt",
    "coverage.quantized.bed.gz",
    "coverage.regions.bed.gz",
    "coverage.thresholds.bed.gz",
)


class CoverageEntryTests(unittest.TestCase):
    """Whole-genome coverage: mosdepth in windows or by targets, and indexcov's overview."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bam = self.root / "cache" / "bam" / "aligned.bam"
        self.csi = self.root / "index" / "aligned.bam.csi"
        self.bai = self.root / "bai" / "aligned.bam.bai"
        self.targets = self.root / "targets" / "targets.bed"
        for path in (self.bam, self.csi, self.bai, self.targets):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode())

    def _mosdepth_argv(self, by):
        return (
            "mosdepth",
            "--no-per-base",
            "--threads",
            "4",
            "--by",
            by,
            "--thresholds",
            "1,10,20,30",
            "--quantize",
            "0:1:10:",
            "--mapq",
            "20",
            "--flag",
            "1796",
            "/outputs/coverage",
            "/inputs/aligned.bam",
        )

    def test_mosdepth_runs_in_fixed_windows_when_no_target_bed_is_given(self):
        runner = FakeInstalledRunner(files=MOSDEPTH_OUTPUTS)

        record = run_tool(
            "mosdepth",
            {"bam": self.bam, "bam_index": self.csi},
            self.root / "coverage",
            runner=runner,
        )

        self.assertEqual(record.argv, self._mosdepth_argv("500"))
        self.assertEqual(
            record.inputs, {"bam": str(self.bam), "bam_index": str(self.csi)}
        )
        self.assertEqual(sorted(record.outputs), list(MOSDEPTH_OUTPUTS))
        self.assertEqual(
            record.identity,
            {
                "executable_sha256": FakeInstalledRunner.executable_sha256,
                "version_line": "mosdepth 0.3.14",
            },
        )

    def test_a_target_bed_switches_the_same_entry_to_depth_by_target(self):
        runner = FakeInstalledRunner(files=MOSDEPTH_OUTPUTS)

        record = run_tool(
            "mosdepth",
            {"bam": self.bam, "bam_index": self.csi, "targets": self.targets},
            self.root / "coverage",
            runner=runner,
        )

        self.assertEqual(record.argv, self._mosdepth_argv("/inputs/targets.bed"))
        self.assertEqual(record.inputs["targets"], str(self.targets))
        (invocation,) = runner.invocations
        self.assertIn(
            Mount(
                host_path=self.targets,
                container_path="/inputs/targets.bed",
                read_only=True,
            ),
            invocation.mounts,
        )

    def test_an_input_neither_required_nor_optional_is_refused_before_contact(self):
        for inputs in (
            {"bam": self.bam, "targets": self.targets},
            {"bam": self.bam, "bam_index": self.csi, "bed": self.targets},
        ):
            with self.subTest(inputs=sorted(inputs)):
                runner = FakeInstalledRunner(files=MOSDEPTH_OUTPUTS)

                with self.assertRaises(RunRefused) as refused:
                    run_tool("mosdepth", inputs, self.root / "coverage", runner=runner)

                self.assertEqual(refused.exception.code, "inputs_mismatch")
                self.assertIn(
                    "['bam', 'bam_index'] and optionally ['targets']",
                    refused.exception.detail,
                )
                self.assertEqual(runner.invocations, [])
                self.assertFalse((self.root / "coverage").exists())

    def test_the_depth_threshold_and_read_filters_come_from_coverage_defaults(self):
        entry = coverage_entry(
            CoverageDefaults(
                minimum_depth=15,
                minimum_mapping_quality=30,
                excluded_flags=("DUP", "SECONDARY"),
            )
        )

        argv = entry.argv
        self.assertEqual(argv[argv.index("--quantize") + 1], "0:1:15:")
        self.assertEqual(argv[argv.index("--thresholds") + 1], "1,15,20,30")
        self.assertEqual(argv[argv.index("--mapq") + 1], "30")
        self.assertEqual(argv[argv.index("--flag") + 1], "1280")
        self.assertEqual(TOOLS["mosdepth"], coverage_entry(CoverageDefaults()))

    def test_the_callable_bins_keep_their_order_at_the_lowest_minimum_depths(self):
        for depth, quantize in ((0, "0:"), (1, "0:1:"), (2, "0:1:2:")):
            with self.subTest(minimum_depth=depth):
                argv = coverage_entry(CoverageDefaults(minimum_depth=depth)).argv

                self.assertEqual(argv[argv.index("--quantize") + 1], quantize)

    def test_coverage_settings_mosdepth_cannot_apply_are_refused(self):
        for setting, value in (
            ("overlapping_pairs", "count_each"),
            ("deletion_spanning_reads", "included"),
            ("applies_baq", True),
            ("excluded_flags", ("DUPLICATE",)),
        ):
            with self.subTest(setting=setting):
                with self.assertRaises(ValueError) as refused:
                    coverage_entry(CoverageDefaults(**{setting: value}))

                self.assertIn(setting, str(refused.exception))

    def test_indexcov_reads_the_bai_with_its_excluded_contigs_and_no_sex(self):
        outputs = (
            "indexcov/index.html",
            "indexcov/indexcov-indexcov.bed.gz",
            "indexcov/indexcov-indexcov.ped",
            "indexcov/indexcov-indexcov.roc",
        )
        runner = FakeInstalledRunner(files=outputs)

        record = run_tool(
            "goleft_indexcov",
            {"bam": self.bam, "bam_index": self.bai},
            self.root / "indexcov",
            runner=runner,
            tools={"goleft_indexcov": indexcov_entry("^NT_|^NW_")},
        )

        self.assertEqual(
            record.argv,
            (
                "goleft",
                "indexcov",
                "--directory",
                "/outputs/indexcov",
                "--excludepatt",
                "^NT_|^NW_",
                "--sex",
                "",
                "/inputs/aligned.bam",
            ),
        )
        self.assertEqual(record.image_reference, PINS["goleft"].image_reference)
        self.assertEqual(sorted(record.outputs), list(outputs))


def _scripted_docker(directory, *, run_exit=0, inspect=None):
    """A `docker` executable that logs its argv and answers `run` and `image inspect`."""
    log = directory / "docker-argv.jsonl"
    script = directory / "docker"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"with open({str(log)!r}, 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'run':\n"
        "    sys.stdout.write('samtools 1.24\\n')\n"
        "    sys.stderr.write('Pulling from biocontainers/samtools\\n')\n"
        f"    sys.exit({run_exit})\n"
        f"inspect = {inspect!r}\n"
        "if inspect is None:\n"
        "    sys.stderr.write('Error: No such image\\n')\n"
        "    sys.exit(1)\n"
        "print(json.dumps(inspect))\n"
    )
    script.chmod(0o755)
    return script, log


class DockerHostRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bam = self.root / "aligned.bam"
        self.bam.write_bytes(b"BAM\x01")
        (self.root / "health").mkdir()
        self.invocation = Invocation(
            pin=_pin(),
            argv=("samtools", "stats", "/inputs/aligned.bam"),
            mounts=(
                Mount(
                    host_path=self.bam,
                    container_path="/inputs/aligned.bam",
                    read_only=True,
                ),
                Mount(
                    host_path=self.root / "health",
                    container_path="/outputs",
                    read_only=False,
                ),
            ),
            workdir="/outputs",
            stdout_path=self.root / "health" / "stats.txt",
        )

    def test_runs_the_pinned_image_offline_and_reports_its_repository_digest(self):
        docker, log = _scripted_docker(
            self.root,
            inspect=[
                "quay.io/biocontainers/other@sha256:" + "c" * 64,
                "quay.io/biocontainers/samtools@" + PINNED,
            ],
        )

        result = DockerHostRunner(docker=str(docker)).run(self.invocation)

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.identity, {"image_digest": PINNED})
        self.assertEqual(result.stderr, b"Pulling from biocontainers/samtools\n")
        self.assertEqual(
            (self.root / "health" / "stats.txt").read_bytes(), b"samtools 1.24\n"
        )
        run, inspect = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(result.executed_argv, (str(docker), *run))
        self.assertEqual(
            run,
            [
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--workdir",
                "/outputs",
                "--mount",
                f"type=bind,source={self.bam},target=/inputs/aligned.bam,readonly",
                "--mount",
                f"type=bind,source={self.root / 'health'},target=/outputs",
                "quay.io/biocontainers/samtools@" + PINNED,
                "samtools",
                "stats",
                "/inputs/aligned.bam",
            ],
        )
        self.assertEqual(
            inspect,
            [
                "image",
                "inspect",
                "--format",
                "{{json .RepoDigests}}",
                "quay.io/biocontainers/samtools@" + PINNED,
            ],
        )

    def test_a_failed_pull_reports_the_docker_exit_and_no_digest(self):
        docker, _ = _scripted_docker(self.root, run_exit=125, inspect=None)

        result = DockerHostRunner(docker=str(docker)).run(self.invocation)

        self.assertEqual(result.exit_code, 125)
        self.assertEqual(result.identity, {"image_digest": None})


def _scripted_env(
    prefix, *, version="samtools 1.24\nUsing htslib 1.24\n", version_exit=0
):
    """An installed `samtools` env whose executable logs each call and answers --version.

    A call is logged as [argv, cwd, real path of the last argument]; a stats call prints
    the last argument's file name and whether `<it>.csi` sits beside it.
    """
    executable = prefix / "envs" / "samtools" / "bin" / "samtools"
    executable.parent.mkdir(parents=True)
    log = prefix / "samtools-calls.jsonl"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "target = sys.argv[-1]\n"
        f"with open({str(log)!r}, 'a') as log:\n"
        "    log.write(json.dumps([sys.argv[1:], os.getcwd(), os.path.realpath(target)])"
        " + '\\n')\n"
        "if sys.argv[1:] == ['--version']:\n"
        f"    sys.stdout.write({version!r})\n"
        f"    sys.exit({version_exit})\n"
        "beside = os.path.exists(target + '.csi')\n"
        "sys.stdout.write(f'stats of {os.path.basename(target)} index beside: {beside}\\n')\n"
    )
    executable.chmod(0o755)
    return executable, log


def _local_identity(executable, version_line="samtools 1.24"):
    return {
        "executable_sha256": "sha256:"
        + hashlib.sha256(executable.read_bytes()).hexdigest(),
        "version_line": version_line,
    }


class LocalProcessRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.prefix = self.root / "opt"
        self.bam = self.root / "aligned.bam"
        self.bam.write_bytes(b"BAM\x01")
        self.health = self.root / "health"

    def _invocation(self):
        self.health.mkdir()
        return Invocation(
            pin=_pin(),
            argv=("samtools", "stats", "/inputs/aligned.bam"),
            mounts=(
                Mount(
                    host_path=self.bam,
                    container_path="/inputs/aligned.bam",
                    read_only=True,
                ),
                Mount(
                    host_path=self.health, container_path="/outputs", read_only=False
                ),
            ),
            workdir="/outputs",
            stdout_path=self.health / "stats.txt",
        )

    def test_runs_the_installed_executable_in_the_output_dir_and_reports_its_identity(
        self,
    ):
        executable, log = _scripted_env(self.prefix)

        result = LocalProcessRunner(prefix=self.prefix).run(self._invocation())

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.identity, _local_identity(executable))
        self.assertEqual(
            (self.health / "stats.txt").read_text(),
            "stats of aligned.bam index beside: False\n",
        )
        *probes, (argv, cwd, target) = [
            json.loads(line) for line in log.read_text().splitlines()
        ]
        self.assertEqual(result.executed_argv, (str(executable), *argv))
        self.assertEqual([probe[0] for probe in probes], [["--version"]])
        self.assertEqual(argv[0], "stats")
        self.assertEqual(Path(argv[1]).name, "aligned.bam")
        self.assertEqual(cwd, str(self.health))
        self.assertEqual(target, str(self.bam))

    def test_inputs_from_different_directories_sit_side_by_side_for_the_tool(self):
        _scripted_env(self.prefix)
        index = self.root / "indexes" / "aligned.bam.csi"
        index.parent.mkdir()
        index.write_bytes(b"CSI\x01")
        invocation = self._invocation()
        invocation = invocation.model_copy(
            update={
                "mounts": (
                    *invocation.mounts,
                    Mount(
                        host_path=index,
                        container_path="/inputs/aligned.bam.csi",
                        read_only=True,
                    ),
                )
            }
        )

        result = LocalProcessRunner(prefix=self.prefix).run(invocation)

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            (self.health / "stats.txt").read_text(),
            "stats of aligned.bam index beside: True\n",
        )

    def test_a_relative_prefix_is_taken_from_the_working_directory(self):
        executable, _ = _scripted_env(self.prefix)
        cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, cwd)

        result = LocalProcessRunner(prefix=Path("opt")).run(self._invocation())

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.identity, _local_identity(executable))

    def test_a_missing_executable_reports_exit_127_and_observes_nothing(self):
        result = LocalProcessRunner(prefix=self.prefix).run(self._invocation())

        self.assertEqual(result.exit_code, 127)
        self.assertEqual(
            result.identity, {"executable_sha256": None, "version_line": None}
        )
        self.assertIn(
            str(self.prefix / "envs" / "samtools" / "bin" / "samtools"),
            result.stderr.decode(),
        )

    def test_a_version_probe_that_exits_nonzero_still_reports_its_first_line(self):
        # goleft 0.2.6 prints its version line, then its usage, and exits 1.
        executable, _ = _scripted_env(
            self.prefix, version="samtools 1.24\n\nusage\n", version_exit=1
        )

        result = LocalProcessRunner(prefix=self.prefix).run(self._invocation())

        self.assertEqual(result.identity, _local_identity(executable))

    def test_a_version_line_other_than_the_pin_is_refused_with_both_values(self):
        executable, _ = _scripted_env(self.prefix, version="samtools 1.23\n")
        entry = _stats_entry(version_line="samtools 1.24")

        with self.assertRaises(RunRefused) as refused:
            run_tool(
                "samtools_stats",
                {"bam": self.bam},
                self.health,
                runner=LocalProcessRunner(prefix=self.prefix),
                tools={"samtools_stats": entry},
            )

        self.assertEqual(refused.exception.code, "identity_mismatch")
        self.assertIn(
            "version_line: pinned samtools 1.24, ran samtools 1.23",
            refused.exception.detail,
        )
        self.assertNotIn("executable_sha256", refused.exception.detail)


def _probed_bin(directory):
    """`bio-neb-<tool>` entry points as a Run's environment has them, scripted.

    Each logs [argv, cwd]; `--version` prints the version in `<bin>/version-<tool>`
    (so it can change without changing the executable); any other call prints the last
    argument's file name and whether `<it>.csi` sits beside it.
    """
    bin_directory = directory / "bin"
    bin_directory.mkdir(parents=True)
    log = directory / "calls.jsonl"
    for tool in ("bcftools", "samtools"):
        (bin_directory / f"version-{tool}").write_text("1.24")
        executable = bin_directory / f"bio-neb-{tool}"
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            f"with open({str(log)!r}, 'a') as log:\n"
            "    log.write(json.dumps([sys.argv, os.getcwd()]) + '\\n')\n"
            "if sys.argv[1:] == ['--version']:\n"
            f"    version = open({str(bin_directory / f'version-{tool}')!r}).read()\n"
            f"    sys.stdout.write(f'{tool} {{version}}\\nUsing htslib 1.24\\n')\n"
            "    sys.exit(0)\n"
            "target = sys.argv[-1]\n"
            "beside = os.path.exists(target + '.csi')\n"
            "sys.stdout.write(f'{sys.argv[1]} of {os.path.basename(target)} '\n"
            "                 f'index beside: {beside}\\n')\n"
        )
        executable.chmod(0o755)
    return bin_directory, log


def _probe_of(bin_directory, directory):
    """The Run's probe of `bin_directory`, through main's `probe_observation_tools`."""
    pysam = ObservationTool(
        package="pysam", package_version=importlib.metadata.version("pysam")
    )
    return probe_observation_tools(
        InvestigationTools(bcftools=pysam, samtools=pysam),
        directory,
        bin_directory=bin_directory,
    )


class ProbedProcessRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bin, self.log = _probed_bin(self.root)
        self.probe = _probe_of(self.bin, self.root / "tools")
        self.bam = self.root / "cache" / "aligned.bam"
        self.index = self.root / "index" / "aligned.bam.csi"
        for path in (self.bam, self.index):
            path.parent.mkdir()
            path.write_bytes(b"\x01")
        self.health = self.root / "health"

    def _idxstats(self):
        return run_tool(
            "samtools_idxstats",
            {"bam": self.bam, "bam_index": self.index},
            self.health,
            runner=ProbedProcessRunner(self.probe),
        )

    def test_runs_the_probed_executable_with_its_inputs_side_by_side(self):
        record = self._idxstats()

        printed = b"idxstats of aligned.bam index beside: True\n"
        self.assertEqual(record.runner, "probed-process")
        self.assertEqual(
            record.outputs,
            {
                "samtools-idxstats.tsv": OutputFile(
                    sha256=hashlib.sha256(printed).hexdigest(), byte_count=len(printed)
                )
            },
        )
        *_, (argv, cwd) = [
            json.loads(line) for line in self.log.read_text().splitlines()
        ]
        self.assertEqual(argv[0], str(self.bin / "bio-neb-samtools"))
        self.assertEqual(record.executed_argv, tuple(argv))
        self.assertEqual(argv[1], "idxstats")
        self.assertEqual(Path(argv[2]).name, "aligned.bam")
        self.assertNotEqual(Path(argv[2]).parent, self.bam.parent)
        self.assertEqual(cwd, str(self.health))

    def test_an_executable_changed_after_the_probe_is_refused(self):
        executable = self.bin / "bio-neb-samtools"
        executable.write_text(executable.read_text() + "# changed\n")

        with self.assertRaises(RunRefused) as refused:
            self._idxstats()

        self.assertEqual(refused.exception.code, "identity_mismatch")
        changed = hashlib.sha256(executable.read_bytes()).hexdigest()
        self.assertIn(
            f"executable_sha256: probed {self.probe.receipt.tool('samtools').executable_sha256}, "
            f"ran {changed}",
            refused.exception.detail,
        )

    def test_a_version_other_than_the_probed_one_is_refused(self):
        (self.bin / "version-samtools").write_text("1.23")

        with self.assertRaises(RunRefused) as refused:
            self._idxstats()

        self.assertEqual(refused.exception.code, "identity_mismatch")
        self.assertIn("tool_version: probed 1.24, ran 1.23", refused.exception.detail)
        self.assertNotIn("executable_sha256", refused.exception.detail)

    def test_a_missing_probed_executable_is_refused_as_exit_127(self):
        (self.bin / "bio-neb-samtools").unlink()

        with self.assertRaises(RunRefused) as refused:
            self._idxstats()

        self.assertEqual(refused.exception.code, "tool_failed")
        self.assertIn("exit 127", refused.exception.detail)
        self.assertIn(str(self.bin / "bio-neb-samtools"), refused.exception.detail)


VCF_BYTES = b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
BAM_BYTES = b"bounded synthetic BAM acquisition bytes"


class _ObjectReads:
    """Object-read port fake for the Run's bucket: one exact GET per key.

    The FASTA is written as the Run's own reference bytes; any other object as
    `reference <key>`. Each is reported with the identity it was asked for, as the
    product's exact read does after checking the stream (`report` overrides that
    reported sha256 per key). A key ending in `fail` raises the
    product's transfer failure instead, as a stalled stream does.
    """

    def __init__(self, *, report=None, fail=None):
        self.report = report or {}
        self.fail = fail
        self.calls = []

    def __call__(self, packet):
        return self

    def download(
        self,
        object_key,
        destination,
        *,
        expected_byte_count,
        expected_sha256,
        observe_part=None,
    ):
        self.calls.append(object_key)
        if self.fail is not None and object_key.endswith(self.fail):
            raise StorageTransferError("get", detail="Read timeout on endpoint URL")
        destination.write(
            SOURCE_PLAIN
            if object_key.endswith("reference.fa")
            else b"reference " + object_key.encode()
        )
        return SimpleNamespace(
            byte_count=expected_byte_count,
            sha256=self.report.get(object_key, expected_sha256),
            credential_receipt=credential_receipt(),
            part_bytes=32 * 1024 * 1024,
            part_count=1,
        )


class RunCacheTests(unittest.TestCase):
    """A Run's reference is copied once into the lane's local cache, by main's read;
    its BAM and VCF stay the Run's own (#406)."""

    @classmethod
    def setUpClass(cls):
        cls.executed = prerequisite(
            "executed", output_payloads={"VCF": VCF_BYTES, "aligned_bam": BAM_BYTES}
        )

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cache = self.root / "cache" / "run"

    def _cache(self, reads, *, resume=False):
        return cache_run_files(
            Path(self.executed.journey_file),
            self.cache,
            object_read_factory=reads,
            resume=resume,
        )

    def _object_keys(self):
        """The FASTA and `.fai` keys, in the order a full cache reads them."""
        reads = _ObjectReads()
        cache_run_files(
            Path(self.executed.journey_file),
            self.root / "cache" / "keys",
            object_read_factory=reads,
        )
        return reads.calls

    def test_a_reference_read_that_fails_is_refused_with_its_cause(self):
        fasta, _ = self._object_keys()
        reads = _ObjectReads(fail="reference.fa")

        with self.assertRaises(RunRefused) as refused:
            self._cache(reads)

        self.assertEqual(refused.exception.code, "reference_read_failed")
        self.assertIn(fasta, refused.exception.detail)
        self.assertIn("transfer_failure", refused.exception.detail)
        self.assertIn("Read timeout on endpoint URL", refused.exception.detail)
        self.assertIsInstance(refused.exception.__cause__, StorageTransferError)
        self.assertEqual(reads.calls, [fasta])
        self.assertFalse((self.cache / "cache-manifest.json").exists())

    def test_a_resumed_cache_rereads_the_files_it_holds_and_copies_the_rest(self):
        fasta, fai = self._object_keys()
        with self.assertRaises(RunRefused):
            self._cache(_ObjectReads(fail="reference.fa.fai"))
        # Main's object read removed the failed file; the FASTA stays.
        self.assertEqual(
            [path.name for path in (self.cache / "reference").iterdir()],
            ["reference.fa"],
        )
        reads = _ObjectReads()

        manifest = self._cache(reads, resume=True)

        self.assertEqual(reads.calls, [fai])
        self.assertEqual(
            {role: cached.kept for role, cached in manifest.files.items()},
            {"reference_fasta": True, "reference_fai": False},
        )
        self.assertEqual(
            Path(manifest.files["reference_fasta"].path).read_bytes(), SOURCE_PLAIN
        )
        self.assertTrue((self.cache / "cache-manifest.json").is_file())

    def test_a_kept_file_whose_bytes_changed_is_refused_before_contact(self):
        with self.assertRaises(RunRefused):
            self._cache(_ObjectReads(fail="reference.fa.fai"))
        fasta = self.cache / "reference" / "reference.fa"
        fasta.chmod(0o644)
        fasta.write_bytes(b"changed")
        fasta.chmod(0o444)
        reads = _ObjectReads()

        with self.assertRaises(RunRefused) as refused:
            self._cache(reads, resume=True)

        self.assertEqual(refused.exception.code, "kept_file_mismatch")
        self.assertIn(str(fasta), refused.exception.detail)
        self.assertEqual(reads.calls, [])
        self.assertFalse((self.cache / "cache-manifest.json").exists())

    def test_a_kept_file_refusal_is_printed_by_the_command(self):
        with self.assertRaises(RunRefused):
            self._cache(_ObjectReads(fail="reference.fa.fai"))
        fasta = self.cache / "reference" / "reference.fa"
        fasta.chmod(0o644)
        reads = _ObjectReads()
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            status = main(
                [
                    "cache",
                    "--journey",
                    str(self.executed.journey_file),
                    "--cache-dir",
                    str(self.cache),
                    "--resume",
                ],
                object_read_factory=reads,
            )

        self.assertEqual(status, 2)
        printed = json.loads(stdout.getvalue())
        # A kept file must be sealed read-only, as main's read left it.
        self.assertEqual(printed["refused"], "candidate_mutable")
        self.assertIn(str(fasta), printed["detail"])
        self.assertEqual(reads.calls, [])
        self.assertFalse((self.cache / "cache-manifest.json").exists())

    def test_a_resume_of_a_complete_cache_is_refused(self):
        self._cache(_ObjectReads())
        reads = _ObjectReads()

        with self.assertRaises(RunRefused) as refused:
            self._cache(reads, resume=True)

        self.assertEqual(refused.exception.code, "cache_complete")
        self.assertEqual(reads.calls, [])

    def test_the_cache_holds_the_runs_reference_with_its_object_identity(self):
        from bio_neb.parabricks.result import read_completed_parabricks_result

        reads = _ObjectReads()

        manifest = self._cache(reads)

        outcome = read_completed_parabricks_result(str(self.executed.journey_file))
        self.assertEqual(manifest.journey_id, outcome.journey_id)
        self.assertEqual(manifest.journey_file, outcome.journey_file)
        self.assertEqual(manifest.bucket, outcome.storage.bucket)
        self.assertEqual(set(manifest.files), {"reference_fasta", "reference_fai"})
        fasta, fai = manifest.files["reference_fasta"], manifest.files["reference_fai"]
        self.assertEqual(fasta.sha256, outcome.reference.plain_fasta_sha256)
        self.assertEqual(fasta.byte_count, outcome.reference.plain_fasta_byte_count)
        self.assertTrue(fasta.object_key.endswith("/reference.fa"))
        self.assertTrue(fai.object_key.endswith("/reference.fa.fai"))
        self.assertEqual(
            Path(fai.path).read_bytes(), b"reference " + fai.object_key.encode()
        )
        self.assertEqual(reads.calls, [fasta.object_key, fai.object_key])
        for cached in manifest.files.values():
            self.assertTrue(Path(cached.path).is_relative_to(self.cache))
            self.assertEqual(os.stat(cached.path).st_mode & 0o222, 0)
        self.assertEqual(
            json.loads((self.cache / "cache-manifest.json").read_text()),
            manifest.model_dump(mode="json"),
        )

    def test_a_reference_read_reporting_another_identity_is_refused(self):
        _, fai_key = self._object_keys()

        with self.assertRaises(RunRefused) as refused:
            self._cache(_ObjectReads(report={fai_key: "f" * 64}))

        self.assertEqual(refused.exception.code, "reference_read_mismatch")
        self.assertIn(fai_key, refused.exception.detail)
        self.assertFalse((self.cache / "cache-manifest.json").exists())

    def test_an_existing_cache_is_refused_before_contact(self):
        self.cache.mkdir(parents=True)
        reads = _ObjectReads()

        with self.assertRaises(RunRefused) as refused:
            self._cache(reads)

        self.assertEqual(refused.exception.code, "cache_exists")
        self.assertEqual(reads.calls, [])

    def test_the_cache_command_prints_the_manifest_it_wrote(self):
        stdout = io.StringIO()
        arguments = [
            "cache",
            "--journey",
            str(self.executed.journey_file),
            "--cache-dir",
            str(self.cache),
        ]
        with contextlib.redirect_stdout(stdout):
            status = main(arguments, object_read_factory=_ObjectReads())

        self.assertEqual(status, 0)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            json.loads((self.cache / "cache-manifest.json").read_text()),
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main(arguments, object_read_factory=_ObjectReads())

        self.assertEqual(status, 2)
        self.assertEqual(json.loads(stdout.getvalue())["refused"], "cache_exists")

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main([*arguments, "--resume"], object_read_factory=_ObjectReads())

        self.assertEqual(status, 2)
        self.assertEqual(json.loads(stdout.getvalue())["refused"], "cache_complete")


class CommandTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bam = self.root / "aligned.bam"
        self.bam.write_bytes(b"BAM\x01")

    def _main(self, *argv, tools):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main(list(argv), tools=tools)
        return status, json.loads(stdout.getvalue())

    def test_run_prints_the_record_of_a_docker_host_run(self):
        docker, _ = _scripted_docker(
            self.root, inspect=["quay.io/biocontainers/samtools@" + PINNED]
        )

        status, printed = self._main(
            "run",
            "samtools_stats",
            "--input",
            f"bam={self.bam}",
            "--output-dir",
            str(self.root / "health"),
            "--runner",
            "docker",
            "--docker",
            str(docker),
            tools={"samtools_stats": _stats_entry()},
        )

        self.assertEqual(status, 0)
        executed = printed.pop("executed_argv")
        self.assertEqual(executed[:2], [str(docker), "run"])
        self.assertEqual(
            executed[-4:],
            [
                "quay.io/biocontainers/samtools@" + PINNED,
                "samtools",
                "stats",
                "/inputs/aligned.bam",
            ],
        )
        self.assertEqual(
            printed,
            {
                "schema_version": "bio-neb.run-health-record/1",
                "tool": "samtools_stats",
                "runner": "owner-host-docker",
                "image_reference": "quay.io/biocontainers/samtools:1.24",
                "identity": {"image_digest": PINNED},
                "argv": ["samtools", "stats", "/inputs/aligned.bam"],
                "inputs": {"bam": str(self.bam)},
                "outputs": {
                    "stats.txt": {
                        "sha256": hashlib.sha256(b"samtools 1.24\n").hexdigest(),
                        "byte_count": 14,
                    }
                },
            },
        )

    def test_run_prints_the_record_of_a_local_process_run(self):
        executable, _ = _scripted_env(self.root / "opt")
        identity = _local_identity(executable)
        printed_stats = b"stats of aligned.bam index beside: False\n"

        status, printed = self._main(
            "run",
            "samtools_stats",
            "--input",
            f"bam={self.bam}",
            "--output-dir",
            str(self.root / "health"),
            "--runner",
            "local",
            "--prefix",
            str(self.root / "opt"),
            # The pin names only the version: this install's digest is recorded.
            tools={
                "samtools_stats": _stats_entry(version_line=identity["version_line"])
            },
        )

        self.assertEqual(status, 0)
        executed = printed.pop("executed_argv")
        self.assertEqual(executed[:2], [str(executable), "stats"])
        self.assertEqual(Path(executed[2]).name, "aligned.bam")
        self.assertNotEqual(Path(executed[2]).parent, self.bam.parent)
        self.assertEqual(
            printed,
            {
                "schema_version": "bio-neb.run-health-record/1",
                "tool": "samtools_stats",
                "runner": "local-process",
                "image_reference": "quay.io/biocontainers/samtools:1.24",
                "identity": identity,
                "argv": ["samtools", "stats", "/inputs/aligned.bam"],
                "inputs": {"bam": str(self.bam)},
                "outputs": {
                    "stats.txt": {
                        "sha256": hashlib.sha256(printed_stats).hexdigest(),
                        "byte_count": len(printed_stats),
                    }
                },
            },
        )

    def test_run_prints_a_refusal_and_never_calls_docker(self):
        docker, log = _scripted_docker(self.root, inspect=[])

        status, printed = self._main(
            "run",
            "samtools_stats",
            "--input",
            f"bam={self.bam}",
            "--output-dir",
            str(self.root / "health"),
            "--runner",
            "docker",
            "--docker",
            str(docker),
            tools={"samtools_stats": _stats_entry(image_digest=None)},
        )

        self.assertEqual(status, 2)
        self.assertEqual(printed["refused"], "not_pinned")
        self.assertIn("samtools 1.24 has no image_digest", printed["detail"])
        self.assertFalse(log.exists())

    def test_run_prints_the_record_of_a_run_held_to_a_kept_probe(self):
        bin_directory, _ = _probed_bin(self.root)
        probe = _probe_of(bin_directory, self.root / "tools")
        printed_stats = b"stats of aligned.bam index beside: False\n"

        status, printed = self._main(
            "run",
            "samtools_stats",
            "--input",
            f"bam={self.bam}",
            "--output-dir",
            str(self.root / "health"),
            "--runner",
            "probed",
            "--probe-dir",
            str(self.root / "tools"),
            tools=TOOLS,
        )

        self.assertEqual(status, 0)
        executed = printed.pop("executed_argv")
        self.assertEqual(
            executed[:2], [str(bin_directory / "bio-neb-samtools"), "stats"]
        )
        self.assertEqual(
            printed,
            {
                "schema_version": "bio-neb.run-health-record/1",
                "tool": "samtools_stats",
                "runner": "probed-process",
                "image_reference": None,
                "identity": {
                    "probe_receipt": probe.receipt_path,
                    "probe_receipt_sha256": probe.receipt_sha256,
                },
                "argv": ["samtools", "stats", "/inputs/aligned.bam"],
                "inputs": {"bam": str(self.bam)},
                "outputs": {
                    "samtools-stats.txt": {
                        "sha256": hashlib.sha256(printed_stats).hexdigest(),
                        "byte_count": len(printed_stats),
                    }
                },
            },
        )

    def test_a_fresh_probe_reads_the_runs_tool_section_from_its_runtime_file(self):
        runtime = self.root / "runtime.json"
        runtime.write_text(json.dumps(runtime_configuration()))
        tools = self.root / "tools"

        status, printed = self._main(
            "run",
            "samtools_index",
            "--input",
            f"bam={self.bam}",
            "--output-dir",
            str(self.root / "health"),
            "--runner",
            "probed",
            "--probe-dir",
            str(tools),
            "--runtime-config",
            str(runtime),
            tools={
                "samtools_index": TOOLS["samtools_index"].model_copy(
                    update={"argv": ("samtools", "--version"), "outputs": ()}
                )
            },
        )

        self.assertEqual(status, 0)
        receipt = ObservationToolsReceipt.model_validate_json(
            (tools / RECEIPT_NAME).read_bytes()
        )
        self.assertEqual(
            printed["executed_argv"], [receipt.tool("samtools").executable, "--version"]
        )
        self.assertEqual(
            Path(receipt.tool("samtools").executable),
            Path(sys.prefix) / "bin" / "bio-neb-samtools",
        )
        self.assertEqual(
            printed["identity"]["probe_receipt_sha256"],
            hashlib.sha256((tools / RECEIPT_NAME).read_bytes()).hexdigest(),
        )

    def test_a_fresh_probe_without_the_runs_tool_section_is_refused(self):
        status, printed = self._main(
            "run",
            "samtools_stats",
            "--input",
            f"bam={self.bam}",
            "--output-dir",
            str(self.root / "health"),
            "--runner",
            "probed",
            "--probe-dir",
            str(self.root / "tools"),
            tools=TOOLS,
        )

        self.assertEqual(status, 2)
        self.assertEqual(printed["refused"], "not_probed")
        self.assertIn("runtime_config_invalid", printed["detail"])
        self.assertIn("investigation_tools", printed["detail"])
        self.assertFalse((self.root / "tools").exists())
        self.assertFalse((self.root / "health").exists())


if __name__ == "__main__":
    unittest.main()


class RunModelTests(unittest.TestCase):
    def test_the_model_comes_from_the_registered_selections_not_the_file(self):
        # The Run's registered selections are its only configuration; its
        # runtime file is never read again, here not even present.
        from unittest.mock import patch

        from bio_neb.application.commands import (
            RegisterFastqInput,
            RegisterRunCommand,
            register_run,
        )
        from tools.run_health.check import run_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prompt, config = root / "prompt.txt", root / "runtime.json"
            prompt.write_text("inspect my registered inputs", encoding="utf-8")
            config.write_text(json.dumps(runtime_configuration()), encoding="utf-8")
            fastqs = (root / "r1.fastq", root / "r2.fastq")
            for fastq in fastqs:
                fastq.write_bytes(b"reads")
            receipt = register_run(
                RegisterRunCommand(
                    run_root=str(root / "runs"),
                    prompt_file=str(prompt),
                    runtime_config_file=str(config),
                    inputs=tuple(
                        RegisterFastqInput(role=f"read_{index}", registered_path=str(p))
                        for index, p in enumerate(fastqs, start=1)
                    ),
                ),
                repository_root=Path.cwd(),
            )
            config.unlink()
            with patch.dict(os.environ, {"LOCAL_MODEL_KEY": "offline"}):
                model = run_model(Path(receipt.journey_file))
            with (
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": ""}),
                self.assertRaises(RunRefused) as refused,
            ):
                run_model(Path(receipt.journey_file))
        self.assertEqual(model.model_name, "Qwen/Qwen3.5-397B-A17B")
        self.assertEqual(refused.exception.code, "model_credential_missing")
