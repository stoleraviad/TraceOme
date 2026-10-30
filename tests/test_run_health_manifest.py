"""One health report per Run: `run_health` runs every whole-Run health tool over the Run's
own BAM and VCF, then MultiQC over their summaries, and returns one manifest (#317, #406).

The runner ports are fakes that write each tool's files by name; the Run is an executed
prerequisite whose verified BAM and VCF are the bytes written here, downloaded beside its
Journey by main's own copies as its Investigation does (#363), so every digest and
`run-artifacts` role below is read independently of the code under test.
"""

import contextlib
import gzip
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bio_neb.application.run_reader import read_run_artifacts
from bio_neb.investigation.vcf_materialization import (
    MaterializeCurrentVcfCommand,
    materialize_current_vcf,
)
from bio_neb.parabricks.result import read_completed_parabricks_result
from tests.parabricks_operation_fakes import BAM_BYTES, VCF_BYTES
from tests.run_health_fakes import ObjectGet, local_run_inputs
from tests.run_prerequisites import RunPrerequisites, prerequisite
from tools.run_health import (
    Pin,
    RunRefused,
    RunResult,
    ToolEntry,
    pinned_binding,
    probe_binding,
    run_tool,
)
from tools.run_health.__main__ import main
from tools.run_health.cache import MANIFEST_NAME as CACHE_MANIFEST
from tools.run_health.cache import CachedFile, RunCache
from tools.run_health.health import (
    COVERED_NAME,
    MANIFEST_NAME,
    SAMPLE_NAMES_NAME,
    HealthManifest,
    run_health,
    run_probe,
)
from tools.run_health.replay import ReplayRunner

# What each tool writes, by program; samtools and bcftools summaries go to stdout.
_WRITES = {
    "mosdepth": tuple(
        f"coverage.{name}"
        for name in (
            "mosdepth.global.dist.txt",
            "mosdepth.region.dist.txt",
            "mosdepth.summary.txt",
            "regions.bed.gz",
            "regions.bed.gz.csi",
            "thresholds.bed.gz",
            "thresholds.bed.gz.csi",
            "quantized.bed.gz",
            "quantized.bed.gz.csi",
        )
    ),
    "goleft": (
        "indexcov/index.html",
        "indexcov/indexcov-indexcov.bed.gz",
        "indexcov/indexcov-indexcov.ped",
        "indexcov/indexcov-indexcov.roc",
        "indexcov/indexcov-indexcov-depth-NC_000001.11.html",
        "indexcov/indexcov-indexcov-depth-NC_000001.11.png",
    ),
    "multiqc": (
        "multiqc.html",
        "multiqc_data/multiqc_data.json",
        "multiqc_data/multiqc.log",
        "multiqc_data/samtools-flagstat-table.json",
    ),
}

# The agreed role of every file the health run writes (path under its directory).
EXPECTED_ROLES = {
    "samtools_index/aligned.bam.csi": "bam_csi",
    "samtools_stats/samtools-stats.txt": "samtools_stats",
    "samtools_flagstat/samtools-flagstat.txt": "samtools_flagstat",
    "samtools_idxstats/samtools-idxstats.tsv": "samtools_idxstats",
    "bcftools_stats/bcftools-stats.txt": "bcftools_stats",
    "mosdepth/coverage.mosdepth.global.dist.txt": "mosdepth_global_dist",
    "mosdepth/coverage.mosdepth.region.dist.txt": "mosdepth_region_dist",
    "mosdepth/coverage.mosdepth.summary.txt": "mosdepth_summary",
    "mosdepth/coverage.regions.bed.gz": "mosdepth_regions",
    "mosdepth/coverage.regions.bed.gz.csi": "mosdepth_bed_index",
    "mosdepth/coverage.thresholds.bed.gz": "mosdepth_thresholds",
    "mosdepth/coverage.thresholds.bed.gz.csi": "mosdepth_bed_index",
    "mosdepth/coverage.quantized.bed.gz": "mosdepth_quantized",
    "mosdepth/coverage.quantized.bed.gz.csi": "mosdepth_bed_index",
    "samtools_index_bai/aligned.bam.bai": "bam_bai",
    "goleft_indexcov/indexcov/index.html": "indexcov_report",
    "goleft_indexcov/indexcov/indexcov-indexcov.bed.gz": "indexcov_bed",
    "goleft_indexcov/indexcov/indexcov-indexcov.ped": "indexcov_ped",
    "goleft_indexcov/indexcov/indexcov-indexcov.roc": "indexcov_roc",
    "goleft_indexcov/indexcov/indexcov-indexcov-depth-NC_000001.11.html": (
        "indexcov_chromosome_plot"
    ),
    "goleft_indexcov/indexcov/indexcov-indexcov-depth-NC_000001.11.png": (
        "indexcov_chromosome_plot"
    ),
    "multiqc/multiqc.html": "multiqc_report",
    "multiqc/multiqc_data/multiqc_data.json": "multiqc_data",
    "multiqc/multiqc_data/multiqc.log": "multiqc_data_file",
    "multiqc/multiqc_data/samtools-flagstat-table.json": "multiqc_data_file",
}


# What samtools idxstats prints (contig, length, mapped, unmapped reads) and mosdepth's
# 500-base windows with their mean depth: the reads lie in NC_000001.11:1-1500 and
# 2001-2500 only (the window 1501-2000 has a mean depth below 1), none on the scaffold.
IDXSTATS = b"NC_000001.11\t3000\t120\t0\nNT_187361.1\t500\t0\t0\n*\t0\t0\t4\n"
WINDOWS = (
    b"NC_000001.11\t0\t500\t30.00\n"
    b"NC_000001.11\t500\t1000\t12.50\n"
    b"NC_000001.11\t1000\t1500\t1.00\n"
    b"NC_000001.11\t1500\t2000\t0.40\n"
    b"NC_000001.11\t2000\t2500\t8.00\n"
    b"NC_000001.11\t2500\t3000\t0.00\n"
    b"NT_187361.1\t0\t500\t0.00\n"
)
_CONTENT = {
    "samtools-idxstats.tsv": IDXSTATS,
    "coverage.regions.bed.gz": gzip.compress(WINDOWS),
}


class _FakeTool:
    """One runner port fake: writes what the invoked tool writes and reports the bound
    identity. `ran` replaces the reported identity of the program it names."""

    def __init__(self, name, binder, identity, *, ran=None):
        self.name = name
        self._binder = binder
        self._identity = identity
        self._ran = ran or {}
        self.invocations = []

    def bind(self, entry):
        return self._binder(entry)

    def run(self, invocation):
        self.invocations.append(invocation)
        argv = invocation.argv
        (outputs,) = [m.host_path for m in invocation.mounts if not m.read_only]
        if invocation.stdout_path is not None:
            invocation.stdout_path.write_bytes(
                _CONTENT.get(invocation.stdout_path.name, " ".join(argv).encode())
            )
        written = list(_WRITES.get(argv[0], ()))
        if argv[:2] == ("samtools", "index"):
            written.append(argv[argv.index("-o") + 1].removeprefix("/outputs/"))
        for name in written:
            (outputs / name).parent.mkdir(parents=True, exist_ok=True)
            (outputs / name).write_bytes(
                _CONTENT.get(name, f"{argv[0]} {name}".encode())
            )
        return RunResult(
            exit_code=0,
            identity=self._ran.get(argv[0], self._identity(invocation)),
            executed_argv=argv,
            stderr=b"",
        )


def _pinned_runner(**ran):
    return _FakeTool(
        "fake-local",
        lambda entry: pinned_binding(
            entry, runner="fake-local", fields=("version_line",)
        ),
        lambda invocation: {
            "executable_sha256": "sha256:" + "9" * 64,
            "version_line": invocation.pin.version_line,
        },
        ran=ran,
    )


def _probed_runner(probe, **ran):
    def identity(invocation):
        tool = probe.receipt.tool(invocation.probed)
        return {
            "executable_sha256": tool.executable_sha256,
            "tool_version": tool.tool_version,
        }

    return _FakeTool(
        "fake-probed",
        lambda entry: probe_binding(entry, probe, runner="fake-probed"),
        identity,
        ran=ran,
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The reference's contigs: an assembled chromosome and an unplaced scaffold.
REFERENCE_FAI = b"NC_000001.11\t1000\t14\t60\t61\nNT_187361.1\t500\t1046\t60\t61\n"


def reference_cache(cache_dir, outcome, *, journey_id=None):
    """The lane's cache of the Run's reference: its `.fai` (and a FASTA entry), as
    `cache_run_files` records them."""
    files = {}
    for role, name, body in (
        ("reference_fasta", "reference.fa", b">NC_000001.11\n"),
        ("reference_fai", "reference.fa.fai", REFERENCE_FAI),
    ):
        path = cache_dir / "reference" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        files[role] = CachedFile(
            path=str(path),
            object_key=f"reference/{name}",
            byte_count=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            kept=False,
        )
    (cache_dir / CACHE_MANIFEST).write_text(
        RunCache(
            journey_id=journey_id or outcome.journey_id,
            journey_file=outcome.journey_file,
            bucket=outcome.storage.bucket,
            files=files,
        ).model_dump_json()
    )


class RunHealthTests(RunPrerequisites, unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.journey_file = self.prerequisite(
            "executed",
            writable=True,
            output_payloads={"VCF": VCF_BYTES, "aligned_bam": BAM_BYTES},
        ).journey_file
        self.investigation = local_run_inputs(
            self.journey_file, vcf=VCF_BYTES, bam=BAM_BYTES
        )
        self.outcome = read_completed_parabricks_result(str(self.journey_file))
        self.artifacts = read_run_artifacts(str(self.journey_file))
        reference_cache(self.root / "cache", self.outcome)
        self.pinned = _pinned_runner()
        self.probed = _probed_runner(run_probe(self.outcome))
        self.output = self.root / "health"

    def _run(self):
        return run_health(
            self.journey_file,
            self.root / "cache",
            self.output,
            pinned=self.pinned,
            probed=self.probed,
        )

    def _refused(self):
        with self.assertRaises(RunRefused) as refused:
            self._run()
        self.assertEqual(self.pinned.invocations + self.probed.invocations, [])
        self.assertFalse(self.output.exists())
        return refused.exception

    def test_every_health_file_is_listed_with_its_role_digest_and_run_record(self):
        manifest = self._run()

        written = {
            path.relative_to(self.output).as_posix(): path
            for path in self.output.rglob("*")
            if path.is_file()
            and path.name not in (MANIFEST_NAME, SAMPLE_NAMES_NAME, COVERED_NAME)
        }
        self.assertEqual(set(written), set(EXPECTED_ROLES))
        listed = {entry.path: entry for entry in manifest.files}
        self.assertEqual(set(listed), set(EXPECTED_ROLES))
        for path, entry in listed.items():
            with self.subTest(path=path):
                self.assertEqual(entry.role, EXPECTED_ROLES[path])
                self.assertEqual(entry.sha256, _sha256(written[path]))
                self.assertEqual(entry.byte_count, written[path].stat().st_size)
                record = manifest.records[entry.record]
                self.assertEqual(
                    record.outputs[path.removeprefix(entry.record + "/")].sha256,
                    entry.sha256,
                )
        self.assertEqual(set(manifest.entry_seconds), set(manifest.records))
        self.assertEqual(
            HealthManifest.model_validate_json(
                (self.output / MANIFEST_NAME).read_text()
            ),
            manifest,
        )

    def test_the_inputs_are_the_runs_own_copies_under_their_run_artifacts_roles(self):
        manifest = self._run()

        listed = {
            artifact.role: artifact
            for artifact in self.artifacts.outputs
            if artifact.role in ("aligned_bam", "VCF")
        }
        self.assertEqual(
            [
                (item.role, item.location, item.byte_count, item.sha256)
                for item in manifest.inputs
            ],
            [
                (
                    role,
                    listed[role].location,
                    listed[role].byte_count,
                    listed[role].sha256,
                )
                for role in ("aligned_bam", "VCF")
            ],
        )
        self.assertEqual(
            [(item.path, item.receipt) for item in manifest.inputs],
            [
                (
                    str(self.investigation / "bam" / "aligned.bam"),
                    str(
                        self.investigation
                        / "bam"
                        / "current-bam-acquisition-receipt.json"
                    ),
                ),
                (
                    str(self.investigation / "vcf" / "variants.vcf"),
                    str(
                        self.investigation / "vcf" / "vcf-materialization-receipt.json"
                    ),
                ),
            ],
        )
        self.assertEqual(Path(manifest.inputs[0].path).read_bytes(), BAM_BYTES)
        run_roles = {
            artifact.role
            for part in (
                self.artifacts.outputs,
                self.artifacts.inputs,
                self.artifacts.reference,
                self.artifacts.local,
            )
            for artifact in part
        }
        self.assertTrue(run_roles)
        self.assertEqual(run_roles & {entry.role for entry in manifest.files}, set())
        self.assertEqual(
            (
                manifest.journey_id,
                manifest.journey_file,
                manifest.run_root,
                manifest.execution_id,
                manifest.outcome_event_id,
            ),
            (
                self.artifacts.journey_id,
                self.artifacts.journey_file,
                self.artifacts.run_root,
                self.artifacts.execution_id,
                self.artifacts.outcome_event_id,
            ),
        )

    def test_each_tool_reads_the_files_earlier_runs_wrote_and_multiqc_reads_every_summary(
        self,
    ):
        self._run()

        ran = {
            invocation.argv[0] + " " + invocation.argv[1]: invocation
            for runner in (self.pinned, self.probed)
            for invocation in runner.invocations
        }
        bam = str(self.investigation / "bam" / "aligned.bam")
        health = self.output

        def inputs(command):
            return {
                Path(mount.host_path).as_posix()
                for mount in ran[command].mounts
                if mount.read_only
            }

        self.assertEqual(
            inputs("samtools idxstats"),
            {bam, f"{health}/samtools_index/aligned.bam.csi"},
        )
        self.assertEqual(
            inputs("mosdepth --no-per-base"),
            {bam, f"{health}/samtools_index/aligned.bam.csi"},
        )
        self.assertEqual(
            inputs("goleft indexcov"),
            {bam, f"{health}/samtools_index_bai/aligned.bam.bai"},
        )
        self.assertEqual(
            inputs("bcftools stats"),
            {str(self.investigation / "vcf" / "variants.vcf")},
        )
        self.assertEqual(
            inputs("multiqc --no-version-check"),
            {
                f"{health}/{SAMPLE_NAMES_NAME}",
                f"{health}/samtools_stats/samtools-stats.txt",
                f"{health}/samtools_flagstat/samtools-flagstat.txt",
                f"{health}/samtools_idxstats/samtools-idxstats.tsv",
                f"{health}/bcftools_stats/bcftools-stats.txt",
                f"{health}/mosdepth/coverage.mosdepth.global.dist.txt",
                f"{health}/mosdepth/coverage.mosdepth.region.dist.txt",
                f"{health}/mosdepth/coverage.mosdepth.summary.txt",
                f"{health}/goleft_indexcov/indexcov/indexcov-indexcov.ped",
            },
        )
        self.assertEqual(
            {invocation.probed for invocation in self.probed.invocations},
            {"samtools", "bcftools"},
        )
        self.assertEqual(
            {invocation.pin.package for invocation in self.pinned.invocations},
            {"mosdepth", "goleft", "multiqc"},
        )

    def test_indexcov_and_multiqc_run_with_the_arguments_the_run_supplies(self):
        manifest = self._run()

        (indexcov,) = [i for i in self.pinned.invocations if i.argv[0] == "goleft"]
        argv = indexcov.argv
        # No contig is left out by its name's shape (ADR 0049); no sex is
        # inferred.
        self.assertEqual(argv[argv.index("--excludepatt") + 1], "^$")
        self.assertEqual(argv[argv.index("--sex") + 1], "")
        self.assertEqual(
            (self.output / SAMPLE_NAMES_NAME).read_text().splitlines(),
            [
                "coverage\taligned_bam",
                "samtools-flagstat\taligned_bam",
                "samtools-idxstats\taligned_bam",
                "samtools-stats\taligned_bam",
                "variants\tVCF",
            ],
        )
        self.assertEqual(
            manifest.arguments.reference_index,
            str(self.root / "cache" / "reference" / "reference.fa.fai"),
        )

    def test_the_regions_the_reads_cover_are_listed_by_contig_and_in_a_bed(self):
        manifest = self._run()

        covered = manifest.covered
        self.assertEqual((covered.window_bases, covered.minimum_mean_depth), (500, 1.0))
        self.assertEqual(
            [
                (
                    contig.contig,
                    contig.length,
                    contig.mapped_reads,
                    contig.covered_bases,
                    contig.first_covered,
                    contig.last_covered,
                )
                for contig in covered.contigs
            ],
            [
                # The scaffold has no read and no covered window: not listed.
                ("NC_000001.11", 3000, 120, 2000, 1, 2500),
            ],
        )
        bed = self.output / covered.path
        self.assertEqual(
            bed.read_text(), "NC_000001.11\t0\t1500\nNC_000001.11\t2000\t2500\n"
        )
        self.assertEqual(
            (covered.byte_count, covered.sha256),
            (bed.stat().st_size, _sha256(bed)),
        )

    def test_a_recorded_report_replays_to_the_same_files_without_running_a_tool(self):
        recorded = self._run()
        ran = len(self.pinned.invocations + self.probed.invocations)
        again = self.root / "replayed"
        replay = ReplayRunner.of_health(self.output, again)

        replayed = run_health(
            self.journey_file,
            self.root / "cache",
            again,
            pinned=replay,
            probed=replay,
        )

        self.assertEqual(len(self.pinned.invocations + self.probed.invocations), ran)
        self.assertEqual(replayed.files, recorded.files)
        self.assertEqual(
            (replayed.inputs, replayed.arguments, replayed.covered),
            (recorded.inputs, recorded.arguments, recorded.covered),
        )
        for entry, record in replayed.records.items():
            with self.subTest(entry=entry):
                self.assertEqual(record.runner, "replay")
                self.assertEqual(record.argv, recorded.records[entry].argv)

    def test_a_replay_whose_command_differs_from_the_recorded_run_is_refused(self):
        recorded = self._run()
        mosdepth = recorded.records["mosdepth"]
        changed = recorded.model_copy(
            update={
                "records": recorded.records
                | {
                    "mosdepth": mosdepth.model_copy(
                        update={"argv": (*mosdepth.argv[:-1], "/inputs/other.bam")}
                    )
                }
            }
        )
        manifest = self.output / MANIFEST_NAME
        manifest.write_text(changed.model_dump_json())
        again = self.root / "replayed"
        replay = ReplayRunner.of_health(self.output, again)

        with self.assertRaises(RunRefused) as refused:
            run_health(
                self.journey_file,
                self.root / "cache",
                again,
                pinned=replay,
                probed=replay,
            )

        self.assertEqual(refused.exception.code, "tool_failed")
        self.assertTrue(refused.exception.detail.startswith("mosdepth: "))
        self.assertIn("replay: mosdepth ran", refused.exception.detail)
        self.assertFalse((again / MANIFEST_NAME).exists())

    def test_a_run_without_its_local_bam_is_refused_before_anything_runs(self):
        bam = self.investigation / "bam"
        for path in bam.iterdir():
            path.unlink()
        bam.rmdir()

        refused = self._refused()

        self.assertEqual(refused.code, "run_file_missing")
        self.assertIn("aligned_bam", refused.detail)
        self.assertIn("before #363", refused.detail)

    def test_a_receipt_of_another_runs_file_is_refused_before_anything_runs(self):
        changed = VCF_BYTES + b"1\t1\t.\tA\tG\t.\t.\t.\n"
        other = prerequisite(
            "executed", output_payloads={"VCF": changed, "aligned_bam": BAM_BYTES}
        )
        receipt = self.investigation / "vcf" / "vcf-materialization-receipt.json"
        foreign = self.root / "foreign"
        materialize_current_vcf(
            MaterializeCurrentVcfCommand(
                execution_journey_file=str(other.journey_file),
                output_directory=str(foreign),
            ),
            object_read_factory=ObjectGet(changed),
        )
        receipt.chmod(0o644)
        receipt.write_bytes((foreign / receipt.name).read_bytes())

        refused = self._refused()

        self.assertEqual(refused.code, "file_changed")
        self.assertIn(hashlib.sha256(changed).hexdigest(), refused.detail)
        self.assertIn(hashlib.sha256(VCF_BYTES).hexdigest(), refused.detail)

    def test_a_receipt_that_is_not_mains_is_refused_before_anything_runs(self):
        receipt = self.investigation / "bam" / "current-bam-acquisition-receipt.json"
        receipt.chmod(0o644)
        receipt.write_text("{}")

        refused = self._refused()

        self.assertEqual(refused.code, "receipt_invalid")
        self.assertIn(str(receipt), refused.detail)

    def test_a_receipt_that_cannot_be_read_is_refused_by_name(self):
        receipt = self.investigation / "bam" / "current-bam-acquisition-receipt.json"
        receipt.chmod(0o644)
        receipt.unlink()
        receipt.mkdir()

        refused = self._refused()

        self.assertEqual(refused.code, "run_file_unreadable")
        self.assertIn(str(receipt), refused.detail)

    def test_a_kept_probe_that_is_not_mains_is_refused_by_name(self):
        probe = self.investigation / "tools" / "observation-tools-probe.json"
        probe.chmod(0o644)
        probe.write_text("{}")

        with self.assertRaises(RunRefused) as refused:
            run_health(
                self.journey_file, self.root / "cache", self.output, pinned=self.pinned
            )

        self.assertEqual(refused.exception.code, "not_probed")
        self.assertIn(str(probe.parent), refused.exception.detail)
        self.assertEqual(self.pinned.invocations, [])
        self.assertFalse(self.output.exists())

    def test_a_cache_of_another_run_is_refused_before_anything_runs(self):
        cache = self.root / "cache"
        (cache / CACHE_MANIFEST).unlink()
        reference_cache(cache, self.outcome, journey_id="another-run")

        refused = self._refused()

        self.assertEqual(refused.code, "cache_other_run")
        self.assertIn("another-run", refused.detail)

    def test_a_refused_tool_run_stops_the_rest_and_no_manifest_is_written(self):
        self.pinned = _pinned_runner(
            mosdepth={
                "executable_sha256": "sha256:" + "0" * 64,
                "version_line": "mosdepth 0.3.13",
            }
        )

        with self.assertRaises(RunRefused) as refused:
            self._run()

        self.assertEqual(refused.exception.code, "identity_mismatch")
        self.assertTrue(refused.exception.detail.startswith("mosdepth: "))
        self.assertEqual(
            [invocation.argv[0] for invocation in self.pinned.invocations],
            ["mosdepth"],
        )
        self.assertNotIn("goleft", [i.argv[0] for i in self.probed.invocations])
        self.assertFalse((self.output / MANIFEST_NAME).exists())
        self.assertFalse((self.output / "samtools_index_bai").exists())

    def test_an_existing_output_directory_is_refused_before_anything_runs(self):
        self.output.mkdir()

        with self.assertRaises(RunRefused) as refused:
            self._run()

        self.assertEqual(refused.exception.code, "output_dir_exists")
        self.assertEqual(self.pinned.invocations + self.probed.invocations, [])
        self.assertEqual(list(self.output.iterdir()), [])

    def test_a_file_a_run_writes_without_a_health_role_is_refused(self):
        _WRITES["bcftools"] = ("bcftools-extra.txt",)
        self.addCleanup(_WRITES.pop, "bcftools")

        with self.assertRaises(RunRefused) as refused:
            self._run()

        self.assertEqual(refused.exception.code, "output_unnamed")
        self.assertIn("bcftools-extra.txt", refused.exception.detail)
        self.assertFalse((self.output / MANIFEST_NAME).exists())


class _Recorder:
    """A pinned-runner fake that writes `out.txt` with what it was given."""

    name = "recorder"

    def bind(self, entry):
        return pinned_binding(entry, runner=self.name, fields=("version_line",))

    def run(self, invocation):
        invocation.stdout_path.write_text("stats of " + " ".join(invocation.argv))
        return RunResult(
            exit_code=0,
            identity={"version_line": invocation.pin.version_line},
            executed_argv=invocation.argv,
            stderr=b"",
        )


class ReplayRunnerTests(unittest.TestCase):
    """A replay answers only the recorded run: same command, same inputs."""

    ENTRY = ToolEntry(
        pin=Pin(package="stats", version="1", version_line="stats 1"),
        argv=("stats", "{bam}", "{windows}"),
        inputs=("bam", "windows"),
        stdout="out.txt",
        outputs=("out.txt",),
    )

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bam = self.root / "run-a" / "aligned.bam"
        self.bam.parent.mkdir()
        self.bam.write_bytes(b"BAM A")
        self.recorded = self.root / "recorded"
        self.again = self.root / "again"
        for output_root in (self.recorded, self.again):
            output_root.mkdir()
            (output_root / "windows.bed").write_text("NC_1\t0\t10\n")
        self.record = run_tool(
            "stats",
            {"bam": self.bam, "windows": self.recorded / "windows.bed"},
            self.recorded / "stats",
            runner=_Recorder(),
            tools={"stats": self.ENTRY},
        )

    def _replay(self, bam, windows=None):
        return run_tool(
            "stats",
            {"bam": bam, "windows": windows or self.again / "windows.bed"},
            self.again / "stats",
            runner=ReplayRunner(self.recorded, self.again, {"stats": self.record}),
            tools={"stats": self.ENTRY},
        )

    def test_the_same_inputs_replay_the_recorded_outputs(self):
        replayed = self._replay(self.bam)

        self.assertEqual(replayed.outputs, self.record.outputs)
        self.assertEqual(replayed.runner, "replay")

    def test_another_runs_file_of_the_same_name_is_refused(self):
        other = self.root / "run-b" / "aligned.bam"
        other.parent.mkdir()
        other.write_bytes(b"BAM B")

        with self.assertRaises(RunRefused) as refused:
            self._replay(other)

        self.assertEqual(refused.exception.code, "tool_failed")
        self.assertIn(f"/inputs/aligned.bam is {other}", refused.exception.detail)
        self.assertFalse((self.again / "stats" / "out.txt").exists())

    def test_another_runs_file_is_refused_without_reading_its_bytes(self):
        other = self.root / "run-b" / "aligned.bam"
        other.parent.mkdir()
        other.write_bytes(b"BAM B")
        digested = []
        file_digest = hashlib.file_digest

        def recording(stream, name):
            digested.append(stream.name)
            return file_digest(stream, name)

        with (
            mock.patch.object(hashlib, "file_digest", recording),
            self.assertRaises(RunRefused),
        ):
            self._replay(other)

        self.assertNotIn(str(other), digested)

    def test_a_generated_input_with_other_bytes_is_refused(self):
        (self.again / "windows.bed").write_text("NC_1\t10\t20\n")

        with self.assertRaises(RunRefused) as refused:
            self._replay(self.bam)

        self.assertEqual(refused.exception.code, "tool_failed")
        self.assertIn("/inputs/windows.bed is", refused.exception.detail)

    def test_recorded_outputs_that_cannot_be_copied_are_a_failed_run(self):
        shutil.rmtree(self.recorded / "stats")

        with self.assertRaises(RunRefused) as refused:
            self._replay(self.bam)

        self.assertEqual(refused.exception.code, "tool_failed")
        self.assertIn(
            f"{self.recorded / 'stats'} cannot be copied", refused.exception.detail
        )


class HealthCommandTests(RunPrerequisites, unittest.TestCase):
    """`python -m tools.run_health health` prints the manifest or the typed refusal."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _main(self, journey_file, cache_dir):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main(
                [
                    "health",
                    "--journey",
                    str(journey_file),
                    "--cache-dir",
                    str(cache_dir),
                    "--output-dir",
                    str(self.root / "health"),
                ]
            )
        return status, json.loads(stdout.getvalue())

    def test_a_run_without_a_completed_execution_is_refused(self):
        approved = self.prerequisite("approved").journey_file

        status, printed = self._main(approved, self.root / "cache")

        self.assertEqual(status, 2)
        self.assertEqual(printed["refused"], "outcome_unavailable")
        self.assertFalse((self.root / "health").exists())

    def test_a_journey_that_cannot_be_read_is_refused_not_raised(self):
        journey = self.root / "run" / "journey.jsonl"
        journey.parent.mkdir()
        journey.write_text("not a journey\n")

        status, printed = self._main(journey, self.root / "cache")

        self.assertEqual(status, 2)
        self.assertEqual(printed["refused"], "run_unreadable")
        self.assertIn(str(journey), printed["detail"])
        self.assertFalse((self.root / "health").exists())

    def test_a_run_from_before_its_local_files_is_refused_by_name(self):
        executed = prerequisite(
            "executed", output_payloads={"VCF": VCF_BYTES, "aligned_bam": BAM_BYTES}
        )

        status, printed = self._main(executed.journey_file, self.root / "cache")

        self.assertEqual(status, 2)
        self.assertEqual(printed["refused"], "run_file_missing")
        self.assertIn("aligned_bam", printed["detail"])
        self.assertFalse((self.root / "health").exists())

    def test_a_missing_cache_and_a_run_without_its_probe_are_refused(self):
        journey_file = self.prerequisite(
            "executed",
            writable=True,
            output_payloads={"VCF": VCF_BYTES, "aligned_bam": BAM_BYTES},
        ).journey_file
        investigation = local_run_inputs(journey_file, vcf=VCF_BYTES, bam=BAM_BYTES)

        status, printed = self._main(journey_file, self.root / "cache")

        self.assertEqual(status, 2)
        self.assertEqual(printed["refused"], "file_missing")
        self.assertIn("no cache manifest", printed["detail"])

        outcome = read_completed_parabricks_result(str(journey_file))
        reference_cache(self.root / "cache", outcome)
        probe = investigation / "tools"
        for path in probe.iterdir():
            path.unlink()

        status, printed = self._main(journey_file, self.root / "cache")

        self.assertEqual(status, 2)
        self.assertEqual(printed["refused"], "not_probed")
        self.assertIn(str(probe), printed["detail"])
        self.assertFalse((self.root / "health").exists())
