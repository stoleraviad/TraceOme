"""Coverage and GC over one indexed slice of a Run's BAM and reference (lane H, #318).

The Run's probed samtools reads synthetic files whose depth and GC are known by
construction, so every expected value below is counted from the layout, not computed
the way the tool does.
"""

import contextlib
import importlib.metadata
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import pysam

from bio_neb.agentic_journey.runtime_config import InvestigationTools, ObservationTool
from bio_neb.investigation.item_evidence import CoverageMeasurement
from bio_neb.investigation.observation_defaults import CoverageDefaults
from bio_neb.investigation.observation_tools import probe_observation_tools
from tests.run_health_fakes import pysam_probed_runner
from tools.run_health import (
    TOOLS,
    ProbedProcessRunner,
    RunRefused,
    run_tool,
)
from tools.run_health.__main__ import main
from tools.run_health.region import RegionFiles, measure_item, measure_range
from tools.run_health.table import depth_entry

CONTIG = "NC_000099.1"
# Bases 1-50 are G/C, 51-100 are A/T; 101-120 are N.
SEQUENCE = "GC" * 25 + "AT" * 25 + "N" * 20


def _write_run_files(directory: Path) -> RegionFiles:
    """A reference with one contig and a BAM over it, each indexed beside itself.

    Reads (20 bases each, base quality 30, mapping quality 60 unless named):
    three at 11-30; a duplicate at 11-30; mapping quality 10 at 41-60; base
    quality 10 at 61-80. Under `CoverageDefaults` only the first three count.
    """
    reference = directory / "reference" / "reference.fa"
    reference.parent.mkdir(parents=True)
    reference.write_text(f">{CONTIG}\n{SEQUENCE}\n")
    pysam.faidx(str(reference))
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": CONTIG, "LN": len(SEQUENCE)}],
    }
    unsorted = directory / "unsorted.bam"
    reads = [
        ("kept-1", 10, 60, 30, 0),
        ("kept-2", 10, 60, 30, 0),
        ("kept-3", 10, 60, 30, 0),
        ("duplicate", 10, 60, 30, 0x400),
        ("low-mapq", 40, 10, 30, 0),
        ("low-baseq", 60, 60, 10, 0),
    ]
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as bam:
        for name, start, mapq, baseq, flag in reads:
            read = pysam.AlignedSegment(bam.header)
            read.query_name = name
            read.reference_id = 0
            read.reference_start = start
            read.mapping_quality = mapq
            read.flag = flag
            read.cigarstring = "20M"
            read.query_sequence = SEQUENCE[start : start + 20]
            read.query_qualities = pysam.qualitystring_to_array(chr(33 + baseq) * 20)
            bam.write(read)
    bam_path = directory / "bam" / "aligned.bam"
    bam_path.parent.mkdir()
    pysam.sort("-o", str(bam_path), str(unsorted))
    index = directory / "index" / "aligned.bam.csi"
    index.parent.mkdir()
    pysam.index("-c", "-o", str(index), str(bam_path))
    return RegionFiles(
        bam=bam_path,
        bam_index=index,
        reference=reference,
        reference_index=reference.with_name("reference.fa.fai"),
    )


class RegionCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.root = Path(directory.name)
        cls.files = _write_run_files(cls.root / "run")
        cls.runner = pysam_probed_runner(cls.root / "tools")

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "out"

    def test_a_typed_range_reports_depth_and_gc_under_the_coverage_defaults(self):
        coverage = measure_range(
            self.files, f"{CONTIG}:1-40", runner=self.runner, output_dir=self.output
        )

        summary = coverage.summary
        self.assertEqual((summary.contig, summary.start, summary.end), (CONTIG, 1, 40))
        self.assertEqual(summary.bases, 40)
        # Three reads over 20 of the 40 bases; the duplicate is excluded.
        self.assertEqual(summary.mean_depth, 1.5)
        self.assertEqual(summary.median_depth, 1.5)
        self.assertEqual((summary.min_depth, summary.max_depth), (0, 3))
        self.assertEqual(
            summary.fraction_at_least, {"1": 0.5, "10": 0.0, "20": 0.0, "30": 0.0}
        )
        self.assertEqual(summary.gc_fraction, 1.0)
        self.assertEqual(summary.n_bases, 0)
        self.assertEqual(summary.parameters, CoverageDefaults())

    def test_reads_below_the_quality_minimums_are_left_out_of_the_depth(self):
        coverage = measure_range(
            self.files, f"{CONTIG}:41-80", runner=self.runner, output_dir=self.output
        )

        # The mapping quality 10 read covers 41-60, the base quality 10 read 61-80.
        self.assertEqual(coverage.summary.max_depth, 0)
        # 41-50 are G/C, 51-80 A/T.
        self.assertEqual(coverage.summary.gc_fraction, 0.25)

    def test_the_read_filters_are_the_given_defaults_and_appear_in_the_record(self):
        defaults = CoverageDefaults(
            minimum_base_quality=0, minimum_mapping_quality=0, excluded_flags=("UNMAP",)
        )

        coverage = measure_range(
            self.files,
            f"{CONTIG}:11-80",
            runner=self.runner,
            output_dir=self.output,
            defaults=defaults,
        )

        # The duplicate now counts at 11-30, and both low-quality reads count.
        self.assertEqual(coverage.summary.max_depth, 4)
        self.assertEqual(coverage.summary.min_depth, 0)
        self.assertEqual(coverage.summary.parameters, defaults)
        self.assertEqual(
            coverage.depth_record.argv,
            (
                "samtools",
                "depth",
                "-a",
                "-s",
                "-q",
                "0",
                "-Q",
                "0",
                "-g",
                "UNMAP,SECONDARY,QCFAIL,DUP",
                "-G",
                "UNMAP",
                "-r",
                f"{CONTIG}:11-80",
                "/inputs/aligned.bam",
            ),
        )
        self.assertEqual(
            coverage.reference_record.argv,
            ("samtools", "faidx", "/inputs/reference.fa", f"{CONTIG}:11-80"),
        )
        self.assertIn("probe_receipt_sha256", coverage.depth_record.identity)

    def test_gc_counts_only_called_bases_and_n_bases_are_counted_apart(self):
        mixed = measure_range(
            self.files, f"{CONTIG}:91-120", runner=self.runner, output_dir=self.output
        )
        only_n = measure_range(
            self.files,
            f"{CONTIG}:101-120",
            runner=self.runner,
            output_dir=self.output.with_name("only-n"),
        )

        self.assertEqual((mixed.summary.gc_fraction, mixed.summary.n_bases), (0.0, 20))
        self.assertEqual(
            (only_n.summary.gc_fraction, only_n.summary.n_bases), (None, 20)
        )

    def test_an_item_gets_the_depth_at_its_position_and_its_window_summary(self):
        item = measure_item(
            self.files,
            CONTIG,
            20,
            flank_bases=15,
            runner=self.runner,
            output_dir=self.output,
        )

        self.assertEqual(
            item.measurement,
            CoverageMeasurement(
                parameters=CoverageDefaults(),
                depth=3,
                mean_depth=3.0,
                meets_minimum_depth=False,
            ),
        )
        window = item.window.summary
        self.assertEqual((window.start, window.end, window.bases), (5, 35, 31))
        # Positions 11-30 at depth 3 of 31.
        self.assertEqual(window.fraction_at_least["1"], round(20 / 31, 6))

    def test_an_item_meets_the_minimum_depth_it_was_measured_under(self):
        item = measure_item(
            self.files,
            CONTIG,
            11,
            flank_bases=0,
            runner=self.runner,
            output_dir=self.output,
            defaults=CoverageDefaults(minimum_depth=3),
        )

        self.assertEqual(
            (item.measurement.depth, item.measurement.meets_minimum_depth), (3, True)
        )
        self.assertEqual(item.window.summary.fraction_at_least["3"], 1.0)

    def test_an_item_window_stays_within_its_contig(self):
        item = measure_item(
            self.files,
            CONTIG,
            118,
            flank_bases=10,
            runner=self.runner,
            output_dir=self.output,
        )

        window = item.window.summary
        self.assertEqual((window.start, window.end), (108, 120))

    def test_a_bad_range_or_missing_index_is_refused_with_its_reason_before_any_run(
        self,
    ):
        missing = self.root / "absent" / "aligned.bam.csi"
        cases = {
            "NC_000099.1": "range_invalid",
            f"{CONTIG}:,-10": "range_invalid",
            f"{CONTIG}:1-,": "range_invalid",
            f"{CONTIG}:0-10": "range_invalid",
            f"{CONTIG}:30-20": "range_invalid",
            "NC_000098.1:1-10": "contig_unknown",
            f"{CONTIG}:100-121": "range_beyond_contig",
        }
        for text, code in cases.items():
            with self.subTest(text=text), self.assertRaises(RunRefused) as refused:
                measure_range(
                    self.files, text, runner=self.runner, output_dir=self.output
                )
            self.assertEqual(refused.exception.code, code)
        for name in ("bam_index", "reference_index"):
            files = self.files.model_copy(update={name: missing})
            with self.subTest(name=name), self.assertRaises(RunRefused) as refused:
                measure_range(
                    files, f"{CONTIG}:1-10", runner=self.runner, output_dir=self.output
                )
            self.assertEqual(refused.exception.code, "index_missing")
            self.assertIn(name, refused.exception.detail)
        self.assertFalse(self.output.exists())

    def test_a_range_longer_than_one_query_reads_is_refused(self):
        with mock.patch("tools.run_health.region.MAX_RANGE_BASES", 50):
            with self.assertRaises(RunRefused) as refused:
                measure_range(
                    self.files,
                    f"{CONTIG}:1-51",
                    runner=self.runner,
                    output_dir=self.output,
                )

        self.assertEqual(refused.exception.code, "range_too_long")
        self.assertFalse(self.output.exists())


class DepthEntryTests(unittest.TestCase):
    def test_a_region_entry_run_without_its_region_is_refused_before_any_run(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        files = _write_run_files(root / "run")

        for parameters in ({}, {"region": f"{CONTIG}:1-2", "flank": "3"}):
            with (
                self.subTest(parameters=parameters),
                self.assertRaises(RunRefused) as refused,
            ):
                run_tool(
                    "samtools_faidx_region",
                    {
                        "reference": files.reference,
                        "reference_index": files.reference_index,
                    },
                    root / "out",
                    runner=pysam_probed_runner(root / "tools"),
                    parameters=parameters,
                )
            self.assertEqual(refused.exception.code, "parameters_mismatch")
        self.assertFalse((root / "out").exists())

    def test_settings_samtools_depth_cannot_apply_are_refused(self):
        for defaults in (
            CoverageDefaults(applies_baq=True),
            CoverageDefaults(excluded_flags=("DUP", "NOT_A_FLAG")),
        ):
            with self.subTest(defaults=defaults), self.assertRaises(ValueError):
                depth_entry(defaults)

    def test_overlap_and_deletion_settings_choose_the_depth_options(self):
        argv = depth_entry(
            CoverageDefaults(
                overlapping_pairs="count_each", deletion_spanning_reads="included"
            )
        ).argv

        self.assertNotIn("-s", argv)
        self.assertIn("-J", argv)
        self.assertIn("-s", TOOLS["samtools_depth_region"].argv)
        self.assertNotIn("-J", TOOLS["samtools_depth_region"].argv)


def _stalled_samtools(directory: Path) -> ProbedProcessRunner:
    """A probe of `bio-neb-<tool>` entry points whose samtools answers `--version` and
    then stalls on any read."""
    bin_directory = directory / "bin"
    bin_directory.mkdir(parents=True)
    for tool in ("bcftools", "samtools"):
        executable = bin_directory / f"bio-neb-{tool}"
        executable.write_text(
            f"#!{sys.executable}\n"
            "import sys, time\n"
            "if sys.argv[1:] == ['--version']:\n"
            f"    sys.stdout.write('{tool} 1.24\\nUsing htslib 1.24\\n')\n"
            "    sys.exit(0)\n"
            "time.sleep(60)\n"
        )
        executable.chmod(0o755)
    pysam_tool = ObservationTool(
        package="pysam", package_version=importlib.metadata.version("pysam")
    )
    return ProbedProcessRunner(
        probe_observation_tools(
            InvestigationTools(bcftools=pysam_tool, samtools=pysam_tool),
            directory / "tools",
            bin_directory=bin_directory,
        )
    )


class StalledReadTests(unittest.TestCase):
    def test_a_read_past_the_coverage_timeout_is_refused_and_records_nothing(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        files = _write_run_files(root / "run")
        started = time.monotonic()

        # The lane's own process bound, set low for the test.
        with (
            mock.patch("tools.run_health.region.TIMEOUT_SECONDS", 1),
            self.assertRaises(RunRefused) as refused,
        ):
            measure_range(
                files,
                f"{CONTIG}:1-40",
                runner=_stalled_samtools(root / "stalled"),
                output_dir=root / "out",
            )

        self.assertEqual(refused.exception.code, "tool_timed_out")
        self.assertIn("past 1 s", refused.exception.detail)
        self.assertLess(time.monotonic() - started, 30)


class RangeCommandTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.files = _write_run_files(self.root / "run")
        pysam_probed_runner(self.root / "tools")

    def _main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = main(list(argv))
        return status, json.loads(out.getvalue())

    def _files(self):
        return (
            "--bam", str(self.files.bam),
            "--bam-index", str(self.files.bam_index),
            "--reference", str(self.files.reference),
            "--reference-index", str(self.files.reference_index),
            "--probe-dir", str(self.root / "tools"),
        )  # fmt: skip

    def test_a_typed_range_prints_its_summary_and_records(self):
        status, printed = self._main(
            "range",
            f"{CONTIG}:1-40",
            *self._files(),
            "--output-dir",
            str(self.root / "q"),
        )

        self.assertEqual(status, 0)
        self.assertEqual(printed["summary"]["mean_depth"], 1.5)
        self.assertEqual(printed["depth_record"]["tool"], "samtools_depth_region")

    def test_a_bad_range_prints_its_refusal(self):
        for text in (f"{CONTIG}:40-1", f"{CONTIG}:,-10"):
            with self.subTest(text=text):
                status, printed = self._main(
                    "range",
                    text,
                    *self._files(),
                    "--output-dir",
                    str(self.root / "q"),
                )

                self.assertEqual(status, 2)
                self.assertEqual(printed["refused"], "range_invalid")
                self.assertFalse((self.root / "q").exists())
