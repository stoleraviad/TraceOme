"""One batched coverage pass over a Run's Work items (lane H, #405).

The fixture is cut from HG001's local files (#405, `cut_fixture.py` in its spike folder):
five rebased slices around six of its items. `items.json` keeps each item's depths as
today's one-process-per-item read reported them on the original BAM, so every expected
depth below comes from the original data, not from the pass under test.
"""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pysam
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from bio_neb.investigation.observation_defaults import CoverageDefaults
from bio_neb.parabricks.result import read_completed_parabricks_result
from tests.parabricks_operation_fakes import VCF_BYTES
from tests.run_health_fakes import local_run_inputs, pysam_probed_runner
from tests.run_prerequisites import RunPrerequisites
from tools.run_health import RunRefused
from tools.run_health.__main__ import main
from tools.run_health.cache import MANIFEST_NAME as CACHE_MANIFEST
from tools.run_health.cache import CachedFile, RunCache
from tools.run_health.health import (
    MANIFEST_NAME,
    CoveredRegions,
    HealthArguments,
    HealthFile,
    HealthInput,
    HealthManifest,
    derive_arguments,
)
from tools.run_health.items import (
    ITEMS_NAME,
    ItemTarget,
    measure_items,
    measure_items_alone,
)
from tools.run_health.region import RegionFiles, measure_range
from tools.run_health.replay import ReplayRunner

FIXTURE = Path(__file__).parent / "fixtures" / "run_health" / "hg001-slices"
FLANK = 100


def _targets(rows) -> tuple[ItemTarget, ...]:
    return tuple(
        ItemTarget(
            work_item_id=row["work_item_id"],
            contig=row["original"]["contig"],
            one_based_position=row["rebased_position"],
            ref=row["original"]["ref"],
        )
        for row in rows
    )


class BatchedItemCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        root = Path(directory.name)
        run = root / "run"
        shutil.copytree(FIXTURE, run)
        pysam.index("-c", str(run / "aligned.bam"))
        pysam.faidx(str(run / "reference.fa"))
        cls.files = RegionFiles(
            bam=run / "aligned.bam",
            bam_index=run / "aligned.bam.csi",
            reference=run / "reference.fa",
            reference_index=run / "reference.fa.fai",
        )
        cls.rows = json.loads((FIXTURE / "items.json").read_text())["items"]
        cls.runner = pysam_probed_runner(root / "tools")
        cls.output = root / "out"
        cls.coverage = measure_items(
            cls.files,
            _targets(cls.rows),
            flank_bases=FLANK,
            runner=cls.runner,
            output_dir=cls.output,
        )

    def _row(self, label):
        return next(row for row in self.rows if row["display_label"] == label)

    def _span_of(self, row):
        (span,) = (
            span
            for span in self.coverage.spans
            if row["work_item_id"] in span.work_item_ids
        )
        return span

    def test_one_slice_one_depth_and_one_reference_run_read_every_window(self):
        records = (
            self.coverage.slice_record,
            self.coverage.depth_record,
            self.coverage.reference_record,
        )

        self.assertEqual(
            [record.tool for record in records],
            [
                "samtools_view_windows",
                "samtools_depth_windows",
                "samtools_faidx_windows",
            ],
        )
        view, depth, reference = (record.argv for record in records)
        self.assertEqual(
            view[:5], ("samtools", "view", "-M", "-L", "/inputs/windows.bed")
        )
        self.assertEqual(
            depth[-3:], ("-b", "/inputs/windows.bed", "/inputs/windows.bam")
        )
        self.assertNotIn("-a", depth)
        self.assertEqual(reference[-2:], ("-r", "/inputs/windows.txt"))
        self.assertEqual(
            sorted((self.output / "windows.bed").read_text().splitlines()),
            sorted(
                f"{span.contig}\t{span.window.start - 1}\t{span.window.end}"
                for span in self.coverage.spans
            ),
        )

    def test_each_window_holds_the_depths_the_original_bam_reported(self):
        for row in self.rows:
            with self.subTest(item=row["display_label"]):
                span = self._span_of(row)
                depths = row["original_depths"]
                window = span.window
                self.assertEqual(window.bases, len(depths))
                self.assertEqual(
                    (window.min_depth, window.max_depth), (min(depths), max(depths))
                )
                self.assertEqual(window.mean_depth, round(sum(depths) / len(depths), 6))

    def test_each_window_summary_equals_todays_read_of_that_range(self):
        for index, span in enumerate(self.coverage.spans):
            with self.subTest(span=f"{span.contig}:{span.span_start}"):
                alone = measure_range(
                    self.files,
                    f"{span.contig}:{span.window.start}-{span.window.end}",
                    runner=self.runner,
                    output_dir=self.output.with_name(f"alone-{index}"),
                )
                self.assertEqual(span.window, alone.summary)

    def test_an_item_reads_its_least_depth_over_its_span(self):
        snv, deletion = self._row("rs429358"), self._row("rs756427959")

        for row, width in ((snv, 1), (deletion, 3)):
            with self.subTest(item=row["display_label"]):
                span = self._span_of(row)
                self.assertEqual(
                    (span.span_start, span.span_end),
                    (row["rebased_position"], row["rebased_position"] + width - 1),
                )
                start = span.span_start - span.window.start
                expected = min(row["original_depths"][start : start + width])
                self.assertEqual(span.measurement.depth, expected)
                self.assertEqual(
                    span.measurement.meets_minimum_depth,
                    expected >= CoverageDefaults().minimum_depth,
                )

    def test_an_item_with_no_reads_reports_zero_depth_over_its_whole_window(self):
        span = self._span_of(self._row("rs2032652"))

        self.assertEqual(span.measurement.depth, 0)
        self.assertFalse(span.measurement.meets_minimum_depth)
        self.assertEqual((span.window.max_depth, span.window.mean_depth), (0, 0.0))
        self.assertEqual(span.window.fraction_at_least["1"], 0.0)
        self.assertIsNotNone(span.window.gc_fraction)

    def test_a_recorded_batch_replays_to_the_same_spans_without_running_samtools(self):
        again = self.output.with_name("replayed")
        replay = ReplayRunner(
            self.output,
            again,
            {
                "slice": self.coverage.slice_record,
                "depth": self.coverage.depth_record,
                "reference": self.coverage.reference_record,
            },
        )

        replayed = measure_items(
            self.files,
            _targets(self.rows),
            flank_bases=FLANK,
            runner=replay,
            output_dir=again,
        )

        self.assertEqual(
            (replayed.spans, replayed.flagged, replayed.unsure),
            (self.coverage.spans, self.coverage.flagged, self.coverage.unsure),
        )
        self.assertEqual(replayed.slice_record.runner, "replay")
        self.assertEqual(replayed.depth_record.argv, self.coverage.depth_record.argv)

    def test_the_batch_records_its_wall_clock_seconds(self):
        self.assertGreater(self.coverage.seconds, 0)

    def test_each_span_lists_its_positions_without_coverage(self):
        covered, uncovered = self._row("rs429358"), self._row("rs2032652")

        self.assertEqual(
            (
                self._span_of(covered).uncovered_positions,
                self._span_of(covered).uncovered,
            ),
            ((), "none"),
        )
        span = self._span_of(uncovered)
        self.assertEqual(
            (span.uncovered_positions, span.uncovered),
            ((uncovered["rebased_position"],), "whole"),
        )

    def test_a_span_partly_without_coverage_names_the_uncovered_positions(self):
        row = self._row("rs372913400")
        start = row["rebased_position"] - FLANK
        depths = row["original_depths"]
        # The first base with no read after one with a read, in the original BAM.
        (zero,) = [
            start + index
            for index in range(1, len(depths))
            if depths[index - 1] > 0 and depths[index] == 0
        ][:1]
        deletion = ItemTarget(
            work_item_id="work-item:deletion",
            contig=row["original"]["contig"],
            one_based_position=zero - 1,
            ref="AA",
        )

        coverage = measure_items(
            self.files,
            (deletion,),
            flank_bases=FLANK,
            runner=self.runner,
            output_dir=self.output.with_name("partly"),
        )

        (span,) = coverage.spans
        self.assertEqual((span.uncovered_positions, span.uncovered), ((zero,), "part"))
        self.assertEqual(span.measurement.depth, 0)

    def test_low_depth_with_reads_nearby_is_unsure_and_no_reads_at_all_is_not(self):
        low = self._row("rs7412")
        depths = low["original_depths"]

        self.assertLess(depths[FLANK], CoverageDefaults().minimum_depth)
        self.assertGreater(max(depths), 0)
        self.assertEqual(
            [(flag.work_item_id, flag.code) for flag in self.coverage.unsure],
            [(low["work_item_id"], "low_depth")],
        )

    def test_an_n_base_in_a_span_makes_its_item_unsure(self):
        row = self._row("rs429358")
        reference = self.files.reference.parent / "n-reference" / "reference.fa"
        reference.parent.mkdir(exist_ok=True)
        lines = (FIXTURE / "reference.fa").read_text().splitlines()
        contig = f">{row['original']['contig']}"
        header = lines.index(contig)
        # 60 bases per line: the item's base is line `(position - 1) // 60` after it.
        line = header + 1 + (row["rebased_position"] - 1) // 60
        column = (row["rebased_position"] - 1) % 60
        lines[line] = lines[line][:column] + "N" + lines[line][column + 1 :]
        reference.write_text("\n".join(lines) + "\n")
        pysam.faidx(str(reference))
        files = self.files.model_copy(
            update={
                "reference": reference,
                "reference_index": reference.with_name("reference.fa.fai"),
            }
        )

        coverage = measure_items(
            files,
            _targets([row]),
            flank_bases=FLANK,
            runner=self.runner,
            output_dir=self.output.with_name("n-base"),
        )

        self.assertEqual(
            [(flag.work_item_id, flag.code) for flag in coverage.unsure],
            [(row["work_item_id"], "reference_n")],
        )

    def test_work_items_at_one_span_share_one_measurement(self):
        ids = [
            row["work_item_id"]
            for row in self.rows
            if row["display_label"] == "rs4149056"
        ]

        (span,) = (span for span in self.coverage.spans if ids[0] in span.work_item_ids)
        self.assertEqual(sorted(span.work_item_ids), sorted(ids))
        self.assertEqual(len(self.coverage.spans), len(self.rows) - 1)

    def test_an_item_off_the_reference_is_flagged_and_the_others_are_measured(self):
        known = _targets(self.rows[:1])
        unknown = ItemTarget(
            work_item_id="work-item:unknown",
            contig="NC_000098.1",
            one_based_position=5,
            ref="A",
        )
        beyond = ItemTarget(
            work_item_id="work-item:beyond",
            contig=known[0].contig,
            one_based_position=10_000_000,
            ref="A",
        )

        coverage = measure_items(
            self.files,
            (*known, unknown, beyond),
            flank_bases=FLANK,
            runner=self.runner,
            output_dir=self.output.with_name("flagged"),
        )

        self.assertEqual(
            [(flag.work_item_id, flag.code) for flag in coverage.flagged],
            [
                ("work-item:unknown", "contig_unknown"),
                ("work-item:beyond", "span_beyond_contig"),
            ],
        )
        self.assertEqual(
            [span.work_item_ids for span in coverage.spans], [(known[0].work_item_id,)]
        )

    def test_a_list_with_nothing_to_measure_runs_nothing(self):
        unknown = ItemTarget(
            work_item_id="work-item:unknown",
            contig="NC_000098.1",
            one_based_position=5,
            ref="A",
        )
        output = self.output.with_name("nothing")

        coverage = measure_items(
            self.files,
            (unknown,),
            flank_bases=FLANK,
            runner=self.runner,
            output_dir=output,
        )

        self.assertEqual(
            (
                coverage.spans,
                coverage.slice_record,
                coverage.depth_record,
                coverage.reference_record,
            ),
            ((), None, None, None),
        )
        self.assertEqual(len(coverage.flagged), 1)
        self.assertEqual(list(output.iterdir()), [])

    def test_a_batch_with_nothing_to_measure_still_refuses_an_existing_directory(self):
        output = self.output.with_name("kept-empty")
        output.mkdir()
        (output / "items-coverage.json").write_text("original record")

        with self.assertRaises(RunRefused) as refused:
            measure_items(
                self.files, (), flank_bases=FLANK, runner=self.runner, output_dir=output
            )

        self.assertEqual(refused.exception.code, "output_dir_exists")
        self.assertEqual(
            (output / "items-coverage.json").read_text(), "original record"
        )

    def test_distinct_spans_that_share_one_window_each_read_it_once(self):
        # Derived setup: the fixture's reference with one N at the start of
        # NC_000019.10, so a window read twice would count two N bases.
        root = self.output.with_name("shared")
        run = root / "run"
        shutil.copytree(self.files.bam.parent, run)
        fasta = run / "reference.fa"
        lines = fasta.read_text().splitlines()
        first = lines.index(">NC_000019.10") + 1
        lines[first] = "N" + lines[first][1:]
        fasta.write_text("\n".join(lines) + "\n")
        files = self.files.model_copy(
            update={
                "bam": run / "aligned.bam",
                "bam_index": run / "aligned.bam.csi",
                "reference": fasta,
                "reference_index": run / "reference.fa.fai",
            }
        )
        spans = tuple(
            ItemTarget(
                work_item_id=f"work-item:{position}",
                contig="NC_000019.10",
                one_based_position=position,
                ref="T",
            )
            for position in (601, 602)
        )

        coverage = measure_items(
            files, spans, flank_bases=2000, runner=self.runner, output_dir=root / "out"
        )
        whole = measure_range(
            files,
            f"NC_000019.10:1-{coverage.spans[0].window.end}",
            runner=self.runner,
            output_dir=root / "whole",
        )

        self.assertEqual(
            [span.work_item_ids for span in coverage.spans],
            [("work-item:601",), ("work-item:602",)],
        )
        self.assertEqual(whole.summary.n_bases, 1)
        for span in coverage.spans:
            self.assertEqual(span.window, whole.summary)
        self.assertEqual(
            len((root / "out" / "windows.txt").read_text().splitlines()), 1
        )

    def test_an_existing_output_directory_is_refused_and_left_as_it_was(self):
        output = self.output.with_name("existing")
        output.mkdir()
        (output / "kept.txt").write_text("kept")

        with self.assertRaises(RunRefused) as refused:
            measure_items(
                self.files,
                _targets(self.rows),
                flank_bases=FLANK,
                runner=self.runner,
                output_dir=output,
            )
        self.assertEqual(refused.exception.code, "output_dir_exists")
        with self.assertRaises(RunRefused) as refused:
            measure_items_alone(
                self.files,
                _targets(self.rows),
                flank_bases=FLANK,
                runner=self.runner,
                output_dir=output,
            )
        self.assertEqual(refused.exception.code, "output_dir_exists")
        self.assertEqual([path.name for path in output.iterdir()], ["kept.txt"])

    def test_a_missing_index_is_refused_before_any_run(self):
        files = self.files.model_copy(update={"bam_index": self.output / "absent.csi"})
        output = self.output.with_name("refused")

        with self.assertRaises(RunRefused) as refused:
            measure_items(
                files,
                _targets(self.rows),
                flank_bases=FLANK,
                runner=self.runner,
                output_dir=output,
            )

        self.assertEqual(refused.exception.code, "index_missing")
        self.assertFalse(output.exists())

    def test_each_window_read_by_itself_equals_its_batch_window(self):
        alone = measure_items_alone(
            self.files,
            _targets(self.rows),
            flank_bases=FLANK,
            runner=self.runner,
            output_dir=self.output.with_name("alone"),
        )

        self.assertEqual(
            [read.summary for read in alone],
            [span.window for span in self.coverage.spans],
        )


class RunItemsTests(RunPrerequisites, unittest.TestCase):
    """`python -m tools.run_health items` measures a Run's items over the Run's own
    BAM, the index its health report wrote and the cached reference."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bam = (FIXTURE / "aligned.bam").read_bytes()
        self.journey_file = self.prerequisite(
            "executed",
            writable=True,
            output_payloads={"VCF": VCF_BYTES, "aligned_bam": self.bam},
        ).journey_file
        investigation = local_run_inputs(
            self.journey_file, vcf=VCF_BYTES, bam=self.bam, tools=pysam_probed_runner
        )
        outcome = read_completed_parabricks_result(str(self.journey_file))
        self.cache = self.root / "cache"
        (self.cache / "reference").mkdir(parents=True)
        shutil.copy(FIXTURE / "reference.fa", self.cache / "reference")
        pysam.faidx(str(self.cache / "reference" / "reference.fa"))
        self.cache.joinpath(CACHE_MANIFEST).write_text(
            RunCache(
                journey_id=outcome.journey_id,
                journey_file=outcome.journey_file,
                bucket=outcome.storage.bucket,
                files={
                    role: CachedFile(
                        path=str(self.cache / "reference" / name),
                        object_key=name,
                        byte_count=1,
                        sha256="0" * 64,
                        kept=False,
                    )
                    for role, name in (
                        ("reference_fasta", "reference.fa"),
                        ("reference_fai", "reference.fa.fai"),
                    )
                },
            ).model_dump_json()
        )
        self.health = self.root / "health"
        self._health_report(outcome, investigation / "bam" / "aligned.bam")
        self.rows = json.loads((FIXTURE / "items.json").read_text())["items"]
        targets = mock.patch(
            "tools.run_health.items.work_item_targets",
            return_value=_targets(self.rows),
        )
        targets.start()
        self.addCleanup(targets.stop)

    def _health_report(self, outcome, bam, *, sha256=None):
        """A health report of the Run's BAM: its CSI index and its manifest."""
        index = self.health / "samtools_index" / "aligned.bam.csi"
        index.parent.mkdir(parents=True)
        pysam.index("-c", "-o", str(index), str(bam))
        (verified,) = [
            item for item in outcome.verified_objects if item.role == "aligned_bam"
        ]
        manifest = HealthManifest(
            journey_id=outcome.journey_id,
            journey_file=outcome.journey_file,
            run_root=outcome.run_root,
            execution_id=outcome.execution_id,
            outcome_event_id=outcome.outcome_event_id,
            inputs=(
                HealthInput(
                    role="aligned_bam",
                    location="s3://bucket/aligned.bam",
                    byte_count=verified.byte_count,
                    sha256=sha256 or verified.sha256,
                    path=str(bam),
                    receipt="receipt.json",
                ),
            ),
            files=(
                HealthFile(
                    role="bam_csi",
                    path="samtools_index/aligned.bam.csi",
                    byte_count=index.stat().st_size,
                    sha256="0" * 64,
                    record="samtools_index",
                ),
            ),
            records={},
            arguments=HealthArguments(
                reference_index="reference.fa.fai",
                indexcov_excluded="^$",
                sample_names=(),
            ),
            covered=CoveredRegions(
                window_bases=500,
                minimum_mean_depth=1.0,
                path="covered-regions.bed",
                byte_count=0,
                sha256="0" * 64,
                contigs=(),
            ),
            entry_seconds={},
        )
        (self.health / MANIFEST_NAME).write_text(manifest.model_dump_json())

    def _command(self, *extra):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = main(
                [
                    "items",
                    "--journey",
                    str(self.journey_file),
                    "--cache-dir",
                    str(self.cache),
                    "--health-dir",
                    str(self.health),
                    "--output-dir",
                    str(self.root / "items"),
                    *extra,
                ]
            )
        return code, json.loads(printed.getvalue())

    def test_the_command_prints_the_batch_over_the_runs_bam_with_a_probe(self):
        row = next(row for row in self.rows if row["display_label"] == "rs2032652")
        probe = f"{row['original']['contig']}:{row['rebased_position'] + 50}"

        code, printed = self._command("--probe", probe)

        self.assertEqual(code, 0)
        spans = {tuple(span["work_item_ids"]): span for span in printed["spans"]}
        self.assertEqual(spans[(f"probe:{probe}",)]["measurement"]["depth"], 0)
        for item in self.rows:
            with self.subTest(item=item["display_label"]):
                (span,) = [s for ids, s in spans.items() if item["work_item_id"] in ids]
                window = item["original_depths"]
                self.assertEqual(span["window"]["max_depth"], max(window))
        self.assertEqual(
            printed["slice_record"]["inputs"]["bam"],
            str(self.journey_file.parent / "investigation" / "bam" / "aligned.bam"),
        )

    def test_check_reads_the_batch_and_checks_its_unsure_item_with_the_runs_files(
        self,
    ):
        code, batch = self._command()
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads((self.root / "items" / ITEMS_NAME).read_text()), batch
        )
        (unsure,) = batch["unsure"]

        (span,) = [
            span
            for span in batch["spans"]
            if unsure["work_item_id"] in span["work_item_ids"]
        ]
        window = f"{span['contig']}:{span['window']['start']}-{span['window']['end']}"

        def respond(messages, info):
            if not any(isinstance(message, ModelResponse) for message in messages):
                return ModelResponse(
                    parts=[ToolCallPart("measure_range", {"region": window})]
                )
            return ModelResponse(
                parts=[
                    TextPart(
                        json.dumps(
                            {
                                "verdict": "low_coverage",
                                "region": window,
                                "reason": "scripted",
                            }
                        )
                    )
                ]
            )

        printed = io.StringIO()
        with (
            mock.patch(
                "tools.run_health.check.run_model", return_value=FunctionModel(respond)
            ),
            contextlib.redirect_stdout(printed),
        ):
            code = main(
                [
                    "check",
                    "--journey",
                    str(self.journey_file),
                    "--cache-dir",
                    str(self.cache),
                    "--health-dir",
                    str(self.health),
                    "--items-dir",
                    str(self.root / "items"),
                    "--output-dir",
                    str(self.root / "checks"),
                    "--request-cap",
                    "12",
                ]
            )

        self.assertEqual(code, 0)
        checks = json.loads(printed.getvalue())
        self.assertEqual(checks["request_cap"], 12)
        (check,) = checks["checks"]
        self.assertEqual(check["flags"], [unsure])
        self.assertEqual(check["verdict"]["verdict"], "low_coverage")
        self.assertEqual(check["result"]["region"], window)

    def test_a_batch_result_that_is_not_one_is_refused_before_any_model_request(self):
        (self.root / "items").mkdir()
        (self.root / "items" / ITEMS_NAME).write_text("{}")
        printed = io.StringIO()

        with (
            mock.patch("tools.run_health.check.run_model") as run_model,
            contextlib.redirect_stdout(printed),
        ):
            code = main(
                [
                    "check",
                    "--journey",
                    str(self.journey_file),
                    "--cache-dir",
                    str(self.cache),
                    "--health-dir",
                    str(self.health),
                    "--items-dir",
                    str(self.root / "items"),
                    "--output-dir",
                    str(self.root / "checks"),
                ]
            )

        refused = json.loads(printed.getvalue())
        self.assertEqual((code, refused["refused"]), (2, "items_unreadable"))
        self.assertIn(ITEMS_NAME, refused["detail"])
        run_model.assert_not_called()
        self.assertFalse((self.root / "checks").exists())

    def test_a_health_report_of_another_bam_is_refused_before_any_run(self):
        shutil.rmtree(self.health)
        outcome = read_completed_parabricks_result(str(self.journey_file))
        self._health_report(
            outcome,
            self.journey_file.parent / "investigation" / "bam" / "aligned.bam",
            sha256="e" * 64,
        )

        code, printed = self._command()

        self.assertEqual((code, printed["refused"]), (2, "health_other_bam"))
        self.assertIn("e" * 64, printed["detail"])
        self.assertFalse((self.root / "items").exists())


class DerivedHealthArgumentsTests(unittest.TestCase):
    """indexcov's contigs and MultiQC's renames come from the Run's reference and files."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        shutil.copy(FIXTURE / "reference.fa", self.root)
        pysam.faidx(str(self.root / "reference.fa"))
        self.fai = self.root / "reference.fa.fai"
        self.inputs = tuple(
            HealthInput(
                role=role,
                location=f"s3://bucket/{name}",
                byte_count=1,
                sha256="0" * 64,
                path=f"/run/investigation/{directory}/{name}",
                receipt=f"/run/investigation/{directory}/receipt.json",
            )
            for role, directory, name in (
                ("aligned_bam", "bam", "aligned.bam"),
                ("VCF", "vcf", "variants.vcf"),
            )
        )

    def _fai(self, keep):
        lines = [
            line
            for line in self.fai.read_text().splitlines()
            if keep(line.split("\t")[0])
        ]
        path = self.root / "kept.fai"
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_indexcov_leaves_out_no_contig_whatever_its_name(self):
        # ADR 0049: no pattern judges a sequence name by its shape, so RefSeq
        # scaffolds and other names alike stay in.
        named = self.root / "named.fai"
        named.write_text("chr1\t100\t6\t60\t61\nNC_000002.12\t100\t200\t60\t61\n")
        for fai in (self.fai, named):
            with self.subTest(fai=fai.name):
                self.assertEqual(
                    derive_arguments(fai, self.inputs).indexcov_excluded, "^$"
                )

    def test_each_sample_name_multiqc_uses_maps_to_its_run_role(self):
        names = derive_arguments(self.fai, self.inputs).sample_names

        # The names MultiQC gives the product's summary files: no rename of a name
        # MultiQC does not use, and indexcov's sample kept.
        self.assertEqual(
            names,
            (
                ("coverage", "aligned_bam"),
                ("samtools-flagstat", "aligned_bam"),
                ("samtools-idxstats", "aligned_bam"),
                ("samtools-stats", "aligned_bam"),
                ("variants", "VCF"),
            ),
        )

    def test_bcftools_sample_follows_the_runs_vcfs_own_name(self):
        bam, vcf = self.inputs
        inputs = (
            bam,
            vcf.model_copy(update={"path": "/run/investigation/vcf/sample-1.vcf"}),
        )

        names = dict(derive_arguments(self.fai, inputs).sample_names)

        self.assertEqual(names["sample-1"], "VCF")
        self.assertNotIn("variants", names)


if __name__ == "__main__":
    unittest.main()
