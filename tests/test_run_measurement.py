"""The Run's measurement over a real BAM with the Run's own samtools (#414).

Each test runs the product's probed ``bio-neb-samtools`` (and ``bio-neb-bcftools``
for Ti/Tv) over the tiny BAM of ``tests.tiny_bam``, whose reads are known: twelve
100-base reads starting one base apart at ``FIRST_READ_START``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bio_neb.investigation.local_bcftools import LocalBcftoolsAdapter
from bio_neb.investigation.run_measurement import (
    MANIFEST_NAME,
    ItemSpan,
    MeasuredFiles,
    MeasurementFault,
    measure_health,
    measure_items,
)
from bio_neb.investigation.vcf_observation import DeterministicVcfObservationTool
from tests.tiny_bam import (
    COVERED_POSITION,
    FIRST_READ_START,
    READ_COUNT,
    READ_LENGTH,
    SEQUENCES,
    tiny_bam,
)

_BIN = Path(sys.executable).parent
CONTIG, CONTIG_LENGTH = SEQUENCES[0]
MITOCHONDRION, MITOCHONDRION_LENGTH = SEQUENCES[1]


class CountingAdapter(LocalBcftoolsAdapter):
    """The local process adapter, counting each samtools subcommand; a named
    subcommand (with ``-b``: the batch) times out."""

    def __init__(self, fail=None) -> None:
        super().__init__()
        self.fail = fail
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv, *, timeout_seconds, stdout_path=None):
        self.calls.append(tuple(argv))
        if self.fail is not None and self.fail(argv):
            raise subprocess.TimeoutExpired(argv, timeout_seconds)
        return super().run(
            argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
        )


class MeasurementTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        bam = root / "aligned.bam"
        bam.write_bytes(tiny_bam())
        (root / "health").mkdir()
        self.files = MeasuredFiles(
            bam=bam,
            bam_sha256=hashlib.sha256(tiny_bam()).hexdigest(),
            vcf_sha256="b" * 64,
            samtools=str(_BIN / "bio-neb-samtools"),
            preparation=None,
            directory=root / "health",
        )

    def test_each_item_gets_its_spans_depth_or_why_none(self) -> None:
        last_read_end = FIRST_READ_START + READ_COUNT - 1 + READ_LENGTH - 1
        spans = [
            # Inside all twelve reads: depth 12, the minimum of 10 met.
            ItemSpan("covered", CONTIG, COVERED_POSITION, COVERED_POSITION),
            # The reads' first bases: depth 1 to 11, below the minimum with reads.
            ItemSpan("edge", CONTIG, FIRST_READ_START, FIRST_READ_START + 10),
            # No read: depth zero, a plain absence, not unsure.
            ItemSpan("empty", CONTIG, 100, 120),
            # A region whose second half lies past the last read.
            ItemSpan("region", CONTIG, last_read_end - 9, last_read_end + 10),
            ItemSpan("unknown", "chrX", 5, 5),
            ItemSpan(
                "beyond", MITOCHONDRION, MITOCHONDRION_LENGTH, MITOCHONDRION_LENGTH + 1
            ),
        ]
        outcomes = measure_items(self.files, spans, LocalBcftoolsAdapter())

        def depth(name):
            measurement = outcomes[name]["measurement"]
            return (
                measurement.depth,
                measurement.mean_depth,
                measurement.meets_minimum_depth,
                outcomes[name]["batch_flags"],
            )

        self.assertEqual(depth("covered"), (12, 12.0, True, ()))
        self.assertEqual(depth("edge"), (1, 6.0, False, ("low_depth",)))
        self.assertEqual(depth("empty"), (0, 0.0, False, ()))
        # The reads end one base apart: the region's first ten bases are 10 down
        # to 1 deep (55 in all), then ten past every read; 55 over 20 bases.
        self.assertEqual(depth("region"), (0, 2.75, False, ("low_depth",)))
        self.assertEqual(outcomes["unknown"]["not_measured"]["code"], "contig_unknown")
        self.assertEqual(
            outcomes["beyond"]["not_measured"]["code"], "span_beyond_contig"
        )

    def test_a_failed_batch_reads_each_leftover_item_alone(self) -> None:
        adapter = CountingAdapter(fail=lambda argv: "-b" in argv)
        spans = [
            ItemSpan("covered", CONTIG, COVERED_POSITION, COVERED_POSITION),
            ItemSpan("empty", CONTIG, 100, 120),
        ]
        outcomes = measure_items(self.files, spans, adapter)
        self.assertEqual(outcomes["covered"]["measurement"].depth, 12)
        self.assertEqual(outcomes["empty"]["measurement"].depth, 0)
        alone = [argv for argv in adapter.calls if "-r" in argv]
        self.assertEqual(
            [argv[argv.index("-r") + 1] for argv in alone],
            [
                f"{CONTIG}:{COVERED_POSITION}-{COVERED_POSITION}",
                f"{CONTIG}:100-120",
            ],
        )

    def test_an_item_whose_own_read_fails_records_why(self) -> None:
        adapter = CountingAdapter(fail=lambda argv: "depth" in argv)
        outcomes = measure_items(
            self.files,
            [ItemSpan("covered", CONTIG, COVERED_POSITION, COVERED_POSITION)],
            adapter,
        )
        self.assertEqual(
            outcomes["covered"]["not_measured"]["code"], "measurement_failed"
        )
        self.assertEqual(
            outcomes["covered"]["not_measured"]["detail"],
            "timeout: bio-neb-samtools depth ran past 3600 s",
        )

    def test_health_states_samtools_numbers_and_keeps_its_manifest(self) -> None:
        numbers = measure_health(self.files, LocalBcftoolsAdapter())
        length = sum(length for _, length in SEQUENCES)
        covered = READ_COUNT - 1 + READ_LENGTH
        self.assertEqual(numbers.reads, READ_COUNT)
        self.assertEqual(numbers.mapped_percent, 100.0)
        self.assertEqual(numbers.covered_contigs, 1)
        self.assertEqual(numbers.covered_bases, covered)
        self.assertAlmostEqual(numbers.covered_fraction, covered / length)
        # Every read base passes the filters: the genome's mean depth is the read
        # bases over its length (samtools states each sequence's mean to six
        # significant digits).
        self.assertAlmostEqual(
            numbers.mean_depth * length, READ_COUNT * READ_LENGTH, delta=0.01
        )
        # No prepared VCF: Ti/Tv was not read, and the fact says why (ST1).
        self.assertEqual(
            (numbers.ti_tv, numbers.ti_tv_reason), (None, "vcf_unprepared")
        )
        (manifest,) = self.files.directory.glob(f"run-*/{MANIFEST_NAME}")
        content = manifest.read_bytes()
        self.assertEqual(numbers.manifest_sha256, hashlib.sha256(content).hexdigest())
        self.assertEqual(
            [row["contig"] for row in json.loads(content)["contigs"]],
            [name for name, _ in SEQUENCES],
        )


class FaultTests(unittest.TestCase):
    """Only a measuring process's own fault is routed; an error of ours is not
    (ADR 0031: an invariant holds the Run until a code fix)."""

    setUp = MeasurementTests.setUp

    def test_a_process_that_exits_nonzero_is_a_typed_fault(self) -> None:
        files = MeasuredFiles(
            **{**self.files.__dict__, "bam": self.files.bam.with_name("absent.bam")}
        )
        with self.assertRaises(MeasurementFault) as fault:
            measure_health(files, LocalBcftoolsAdapter())
        self.assertEqual(fault.exception.code, "process_failed")
        self.assertIn("bio-neb-samtools flagstat exited", fault.exception.detail)

    def test_a_timeout_is_a_typed_fault(self) -> None:
        adapter = CountingAdapter(fail=lambda argv: "flagstat" in argv)
        with self.assertRaises(MeasurementFault) as fault:
            measure_health(self.files, adapter)
        self.assertEqual(fault.exception.code, "timeout")

    def test_an_error_of_ours_propagates_and_records_no_item(self) -> None:
        class Broken(LocalBcftoolsAdapter):
            def run(self, argv, *, timeout_seconds, stdout_path=None):
                if "depth" in argv:
                    raise KeyError("a defect, not the process")
                return super().run(
                    argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
                )

        with self.assertRaises(KeyError):
            measure_items(
                self.files,
                [ItemSpan("covered", CONTIG, COVERED_POSITION, COVERED_POSITION)],
                Broken(),
            )


class TransitionTransversionTests(unittest.TestCase):
    def _ti_tv(self, records: str):
        with tempfile.TemporaryDirectory() as temporary:
            vcf = Path(temporary) / "variants.vcf"
            vcf.write_text(
                "##fileformat=VCFv4.2\n##contig=<ID=c1,length=1000>\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + records
            )
            preparation = SimpleNamespace(
                runtime=SimpleNamespace(executable_path=str(_BIN / "bio-neb-bcftools")),
                derived_bgzf=SimpleNamespace(path=str(vcf)),
            )
            return DeterministicVcfObservationTool().transition_transversion(
                preparation, LocalBcftoolsAdapter(), timeout_seconds=60
            )

    def test_ti_tv_is_copied_from_bcftools_stats(self) -> None:
        # Two transitions (A>G, C>T) and one transversion (A>C).
        self.assertEqual(
            self._ti_tv(
                "c1\t10\t.\tA\tG\t50\tPASS\t.\nc1\t20\t.\tA\tC\t50\tPASS\t.\n"
                "c1\t30\t.\tC\tT\t50\tPASS\t.\n"
            ),
            2.0,
        )

    def test_a_vcf_without_a_transversion_has_no_ti_tv(self) -> None:
        self.assertIsNone(self._ti_tv("c1\t10\t.\tA\tG\t50\tPASS\t.\n"))


if __name__ == "__main__":
    unittest.main()
