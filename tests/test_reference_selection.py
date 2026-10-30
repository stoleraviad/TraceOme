"""The Run's reference sequences as one streaming selection (#464, owner 2026-10-29:
"primey for the ref we are running"): the records of the sequences outside the Run's
reference (alt loci, patches) are dropped whole, whatever the chunking. Input storage's
stream measure and the CPU worker call the same selection, so both derive the same
plain FASTA."""

from __future__ import annotations

import gzip
import hashlib
import io
import random
import unittest

from bio_neb.fasta_selection import FastaRecordSelection
from bio_neb.providers.gzip_content import GzipContentMeasure

FASTA = (
    b">NC_000001.11 Homo sapiens chromosome 1\nACGT\nTTGA\n"
    b">NT_187515.1 alt locus\nGGGG\n"
    b">NC_012920.1 mitochondrion\nCCAT\n"
)
# The worked example: the alt locus's header and sequence lines are gone, nothing else.
SELECTED = (
    b">NC_000001.11 Homo sapiens chromosome 1\nACGT\nTTGA\n"
    b">NC_012920.1 mitochondrion\nCCAT\n"
)


def _select(chunks, excluded=("NT_187515.1",)) -> bytes:
    selection = FastaRecordSelection(excluded)
    out = b"".join(selection.feed(chunk) for chunk in chunks)
    return out + selection.finish()


def _by_lines(fasta: bytes, excluded: set[bytes]) -> bytes:
    """An independent line-by-line reference: a header line opens a record, which is
    dropped when its first word is excluded."""
    out, drop = [], False
    for line in fasta.splitlines(keepends=True):
        if line.startswith(b">"):
            words = line[1:].split(None, 1)
            drop = bool(words) and words[0] in excluded
        if not drop:
            out.append(line)
    return b"".join(out)


class FastaRecordSelectionTests(unittest.TestCase):
    def test_an_excluded_record_is_dropped_whole_and_the_rest_pass_in_order(self):
        self.assertEqual(_select([FASTA]), SELECTED)

    def test_any_chunking_gives_the_same_bytes(self):
        for size in (1, 2, 3, 5, 7, 11, 64):
            with self.subTest(size=size):
                chunks = [FASTA[i : i + size] for i in range(0, len(FASTA), size)]
                self.assertEqual(_select(chunks), SELECTED)

    def test_no_exclusions_pass_every_byte(self):
        self.assertEqual(_select([FASTA[:20], FASTA[20:]], excluded=()), FASTA)

    def test_a_last_header_without_a_line_break_is_judged_at_finish(self):
        for excluded, tail in (((), b">NC_X.1 last"), (("NC_X.1",), b"")):
            with self.subTest(excluded=excluded):
                self.assertEqual(
                    _select([SELECTED, b">NC_X", b".1 last"], excluded=excluded),
                    SELECTED + tail,
                )

    def test_a_large_synthetic_fasta_matches_the_line_reference_in_random_chunks(self):
        rng = random.Random(464)
        records, excluded = [], set()
        for i in range(200):
            name = f"SEQ_{i}.1"
            if i % 3 == 0:
                excluded.add(name)
            width = rng.choice((60, 70, 80))
            sequence = bytes(
                rng.choice(b"ACGTN") for _ in range(rng.randrange(1, 9000))
            )
            lines = b"\n".join(
                sequence[j : j + width] for j in range(0, len(sequence), width)
            )
            records.append(b">" + name.encode() + b" synthetic\n" + lines + b"\n")
        fasta = b"".join(records)
        chunks, position = [], 0
        while position < len(fasta):
            size = rng.choice((1, 2, 61, 4096, 65536))
            chunks.append(fasta[position : position + size])
            position += size

        expected = _by_lines(fasta, {name.encode() for name in excluded})
        self.assertLess(len(expected), len(fasta))
        self.assertEqual(_select(chunks, excluded=excluded), expected)


class ControllerMeasureTests(unittest.TestCase):
    """The controller's gzip measure counts exactly the bytes the selection derives,
    the final flush once (#464 review r2 ST5: one derivation, ADR 0029/0035)."""

    def measured(self, plain: bytes, excluded=(), chunk=3) -> tuple[int, str]:
        measure = GzipContentMeasure(io.BytesIO(gzip.compress(plain)), excluded)
        while measure.read(chunk):
            pass
        return measure.identity()

    def test_a_final_header_without_a_line_break_is_measured_as_selected(self):
        for plain, excluded, expected in (
            (b">kept\nACGT\n>last", (), b">kept\nACGT\n>last"),
            (b">kept\nACGT\n>last", ("last",), b">kept\nACGT\n"),
        ):
            with self.subTest(excluded=excluded):
                self.assertEqual(
                    self.measured(plain, excluded),
                    (len(expected), hashlib.sha256(expected).hexdigest()),
                )

    def test_identity_is_stable_when_read_twice(self):
        measure = GzipContentMeasure(io.BytesIO(gzip.compress(b">a\nAC\n>b")), ())
        while measure.read(4):
            pass
        self.assertEqual(measure.identity(), measure.identity())


if __name__ == "__main__":
    unittest.main()
