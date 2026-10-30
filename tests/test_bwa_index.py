"""The one BWA-index reader (#449): what a FASTA and its index files state, and every
finding against them, one table per file kind.

Each case starts from the tiny valid eight-role bundle (`local_bundle_documents`: one
contig `chr1`, ACGTACGT) and changes one or two files, so each expected finding follows
from the change made. Admission refuses on the first finding and the reference-index fit
reports every one; those callers keep their own cases (`tests/test_reference_index.py`).
"""

from __future__ import annotations

import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from bio_neb.parabricks.bwa_index import (
    Ann,
    Contig,
    FormatInvalid,
    read_ann,
    read_bwa_index,
    read_fai,
    read_fasta,
    sequence_dictionary,
)
from tests.bwa_bundle_fakes import local_bundle_documents

M5 = hashlib.md5(b"ACGTACGT", usedforsecurity=False).hexdigest()


class _Bundle(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.paths = {
            role: self.root / name
            for role, (name, _) in local_bundle_documents().items()
        }
        self.write()

    def write(self, changes=()):
        """The valid bundle again, with `changes` (role, bytes) made to it."""
        bodies = {role: body for role, (_, body) in local_bundle_documents().items()}
        for role, body in {**bodies, **dict(changes)}.items():
            self.paths[role].write_bytes(body)

    def read(self, *, changes=(), dict_file=True):
        self.write(changes)
        files = {
            role: path
            for role, path in self.paths.items()
            if role != "genomic_fasta" and (dict_file or role != "reference_dict")
        }
        return read_bwa_index(self.paths["genomic_fasta"], files)

    def findings(self, reading):
        return [(f.kind, f.reason, f.role) for f in reading.findings]


class ValidIndexTests(_Bundle):
    def test_a_valid_index_states_its_contigs_and_files_with_no_finding(self):
        reading = self.read()

        self.assertEqual(reading.findings, ())
        self.assertEqual(reading.contigs, (Contig("chr1", 8, M5),))
        self.assertEqual((reading.forward_base_count, reading.pac_trailer), (8, 0))
        self.assertEqual(reading.ann, Ann((8, 1, 11), ("chr1",), ((0, 8, 0),)))
        documents = local_bundle_documents()
        self.assertEqual(
            [(f.role, f.byte_count, f.sha256) for f in (reading.fasta, *reading.files)],
            [
                (role, len(body), hashlib.sha256(body).hexdigest())
                for role, (_, body) in documents.items()
            ],
        )

    def test_an_index_without_its_dict_is_read_without_the_m5_check(self):
        reading = self.read(
            changes=[
                ("reference_dict", b"@SQ\tSN:chr1\tLN:8\tM5:" + b"0" * 32 + b"\n")
            ],
            dict_file=False,
        )

        self.assertEqual(reading.findings, ())
        self.assertNotIn("reference_dict", [f.role for f in reading.files])

    def test_an_index_missing_a_bwa_role_is_refused_before_any_read(self):
        with self.assertRaisesRegex(ValueError, "a BWA index has the roles"):
            read_bwa_index(
                self.paths["genomic_fasta"], {"fasta_fai": self.paths["fasta_fai"]}
            )


class FastaFindingTests(_Bundle):
    def test_a_fasta_that_is_not_one_is_the_only_finding(self):
        cases = (
            (b">\xff\nACGT\n", UnicodeDecodeError),
            (b"> \t\nACGT\n", type(None)),
            (b">chr1\nACGTXCGT\n", type(None)),
            (b"ACGT\n", type(None)),
        )
        for body, cause in cases:
            with self.subTest(body=body):
                reading = self.read(
                    changes=[("genomic_fasta", body), ("bwa_pac", b"wrong")]
                )

                self.assertEqual(
                    self.findings(reading),
                    [("structure", "fasta_structure_invalid", "genomic_fasta")],
                )
                self.assertIsInstance(reading.findings[0].cause, cause)
                self.assertEqual(reading.files, ())

    def test_read_fasta_gives_each_contigs_m5_of_the_upper_cased_sequence(self):
        path = self.root / "masked.fa"
        path.write_bytes(b">one first\nacgt\nACGT\n>two\nNNAC\n")

        reading = read_fasta(path)

        self.assertEqual(
            reading.contigs,
            (
                Contig("one", 8, M5),
                Contig("two", 4, hashlib.md5(b"NNAC").hexdigest()),
            ),
        )
        self.assertEqual(reading.file.byte_count, path.stat().st_size)


class TextFileFindingTests(_Bundle):
    """`.fai`, `.dict`, `.ann` and `.amb`: each file's structure, then its comparison."""

    def test_each_text_file_kind_has_its_findings(self):
        # A file that disagrees also fails the comparisons made against it.
        fai_and_ann = [
            ("mismatch", "fasta_fai_mismatch", "fasta_fai"),
            ("mismatch", "ann_fasta_mismatch", "bwa_ann"),
        ]
        ann_and_amb = [
            ("mismatch", "ann_fasta_mismatch", "bwa_ann"),
            ("mismatch", "amb_ann_mismatch", "bwa_amb"),
        ]
        cases = (
            ("fasta_fai", b"chr1\t9\t14\t8\t9\n", fai_and_ann),
            ("fasta_fai", b"chr1\t8\t14\t8\n", "structure", "fai_structure_invalid"),
            ("fasta_fai", b"chr1\tx\t14\t8\t9\n", "structure", "fai_structure_invalid"),
            ("fasta_fai", b"\xffchr1\n", "structure", "fai_structure_invalid"),
            (
                "reference_dict",
                b"@SQ\tSN:chr1\tLN:8\tM5:" + b"0" * 32 + b"\n",
                "mismatch",
                "fasta_dict_m5_mismatch",
            ),
            (
                "reference_dict",
                b"@SQ\tSN:chr1\tLN:8\n",
                "structure",
                "dict_structure_invalid",
            ),
            (
                "reference_dict",
                b"@SQ\tSN:chr1\tLN:x\tM5:" + M5.encode() + b"\n",
                "structure",
                "dict_structure_invalid",
            ),
            (
                "bwa_ann",
                b"8 1 11\n0 chr2 fixture\n0 8 0\n",
                "mismatch",
                "ann_fasta_mismatch",
            ),
            ("bwa_ann", b"9 1 11\n0 chr1 fixture\n0 8 0\n", ann_and_amb),
            ("bwa_ann", b"8 0 11\n", ann_and_amb),
            (
                "bwa_ann",
                b"8 1 11\n0 chr1 fixture\n",
                "structure",
                "ann_structure_invalid",
            ),
            ("bwa_amb", b"9 1 0\n", "mismatch", "amb_ann_mismatch"),
            ("bwa_amb", b"8 1 1\n", "mismatch", "amb_ann_mismatch"),
            ("bwa_amb", b"8 1 1\n3 1 N\n", "mismatch", "amb_ann_mismatch"),
            ("bwa_amb", b"8 1 1\n0 1\n", "structure", "amb_structure_invalid"),
            ("bwa_amb", b"8 1\n", "structure", "amb_structure_invalid"),
        )
        for role, body, *expected in cases:
            with self.subTest(role=role, body=body):
                reading = self.read(changes=[(role, body)])

                self.assertEqual(
                    self.findings(reading),
                    expected[0] if len(expected) == 1 else [(*expected, role)],
                )

    def test_a_number_that_does_not_parse_keeps_its_parser_cause(self):
        # A number that does not parse: the role's structure miss, the ValueError its
        # cause.
        cases = (
            ("bwa_ann", b"8 x 11\n0 chr1 fixture\n0 8 0\n", "ann_structure_invalid"),
            ("fasta_fai", b"chr1\tx\t14\t8\t9\n", "fai_structure_invalid"),
            (
                "reference_dict",
                b"@SQ\tSN:chr1\tLN:x\tM5:" + M5.encode() + b"\n",
                "dict_structure_invalid",
            ),
        )
        for role, body, reason in cases:
            with self.subTest(role=role):
                reading = self.read(changes=[(role, body)])

                finding = reading.findings[0]
                self.assertEqual((finding.reason, finding.role), (reason, role))
                self.assertIsInstance(finding.cause, ValueError)

    def test_a_comparison_needing_an_unreadable_file_is_not_made(self):
        # The .ann compares with the .fai, and the .amb with the .ann.
        reading = self.read(changes=[("fasta_fai", b"chr1\t8\n"), ("bwa_ann", b"\xff")])

        self.assertEqual(
            self.findings(reading),
            [
                ("structure", "fai_structure_invalid", "fasta_fai"),
                ("structure", "ann_structure_invalid", "bwa_ann"),
            ],
        )


class BinaryFileFindingTests(_Bundle):
    """`.pac` and the `.bwt`/`.sa` headers."""

    def test_each_binary_file_kind_has_its_findings(self):
        header = (3, 1, 4, 8, 16)
        cases = (
            ("bwa_pac", b"wrong", "mismatch", "pac_length_or_trailer_mismatch"),
            (
                "bwa_pac",
                b"\x1b\x1b\0\x01",
                "mismatch",
                "pac_length_or_trailer_mismatch",
            ),
            ("bwa_bwt", b"\0" * 8, "structure", "bwt_sa_header_truncated"),
            ("bwa_sa", b"\0" * 48, "structure", "bwt_sa_header_truncated"),
            (
                "bwa_bwt",
                struct.pack("<5Q", 3, 1, 4, 8, 15) + b"\0" * 4,
                [
                    ("mismatch", "bwt_header_invalid", "bwa_bwt"),
                    # The .sa header repeats the .bwt's.
                    ("mismatch", "sa_header_invalid", "bwa_sa"),
                ],
            ),
            (
                "bwa_sa",
                struct.pack("<7Q", *header, 16, 16) + b"\0" * 8,
                "mismatch",
                "sa_header_invalid",
            ),
            (
                "bwa_sa",
                struct.pack("<7Q", *header, 32, 16) + b"\0" * 9,
                "mismatch",
                "bwt_sa_alignment_invalid",
            ),
        )
        for role, body, *expected in cases:
            with self.subTest(role=role, body=body):
                reading = self.read(changes=[(role, body)])

                self.assertEqual(
                    self.findings(reading),
                    expected[0] if len(expected) == 1 else [(*expected, role)],
                )

    def test_every_finding_is_returned_in_role_order(self):
        reading = self.read(
            changes=[
                ("bwa_sa", struct.pack("<7Q", 3, 1, 4, 8, 16, 16, 16) + b"\0" * 9),
                ("bwa_pac", b"wrong"),
                ("fasta_fai", b"chr1\t9\t14\t8\t9\n"),
            ]
        )

        self.assertEqual(
            self.findings(reading),
            [
                ("mismatch", "fasta_fai_mismatch", "fasta_fai"),
                ("mismatch", "ann_fasta_mismatch", "bwa_ann"),
                ("mismatch", "pac_length_or_trailer_mismatch", "bwa_pac"),
                ("mismatch", "sa_header_invalid", "bwa_sa"),
                ("mismatch", "bwt_sa_alignment_invalid", "bwa_sa"),
            ],
        )


class ReadFindingTests(_Bundle):
    def test_an_empty_file_stops_the_reading_at_that_role(self):
        reading = self.read(changes=[("bwa_bwt", b""), ("bwa_sa", b"")])

        self.assertEqual(
            self.findings(reading),
            [("read", "local_role_not_positive_regular_file", "bwa_bwt")],
        )
        self.assertEqual(
            [f.role for f in reading.files],
            ["fasta_fai", "reference_dict", "bwa_amb", "bwa_ann"],
        )

    def test_a_missing_file_passes_its_os_error_to_the_caller(self):
        self.paths["bwa_sa"].unlink()
        files = {r: p for r, p in self.paths.items() if r != "genomic_fasta"}

        with self.assertRaises(FileNotFoundError) as raised:
            read_bwa_index(self.paths["genomic_fasta"], files)

        self.assertEqual(raised.exception.filename, str(self.paths["bwa_sa"]))


class ReadAnnTests(unittest.TestCase):
    def test_ann_bytes_give_the_header_names_and_layout(self):
        ann = read_ann(b"20 2 11\n0 one (null)\n0 12 1\n0 two a description\n12 8 0\n")

        self.assertEqual(
            ann, Ann((20, 2, 11), ("one", "two"), ((0, 12, 1), (12, 8, 0)))
        )

    def test_bytes_that_are_not_an_ann_are_refused_with_its_reason(self):
        for body in (b"", b"20 2\n", b"20 1 11\n0 one\n0 12\n", b"20 1 11\n"):
            with self.subTest(body=body):
                with self.assertRaises(FormatInvalid) as raised:
                    read_ann(body)
                self.assertEqual(raised.exception.reason, "ann_structure_invalid")


class FaiAndDictTests(unittest.TestCase):
    def test_a_dict_is_written_from_the_fai_s_contigs_in_its_order(self):
        """Owner I7 (#464): a streamed prebuilt index's `.dict` is written from its
        renamed `.fai`, `@HD` then one `@SQ SN LN` per contig."""
        contigs = read_fai(b"NC_000001.11\t80\t14\t60\t61\nchrM\t9\t166\t60\t61\n")
        self.assertEqual(contigs, [("NC_000001.11", 80), ("chrM", 9)])
        self.assertEqual(
            sequence_dictionary(contigs),
            b"@HD\tVN:1.0\tSO:unsorted\n"
            b"@SQ\tSN:NC_000001.11\tLN:80\n"
            b"@SQ\tSN:chrM\tLN:9\n",
        )

    def test_bytes_that_are_not_a_fai_are_refused_with_its_reason(self):
        for body in (b"chr1\teighty\t6\t60\t61\n", b"chr1\t80\t6\n"):
            with self.subTest(body=body):
                with self.assertRaises(FormatInvalid) as raised:
                    read_fai(body)
                self.assertEqual(raised.exception.reason, "fai_structure_invalid")


if __name__ == "__main__":
    unittest.main()
