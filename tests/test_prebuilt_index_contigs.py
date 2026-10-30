"""A prebuilt index's contigs beside NCBI's copied sequence rows (#464, ADR 0049): each
of the Run's requested sequences paired, missing or of another length, and each extra
contig with the NCBI role of the row that names it. Facts only: which index is the
Run's is the planning model's choice."""

from __future__ import annotations

import unittest

from bio_neb.failure import ProviderError
from bio_neb.prebuilt_index.contigs import pair_contigs, requested_sequences
from bio_neb.providers.ncbi_assembly_report import SequenceRecord
from bio_neb.providers.ncbi_datasets import parse_assembly_report

ACCESSION = "GCF_000001405.40"
REPORT = (
    "# Sequence-Name\tSequence-Role\tAssigned-Molecule\tAssigned-Molecule-Location/Type"
    "\tGenBank-Accn\tRelationship\tRefSeq-Accn\tAssembly-Unit\tSequence-Length"
    "\tUCSC-style-name\n"
    "1\tassembled-molecule\t1\tChromosome\tCM000663.2\t=\tNC_000001.11"
    "\tPrimary Assembly\t248956422\tchr1\n"
    "HSCHR1_CTG1_UNLOCALIZED\tunlocalized-scaffold\t1\tChromosome\tKI270706.1\t="
    "\tNT_187361.1\tPrimary Assembly\t175055\tchr1_KI270706v1_random\n"
    "MT\tassembled-molecule\tMT\tMitochondrion\tJ01415.2\t=\tNC_012920.1"
    "\tnon-nuclear\t16569\tchrM\n"
    "HSCHR1_1_CTG3\talt-scaffold\t1\tChromosome\tKI270762.1\t=\tNT_187515.1"
    "\tALT_REF_LOCI_1\t354444\tchr1_KI270762v1_alt\n"
)
SEQUENCES = parse_assembly_report(REPORT.encode())
UCSC = (
    ("chr1", 248956422),
    ("chr1_KI270706v1_random", 175055),
    ("chrM", 16569),
)


class PairingTests(unittest.TestCase):
    def test_every_requested_sequence_pairs_by_a_name_ncbi_lists_it_under(self):
        contigs = pair_contigs(ACCESSION, SEQUENCES, UCSC)

        self.assertEqual(
            [(p.requested_name, p.candidate_name, p.status) for p in contigs.requested],
            [
                ("NC_000001.11", "chr1", "paired"),
                ("NT_187361.1", "chr1_KI270706v1_random", "paired"),
                ("NC_012920.1", "chrM", "paired"),
            ],
        )
        self.assertEqual((contigs.renamed, contigs.extras), (3, ()))
        self.assertTrue(contigs.holds_every_requested_sequence)

    def test_an_extra_contig_carries_the_role_of_the_ncbi_row_naming_it(self):
        # The human alt loci: extras, disclosed with NCBI's role, not a refusal.
        contigs = pair_contigs(
            ACCESSION,
            SEQUENCES,
            (*UCSC, ("chr1_KI270762v1_alt", 354444), ("chrEBV", 171823)),
        )

        self.assertEqual(
            [(e.name, e.length, e.ncbi_role) for e in contigs.extras],
            [
                ("chr1_KI270762v1_alt", 354444, "alt-scaffold"),
                ("chrEBV", 171823, None),
            ],
        )
        self.assertTrue(contigs.holds_every_requested_sequence)

    def test_each_contig_s_ncbi_name_in_the_run_s_naming(self):
        """Owner I7 (#464): execution renames each contig to NCBI's name, a pair to
        its requested name and an extra to its row's RefSeq name (GenBank for a GCA);
        an extra no row names keeps its own."""
        extras = (("chr1_KI270762v1_alt", 354444), ("chrEBV", 171823))
        contigs = pair_contigs(ACCESSION, SEQUENCES, (*UCSC, *extras))
        self.assertEqual(
            [(e.name, e.ncbi_name) for e in contigs.extras],
            [("chr1_KI270762v1_alt", "NT_187515.1"), ("chrEBV", None)],
        )
        self.assertEqual(
            contigs.ncbi_names(),
            {
                "chr1": "NC_000001.11",
                "chr1_KI270706v1_random": "NT_187361.1",
                "chrM": "NC_012920.1",
                "chr1_KI270762v1_alt": "NT_187515.1",
            },
        )
        genbank = pair_contigs("GCA_000001405.29", SEQUENCES, (*UCSC, *extras))
        self.assertEqual(
            [(e.name, e.ncbi_name) for e in genbank.extras],
            [("chr1_KI270762v1_alt", "KI270762.1"), ("chrEBV", None)],
        )
        # An index already under NCBI's names renames nothing.
        named = pair_contigs(
            ACCESSION,
            SEQUENCES,
            (
                ("NC_000001.11", 248956422),
                ("NT_187361.1", 175055),
                ("NC_012920.1", 16569),
            ),
        )
        self.assertEqual(named.ncbi_names(), {})

    def test_a_missing_or_other_length_sequence_is_stated_and_not_held(self):
        contigs = pair_contigs(
            ACCESSION, SEQUENCES, (("chr1", 248956422), ("chrM", 16570))
        )

        self.assertEqual(
            [(p.requested_name, p.status) for p in contigs.requested],
            [
                ("NC_000001.11", "paired"),
                ("NT_187361.1", "missing"),
                ("NC_012920.1", "other_length"),
            ],
        )
        self.assertFalse(contigs.holds_every_requested_sequence)

    def test_the_fit_reads_the_main_assembly_and_shows_a_missing_minor_scaffold(self):
        """Owner, #459 (2026-10-30: "we are looking for main asmenlby oll the minonr
        one areless importedt"): an index holds the Run's reference when each of its
        assembled molecules pairs at its length; a missing or other-length unlocalized
        or unplaced scaffold is stated with its NCBI role, never a refusal."""
        held = pair_contigs(
            ACCESSION, SEQUENCES, (("chr1", 248956422), ("chrM", 16569))
        )
        self.assertEqual(
            [(p.requested_name, p.role, p.status) for p in held.requested],
            [
                ("NC_000001.11", "assembled-molecule", "paired"),
                ("NT_187361.1", "unlocalized-scaffold", "missing"),
                ("NC_012920.1", "assembled-molecule", "paired"),
            ],
        )
        self.assertTrue(held.holds_every_requested_sequence)
        # A chromosome of another length is not the Run's chromosome.
        other = pair_contigs(
            ACCESSION, SEQUENCES, (("chr1", 248956421), ("chrM", 16569))
        )
        self.assertFalse(other.holds_every_requested_sequence)

    def test_one_sequence_named_by_two_contigs_is_ambiguous(self):
        contigs = pair_contigs(ACCESSION, SEQUENCES, (*UCSC, ("1", 248956422)))

        self.assertEqual(contigs.ambiguous, ("NC_000001.11",))
        self.assertFalse(contigs.holds_every_requested_sequence)

    def test_a_genbank_assembly_is_requested_and_paired_by_its_genbank_names(self):
        contigs = pair_contigs(
            "GCA_000001405.29",
            SEQUENCES,
            (("CM000663.2", 248956422), ("chr1_KI270706v1_random", 175055)),
        )

        self.assertEqual(
            [(p.requested_name, p.candidate_name, p.status) for p in contigs.requested],
            [
                ("CM000663.2", "CM000663.2", "paired"),
                ("KI270706.1", "chr1_KI270706v1_random", "paired"),
                ("J01415.2", None, "missing"),
            ],
        )
        self.assertEqual(contigs.renamed, 1)


C_ELEGANS = (
    "# Assembly name:  WBcel235\n"
    "# Sequence-Name\tSequence-Role\tAssigned-Molecule\tAssigned-Molecule-Location/Type"
    "\tGenBank-Accn\tRelationship\tRefSeq-Accn\tAssembly-Unit\tSequence-Length"
    "\tUCSC-style-name\n"
    "I\tassembled-molecule\tI\tChromosome\tBX284601.5\t=\tNC_003279.8"
    "\tPrimary Assembly\t15072434\tchrI\n"
    "II\tassembled-molecule\tII\tChromosome\tBX284602.5\t=\tNC_003280.10"
    "\tPrimary Assembly\t15279421\tchrII\n"
    "MtDNA\tassembled-molecule\tMtDNA\tMitochondrion\tX54252.1\t=\tNC_001328.1"
    "\tnon-nuclear\t13794\tchrM\n"
)


class ReferenceSequenceTests(unittest.TestCase):
    def test_the_run_s_reference_is_its_assembled_and_placed_scaffold_sequences_by_role(
        self,
    ) -> None:
        # The Run's reference is every sequence whose NCBI role is an assembled
        # molecule or an unlocalized or unplaced scaffold, whatever its unit's name;
        # alt loci and patches are left out (ADR 0049: a role NCBI's report copies,
        # no unit-name set).
        alt = (
            "HSCHR1_ALT\talt-scaffold\tI\tChromosome\tKI1.1\t=\tNT_1.1"
            "\tALT_REF_LOCI_1\t5000\tchrI_alt\n"
        )
        unlocalized = (
            "I_random\tunlocalized-scaffold\tI\tChromosome\tKI9.1\t=\tNT_9.1"
            "\tC57BL/6J\t4000\tchrI_random\n"
        )
        patch = (
            "HG1_PATCH\tfix-patch\tI\tChromosome\tKN1.1\t=\tNW_1.1"
            "\tPATCHES\t3000\tchrI_fix\n"
        )
        sequences = parse_assembly_report(
            (C_ELEGANS + alt + unlocalized + patch).encode()
        )
        self.assertEqual(
            [name for name, _ in requested_sequences("GCF_000002985.6", sequences)],
            ["NC_003279.8", "NC_003280.10", "NC_001328.1", "NT_9.1"],
        )


class AssemblyReportTests(unittest.TestCase):
    """NCBI's assembly report copied row by row (#365, #467), the rows the Run's
    reference sequences and a candidate's pairing read."""

    def test_each_sequence_keeps_its_names_and_length(self) -> None:
        sequences = parse_assembly_report(C_ELEGANS.encode())
        self.assertEqual(
            sequences[0],
            SequenceRecord(
                sequence_name="I",
                sequence_role="assembled-molecule",
                assigned_molecule="I",
                location_type="Chromosome",
                genbank="BX284601.5",
                relationship="=",
                refseq="NC_003279.8",
                assembly_unit="Primary Assembly",
                length=15072434,
                ucsc="chrI",
            ),
        )
        self.assertEqual(len(sequences), 3)

    def test_na_is_no_name_and_a_short_row_is_refused(self) -> None:
        row = "U\tunplaced-scaffold\tna\tna\tKN1.1\t<>\tna\tPrimary Assembly\t10\tna\n"
        (sequence,) = parse_assembly_report(row.encode())
        self.assertEqual((sequence.refseq, sequence.ucsc), (None, None))
        with self.assertRaises(ProviderError):
            parse_assembly_report(b"I\tassembled-molecule\t15\n")


if __name__ == "__main__":
    unittest.main()
