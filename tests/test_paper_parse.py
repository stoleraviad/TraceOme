"""The general parser: one passage read into the typed slots the next readers check (#395).

Owner decisions on #395 (2026-10-19): the model's reading is the source and code keeps a
value only where this parser reads it (decision 1); the parser reads the slots main's request
item and lane Rf's change type, by shape and not by paper layout (decision 2).

Passages are copied from #349's retained answers (h1, s3, s17) and #333's s16 Methods line;
``Hg38`` is a synthetic spelling (CASQ2, the paper that writes it, is #396's unseen case).
"""

import unittest
from decimal import Decimal

from bio_neb.target_request.paper_text import read, written_number

H1_KCNH2_ROW = (
    "KCNH2\tchr7:150951552/NM_000238.4\tc.1841C>T p. (Ala614Val)\tMissense variant"
    "\tD/PD\trs199472944\tPathogenic"
)
S17_SENTENCE = (
    "The only associated ‘HIGH’ effect nonsense variant was on chr4 (EquCab3.0 "
    "chr4:54108297) in exon 26 of RAPGEF5 (c.2624C>A p.Ser875*) and was exclusive to TBs."
)
S3_SENTENCE = (
    "the previously-described point mutations at PBP2 (Y127X), SGF73 (E294X), and UPF3 "
    "(G6W) [4], and an additional point mutation intergenic to RRN3 and YPK1 "
    "(chr11: 207,469:A>C)."
)
S17_S6_ROW = "CM009151.1\t4\t45478863\t0\tA\tG\t825\t0\tAB=0.695652;TYPE=snp"


class GeneralParserTests(unittest.TestCase):
    def test_a_table_row_is_read_cell_by_cell_into_its_slots(self):
        readings = read(H1_KCNH2_ROW)

        self.assertIn(150951552, readings.positions)
        self.assertIn("chr7", readings.contigs)
        self.assertIn("rs199472944", readings.rsids)
        self.assertEqual(readings.transcripts, frozenset({"NM_000238.4"}))
        self.assertEqual(readings.changes, frozenset({"c.1841C>T", "p. (Ala614Val)"}))
        # A coding change states transcript bases, not the genome's REF and ALT.
        self.assertEqual(readings.allele_pairs, frozenset())

    def test_a_sentence_gives_its_version_locus_and_changes_as_written(self):
        readings = read(S17_SENTENCE)

        self.assertIn("EquCab3.0", readings.versions)
        self.assertIn("chr4", readings.contigs)
        self.assertIn(54108297, readings.positions)
        self.assertTrue({"c.2624C>A", "p.Ser875*"} <= readings.changes)
        # The closing parenthesis belongs to the sentence, not to the change.
        self.assertNotIn("p.Ser875*)", readings.changes)

    def test_protein_changes_in_one_letter_code_and_a_grouped_position_are_read(self):
        readings = read(S3_SENTENCE)

        self.assertTrue({"Y127X", "E294X", "G6W"} <= readings.changes)
        self.assertIn("chr11", readings.contigs)
        self.assertIn(207469, readings.positions)
        self.assertIn(("A", "C"), readings.allele_pairs)

    def test_a_vcf_rows_base_cells_state_no_direction(self):
        readings = read(S17_S6_ROW)

        self.assertTrue({"CM009151.1", "4"} <= readings.contigs)
        self.assertIn(45478863, readings.positions)
        # Two base cells alone state no direction (Standards r1 S1); the model names a
        # row's REF and ALT cells, whatever its columns are called.
        self.assertEqual(readings.allele_pairs, frozenset())

    def test_lane_ps_own_contig_rule_holds_its_spellings(self):
        """The reader keeps lane P's spelling rule as it ran (owner #383: a move), kept
        by the owner as a decision row (#486 Q4 a; ADR 0049, owner #467 R2 a, R4 a)."""
        from bio_neb.paper_reading.answer import contig_name

        self.assertEqual(
            {contig_name(c) for c in ("chr4", "4", "Chromosome 4", "CHR4")}, {"4"}
        )
        self.assertNotEqual(contig_name("chr4"), contig_name("chr14"))

    def test_version_names_are_read_by_shape_in_any_case(self):
        for text, name in (
            ("Reads were aligned to Hg38 with BWA.", "Hg38"),
            ("We used the dog CanFam 3.1 reference genome assembly.", "CanFam 3.1"),
            ("on GRCh38.p14 and GCF_000001405.40.", "GRCh38.p14"),
            ("on GRCh38.p14 and GCF_000001405.40.", "GCF_000001405.40"),
            ("the ROS_Cfam_1.0 assembly", "ROS_Cfam_1.0"),
            ("Reads were aligned to canfam 3.1.", "canfam 3.1"),
        ):
            with self.subTest(name=name):
                self.assertIn(name, read(text).versions)
        # A trailing full stop ends the sentence, not the name.
        self.assertNotIn("GCF_000001405.40.", read("and GCF_000001405.40.").versions)

    def test_a_value_not_written_exactly_is_not_read(self):
        readings = read("A 1.2 Mb deletion near 54.1 Mb on chromosome 4 in 4 dogs.")

        self.assertNotIn(54100000, readings.positions)
        self.assertNotIn(1200000, readings.positions)
        # "chromosome 4" names a contig; a bare count does not.
        self.assertEqual(readings.contigs, frozenset({"chromosome 4", "4"}))
        # A lone capital letter in prose is a word, not an allele.
        self.assertEqual(readings.allele_pairs, frozenset())

    def test_a_genotype_states_no_ref_and_alt(self):
        # "A/C" is a genotype: two bases with no direction, so no REF and ALT.
        readings = read(
            "Both SNPs (chr 3: 1,234,567) held true. Wild-type birds were A/A at the locus;"
            " carriers were A/C or C/C."
        )

        self.assertEqual(readings.allele_pairs, frozenset())
        self.assertEqual(read("rs123 G>A").allele_pairs, frozenset({("G", "A")}))

    def test_a_genomic_change_gives_its_contig_position_and_alleles(self):
        readings = read("the variant NC_000004.12:g.54108297G>T was absent")

        self.assertIn("NC_000004.12", readings.contigs)
        self.assertIn(54108297, readings.positions)
        self.assertIn(("G", "T"), readings.allele_pairs)
        self.assertIn("NC_000004.12:g.54108297G>T", readings.changes)


class WrittenNumberTests(unittest.TestCase):
    """A number is read from its written text with a form tag (#404 d3): code never invents
    digits, and two written texts never become one value by their digits alone."""

    def test_each_written_number_is_read_with_its_form(self):
        for written, value, form in (
            ("86770373", 86770373, "as_written"),
            ("86,770,373", 86770373, "as_written"),
            ("86\u2009770\u2009373", 86770373, "as_written"),
            # c2's Results paragraph (PMC6903725): the stray space is in the paper.
            ("86,770, 373", 86770373, "damaged_thousands_group"),
            ("3.01", Decimal("3.01"), "as_written"),
            ("1.2e-8", Decimal("1.2e-8"), "scientific"),
        ):
            with self.subTest(written=written):
                number = written_number(written)
                self.assertEqual((number.value, number.form), (value, form))

    def test_3_01_and_30_1_stay_two_values(self):
        self.assertNotEqual(written_number("3.01").value, written_number("30.1").value)
        self.assertNotEqual(written_number("3.01").value, 301)

    def test_a_group_that_is_no_thousands_group_is_ambiguous(self):
        # "3,01" is a decimal comma or a damaged group; the paper does not say which.
        number = written_number("3,01")

        self.assertEqual((number.value, number.form), (None, "ambiguous"))

    def test_a_leading_zero_or_a_thin_space_loose_group_is_ambiguous(self):
        for written in ("012", "00,123", "3\u202f01"):
            with self.subTest(written=written):
                self.assertEqual(written_number(written).form, "ambiguous")

    def test_text_that_is_no_number_is_not_read(self):
        # Digits other than 0-9 are no digits a position is written in.
        for written in ("12a", "chr 2", "", "1,234,", "\u0661\u0662\u0663"):
            with self.subTest(written=written):
                self.assertIsNone(written_number(written))


if __name__ == "__main__":
    unittest.main()
