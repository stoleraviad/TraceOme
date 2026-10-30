"""The gene annotation G15 reads (#376): the Run's assembly's own GFF3, every organism's
alike (#469, owner Q4 a), turned into transcripts with CDS segments on the Run's assembly,
each with its translation table and its tags copied (ADR 0049). Inputs are small
hand-written documents in the source's shape, as the development GETs under
docs/development/reviews/issue376-gene-change-targets/devgets/ show it."""

from __future__ import annotations

import io
import unittest
import zipfile

from bio_neb.target_request.annotation import gene_from_gff
from bio_neb.target_request.gene_change import Transcript

_GFF = "\n".join(
    [
        "##gff-version 3",
        "NC_A.1\tRefSeq\tregion\t1\t9000\t.\t+\t.\tID=NC_A.1:1..9000",
        # A plus-strand gene with one mRNA, two CDS segments.
        "NC_A.1\tRefSeq\tgene\t100\t900\t.\t+\t.\tID=gene-G1;Name=PBP2;gene=PBP2;"
        "gene_synonym=HEK1;gene_biotype=protein_coding",
        "NC_A.1\tRefSeq\tmRNA\t100\t900\t.\t+\t.\tID=rna-NM_1.1;Parent=gene-G1;gene=PBP2;"
        "tag=MANE Select,RefSeq Select;transcript_id=NM_1.1",
        "NC_A.1\tRefSeq\tCDS\t110\t200\t.\t+\t0\tID=cds-NP_1.1;Parent=rna-NM_1.1;"
        "gene=PBP2;protein_id=NP_1.1",
        "NC_A.1\tRefSeq\tCDS\t300\t400\t.\t+\t2\tID=cds-NP_1.1;Parent=rna-NM_1.1;"
        "gene=PBP2;protein_id=NP_1.1",
        # A minus-strand gene with two models; X2 starts mid-codon (5' partial).
        "NC_A.1\tGnomon\tgene\t2000\t5000\t.\t-\t.\tID=gene-R5;Name=RAPGEF5;gene=RAPGEF5",
        "NC_A.1\tGnomon\tmRNA\t2000\t5000\t.\t-\t.\tID=rna-XM_1.1;Parent=gene-R5;"
        "gene=RAPGEF5;transcript_id=XM_1.1",
        "NC_A.1\tGnomon\tCDS\t2100\t2200\t.\t-\t0\tID=cds-XP_1.1;Parent=rna-XM_1.1;"
        "gene=RAPGEF5;protein_id=XP_1.1",
        "NC_A.1\tGnomon\tCDS\t4000\t4100\t.\t-\t0\tID=cds-XP_1.1;Parent=rna-XM_1.1;"
        "gene=RAPGEF5;protein_id=XP_1.1",
        "NC_A.1\tGnomon\tmRNA\t2000\t4500\t.\t-\t.\tID=rna-XM_2.1;Parent=gene-R5;"
        "gene=RAPGEF5;transcript_id=XM_2.1",
        "NC_A.1\tGnomon\tCDS\t2100\t4400\t.\t-\t1\tID=cds-XP_2.1;Parent=rna-XM_2.1;"
        "gene=RAPGEF5;protein_id=XP_2.1",
        # A non-coding transcript of the same gene: no CDS, not a transcript here.
        "NC_A.1\tGnomon\tlnc_RNA\t2000\t2500\t.\t-\t.\tID=rna-XR_1.1;Parent=gene-R5;"
        "gene=RAPGEF5;transcript_id=XR_1.1",
        # A mitochondrial gene whose CDS writes its translation table (NCBI table 2).
        "NC_M.1\tRefSeq\tgene\t10\t60\t.\t+\t.\tID=gene-M1;Name=COX1;gene=COX1",
        "NC_M.1\tRefSeq\tCDS\t10\t60\t.\t+\t0\tID=cds-YP_1.1;Parent=gene-M1;gene=COX1;"
        "protein_id=YP_1.1;transl_table=2",
        # A gene whose two CDS rows write different tables.
        "NC_M.1\tRefSeq\tgene\t100\t300\t.\t+\t.\tID=gene-M2;Name=ND9;gene=ND9",
        "NC_M.1\tRefSeq\tCDS\t100\t150\t.\t+\t0\tID=cds-YP_2.1;Parent=gene-M2;gene=ND9;"
        "protein_id=YP_2.1;transl_table=2",
        "NC_M.1\tRefSeq\tCDS\t200\t300\t.\t+\t0\tID=cds-YP_2.1;Parent=gene-M2;gene=ND9;"
        "protein_id=YP_2.1;transl_table=5",
        # Two genes sharing one symbol.
        "NC_A.1\tRefSeq\tgene\t6000\t6300\t.\t+\t.\tID=gene-D1;Name=DUP;gene=DUP",
        "NC_A.2\tRefSeq\tgene\t6000\t6300\t.\t+\t.\tID=gene-D2;Name=DUP;gene=DUP",
        "",
    ]
)


def _zip(gff=_GFF, member="ncbi_dataset/data/GCF_000000001.1/genomic.gff"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("ncbi_dataset/data/assembly_data_report.jsonl", "{}\n")
        archive.writestr(member, gff)
    return buffer.getvalue()


class GffTests(unittest.TestCase):
    def test_a_plus_strand_gene_gives_its_mrna_with_cds_segments_in_order(self):
        gene = gene_from_gff(_zip(), "PBP2")

        self.assertEqual(
            gene.transcripts,
            (
                Transcript(
                    accession="NM_1.1",
                    protein="NP_1.1",
                    gene="PBP2",
                    contig="NC_A.1",
                    strand="+",
                    cds=((110, 200), (300, 400)),
                    # No transl_table written: the INSDC rule's table 1.
                    genetic_code=1,
                    genetic_code_origin="default",
                    tags=("MANE Select", "RefSeq Select"),
                ),
            ),
        )
        self.assertEqual(gene.reason, None)

    def test_a_cds_translation_table_is_copied(self):
        """The genetic code is the source's (#469, owner RC2 2): the CDS's own
        transl_table, marked copied, never a code default for a mitochondrial gene."""
        [transcript] = gene_from_gff(_zip(), "COX1").transcripts

        self.assertEqual(
            (transcript.accession, transcript.genetic_code),
            ("YP_1.1", 2),
        )
        self.assertEqual(transcript.genetic_code_origin, "copied")
        self.assertEqual(transcript.tags, ())

    def test_cds_rows_naming_two_tables_are_unreadable(self):
        """An annotation whose one CDS names two translation tables contradicts itself:
        the download does not read as one (ADR 0049 copies, it never picks one)."""
        with self.assertRaisesRegex(ValueError, "translation tables"):
            gene_from_gff(_zip(), "ND9")

    def test_a_minus_strand_gene_lists_segments_five_prime_first_and_keeps_partials(
        self,
    ):
        """Every transcript is copied (#469, owner Q5 a; review r1 SP3): a 5'-partial
        CDS keeps its phase, its tags and genetic code, not dropped as prose."""
        gene = gene_from_gff(_zip(), "RAPGEF5")

        self.assertEqual(
            [(t.accession, t.strand, t.cds, t.cds_phase) for t in gene.transcripts],
            [
                ("XM_1.1", "-", ((4000, 4100), (2100, 2200)), 0),
                ("XM_2.1", "-", ((2100, 4400),), 1),
            ],
        )
        self.assertEqual(
            {(t.genetic_code, t.genetic_code_origin) for t in gene.transcripts},
            {(1, "default")},
        )

    def test_a_synonym_finds_the_gene_under_its_own_symbol(self):
        gene = gene_from_gff(_zip(), "hek1")

        self.assertEqual([t.gene for t in gene.transcripts], ["PBP2"])

    def test_an_unknown_symbol_is_its_reason(self):
        gene = gene_from_gff(_zip(), "NOPE")

        self.assertEqual(gene.transcripts, ())
        self.assertEqual(gene.outcome, "not_found")
        self.assertEqual(gene.reason, "the assembly's annotation names no gene NOPE")

    def test_a_symbol_naming_two_genes_is_ambiguous(self):
        gene = gene_from_gff(_zip(), "DUP")

        self.assertEqual(gene.outcome, "ambiguous")
        self.assertEqual(
            gene.reason,
            "the assembly's annotation names 2 genes DUP: gene-D1 (NC_A.1:6000-6300), "
            "gene-D2 (NC_A.2:6000-6300)",
        )

    def test_a_download_without_a_gff_is_malformed(self):
        with self.assertRaisesRegex(ValueError, "no genomic.gff"):
            gene_from_gff(_zip(member="ncbi_dataset/data/README.md"), "PBP2")
