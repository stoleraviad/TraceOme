"""A gene plus protein or coding change as candidate position targets (#376, G15).

Lane Rf's step from a claim such as s3's ``PBP2 (Y127X)`` or s17's
``RAPGEF5 (c.2624C>A p.Ser875*)`` to every genomic change on the Run's assembly that gives
it, each a ``bio-neb.investigation-request/2`` position target. Expected values come from the
genetic code and hand-built transcripts, not from the code under test.
"""

from __future__ import annotations

import unittest

from bio_neb.target_request.gene_change import (
    CodingChange,
    ProteinChange,
    Transcript,
    decide,
    read_transcript,
)


def _plus(accession="NM_TEST.1", cds=((101, 130),), contig="NC_TEST.1", **more):
    return Transcript(
        accession=accession,
        protein="NP_TEST.1",
        gene="GENE",
        contig=contig,
        strand="+",
        cds=cds,
        **{"genetic_code": 1, "genetic_code_origin": "default", **more},
    )


class _Genome:
    """Plus-strand bases by 1-based position, scripted per test; every read is kept."""

    def __init__(self, bases):
        self.bases = dict(bases)
        self.reads = []

    def __call__(self, contig, start, length):
        self.reads.append((contig, start, length))
        return "".join(
            self.bases[position] for position in range(start, start + length)
        )


def _codon_at(start, codon):
    return {start + offset: base for offset, base in enumerate(codon)}


class TypedChangeTests(unittest.TestCase):
    """The model types a change from its words, and code reads no change text (#473
    owner Q2 a, ADR 0049): these replace ``parse_change``'s grammar tests (Q10 a)."""

    def test_a_typed_protein_and_coding_change_are_read_as_the_model_typed_them(self):
        from bio_neb.investigation.investigation_request import (
            CodingChangeDraft,
            ProteinChangeDraft,
            StatedChange,
        )
        from bio_neb.target_request.ncbi_sources import typed_change
        from bio_neb.target_request.records import GeneChangeClaim

        def claim(**typed):
            return GeneChangeClaim(
                label="s17",
                change=StatedChange(
                    gene="RAPGEF5", change="c.2624C>A p.Ser875*", **typed
                ),
                request_text="RAPGEF5 (c.2624C>A p.Ser875*)",
            )

        protein = ProteinChangeDraft(ref="S", codon=875, alt="*")
        coding = CodingChangeDraft(cds_position=2624, ref="C", alt="A")
        self.assertEqual(
            typed_change(claim(protein=protein, coding=coding)),
            (CodingChange(2624, "C", "A"), ProteinChange("S", 875, "*")),
        )
        self.assertEqual(
            typed_change(claim(protein=protein)), ProteinChange("S", 875, "*")
        )
        self.assertIsNone(typed_change(claim()))

    def test_a_coding_and_protein_change_naming_two_codons_is_refused(self):
        """Typed values that name two codons are the model's argument to repair: c.4 is
        codon 2 (#473 Q2 a)."""
        from pydantic import ValidationError

        from bio_neb.investigation.investigation_request import StatedChange

        with self.assertRaisesRegex(ValidationError, "c.4 is codon 2"):
            StatedChange(
                gene="PBP2",
                change="c.4A>G p.Lys3Glu",
                coding={"cds_position": 4, "ref": "A", "alt": "G"},
                protein={"ref": "K", "codon": 3, "alt": "E"},
            )

    def test_a_protein_change_to_the_same_residue_is_refused(self):
        """A stated change names another residue (#473 Q2 a)."""
        from pydantic import ValidationError

        from bio_neb.investigation.investigation_request import ProteinChangeDraft

        with self.assertRaisesRegex(ValidationError, "another residue"):
            ProteinChangeDraft(ref="K", codon=3, alt="K")

    def test_a_typed_change_stands_in_the_quote_that_writes_it(self):
        from bio_neb.investigation.investigation_request import StatedChange

        braf = StatedChange(
            gene="BRAF",
            change="V600E",
            transcript="NM_004333.6",
            protein={"ref": "V", "codon": 600, "alt": "E"},
        )
        quote = "Does my sample carry BRAF V600E (NM_004333.6)?"
        self.assertTrue(braf.in_quote(quote))
        # Membership only: every typed value is written, on whole words.
        self.assertFalse(
            braf.in_quote("Does my sample carry BRAFX V600E (NM_004333.6)?")
        )
        self.assertFalse(braf.in_quote("Does my sample carry BRAF V600E?"))
        self.assertFalse(
            braf.model_copy(
                update={"protein": braf.protein.model_copy(update={"codon": 601})}
            ).in_quote(quote)
        )


class ProteinStopOnPlusStrandTests(unittest.TestCase):
    def test_tac_to_stop_gives_both_single_base_snvs_and_the_two_base_tga(self):
        # Codon 2 of a CDS starting at 101 is 104-106; TAC encodes Y.
        genome = _Genome(_codon_at(104, "TAC"))

        reading = read_transcript(_plus(), ProteinChange("Y", 2, "*"), genome)

        self.assertEqual(reading.reference_codon, "TAC")
        self.assertEqual(reading.reference_residue, "Y")
        self.assertTrue(reading.matches)
        self.assertEqual(
            [
                (c.contig, c.position, c.ref, c.alt, c.bases_changed, c.alt_codon)
                for c in reading.candidates
            ],
            [
                ("NC_TEST.1", 106, "C", "A", 1, "TAA"),
                ("NC_TEST.1", 106, "C", "G", 1, "TAG"),
                ("NC_TEST.1", 105, "AC", "GA", 2, "TGA"),
            ],
        )
        self.assertEqual(genome.reads, [("NC_TEST.1", 104, 3)])


def _minus(accession="XM_TEST.1", cds=((201, 230),), contig="NC_TEST.1", **more):
    return Transcript(
        accession=accession,
        protein="XP_TEST.1",
        gene="GENE",
        contig=contig,
        strand="-",
        cds=cds,
        **{"genetic_code": 1, "genetic_code_origin": "default", **more},
    )


class MinusStrandTests(unittest.TestCase):
    def test_a_coding_change_on_the_minus_strand_is_its_complement_on_the_plus_strand(
        self,
    ):
        # CDS 201-230 on the minus strand: c.1 is 230, so codon 2 (c.4-6) is 227, 226,
        # 225. Coding TCA (Ser) reads TGA on the plus strand from 225 to 227.
        genome = _Genome(_codon_at(225, "TGA"))

        reading = read_transcript(
            _minus(),
            (CodingChange(5, "C", "A"), ProteinChange("S", 2, "*")),
            genome,
        )

        self.assertEqual(
            (reading.reference_codon, reading.reference_residue), ("TCA", "S")
        )
        self.assertEqual(
            [(c.position, c.ref, c.alt, c.alt_codon) for c in reading.candidates],
            [(226, "G", "T", "TAA")],
        )

    def test_a_coding_name_whose_protein_name_disagrees_matches_nothing(self):
        genome = _Genome(_codon_at(225, "TGA"))

        reading = read_transcript(
            _minus(),
            (CodingChange(5, "C", "A"), ProteinChange("S", 2, "W")),
            genome,
        )

        self.assertFalse(reading.matches)
        self.assertEqual(reading.candidates, ())
        self.assertEqual(
            reading.reason,
            "on XM_TEST.1, c.5C>A gives *, not the stated W",
        )


class CodonAcrossAnIntronTests(unittest.TestCase):
    def test_a_split_codon_is_read_in_two_runs_and_a_change_across_the_intron_is_unplaced(
        self,
    ):
        # CDS 101-105 then 201-225: codon 2 is 104, 105 | 201, so TAC is read as TA at
        # 104-105 and C at 201.
        genome = _Genome({104: "T", 105: "A", 201: "C"})
        transcript = _plus(cds=((101, 105), (201, 225)))

        reading = read_transcript(transcript, ProteinChange("Y", 2, "*"), genome)

        self.assertEqual(reading.reference_codon, "TAC")
        self.assertEqual(genome.reads, [("NC_TEST.1", 104, 2), ("NC_TEST.1", 201, 1)])
        self.assertEqual(
            [(c.position, c.ref, c.alt) for c in reading.candidates],
            [(201, "C", "A"), (201, "C", "G")],
        )
        # TGA changes 105 (A>G) and 201 (C>A): an intron lies between them.
        self.assertEqual(reading.unplaced, ("TGA",))


class ReferenceMismatchTests(unittest.TestCase):
    def test_a_transcript_whose_codon_is_another_residue_matches_nothing(self):
        genome = _Genome(_codon_at(104, "TGG"))

        reading = read_transcript(_plus(), ProteinChange("Y", 2, "*"), genome)

        self.assertFalse(reading.matches)
        self.assertEqual(reading.candidates, ())
        self.assertEqual(
            reading.reason, "NM_TEST.1 codes W at codon 2; the claim states Y"
        )

    def test_a_codon_past_the_cds_is_not_read(self):
        genome = _Genome({})

        reading = read_transcript(_plus(), ProteinChange("Y", 11, "*"), genome)

        self.assertFalse(reading.matches)
        self.assertEqual(reading.reason, "NM_TEST.1's CDS ends before codon 11")
        self.assertEqual(genome.reads, [])


class DecideTests(unittest.TestCase):
    """One claim over the gene's transcripts: one answer, never a dropped claim."""

    def test_transcripts_that_agree_on_the_codon_give_one_candidate_set_naming_both(
        self,
    ):
        genome = _Genome(_codon_at(104, "TAC"))
        x1 = _plus("XM_A.1")
        x2 = _plus("XM_B.1")

        decision = decide(ProteinChange("Y", 2, "*"), (x1, x2), genome)

        self.assertEqual(decision.outcome, "candidates")
        self.assertEqual(
            [
                (c.position, c.alt, transcripts)
                for c, transcripts in decision.candidates
            ],
            [
                (106, "A", ("XM_A.1", "XM_B.1")),
                (106, "G", ("XM_A.1", "XM_B.1")),
                (105, "GA", ("XM_A.1", "XM_B.1")),
            ],
        )

    def test_only_the_transcripts_holding_the_stated_residue_give_candidates(self):
        # X2's codon 2 starts 3 bases later (TGG, W): only X1 holds Y there.
        genome = _Genome({**_codon_at(104, "TAC"), **_codon_at(107, "TGG")})
        x1 = _plus("XM_A.1")
        x2 = _plus("XM_B.1", cds=((104, 130),))

        decision = decide(ProteinChange("Y", 2, "*"), (x1, x2), genome)

        self.assertEqual(decision.outcome, "candidates")
        self.assertEqual({t for _, ts in decision.candidates for t in ts}, {"XM_A.1"})
        self.assertEqual(
            [(r.transcript.accession, r.matches) for r in decision.readings],
            [("XM_A.1", True), ("XM_B.1", False)],
        )

    def test_transcripts_placing_the_stated_residue_at_different_codons_are_ambiguous(
        self,
    ):
        genome = _Genome({**_codon_at(104, "TAC"), **_codon_at(107, "TAT")})
        x1 = _plus("XM_A.1")
        x2 = _plus("XM_B.1", cds=((104, 130),))

        decision = decide(ProteinChange("Y", 2, "*"), (x1, x2), genome)

        self.assertEqual(decision.outcome, "ambiguous")
        self.assertEqual(
            decision.reason,
            "GENE's transcripts place Y2 at different codons: XM_A.1 at "
            "NC_TEST.1:104-106, XM_B.1 at NC_TEST.1:107-109",
        )
        self.assertEqual(
            sorted({(c.position, ts) for c, ts in decision.candidates}),
            [
                (105, ("XM_A.1",)),
                (106, ("XM_A.1",)),
                (108, ("XM_B.1",)),
                (109, ("XM_B.1",)),
            ],
        )

    def test_no_transcript_holding_the_stated_residue_is_not_found_with_each_reason(
        self,
    ):
        genome = _Genome(_codon_at(104, "TGG"))

        decision = decide(ProteinChange("Y", 2, "*"), (_plus(),), genome)

        self.assertEqual(decision.outcome, "not_found")
        self.assertEqual(decision.candidates, ())
        self.assertEqual(
            decision.reason, "NM_TEST.1 codes W at codon 2; the claim states Y"
        )

    def test_every_transcript_is_read_whatever_its_tags(self):
        """No transcript preference in code (#469, owner Q5 a, ADR 0049): a MANE Select
        tag is copied, not a choice; every transcript is read, and the two codons they
        place the residue at make the claim ambiguous for the model's pick (#473)."""
        genome = _Genome({**_codon_at(104, "TAC"), **_codon_at(107, "TAC")})
        mane = _plus("NM_A.1", tags=("MANE Select",))
        other = _plus("NM_B.1", cds=((104, 130),))

        decision = decide(ProteinChange("Y", 2, "*"), (mane, other), genome)

        self.assertEqual(
            [r.transcript.accession for r in decision.readings], ["NM_A.1", "NM_B.1"]
        )
        self.assertEqual(decision.readings[0].transcript.tags, ("MANE Select",))
        self.assertEqual(decision.outcome, "ambiguous")

    def test_a_transcript_is_translated_with_its_own_genetic_code(self):
        """The copied genetic code translates (#469, owner RC2 2): under NCBI table 2
        (vertebrate mitochondrial) TGA is Trp, and AGA and AGG are stops beside TAA
        and TAG; under table 1 the same TGA is a stop."""
        genome = _Genome(_codon_at(104, "TGA"))
        mitochondrial = _plus(genetic_code=2, genetic_code_origin="copied")

        reading = read_transcript(mitochondrial, ProteinChange("W", 2, "*"), genome)

        self.assertTrue(reading.matches, reading.reason)
        self.assertEqual(reading.reference_residue, "W")
        self.assertEqual(
            {candidate.alt_codon for candidate in reading.candidates},
            {"TAA", "TAG", "AGA", "AGG"},
        )
        standard = read_transcript(_plus(), ProteinChange("W", 2, "*"), genome)
        self.assertEqual(standard.reference_residue, "*")
        self.assertFalse(standard.matches)

    def test_a_cds_starting_mid_codon_is_read_as_its_reason(self):
        """A 5'-partial CDS is copied with its phase (#469 review r1 SP3); its codons
        cannot be counted from its start, so its reading names why and reads no base."""
        genome = _Genome(_codon_at(104, "TAC"))

        reading = read_transcript(
            _plus(cds_phase=1), ProteinChange("Y", 2, "*"), genome
        )

        self.assertFalse(reading.matches)
        self.assertEqual(
            reading.reason,
            "NM_TEST.1's CDS starts mid-codon (phase 1): its codons are not counted "
            "from it",
        )
        self.assertEqual(genome.reads, [])

    def test_a_table_ncbi_does_not_define_is_its_reason(self):
        """A copied table id NCBI's tables lack is read as no codon, with its reason
        (ADR 0049: the copied value is kept, never replaced by a default)."""
        genome = _Genome(_codon_at(104, "TGA"))

        reading = read_transcript(
            _plus(genetic_code=7, genetic_code_origin="copied"),
            ProteinChange("W", 2, "*"),
            genome,
        )

        self.assertFalse(reading.matches)
        self.assertEqual(
            reading.reason,
            "NM_TEST.1's annotation names translation table 7, which NCBI's tables do "
            "not define",
        )
        self.assertEqual(genome.reads, [])

    def test_a_transcript_the_paper_names_with_its_version_is_the_only_one_read(self):
        genome = _Genome(_codon_at(107, "TAT"))
        x1 = _plus("XM_A.1")
        x2 = _plus("XM_B.2", cds=((104, 130),))

        decision = decide(
            ProteinChange("Y", 2, "*"), (x1, x2), genome, transcript="XM_B.2"
        )

        self.assertEqual(
            [r.transcript.accession for r in decision.readings], ["XM_B.2"]
        )
        self.assertEqual(decision.outcome, "candidates")

    def test_a_transcript_named_without_a_version_reads_the_annotations_version(self):
        genome = _Genome(_codon_at(107, "TAT"))
        x2 = _plus("XM_B.2", cds=((104, 130),))

        decision = decide(
            ProteinChange("Y", 2, "*"), (_plus("XM_A.1"), x2), genome, transcript="XM_B"
        )

        self.assertEqual(
            [r.transcript.accession for r in decision.readings], ["XM_B.2"]
        )

    def test_a_named_version_the_annotation_does_not_hold_is_not_found_and_not_read(
        self,
    ):
        # Another version of a transcript can have other exons: XM_B.2's coordinates are
        # not XM_B.1's.
        genome = _Genome({})
        x2 = _plus("XM_B.2", cds=((104, 130),))

        decision = decide(
            ProteinChange("Y", 2, "*"),
            (_plus("XM_A.1"), x2),
            genome,
            transcript="XM_B.1",
        )

        self.assertEqual(decision.outcome, "not_found")
        self.assertEqual(
            decision.reason,
            "the claim names XM_B.1; GENE's annotation on the Run's assembly holds "
            "XM_B.2, another version of it",
        )
        self.assertEqual((decision.readings, genome.reads), ((), []))

    def test_a_named_transcript_the_annotation_lacks_is_not_found(self):
        decision = decide(
            ProteinChange("Y", 2, "*"), (_plus(),), _Genome({}), transcript="NM_GONE.3"
        )

        self.assertEqual(decision.outcome, "not_found")
        self.assertEqual(
            decision.reason,
            "the claim names NM_GONE.3; GENE has no such transcript on the Run's "
            "assembly (NM_TEST.1)",
        )


class CandidatePropertyTests(unittest.TestCase):
    """The same claim-to-candidates step over many random genes, strands, codons and
    residue changes gives the same kind of answer (#346 D5). The check is independent of
    the code under test: each candidate is applied to the genome, the whole CDS is spliced
    and translated with Python's own table here, and the residue at the stated codon must
    be the stated one; every single-base change giving it must be among the candidates."""

    _TABLE = {
        a + b + c: "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"[
            16 * "TCAG".index(a) + 4 * "TCAG".index(b) + "TCAG".index(c)
        ]
        for a in "TCAG"
        for b in "TCAG"
        for c in "TCAG"
    }

    def _protein(self, genome, transcript):
        complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
        coding = ""
        for start, end in transcript.cds:
            piece = "".join(genome[p] for p in range(start, end + 1))
            if transcript.strand == "-":
                piece = "".join(complement[b] for b in reversed(piece))
            coding += piece
        return "".join(
            self._TABLE[coding[i : i + 3]] for i in range(0, len(coding) - 2, 3)
        )

    def test_every_candidate_gives_the_stated_residue_and_no_snv_is_missed(self):
        import random

        rng = random.Random(376)
        checked = 0
        for _ in range(200):
            genome = {p: rng.choice("ACGT") for p in range(1, 401)}
            # Two or three CDS segments with introns between, lengths summing to 3n.
            cuts = sorted(rng.sample(range(20, 380), 4))
            segments = [(cuts[0], cuts[1]), (cuts[2], cuts[3])]
            length = sum(end - start + 1 for start, end in segments)
            last = segments[-1]
            segments[-1] = (last[0], last[1] - length % 3)
            strand = rng.choice("+-")
            ordered = tuple(segments if strand == "+" else reversed(segments))
            transcript = Transcript(
                accession="T.1",
                protein=None,
                gene="G",
                contig="C.1",
                strand=strand,
                cds=ordered,
                genetic_code=1,
                genetic_code_origin="default",
            )
            protein = self._protein(genome, transcript)
            number = rng.randrange(1, len(protein) + 1)
            ref = protein[number - 1]
            if ref == "*":
                continue
            alt = rng.choice([r for r in "ACDEFGHIKLMNPQRSTVWY*" if r != ref])

            reading = read_transcript(
                transcript,
                ProteinChange(ref, number, alt),
                lambda contig, start, size, genome=genome: "".join(
                    genome[p] for p in range(start, start + size)
                ),
            )

            self.assertTrue(reading.matches, reading.reason)
            for candidate in reading.candidates:
                changed = dict(genome)
                for offset, base in enumerate(candidate.alt):
                    changed[candidate.position + offset] = base
                self.assertEqual(
                    self._protein(changed, transcript)[number - 1], alt, candidate
                )
            snvs = set()
            coding = [p for start, end in segments for p in range(start, end + 1)]
            for position in coding:
                for base in "ACGT":
                    if base == genome[position]:
                        continue
                    changed = dict(genome)
                    changed[position] = base
                    if self._protein(changed, transcript)[number - 1] == alt:
                        snvs.add((position, genome[position], base))
            self.assertLessEqual(
                snvs,
                {
                    (c.position, c.ref, c.alt)
                    for c in reading.candidates
                    if c.bases_changed == 1
                },
            )
            checked += 1
        self.assertGreater(checked, 150)


class SelectionAndPlacementTests(unittest.TestCase):
    def test_a_change_every_codon_of_which_spans_an_intron_is_not_checked(self):
        # CDS 100-103 | 200-204: codon 2 is 103 | 200, 201, AAA (Lys). Phe is TTT or TTC:
        # both change 103 and 200 together, which no one VCF record holds.
        genome = _Genome({103: "A", 200: "A", 201: "A"})
        transcript = _plus(cds=((100, 103), (200, 204)))

        decision = decide(ProteinChange("K", 2, "F"), (transcript,), genome)

        self.assertEqual(decision.outcome, "not_checked")
        self.assertEqual(decision.candidates, ())
        self.assertEqual(
            decision.reason,
            "every change giving F at codon 2 changes bases an intron separates "
            "(NM_TEST.1: TTC, TTT); no one VCF record holds it",
        )
