"""Paper findings as proposed targets: the typed ``PaperTargets`` record (#350, #395).

The answers are #349's retained staged-spike answers, each verified against its replayed
paper (``docs/development/reviews/issue349-paper-spikes/runs/<run>/answer.json`` at
``a60418a3``, copied to ``tests/fixtures/paper_retrieval/answers/``). Expected targets are read off those answers by hand: the line that states each
rsID, the exact span of each text-only change.

The tests that sent these targets through main's request file, lookup and intent check were
retired with P's ``request.json`` bridge (owner, map #307 Q8); their guarantees return in
#386's chain test, where P's record meets lane Rf's ``place``.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from bio_neb.investigation.investigation_request import (
    InvestigationRequestItem,
    StatedAssembly,
    StatedPosition,
)
from bio_neb.paper_reading.answer import judge
from bio_neb.paper_reading.contracts import FindingPointer, KeptFinding
from bio_neb.paper_reading.targets import paper_targets
from bio_neb.target_request.paper_targets import (
    Finding,
    PaperIdentity,
    PaperTargets,
    SourceLine,
    TargetChange,
)
from tests.paper_targets_support import (
    AKAP9_ROW,
    H1_PAPER,
    KCNH2_ROW,
    S3_LOCATED,
    S3_METHODS,
    S3_READING,
    S3_SENTENCE,
    TTN_ROW,
    located,
    paper_of,
    read_as,
    recorded_answer,
    s3_answer,
    s17_answer,
    stated,
    synthetic_answer,
    synthetic_reference,
)
from tools.paper_retrieval.__main__ import main


class PaperTargetTests(unittest.TestCase):
    def test_h1_rsid_findings_become_targets_the_paper_asserts_on_their_table_rows(
        self,
    ):
        # A recorded h1 answer: an rsID target quotes the row that states its rsID.
        targets = paper_targets(recorded_answer("h1-run1"))

        self.assertEqual(targets.asserted_by, H1_PAPER)
        self.assertEqual(
            [
                (t.request.rsid, t.request.request_text, t.kind, t.not_checked)
                for t in targets.targets
            ],
            [
                ("rs199472944", KCNH2_ROW, "snv", None),
                ("rs779447911", AKAP9_ROW, "indel", None),
                ("rs773002407", TTN_ROW, "snv", None),
            ],
        )
        # The quote is the paper's retrieved row, at its place in the main text.
        self.assertEqual(
            [
                (t.quote.document_id, t.quote.start, t.quote.end)
                for t in targets.targets
            ],
            [("main", 10615, 10717), ("main", 10718, 10841), ("main", 10842, 10940)],
        )
        self.assertEqual(
            [[f.gene for f in t.findings] for t in targets.targets],
            [["KCNH2"], ["AKAP9"], ["TTN"]],
        )

    def test_an_rsid_the_model_states_makes_its_target_on_the_row_that_writes_it(self):
        text, paper = paper_of(KCNH2_ROW, TTN_ROW)
        # The two rows are one table: KCNH2's row is t1:r1, TTN's t1:r2.
        answer = read_as(
            text,
            paper,
            [
                stated("contig", "chr7", "t1:r1"),
                stated("position", "150951552", "t1:r1", number=150951552),
                stated("rsid", "rs199472944", "t1:r1"),
                stated("change", "c.1841C>T", "t1:r1"),
                stated("change", "p. (Ala614Val)", "t1:r1"),
                stated("gene", "KCNH2", "t1:r1"),
            ],
            [stated("rsid", "rs773002407", "t1:r2"), stated("gene", "TTN", "t1:r2")],
        )

        targets = paper_targets(answer)

        # The row stating the rsID is the request, whatever words the model quoted.
        self.assertEqual(
            [
                (t.request.rsid, t.request.request_text, t.kind, t.request.position)
                for t in targets.targets
            ],
            [
                (
                    "rs199472944",
                    KCNH2_ROW,
                    "snv",
                    StatedPosition(contig="chr7", position=150951552),
                ),
                ("rs773002407", TTN_ROW, "not_stated", None),
            ],
        )

    def test_s3_text_only_findings_become_targets_on_their_quotes_and_changes(self):
        targets = paper_targets(s3_answer())

        self.assertEqual(
            [
                (t.request.request_text, t.request.rsid, t.not_checked, t.kind)
                for t in targets.targets
            ],
            [
                # Three text-only findings in one sentence: each quote is the paper's
                # words for its own gene and change, and nothing locates it.
                ("PBP2 (Y127X)", None, "not_located", "not_stated"),
                ("SGF73 (E294X)", None, "not_located", "not_stated"),
                ("UPF3 (G6W)", None, "not_located", "not_stated"),
                # The located one has no rsID either: main checks it by its position.
                (S3_LOCATED, None, None, "snv"),
            ],
        )
        self.assertEqual(
            [t.quote.start for t in targets.targets],
            [
                S3_SENTENCE.index(w) + len("## Results\n")
                for w in ("PBP2 (Y127X)", "SGF73 (E294X)", "UPF3 (G6W)", S3_LOCATED)
            ],
        )
        # Each request carries only the typed fields its finding states; the located one
        # its position, as written, on the paper's own sacCer1, cited.
        self.assertEqual(
            [t.request for t in targets.targets],
            [
                InvestigationRequestItem(request_text="PBP2 (Y127X)"),
                InvestigationRequestItem(request_text="SGF73 (E294X)"),
                InvestigationRequestItem(request_text="UPF3 (G6W)"),
                InvestigationRequestItem(
                    request_text=S3_LOCATED,
                    position=StatedPosition(
                        contig="chr11", position=207469, ref="A", alt="C"
                    ),
                    stated_assembly=StatedAssembly(
                        name="sacCer1", cited_line=S3_METHODS
                    ),
                ),
            ],
        )

    def test_findings_of_one_change_join_into_one_target_by_main_identity(self):
        # As in s3 run 2: two reading items each gave the paper's four point mutations.
        text, paper = paper_of(S3_SENTENCE, S3_METHODS)
        answer = read_as(text, paper, *S3_READING, *S3_READING, reference="sacCer1")

        targets = paper_targets(answer)

        self.assertEqual(
            [t.request.request_text for t in targets.targets],
            ["PBP2 (Y127X)", "SGF73 (E294X)", "UPF3 (G6W)", S3_LOCATED],
        )
        # paper-targets/1 keeps its finding shape: the kept values stay in the
        # answer.
        as_target = [
            Finding.model_validate(f.model_dump(exclude={"values"}))
            for f in answer.findings
        ]
        self.assertEqual(
            [t.findings for t in targets.targets],
            [(as_target[i], as_target[i + 4]) for i in range(4)],
        )
        self.assertNotIn('"values"', targets.model_dump_json())

    def test_two_changes_of_one_gene_in_one_paragraph_stay_two_targets(self):
        text, paper = paper_of(
            "The evolved strain carries PBP2 (Y127X) and PBP2 (Q30X) in one clone."
        )
        answer = read_as(
            text,
            paper,
            [stated("gene", "PBP2", "main:p2"), stated("change", "Y127X", "main:p2")],
            [stated("gene", "PBP2", "main:p2"), stated("change", "Q30X", "main:p2")],
        )

        targets = paper_targets(answer)

        self.assertEqual(
            [t.request.request_text for t in targets.targets],
            ["PBP2 (Y127X)", "PBP2 (Q30X)"],
        )
        self.assertEqual(
            [t.change for t in targets.targets],
            [
                TargetChange(gene="PBP2", text="Y127X"),
                TargetChange(gene="PBP2", text="Q30X"),
            ],
        )

    def test_located_findings_on_one_line_stay_two_targets_by_their_quotes(self):
        for text_, second in (
            (
                "Both had GENE1 chr1:1234567 A>G and GENE2 chr1:7654321 C>T.",
                "chr1:7654321 C>T",
            ),
            (
                "Both had GENE1 chr1:1234567 A>G and GENE2 chr2:1234567 A>G.",
                "chr2:1234567 A>G",
            ),
        ):
            with self.subTest(second=second):
                text, paper = paper_of(text_)
                contig, rest = second.split(":")
                position, alleles = rest.split(" ")
                ref, alt = alleles.split(">")
                answer = read_as(
                    text,
                    paper,
                    [
                        stated("contig", "chr1", "main:p2"),
                        stated("position", "1234567", "main:p2", number=1234567),
                        stated("alleles", "A>G", "main:p2"),
                    ],
                    [
                        stated("contig", contig, "main:p2"),
                        stated("position", position, "main:p2", number=int(position)),
                        stated("alleles", f"{ref}>{alt}", "main:p2"),
                    ],
                )

                targets = paper_targets(answer)

                self.assertEqual(
                    [
                        (t.request.request_text, t.request.position.contig)
                        for t in targets.targets
                    ],
                    [("chr1:1234567 A>G", "chr1"), (second, contig)],
                )

    def test_a_locus_left_partly_unstated_is_refused_and_located_once_stated(self):
        # Standards r4 S6, through the target path: chr11's locus needs its contig.
        text, paper = paper_of(S3_SENTENCE)
        partial = FindingPointer.model_validate(
            {"values": S3_READING[3][2:], "reason": "R."}
        )

        self.assertEqual(
            judge(partial, paper).reasons, ("value_not_stated:contig=chr11",)
        )
        [target] = paper_targets(read_as(text, paper, S3_READING[3])).targets
        self.assertEqual(
            target.request.position,
            StatedPosition(contig="chr11", position=207469, ref="A", alt="C"),
        )

    def test_one_locus_spelled_4_and_chr4_keeps_its_position_through_targets(self):
        # Standards r1 S3: the join and the target use one locus rule.
        row = "4\t54108297\tG\tT\trs1"
        prose = "The rs1 variant on chr4:54108297 was found."
        text, paper = paper_of(row, prose)
        # The row is a one-row table (t1:r1); the prose is main:p3.
        answer = read_as(
            text,
            paper,
            [
                stated("contig", "4", "t1:r1"),
                stated("position", "54108297", "t1:r1", number=54108297),
                stated("rsid", "rs1", "t1:r1"),
            ],
            [
                stated("contig", "chr4", "main:p3"),
                stated("position", "54108297", "main:p3", number=54108297),
                stated("rsid", "rs1", "main:p3"),
            ],
        )

        [target] = paper_targets(answer).targets

        self.assertEqual(
            target.request.position, StatedPosition(contig="4", position=54108297)
        )

    def test_a_located_target_carries_its_position_and_the_papers_cited_version(self):
        targets = paper_targets(s17_answer())

        [target] = targets.targets
        self.assertEqual(
            (target.request.request_text, target.request.rsid, target.not_checked),
            ("RAPGEF5 (c.2624C>A p.Ser875*)", None, None),
        )
        # The row states the locus with its VCF alleles; the Methods line the version.
        self.assertEqual(
            target.request.position,
            StatedPosition(contig="4", position=54108297, ref="G", alt="T"),
        )
        self.assertEqual(
            target.request.stated_assembly,
            StatedAssembly(
                name="EquCab3.0", cited_line="Reads were mapped to EquCab3.0."
            ),
        )
        self.assertEqual(
            (targets.reference.candidate.name, targets.reference.reason),
            ("EquCab3.0", "The Methods name it."),
        )

    def test_the_record_names_its_schema_and_each_change_as_the_paper_writes_it(self):
        # Placement reads targets.json as data (it declares its own model of this
        # record, not ours), so the record names its schema, and each target gives the
        # gene and the change in the paper's own spelling for placement to resolve.
        s3 = paper_targets(s3_answer())
        s17 = paper_targets(s17_answer())
        h1 = paper_targets(recorded_answer("h1-run1"))

        self.assertEqual(s3.schema_version, "bio-neb.paper-targets/1")
        self.assertEqual(
            [t.change for t in s3.targets],
            [
                TargetChange(gene="PBP2", text="Y127X"),
                TargetChange(gene="SGF73", text="E294X"),
                TargetChange(gene="UPF3", text="G6W"),
                # Its finding states no change name.
                None,
            ],
        )
        self.assertEqual(
            [t.change for t in s17.targets],
            [TargetChange(gene="RAPGEF5", text="c.2624C>A p.Ser875*")],
        )
        # h1's rows write each gene with its coding change and transcript (the
        # recorded protein names are a normal form the rows do not write).
        self.assertEqual(
            [t.change for t in h1.targets],
            [
                TargetChange(gene="KCNH2", text="c.1841C>T", transcript="NM_000238.4"),
                TargetChange(
                    gene="AKAP9", text="c.4960_4961del", transcript="NM_005751.5"
                ),
                # "NM_173,648.4" is written with a comma: no transcript id.
                TargetChange(gene="TTN", text="c.103957C>T"),
            ],
        )
        record = json.loads(s3.model_dump_json())
        self.assertEqual(record["schema_version"], "bio-neb.paper-targets/1")
        self.assertEqual(
            record["targets"][0]["change"],
            {"gene": "PBP2", "text": "Y127X", "transcript": None},
        )

    def test_a_change_names_the_one_transcript_its_line_states(self):
        def change(line: str, words: str, name: str) -> TargetChange | None:
            text, paper = paper_of(line)
            answer = read_as(
                text,
                paper,
                [stated("gene", "BRCA1", "main:p2"), stated("change", name, "main:p2")],
            )
            self.assertEqual(paper_targets(answer).targets[0].quote.text, words)
            [target] = paper_targets(answer).targets
            return target.change

        self.assertEqual(
            change(
                "We found BRCA1 (c.68_69delAG) on NM_007294.4 in the proband.",
                "BRCA1 (c.68_69delAG)",
                "c.68_69delAG",
            ),
            TargetChange(gene="BRCA1", text="c.68_69delAG", transcript="NM_007294.4"),
        )
        self.assertEqual(
            change(
                "BRCA1 (p.Glu23fs) on NM_007294.4 or NM_007300.4.",
                "BRCA1 (p.Glu23fs)",
                "p.Glu23fs",
            ),
            TargetChange(gene="BRCA1", text="p.Glu23fs"),
        )
        # Only NM_, XM_ and ENST ids count: a non-coding NR_ or XR_ id is no
        # transcript of a change, and does not hide the one coding id the line names.
        self.assertEqual(
            change(
                "BRCA1 (c.68_69delAG) on NR_123456.1.",
                "BRCA1 (c.68_69delAG)",
                "c.68_69delAG",
            ),
            TargetChange(gene="BRCA1", text="c.68_69delAG"),
        )
        self.assertEqual(
            change(
                "BRCA1 (c.68_69delAG) on NM_007294.4; a control used XR_123456.1.",
                "BRCA1 (c.68_69delAG)",
                "c.68_69delAG",
            ),
            TargetChange(gene="BRCA1", text="c.68_69delAG", transcript="NM_007294.4"),
        )
        self.assertEqual(
            change(
                "BRCA1 (c.68_69delAG) on ENST00000357654.9.",
                "BRCA1 (c.68_69delAG)",
                "c.68_69delAG",
            ),
            TargetChange(
                gene="BRCA1", text="c.68_69delAG", transcript="ENST00000357654.9"
            ),
        )

    def test_findings_stating_two_loci_for_one_rsid_give_their_target_no_position(self):
        row = "GENE1\tchr1:1234567\tA>G\trs123"
        prose = "GENE1 rs123 lies at chr1:1234570 A>G."
        text, paper = paper_of(row, prose, "Aligned to GRCh38.")
        answer = read_as(
            text,
            paper,
            [
                stated("contig", "chr1", "t1:r1"),
                stated("position", "1234567", "t1:r1", number=1234567),
                stated("alleles", "A>G", "t1:r1"),
                stated("rsid", "rs123", "t1:r1"),
            ],
            [
                stated("contig", "chr1", "main:p3"),
                stated("position", "1234570", "main:p3", number=1234570),
                stated("alleles", "A>G", "main:p3"),
                stated("rsid", "rs123", "main:p3"),
            ],
            reference="GRCh38",
        )

        [target] = paper_targets(answer).targets

        self.assertEqual(
            (
                target.request.rsid,
                target.request.position,
                target.request.stated_assembly,
            ),
            ("rs123", None, None),
        )
        self.assertIsNone(target.not_checked)

    def test_an_rsid_target_takes_only_a_position_a_finding_of_that_rsid_states(self):
        text, paper = paper_of(
            "The carrier had rs123 in GENE1 and chr1:1234567 A>G nearby."
        )
        answer = read_as(
            text,
            paper,
            [stated("rsid", "rs123", "main:p2")],
            [
                stated("contig", "chr1", "main:p2"),
                stated("position", "1234567", "main:p2", number=1234567),
                stated("alleles", "A>G", "main:p2"),
            ],
        )

        rsid, located_ = paper_targets(answer).targets

        self.assertEqual((rsid.request.rsid, rsid.request.position), ("rs123", None))
        self.assertEqual(located_.request.position.position, 1234567)

    def test_alleles_not_written_in_vcf_bases_leave_a_position_main_cannot_check(self):
        # Synthetic record: a deletion written with "-" as its ALT; main takes VCF bases
        # only, so the position goes without alleles, and with no rsID main asks.
        text = "GENE1 chr1:1234567 A>- was found."
        line = SourceLine(
            document_id="main", section="R", start=0, end=len(text), text=text
        )
        answer = synthetic_answer(
            KeptFinding(
                variants=(located(line, "1", 1234567, "A", "-"),),
                passages=(line,),
                reason="GENE1 chr1:1234567.",
                gene="GENE1",
            )
        ).model_copy(update={"reference": synthetic_reference("Aligned to GRCh38.")})

        [target] = paper_targets(answer).targets

        self.assertEqual(
            target.request.position, StatedPosition(contig="1", position=1234567)
        )
        self.assertEqual(target.not_checked, "no_alleles")
        self.assertEqual(target.request.stated_assembly.name, "GRCh38")

    def test_a_reference_line_longer_than_mains_limit_is_cited_by_its_span(self):
        # Synthetic record: the only line naming the reference is longer than main's
        # 4,000 characters; the citation is that line's exact span around the name.
        long_line = "x" * 3000 + " Reads were aligned to GRCh38 here. " + "y" * 3000
        reference = synthetic_reference(long_line)
        text = "GENE1 chr1:1234567 A>G was found."
        line = SourceLine(
            document_id="main", section="R", start=0, end=len(text), text=text
        )
        answer = synthetic_answer(
            KeptFinding(
                variants=(located(line, "1", 1234567, "A", "G"),),
                passages=(line,),
                reason="GENE1 chr1:1234567 A>G.",
                gene="GENE1",
            )
        ).model_copy(update={"reference": reference})

        [target] = paper_targets(answer).targets

        cited = target.request.stated_assembly.cited_line
        self.assertLessEqual(len(cited), 4000)
        self.assertIn("aligned to GRCh38", cited)
        self.assertIn(cited, long_line)

    def test_a_paper_with_no_identifier_cannot_assert_a_target(self):
        """Owner, #383 A1b: a paper asserts its targets by DOI, PMID, PMCID or, when
        the researcher supplied it with none, the Run's paper_id; with none of them it
        asserts nothing."""
        answer = recorded_answer("s3-run1").model_copy(
            update={
                "paper": PaperIdentity(
                    title="An evolved yeast", resolved_by="web_search"
                )
            }
        )

        with self.assertRaisesRegex(
            ValueError, "names its paper by DOI, PMID, PMCID or paper_id"
        ):
            paper_targets(answer)

    def test_the_paper_is_named_by_its_pmid_when_it_has_no_doi(self):
        answer = recorded_answer("s3-run1")
        answer = answer.model_copy(
            update={"paper": answer.paper.model_copy(update={"doi": None})}
        )

        self.assertEqual(paper_targets(answer).asserted_by, "paper:pmid:20128923")

    def test_no_finding_gives_no_target(self):
        targets = paper_targets(recorded_answer("s15-run1"))

        self.assertEqual(targets.targets, ())
        self.assertTrue(targets.not_answered.startswith("The paper does not report"))


class TargetsAdapterTests(unittest.TestCase):
    """The ``targets`` command only adapts ``paper_targets``: it writes the record as
    ``targets.json`` and nothing else (owner Q7: P writes no ``/2``; Rf's ``place`` does)."""

    def test_targets_writes_only_the_record_paper_targets_returns(self):
        answer = recorded_answer("h1-run1")
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as given,
        ):
            # The retained answer as the reader writes it today (``recorded_answer``).
            answer_file = Path(given, "answer.json")
            answer_file.write_text(answer.model_dump_json(), "utf-8")
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                code = main(
                    ["targets", "--answer", str(answer_file), "--out", directory]
                )
            written = sorted(p.name for p in Path(directory).iterdir())
            record = PaperTargets.model_validate_json(
                Path(directory, "targets.json").read_text("utf-8")
            )

        self.assertEqual(code, 0)
        self.assertEqual(written, ["targets.json"])
        self.assertEqual(record, paper_targets(answer))
        self.assertEqual(
            json.loads(printed.getvalue()),
            {"asserted_by": H1_PAPER, "targets": 3, "not_checked": 0},
        )


if __name__ == "__main__":
    unittest.main()
