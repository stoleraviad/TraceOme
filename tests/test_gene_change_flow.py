"""G15's claim flow (#376) over scripted NCBI answers: the Run's assembly report, the gene's
annotation (the assembly's own GFF3, every organism's alike: #469, owner Q4 a), the codon's
reference bases through the product's ``ncbi_reference_span``, and the Variation service's
reading of a transcript ``c.`` name. Every GET goes through the product's NCBI
transport and failure module; the answers' shapes follow the development GETs
under docs/development/reviews/issue376-gene-change-targets/devgets/."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import unittest.mock
import zipfile
from pathlib import Path

from bio_neb.failure import run_failure_tuning
from bio_neb.investigation.investigation_request import StatedChange
from bio_neb.providers.ncbi_datasets import UrllibNcbiTransport
from bio_neb.providers.ncbi_investigation import NcbiInvestigationSourceAdapter
from bio_neb.target_request.ncbi_sources import NcbiSources, answer_claim
from bio_neb.target_request.records import GeneChangeClaim, StatedVersion
from tests.gene_change_support import (
    API,
    HUMAN,
    NO_BACKOFF,
    YEAST,
    YEAST_GFF,
    YEAST_GFF_TEXT,
    YEAST_REPORT,
    Answer,
    Opener,
    dataset_report,
    gff_zip,
    names,
    span,
    span_url,
    yeast_answers,
)

SACCER1 = StatedVersion(
    name="sacCer1",
    cited_line="Reads were aligned to the UCSC sacCer1 reference sequence using Maq",
)


def _change(gene, change, transcript=None, protein=None, coding=None):
    """The gene change as the model types it from its words (#473 Q2 a)."""
    return StatedChange(
        gene=gene,
        change=change,
        transcript=transcript,
        protein=None
        if protein is None
        else dict(zip(("ref", "codon", "alt"), protein, strict=True)),
        coding=None
        if coding is None
        else dict(zip(("cds_position", "ref", "alt"), coding, strict=True)),
    )


def _claim(gene="PBP2", change="Y2X", protein=("Y", 2, "*"), **fields):
    stated = {
        "label": "s3-pbp2",
        "stated_assembly": SACCER1,
        "request_text": "the previously-described point mutations at PBP2 (Y2X)",
    }
    typed = {key: fields.pop(key) for key in ("transcript", "coding") if key in fields}
    return GeneChangeClaim(
        change=_change(gene, change, protein=protein, **typed), **(stated | fields)
    )


class _Tool:
    def __init__(self, test, answers):
        test.enterContext(run_failure_tuning(NO_BACKOFF))
        self.root = Path(test.enterContext(tempfile.TemporaryDirectory()))
        self.opener = Opener(answers)
        self.sources = NcbiSources(
            NcbiInvestigationSourceAdapter(UrllibNcbiTransport(opener=self.opener)),
            self.root,
        )

    def answer(self, assembly, claim):
        return answer_claim(self.sources, assembly, claim)


class NonHumanFlowTests(unittest.TestCase):
    def test_a_stop_claim_is_placed_on_the_runs_assembly_from_its_own_annotation(self):
        tool = _Tool(self, yeast_answers())

        answer = tool.answer(YEAST, _claim())

        self.assertEqual(answer.outcome, "candidates")
        self.assertEqual(
            [
                (c.contig, c.position, c.ref, c.alt, c.transcripts)
                for c in answer.candidates
            ],
            [
                ("NC_001134.8", 1006, "C", "A", ("NM_1.3",)),
                ("NC_001134.8", 1006, "C", "G", ("NM_1.3",)),
                ("NC_001134.8", 1005, "AC", "GA", ("NM_1.3",)),
            ],
        )
        self.assertEqual(
            tool.opener.sent,
            [YEAST_REPORT, YEAST_GFF, span_url("NC_001134.8", 1004, 3)],
        )

    def test_a_paper_version_naming_another_assembly_is_recorded_unmatched(self):
        tool = _Tool(self, yeast_answers())

        answer = tool.answer(YEAST, _claim())

        self.assertEqual(
            (answer.version.stated, answer.version.run_name, answer.version.matched),
            ("sacCer1", "R64", None),
        )

    def test_the_runs_paired_genbank_accession_is_the_runs_version(self):
        tool = _Tool(self, yeast_answers())
        stated = StatedVersion(name="GCA_000146045.2", cited_line="the line")

        answer = tool.answer(YEAST, _claim(stated_assembly=stated))

        self.assertEqual(answer.version.matched, "paired_accession")

    def test_the_annotation_and_a_codons_bases_are_asked_once_per_run(self):
        tool = _Tool(self, yeast_answers())

        tool.answer(YEAST, _claim())
        tool.answer(YEAST, _claim(label="again"))

        self.assertEqual(tool.opener.sent.count(YEAST_GFF), 1)
        self.assertEqual(tool.opener.sent.count(YEAST_REPORT), 1)
        self.assertEqual(tool.opener.sent.count(span_url("NC_001134.8", 1004, 3)), 1)

    def test_an_untyped_change_is_not_checked_before_any_annotation_contact(self):
        """Code reads no change text (#473 owner Q2 a, ADR 0049): a change the model
        typed no part of, as a paper's stays until #383 (owner Q12 a), is not read."""
        tool = _Tool(self, yeast_answers())

        answer = tool.answer(YEAST, _claim(change="exon 3 deletion", protein=None))

        self.assertEqual(answer.outcome, "not_checked")
        self.assertEqual(
            answer.reason,
            "the change 'exon 3 deletion' is not typed: the model types its protein "
            "or coding change from the words that state it",
        )
        self.assertEqual(tool.opener.sent, [YEAST_REPORT])

    def test_an_unknown_gene_is_not_found(self):
        tool = _Tool(self, yeast_answers())

        answer = tool.answer(YEAST, _claim(gene="NOPE"))

        self.assertEqual(answer.outcome, "not_found")
        self.assertEqual(answer.reason, "the assembly's annotation names no gene NOPE")

    def test_an_annotation_download_without_a_gff_is_the_claims_semantic_failure(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("ncbi_dataset/data/README.md", "no annotation")
        tool = _Tool(self, yeast_answers(**{YEAST_GFF: Answer(200, buffer.getvalue())}))

        first = tool.answer(YEAST, _claim())
        second = tool.answer(YEAST, _claim(label="again"))

        for answer in (first, second):
            self.assertEqual(
                (
                    answer.failure.tool,
                    answer.failure.failure_class,
                    answer.failure.route,
                ),
                ("ncbi_genome_annotation", "semantic", "ask"),
            )
        self.assertEqual(tool.opener.sent.count(YEAST_GFF), 1)

    def test_a_symbol_naming_two_genes_is_ambiguous_with_no_candidates(self):
        gff = YEAST_GFF_TEXT + (
            "NC_001134.8\tRefSeq\tgene\t5001\t5030\t.\t+\t.\tID=gene-YB2;Name=PBP2;"
            "gene=PBP2\n"
        )
        tool = _Tool(self, yeast_answers(**{YEAST_GFF: Answer(200, gff_zip(gff))}))

        answer = tool.answer(YEAST, _claim())

        self.assertEqual((answer.outcome, answer.candidates), ("ambiguous", ()))
        self.assertEqual(
            answer.reason,
            "the assembly's annotation names 2 genes PBP2: gene-YB1 "
            "(NC_001134.8:1001-1030), gene-YB2 (NC_001134.8:5001-5030)",
        )

    def test_a_failed_annotation_download_is_the_claims_failure_after_its_retries(self):
        tool = _Tool(
            self,
            yeast_answers(**{YEAST_GFF: [Answer(503), Answer(503), Answer(503)]}),
        )

        answer = tool.answer(YEAST, _claim())

        self.assertIsNone(answer.outcome)
        self.assertEqual(
            (answer.failure.tool, answer.failure.failure_class, answer.failure.route),
            ("ncbi_genome_annotation", "exhausted", "ask"),
        )
        self.assertEqual(tool.opener.sent.count(YEAST_GFF), 3)

    def test_a_reference_span_that_is_not_the_asked_span_is_a_failure(self):
        tool = _Tool(
            self,
            yeast_answers(
                **{
                    span_url("NC_001134.8", 1004, 3): Answer(
                        200, span("NC_001134.8", 1005, "TAC")
                    )
                }
            ),
        )

        answer = tool.answer(YEAST, _claim())

        self.assertEqual(
            (answer.failure.tool, answer.failure.failure_class),
            ("ncbi_reference_span", "semantic"),
        )


HUMAN_REPORT = f"{API}/datasets/v2/genome/accession/{HUMAN}/dataset_report"
HUMAN_GFF = (
    f"{API}/datasets/v2/genome/accession/{HUMAN}/download"
    "?include_annotation_type=GENOME_GFF"
)
# APOE's MANE Select transcript as GRCh38's own GFF3 writes it, tag and all: one CDS
# segment 501-530, so codon 2 is 504-506. No transl_table is written for a nuclear CDS.
APOE_GFF_TEXT = "\n".join(
    [
        "##gff-version 3",
        "NC_000019.10\tBestRefSeq\tgene\t501\t530\t.\t+\t.\tID=gene-APOE;Name=APOE;"
        "gene=APOE",
        "NC_000019.10\tBestRefSeq\tmRNA\t501\t530\t.\t+\t.\tID=rna-NM_000041.4;"
        "Parent=gene-APOE;gene=APOE;tag=MANE Select;transcript_id=NM_000041.4",
        "NC_000019.10\tBestRefSeq\tCDS\t501\t530\t.\t+\t0\tID=cds-NP_000032.1;"
        "Parent=rna-NM_000041.4;gene=APOE;protein_id=NP_000032.1;tag=MANE Select",
        "",
    ]
)


def _human_answers(**more):
    return {
        HUMAN_REPORT: Answer(
            200, dataset_report(HUMAN, "GRCh38.p14", "hg38", "GCA_000001405.29", 9606)
        ),
        **names(
            HUMAN,
            json.dumps(
                {"reports": [{"refseq_accession": "NC_000019.10", "chr_name": "19"}]}
            ).encode(),
        ),
        HUMAN_GFF: Answer(200, gff_zip(APOE_GFF_TEXT)),
        span_url("NC_000019.10", 504, 3): Answer(200, span("NC_000019.10", 504, "TGC")),
        **more,
    }


def _hgvs_url(name):
    return f"{API}/variation/v0/hgvs/{name.replace('>', '%3E')}/contextuals"


class HumanFlowTests(unittest.TestCase):
    def _claim(self, change="p.Cys2Arg", protein=("C", 2, "R"), **typed):
        return GeneChangeClaim(
            label="apoe",
            change=_change("APOE", change, protein=protein, **typed),
            request_text=f"APOE {change}",
        )

    def test_a_human_claim_reads_its_gene_from_the_runs_own_annotation(self):
        """No human route (#469, owner Q4 a): a human Run's gene comes from its
        assembly's own GFF3, as every Run's does, its MANE Select tag copied."""
        tool = _Tool(self, _human_answers())

        answer = tool.answer(HUMAN, self._claim())

        self.assertEqual(
            tool.opener.sent,
            [HUMAN_REPORT, HUMAN_GFF, span_url("NC_000019.10", 504, 3)],
        )
        [read] = answer.transcripts
        self.assertEqual(
            (read.accession, read.tags, read.genetic_code, read.genetic_code_origin),
            ("NM_000041.4", ("MANE Select",), 1, "default"),
        )
        self.assertEqual(answer.outcome, "candidates")
        # TGC (Cys) to CGC, CGT, CGA, CGG, AGA, AGG (Arg): one base gives CGC.
        self.assertEqual(
            [(c.position, c.ref, c.alt, c.bases_changed) for c in answer.candidates][0],
            (504, "T", "C", 1),
        )
        self.assertEqual(len(answer.candidates), 6)
        self.assertEqual(answer.version.matched, None)
        self.assertEqual(answer.version.stated, None)

    def test_a_transcript_coding_name_is_checked_by_the_variation_service(self):
        name = "NM_000041.4:c.4T>C"
        spdi = {
            "seq_id": "NM_000041.4",
            "position": 3,
            "deleted_sequence": "T",
            "inserted_sequence": "C",
        }
        equivalents = (
            f"{API}/variation/v0/spdi/NM_000041.4:3:T:C/all_equivalent_contextual"
        )
        tool = _Tool(
            self,
            _human_answers(
                **{
                    _hgvs_url(name): Answer(200, {"data": {"spdis": [spdi]}}),
                    equivalents: Answer(
                        200,
                        {
                            "data": {
                                "spdis": [
                                    spdi,
                                    {
                                        "seq_id": "NC_000019.10",
                                        "position": 503,
                                        "deleted_sequence": "T",
                                        "inserted_sequence": "C",
                                    },
                                ]
                            }
                        },
                    ),
                }
            ),
        )

        answer = tool.answer(
            HUMAN,
            self._claim(
                change="c.4T>C",
                protein=None,
                coding=(4, "T", "C"),
                transcript="NM_000041.4",
            ),
        )

        self.assertEqual(answer.outcome, "candidates")
        self.assertEqual(
            [(c.position, c.ref, c.alt) for c in answer.candidates], [(504, "T", "C")]
        )
        # NCBI's answer is copied with its status and body.
        hgvs = answer.hgvs.model_dump()
        self.assertEqual(json.loads(hgvs.pop("body")), {"data": {"spdis": [spdi]}})
        self.assertEqual(
            hgvs,
            {
                "hgvs": name,
                "status_code": 200,
                "spdis": ("NM_000041.4:3:T:C",),
                "on_run": ("NC_000019.10:503:T:C",),
                "candidates": ("NC_000019.10:503:T:C",),
            },
        )

    def test_a_variation_service_reading_elsewhere_is_copied_for_the_model(self):
        """No code verdict on NCBI's answer (#473 owner Q3 a, ADR 0049): a reading
        elsewhere names no candidate, the annotation's outcome stands, and a
        disagreement is the model's to open the member's Review on."""
        name = "NM_000041.4:c.4T>C"
        spdi = {
            "seq_id": "NM_000041.4",
            "position": 3,
            "deleted_sequence": "T",
            "inserted_sequence": "C",
        }
        equivalents = (
            f"{API}/variation/v0/spdi/NM_000041.4:3:T:C/all_equivalent_contextual"
        )
        elsewhere = {
            "seq_id": "NC_000019.10",
            "position": 900,
            "deleted_sequence": "T",
            "inserted_sequence": "C",
        }
        tool = _Tool(
            self,
            _human_answers(
                **{
                    _hgvs_url(name): Answer(200, {"data": {"spdis": [spdi]}}),
                    equivalents: Answer(200, {"data": {"spdis": [spdi, elsewhere]}}),
                }
            ),
        )

        answer = tool.answer(
            HUMAN,
            self._claim(
                change="c.4T>C",
                protein=None,
                coding=(4, "T", "C"),
                transcript="NM_000041.4",
            ),
        )

        self.assertEqual((answer.outcome, answer.reason), ("candidates", None))
        self.assertEqual(
            (answer.hgvs.on_run, answer.hgvs.candidates),
            (("NC_000019.10:900:T:C",), ()),
        )

    def test_a_named_transcript_is_asked_even_when_the_annotation_has_no_gene(self):
        """Whenever a transcript is named, NCBI is asked (#473 owner Q3 a; review r1
        SP2): a coding name needs no annotation."""
        name = "NM_000041.4:c.4T>C"
        tool = _Tool(self, _human_answers(**{_hgvs_url(name): Answer(400, {"e": 1})}))

        answer = tool.answer(
            HUMAN,
            GeneChangeClaim(
                label="nope",
                change=_change(
                    "NOPE", "c.4T>C", transcript="NM_000041.4", coding=(4, "T", "C")
                ),
                request_text="NOPE c.4T>C on NM_000041.4",
            ),
        )

        self.assertEqual(
            (answer.outcome, answer.hgvs.hgvs, answer.hgvs.status_code),
            ("not_found", name, 400),
        )

    def test_every_contextual_allele_ncbi_names_is_kept(self):
        """No cap on NCBI's answer (ADR 0049; review r1 SP3): two contextual alleles
        are both kept, with each one's equivalents on the Run's sequences."""
        name = "NM_000041.4:c.4T>C"
        first = {
            "seq_id": "NM_000041.4",
            "position": 3,
            "deleted_sequence": "T",
            "inserted_sequence": "C",
        }
        second = {**first, "seq_id": "NM_000041.5"}
        on_run = {**first, "seq_id": "NC_000019.10", "position": 503}
        tool = _Tool(
            self,
            _human_answers(
                **{
                    _hgvs_url(name): Answer(200, {"data": {"spdis": [first, second]}}),
                    f"{API}/variation/v0/spdi/NM_000041.4:3:T:C/"
                    "all_equivalent_contextual": Answer(
                        200, {"data": {"spdis": [first, on_run]}}
                    ),
                    f"{API}/variation/v0/spdi/NM_000041.5:3:T:C/"
                    "all_equivalent_contextual": Answer(
                        200, {"data": {"spdis": [second, on_run]}}
                    ),
                }
            ),
        )

        answer = tool.answer(
            HUMAN,
            self._claim(
                change="c.4T>C",
                protein=None,
                coding=(4, "T", "C"),
                transcript="NM_000041.4",
            ),
        )

        self.assertEqual(
            (answer.hgvs.spdis, answer.hgvs.on_run, answer.hgvs.candidates),
            (
                ("NM_000041.4:3:T:C", "NM_000041.5:3:T:C"),
                ("NC_000019.10:503:T:C",),
                ("NC_000019.10:503:T:C",),
            ),
        )

    def test_a_protein_change_on_a_named_transcript_asks_for_its_protein_name(self):
        """With no coding change typed, the name is the typed protein change on the
        protein the annotation gives the named transcript (#473 owner Q3 a)."""
        name = "NP_000032.1:p.C2R"
        tool = _Tool(
            self,
            _human_answers(
                **{_hgvs_url(name): Answer(400, {"error": {"message": "unreadable"}})}
            ),
        )

        answer = tool.answer(HUMAN, self._claim(transcript="NM_000041.4"))

        self.assertEqual(tool.opener.sent[-1], _hgvs_url(name))
        self.assertEqual(
            (answer.outcome, answer.hgvs.hgvs, answer.hgvs.status_code),
            ("candidates", name, 400),
        )

    def test_a_non_human_transcript_coding_name_is_checked_too(self):
        """The Variation service's check has no human gate (#469, owner Q4 a): a yeast
        claim naming its transcript's ``c.`` change is read by the service as well."""
        name = "NM_1.3:c.4T>G"
        spdi = {
            "seq_id": "NM_1.3",
            "position": 3,
            "deleted_sequence": "T",
            "inserted_sequence": "G",
        }
        equivalents = f"{API}/variation/v0/spdi/NM_1.3:3:T:G/all_equivalent_contextual"
        on_run = {
            "seq_id": "NC_001134.8",
            "position": 1003,
            "deleted_sequence": "T",
            "inserted_sequence": "G",
        }
        tool = _Tool(
            self,
            yeast_answers(
                **names(
                    YEAST,
                    json.dumps(
                        {"reports": [{"refseq_accession": "NC_001134.8"}]}
                    ).encode(),
                ),
                **{
                    _hgvs_url(name): Answer(200, {"data": {"spdis": [spdi]}}),
                    equivalents: Answer(200, {"data": {"spdis": [spdi, on_run]}}),
                },
            ),
        )

        answer = tool.answer(
            YEAST,
            _claim(
                change="c.4T>G",
                protein=None,
                coding=(4, "T", "G"),
                transcript="NM_1.3",
                stated_assembly=None,
            ),
        )

        self.assertEqual(answer.outcome, "candidates")
        self.assertEqual(
            (answer.hgvs.status_code, answer.hgvs.on_run, answer.hgvs.candidates),
            (200, ("NC_001134.8:1003:T:G",), ("NC_001134.8:1003:T:G",)),
        )

    def test_ncbis_not_human_answer_is_copied_and_the_annotation_still_answers(self):
        """NCBI's HGVS service answers only human sequences: its 400 is copied as its
        answer, never skipped by code and never a failure (#473 owner Q3 a, owner
        2026-10-28: not human-only), and the annotation's candidates stand."""
        name = "NM_1.3:c.4T>G"
        refusal = {
            "error": {"code": 400, "message": "The submitted sequence is not human."}
        }
        tool = _Tool(
            self,
            yeast_answers(
                **names(
                    YEAST,
                    json.dumps(
                        {"reports": [{"refseq_accession": "NC_001134.8"}]}
                    ).encode(),
                ),
                **{_hgvs_url(name): Answer(400, refusal)},
            ),
        )

        answer = tool.answer(
            YEAST,
            _claim(
                change="c.4T>G",
                protein=None,
                coding=(4, "T", "G"),
                transcript="NM_1.3",
                stated_assembly=None,
            ),
        )

        self.assertEqual(answer.outcome, "candidates")
        self.assertIsNone(answer.failure)
        self.assertEqual(
            (
                answer.hgvs.status_code,
                json.loads(answer.hgvs.body),
                answer.hgvs.spdis,
                answer.hgvs.on_run,
            ),
            (400, refusal, (), ()),
        )


class _CommandCase(unittest.TestCase):
    def setUp(self):
        self.enterContext(run_failure_tuning(NO_BACKOFF))
        self.cache = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out = io.StringIO()

    def _main(self, argv, answers):
        from tools.variant_sources import __main__ as command

        self.opener = Opener(answers)
        with (
            contextlib.redirect_stdout(self.out),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return command.main(["--cache", str(self.cache), *argv], opener=self.opener)


class VersionCommandTests(_CommandCase):
    """``python -m tools.variant_sources version``: stated names against the Run's
    assembly. G15's claims are placed through the product's `place` since #367
    (`tests/test_paper_targets_placement.py` `GeneChangeTests`)."""

    def test_version_resolves_each_name_against_the_runs_assembly(self):
        code = self._main(
            [
                "version",
                "--assembly",
                YEAST,
                "--name",
                "GCA_000146045.2",
                "--name",
                "sacCer3",
                "--name",
                "sacCer1",
            ],
            yeast_answers(),
        )

        self.assertEqual(code, 0)
        self.assertEqual(
            self.out.getvalue().splitlines(),
            [
                "GCA_000146045.2\tR64 (GCF_000146045.2)\tpaired_accession",
                "sacCer3\tR64 (GCF_000146045.2)\tsynonym",
                "sacCer1\tR64 (GCF_000146045.2)\tnot a name NCBI's report gives the Run "
                "(R64, sacCer3, GCF_000146045.2, GCA_000146045.2)",
            ],
        )
