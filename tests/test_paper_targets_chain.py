"""Where lane P's writer meets lane Rf's reader (#386, owner map #307 lane P Q8 and Q9).

Lane P's real ``targets`` command writes ``targets.json`` from an answer; lane Rf's real
``place-targets`` places it on the Run's assembly and writes the Run's ``/2`` request file;
main reads that file, plans with the paper as asserter, and runs its lookup and intent check.
These tests restore the guarantees of lane P's tests retired with its ``request.json`` bridge
(``1b8b220b:tests/test_paper_targets.py`` ``PaperTargetVerificationTests``) and pin lane P's
writer against lane Rf's reader, which reads the record as data.

The answers are #349's retained h1 answer and answers code keeps from model readings
(``fill``), never the pre-#404 s3 and s17 recordings (#386 compare B8). NCBI is scripted
(``tests/gene_change_support.py``); the RefSNP bodies are synthetic as in the retired tests.
"""

import contextlib
import copy
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from bio_neb.application.advancement import (
    ResumeRun,
    advance,
    use_run_advancement_composition,
)
from bio_neb.content import canonical_json
from bio_neb.failure import run_failure_tuning
from bio_neb.investigation.investigation import (
    InvestigationInput,
    InvestigationVcfIdentity,
    UserAssertion,
)
from bio_neb.investigation.investigation_request import (
    current_investigation_request,
    read_investigation_request_document,
)
from bio_neb.journey import Journey
from bio_neb.paper_reading.contracts import PaperAnswer
from bio_neb.paper_reading.targets import paper_targets as lane_p_targets
from bio_neb.recording import read_evidence
from bio_neb.target_request.paper_targets import PaperTargets as WrittenTargets
from tests.gene_change_support import (
    HORSE,
    HUMAN,
    NO_BACKOFF,
    NO_EQUIVALENT,
    SACCER1_TABLE,
    YEAST,
    YEAST_GFF,
    Answer,
    Opener,
    dataset_report,
    equivalents_url,
    gff_zip,
    horse_answers,
    human_answers,
    paper_target,
    paper_targets,
    span,
    span_url,
    yeast_answers,
)
from tests.location_retained_bodies import GRCH38_REPORT
from tests.paper_targets_support import (
    AKAP9_ROW,
    H1_PAPER,
    KCNH2_ROW,
    TTN_ROW,
    paper_of,
    read_as,
    recorded_answer,
    s3_answer,
    s17_answer,
    stated,
)
from tests.planning_assignment_fakes import configured_planning_run
from tests.reference_index_support import INSTALL, SyntheticPair, choice_record
from tests.refsnp_retained_bodies import (
    RS429358_DOCUMENT,
    RS4149056,
    RS4149056_DOCUMENT,
    RS60037639,
    refsnp_body,
)
from tests.sequence_naming_support import assembly_report_text
from tests.sequential_investigation_support import (
    SequentialAdaptiveModel,
    SourceResponsesOnly,
    input_and_assertion,
    run_attempt,
)
from tools.paper_retrieval.__main__ import main as paper_main
from tools.reference_index.__main__ import main as reference_main


def h1_body(
    rsid: str, seq_id: str, position: int, ref: str, alts: tuple[str, ...]
) -> bytes:
    """Synthetic RefSNP body (the retired tests' ``h1_body``): #300's retained rs429358
    record with the paper's rsID and its Table 1 locus on GRCh38, no merges and no GRCh37
    placement. The alleles are supplied, not dbSNP's."""

    document = copy.deepcopy(RS429358_DOCUMENT)
    document["refsnp_id"] = rsid.removeprefix("rs")
    document["dbsnp1_merges"] = []
    snapshot = document["primary_snapshot_data"]
    [placement] = [p for p in snapshot["placements_with_allele"] if p["is_ptlp"]]
    placement["seq_id"] = seq_id
    placement["alleles"] = [
        {
            "allele": {
                "spdi": {
                    "seq_id": seq_id,
                    "position": position - 1,
                    "deleted_sequence": ref,
                    "inserted_sequence": inserted,
                }
            }
        }
        for inserted in (ref, *alts)
    ]
    snapshot["placements_with_allele"] = [placement]
    return refsnp_body(document)


# EquCab3.0's reports as main's lookup reads them: two retained development GETs, with the
# status the first records for GCF_002863925.1 (``suppressed``), which main's report
# reader requires.
_EQUCAB3_REPORT = (
    b'{"reports":[{"accession":"GCF_002863925.1","paired_accession":"GCA_002863925.1",'
    b'"assembly_info":{"assembly_name":"EquCab3.0","synonym":"equCab3",'
    b'"assembly_status":"suppressed"}}],"total_count":1}'
)
_EQUCAB3_CHR4 = (
    b'{"reports":[{"assembly_accession":"GCF_002863925.1","chr_name":"4","ucsc_style_name'
    b'":"chr4","sort_order":4,"assigned_molecule_location_type":"Chromosome","refseq_acces'
    b'sion":"NC_009147.3","assembly_unit":"Primary Assembly","length":109462549,"genbank_a'
    b'ccession":"CM009151.1","gc_count":"43561424","gc_percent":40,"role":"assembled-molec'
    b'ule","sequence_name":"chr4"}],"total_count":4701}'
)


_UNTYPED = (
    "the change '{}' is not typed: the model types its protein or coding change "
    "from the words that state it"
)


class HorseSources(SourceResponsesOnly):
    """A horse Run's NCBI reports (above) and EVA's listing at a position, synthetic: EVA
    lists no variant there. dbSNP is never asked."""

    def __init__(self, **kwargs) -> None:
        super().__init__(assembly_body=_EQUCAB3_REPORT, **kwargs)

    def fetch_refsnp(self, rsid: str):  # noqa: ANN201
        raise AssertionError("a non-human Run never reaches dbSNP")

    def fetch_sequence_names(self, assembly_accession: str):  # noqa: ANN201
        return self._answer(
            "ncbi_sequence_names",
            assembly_accession,
            "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
            f"{assembly_accession}/assembly_report.txt",
            200,
            assembly_report_text(_EQUCAB3_CHR4),
        )

    def fetch_eva_clustered_variants_at(self, key: str):  # noqa: ANN201
        return self._answer(
            "eva_clustered_variants_at",
            key,
            "https://www.ebi.ac.uk/eva/webservices/identifiers/v1/clustered-variants/"
            f"at/{key}",
            200,
            b"[]",
        )


def _gene(contig: str, start: int, codons: int, gene: str, number: int) -> list[str]:
    """One plus-strand gene with one mRNA and its CDS, in NCBI's GFF3 rows."""
    end = start + 3 * codons - 1
    rna, cds = f"NM_{number}.1", f"NP_{number}.1"
    return [
        f"{contig}\tRefSeq\tgene\t{start}\t{end}\t.\t+\t.\tID=gene-{gene};Name={gene};"
        f"gene={gene}",
        f"{contig}\tRefSeq\tmRNA\t{start}\t{end}\t.\t+\t.\tID=rna-{rna};"
        f"Parent=gene-{gene};gene={gene};transcript_id={rna}",
        f"{contig}\tRefSeq\tCDS\t{start}\t{end}\t.\t+\t0\tID=cds-{cds};"
        f"Parent=rna-{rna};gene={gene};protein_id={cds}",
    ]


# Synthetic R64 annotation holding s3's three stated codons: PBP2 codon 127 (TAC) at
# NC_001134.8:1379, SGF73 codon 294 (GAA) at NC_001140.6:5880, UPF3 codon 6 (GGT) at
# NC_001144.5:9016. Not NCBI's annotation: it places s3's changes, not their real loci.
S3_GFF = "\n".join(
    [
        "##gff-version 3",
        *_gene("NC_001134.8", 1001, 127, "PBP2", 2),
        *_gene("NC_001140.6", 5001, 294, "SGF73", 3),
        *_gene("NC_001144.5", 9001, 10, "UPF3", 4),
        "",
    ]
)


class PaperTargetsChainTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(run_failure_tuning(NO_BACKOFF))
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.request = self.root / "request.json"

    def written(self, answer: PaperAnswer | Path) -> Path:
        """Lane P's ``targets`` command on the answer: its ``targets.json``."""
        if isinstance(answer, PaperAnswer):
            path = self.root / "answer.json"
            path.write_text(answer.model_dump_json(), "utf-8")
            answer = path
        out = self.root / "paper"
        with contextlib.redirect_stdout(io.StringIO()):
            code = paper_main(["targets", "--answer", str(answer), "--out", str(out)])
        self.assertEqual(code, 0)
        return out / "targets.json"

    def placed(self, targets: Path, assembly: str, answers: dict, *options: str):
        """Lane Rf's ``place-targets`` on lane P's file: the exit code and its record."""
        self.opener = Opener(answers)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = reference_main(
                [
                    "place-targets",
                    "--assembly",
                    assembly,
                    "--paper-targets",
                    str(targets),
                    "--request",
                    str(self.request),
                    "--cache",
                    str(self.root / "reference-index"),
                    "--variant-cache",
                    str(self.root / "variant-sources"),
                    *options,
                ],
                opener=self.opener,
                tools=INSTALL,
            )
        return code, json.loads(out.getvalue())

    def steps(self, record):
        return [(a["label"], a["answer"], a["step"]) for a in record["answers"]]

    def assertion_on(self, assembly: str):
        """Main's assertion from the written ``/2`` file, read by main's reader."""
        request = read_investigation_request_document(str(self.request))[1]
        investigation_input = InvestigationInput.create(
            vcf=InvestigationVcfIdentity.create(
                bucket="accepted-run-bucket",
                object_key="outputs/SAMPLE01.vcf.gz",
                byte_count=1234,
                sha256="d" * 64,
            ),
            sample_id="SAMPLE01",
            assembly_accession=assembly,
        )
        return (
            request,
            investigation_input,
            UserAssertion.create_collection(
                asserted_by=request.asserted_by,
                requests=request.items,
                predecessor_ids=(investigation_input.input_id,),
            ),
        )

    def test_h1s_targets_are_kept_and_planning_cites_the_file_the_paper_asserts(self):
        targets = self.written(recorded_answer("h1-run1"))

        code, record = self.placed(targets, HUMAN, human_answers())

        self.assertEqual(code, 0, record)
        self.assertEqual(
            self.steps(record),
            [(f"t{n}", "kept", "same_version") for n in (1, 2, 3)],
        )
        receipt, composition, _ = configured_planning_run(
            self.root / "planning",
            investigation_request=json.loads(self.request.read_text()),
        )
        with use_run_advancement_composition(composition):
            advance(ResumeRun(journey_file=receipt.journey_file))
        planned = current_investigation_request(
            Journey.open(Path(receipt.journey_file))
        )
        self.assertEqual(planned.request.asserted_by, H1_PAPER)
        self.assertEqual(
            # Each item is the paper's line, then the placement's note.
            [
                (item.request_text.split(" (placement: ")[0], item.rsid)
                for item in planned.request.items
            ],
            [
                (KCNH2_ROW, "rs199472944"),
                (AKAP9_ROW, "rs779447911"),
                (TTN_ROW, "rs773002407"),
            ],
        )
        # The paper's hg38 reaches planning as stated, each position kept.
        self.assertEqual(
            [
                (
                    item.position.position,
                    item.stated_assembly.name,
                    item.placement.step,
                    item.placed.position,
                )
                for item in planned.request.items
            ],
            [
                (150951552, "hg38", "same_version", 150951552),
                (92042078, "hg38", "same_version", 92042078),
                (178532658, "hg38", "same_version", 178532658),
            ],
        )

    def test_h1s_targets_enter_mains_verification_and_its_intent_check_passes(self):
        self.placed(self.written(recorded_answer("h1-run1")), HUMAN, human_answers())
        request = read_investigation_request_document(str(self.request))[1]
        investigation_input, _ = input_and_assertion()
        assertion = UserAssertion.create_collection(
            asserted_by=request.asserted_by,
            requests=request.items,
            predecessor_ids=(investigation_input.input_id,),
        )
        sources = SourceResponsesOnly(
            refsnp_bodies={
                "rs199472944": h1_body(
                    "rs199472944", "NC_000007.14", 150951552, "G", ("A",)
                ),
                "rs779447911": h1_body(
                    "rs779447911", "NC_000007.14", 92042078, "AG", ("",)
                ),
                "rs773002407": h1_body(
                    "rs773002407", "NC_000002.12", 178532658, "G", ("A",)
                ),
            },
            assembly_body=GRCH38_REPORT,
        )
        run_root = self.root / "run"
        run_root.mkdir()

        result, journey = run_attempt(
            run_root,
            SequentialAdaptiveModel(),
            sources,
            investigation_input=investigation_input,
            assertion=assertion,
        )

        events = read_evidence(journey)
        self.assertEqual(
            [
                event["payload"]["status"]
                for event in events
                if event["event_type"] == "AgentOutputValidated"
            ],
            ["valid"] * 3,
        )
        self.assertEqual(result.status, "review_pending")
        [started] = [e for e in events if e["event_type"] == "InvestigationStarted"]
        self.assertEqual(started["payload"]["assertion"]["asserted_by"], H1_PAPER)
        self.assertEqual(
            [
                (w.target.contig, w.target.one_based_position, w.target.ref)
                for w in result.pending_request.proposal.work_items
            ],
            [
                ("NC_000007.14", 150951552, "G"),
                ("NC_000007.14", 92042078, "AG"),
                ("NC_000002.12", 178532658, "G"),
            ],
        )

    def test_s17s_target_is_kept_on_equcab3_and_becomes_its_own_work_item(self):
        code, record = self.placed(self.written(s17_answer()), HORSE, horse_answers())

        self.assertEqual(code, 0, record)
        self.assertEqual(self.steps(record), [("t1", "kept", "same_version")])
        _, investigation_input, assertion = self.assertion_on(HORSE)
        sources = HorseSources(reference_bases={"NC_009147.3:54108296:1": "G"})
        run_root = self.root / "run"
        run_root.mkdir()
        result, journey = run_attempt(
            run_root,
            SequentialAdaptiveModel(),
            sources,
            investigation_input=investigation_input,
            assertion=assertion,
        )
        self.assertEqual(
            [
                event["payload"]["status"]
                for event in read_evidence(journey)
                if event["event_type"] == "AgentOutputValidated"
            ],
            ["valid"],
        )
        self.assertEqual(
            [
                (
                    w.target.contig,
                    w.target.one_based_position,
                    w.target.ref,
                    w.target.alt,
                )
                for w in result.pending_request.proposal.work_items
            ],
            [("NC_009147.3", 54108297, "G", "T")],
        )

    def test_a_placed_target_is_checked_at_its_placed_position_never_as_stated(self):
        # A /3 item placed on EquCab3.0 from a version the Run is not (moved by a table
        # step), and one kept by an alias the Run's report does not give, each pass the
        # Run's own lookup and intent check at the placed position.
        stated = {"contig": "4", "position": 54108300, "ref": "G", "alt": "T"}
        placed = {"contig": "4", "position": 54108297, "ref": "G", "alt": "T"}
        kept = {"contig": "4", "position": 54108297, "ref": "G", "alt": "T"}
        cases = (
            ("table", "EquCab2.0", "EquCab2.0", stated, placed),
            ("same_version", "the horse genome v3", "EquCab3.0", kept, kept),
        )
        for step, version, matched, stated_at, placed_at in cases:
            with self.subTest(step=step):
                self.request.write_text(
                    json.dumps(
                        {
                            "schema_version": "bio-neb.investigation-request/3",
                            "asserted_by": "paper:doi:10.1000/horse",
                            "requests": [
                                {
                                    "request": "RAPGEF5 4:54108300 G>T (placement)",
                                    "target": {"position": stated_at},
                                    "stated_assembly": {
                                        "name": version,
                                        "cited_line": f"aligned to {version}",
                                    },
                                    "placement": {
                                        "step": step,
                                        "matched_name": matched,
                                        "position": placed_at,
                                        "reason": "placed for the test",
                                    },
                                }
                            ],
                        }
                    )
                )
                _, investigation_input, assertion = self.assertion_on(HORSE)
                sources = HorseSources(reference_bases={"NC_009147.3:54108296:1": "G"})
                run_root = self.root / f"run-{step}"
                run_root.mkdir()

                result, journey = run_attempt(
                    run_root,
                    SequentialAdaptiveModel(),
                    sources,
                    investigation_input=investigation_input,
                    assertion=assertion,
                )

                events = read_evidence(journey)
                self.assertEqual(
                    [
                        e["payload"]["status"]
                        for e in events
                        if e["event_type"] == "AgentOutputValidated"
                    ],
                    ["valid"],
                )
                # The placed contig `4` is the paper's: the model names its Run
                # sequence after the first lookup asks.
                unnamed, named = [
                    e["payload"]["result"]
                    for e in events
                    if e["event_type"] == "RefSnpToolResultRecorded"
                ]
                self.assertEqual(unnamed["status"], "sequence_unnamed")
                location = named["location"]
                self.assertEqual(
                    [
                        (item["record"], item["match"])
                        for item in location["sequence_namings"]
                    ],
                    [("NC_009147.3", "model_matched")],
                )
                self.assertEqual(
                    (location["stated"], location["placed"]["position"]),
                    (stated_at, 54108297),
                )
                self.assertEqual(
                    [
                        (w.target.contig, w.target.one_based_position)
                        for w in result.pending_request.proposal.work_items
                    ],
                    [("NC_009147.3", 54108297)],
                )

    def test_s3s_four_targets_are_each_answered_and_main_reads_the_file(self):
        # The paper reader here types no change, so each is answered not checked
        # until a reader types it, and the sacCer1-era chromosome XI is not in this
        # test's cache.
        answers = yeast_answers(
            **{equivalents_url("NC_001143.4:207468:A:C"): NO_EQUIVALENT}
        )

        code, record = self.placed(
            self.written(s3_answer()),
            YEAST,
            answers,
            "--stated-sequences",
            str(SACCER1_TABLE),
        )

        self.assertEqual(code, 0, record)
        self.assertEqual(
            self.steps(record),
            [
                ("t1", "not_checked", "g15"),
                ("t2", "not_checked", "g15"),
                ("t3", "not_checked", "g15"),
                ("t4", "not_checked", "none"),
            ],
        )
        self.assertEqual(
            [a["reason"] for a in record["answers"][:3]],
            [_UNTYPED.format(change) for change in ("Y127X", "E294X", "G6W")],
        )
        # P writes the paper's `chr11`; the table's row `chr11`, as written, names its
        # sequence (a row matches its contig exactly).
        self.assertIn(
            "sacCer1's chr11 is NC_001143.4, which is not in the reference-index cache: "
            "python -m tools.reference_index fetch-sequence NC_001143.4",
            record["answers"][3]["reason"],
        )
        request, _, _ = self.assertion_on(YEAST)
        self.assertEqual(len(request.items), 4)
        self.assertEqual(request.asserted_by, "paper:doi:10.1000/synthetic")

    def stored_pair(self) -> SyntheticPair:
        """#332's synthetic pair in the reference-index cache, as lane Rf's move tests
        hold it: A's ``chrA`` is the sacCer1-era ``NC_001143.4`` (chromosome 11 by the
        fixture table's row), B the Run's stored index, which a choice record names for
        R64."""
        (self.root / "pair").mkdir()
        pair = SyntheticPair(self.root / "pair")
        cache = self.root / "reference-index"
        sequence = cache / "ncbi" / "sequences" / "NC_001143.4.fa"
        sequence.parent.mkdir(parents=True)
        sequence.write_text(pair.source.read_text().replace(">chrA", ">NC_001143.4", 1))
        digest = hashlib.sha256(pair.target.read_bytes()).hexdigest()
        store = cache / "store" / "sha256" / digest
        store.mkdir(parents=True)
        for suffix in ("", ".amb", ".ann", ".bwt", ".pac", ".sa", ".fai"):
            shutil.copy(f"{pair.target}{suffix}", store / f"reference.fa{suffix}")
        choice = cache / "choices" / "GCF_900000001.1.json"
        choice.parent.mkdir(parents=True)
        choice.write_text(
            canonical_json(
                choice_record("GCF_900000001.1", YEAST, digest, []).model_dump(
                    mode="json"
                )
            )
        )
        return pair

    def test_s3s_changes_wait_for_a_model_and_its_position_is_moved_onto_r64(self):
        # s3's sentence with its own changes, and its sacCer1 position on the
        # synthetic pair (segment 0). The paper reader here types no change, so its
        # three changes are not checked until a reader types them; the position is
        # moved as before.
        pair = self.stored_pair()
        position, ref, alt = pair.positions[0], pair.refs[0], pair.alts[0]
        sentence = (
            "We detected four single-point differences: the previously-described point "
            "mutations at PBP2 (Y127X), SGF73 (E294X), and UPF3 (G6W) [4], and an "
            "additional point mutation intergenic to RRN3 and YPK1 "
            f"(chr11: {position}:{ref}>{alt})."
        )
        text, paper = paper_of(
            sentence, "Reads were aligned to the UCSC sacCer1 genome."
        )
        answer = read_as(
            text,
            paper,
            [stated("gene", "PBP2", "main:p2"), stated("change", "Y127X", "main:p2")],
            [stated("gene", "SGF73", "main:p2"), stated("change", "E294X", "main:p2")],
            [stated("gene", "UPF3", "main:p2"), stated("change", "G6W", "main:p2")],
            [
                stated("gene", "YPK1", "main:p2"),
                stated("contig", "chr11", "main:p2"),
                stated("position", str(position), "main:p2", number=position),
                stated("alleles", f"{ref}>{alt}", "main:p2"),
            ],
            reference="sacCer1",
        )
        answers = yeast_answers(
            **{
                YEAST_GFF: Answer(200, gff_zip(S3_GFF)),
                span_url("NC_001134.8", 1379, 3): Answer(
                    200, span("NC_001134.8", 1379, "TAC")
                ),
                span_url("NC_001140.6", 5880, 3): Answer(
                    200, span("NC_001140.6", 5880, "GAA")
                ),
                span_url("NC_001144.5", 9016, 3): Answer(
                    200, span("NC_001144.5", 9016, "GGT")
                ),
                equivalents_url(
                    f"NC_001143.4:{position - 1}:{ref}:{alt}"
                ): NO_EQUIVALENT,
            }
        )

        code, record = self.placed(
            self.written(answer),
            YEAST,
            answers,
            "--stated-sequences",
            str(SACCER1_TABLE),
        )

        self.assertEqual(code, 0, record)
        self.assertEqual(
            self.steps(record),
            [
                ("t1", "not_checked", "g15"),
                ("t2", "not_checked", "g15"),
                ("t3", "not_checked", "g15"),
                ("t4", "moved", "table"),
            ],
        )
        moved = record["answers"][3]["position"]
        receipt, composition, _ = configured_planning_run(
            self.root / "planning",
            investigation_request=json.loads(self.request.read_text()),
        )
        with use_run_advancement_composition(composition):
            advance(ResumeRun(journey_file=receipt.journey_file))
        planned = current_investigation_request(
            Journey.open(Path(receipt.journey_file))
        )
        self.assertEqual(planned.request.asserted_by, "paper:doi:10.1000/synthetic")
        # The moved item reaches planning placed on R64, never stated there; each
        # change reaches it unplaced, with its gene and change as written.
        located = [
            (i.placed.contig, i.placed.position, i.placed.ref, i.placed.alt)
            for i in planned.request.items
            if i.placed is not None
        ]
        self.assertEqual(
            located,
            [(moved["contig"], moved["position"], moved["ref"], moved["alt"])],
        )
        self.assertEqual(
            [
                (i.gene_change.change, i.placement.step)
                for i in planned.request.items
                if i.gene_change is not None
            ],
            [("Y127X", "none"), ("E294X", "none"), ("G6W", "none")],
        )

    def test_a_table_row_spelled_as_the_paper_writes_it_is_kept_on_grch38(self):
        row = "KCNH2 chr7:150951552 c.1841C>T p. (Ala614Val) G>A rs199472944"
        text, paper = paper_of(row, "Reads were aligned to hg38.")
        answer = read_as(
            text,
            paper,
            [
                stated("gene", "KCNH2", "main:p2"),
                stated("contig", "chr7", "main:p2"),
                stated("position", "150951552", "main:p2", number=150951552),
                stated("change", "c.1841C>T", "main:p2"),
                stated("change", "p. (Ala614Val)", "main:p2"),
                stated("alleles", "G>A", "main:p2"),
                stated("rsid", "rs199472944", "main:p2"),
            ],
            reference="hg38",
        )

        code, record = self.placed(self.written(answer), HUMAN, human_answers())

        self.assertEqual(code, 0, record)
        self.assertEqual(self.steps(record), [("t1", "kept", "same_version")])
        request, _, _ = self.assertion_on(HUMAN)
        [item] = request.items
        self.assertEqual(
            (item.rsid, item.position.contig, item.position.position),
            ("rs199472944", "chr7", 150951552),
        )

    def test_a_version_named_by_its_cultivar_is_kept_by_the_alias_the_paper_writes(
        self,
    ):
        # C3's Methods line names the rice version two ways, the model put the
        # cultivar's in name and IRGSP-1.0 in aliases; lane P now carries the aliases
        # and lane Rf tries them, so the target is kept as L7 kept it.
        methods = (
            "High-quality reads were then mapped to the reference genome Nipponbare 1.0 "
            "(IRGSP-1.0, release 52) using bwa-mem (v0.7.17) with default parameters24."
        )
        text, paper = paper_of(
            "CHROM\t4\t6",
            "Position\t22,375,729\t20,640,731",
            "Reference\tG\tC",
            "ALT\tT\tT",
            methods,
        )
        answer = read_as(
            text,
            paper,
            [
                stated("contig", "4", "t1:r1", column="4"),
                stated("position", "22,375,729", "t1:r2", column="4", number=22375729),
                stated("ref", "G", "t1:r3", column="4"),
                stated("alt", "T", "t1:r4", column="4"),
            ],
            reference="Nipponbare 1.0",
            aliases=("IRGSP-1.0",),
        )
        targets = self.written(answer)
        rice = "GCF_001433935.1"
        report = f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{rice}/dataset_report"
        answers = {
            report: Answer(
                200, dataset_report(rice, "IRGSP-1.0", None, "GCA_001433935.1", 39947)
            )
        }

        code, record = self.placed(targets, rice, answers)

        self.assertEqual(
            json.loads(targets.read_text())["reference"]["aliases"], ["IRGSP-1.0"]
        )
        self.assertEqual(code, 0, record)
        self.assertEqual(self.steps(record), [("t1", "kept", "same_version")])
        request, _, _ = self.assertion_on(rice)
        [item] = request.items
        # The paper's own version stays stated; the alias that matched is typed.
        self.assertEqual(
            (
                item.position.contig,
                item.position.position,
                item.stated_assembly.name,
                item.placement.step,
                item.placement.matched_name,
            ),
            ("4", 22375729, "Nipponbare 1.0", "same_version", "IRGSP-1.0"),
        )

    def test_a_version_named_inside_an_alias_phrase_is_kept_with_no_listing(self):
        # C3 run 1's first try, name "Nipponbare 1.0" and alias "IRGSP-1.0, release 52"
        # as the paper writes it, is accepted (LF1) and carried; lane Rf reads the
        # version name inside the alias by lane P's parser.
        methods = (
            "High-quality reads were then mapped to the reference genome Nipponbare 1.0 "
            "(IRGSP-1.0, release 52) using bwa-mem (v0.7.17) with default parameters24."
        )
        text, paper = paper_of(
            "CHROM\t4\t6",
            "Position\t22,375,729\t20,640,731",
            "Reference\tG\tC",
            "ALT\tT\tT",
            methods,
        )
        answer = read_as(
            text,
            paper,
            [
                stated("contig", "4", "t1:r1", column="4"),
                stated("position", "22,375,729", "t1:r2", column="4", number=22375729),
                stated("ref", "G", "t1:r3", column="4"),
                stated("alt", "T", "t1:r4", column="4"),
            ],
            reference="Nipponbare 1.0",
            aliases=("IRGSP-1.0, release 52",),
        )
        rice = "GCF_001433935.1"
        report = f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{rice}/dataset_report"
        answers = {
            report: Answer(
                200, dataset_report(rice, "IRGSP-1.0", None, "GCA_001433935.1", 39947)
            )
        }

        code, record = self.placed(self.written(answer), rice, answers)

        self.assertEqual(code, 0, record)
        self.assertEqual(self.steps(record), [("t1", "kept", "same_version")])
        self.assertIn(
            "IRGSP-1.0 (in its alias IRGSP-1.0, release 52 of Nipponbare 1.0)",
            record["answers"][0]["reason"],
        )
        self.assertEqual(self.opener.sent, [report])
        request, _, _ = self.assertion_on(rice)
        [item] = request.items
        self.assertEqual(
            (item.stated_assembly.name, item.placement.matched_name),
            ("Nipponbare 1.0", "IRGSP-1.0"),
        )

    def test_a_paper_gene_change_is_not_checked_until_a_model_types_it(self):
        """No code reads a paper's change text (ADR 0049; #473 retires parse_change,
        owner Q10 a); lane P's reader types none before #383, so G15 answers it not
        checked and the request file keeps it unplaced (owner Q12 a)."""
        sentence = (
            "The screen found a point mutation at PBP2 p. (Tyr2Ter) in every clone."
        )
        text, paper = paper_of(sentence, "Reads were aligned to sacCer1.")
        answer = read_as(
            text,
            paper,
            [
                stated("gene", "PBP2", "main:p2"),
                stated("change", "p. (Tyr2Ter)", "main:p2"),
            ],
            reference="sacCer1",
        )

        code, record = self.placed(self.written(answer), YEAST, yeast_answers())

        self.assertEqual(code, 0, record)
        self.assertEqual(self.steps(record), [("t1", "not_checked", "g15")])
        self.assertEqual(
            record["answers"][0]["reason"], _UNTYPED.format("p. (Tyr2Ter)")
        )
        request, _, _ = self.assertion_on(YEAST)
        [item] = request.items
        self.assertEqual(
            (item.placed, item.placement.step, item.gene_change.change),
            (None, "none", "p. (Tyr2Ter)"),
        )

    def slco1b1_assertion(self):
        """A paper's rsID that dbSNP has merged, through lane P's writer and lane Rf's
        ``place-targets`` (an rsID alone passes), as main's assertion on GRCh38."""
        text, paper = paper_of("SLCO1B1 rs60037639 was associated with myopathy.")
        answer = read_as(
            text,
            paper,
            [
                stated("gene", "SLCO1B1", "main:p2"),
                stated("rsid", "rs60037639", "main:p2"),
            ],
        )
        code, record = self.placed(self.written(answer), HUMAN, {})
        self.assertEqual(code, 0, record)
        self.assertEqual(self.steps(record), [("t1", "passed", "rsid_only")])
        request = read_investigation_request_document(str(self.request))[1]
        investigation_input, _ = input_and_assertion()
        return (
            request,
            investigation_input,
            UserAssertion.create_collection(
                asserted_by=request.asserted_by,
                requests=request.items,
                predecessor_ids=(investigation_input.input_id,),
            ),
        )

    def test_a_paper_rsid_dbsnp_merged_is_accepted_as_a_documented_rename(self):
        # The retained bodies: rs60037639 merged into rs4149056, whose record lists it.
        request, investigation_input, assertion = self.slco1b1_assertion()
        sources = SourceResponsesOnly(
            refsnp_bodies={"rs60037639": RS60037639, "rs4149056": RS4149056}
        )
        run_root = self.root / "run"
        run_root.mkdir()

        result, journey = run_attempt(
            run_root,
            SequentialAdaptiveModel(),
            sources,
            investigation_input=investigation_input,
            assertion=assertion,
        )

        self.assertEqual(
            sources.operations[:2],
            ["fetch_refsnp:rs60037639", "fetch_refsnp:rs4149056"],
        )
        self.assertNotIn(
            "invalid",
            [
                e["payload"]["status"]
                for e in read_evidence(journey)
                if e["event_type"] == "AgentOutputValidated"
            ],
        )
        self.assertIsNone(result.pending_request.question)
        [member] = result.pending_request.proposal.request_members
        self.assertEqual(
            (member.request.request_text, member.refsnp.verified_rsid),
            (request.items[0].request_text, "rs4149056"),
        )
        self.assertIn("rs60037639", member.refsnp.merged_rsids)

    def test_a_paper_rsid_moved_without_a_merge_record_returns_as_a_correction(self):
        # Derived from the retained rs4149056 record without its rs60037639 merge
        # entry: dbSNP moved the paper's rsID to a record that does not document it.
        request, investigation_input, assertion = self.slco1b1_assertion()
        current = copy.deepcopy(RS4149056_DOCUMENT)
        current["dbsnp1_merges"] = [
            m for m in current["dbsnp1_merges"] if m["merged_rsid"] != "60037639"
        ]
        sources = SourceResponsesOnly(
            refsnp_bodies={"rs60037639": RS60037639, "rs4149056": refsnp_body(current)}
        )
        model = SequentialAdaptiveModel()
        run_root = self.root / "run"
        run_root.mkdir()

        result, journey = run_attempt(
            run_root,
            model,
            sources,
            investigation_input=investigation_input,
            assertion=assertion,
        )

        refused = [
            e
            for e in read_evidence(journey)
            if e["event_type"] == "AgentOutputValidated"
            and e["payload"]["status"] == "invalid"
        ]
        reason = "requested rs60037639, retrieved rs4149056"
        # The refusal names the question Tool as its next action.
        ask = (
            "if the request means another target, ask the researcher which one "
            "with request_investigation_information"
        )
        self.assertEqual(
            [e["payload"]["errors"] for e in refused],
            # A refused output is a failed turn: the fourth in a row stops the
            # member and asks its question (one failure rule). The difference
            # is typed after the code.
            [["retrieved_target_not_requested", "rsid", reason, ask]] * 4,
        )
        self.assertIn(reason, refused[0]["payload"]["feedback"])
        self.assertIn(reason, model.visible_messages[-1])
        self.assertEqual(result.proposals, ())
        question = result.pending_request.question
        # The correction returns to the paper's own request, as lane Rf wrote it.
        self.assertEqual(question.request.request_text, request.items[0].request_text)
        self.assertEqual(question.refused_output_event_id, refused[-1]["event_id"])

    def test_a_lone_rsid_passes_to_main_with_no_contact(self):
        # A paper that writes only `RS12345`.
        text, paper = paper_of("Carriers of RS12345 had the trait.")
        answer = read_as(text, paper, [stated("rsid", "RS12345", "main:p2")])

        code, record = self.placed(self.written(answer), HUMAN, {})

        self.assertEqual(code, 0, record)
        self.assertEqual(self.steps(record), [("t1", "passed", "rsid_only")])
        self.assertEqual(self.opener.sent, [])
        request, _, _ = self.assertion_on(HUMAN)
        [item] = request.items
        self.assertEqual((item.rsid, item.position), ("rs12345", None))


class PaperTargetsRecordTests(unittest.TestCase):
    """Paper reading writes and placement reads one model of the record
    (`bio_neb.target_request.paper_targets.PaperTargets`, #367); the second model placement
    once kept is retired, so no pin between two models is needed."""

    def test_the_record_reads_back_with_its_empty_fields_left_out(self):
        for answer in (recorded_answer("h1-run1"), s3_answer(), s17_answer()):
            with self.subTest(paper=answer.paper.doi):
                lean = lane_p_targets(answer).model_dump_json(exclude_none=True)

                WrittenTargets.model_validate_json(lean)

    def test_the_hand_built_test_record_is_one_paper_reading_writes(self):
        change = {"gene": "PBP2", "text": "Y2X", "transcript": None}
        built = paper_targets([paper_target("point mutations at PBP2 (Y2X)", change)])

        WrittenTargets.model_validate(built)
