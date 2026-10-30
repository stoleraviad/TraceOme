"""A paper's targets onto the Run's assembly: the product's `place` and its one request
writer (#385, #393, #367).

`python -m tools.reference_index place-targets` (the product's developer entry with the
reference-index tool's flank move behind the placement port) over a hand-built
``bio-neb.paper-targets/1`` record (written as data in `tests/gene_change_support.py`), scripted
NCBI answers (G15's, `tests/test_gene_change_flow.py`, and #393's development GETs under
`tests/fixtures/paper_targets/ncbi/`) and, for a flank move, #332's synthetic pair stored as the
Run's index. Each expected answer follows from how the inputs were built; every written ``/3``
file is read back by the Run's own reader.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from bio_neb.content import canonical_json
from bio_neb.failure import run_failure_tuning
from bio_neb.investigation.investigation_request import (
    read_investigation_request_document,
)
from bio_neb.providers.ncbi_datasets import UrllibNcbiTransport
from bio_neb.providers.ncbi_investigation import NcbiInvestigationSourceAdapter
from bio_neb.target_request.ncbi_sources import NcbiSources
from bio_neb.target_request.paper_targets import candidate_id, reference_key
from bio_neb.target_request.placement import (
    PlacementRefused,
    PlacementRun,
    PlacementSources,
    Resume,
    TargetsFile,
    place,
    read_stated_sequences,
)
from bio_neb.target_request.records import PaperTargetsPlaced, StatedSequences
from bio_neb.target_request.request_document import write_request_file
from tests.gene_change_support import (
    API,
    HG38_LINE,
    HORSE,
    HORSE_REPORT,
    HUMAN,
    HUMAN_REPORT,
    LINE,
    NCBI,
    NO_BACKOFF,
    NO_EQUIVALENT,
    SACCER1_TABLE,
    YEAST,
    YEAST_GFF,
    YEAST_GFF_TEXT,
    YEAST_REPORT,
    Answer,
    Opener,
    dataset_report,
    equivalents_url,
    gff_zip,
    human_answers,
    listing_url,
    names,
    names_urls,
    paper_target,
    paper_targets,
    span,
    span_url,
    yeast_answers,
    yeast_listing,
)
from tests.location_retained_bodies import GRCH38_MOLECULES
from tests.reference_index_support import (
    INSTALL,
    INSTALLED,
    SyntheticPair,
    choice_record,
)
from tools.reference_index.__main__ import main


class _PlaceCase(unittest.TestCase):
    def setUp(self):
        self.enterContext(run_failure_tuning(NO_BACKOFF))
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cache = self.root / "reference-index"
        self.variant_cache = self.root / "variant-sources"
        self.targets = self.root / "targets.json"
        self.request = self.root / "request.json"

    def place(self, document, assembly, answers, *options):
        if document is not None:
            self.targets.write_text(json.dumps(document))
        self.opener = Opener(answers)
        out, self.err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(self.err):
            code = main(
                [
                    "place-targets",
                    "--assembly",
                    assembly,
                    "--paper-targets",
                    str(self.targets),
                    "--request",
                    str(self.request),
                    "--cache",
                    str(self.cache),
                    "--variant-cache",
                    str(self.variant_cache),
                    *options,
                ],
                opener=self.opener,
                tools=INSTALL,
            )
        return code, json.loads(out.getvalue())

    def records(self):
        directory = self.variant_cache / "paper-targets" / "records"
        return sorted(directory.glob("*.json"))

    def requests(self):
        """The written file as main reads it, and its items as written."""
        _, request = read_investigation_request_document(str(self.request))
        return request, json.loads(self.request.read_text())["requests"]

    def answers(self, record):
        return [(a["label"], a["answer"]) for a in record["answers"]]


class SchemaTests(_PlaceCase):
    def test_another_schema_version_stops_the_run_before_any_contact(self):
        # An unreadable targets.json is a plan-level stop: the record is written with
        # `stopped`, and no request file.
        document = paper_targets([]) | {"schema_version": "bio-neb.paper-targets/2"}

        code, record = self.place(document, YEAST, yeast_answers())

        self.assertEqual(code, 1)
        self.assertIn("bio-neb.paper-targets/1", record["stopped"]["error"])
        self.assertEqual(
            (record["answers"], record["asserted_by"], record["schema_version"]),
            ([], None, "bio-neb.paper-targets-placed/6"),
        )
        self.assertEqual(self.opener.sent, [])
        self.assertFalse(self.request.exists())
        self.assertEqual(len(self.records()), 1)

    def test_a_targets_file_that_cannot_be_read_stops_the_run_with_a_record(self):
        # Spec r1 F1: the file itself, not only its content, is the run's plan-level input.
        self.targets.unlink(missing_ok=True)

        code, record = self.place(None, YEAST, {})

        self.assertEqual(code, 1)
        self.assertIn("FileNotFoundError", record["stopped"]["error"])
        self.assertEqual(
            (record["answers"], record["paper_targets_sha256"], record["asserted_by"]),
            ([], None, None),
        )
        self.assertEqual(self.opener.sent, [])
        self.assertFalse(self.request.exists())
        self.assertEqual(len(self.records()), 1)

    def test_the_writer_uses_the_bytes_place_read_even_if_the_file_changes(self):
        # Spec r2 F3: one reading of the targets serves both place and the request writer.
        document = paper_targets(
            [
                paper_target(
                    "rs4 at chr7:150951552",
                    rsid="rs4",
                    position={"contig": "7", "position": 150951552},
                    stated={"name": "hg38", "cited_line": HG38_LINE},
                )
            ]
        )
        changed = paper_targets(
            [
                paper_target(
                    "rs5 at chr7:42",
                    rsid="rs5",
                    position={"contig": "7", "position": 42},
                    stated={"name": "hg38", "cited_line": HG38_LINE},
                )
            ]
        )
        targets = self.targets

        class Rewriting(Answer):
            def read(self, limit=-1):
                targets.write_text(json.dumps(changed))  # during the report's contact
                return super().read(limit)

        report = human_answers()[HUMAN_REPORT]
        self.targets.write_text(json.dumps(document))
        code, record = self.place(
            None, HUMAN, {HUMAN_REPORT: Rewriting(report.status, report.body)}
        )

        self.assertEqual(code, 0, record)
        request, _ = self.requests()
        [item] = request.items
        self.assertEqual((item.rsid, item.position.position), ("rs4", 150951552))
        self.assertTrue(item.request_text.startswith("rs4 at chr7:150951552"))

    def test_the_writer_refuses_targets_other_than_the_records(self):
        # Standards r2 S2: the adapter writes only the input place resolved.
        document = paper_targets([paper_target("rs4 here", rsid="rs4")])
        code, record = self.place(document, HUMAN, human_answers())
        self.assertEqual(code, 0)
        written = self.request.read_bytes()
        other = self.root / "other.json"
        other.write_text(json.dumps(paper_targets([paper_target("rs9", rsid="rs9")])))
        placed = PaperTargetsPlaced.model_validate(record)

        for targets in (TargetsFile.read(other), TargetsFile.read(self.root / "gone")):
            with self.subTest(unread=targets.unread is not None):
                with self.assertRaises(PlacementRefused) as refused:
                    write_request_file(placed, targets, self.request)
                self.assertEqual(refused.exception.reason, "targets_mismatch")
                self.assertEqual(self.request.read_bytes(), written)

    def test_an_unreadable_table_is_refused_before_any_contact(self):
        table = self.root / "table.json"
        table.write_text(json.dumps({"schema_version": "bio-neb.stated-sequences/1"}))

        code, refused = self.place(
            paper_targets([]), YEAST, {}, "--stated-sequences", str(table)
        )

        self.assertEqual((code, refused["refused"]), (2, "stated_sequences_unreadable"))
        self.assertEqual(self.opener.sent, [])
        self.assertEqual(self.records(), [])


CHICKEN5 = "GCF_000002315.4"
CHICKEN5_REPORT = f"{API}/datasets/v2/genome/accession/{CHICKEN5}/dataset_report"
CHICKEN5_ANY_VERSION = f"{CHICKEN5_REPORT}?filters.assembly_version=all_assemblies"
GALLUS5_LINE = "SNPs were called on Gallus_gallus-5.0"


def _chicken_target(position):
    return paper_target(
        f"a SNP at chr 2: {position}",
        position={"contig": "chr 2", "position": position},
        stated={"name": "Gallus_gallus-5.0", "cited_line": GALLUS5_LINE},
    )


class ReplacedRunAssemblyTests(_PlaceCase):
    """#386 live L5-F1: NCBI answers ``{}`` (2 bytes, HTTP 200; #386 live L5's recorded
    body) to the plain report of an assembly that is no longer current, here
    Gallus_gallus-5.0, and describes it only with ``filters.assembly_version=all_assemblies``
    (as ``tools/reference_index/datasets.py`` ``Datasets.describe`` reads it), with the
    status NCBI gave it there (``suppressed``, #386 live L5a2's recorded any-version
    body)."""

    def test_a_run_on_a_replaced_assembly_reads_its_any_version_report(self):
        answers = {
            CHICKEN5_REPORT: Answer(200, b"{}"),
            CHICKEN5_ANY_VERSION: Answer(
                200,
                dataset_report(
                    CHICKEN5,
                    "Gallus_gallus-5.0",
                    "galGal5",
                    "GCA_000002315.5",
                    9031,
                    status="suppressed",
                ),
            ),
        }

        code, record = self.place(
            paper_targets([_chicken_target(86770373), _chicken_target(86870271)]),
            CHICKEN5,
            answers,
        )

        self.assertEqual(code, 0, record)
        self.assertEqual(
            [(a["answer"], a["step"]) for a in record["answers"]],
            [("kept", "same_version")] * 2,
        )
        # One read of the Run's report per run, shared by both targets.
        self.assertEqual(self.opener.sent, [CHICKEN5_REPORT, CHICKEN5_ANY_VERSION])

    def test_an_unreadable_any_version_report_is_each_targets_failure(self):
        answers = {
            CHICKEN5_REPORT: Answer(200, b"{}"),
            CHICKEN5_ANY_VERSION: Answer(200, b"{}"),
        }

        code, record = self.place(
            paper_targets([_chicken_target(86770373), _chicken_target(86870271)]),
            CHICKEN5,
            answers,
        )

        self.assertEqual(code, 2, record)
        self.assertEqual(record["waiting"], ["t1", "t2"])
        self.assertIsNone(record["request_file"])
        self.assertEqual(
            {
                (a["failure"]["tool"], a["failure"]["failure_class"])
                for a in record["answers"]
            },
            {("fetch_assembly_report", "semantic")},
        )
        self.assertEqual(self.opener.sent, [CHICKEN5_REPORT, CHICKEN5_ANY_VERSION])

    def test_a_rejected_report_keeps_its_rejection_whatever_its_body(self):
        # Only a successful empty report means "not current"; a 404 is the source's
        # rejection, with no second GET.
        answers = {
            CHICKEN5_REPORT: Answer(404, b"{}"),
            CHICKEN5_ANY_VERSION: Answer(
                200,
                dataset_report(
                    CHICKEN5,
                    "Gallus_gallus-5.0",
                    "galGal5",
                    "GCA_000002315.5",
                    9031,
                    status="suppressed",
                ),
            ),
        }

        code, record = self.place(
            paper_targets([_chicken_target(86770373), _chicken_target(86870271)]),
            CHICKEN5,
            answers,
        )

        self.assertEqual(code, 2, record)
        self.assertEqual(record["waiting"], ["t1", "t2"])
        self.assertIsNone(record["request_file"])
        self.assertEqual(
            {
                (
                    a["failure"]["tool"],
                    a["failure"]["failure_class"],
                    a["failure"]["detail"],
                )
                for a in record["answers"]
            },
            {("fetch_assembly_report", "rejected", "NCBI answered 404.")},
        )
        self.assertEqual(self.opener.sent, [CHICKEN5_REPORT])

    def test_a_report_that_is_not_the_empty_answer_keeps_its_own_failure(self):
        answers = {CHICKEN5_REPORT: Answer(200, b'{"reports": []}')}

        code, record = self.place(
            paper_targets([_chicken_target(86770373)]), CHICKEN5, answers
        )

        self.assertEqual(code, 2, record)
        self.assertEqual(record["answers"][0]["failure"]["failure_class"], "semantic")
        self.assertEqual(self.opener.sent, [CHICKEN5_REPORT])


class SameVersionTests(_PlaceCase):
    def test_a_position_stated_on_the_runs_version_keeps_the_papers_version(self):
        text = "KCNH2 chr7:150951552 c.1841C>T rs199472944"
        document = paper_targets(
            [
                paper_target(
                    text,
                    rsid="rs199472944",
                    position={"contig": "7", "position": 150951552},
                    stated={"name": "hg38", "cited_line": HG38_LINE},
                )
            ]
        )

        code, record = self.place(document, HUMAN, human_answers())

        self.assertEqual(code, 0)
        self.assertEqual(self.answers(record), [("t1", "kept")])
        request, written = self.requests()
        [item] = request.items
        self.assertEqual(
            (item.rsid, item.position.contig, item.position.position),
            ("rs199472944", "7", 150951552),
        )
        self.assertIsNone(item.position.ref)
        # The paper's own version is kept as stated: never the Run's accession.
        self.assertEqual(
            (item.stated_assembly.name, item.stated_assembly.cited_line),
            ("hg38", HG38_LINE),
        )
        self.assertEqual(
            (
                item.placement.step,
                item.placement.matched_name,
                item.placed.model_dump(),
            ),
            ("same_version", "hg38", item.position.model_dump()),
        )
        self.assertTrue(item.request_text.startswith(text))
        self.assertEqual(request.asserted_by, document["asserted_by"])
        self.assertEqual(record["answers"][0]["requests"], [0])
        self.assertEqual(
            (record["answers"][0]["step"], record["answers"][0]["position"]),
            (
                "same_version",
                {"contig": "7", "position": 150951552, "ref": None, "alt": None},
            ),
        )
        self.assertEqual(self.opener.sent, [HUMAN_REPORT])

    def test_a_version_the_paper_also_names_by_an_alias_is_kept_by_that_alias(self):
        # c3 stated "Nipponbare 1.0" with alias "IRGSP-1.0" (both written in its Methods
        # line); the Run's assembly name is IRGSP-1.0, so the five targets went
        # not_checked. Placement tries the paper's name, then each alias.
        rice = "GCF_001433935.1"
        line = (
            "High-quality reads were then mapped to the reference genome Nipponbare 1.0 "
            "(IRGSP-1.0, release 52) using bwa-mem (v0.7.17) with default parameters24."
        )
        reference = {
            "candidate": {
                "candidate_id": candidate_id("r", reference_key("Nipponbare 1.0")),
                "name": "Nipponbare 1.0",
                "spellings": ["Nipponbare 1.0"],
                "sources": [
                    {
                        "document_id": "main",
                        "section": "Methods",
                        "start": 0,
                        "end": len(line),
                        "text": line,
                    }
                ],
            },
            "reason": "the reads were mapped to it",
            "aliases": ["IRGSP-1.0"],
        }
        target = paper_target(
            "22,375,729",
            position={"contig": "4", "position": 22375729, "ref": "G", "alt": "T"},
            stated={"name": "Nipponbare 1.0", "cited_line": line},
        )
        report = f"{API}/datasets/v2/genome/accession/{rice}/dataset_report"
        answers = {
            report: Answer(
                200, dataset_report(rice, "IRGSP-1.0", None, "GCA_001433935.1", 39947)
            )
        }

        code, record = self.place(paper_targets([target], reference), rice, answers)

        self.assertEqual(code, 0, self.err.getvalue())
        [answer] = record["answers"]
        self.assertEqual((answer["answer"], answer["step"]), ("kept", "same_version"))
        self.assertIn(
            "the paper's IRGSP-1.0 (its alias of Nipponbare 1.0)", answer["reason"]
        )
        # The written name that matched is one typed field.
        request, _ = self.requests()
        [item] = request.items
        self.assertEqual(
            (item.stated_assembly.name, item.placement.matched_name),
            ("Nipponbare 1.0", "IRGSP-1.0"),
        )
        self.assertEqual(self.opener.sent, [report])


class RunPlanTests(_PlaceCase):
    """What the record says about a run's own source contacts. Outside a Run nothing
    records contacts (owner, #417): the record counts this run's GETs and names no total
    (the retired ``--max-contacts`` monitored total and its per-contact log)."""

    def test_the_record_counts_this_runs_gets_and_names_no_total(self):
        document = paper_targets([paper_target("rs4 here", rsid="rs4"), _pbp2()])
        self.place(document, YEAST, yeast_answers())

        # A second run on the same caches: its count is its own GETs, not the first's.
        code, record = self.place(document, YEAST, yeast_answers())

        self.assertEqual(code, 0, record)
        self.assertEqual(
            self.opener.sent, [YEAST_REPORT, span_url("NC_001134.8", 1004, 3)]
        )
        self.assertEqual(
            (record["contact_count"], record["failure_count"]),
            (len(self.opener.sent), 0),
        )
        self.assertFalse({"contacts", "max_contacts"} & record.keys())


class RequestTextTests(_PlaceCase):
    """Main takes a request line of at most 4,000 characters (`InvestigationRequestItem`)."""

    def test_a_longest_paper_line_keeps_lane_rfs_answer_and_reason(self):
        text = "a deletion " + "x" * 3989
        document = paper_targets([paper_target(text)])

        self.place(document, HUMAN, human_answers())

        request, _ = self.requests()
        [item] = request.items
        self.assertLessEqual(len(item.request_text), 4000)
        self.assertTrue(
            item.request_text.endswith(
                "(placement: not_checked: the paper states no rsID, position or gene "
                "change for it)"
            ),
            item.request_text[-200:],
        )
        self.assertTrue(item.request_text.startswith("a deletion xxx"))

    def test_a_cut_paper_line_still_states_its_rsid(self):
        text = "y" * 3990 + " rs12345"
        document = paper_targets(
            [
                paper_target(
                    text,
                    rsid="rs12345",
                    position={"contig": "III", "position": 1000},
                    stated={"name": "ce11", "cited_line": "aligned to ce11"},
                )
            ]
        )

        code, record = self.place(
            document, YEAST, yeast_answers(**yeast_listing("ce11"))
        )

        self.assertEqual((code, record["answers"][0]["answer"]), (0, "passed"))
        request, _ = self.requests()
        [item] = request.items
        self.assertEqual(item.rsid, "rs12345")
        self.assertLessEqual(len(item.request_text), 4000)
        self.assertIn(
            "(placement: passed: its position is not used: ce11", item.request_text
        )
        self.assertIn("the paper's line states rs12345", item.request_text)


class PassedAndUnplacedTests(_PlaceCase):
    def test_an_rsid_with_nothing_else_is_passed_for_main_to_look_up(self):
        text = "carriers of rs429358 had higher risk"
        document = paper_targets([paper_target(text, rsid="rs429358")])

        code, record = self.place(document, HUMAN, human_answers())

        self.assertEqual(code, 0)
        self.assertEqual(self.answers(record), [("t1", "passed")])
        request, written = self.requests()
        [item] = request.items
        self.assertEqual(
            (item.rsid, item.position, item.stated_assembly), ("rs429358", None, None)
        )
        self.assertEqual(written[0]["target"], {"rsid": "rs429358"})
        self.assertEqual((item.request_text, item.placement), (text, None))

    def test_a_target_stating_nothing_placement_reads_is_written_unplaced(self):
        text = "a deletion upstream of the promoter"
        document = paper_targets([paper_target(text)])

        code, record = self.place(document, HUMAN, human_answers())

        self.assertEqual(code, 0)
        [answer] = record["answers"]
        self.assertEqual(
            (answer["answer"], answer["reason"]),
            ("not_checked", "the paper states no rsID, position or gene change for it"),
        )
        request, written = self.requests()
        [item] = request.items
        self.assertEqual(
            (item.rsid, item.position, item.stated_assembly), (None, None, None)
        )
        self.assertNotIn("target", written[0])
        self.assertEqual(
            (item.placement.step, item.placement.reason, item.placed),
            (
                "none",
                "not_checked: the paper states no rsID, position or gene change for it",
                None,
            ),
        )
        self.assertEqual(
            item.request_text,
            f"{text} (placement: not_checked: the paper states no rsID, position or gene "
            "change for it)",
        )

    def test_a_position_on_a_version_placement_cannot_name_is_written_unplaced(self):
        text = "chrIII:1000 G>A"
        document = paper_targets(
            [
                paper_target(
                    text,
                    position={
                        "contig": "III",
                        "position": 1000,
                        "ref": "G",
                        "alt": "A",
                    },
                    stated={"name": "ce11", "cited_line": "aligned to ce11"},
                )
            ]
        )

        code, record = self.place(
            document, YEAST, yeast_answers(**yeast_listing("ce11"))
        )

        self.assertEqual(code, 0)
        [answer] = record["answers"]
        self.assertEqual((answer["answer"], answer["step"]), ("not_checked", "none"))
        # NCBI's search lists no yeast assembly literally named ce11, and placement
        # with no model names none by any other spelling.
        self.assertEqual(
            answer["reason"],
            "ce11 is not a name NCBI's report gives the Run's R64 (GCF_000146045.2); "
            "table: no table row (the Run gives no table) for ce11 III; NCBI: NCBI's "
            "search lists no assembly of taxon 559292 literally named ce11",
        )
        self.assertIsNone(answer["version"]["matched"])
        self.assertEqual(self.opener.sent, [YEAST_REPORT, listing_url(559292, "ce11")])
        request, _ = self.requests()
        [item] = request.items
        # The paper's position stays stated on ce11; nothing places it on the Run's.
        self.assertEqual(
            (item.position.contig, item.stated_assembly.name, item.rsid, item.placed),
            ("III", "ce11", None, None),
        )
        self.assertEqual(
            (item.placement.step, item.placement.reason),
            ("none", f"not_checked: {answer['reason']}"),
        )

    def test_a_position_that_cannot_be_moved_passes_its_rsid_instead(self):
        text = "rs1 at chrIII:1000"
        document = paper_targets(
            [
                paper_target(
                    text,
                    rsid="rs1",
                    position={"contig": "III", "position": 1000},
                    stated={"name": "ce11", "cited_line": "aligned to ce11"},
                )
            ]
        )

        code, record = self.place(
            document, YEAST, yeast_answers(**yeast_listing("ce11"))
        )

        self.assertEqual(code, 0)
        [answer] = record["answers"]
        self.assertEqual((answer["answer"], answer["step"]), ("passed", "rsid_only"))
        self.assertTrue(
            answer["reason"].startswith(
                "its position is not used: ce11 is not a name NCBI's report gives"
            ),
            answer["reason"],
        )
        request, _ = self.requests()
        [item] = request.items
        self.assertEqual(
            (item.rsid, item.position.position, item.placement.step, item.placed),
            ("rs1", 1000, "none", None),
        )

    def test_a_position_without_a_stated_version_is_not_checked(self):
        document = paper_targets(
            [
                paper_target(
                    "chr2:5 A>G",
                    position={"contig": "2", "position": 5, "ref": "A", "alt": "G"},
                )
            ],
            reference=None,
        )

        code, record = self.place(document, YEAST, yeast_answers())

        self.assertEqual(code, 0)
        [answer] = record["answers"]
        self.assertEqual(
            (answer["answer"], answer["reason"]),
            ("not_checked", "the paper states no genome version for its position"),
        )


Y2X = {"ref": "Y", "codon": 2, "alt": "*"}


def _pbp2(change="Y2X", gene="PBP2", protein=Y2X, coding=None):
    """A paper's gene change with the parts a model typed from it (#473 Q2 a); lane P's
    reader types none before #383 (owner Q12 a)."""
    return paper_target(
        f"point mutations at {gene} ({change})",
        {
            "gene": gene,
            "text": change,
            "transcript": None,
            "protein": protein,
            "coding": coding,
        },
    )


class GeneChangeTests(_PlaceCase):
    """G15's scripted yeast gene (`yeast_answers`): PBP2's codon 2 is TAC at NC_001134.8:1004."""

    def test_a_gene_and_change_with_no_position_is_placed_by_g15(self):
        code, record = self.place(paper_targets([_pbp2()]), YEAST, yeast_answers())

        self.assertEqual(code, 0)
        [answer] = record["answers"]
        self.assertEqual((answer["label"], answer["answer"]), ("t1", "placed"))
        self.assertEqual(answer["gene_change"]["outcome"], "candidates")
        self.assertEqual(answer["requests"], [0, 1, 2])
        request, _ = self.requests()
        # Each candidate is a placed position; the paper states only its gene and change.
        self.assertEqual(
            [
                (i.placed.contig, i.placed.position, i.placed.ref, i.placed.alt)
                for i in request.items
            ],
            [
                ("NC_001134.8", 1006, "C", "A"),
                ("NC_001134.8", 1006, "C", "G"),
                ("NC_001134.8", 1005, "AC", "GA"),
            ],
        )
        self.assertEqual(
            {
                (
                    i.position,
                    i.gene_change.gene,
                    i.gene_change.change,
                    i.placement.step,
                    i.placement.matched_name,
                )
                for i in request.items
            },
            {(None, "PBP2", "Y2X", "gene_change", None)},
        )
        self.assertEqual(
            {
                (i.stated_assembly.name, i.stated_assembly.cited_line)
                for i in request.items
            },
            {("sacCer1", LINE)},
        )

    def test_a_reference_line_past_mains_limit_is_cut_around_the_version_name(self):
        # Lane P cites the paper's line as written; main reads at most 4,000.
        methods = "x " * 2100 + LINE + " y" * 2100
        reference = {
            "candidate": {
                "candidate_id": "r-b8a9216d7b93",
                "name": "sacCer1",
                "spellings": ["sacCer1"],
                "sources": [
                    {
                        "document_id": "main",
                        "section": "Methods",
                        "start": 0,
                        "end": len(methods),
                        "text": methods,
                    }
                ],
            },
            "reason": "the Methods state it",
        }

        code, record = self.place(
            paper_targets([_pbp2()], reference), YEAST, yeast_answers()
        )

        self.assertEqual(code, 0, record)
        self.assertEqual(record["answers"][0]["answer"], "placed")
        request, _ = self.requests()
        [cited] = {item.stated_assembly.cited_line for item in request.items}
        self.assertLessEqual(len(cited), 4_000)
        self.assertIn(LINE, cited)

    def test_an_untyped_paper_change_is_not_checked_until_a_model_types_it(self):
        """No code reads a paper's change text (ADR 0049); no reader types it before
        #383, so G15 answers it not checked (#473 owner Q12 a)."""
        code, record = self.place(
            paper_targets([_pbp2(protein=None), _pbp2()]), YEAST, yeast_answers()
        )

        self.assertEqual(code, 0, record)
        self.assertEqual(
            self.answers(record), [("t1", "not_checked"), ("t2", "placed")]
        )
        self.assertEqual(
            record["answers"][0]["reason"],
            "the change 'Y2X' is not typed: the model types its protein or coding "
            "change from the words that state it",
        )

    def test_a_change_whose_numbers_disagree_is_not_checked_and_the_others_go_on(self):
        # Migrated from the retired `variant_sources targets` command: c.4T>A is codon
        # 2's first base, never K3X's codon 3; the typed change is refused.
        code, record = self.place(
            paper_targets(
                [
                    _pbp2(
                        change="c.4T>A K3X",
                        protein={"ref": "K", "codon": 3, "alt": "*"},
                        coding={"cds_position": 4, "ref": "T", "alt": "A"},
                    ),
                    _pbp2(),
                ]
            ),
            YEAST,
            yeast_answers(),
        )

        self.assertEqual(code, 0, record)
        self.assertEqual(
            self.answers(record), [("t1", "not_checked"), ("t2", "placed")]
        )
        request, _ = self.requests()
        self.assertEqual(
            [i.placement.step for i in request.items],
            ["none", "gene_change", "gene_change", "gene_change"],
        )

    def test_a_symbol_naming_two_genes_is_ambiguous_and_the_others_go_on(self):
        # Migrated from the retired `variant_sources targets` command.
        gff = YEAST_GFF_TEXT + (
            "NC_001134.8\tRefSeq\tgene\t5001\t5030\t.\t+\t.\tID=gene-YB2;Name=DUP;"
            "gene=DUP\n"
            "NC_001134.8\tRefSeq\tgene\t6001\t6030\t.\t+\t.\tID=gene-YB3;Name=DUP;"
            "gene=DUP\n"
        )

        code, record = self.place(
            paper_targets([_pbp2(gene="DUP"), _pbp2()]),
            YEAST,
            yeast_answers(**{YEAST_GFF: Answer(200, gff_zip(gff))}),
        )

        self.assertEqual(code, 0, record)
        self.assertIsNone(record["stopped"])
        self.assertEqual(self.answers(record), [("t1", "ambiguous"), ("t2", "placed")])
        request, _ = self.requests()
        # No candidate for DUP: it is written unplaced, with G15's reason.
        self.assertEqual(
            (request.items[0].gene_change.gene, request.items[0].placement.step),
            ("DUP", "none"),
        )

    def test_a_request_text_cut_at_mains_limit_is_one_canonical_line(self):
        # Either parity of the cut lands once on a space between two words.
        for pad in ("", "b"):
            with self.subTest(pad=pad):
                text = pad + " ".join(["a"] * 1950)
                target = paper_target(
                    text,
                    {"gene": "PBP2", "text": "Y2X", "transcript": None, "protein": Y2X},
                )

                code, record = self.place(
                    paper_targets([target]), YEAST, yeast_answers()
                )

                self.assertEqual(code, 0, record)
                _, items = self.requests()
                self.assertTrue(all(len(item["request"]) <= 4_000 for item in items))

    def test_g15_reads_the_targets_gene_change_transcript_and_the_papers_line(self):
        upf3 = paper_target(
            "point mutations at UPF3 (G6W)",
            {
                "gene": "UPF3",
                "text": "G6W",
                "transcript": "NM_1.2",
                "protein": {"ref": "G", "codon": 6, "alt": "W"},
            },
        )

        _, record = self.place(paper_targets([_pbp2(), upf3]), YEAST, yeast_answers())

        self.assertEqual(
            [
                (
                    c["label"],
                    c["change"]["gene"],
                    c["change"]["change"],
                    c["change"].get("transcript"),
                    c["request_text"],
                )
                for c in (a["gene_change"]["claim"] for a in record["answers"])
            ],
            [
                ("t1-PBP2", "PBP2", "Y2X", None, "point mutations at PBP2 (Y2X)"),
                ("t2-UPF3", "UPF3", "G6W", "NM_1.2", "point mutations at UPF3 (G6W)"),
            ],
        )
        self.assertEqual(
            {
                (stated["name"], stated["cited_line"])
                for stated in (
                    a["gene_change"]["claim"]["stated_assembly"]
                    for a in record["answers"]
                )
            },
            {("sacCer1", LINE)},
        )

    def test_a_paper_stating_no_version_places_with_no_stated_version(self):
        document = paper_targets([_pbp2()], reference=None)

        code, record = self.place(document, YEAST, yeast_answers())

        self.assertEqual((code, record["answers"][0]["answer"]), (0, "placed"))
        request, written = self.requests()
        self.assertEqual({i.stated_assembly for i in request.items}, {None})
        self.assertNotIn("stated_assembly", written[0])

    def test_a_placed_targets_rsid_is_kept_on_every_candidate(self):
        target = paper_target(
            "PBP2 (Y2X), rs7",
            {"gene": "PBP2", "text": "Y2X", "transcript": None, "protein": Y2X},
            rsid="rs7",
        )

        code, record = self.place(paper_targets([target]), YEAST, yeast_answers())

        self.assertEqual((code, record["answers"][0]["answer"]), (0, "placed"))
        request, _ = self.requests()
        self.assertEqual(
            [(i.rsid, i.placed.position) for i in request.items],
            [("rs7", 1006), ("rs7", 1006), ("rs7", 1005)],
        )

    def test_an_ambiguous_targets_rsid_is_kept_on_every_candidate(self):
        # A second PBP2 transcript whose codon 2 is TAT at 1007-1009 places Y2X apart.
        gff = YEAST_GFF_TEXT + (
            "NC_001134.8\tRefSeq\tmRNA\t1004\t1033\t.\t+\t.\tID=rna-NM_2.1;"
            "Parent=gene-YB1;gene=PBP2;transcript_id=NM_2.1\n"
            "NC_001134.8\tRefSeq\tCDS\t1004\t1033\t.\t+\t0\tID=cds-NP_2.1;"
            "Parent=rna-NM_2.1;gene=PBP2;protein_id=NP_2.1\n"
        )
        answers = yeast_answers(
            **{
                YEAST_GFF: Answer(200, gff_zip(gff)),
                span_url("NC_001134.8", 1007, 3): Answer(
                    200, span("NC_001134.8", 1007, "TAT")
                ),
            }
        )
        target = paper_target(
            "PBP2 (Y2X), rs7",
            {"gene": "PBP2", "text": "Y2X", "transcript": None, "protein": Y2X},
            rsid="rs7",
        )

        code, record = self.place(paper_targets([target]), YEAST, answers)

        self.assertEqual((code, record["answers"][0]["answer"]), (0, "ambiguous"))
        request, _ = self.requests()
        self.assertGreater(len(request.items), 3)
        self.assertEqual({i.rsid for i in request.items}, {"rs7"})
        self.assertLessEqual(
            {1005, 1006, 1009}, {i.placed.position for i in request.items}
        )

    def test_a_gene_g15_does_not_find_is_written_unplaced(self):
        code, record = self.place(
            paper_targets([_pbp2(gene="NOPE")]), YEAST, yeast_answers()
        )

        self.assertEqual(code, 0)
        [answer] = record["answers"]
        self.assertEqual(
            (answer["answer"], answer["reason"]),
            ("not_found", "the assembly's annotation names no gene NOPE"),
        )
        request, written = self.requests()
        self.assertEqual(
            written[0]["target"], {"gene_change": {"gene": "NOPE", "change": "Y2X"}}
        )
        self.assertEqual(
            (request.items[0].placement.step, request.items[0].placed),
            ("none", None),
        )
        self.assertIn("NOPE (Y2X)", request.items[0].request_text)

    def test_a_contact_that_asks_waits_for_you_and_the_other_targets_are_answered(self):
        # Each target fails on its own (Q18).
        answers = yeast_answers(**{YEAST_GFF: [Answer(503) for _ in range(3)]})
        document = paper_targets([_pbp2(), paper_target("rs5 here", rsid="rs5")])

        code, record = self.place(document, YEAST, answers)

        self.assertEqual(code, 2)
        failed, passed = record["answers"]
        self.assertEqual(
            (failed["answer"], failed["step"], failed["failure"]["tool"]),
            (None, "g15", "ncbi_genome_annotation"),
        )
        self.assertEqual(passed["answer"], "passed")
        self.assertEqual(
            (record["waiting"], record["failure_count"], record["stopped"]),
            (["t1"], 1, None),
        )
        self.assertIsNone(record["request_file"])
        self.assertFalse(self.request.exists())
        self.assertIn(
            "t1 asks: exhausted after 3 attempts on ncbi_genome_annotation",
            self.err.getvalue(),
        )
        self.assertIn("--retry LABEL or --accept LABEL", self.err.getvalue())

    def test_the_runs_assembly_report_failing_is_each_dependent_targets_failure(self):
        answers = {YEAST_REPORT: [Answer(503) for _ in range(3)]}
        document = paper_targets([_pbp2(), paper_target("rs5 here", rsid="rs5")])

        code, record = self.place(document, YEAST, answers)

        self.assertEqual(code, 2)
        self.assertEqual(
            [
                (a["label"], a["answer"], (a["failure"] or {}).get("tool"))
                for a in record["answers"]
            ],
            [("t1", None, "fetch_assembly_report"), ("t2", "passed", None)],
        )
        self.assertEqual(self.opener.sent, [YEAST_REPORT] * 3)
        self.assertFalse(self.request.exists())


SACCER1 = {"name": "sacCer1", "cited_line": LINE}


@unittest.skipUnless(INSTALLED, "needs #319's bwa and samtools install")
class MoveTests(_PlaceCase):
    """#332's synthetic pair: A's `chrA` is the cached sacCer1-era `NC_001143.4` (chromosome
    11 by the fixture table's row), B the Run's stored index, which a choice record names for
    R64. Each segment's place on B follows from how B was built (`SyntheticPair`). The
    Variation service covers no yeast sequence (#393's development GET d6)."""

    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.pair = SyntheticPair(Path(directory.name))

    def setUp(self):
        super().setUp()
        sequence = self.cache / "ncbi" / "sequences" / "NC_001143.4.fa"
        sequence.parent.mkdir(parents=True)
        sequence.write_text(
            self.pair.source.read_text().replace(">chrA", ">NC_001143.4", 1)
        )
        self.digest = hashlib.sha256(self.pair.target.read_bytes()).hexdigest()
        store = self.cache / "store" / "sha256" / self.digest
        store.mkdir(parents=True)
        for suffix in ("", ".amb", ".ann", ".bwt", ".pac", ".sa", ".fai"):
            shutil.copy(f"{self.pair.target}{suffix}", store / f"reference.fa{suffix}")
        self.choose(YEAST)

    def choose(self, target):
        choice = choice_record("GCF_900000001.1", target, self.digest, [])
        path = self.cache / "choices" / "GCF_900000001.1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(choice.model_dump(mode="json")))

    def segment(self, k, rsid=None, **position):
        text = f"{rsid or 'a change'} at chr11:{self.pair.positions[k]}"
        stated = {
            "contig": "11",
            "position": self.pair.positions[k],
            "ref": self.pair.refs[k],
            "alt": self.pair.alts[k],
        } | position
        return paper_target(text, position=stated, stated=SACCER1, rsid=rsid)

    def placed(self, *targets):
        equivalents = {
            equivalents_url(
                f"NC_001143.4:{position['position'] - 1}:{position['ref']}:"
                f"{position['alt']}"
            ): NO_EQUIVALENT
            for position in (target["request"]["position"] for target in targets)
            if position.get("ref")
        }
        code, record = self.place(
            paper_targets(list(targets)),
            YEAST,
            yeast_answers(**equivalents),
            "--stated-sequences",
            str(SACCER1_TABLE),
        )
        self.assertEqual(code, 0, record)
        return record

    def test_the_tables_row_answers_only_the_contig_as_it_writes_it(self):
        """No code equates contig spellings (ADR 0049): a table row matches its contig
        exactly (owner #467 R4 a; #459/#414 replace the table). Its rows are `11` and
        the paper's own `chr11`."""
        record = self.placed(self.segment(0, contig="11"))
        [answer] = record["answers"]
        self.assertEqual((answer["answer"], answer["step"]), ("moved", "table"))
        table = StatedSequences.model_validate_json(SACCER1_TABLE.read_bytes())
        self.assertEqual(table.row("SACCER1", "11").accession, "NC_001143.4")
        # The paper's own `chr11` has its row as written; no other spelling matches.
        self.assertEqual(table.row("sacCer1", "chr11").accession, "NC_001143.4")
        for contig in ("Chr11", "CHR11", "chr 11", "chromosome 11"):
            with self.subTest(contig=contig):
                self.assertIsNone(table.row("sacCer1", contig))

    def test_a_position_on_another_version_is_moved_onto_the_runs_assembly(self):
        record = self.placed(self.segment(0, rsid="rs7"))

        [answer] = record["answers"]
        self.assertEqual((answer["answer"], answer["step"]), ("moved", "table"))
        self.assertEqual(
            (answer["move"]["result"], answer["move"]["place"]),
            ("unique", {"contig": "same", "position": 899}),
        )
        self.assertEqual(
            answer["stated_sequence"],
            {
                "accession": "NC_001143.4",
                "named_by": "table",
                "assembly_accession": None,
            },
        )
        table = SACCER1_TABLE.read_bytes()
        self.assertEqual(
            record["table"],
            {
                "name": "s3 sacCer1 (test fixture, #332's proof)",
                "sha256": hashlib.sha256(table).hexdigest(),
            },
        )
        request, _ = self.requests()
        [item] = request.items
        # The placed position is the Run's; the paper's stays stated on sacCer1.
        self.assertEqual(
            (item.rsid, item.placed.contig, item.placed.position),
            ("rs7", "same", 899),
        )
        self.assertEqual(
            (item.placed.ref, item.placed.alt),
            (self.pair.refs[0], self.pair.alts[0]),
        )
        self.assertEqual(
            (item.position.contig, item.position.position),
            ("11", self.pair.positions[0]),
        )
        self.assertEqual(
            (
                item.stated_assembly.name,
                item.stated_assembly.cited_line,
                item.placement.step,
                item.placement.matched_name,
            ),
            ("sacCer1", LINE, "table", "sacCer1"),
        )
        self.assertEqual(self.opener.sent, [YEAST_REPORT])
        self.assertEqual(len(list((self.cache / "moves" / YEAST).glob("*.json"))), 1)

    def test_a_move_onto_the_reverse_strand_writes_the_runs_alleles(self):
        complement = {"A": "T", "C": "G", "G": "C", "T": "A"}

        record = self.placed(self.segment(1))

        self.assertEqual(record["answers"][0]["answer"], "moved")
        request, _ = self.requests()
        position = request.items[0].placed
        self.assertEqual(
            (position.contig, position.position, position.ref, position.alt),
            (
                "flipped",
                900,
                complement[self.pair.refs[1]],
                complement[self.pair.alts[1]],
            ),
        )

    def test_a_move_without_one_place_is_answered_and_written_unplaced(self):
        record = self.placed(self.segment(3), self.segment(4), self.segment(2))

        self.assertEqual(
            [(a["answer"], a["step"], a["move"]["result"]) for a in record["answers"]],
            [
                ("ambiguous", "table", "several_places"),
                ("not_found", "table", "not_found"),
                ("not_checked", "table", "ref_changed"),
            ],
        )
        request, _ = self.requests()
        self.assertEqual(
            [(item.placement.step, item.placed) for item in request.items],
            [("none", None)] * 3,
        )

    def test_a_changed_ref_passes_the_rsid_instead(self):
        record = self.placed(self.segment(2, rsid="rs9"))

        [answer] = record["answers"]
        self.assertEqual((answer["answer"], answer["step"]), ("passed", "rsid_only"))
        self.assertIn("B has", answer["reason"])
        request, _ = self.requests()
        self.assertEqual(
            (request.items[0].rsid, request.items[0].placed), ("rs9", None)
        )

    def test_what_the_move_lacks_is_not_checked_and_each_step_named(self):
        cases = [
            (
                self.segment(0, ref=None, alt=None),
                "table: a position without its alleles cannot be moved; database "
                "equivalence: a position without its alleles has no allele to ask for",
            ),
            (
                self.segment(0, ref="AC", alt="A"),
                "table: only an SNV moves; the paper states AC>A; database equivalence: "
                "NCBI's Variation service gives NC_001143.4:",
            ),
        ]
        for target, reason in cases:
            with self.subTest(reason=reason):
                record = self.placed(target)
                [answer] = record["answers"]
                self.assertEqual(
                    (answer["answer"], answer["step"]), ("not_checked", "none")
                )
                self.assertIn(reason, answer["reason"])

    def test_a_run_assembly_with_no_stored_index_is_not_checked(self):
        self.choose("GCF_900000009.1")

        record = self.placed(self.segment(0))

        [answer] = record["answers"]
        self.assertEqual(answer["answer"], "not_checked")
        self.assertIn("table: the move refused it (no_stored_index", answer["reason"])
        self.assertIn("no equivalent on the Run's sequences", answer["reason"])
        # The reason names the commands that store the Run's index.
        self.assertIn(
            f"python -m tools.reference_index catalogues, then python -m "
            f"tools.reference_index ask {YEAST}",
            answer["reason"],
        )

    def test_a_curated_sequence_missing_from_the_cache_is_named_with_its_fetch(self):
        (self.cache / "ncbi" / "sequences" / "NC_001143.4.fa").unlink()

        record = self.placed(self.segment(0))

        [answer] = record["answers"]
        self.assertEqual(answer["answer"], "not_checked")
        self.assertIn(
            "table: sacCer1's 11 is NC_001143.4, which is not in the reference-index "
            "cache: python -m tools.reference_index fetch-sequence NC_001143.4",
            answer["reason"],
        )


GRCH37_P13 = {"name": "GRCh37.p13", "cited_line": "reads were aligned to GRCh37.p13"}
GRCH37 = "GCF_000001405.25"
APOE_HG19_SPDI = "NC_000019.9:45411940:T:C"


def _apoe_grch37(rsid=None):
    """APOE's e4 allele as a paper states it on GRCh37.p13 (19:45411941 T>C; GRCh38
    19:44908684), with no rsID unless given, so a step must place its position. The
    paper names the version as NCBI does, so placement with no model finds it
    literally (#471 R4 a)."""
    return paper_target(
        "APOE e4 at chr19:45411941 T>C (GRCh37.p13)",
        position={"contig": "19", "position": 45411941, "ref": "T", "alt": "C"},
        stated=GRCH37_P13,
        rsid=rsid,
    )


def _grch37_answers(**more):
    """The GETs the resolver makes for a GRCh37.p13 position on GRCh38.p14: #393's
    development GETs d2, d4 and d5 (its recorded listing of every human RefSeq version
    standing in for NCBI's search for GRCh37.p13, synthetic), and #369's retained
    GRCh38 molecules."""
    return {
        **human_answers(),
        listing_url(9606, "GRCh37.p13"): Answer(
            200, (NCBI / "taxon-9606-refseq-all-versions.json").read_bytes()
        ),
        **names(GRCH37, (NCBI / "gcf-000001405.25-sequence-report.json").read_bytes()),
        equivalents_url(APOE_HG19_SPDI): Answer(
            200,
            (NCBI / "spdi-NC_000019.9-45411940-T-C-equivalents.json").read_bytes(),
        ),
        **names(HUMAN, GRCH38_MOLECULES),
        **more,
    }


class DatabaseEquivalenceTests(_PlaceCase):
    """Step 3 (owner, #393 Q6, Q8): NCBI names the stated sequence, and its Variation
    service gives the stated allele one equivalent on the Run's sequences."""

    def test_ncbi_names_the_stated_sequence_its_contig_literally_names(self):
        """Paper placement has no model: NCBI's record literally named the contig is
        the stated sequence; any other spelling is left with its reason for the model
        (owner #467 R4 a, #383; ADR 0049)."""
        target = _apoe_grch37()
        target["request"]["position"]["contig"] = "chr19"
        code, record = self.place(paper_targets([target]), HUMAN, _grch37_answers())
        self.assertEqual(code, 0, record)
        [answer] = record["answers"]
        self.assertEqual(
            (answer["answer"], answer["stated_sequence"]["accession"]),
            ("moved", "NC_000019.9"),
        )

        target["request"]["position"]["contig"] = "Chromosome 19"
        code, record = self.place(paper_targets([target]), HUMAN, _grch37_answers())
        self.assertEqual(code, 0, record)
        [answer] = record["answers"]
        self.assertNotEqual(answer["answer"], "moved")
        self.assertIn("literally named Chromosome 19", answer["reason"])

    def test_a_grch37_position_on_grch38_is_moved_by_database_equivalence(self):
        code, record = self.place(
            paper_targets([_apoe_grch37()]), HUMAN, _grch37_answers()
        )

        self.assertEqual(code, 0, record)
        [answer] = record["answers"]
        self.assertEqual(
            (answer["answer"], answer["step"]), ("moved", "database_equivalence")
        )
        self.assertEqual(
            answer["position"],
            {"contig": "NC_000019.10", "position": 44908684, "ref": "T", "alt": "C"},
        )
        self.assertEqual(
            answer["stated_sequence"],
            {
                "accession": "NC_000019.9",
                "named_by": "ncbi",
                "assembly_accession": GRCH37,
            },
        )
        self.assertEqual(
            answer["equivalence"],
            {"spdi": APOE_HG19_SPDI, "on_run": ["NC_000019.10:44908683:T:C"]},
        )
        request, _ = self.requests()
        [item] = request.items
        self.assertEqual(
            (
                item.placed.contig,
                item.placed.position,
                item.placed.ref,
                item.placed.alt,
            ),
            ("NC_000019.10", 44908684, "T", "C"),
        )
        self.assertEqual(
            (item.position.contig, item.position.position),
            ("19", 45411941),
        )
        self.assertEqual(
            (
                item.stated_assembly.name,
                item.stated_assembly.cited_line,
                item.placement.step,
                item.placement.matched_name,
            ),
            (
                "GRCh37.p13",
                GRCH37_P13["cited_line"],
                "database_equivalence",
                "GRCh37.p13",
            ),
        )
        self.assertEqual(
            self.opener.sent,
            [
                HUMAN_REPORT,
                listing_url(9606, "GRCh37.p13"),
                *names_urls(GRCH37),
                equivalents_url(APOE_HG19_SPDI),
                *names_urls(HUMAN),
            ],
        )

    def test_several_equivalents_on_the_runs_sequences_are_ambiguous(self):
        several = {
            "data": {
                "spdis": [
                    {
                        "seq_id": "NC_000019.10",
                        "position": at,
                        "deleted_sequence": "T",
                        "inserted_sequence": "C",
                    }
                    for at in (44908683, 44999999)
                ]
            }
        }
        answers = _grch37_answers(
            **{equivalents_url(APOE_HG19_SPDI): Answer(200, several)}
        )

        code, record = self.place(paper_targets([_apoe_grch37()]), HUMAN, answers)

        self.assertEqual(code, 0)
        [answer] = record["answers"]
        self.assertEqual(
            (answer["answer"], answer["step"], answer["position"]),
            ("ambiguous", "database_equivalence", None),
        )
        self.assertIn("2 equivalents on the Run's sequences", answer["reason"])
        request, _ = self.requests()
        self.assertEqual(
            (request.items[0].placement.step, request.items[0].placed), ("none", None)
        )

    def test_equivalents_that_are_not_substitutions_are_a_miss_not_an_answer(self):
        # Spec r1 F2: step 3 answers a same-length substitution only, one or several.
        def deletion(at):
            return {
                "seq_id": "NC_000019.10",
                "position": at,
                "deleted_sequence": "TG",
                "inserted_sequence": "",
            }

        for count in (1, 2):
            with self.subTest(count=count):
                body = {
                    "data": {"spdis": [deletion(44908683 + n) for n in range(count)]}
                }
                answers = _grch37_answers(
                    **{equivalents_url(APOE_HG19_SPDI): Answer(200, body)}
                )

                code, record = self.place(
                    paper_targets([_apoe_grch37()]), HUMAN, answers
                )

                self.assertEqual(code, 0)
                [answer] = record["answers"]
                self.assertEqual(
                    (answer["answer"], answer["step"]), ("not_checked", "none")
                )
                self.assertIn(
                    "database equivalence: NCBI's Variation service gives "
                    "NC_000019.9:45411940:T:C no substitution on the Run's sequences "
                    "(NC_000019.10:44908683:TG:",
                    answer["reason"],
                )
                self.assertIn(
                    "flank alignment: GRCh37.p13's 19 is NC_000019.9", answer["reason"]
                )

    def test_a_malformed_equivalent_is_the_targets_semantic_failure(self):
        # Standards r1 S1: the source's fields are read inside its failure boundary.
        malformed = {
            "data": {
                "spdis": [
                    {
                        "seq_id": "NC_000019.10",
                        "position": "bad",
                        "deleted_sequence": "T",
                        "inserted_sequence": "C",
                    }
                ]
            }
        }
        answers = _grch37_answers(
            **{equivalents_url(APOE_HG19_SPDI): Answer(200, malformed)}
        )
        document = paper_targets([_apoe_grch37(), paper_target("rs5", rsid="rs5")])

        code, record = self.place(document, HUMAN, answers)

        self.assertEqual(code, 2)
        failed, passed = record["answers"]
        self.assertEqual(
            (
                failed["failure"]["tool"],
                failed["failure"]["failure_class"],
                failed["failure"]["route"],
            ),
            ("ncbi_spdi_equivalents", "semantic", "ask"),
        )
        self.assertEqual(passed["answer"], "passed")
        self.assertEqual((record["waiting"], record["stopped"]), (["t1"], None))
        self.assertFalse(self.request.exists())

    def test_no_equivalent_and_no_stored_index_is_not_checked_naming_each_step(self):
        answers = _grch37_answers(**{equivalents_url(APOE_HG19_SPDI): NO_EQUIVALENT})

        code, record = self.place(paper_targets([_apoe_grch37()]), HUMAN, answers)

        self.assertEqual(code, 0)
        [answer] = record["answers"]
        self.assertEqual((answer["answer"], answer["step"]), ("not_checked", "none"))
        self.assertEqual(
            answer["reason"],
            "GRCh37.p13 is not a name NCBI's report gives the Run's GRCh38.p14 "
            "(GCF_000001405.40); table: no table row (the Run gives no table) for "
            "GRCh37.p13 19; database equivalence: NCBI's Variation service gives "
            "NC_000019.9:45411940:T:C no equivalent on the Run's sequences; flank "
            "alignment: GRCh37.p13's 19 is NC_000019.9, which is not in the "
            "reference-index cache: python -m tools.reference_index fetch-sequence "
            "NC_000019.9",
        )

    def test_a_failed_equivalence_contact_is_the_targets_own_failure(self):
        answers = _grch37_answers(
            **{equivalents_url(APOE_HG19_SPDI): [Answer(503) for _ in range(3)]}
        )
        document = paper_targets([_apoe_grch37(), paper_target("rs5", rsid="rs5")])

        code, record = self.place(document, HUMAN, answers)

        self.assertEqual(code, 2)
        failed, passed = record["answers"]
        self.assertEqual(
            (failed["step"], failed["failure"]["tool"], failed["failure"]["route"]),
            ("same_version", "ncbi_spdi_equivalents", "ask"),
        )
        self.assertEqual(passed["answer"], "passed")
        self.assertEqual(record["waiting"], ["t1"])


class NcbiNamingTests(_PlaceCase):
    """The stated sequence when the Run's table has no row (owner, #393 G1 A)."""

    def listing(self, reports, **more):
        return {"reports": reports, **more}

    def report(self, accession, name, synonym):
        # Synthetic listing rows; NCBI's report states each version's status.
        return {
            "accession": accession,
            "assembly_info": {
                "assembly_name": name,
                "synonym": synonym,
                "assembly_status": "current",
            },
        }

    def test_a_name_ncbi_gives_several_assemblies_names_none(self):
        """Two rows NCBI's search lists are both literally named the stated version:
        placement with no model picks none of them and names both in its reason (the
        owner's conflict rule: a real ambiguity is never silently picked; #471 R4 a,
        #393 G1 A's series rule superseded, #375 B19)."""
        listing = self.listing(
            [
                self.report("GCF_000000001.1", "Old-1", "xx1"),
                self.report("GCF_000000002.1", "Other-1", "xx1"),
            ]
        )
        target = paper_target(
            "chr1:5 A>G",
            position={"contig": "1", "position": 5, "ref": "A", "alt": "G"},
            stated={"name": "xx1", "cited_line": "aligned to xx1"},
        )
        answers = {**human_answers(), listing_url(9606, "xx1"): Answer(200, listing)}

        code, record = self.place(paper_targets([target]), HUMAN, answers)

        self.assertEqual(code, 0)
        self.assertIn(
            "NCBI: NCBI lists several assemblies of taxon 9606 named xx1: Old-1 "
            "(GCF_000000001.1), Other-1 (GCF_000000002.1)",
            record["answers"][0]["reason"],
        )

    def test_a_search_past_one_page_reads_its_next_page(self):
        """NCBI's search is read page by page, by its own next page token: a version
        listed on the second page is found, with no refusal past one page (#471 R4 a;
        #393 G1 A's one-page refusal superseded, #375 B20). Synthetic pages."""
        first = self.listing([], next_page_token="more", total_count=1)
        second = self.listing(
            [self.report(GRCH37, "GRCh37.p13", "hg19")], total_count=1
        )
        answers = _grch37_answers(
            **{
                listing_url(9606, "GRCh37.p13"): Answer(200, first),
                listing_url(9606, "GRCh37.p13", "more"): Answer(200, second),
            }
        )

        code, record = self.place(paper_targets([_apoe_grch37()]), HUMAN, answers)

        self.assertEqual(code, 0, record)
        [answer] = record["answers"]
        self.assertEqual(
            (answer["answer"], answer["stated_sequence"]["assembly_accession"]),
            ("moved", GRCH37),
        )
        self.assertIn(listing_url(9606, "GRCh37.p13", "more"), self.opener.sent)

    def test_a_ucsc_name_ncbis_search_does_not_cover_waits_for_the_model(self):
        """A paper that writes hg19: NCBI's search covers no UCSC name, so placement
        with no model lists no assembly literally named it and the position stays
        unplaced with that reason until #383's model step names the version
        (#471 R4 a). NCBI's answer here is synthetic (an empty search)."""
        target = _apoe_grch37()
        target["request"]["stated_assembly"] = {
            "name": "hg19",
            "cited_line": "reads were aligned to hg19",
        }
        answers = {
            **human_answers(),
            listing_url(9606, "hg19"): Answer(200, self.listing([], total_count=0)),
        }

        code, record = self.place(paper_targets([target]), HUMAN, answers)

        self.assertEqual(code, 0, record)
        [answer] = record["answers"]
        self.assertEqual((answer["answer"], answer["step"]), ("not_checked", "none"))
        self.assertIn(
            "NCBI: NCBI's search lists no assembly of taxon 9606 literally named hg19",
            answer["reason"],
        )

    def test_the_runs_table_row_names_the_sequence_before_ncbi(self):
        table = self.root / "hg19.json"
        table.write_text(
            json.dumps(
                {
                    "schema_version": "bio-neb.stated-sequences/1",
                    "name": "hg19 chr19 (test)",
                    "rows": [
                        {
                            # The contig as the target writes it: a row matches
                            # exactly.
                            "version": "GRCh37.p13",
                            "contig": "19",
                            "accession": "NC_000019.9",
                            "proof": {"source": "a test", "window": "the test's own"},
                        }
                    ],
                }
            )
        )

        code, record = self.place(
            paper_targets([_apoe_grch37()]),
            HUMAN,
            _grch37_answers(),
            "--stated-sequences",
            str(table),
        )

        self.assertEqual(code, 0)
        [answer] = record["answers"]
        self.assertEqual(
            (answer["answer"], answer["step"], answer["stated_sequence"]["named_by"]),
            ("moved", "database_equivalence", "table"),
        )
        self.assertNotIn(listing_url(9606, "GRCh37.p13"), self.opener.sent)
        self.assertEqual(record["table"]["name"], "hg19 chr19 (test)")


class DevtoolsEntryTests(_PlaceCase):
    """``python -m bio_neb.devtools.target_placement``: the same placement with no flank
    mover behind the port; a flank step says so and the next step runs."""

    def test_the_entry_places_and_a_flank_step_names_its_missing_mover(self):
        from bio_neb.devtools.target_placement import main as entry

        kept = paper_target(
            "rs4 at chr7:150951552",
            rsid="rs4",
            position={"contig": "7", "position": 150951552},
            stated={"name": "hg38", "cited_line": HG38_LINE},
        )
        moved = paper_target(
            "chr11:899 A>G",
            position={"contig": "11", "position": 899, "ref": "A", "alt": "G"},
            stated={"name": "sacCer1", "cited_line": LINE},
        )
        cases = (
            (kept, HUMAN, human_answers(), ("kept", "same_version"), None),
            (
                moved,
                YEAST,
                yeast_answers(
                    **{equivalents_url("NC_001143.4:898:A:G"): NO_EQUIVALENT}
                ),
                ("not_checked", "none"),
                "table: sacCer1's 11 is NC_001143.4, which this entry holds no flank "
                "mover for",
            ),
        )
        for target, assembly, answers, expected, reason in cases:
            with self.subTest(expected=expected):
                self.targets.write_text(json.dumps(paper_targets([target])))
                self.opener = Opener(answers)
                out = io.StringIO()
                with (
                    contextlib.redirect_stdout(out),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    code = entry(
                        [
                            "--assembly",
                            assembly,
                            "--paper-targets",
                            str(self.targets),
                            "--request",
                            str(self.request),
                            "--variant-cache",
                            str(self.variant_cache),
                            "--stated-sequences",
                            str(SACCER1_TABLE),
                        ],
                        opener=self.opener,
                    )

                record = json.loads(out.getvalue())
                self.assertEqual(code, 0, record)
                [answer] = record["answers"]
                self.assertEqual((answer["answer"], answer["step"]), expected)
                if reason is not None:
                    self.assertIn(reason, answer["reason"])
                request, _ = self.requests()
                self.assertEqual(len(request.items), 1)


class RunSeamTests(_PlaceCase):
    """A Run passes no flank mover: the move is development-only (owner, #449 Q1 a). A
    target only a flank move would place gets the failure package's typed failure and waits
    for its Review (retry or accept), never a plain `not_checked` miss; the other targets
    are answered and the run goes on."""

    def run_place(self, document, assembly, answers, *, table=None, resume=None):
        self.targets.write_text(json.dumps(document))
        self.opener = Opener(answers)
        used = None
        if table is not None:
            table, used = read_stated_sequences(table.read_bytes())
        ncbi = NcbiSources(
            NcbiInvestigationSourceAdapter(UrllibNcbiTransport(opener=self.opener)),
            self.variant_cache / "gene-change",
        )
        return place(
            TargetsFile.read(self.targets),
            PlacementRun(
                assembly=assembly, table=table, table_used=used, resume=resume
            ),
            PlacementSources(ncbi=ncbi, mover=None),
        )

    def sacCer1(self, rsid=None):
        return paper_target(
            f"{rsid or 'a change'} at chr11:899 A>G",
            position={"contig": "11", "position": 899, "ref": "A", "alt": "G"},
            stated=SACCER1,
            rsid=rsid,
        )

    def test_a_target_only_a_flank_move_places_fails_and_waits_for_its_review(self):
        cases = (
            (
                "table",
                [self.sacCer1(), paper_target("rs5", rsid="rs5")],
                YEAST,
                yeast_answers(
                    **{equivalents_url("NC_001143.4:898:A:G"): NO_EQUIVALENT}
                ),
                SACCER1_TABLE,
                {
                    "accession": "NC_001143.4",
                    "named_by": "table",
                    "assembly_accession": None,
                },
            ),
            (
                "flank_alignment",
                [_apoe_grch37(), paper_target("rs5", rsid="rs5")],
                HUMAN,
                _grch37_answers(**{equivalents_url(APOE_HG19_SPDI): NO_EQUIVALENT}),
                None,
                {
                    "accession": "NC_000019.9",
                    "named_by": "ncbi",
                    "assembly_accession": GRCH37,
                },
            ),
        )
        for step, targets, assembly, answers, table, sequence in cases:
            with self.subTest(step=step):
                record = self.run_place(
                    paper_targets(targets), assembly, answers, table=table
                )

                failed, passed = record.answers
                self.assertEqual((failed.answer, failed.step), (None, step))
                failure = failed.failure.model_dump(mode="json")
                self.assertEqual(
                    (failure["kind"], failure["failure_class"], failure["route"]),
                    ("flank_move_unavailable", "rejected", "ask"),
                )
                self.assertEqual(failure["stated_sequence"], sequence)
                self.assertIn("no equivalent on the Run's sequences", failed.reason)
                self.assertIn("a Run holds no flank mover", failed.reason)
                self.assertEqual(passed.answer, "passed")
                self.assertEqual(
                    (record.waiting, record.failure_count, record.stopped),
                    (("t1",), 1, None),
                )

    def test_accept_drops_the_target_as_unplaced_with_its_failure_named(self):
        document = paper_targets([self.sacCer1()])
        answers = yeast_answers(
            **{equivalents_url("NC_001143.4:898:A:G"): NO_EQUIVALENT}
        )
        waiting = self.run_place(document, YEAST, answers, table=SACCER1_TABLE)
        content = canonical_json(waiting.model_dump(mode="json")).encode()
        resume = Resume(
            record=PaperTargetsPlaced.model_validate_json(content),
            record_sha256=hashlib.sha256(content).hexdigest(),
            retry=frozenset(),
            accept=frozenset({"t1"}),
        )

        record = self.run_place(document, YEAST, {}, table=SACCER1_TABLE, resume=resume)
        record = write_request_file(
            record, TargetsFile.read(self.targets), self.request
        )

        self.assertEqual((record.waiting, record.answers[0].accepted), ((), True))
        self.assertEqual(self.opener.sent, [])
        request, _ = self.requests()
        [item] = request.items
        self.assertEqual((item.placement.step, item.placed), ("none", None))
        self.assertIn(
            "(placement: accepted failure: rejected on flank_move: sacCer1's 11 is "
            "NC_001143.4",
            item.request_text,
        )

    def test_a_target_needing_a_flank_move_with_an_rsid_passes_it(self):
        record = self.run_place(
            paper_targets([self.sacCer1(rsid="rs9")]),
            YEAST,
            yeast_answers(**{equivalents_url("NC_001143.4:898:A:G"): NO_EQUIVALENT}),
            table=SACCER1_TABLE,
        )

        [answer] = record.answers
        self.assertEqual(
            (answer.answer, answer.step, answer.failure), ("passed", "rsid_only", None)
        )
        self.assertIn("a Run holds no flank mover", answer.reason)
        self.assertEqual(record.waiting, ())


class ResumeTests(_PlaceCase):
    """The owner answers each waiting target (#393 Q18): retry reruns it, accept writes it
    as unplaced with the failure named; every other answer is reused, with no contact."""

    def wait(self):
        # t1's annotation download fails three times; t2's, the fourth, is answered.
        answers = yeast_answers(
            **{
                YEAST_GFF: [
                    Answer(503),
                    Answer(503),
                    Answer(503),
                    Answer(200, gff_zip(YEAST_GFF_TEXT)),
                ]
            }
        )
        document = paper_targets([_pbp2(), _pbp2(gene="NOPE")])
        code, record = self.place(document, YEAST, answers)
        self.assertEqual((code, record["waiting"]), (2, ["t1"]))
        [path] = self.records()
        return document, record, path

    def test_retry_reruns_the_waiting_target_and_reuses_the_others(self):
        document, waiting, path = self.wait()

        code, record = self.place(
            None, YEAST, yeast_answers(), "--resume", str(path), "--retry", "t1"
        )

        self.assertEqual(code, 0, record)
        self.assertEqual(
            [(a["label"], a["answer"]) for a in record["answers"]],
            [("t1", "placed"), ("t2", "not_found")],
        )
        self.assertEqual(record["answers"][1] | {"requests": []}, waiting["answers"][1])
        self.assertEqual(
            record["resumed_from"], hashlib.sha256(path.read_bytes()).hexdigest()
        )
        # The annotation t2 downloaded is read from the cache: t1 asks only its codon.
        self.assertEqual(
            self.opener.sent, [YEAST_REPORT, span_url("NC_001134.8", 1004, 3)]
        )
        request, _ = self.requests()
        self.assertEqual(len(request.items), 4)

    def test_accept_writes_the_failure_as_unplaced_with_no_contact(self):
        _, _, path = self.wait()

        code, record = self.place(
            None, YEAST, {}, "--resume", str(path), "--accept", "t1"
        )

        self.assertEqual(code, 0, record)
        accepted = record["answers"][0]
        self.assertEqual(
            (accepted["accepted"], accepted["answer"], record["waiting"]),
            (True, None, []),
        )
        self.assertEqual(record["failure_count"], 1)
        self.assertEqual(self.opener.sent, [])
        request, written = self.requests()
        self.assertEqual(
            (written[0]["target"], request.items[0].placement.step),
            ({"gene_change": {"gene": "PBP2", "change": "Y2X"}}, "none"),
        )
        self.assertIn(
            "(placement: accepted failure: exhausted after 3 attempts on "
            "ncbi_genome_annotation",
            request.items[0].request_text,
        )

    def test_an_accepted_failure_stays_accepted_through_a_later_resume(self):
        fails = yeast_answers(**{YEAST_GFF: [Answer(503) for _ in range(6)]})
        code, _ = self.place(paper_targets([_pbp2(), _pbp2()]), YEAST, fails)
        self.assertEqual(code, 2)
        [first] = self.records()
        code, record = self.place(
            None,
            YEAST,
            yeast_answers(**{YEAST_GFF: [Answer(503) for _ in range(3)]}),
            "--resume",
            str(first),
            "--accept",
            "t1",
            "--retry",
            "t2",
        )
        self.assertEqual((code, record["waiting"]), (2, ["t2"]))
        [second] = [path for path in self.records() if path != first]

        code, record = self.place(
            None, YEAST, yeast_answers(), "--resume", str(second), "--retry", "t2"
        )

        self.assertEqual(code, 0, record)
        self.assertEqual(
            [(a["label"], a["answer"], a["accepted"]) for a in record["answers"]],
            [("t1", None, True), ("t2", "placed", False)],
        )
        self.assertTrue(self.request.exists())

    def test_answers_that_miss_a_waiting_target_are_refused_before_contact(self):
        _, _, path = self.wait()
        cases = {
            "none": (),
            "both": ("--retry", "t1", "--accept", "t1"),
            "another": ("--retry", "t2"),
        }
        for name, answers in cases.items():
            with self.subTest(name):
                code, refused = self.place(
                    None, YEAST, {}, "--resume", str(path), *answers
                )
                self.assertEqual((code, refused["refused"]), (2, "resume_answers"))
                self.assertEqual(self.opener.sent, [])

    def test_a_waiting_record_of_other_targets_is_refused(self):
        _, _, path = self.wait()

        code, refused = self.place(
            paper_targets([_pbp2()]), YEAST, {}, "--resume", str(path), "--retry", "t1"
        )

        self.assertEqual((code, refused["refused"]), (2, "resume_mismatch"))
        self.assertEqual(self.opener.sent, [])

    def test_retry_or_accept_without_a_waiting_record_is_refused(self):
        code, refused = self.place(paper_targets([_pbp2()]), YEAST, {}, "--retry", "t1")

        self.assertEqual((code, refused["refused"]), (2, "resume_answers"))
        self.assertEqual(self.records(), [])


LANE_P_RECORDS = Path(__file__).parent / "fixtures" / "paper_targets"


class LanePRecordsTests(_PlaceCase):
    """Lane P's retained #349 records (copied by #376): every target gets one answer, every
    target is written, and main reads the file."""

    def place_record(self, name, assembly, answers, *options):
        document = json.loads(
            (LANE_P_RECORDS / f"{name}-run1-targets.json").read_text()
        )
        code, record = self.place(document, assembly, answers, *options)
        self.assertEqual(code, 0, record)
        request, written = self.requests()
        self.assertEqual(
            [a["label"] for a in record["answers"]],
            [f"t{n}" for n in range(1, len(document["targets"]) + 1)],
        )
        self.assertEqual(
            sorted(i for a in record["answers"] for i in a["requests"]),
            list(range(len(written))),
        )
        self.assertEqual(len(request.items), len(written))
        return document, record, request

    def test_h1s_hg38_positions_are_kept_on_grch38_with_their_rsids(self):
        document, record, request = self.place_record("h1", HUMAN, human_answers())

        self.assertEqual({a["answer"] for a in record["answers"]}, {"kept"})
        self.assertEqual(
            [(i.rsid, i.position.contig, i.position.position) for i in request.items],
            [
                (
                    t["request"]["rsid"],
                    "7" if n < 2 else "2",
                    t["request"]["position"]["position"],
                )
                for n, t in enumerate(document["targets"])
            ],
        )
        # The paper's hg38 stays its stated version; same_version keeps each position.
        self.assertEqual(
            {
                (i.stated_assembly.name, i.placement.step, i.placement.matched_name)
                for i in request.items
            },
            {("hg38", "same_version", "hg38")},
        )
        self.assertEqual(
            [i.placed.model_dump() for i in request.items],
            [i.position.model_dump() for i in request.items],
        )

    def test_s17s_equcab3_position_is_kept_with_its_alleles(self):
        answers = {
            HORSE_REPORT: Answer(
                200,
                dataset_report(HORSE, "EquCab3.0", "equCab3", "GCA_002863925.1", 9796),
            )
        }

        _, record, request = self.place_record("s17", HORSE, answers)

        self.assertEqual([a["answer"] for a in record["answers"]], ["kept"])
        position = request.items[0].position
        self.assertEqual(
            (position.contig, position.position, position.ref, position.alt),
            ("4", 54108297, "G", "T"),
        )

    def test_s3s_four_targets_are_each_answered_and_written(self):
        # Lane P's reader types no change, so G15 answers each gene change not
        # checked, and the sacCer1-era chromosome XI is not in this test's cache: no
        # target is placed, none is dropped.
        answers = yeast_answers(
            **{equivalents_url("NC_001143.4:207468:A:C"): NO_EQUIVALENT}
        )
        _, record, request = self.place_record(
            "s3", YEAST, answers, "--stated-sequences", str(SACCER1_TABLE)
        )

        self.assertEqual(
            [(a["label"], a["answer"], a["step"]) for a in record["answers"]],
            [
                ("t1", "not_checked", "g15"),
                ("t2", "not_checked", "g15"),
                ("t3", "not_checked", "g15"),
                ("t4", "not_checked", "none"),
            ],
        )
        self.assertIn("fetch-sequence NC_001143.4", record["answers"][3]["reason"])
        # No item is placed or has an rsID: each is written unplaced with its reason.
        self.assertEqual(
            {(i.rsid, i.placed, i.placement.step) for i in request.items},
            {(None, None, "none")},
        )
