"""An assembly's sequence names are copied, and the model names the sequence (#467).

ADR 0049: code copies NCBI's assembly report (every row, every column), the model names
which sequence a written contig means, and code checks only membership: the cited record
is one the report holds and the quote stands in the researcher's words and holds the
contig (owner D1 a, RC5 1a). Answers are #466's recorded NCBI answers, replayed beneath
the real NCBI port (`tests/recorded_sources.py`).
"""

from __future__ import annotations

import json
import unittest

from bio_neb.agentic_journey.tool_execution import tool_model_view
from bio_neb.failure import FailureOverrides, FailureTuning, run_failure_tuning
from bio_neb.investigation.investigation import (
    RetrievedSource,
    SequenceChoice,
    SequenceNaming,
)
from bio_neb.investigation.investigation_agent import LOOKUP_VIEW
from bio_neb.investigation.investigation_request import StatedPosition
from bio_neb.investigation.investigation_sources import look_up_stated_position
from bio_neb.investigation.lookup_origins import SequenceUnnamed
from bio_neb.providers.eva_investigation import fetch_source
from bio_neb.providers.ncbi_assembly_report import (
    ReportUnreadable,
    SequenceRecord,
    record_of,
    report_header,
    report_rows,
    search_rows,
)
from bio_neb.providers.ncbi_investigation import InvestigationSourceResponse
from tests.recorded_sources import fixture, recorded_ports

NO_BACKOFF = FailureTuning(operation=FailureOverrides(backoff_seconds=0))
HUMAN = "GCF_000001405.40"


class _AlleleAnswers:
    """The recorded Investigation sources, plus synthetic answers for the two GETs a
    stated allele adds once its sequence is named: the Run's reference holds the
    stated REF, and dbSNP lists no rsID there. A position with neither alleles nor an
    rsID is no longer looked up at all (owner #480 Q1), so these naming tests state
    the allele; nothing here was read from NCBI."""

    def __init__(self, sources, ref: str = "A") -> None:
        self._sources = sources
        self._ref = ref

    def __getattr__(self, name):
        return getattr(self._sources, name)

    def fetch_reference_span(self, span: str) -> InvestigationSourceResponse:
        sequence, start, length = span.split(":")
        body = json.dumps(
            {
                "data": {
                    "seq_id": sequence,
                    "position": int(start),
                    "deleted_sequence": self._ref * int(length),
                    "inserted_sequence": "N" * int(length),
                }
            }
        ).encode()
        return _synthetic("ncbi_reference_span", span, body)

    def fetch_spdi_rsids(self, spdi: str) -> InvestigationSourceResponse:
        return _synthetic("ncbi_spdi_rsids", spdi, b'{"data":{"rsids":[]}}')


def _synthetic(operation: str, key: str, body: bytes) -> InvestigationSourceResponse:
    return InvestigationSourceResponse(
        f"https://synthetic.example/{operation}/{key}",
        200,
        body,
        "2026-10-29T00:00:00Z",
        1,
    )


def _allele_fetch(fixture_id: str):
    """A Run's `SourceFetch` over `_AlleleAnswers` of a fixture's recorded GETs."""
    sources = _AlleleAnswers(recorded_ports(fixture_id).investigation)

    def fetch(operation: str, key: str):
        response = fetch_source(sources, operation, key)
        source = RetrievedSource.create(
            operation=operation,
            request_identity=response.request_identity,
            status_code=response.status_code,
            observed_at=response.observed_at,
            elapsed_time_ms=response.elapsed_time_ms,
            raw_response=response.body,
        )
        return source, response.body

    return fetch


def _report(fixture_id: str) -> bytes:
    """A fixture's recorded FTP assembly report (its last answer)."""
    return fixture(fixture_id).answers[-1].body


class CopiedReportTests(unittest.TestCase):
    """The one reader copies every row and column as NCBI writes them."""

    def test_every_row_of_a_large_report_is_copied_in_its_order(self) -> None:
        rows = list(report_rows(_report("naming-wgs-scaffold")))

        # NCBI's dog CanFam3.1 report lists 3,268 sequences (spike).
        self.assertEqual(len(rows), 3268)
        self.assertEqual(rows[0].sequence_name, "chr01")

    def test_an_unplaced_scaffold_keeps_all_ten_columns(self) -> None:
        (scaffold,) = [
            row
            for row in report_rows(_report("naming-wgs-scaffold"))
            if row.genbank == "AAEX03018827.1"
        ]

        self.assertEqual(
            scaffold,
            SequenceRecord(
                sequence_name="chrUn_16547365",
                sequence_role="unplaced-scaffold",
                assigned_molecule=None,
                location_type=None,
                genbank="AAEX03018827.1",
                relationship="=",
                refseq="NW_003726146.1",
                assembly_unit="Primary Assembly",
                length=405216,
                ucsc="chrUn_AAEX03018827",
            ),
        )

    def test_a_name_ncbi_writes_na_is_absent_not_a_name(self) -> None:
        (chromosome,) = [
            row
            for row in report_rows(_report("naming-plant-chromosome"))
            if row.sequence_name == "1"
        ]

        self.assertIsNone(chromosome.ucsc)
        self.assertEqual(chromosome.names, {"1", "CP002684.1", "NC_003070.9"})

    def test_an_nz_refseq_name_is_copied(self) -> None:
        names = {
            name
            for row in report_rows(_report("naming-nz-sequence"))
            for name in row.names
        }

        self.assertIn("NZ_CP009072.1", names)

    def test_the_report_header_is_copied_as_written(self) -> None:
        header = report_header(_report("naming-mitochondrion"))

        self.assertIn("Assembly name:  GRCh38.p14", header)
        self.assertIn("Taxid:          9606", header)

    def test_a_row_that_is_not_ten_columns_is_unreadable(self) -> None:
        """A malformed report is the reader's typing of what it cannot read (#458)."""
        with self.assertRaisesRegex(ReportUnreadable, "not ten columns"):
            list(report_rows(b"# Sequence-Name\nI\tassembled-molecule\t15\n"))


class SearchAndMembershipTests(unittest.TestCase):
    """The searchable view names its cut; membership is an exact record id."""

    def test_a_search_names_how_many_rows_match_and_pages_the_rest(self) -> None:
        body = _report("naming-wgs-scaffold")
        first, total = search_rows(body, "chrUn_", offset=0, limit=2)
        second, again = search_rows(body, "chrUn_", offset=2, limit=2)

        self.assertEqual(len(first), 2)
        self.assertEqual(total, again)
        self.assertGreater(total, 3000)
        self.assertNotEqual(first, second)

    def test_a_search_matches_any_column_whatever_its_case(self) -> None:
        rows, total = search_rows(
            _report("naming-mitochondrion"), "CHRM", offset=0, limit=5
        )

        self.assertEqual(total, 1)
        self.assertEqual(rows[0].refseq, "NC_012920.1")

    def test_a_record_is_found_by_either_accession_only(self) -> None:
        body = _report("naming-mitochondrion")

        self.assertEqual(record_of(body, "J01415.2").sequence_name, "MT")
        self.assertEqual(record_of(body, "NC_012920.1").sequence_name, "MT")
        self.assertIsNone(record_of(body, "chrM"))


class NcbiPortTests(unittest.TestCase):
    def test_the_report_is_read_from_the_directory_ncbi_links_name(self) -> None:
        """TAIR10's directory is `GCF_000001735.4_TAIR10.1`, not derivable from its name."""
        response = recorded_ports("naming-plant-chromosome").ncbi.fetch_sequence_names(
            "GCF_000001735.4"
        )

        self.assertTrue(
            response.request_identity.endswith("TAIR10.1_assembly_report.txt")
        )
        self.assertEqual(len(response.preceding), 1)
        self.assertTrue(response.preceding[0].request_identity.endswith("/links"))
        self.assertEqual(len(list(report_rows(response.body))), 7)


class ModelNamedSequenceTests(unittest.TestCase):
    """The lookup takes the model's named sequence, checked by membership only (owner
    #467 D1 a, RC5 1a; ADR 0049)."""

    def setUp(self) -> None:
        self.enterContext(run_failure_tuning(NO_BACKOFF))

    def locate(
        self,
        fixture_id: str,
        accession: str,
        contig: str,
        choices: tuple[SequenceChoice, ...],
        words: tuple[str, ...],
        position: int = 3243,
    ):
        return look_up_stated_position(
            # The allele lets the lookup name the sequence.
            position=StatedPosition(contig=contig, position=position, ref="A", alt="G"),
            stated_assembly=None,
            rsid=None,
            requested_rsid=None,
            assembly_accession=accession,
            fetch=_allele_fetch(fixture_id),
            choices=choices,
            words=words,
            source="ncbi_dbsnp",
        )

    def test_a_non_literal_contig_is_named_by_the_model_and_marked(self) -> None:
        result = self.locate(
            "naming-mitochondrion",
            HUMAN,
            "chrMT",
            (
                SequenceChoice(
                    assembly_accession=HUMAN, record="NC_012920.1", quote="chrMT:3243"
                ),
            ),
            ("Check chrMT:3243 in this sample.",),
        )

        self.assertEqual(result.location.run_contig, "NC_012920.1")
        self.assertEqual(
            result.location.sequence_namings,
            (
                SequenceNaming(
                    assembly_accession=HUMAN,
                    written="chrMT",
                    record="NC_012920.1",
                    sequence_name="MT",
                    quote="chrMT:3243",
                    match="model_matched",
                ),
            ),
        )

    def test_a_contig_that_is_one_of_the_records_names_is_literal(self) -> None:
        result = self.locate(
            "naming-mitochondrion",
            HUMAN,
            "MT",
            (
                SequenceChoice(
                    assembly_accession=HUMAN, record="J01415.2", quote="MT 3243"
                ),
            ),
            ("MT 3243",),
        )

        self.assertEqual(result.location.sequence_namings[0].match, "literal")
        self.assertEqual(result.location.run_contig, "NC_012920.1")

    def test_arabic_for_roman_and_a_tair_name_are_the_models_matches(self) -> None:
        cases = (
            (
                "naming-roman-chromosome",
                "GCF_000146045.2",
                "chr4",
                "NC_001136.10",
                "IV",
            ),
            ("naming-plant-chromosome", "GCF_000001735.4", "Chr1", "CP002684.1", "1"),
        )
        for fixture_id, accession, written, record, name in cases:
            with self.subTest(written):
                result = self.locate(
                    fixture_id,
                    accession,
                    written,
                    (
                        SequenceChoice(
                            assembly_accession=accession, record=record, quote=written
                        ),
                    ),
                    (f"{written} at 1000",),
                    position=1000,
                )

                (naming,) = result.location.sequence_namings
                self.assertEqual(
                    (naming.sequence_name, naming.match), (name, "model_matched")
                )

    def test_no_named_sequence_asks_the_model_to_name_one(self) -> None:
        result = self.locate(
            "naming-mitochondrion", HUMAN, "chrMT", (), ("chrMT:3243",)
        )

        self.assertEqual(result.status, "sequence_unnamed")
        # The fact states only what was observed: no sequence named yet.
        self.assertEqual(
            result.conflicts,
            (f"No sequence is named for the written contig 'chrMT' on {HUMAN}.",),
        )
        self.assertEqual(
            result.origin, SequenceUnnamed(contig="chrMT", assemblies=(HUMAN,))
        )
        self.assertEqual(result.location.sequence_namings, ())
        # The asking moved to the Tool's model view: it states the observation and
        # its next step names the search Tool.
        view = tool_model_view(
            LOOKUP_VIEW,
            {
                "event_id": "lookup-1",
                "payload": {"result": result.model_dump(mode="json")},
            },
        )
        self.assertEqual(view["status"], "sequence_unnamed")
        self.assertEqual(view["statement"], result.conflicts[0])
        self.assertIn("search_sequence_names", view["next"])
        self.assertNotIn("search_sequence_names", result.conflicts[0])

    def test_membership_refusals_name_their_reason(self) -> None:
        """Code checks only membership (ADR 0049): a record the report does not hold, a
        quote outside the researcher's words, a quote without the written contig."""
        cases = (
            ("NC_999999.1", "chrMT:3243", "has no sequence NC_999999.1"),
            ("NC_012920.1", "chrMT 3243 heteroplasmy", "not in the researcher's words"),
            ("NC_012920.1", "3243", "does not hold the written contig"),
        )
        for record, quote, reason in cases:
            with self.subTest(reason):
                result = self.locate(
                    "naming-mitochondrion",
                    HUMAN,
                    "chrMT",
                    (
                        SequenceChoice(
                            assembly_accession=HUMAN, record=record, quote=quote
                        ),
                    ),
                    ("Check chrMT:3243 in this sample.",),
                )

                self.assertEqual(result.status, "sequence_unnamed")
                self.assertIn(reason, result.conflicts[0])
                self.assertEqual(result.location.sequence_namings, ())


if __name__ == "__main__":
    unittest.main()


class RecordedAgentFlowTests(unittest.TestCase):
    """Through the Investigation Agent over NCBI's recorded answers (#467): the real
    adapter reads the FTP assembly report the `links` answer names, the scripted model
    names the sequence after `sequence_unnamed`, and the Attempt's Journey reopens with
    the report retained as text."""

    def test_a_chrmt_position_is_named_on_grch38s_recorded_report(self) -> None:
        """The pending issue Review the lookup leads to shows the match (owner Q2 a;
        Spec r1 SP1). The position states its allele (owner #480 Q1), dbSNP lists
        no rsID there and no version is stated, so the finding is `ambiguous`."""
        import tempfile
        from pathlib import Path

        from bio_neb.investigation.investigation import UserAssertion
        from bio_neb.investigation.investigation_request import InvestigationRequestItem
        from bio_neb.recording import read_evidence
        from tests.sequential_investigation_support import (
            SequentialAdaptiveModel,
            input_and_assertion,
            run_attempt,
        )

        investigation_input, _ = input_and_assertion()
        request = InvestigationRequestItem(
            request_text="Check chrMT:3243 in this sample.",
            position=StatedPosition(contig="chrMT", position=3243, ref="A", alt="G"),
        )
        assertion = UserAssertion.create_collection(
            asserted_by="researcher",
            requests=(request,),
            predecessor_ids=(investigation_input.input_id,),
        )
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with run_failure_tuning(NO_BACKOFF):
            projection, journey = run_attempt(
                directory,
                # The scripted model judges chrMT means molecule MT.
                SequentialAdaptiveModel({"chrMT": "MT"}),
                _AlleleAnswers(recorded_ports("naming-mitochondrion").investigation),
                investigation_input=investigation_input,
                assertion=assertion,
            )
        events = read_evidence(journey)
        lookups = [
            e["payload"]["result"]
            for e in events
            if e["event_type"] == "RefSnpToolResultRecorded"
        ]

        self.assertEqual(lookups[0]["status"], "sequence_unnamed")
        self.assertEqual(
            [
                (naming["record"], naming["sequence_name"], naming["match"])
                for naming in lookups[-1]["location"]["sequence_namings"]
            ],
            [("NC_012920.1", "MT", "model_matched")],
        )
        retained = [
            reference["media_type"]
            for e in events
            if e["event_type"] == "InvestigationSourceRetrieved"
            and e["payload"]["source"]["operation"] == "ncbi_sequence_names"
            for reference in e["artifact_references"]
        ]
        self.assertIn("text/plain", retained)
        self.assertIn("application/json", retained)
        review = projection.pending_request
        self.assertEqual(review.question.issue.key, "refsnp: ambiguous")
        self.assertEqual(
            tuple(name.audit() for name in review.model_matched),
            ("chrMT -> MT (NC_012920.1) on GCF_000001405.40",),
        )


class _CitesAnEarlierLookup:
    """The scripted model, which compiles its `ambiguous` finding citing that lookup
    (Spec r2 SP2). Since #479 G2.2 a decision answers only a result that asks for it,
    so no later lookup can name the contig another way after the finding; the Review
    still reads the match of the lookup the finding cites."""

    def __init__(self) -> None:
        from tests.sequential_investigation_support import SequentialAdaptiveModel

        self.inner = SequentialAdaptiveModel({"chrMT": "MT"})
        self.cited = None
        from tests.model_endpoint_fakes import ScriptedModelEndpoint

        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        from tests.sequential_investigation_support import _current_member_messages

        _, lookup, _, basis, *_ = _current_member_messages(messages)
        if lookup is not None and lookup["status"] == "ambiguous":
            self.cited = basis["refsnp_event_id"]
        return await self.inner.respond(messages, info)


class CitedLookupDisclosureTests(unittest.TestCase):
    def test_the_review_shows_the_match_of_the_lookup_its_finding_cites(self) -> None:
        """The cited lookup's own match (#467 Spec r2 SP2; owner Q2 a)."""
        import tempfile
        from pathlib import Path

        from bio_neb.investigation.investigation import UserAssertion
        from bio_neb.investigation.investigation_request import InvestigationRequestItem
        from bio_neb.recording import read_evidence
        from tests.sequential_investigation_support import (
            input_and_assertion,
            run_attempt,
        )

        investigation_input, _ = input_and_assertion()
        request = InvestigationRequestItem(
            request_text="Check chrMT:3243 in this sample.",
            position=StatedPosition(contig="chrMT", position=3243, ref="A", alt="G"),
        )
        assertion = UserAssertion.create_collection(
            asserted_by="researcher",
            requests=(request,),
            predecessor_ids=(investigation_input.input_id,),
        )
        model = _CitesAnEarlierLookup()
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with run_failure_tuning(NO_BACKOFF):
            projection, journey = run_attempt(
                directory,
                model,
                _AlleleAnswers(recorded_ports("naming-mitochondrion").investigation),
                investigation_input=investigation_input,
                assertion=assertion,
            )
        lookups = [
            e["payload"]["result"]["status"]
            for e in read_evidence(journey)
            if e["event_type"] == "RefSnpToolResultRecorded"
        ]

        self.assertEqual(lookups[-1], "ambiguous")
        review = projection.pending_request
        self.assertEqual(review.question.issue.evidence_event_ids[0], model.cited)
        self.assertEqual(
            tuple(name.audit() for name in review.model_matched),
            ("chrMT -> MT (NC_012920.1) on GCF_000001405.40",),
        )
