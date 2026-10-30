"""A Run's assembly is read from NCBI's current report or, for a replaced or suppressed
assembly, its any-version report (#417), with no contact: lane Rf's live L5 bodies of
Gallus_gallus-5.0 (`tests/location_retained_bodies.py`), whose plain report is ``{}``."""

from __future__ import annotations

import hashlib
import json
import unittest
from datetime import UTC, datetime

from bio_neb.investigation.investigation import RetrievedSource
from bio_neb.investigation.investigation_request import (
    StatedAssembly,
    StatedPosition,
    literal_version_form,
)
from bio_neb.investigation.investigation_sources import (
    AssemblyReport,
    look_up_stated_position,
    normalize_assembly_report,
    read_assembly_report,
)
from bio_neb.providers.eva_investigation import fetch_source
from bio_neb.providers.ncbi_datasets import MetadataResponse
from bio_neb.providers.ncbi_investigation import NcbiInvestigationSourceAdapter
from tests.location_retained_bodies import (
    GALGAL5_ANY_VERSION_REPORT,
    GALGAL5_PLAIN_REPORT,
    GRCH38_REPORT,
)

GALGAL5 = "GCF_000002315.4"
REPORT = (
    "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
    f"{GALGAL5}/dataset_report"
)
ANY_VERSION = f"{REPORT}?filters.assembly_version=all_assemblies"


class _Answers:
    """NCBI's answers by URL, in the order they are asked; an unasked URL fails."""

    def __init__(self, answers: dict[str, bytes]) -> None:
        self.answers = answers
        self.urls: list[str] = []
        self.contact_count = 0

    def get(self, url: str, *, max_bytes: int, accept=None) -> MetadataResponse:
        self.contact_count += 1
        self.urls.append(url)
        return MetadataResponse(
            request_identity=url,
            citation_url=url,
            status_code=200,
            body=self.answers[url],
            source_version=None,
            observed_at=datetime(2026, 10, 21, tzinfo=UTC),
            elapsed_time_ms=1,
        )


def _galgal5() -> _Answers:
    return _Answers(
        {REPORT: GALGAL5_PLAIN_REPORT, ANY_VERSION: GALGAL5_ANY_VERSION_REPORT}
    )


def _retained(response) -> tuple[RetrievedSource, bytes]:
    source = RetrievedSource.create(
        operation="fetch_assembly_report",
        request_identity=response.request_identity,
        status_code=response.status_code,
        observed_at=response.observed_at,
        elapsed_time_ms=response.elapsed_time_ms,
        raw_response=response.body,
    )
    return source, response.body


class ReplacedAssemblyTests(unittest.TestCase):
    def test_the_recorded_bodies_are_lane_rf_s_live_l5_bytes(self) -> None:
        self.assertEqual(
            hashlib.sha256(GALGAL5_ANY_VERSION_REPORT).hexdigest(),
            "b01d9abc13a7ca27f8fda43ee5f518812d49c4f55db55143099aa51b041f4725",
        )
        self.assertEqual(
            hashlib.sha256(GALGAL5_PLAIN_REPORT).hexdigest()[:8], "44136fa3"
        )

    def test_a_replaced_assembly_s_report_is_asked_again_for_any_version(self):
        answers = _galgal5()
        response = fetch_source(
            NcbiInvestigationSourceAdapter(answers), "fetch_assembly_report", GALGAL5
        )
        self.assertEqual(answers.urls, [REPORT, ANY_VERSION])
        self.assertEqual(response.request_identity, ANY_VERSION)
        self.assertEqual(response.body, GALGAL5_ANY_VERSION_REPORT)
        # The plain report's ``{}`` is the operation's earlier answer, kept for the
        # Tool to retain and show.
        [plain] = response.preceding
        self.assertEqual(
            (plain.request_identity, plain.body), (REPORT, GALGAL5_PLAIN_REPORT)
        )

    def test_a_current_assembly_s_report_is_one_get(self) -> None:
        url = REPORT.replace(GALGAL5, "GCF_000001405.40")
        answers = _Answers({url: GRCH38_REPORT})
        NcbiInvestigationSourceAdapter(answers).fetch_assembly_report(
            "GCF_000001405.40"
        )
        self.assertEqual(answers.urls, [url])

    def test_the_any_version_report_names_the_assembly_its_pair_and_status(self):
        self.assertEqual(
            read_assembly_report(json.loads(GALGAL5_ANY_VERSION_REPORT), GALGAL5),
            AssemblyReport(
                accession=GALGAL5,
                name="Gallus_gallus-5.0",
                synonym=None,
                paired_accession="GCA_000002315.3",
                status="suppressed",
                taxid=9031,
            ),
        )

    def test_a_position_lookup_on_the_replaced_assembly_reads_it(self) -> None:
        adapter = NcbiInvestigationSourceAdapter(_galgal5())

        def fetch(operation: str, key: str):
            if operation != "fetch_assembly_report":
                # The lookup goes on past the assembly; no sequence names are kept.
                return (
                    RetrievedSource.create(
                        operation=operation,
                        request_identity=f"https://source.example/{key}",
                        status_code=404,
                        observed_at="2026-10-21T00:00:00Z",
                        elapsed_time_ms=1,
                        raw_response=b"{}",
                    ),
                    b"{}",
                )
            return _retained(fetch_source(adapter, operation, key))

        result = look_up_stated_position(
            position=StatedPosition(contig="1", position=100, ref="A", alt="G"),
            stated_assembly=StatedAssembly(
                name="GCA_000002315.3", cited_line="On GCA_000002315.3."
            ),
            rsid=None,
            requested_rsid=None,
            assembly_accession=GALGAL5,
            fetch=fetch,
            # A chicken Run: the model's choice would be EVA; the test reads only
            # the assembly report.
            source="eva",
        )

        self.assertNotEqual(
            result.conflicts,
            ("NCBI's report of the Run's assembly was malformed.",),
        )
        self.assertEqual(result.location.run_assembly_name, "Gallus_gallus-5.0")
        self.assertEqual(result.location.run_assembly_status, "suppressed")
        self.assertEqual(
            result.location.run_assembly_paired_accession, "GCA_000002315.3"
        )
        # The stated paired accession is the Run's version: the lookup went on to
        # place the position on the Run's sequences. The ending states what the
        # source answered, in its own words, and holds no instruction.
        self.assertEqual(
            result.conflicts,
            ("NCBI's sequence names of Gallus_gallus-5.0 answered HTTP 404: {}.",),
        )
        self.assertEqual(result.origin.code, "source_answered")
        self.assertEqual(result.origin.status_code, 404)
        self.assertEqual(result.origin.words_sent, "{}")
        self.assertEqual(result.assessment.rationale, result.conflicts[0])

    def test_the_assembly_tool_confirms_a_replaced_assembly_with_its_status(self):
        response = fetch_source(
            NcbiInvestigationSourceAdapter(_galgal5()), "fetch_assembly_report", GALGAL5
        )
        source, body = _retained(response)

        result = normalize_assembly_report(
            expected_assembly_accession=GALGAL5,
            expected_assembly_name="Gallus_gallus-5.0",
            source=source,
            raw_response=body,
        )

        self.assertEqual(result.status, "compatible")
        self.assertEqual(result.observed_assembly_status, "suppressed")
        self.assertEqual(result.source.request_identity, ANY_VERSION)


class StatedVersionTests(unittest.TestCase):
    """A stated version is the Run's literally by one of its own names, its paired
    accession among them; any other spelling, a version without its patch suffix or in
    another case, is the model's to name (#471 Q3 a, Q4 a; #417's patch strip and case
    folding are superseded by ADR 0049). Another version of the pair is not the Run's."""

    def test_each_name_of_the_run_s_version_is_found_by_its_form(self) -> None:
        run = ("GRCh38.p14", "hg38", "GCF_000001405.40", "GCA_000001405.29")
        cases = {
            "GRCh38.p14": "assembly_name",
            "hg38": "synonym",
            "GCF_000001405.40": "accession",
            "GCA_000001405.29": "paired_accession",
            "GRCh38": None,
            "grch38.P14": None,
            "HG38": None,
            "GCA_000001405.28": None,
            "GCF_000001405.39": None,
            "hg19": None,
        }
        for stated, form in cases.items():
            with self.subTest(stated=stated):
                self.assertEqual(literal_version_form(stated, *run), form)


if __name__ == "__main__":
    unittest.main()
