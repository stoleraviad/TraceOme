"""A paper's open copy downloads through the product source operations: Europe PMC
resolves it, the PMC open-data bucket and Europe PMC give its documents, each with the
digest a Run's artifact store names it by, each GET on its own operation's surface."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
import urllib.parse
import uuid
import zipfile
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch

from bio_neb.failure import (
    ContactLimitOverrides,
    FailureOverrides,
    FailureSurface,
    FailureTuning,
    limits_for,
    run_failure_tuning,
)
from bio_neb.providers.paper_sources import (
    PaperDownloadStopped,
    PaperSourceAdapter,
    download_paper,
    parse_pmc_open_data_listing,
    stated_identifiers,
)
from tests.source_limits_support import declaring

DOI = "10.1371/journal.pgen.1006651"
PMCID = "PMC5352138"
REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
SEARCH = (
    f"{REST}/search?query=DOI%3A%2210.1371%2Fjournal.pgen.1006651%22"
    "&format=json&resultType=lite&pageSize=25&cursorMark=%2A"
)
LISTING = f"{BUCKET}/?list-type=2&prefix={PMCID}."
JATS = f"{BUCKET}/PMC5352138.1/PMC5352138.1.xml"
TABLE = f"{BUCKET}/PMC5352138.1/pgen.1006651.s001.csv"
FIGURE = f"{BUCKET}/PMC5352138.1/pgen.1006651.s002.pdf"
FULL_TEXT = f"{REST}/{PMCID}/fullTextXML"
ZIP = f"{REST}/{PMCID}/supplementaryFiles"

ARTICLE = (
    b'<article xmlns:xlink="http://www.w3.org/1999/xlink"><body><p>A variant.</p>'
    b"</body><back><supplementary-material>"
    b'<media xlink:href="pgen.1006651.s001.csv"/></supplementary-material>'
    b'<supplementary-material><media xlink:href="pgen.1006651.s002.pdf"/>'
    b"</supplementary-material></back></article>"
)
TABLE_BYTES = b"Chromosome,Position,Ref,Alt\n10,68587027,A,G\n"
FIGURE_BYTES = b"%PDF-1.4 figure"
GENERIC = "binary/octet-stream"


def zipped(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def listing(keys: list[str]) -> bytes:
    contents = "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"{contents}</ListBucketResult>"
    ).encode()


def search(*results: dict) -> bytes:
    return json.dumps({"resultList": {"result": list(results)}}).encode()


def dog() -> dict[str, tuple[int, bytes, str]]:
    """The dog paper's answers, as the hosts served them (their content types)."""
    return {
        SEARCH: (
            200,
            search({"pmcid": PMCID, "doi": DOI, "isOpenAccess": "Y"}),
            "application/json;charset=UTF-8",
        ),
        LISTING: (
            200,
            listing(
                [
                    "PMC5352138.1/PMC5352138.1.xml",
                    "PMC5352138.1/pgen.1006651.g001.jpg",
                    "PMC5352138.1/pgen.1006651.s001.csv",
                    "PMC5352138.1/pgen.1006651.s002.pdf",
                ]
            ),
            "application/xml",
        ),
        JATS: (200, ARTICLE, GENERIC),
        TABLE: (200, TABLE_BYTES, GENERIC),
        FIGURE: (200, FIGURE_BYTES, GENERIC),
    }


def hosts(answers: dict[str, tuple[int, bytes, str]]):
    """An adapter over recorded answers: a URL never recorded answers 404, and an
    answer may be an exception the connection raises."""
    asked: list[str] = []

    def opener(request, timeout):
        url = request.full_url
        asked.append(url)
        answer = answers.get(url, (404, b"", "text/plain"))
        if isinstance(answer, list):
            answer = answer.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        status, body, kind = answer
        headers = Message()
        headers["Content-Type"] = kind
        if status >= 400:
            raise urllib.error.HTTPError(url, status, "", headers, io.BytesIO(body))
        response = MagicMock(status=status, headers=headers)
        response.__enter__.return_value = response
        response.read.side_effect = lambda size=-1: body if size < 0 else body[:size]
        return response

    return PaperSourceAdapter(opener=opener), asked


def journey_digest(content: bytes) -> str:
    """The digest a Run's artifact store names these bytes by."""
    from bio_neb.journey import Journey

    with tempfile.TemporaryDirectory() as directory:
        journey = Journey.create(Path(directory))
        event = journey.record_with_artifact(
            "JourneyStarted", {}, content, "application/octet-stream"
        )
        (reference,) = event["artifact_references"]
        return reference["artifact_id"].removeprefix("sha256:")


NO_BACKOFF = FailureTuning(operation=FailureOverrides(backoff_seconds=0))


class DownloadTests(unittest.TestCase):
    def test_a_pmc_paper_comes_from_the_bucket_each_document_with_its_digest(self):
        adapter, asked = hosts(dog())

        download = download_paper(f"doi:{DOI}", adapter)

        # The bucket gave the article and every file it names: no Europe PMC copy.
        self.assertEqual(asked, [SEARCH, LISTING, JATS, TABLE, FIGURE])
        self.assertEqual(adapter.contact_count, 5)
        self.assertEqual((download.resolved.pmcid, download.resolved.doi), (PMCID, DOI))
        self.assertEqual(
            [(d.name, d.url, d.route, d.role) for d in download.documents],
            [
                ("PMC5352138.1.xml", JATS, "pmc_open_data", "article"),
                ("pgen.1006651.s001.csv", TABLE, "pmc_open_data", "supplement"),
                ("pgen.1006651.s002.pdf", FIGURE, "pmc_open_data", "supplement"),
            ],
        )
        self.assertEqual(
            [d.content for d in download.documents],
            [ARTICLE, TABLE_BYTES, FIGURE_BYTES],
        )
        self.assertEqual({d.media_type for d in download.documents}, {GENERIC})
        for document in download.documents:
            with self.subTest(document=document.name):
                self.assertEqual(document.sha256, journey_digest(document.content))
        self.assertEqual(download.unfetched, ())
        self.assertIsNone(download.stop)

    def test_without_the_bucket_europe_pmc_gives_the_article_and_the_zip(self):
        answers = dog()
        answers[LISTING] = (200, listing([]), "application/xml")
        answers[FULL_TEXT] = (200, ARTICLE, "application/xml")
        supplements = zipped({"pgen.1006651.s001.csv": TABLE_BYTES})
        answers[ZIP] = (200, supplements, "application/zip")
        adapter, asked = hosts(answers)

        download = download_paper(f"doi:{DOI}", adapter)

        self.assertEqual(asked, [SEARCH, LISTING, FULL_TEXT, ZIP])
        self.assertEqual(
            [(d.name, d.route, d.role, d.content) for d in download.documents],
            [
                ("fullTextXML", "europe_pmc_full_text", "article", ARTICLE),
                (
                    "supplementaryFiles",
                    "europe_pmc_supplementary_files",
                    "supplement_archive",
                    supplements,
                ),
            ],
        )

    def test_a_named_file_the_bucket_lacks_is_absent_and_the_zip_is_asked(self):
        answers = dog()
        del answers[FIGURE]
        answers[ZIP] = (200, zipped({"x.csv": b"a"}), "application/zip")
        adapter, asked = hosts(answers)

        download = download_paper(f"doi:{DOI}", adapter)

        self.assertEqual(asked, [SEARCH, LISTING, JATS, TABLE, FIGURE, ZIP])
        self.assertEqual(
            [(u.name, u.url, u.route, u.reason) for u in download.unfetched],
            [
                (
                    "pgen.1006651.s002.pdf",
                    FIGURE,
                    "pmc_open_data",
                    "HTTP 404: the host holds none",
                )
            ],
        )
        self.assertEqual(download.documents[-1].role, "supplement_archive")

    def test_a_named_file_the_listing_lacks_is_unfetched_with_its_reason(self):
        # Spec r1 SP1: a file the article names keeps its name and reason.
        answers = dog()
        answers[LISTING] = (
            200,
            listing(
                ["PMC5352138.1/PMC5352138.1.xml", "PMC5352138.1/pgen.1006651.s001.csv"]
            ),
            "application/xml",
        )
        adapter, asked = hosts(answers)

        download = download_paper(f"doi:{DOI}", adapter)

        # The zip is asked for the missing file, and answers 404.
        self.assertEqual(asked, [SEARCH, LISTING, JATS, TABLE, ZIP])
        self.assertEqual(
            [(u.name, u.url, u.reason) for u in download.unfetched],
            [
                ("pgen.1006651.s002.pdf", None, "not in the bucket listing"),
                ("supplementaryFiles", ZIP, "HTTP 404: the host holds none"),
            ],
        )
        self.assertIsNone(download.stop)

    def test_names_from_europe_pmc_s_full_text_are_unfetched_when_no_copy_holds_them(
        self,
    ):
        # Spec r2 SP1: the full-text fallback's names keep their reasons too.
        answers = dog()
        answers[LISTING] = (200, listing([]), "application/xml")
        answers[FULL_TEXT] = (200, ARTICLE, "application/xml")
        adapter, asked = hosts(answers)

        download = download_paper(f"doi:{DOI}", adapter)

        self.assertEqual(asked, [SEARCH, LISTING, FULL_TEXT, ZIP])
        self.assertEqual(
            [(u.name, u.url, u.reason) for u in download.unfetched],
            [
                ("pgen.1006651.s001.csv", None, "not in the bucket listing"),
                ("pgen.1006651.s002.pdf", None, "not in the bucket listing"),
                ("supplementaryFiles", ZIP, "HTTP 404: the host holds none"),
            ],
        )

    def test_a_paper_without_a_pmcid_is_resolved_and_nothing_is_downloaded(self):
        answers = {SEARCH: (200, search(), "application/json")}
        adapter, asked = hosts(answers)

        download = download_paper(f"https://doi.org/{DOI}", adapter)

        self.assertEqual(asked, [SEARCH])
        self.assertEqual(
            (download.resolved.doi, download.resolved.resolved_by),
            (DOI, "user_supplied"),
        )
        self.assertEqual(download.documents, ())
        self.assertIn("no PMCID", download.note)

    def test_a_title_search_gives_every_row_copied_in_source_order(self):
        """A reference that states no id resolves to no paper by code: every row
        Europe PMC lists comes back as a candidate, as listed (ADR 0049: no ranking
        or cut on a source's answer; #420 removed the title-overlap scoring)."""
        title = "A de novo variant in the ASPRV1 gene in a dog with ichthyosis"
        query = f'TITLE:"{title}"'
        url = (
            f"{REST}/search?query={urllib.parse.quote_plus(query)}"
            "&format=json&resultType=lite&pageSize=25&cursorMark=%2A"
        )
        answers = {
            url: (
                200,
                search(
                    # The row whose title the reference writes comes second: code
                    # neither moves it first nor settles on it.
                    {"pmcid": "PMC2222222", "title": "A dog gene panel"},
                    {"pmcid": "PMC1111111", "title": title, "authorString": "Smith J."},
                    {"doi": "10.1000/x", "title": "Ichthyosis in cattle"},
                ),
                "application/json",
            )
        }
        adapter, asked = hosts(answers)

        download = download_paper(title, adapter)

        self.assertEqual(asked, [url])
        self.assertIsNone(download.resolved)
        self.assertEqual(
            [(c.pmcid, c.doi, c.title, c.authors) for c in download.candidates],
            [
                ("PMC2222222", None, "A dog gene panel", None),
                ("PMC1111111", None, title, "Smith J."),
                (None, "10.1000/x", "Ichthyosis in cattle", None),
            ],
        )
        self.assertEqual(download.documents, ())


class StopTests(unittest.TestCase):
    """A failure the policy gives up on ends the download: no later GET to any host
    (owner, #419 Q7)."""

    def stopped(self, answers):
        adapter, asked = hosts(answers)
        with (
            run_failure_tuning(NO_BACKOFF),
            self.assertRaises(PaperDownloadStopped) as stopped,
        ):
            download_paper(f"doi:{DOI}", adapter)
        return stopped.exception, asked

    def test_a_403_is_not_retried_and_nothing_more_is_asked(self):
        answers = dog()
        answers[TABLE] = (403, b"Forbidden", "text/plain")

        error, asked = self.stopped(answers)

        self.assertEqual(asked, [SEARCH, LISTING, JATS, TABLE])
        self.assertEqual(error.failure_class, "rejected")
        stop = error.download.stop
        self.assertEqual(
            (stop.host, stop.operation, stop.failure_class),
            ("pmc-oa-opendata.s3.amazonaws.com", "pmc_open_data_file", "rejected"),
        )
        self.assertIn("HTTP 403", stop.detail)
        # What it had before the stop stays the download's, and each named file it
        # does not hold says why (Spec r1 SP1).
        self.assertEqual(
            [d.name for d in error.download.documents], ["PMC5352138.1.xml"]
        )
        self.assertEqual(
            [(u.name, u.url, u.reason[:25]) for u in error.download.unfetched],
            [
                ("pgen.1006651.s001.csv", TABLE, "failed: Source rejected i"),
                ("pgen.1006651.s002.pdf", FIGURE, "not requested: the downlo"),
            ],
        )

    def test_a_stop_keeps_a_named_file_the_listing_lacks(self):
        # Spec r3 SP1: a name the listing lacks is accounted before a later GET stops.
        answers = dog()
        answers[LISTING] = (
            200,
            listing(
                ["PMC5352138.1/PMC5352138.1.xml", "PMC5352138.1/pgen.1006651.s001.csv"]
            ),
            "application/xml",
        )
        answers[TABLE] = (403, b"Forbidden", "text/plain")

        error, asked = self.stopped(answers)

        self.assertEqual(asked, [SEARCH, LISTING, JATS, TABLE])
        self.assertEqual(
            [(u.name, u.url, u.reason[:25]) for u in error.download.unfetched],
            [
                ("pgen.1006651.s002.pdf", None, "not in the bucket listing"),
                ("pgen.1006651.s001.csv", TABLE, "failed: Source rejected i"),
            ],
        )

    def test_a_429_is_retried_by_the_policy_then_ends_the_download(self):
        answers = dog()
        answers[TABLE] = [(429, b"slow down", "text/plain")] * 3

        error, asked = self.stopped(answers)

        self.assertEqual(asked, [SEARCH, LISTING, JATS, TABLE, TABLE, TABLE])
        self.assertEqual(error.failure_class, "exhausted")
        self.assertEqual(error.download.stop.host, "pmc-oa-opendata.s3.amazonaws.com")

    def test_a_429_answered_on_a_retry_downloads_on(self):
        answers = dog()
        answers[TABLE] = [(429, b"slow down", "text/plain"), dog()[TABLE]]
        adapter, asked = hosts(answers)

        with run_failure_tuning(NO_BACKOFF):
            download = download_paper(f"doi:{DOI}", adapter)

        self.assertEqual(asked, [SEARCH, LISTING, JATS, TABLE, TABLE, FIGURE])
        self.assertEqual(len(download.documents), 3)

    def test_a_lost_connection_after_its_retries_ends_the_download(self):
        answers = dog()
        answers[LISTING] = [TimeoutError("no route")] * 3

        error, asked = self.stopped(answers)

        self.assertEqual(asked, [SEARCH, LISTING, LISTING, LISTING])
        self.assertEqual(error.failure_class, "exhausted")
        self.assertEqual(error.download.stop.operation, "pmc_open_data_listing")
        self.assertEqual(error.download.documents, ())


class OperationTests(unittest.TestCase):
    """Each GET is a registered operation on its own surface, with the limits the
    owner set (#419 Q6)."""

    OPERATIONS = {
        "europe_pmc_search",
        "europe_pmc_full_text",
        "europe_pmc_supplementary_files",
        "pmc_open_data_listing",
        "pmc_open_data_file",
    }

    def test_download_paper_is_one_tool_whose_gets_run_on_their_own_surfaces(self):
        from bio_neb.agentic_journey.source_contacts import (
            contact_surfaces,
            source_tools,
        )

        self.assertEqual(contact_surfaces("download_paper"), self.OPERATIONS)
        self.assertLessEqual({"download_paper", *self.OPERATIONS}, source_tools())

    def test_a_call_makes_16_contacts_and_a_document_may_be_64_mib(self):
        mib = 1024 * 1024
        resolved = {
            name: limits_for(FailureSurface(scope="tool", name=name)).effective
            for name in ("download_paper", *self.OPERATIONS)
        }
        self.assertEqual(resolved["download_paper"].contacts, 16)
        self.assertEqual(
            {name: limits.response_bytes for name, limits in resolved.items()},
            {
                "download_paper": 4 * mib,
                "europe_pmc_search": 4 * mib,
                "pmc_open_data_listing": 4 * mib,
                "europe_pmc_full_text": 64 * mib,
                "europe_pmc_supplementary_files": 64 * mib,
                "pmc_open_data_file": 64 * mib,
            },
        )

    def test_a_document_over_its_bound_ends_the_download_unread(self):
        adapter, asked = hosts(dog())
        small = {"pmc_open_data_file": ContactLimitOverrides(response_bytes=20)}
        with (
            declaring(small),
            self.assertRaises(PaperDownloadStopped) as stopped,
        ):
            download_paper(f"doi:{DOI}", adapter)

        self.assertEqual(asked, [SEARCH, LISTING, JATS])
        self.assertEqual(stopped.exception.failure_class, "rejected")
        self.assertIn("byte bound", stopped.exception.download.stop.detail)


class RunJournalTests(unittest.TestCase):
    """Inside a Run's source scope every GET is a row of the contact journal, and the
    Tool's count is enforced."""

    def downloaded(self, answers, **limits):
        from bio_neb.agentic_journey.lifecycle import ActivityLifecycle
        from bio_neb.agentic_journey.source_contacts import observed_source_operation
        from bio_neb.journey import Journey
        from bio_neb.recording import processing_pass, read_evidence

        adapter, asked = hosts(answers)
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory))
            origin = journey.record("JourneyStarted", {})
            outcome = None
            with (
                declaring(limits),
                run_failure_tuning(NO_BACKOFF),
                processing_pass(journey),
            ):
                lifecycle = ActivityLifecycle(journey, causal_head=origin["event_id"])
                activity = lifecycle.start(
                    adapter="ncbi_datasets", stage="resolve_taxon"
                )
                try:
                    outcome = observed_source_operation(
                        activity,
                        str(uuid.uuid4()),
                        "call-1",
                        "download_paper",
                        lambda: download_paper(f"doi:{DOI}", adapter),
                    )()
                except PaperDownloadStopped as error:
                    outcome = error
            return outcome, asked, read_evidence(journey)

    def test_each_get_is_a_journal_row_finished_as_it_ended(self):
        download, asked, events = self.downloaded(dog())

        started = [
            e["payload"]
            for e in events
            if e["event_type"] == "SourcePhysicalContactStarted"
        ]
        finished = [
            e["payload"]["disposition"]
            for e in events
            if e["event_type"] == "SourcePhysicalContactFinished"
        ]
        self.assertEqual([s["request_identity"] for s in started], asked)
        self.assertEqual({s["method"] for s in started}, {"GET"})
        self.assertEqual(finished, ["succeeded"] * 5)
        (scope,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "SourceContactScopeStarted"
        ]
        self.assertEqual(scope["tool_name"], "download_paper")
        self.assertEqual(len(download.documents), 3)

    def test_the_call_s_contact_count_refuses_the_get_past_it(self):
        error, asked, events = self.downloaded(
            dog(), download_paper=ContactLimitOverrides(contacts=3)
        )

        self.assertIsInstance(error, PaperDownloadStopped)
        self.assertEqual(asked, [SEARCH, LISTING, JATS])
        (breach,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "SourceContactLimitReached"
        ]
        self.assertEqual(
            (breach["limit"], breach["bound"], breach["limits"]["surface"]["name"]),
            ("contacts", 3, "download_paper"),
        )
        self.assertEqual(error.failure_class, "exhausted")


class MalformedAnswerTests(unittest.TestCase):
    """Standards r1 S1: a search or listing answer that does not read as one fails its
    own operation as `semantic`, recorded on its surface, and ends the download."""

    def stopped(self, answers):
        adapter, asked = hosts(answers)
        with self.assertRaises(PaperDownloadStopped) as stopped:
            download_paper(f"doi:{DOI}", adapter)
        return stopped.exception, asked

    def test_a_search_answer_that_is_no_result_list_is_semantic(self):
        for body in (
            b"not json",
            b'{"resultList": {"result": {}}}',
            b'{"resultList": {"result": ["x"]}}',
            json.dumps({"resultList": {"result": [{"pmcid": 5352138}]}}).encode(),
            json.dumps({"resultList": {"result": [{"pmcid": "5352138"}]}}).encode(),
        ):
            with self.subTest(body=body):
                answers = dog()
                answers[SEARCH] = (200, body, "application/json")

                error, asked = self.stopped(answers)

                self.assertEqual(asked, [SEARCH])
                self.assertEqual(error.failure_class, "semantic")
                self.assertEqual(
                    (error.download.stop.operation, error.evidence["surface"]),
                    ("europe_pmc_search", "europe_pmc_search"),
                )

    def test_a_listing_that_is_no_complete_bucket_listing_is_semantic(self):
        for body in (
            b"not XML",
            b"<html>blocked</html>",
            b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            b"<IsTruncated>true</IsTruncated></ListBucketResult>",
        ):
            with self.subTest(body=body):
                answers = dog()
                answers[LISTING] = (200, body, "application/xml")

                error, asked = self.stopped(answers)

                self.assertEqual(asked, [SEARCH, LISTING])
                self.assertEqual(error.failure_class, "semantic")
                self.assertEqual(error.download.stop.operation, "pmc_open_data_listing")

    def test_an_empty_listing_and_missing_optional_ids_stay_lawful(self):
        answers = dog()
        answers[SEARCH] = (200, search({"pmcid": PMCID}), "application/json")
        answers[LISTING] = (200, listing([]), "application/xml")
        answers[FULL_TEXT] = (200, ARTICLE, "application/xml")
        adapter, _ = hosts(answers)

        download = download_paper(f"doi:{DOI}", adapter)

        self.assertEqual(
            (download.resolved.pmcid, download.resolved.doi), (PMCID, None)
        )
        self.assertEqual(download.article.route, "europe_pmc_full_text")

    def test_a_malformed_answer_is_recorded_on_its_surface_in_a_run(self):
        answers = dog()
        answers[LISTING] = (200, b"not XML", "application/xml")
        error, asked, events = RunJournalTests.downloaded(None, answers)

        self.assertIsInstance(error, PaperDownloadStopped)
        self.assertEqual(asked, [SEARCH, LISTING])
        (failure,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "SourceOperationFailureRecorded"
        ]
        self.assertEqual(
            (failure["failure_class"], failure["policy"]["surface"]["name"]),
            ("semantic", "pmc_open_data_listing"),
        )


class ParsingTests(unittest.TestCase):
    def test_open_data_listing_gives_the_latest_article_version_files(self):
        # S3 ListObjectsV2 shape; the per-article folder layout is the assumption the
        # contact run checks.
        listing = (
            b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            b"<Name>pmc-oa-opendata</Name><Prefix>PMC5352138.</Prefix>"
            b"<IsTruncated>false</IsTruncated>"
            b"<Contents><Key>PMC5352138.1/PMC5352138.1.xml</Key><Size>90</Size></Contents>"
            b"<Contents><Key>PMC5352138.1/pgen.1006651.s001.xlsx</Key><Size>9</Size></Contents>"
            b"<Contents><Key>PMC5352138.2/PMC5352138.2.xml</Key><Size>91</Size></Contents>"
            b"<Contents><Key>PMC5352138.2/pgen.1006651.s001.xlsx</Key><Size>9</Size></Contents>"
            b"<Contents><Key>PMC5352138.2/pgen.1006651.g001.jpg</Key><Size>9</Size></Contents>"
            b"</ListBucketResult>"
        )

        self.assertEqual(
            parse_pmc_open_data_listing(listing, "PMC5352138"),
            [
                "PMC5352138.2/PMC5352138.2.xml",
                "PMC5352138.2/pgen.1006651.s001.xlsx",
                "PMC5352138.2/pgen.1006651.g001.jpg",
            ],
        )
        self.assertEqual(
            parse_pmc_open_data_listing(b"<ListBucketResult/>", "PMC1"), []
        )

    def test_stated_identifiers(self):
        self.assertEqual(
            stated_identifiers(
                "see https://doi.org/10.1038/s41587-019-0074-6. PMID: 30858580"
            ),
            {"doi": "10.1038/s41587-019-0074-6", "pmid": "30858580"},
        )
        self.assertEqual(stated_identifiers("pmc6500473"), {"pmcid": "PMC6500473"})

    def test_a_free_text_title_with_one_europe_pmc_hit_is_its_one_candidate(self):
        """Even a single row is the model's to choose: code settles no title (ADR
        0049; migrated from the one-hit rule #420 retired)."""
        title = "A de novo variant in the ASPRV1 gene in a dog with ichthyosis"
        url = (
            f"{REST}/search?query=TITLE%3A%22A+de+novo+variant+in+the+ASPRV1+gene+in+"
            "a+dog+with+ichthyosis%22&format=json&resultType=lite&pageSize=25&cursorMark=%2A"
        )
        adapter, _ = hosts(
            {
                url: (
                    200,
                    search({"pmid": "28249031", "pmcid": None}),
                    "application/json",
                )
            }
        )

        download = download_paper(title, adapter)

        self.assertIsNone(download.resolved)
        self.assertEqual([c.pmid for c in download.candidates], ["28249031"])
        self.assertEqual(download.documents, ())


class DeveloperEntryTests(unittest.TestCase):
    def test_the_entry_writes_each_file_with_its_digests_and_the_gets_sent(self):
        import contextlib
        import hashlib

        from bio_neb.devtools import paper_download

        adapter, asked = hosts(dog())
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(paper_download, "PaperSourceAdapter", return_value=adapter),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            out = Path(directory)
            code = paper_download.main([f"doi:{DOI}", "--out", str(out)])
            record = json.loads((out / "download.json").read_text())
            table = (out / "files" / "pgen.1006651.s001.csv").read_bytes()

        self.assertEqual(code, 0)
        self.assertEqual(table, TABLE_BYTES)
        self.assertEqual((record["sent"], record["stop"]), (len(asked), None))
        self.assertEqual(record["resolved"]["pmcid"], PMCID)
        self.assertEqual(
            [(d["name"], d["bytes"], d["md5"]) for d in record["documents"]],
            [
                ("PMC5352138.1.xml", len(ARTICLE), hashlib.md5(ARTICLE).hexdigest()),
                (
                    "pgen.1006651.s001.csv",
                    len(TABLE_BYTES),
                    hashlib.md5(TABLE_BYTES).hexdigest(),
                ),
                (
                    "pgen.1006651.s002.pdf",
                    len(FIGURE_BYTES),
                    hashlib.md5(FIGURE_BYTES).hexdigest(),
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
