"""The one source transport family: every public source GET, NCBI's, EVA's and a
reference download's, leaves through one no-redirect opener with one user agent and
one host check, within the timeout and the byte bound of its surface."""

from __future__ import annotations

import io
import unittest
import urllib.error
from contextlib import contextmanager
from email.message import Message
from unittest.mock import MagicMock, patch

from bio_neb.failure import ProviderError
from bio_neb.providers.eva_investigation import UrllibEvaTransport
from bio_neb.providers.ncbi_datasets import UrllibNcbiTransport
from bio_neb.providers.source_transport import (
    USER_AGENT,
    SourceTransport,
)
from bio_neb.reference_discovery.reference_identity_tool import ReferenceAdapterError

NCBI_REPORT = (
    "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
    "GCF_000002315.4/dataset_report"
)
EVA_CLUSTERED = (
    "https://www.ebi.ac.uk/eva/webservices/identifiers/v1/clustered-variants/9"
)


def _answered(body: bytes = b"{}", status: int = 200):
    response = MagicMock(status=status)
    response.headers = {}
    response.read.side_effect = lambda size=-1: body if size < 0 else body[:size]
    return response


class OneFamilyTests(unittest.TestCase):
    def test_ncbi_and_eva_send_with_one_user_agent_through_a_no_redirect_opener(
        self,
    ) -> None:
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.return_value.__enter__.return_value = _answered()
            UrllibNcbiTransport().get(NCBI_REPORT, max_bytes=10)
            UrllibEvaTransport().get(EVA_CLUSTERED)
        sent = [call.args[0] for call in build.return_value.open.call_args_list]
        self.assertEqual(
            [request.get_header("User-agent") for request in sent],
            [USER_AGENT, USER_AGENT],
        )
        # Each transport builds its opener with the family's one redirect refusal.
        handlers = {type(call.args[0]) for call in build.call_args_list}
        self.assertEqual(len(handlers), 1)
        self.assertEqual(
            [call.kwargs["timeout"] for call in build.return_value.open.call_args_list],
            [15.0, 15.0],
        )

    def test_a_url_outside_the_family_hosts_is_refused_before_contact(self) -> None:
        refused = (
            "https://evil.example/datasets/v2/genome",
            "http://api.ncbi.nlm.nih.gov/datasets/v2/genome",
            "https://user:pw@api.ncbi.nlm.nih.gov/datasets/v2/genome",
            "https://www.ebi.ac.uk/ena/browser/api/9",
        )
        with patch("urllib.request.build_opener") as build:
            transport = SourceTransport()
            for url in refused:
                with self.subTest(url=url), self.assertRaises(ProviderError) as error:
                    transport.exchange(url, max_bytes=10)
                self.assertEqual(error.exception.failure_class, "rejected")
                self.assertIn("outside the source hosts", error.exception.detail)
        build.return_value.open.assert_not_called()
        self.assertEqual(transport.contact_count, 0)

    def test_galaxy_s_plain_http_datacache_is_admitted_and_nothing_else_plain(
        self,
    ) -> None:
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.return_value.__enter__.return_value = _answered(
                b"loc"
            )
            answer = SourceTransport().exchange(
                "http://datacache.galaxyproject.org/vgp/config/bwa_mem_index.loc",
                max_bytes=10,
            )
        self.assertEqual((answer.status, answer.body), (200, b"loc"))

    def test_an_http_status_is_an_answer_and_a_redirect_is_never_followed(
        self,
    ) -> None:
        moved = urllib.error.HTTPError(
            NCBI_REPORT, 301, "Moved", Message(), io.BytesIO(b"elsewhere")
        )
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.side_effect = [moved, moved]
            answer = SourceTransport().exchange(NCBI_REPORT, max_bytes=100)
            with self.assertRaises(ReferenceAdapterError) as refused:
                UrllibNcbiTransport().get(NCBI_REPORT, max_bytes=100)
        self.assertEqual((answer.status, answer.body), (301, b"elsewhere"))
        # NCBI's view refuses the contact but names it the source's answer, for
        # its reader; nothing is followed.
        self.assertEqual(refused.exception.failure.code, "source_answered")
        self.assertFalse(refused.exception.failure.retryable)
        self.assertEqual(build.return_value.open.call_count, 2)

    def test_a_lost_connection_is_transient(self) -> None:
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.side_effect = TimeoutError("slow")
            transport = SourceTransport()
            with self.assertRaises(ProviderError) as lost:
                transport.exchange(NCBI_REPORT, max_bytes=10)
        self.assertEqual(lost.exception.failure_class, "transient")
        self.assertEqual(transport.contact_count, 1)

    def test_the_opener_is_looked_up_when_a_request_is_sent(self) -> None:
        # A transport built before a test replaces the opener's `open` never sends
        # through the real one: an offline test stays contact-free.
        transport = SourceTransport()
        with patch("urllib.request.OpenerDirector.open") as sent:
            sent.return_value.__enter__.return_value = _answered(b"{}")
            answer = transport.exchange(NCBI_REPORT, max_bytes=10)
        self.assertEqual(sent.call_count, 1)
        self.assertEqual(answer.body, b"{}")

    def test_the_timeout_is_fifteen_seconds_and_never_above_thirty(self) -> None:
        with self.assertRaises(ValueError):
            SourceTransport(timeout_seconds=31)
        with self.assertRaises(ValueError):
            SourceTransport(timeout_seconds=0)

    def test_an_answer_over_its_byte_bound_is_rejected(self) -> None:
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.return_value.__enter__.return_value = _answered(
                b"x" * 11
            )
            with self.assertRaises(ProviderError) as refused:
                SourceTransport().exchange(NCBI_REPORT, max_bytes=10)
        self.assertEqual(refused.exception.failure_class, "rejected")
        self.assertIn("byte bound", refused.exception.detail)

    def test_a_download_streams_its_body_with_the_caller_s_headers(self) -> None:
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.return_value.__enter__.return_value = _answered(
                b"ACGT", status=206
            )
            transport = SourceTransport()
            with transport.stream(
                "https://ftp.ncbi.nlm.nih.gov/genomes/all/x.fna.gz",
                headers={"Range": "bytes=4-7"},
            ) as response:
                body = response.read()
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.get_header("Range"), "bytes=4-7")
        self.assertEqual(request.get_header("User-agent"), USER_AGENT)
        self.assertEqual((response.status, body), (206, b"ACGT"))
        self.assertEqual(transport.contact_count, 1)


class ContactOutcomeTests(unittest.TestCase):
    """Review r1 (S1-S3, SP1): an answer keeps its headers' meaning, a stalled error
    body is a lost contact, and a streamed body is read inside its physical contact."""

    def test_an_answers_headers_are_read_in_any_case(self) -> None:
        headers = Message()
        headers["content-length"] = "12345"
        headers["etag"] = '"source-version"'
        response = MagicMock(status=200, headers=headers)
        response.__enter__.return_value = response
        observed = UrllibNcbiTransport(opener=lambda *a, **k: response).head(
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/file.fna.gz"
        )
        self.assertEqual(observed.content_length, 12345)
        self.assertEqual(observed.source_version, '"source-version"')

    def test_a_stalled_error_body_is_a_lost_contact_the_policy_retries(self) -> None:
        from bio_neb.agentic_journey.source_contacts import call_source
        from bio_neb.failure import FailureOverrides, FailureTuning, run_failure_tuning
        from bio_neb.providers.ncbi_investigation import NcbiInvestigationSourceAdapter

        class StalledBody(io.BytesIO):
            def read(self, *args):
                raise TimeoutError("the error body stalled")

        sent = []

        def opener(request, timeout):
            sent.append(request.full_url)
            raise urllib.error.HTTPError(
                request.full_url, 503, "unavailable", Message(), StalledBody()
            )

        adapter = NcbiInvestigationSourceAdapter(UrllibNcbiTransport(opener=opener))
        tuning = FailureTuning(operation=FailureOverrides(backoff_seconds=0))
        with run_failure_tuning(tuning), self.assertRaises(ProviderError) as refused:
            call_source("fetch_refsnp", lambda: adapter.fetch_refsnp("rs429358"))
        self.assertEqual(refused.exception.failure_class, "exhausted")
        self.assertEqual(len(sent), 3)

    def test_ncbi_s_answer_about_the_request_reaches_its_reader(self) -> None:
        # NCBI's view fails the contact of an error status; an answer about the
        # request keeps its body for the reader, an access fault is refused.
        from bio_neb.providers.eva_investigation import fetch_source
        from bio_neb.providers.ncbi_investigation import NcbiInvestigationSourceAdapter

        statuses = iter((400, 410, 403))

        def opener(request, timeout):
            status = next(statuses)
            raise urllib.error.HTTPError(
                request.full_url,
                status,
                "answer",
                Message(),
                io.BytesIO(b'{"error":{"code":%d}}' % status),
            )

        adapter = NcbiInvestigationSourceAdapter(UrllibNcbiTransport(opener=opener))
        for status in (400, 410):
            answer = fetch_source(adapter, "ncbi_spdi_rsids", "NC_000019.10:1:C:T")
            self.assertEqual(
                (answer.status_code, answer.body),
                (status, b'{"error":{"code":%d}}' % status),
            )
        with self.assertRaises(ReferenceAdapterError) as refused:
            fetch_source(adapter, "ncbi_spdi_rsids", "NC_000019.10:1:C:T")
        self.assertEqual(refused.exception.failure.code, "http_error")

    def test_a_runs_lookup_reaches_the_placement_gets_by_their_text_keys(self) -> None:
        # A Run's lookup keys each GET by its text, so the taxon id arrives as "9606";
        # the Run's sources ask NCBI's listing and equivalents with it.
        from bio_neb.providers.eva_investigation import (
            PublicInvestigationSources,
            fetch_source,
        )
        from bio_neb.providers.ncbi_investigation import NcbiInvestigationSourceAdapter

        with patch("urllib.request.build_opener") as build:
            sources = PublicInvestigationSources(ncbi=NcbiInvestigationSourceAdapter())
            build.return_value.open.return_value.__enter__.return_value = _answered(
                b'{"reports":[]}'
            )
            fetch_source(sources, "ncbi_taxon_assemblies", "9606")
            fetch_source(sources, "ncbi_spdi_equivalents", "NC_000019.9:45411940:T:C")
            with self.assertRaises(ValueError):
                fetch_source(sources, "ncbi_taxon_assemblies", "09606")
        asked = [
            call.args[0].full_url for call in build.return_value.open.call_args_list
        ]

        self.assertEqual(
            [url.split("?")[0] for url in asked],
            [
                "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/taxon/9606/"
                "dataset_report",
                "https://api.ncbi.nlm.nih.gov/variation/v0/spdi/"
                "NC_000019.9:45411940:T:C/all_equivalent_contextual",
            ],
        )

    def test_a_streamed_body_is_read_inside_its_physical_contact(self) -> None:
        events = []

        @contextmanager
        def contact(*args, **kwargs):
            events.append("start")
            try:
                yield
            except TimeoutError:
                events.append("failed")
                raise
            else:
                events.append("succeeded")

        response = MagicMock(status=200, headers={})
        response.__enter__.return_value = response

        def read(*args):
            events.append("read")
            raise TimeoutError("the stream stalled")

        response.read.side_effect = read
        transport = SourceTransport(opener=lambda *args, **kwargs: response)
        with patch(
            "bio_neb.providers.source_transport.source_physical_attempt", contact
        ):
            # The caller's own read fault reaches it unchanged.
            with self.assertRaises(TimeoutError):
                with transport.stream(
                    "https://ftp.ncbi.nlm.nih.gov/genomes/all/file.fna.gz"
                ) as opened:
                    opened.read()
        self.assertEqual(events, ["start", "read", "failed"])


class OperationRegistrationTests(unittest.TestCase):
    """Each source operation's definition registers its own failure surface;
    `source_tools()` is built from those registrations and the source Tools'."""

    def test_lane_rf_s_operations_are_registered_product_operations(self) -> None:
        from bio_neb.agentic_journey.source_contacts import source_tools
        from bio_neb.providers.source_operations import SOURCE_OPERATIONS

        rf = {
            "ncbi_genome_annotation",
            "ncbi_hgvs_contextuals",
            "ncbi_spdi_equivalents",
            "ncbi_taxon_assemblies",
        }
        self.assertLessEqual(rf, set(SOURCE_OPERATIONS))
        self.assertLessEqual(set(SOURCE_OPERATIONS), source_tools())

    def test_a_surface_s_declared_bound_is_its_resolved_bound(self) -> None:
        from bio_neb.agentic_journey.source_contacts import (
            declared_limits,
            source_tools,
        )
        from bio_neb.failure import ContactLimits, FailureSurface, limits_for

        # A surface's declaration: its row's or registration's supplied fields.
        declarations = {
            name: declared_limits(name)
            for name in source_tools()
            if declared_limits(name).model_dump()
        }
        for name in source_tools():
            with self.subTest(surface=name):
                declared = declarations.get(name)
                expected = (
                    declared.response_bytes
                    if declared is not None
                    else ContactLimits().response_bytes
                )
                resolved = limits_for(FailureSurface(scope="tool", name=name))
                self.assertEqual(resolved.effective.response_bytes, expected)
        # The caps the tools gave these surfaces are kept.
        mib = 1024 * 1024
        self.assertEqual(
            {name: limits.response_bytes for name, limits in declarations.items()},
            {
                # The whole assembly report beside the FASTA, and an assembly's
                # sequence names.
                "describe_companion": 256 * mib,
                "ncbi_sequence_names": 256 * mib,
                "ncbi_taxon_assemblies": 16 * mib,
                "ncbi_genome_annotation": 256 * mib,
                "ncbi_hgvs_contextuals": 16 * mib,
                "ncbi_spdi_equivalents": 16 * mib,
                # A paper's documents; its search and listing keep the default, and
                # the Tool declares only its count.
                "europe_pmc_full_text": 64 * mib,
                "europe_pmc_supplementary_files": 64 * mib,
                "pmc_open_data_file": 64 * mib,
                "download_paper": 4 * mib,
                # The planning Tool that downloads a cited paper takes the download's
                # declaration.
                "retrieve_paper": 4 * mib,
            },
        )

    def test_a_scope_reads_only_a_registered_source_tool(self) -> None:
        from pydantic import ValidationError

        from bio_neb.agentic_journey.source_contacts import SourceContactScopeStarted

        retained = SourceContactScopeStarted(
            attempt_id="a", tool_call_id="c", tool_name="ncbi_genome_annotation"
        )
        self.assertEqual(retained.tool_name, "ncbi_genome_annotation")
        with self.assertRaises(ValidationError) as refused:
            SourceContactScopeStarted(
                attempt_id="a", tool_call_id="c", tool_name="fetch_anything"
            )
        self.assertIn("no registered source Tool", str(refused.exception))


if __name__ == "__main__":
    unittest.main()
