"""The EVA source adapter: one bounded GET per operation on EVA's documented paths,
a lawful 404, and each fault classified for the failure module (#366)."""

from __future__ import annotations

import io
import unittest
import urllib.error
from email.message import Message
from unittest.mock import MagicMock, patch

from bio_neb.failure import ProviderError
from bio_neb.providers.eva_investigation import (
    EvaInvestigationSourceAdapter,
    UrllibEvaTransport,
    checked_eva_response,
)
from tests.eva_retained_bodies import (
    EVA_CONTIG_ALIAS,
    EVA_IDENTIFIERS,
    PIG_RS5334475172,
)


def _answered(body: bytes, status: int = 200):
    response = MagicMock(status=status)
    response.headers = {}
    response.read.return_value = body
    return response


_ALIASES = "https://www.ebi.ac.uk/eva/webservices/contig-alias/v1"


class EvaAdapterTests(unittest.TestCase):
    def test_each_operation_gets_its_documented_path(self) -> None:
        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value = _answered(
                PIG_RS5334475172
            )
            adapter = EvaInvestigationSourceAdapter(UrllibEvaTransport())
            responses = [
                adapter.fetch_eva_clustered_variant("rs5334475172"),
                adapter.fetch_eva_submitted_variants("rs5334475172"),
                adapter.fetch_eva_contig_alias("CM000823.5"),
            ]
        sent = [call.args[0] for call in opener.return_value.open.call_args_list]
        self.assertEqual(
            [request.full_url for request in sent],
            [
                f"{EVA_IDENTIFIERS}/clustered-variants/5334475172",
                f"{EVA_IDENTIFIERS}/clustered-variants/5334475172/submitted",
                f"{EVA_CONTIG_ALIAS}/chromosomes/genbank/CM000823.5",
            ],
        )
        self.assertEqual({r.get_header("Accept") for r in sent}, {"application/json"})
        self.assertEqual([r.status_code for r in responses], [200, 200, 200])
        self.assertEqual(responses[0].body, PIG_RS5334475172)
        self.assertEqual(adapter.contact_count, 3)

    def test_malformed_keys_are_refused_before_contact(self) -> None:
        with patch("urllib.request.build_opener") as opener:
            adapter = EvaInvestigationSourceAdapter(UrllibEvaTransport())
            for call in (
                lambda: adapter.fetch_eva_clustered_variant("rs01"),
                lambda: adapter.fetch_eva_submitted_variants("5334475172"),
                lambda: adapter.fetch_eva_contig_alias(""),
            ):
                with self.assertRaises(ValueError):
                    call()
        opener.return_value.open.assert_not_called()

    def test_a_contig_key_is_percent_encoded_never_judged_by_its_shape(self) -> None:
        """URL safety by encoding, not by a sequence name's shape (#467, ADR 0049)."""
        sent = []

        class _Transport:
            contact_count = 0

            def get(self, url):
                sent.append(url)
                return None

        adapter = EvaInvestigationSourceAdapter(_Transport())
        adapter.fetch_eva_contig_alias("CM000823.5/../x")
        adapter.fetch_eva_contig_alias("NC_010454.4")

        self.assertEqual(
            sent,
            [
                f"{_ALIASES}/chromosomes/genbank/CM000823.5%2F..%2Fx",
                f"{_ALIASES}/chromosomes/genbank/NC_010454.4",
            ],
        )

    def test_evas_own_options_pass_encoded_and_no_other_is_taken(self) -> None:
        """A key may carry EVA's one documented option for its GET (#468, owner R3):
        the clustered and submitted variants' contigNamingConvention, the alias's page.
        Its value is percent-encoded, never judged; any other option is refused."""
        sent = []

        class _Transport:
            contact_count = 0

            def get(self, url):
                sent.append(url)
                return None

        adapter = EvaInvestigationSourceAdapter(_Transport())
        adapter.fetch_eva_clustered_variant("rs5?contigNamingConvention=REFSEQ")
        adapter.fetch_eva_submitted_variants("rs5?contigNamingConvention=a b")
        adapter.fetch_eva_contig_alias("CM000823.5?page=2")
        for key in (
            "rs5?page=1",
            "rs5?contigNamingConvention=REFSEQ&size=9",
            "CM000823.5?size=100",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                (
                    adapter.fetch_eva_contig_alias
                    if key.startswith("CM")
                    else adapter.fetch_eva_clustered_variant
                )(key)

        self.assertEqual(
            sent,
            [
                f"{EVA_IDENTIFIERS}/clustered-variants/5?contigNamingConvention=REFSEQ",
                f"{EVA_IDENTIFIERS}/clustered-variants/5/submitted"
                "?contigNamingConvention=a+b",
                f"{_ALIASES}/chromosomes/genbank/CM000823.5?page=2",
            ],
        )

    def test_a_status_that_is_no_answer_goes_to_the_failure_module_for_every_lookup_get(
        self,
    ) -> None:
        """A rate limit, a server fault or an access fault is no source answer: every
        GET of the three variant lookup Tools raises it at the one fetch seam, for the
        failure module to retry or ask, with no handling of the lookup's own (owner #468
        SP4 a, #458)."""
        from types import SimpleNamespace

        from bio_neb.agentic_journey.tool_execution import function_tool_contracts
        from bio_neb.providers.eva_investigation import fetch_source
        from bio_neb.providers.ncbi_investigation import InvestigationSourceResponse
        from bio_neb.providers.source_operations import SOURCE_OPERATIONS

        operations = sorted(
            {
                operation
                for contract in function_tool_contracts("investigation_planner")
                if contract.tool_name
                in {"fetch_refsnp", "fetch_eva_variant", "locate_variant"}
                for operation in contract.source_surface.operations
            }
        )
        self.assertIn("eva_contig_alias", operations)
        for operation in operations:
            method = SOURCE_OPERATIONS[operation].method
            for status in (401, 403, 407, 429, 500, 503):
                with self.subTest(operation=operation, status=status):
                    answer = InvestigationSourceResponse(
                        request_identity="https://api.ncbi.nlm.nih.gov/x",
                        status_code=status,
                        body=b'{"message":"busy"}',
                        observed_at="2026-10-27T12:00:00Z",
                        elapsed_time_ms=1,
                    )
                    adapter = SimpleNamespace(
                        **{method: lambda *_key, answer=answer: answer}
                    )

                    with self.assertRaises(ProviderError):
                        fetch_source(adapter, operation, "key")
            answered = InvestigationSourceResponse(
                request_identity="https://api.ncbi.nlm.nih.gov/x",
                status_code=410,
                body=b"{}",
                observed_at="2026-10-27T12:00:00Z",
                elapsed_time_ms=1,
            )
            adapter = SimpleNamespace(
                **{method: lambda *_key, answered=answered: answered}
            )
            with self.subTest(operation=operation, status=410):
                self.assertIs(fetch_source(adapter, operation, "key"), answered)

    def test_an_http_status_is_an_answer_and_a_lost_connection_is_transient(
        self,
    ) -> None:
        not_found = urllib.error.HTTPError(
            "u", 404, "Not Found", Message(), io.BytesIO(b'{"message":"none"}')
        )
        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = [not_found, TimeoutError("slow")]
            transport = UrllibEvaTransport()
            answered = EvaInvestigationSourceAdapter(
                transport
            ).fetch_eva_clustered_variant("rs9")
            with self.assertRaises(ProviderError) as lost:
                EvaInvestigationSourceAdapter(transport).fetch_eva_clustered_variant(
                    "rs9"
                )
        self.assertEqual(
            (answered.status_code, answered.body), (404, b'{"message":"none"}')
        )
        self.assertEqual(lost.exception.failure_class, "transient")
        self.assertEqual(transport.contact_count, 2)

    def test_a_body_over_its_surfaces_bound_is_rejected(self) -> None:
        # The bound is the operation surface's limit, a Tool's own in code.
        from bio_neb.agentic_journey.source_contacts import call_source
        from bio_neb.failure import ContactLimitOverrides
        from tests.source_limits_support import declaring

        url = f"{EVA_IDENTIFIERS}/clustered-variants/9"
        with (
            patch("urllib.request.build_opener") as opener,
            declaring(
                {"eva_clustered_variant": ContactLimitOverrides(response_bytes=10)}
            ),
        ):
            answered = opener.return_value.open.return_value.__enter__.return_value
            answered.status = 200
            answered.read.side_effect = lambda size: b"x" * min(size, 11)
            transport = UrllibEvaTransport()
            with self.assertRaises(ProviderError) as refused:
                call_source("eva_clustered_variant", lambda: transport.get(url))
            self.assertEqual(answered.read.call_args.args, (11,))
            answered.read.side_effect = lambda size: b"x" * 10
            self.assertEqual(
                call_source("eva_clustered_variant", lambda: transport.get(url)).body,
                b"x" * 10,
            )
        self.assertEqual(refused.exception.failure_class, "rejected")
        self.assertIn("byte bound", refused.exception.detail)

    def test_other_hosts_and_paths_are_refused_before_contact(self) -> None:
        with patch("urllib.request.build_opener") as opener:
            transport = UrllibEvaTransport()
            for url in (
                "http://www.ebi.ac.uk/eva/webservices/identifiers/v1/clustered-variants/9",
                "https://www.ebi.ac.uk/ena/browser/api/9",
                "https://evil.example/eva/webservices/identifiers/v1/x",
            ):
                with self.subTest(url=url), self.assertRaises(ProviderError):
                    transport.get(url)
        opener.return_value.open.assert_not_called()


class EnvelopeTests(unittest.TestCase):
    def test_an_answer_about_the_request_passes_and_faults_are_classified(
        self,
    ) -> None:
        # A source's answer about the request, a 410 among them, is the lookup's to
        # read, a redirect's body too, never followed (Spec r1 SP4); a rate limit
        # or server fault is transient and an access fault is not an answer about
        # the record.
        from bio_neb.providers.ncbi_investigation import InvestigationSourceResponse

        def response(status):
            return InvestigationSourceResponse(
                "u", status, b"[]", "2026-10-17T00:00:00Z", 1
            )

        for status in (200, 301, 302, 400, 404, 410):
            self.assertEqual(checked_eva_response(response(status)).status_code, status)
        for status, failure_class in (
            (503, "transient"),
            (429, "transient"),
            (401, "rejected"),
            (403, "rejected"),
            (407, "rejected"),
        ):
            with self.subTest(status=status), self.assertRaises(ProviderError) as error:
                checked_eva_response(response(status))
            self.assertEqual(error.exception.failure_class, failure_class)


if __name__ == "__main__":
    unittest.main()
