"""Offline contracts for bounded NCBI reference-metadata parsing and transport."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import unittest
import urllib.parse
from datetime import UTC, datetime

from bio_neb.failure import ContactLimits
from bio_neb.providers.ncbi_datasets import (
    MetadataResponse,
    NcbiDatasetsReferenceAdapter,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    AssemblyQuery,
    ReferenceAdapterError,
    ReferenceQueryRefused,
)

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)

# The failure module's default byte bound for a source Tool.
_DEFAULT_RESPONSE_BYTES = ContactLimits().response_bytes


class NcbiReferenceMetadataAdapterTests(unittest.TestCase):
    def test_explicit_taxonomy_no_match_is_a_scientific_zero_candidate_result(
        self,
    ) -> None:
        query = "Escherichia coli K-12 MG1655 (TaxId 511145)"
        no_match = {
            "reports": [
                {
                    "query": [query],
                    "errors": [
                        {
                            "invalid_identifiers": [query],
                            "reason": "No taxonomy nodes matched the supplied identifier.",
                        }
                    ],
                }
            ]
        }
        body = json.dumps(no_match, separators=(",", ":")).encode()
        transport = RecordedTransport((body,))

        observation = NcbiDatasetsReferenceAdapter(transport).resolve_taxon(query)

        self.assertEqual(observation.records, ())
        self.assertEqual(observation.source.authority, "ncbi_taxonomy")
        self.assertEqual(
            observation.source.response_sha256,
            hashlib.sha256(body).hexdigest(),
        )
        self.assertEqual(transport.contact_count, 1)
        encoded = urllib.parse.quote(query, safe="")
        self.assertEqual(
            transport.urls,
            [
                "https://api.ncbi.nlm.nih.gov/datasets/v2/taxonomy/taxon/"
                f"{encoded}/dataset_report"
            ],
        )
        self.assertEqual(observation.source.request_identity, transport.urls[0])

    def test_a_term_naming_several_taxa_is_refused_before_contact(self) -> None:
        # NCBI Taxonomy reads a comma in its `{taxons}` path segment as a list of
        # taxa (a recorded request sent "Homo sapiens, NCBI taxon identifier
        # 9606").
        term = "Homo sapiens, NCBI taxon identifier 9606"
        transport = RecordedTransport(())
        adapter = NcbiDatasetsReferenceAdapter(transport)

        self.assertEqual(adapter.taxon_term_refusal(term), "taxon_term_not_one_taxon")
        self.assertIsNone(adapter.taxon_term_refusal("Homo sapiens"))
        self.assertIsNone(adapter.taxon_term_refusal("9606"))
        with self.assertRaises(ReferenceQueryRefused) as refused:
            adapter.resolve_taxon(term)

        self.assertEqual(refused.exception.code, "taxon_term_not_one_taxon")
        self.assertEqual(refused.exception.term, term)
        self.assertNotIsInstance(refused.exception, ReferenceAdapterError)
        self.assertEqual(transport.contact_count, 0)

    def test_malformed_taxonomy_payloads_are_operational_failures(self) -> None:
        query = "Escherichia coli K-12 MG1655 (TaxId 511145)"
        no_match = {
            "reports": [
                {
                    "query": [query],
                    "errors": [
                        {
                            "invalid_identifiers": [query],
                            "reason": "No taxonomy nodes matched the supplied identifier.",
                        }
                    ],
                }
            ]
        }
        valid_node = {
            "taxonomy": {
                "tax_id": 511145,
                "current_scientific_name": {"name": "Escherichia coli K-12 MG1655"},
            }
        }
        invalid_reports = (
            {"reports": [{**no_match["reports"][0], "unknown": True}]},
            {
                "reports": [
                    {
                        "query": [query],
                        "errors": [
                            {
                                "invalid_identifiers": query,
                                "reason": "Malformed invalid identifier collection.",
                            }
                        ],
                    }
                ]
            },
            {"reports": [valid_node, no_match["reports"][0]]},
            {**no_match, "unexpected": True},
            {
                "reports": [
                    {
                        "query": [query, "511145"],
                        "errors": [
                            {
                                "invalid_identifiers": [query, "511145"],
                                "reason": "Unexpected extra identifier echo.",
                            }
                        ],
                    }
                ]
            },
        )
        for invalid in invalid_reports:
            with self.subTest(invalid=invalid):
                transport = RecordedTransport((json.dumps(invalid).encode(),))
                with self.assertRaisesRegex(
                    ReferenceAdapterError, "malformed taxonomy node"
                ):
                    NcbiDatasetsReferenceAdapter(transport).resolve_taxon(query)
                self.assertEqual(transport.contact_count, 1)

    def test_current_routes_parse_direct_taxonomy_lineage_and_source_identity(
        self,
    ) -> None:
        taxonomy_body = json.dumps(
            {
                "taxonomy_nodes": [
                    {
                        "taxonomy": {
                            "tax_id": 9606,
                            "current_scientific_name": {"name": "Homo sapiens"},
                            "rank": "species",
                            "lineage": [{"tax_id": 2759, "name": "Eukaryota"}],
                        }
                    }
                ]
            }
        ).encode()
        assembly_body = json.dumps(
            {
                "reports": [
                    {
                        "accession": "GCF_000001405.40",
                        "organism": {"organism_name": "Homo sapiens", "tax_id": 9606},
                        "assembly_info": {
                            "assembly_name": "GRCh38.p14",
                            "assembly_status": "current",
                            "refseq_category": "reference genome",
                            "assembly_level": "chromosome",
                        },
                    }
                ],
                "total_count": 1,
            }
        ).encode()
        transport = RecordedTransport((taxonomy_body, assembly_body, assembly_body))
        adapter = NcbiDatasetsReferenceAdapter(transport)

        taxon = adapter.resolve_taxon("9606")
        discovery = adapter.discover_assemblies(
            9606, AssemblyQuery(reference_only=True, assembly_source="refseq")
        )
        exact = adapter.describe_assembly("GCF_000001405.40")

        self.assertEqual(taxon.records[0].scientific_name, "Homo sapiens")
        self.assertEqual(taxon.records[0].lineage, ("Eukaryota",))
        self.assertEqual(taxon.records[0].lineage_tax_ids, (2759,))
        self.assertEqual(discovery.records[0].assembly_accession, "GCF_000001405.40")
        self.assertEqual(exact.records[0].assembly_status, "current")
        self.assertEqual(adapter.contact_count, 3)
        self.assertEqual(
            transport.urls,
            [
                "https://api.ncbi.nlm.nih.gov/datasets/v2/taxonomy/taxon/9606/"
                "dataset_report",
                "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/taxon/9606/"
                "dataset_report?filters.reference_only=true"
                "&filters.assembly_source=refseq&page_size=100",
                "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
                "GCF_000001405.40/dataset_report",
            ],
        )
        self.assertEqual(taxon.source.request_identity, transport.urls[0])
        self.assertEqual(discovery.source.request_identity, transport.urls[1])
        self.assertEqual(exact.source.request_identity, transport.urls[2])
        self.assertEqual(taxon.source.authority, "ncbi_taxonomy")
        self.assertEqual(discovery.source.authority, "ncbi_datasets")
        self.assertEqual(exact.source.authority, "ncbi_datasets")

    def test_a_replaced_assembly_is_described_from_its_any_version_report(
        self,
    ) -> None:
        body = json.dumps(
            {
                "reports": [
                    {
                        "accession": "GCF_000003025.6",
                        "organism": {"organism_name": "Sus scrofa", "tax_id": 9823},
                        "assembly_info": {
                            "assembly_name": "Sscrofa11.1",
                            "assembly_status": "suppressed",
                            "assembly_level": "chromosome",
                        },
                    }
                ],
                "total_count": 1,
            }
        ).encode()
        # NCBI's plain report of a suppressed assembly is exactly `{}`.
        transport = RecordedTransport((b"{}", body))
        adapter = NcbiDatasetsReferenceAdapter(transport)

        described = adapter.describe_assembly("GCF_000003025.6")

        plain = (
            "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
            "GCF_000003025.6/dataset_report"
        )
        url = f"{plain}?filters.assembly_version=all_assemblies"
        self.assertEqual(transport.urls, [plain, url])
        self.assertEqual(
            [(r.assembly_accession, r.assembly_status) for r in described.records],
            [("GCF_000003025.6", "suppressed")],
        )
        self.assertEqual(described.total_count, 1)
        self.assertEqual(described.source.request_identity, url)
        self.assertEqual(described.source.authority, "ncbi_datasets")
        self.assertEqual(
            described.source.response_sha256, hashlib.sha256(body).hexdigest()
        )

    def test_a_replaced_description_refuses_what_the_exact_description_refuses(
        self,
    ) -> None:
        """Both descriptions refuse an answer with no report list or a row with no
        exact accession, and neither caps how many reports NCBI lists: an
        any-version answer with eleven is copied whole (#471 Q5 a, #375 C13)."""
        record = {
            "accession": "GCF_000003025.6",
            "organism": {"organism_name": "Sus scrofa", "tax_id": 9823},
            "assembly_info": {
                "assembly_name": "Sscrofa11.1",
                "assembly_status": "suppressed",
            },
        }
        cases = {
            "no report list": {"total_count": 0},
            "a malformed report": {"reports": [record, {"accession": "GCF_1.1"}]},
            "a report with no accession": {"reports": [record, {"organism": {}}]},
        }
        for name, payload in cases.items():
            with self.subTest(name):
                transport = RecordedTransport((json.dumps(payload).encode(),))
                adapter = NcbiDatasetsReferenceAdapter(transport)
                with self.assertRaises(ReferenceAdapterError) as raised:
                    adapter.describe_assembly("GCF_000003025.6")
                self.assertEqual(raised.exception.failure.code, "invalid_response")
                self.assertEqual(transport.contact_count, 1)
        with self.subTest("eleven reports"):
            versions = [
                {**record, "accession": f"GCF_000003025.{version}"}
                for version in range(1, 12)
            ]
            transport = RecordedTransport((json.dumps({"reports": versions}).encode(),))
            described = NcbiDatasetsReferenceAdapter(transport).describe_assembly(
                "GCF_000003025.6"
            )
            self.assertEqual(
                [r.assembly_accession for r in described.records],
                [item["accession"] for item in versions],
            )
        # An any-version report that is `{}` too describes nothing.
        transport = RecordedTransport((b"{}", b"{}"))
        with self.assertRaises(ReferenceAdapterError) as raised:
            NcbiDatasetsReferenceAdapter(transport).describe_assembly("GCF_000003025.6")
        self.assertEqual(raised.exception.failure.code, "invalid_response")
        self.assertEqual(transport.contact_count, 2)

    def test_an_empty_report_from_another_request_is_refused_before_a_second_get(
        self,
    ) -> None:
        """An empty plain report decides the any-version GET only when it is the
        exact answer to this request; one that names another request is
        ``source_disagreement`` after one contact (#471 review r6 ST5)."""

        class Misidentified(RecordedTransport):
            def get(self, url: str, *, max_bytes: int) -> MetadataResponse:
                answered = super().get(url, max_bytes=max_bytes)
                other = url.replace("GCF_000001405.25", "GCF_000002985.6")
                return dataclasses.replace(
                    answered, request_identity=other, citation_url=other
                )

        transport = Misidentified((b"{}", b'{"reports": []}'))
        with self.assertRaises(ReferenceAdapterError) as raised:
            NcbiDatasetsReferenceAdapter(transport).describe_assembly(
                "GCF_000001405.25"
            )
        self.assertEqual(raised.exception.failure.code, "source_disagreement")
        self.assertEqual(transport.contact_count, 1)

    def test_invalid_json_makes_one_contact_and_no_hidden_retry(self) -> None:
        transport = RecordedTransport((b"not-json",))
        adapter = NcbiDatasetsReferenceAdapter(transport)
        with self.assertRaisesRegex(ReferenceAdapterError, "valid JSON"):
            adapter.resolve_taxon("9606")
        self.assertEqual(transport.contact_count, 1)

    def test_transport_cannot_bypass_the_adapter_response_byte_bound(self) -> None:
        transport = OversizedTransport()
        adapter = NcbiDatasetsReferenceAdapter(transport)
        with self.assertRaisesRegex(ReferenceAdapterError, "byte bound"):
            adapter.resolve_taxon("9606")
        self.assertEqual(transport.contact_count, 1)


class RecordedTransport:
    def __init__(self, bodies: tuple[bytes, ...]) -> None:
        self._bodies = list(bodies)
        self.contact_count = 0
        self.urls: list[str] = []

    def get(self, url: str, *, max_bytes: int) -> MetadataResponse:
        self.contact_count += 1
        self.urls.append(url)
        body = self._bodies.pop(0)
        if len(body) > max_bytes:
            raise AssertionError("fixture exceeded requested byte bound")
        return MetadataResponse(
            request_identity=url,
            citation_url=url,
            status_code=200,
            body=body,
            source_version="recorded-ncbi-v1",
            observed_at=NOW,
            elapsed_time_ms=3,
        )


class OversizedTransport:
    def __init__(self) -> None:
        self.contact_count = 0

    def get(self, url: str, *, max_bytes: int) -> MetadataResponse:
        del max_bytes
        self.contact_count += 1
        return MetadataResponse(
            request_identity=url,
            citation_url=url,
            status_code=200,
            body=b"x" * (_DEFAULT_RESPONSE_BYTES + 1),
            source_version="violating-transport",
            observed_at=NOW,
            elapsed_time_ms=1,
        )


if __name__ == "__main__":
    unittest.main()
