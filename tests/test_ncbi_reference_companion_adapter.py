from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from bio_neb.failure import ContactLimits
from bio_neb.providers.ncbi_datasets import (
    MetadataResponse,
    NcbiDatasetsReferenceAdapter,
    UrllibNcbiTransport,
)
from bio_neb.reference_discovery.reference_identity_tool import ReferenceAdapterError

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)

# The failure module's default byte bound for a source Tool.
_DEFAULT_RESPONSE_BYTES = ContactLimits().response_bytes


class RecordedTransport:
    contact_count = 0

    def __init__(self, payload: object):
        self.payload = payload
        self.requests: list[str] = []

    def get(self, url: str, *, max_bytes: int) -> MetadataResponse:
        self.requests.append(url)
        self.contact_count += 1
        assert max_bytes == _DEFAULT_RESPONSE_BYTES
        return MetadataResponse(
            request_identity=url,
            citation_url=url,
            status_code=200,
            body=json.dumps(self.payload).encode(),
            source_version="datasets/v2-test",
            observed_at=NOW,
            elapsed_time_ms=3,
        )


_PIG = "GCF_054392235.1"
_PIG_SUMMARY = (
    "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
    "GCF_054392235.1/download_summary"
)
_PIG_REPORT = (
    "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
    "GCF_054392235.1/dataset_report"
)
_PIG_DIRECTORY = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/054/392/235/"
    "GCF_054392235.1_Sscrofa_HxYL_maternal_Yorkshire_Landrace_v2.0"
)
_PIG_FASTA = (
    _PIG_DIRECTORY
    + "/GCF_054392235.1_Sscrofa_HxYL_maternal_Yorkshire_Landrace_v2.0_genomic.fna.gz"
)
_PIG_CHECKSUMS = _PIG_DIRECTORY + "/md5checksums.txt"
# Recorded 2026-10-13T17:14:55Z from _PIG_CHECKSUMS (HTTP 200, 34,876 bytes, sha256
# 5727972de718d63d767cd1824fc48252aab14f6d51767c18314fc339245b0f17).
_PIG_CHECKSUMS_BODY = (
    Path(__file__).parent / "fixtures/ncbi/GCF_054392235.1-md5checksums.txt"
).read_bytes()
# HEAD Content-Length recorded for _PIG_FASTA by the NemoEval link pass.
_PIG_FASTA_BYTES = 796_779_932


def _pig_report(assembly_name: str) -> bytes:
    return json.dumps(
        {
            "reports": [
                {
                    "accession": _PIG,
                    "organism": {"organism_name": "Sus scrofa", "tax_id": 9823},
                    "assembly_info": {
                        "assembly_name": assembly_name,
                        "assembly_status": "current",
                    },
                }
            ]
        }
    ).encode()


_PIG_SUMMARY_BODY = json.dumps(
    {
        "resource_updated_on": "2026-09-28T08:23:24Z",
        "available_files": {"all_genomic_fasta": {"file_count": 1, "size_mb": 760.4}},
    }
).encode()


class SiteTransport:
    """Recorded NCBI responses keyed by exact URL; anything else is a contact fault."""

    def __init__(self, gets: dict[str, bytes], heads: dict[str, int | None]):
        self.gets = gets
        self.heads = heads
        self.requests: list[tuple[str, str]] = []
        self.contact_count = 0

    def get(self, url: str, *, max_bytes: int) -> MetadataResponse:
        self.requests.append(("GET", url))
        self.contact_count += 1
        if url not in self.gets:
            raise ReferenceAdapterError(
                "http_error",
                "Official NCBI metadata returned HTTP 404.",
                retryable=False,
            )
        body = self.gets[url]
        assert len(body) <= max_bytes
        return MetadataResponse(
            request_identity=url,
            citation_url=url,
            status_code=200,
            body=body,
            source_version="recorded",
            observed_at=NOW,
            elapsed_time_ms=3,
        )

    def head(self, url: str) -> MetadataResponse:
        self.requests.append(("HEAD", url))
        self.contact_count += 1
        if url not in self.heads:
            raise ReferenceAdapterError(
                "http_error",
                "Official NCBI metadata returned HTTP 404.",
                retryable=False,
            )
        return MetadataResponse(
            request_identity=url,
            citation_url=url,
            status_code=200,
            body=b"",
            source_version="recorded",
            observed_at=NOW,
            elapsed_time_ms=3,
            content_length=self.heads[url],
        )


def _pig_site(**changes: object) -> SiteTransport:
    gets: dict[str, bytes] = {
        _PIG_SUMMARY: _PIG_SUMMARY_BODY,
        _PIG_REPORT: _pig_report("Sscrofa HxYL maternal Yorkshire Landrace v2.0"),
        _PIG_CHECKSUMS: _PIG_CHECKSUMS_BODY,
    }
    heads: dict[str, int | None] = {_PIG_FASTA: _PIG_FASTA_BYTES}
    gets.update(changes.get("gets", {}))  # type: ignore[arg-type]
    heads.update(changes.get("heads", {}))  # type: ignore[arg-type]
    return SiteTransport(gets, heads)


class GenomicFastaSourceIdentityTests(unittest.TestCase):
    def test_genomic_fasta_carries_the_file_url_publisher_md5_and_head_size(self):
        transport = _pig_site()
        observed = NcbiDatasetsReferenceAdapter(transport).describe_companions(
            _PIG, "genomic_fasta"
        )
        (record,) = observed.records
        self.assertEqual(record.asset_id, "all_genomic_fasta")
        self.assertEqual(record.source_locator, _PIG_FASTA)
        self.assertEqual(record.checksum_algorithm, "md5")
        self.assertEqual(record.checksum, "56b9f3d9c185e35e2410611e424ec6eb")
        self.assertEqual(record.checksum_authority_locator, _PIG_CHECKSUMS)
        self.assertEqual(record.byte_size, 796_779_932)
        self.assertEqual(observed.source.request_identity, _PIG_SUMMARY)
        self.assertEqual(
            transport.requests,
            [
                ("GET", _PIG_SUMMARY),
                ("GET", _PIG_REPORT),
                ("GET", _PIG_CHECKSUMS),
                ("HEAD", _PIG_FASTA),
            ],
        )

    def test_a_listing_without_the_exact_genomic_fasta_entry_is_refused(self):
        # One near-miss remains: the RNA FASTA whose name also ends in genomic.fna.gz.
        decoys_only = b"".join(
            line + b"\n"
            for line in _PIG_CHECKSUMS_BODY.splitlines()
            if not line.endswith((b"_v2.0_genomic.fna.gz", b"_cds_from_genomic.fna.gz"))
        )
        self.assertEqual(decoys_only.count(b"genomic.fna.gz"), 1)
        transport = _pig_site(gets={_PIG_CHECKSUMS: decoys_only})
        with self.assertRaises(ReferenceAdapterError) as raised:
            NcbiDatasetsReferenceAdapter(transport).describe_companions(
                _PIG, "genomic_fasta"
            )
        self.assertEqual(raised.exception.failure.code, "invalid_response")
        self.assertIn("no single md5", raised.exception.failure.public_message)
        self.assertNotIn(("HEAD", _PIG_FASTA), transport.requests)

    def test_a_failed_or_sizeless_head_is_refused(self):
        for heads, code in (
            ({_PIG_FASTA: None}, "invalid_response"),
            ({}, "http_error"),
        ):
            with self.subTest(code=code):
                transport = _pig_site()
                transport.heads = heads
                with self.assertRaises(ReferenceAdapterError) as raised:
                    NcbiDatasetsReferenceAdapter(transport).describe_companions(
                        _PIG, "genomic_fasta"
                    )
                self.assertEqual(raised.exception.failure.code, code)
                self.assertEqual(transport.requests[-1], ("HEAD", _PIG_FASTA))

    def test_an_assembly_name_the_source_does_not_report_once_is_refused(self):
        two_names = json.loads(_pig_report("Sscrofa v2.0"))
        two_names["reports"].append(
            json.loads(_pig_report("Sscrofa v2.1"))["reports"][0]
        )
        for reports in ({"reports": []}, two_names):
            with self.subTest(count=len(reports["reports"])):
                transport = _pig_site(gets={_PIG_REPORT: json.dumps(reports).encode()})
                with self.assertRaises(ReferenceAdapterError) as raised:
                    NcbiDatasetsReferenceAdapter(transport).describe_companions(
                        _PIG, "genomic_fasta"
                    )
                self.assertEqual(raised.exception.failure.code, "source_disagreement")
                self.assertEqual(
                    transport.requests, [("GET", _PIG_SUMMARY), ("GET", _PIG_REPORT)]
                )

    def test_annotation_contacts_only_the_download_summary(self):
        transport = _pig_site(
            gets={
                _PIG_SUMMARY: json.dumps(
                    {"available_files": {"genome_gff": {"file_count": 1}}}
                ).encode()
            }
        )
        observed = NcbiDatasetsReferenceAdapter(transport).describe_companions(
            _PIG, "annotation"
        )
        self.assertEqual(observed.records[0].asset_id, "genome_gff")
        self.assertIsNone(observed.records[0].source_locator)
        self.assertEqual(transport.requests, [("GET", _PIG_SUMMARY)])


class GenomesHostTransportTests(unittest.TestCase):
    def test_head_sends_one_bodiless_request_and_reports_the_length(self):
        response = MagicMock(status=200)
        response.headers = {"Content-Length": "796779932", "Last-Modified": "Mon"}
        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value = response
            observed = UrllibNcbiTransport().head(_PIG_FASTA)
        (request,), _ = opener.return_value.open.call_args
        self.assertEqual(request.get_method(), "HEAD")
        self.assertEqual(request.full_url, _PIG_FASTA)
        self.assertEqual(request.get_header("Accept"), "text/plain")
        response.read.assert_not_called()
        self.assertEqual(observed.content_length, 796_779_932)
        self.assertEqual(observed.body, b"")

    def test_genomes_host_get_and_other_hosts_refused_before_contact(self):
        response = MagicMock(status=200)
        response.headers = {}
        response.read.return_value = b"abc  ./x\n"
        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value = response
            transport = UrllibNcbiTransport()
            observed = transport.get(_PIG_CHECKSUMS, max_bytes=1024)
            for url in (
                _PIG_FASTA.replace("ftp.ncbi", "ftp-trace.ncbi"),
                _PIG_FASTA.replace("https://", "http://"),
            ):
                with self.subTest(url=url), self.assertRaises(ReferenceAdapterError):
                    transport.head(url)
        self.assertEqual(observed.body, b"abc  ./x\n")
        self.assertEqual(opener.return_value.open.call_count, 1)
        self.assertEqual(transport.contact_count, 1)


class NcbiReferenceCompanionAdapterTests(unittest.TestCase):
    def test_download_summary_preserves_identity_and_exact_source_scope(self):
        opaque = "ncbi-datasets:GCF_000001405.40:GENOMIC_NUCLEOTIDE_FASTA"
        transport = RecordedTransport(
            {
                "reports": [
                    {
                        "accession": "GCF_000001405.40",
                        "files": [
                            {
                                "file_type": "GENOMIC_NUCLEOTIDE_FASTA",
                                "file_name": opaque,
                            },
                            {"file_type": "GFF3", "file_name": "annotation-id"},
                        ],
                    }
                ]
            }
        )
        observed = NcbiDatasetsReferenceAdapter(transport).describe_companions(
            "GCF_000001405.40", "genomic_fasta"
        )
        self.assertEqual(transport.contact_count, 1)
        self.assertEqual(observed.records[0].asset_id, opaque)
        self.assertEqual(observed.records[0].assembly_accession, "GCF_000001405.40")
        self.assertIn(
            "GCF_000001405.40/download_summary", observed.source.request_identity
        )
        self.assertEqual(observed.records[0].release_version, "GCF_000001405.40")
        self.assertIsNone(observed.records[0].checksum)
        self.assertNotEqual(
            observed.source.response_sha256, observed.records[0].checksum
        )

    def test_current_aggregate_preserves_exact_source_key(self):
        summary = {
            "record_count": 1,
            "resource_updated_on": "2026-08-06",
            "hydrated": "https://example.invalid/not-contacted.zip",
            "dehydrated": "https://example.invalid/not-contacted.zip",
            "available_files": {
                "all_genomic_fasta": {"file_count": 1, "size_mb": 4.7},
                "all_protein_fasta": {"file_count": 1, "size_mb": 1.4},
                "cds_fasta": {"file_count": 1, "size_mb": 1.3},
                "genome_gbff": {"file_count": 1, "size_mb": 9.1},
                "genome_gff": {"file_count": 1, "size_mb": 1.8},
                "genome_gtf": {"file_count": 1, "size_mb": 1.7},
                "rna_fasta": {},
                "sequence_report": {"file_count": 1, "size_mb": 0.01},
            },
        }
        transport = _pig_site(gets={_PIG_SUMMARY: json.dumps(summary).encode()})
        observed = NcbiDatasetsReferenceAdapter(transport).describe_companions(
            _PIG, "genomic_fasta"
        )
        self.assertEqual(len(observed.records), 1)
        candidate = observed.records[0]
        self.assertEqual(candidate.asset_id, "all_genomic_fasta")
        self.assertEqual(candidate.representation, "all_genomic_fasta")
        self.assertEqual(candidate.assembly_accession, _PIG)
        self.assertEqual(candidate.role, "genomic_fasta")
        self.assertEqual(candidate.release_version, "2026-08-06")
        self.assertEqual(candidate.authority_locator, _PIG_SUMMARY)
        self.assertNotIn(
            ("GET", "https://example.invalid/not-contacted.zip"), transport.requests
        )

    def test_empty_aggregate_metadata_means_zero_files_and_no_records(self):
        transport = RecordedTransport({"available_files": {"rna_fasta": {}}})
        observed = NcbiDatasetsReferenceAdapter(transport).describe_companions(
            "GCF_000005845.2", "annotation"
        )
        self.assertEqual(transport.contact_count, 1)
        self.assertEqual(observed.records, ())

        invalid_entries = (
            {"": {}},
            {"rna_fasta": []},
            {"rna_fasta": {"size_mb": 0.0}},
            {"rna_fasta": {"file_count": True, "size_mb": 0.0}},
            {"rna_fasta": {"file_count": -1, "size_mb": 0.0}},
            {"rna_fasta": {"file_count": "0", "size_mb": 0.0}},
        )
        for available_files in invalid_entries:
            with self.subTest(available_files=available_files):
                adapter = NcbiDatasetsReferenceAdapter(
                    RecordedTransport({"available_files": available_files})
                )
                with self.assertRaises(ReferenceAdapterError):
                    adapter.describe_companions("GCF_000005845.2", "genomic_fasta")

    def test_current_aggregate_annotation_keys_keep_exact_adapter_mapping(self):
        transport = RecordedTransport(
            {
                "resource_updated_on": "2026-08-06",
                "available_files": {
                    "genome_gff": {"file_count": 1, "size_mb": 1.5},
                    "genome_gtf": {"file_count": 1, "size_mb": 1.4},
                    "rna_fasta": {"file_count": 2, "size_mb": 0.8},
                    "prot_fasta": {"file_count": 0, "size_mb": 0.0},
                    "all_genomic_fasta": {"file_count": 1, "size_mb": 4.25},
                },
            }
        )
        observed = NcbiDatasetsReferenceAdapter(transport).describe_companions(
            "GCF_000001405.40", "annotation"
        )
        self.assertEqual(transport.contact_count, 1)
        self.assertEqual(
            tuple(record.asset_id for record in observed.records),
            ("genome_gff", "genome_gtf", "rna_fasta"),
        )
        self.assertTrue(all(record.byte_size is None for record in observed.records))
        self.assertTrue(all(record.checksum is None for record in observed.records))

    def test_download_summary_never_normalizes_opaque_identity(self):
        transport = RecordedTransport(
            {
                "reports": [
                    {
                        "accession": "GCF_000001405.40",
                        "files": [
                            {
                                "file_type": "GENOMIC_NUCLEOTIDE_FASTA",
                                "file_name": " source-owned-id ",
                            }
                        ],
                    }
                ]
            }
        )
        adapter = NcbiDatasetsReferenceAdapter(transport)
        with self.assertRaisesRegex(ValidationError, "canonical string"):
            adapter.describe_companions("GCF_000001405.40", "genomic_fasta")
        self.assertEqual(transport.contact_count, 1)


if __name__ == "__main__":
    unittest.main()
