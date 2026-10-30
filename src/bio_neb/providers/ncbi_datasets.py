"""Bounded, retry-free NCBI metadata mechanics for reference identity Tools."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Mapping, Protocol, Sequence

from bio_neb.agentic_journey.source_answers import answer_received, source_answered
from bio_neb.agentic_journey.source_contacts import (
    response_byte_bound,
    response_over_bound,
)
from bio_neb.providers.ncbi_assembly_report import (
    ReportUnreadable,
    SequenceRecord,
    report_rows,
    report_url,
)
from bio_neb.providers.source_transport import (
    TIMEOUT_SECONDS,
    AnsweredStatus,
    Opener,
    SourceAnswerTooLarge,
    SourceContactLost,
    SourceTransport,
    SourceUrlRefused,
)
from bio_neb.reference_discovery.reference_companions_tool import (
    CompanionAdapterObservation,
    CompanionRole,
    CompanionSourceRecord,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    AssemblyAdapterObservation,
    AssemblyQuery,
    AssemblySourceRecord,
    InfraspecificNames,
    ReferenceAdapterError,
    ReferenceQueryRefused,
    SourceObservation,
    TaxonomyAdapterObservation,
    TaxonomySourceRecord,
)

ADAPTER_VERSION = "ncbi-datasets-reference-metadata/v1"
_OFFICIAL_HOST = "api.ncbi.nlm.nih.gov"
_GENOMES_HOST = "ftp.ncbi.nlm.nih.gov"
_GENOMES_ACCESSION = re.compile(r"^(GC[AF])_(\d{3})(\d{3})(\d{3})\.\d+$")
_CURRENT_COMPANION_KEYS: dict[CompanionRole, frozenset[str]] = {
    "genomic_fasta": frozenset({"all_genomic_fasta"}),
    "annotation": frozenset({"genome_gff", "genome_gtf", "rna_fasta", "prot_fasta"}),
}


@dataclass(frozen=True)
class MetadataResponse:
    request_identity: str
    citation_url: str
    status_code: int
    body: bytes
    source_version: str | None
    observed_at: datetime
    elapsed_time_ms: int
    content_length: int | None = None


class MetadataTransport(Protocol):
    contact_count: int

    def get(
        self, url: str, *, max_bytes: int, accept: str | None = None
    ) -> MetadataResponse: ...

    def head(self, url: str) -> MetadataResponse: ...


class UrllibNcbiTransport:
    """NCBI's metadata view of the source transport family: one bounded GET or HEAD
    per call; nothing is retried. Each GET the source answered is reported to the
    Tool collecting answers (`source_answers.answer_received`). A redirect or a
    client status is the source's answer, refused as ``source_answered`` for its reader; any other
    non-success is ``http_error``."""

    def __init__(
        self, *, timeout_seconds: float = TIMEOUT_SECONDS, opener: Opener | None = None
    ) -> None:
        self._transport = SourceTransport(
            (_OFFICIAL_HOST, _GENOMES_HOST),
            timeout_seconds=timeout_seconds,
            opener=opener,
        )

    @property
    def contact_count(self) -> int:
        return self._transport.contact_count

    def get(
        self, url: str, *, max_bytes: int, accept: str | None = None
    ) -> MetadataResponse:
        if max_bytes < 1:
            raise ValueError("metadata byte limit is outside the bounded contract")
        return self._send(url, method="GET", max_bytes=max_bytes, accept=accept)

    def head(self, url: str) -> MetadataResponse:
        """Observe one file's headers; the body is never requested."""
        return self._send(url, method="HEAD", max_bytes=0, accept=None)

    def _send(
        self,
        url: str,
        *,
        method: Literal["GET", "HEAD"],
        max_bytes: int,
        accept: str | None,
    ) -> MetadataResponse:
        try:
            answer = self._transport.exchange(
                url,
                method=method,
                max_bytes=max_bytes,
                accept=accept,
                status_fails=True,
            )
        except AnsweredStatus as status:
            code = status.answer.status
            if method == "GET" and source_answered(code):
                answer_received(url, code, status.answer.body)
                raise ReferenceAdapterError(
                    "source_answered",
                    f"Official NCBI metadata answered HTTP {code}.",
                    retryable=False,
                ) from status
            raise ReferenceAdapterError(
                "http_error",
                "Official NCBI metadata redirect was refused."
                if 300 <= code < 400
                else f"Official NCBI metadata returned HTTP {code}.",
                retryable=code == 429 or code >= 500,
            ) from status
        except SourceUrlRefused as error:
            raise ReferenceAdapterError(
                "transport_error",
                "Metadata URL is outside the official NCBI allowlist.",
                retryable=False,
            ) from error
        except SourceAnswerTooLarge as error:
            raise ReferenceAdapterError(
                "response_too_large",
                "Official NCBI metadata exceeded the configured byte bound.",
                retryable=False,
            ) from error
        except SourceContactLost as error:
            raise ReferenceAdapterError(
                "transport_error",
                f"Official NCBI metadata request failed: {error.detail}",
                retryable=True,
            ) from error
        if method == "GET":
            answer_received(url, answer.status, answer.body)
        return MetadataResponse(
            request_identity=url,
            citation_url=url,
            status_code=answer.status,
            body=answer.body,
            source_version=(
                answer.headers.get("X-Datasets-Version")
                or answer.headers.get("Last-Modified")
                or answer.headers.get("ETag")
            ),
            observed_at=answer.observed_at,
            elapsed_time_ms=answer.elapsed_time_ms,
            content_length=_positive_or_zero_int(answer.headers.get("Content-Length")),
        )


class NcbiDatasetsReferenceAdapter:
    """Translate current NCBI JSON mechanics into strict source observations."""

    adapter_version = ADAPTER_VERSION

    def __init__(self, transport: MetadataTransport | None = None) -> None:
        self._transport = transport or UrllibNcbiTransport()

    @property
    def contact_count(self) -> int:
        return self._transport.contact_count

    def taxon_term_refusal(self, term: str) -> str | None:
        """NCBI Taxonomy reads a comma in its ``{taxons}`` path as a list of taxa."""
        return "taxon_term_not_one_taxon" if "," in term else None

    def resolve_taxon(self, query: str) -> TaxonomyAdapterObservation:
        if (refusal := self.taxon_term_refusal(query)) is not None:
            raise ReferenceQueryRefused(refusal, query)
        encoded = urllib.parse.quote(query, safe="")
        response, payload, source = self._json_get(
            f"https://{_OFFICIAL_HOST}/datasets/v2/taxonomy/taxon/{encoded}/dataset_report",
            authority="ncbi_taxonomy",
        )
        del response
        raw_nodes = payload.get(
            "taxonomy_nodes", payload.get("taxonomyNodes", payload.get("reports", ()))
        )
        if not isinstance(raw_nodes, list) or len(raw_nodes) > 10:
            raise ReferenceAdapterError(
                "invalid_response",
                "NCBI Taxonomy did not return a bounded taxonomy-node list.",
                retryable=False,
            )
        records = tuple(
            record for raw in raw_nodes if (record := _taxonomy_record(raw)) is not None
        )
        if (
            not records
            and len(raw_nodes) == 1
            and set(payload) == {"reports"}
            and _explicit_taxonomy_no_match(raw_nodes[0], query=query)
        ):
            return TaxonomyAdapterObservation(records=(), source=source)
        if len(records) != len(raw_nodes):
            raise ReferenceAdapterError(
                "invalid_response",
                "NCBI Taxonomy returned a malformed taxonomy node.",
                retryable=False,
            )
        return TaxonomyAdapterObservation(records=records, source=source)

    def discover_assemblies(
        self, ncbi_tax_id: int, query: AssemblyQuery
    ) -> AssemblyAdapterObservation:
        """One page of NCBI's genome reports for the taxon under the model's query,
        each row copied, with NCBI's ``total_count`` and next page token (#471)."""
        _response, payload, source = self._json_get(
            genome_report_url(
                f"{ncbi_tax_id}?{urllib.parse.urlencode(query.parameters())}"
            ),
            authority="ncbi_datasets",
        )
        records, total_count, next_page_token = _genome_report_page(payload)
        return AssemblyAdapterObservation(
            records=records,
            source=source,
            total_count=total_count,
            next_page_token=next_page_token,
        )

    def describe_assembly(self, assembly_accession: str) -> AssemblyAdapterObservation:
        """Describe the accession, current or not: when NCBI says it is no longer
        current (`not_current`), the replaced or suppressed version comes from its
        any-version report, two GETs then on the caller's one surface. A discovery
        page may list a replaced version the model chooses (#471 live a2)."""
        url = assembly_report_url(assembly_accession)
        response = self._transport.get(url, max_bytes=response_byte_bound())
        if not not_current(response.status_code, response.body):
            return self._describe_exact(url, response)
        # An empty report decides a second contact only once it is the exact
        # answer to this request.
        _exact_response(response, url)
        return self._describe_exact(f"{url}?{ASSEMBLY_ANY_VERSION}")

    def _describe_exact(
        self, url: str, response: MetadataResponse | None = None
    ) -> AssemblyAdapterObservation:
        _response, payload, source = self._json_get(
            url, authority="ncbi_datasets", response=response
        )
        records = _assembly_records(payload, "exact-assembly report list")
        return AssemblyAdapterObservation(
            records=records, source=source, total_count=len(records)
        )

    def describe_companions(
        self, assembly_accession: str, role: CompanionRole
    ) -> CompanionAdapterObservation:
        """Describe an exact assembly-qualified package slice without acquiring bytes."""

        encoded = urllib.parse.quote(assembly_accession, safe="_.")
        _response, payload, source = self._json_get(
            f"https://{_OFFICIAL_HOST}/datasets/v2/genome/accession/{encoded}/download_summary",
            authority="ncbi_datasets",
        )
        if "available_files" in payload:
            records = _current_companion_records(
                payload,
                assembly_accession=assembly_accession,
                role=role,
                source=source,
            )
            if role == "genomic_fasta" and records:
                publisher_file = self._publisher_genomic_fasta(assembly_accession)
                records = tuple(
                    CompanionSourceRecord.model_validate(
                        {**record.model_dump(), **publisher_file}
                    )
                    for record in records
                )
            return CompanionAdapterObservation(
                records=records,
                source=source,
                source_truncated=False,
            )

        raw_reports = payload.get("reports")
        if not isinstance(raw_reports, list):
            raise ReferenceAdapterError(
                "invalid_response",
                "NCBI Datasets did not return a download-summary report list.",
                retryable=False,
            )
        role_tokens = {
            "genomic_fasta": {
                "GENOME_FASTA",
                "GENOMIC_FASTA",
                "GENOMIC_NUCLEOTIDE_FASTA",
            },
            "annotation": {"GFF3", "GTF", "RNA_FASTA", "PROTEIN_FASTA"},
        }[role]
        records: list[CompanionSourceRecord] = []
        for report in raw_reports:
            if not isinstance(report, Mapping):
                raise ReferenceAdapterError(
                    "invalid_response",
                    "NCBI Datasets returned a malformed download-summary report.",
                    retryable=False,
                )
            report_accession = _first(report, (("accession",),)) or assembly_accession
            files = report.get("files", ())
            if not isinstance(files, list) or len(files) > 100:
                raise ReferenceAdapterError(
                    "invalid_response",
                    "NCBI Datasets returned a malformed download-summary file list.",
                    retryable=False,
                )
            for raw_file in files:
                if not isinstance(raw_file, Mapping):
                    raise ReferenceAdapterError(
                        "invalid_response",
                        "NCBI Datasets returned malformed companion metadata.",
                        retryable=False,
                    )
                file_type = _first(
                    raw_file, (("file_type",), ("fileType",), ("type",), ("role",))
                )
                if file_type is None or file_type.upper() not in role_tokens:
                    continue
                # Preserve the source-owned identity byte-for-byte after canonical whitespace
                # validation.  Never replace it with a filename/path-derived digest.
                asset_id = _opaque_asset_id(
                    raw_file,
                    (
                        ("asset_id",),
                        ("assetId",),
                        ("file_id",),
                        ("fileId",),
                        ("file_name",),
                    ),
                )
                if asset_id is None:
                    continue
                checksum_algorithm, checksum = _companion_checksum(raw_file)
                records.append(
                    CompanionSourceRecord(
                        asset_id=asset_id,
                        assembly_accession=report_accession,
                        role=role,
                        publisher="NCBI Datasets",
                        release_version=(
                            _first(
                                raw_file,
                                (
                                    ("release_version",),
                                    ("releaseVersion",),
                                    ("version",),
                                ),
                            )
                            or report_accession
                        ),
                        representation=file_type,
                        byte_size=_positive_or_zero_int(
                            raw_file.get("file_size", raw_file.get("fileSize"))
                        ),
                        checksum_algorithm=checksum_algorithm,
                        checksum=checksum,
                        authority_locator=source.request_identity,
                    )
                )
        return CompanionAdapterObservation(
            records=tuple(records),
            source=source,
            source_truncated=bool(
                payload.get("next_page_token") or payload.get("nextPageToken")
            ),
        )

    def _publisher_genomic_fasta(self, assembly_accession: str) -> dict[str, object]:
        """Derive the genomes-FTP file, then observe its listed md5 and HEAD size.

        A missing listing entry or size is a typed refusal, never a filled-in value.
        """

        directory = _genomes_directory(
            assembly_accession, self._assembly_name(assembly_accession)
        )
        stem = directory.rsplit("/", 1)[1]
        fasta = f"{directory}/{stem}_genomic.fna.gz"
        listing = f"{directory}/md5checksums.txt"
        checksums = _exact_response(
            self._transport.get(listing, max_bytes=response_byte_bound()),
            listing,
        )
        if response_over_bound(len(checksums.body)):
            raise ReferenceAdapterError(
                "response_too_large",
                "NCBI md5checksums.txt exceeded the configured byte bound.",
                retryable=False,
            )
        md5 = _listed_md5(checksums.body, f"./{stem}_genomic.fna.gz")
        head = _exact_response(self._transport.head(fasta), fasta)
        if not head.content_length:
            raise ReferenceAdapterError(
                "invalid_response",
                "NCBI did not report the genomic FASTA's byte size.",
                retryable=False,
            )
        return {
            "source_locator": fasta,
            "byte_size": head.content_length,
            "checksum_algorithm": "md5",
            "checksum": md5,
            "checksum_authority_locator": listing,
        }

    def fetch_assembly_sequences(
        self, genomic_fasta_locator: str
    ) -> tuple[SequenceRecord, ...]:
        """The sequences of the assembly whose genomic FASTA is ``genomic_fasta_locator``,
        from the assembly report beside it: each one's names and length (#365)."""
        prefix = f"https://{_GENOMES_HOST}/genomes/all/"
        if not genomic_fasta_locator.startswith(
            prefix
        ) or not genomic_fasta_locator.endswith("_genomic.fna.gz"):
            raise ReferenceAdapterError(
                "invalid_response",
                "The genomic FASTA is not an NCBI genomes file.",
                retryable=False,
            )
        url = genomic_fasta_locator.removesuffix("_genomic.fna.gz")
        url += "_assembly_report.txt"
        response = _exact_response(
            self._transport.get(url, max_bytes=response_byte_bound()), url
        )
        return parse_assembly_report(response.body)

    def fetch_accession_sequences(
        self, assembly_accession: str
    ) -> tuple[SequenceRecord, ...]:
        """The sequences of ``assembly_accession`` from its assembly report, in the FTP
        directory NCBI Datasets' ``links`` answer names (`report_url`, #467): two GETs,
        for an index read before any companion is described (owner, #459 D1)."""
        encoded = urllib.parse.quote(assembly_accession, safe="_.")
        url = f"https://{_OFFICIAL_HOST}/datasets/v2/genome/accession/{encoded}/links"
        links = _exact_response(
            self._transport.get(url, max_bytes=response_byte_bound()), url
        )
        try:
            report = report_url(links.body)
        except ReportUnreadable as error:
            raise ReferenceAdapterError(
                "invalid_response", str(error), retryable=False
            ) from error
        response = _exact_response(
            self._transport.get(report, max_bytes=response_byte_bound()), report
        )
        return parse_assembly_report(response.body)

    def _assembly_name(self, assembly_accession: str) -> str:
        names = {
            record.assembly_name
            for record in self.describe_assembly(assembly_accession).records
            if record.assembly_accession == assembly_accession
        }
        if len(names) != 1:
            raise ReferenceAdapterError(
                "source_disagreement",
                "NCBI Datasets did not report one assembly name for the accession.",
                retryable=False,
            )
        return names.pop()

    def _json_get(
        self,
        url: str,
        *,
        authority: str,
        response: MetadataResponse | None = None,
    ) -> tuple[MetadataResponse, Mapping[str, object], SourceObservation]:
        """Read one JSON answer: the one already received, or one GET of ``url``."""
        if response is None:
            response = self._transport.get(url, max_bytes=response_byte_bound())
        if response_over_bound(len(response.body)):
            raise ReferenceAdapterError(
                "response_too_large",
                "Official NCBI metadata exceeded the configured byte bound.",
                retryable=False,
            )
        _exact_response(response, url)
        try:
            payload = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReferenceAdapterError(
                "invalid_response",
                "Official NCBI metadata was not valid JSON.",
                retryable=False,
            ) from error
        if not isinstance(payload, Mapping):
            raise ReferenceAdapterError(
                "invalid_response",
                "Official NCBI metadata JSON must be an object.",
                retryable=False,
            )
        digest = hashlib.sha256(response.body).hexdigest()
        source = SourceObservation(
            authority=authority,
            adapter_version=self.adapter_version,
            source_version=response.source_version or f"content-sha256:{digest}",
            request_identity=response.request_identity,
            response_sha256=digest,
            observed_at=response.observed_at,
            citation_url=response.citation_url,
            elapsed_time_ms=response.elapsed_time_ms,
            response_bytes=len(response.body),
        )
        return response, payload, source


# NCBI Datasets' documented query parameters of a taxon's genome reports (OpenAPI v2,
# `/genome/taxon/{taxons}/dataset_report`), the ones a query may send.
GENOME_REPORT_PARAMETERS = frozenset(
    {
        "filters.reference_only",
        "filters.assembly_source",
        "filters.has_annotation",
        "filters.exclude_paired_reports",
        "filters.exclude_atypical",
        "filters.assembly_version",
        "filters.assembly_level",
        "filters.first_release_date",
        "filters.last_release_date",
        "filters.search_text",
        "filters.is_metagenome_derived",
        "filters.is_type_material",
        "filters.is_ictv_exemplar",
        "filters.exclude_multi_isolate",
        "filters.type_material_category",
        "tax_exact_match",
        "sort.field",
        "sort.direction",
        "page_size",
        "page_token",
    }
)


def genome_report_url(taxon_query: str) -> str:
    """The URL of a taxon's genome reports for ``<taxid>?<query>``: the query's
    parameters are NCBI's documented ones, each value percent-encoded as sent."""
    taxid, _, query = taxon_query.partition("?")
    if not re.fullmatch(r"[1-9][0-9]*", taxid):
        raise ValueError("a taxon's genome reports take one NCBI taxon id")
    pairs = urllib.parse.parse_qsl(
        query, keep_blank_values=True, strict_parsing=bool(query)
    )
    if any(name not in GENOME_REPORT_PARAMETERS for name, _ in pairs):
        raise ValueError("a genome-report query sends only NCBI's documented options")
    encoded = urllib.parse.urlencode(pairs)
    return (
        f"https://{_OFFICIAL_HOST}/datasets/v2/genome/taxon/{taxid}/dataset_report"
        + (f"?{encoded}" if encoded else "")
    )


def genome_report_page(
    body: bytes,
) -> tuple[tuple[AssemblySourceRecord, ...], int | None, str | None]:
    """A genome-report answer's copied rows, NCBI's ``total_count`` and next page
    token; an answer that is not one raises ``ValueError``."""
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("NCBI's genome reports were not JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("NCBI's genome reports were not a JSON object")
    try:
        return _genome_report_page(payload)
    except ReferenceAdapterError as error:
        raise ValueError(error.failure.public_message) from error


def _genome_report_page(
    payload: Mapping[str, object],
) -> tuple[tuple[AssemblySourceRecord, ...], int | None, str | None]:
    return (
        _assembly_records(payload, "assembly-report list", listed=True),
        _positive_or_zero_int(payload.get("total_count")),
        _first(payload, (("next_page_token",),)),
    )


ASSEMBLY_ANY_VERSION = "filters.assembly_version=all_assemblies"
"""The query that lists every version of an assembly or a taxon: the current one and
the replaced and suppressed ones."""


def assembly_report_url(assembly_accession: str) -> str:
    """NCBI Datasets' report of one versioned assembly."""
    encoded = urllib.parse.quote(assembly_accession, safe="_.")
    return f"https://{_OFFICIAL_HOST}/datasets/v2/genome/accession/{encoded}/dataset_report"


def not_current(status: int, body: bytes) -> bool:
    """Whether an assembly's plain report says it is no longer current.

    NCBI answers the plain report of a replaced or suppressed assembly with HTTP 200
    and exactly ``{}`` (seen for ``GCF_000003025.6`` and ``GCF_000002315.4``) and
    describes it only in the any-version report; any other answer keeps its meaning.
    """
    return status == 200 and body == b"{}"


def parse_assembly_report(body: bytes) -> tuple[SequenceRecord, ...]:
    """Every row of an NCBI ``_assembly_report.txt``, copied by the one reader
    (`ncbi_assembly_report`, #467); a report it cannot read is NCBI's invalid response."""
    try:
        return tuple(report_rows(body))
    except ReportUnreadable as unreadable:
        raise ReferenceAdapterError(
            "invalid_response", str(unreadable), retryable=False
        ) from unreadable


def _exact_response(response: MetadataResponse, url: str) -> MetadataResponse:
    if response.request_identity != url or response.citation_url != url:
        raise ReferenceAdapterError(
            "source_disagreement",
            "NCBI transport changed the exact request identity.",
            retryable=False,
        )
    if response.status_code != 200:
        raise ReferenceAdapterError(
            "http_error",
            f"Official NCBI metadata returned HTTP {response.status_code}.",
            retryable=response.status_code >= 500,
        )
    return response


def _genomes_directory(assembly_accession: str, assembly_name: str) -> str:
    """NCBI's genomes/all layout: accession digits in threes, then accession_name."""

    match = _GENOMES_ACCESSION.fullmatch(assembly_accession)
    if match is None:
        raise ReferenceAdapterError(
            "invalid_response",
            "The assembly accession has no NCBI genomes directory.",
            retryable=False,
        )
    prefix, first, second, third = match.groups()
    # The directory spells the name with every character outside [A-Za-z0-9._-] as "_".
    stem = f"{assembly_accession}_{re.sub(r'[^A-Za-z0-9._-]', '_', assembly_name)}"
    return (
        f"https://{_GENOMES_HOST}/genomes/all/{prefix}/{first}/{second}/{third}/{stem}"
    )


def _listed_md5(body: bytes, listed_name: str) -> str:
    """Return the one md5 the listing publishes for exactly ``listed_name``."""

    try:
        lines = body.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ReferenceAdapterError(
            "invalid_response",
            "NCBI md5checksums.txt was not ASCII text.",
            retryable=False,
        ) from error
    listed = {
        checksum
        for line in lines
        if len(parts := line.split()) == 2
        and parts[1] == listed_name
        and len(checksum := parts[0].lower()) == 32
        and all(character in "0123456789abcdef" for character in checksum)
    }
    if len(listed) != 1:
        raise ReferenceAdapterError(
            "invalid_response",
            "NCBI md5checksums.txt has no single md5 for the genomic FASTA.",
            retryable=False,
        )
    return listed.pop()


def _nested(value: Mapping[str, object], *keys: str) -> object | None:
    current: object = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first(value: Mapping[str, object], paths: Sequence[tuple[str, ...]]) -> str | None:
    for path in paths:
        selected = _nested(value, *path)
        if selected not in (None, ""):
            text = str(selected).strip()
            if text:
                return text
    return None


def _opaque_asset_id(
    value: Mapping[str, object], paths: Sequence[tuple[str, ...]]
) -> str | None:
    """Return only an exact source string; canonical validation belongs to the contract."""

    for path in paths:
        selected = _nested(value, *path)
        if isinstance(selected, str) and selected:
            return selected
    return None


def _current_companion_records(
    payload: Mapping[str, object],
    *,
    assembly_accession: str,
    role: CompanionRole,
    source: SourceObservation,
) -> tuple[CompanionSourceRecord, ...]:
    """Adapt the aggregate v2 download summary without inventing file-level facts."""

    available = payload.get("available_files")
    if not isinstance(available, Mapping) or len(available) > 100:
        raise ReferenceAdapterError(
            "invalid_response",
            "NCBI Datasets returned malformed aggregate companion metadata.",
            retryable=False,
        )
    release_version = (
        _first(payload, (("resource_updated_on",), ("resourceUpdatedOn",)))
        or source.source_version
    )
    records: list[CompanionSourceRecord] = []
    for asset_id, metadata in available.items():
        if (
            not isinstance(asset_id, str)
            or not asset_id.isascii()
            or not 1 <= len(asset_id) <= 100
            or not asset_id[0].isalpha()
            or asset_id != asset_id.lower()
            or any(
                not (character.isalnum() or character == "_") for character in asset_id
            )
            or not isinstance(metadata, Mapping)
        ):
            raise ReferenceAdapterError(
                "invalid_response",
                "NCBI Datasets returned malformed aggregate companion metadata.",
                retryable=False,
            )
        file_count = (
            0 if not metadata else metadata.get("file_count", metadata.get("fileCount"))
        )
        if (
            isinstance(file_count, bool)
            or not isinstance(file_count, int)
            or file_count < 0
        ):
            raise ReferenceAdapterError(
                "invalid_response",
                "NCBI Datasets returned an invalid aggregate companion count.",
                retryable=False,
            )
        if file_count == 0 or asset_id not in _CURRENT_COMPANION_KEYS[role]:
            continue
        records.append(
            CompanionSourceRecord(
                asset_id=asset_id,
                assembly_accession=assembly_accession,
                role=role,
                publisher="NCBI Datasets",
                release_version=release_version,
                representation=asset_id,
                # ``size_mb`` is aggregate/approximate, not an exact byte count.
                byte_size=None,
                checksum_algorithm=None,
                checksum=None,
                authority_locator=source.request_identity,
            )
        )
    return tuple(records)


def _positive_int(value: object) -> int | None:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _positive_or_zero_int(value: object) -> int | None:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _companion_checksum(
    raw: Mapping[str, object],
) -> tuple[Literal["md5", "sha256", "sha512"] | None, str | None]:
    for algorithm, keys, size in (
        ("md5", ("md5", "md5_checksum"), 32),
        ("sha256", ("sha256", "sha256_checksum"), 64),
        ("sha512", ("sha512", "sha512_checksum"), 128),
    ):
        for key in keys:
            value = raw.get(key)
            if isinstance(value, str):
                canonical = value.strip().lower()
                if len(canonical) == size and all(
                    character in "0123456789abcdef" for character in canonical
                ):
                    return algorithm, canonical  # type: ignore[return-value]
    checksum = raw.get("checksum")
    algorithm = raw.get("checksum_algorithm", raw.get("checksumAlgorithm"))
    if isinstance(checksum, str) and isinstance(algorithm, str):
        selected = algorithm.strip().lower()
        if selected in {"md5", "sha256", "sha512"}:
            size = {"md5": 32, "sha256": 64, "sha512": 128}[selected]
            canonical = checksum.strip().lower()
            if len(canonical) == size and all(
                character in "0123456789abcdef" for character in canonical
            ):
                return selected, canonical  # type: ignore[return-value]
    return None, None


def _taxonomy_record(raw: object) -> TaxonomySourceRecord | None:
    if not isinstance(raw, Mapping):
        return None
    taxonomy = raw.get("taxonomy", raw)
    if not isinstance(taxonomy, Mapping):
        return None
    tax_id = _positive_int(taxonomy.get("tax_id", taxonomy.get("taxId")))
    scientific_name = _first(
        taxonomy,
        (
            ("current_scientific_name", "name"),
            ("currentScientificName", "name"),
            ("organism_name",),
            ("organismName",),
            ("scientific_name",),
            ("scientificName",),
            ("name",),
        ),
    )
    if tax_id is None or scientific_name is None:
        return None
    lineage_raw = taxonomy.get("lineage", ())
    lineage_names: list[str] = []
    lineage_ids: list[int] = []
    if isinstance(lineage_raw, list):
        for item in lineage_raw:
            if isinstance(item, Mapping):
                name = _first(
                    item, (("scientific_name",), ("scientificName",), ("name",))
                )
                lineage_id = _positive_int(item.get("tax_id", item.get("taxId")))
            else:
                name = str(item).strip() or None
                lineage_id = _positive_int(item)
            if name is not None:
                lineage_names.append(name[:300])
            if lineage_id is not None:
                lineage_ids.append(lineage_id)
    merged_raw = taxonomy.get(
        "merged_or_secondary_tax_ids",
        taxonomy.get("mergedOrSecondaryTaxIds", ()),
    )
    merged = (
        tuple(
            value for item in merged_raw if (value := _positive_int(item)) is not None
        )
        if isinstance(merged_raw, list)
        else ()
    )
    return TaxonomySourceRecord(
        ncbi_tax_id=tax_id,
        scientific_name=scientific_name[:300],
        rank=_first(taxonomy, (("rank",),)),
        lineage=tuple(lineage_names),
        lineage_tax_ids=tuple(lineage_ids),
        merged_or_secondary_tax_ids=merged,
        provisional=bool(taxonomy.get("provisional")),
        deleted=bool(taxonomy.get("deleted")),
    )


def _explicit_taxonomy_no_match(raw: object, *, query: str) -> bool:
    """Recognize only NCBI's bounded explicit invalid-identifier report."""

    if not isinstance(raw, Mapping) or set(raw) != {"query", "errors"}:
        return False
    queries = raw.get("query")
    errors = raw.get("errors")
    if (
        not isinstance(queries, list)
        or not 1 <= len(queries) <= 10
        or not isinstance(errors, list)
        or not 1 <= len(errors) <= 10
    ):
        return False
    if queries != [query]:
        return False
    for error in errors:
        if not isinstance(error, Mapping) or set(error) != {
            "invalid_identifiers",
            "reason",
        }:
            return False
        invalid = error.get("invalid_identifiers")
        reason = error.get("reason")
        if (
            invalid != [query]
            or not isinstance(reason, str)
            or not reason
            or reason.strip() != reason
            or len(reason) > 500
        ):
            return False
    return True


def _assembly_records(
    payload: Mapping[str, object], what: str, *, listed: bool = False
) -> tuple[AssemblySourceRecord, ...]:
    """Every row of a genome-report answer, copied; a row with no exact accession is
    no row NCBI's answer can be cited by, and makes the answer NCBI's invalid
    response. A listing that matched nothing (``listed``, ``total_count`` 0) writes
    no report list; an exact report always writes one."""
    raw_reports = payload.get("reports")
    if raw_reports is None and listed and payload.get("total_count") in (None, 0):
        raw_reports = []
    if not isinstance(raw_reports, list):
        raise ReferenceAdapterError(
            "invalid_response",
            f"NCBI Datasets did not return an {what}.",
            retryable=False,
        )
    try:
        records = tuple(_assembly_record(raw) for raw in raw_reports)
    except ValueError:
        records = (None,)
    if None in records:
        raise ReferenceAdapterError(
            "invalid_response",
            f"NCBI Datasets returned an {what} with a row that names no accession.",
            retryable=False,
        )
    return records  # type: ignore[return-value]


def _assembly_record(raw: object) -> AssemblySourceRecord | None:
    """One genome report's fields as NCBI writes them (#471): only the accession is
    required, and nothing is read for a verdict."""
    if not isinstance(raw, Mapping) or _first(raw, (("accession",),)) is None:
        return None
    info = ("assembly_info",)
    names = _nested(raw, "organism", "infraspecific_names")
    names = names if isinstance(names, Mapping) else {}
    notes = _nested(raw, "assembly_info", "genome_notes")
    return AssemblySourceRecord(
        assembly_accession=_first(raw, (("accession",),)),
        current_accession=_first(raw, (("current_accession",),)),
        paired_accession=_first(
            raw, (("paired_accession",), (*info, "paired_assembly", "accession"))
        ),
        source_database=_first(raw, (("source_database",),)),
        organism_name=_first(raw, (("organism", "organism_name"),)),
        common_name=_first(raw, (("organism", "common_name"),)),
        ncbi_tax_id=_positive_int(_nested(raw, "organism", "tax_id")),
        infraspecific_names=InfraspecificNames(
            **{
                field: names[field]
                for field in InfraspecificNames.model_fields
                if isinstance(names.get(field), str) and names[field]
            }
        ),
        assembly_name=_first(raw, ((*info, "assembly_name"),)),
        synonym=_first(raw, ((*info, "synonym"),)),
        assembly_status=_first(raw, ((*info, "assembly_status"),)),
        paired_assembly_status=_first(raw, ((*info, "paired_assembly", "status"),)),
        suppression_reason=_first(raw, ((*info, "suppression_reason"),)),
        refseq_category=_first(raw, ((*info, "refseq_category"),)),
        assembly_level=_first(raw, ((*info, "assembly_level"),)),
        assembly_type=_first(raw, ((*info, "assembly_type"),)),
        release_date=_first(raw, ((*info, "release_date"),)),
        submitter=_first(raw, ((*info, "submitter"),)),
        genome_notes=tuple(
            note
            for note in (notes if isinstance(notes, list) else ())
            if isinstance(note, str) and note
        ),
    )


__all__ = [
    "parse_assembly_report",
    "ADAPTER_VERSION",
    "ASSEMBLY_ANY_VERSION",
    "MetadataResponse",
    "MetadataTransport",
    "NcbiDatasetsReferenceAdapter",
    "UrllibNcbiTransport",
    "assembly_report_url",
    "not_current",
]
