"""Bounded, retry-free NCBI mechanics for Investigation source Tools."""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, replace
from typing import Protocol

from bio_neb.agentic_journey.source_answers import source_answered
from bio_neb.agentic_journey.source_contacts import response_byte_bound
from bio_neb.failure import ProviderError
from bio_neb.providers.ncbi_assembly_report import ReportUnreadable, report_url
from bio_neb.providers.ncbi_datasets import (
    ASSEMBLY_ANY_VERSION,
    MetadataTransport,
    UrllibNcbiTransport,
    assembly_report_url,
    genome_report_url,
    not_current,
)
from bio_neb.providers.source_transport import AnsweredStatus


@dataclass(frozen=True)
class InvestigationSourceResponse:
    request_identity: str
    status_code: int
    body: bytes
    observed_at: str
    elapsed_time_ms: int
    # The same operation's earlier GETs, in call order, when it makes more than one
    # (an assembly's plain report, then its any-version one): retained and shown too.
    preceding: tuple[InvestigationSourceResponse, ...] = ()
    # What the body is, as it is retained: NCBI's FTP assembly report is text.
    media_type: str = "application/json"


def unanswered(status_code: int, detail: str) -> ProviderError:
    """The fault of a status that is no answer: transient for a rate limit or a
    server fault, rejected otherwise."""
    return ProviderError(
        "transient" if status_code == 429 or status_code >= 500 else "rejected",
        f"{detail} with HTTP {status_code}.",
    )


def checked_source_response(
    response: InvestigationSourceResponse,
) -> InvestigationSourceResponse:
    """Classify the provider envelope before biological normalization.

    A source's answer about the request passes with its body, whatever the body holds:
    the lookup ends a non-2xx one `source_answered`, the reader types only what it
    cannot read (`malformed`), and the Tool retains and shows the source's own bytes
    (#458 Spec r1 SP3; #468 R1 b, R4 a). Any other status (429, 5xx, an access fault)
    raises here, for the failure module to retry or ask (owner #468 SP4 a).
    """
    if not source_answered(response.status_code):
        raise unanswered(response.status_code, "Source rejected its metadata request")
    return response


class InvestigationSourceAdapter(Protocol):
    contact_count: int

    def fetch_refsnp(self, rsid: str) -> InvestigationSourceResponse: ...

    def fetch_assembly_report(
        self, assembly_accession: str
    ) -> InvestigationSourceResponse: ...

    def fetch_sequence_names(
        self, assembly_accession: str
    ) -> InvestigationSourceResponse: ...

    def fetch_reference_span(self, span: str) -> InvestigationSourceResponse: ...

    def fetch_spdi_rsids(self, spdi: str) -> InvestigationSourceResponse: ...

    def fetch_taxon_assemblies(
        self, taxon_query: str
    ) -> InvestigationSourceResponse: ...

    def fetch_spdi_equivalents(self, spdi: str) -> InvestigationSourceResponse: ...

    def fetch_genome_annotation(
        self, assembly_accession: str
    ) -> InvestigationSourceResponse: ...

    def fetch_hgvs_contextuals(self, name: str) -> InvestigationSourceResponse: ...


_ACCESSION = re.compile(r"^GC[AF]_[0-9]{9}\.[0-9]+$")
_SPAN = re.compile(
    r"^(?P<sequence>[^:]+):(?P<start>0|[1-9][0-9]*):(?P<length>[1-9][0-9]*)$"
)
_SPDI = re.compile(r"^(?P<sequence>[^:]+):(?P<start>0|[1-9][0-9]*):[ACGTN]*:[ACGTN]*$")
_VARIATION = "https://api.ncbi.nlm.nih.gov/variation/v0"
_DATASETS = "https://api.ncbi.nlm.nih.gov/datasets/v2"


def _segment(key: str) -> str:
    """A key as one URL path segment: percent-encoded, its SPDI ``:`` kept (ADR 0049:
    URL safety by encoding, never by the shape of a sequence name)."""
    return urllib.parse.quote(key, safe=":")


class NcbiInvestigationSourceAdapter:
    """Perform only NCBI's Investigation GETs: a RefSNP record, an assembly's report
    and sequence names, reference bases at a span and the rsIDs at an SPDI allele."""

    def __init__(self, transport: MetadataTransport | None = None) -> None:
        self._transport = transport or UrllibNcbiTransport()

    @property
    def contact_count(self) -> int:
        return self._transport.contact_count

    def fetch_refsnp(self, rsid: str) -> InvestigationSourceResponse:
        numeric = rsid.removeprefix("rs")
        if not numeric.isdigit() or not numeric or numeric.startswith("0"):
            raise ValueError("RefSNP request requires one canonical rsID")
        return self._get("https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/" + numeric)

    def fetch_assembly_report(
        self, assembly_accession: str
    ) -> InvestigationSourceResponse:
        """The assembly's report: its current one or, when NCBI says it is no longer
        current (`not_current`), its any-version report, a second GET on the same
        surface whose URL names the query it asked."""
        url = assembly_report_url(assembly_accession)
        response = self._get(url)
        if not not_current(response.status_code, response.body):
            return response
        return replace(
            self._get(f"{url}?{ASSEMBLY_ANY_VERSION}"), preceding=(response,)
        )

    def fetch_sequence_names(
        self, assembly_accession: str
    ) -> InvestigationSourceResponse:
        """The assembly's whole assembly report, every sequence with every name and role
        (#467): NCBI Datasets' ``links`` answer names its FTP directory, then the report
        is read there. An unreadable ``links`` answer is the operation's answer."""
        if not _ACCESSION.fullmatch(assembly_accession):
            raise ValueError("a sequence-names report takes one assembly accession")
        links = self._get(
            f"{_DATASETS}/genome/accession/"
            f"{urllib.parse.quote(assembly_accession, safe='')}/links"
        )
        if not 200 <= links.status_code < 300:
            return links
        try:
            url = report_url(links.body)
        except ReportUnreadable:
            return links
        return replace(self._get(url), preceding=(links,), media_type="text/plain")

    def fetch_reference_span(self, span: str) -> InvestigationSourceResponse:
        """A sequence's bases at ``sequence:start0:length``: the Variation service's
        contextual form of a substitution by as many ``N``, whose deleted sequence is
        the reference itself (an ``N`` never shares a reference base)."""
        match = _SPAN.fullmatch(span)
        if match is None:
            raise ValueError("a reference span takes sequence:start:length")
        substitution = f"{span}:{'N' * int(match['length'])}"
        return self._get(f"{_VARIATION}/spdi/{_segment(substitution)}/contextual")

    def fetch_spdi_rsids(self, spdi: str) -> InvestigationSourceResponse:
        """The rsIDs dbSNP lists for one SPDI allele; a 404 lists none."""
        match = _SPDI.fullmatch(spdi)
        if match is None:
            raise ValueError("an rsID listing takes one SPDI allele")
        return self._get(f"{_VARIATION}/spdi/{_segment(spdi)}/rsids")

    def fetch_taxon_assemblies(self, taxon_query: str) -> InvestigationSourceResponse:
        """One page of NCBI's genome reports for ``<taxid>?<query>``, the model's
        NCBI options as sent (#471 R1 a); code adds no filter of its own."""
        return self._get(genome_report_url(taxon_query))

    def fetch_genome_annotation(
        self, assembly_accession: str
    ) -> InvestigationSourceResponse:
        """An assembly's GFF3 annotation, downloaded as NCBI Datasets' zip package,
        retained as the zip it is (#473: a Run's placement keeps it in the Journey)."""
        if not _ACCESSION.fullmatch(assembly_accession):
            raise ValueError("an annotation download takes one assembly accession")
        response = self._get(
            f"{_DATASETS}/genome/accession/{assembly_accession}/download"
            "?include_annotation_type=GENOME_GFF",
            accept="application/zip",
        )
        return replace(response, media_type="application/zip")

    def fetch_hgvs_contextuals(self, name: str) -> InvestigationSourceResponse:
        """The Variation service's reading of one HGVS name as contextual alleles. The
        name is one percent-encoded path segment (ADR 0049: URL safety by encoding,
        never by an HGVS grammar; NCBI answers what it cannot read)."""
        return self._get(
            f"{_VARIATION}/hgvs/{urllib.parse.quote(name, safe=':.')}/contextuals"
        )

    def fetch_spdi_equivalents(self, spdi: str) -> InvestigationSourceResponse:
        """The Variation service's equivalents of one SPDI allele on every sequence."""
        match = _SPDI.fullmatch(spdi)
        if match is None:
            raise ValueError("an SPDI's equivalents take one SPDI allele")
        return self._get(
            f"{_VARIATION}/spdi/{_segment(spdi)}/all_equivalent_contextual"
        )

    def _get(self, url: str, **options: str) -> InvestigationSourceResponse:
        try:
            response = self._transport.get(
                url, max_bytes=response_byte_bound(), **options
            )
        except ProviderError as error:
            # NCBI's transport refuses a status as its own error type; the source's
            # answer about the request is the status it carries.
            cause = error.__cause__
            if not isinstance(cause, AnsweredStatus) or not source_answered(
                cause.answer.status
            ):
                raise
            # The source's answer about the request: its body is the evidence.
            answer = cause.answer
            return InvestigationSourceResponse(
                request_identity=url,
                status_code=answer.status,
                body=answer.body,
                observed_at=answer.observed_at.isoformat().replace("+00:00", "Z"),
                elapsed_time_ms=answer.elapsed_time_ms,
            )
        return InvestigationSourceResponse(
            request_identity=response.request_identity,
            status_code=response.status_code,
            body=response.body,
            observed_at=response.observed_at.isoformat().replace("+00:00", "Z"),
            elapsed_time_ms=response.elapsed_time_ms,
        )


__all__ = [
    "InvestigationSourceAdapter",
    "InvestigationSourceResponse",
    "NcbiInvestigationSourceAdapter",
    "checked_source_response",
    "unanswered",
]
