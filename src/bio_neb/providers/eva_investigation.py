"""Bounded, retry-free European Variation Archive mechanics for the EVA lookup Tool.

EVA answers a non-human Run's rsID lookups (#366). Each method is one GET on a path EVA
documents: its identifiers service (a clustered variant and its submitted variants,
`/eva/webservices/identifiers/v3/api-docs`) and its contig-alias service (an INSDC
sequence's other names and its GenBank/RefSeq assembly pair). An answer about the
request (a 404, a 410) is the lookup's to read; the failure module owns retries (`source_contacts.call_source`).
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Protocol

from bio_neb.agentic_journey.source_answers import answer_received, source_answered
from bio_neb.agentic_journey.source_contacts import response_byte_bound
from bio_neb.failure import ProviderError
from bio_neb.providers.ncbi_investigation import (
    InvestigationSourceAdapter,
    InvestigationSourceResponse,
    NcbiInvestigationSourceAdapter,
    checked_source_response,
    unanswered,
)
from bio_neb.providers.source_operations import SOURCE_OPERATIONS
from bio_neb.providers.source_transport import (
    TIMEOUT_SECONDS,
    Opener,
    SourceTransport,
)

EVA_HOST = "www.ebi.ac.uk"
EVA_IDENTIFIERS = f"https://{EVA_HOST}/eva/webservices/identifiers/v1"
EVA_CONTIG_ALIAS = f"https://{EVA_HOST}/eva/webservices/contig-alias/v1"


class UrllibEvaTransport:
    """EVA's view of the source transport family: one bounded GET per call under EVA's
    web services; an HTTP status is an answer, a lost connection a transient fault,
    and nothing is retried here."""

    def __init__(
        self, *, timeout_seconds: float = TIMEOUT_SECONDS, opener: Opener | None = None
    ) -> None:
        self._transport = SourceTransport(
            (EVA_HOST,), timeout_seconds=timeout_seconds, opener=opener
        )

    @property
    def contact_count(self) -> int:
        return self._transport.contact_count

    def get(self, url: str) -> InvestigationSourceResponse:
        # The bound of the operation surface whose contact runs now.
        answer = self._transport.exchange(url, max_bytes=response_byte_bound())
        # Reported to the Tool collecting answers, as NCBI's transport reports its own
        # (`source_answers.answer_received`).
        answer_received(url, answer.status, answer.body)
        return InvestigationSourceResponse(
            request_identity=url,
            status_code=answer.status,
            body=answer.body,
            observed_at=answer.observed_at.isoformat().replace("+00:00", "Z"),
            elapsed_time_ms=answer.elapsed_time_ms,
        )


def checked_eva_response(
    response: InvestigationSourceResponse,
) -> InvestigationSourceResponse:
    """EVA's answer about the request, a 404 or a 410 among them, is the lookup's to
    read with its body (#458); a status that is no answer is a fault."""
    if source_answered(response.status_code):
        return response
    raise unanswered(response.status_code, "EVA rejected its request")


def checked_download(
    response: InvestigationSourceResponse,
) -> InvestigationSourceResponse:
    """Downloaded bytes are only a success's: a 429 or 5xx is transient, any other
    status, a 404 included, rejected."""
    if 200 <= response.status_code < 300:
        return response
    raise ProviderError(
        "transient"
        if response.status_code == 429 or response.status_code >= 500
        else "rejected",
        f"Source rejected its download with HTTP {response.status_code}.",
    )


def checked_document(response):
    """A paper's document is a success's; a 404 says the host holds none. A 429 or 5xx
    is transient, any other status rejected."""
    if response.status_code == 404 or 200 <= response.status_code < 300:
        return response
    raise ProviderError(
        "transient"
        if response.status_code == 429 or response.status_code >= 500
        else "rejected",
        f"Source rejected its document request with HTTP {response.status_code}.",
    )


_ANSWERS = {
    "json": checked_source_response,
    "eva": checked_eva_response,
    "bytes": checked_download,
    "document": checked_document,
}


def fetch_source(adapter: object, operation: str, *key: object):
    """One registered operation's GET through ``adapter``, its envelope checked: the
    one source-fetch seam a Run's lookups and the developer entries share. The
    caller runs it on the operation's surface (`source_contacts.call_source`)."""
    definition = SOURCE_OPERATIONS[operation]
    return _ANSWERS[definition.answer](getattr(adapter, definition.method)(*key))


class EvaTransport(Protocol):
    contact_count: int

    def get(self, url: str) -> InvestigationSourceResponse: ...


class EvaInvestigationSourceAdapter:
    """Perform only EVA's GETs: an rsID's three lookups and the variants at a
    position."""

    def __init__(self, transport: EvaTransport | None = None) -> None:
        self._transport = transport or UrllibEvaTransport()

    @property
    def contact_count(self) -> int:
        return self._transport.contact_count

    def fetch_eva_clustered_variant(self, rsid: str) -> InvestigationSourceResponse:
        """``rsid``, or ``rsid?contigNamingConvention=<EVA's value>`` (#468, owner R3)."""
        number, query = _keyed(rsid, "contigNamingConvention")
        return self._transport.get(
            f"{EVA_IDENTIFIERS}/clustered-variants/{_number(number)}{query}"
        )

    def fetch_eva_submitted_variants(self, rsid: str) -> InvestigationSourceResponse:
        """``rsid``, or ``rsid?contigNamingConvention=<EVA's value>`` (#468, owner R3)."""
        number, query = _keyed(rsid, "contigNamingConvention")
        return self._transport.get(
            f"{EVA_IDENTIFIERS}/clustered-variants/{_number(number)}/submitted{query}"
        )

    def fetch_eva_contig_alias(self, insdc_contig: str) -> InvestigationSourceResponse:
        """An INSDC contig, or ``contig?page=<n>`` for a later page of EVA's answer."""
        contig, query = _keyed(insdc_contig, "page")
        if not contig:
            raise ValueError("an EVA contig alias takes one INSDC sequence accession")
        return self._transport.get(
            f"{EVA_CONTIG_ALIAS}/chromosomes/genbank/"
            f"{urllib.parse.quote(contig, safe='')}{query}"
        )

    def fetch_eva_clustered_variants_at(
        self, place: str
    ) -> InvestigationSourceResponse:
        """EVA's clustered variants at ``assembly:contig:start:type`` (#369), the
        position in EVA's own form, its contig percent-encoded, or with
        ``?contigNamingConvention=<EVA's value>``, the convention its contig is written
        in (#469, owner SP4 a)."""
        place, option = _keyed(place, "contigNamingConvention")
        match = _PLACE.fullmatch(place)
        if match is None:
            raise ValueError(
                "EVA's variants at a position take assembly:contig:start:type"
            )
        query = urllib.parse.urlencode(
            {
                "assemblyId": match["assembly"],
                "referenceName": urllib.parse.unquote(match["contig"]),
                "start": match["start"],
                "variantType": match["kind"],
            }
        ) + option.replace("?", "&", 1)
        return self._transport.get(f"{EVA_IDENTIFIERS}/clustered-variants?{query}")


_PLACE = re.compile(
    r"^(?P<assembly>GCA_[0-9]{9}\.[0-9]+):(?P<contig>[^:]+):(?P<start>[1-9][0-9]*)"
    r":(?P<kind>[A-Z_]+)$"
)


def _keyed(key: str, option: str) -> tuple[str, str]:
    """A key and the one EVA option it may carry, as EVA's own query (#468): the
    option's value is passed verbatim and percent-encoded; EVA answers what it accepts."""
    value, _, query = key.partition("?")
    if not query:
        return value, ""
    options = urllib.parse.parse_qs(query, keep_blank_values=True, strict_parsing=True)
    if list(options) != [option] or len(options[option]) != 1:
        raise ValueError(f"an EVA key carries only its {option} option")
    return value, "?" + urllib.parse.urlencode({option: options[option][0]})


def _number(rsid: str) -> str:
    """The rsID's number, the key EVA's path takes: ``rs`` and digits with no leading
    zero, as the NCBI adapter's RefSNP key is checked; the adapter's own key rule."""
    number = rsid.removeprefix("rs")
    if not rsid.startswith("rs") or not number.isdigit() or number.startswith("0"):
        raise ValueError("an EVA clustered-variant request takes one canonical rsID")
    return number


class EvaSourceAdapter(Protocol):
    def fetch_eva_clustered_variant(self, rsid: str) -> InvestigationSourceResponse: ...

    def fetch_eva_submitted_variants(
        self, rsid: str
    ) -> InvestigationSourceResponse: ...

    def fetch_eva_contig_alias(
        self, insdc_contig: str
    ) -> InvestigationSourceResponse: ...

    def fetch_eva_clustered_variants_at(
        self, place: str
    ) -> InvestigationSourceResponse: ...


class PublicInvestigationSources:
    """A Run's public variant sources: NCBI for dbSNP and assembly reports, EVA for
    non-human rsIDs. Which one a lookup reads is the Run's route, not this object's."""

    def __init__(
        self,
        ncbi: InvestigationSourceAdapter | None = None,
        eva: EvaSourceAdapter | None = None,
    ) -> None:
        self._ncbi = ncbi or NcbiInvestigationSourceAdapter()
        self._eva = eva or EvaInvestigationSourceAdapter()

    @property
    def contact_count(self) -> int:
        return self._ncbi.contact_count + self._eva.contact_count

    def fetch_refsnp(self, rsid: str) -> InvestigationSourceResponse:
        return self._ncbi.fetch_refsnp(rsid)

    def fetch_assembly_report(
        self, assembly_accession: str
    ) -> InvestigationSourceResponse:
        return self._ncbi.fetch_assembly_report(assembly_accession)

    def fetch_eva_clustered_variant(self, rsid: str) -> InvestigationSourceResponse:
        return self._eva.fetch_eva_clustered_variant(rsid)

    def fetch_eva_submitted_variants(self, rsid: str) -> InvestigationSourceResponse:
        return self._eva.fetch_eva_submitted_variants(rsid)

    def fetch_eva_contig_alias(self, insdc_contig: str) -> InvestigationSourceResponse:
        return self._eva.fetch_eva_contig_alias(insdc_contig)

    def fetch_eva_clustered_variants_at(
        self, place: str
    ) -> InvestigationSourceResponse:
        return self._eva.fetch_eva_clustered_variants_at(place)

    def fetch_sequence_names(
        self, assembly_accession: str
    ) -> InvestigationSourceResponse:
        return self._ncbi.fetch_sequence_names(assembly_accession)

    def fetch_reference_span(self, span: str) -> InvestigationSourceResponse:
        return self._ncbi.fetch_reference_span(span)

    def fetch_spdi_rsids(self, spdi: str) -> InvestigationSourceResponse:
        return self._ncbi.fetch_spdi_rsids(spdi)

    def fetch_taxon_assemblies(self, taxon_query: str) -> InvestigationSourceResponse:
        return self._ncbi.fetch_taxon_assemblies(taxon_query)

    def fetch_spdi_equivalents(self, spdi: str) -> InvestigationSourceResponse:
        return self._ncbi.fetch_spdi_equivalents(spdi)

    def fetch_genome_annotation(
        self, assembly_accession: str
    ) -> InvestigationSourceResponse:
        return self._ncbi.fetch_genome_annotation(assembly_accession)

    def fetch_hgvs_contextuals(self, name: str) -> InvestigationSourceResponse:
        return self._ncbi.fetch_hgvs_contextuals(name)


__all__ = [
    "EvaInvestigationSourceAdapter",
    "EvaSourceAdapter",
    "PublicInvestigationSources",
    "UrllibEvaTransport",
    "checked_download",
    "checked_eva_response",
    "fetch_source",
]
