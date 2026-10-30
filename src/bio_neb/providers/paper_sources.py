"""A paper's open copy, downloaded through the public source operations.

Europe PMC resolves a reference to its ids, and a paper with a PMCID is downloaded from
the PMC open-data bucket and Europe PMC's REST service (owner, #419 Q1): the bucket's
JATS and every file it names first, Europe PMC's full text when the bucket holds no
JATS, and Europe PMC's supplement zip when the bucket missed a named file. Each GET is
one registered operation on its own failure surface (`call_source`), through the source
transport family; inside a Run's source scope each is a row in the contact journal.

A download returns each document's bytes with their sha256, the digest a Run's
artifact store names them by (`Journey.record_with_artifact`); storing them is the
Run's (1b2). Reading their text is the developer converters'. A 404 says the host
holds no such document, and a document the download knew of and does not hold is
``unfetched`` with its reason. A search or listing answer that does not read as one is
`semantic`, inside its operation. Any other failure the policy gives up on, after its retries
(ADR 0031), ends the download: no later GET is sent to any host, and the download
raises it with what it had (`PaperDownloadStopped`).
"""

from __future__ import annotations

import json
import posixpath
import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

from bio_neb.agentic_journey.source_answers import answer_received
from bio_neb.agentic_journey.source_contacts import call_source, response_byte_bound
from bio_neb.content import sha256_bytes
from bio_neb.failure import ContactLimitOverrides, ProviderError
from bio_neb.providers.eva_investigation import fetch_source
from bio_neb.providers.source_operations import SourceSurface
from bio_neb.providers.source_transport import (
    TIMEOUT_SECONDS,
    Opener,
    SourceTransport,
)

EUROPE_PMC_HOST = "www.ebi.ac.uk"
EUROPE_PMC_REST = f"https://{EUROPE_PMC_HOST}/europepmc/webservices/rest"
PMC_OPEN_DATA_HOST = "pmc-oa-opendata.s3.amazonaws.com"
PMC_OPEN_DATA = f"https://{PMC_OPEN_DATA_HOST}"
# A search page's rows: Europe PMC's own pageSize, 25 by default and 100 at most, the
# bound one page puts on what the model reads.
SEARCH_PAGE_SIZE = 25
MAX_SEARCH_PAGE_SIZE = 100
# A paper's open copy: one call downloads one paper, each GET on its own operation's
# surface. It is no Agent Tool, so this module declares its surface. Its 16 contacts are
# lane P's fetch budget per paper (`PaperWorkspace.max_fetches`); raise it to 32 if a
# paper reaches it.
PAPER_DOWNLOAD_SURFACES = {
    "download_paper": SourceSurface(
        frozenset(
            {
                "europe_pmc_search",
                "europe_pmc_full_text",
                "europe_pmc_supplementary_files",
                "pmc_open_data_listing",
                "pmc_open_data_file",
            }
        ),
        ContactLimitOverrides(contacts=16),
    )
}

_PMCID = re.compile(r"PMC\d{4,9}")
_BUCKET_KEY = re.compile(r"PMC\d{4,9}\.\d+/[^/]+")
_IMAGE = re.compile(r"\.(?:jpe?g|gif|png|tiff?)$", re.IGNORECASE)

Route = Literal[
    "pmc_open_data", "europe_pmc_full_text", "europe_pmc_supplementary_files"
]
Role = Literal["article", "supplement", "supplement_archive"]


@dataclass(frozen=True)
class PaperSourceResponse:
    request_identity: str
    status_code: int
    body: bytes
    media_type: str


class PaperSourceAdapter:
    """Perform only a paper download's GETs: Europe PMC's search, full text and
    supplement zip, and the PMC open-data bucket's listing and files."""

    def __init__(
        self, *, timeout_seconds: float = TIMEOUT_SECONDS, opener: Opener | None = None
    ) -> None:
        self._transport = SourceTransport(
            (EUROPE_PMC_HOST, PMC_OPEN_DATA_HOST),
            timeout_seconds=timeout_seconds,
            opener=opener,
        )

    @property
    def contact_count(self) -> int:
        return self._transport.contact_count

    def fetch_europe_pmc_search(
        self,
        query: str,
        page_size: int = SEARCH_PAGE_SIZE,
        cursor_mark: str = "*",
        sort: str | None = None,
    ) -> PaperSourceResponse:
        """One page of Europe PMC's search, its documented options passed as given:
        ``query`` in its search syntax, ``pageSize``, ``cursorMark`` and ``sort``."""
        if not 1 <= page_size <= MAX_SEARCH_PAGE_SIZE:
            raise ValueError("a Europe PMC search page holds 1 to 100 rows")
        parameters = {
            "query": query,
            "format": "json",
            "resultType": "lite",
            "pageSize": str(page_size),
            "cursorMark": cursor_mark,
        }
        if sort is not None:
            parameters["sort"] = sort
        return self._get(
            f"{EUROPE_PMC_REST}/search?{urllib.parse.urlencode(parameters)}",
            "application/json",
        )

    def fetch_europe_pmc_full_text(self, pmcid: str) -> PaperSourceResponse:
        return self._get(f"{EUROPE_PMC_REST}/{_pmcid(pmcid)}/fullTextXML")

    def fetch_europe_pmc_supplementary_files(self, pmcid: str) -> PaperSourceResponse:
        return self._get(f"{EUROPE_PMC_REST}/{_pmcid(pmcid)}/supplementaryFiles")

    def fetch_pmc_open_data_listing(self, pmcid: str) -> PaperSourceResponse:
        return self._get(f"{PMC_OPEN_DATA}/?list-type=2&prefix={_pmcid(pmcid)}.")

    def fetch_pmc_open_data_file(self, key: str) -> PaperSourceResponse:
        if not _BUCKET_KEY.fullmatch(key):
            raise ValueError("a bucket file takes one <pmcid>.<version>/<name> key")
        return self._get(f"{PMC_OPEN_DATA}/{urllib.parse.quote(key)}")

    def _get(self, url: str, accept: str = "*/*") -> PaperSourceResponse:
        # The bound of the operation surface whose contact runs now.
        answer = self._transport.exchange(
            url, max_bytes=response_byte_bound(), accept=accept
        )
        # Each GET's answer reaches the source Tool collecting them.
        answer_received(url, answer.status, answer.body)
        return PaperSourceResponse(
            request_identity=url,
            status_code=answer.status,
            body=answer.body,
            media_type=answer.headers.get("Content-Type") or "application/octet-stream",
        )


def _pmcid(value: str) -> str:
    if not _PMCID.fullmatch(value):
        raise ValueError("a paper's open copy takes one PMCID")
    return value


@dataclass(frozen=True)
class PaperIds:
    """What Europe PMC, or the reference itself, says the paper is; `PaperIdentity`'s
    names. Absent values are unknown."""

    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    title: str | None = None
    authors: str | None = None
    journal: str | None = None
    year: str | None = None
    open_access: bool | None = None
    resolved_by: Literal["europe_pmc", "user_supplied"] = "europe_pmc"


@dataclass(frozen=True)
class EuropePmcRow:
    """One row of a Europe PMC search page, copied: its record id (``source:id``, as
    Europe PMC names the record) and the ids it lists."""

    candidate_id: str
    ids: PaperIds


@dataclass(frozen=True)
class EuropePmcPage:
    """One page of Europe PMC's search answer, copied: its total ``hitCount``, the
    ``nextCursorMark`` that reads the next page, and every row in its order."""

    hit_count: int
    next_cursor_mark: str | None
    rows: tuple[EuropePmcRow, ...]


@dataclass(frozen=True)
class DownloadedDocument:
    name: str
    url: str
    route: Route
    role: Role
    media_type: str
    sha256: str
    content: bytes = field(repr=False)


@dataclass(frozen=True)
class UnfetchedDocument:
    """A document the download knew of and does not hold, and why: its host answered
    404, the bucket listing lacks a file the article names, its GET failed, or a stop
    came before it was asked for. ``url`` is None for a file no route was asked for."""

    name: str
    url: str | None
    route: Route
    reason: str


@dataclass(frozen=True)
class DownloadStop:
    """The failure that ended a download, on the host and operation it came from."""

    host: str
    operation: str
    failure_class: str
    detail: str


@dataclass(frozen=True)
class PaperDownload:
    reference: str
    resolved: PaperIds | None = None
    # A reference that states no id: every row Europe PMC's search lists, copied in
    # its order, for the model or the researcher to choose from (ADR 0049).
    candidates: tuple[PaperIds, ...] = ()
    documents: tuple[DownloadedDocument, ...] = ()
    unfetched: tuple[UnfetchedDocument, ...] = ()
    stop: DownloadStop | None = None
    note: str = ""

    @property
    def article(self) -> DownloadedDocument | None:
        return next((d for d in self.documents if d.role == "article"), None)


class PaperDownloadStopped(ProviderError):
    """A failure the policy gave up on ended the download; ``download`` keeps what it
    had, its ``stop`` naming the failure."""

    def __init__(self, error: ProviderError, download: PaperDownload) -> None:
        super().__init__(error.failure_class, error.detail, dict(error.evidence))
        self.download = download


DOI_PATTERN = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+[^\s\"<>.,;)\]])", re.IGNORECASE)
PMCID_PATTERN = re.compile(r"\bPMC\d{4,9}\b", re.IGNORECASE)
PMID_PATTERN = re.compile(r"\b(?:PMID:?\s*|pubmed/)(\d{5,9})\b", re.IGNORECASE)


def stated_identifiers(text: str) -> dict[str, str]:
    """DOI, PMCID and PMID written in a reference string (first of each)."""

    found: dict[str, str] = {}
    doi = DOI_PATTERN.search(urllib.parse.unquote(text))
    if doi:
        found["doi"] = doi.group(1)
    pmcid = PMCID_PATTERN.search(text)
    if pmcid:
        found["pmcid"] = pmcid.group(0).upper()
    pmid = PMID_PATTERN.search(text)
    if pmid:
        found["pmid"] = pmid.group(1)
    return found


def _malformed(detail: str) -> ProviderError:
    return ProviderError("semantic", detail)


_SEARCH_FIELDS = {
    "doi": "doi",
    "pmid": "pmid",
    "pmcid": "pmcid",
    "title": "title",
    "authors": "authorString",
    "journal": "journalTitle",
    "year": "pubYear",
}


def read_europe_pmc_search(body: bytes) -> EuropePmcPage:
    """Europe PMC's lite search page, copied in its order, with its ``hitCount`` and
    ``nextCursorMark``. An answer that is not its result list with its hit count, or a
    record whose ids are not text (a PMCID not one), is malformed: `semantic`."""

    answer, records = _search_answer(body)
    hit_count = answer.get("hitCount")
    cursor = answer.get("nextCursorMark")
    if (
        not isinstance(hit_count, int)
        or isinstance(hit_count, bool)
        or not (cursor is None or isinstance(cursor, str))
    ):
        raise _malformed("Europe PMC's search answer has no hit count or cursor.")
    rows = tuple(_row(record) for record in records)
    if any(not row.candidate_id for row in rows):
        raise _malformed("Europe PMC's search answer holds a record with no id.")
    return EuropePmcPage(hit_count, cursor, rows)


def search_europe_pmc(
    adapter: PaperSourceAdapter,
    query: str,
    page_size: int = SEARCH_PAGE_SIZE,
    cursor_mark: str = "*",
    sort: str | None = None,
) -> tuple[int, EuropePmcPage | None]:
    """One page of Europe PMC's search on its operation's surface, its options passed
    as given: the status and the copied page, or no page when Europe PMC answered the
    query with another status (its body is the caller's collected answer, #458)."""

    def fetch() -> tuple[int, EuropePmcPage | None]:
        response = fetch_source(
            adapter, "europe_pmc_search", query, page_size, cursor_mark, sort
        )
        if not 200 <= response.status_code < 300:
            return response.status_code, None
        return response.status_code, read_europe_pmc_search(response.body)

    return call_source("europe_pmc_search", fetch)


def parse_europe_pmc_search(body: bytes) -> list[PaperIds]:
    """The ids of each row of Europe PMC's search answer, in its order."""
    return [_row(record).ids for record in _search_answer(body)[1]]


def _search_answer(body: bytes) -> tuple[dict, list]:
    try:
        answer = json.loads(body)
    except (ValueError, UnicodeError) as error:
        raise _malformed("Europe PMC's search answer is not JSON.") from error
    result_list = answer.get("resultList") if isinstance(answer, dict) else None
    records = result_list.get("result") if isinstance(result_list, dict) else None
    if not isinstance(records, list):
        raise _malformed("Europe PMC's search answer is not its result list.")
    return answer, records


def _row(record: object) -> EuropePmcRow:
    if not isinstance(record, dict):
        raise _malformed("Europe PMC's search answer holds a record that is no object.")
    values = {field: record.get(name) for field, name in _SEARCH_FIELDS.items()}
    open_access = record.get("isOpenAccess")
    source, record_id = record.get("source"), record.get("id")
    if any(
        v is not None and not isinstance(v, str)
        for v in (*values.values(), source, record_id)
    ) or (open_access not in (None, "Y", "N")):
        raise _malformed("Europe PMC's search answer holds an id that is not text.")
    if values["pmcid"] is not None and not _PMCID.fullmatch(values["pmcid"]):
        raise _malformed("Europe PMC's search answer holds a malformed PMCID.")
    return EuropePmcRow(
        # Europe PMC names each record by its source and id (MED, PMC, PPR...).
        candidate_id=f"{source}:{record_id}" if source and record_id else "",
        ids=PaperIds(
            **values,
            open_access=None if open_access is None else open_access == "Y",
        ),
    )


def _xml_local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_pmc_open_data_listing(body: bytes, pmcid: str) -> list[str]:
    """Keys in the newest ``<pmcid>.<version>/`` folder of a bucket listing, in listed
    order; an empty listing has none. An answer that is no complete bucket listing is
    malformed: `semantic`."""

    try:
        root = ET.fromstring(body)
    except ET.ParseError as error:
        raise _malformed("The bucket's answer is not XML.") from error
    if _xml_local(root.tag) != "ListBucketResult":
        raise _malformed("The bucket's answer is not a bucket listing.")
    truncated = next(
        (n.text for n in root if _xml_local(n.tag) == "IsTruncated"), "false"
    )
    if (truncated or "").strip().lower() == "true":
        raise _malformed("The bucket's listing is truncated.")
    folders: dict[int, list[str]] = {}
    for node in root.iter():
        if _xml_local(node.tag) != "Key" or not node.text:
            continue
        match = re.fullmatch(rf"{re.escape(pmcid)}\.(\d+)/(.+)", node.text)
        if match:
            folders.setdefault(int(match.group(1)), []).append(node.text)
    return folders[max(folders)] if folders else []


def jats_supplement_files(body: bytes) -> list[str] | None:
    """The file names a JATS article's supplementary material points to, once each, or
    None when the bytes are no XML."""

    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None
    hrefs = []
    for node in root.iter():
        if _xml_local(node.tag) in ("supplementary-material", "media"):
            href = next((v for k, v in node.attrib.items() if k.endswith("href")), None)
            if href:
                hrefs.append(href)
    return list(dict.fromkeys(hrefs))


class _Download:
    """One download's GETs: each on its operation's surface, none after a stop."""

    def __init__(self, adapter: PaperSourceAdapter, reference: str) -> None:
        self._adapter = adapter
        self.result = PaperDownload(reference=reference)
        # The named files still to be asked for, unfetched if a stop comes first.
        self.pending: list[UnfetchedDocument] = []

    def get(
        self,
        operation: str,
        *key: str,
        read: Callable[[bytes], object] | None = None,
        item: UnfetchedDocument | None = None,
    ) -> tuple[PaperSourceResponse, object] | None:
        """The operation's answer and, with ``read``, what it reads in it, inside the
        operation's surface; None for the host's 404. A failure ends the download:
        ``item`` and the pending files are unfetched, and the stop names it."""

        def fetch() -> tuple[PaperSourceResponse, object]:
            response = fetch_source(self._adapter, operation, *key)
            if response.status_code == 404 or read is None:
                return response, None
            return response, read(response.body)

        try:
            response, read_value = call_source(operation, fetch)
        except ProviderError as error:
            stopped = "not requested: the download stopped"
            self.update(
                stop=DownloadStop(
                    _OPERATION_HOSTS[operation],
                    operation,
                    error.failure_class,
                    error.detail,
                ),
                unfetched=(
                    *self.result.unfetched,
                    *(
                        (replace(item, reason=f"failed: {error.detail}"),)
                        if item
                        else ()
                    ),
                    *(replace(p, reason=stopped) for p in self.pending),
                ),
            )
            self.pending = []
            raise PaperDownloadStopped(error, self.result) from error
        if response.status_code == 404:
            if item is not None:
                self.missing(replace(item, reason="HTTP 404: the host holds none"))
            return None
        return response, read_value

    def update(self, **changes) -> None:
        self.result = replace(self.result, **changes)

    def keep(
        self, response: PaperSourceResponse, name: str, route: Route, role: Role
    ) -> None:
        document = DownloadedDocument(
            name=name,
            url=response.request_identity,
            route=route,
            role=role,
            media_type=response.media_type,
            sha256=sha256_bytes(response.body),
            content=response.body,
        )
        self.update(documents=(*self.result.documents, document))

    def missing(self, item: UnfetchedDocument) -> None:
        self.update(unfetched=(*self.result.unfetched, item))


_OPERATION_HOSTS = {
    "europe_pmc_search": EUROPE_PMC_HOST,
    "europe_pmc_full_text": EUROPE_PMC_HOST,
    "europe_pmc_supplementary_files": EUROPE_PMC_HOST,
    "pmc_open_data_listing": PMC_OPEN_DATA_HOST,
    "pmc_open_data_file": PMC_OPEN_DATA_HOST,
}
"""Each paper operation's one host."""


def _resolve(download: _Download, reference: str) -> None:
    """PMCID, then DOI, then PMID, through Europe PMC's search. A reference that states
    none of them is searched by its title, and every row Europe PMC lists comes back
    as a candidate, copied in its order: which one is the paper is not code's to say
    (ADR 0049)."""
    stated = stated_identifiers(reference)
    if "pmcid" in stated:
        query = f"PMCID:{stated['pmcid']}"
    elif "doi" in stated:
        query = f'DOI:"{stated["doi"]}"'
    elif "pmid" in stated:
        query = f"EXT_ID:{stated['pmid']} AND SRC:MED"
    else:
        query = f'TITLE:"{reference.strip()[:300]}"'
    answer = download.get("europe_pmc_search", query, read=parse_europe_pmc_search)
    identities = [] if answer is None else answer[1]
    if stated and identities:
        download.update(resolved=identities[0])
    elif "doi" in stated:
        download.update(
            resolved=PaperIds(doi=stated["doi"], resolved_by="user_supplied")
        )
    elif not stated and identities:
        download.update(
            candidates=tuple(identities),
            note="Europe PMC's title search lists these papers",
        )
    else:
        download.update(note="Europe PMC names no paper for the reference")


def _open_copy(download: _Download, pmcid: str) -> None:
    keys: list[str] = []
    listing = download.get(
        "pmc_open_data_listing",
        pmcid,
        read=lambda body: parse_pmc_open_data_listing(body, pmcid),
    )
    if listing is not None:
        keys = listing[1]
    named: list[str] | None = None
    fetched: set[str] = set()
    if keys:
        folder = keys[0].split("/", 1)[0]
        jats_key = f"{folder}/{folder}.xml"
        if jats_key in keys:
            jats = _bucket_file(download, jats_key, "article")
            if jats is not None:
                named = jats_supplement_files(jats.body)
        by_name = {posixpath.basename(key): key for key in keys}
        wanted = (
            [by_name[n] for n in map(posixpath.basename, named) if n in by_name]
            if named is not None
            else [
                key
                for name, key in by_name.items()
                if not name.startswith(f"{folder}.") and not _IMAGE.search(name)
            ]
        )
        # The names no listed key gives, before a later GET can stop the download.
        _account_named(download, named, {posixpath.basename(k) for k in wanted})
        for index, key in enumerate(wanted):
            download.pending = [_bucket_item(k) for k in wanted[index + 1 :]]
            if _bucket_file(download, key, "supplement") is not None:
                fetched.add(posixpath.basename(key))
        download.pending = []
    if download.result.article is None:
        _europe_pmc(download, "europe_pmc_full_text", pmcid, "article")
        article = download.result.article
        if named is None and article is not None:
            named = jats_supplement_files(article.content)
    _account_named(download, named, fetched)
    if named is None or any(posixpath.basename(n) not in fetched for n in named):
        _europe_pmc(
            download, "europe_pmc_supplementary_files", pmcid, "supplement_archive"
        )


def _account_named(
    download: _Download, named: list[str] | None, held: set[str]
) -> None:
    """Each file the article names that the bucket does not hold (``held``) and nothing
    accounts for yet is unfetched: after the bucket's listing and article, before any
    of its file GETs, and again after Europe PMC's full text."""
    accounted = {item.name for item in download.result.unfetched}
    for name in dict.fromkeys(map(posixpath.basename, named or ())):
        if name not in held and name not in accounted:
            download.missing(
                UnfetchedDocument(
                    name, None, "pmc_open_data", "not in the bucket listing"
                )
            )


def _bucket_item(key: str) -> UnfetchedDocument:
    return UnfetchedDocument(
        posixpath.basename(key),
        f"{PMC_OPEN_DATA}/{urllib.parse.quote(key)}",
        "pmc_open_data",
        "",
    )


def _bucket_file(
    download: _Download, key: str, role: Role
) -> PaperSourceResponse | None:
    answer = download.get("pmc_open_data_file", key, item=_bucket_item(key))
    if answer is not None:
        download.keep(answer[0], posixpath.basename(key), "pmc_open_data", role)
    return None if answer is None else answer[0]


_EUROPE_PMC_ROUTES: dict[str, tuple[Route, Callable[[str], str]]] = {
    "europe_pmc_full_text": (
        "europe_pmc_full_text",
        lambda pmcid: f"{EUROPE_PMC_REST}/{pmcid}/fullTextXML",
    ),
    "europe_pmc_supplementary_files": (
        "europe_pmc_supplementary_files",
        lambda pmcid: f"{EUROPE_PMC_REST}/{pmcid}/supplementaryFiles",
    ),
}


def _europe_pmc(download: _Download, operation: str, pmcid: str, role: Role) -> None:
    route, url = _EUROPE_PMC_ROUTES[operation]
    item = UnfetchedDocument(posixpath.basename(url(pmcid)), url(pmcid), route, "")
    answer = download.get(operation, pmcid, item=item)
    if answer is not None:
        download.keep(answer[0], item.name, route, role)


def download_paper(reference: str, adapter: PaperSourceAdapter) -> PaperDownload:
    """Resolve the reference and download its open copy, each GET on its operation's
    surface. A paper without a PMCID has no open copy here (owner, #419 Q1)."""

    download = _Download(adapter, reference)
    _resolve(download, reference)
    resolved = download.result.resolved
    if resolved is not None and resolved.pmcid:
        _open_copy(download, resolved.pmcid)
    elif resolved is not None:
        download.update(note="the paper has no PMCID, so no open copy is downloaded")
    return download.result


__all__ = [
    "EUROPE_PMC_HOST",
    "PAPER_DOWNLOAD_SURFACES",
    "PMC_OPEN_DATA_HOST",
    "MAX_SEARCH_PAGE_SIZE",
    "SEARCH_PAGE_SIZE",
    "DownloadStop",
    "DownloadedDocument",
    "EuropePmcPage",
    "EuropePmcRow",
    "PaperDownload",
    "PaperDownloadStopped",
    "PaperIds",
    "PaperSourceAdapter",
    "PaperSourceResponse",
    "UnfetchedDocument",
    "download_paper",
    "jats_supplement_files",
    "parse_europe_pmc_search",
    "parse_pmc_open_data_listing",
    "read_europe_pmc_search",
    "search_europe_pmc",
    "stated_identifiers",
]
