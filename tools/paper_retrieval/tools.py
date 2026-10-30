"""A paper's documents as text, over the product's download.

The product resolves the paper and downloads its open copy
(`bio_neb.providers.paper_sources.download_paper`); this reads each downloaded document
with the product converters (`bio_neb.paper_intake.documents`) into the documents the reading Tools read. Nothing here
contacts a host itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from bio_neb.paper_intake.documents import SourceFile, document_texts
from bio_neb.paper_reading.contracts import PaperDocument
from bio_neb.providers.paper_sources import (
    DownloadedDocument,
    PaperDownload,
    PaperDownloadStopped,
    PaperIds,
    PaperSourceAdapter,
    download_paper,
)
from bio_neb.target_request.paper_targets import PaperIdentity

FetchVia = Literal["direct_http", "recorded"]


@dataclass(frozen=True)
class RetrievalFailed:
    """Why the download gave no readable article."""

    code: Literal["fetch_failed", "unsupported_content"]
    message: str


@dataclass
class ReadFile:
    """One downloaded file, or one member of a downloaded zip, and what was read."""

    name: str
    url: str
    route: str
    read: bool
    characters: int = 0
    note: str = ""
    text: str = field(default="", repr=False)


@dataclass
class RetrievalBundle:
    reference: str
    identity: PaperIdentity | None
    download: PaperDownload
    via: FetchVia
    main: PaperDocument | RetrievalFailed | None
    supplementary: list[PaperDocument]
    files: list[ReadFile]


def paper_identity(ids: PaperIds) -> PaperIdentity:
    return PaperIdentity(
        doi=ids.doi,
        pmid=ids.pmid,
        pmcid=ids.pmcid,
        title=ids.title,
        journal=ids.journal,
        year=ids.year,
        open_access=ids.open_access,
        resolved_by=ids.resolved_by,
        landing_url=(
            f"https://doi.org/{ids.doi}"
            if ids.resolved_by == "user_supplied" and ids.doi
            else None
        ),
    )


def _source(document: DownloadedDocument) -> SourceFile:
    return SourceFile(
        document.name,
        document.url,
        document.role,
        document.media_type,
        document.content,
    )


def paper_documents(
    download: PaperDownload,
) -> tuple[PaperDocument | RetrievalFailed | None, list[PaperDocument], list[ReadFile]]:
    """The article and its readable supplements, as the product reads them
    (`bio_neb.paper_intake.documents.document_texts`): ``main`` and one
    ``supp:<file name>`` document per readable file."""

    pmcid = download.resolved.pmcid if download.resolved else None
    texts = document_texts([_source(d) for d in download.documents])
    by_id = {document.document_id: document for document in texts.documents}
    article = download.article
    main: PaperDocument | RetrievalFailed | None = None
    if pmcid is not None and article is None:
        main = RetrievalFailed("fetch_failed", f"no JATS for {pmcid} by either route")
    elif pmcid is not None and "main" not in by_id:
        main = RetrievalFailed("unsupported_content", f"{article.url}: not JATS")
    elif pmcid is not None:
        main = PaperDocument(
            document_id="main",
            kind="main_text",
            url=article.url,
            text=by_id["main"].text,
        )
    files = []
    for reading in texts.readings:
        document = download.documents[reading.source]
        if document.role == "article":
            continue
        files.append(
            ReadFile(
                reading.name,
                document.url
                if reading.name == document.name
                else f"{document.url}#{reading.name}",
                document.route,
                reading.text is not None,
                len(reading.text or ""),
                reading.reason,
                reading.text or "",
            )
        )
    supplements = [
        PaperDocument(
            document_id=document.document_id,
            kind="supplementary",
            url=files_url(download, document),
            title=document.document_id.removeprefix("supp:"),
            text=document.text,
        )
        for document in texts.documents
        if document.document_id != "main"
    ]
    return main, supplements, files


def files_url(download: PaperDownload, document) -> str:
    source = download.documents[document.source]
    return (
        source.url if document.name == source.name else f"{source.url}#{document.name}"
    )


def retrieve(
    reference: str, adapter: PaperSourceAdapter, *, via: FetchVia = "direct_http"
) -> RetrievalBundle:
    """Resolve and download the paper through the product, then read its documents. A
    download a failure ended keeps what it had; its ``stop`` says why."""

    try:
        download = download_paper(reference, adapter)
    except PaperDownloadStopped as stopped:
        download = stopped.download
    main, supplements, files = paper_documents(download)
    return RetrievalBundle(
        reference=reference,
        identity=paper_identity(download.resolved) if download.resolved else None,
        download=download,
        via=via,
        main=main,
        supplementary=supplements,
        files=files,
    )
