"""Download one paper's open copy through the product's source operations; no Run.

  python -m bio_neb.devtools.paper_download REFERENCE --out DIR

REFERENCE is a DOI, PMCID, PMID or title. Only Europe PMC and the PMC open-data bucket
are contacted, through `paper_sources.download_paper`: the product's operations,
transport, retries and byte bounds. Outside a Run no contact journal is written (owner,
#417 Q4), so ``download.json`` counts the GETs sent. It writes each downloaded document
under ``files/`` and ``download.json``: the resolved ids, each document's route, role,
media type, byte count, sha256 (the Run's artifact digest) and md5 (the light check a
re-download is compared by, owner #419 Q3), each document it knew of and does not hold,
with the reason, and any stop. A stop exits 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from bio_neb.providers.paper_sources import (
    PaperDownload,
    PaperDownloadStopped,
    PaperSourceAdapter,
    download_paper,
)


def download_record(download: PaperDownload, sent: int) -> dict:
    """What a download returned, without its bytes."""
    return {
        "reference": download.reference,
        "resolved": asdict(download.resolved) if download.resolved else None,
        "candidates": [asdict(candidate) for candidate in download.candidates],
        "note": download.note,
        "documents": [
            {
                "name": document.name,
                "url": document.url,
                "route": document.route,
                "role": document.role,
                "media_type": document.media_type,
                "bytes": len(document.content),
                "sha256": document.sha256,
                "md5": hashlib.md5(document.content).hexdigest(),
            }
            for document in download.documents
        ],
        "unfetched": [asdict(item) for item in download.unfetched],
        "stop": asdict(download.stop) if download.stop else None,
        "sent": sent,
    }


def write_download(download: PaperDownload, sent: int, out: Path) -> dict:
    """Each document's bytes under ``files/`` and the record as ``download.json``."""
    files = out / "files"
    files.mkdir(parents=True, exist_ok=True)
    for document in download.documents:
        (files / document.name).write_bytes(document.content)
    record = download_record(download, sent)
    (out / "download.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bio_neb.devtools.paper_download")
    parser.add_argument("reference", help="a DOI, PMCID, PMID or title")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    adapter = PaperSourceAdapter()
    try:
        download = download_paper(args.reference, adapter)
    except PaperDownloadStopped as stopped:
        download = stopped.download
    record = write_download(download, adapter.contact_count, args.out)
    print(json.dumps({k: record[k] for k in ("resolved", "stop", "sent")}, indent=2))
    return 1 if download.stop else 0


if __name__ == "__main__":
    sys.exit(main())
