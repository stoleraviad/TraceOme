"""Read-only downloads into the local cache through the product's source transport.

Every GET, each Range part included, leaves through `SourceTransport.stream`: the product's
one no-redirect opener, user agent and host check, with the hosts the command names (a host
outside them is refused before any request). This module keeps only the whole-file request:
a redirect is followed as a separately checked hop, at most `MAX_HOPS`; a body shorter than
its declared length is a failure (http.client returns it without one), its bytes stay in a
partial file and the rest comes through the product's `SourceDownload` (#365): Range parts
of main's part size, each retried by the bound failure policy, a part that is not the one
asked for refused. Each whole file keeps a receipt beside it (`<file>.fetched.json`: where
it came from, its size and digests); no contact ledger is kept (owner, #417).
"""

from __future__ import annotations

import hashlib
import http.client
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urljoin

from bio_neb.providers.nebius_object_storage import MULTIPART_PART_BYTES
from bio_neb.providers.source_transport import SourceContactLost, SourceTransport
from tools.reference_index.source_download import SourceDownload, SourcePartFailed

from .cache import ReferenceIndexCache
from .contracts import Fetched

MAX_HOPS = 3
_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_TRANSPORT_FAULTS = (TimeoutError, ConnectionError, http.client.HTTPException)


class FetchFailed(Exception):
    """The request was sent and failed."""


class FetchRefused(Exception):
    """A download that would replace a file already in the cache; nothing is sent."""


class Fetcher:
    def __init__(
        self,
        cache: ReferenceIndexCache,
        transport: SourceTransport,
        *,
        part_bytes: int = MULTIPART_PART_BYTES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache = cache
        self.transport = transport
        self.part_bytes = part_bytes
        self.sleep = sleep

    def get(self, url: str, destination: Path) -> Fetched:
        """Write the whole entity to `destination` (which must not exist yet),
        following at most `MAX_HOPS` redirects as separately checked requests. A body
        that ends short of its declared length, or a partial file left by an earlier
        such request, continues in Range parts (`_parts`). The result is the whole
        file's receipt, also kept beside it."""
        if destination.exists():
            raise FetchRefused(f"{destination} already exists")
        partial = _partial(destination)
        if partial.exists():
            return self._parts(url, destination, _Hashes.of(partial), total=None)
        for _ in range(MAX_HOPS + 1):
            try:
                sent = self._send(url, destination)
            except _Short as short:
                return self._parts(url, destination, short.hashes, total=short.declared)
            if isinstance(sent, Fetched):
                return sent
            url = urljoin(url, sent)
        raise FetchFailed(f"more than {MAX_HOPS} redirects")

    def _send(self, url: str, destination: Path) -> Fetched | str:
        """One request: the whole file once its body is saved, or the target of the
        redirect it answered with. A body that ends short of its declared length is a
        failure; the bytes it brought stay in the partial file (`_Short`)."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = _partial(destination)
        hashes = _Hashes()
        declared: int | None = None
        # The partial stays only for a short body that `_parts` continues.
        keep = False
        try:
            with (
                self.transport.stream(url) as response,
                partial.open("wb") as target,
            ):
                status = response.status
                if status in _REDIRECTS and response.headers.get("Location"):
                    return response.headers["Location"]
                if status == 206:
                    raise _NotThePart("206 to a GET without Range: not a whole body")
                if not 200 <= status < 300:
                    raise FetchFailed(f"{url}: {status}")
                declared = _declared_length(response.headers)
                interrupted = None
                try:
                    for block in iter(lambda: response.read(1 << 20), b""):
                        target.write(block)
                        hashes.update(block)
                except _TRANSPORT_FAULTS as fault:
                    interrupted = fault
            count = hashes.byte_count
            if interrupted is not None or (declared is not None and count != declared):
                raise _Short(hashes, declared, interrupted)
            try:
                partial.rename(destination)
            except OSError as error:
                raise FetchFailed(f"{url}: received but not saved: {error}") from error
        except _Short as short:
            if hashes.byte_count and declared is not None:
                keep = True
                raise
            raise FetchFailed(f"{url}: {short.reason}") from short
        except (_NotThePart, SourceContactLost) as error:
            raise FetchFailed(f"{url}: {error}") from error
        finally:
            if not keep:
                partial.unlink(missing_ok=True)
        return self._kept(url, destination, hashes)

    def _parts(
        self, url: str, destination: Path, hashes: _Hashes, *, total: int | None
    ) -> Fetched:
        """The rest of the file after the partial's bytes, read through the product's
        `SourceDownload`: `part_bytes` Range parts in order, each retried by the bound
        failure policy, appended to the partial as they arrive. The parts' total must
        be the one the first answer declared; the file is whole when the download ends
        and is then renamed into place. The prefix's digests come from the request that
        wrote it, or one read of a partial an earlier run left."""
        partial = _partial(destination)
        download = SourceDownload(
            self.transport,
            url,
            start=hashes.byte_count,
            part_bytes=self.part_bytes,
            sleep=self.sleep,
        )
        try:
            with partial.open("ab") as target:
                for block in iter(lambda: download.read(1 << 20), b""):
                    if total is not None and download.total != total:
                        raise FetchFailed(
                            f"{url}: the parts' total {download.total} is not the "
                            f"declared {total} bytes"
                        )
                    target.write(block)
                    hashes.update(block)
            partial.rename(destination)
        except SourcePartFailed as failed:
            raise FetchFailed(f"{url}: {failed.detail}") from failed
        except OSError as error:
            raise FetchFailed(f"{url}: received but not saved: {error}") from error
        return self._kept(url, destination, hashes)

    def _kept(self, url: str, destination: Path, hashes: _Hashes) -> Fetched:
        """The whole file's receipt, written beside it."""
        fetched = Fetched(
            url=url,
            destination=str(destination),
            byte_count=hashes.byte_count,
            sha256=hashes.sha256.hexdigest(),
            md5=hashes.md5.hexdigest(),
        )
        self.cache.keep_receipt(fetched)
        return fetched


class _Hashes:
    """Running size, sha256 and md5 of a file as its bytes are written."""

    def __init__(self) -> None:
        self.byte_count = 0
        self.sha256 = hashlib.sha256()
        self.md5 = hashlib.md5(usedforsecurity=False)

    @classmethod
    def of(cls, path: Path) -> _Hashes:
        hashes = cls()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                hashes.update(block)
        return hashes

    def update(self, block: bytes) -> None:
        self.byte_count += len(block)
        self.sha256.update(block)
        self.md5.update(block)


class _NotThePart(Exception):
    """A GET without Range answered with a 206: a part, not the whole body."""


class _Short(Exception):
    """An answered request whose body stopped before its declared length."""

    def __init__(
        self, hashes: _Hashes, declared: int | None, fault: BaseException | None
    ) -> None:
        self.hashes = hashes
        self.declared = declared
        received = hashes.byte_count
        self.reason = (
            f"short body: {received} of {declared} bytes"
            if fault is None
            else f"interrupted after {received} bytes: {fault}"[:300]
        )
        super().__init__(self.reason)


def _partial(destination: Path) -> Path:
    return destination.with_name(destination.name + ".part")


def _declared_length(headers) -> int | None:
    value = headers.get("Content-Length")
    return int(value) if value is not None and value.isdigit() else None


def ncbi_md5(listing: str, file_name: str) -> str:
    """The md5 NCBI's `md5checksums.txt` publishes for one file of the folder."""
    for line in listing.splitlines():
        digest, _, name = line.strip().partition("  ")
        if name.removeprefix("./") == file_name:
            return digest.lower()
    raise ValueError(f"{file_name} is not listed in md5checksums.txt")
