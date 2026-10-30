"""Measure a gzip stream's decompressed bytes while the stream passes through."""

from __future__ import annotations

import hashlib
import zlib
from collections.abc import Iterable
from typing import Any

from bio_neb.failure import ProviderError
from bio_neb.fasta_selection import FastaRecordSelection

_GZIP = 16 + zlib.MAX_WBITS


class GzipContentMeasure:
    """A reader returning the source bytes unchanged; it also inflates them.

    Multi-member gzip (as concatenated publisher files are) decompresses as one
    content; zero padding after a member is skipped, as Python's gzip reader (the
    CPU worker's) does. Reads never fail on content, so the upload's own failure handling is
    unchanged; ``identity`` refuses a stream that is not complete gzip. The measured
    content is the FASTA without the records named in ``excluded`` (#464), selected as
    the CPU worker selects it.
    """

    def __init__(self, source: Any, excluded: Iterable[str] = ()) -> None:
        self._source = source
        self._selection = FastaRecordSelection(excluded)
        self._inflate = zlib.decompressobj(_GZIP)
        self._member_open = False
        self._digest = hashlib.sha256()
        self._byte_count = 0
        self._invalid = False
        self._members = 0
        self._finished = False

    def read(self, size: int = -1) -> bytes:
        chunk = self._source.read(size)
        data = b"" if self._invalid else chunk
        try:
            while data:
                if not self._member_open and self._members:
                    # Zero padding after a member is skipped, as the gzip reader does.
                    data = data.lstrip(b"\0")
                    if not data:
                        break
                self._member_open = True
                self._count(self._inflate.decompress(data))
                if not self._inflate.eof:
                    break
                self._member_open = False
                self._members += 1
                data = self._inflate.unused_data
                self._inflate = zlib.decompressobj(_GZIP)
        except zlib.error:
            self._invalid = True
        return chunk

    def _count(self, plain: bytes) -> None:
        self._account(self._selection.feed(plain))

    def _account(self, selected: bytes) -> None:
        self._byte_count += len(selected)
        self._digest.update(selected)

    def identity(self) -> tuple[int, str]:
        if not self._invalid and not self._member_open and not self._finished:
            # The selection's final flush is already selected: count it as it is.
            self._finished = True
            self._account(self._selection.finish())
        if self._invalid or self._member_open or self._byte_count == 0:
            raise ProviderError("semantic", "source_not_gzip")
        return self._byte_count, self._digest.hexdigest()
