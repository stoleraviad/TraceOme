"""A source file read in Range parts through the one source transport.

A large public file (a prebuilt index's `.bwt` or `.sa`, #365) moves as object storage
moves its parts (`nebius_object_storage`: `MULTIPART_PART_BYTES`, `_part_attempts`):
each part is one ranged GET through `SourceTransport.stream`, and a transient fault (a
5xx, a lost connection, a body that stops before its part ends) asks for the same part
again by the bound failure policy (owner, #363; #365 Q10). Any other answer (a 3xx or
4xx, a 200 that ignored the Range, a Content-Range that is not the part asked for or
whose total changed) is the source's and is refused unretried. A download continues a
partial file from its size (`start`). Lane Rf's development tool (`tools/reference_index`)
reads its sources through it; the product streams a prebuilt index into the Run's bucket
instead (#464, owner I7).
"""

from __future__ import annotations

import http.client
import random
import re
import time
import urllib.error
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Any, Protocol

from bio_neb.failure import (
    ProviderError,
    bound_operation_policy,
    operation_retry_decision,
)
from bio_neb.providers.nebius_object_storage import MULTIPART_PART_BYTES
from bio_neb.providers.source_transport import SourceContactLost

_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
_READ_FAULTS = (
    TimeoutError,
    ConnectionError,
    http.client.HTTPException,
    urllib.error.URLError,
)


class PartTransport(Protocol):
    """`SourceTransport.stream`, or a view of it that observes each GET."""

    def stream(
        self, url: str, *, headers: Mapping[str, str] | None = None
    ) -> AbstractContextManager[Any]: ...


class SourcePartFailed(ProviderError):
    def __init__(self, failure_class, url: str, span: str, reason: str) -> None:
        super().__init__(
            failure_class,
            f"The source's {span} failed: {reason}.",
            evidence={"url": url, "range": span},
        )


class _Transient(Exception):
    """A part attempt that the failure policy may repeat."""


class SourceDownload:
    """One source file as a readable stream, fetched one part at a time."""

    def __init__(
        self,
        transport: PartTransport,
        url: str,
        *,
        start: int = 0,
        part_bytes: int = MULTIPART_PART_BYTES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if start < 0 or part_bytes < 1:
            raise ValueError("a download starts at a size and moves positive parts")
        self.url = url
        self.total: int | None = None
        """The file's size, as its first part's Content-Range states it."""
        self._transport = transport
        self._offset = start
        self._part_bytes = part_bytes
        self._sleep = sleep
        self._part = b""
        self._at = 0

    def read(self, size: int = -1) -> bytes:
        """At most `size` bytes (all that remain when negative); b"" once whole."""
        blocks = []
        wanted = size
        while wanted != 0:
            if self._at == len(self._part):
                if self.total is not None and self._offset >= self.total:
                    break
                self._part, self._at = self._next_part(), 0
            left = len(self._part) - self._at
            take = left if wanted < 0 else min(wanted, left)
            blocks.append(self._part[self._at : self._at + take])
            self._at += take
            if wanted > 0:
                wanted -= take
        return b"".join(blocks)

    def _next_part(self) -> bytes:
        start = self._offset
        end = start + self._part_bytes - 1
        if self.total is not None:
            end = min(end, self.total - 1)
        span = f"bytes={start}-{end}"
        policy = bound_operation_policy()
        ordinal = 0
        while True:
            ordinal += 1
            try:
                body = self._part_once(start, end, span)
            except _Transient as transient:
                _, delay = operation_retry_decision("transient", ordinal, policy)
                if delay is None:
                    raise SourcePartFailed(
                        "transient", self.url, span, str(transient)
                    ) from transient.__cause__
                self._sleep(random.uniform(0, delay))
                continue
            self._offset += len(body)
            return body

    def _part_once(self, start: int, end: int, span: str) -> bytes:
        try:
            with self._transport.stream(self.url, headers={"Range": span}) as response:
                status = int(response.status)
                if status >= 500:
                    raise _Transient(f"HTTP {status}")
                served = _CONTENT_RANGE.fullmatch(
                    response.headers.get("Content-Range") or ""
                )
                if status != 206 or served is None:
                    raise SourcePartFailed(
                        "rejected", self.url, span, f"HTTP {status} is not the part"
                    )
                first, last, total = map(int, served.groups())
                if (
                    first != start
                    or last != min(end, total - 1)
                    or (self.total is not None and total != self.total)
                ):
                    raise SourcePartFailed(
                        "rejected",
                        self.url,
                        span,
                        f"Content-Range {served[0]} is not the part",
                    )
                body = response.read(last - first + 1)
        except SourceContactLost as lost:
            raise _Transient(lost.detail) from lost
        except _READ_FAULTS as fault:
            raise _Transient(f"the body's read failed: {type(fault).__name__}") from (
                fault
            )
        if len(body) != last - first + 1:
            raise _Transient(f"short body: {len(body)} of {last - first + 1} bytes")
        self.total = total
        return body


__all__ = ["PartTransport", "SourceDownload", "SourcePartFailed"]
