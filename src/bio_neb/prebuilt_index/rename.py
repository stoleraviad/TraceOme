"""A prebuilt index's contig names made NCBI's while it streams into the Run's bucket
(#464, owner I7: trust Galaxy fully; stream, don't download).

A `ContigRename` states one file's format and each (Galaxy, NCBI) name pair. A FASTA's
header lines have their first word rewritten; an `.ann`'s descriptor lines (`gi name
annotation`) their name; a `.fai`'s lines their name, with the offset shifted by the
change in header length up to and including that record. Every other byte passes
through unchanged, and a line this module cannot read passes unchanged too: the caller
reads the result through the one reader (`parabricks.bwa_index.read_fai`).

Each contig is named once in a FASTA and once in an `.ann`, so either file grows by the
same known length (`renamed_byte_count`) and streams with its placed size stated, as an
upload needs. A `.fai` offset can gain or lose a digit, so a `.fai`, which is small, is
rewritten whole. Standard library only: the providers stream through it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

RenamedFormat = Literal["fasta", "fai", "ann"]
_CHUNK = 1 << 20
_HEADER_NAME = re.compile(rb">(\S+)")


@dataclass(frozen=True)
class ContigRename:
    """One streamed file's renames: its format, each (Galaxy, NCBI) name pair, and the
    placed size, stated for a FASTA or `.ann` and absent for a `.fai`."""

    format: RenamedFormat
    names: tuple[tuple[str, str], ...]
    placed_byte_count: int | None = None

    def __post_init__(self) -> None:
        if (
            self.format not in ("fasta", "fai", "ann")
            or (self.placed_byte_count is None) != (self.format == "fai")
            or (self.placed_byte_count is not None and self.placed_byte_count <= 0)
            or len({old for old, _ in self.names}) != len(self.names)
        ):
            raise ValueError(
                "a rename states its format, distinct names and, except for a .fai,"
                " its positive placed size"
            )

    def open(self, source: Any) -> RenamedStream:
        """A reader of ``source`` with the names rewritten."""
        return RenamedStream(source, self)

    def whole(self, data: bytes) -> bytes:
        """``data`` with the names rewritten."""
        rewriter = _rewriter(self)
        return rewriter.feed(data) + rewriter.finish()


def renamed_byte_count(byte_count: int, names: Iterable[tuple[str, str]]) -> int:
    """The size of a FASTA or `.ann` of ``byte_count`` bytes once each contig of
    ``names`` is renamed: each is named once."""
    return byte_count + sum(len(new.encode()) - len(old.encode()) for old, new in names)


class RenamedStream:
    """Reads the renamed bytes of ``source``; counts and hashes the source's own."""

    def __init__(self, source: Any, rename: ContigRename) -> None:
        self._source = source
        self._rewriter = _rewriter(rename)
        self._pending = b""
        self._offset = 0
        self._ended = False
        self._digest = hashlib.sha256()
        self._byte_count = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            parts = [self._pending[self._offset :]]
            self._pending, self._offset = b"", 0
            while not self._ended:
                parts.append(self._next())
            return b"".join(parts)
        while self._offset == len(self._pending) and not self._ended:
            self._pending, self._offset = self._next(), 0
        start = self._offset
        self._offset = min(len(self._pending), start + size)
        return self._pending[start : self._offset]

    def _next(self) -> bytes:
        chunk = self._source.read(_CHUNK)
        if not chunk:
            self._ended = True
            return self._rewriter.finish()
        self._byte_count += len(chunk)
        self._digest.update(chunk)
        return self._rewriter.feed(chunk)

    def source_identity(self) -> tuple[int, str]:
        """The source bytes read so far: their count and SHA-256."""
        return self._byte_count, self._digest.hexdigest()


def _rewriter(rename: ContigRename):
    names = {old.encode(): new.encode() for old, new in rename.names}
    if rename.format == "fasta":
        return _FastaHeaders(names)
    return _Lines(_FaiRow(names) if rename.format == "fai" else _AnnRow(names))


class _FastaHeaders:
    """Header lines renamed; sequence lines pass through in bulk."""

    def __init__(self, names: dict[bytes, bytes]) -> None:
        self._names = names
        self._header: bytearray | None = None
        self._line_start = True

    def feed(self, data: bytes) -> bytes:
        out, start = [], 0
        while start < len(data):
            if self._header is not None:
                end = data.find(b"\n", start)
                if end < 0:
                    self._header += data[start:]
                    break
                self._header += data[start : end + 1]
                out.append(self._renamed(bytes(self._header)))
                self._header, self._line_start, start = None, True, end + 1
            elif self._line_start and data[start] == 0x3E:
                self._header = bytearray()
            else:
                header = data.find(b"\n>", start)
                if header < 0:
                    out.append(data[start:])
                    self._line_start = data.endswith(b"\n")
                    break
                out.append(data[start : header + 1])
                self._line_start, start = True, header + 1
        return b"".join(out)

    def finish(self) -> bytes:
        header, self._header = self._header, None
        return b"" if header is None else self._renamed(bytes(header))

    def _renamed(self, line: bytes) -> bytes:
        match = _HEADER_NAME.match(line)
        if match is None or match.group(1) not in self._names:
            return line
        return b">" + self._names[match.group(1)] + line[match.end() :]


class _Lines:
    """Whole lines, each rewritten by ``row``; a last line without a newline too."""

    def __init__(self, row) -> None:
        self._row = row
        self._rest = b""

    def feed(self, data: bytes) -> bytes:
        data = self._rest + data
        cut = data.rfind(b"\n") + 1
        self._rest = data[cut:]
        return b"".join(
            self._row(line) + b"\n" for line in data[:cut].split(b"\n")[:-1]
        )

    def finish(self) -> bytes:
        rest, self._rest = self._rest, b""
        return self._row(rest) if rest else b""


class _FaiRow:
    """`name length offset line-bases line-width`: the name renamed and the offset
    shifted by every header's change so far, this record's included."""

    def __init__(self, names: dict[bytes, bytes]) -> None:
        self._names = names
        self._shift = 0

    def __call__(self, line: bytes) -> bytes:
        fields = line.split(b"\t")
        if len(fields) < 5 or not fields[2].isdigit():
            return line
        new = self._names.get(fields[0], fields[0])
        self._shift += len(new) - len(fields[0])
        fields[0], fields[2] = new, str(int(fields[2]) + self._shift).encode()
        return b"\t".join(fields)


class _AnnRow:
    """`.ann`: a header line, then per contig a descriptor line (`gi name annotation`),
    whose name is renamed, and a position line."""

    def __init__(self, names: dict[bytes, bytes]) -> None:
        self._names = names
        self._number = 0

    def __call__(self, line: bytes) -> bytes:
        self._number += 1
        if self._number % 2:
            return line
        body = line.rstrip(b"\r")
        fields = body.split(b" ", 2)
        if len(fields) < 2:
            return line
        fields[1] = self._names.get(fields[1], fields[1])
        return b" ".join(fields) + line[len(body) :]


__all__ = [
    "ContigRename",
    "RenamedFormat",
    "RenamedStream",
    "renamed_byte_count",
]
