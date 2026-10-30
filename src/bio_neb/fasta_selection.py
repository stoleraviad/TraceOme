"""The Run's reference sequences as one streaming selection of a plain FASTA (#464).

The records of the excluded names (the assembly's sequences outside the Run's reference,
`prebuilt_index.contigs.REFERENCE_ROLES`) are dropped whole; every other byte passes
unchanged and in order. Input storage's stream measure and the CPU worker feed it the
same plain bytes, so both derive the same FASTA. It is standard-library only: the worker
image copies this file beside `parabricks/worker_exchange.py`, and providers may import
it (ADR 0020 keeps them off `bio_neb.parabricks`).
"""

from __future__ import annotations

from collections.abc import Iterable


class FastaRecordSelection:
    """Feed plain FASTA bytes in any chunking, then `finish`; the output does not
    depend on the chunking. A record's name is its header line's first word, as bwa
    and samtools read it. Sequence bytes are passed as slices; only header lines are
    read, so the cost is a scan for `\\n>` per chunk."""

    def __init__(self, excluded: Iterable[str]) -> None:
        self._excluded = frozenset(name.encode() for name in excluded)
        self._drop = False
        # A header line not yet ended by its line break.
        self._header = b""
        self._at_line_start = True

    def feed(self, data: bytes) -> bytes:
        if self._header:
            data, self._header = self._header + data, b""
        out = []
        position, end = 0, len(data)
        while position < end:
            if self._at_line_start and data[position] == 0x3E:  # ">"
                line_end = data.find(b"\n", position)
                if line_end < 0:
                    self._header = data[position:]
                    break
                self._name(data, position, line_end)
                if not self._drop:
                    out.append(data[position : line_end + 1])
                position = line_end + 1
                continue
            header = data.find(b"\n>", position)
            stop = end if header < 0 else header + 1
            if not self._drop:
                out.append(
                    data if (position, stop) == (0, end) else data[position:stop]
                )
            self._at_line_start = data[stop - 1] == 0x0A  # "\n"
            position = stop
        if len(out) == 1:
            return out[0]
        return b"".join(out)

    def finish(self) -> bytes:
        """The last header line, when the stream ends inside it."""
        header, self._header = self._header, b""
        if not header:
            return b""
        self._name(header, 0, len(header))
        return b"" if self._drop else header

    def _name(self, data: bytes, start: int, stop: int) -> None:
        words = data[start + 1 : stop].split(None, 1)
        self._drop = bool(words) and words[0] in self._excluded
        self._at_line_start = True


__all__ = ["FastaRecordSelection"]
