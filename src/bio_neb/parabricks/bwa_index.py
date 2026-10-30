"""One reader of a BWA index and its FASTA (owner, #439 C3, #449).

`read_fasta` reads a FASTA once: each contig's name, length and M5 (the SAM ``@SQ`` MD5
of the upper-cased sequence) and the file's identity. `read_bwa_index` reads a FASTA with
its ``.fai``, an optional ``.dict`` and the five BWA files, each once with its stat
identity, and returns what they state and every finding against them. The reader decides
nothing: local-bundle admission refuses on the first finding, and the reference-index fit
reports every one. `read_ann`, `read_fai` and `read_dict` read one ``.ann``, ``.fai`` or
``.dict`` from its bytes, for a caller that compares one it made or reproduced or writes
from it; `sequence_dictionary` writes a ``.dict``.

A finding's `reason` is the miss admission refuses with. Its `kind` says what it is about:

- ``read``: the file is not a positive regular file, or it changed while it was read.
  The reading stops there; so does a FASTA finding, since nothing else can be compared.
- ``structure``: the file is not the format it claims.
- ``mismatch``: the file disagrees with the FASTA or with another index file.

Findings follow the files in role order, each file's structure before its comparison. A
comparison whose input could not be read is not made. An `OSError` passes to the caller.
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bio_neb.parabricks.worker_exchange import PREPARED_ROLES

INDEX_ROLES = PREPARED_ROLES[1:]
"""The roles beside the FASTA, in the order they are read; ``reference_dict`` is optional."""

_RETAINED = frozenset({"fasta_fai", "reference_dict", "bwa_amb", "bwa_ann"})
_IUPAC = frozenset(b"ACGTNMRWSYKVHDB")


class FormatInvalid(ValueError):
    """The bytes are not the format they claim; `reason` names the role's miss."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class IndexFile:
    """One file as it was read: its bytes' count and SHA-256, and its stat identity."""

    role: str
    path: Path
    byte_count: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True)
class Contig:
    name: str
    length: int
    m5: str


@dataclass(frozen=True)
class Ann:
    """The ``.ann`` header (``l_pac n_seqs seed``), each contig's name and its layout
    (``offset length n_ambs``)."""

    header: tuple[int, int, int]
    names: tuple[str, ...]
    layout: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class Finding:
    kind: Literal["read", "structure", "mismatch"]
    reason: str
    role: str
    path: Path
    cause: BaseException | None = None


@dataclass(frozen=True)
class FastaReading:
    file: IndexFile | None
    contigs: tuple[Contig, ...]
    findings: tuple[Finding, ...]


@dataclass(frozen=True)
class BwaIndexReading:
    """What a FASTA and its index files state, and every finding against them."""

    fasta: IndexFile | None
    files: tuple[IndexFile, ...]
    contigs: tuple[Contig, ...]
    ann: Ann | None
    pac_trailer: int | None
    findings: tuple[Finding, ...]

    @property
    def forward_base_count(self) -> int:
        return sum(contig.length for contig in self.contigs)

    def file(self, role: str) -> IndexFile:
        return next(item for item in self.files if item.role == role)


def read_fasta(path: Path) -> FastaReading:
    """The FASTA's contigs and identity, or the finding that it is not one."""
    before = path.stat()
    digest = hashlib.sha256()
    contigs: list[Contig] = []
    name: str | None = None
    length = 0
    md5 = hashlib.md5(usedforsecurity=False)
    try:
        with path.open("rb") as source:
            for raw in source:
                digest.update(raw)
                if raw.startswith(b">"):
                    if name is not None:
                        contigs.append(Contig(name, length, md5.hexdigest()))
                    name = fasta_record_name(raw)
                    length = 0
                    md5 = hashlib.md5(usedforsecurity=False)
                    continue
                sequence = raw.strip().upper()
                if name is None or not _IUPAC.issuperset(sequence):
                    raise FormatInvalid("fasta_structure_invalid")
                length += len(sequence)
                md5.update(sequence)
        if name is None:
            raise FormatInvalid("fasta_structure_invalid")
    except FormatInvalid as invalid:
        finding = _finding("structure", invalid, "genomic_fasta", path)
        return FastaReading(None, (), (finding,))
    contigs.append(Contig(name, length, md5.hexdigest()))
    if not _unchanged(before, path.stat()):
        changed = Finding(
            "read", "local_role_changed_during_admission", "genomic_fasta", path
        )
        return FastaReading(None, (), (changed,))
    file = IndexFile(
        "genomic_fasta",
        path,
        before.st_size,
        digest.hexdigest(),
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
    )
    return FastaReading(file, tuple(contigs), ())


def read_bwa_index(fasta: Path, files: Mapping[str, Path]) -> BwaIndexReading:
    """The FASTA and its index files (``files``: role → path; every role of
    `INDEX_ROLES`, ``reference_dict`` optional), read once each, and every finding."""
    roles = [role for role in INDEX_ROLES if role in files]
    if set(files) - set(roles) or set(INDEX_ROLES) - set(roles) - {"reference_dict"}:
        raise ValueError(f"a BWA index has the roles {INDEX_ROLES}: {sorted(files)}")
    read = read_fasta(fasta)
    if read.findings:
        return BwaIndexReading(None, (), (), None, None, read.findings)
    identities: list[IndexFile] = []
    heads: dict[str, bytes] = {}
    tails: dict[str, bytes] = {}
    kept: dict[str, bytes] = {}
    for role in roles:
        result = _read_once(role, files[role], retain=role in _RETAINED)
        if isinstance(result, Finding):
            return BwaIndexReading(
                read.file, tuple(identities), read.contigs, None, None, (result,)
            )
        identity, heads[role], tails[role], payload = result
        identities.append(identity)
        if payload is not None:
            kept[role] = payload
    findings: list[Finding] = []
    contigs = read.contigs
    total = sum(contig.length for contig in contigs)

    def parsed(role, parse):
        try:
            return parse(kept[role])
        except FormatInvalid as invalid:
            findings.append(_finding("structure", invalid, role, files[role]))
            return None

    def compare(role, agrees, reason):
        if not agrees:
            findings.append(Finding("mismatch", reason, role, files[role]))

    fai = parsed("fasta_fai", read_fai)
    if fai is not None:
        compare(
            "fasta_fai",
            fai == [(contig.name, contig.length) for contig in contigs],
            "fasta_fai_mismatch",
        )
    if "reference_dict" in kept:
        sequences = parsed("reference_dict", read_dict)
        if sequences is not None:
            compare("reference_dict", sequences == contigs, "fasta_dict_m5_mismatch")
    ann = parsed("bwa_ann", read_ann)
    if ann is not None and fai is not None:
        compare(
            "bwa_ann",
            ann.header[:2] == (total, len(fai))
            and list(
                zip(ann.names, (length for _, length, _ in ann.layout), strict=True)
            )
            == fai,
            "ann_fasta_mismatch",
        )
    amb = parsed("bwa_amb", _amb)
    if amb is not None and ann is not None:
        header, holes = amb
        compare(
            "bwa_amb",
            header[:2] == ann.header[:2]
            and header[2] == holes == sum(n for _, _, n in ann.layout),
            "amb_ann_mismatch",
        )
    sizes = {item.role: item.byte_count for item in identities}
    compare(
        "bwa_pac",
        sizes["bwa_pac"] == total // 4 + 2 and tails["bwa_pac"][-1] == total % 4,
        "pac_length_or_trailer_mismatch",
    )
    short = [
        role
        for role, words in (("bwa_bwt", 5), ("bwa_sa", 7))
        if len(heads[role]) < 8 * words
    ]
    for role in short:
        findings.append(
            Finding("structure", "bwt_sa_header_truncated", role, files[role])
        )
    if not short:
        bwt = struct.unpack("<5Q", heads["bwa_bwt"][:40])
        sa = struct.unpack("<7Q", heads["bwa_sa"][:56])
        indexed = 2 * total
        compare(
            "bwa_bwt",
            bwt[0] <= indexed and 0 <= bwt[1] <= bwt[2] <= bwt[3] <= bwt[4] == indexed,
            "bwt_header_invalid",
        )
        compare(
            "bwa_sa",
            sa[:5] == bwt and sa[5] == 32 and sa[6] == indexed,
            "sa_header_invalid",
        )
        compare(
            "bwa_sa",
            not ((sizes["bwa_bwt"] - 40) % 4 or (sizes["bwa_sa"] - 56) % 8),
            "bwt_sa_alignment_invalid",
        )
    return BwaIndexReading(
        read.file,
        tuple(identities),
        contigs,
        ann,
        tails["bwa_pac"][-1],
        tuple(findings),
    )


def read_ann(data: bytes) -> Ann:
    """An ``.ann``: its header, then per contig a descriptor line (``gi name
    annotation``) and a layout line (``offset length n_ambs``)."""
    lines = _lines(data, "ann_structure_invalid")
    try:
        header = tuple(map(int, lines[0].split()))
    except (IndexError, ValueError) as error:
        raise FormatInvalid("ann_structure_invalid") from error
    if len(header) != 3 or len(lines) != 1 + 2 * header[1]:
        raise FormatInvalid("ann_structure_invalid")
    names, layout = [], []
    for index in range(header[1]):
        descriptor = lines[1 + 2 * index].split(" ", 2)
        try:
            location = tuple(map(int, lines[2 + 2 * index].split()))
        except ValueError as error:
            raise FormatInvalid("ann_structure_invalid") from error
        if len(descriptor) < 2 or len(location) != 3:
            raise FormatInvalid("ann_structure_invalid")
        names.append(descriptor[1])
        layout.append(location)
    return Ann(header, tuple(names), tuple(layout))


def fasta_record_name(header: bytes) -> str:
    """A FASTA record's name: its header line's first word."""
    words = header[1:].split(None, 1)
    if not header.startswith(b">") or not words:
        raise FormatInvalid("fasta_structure_invalid")
    try:
        return words[0].decode("ascii")
    except UnicodeDecodeError as error:
        raise FormatInvalid("fasta_structure_invalid") from error


def read_fai(data: bytes) -> list[tuple[str, int]]:
    """A ``.fai``: each row's contig name and length, in its order."""
    rows = []
    for line in _lines(data, "fai_structure_invalid"):
        fields = line.split("\t")
        if len(fields) != 5:
            raise FormatInvalid("fai_structure_invalid")
        rows.append((fields[0], _count(fields[1], "fai_structure_invalid")))
    return rows


def read_dict(data: bytes) -> tuple[Contig, ...]:
    """A ``.dict``: each ``@SQ`` line's name, length and M5 (lower-cased)."""
    sequences = []
    for line in _lines(data, "dict_structure_invalid"):
        if not line.startswith("@SQ\t"):
            continue
        fields = dict(
            item.split(":", 1) for item in line.split("\t")[1:] if ":" in item
        )
        if not {"SN", "LN", "M5"} <= fields.keys():
            raise FormatInvalid("dict_structure_invalid")
        length = _count(fields["LN"], "dict_structure_invalid")
        sequences.append(Contig(fields["SN"], length, fields["M5"].lower()))
    return tuple(sequences)


def sequence_dictionary(contigs: Iterable[tuple[str, int]]) -> bytes:
    """A ``.dict`` of ``contigs`` (name, length), headed as ``samtools dict`` heads
    one: ``@HD``, then one ``@SQ`` with the name and length per contig (owner I7, #464:
    a streamed prebuilt index's `.dict`, from its renamed `.fai`)."""
    lines = ["@HD\tVN:1.0\tSO:unsorted\n"]
    lines.extend(f"@SQ\tSN:{name}\tLN:{length}\n" for name, length in contigs)
    return "".join(lines).encode()


def _amb(data: bytes) -> tuple[tuple[int, int, int], int]:
    """The ``.amb`` header (``l_pac n_seqs n_holes``) and its count of hole rows, each
    ``offset length letter``."""
    lines = _lines(data, "amb_structure_invalid")
    try:
        header = tuple(map(int, lines[0].split()))
        holes = [line.split() for line in lines[1:]]
        for offset, length, letter in holes:
            int(offset), int(length)
            if len(letter) != 1:
                raise ValueError(letter)
    except (IndexError, ValueError) as error:
        raise FormatInvalid("amb_structure_invalid") from error
    if len(header) != 3:
        raise FormatInvalid("amb_structure_invalid")
    return header, len(holes)


def _count(text: str, reason: str) -> int:
    """A length written in decimal digits, or the role's structure miss; a number that
    does not parse keeps its ValueError as the cause (owner, #449 Q3 rule 3)."""
    try:
        value = int(text)
    except ValueError as error:
        raise FormatInvalid(reason) from error
    if not text.isdigit():
        raise FormatInvalid(reason)
    return value


def _lines(data: bytes, reason: str) -> list[str]:
    try:
        return data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise FormatInvalid(reason) from error


def _read_once(
    role: str, path: Path, *, retain: bool
) -> tuple[IndexFile, bytes, bytes, bytes | None] | Finding:
    """The file's identity, its first and last 64 bytes and, when retained, its bytes."""
    before = path.stat()
    if not path.is_file() or before.st_size <= 0:
        return Finding("read", "local_role_not_positive_regular_file", role, path)
    digest = hashlib.sha256()
    head = bytearray()
    tail = b""
    retained = bytearray()
    count = 0
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            count += len(chunk)
            digest.update(chunk)
            if len(head) < 64:
                head.extend(chunk[: 64 - len(head)])
            tail = (tail + chunk)[-64:]
            if retain:
                retained.extend(chunk)
    if not _unchanged(before, path.stat()) or count != before.st_size:
        return Finding("read", "local_role_changed_during_admission", role, path)
    identity = IndexFile(
        role,
        path,
        count,
        digest.hexdigest(),
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
    )
    return identity, bytes(head), tail, bytes(retained) if retain else None


def _unchanged(before: os.stat_result, after: os.stat_result) -> bool:
    return (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )


def _finding(kind, invalid: FormatInvalid, role: str, path: Path) -> Finding:
    return Finding(kind, invalid.reason, role, path, invalid.__cause__)


__all__ = [
    "INDEX_ROLES",
    "Ann",
    "BwaIndexReading",
    "Contig",
    "FastaReading",
    "Finding",
    "FormatInvalid",
    "IndexFile",
    "read_ann",
    "read_bwa_index",
    "read_dict",
    "read_fasta",
    "fasta_record_name",
]
