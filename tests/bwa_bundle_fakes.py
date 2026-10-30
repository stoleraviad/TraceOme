"""The tiny structurally valid eight-role BWA bundle for offline reader tests."""

from __future__ import annotations

import hashlib
import struct

_FASTA = b">chr1 fixture\nACGTACGT\n"
_CONTIG_MD5 = hashlib.md5(b"ACGTACGT", usedforsecurity=False).hexdigest()


def local_bundle_documents(
    *, dict_uri: str = "file:///producer/reference.fa"
) -> dict[str, tuple[str, bytes]]:
    """The tiny structurally valid eight-role bundle, role -> (file name, bytes)."""

    return {
        "genomic_fasta": ("reference.fa", _FASTA),
        "fasta_fai": ("reference.fa.fai", b"chr1\t8\t14\t8\t9\n"),
        "reference_dict": (
            "reference.dict",
            f"@HD\tVN:1.0\n@SQ\tSN:chr1\tLN:8\tM5:{_CONTIG_MD5}\tUR:{dict_uri}\n".encode(),
        ),
        "bwa_amb": ("reference.fa.amb", b"8 1 0\n"),
        "bwa_ann": ("reference.fa.ann", b"8 1 11\n0 chr1 fixture\n0 8 0\n"),
        "bwa_bwt": (
            "reference.fa.bwt",
            struct.pack("<5Q", 3, 1, 4, 8, 16) + b"\0" * 4,
        ),
        "bwa_pac": ("reference.fa.pac", b"\x1b\x1b\0\0"),
        "bwa_sa": (
            "reference.fa.sa",
            struct.pack("<7Q", 3, 1, 4, 8, 16, 32, 16) + b"\0" * 8,
        ),
    }
