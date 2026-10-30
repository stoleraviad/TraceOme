"""Synthetic NCBI- and Galaxy-shaped FASTAs and BWA bundles for the prebuilt-index fit
(#319, moved into the product by #365). The bundle files are written as `bwa fa2pac -f` and
`bwa index` lay them out, so the product's reader reads them as real ones."""

from __future__ import annotations

import struct
from pathlib import Path


def write_fasta(path, contigs, *, width, lower=False):
    lines = []
    for name, description, sequence in contigs:
        lines.append(f">{name} {description}".rstrip())
        text = sequence.lower() if lower else sequence
        lines.extend(text[i : i + width] for i in range(0, len(text), width))
    path.write_text("\n".join(lines) + "\n")


# Synthetic sequences.
CHROMOSOME_I = "ACGTTGCAAC" * 13 + "GGA"
CHROMOSOME_II = "TTAGGCATCG" * 9 + "ACGTA"
MITOCHONDRION = "CCGATAGCTA" * 6 + "TT"

NCBI = (
    ("NC_000001.1", "Synthetic chromosome I, complete sequence", CHROMOSOME_I),
    ("NC_000002.1", "Synthetic chromosome II, complete sequence", CHROMOSOME_II),
    ("NC_000003.1", "Synthetic mitochondrion, complete genome", MITOCHONDRION),
)


def galaxy_contigs(*overrides):
    """Galaxy-style contigs: UCSC names, no description, the NCBI sequences."""
    names = {"NC_000001.1": "chrI", "NC_000002.1": "chrII", "NC_000003.1": "chrM"}
    contigs = [(names[name], "", sequence) for name, _, sequence in NCBI]
    for index, contig in overrides:
        contigs[index] = contig
    return tuple(contigs)


def fai_text(contigs, *, width):
    rows, offset = [], 0
    for name, description, sequence in contigs:
        offset += len(f">{name} {description}".rstrip()) + 1
        rows.append(f"{name}\t{len(sequence)}\t{offset}\t{width}\t{width + 1}\n")
        offset += len(sequence) + -(-len(sequence) // width)
    return "".join(rows)


def bwa_text_files(contigs):
    """`.ann`, `.amb` and `.pac` as `bwa fa2pac -f` writes them (bwa 0.7 bntseq.c):
    2-bit A0 C1 G2 T3, four bases per byte from the high bits, a zero byte when the
    length divides by four, then one byte holding length % 4; seed 11. A run of one
    ambiguous letter is one `.amb` hole row `offset length letter`, counted per contig
    in `.ann`. bwa packs a random base for each ambiguous one (lrand48, seed 11); this
    writer packs A, which is enough because the fit check only compares two outputs
    of the same writer."""
    total = sum(len(sequence) for _, _, sequence in contigs)
    ann, holes, offset = [f"{total} {len(contigs)} 11\n"], [], 0
    for name, description, sequence in contigs:
        runs = [
            (offset + start, len(run), run[0])
            for start, run in _ambiguous_runs(sequence.upper())
        ]
        holes.extend(runs)
        ann.append(
            f"0 {name} {description or '(null)'}\n{offset} {len(sequence)} {len(runs)}\n"
        )
        offset += len(sequence)
    packed = bytearray(-(-total // 4))
    for position, base in enumerate("".join(s for _, _, s in contigs).upper()):
        code = "ACGT".index(base) if base in "ACGT" else 0
        packed[position >> 2] |= code << ((~position & 3) << 1)
    if total % 4 == 0:
        packed.append(0)
    packed.append(total % 4)
    amb = [f"{total} {len(contigs)} {len(holes)}\n"]
    amb.extend(f"{start} {length} {letter}\n" for start, length, letter in holes)
    return {
        ".ann": "".join(ann).encode(),
        ".amb": "".join(amb).encode(),
        ".pac": bytes(packed),
    }


def _ambiguous_runs(sequence):
    start = None
    for position, base in enumerate(sequence + "A"):
        if start is not None and base != sequence[start]:
            yield start, sequence[start:position]
            start = None
        if start is None and base not in "ACGT" and position < len(sequence):
            start = position


def write_bundle(fasta, contigs, *, width=50, lower=False):
    """A Galaxy-shaped bundle: `<x>.fa`, its `.fai` and the five BWA files beside it."""
    write_fasta(fasta, contigs, width=width, lower=lower)
    Path(f"{fasta}.fai").write_text(fai_text(contigs, width=width))
    for suffix, payload in bwa_text_files(contigs).items():
        Path(f"{fasta}{suffix}").write_bytes(payload)
    indexed = 2 * sum(len(sequence) for _, _, sequence in contigs)
    header = (7, indexed // 4, indexed // 2, 3 * indexed // 4, indexed)
    Path(f"{fasta}.bwt").write_bytes(struct.pack("<5Q", *header) + b"\0" * 8)
    Path(f"{fasta}.sa").write_bytes(
        struct.pack("<7Q", *header, 32, indexed) + b"\0" * 16
    )
    return fasta


# A Galaxy datacache for the planning fakes' reference (`reference_fakes.SOURCE_PLAIN`,
# one 16-base contig named chr-offline, as `reference_fakes.OFFLINE_ASSEMBLY_REPORT`
# names it), served through the product adapter's opener.
GALAXY_ACCESSION = "GCF_000001405.40"
GALAXY_DATACACHE = "http://datacache.galaxyproject.org"


def galaxy_file(accession: str, kind: str, *, root: str = GALAXY_DATACACHE) -> str:
    """BRC's file of ``accession``: its BWA-MEM index (``bwa_mem_index``) or its FASTA
    (``sam_fasta_index``), on the datacache or (``root``) its CVMFS path."""
    return f"{root}/data/genomes/{accession}/{kind}/v1/{accession}/{accession}.fa"


GALAXY_INDEX = galaxy_file(
    GALAXY_ACCESSION, "bwa_mem_index", root=f"{GALAXY_DATACACHE}/brc"
)
GALAXY_FASTA = galaxy_file(
    GALAXY_ACCESSION, "sam_fasta_index", root=f"{GALAXY_DATACACHE}/brc"
)


class GalaxyOpener:
    """Galaxy's datacache, its four repositories' catalogues: BRC's list ``accession``
    (`GALAXY_ACCESSION`) as ``name`` (``listed``) and the others nothing; any file it
    holds in ``files`` is served whole or by Range, and each file's size on a HEAD."""

    def __init__(
        self, *, files=None, listed=True, accession=GALAXY_ACCESSION, name="Human"
    ):
        from bio_neb.prebuilt_index.catalogue import REPOSITORIES

        cvmfs = "/cvmfs/brc.galaxyproject.org"

        # A catalogue that lists nothing still has a comment line.
        self.bodies = {
            url: b"# This catalogue lists no index.\n"
            for repository in REPOSITORIES.values()
            for url in (repository.index_catalogue, repository.fasta_catalogue)
        }
        if listed:
            for catalogue, kind in (
                ("bwa_mem_index", "bwa_mem_index"),
                ("fasta_indexes", "sam_fasta_index"),
            ):
                path = galaxy_file(accession, kind, root=cvmfs)
                self.bodies[f"{GALAXY_DATACACHE}/brc/config/{catalogue}.loc"] = (
                    f"{accession}\t{accession}\t{name}\t{path}\n".encode()
                )
        self.bodies.update(files or {})
        self.sent: list[tuple[str, str]] = []

    def __call__(self, request, timeout):  # noqa: ANN001, ANN204
        import re
        from email.message import Message
        from unittest.mock import MagicMock

        url, method = request.full_url, request.get_method()
        self.sent.append((method, url))
        body = self.bodies.get(url, b"x" * 1000)
        headers = Message()
        status = 200
        span = request.get_header("Range")
        if method == "HEAD":
            headers["Content-Length"] = str(len(body))
            body = b""
        elif span:
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", span).groups())
            end = min(end, len(body) - 1)
            headers["Content-Range"] = f"bytes {start}-{end}/{len(body)}"
            body, status = body[start : end + 1], 206
        response = MagicMock(status=status)
        response.headers = headers
        response.read.side_effect = lambda size=-1: body if size < 0 else body[:size]
        response.__enter__.return_value = response
        return response


def offline_galaxy_files(
    root, *, extras=(), accession=GALAXY_ACCESSION, name="chr-offline"
) -> dict[str, bytes]:
    """Galaxy's prebuilt bundle of the planning fakes' reference under BRC's
    ``accession``, by URL: ``name`` (NCBI's chr-offline, or another name NCBI's report
    lists for it, such as its UCSC chrOffline), the same sequence as
    `reference_fakes.SOURCE_PLAIN`, then each of ``extras`` (name, sequence), contigs
    that are no requested sequence."""
    index = galaxy_file(accession, "bwa_mem_index", root=f"{GALAXY_DATACACHE}/brc")
    fasta_url = galaxy_file(
        accession, "sam_fasta_index", root=f"{GALAXY_DATACACHE}/brc"
    )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fasta = write_bundle(
        root / f"{accession}.fa",
        (
            (name, "", "ACGTACGTACGTACGT"),
            *((name, "", sequence) for name, sequence in extras),
        ),
    )
    files = {fasta_url: fasta.read_bytes()}
    files[f"{fasta_url}.fai"] = Path(f"{fasta}.fai").read_bytes()
    for suffix in (".amb", ".ann", ".bwt", ".pac", ".sa"):
        files[f"{index}{suffix}"] = Path(f"{fasta}{suffix}").read_bytes()
    return files
