"""The tool's steps over the local cache: list candidates, fetch a bundle or an NCBI FASTA,
and fit a fetched candidate. The command line and the one-step choice (`choice.py`) both
call them; every GET goes through the caller's `Fetcher`, to the plan's hosts only.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

from bio_neb.prebuilt_index.catalogue import (
    INDEX_SUFFIXES,
    CatalogueRow,
    GalaxyCandidate,
    parse_loc,
    parse_tool_data,
)
from tools.reference_index.fit import FitReport, check_fit

from .cache import ReferenceIndexCache
from .catalogue import attach_fit, carry_fits, find
from .contracts import Candidate, CandidateList, Fetched
from .fetch import Fetcher, ncbi_md5
from .galaxy_rule import FASTA_CATALOGUE, INDEX_CATALOGUES, galaxy_candidates

USEGALAXY_CATALOGUE = (
    "usegalaxy-bwa_mem_indexes.json",
    "https://usegalaxy.org/api/tool_data/bwa_mem_indexes",
)
"""Tool-only until #450: usegalaxy.org's list, whose paths are not datacache paths, read
only for the card's older and same-species rows (`choice.py`), never for `find`."""
BWA_CATALOGUES = {**INDEX_CATALOGUES, USEGALAXY_CATALOGUE[0]: USEGALAXY_CATALOGUE[1]}
NCBI_ALL = "https://ftp.ncbi.nlm.nih.gov/genomes/all"


class Refused(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def index_rows(cache: ReferenceIndexCache) -> list[CatalogueRow]:
    """The saved datacache `.loc` index catalogues' rows, the product rule's input."""
    return [
        row
        for name in INDEX_CATALOGUES
        if (path := cache.root / "catalogues" / name).exists()
        for row in parse_loc(path.read_text(), catalogue=name)
    ]


def fasta_rows(cache: ReferenceIndexCache) -> list[CatalogueRow]:
    """The saved BRC `fasta_indexes.loc` rows: the FASTA each index was built from."""
    name = FASTA_CATALOGUE[0]
    path = cache.root / "catalogues" / name
    return list(parse_loc(path.read_text(), catalogue=name)) if path.exists() else []


def catalogue_rows(cache: ReferenceIndexCache) -> list[CatalogueRow]:
    """Tool-only until #450: every saved index row, usegalaxy.org's list after the
    `.loc` rows, for the card's older and same-species rows (`choice.py`)."""
    path = cache.root / "catalogues" / USEGALAXY_CATALOGUE[0]
    usegalaxy = (
        parse_tool_data(path.read_text(), catalogue=USEGALAXY_CATALOGUE[0])
        if path.exists()
        else ()
    )
    return [*index_rows(cache), *usegalaxy]


def find_candidates(cache: ReferenceIndexCache, accession: str) -> CandidateList:
    """The product's candidates from the saved catalogues, with the names of each
    fetched `.fai`, keeping the fits a saved list already carries."""
    bundles = cache.root / "bundles"
    names = {
        ":".join(fai.parent.relative_to(bundles).parts): [
            line.split("\t", 1)[0] for line in fai.read_text().splitlines() if line
        ]
        for fai in sorted(bundles.glob("*/*/*/*.fai"))
    }
    listed = find(accession, index_rows(cache), fasta_rows(cache), fai_names=names)
    saved = cache.root / "candidates" / f"{accession}.json"
    return (
        carry_fits(listed, cache.load_candidates(accession))
        if saved.exists()
        else listed
    )


def fetch_missing(fetcher: Fetcher, gets: list[tuple[str, Path]]) -> list[dict]:
    """GET each file not yet in the cache; a file already there costs no request."""
    return [
        {"url": url, "outcome": "already_cached", "destination": str(target)}
        if target.exists()
        else fetcher.get(url, target).model_dump(mode="json")
        for url, target in gets
    ]


def fetch_bundle(
    cache: ReferenceIndexCache,
    fetcher: Fetcher,
    accession: str,
    candidate_id: str,
    *,
    fai_only: bool,
) -> list[dict]:
    """A listed candidate's files where the product's candidate gives them: the FASTA's
    `.fai`, then (unless `fai_only`) the FASTA and the five BWA files, all saved beside
    one another under the index's file name."""
    candidate = listed_candidate(cache, accession, candidate_id)
    urls = galaxy_candidate(cache, candidate).files()
    fasta = candidate_fasta(cache, candidate)
    roles = (
        ("fasta_fai",) if fai_only else ("fasta_fai", "genomic_fasta", *INDEX_SUFFIXES)
    )
    fetched = fetch_missing(
        fetcher,
        [(urls[role], Path(f"{fasta}{_SUFFIXES[role]}")) for role in roles],
    )
    if not fai_only:
        sizes = fasta_sizes(Path(f"{fasta}.fai"))
        if fasta.stat().st_size not in sizes:
            raise Refused(
                "fasta_size_differs_from_fai",
                f"{fasta}: {fasta.stat().st_size} bytes, "
                f"its .fai gives {sizes[0]} ({sizes[1]} without a final line end)",
            )
    return fetched


_SUFFIXES = {"genomic_fasta": "", "fasta_fai": ".fai", **INDEX_SUFFIXES}


def galaxy_candidate(
    cache: ReferenceIndexCache, candidate: Candidate
) -> GalaxyCandidate:
    """The product's candidate a listed one stands for, from the saved catalogues."""
    for found in galaxy_candidates(
        candidate.assembly_accession, index_rows(cache), fasta_rows(cache)
    ):
        if found.candidate_id == candidate.candidate_id:
            return found
    raise Refused("not_a_catalogue_candidate", candidate.candidate_id)


def fai_rows(fai: Path) -> list[tuple[str, int, int, int, int]]:
    """A `.fai`'s rows (name, length, offset, bases per line, bytes per line), or a
    `fai_structure_invalid` refusal naming the file."""
    rows = []
    for line in fai.read_text(errors="replace").splitlines():
        fields = line.split("\t")
        numbers = fields[1:5]
        if len(fields) < 5 or not all(number.isdigit() for number in numbers):
            raise Refused("fai_structure_invalid", f"{fai}: {line[:80]!r}")
        length, offset, line_bases, line_bytes = map(int, numbers)
        if line_bases == 0 or line_bytes < line_bases:
            raise Refused("fai_structure_invalid", f"{fai}: {line[:80]!r}")
        rows.append((fields[0], length, offset, line_bases, line_bytes))
    if not rows:
        raise Refused("fai_structure_invalid", f"{fai}: no rows")
    return rows


def fasta_sizes(fai: Path) -> tuple[int, int]:
    """The FASTA's exact size from its `.fai` (the last contig's offset, its bases and
    its line ends), and the same without a line end after its last line."""
    _, length, offset, line_bases, line_bytes = fai_rows(fai)[-1]
    size = offset + length + -(-length // line_bases) * (line_bytes - line_bases)
    return size, size - (line_bytes - line_bases if length else 0)


def listed_candidate(
    cache: ReferenceIndexCache, accession: str, candidate_id: str
) -> Candidate:
    listed = cache.load_candidates(accession)
    matches = [c for c in listed.candidates if c.candidate_id == candidate_id]
    if not matches:
        raise Refused("not_a_listed_candidate", candidate_id)
    return matches[0]


def fetch_ncbi(
    cache: ReferenceIndexCache, fetcher: Fetcher, accession: str, assembly_name: str
) -> dict:
    prefix, number = accession[:3], accession[4:13]
    stem = f"{accession}_{assembly_name}"
    folder = f"{NCBI_ALL}/{prefix}/{number[:3]}/{number[3:6]}/{number[6:9]}/{stem}/"
    target = cache.root / "ncbi" / accession
    listing = fetcher.get(folder + "md5checksums.txt", target / "md5checksums.txt")
    packed = fetcher.get(
        folder + f"{stem}_genomic.fna.gz", target / f"{stem}_genomic.fna.gz"
    )
    published = ncbi_md5(
        (target / "md5checksums.txt").read_text(), f"{stem}_genomic.fna.gz"
    )
    if packed.md5 != published:
        raise Refused(
            "md5_mismatch",
            f"{stem}_genomic.fna.gz md5 {packed.md5}, md5checksums.txt {published}",
        )
    plain = target / f"{stem}_genomic.fna"
    with (
        gzip.open(target / f"{stem}_genomic.fna.gz", "rb") as source,
        plain.open("xb") as sink,
    ):
        shutil.copyfileobj(source, sink)
    return {
        "accession": accession,
        "md5_checked": published,
        "fetched": [
            listing.model_dump(mode="json"),
            packed.model_dump(mode="json"),
        ],
        "plain_fasta": str(plain),
    }


EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_sequence(
    cache: ReferenceIndexCache, fetcher: Fetcher, accession: str
) -> dict:
    """One sequence by its accession.version, as NCBI efetch's FASTA, for an assembly NCBI
    keeps no assembly record of (#332: UCSC sacCer1's chromosome XI)."""
    target = cache.root / "ncbi" / "sequences" / f"{accession}.fa"
    fetched = fetcher.get(
        f"{EFETCH}?db=nuccore&id={accession}&rettype=fasta&retmode=text", target
    )
    with target.open("rb") as fasta:
        header = fasta.readline().decode(errors="replace").rstrip()
    if header.split(" ", 1)[0] != f">{accession}":
        cache.forget(target)
        raise Refused(
            "not_the_sequence",
            f"efetch answered {header[:80]!r} for {accession} (sha256 {fetched.sha256})",
        )
    return {
        "accession": accession,
        "path": str(target),
        "byte_count": fetched.byte_count,
        "sha256": fetched.sha256,
        "fetched": fetched.model_dump(mode="json"),
    }


def fit_candidate(
    cache: ReferenceIndexCache, accession: str, candidate_id: str
) -> FitReport:
    """Check a fetched Galaxy candidate against the cached NCBI FASTA, save the report
    and attach it to the saved candidate list."""
    listed = cache.load_candidates(accession)
    candidate = listed_candidate(cache, accession, candidate_id)
    requested = sorted((cache.root / "ncbi" / accession).glob("*_genomic.fna"))
    if len(requested) != 1:
        raise Refused("ncbi_fasta_missing", str(cache.root / "ncbi" / accession))
    report = check_fit(
        requested=requested[0],
        requested_accession=accession,
        candidate=candidate_fasta(cache, candidate),
        candidate_id=candidate_id,
    )
    cache.save_fit(report)
    cache.save_candidates(attach_fit(listed, report))
    return report


def fit_sources(
    cache: ReferenceIndexCache, accession: str, candidate_id: str
) -> tuple[Fetched, ...]:
    """The receipts of the fetched files a candidate's fit reads: NCBI's FASTA and the
    candidate's bundle."""
    candidate = listed_candidate(cache, accession, candidate_id)
    return cache.receipts_in(
        cache.root / "ncbi" / accession, candidate_fasta(cache, candidate).parent
    )


def candidate_fasta(cache: ReferenceIndexCache, candidate: Candidate) -> Path:
    """Where a fetched candidate's FASTA sits in the cache: `bundles/galaxy/<catalogue
    source>/<accession>/`, under its index's file name."""
    return (
        cache.root.joinpath("bundles", *candidate.candidate_id.split(":"))
        / candidate.location.rsplit("/", 1)[1]
    )
