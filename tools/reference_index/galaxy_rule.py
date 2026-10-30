"""Tool-only: the lane Rf choice card's Galaxy candidate rule, as #365 left it.

The product's catalogue (`bio_neb.prebuilt_index.catalogue`) no longer names an
assembly series or a candidate by accession (#464: the planning model searches Galaxy's
four repositories and chooses). This developer tool keeps its BRC/VGP view, its BRC
FASTA catalogue and its series listing until #459 internalizes or retires the card.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from bio_neb.prebuilt_index.catalogue import (
    DATACACHE,
    CatalogueRow,
    GalaxyCandidate,
)

INDEX_CATALOGUES = {
    "brc-bwa_mem_index.loc": f"{DATACACHE}/brc/config/bwa_mem_index.loc",
    "vgp-bwa_mem_index.loc": f"{DATACACHE}/vgp/config/bwa_mem_index.loc",
}
FASTA_CATALOGUE = ("brc-fasta_indexes.loc", f"{DATACACHE}/brc/config/fasta_indexes.loc")
_CVMFS = re.compile(r"^/cvmfs/(brc|vgp)\.galaxyproject\.org/(.+)$")
_SERIES = re.compile(r"^GC[AF]_(\d{9})\.\d+$")


def datacache_url(path: str) -> str | None:
    """The HTTP mirror of a Galaxy BRC/VGP CVMFS path, or None for another path."""
    match = _CVMFS.match(path)
    return None if match is None else f"{DATACACHE}/{match[1]}/{match[2]}"


def candidate_id(catalogue: str, accession: str) -> str:
    return f"galaxy:{catalogue.split('-', 1)[0]}:{accession}"


def galaxy_candidates(
    accession: str,
    index_rows: Sequence[CatalogueRow],
    fasta_rows: Sequence[CatalogueRow],
) -> tuple[GalaxyCandidate, ...]:
    """Each BRC/VGP publication of exactly `accession`, with one BRC FASTA row."""
    candidates: dict[str, GalaxyCandidate] = {}
    for row in index_rows:
        identity = candidate_id(row.catalogue, row.value)
        if row.value != accession or identity in candidates:
            continue
        index_url = datacache_url(row.path)
        fastas = [
            url
            for fasta in fasta_rows
            if fasta.value == row.value and (url := datacache_url(fasta.path))
        ]
        if index_url is None or len(fastas) != 1:
            continue
        candidates[identity] = GalaxyCandidate(
            candidate_id=identity,
            catalogue=row.catalogue,
            value=accession,
            dbkey=row.dbkey,
            display_name=row.name,
            index_url=index_url,
            fasta_url=fastas[0],
        )
    return tuple(candidates.values())


def assembly_series(accession: str) -> str | None:
    """The nine-digit GCA/GCF number shared by every prefix and version, or None."""
    match = _SERIES.match(accession)
    return match.group(1) if match else None


def series_candidates(
    accession: str,
    index_rows: Sequence[CatalogueRow],
    fasta_rows: Sequence[CatalogueRow],
) -> tuple[GalaxyCandidate, ...]:
    """The publications of every version and prefix of `accession`'s series, the
    exact accession's first."""
    series = assembly_series(accession)
    if series is None:
        raise ValueError(f"not an assembly accession: {accession}")
    accessions = sorted(
        {row.value for row in index_rows if assembly_series(row.value) == series},
        key=lambda value: (value != accession, value),
    )
    return tuple(
        candidate
        for value in accessions
        for candidate in galaxy_candidates(value, index_rows, fasta_rows)
    )
