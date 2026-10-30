"""Galaxy's prebuilt BWA index catalogues and the candidates their rows list.

Galaxy's reference data catalogues (the `.loc` files of its four datacache repositories,
`REPOSITORIES`, and usegalaxy.org's tool-data table) name each prebuilt index by a value,
a dbkey, a display name and a path, copied here as published (`CatalogueRow`). A row
whose index and FASTA both have a datacache path is a candidate, one per publication
(`galaxy:<repository>:<value>`). Which row is the Run's index is the planning model's
choice, which execution trusts (owner I7, #464). NCBI stays the identity source (#464,
ADR 0049).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from bio_neb.contracts import FrozenContract

DATACACHE = "http://datacache.galaxyproject.org"


class Repository(FrozenContract):
    """One of Galaxy's reference-data repositories as its datacache mirror serves it:
    the CVMFS prefix its catalogue paths carry, that prefix's HTTP mirror, and the
    folder holding its `.loc` catalogues."""

    cvmfs: str
    mirror: str
    catalogues: str

    @property
    def index_catalogue(self) -> str:
        return self.catalogues + "bwa_mem_index.loc"

    @property
    def fasta_catalogue(self) -> str:
        return self.catalogues + "fasta_indexes.loc"


REPOSITORIES = {
    "indexes": Repository(
        cvmfs="/cvmfs/data.galaxyproject.org/byhand/",
        mirror=f"{DATACACHE}/indexes/",
        catalogues=f"{DATACACHE}/indexes/location/",
    ),
    "managed": Repository(
        cvmfs="/cvmfs/data.galaxyproject.org/managed/",
        mirror=f"{DATACACHE}/managed/",
        catalogues=f"{DATACACHE}/managed/location/",
    ),
    "brc": Repository(
        cvmfs="/cvmfs/brc.galaxyproject.org/",
        mirror=f"{DATACACHE}/brc/",
        catalogues=f"{DATACACHE}/brc/config/",
    ),
    "vgp": Repository(
        cvmfs="/cvmfs/vgp.galaxyproject.org/",
        mirror=f"{DATACACHE}/vgp/",
        catalogues=f"{DATACACHE}/vgp/config/",
    ),
}
"""Galaxy's published reference-data repositories on its datacache, by Galaxy's name:
the hand-built genomes (`indexes`, CVMFS `byhand`), the data managers' (`managed`),
and the BRC and VGP genome repositories."""
INDEX_SUFFIXES = {
    "bwa_amb": ".amb",
    "bwa_ann": ".ann",
    "bwa_bwt": ".bwt",
    "bwa_pac": ".pac",
    "bwa_sa": ".sa",
}


class CatalogueUnreadable(Exception):
    """A catalogue response is not the shape it is read as."""


class CatalogueRow(FrozenContract):
    catalogue: str
    value: str
    dbkey: str
    name: str
    path: str


class GalaxyCandidate(FrozenContract):
    """A prebuilt index one Galaxy catalogue row lists, with where its files are."""

    candidate_id: str
    catalogue: str
    value: str
    dbkey: str
    display_name: str
    index_url: str
    fasta_url: str

    def files(self) -> dict[str, str]:
        """Each file's URL by prepared role: the FASTA, its `.fai`, the five BWA files."""
        return {
            "genomic_fasta": self.fasta_url,
            "fasta_fai": self.fasta_url + ".fai",
            **{
                role: self.index_url + suffix for role, suffix in INDEX_SUFFIXES.items()
            },
        }


def parse_loc(text: str, *, catalogue: str) -> tuple[CatalogueRow, ...]:
    """Rows of a Galaxy `.loc` file: value, dbkey, display name, path, tab-separated."""
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 4 or not all(field.strip() for field in fields[:4]):
            raise CatalogueUnreadable(f"{catalogue} line {number} is not a loc row")
        value, dbkey, name, path = (field.strip() for field in fields[:4])
        rows.append(
            CatalogueRow(
                catalogue=catalogue, value=value, dbkey=dbkey, name=name, path=path
            )
        )
    return tuple(rows)


def parse_tool_data(text: str, *, catalogue: str) -> tuple[CatalogueRow, ...]:
    """Rows of a Galaxy `/api/tool_data/<table>` response: `columns` names the fields of
    each `fields` row. usegalaxy.org gives paths relative to its own data folders."""
    try:
        table = json.loads(text)
        columns = table["columns"]
        rows = [dict(zip(columns, values, strict=True)) for values in table["fields"]]
        return tuple(
            CatalogueRow(
                catalogue=catalogue,
                value=row["value"],
                dbkey=row["dbkey"],
                name=row["name"],
                path=row["path"],
            )
            for row in rows
        )
    except (ValueError, KeyError, TypeError) as error:
        raise CatalogueUnreadable(f"{catalogue} is not a tool-data table") from error


def datacache_url(path: str) -> str | None:
    """The datacache mirror of a path in one of Galaxy's repositories, or None for a
    path outside them."""
    for repository in REPOSITORIES.values():
        if path.startswith(repository.cvmfs) and len(path) > len(repository.cvmfs):
            return repository.mirror + path[len(repository.cvmfs) :]
    return None


def repository_of(catalogue: str) -> str:
    """The repository a catalogue name belongs to (`brc-bwa_mem_index.loc`: `brc`)."""
    return catalogue.split("-", 1)[0]


def candidate_id(catalogue: str, value: str) -> str:
    """A publication's identity: its repository and the catalogue row's value."""
    return f"galaxy:{repository_of(catalogue)}:{value}"


def row_candidate(
    row: CatalogueRow, fasta_rows: Sequence[CatalogueRow]
) -> GalaxyCandidate | None:
    """The publication an index row lists: its index and FASTA (the one
    `fasta_indexes.loc` row of the same value in the same repository) both on the
    datacache; None when either is not."""
    index_url = datacache_url(row.path)
    fastas = [
        url
        for fasta in fasta_rows
        if fasta.value == row.value
        and repository_of(fasta.catalogue) == repository_of(row.catalogue)
        and (url := datacache_url(fasta.path))
    ]
    if index_url is None or len(fastas) != 1:
        return None
    return GalaxyCandidate(
        candidate_id=candidate_id(row.catalogue, row.value),
        catalogue=row.catalogue,
        value=row.value,
        dbkey=row.dbkey,
        display_name=row.name,
        index_url=index_url,
        fasta_url=fastas[0],
    )


def galaxy_candidates(
    value: str,
    index_rows: Sequence[CatalogueRow],
    fasta_rows: Sequence[CatalogueRow],
) -> tuple[GalaxyCandidate, ...]:
    """Each publication of the catalogue value `value`, in catalogue order, one per
    repository (`row_candidate`)."""
    candidates: dict[str, GalaxyCandidate] = {}
    for row in index_rows:
        identity = candidate_id(row.catalogue, row.value)
        if row.value != value or identity in candidates:
            continue
        found = row_candidate(row, fasta_rows)
        if found is not None:
            candidates[identity] = found
    return tuple(candidates.values())


PAGE_ROWS = 25
"""The index rows one search shows, from its offset: a named cut, the rest reachable
by the next offset (ADR 0049 bounded views)."""


def matches(row: CatalogueRow, search_text: str | None) -> bool:
    """Galaxy's own fields searched as written: the text, case-insensitive, within
    the row's value, dbkey or display name; no text matches every row."""
    if search_text is None:
        return True
    text = search_text.casefold()
    return any(text in field.casefold() for field in (row.value, row.dbkey, row.name))
