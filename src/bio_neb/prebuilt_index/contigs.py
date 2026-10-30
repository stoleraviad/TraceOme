"""A prebuilt index's contigs beside NCBI's copied sequence rows (#464, ADR 0049).

The Run's reference is its assembly's sequences of `REFERENCE_ROLES`, each named as in
NCBI's FASTA (`requested_sequences`). A candidate index's contigs (names and lengths, from
its `.ann`) are paired with them by any name NCBI's report lists a sequence under, an
exact membership read. The result states facts only: each requested sequence paired,
missing or of another length, each extra contig with the role of the NCBI row naming it,
and any name that pairs ambiguously. Which index is the Run's is the planning model's
choice; execution trusts it and renames its contigs to NCBI's in flight
(`IndexContigs.ncbi_names`, `rename`; owner I7, #464).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field, StrictInt, StrictStr

from bio_neb.contracts import FrozenContract

REFERENCE_ROLES = frozenset(
    {"assembled-molecule", "unlocalized-scaffold", "unplaced-scaffold"}
)
"""The NCBI sequence roles a Run's reference holds, in any assembly unit: alt loci and
patch scaffolds are left out (owner, #464; ADR 0049 retires #365 A's unit-name set)."""
MAIN_ROLE = "assembled-molecule"
"""The role an index must hold to fit: the main assembly's chromosomes and organelles;
the minor scaffolds are shown, never required (owner, #459, 2026-10-30)."""


class RequestedContig(FrozenContract):
    name: StrictStr
    length: StrictInt = Field(gt=0)


class ContigPair(FrozenContract):
    """One requested sequence and the candidate contig that names it, if any."""

    requested_name: StrictStr
    role: StrictStr
    """NCBI's role of the requested sequence, copied."""
    length: StrictInt = Field(gt=0)
    candidate_name: StrictStr | None
    candidate_length: StrictInt | None
    status: Literal["paired", "other_length", "missing"]


class ExtraContig(FrozenContract):
    """A candidate contig that is no requested sequence, with the NCBI row naming it
    and that row's name in the Run's naming, if it has one (`ncbi_name`: RefSeq for a
    GCF assembly, GenBank for a GCA)."""

    name: StrictStr
    length: StrictInt = Field(gt=0)
    ncbi_sequence: StrictStr | None
    ncbi_role: StrictStr | None
    ncbi_name: StrictStr | None


class IndexContigs(FrozenContract):
    requested: tuple[ContigPair, ...]
    extras: tuple[ExtraContig, ...]
    renamed: StrictInt = Field(ge=0)
    """Paired contigs whose name is not NCBI's FASTA name."""
    ambiguous: tuple[StrictStr, ...]
    """NCBI sequences named by several candidate contigs, or a candidate name NCBI
    gives several sequences."""

    def ncbi_names(self) -> dict[str, str]:
        """Each candidate name that is not NCBI's, and NCBI's name for it: a paired
        contig's requested name, an extra's `ncbi_name`; an extra without one keeps
        its own (owner I7, #464)."""
        names = {
            pair.candidate_name: pair.requested_name
            for pair in self.requested
            if pair.candidate_name is not None
        }
        names.update(
            (extra.name, extra.ncbi_name)
            for extra in self.extras
            if extra.ncbi_name is not None
        )
        return {old: new for old, new in names.items() if old != new}

    @property
    def holds_every_requested_sequence(self) -> bool:
        """Every main-assembly sequence (`MAIN_ROLE`) paired at its length, and no
        name ambiguous: what a choice needs. A minor scaffold missing or of another
        length is shown in the pairs (owner, #459)."""
        return not self.ambiguous and all(
            pair.status == "paired" for pair in self.requested if pair.role == MAIN_ROLE
        )


def requested_sequences(accession: str, sequences: Sequence[Any]) -> list:
    """Each sequence of the Run's reference (`REFERENCE_ROLES`) by its name in NCBI's
    FASTA: the RefSeq accession for a GCF assembly, the GenBank one for a GCA."""
    return reference_partition(accession, sequences)[0]


def reference_partition(
    accession: str, sequences: Sequence[Any]
) -> tuple[list, tuple[str, ...]]:
    """The report's rows split by `REFERENCE_ROLES`, each by its name in NCBI's FASTA
    (the RefSeq accession for a GCF assembly, the GenBank one for a GCA): the Run's
    reference sequences as (name, row), and the names of the others (alt loci and
    patches, owner #464), which the CPU build and input storage's measure drop from
    the FASTA. A row without such a name is in neither."""
    requested, excluded = [], []
    for sequence in sequences:
        name = sequence.refseq if accession.startswith("GCF_") else sequence.genbank
        if not name:
            continue
        if sequence.sequence_role in REFERENCE_ROLES:
            requested.append((name, sequence))
        else:
            excluded.append(name)
    return requested, tuple(excluded)


def pair_contigs(
    accession: str, sequences: Sequence[Any], contigs: Sequence[tuple[str, int]]
) -> IndexContigs:
    """The candidate's contigs (name, length) against NCBI's rows of the Run's
    assembly, every row's names read exactly."""
    requested = requested_sequences(accession, sequences)
    naming = "refseq" if accession.startswith("GCF_") else "genbank"
    index_of = {id(sequence): i for i, (_, sequence) in enumerate(requested)}
    rows_named: dict[str, list[Any]] = {}
    for sequence in sequences:
        for name in sequence.names:
            rows_named.setdefault(name, []).append(sequence)
    paired: dict[int, list[tuple[str, int]]] = {}
    extras, ambiguous = [], []
    for name, length in contigs:
        rows = rows_named.get(name, [])
        if len(rows) > 1:
            ambiguous.append(name)
            continue
        row = rows[0] if rows else None
        if row is not None and id(row) in index_of:
            paired.setdefault(index_of[id(row)], []).append((name, length))
            continue
        extras.append(
            ExtraContig(
                name=name,
                length=length,
                ncbi_sequence=row.sequence_name if row else None,
                ncbi_role=row.sequence_role if row else None,
                ncbi_name=getattr(row, naming) if row else None,
            )
        )
    pairs = []
    for i, (name, sequence) in enumerate(requested):
        named = paired.get(i, [])
        if len(named) > 1:
            ambiguous.append(name)
        candidate = named[0] if named else None
        pairs.append(
            ContigPair(
                requested_name=name,
                role=sequence.sequence_role,
                length=sequence.length,
                candidate_name=candidate[0] if candidate else None,
                candidate_length=candidate[1] if candidate else None,
                status="missing"
                if candidate is None
                else "paired"
                if candidate[1] == sequence.length
                else "other_length",
            )
        )
    return IndexContigs(
        requested=tuple(pairs),
        extras=tuple(extras),
        renamed=sum(
            1
            for pair in pairs
            if pair.candidate_name not in {None, pair.requested_name}
        ),
        ambiguous=tuple(ambiguous),
    )


__all__ = [
    "MAIN_ROLE",
    "REFERENCE_ROLES",
    "ContigPair",
    "ExtraContig",
    "IndexContigs",
    "RequestedContig",
    "pair_contigs",
    "reference_partition",
    "requested_sequences",
]
