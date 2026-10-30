"""Variant placements, copied: the one reader of NCBI dbSNP's and EVA's placements (#468).

A RefSNP record's `placements_with_allele` and EVA's clustered and submitted variants are copied
into one row shape, `VariantPlacement`, with each source's own fields under that source's
name. Every placement is kept, on every sequence and assembly: no top-level or primary filter,
no cap and no ranking (ADR 0049). Only a value missing from the answer reads as ``None``. An
answer that does not hold the shape its source documents raises `PlacementsUnreadable`, the
reader's typing of what it cannot read, never a verdict on a value it holds.

Which placement is on the Run's assembly, and the Run's name for its sequence, are the lookup's
to mark (`investigation_sources`); a row read here is marked off the Run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, StrictBool, StrictInt, StrictStr, model_validator

from bio_neb.contracts import StrictFrozenContract


class PlacementsUnreadable(ValueError):
    """An answer that does not hold the placements its source documents."""


def _flag(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


class SpdiAllele(StrictFrozenContract):
    """One allele of a dbSNP placement as dbSNP writes it: its SPDI (a zero-based
    position) and its HGVS name."""

    seq_id: StrictStr
    position: StrictInt = Field(ge=0)
    deleted_sequence: StrictStr
    inserted_sequence: StrictStr
    hgvs: StrictStr | None


class DbsnpAssembly(StrictFrozenContract):
    """One entry of a placement's `seq_id_traits_by_assembly`, copied."""

    assembly_name: StrictStr
    assembly_accession: StrictStr
    is_top_level: StrictBool | None
    is_alt: StrictBool | None
    is_patch: StrictBool | None
    is_chromosome: StrictBool | None


class DbsnpPlacement(StrictFrozenContract):
    """A RefSNP placement's own fields, copied."""

    is_ptlp: StrictBool | None
    seq_type: StrictStr | None
    mol_type: StrictStr | None
    is_aln_opposite_orientation: StrictBool | None
    is_mismatch: StrictBool | None
    assemblies: tuple[DbsnpAssembly, ...]
    alleles: tuple[SpdiAllele, ...]


class EvaSubmittedVariant(StrictFrozenContract):
    """One submitted variant (ss) EVA keeps at a clustered placement, in EVA's form: no
    context base, an empty side for an insertion or deletion."""

    ss: StrictStr
    reference: StrictStr
    alternate: StrictStr
    project: StrictStr | None


class EvaClusteredVariant(StrictFrozenContract):
    """A clustered-variant record's own fields, copied, with the submitted variants EVA
    keeps at its place."""

    accession: StrictInt = Field(gt=0)
    version: StrictInt | None
    assembly_accession: StrictStr
    taxonomy_accession: StrictInt | None
    start: StrictInt = Field(gt=0)
    type: StrictStr
    validated: StrictBool | None
    map_weight: StrictInt | None
    created_date: StrictStr | None
    submitted: tuple[EvaSubmittedVariant, ...] = ()


class VariantPlacement(StrictFrozenContract):
    """One placement of a variant as its source states it: its sequence (dbSNP's
    `seq_id`, EVA's `contig`), its source's own fields, and whether the lookup found it on
    the Run's assembly, with the Run's name for its sequence when it has one."""

    sequence: StrictStr
    on_run_assembly: StrictBool = False
    run_contig: StrictStr | None = None
    dbsnp: DbsnpPlacement | None = Field(default=None, exclude_if=lambda v: v is None)
    eva: EvaClusteredVariant | None = Field(
        default=None, exclude_if=lambda v: v is None
    )

    @model_validator(mode="after")
    def one_source(self) -> "VariantPlacement":
        if (self.dbsnp is None) == (self.eva is None):
            raise ValueError("a placement holds exactly one source's fields")
        if self.run_contig is not None and not self.on_run_assembly:
            raise ValueError("only a placement on the Run's assembly has a Run contig")
        return self


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlacementsUnreadable(what)
    return value


def _list(value: Any, what: str) -> list:
    if not isinstance(value, list):
        raise PlacementsUnreadable(what)
    return value


def refsnp_placements(document: Mapping[str, Any]) -> tuple[VariantPlacement, ...]:
    """Every placement of a current RefSNP record (`primary_snapshot_data`), in dbSNP's
    order, every field copied."""
    snapshot = _mapping(
        document.get("primary_snapshot_data"), "the record has no primary snapshot"
    )
    placements = _list(
        snapshot.get("placements_with_allele", []),
        "dbSNP's placements_with_allele is not a list",
    )
    return tuple(_refsnp_placement(item) for item in placements)


def _refsnp_placement(item: Any) -> VariantPlacement:
    unreadable = "a dbSNP placement does not hold its sequence, annotation and alleles"
    placement = _mapping(item, unreadable)
    annotation = _mapping(placement.get("placement_annot", {}), unreadable)
    sequence = placement.get("seq_id")
    if not isinstance(sequence, str) or not sequence:
        raise PlacementsUnreadable(unreadable)
    assemblies = []
    for trait in _list(annotation.get("seq_id_traits_by_assembly", []), unreadable):
        trait = _mapping(trait, unreadable)
        name, accession = trait.get("assembly_name"), trait.get("assembly_accession")
        if not isinstance(name, str) or not isinstance(accession, str):
            raise PlacementsUnreadable(unreadable)
        assemblies.append(
            DbsnpAssembly(
                assembly_name=name,
                assembly_accession=accession,
                is_top_level=_flag(trait.get("is_top_level")),
                is_alt=_flag(trait.get("is_alt")),
                is_patch=_flag(trait.get("is_patch")),
                is_chromosome=_flag(trait.get("is_chromosome")),
            )
        )
    alleles = []
    for allele in _list(placement.get("alleles", []), unreadable):
        allele = _mapping(allele, unreadable)
        spdi = _mapping(
            _mapping(allele.get("allele"), unreadable).get("spdi"), unreadable
        )
        values = (
            spdi.get("seq_id"),
            spdi.get("position"),
            spdi.get("deleted_sequence"),
            spdi.get("inserted_sequence"),
        )
        if not (
            isinstance(values[0], str)
            and _count(values[1]) is not None
            and values[1] >= 0
            and isinstance(values[2], str)
            and isinstance(values[3], str)
        ):
            raise PlacementsUnreadable(unreadable)
        alleles.append(
            SpdiAllele(
                seq_id=values[0],
                position=values[1],
                deleted_sequence=values[2],
                inserted_sequence=values[3],
                hgvs=_text(allele.get("hgvs")),
            )
        )
    return VariantPlacement(
        sequence=sequence,
        dbsnp=DbsnpPlacement(
            is_ptlp=_flag(placement.get("is_ptlp")),
            seq_type=_text(annotation.get("seq_type")),
            mol_type=_text(annotation.get("mol_type")),
            is_aln_opposite_orientation=_flag(
                annotation.get("is_aln_opposite_orientation")
            ),
            is_mismatch=_flag(annotation.get("is_mismatch")),
            assemblies=tuple(assemblies),
            alleles=tuple(alleles),
        ),
    )


def eva_clustered_placements(document: Any) -> tuple[VariantPlacement, ...]:
    """Every clustered-variant record EVA answered, in EVA's order, every field copied."""
    return tuple(
        _eva_record(item)
        for item in _list(document, "EVA's clustered-variant answer is not a list")
    )


def _eva_record(item: Any) -> VariantPlacement:
    unreadable = "EVA's clustered-variant records were malformed."
    record = _mapping(item, unreadable)
    data = _mapping(record.get("data"), unreadable)
    accession, start = _count(record.get("accession")), _count(data.get("start"))
    assembly, contig = data.get("assemblyAccession"), data.get("contig")
    kind = data.get("type")
    if not (
        accession is not None
        and accession > 0
        and start is not None
        and start > 0
        and isinstance(assembly, str)
        and assembly
        and _one_line(contig)
        and isinstance(kind, str)
    ):
        raise PlacementsUnreadable(unreadable)
    return VariantPlacement(
        sequence=contig,
        eva=EvaClusteredVariant(
            accession=accession,
            version=_count(record.get("version")),
            assembly_accession=assembly,
            taxonomy_accession=_count(data.get("taxonomyAccession")),
            start=start,
            type=kind,
            validated=_flag(data.get("validated")),
            map_weight=_count(data.get("mapWeight")),
            created_date=_text(data.get("createdDate")),
        ),
    )


def eva_submitted_variants(
    document: Any, accession: int
) -> tuple[tuple[tuple[Any, Any, Any], EvaSubmittedVariant], ...]:
    """Each submitted variant EVA answered for clustered variant ``accession``, with the
    place EVA states it at (its assembly, contig and start). A submitted variant naming
    another cluster is not this cluster's answer."""
    unreadable = "EVA's submitted-variant records were malformed."
    read = []
    for item in _list(document, "EVA's submitted-variant answer was not a list."):
        record = _mapping(item, unreadable)
        data = _mapping(record.get("data"), unreadable)
        ss = _count(record.get("accession"))
        if ss is None or ss <= 0:
            raise PlacementsUnreadable(unreadable)
        if data.get("clusteredVariantAccession") != accession:
            raise PlacementsUnreadable(
                f"EVA's submitted variant ss{ss} names another cluster than "
                f"rs{accession}."
            )
        reference, alternate = data.get("referenceAllele"), data.get("alternateAllele")
        if not isinstance(reference, str) or not isinstance(alternate, str):
            raise PlacementsUnreadable(unreadable)
        read.append(
            (
                (
                    data.get("referenceSequenceAccession"),
                    data.get("contig"),
                    data.get("start"),
                ),
                EvaSubmittedVariant(
                    ss=f"ss{ss}",
                    reference=reference,
                    alternate=alternate,
                    project=_text(data.get("projectAccession")),
                ),
            )
        )
    return tuple(read)


def _one_line(value: Any) -> bool:
    """A source-stated name a row can hold: non-empty, one line."""
    return (
        isinstance(value, str)
        and bool(value)
        and not any(character in value for character in "\r\n\0")
    )
