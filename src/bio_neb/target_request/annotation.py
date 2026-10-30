"""The gene annotations G15 reads (#376), turned into transcripts on the Run's assembly.

Every Run's gene comes from that assembly's own GFF3, NCBI Datasets'
``include_annotation_type=GENOME_GFF`` download (#469, owner Q4 a: no human route):
NCBI's gene reports carry only its latest annotation, which may be on a newer assembly
than the Run's (horse RAPGEF5 is on ``NC_091687.1``, not EquCab3.0's ``NC_009147.3``;
development GET d2), and the assembly's ``annotation_report`` names transcripts without
their exons (d4). Each transcript's CDS segments come in transcript order, with its
translation table and its tags copied from the annotation (ADR 0049).
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

from .gene_change import Transcript

GeneOutcome = Literal["found", "not_found", "ambiguous", "not_checked"]


@dataclass(frozen=True)
class AnnotatedGene:
    """A gene symbol on the Run's assembly: its protein-coding transcripts, or why none."""

    symbol: str
    outcome: GeneOutcome
    reason: str | None = None
    transcripts: tuple[Transcript, ...] = ()


@dataclass(frozen=True)
class _Feature:
    contig: str
    kind: str
    start: int
    end: int
    strand: str
    phase: str
    attributes: dict[str, str]

    @property
    def id(self) -> str | None:
        return self.attributes.get("ID")

    @property
    def parents(self) -> tuple[str, ...]:
        parent = self.attributes.get("Parent")
        return tuple(parent.split(",")) if parent else ()


def gene_from_gff(download: bytes | Path, symbol: str) -> AnnotatedGene:
    """The gene's transcripts from a Datasets GFF3 download (a zip holding
    ``genomic.gff``), found by its symbol, else by a synonym."""
    wanted = symbol.casefold()
    by_symbol, by_synonym = [], []
    for feature in _features(download):
        if feature.kind != "gene":
            continue
        names = {feature.attributes.get(key, "").casefold() for key in ("gene", "Name")}
        synonyms = {
            name.casefold()
            for name in feature.attributes.get("gene_synonym", "").split(",")
        }
        if wanted in names:
            by_symbol.append(feature)
        elif wanted in synonyms:
            by_synonym.append(feature)
    genes = by_symbol or by_synonym
    if not genes:
        return AnnotatedGene(
            symbol, "not_found", f"the assembly's annotation names no gene {symbol}"
        )
    if len(genes) > 1:
        listed = ", ".join(
            f"{gene.id} ({gene.contig}:{gene.start}-{gene.end})" for gene in genes
        )
        return AnnotatedGene(
            symbol,
            "ambiguous",
            f"the assembly's annotation names {len(genes)} genes {symbol}: {listed}",
        )
    [gene] = genes
    # The gene's transcripts, then their CDS segments: a pass each, so no file order is
    # assumed.
    parents = {
        feature.id: feature
        for feature in _features(download)
        if gene.id in feature.parents and feature.kind != "CDS"
    }
    segments: dict[str, list[_Feature]] = {}
    for feature in _features(download):
        if feature.kind == "CDS" and (
            gene.id in feature.parents or set(feature.parents) & parents.keys()
        ):
            key = next(
                (parent for parent in feature.parents if parent in parents), gene.id
            )
            segments.setdefault(key, []).append(feature)
    primary = gene.attributes.get("gene") or gene.attributes.get("Name") or symbol
    transcripts = []
    for key, cds in segments.items():
        parent = parents.get(key)
        accession = (
            (parent.attributes.get("transcript_id") if parent else None)
            or cds[0].attributes.get("protein_id")
            or key
        )
        strand = cds[0].strand
        ordered = sorted(cds, key=lambda f: f.start, reverse=strand == "-")
        # The CDS's own translation table, copied; with none written, the INSDC
        # rule's table 1. The annotation's tags are copied whole.
        tables = {f.attributes.get("transl_table") for f in cds}
        if len(tables) > 1:
            raise ValueError(f"{accession}'s CDS rows name translation tables {tables}")
        [table] = tables
        tags = tuple(
            dict.fromkeys(
                tag
                for feature in ((parent,) if parent else ()) + tuple(cds)
                for tag in feature.attributes.get("tag", "").split(",")
                if tag
            )
        )
        transcripts.append(
            Transcript(
                accession=accession,
                protein=cds[0].attributes.get("protein_id"),
                gene=primary,
                contig=cds[0].contig,
                strand=strand,
                cds=tuple((f.start, f.end) for f in ordered),
                genetic_code=1 if table is None else int(table),
                genetic_code_origin="default" if table is None else "copied",
                tags=tags,
                # A 5'-partial CDS starts mid-codon: copied as written, and its reading
                # says why its codons are not counted.
                cds_phase=int(ordered[0].phase),
            )
        )
    return AnnotatedGene(symbol, "found", transcripts=tuple(transcripts))


def _features(download: bytes | Path) -> Iterator[_Feature]:
    source = io.BytesIO(download) if isinstance(download, bytes) else download
    with zipfile.ZipFile(source) as archive:
        members = [n for n in archive.namelist() if n.endswith("/genomic.gff")]
        if len(members) != 1:
            raise ValueError("the annotation download holds no genomic.gff")
        with archive.open(members[0]) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                if line.startswith("#") or not line.strip():
                    continue
                columns = line.rstrip("\n").split("\t")
                if len(columns) != 9:
                    raise ValueError(f"a GFF line has {len(columns)} columns")
                attributes = dict(
                    (key, unquote(value))
                    for key, _, value in (
                        item.partition("=") for item in columns[8].split(";") if item
                    )
                )
                yield _Feature(
                    contig=columns[0],
                    kind=columns[2],
                    start=int(columns[3]),
                    end=int(columns[4]),
                    strand=columns[6],
                    phase=columns[7],
                    attributes=attributes,
                )
