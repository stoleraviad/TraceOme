"""Prebuilt BWA index candidates for an assembly accession.

The candidates are the product's (`bio_neb.prebuilt_index.catalogue.series_candidates`,
#365): each publication of Galaxy's datacache catalogues whose index and FASTA both have
datacache paths, for every accession of the requested assembly series, retrieved at run
time (owner, #365 SP3 a). This module only adds the tool's display: naming style and
content parts read from a fetched `.fai`, the relation to the request and the index
format. NCBI stays the identity source: whether a candidate fits is the product fit's
answer (`tools.reference_index.fit.check_fit`), and only an all-contig M5 match sets
`auto_pick` (`attach_fit`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from bio_neb.content import canonical_json_digest
from bio_neb.prebuilt_index.catalogue import CatalogueRow
from tools.reference_index.fit import FitReport

from .contracts import (
    Candidate,
    CandidateList,
    ContentPart,
    FitSummary,
    NamingStyle,
)
from .galaxy_rule import series_candidates

_STYLES: tuple[tuple[NamingStyle, re.Pattern[str]], ...] = (
    ("refseq", re.compile(r"^(AC|NC|NG|NT|NW|NZ)_[A-Z0-9]+\.\d+$")),
    ("genbank", re.compile(r"^[A-Z]{1,6}\d{5,10}\.\d+$")),
    ("ucsc", re.compile(r"^chr")),
    ("ensembl", re.compile(r"^(\d+|[IVXYWZ]+|MT|Mito)$")),
)
_PARTS: tuple[tuple[ContentPart, re.Pattern[str]], ...] = (
    ("alts", re.compile(r"_alt$")),
    ("patches", re.compile(r"_(fix|novel)$")),
    ("decoys", re.compile(r"_decoy$")),
    ("hla", re.compile(r"^HLA-")),
    ("ebv", re.compile(r"^(chrEBV|EBV|NC_007605\.1)$")),
)
_GALAXY_INDEX = "bwa 0.7.x (Galaxy data manager; the exact version is not published)"


def find(
    accession: str,
    index_rows: Sequence[CatalogueRow],
    fasta_rows: Sequence[CatalogueRow],
    *,
    fai_names: Mapping[str, Sequence[str]] | None = None,
) -> CandidateList:
    """The product's candidates of the requested accession's assembly series (same
    GCA/GCF number, any prefix or version), exact matches first, each with the names
    its fetched `.fai` gives (`fai_names`, by candidate id) when there is one."""
    observed = fai_names or {}
    return CandidateList(
        requested_accession=accession,
        candidates=tuple(
            _candidate(
                candidate_id=found.candidate_id,
                source="galaxy",
                catalogue=found.catalogue,
                assembly_accession=found.value,
                requested=accession,
                display_name=found.display_name,
                location=found.index_url,
                names=observed.get(found.candidate_id),
                index_format=_GALAXY_INDEX,
            )
            for found in series_candidates(accession, index_rows, fasta_rows)
        ),
    )


def attach_fit(listed: CandidateList, report: FitReport) -> CandidateList:
    """The list with the report's candidate carrying its fit, and `auto_pick` set
    exactly when every contig matched by M5 and the index checks passed."""
    ids = [candidate.candidate_id for candidate in listed.candidates]
    if report.requested_accession != listed.requested_accession:
        raise ValueError(
            f"the report checks {report.requested_accession}, "
            f"the list is for {listed.requested_accession}"
        )
    if report.candidate_id not in ids:
        raise ValueError(f"{report.candidate_id} is not a listed candidate")
    fit = FitSummary(
        verdict=report.verdict,
        sequence_match=report.sequence_match,
        report_sha256=canonical_json_digest(report.model_dump(mode="json")),
    )
    return listed.model_copy(
        update={
            "candidates": tuple(
                Candidate.model_validate(
                    {
                        **candidate.model_dump(),
                        "fit": fit.model_dump(),
                        "auto_pick": fit.verdict != "refused",
                    }
                )
                if candidate.candidate_id == report.candidate_id
                else candidate
                for candidate in listed.candidates
            )
        }
    )


def carry_fits(listed: CandidateList, previous: CandidateList) -> CandidateList:
    """A refreshed list keeps the fit each still-listed candidate already carried.
    The retained `FitSummary` (with its report digest) is trusted as recorded (ADR
    0029); no report is re-read or re-hashed."""
    if previous.requested_accession != listed.requested_accession:
        raise ValueError(
            f"the previous list is for {previous.requested_accession}, "
            f"not {listed.requested_accession}"
        )
    fits = {c.candidate_id: c.fit for c in previous.candidates if c.fit is not None}
    return listed.model_copy(
        update={
            "candidates": tuple(
                Candidate.model_validate(
                    {
                        **candidate.model_dump(),
                        "fit": fits[candidate.candidate_id].model_dump(),
                        "auto_pick": fits[candidate.candidate_id].verdict != "refused",
                    }
                )
                if candidate.candidate_id in fits
                else candidate
                for candidate in listed.candidates
            )
        }
    )


def style_of_names(names: Sequence[str]) -> NamingStyle:
    """The one style every contig name follows (HLA allele contigs aside), else mixed."""
    styles = {
        next((style for style, pattern in _STYLES if pattern.match(name)), "other")
        for name in names
        if not name.startswith("HLA-")
    }
    if not styles:
        return "unknown"
    return styles.pop() if len(styles) == 1 and "other" not in styles else "mixed"


def class_of_names(names: Sequence[str]) -> tuple[ContentPart, ...]:
    """Primary sequence plus each extra kind the contig names show."""
    found = {part for name in names for part, pattern in _PARTS if pattern.search(name)}
    return ("primary", *(part for part, _ in _PARTS if part in found))


def _candidate(
    *, requested: str, names: Sequence[str] | None, **fields: str
) -> Candidate:
    relation = (
        "exact" if fields["assembly_accession"] == requested else "same_assembly_series"
    )
    return Candidate(
        **fields,
        relation=relation,
        naming_style=style_of_names(names) if names else "unknown",
        contig_count=len(names) if names else None,
        contig_count_basis="observed_fai" if names else "not_measured",
        content_class=class_of_names(names) if names else ("unknown",),
        content_basis="observed_fai" if names else "not_measured",
    )
