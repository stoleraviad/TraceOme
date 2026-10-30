"""Does a prebuilt BWA index fit the requested assembly's FASTA?

Two questions, kept apart:

- **Sequence.** Is the candidate's FASTA the requested sequence? Contigs are paired by
  their M5 (the SAM `@SQ` MD5 of the upper-cased sequence), so names, descriptions, line
  width and soft-masking do not matter.
- **Index.** Was the candidate's index built from the candidate's FASTA? The one BWA-index
  reader (`bio_neb.parabricks.bwa_index`, #449) reads the FASTA and its files and returns
  every finding; the fit reports each one, where local-bundle admission refuses on the
  first. The `.pac` content itself is not compared with the FASTA (owner R2 a, #365): the
  controller has no `bwa fa2pac`; its size and trailer are checked.

A file the reader cannot read as its format refuses the fit (`BundleUnreadable`) with the
reader's reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, get_args

from pydantic import Field, model_validator

from bio_neb.contracts import FrozenContract
from bio_neb.parabricks.bwa_index import (
    BwaIndexReading,
    Contig,
    FastaReading,
    Finding,
    IndexFile,
    read_bwa_index,
    read_fasta,
)

FIT_SCHEMA = "bio-neb.prebuilt-index-fit/2"

ContigStatus = Literal[
    "match", "sequence_differs", "missing_from_candidate", "extra_in_candidate"
]
IndexCheckName = Literal["fasta_fai", "ann_fai", "amb_ann", "pac_size", "bwt_sa_header"]

BUNDLE_SUFFIXES = {
    "fasta_fai": ".fai",
    "bwa_amb": ".amb",
    "bwa_ann": ".ann",
    "bwa_bwt": ".bwt",
    "bwa_pac": ".pac",
    "bwa_sa": ".sa",
}
"""A prebuilt bundle's files beside its FASTA, by prepared role."""
_FINDING_CHECKS = {
    "fasta_fai_mismatch": "fasta_fai",
    "ann_fasta_mismatch": "ann_fai",
    "amb_ann_mismatch": "amb_ann",
    "pac_length_or_trailer_mismatch": "pac_size",
    "bwt_header_invalid": "bwt_sa_header",
    "sa_header_invalid": "bwt_sa_header",
    "bwt_sa_alignment_invalid": "bwt_sa_header",
}
"""The reader's mismatch findings, by the fit's index check that reports them."""


class BundleUnreadable(Exception):
    """A file the fit needs is missing or not the format it claims."""

    def __init__(self, reason: str, path: Path) -> None:
        super().__init__(f"{reason}: {path}")
        self.reason = reason
        self.path = path


class FileDigest(FrozenContract):
    """One file the report read, by its bundle role."""

    role: str
    name: str
    byte_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ContigFit(FrozenContract):
    """One contig of either FASTA, paired by sequence (M5), then by name or position."""

    requested_name: str | None
    candidate_name: str | None
    requested_length: int | None
    candidate_length: int | None
    requested_m5: str | None
    candidate_m5: str | None
    status: ContigStatus

    @model_validator(mode="after")
    def status_follows_both_sides(self) -> ContigFit:
        requested = (self.requested_name, self.requested_length, self.requested_m5)
        candidate = (self.candidate_name, self.candidate_length, self.candidate_m5)
        for side in (requested, candidate):
            if len({value is None for value in side}) != 1:
                raise ValueError("a contig side is given whole or not at all")
        sides = (requested[0] is not None, candidate[0] is not None)
        same = requested[1:] == candidate[1:]
        expected = {
            "match": sides == (True, True) and same,
            "sequence_differs": sides == (True, True) and not same,
            "missing_from_candidate": sides == (True, False),
            "extra_in_candidate": sides == (False, True),
        }
        if not expected[self.status]:
            raise ValueError(f"{self.status} does not follow the two sides given")
        return self


class IndexCheck(FrozenContract):
    """One consistency check of the candidate's index against its own FASTA."""

    check: IndexCheckName
    passed: bool
    detail: str = ""


class BoundExtra(FrozenContract):
    """A candidate contig beyond the requested FASTA that the Execution Review bound,
    by its name and length (#464; spec r1 SP1)."""

    name: str
    length: int = Field(gt=0)


class FitReport(FrozenContract):
    """Whether one prebuilt candidate fits the requested assembly's FASTA."""

    schema_version: Literal["bio-neb.prebuilt-index-fit/2"] = FIT_SCHEMA
    requested_accession: str
    candidate_id: str
    requested_files: tuple[FileDigest, ...]
    candidate_files: tuple[FileDigest, ...]
    contigs: tuple[ContigFit, ...]
    sequence_match: Literal["all", "partial", "none"]
    names: Literal["identical", "renamed"]
    order: Literal["same", "different"]
    index_checks: tuple[IndexCheck, ...]
    bound_extras: tuple[BoundExtra, ...] = ()
    """Candidate contigs beyond the requested FASTA that the Execution Review bound
    (#464): extra at the bound length and still a fit."""
    verdict: Literal["fits", "fits_after_rename", "refused"]
    reasons: tuple[str, ...]

    @model_validator(mode="after")
    def verdict_follows_the_evidence(self) -> FitReport:
        matched = [c for c in self.contigs if c.status == "match"]
        unexpected = [
            c
            for c in self.contigs
            if c.status != "match" and not _bound(c, self.bound_extras)
        ]
        sequence_match = (
            "all"
            if self.contigs and len(matched) == len(self.contigs)
            else "partial"
            if matched
            else "none"
        )
        names = (
            "identical"
            if all(c.requested_name == c.candidate_name for c in matched)
            else "renamed"
        )
        checks = [check.check for check in self.index_checks]
        if sorted(checks) != sorted(get_args(IndexCheckName)):
            raise ValueError("every index check is reported exactly once")
        if (self.sequence_match, self.names) != (sequence_match, names):
            raise ValueError("sequence_match and names must follow the contigs")
        fits = (
            bool(matched)
            and not unexpected
            and all(check.passed for check in self.index_checks)
        )
        expected = (
            "refused"
            if not fits
            else "fits"
            if names == "identical"
            else "fits_after_rename"
        )
        if self.verdict != expected or (self.verdict == "refused") != bool(
            self.reasons
        ):
            raise ValueError("verdict and reasons must follow the contigs and checks")
        return self

    def renames(self) -> dict[str, str]:
        """Each matched candidate contig's name to the requested name it pairs with."""
        return {
            c.candidate_name: c.requested_name
            for c in self.contigs
            if c.status == "match" and c.candidate_name != c.requested_name
        }


def check_fit(
    *,
    requested: Path,
    requested_accession: str,
    candidate: Path,
    candidate_id: str,
    bound_extras: Sequence[BoundExtra] = (),
) -> FitReport:
    """Report every contig pairing and index check; the verdict follows from them.
    A candidate contig in `bound_extras` (the Review bound its name and length, #464)
    may be extra; the same name at another length is refused. The candidate's files
    sit beside its FASTA (`BUNDLE_SUFFIXES`)."""
    bound = tuple(bound_extras)
    requested_fasta = _fasta(requested)
    reading = _bundle(candidate)
    wanted, offered = list(requested_fasta.contigs), list(reading.contigs)
    contigs = _pair(wanted, offered)
    matched = [contig for contig in contigs if contig.status == "match"]
    same_names = all(c.requested_name == c.candidate_name for c in matched)
    index_checks = _index_checks(reading)
    reasons = tuple(
        _contig_reason(contig, bound)
        for contig in contigs
        if contig.status != "match" and not _bound(contig, bound)
    ) + tuple(
        f"{check.check}: {check.detail}" for check in index_checks if not check.passed
    )
    return FitReport(
        requested_accession=requested_accession,
        candidate_id=candidate_id,
        requested_files=(_digest(requested_fasta.file),),
        candidate_files=(
            _digest(reading.fasta),
            *(_digest(reading.file(role)) for role in BUNDLE_SUFFIXES),
        ),
        contigs=contigs,
        sequence_match=(
            "all" if len(matched) == len(contigs) else "partial" if matched else "none"
        ),
        names="identical" if same_names else "renamed",
        order=(
            "same" if [c.m5 for c in wanted] == [c.m5 for c in offered] else "different"
        ),
        index_checks=index_checks,
        bound_extras=bound,
        verdict=(
            "refused" if reasons else "fits" if same_names else "fits_after_rename"
        ),
        reasons=reasons,
    )


def _pair(wanted: list[Contig], offered: list[Contig]) -> tuple[ContigFit, ...]:
    """Pair by M5 first; pair what is left by name, then by position at equal length,
    as a changed sequence; anything still alone is missing from, or extra in, the
    candidate."""
    free = list(range(len(offered)))
    pairs: dict[int, int] = {}
    for i, contig in enumerate(wanted):
        j = next(
            (
                j
                for j in free
                if (offered[j].m5, offered[j].length) == (contig.m5, contig.length)
            ),
            None,
        )
        if j is not None:
            pairs[i] = j
            free.remove(j)
    differs: dict[int, int] = {}
    for same in (
        lambda i, j: wanted[i].name == offered[j].name,
        lambda i, j: i == j and wanted[i].length == offered[j].length,
    ):
        for i in range(len(wanted)):
            if i in pairs or i in differs:
                continue
            j = next((j for j in free if same(i, j)), None)
            if j is not None:
                differs[i] = j
                free.remove(j)
    rows = []
    for i, contig in enumerate(wanted):
        j = pairs.get(i, differs.get(i))
        other = offered[j] if j is not None else None
        rows.append(
            ContigFit(
                requested_name=contig.name,
                candidate_name=other.name if other else None,
                requested_length=contig.length,
                candidate_length=other.length if other else None,
                requested_m5=contig.m5,
                candidate_m5=other.m5 if other else None,
                status=(
                    "match"
                    if i in pairs
                    else "sequence_differs"
                    if other
                    else "missing_from_candidate"
                ),
            )
        )
    for j in free:
        rows.append(
            ContigFit(
                requested_name=None,
                candidate_name=offered[j].name,
                requested_length=None,
                candidate_length=offered[j].length,
                requested_m5=None,
                candidate_m5=offered[j].m5,
                status="extra_in_candidate",
            )
        )
    return tuple(rows)


def _bound(contig: ContigFit, bound: Sequence[BoundExtra]) -> bool:
    """An extra contig the Review bound, at the name and length it bound."""
    return contig.status == "extra_in_candidate" and any(
        (extra.name, extra.length) == (contig.candidate_name, contig.candidate_length)
        for extra in bound
    )


def _contig_reason(contig: ContigFit, bound: Sequence[BoundExtra]) -> str:
    if contig.status == "sequence_differs":
        return (
            f"sequence_differs: {contig.requested_name} "
            f"(candidate {contig.candidate_name})"
        )
    if contig.status == "missing_from_candidate":
        return f"missing_from_candidate: {contig.requested_name}"
    lengths = [extra.length for extra in bound if extra.name == contig.candidate_name]
    if lengths:
        return (
            f"extra_in_candidate: {contig.candidate_name} is "
            f"{contig.candidate_length} bases; the Review bound "
            + ", ".join(map(str, lengths))
        )
    return f"extra_in_candidate: {contig.candidate_name}"


def _index_checks(reading: BwaIndexReading) -> tuple[IndexCheck, ...]:
    failed = {_FINDING_CHECKS[finding.reason] for finding in reading.findings}
    total = reading.forward_base_count
    pac = reading.file("bwa_pac").byte_count
    indexed = 2 * total
    return (
        _check(
            "fasta_fai",
            "fasta_fai" not in failed,
            "the .fai names and lengths differ from the FASTA",
        ),
        _check(
            "ann_fai",
            "ann_fai" not in failed,
            "the .ann names, lengths or total differ from the .fai",
        ),
        _check(
            "amb_ann",
            "amb_ann" not in failed,
            "the .amb header differs from the .ann or its hole count is wrong",
        ),
        _check(
            "pac_size",
            "pac_size" not in failed,
            f".pac is {pac} bytes ending {reading.pac_trailer:02x}; "
            f"{total} bases need {total // 4 + 2} ending {total % 4:02x}",
        ),
        _check(
            "bwt_sa_header",
            "bwt_sa_header" not in failed,
            f"the .bwt/.sa headers do not describe {indexed} indexed bases, "
            "or their payloads are not whole 4- and 8-byte words",
        ),
    )


def _check(name: IndexCheckName, passed: bool, detail: str) -> IndexCheck:
    return IndexCheck(check=name, passed=passed, detail="" if passed else detail)


def _fasta(path: Path) -> FastaReading:
    try:
        reading = read_fasta(path)
    except OSError as error:
        raise BundleUnreadable("fasta_unreadable", path) from error
    _refuse_unreadable(reading.findings)
    return reading


def _bundle(fasta: Path) -> BwaIndexReading:
    """The candidate's FASTA and index files through the product's reader."""
    files = {role: Path(f"{fasta}{suffix}") for role, suffix in BUNDLE_SUFFIXES.items()}
    try:
        reading = read_bwa_index(fasta, files)
    except OSError as error:
        path = Path(error.filename) if error.filename else fasta
        reason = "fasta_unreadable" if path == fasta else "file_unreadable"
        raise BundleUnreadable(reason, path) from error
    _refuse_unreadable(reading.findings)
    return reading


def _refuse_unreadable(findings: Sequence[Finding]) -> None:
    for finding in findings:
        if finding.kind != "mismatch":
            raise BundleUnreadable(finding.reason, finding.path) from finding.cause


def _digest(file: IndexFile) -> FileDigest:
    return FileDigest(
        role=file.role,
        name=file.path.name,
        byte_count=file.byte_count,
        sha256=file.sha256,
    )
