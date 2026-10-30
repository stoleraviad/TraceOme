"""The local store of fitted bundles, keyed by content digest.

A bundle is stored only after its fit report found an all-contig M5 match and every index
check passed. The digest covers what was fetched (the FASTA and the five BWA files) and is
taken from the fit report's file digests, which were hashed when the check read the files
(ADR 0029: no re-hash of trusted inputs). The store keeps bytes only: identical content
fitted for another assembly or candidate reuses the stored bundle, and each `Choice` keeps
its own fit. The folder `store/sha256/<digest>/` is the product's assembly
(`tools.reference_index.bundle.assemble_prepared_roles`, #365): the eight prepared roles,
renamed to NCBI's names when the fit was `fits_after_rename`, the `.dict` made with pysam's
samtools, and the result read back through the product's reader against the fit.
`bundle.json` is written last and marks a complete bundle.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pysam
from pysam import libcsamtools

from bio_neb.content import canonical_json, canonical_json_digest
from tools.reference_index.bundle import (
    AssemblyRefused,
    DerivationRule,
    assemble_prepared_roles,
)
from tools.reference_index.fit import FileDigest, FitReport

from .cache import ReferenceIndexCache
from .contracts import StoredBundle, ToolIdentity

FETCHED = ("genomic_fasta", "bwa_amb", "bwa_ann", "bwa_bwt", "bwa_pac", "bwa_sa")
"""The roles the bundle digest covers: the fetched FASTA and five BWA files."""
NAMED = frozenset({"genomic_fasta", "fasta_fai", "bwa_ann"})
"""The roles whose files name the contigs: rewritten when the fit renames them."""


class StoreRefused(Exception):
    """The bundle cannot be stored; nothing is left in the store."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def bundle_digest(report: FitReport) -> str:
    """The fetched content's digest; a renamed bundle's also covers its renames, as
    the stored FASTA and `.ann` then carry the requested names."""
    digests = {file.role: file.sha256 for file in report.candidate_files}
    material: dict = {role: digests[role] for role in FETCHED}
    if report.renames():
        material["renames"] = report.renames()
    return canonical_json_digest(material)


def bundle_folder(cache: ReferenceIndexCache, digest: str) -> Path:
    return cache.root / "store" / "sha256" / digest


def load_bundle(cache: ReferenceIndexCache, digest: str) -> StoredBundle:
    return StoredBundle.model_validate_json(
        (bundle_folder(cache, digest) / "bundle.json").read_bytes()
    )


def store_bundle(
    cache: ReferenceIndexCache, report: FitReport, fasta: Path
) -> StoredBundle:
    """Store the fitted candidate whose FASTA is `fasta` (its `.fai` and index files
    beside it) as the product assembles a pulled prebuilt."""
    if report.verdict == "refused":
        raise StoreRefused("fit_refused", report.candidate_id)
    digest = bundle_digest(report)
    folder = bundle_folder(cache, digest)
    if (folder / "bundle.json").exists():
        return load_bundle(cache, digest)
    if folder.exists():
        raise StoreRefused("store_incomplete", str(folder))
    try:
        assembled = assemble_prepared_roles(report, fasta, folder)
    except AssemblyRefused as error:
        raise StoreRefused("assembly_refused", str(error)) from error
    try:
        stored = StoredBundle(
            bundle_digest=digest,
            files=_digests(assembled.roles, report, assembled.derivation_rule),
            made_with=_samtools(),
        )
        with (folder / "bundle.json").open("x") as target:
            target.write(canonical_json(stored.model_dump(mode="json")))
    except BaseException:
        shutil.rmtree(folder)
        raise
    return stored


def _digests(
    roles: dict[str, Path], report: FitReport, rule: DerivationRule
) -> tuple[FileDigest, ...]:
    """Each stored file's digest: the fit report's for a file placed as fetched, under
    its prepared name; a file made or renamed here is hashed."""
    reported = {file.role: file for file in report.candidate_files}
    return tuple(
        reported[role].model_copy(update={"name": path.name})
        if role in reported and (rule == "as_published" or role not in NAMED)
        else _digest(role, path)
        for role, path in roles.items()
    )


def _samtools() -> ToolIdentity:
    """pysam's compiled samtools, which made the `.fai` and `.dict`."""
    library = Path(libcsamtools.__file__)
    return ToolIdentity(
        executable=str(library),
        sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        version_line=f"samtools {pysam.__samtools_version__} (pysam {pysam.__version__})",
    )


def _digest(role: str, path: Path) -> FileDigest:
    content = path.read_bytes()
    return FileDigest(
        role=role,
        name=path.name,
        byte_count=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
