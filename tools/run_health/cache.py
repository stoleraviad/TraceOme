"""Copy a Run's reference FASTA and `.fai` once into the lane's local cache.

The Run's BAM and VCF are not cached: its Investigation keeps them local beside its
Journey (#363), and the health tools read them there (`health.read_run`). The reference
is read with main's exact object read in the Run's own bucket and context, its identity
taken from the Run's verified execution manifest. `cache-manifest.json` records each
file's object key, byte count and sha256, written last.

A resumed cache keeps each file an earlier attempt finished: it rereads each kept file in
full against the Run's identity and reads only the files that are absent. A failed read
removes its own file (main's object reads, #363), so no operator step precedes a resume.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from bio_neb.application.artifact_materialization import (
    MaterializationFiles,
    MaterializedFile,
    ObjectMaterializationRefused,
    execution_object_storage,
)
from bio_neb.content import canonical_json
from bio_neb.journey_path import artifact_location
from bio_neb.parabricks.h100_recipe import H100ExecutionManifest
from bio_neb.parabricks.result import (
    ParabricksExecutionIncomplete,
    ParabricksExecutionOutcome,
    ParabricksResultUnavailable,
    read_completed_parabricks_result,
)
from bio_neb.providers.nebius_object_storage import NebiusObjectStorageError
from bio_neb.verified_artifacts import VerifiedObject

from .runner import RunRefused
from .table import StrictModel

CACHE_SCHEMA = "bio-neb.run-health-cache/2"
MANIFEST_NAME = "cache-manifest.json"
_REFERENCE = {"reference_fasta": "genomic_fasta", "reference_fai": "fasta_fai"}

CachedRole = Literal["reference_fasta", "reference_fai"]


class CachedFile(StrictModel):
    """One cached file. `kept` is true when an earlier attempt copied it and this one
    reread it in full against the Run's identity."""

    path: str
    object_key: str
    byte_count: int
    sha256: str
    kept: bool


class RunCache(StrictModel):
    schema_version: Literal["bio-neb.run-health-cache/2"] = CACHE_SCHEMA
    journey_id: str
    journey_file: str
    bucket: str
    files: dict[CachedRole, CachedFile]


_files = MaterializationFiles(
    MaterializedFile,
    lambda reason, *details: RunRefused(reason, " ".join(details)),
)


def cache_run_files(
    journey_file: Path,
    cache_dir: Path,
    *,
    object_read_factory: Callable[[Any], Any] | None = None,
    resume: bool = False,
) -> RunCache:
    """Copy the Run's reference FASTA and `.fai` into `cache_dir/reference/` and write
    its manifest.

    Without `resume` the directory must not exist; with it, the directory holds an
    earlier attempt and no manifest yet. Fail-stop: the first refusal (`RunRefused`)
    stops the copy, and no manifest is written. `object_read_factory` stands in for main's object
    storage in tests.
    """
    journey = str(journey_file.absolute())
    cache_dir = cache_dir.absolute()
    if resume:
        if not cache_dir.is_dir():
            raise RunRefused("cache_missing", f"{cache_dir} holds no earlier attempt")
        if (cache_dir / MANIFEST_NAME).exists():
            raise RunRefused("cache_complete", f"{cache_dir} has its manifest")
    elif cache_dir.exists() or cache_dir.is_symlink():
        raise RunRefused("cache_exists", f"{cache_dir} exists; a cache is made once")
    outcome = read_outcome(Path(journey))
    reference = _reference_objects(outcome)
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=resume)
    directory = cache_dir / "reference"
    if not directory.exists():
        _files.fresh_directory(directory)
    kept = {
        name: _kept(directory / Path(descriptor.object_key).name, descriptor)
        for name, descriptor in reference.items()
        if (directory / Path(descriptor.object_key).name).exists()
    }
    absent = {name: item for name, item in reference.items() if name not in kept}
    try:
        read = _reference(outcome, directory, absent, object_read_factory)
    except ObjectMaterializationRefused as refused:
        raise RunRefused(refused.code, " ".join(refused.details)) from refused
    cache = RunCache(
        journey_id=outcome.journey_id,
        journey_file=outcome.journey_file,
        bucket=outcome.storage.bucket,
        files={name: (kept | read)[name] for name in reference},
    )
    payload = canonical_json(cache.model_dump(mode="json")).encode("utf-8")
    with _files.exclusive_output(cache_dir / MANIFEST_NAME) as stream:
        stream.write(payload)
    os.chmod(cache_dir / MANIFEST_NAME, 0o444)
    return cache


def read_outcome(journey_file: Path) -> ParabricksExecutionOutcome:
    """The Run's verified Execution result. A Journey that cannot be read
    (`run_unreadable`) and a Run without a completed Execution (`outcome_unavailable`)
    are refused by name."""
    try:
        return read_completed_parabricks_result(str(journey_file))
    except (ParabricksResultUnavailable, ParabricksExecutionIncomplete) as error:
        raise RunRefused("outcome_unavailable", f"{journey_file}: {error}") from error
    except (OSError, ValueError) as error:
        raise RunRefused("run_unreadable", f"{journey_file}: {error}") from error


def read_cache(cache_dir: Path, outcome: ParabricksExecutionOutcome) -> RunCache:
    """The Run's cache manifest; a missing manifest or another Run's is refused."""
    manifest = cache_dir / MANIFEST_NAME
    try:
        cache = RunCache.model_validate_json(manifest.read_bytes())
    except FileNotFoundError:
        raise RunRefused("file_missing", f"no cache manifest at {manifest}") from None
    except (OSError, ValueError) as error:
        raise RunRefused(
            "cache_unreadable",
            f"{manifest} is not a readable run-health cache: {error}",
        ) from error
    if cache.journey_id != outcome.journey_id:
        raise RunRefused(
            "cache_other_run",
            f"{manifest} is Run {cache.journey_id}'s; this Run is {outcome.journey_id}",
        )
    return cache


def _reference_objects(outcome: Any) -> dict[str, VerifiedObject]:
    """The prepared reference FASTA and `.fai` the Run's Execution read.

    Their identities come from the verified execution manifest, which the Run keeps as
    a content-addressed Journey artifact; the FASTA's must equal the Run's reference
    result.
    """
    (manifest_object,) = (
        item for item in outcome.verified_objects if item.role == "execution_manifest"
    )
    manifest_file = Path(outcome.journey_file).parent / artifact_location(
        f"sha256:{manifest_object.sha256}"
    )
    observed, content = _files.stable_full_read(
        manifest_file, require_read_only=False, capture=True
    )
    if (observed.byte_count, observed.sha256) != (
        manifest_object.byte_count,
        manifest_object.sha256,
    ):
        raise RunRefused("manifest_identity_mismatch", str(manifest_file))
    manifest = H100ExecutionManifest.model_validate_json(content)
    inputs = {item.role: item for item in manifest.inputs}
    fasta = inputs["genomic_fasta"]
    if (fasta.byte_count, fasta.sha256) != (
        outcome.reference.plain_fasta_byte_count,
        outcome.reference.plain_fasta_sha256,
    ):
        raise RunRefused("reference_identity_mismatch", fasta.object_key)
    return {
        name: VerifiedObject(
            role=role,
            bucket=outcome.storage.bucket,
            object_key=inputs[role].object_key,
            byte_count=inputs[role].byte_count,
            sha256=inputs[role].sha256,
        )
        for name, role in _REFERENCE.items()
    }


def _reference(
    outcome: Any,
    directory: Path,
    objects: dict[str, VerifiedObject],
    object_read_factory: Callable[[Any], Any] | None,
) -> dict[str, CachedFile]:
    """One exact read per absent reference object into `directory`; its identity is
    the transfer's own checked one (the replica is sealed, not reread)."""
    if not objects:
        return {}
    source = (
        object_read_factory(outcome)
        if object_read_factory is not None
        else execution_object_storage(
            outcome, RunRefused("context_missing", outcome.journey_file)
        )
    )
    cached = {}
    for name, descriptor in objects.items():
        path = directory / Path(descriptor.object_key).name
        try:
            read = _files.read_object(path, source, descriptor)
        except NebiusObjectStorageError as error:
            raise RunRefused(
                "reference_read_failed",
                f"{descriptor.object_key}: {error.kind}: {error.detail}",
            ) from error
        if (read.byte_count, read.sha256) != (descriptor.byte_count, descriptor.sha256):
            raise RunRefused(
                "reference_read_mismatch",
                f"{descriptor.object_key}: read {read.byte_count} B "
                f"sha256 {read.sha256}",
            )
        _files.seal_replica(path)
        cached[name] = CachedFile(
            path=str(path),
            object_key=descriptor.object_key,
            byte_count=descriptor.byte_count,
            sha256=descriptor.sha256,
            kept=False,
        )
    return cached


def _kept(path: Path, descriptor: VerifiedObject) -> CachedFile:
    """Reread one sealed file an earlier attempt copied; it must be the Run's object."""
    observed, _ = _files.stable_full_read(path, require_read_only=True)
    if (observed.byte_count, observed.sha256) != (
        descriptor.byte_count,
        descriptor.sha256,
    ):
        raise RunRefused(
            "kept_file_mismatch",
            f"{path}: {observed.byte_count} B sha256 {observed.sha256}, the Run's "
            f"{descriptor.object_key} is {descriptor.byte_count} B sha256 "
            f"{descriptor.sha256}",
        )
    return CachedFile(
        path=str(path),
        object_key=descriptor.object_key,
        byte_count=observed.byte_count,
        sha256=observed.sha256,
        kept=True,
    )
