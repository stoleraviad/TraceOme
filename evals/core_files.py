"""The eval core-file ledger: the reads and reference each scenario's runs use, kept for reuse.

The prep worker downloads a scenario's public FASTQs and reference FASTA once, checks them
against the publisher's checksum, and records them here. Runs take their inputs from this
ledger instead of a copied runtime, so every run of a scenario registers the same bytes. The
files live under the local state root (never in the repository); `path` is relative to it.
No prepared index is recorded: the product builds it. `record` builds a row after the
download's md5 matches the publisher's.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

LEDGER = Path(__file__).resolve().parent / "core-files.yaml"
_SHA256 = r"^[0-9a-f]{64}$"


class CoreFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9-]+\.[a-z0-9_]+$")
    scenario: str
    kind: Literal["reference_fasta", "fastq"]
    role: str | None = None
    assembly_accession: str | None = None
    reference_asset_id: str | None = None
    source_url: str = Field(pattern=r"^https://")
    source_checksum: str = Field(
        description="The publisher's checksum, as `md5:<hex>`."
    )
    source_checksum_locator: str
    path: str = Field(description="Relative to the local state root.")
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256)
    plain_bytes: int | None = Field(default=None, gt=0)
    plain_sha256: str | None = Field(default=None, pattern=_SHA256)
    retrieved_at: str
    retrieved_by: str

    @model_validator(mode="after")
    def shape_by_kind(self) -> CoreFile:
        reference = (
            self.assembly_accession,
            self.reference_asset_id,
            self.plain_bytes,
            self.plain_sha256,
        )
        if self.kind == "reference_fasta" and (self.role or (None in reference)):
            raise ValueError("a reference needs accession, asset id and plain identity")
        if self.kind == "fastq" and (not self.role or any(reference)):
            raise ValueError("a FASTQ needs a role and no reference fields")
        if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ValueError("path must be relative to the local state root")
        return self


class ScenarioCoreFiles(BaseModel):
    model_config = ConfigDict(frozen=True)

    reference: CoreFile
    reads: tuple[CoreFile, ...]


def load_ledger(path: Path = LEDGER) -> tuple[CoreFile, ...]:
    rows = tuple(
        CoreFile.model_validate(row) for row in yaml.safe_load(path.read_text())
    )
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("core-file ids must be unique")
    return rows


def scenario_files(rows: tuple[CoreFile, ...], scenario: str) -> ScenarioCoreFiles:
    """One reference and the reads of a scenario; refuses a scenario without them."""
    mine = [row for row in rows if row.scenario == scenario]
    references = [row for row in mine if row.kind == "reference_fasta"]
    reads = tuple(row for row in mine if row.kind == "fastq")
    if len(references) != 1 or not reads:
        raise ValueError(f"scenario {scenario} has no core reference and reads")
    return ScenarioCoreFiles(reference=references[0], reads=reads)


def _digest(path: Path, *, gunzip: bool = False) -> tuple[int, str]:
    digest, size = hashlib.sha256(), 0
    opener = gzip.open if gunzip else open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def record(local_state: Path, path: str, **fields) -> CoreFile:
    """A ledger row for one downloaded file, after its publisher's md5 matches."""
    absolute = local_state / path
    md5 = hashlib.md5()
    with absolute.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            md5.update(chunk)
    if fields["source_checksum"] != f"md5:{md5.hexdigest()}":
        raise ValueError(f"{path} does not match its publisher's checksum")
    size, sha256 = _digest(absolute)
    if fields["kind"] == "reference_fasta":
        fields["plain_bytes"], fields["plain_sha256"] = _digest(absolute, gunzip=True)
    return CoreFile(path=path, bytes=size, sha256=sha256, **fields)
