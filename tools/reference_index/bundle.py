"""A fitted prebuilt index as the Run's eight prepared roles.

The candidate's FASTA and five BWA files are placed under the prepared-role names, its
`.fai` with them, and a `.dict` is made with samtools (pysam's, as the Run's observation
tools are). When the fit paired the candidate's contigs with NCBI's under other names
(`fits_after_rename`), the FASTA headers and the `.ann` names are rewritten once to
NCBI's names and the `.fai` is made again (owner Q6 a, #365); the `.bwt`, `.sa`, `.pac`
and `.amb` hold no names and are placed as published. The result is read back once
through the one BWA-index reader and must state the fit's pairing, in the candidate's
own order (its BWA offsets depend on it), or nothing is kept; that reading is the
assembled roles' proof, which admission reuses (ADR 0029).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pysam

from bio_neb.parabricks.bwa_index import (
    BwaIndexReading,
    fasta_record_name,
    read_ann,
    read_bwa_index,
)
from bio_neb.parabricks.worker_exchange import PREPARED_ROLES
from tools.reference_index.fit import BUNDLE_SUFFIXES, FitReport

DerivationRule = Literal["as_published", "renamed_to_requested_names"]
PREPARED_NAMES = {
    "genomic_fasta": "reference.fa",
    "fasta_fai": "reference.fa.fai",
    "reference_dict": "reference.dict",
    "bwa_amb": "reference.fa.amb",
    "bwa_ann": "reference.fa.ann",
    "bwa_bwt": "reference.fa.bwt",
    "bwa_pac": "reference.fa.pac",
    "bwa_sa": "reference.fa.sa",
}
_NAMELESS = ("bwa_amb", "bwa_bwt", "bwa_pac", "bwa_sa")


class AssemblyRefused(ValueError):
    """The assembled roles do not state the contigs the fit matched."""


@dataclass(frozen=True)
class AssembledRoles:
    roles: dict[str, Path]
    """Each prepared role's file, in `PREPARED_ROLES` order."""
    derivation_rule: DerivationRule
    reading: BwaIndexReading
    """The one reader's reading of the assembled roles."""


def assemble_prepared_roles(
    report: FitReport, candidate: Path, folder: Path
) -> AssembledRoles:
    """The eight prepared roles of a candidate that fits, written into a new `folder`."""
    if report.verdict == "refused":
        raise ValueError(f"{report.candidate_id}: its fit is refused")
    folder.mkdir(parents=True)
    try:
        roles = {role: folder / name for role, name in PREPARED_NAMES.items()}
        renames = report.renames()
        for role in _NAMELESS:
            _place(Path(f"{candidate}{BUNDLE_SUFFIXES[role]}"), roles[role])
        if renames:
            _renamed_fasta(candidate, roles["genomic_fasta"], renames)
            _renamed_ann(
                Path(f"{candidate}{BUNDLE_SUFFIXES['bwa_ann']}"),
                roles["bwa_ann"],
                renames,
            )
            pysam.faidx(str(roles["genomic_fasta"]))
        else:
            _place(candidate, roles["genomic_fasta"])
            _place(Path(f"{candidate}.fai"), roles["fasta_fai"])
            _place(Path(f"{candidate}{BUNDLE_SUFFIXES['bwa_ann']}"), roles["bwa_ann"])
        roles["reference_dict"].write_text(pysam.dict(str(roles["genomic_fasta"])))
        reading = _states_the_fit(roles, report)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return AssembledRoles(
        roles={role: roles[role] for role in PREPARED_ROLES},
        derivation_rule="renamed_to_requested_names" if renames else "as_published",
        reading=reading,
    )


def _place(source: Path, target: Path) -> None:
    """The published file itself, linked when the filesystem allows."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _renamed_fasta(source: Path, target: Path, renames: dict[str, str]) -> None:
    """The FASTA with each renamed contig's header line replaced by its new name alone."""
    with source.open("rb") as reading, target.open("wb") as writing:
        for line in reading:
            if line.startswith(b">"):
                name = fasta_record_name(line)
                if name in renames:
                    line = f">{renames[name]}\n".encode()
            writing.write(line)


def _renamed_ann(source: Path, target: Path, renames: dict[str, str]) -> None:
    """The `.ann` with each descriptor line's name (`gi name annotation`) replaced;
    its structure is the one reader's (`read_ann`)."""
    data = source.read_bytes()
    read_ann(data)
    lines = data.decode().splitlines(keepends=True)
    for index in range(1, len(lines), 2):
        gi, name, *annotation = lines[index].rstrip("\n").split(" ", 2)
        lines[index] = " ".join([gi, renames.get(name, name), *annotation]) + "\n"
    target.write_text("".join(lines))


def _states_the_fit(roles: dict[str, Path], report: FitReport) -> BwaIndexReading:
    reading = read_bwa_index(
        roles["genomic_fasta"],
        {role: path for role, path in roles.items() if role != "genomic_fasta"},
    )
    if reading.findings or not states_the_fit(reading, report):
        raise AssemblyRefused(
            f"{report.candidate_id}: the assembled roles do not state the fit's contigs"
        )
    return reading


def states_the_fit(reading: BwaIndexReading, report: FitReport) -> bool:
    """Whether a reading holds exactly the fit's contigs, by name, length and M5, in
    any order: each requested contig under its requested name, and each extra the
    Review bound (#464) as the candidate holds it."""
    read = [(c.name, c.length, c.m5) for c in reading.contigs]
    wanted = [
        (c.candidate_name, c.candidate_length, c.candidate_m5)
        if c.status == "extra_in_candidate"
        else (c.requested_name, c.requested_length, c.requested_m5)
        for c in report.contigs
    ]
    return sorted(read) == sorted(wanted)
