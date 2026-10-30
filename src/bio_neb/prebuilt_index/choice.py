"""The prebuilt index the planning model chose, as the Execution Review binds it (#464).

The model searches Galaxy's catalogues and reads a candidate's contigs with the planning
Tools (`reference_discovery.reference_tools`); its membership-checked choice carries this
candidate into the proposal: the catalogue row, each of its seven files with the size a
HEAD reported, and its contigs paired with NCBI's copied rows (`contigs.pair_contigs`),
extras included. After approval execution streams the files into the Run's bucket,
each at its bound size and renamed to NCBI's names in flight, or falls to the CPU Review
(owner I7, #464; ADR 0049: code copies, the model decides).
"""

from __future__ import annotations

from pydantic import Field, StrictInt, StrictStr, model_validator

from bio_neb.contracts import FrozenContract
from bio_neb.prebuilt_index.catalogue import GalaxyCandidate, repository_of
from bio_neb.prebuilt_index.contigs import IndexContigs

GALAXY_BRC_BWA_VERSION = "0.7.17-r1188"
"""The bwa whose `bwa index` of NCBI's FASTA is byte-identical to Galaxy BRC's index
(`docs/development/reviews/issue365-prebuilt-index/spike/`). Galaxy's catalogues state
no version; a pulled BRC producer records this one and is not pinned (owner Q5). An
index of another repository states none."""


class PrebuiltFile(FrozenContract):
    role: StrictStr
    url: StrictStr
    byte_count: StrictInt = Field(gt=0)


class PrebuiltIndexCandidate(FrozenContract):
    """What the Execution Review binds: where each file is, its size, and the
    candidate's contigs beside the Run's sequences (extras with NCBI's role)."""

    candidate: GalaxyCandidate
    files: tuple[PrebuiltFile, ...]
    contigs: IndexContigs
    stated_bwa_version: StrictStr | None

    @model_validator(mode="after")
    def the_candidate_s_files(self) -> PrebuiltIndexCandidate:
        if [(f.role, f.url) for f in self.files] != list(
            self.candidate.files().items()
        ):
            raise ValueError("a prebuilt candidate states each of its files, in order")
        if not self.contigs.holds_every_requested_sequence:
            raise ValueError("a bound prebuilt index holds every requested sequence")
        return self


def stated_bwa_version(candidate: GalaxyCandidate) -> str | None:
    """The bwa a candidate's publisher is known to have built it with: BRC's spike
    version, else none."""
    return (
        GALAXY_BRC_BWA_VERSION if repository_of(candidate.catalogue) == "brc" else None
    )


__all__ = [
    "GALAXY_BRC_BWA_VERSION",
    "PrebuiltFile",
    "PrebuiltIndexCandidate",
    "stated_bwa_version",
]
