"""The placement port's flank move over the reference-index tool (#367 readiness Q6).

`StoredMove` answers the product's development-only `FlankMover` port (no Run passes a mover;
#449) with #332's move of one SNV onto the
Run's stored index (the bundle a choice record names for it), from a stated version's sequence
already in the reference-index cache, where `fetch-sequence` keeps it. The move's own record is
kept under ``moves/``. No rsID is looked up (#385 Q4).
"""

from __future__ import annotations

from pathlib import Path

from bio_neb.target_request.records import FlankMove, FlankMoveRequest, FlankPlace
from bio_neb.target_request.resolver import FlankMoveRefused
from tools.variant_sources.lookup import VariantSources

from .cache import ReferenceIndexCache
from .contracts import SnvToMove
from .local_tools import LocalTools
from .move import MoveRefused, move


class StoredMove:
    """The bwa flank move behind the product's `FlankMover` port."""

    def __init__(
        self, cache: ReferenceIndexCache, tools: LocalTools, lookups: VariantSources
    ) -> None:
        self.cache = cache
        self.tools = tools
        self.lookups = lookups

    def sequence(self, accession: str) -> Path | None:
        path = self.cache.root / "ncbi" / "sequences" / f"{accession}.fa"
        return path if path.exists() else None

    def missing(self, accession: str) -> str:
        return (
            "which is not in the reference-index cache: python -m "
            f"tools.reference_index fetch-sequence {accession}"
        )

    def __call__(
        self, snv: FlankMoveRequest, source: Path, run_assembly: str
    ) -> FlankMove:
        try:
            record = move(
                self.cache,
                self.tools,
                run_assembly,
                source_assembly=snv.assembly,
                source_fasta=source,
                snvs=(
                    SnvToMove(
                        label=snv.label,
                        assembly=snv.assembly,
                        contig=snv.sequence,
                        position=snv.position,
                        ref=snv.ref,
                        alt=snv.alt,
                    ),
                ),
                rsids={},
                lookups=self.lookups,
            )
        except MoveRefused as refused:
            raise FlankMoveRefused(refused.reason, refused.detail) from refused
        moved = record.moves[0]
        return FlankMove(
            result=moved.result,
            place=None
            if moved.place is None
            else FlankPlace(contig=moved.place.contig, position=moved.place.position),
            target_ref=moved.target_ref,
            target_alt=moved.target_alt,
            reason=moved.reason,
        )
