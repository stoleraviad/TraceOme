"""The known targets checked on another assembly before the researcher chooses it (#459).

When Galaxy has no fitting index of the requested assembly X but has one of X's newer
version Y, the planning model suggests Y (`suggest_assembly_with_index`). Before the
researcher answers, each target the Run knows then (the prompt's, a request file's, a
paper planning read) goes through the Investigation's own lookup against Y
(`investigation_sources.look_up_stated_position`, `normalize_refsnp`): NCBI names the
stated sequence, the Variation service gives its one equivalent on Y, and Y's own bases
check its REF (owner, #459 D3, 2026-10-30: "we can try to move the SNV resolve to the
inial run"). The same code runs again in the Investigation for a target that arrives
later (D5); nothing here is a second placement path.

A position that states no version is stated on X, the assembly the researcher asked
for, so it is moved rather than read on Y as written. A moved position keeps its move
as its placement (`RequestPlacement`, step ``database_equivalence``); every other result
is shown with its reason and left to the Investigation, where a dead end opens the
item's issue Review.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import Field, StrictStr

from bio_neb.contracts import FrozenContract
from bio_neb.investigation.investigation_request import (
    InvestigationRequestItem,
    PlacedPosition,
    RequestPlacement,
    StatedAssembly,
    StatedPosition,
)

SuggestedOutcome = Literal["moved", "looked_up", "not_placed", "not_checked"]


class SuggestedTarget(FrozenContract):
    """One known target as the lookup against the suggested assembly answered it."""

    request_text: StrictStr
    rsid: StrictStr | None = Field(default=None, exclude_if=lambda value: value is None)
    position: StatedPosition | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    stated_assembly: StatedAssembly | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The version the lookup read the position on: its own, else the requested
    assembly's (D3)."""
    outcome: SuggestedOutcome
    status: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The lookup's status, copied (`investigation.RefSnpStatus`)."""
    placed: PlacedPosition | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """Where the target is on the suggested assembly, when the lookup placed it."""
    reason: StrictStr = Field(min_length=1, max_length=1500)

    @property
    def placement(self) -> RequestPlacement | None:
        """The placement a moved position keeps once the researcher chooses the
        suggested assembly; None for every other outcome."""
        if self.outcome != "moved":
            return None
        return RequestPlacement(
            step="database_equivalence", position=self.placed, reason=self.reason
        )

    def answers(self, item: InvestigationRequestItem) -> bool:
        """Whether this is ``item``'s result: the same words and typed target."""
        return (
            self.request_text == item.request_text
            and self.rsid == item.rsid
            and self.position == item.position
            and item.stated_assembly in {None, self.stated_assembly}
        )


Fetch = Callable[[str, str], tuple]
"""``fetch(operation, key)``: one retained GET (`investigation_sources.SourceFetch`)."""


def check_known_targets(
    items: Sequence[InvestigationRequestItem],
    *,
    requested: StatedAssembly,
    requested_names: frozenset[str],
    requested_accession: str,
    assembly_accession: str,
    fetch: Fetch,
    source: str,
) -> tuple[SuggestedTarget, ...]:
    """Each known target checked on ``assembly_accession`` with the Investigation's
    lookup, on the variant source the model named (``ncbi_dbsnp`` or ``eva``; #469,
    ADR 0049: code routes no target to a source). ``requested`` is the requested
    assembly as an unversioned position's stated version; a stated version literally
    one of ``requested_names`` is the requested assembly (its accession, name or
    synonym, copied from NCBI). A result that places nothing is shown with its source
    and read again in the Investigation (D5)."""
    from bio_neb.investigation.investigation import (
        ADMISSIBLE_LOOKUP_STATUSES,
        VersionNaming,
    )
    from bio_neb.investigation.investigation_sources import (
        look_up_eva_rsid,
        look_up_stated_position,
        normalize_refsnp,
    )

    checked = []
    for item in items:
        base = {"request_text": item.request_text, "rsid": item.rsid}
        if item.position is not None:
            stated = item.stated_assembly or requested
            version = (
                VersionNaming(
                    written=stated.name,
                    assembly_accession=requested_accession,
                    quote=None,
                    match="literal",
                )
                if stated.name in requested_names
                else None
            )
            result = look_up_stated_position(
                position=item.position,
                stated_assembly=stated,
                rsid=item.rsid,
                requested_rsid=item.rsid,
                assembly_accession=assembly_accession,
                fetch=fetch,
                # No model reads at planning: a written name is taken only literally.
                literal=True,
                version=version,
                source=source,
            )
            placed = (
                result.location.placed if result.location != "not_applicable" else None
            )
            moved = placed is not None and result.status in ADMISSIBLE_LOOKUP_STATUSES
            checked.append(
                SuggestedTarget(
                    **base,
                    position=item.position,
                    stated_assembly=stated,
                    outcome="moved" if moved else "not_placed",
                    status=result.status,
                    placed=placed if moved else None,
                    reason=_reason(result, moved, stated.name, assembly_accession),
                )
            )
        elif item.rsid is not None and item.region is None and item.gene_change is None:
            if source == "eva":
                result = look_up_eva_rsid(
                    rsid=item.rsid,
                    assembly_accession=assembly_accession,
                    fetch=fetch,
                    requested_rsid=item.rsid,
                )
            else:
                record, body = fetch("fetch_refsnp", item.rsid)
                result = normalize_refsnp(
                    rsid=item.rsid,
                    assembly_accession=assembly_accession,
                    source=record,
                    raw_response=body,
                    requested_rsid=item.rsid,
                )
            found = result.status in ADMISSIBLE_LOOKUP_STATUSES and result.candidates
            checked.append(
                SuggestedTarget(
                    **base,
                    outcome="looked_up" if found else "not_placed",
                    status=result.status,
                    placed=PlacedPosition(
                        contig=result.candidates[0].contig,
                        position=result.candidates[0].one_based_position,
                        ref=result.candidates[0].ref,
                        alt=result.candidates[0].alt,
                    )
                    if found
                    else None,
                    reason=(
                        f"{source} places {item.rsid} on {assembly_accession}."
                        if found
                        else _bounded(
                            f"{source}'s {item.rsid} on {assembly_accession}: "
                            + ("; ".join(result.conflicts) or result.status)
                        )
                    ),
                )
            )
        else:
            checked.append(
                SuggestedTarget(
                    **base,
                    outcome="not_checked",
                    reason="A region or gene change is placed in the Investigation "
                    "on the Run's reference.",
                )
            )
    return tuple(checked)


def _reason(result, moved: bool, stated: str, assembly: str) -> str:
    if moved:
        placed = result.location.placed
        return _bounded(
            f"NCBI's Variation service moves the position stated on {stated} to "
            f"{placed.contig}:{placed.position} on {assembly}, where {assembly}'s "
            f"reference holds {placed.ref}."
        )
    return _bounded(
        f"Not placed on {assembly} ({result.status}): "
        + ("; ".join(result.conflicts) or "the lookup ended before a placement")
    )


def _bounded(text: str) -> str:
    return text if len(text) <= 1500 else text[:1499].rstrip() + "…"


__all__ = [
    "SuggestedOutcome",
    "SuggestedTarget",
    "check_known_targets",
]
