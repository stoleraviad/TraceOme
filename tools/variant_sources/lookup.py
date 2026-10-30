"""One lookup entry for any caller: a variant claim in, the source's answer out.

The caller names the source, as the product's model chooses it (#469: no code routes by
assembly; this tool has no model). An rs claim sent to dbSNP goes through the product's
Investigation source seam, following a merged rsID to its current one as the product's Agent
does; one sent to EVA goes to the product's EVA lookup. A claim with a position goes through
the product's own position lookup, `look_up_stated_position` (#369), from the named source: it names the stated sequence on the
claim's assembly by literal membership (#467: this tool has no model), reads the reference bases there and the variants the source lists (#393 Q10).
An HGVS-only claim is answered `not_checked` with the reason. A source contact the failure
module gave up on (`failures.py`) is the claim's answer too.

Every GET goes through the product's one source-fetch seam (`eva_investigation.fetch_source`)
on its operation's surface, over the product's transports; this module keeps no source logic.
"""

from __future__ import annotations

from bio_neb.investigation.investigation import RefSnpToolResult, RetrievedSource
from bio_neb.investigation.investigation_request import StatedAssembly, StatedPosition
from bio_neb.investigation.investigation_sources import (
    look_up_eva_rsid,
    look_up_stated_position,
    normalize_refsnp,
)
from bio_neb.providers.eva_investigation import (
    EvaInvestigationSourceAdapter,
    PublicInvestigationSources,
    UrllibEvaTransport,
    fetch_source,
)
from bio_neb.providers.ncbi_datasets import UrllibNcbiTransport
from bio_neb.providers.ncbi_investigation import NcbiInvestigationSourceAdapter
from bio_neb.providers.source_transport import Opener
from bio_neb.target_request.source_failure import SourceFailed, through_failure_module

from .contracts import ClaimAnswer, Route, VariantClaim


class VariantSources:
    """The product's public variant sources and the one `fetch` the product's lookups
    take: each GET on its own operation's failure surface."""

    def __init__(self, sources: PublicInvestigationSources) -> None:
        self.sources = sources

    @classmethod
    def public(cls, opener: Opener | None = None) -> VariantSources:
        """NCBI and EVA over the product's transports; a test supplies the opener."""
        return cls(
            PublicInvestigationSources(
                ncbi=NcbiInvestigationSourceAdapter(UrllibNcbiTransport(opener=opener)),
                eva=EvaInvestigationSourceAdapter(UrllibEvaTransport(opener=opener)),
            )
        )

    @property
    def contact_count(self) -> int:
        return self.sources.contact_count

    def fetch(self, operation: str, key: str) -> tuple[RetrievedSource, bytes]:
        """One GET as the product's lookups ask for it (`SourceFetch`)."""
        response = through_failure_module(
            operation, lambda: fetch_source(self.sources, operation, key)
        )
        source = RetrievedSource.create(
            operation=operation,
            request_identity=response.request_identity,
            status_code=response.status_code,
            observed_at=response.observed_at,
            elapsed_time_ms=response.elapsed_time_ms,
            raw_response=response.body,
        )
        return source, response.body

    def refsnp(
        self, rsid: str, assembly_accession: str, requested_rsid: str | None = None
    ) -> RefSnpToolResult:
        """One `fetch_refsnp` and its normalization, as the product's Tool makes them;
        `requested_rsid` is the claim's rsID when this lookup follows its merge."""
        source, body = self.fetch("fetch_refsnp", rsid)
        return normalize_refsnp(
            rsid=rsid,
            assembly_accession=assembly_accession,
            source=source,
            raw_response=body,
            requested_variant=None,
            requested_rsid=requested_rsid,
        )

    def answer(self, claim: VariantClaim, source: Route) -> ClaimAnswer:
        """The claim's answer from the source the caller names (#469)."""
        route = source
        reason = f"the caller named {'NCBI dbSNP' if source == 'dbsnp' else 'EVA'}"
        answer = {"claim": claim, "route": route, "route_reason": reason}
        if claim.rsid is None and claim.position is None:
            return ClaimAnswer(
                **answer, not_checked="an HGVS name only: no source here reads HGVS"
            )
        try:
            found = self._source_answer(claim, route)
        except SourceFailed as failed:
            found = {"failure": failed.failure}
        return ClaimAnswer(**answer, **found)

    def _source_answer(self, claim: VariantClaim, route: Route) -> dict:
        if claim.position is not None:
            # The claim's coordinates are on its own assembly: that is the version it
            # states, in the words that state it.
            stated = StatedAssembly(
                name=claim.assembly_accession,
                cited_line=claim.request_text
                or f"the claim states its position on {claim.assembly_accession}",
            )
            return {
                "position": look_up_stated_position(
                    position=StatedPosition(
                        contig=claim.contig,
                        position=claim.position,
                        ref=claim.ref,
                        alt=claim.alt,
                    ),
                    stated_assembly=stated,
                    rsid=claim.rsid,
                    requested_rsid=claim.rsid,
                    assembly_accession=claim.assembly_accession,
                    fetch=self.fetch,
                    # This tool has no model: a sequence is named only when its
                    # written contig is literally one of its names.
                    literal=True,
                    source="ncbi_dbsnp" if route == "dbsnp" else "eva",
                )
            }
        if route == "dbsnp":
            return self._dbsnp_answer(claim)
        return {
            "eva": look_up_eva_rsid(
                rsid=claim.rsid,
                assembly_accession=claim.assembly_accession,
                fetch=self.fetch,
                requested_variant=None,
            )
        }

    def _dbsnp_answer(self, claim: VariantClaim) -> dict:
        """The claim's rsID on dbSNP; a record merged into one current rsID is followed
        there, with the claim's rsID kept as the requested one (`investigation_planner` v5).
        A merge into several is the answer: which one the claim means is the caller's."""
        found = self.refsnp(claim.rsid, claim.assembly_accession)
        if found.status != "merged" or len(found.merged_into) != 1:
            return {"dbsnp": found}
        try:
            current = self.refsnp(
                found.merged_into[0], claim.assembly_accession, claim.rsid
            )
        except SourceFailed as failed:
            return {"merge": found, "failure": failed.failure}
        return {"merge": found, "dbsnp": current}
