"""One of every report qualification's typed origin, with its kind, audit sentence
and plain words (#474), shared by the origin pin and the readable report's render
test; and the audit wording the readable body never shows (#460 LSP6).
"""

import re

from bio_neb.investigation import qualification_origins as origins

ITEM = "Work item work-item:aa (rs1 chr1:10 A>G)"
# (origin, kind, audit sentence for its subject, plain words), one per code and
# one per typed branch its words or audit read.
PINS = (
    (
        origins.RequestRejected(by="researcher:ana", reason="Not this rs1"),
        "rs1",
        "limitation",
        "Request rs1 was rejected by researcher:ana: Not this rs1.",
        'rejected by researcher:ana, who wrote: "Not this rs1"',
    ),
    (
        origins.RequestRejected(by=None, reason="Model budget spent"),
        "rs1",
        "limitation",
        "Request rs1 was rejected Model budget spent.",
        "stopped before it was checked; the VCF was not searched",
    ),
    (
        origins.RequestIssueRejected(
            by="researcher:ana", source="refsnp", verdict="not_found"
        ),
        "rs1",
        "limitation",
        "Request rs1 was rejected by researcher:ana: refsnp: not_found finding.",
        "the variant lookup could not place it on the Run's reference, and "
        "researcher:ana rejected it",
    ),
    (
        origins.RequestSetAside(code="request_removed", by="researcher:ana"),
        "rs1",
        "limitation",
        "Request rs1 was removed by researcher:ana.",
        "removed by researcher:ana; the VCF was not searched",
    ),
    (
        origins.RequestSetAside(code="request_deferred", by="researcher:ana"),
        "rs1",
        "limitation",
        "Request rs1 was deferred by researcher:ana.",
        "deferred by researcher:ana; the VCF was not searched",
    ),
    (
        origins.RequestNotChecked(reason_code="not_placed", reason="No contig named."),
        "rs1",
        "limitation",
        "Request rs1 was not checked: No contig named.",
        "no step could place it on the Run's reference as stated, so it was not "
        "checked",
    ),
    (
        origins.RequestNotChecked(
            reason_code="position_without_alleles",
            reason="it states 19:44908822 without its alleles or an rsID, and the "
            "Run's sources list a variant only at a stated allele, so it was not "
            "looked up",
        ),
        "rs1",
        "limitation",
        "Request rs1 was deferred out of the Run: it states 19:44908822 without its "
        "alleles or an rsID, and the Run's sources list a variant only at a stated "
        "allele, so it was not looked up.",
        "deferred out of the Run: it was stated without its alleles or an rsID, so "
        "no source was asked about it and the VCF was not searched; this does not "
        "say whether the sample carries a variant there",
    ),
    (
        origins.RequestUnresolved(
            source="refsnp", verdict="not_found", decision="approved"
        ),
        "rs1",
        "gap",
        "Request rs1 has no Work item: refsnp: not_found (approved, checked not "
        "verified).",
        "the variant lookup could not place it on the Run's reference; you accepted "
        "that finding as checked but not verified",
    ),
    (
        origins.RequestUnresolved(
            source="assembly_report", verdict="not_found", decision="unanswered"
        ),
        "rs1",
        "gap",
        "Request rs1 has no Work item: assembly_report: not_found, unanswered when "
        "the Run ended.",
        "the assembly report could not confirm its place on the Run's reference; "
        "its question was unanswered when the Run ended",
    ),
    (
        origins.RequestUnresolved(),
        "rs1",
        "gap",
        "Request rs1 has no Work item: unresolved.",
        "it was not placed on the Run's reference",
    ),
    (
        origins.RequestNotProposed(),
        "rs1",
        "gap",
        "Request rs1 was checked but not proposed: the Run ended before the "
        "collection was proposed.",
        "checked, but the Run ended before a VCF search was proposed",
    ),
    (
        origins.RequestNeverRun(),
        "rs1",
        "gap",
        "Request rs1 was never checked: its check did not finish before Finalization.",
        "its check did not finish before the Run ended",
    ),
    (
        origins.ItemApprovedNotObserved(),
        ITEM,
        "gap",
        f"{ITEM} was approved but not observed before Finalization.",
        "approved, but the Run ended before the VCF was searched here",
    ),
    (
        origins.ItemNotApproved(),
        ITEM,
        "gap",
        f"{ITEM} was proposed but not approved before Finalization.",
        "proposed, but not approved for a VCF search",
    ),
    (
        origins.ItemUnselected(),
        ITEM,
        "gap",
        f"{ITEM} is a candidate the approval did not select.",
        "another allele was chosen instead, so this one was not searched",
    ),
    (
        origins.ItemSetAside(code="item_removed", by="researcher:ana"),
        ITEM,
        "limitation",
        f"{ITEM} was removed by researcher:ana.",
        "removed by researcher:ana; the VCF was not searched here",
    ),
    (
        origins.ItemSetAside(code="item_deferred", by="researcher:ana"),
        ITEM,
        "limitation",
        f"{ITEM} was deferred by researcher:ana.",
        "deferred by researcher:ana; the VCF was not searched here",
    ),
    (
        origins.ItemFailed(reason="bcftools exited 1"),
        ITEM,
        "limitation",
        f"{ITEM} failed: bcftools exited 1.",
        "the VCF search failed here",
    ),
    (
        origins.CollectionItem(status="stopped", reason="researcher stop"),
        "work-item:aa",
        "limitation",
        "Collection Work item work-item:aa is stopped: researcher stop.",
        "the VCF search was stopped here",
    ),
    (
        origins.CollectionItem(status="failed", reason="tool error"),
        "work-item:aa",
        "limitation",
        "Collection Work item work-item:aa is failed: tool error.",
        "the VCF search failed here",
    ),
    (
        origins.CollectionItem(status="not_started"),
        "work-item:aa",
        "limitation",
        "Collection Work item work-item:aa is not_started.",
        "the VCF search here did not start",
    ),
    (
        origins.CollectionItem(status="approval_pending"),
        "work-item:aa",
        "limitation",
        "Collection Work item work-item:aa is approval_pending.",
        "its approval was still pending when the Run ended",
    ),
    (
        origins.CollectionItem(status="observation_pending"),
        "work-item:aa",
        "limitation",
        "Collection Work item work-item:aa is observation_pending.",
        "its VCF search was still pending when the Run ended",
    ),
    (
        origins.CoverageNotMeasured(),
        "work-item:aa",
        "gap",
        "Coverage was not measured for Work item work-item:aa.",
        "coverage was not measured here",
    ),
    (
        origins.CoverageNotMeasured(
            reason_code="contig_unknown", detail="chr1 not in the BAM"
        ),
        "work-item:aa",
        "gap",
        "Coverage was not measured for Work item work-item:aa: contig_unknown "
        "(chr1 not in the BAM).",
        "coverage was not measured here",
    ),
    (
        origins.DepthBelowMinimum(depth=4, minimum=10),
        "work-item:aa",
        "limitation",
        "Work item work-item:aa: read depth 4 at the least-covered base of its "
        "span, below the minimum of 10.",
        "read depth here was 4 at its lowest, below the minimum of 10",
    ),
    (
        origins.CoverageNotMeasured(
            reason_code="bam_unavailable", detail="the BAM download was aborted"
        ),
        "work-item:aa",
        "gap",
        "Coverage was not measured for Work item work-item:aa: bam_unavailable "
        "(the BAM download was aborted).",
        "coverage was not measured here",
    ),
    (
        origins.ObservationScope(outcome="exact_allele_observed"),
        "",
        "limitation",
        "Exact allele and genotype observation in this VCF only.",
        "the result holds for this sample's VCF at this target only",
    ),
    (
        origins.ObservationNotInferred(),
        "",
        "limitation",
        "Coverage, callability, biological absence, function, causality, and "
        "diagnosis are not inferred.",
        "coverage, callability, biological absence, function, causality and "
        "diagnosis are not inferred from it",
    ),
    (
        origins.ObservationNotInferredDepthMeasured(),
        "",
        "limitation",
        "Callability, biological absence, function, causality, and diagnosis are "
        "not inferred; read depth is the Run's measured value.",
        "callability, biological absence, function, causality and diagnosis are "
        "not inferred from it; its read depth was measured",
    ),
    (
        origins.ObservationProvenance(run_provenance="unestablished"),
        "",
        "limitation",
        "Run provenance is unestablished; freshness and development-clone status "
        "are not inferred.",
        "whether the Run's analysis was freshly made is not established",
    ),
    (
        origins.ObservationProvenance(run_provenance="causal_child_or_fork"),
        "",
        "limitation",
        "This observation belongs to a Run a Fork created; retained execution is "
        "not fresh child work.",
        "it comes from a Run forked from earlier work, not a fresh analysis",
    ),
    (
        origins.TargetAltNotCarried(),
        "",
        "conflict",
        "The sample does not carry the requested target ALT in this VCF.",
        "the sample does not carry the target allele here",
    ),
    (
        origins.ObservationNotAdmitted(),
        "",
        "gap",
        "Automatic claim admission is unavailable for this outcome.",
        "this result is not admitted as a finding automatically",
    ),
    (
        origins.DiploidCallingJudgedOtherwise(judgment="likely_non_diploid"),
        "",
        "limitation",
        "Variant calling assumed a diploid sample (ploidy 2); the planning model "
        "judged this organism likely not diploid, and the researcher chose diploid "
        "calling.",
        "the variants were called as diploid, though the planning model judged this "
        "organism likely not diploid; you chose diploid calling",
    ),
    (
        origins.DiploidCallingJudgedOtherwise(judgment="unknown"),
        "",
        "limitation",
        "Variant calling assumed a diploid sample (ploidy 2); the planning model "
        "judged this organism of unknown ploidy, and the researcher chose diploid "
        "calling.",
        "the variants were called as diploid, though the planning model judged this "
        "organism of unknown ploidy; you chose diploid calling",
    ),
)
# Audit wording the readable body never shows: a stage, a layout word, a recorded
# state or code (snake_case) or a source's verdict.
AUDIT_WORDING = re.compile(
    r"Work item|Finalization|checked not verified|\b\w+_\w+\b|\brows?\b|\bnot_found\b"
)
