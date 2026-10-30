"""A paper answer's findings as proposed targets: the typed ``PaperTargets`` record.

Every finding becomes a target that the paper, never the researcher, asserts. Its request is
main's ``InvestigationRequestItem``: the paper's line as the quote, the one rsID that line states
by main's own rsID rule, the position its located candidates state and, with a position, the
paper's stated genome version with one line that states it. ``paper_targets`` is the one
callable: a reading's ``PaperAnswerRecorded`` keeps its record, and placement
(``target_request.placement``) places each target and writes the Run's request file; main then
looks each rsID up and checks the target's intent. Nothing here resolves a version name or
names a sequence.

The quote is the paper's exact words, never generated: for each rsID a finding states, the
first of its lines stating it (one stating it alone and its position first, as a table row
does), so main reads that rsID from the paper's own line; otherwise the finding's quote, the
paper's words for the variant that code chose from its kept values (its first passage; owner
#395 decision 4, #404 d1). Main's identity of a target (its rsID, else its words) joins
findings of one target, so the quote tells two changes of one sentence apart.

The position is the one locus the target's findings state (for an rsID target, those stating
that rsID): contig, position, REF and ALT agree where both state them; none when they state
two, or no contig. REF and ALT are written when that locus states both in VCF bases and they
differ. The stated version is the answer's reference: its name as written and its line, when
that line is one line of at most 4,000 characters (else that line's span around the name). A
target with no position carries no version, as main reads a version only with a position.

The change is the gene and the span of the quote holding the finding's change names as
written (``RAPGEF5 (c.2624C>A p.Ser875*)`` gives ``c.2624C>A p.Ser875*``), with the one
transcript id the quote's line writes, read by the general parser.

A target main cannot check by these fields is ``not_checked``: ``not_located`` (no rsID, no
position) or ``no_alleles`` (a position without alleles, no rsID); main asks the researcher.
"""

from __future__ import annotations

import re

from bio_neb.investigation.investigation_request import (
    RSID_RULE,
    InvestigationRequestItem,
    StatedAssembly,
    StatedPosition,
    request_target_key,
)
from bio_neb.paper_reading.answer import contig_name
from bio_neb.paper_reading.contracts import PaperAnswer
from bio_neb.target_request.paper_targets import (
    CITED_LINE_LIMIT,
    Finding,
    PaperIdentity,
    PaperTarget,
    PaperTargets,
    ReferenceCandidate,
    SourceLine,
    TargetChange,
    TargetKind,
    VariantCandidate,
    cited_window,
    is_protein_name,
    not_checked,
)
from bio_neb.target_request.paper_text import read


def paper_targets(answer: PaperAnswer) -> PaperTargets:
    """Every finding of ``answer`` as a proposed target asserted by its paper.

    Findings of one target by main's identity (its rsID, else its words) join into it, in
    the order the answer gives them."""

    if answer.paper is None:
        raise ValueError(
            "a paper target names its paper by DOI, PMID, PMCID or paper_id"
        )
    joined: dict[str, tuple[SourceLine, str | None, list[Finding]]] = {}
    for finding in answer.findings:
        for quote, rsid in _quotes(finding):
            key = request_target_key(quote.text, rsid)
            joined.setdefault(key, (quote, rsid, []))[2].append(finding)
    version = (
        None
        if answer.reference is None
        else _stated_version(answer.reference.candidate)
    )
    targets = []
    for quote, rsid, findings in joined.values():
        position = _position(findings, rsid)
        request = InvestigationRequestItem(
            request_text=quote.text,
            rsid=rsid,
            position=position,
            stated_assembly=None if position is None else version,
        )
        targets.append(
            PaperTarget(
                request=request,
                quote=quote,
                kind=_kind(tuple(v for f in findings for v in f.variants)),
                not_checked=not_checked(request),
                change=_change(findings, quote),
                # paper-targets/1 keeps its finding shape: a kept finding's values stay
                # in the answer.
                findings=tuple(
                    Finding.model_validate(f.model_dump(exclude={"values"}))
                    for f in findings
                ),
            )
        )
    return PaperTargets(
        question=answer.question,
        paper=answer.paper,
        asserted_by=asserted_by(answer.paper),
        reference=answer.reference,
        targets=tuple(targets),
        not_answered=answer.not_answered,
    )


def asserted_by(paper: PaperIdentity) -> str:
    """The paper as main's asserter: its DOI, else its PMID, else its PMCID, else, for a
    paper the researcher supplied with none of them, the Run's ``paper_id`` (#383 A1b)."""

    for kind in ("doi", "pmid", "pmcid", "paper_id"):
        value = getattr(paper, kind)
        if value:
            return f"paper:{kind}:{value}"
    raise ValueError("a paper target names its paper by DOI, PMID, PMCID or paper_id")


def _change(findings: list[Finding], quote: SourceLine) -> TargetChange | None:
    """The gene and the span of the quote holding the findings' change names, verbatim,
    with the one transcript the quote's line writes; None when the quote writes no change
    name of the findings, or there is no gene."""

    gene = next((f.gene for f in findings if f.gene), None)
    names = [
        name
        for f in findings
        for v in f.variants
        for name in v.hgvs
        if name in quote.text
    ]
    if not gene or not names:
        return None
    starts = [quote.text.index(name) for name in names]
    ends = [start + len(name) for start, name in zip(starts, names, strict=True)]
    transcripts = read(_line_of(quote, findings).text).transcripts
    return TargetChange(
        gene=gene,
        text=quote.text[min(starts) : max(ends)],
        transcript=next(iter(transcripts)) if len(transcripts) == 1 else None,
    )


def _line_of(quote: SourceLine, findings: list[Finding]) -> SourceLine:
    """The finding line the quote is a span of (the quote itself when it is a whole line)."""

    lines = (
        line
        for f in findings
        for line in (*f.passages[1:], *(s for v in f.variants for s in v.sources))
    )
    return next(
        (
            line
            for line in lines
            if line.document_id == quote.document_id
            and line.start <= quote.start
            and quote.end <= line.end
        ),
        quote,
    )


Locus = tuple[str | None, int, str | None, str | None]  # contig, position, REF, ALT


def _join(first: Locus, second: Locus) -> Locus | bool:
    """One locus from two statements of it, each value from whichever states it (the first
    spelling of a contig); False when they state two contigs (by ``contig_name``, the one
    rule the answer's join uses), two positions or two allele pairs."""

    joined = []
    for index, (mine, theirs) in enumerate(zip(first, second, strict=True)):
        same = (
            contig_name(mine) == contig_name(theirs)
            if index == 0 and mine is not None and theirs is not None
            else mine == theirs
        )
        if mine is not None and theirs is not None and not same:
            return False
        joined.append(mine if mine is not None else theirs)
    contig, position, ref, alt = joined
    return contig, position, ref, alt


_VCF_BASES = re.compile(r"[ACGTN]+")


def _position(findings: list[Finding], rsid: str | None) -> StatedPosition | None:
    """The one locus the target's located candidates state, by the candidates' own join rule:
    for an rsID target the candidates stating that rsID, else every located candidate. None
    when they state two loci, or none, or no contig. REF and ALT only when it states both in
    VCF bases and they differ."""

    loci = [
        (v.contig, v.position, v.ref, v.alt)
        for f in findings
        for v in f.variants
        if v.position is not None and (rsid is None or rsid in v.rsids)
    ]
    if not loci:
        return None
    locus = loci[0]
    for other in loci[1:]:
        locus = _join(locus, other)
        if locus is False:
            return None
    contig, position, ref, alt = locus
    if not contig:
        return None
    alleles = (
        ref is not None
        and alt is not None
        and ref != alt
        and all(_VCF_BASES.fullmatch(allele) for allele in (ref, alt))
    )
    return StatedPosition(
        contig=contig,
        position=position,
        ref=ref if alleles else None,
        alt=alt if alleles else None,
    )


def _stated_version(reference: ReferenceCandidate) -> StatedAssembly | None:
    """The answer's reference as main's stated version: its first spelling, as written, and
    its first line that is one line of at most 4,000 characters; else the span of its first
    line around that spelling, cut to 4,000 and trimmed. None if neither is one line."""

    for line in reference.sources:
        if _one_line(line.text):
            return StatedAssembly(name=reference.name, cited_line=line.text)
    cut = cited_window(reference.sources[0].text, reference.name)
    if not _one_line(cut):
        return None
    return StatedAssembly(name=reference.name, cited_line=cut)


def _one_line(text: str) -> bool:
    """One canonical line as main reads it, within its cited-line limit."""

    return (
        bool(text)
        and text.strip() == text
        and not any(char in text for char in "\r\n\0")
        and len(text) <= CITED_LINE_LIMIT
    )


def _quotes(finding: Finding) -> list[tuple[SourceLine, str | None]]:
    """The paper's line for each target of one finding, and the rsID it states."""

    rsids = dict.fromkeys(r for v in finding.variants for r in v.rsids)
    if rsids:
        return [
            (
                _line_stating([v for v in finding.variants if rsid in v.rsids], rsid),
                rsid,
            )
            for rsid in rsids
        ]
    return [(finding.passages[0], None)]


def _line_stating(variants: list[VariantCandidate], rsid: str) -> SourceLine:
    """The first of the variants' lines stating ``rsid``: one stating it alone and the
    variant's position first (a table row), then one stating it alone, then any."""

    positions = {str(v.position) for v in variants if v.position is not None}
    stating = [
        line
        for v in variants
        for line in v.sources
        if RSID_RULE.matches(rsid, line.text)
    ]
    if not stating:
        raise ValueError(f"no line of the finding states {rsid} by main's rsID rule")
    return min(
        stating,
        key=lambda line: (
            RSID_RULE.stated(line.text) != (rsid,),
            not any(p in line.text.replace(",", "") for p in positions),
        ),
    )


_DNA_SUBSTITUTION = re.compile(r"\d[ACGT]>[ACGT](?![A-Za-z])")
_DNA_INDEL = re.compile(r"del|ins|dup")


def _kind(variants: tuple[VariantCandidate, ...]) -> TargetKind:
    """The variant kind the paper states: by REF and ALT, else by an HGVS DNA name."""

    for variant in variants:
        if variant.ref and variant.alt:
            if len(variant.ref) == len(variant.alt):
                return "snv" if len(variant.ref) == 1 else "mnv"
            return "indel"
    names = [n for v in variants for n in v.hgvs if not is_protein_name(n)]
    for name in names:
        if _DNA_INDEL.search(name):
            return "indel"
        if _DNA_SUBSTITUTION.search(name):
            return "snv"
    return "not_stated"
