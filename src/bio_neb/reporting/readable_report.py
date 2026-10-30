"""The researcher's readable report: one Markdown page from one published report.

The page is a clean report of the Run, not an interpretation (#491, owner
2026-10-30; the template is ``docs/development/reviews/report-design/README.md``
§0). Sections 1-11 are retrieved facts, each one of: a copied field of the
approved Grounded report, of a Journey fact its accounting holds or of a raw
source answer it names; an owner's fixed words keyed by a typed value (each
state through ``state_words``, each qualification through its typed origin); a
count; or a label (``T1`` for targets in report order, ``P1`` for papers, ``L1``
for the report's limitation lines). Section 12 is the report explainer's: its
title, summary, per-target notes and caveats, each admitted by the explanation's
one check and marked. A long copied text is printed once and a long quote shows
its named cut (ADR 0049). Before the page is returned, a consistency check
compares every results row with the report's own members; a row that disagrees
refuses the page. The page stores nothing: it renders again from the same
report, explanation and Run (owner Q4 a).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import PurePath

from bio_neb.investigation.case_origins import MergedIdentity
from bio_neb.investigation.collection_view import (
    CollectionView,
    RequestedSnp,
    RequestedSnpItem,
    RequestLookup,
)
from bio_neb.investigation.investigation import (
    InvestigationProposal,
    RegionTarget,
    RetrievedSource,
    VariantTarget,
)
from bio_neb.investigation.qualification_origins import RequestNotChecked
from bio_neb.investigation.state_words import item_words, request_words
from bio_neb.paper_reading.report import ReportedPaper, asserter_words
from bio_neb.reporting.explanation import (
    ExplainedRequest,
    ReportExplanationByRequest,
)
from bio_neb.reporting.explanation_rows import numbered_report_qualifications
from bio_neb.reporting.grounded_report import (
    GroundedExactRecordObservation,
    GroundedReport,
)

_AUTHORITY = {"ncbi": "NCBI", "eva": "EVA"}
_NOT_WRITTEN = "Not written"
# A target's label: its rsID, its position, else its words up to this many
# characters (never a paper's paragraph, Mullins F1).
LABEL_LIMIT = 60
# A copied quote inline, before its named cut.
QUOTE_LIMIT = 200
# A source's own answer inline, in bytes, before its named cut.
BODY_LIMIT = 300
# The Run's recorded outcome in plain words; another shows its code.
_OUTCOMES = {
    "investigation_evidence_ready": "The Run checked every target it could; the researcher approved its report.",
    "execution_failed": "The sequencing analysis did not finish; the report accounts for the Run.",
    "investigation_requests_rejected": "Every target was rejected; the report accounts for the Run.",
    "investigation_issues_approved": "Each target ended with an issue the researcher accepted; no VCF search ran.",
    "investigation_requests_not_checked": "No target could be checked as stated; no VCF search ran.",
}
_INDEX = {
    "prebuilt": "a prebuilt index from a public data cache",
    "admitted": "an admitted prepared index",
    "cpu_built": "an index built for this Run",
}
_PROVENANCE = {
    "unestablished": "not established (the Run records no proof that its analysis was freshly made)",
    "causal_child_or_fork": "a child or Fork of an earlier Run",
}


def _answered(action: str) -> str:
    """A Review's answer to an exact proposal or basis: approved, else its action
    as recorded."""
    return "approved" if action == "approve" else f"answered `{action}`"


_FACTS = (
    "Sections 1–11 are facts retrieved from the Run's Journey and its approved "
    "Grounded report; nothing in them is interpreted. Section 12 is the report "
    "explainer's own summary and caveats. What was tested, the traces and what was "
    "not tested are in the Run's trace (`observe --rich`, `journey-feed`)."
)


class ReadableReportInconsistent(ValueError):
    """A rendered results row disagrees with the report it was copied from."""

    def __init__(self, faults: tuple[str, ...]) -> None:
        super().__init__("readable_report_inconsistent")
        self.code = "readable_report_inconsistent"
        self.faults = faults


@dataclass(frozen=True)
class ExplanationState:
    """The report's latest explanation, or why its words are not written."""

    explanation: ReportExplanationByRequest | None
    not_written: str | None = None


@dataclass(frozen=True)
class RunFacts:
    """What the page reads from the Run beside its report: when the report was
    published, and each source's own answer as the report names it (its bytes
    verified against the source's sha256), or None when the report names no
    retained answer. A named answer that does not verify refuses the page."""

    published_at: str | None = None
    source_answer: Callable[[RetrievedSource], bytes | None] = field(
        default=lambda _source: None
    )


@dataclass(frozen=True)
class ReadableItem:
    """One Work item's cell: the recorded outcome or state its words are keyed by,
    what those words state, and where it was placed."""

    work_item_id: str
    key: str
    searched: bool
    carries: bool
    genotype: str | None
    # Where it was placed, as its Work item's target reads (a position with its
    # alleles, or a region's interval), else as the view recorded it.
    placed: str


@dataclass(frozen=True)
class ReadableRow:
    """One request as the page shows it: its target, the facts of each of its
    items, and the source answer its check read."""

    number: int
    request_member_id: str
    # The request as its asserter wrote it, and its label.
    target: str
    label: str
    # How the request was read: its rsID or position, a merge, a placement.
    read_as: str
    state: str | None
    items: tuple[ReadableItem, ...]
    lookup: RequestLookup | None
    # Why a request not checked was set aside, when its origin names it.
    reason_code: str | None = None
    # Who asserts the target when it is not the researcher: a paper's id.
    asserted_by: str | None = None

    @property
    def id(self) -> str:
        return f"T{self.number}"

    @property
    def searched(self) -> bool:
        return any(item.searched for item in self.items)

    @property
    def carries_target(self) -> bool:
        return any(item.carries for item in self.items)

    @property
    def not_searched(self) -> str:
        """Why the request has no VCF search: its items' own state words, or,
        with no Work item, its request state's."""
        if self.items:
            return "; ".join(
                dict.fromkeys(item_words(item.key).cell for item in self.items)
            )
        words = request_words(self.state, self.reason_code).cell
        return words[:1].upper() + words[1:]

    @property
    def result(self) -> str:
        """Each item's short table words, or the request's own state's."""
        if self.items:
            return ", ".join(
                dict.fromkeys(item_words(item.key).table for item in self.items)
            )
        return request_words(self.state, self.reason_code).cell


def _position(contig: str, position: int, ref: str | None, alt: str | None) -> str:
    # A trimmed indel's empty allele reads as "-".
    alleles = (
        f" {ref or '-'}>{alt or '-'}" if ref is not None and alt is not None else ""
    )
    return f"{contig}:{position}{alleles}"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _label(snp: RequestedSnp) -> str:
    """The rsID or position the target states, else its first words: a copied
    field, never the paper's paragraph (Mullins F1)."""
    if snp.requested_rsid:
        return snp.requested_rsid
    stated = snp.stated_position
    if stated is not None:
        return _position(stated.contig, stated.position, stated.ref, stated.alt)
    text = _one_line(snp.request_text)
    if len(text) <= LABEL_LIMIT:
        return text
    return text[: LABEL_LIMIT - 1].rsplit(" ", 1)[0] + " …"


def _read_as(snp: RequestedSnp, view: CollectionView) -> str:
    """How the request was read, from its row's typed facts: a merge in its case's
    own words, a placement from the version it was stated on, else the rsID or
    position it states (#481 H2)."""
    parts = [
        case.origin.words()
        for case in view.cases
        if isinstance(case.origin, MergedIdentity)
        and snp.request_member_id in case.request_member_ids
    ]
    placement = None if snp.lookup is None else snp.lookup.placement
    if placement is not None:
        placed = placement.placed
        at = _position(placed.contig, placed.position, placed.ref, placed.alt)
        stated = placement.stated
        if stated is None:
            parts.append(f"placed at {at}")
        else:
            version = (
                ""
                if placement.stated_version is None
                else f" on {placement.stated_version}"
            )
            parts.append(
                _position(stated.contig, stated.position, stated.ref, stated.alt)
                + f" as stated{version}; placed at {at}"
            )
    if parts:
        return "; ".join(parts)
    stated = snp.stated_position
    if stated is not None:
        return _position(stated.contig, stated.position, stated.ref, stated.alt)
    return snp.verified_rsid or snp.requested_rsid or "—"


def _status(code: int) -> str:
    try:
        return f"HTTP {code} ({HTTPStatus(code).phrase.lower()})"
    except ValueError:
        return f"HTTP {code}"


def _placement(target) -> str | None:
    """A Work item's target in readable form: its position and alleles, or its
    region's interval."""
    if isinstance(target, VariantTarget):
        return _position(
            target.contig, target.one_based_position, target.ref, target.alt
        )
    if isinstance(target, RegionTarget):
        return f"{target.contig}:{target.start}-{target.end}"
    return None


def _item(item: RequestedSnpItem, work_items) -> ReadableItem:
    key = item.outcome or item.state or ""
    words = item_words(key)
    target = work_items.get(item.work_item_id)
    # A candidate outside the observed membership keeps the target the view
    # recorded for it.
    placed = _placement(target) or item.target
    return ReadableItem(
        work_item_id=item.work_item_id,
        key=key,
        searched=words.searched,
        carries=words.carries,
        genotype=item.genotype,
        placed=placed,
    )


def readable_rows(report: GroundedReport) -> tuple[ReadableRow, ...]:
    """Every request of the report's view, its facts copied from its members."""
    view = report.collection_view
    if view is None:
        return ()
    work_items = {
        item.value.work_item_id: item.value.target for item in report.work_items
    }
    return tuple(
        ReadableRow(
            number=number,
            request_member_id=snp.request_member_id,
            target=snp.request_text,
            label=_label(snp),
            read_as=_read_as(snp, view),
            state=snp.state,
            items=tuple(_item(item, work_items) for item in snp.items),
            lookup=snp.lookup,
            reason_code=(
                snp.state_origin.reason_code
                if isinstance(snp.state_origin, RequestNotChecked)
                else None
            ),
            asserted_by=snp.asserted_by,
        )
        for number, snp in enumerate(view.snps, start=1)
    )


def consistency_faults(
    report: GroundedReport, rows: Sequence[ReadableRow]
) -> tuple[str, ...]:
    """Each row whose facts disagree with the report's own members (#460 owner
    item 2): the rows are the report's requests, each holding exactly its own
    request's items, lookup, target and label; each item reads as searched exactly
    when a Sample observation exists, with that observation's outcome, carrying
    the target exactly when it observed the target allele, with the genotype it
    calls and at its Work item's target; and a named source is the retained answer
    of its lookup, by id and authority."""
    observations = {
        item.value.work_item_id: item.value.result
        for item in report.sample_observations
    }
    targets = {
        item.value.work_item_id: _placement(item.value.target)
        for item in report.work_items
    }
    retrieved = {item.value.source_id: item.value for item in report.retrieved_sources}
    snps = (
        {snp.request_member_id: snp for snp in report.collection_view.snps}
        if report.collection_view is not None
        else {}
    )
    faults = []
    if [row.request_member_id for row in rows] != list(snps):
        faults.append("the rows are not the report's requests, in its order")
    for row in rows:
        # A row holds exactly its own request's items and lookup (ADR 0039).
        own = snps.get(row.request_member_id)
        if own is None or [item.work_item_id for item in row.items] != [
            item.work_item_id for item in own.items
        ]:
            faults.append(f"row {row.number}: its items are not its request's")
        if own is None or row.lookup != own.lookup:
            faults.append(f"row {row.number}: its source answer is not its request's")
        if own is None or row.target != own.request_text or row.label != _label(own):
            faults.append(f"row {row.number}: its target is not its request's")
        if own is None or row.read_as != _read_as(own, report.collection_view):
            faults.append(f"row {row.number}: its reading is not its request's")
        for item in row.items:
            observed = observations.get(item.work_item_id)
            fault = f"row {row.number} {item.work_item_id}:"
            if item.searched != (observed is not None):
                faults.append(f"{fault} its VCF search differs from the observations")
            if observed is not None and item.key != observed.outcome:
                faults.append(f"{fault} reads {item.key}, observed {observed.outcome}")
            if item.carries != (
                observed is not None and observed.outcome == "exact_allele_observed"
            ):
                faults.append(
                    f"{fault} its carried allele differs from the observation"
                )
            called = (
                observed.genotype.raw_genotype
                if isinstance(observed, GroundedExactRecordObservation)
                else None
            )
            if item.genotype is not None and item.genotype != called:
                faults.append(f"{fault} genotype {item.genotype} is not called")
            if (
                item.work_item_id in targets
                and item.placed != targets[item.work_item_id]
            ):
                faults.append(f"{fault} placed apart from its Work item")
        lookup = row.lookup
        if lookup is not None:
            source = retrieved.get(lookup.source_id)
            if source is None or source.authority != lookup.authority:
                faults.append(f"row {row.number}: its source answer is not retained")
    return tuple(faults)


# --- words -------------------------------------------------------------------


def _quoted(text: str, limit: int = QUOTE_LIMIT) -> str:
    """A copied quote, whole when short, else its start and a named cut."""
    text = _one_line(text)
    if len(text) <= limit:
        return f"“{text}”"
    head = text[:limit].rsplit(" ", 1)[0]
    return f"“{head} …” [cut: {len(text):,} characters]"


def _cell(text: str) -> str:
    # Copied and model text is free text: a pipe in it stays in its cell.
    return text.replace("|", "\\|")


def _where(line) -> str:
    """Where a copied chunk stands: its section as recorded and its offsets."""
    return f"{line.section}; {line.document_id} {line.start}–{line.end}"


def _within(line, chunks) -> str:
    """The larger copied chunk this one lies inside, named, if any."""
    outer = next(
        (
            other
            for other in chunks
            if other is not line
            and other.document_id == line.document_id
            and other.start <= line.start
            and line.end <= other.end
            and (other.end - other.start) > (line.end - line.start)
        ),
        None,
    )
    return (
        ""
        if outer is None
        else f", within {outer.document_id} {outer.start}–{outer.end}"
    )


@dataclass(frozen=True)
class _Line:
    """One limitation line: the report's qualifications that read alike, by number,
    with the targets their ids concern."""

    id: str
    kind: str
    words: str
    numbers: tuple[int, ...]
    targets: tuple[ReadableRow, ...]


def _limit_lines(
    report: GroundedReport, rows: Sequence[ReadableRow]
) -> tuple[_Line, ...]:
    """Each of the report's qualifications in its typed origin's plain words once,
    led by every target its ids cover (#481 H4): qualifications that read alike
    are one line, numbered in the report's order (#474: no audit sentence)."""
    grouped: dict[str, dict] = {}
    for number, kind, qualification in numbered_report_qualifications(report):
        words = qualification.origin.words()
        line = grouped.setdefault(words, {"kind": kind, "numbers": [], "targets": []})
        line["numbers"].append(number)
        line["targets"] += [
            row
            for row in rows
            if row.request_member_id in qualification.request_member_ids
            or any(
                item.work_item_id in qualification.work_item_ids for item in row.items
            )
        ]
    return tuple(
        _Line(
            id=f"L{index}",
            kind=line["kind"],
            words=words,
            numbers=tuple(line["numbers"]),
            targets=tuple(
                sorted(
                    {row.number: row for row in line["targets"]}.values(),
                    key=lambda row: row.number,
                )
            ),
        )
        for index, (words, line) in enumerate(grouped.items(), start=1)
    )


class _Page:
    def __init__(
        self, report: GroundedReport, state: ExplanationState, run: RunFacts
    ) -> None:
        self.report = report
        self.state = state
        self.explanation = state.explanation
        self.run = run
        self.rows = readable_rows(report)
        self.by_member = {row.request_member_id: row for row in self.rows}
        self.lines = _limit_lines(report, self.rows)
        self.events = [entry.value for entry in report.accounting]
        self.sources = {
            item.value.source_id: item.value for item in report.retrieved_sources
        }
        self.depths = (
            {}
            if report.run_health is None
            else {line.work_item_id: line for line in report.run_health.items}
        )
        self.items = {
            item.work_item_id: item
            for snp in (report.collection_view.snps if report.collection_view else ())
            for item in snp.items
        }
        self.papers = {
            paper.asserted_by: f"P{number}"
            for number, paper in enumerate(report.papers, start=1)
            if paper.asserted_by is not None
        }

    # --- reading the report ----------------------------------------------------

    def payloads(self, event_type: str) -> list[Mapping]:
        return [
            entry.payload for entry in self.events if entry.event_type == event_type
        ]

    def fact(self, kind: str):
        return next(
            (slot for slot in self.report.fact_slots if slot.kind == kind), None
        )

    def limits_of(self, row: ReadableRow) -> str:
        return ", ".join(line.id for line in self.lines if row in line.targets) or "—"

    def from_whom(self, row: ReadableRow) -> str:
        words = asserter_words(row.asserted_by)
        if words is None:
            return "you"
        return self.papers.get(row.asserted_by, words)

    def ended_by(self) -> dict[str, ReadableRow]:
        """Each item Review answer's event id, with the target it ended: the
        report's proposal member, whose rejection or issue approval names the
        answer (#491 D7 a: the stored link, no reason match)."""
        ended = {}
        for entry in self.events:
            if entry.event_type not in (
                "InvestigationProposalRecorded",
                "InvestigationProposalCarriedForward",
            ):
                continue
            proposal = InvestigationProposal.model_validate_json(
                json.dumps(entry.payload["proposal"])
            )
            for member in proposal.request_members:
                row = self.by_member.get(member.request_member_id)
                for decision in (member.rejection, member.issue_approval):
                    if row is not None and decision is not None:
                        ended[decision.response_event_id] = row
        return ended

    # --- the page --------------------------------------------------------------

    def render(self) -> str:
        method = self.report.method
        sample_slot = self.fact("sample_identity")
        sample = (
            method.sample_id
            if method is not None
            else (
                sample_slot.verified_value or sample_slot.requested_value
                if sample_slot is not None
                else (
                    self.report.work_items[0].value.sample_id
                    if self.report.work_items
                    else "the sample"
                )
            )
        )
        out = [f"# Run report: {sample}", "", self.title(), "", _FACTS]
        for section in (
            self.run_section,
            self.summary,
            self.question,
            self.health,
            self.results,
            self.decisions,
            self.papers_section,
            self.not_examined,
            self.limitations,
            self.method,
            self.references,
            self.agent,
        ):
            out += ["", *section()]
        return "\n".join(out) + "\n"

    def not_written(self) -> str:
        return f"{_NOT_WRITTEN}: {self.state.not_written or 'no explanation was published'}"

    def title(self) -> str:
        explanation = self.explanation
        if explanation is None:
            return f"*Title: {self.not_written()}.*"
        if explanation.title is None:
            return (
                f"*Title: {_NOT_WRITTEN}: this explanation was written before titles.*"
            )
        return f"*{_one_line(explanation.title)}* ✓ (the report explainer's title)"

    def run_section(self) -> list[str]:
        report = self.report
        roots = [
            PurePath(payload["run_root"]).name
            for payload in self.payloads("RunInputsRegistered")
        ]
        rows = [
            ("Run", f"`{roots[0]}`" if roots else "—"),
            ("Journey", f"`{report.journey_id}`"),
        ]
        for kind, name in (
            ("sample_identity", "Sample"),
            ("taxon_identity", "Organism (NCBI Taxonomy)"),
            ("reference_assembly_identity", "Reference"),
        ):
            slot = self.fact(kind)
            if slot is not None:
                value = slot.verified_value or slot.requested_value or "—"
                rows.append((name, f"{value} — {slot.summary}"))
        action = report.review_authority.response.action
        rows += [
            ("Outcome", _OUTCOMES.get(report.outcome, f"`{report.outcome}`")),
            (
                "Report basis",
                f"{_answered(action)} by the researcher at Finalization",
            ),
            (
                "Run provenance",
                _PROVENANCE.get(report.run_provenance, report.run_provenance),
            ),
            ("Report published", self.run.published_at or "—"),
        ]
        return [
            "## 1. Run",
            "",
            "| | |",
            "|---|---|",
            *(f"| {name} | {_cell(value)} |" for name, value in rows),
        ]

    def summary(self) -> list[str]:
        rows = self.rows
        searched = sum(row.searched for row in rows)
        carried = sum(row.carries_target for row in rows)
        return [
            "## 2. Summary",
            "",
            f"> **{len(rows)} targets · {searched} examined in the VCF · {carried} with "
            f"the target allele called · {len(rows) - searched} not examined**",
        ]

    def question(self) -> list[str]:
        prompts = self.payloads("ResearchRequestRegistered")
        return [
            "## 3. Question",
            "",
            f"> {_one_line(prompts[0]['prompt'])}" if prompts else "Not recorded.",
        ]

    def health(self) -> list[str]:
        """The Run's own measurement (#414), copied from the report's health
        section: it reports and never changed a decision."""
        out = ["## 4. Run health", ""]
        section = self.report.run_health
        if section is None:
            return out + ["Not measured: no Run health fact is in this report."]
        health = section.health
        if health.measurement is None:
            return out + [f"Not measured: {health.not_measured.detail}."]
        numbers = health.measurement
        ti_tv = (
            f"{numbers.ti_tv}"
            if numbers.ti_tv is not None
            else {
                "no_transversion": "none: the VCF has no transversion",
                "vcf_unprepared": "not read: the VCF could not be prepared",
            }[numbers.ti_tv_reason]
        )
        rows = [
            ("Reads", f"{numbers.reads:,}"),
            ("Mapped", f"{numbers.mapped_percent}%"),
            ("Mean depth", f"{numbers.mean_depth:.1f}"),
            ("Ti/Tv", ti_tv),
            (
                "Covered",
                f"{numbers.covered_fraction:.1%} of bases, on "
                f"{numbers.covered_contigs} sequences",
            ),
        ]
        return out + [
            "| Measure | Value |",
            "|---|---|",
            *(f"| {name} | {value} |" for name, value in rows),
        ]

    def results(self) -> list[str]:
        method = self.report.method
        reference = "the reference" if method is None else method.reference_accession
        out = [
            "## 5. Results",
            "",
            "One row per position proposed for a target; every cell is the VCF's or "
            "the depth line's.",
            "",
            f"| T | Target | From | Position ({reference}) | VCF result | GT | GQ "
            "| DP (AD) | FILTER / QUAL | Depth at site (min) | Limits |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in self.rows:
            for index, item in enumerate(row.items):
                cells = self.items[item.work_item_id]
                depth = "—" if cells.read_depth is None else str(cells.read_depth)
                if cells.allele_depths:
                    depth += (
                        " ("
                        + ",".join(
                            "." if value is None else str(value)
                            for value in cells.allele_depths
                        )
                        + ")"
                    )
                line = self.depths.get(item.work_item_id)
                site = (
                    f"{line.depth} (min {line.minimum_depth}"
                    + "".join(f", {flag}" for flag in line.batch_flags)
                    + ")"
                    if line is not None
                    else {"not_measured": "not measured", "measured": "measured"}.get(
                        cells.coverage or "", "—"
                    )
                )
                name, label = (
                    (f"**{row.id}**", _cell(row.label)) if index == 0 else (row.id, "")
                )
                out.append(
                    f"| {name} | {label} | {self.from_whom(row)} | {item.placed} "
                    f"| {item_words(item.key).table} | {item.genotype or '—'} "
                    f"| {'—' if cells.genotype_quality is None else cells.genotype_quality} "
                    f"| {depth} | {';'.join(cells.filters or ()) or '—'} / "
                    f"{cells.qual or '—'} | {site} | {self.limits_of(row)} |"
                )
        if not any(row.items for row in self.rows):
            out.append("| — | No target was examined in the VCF | | | | | | | | | |")
        # How a target was read when that is more than its label: a dbSNP merge
        # in its case's words, a position placed from another version.
        read = [row for row in self.rows if row.read_as not in (row.label, "—")]
        if read:
            out += [
                "",
                "How each target was read, where it differs from its label:",
                "",
            ]
            out += [f"- **{row.id}** {_cell(row.read_as)}" for row in read]
        return out

    def decisions(self) -> list[str]:
        """Key decisions and metadata, each from the fact that holds it; the full
        Review flow is in the Run's trace (owner Q6)."""
        method = self.report.method
        out = ["## 6. Key decisions and metadata", ""]
        reference = self.fact("reference_assembly_identity")
        if reference is not None:
            index = "" if method is None else f"; index: {_INDEX[method.index_origin]}"
            out.append(
                f"- **Reference:** {reference.verified_value or reference.requested_value} "
                f"({reference.summary}){index}."
            )
        requests = {
            candidate["candidate_id"]: candidate["display_label"]
            for payload in self.payloads("PlanningHitlRequested")
            for candidate in payload["candidates"]
        }
        for payload in self.payloads("PlanningHitlResponded"):
            response = payload["response"]
            chosen = requests.get(response.get("candidate_id"))
            out.append(
                f"- **Planning question answered:** {response['action']}"
                + (f" “{chosen}”" if chosen else "")
                + f" (`{payload['authority']}`)."
            )
        for payload in self.payloads("ExecutionHitlResponded"):
            response = payload["response"]
            ploidy = (
                f", calling ploidy {response['calling_ploidy']}"
                if response.get("calling_ploidy")
                else ""
            )
            out.append(
                "- **Execution:** the exact Parabricks proposal "
                f"{_answered(response['action'])}{ploidy} "
                f"(`{payload['authority']}`)."
            )
        ended = self.ended_by()
        answers = [
            (entry.source_event_id, entry.payload["response"])
            for entry in self.events
            if entry.event_type == "InvestigationReviewResponded"
        ]
        # Target questions in target order, then the proposal's answers, each
        # follow-up after the answer it follows: the accounting is a set.
        follows = {
            entry.source_event_id: entry.payload.get("follows_response_event_id")
            for entry in self.events
            if entry.event_type == "InvestigationReviewRequested"
        }
        asked = {
            entry.source_event_id: entry.payload["request_event_id"]
            for entry in self.events
            if entry.event_type == "InvestigationReviewResponded"
        }

        def depth(event_id: str | None) -> int:
            followed = follows.get(asked.get(event_id))
            return 0 if followed is None else 1 + depth(followed)

        for event_id, response in sorted(
            answers,
            key=lambda answer: (
                "proposal_id" in answer[1],
                ended[answer[0]].number if answer[0] in ended else len(self.rows) + 1,
                depth(answer[0]),
            ),
        ):
            action = response["action"]
            if "proposal_id" in response:
                scope = (
                    ""
                    if response.get("work_item_ids") is None
                    else f" for {len(response['work_item_ids'])} positions"
                )
                out.append(
                    "- **Investigation:** the exact proposal "
                    f"{_answered(action)}{scope}."
                )
                continue
            row = ended.get(event_id)
            where = None if row is None else self._reason_line(row)
            reason = response.get("reason")
            out.append(
                f"- **Target question, {f'{row.id} {row.label}' if row else 'a target'}:** "
                + action
                + (
                    f" (reason in {where})"
                    if where
                    else (f": {_quoted(reason, 10**6)}" if reason else "")
                )
                + f", by {response['asserted_by']}."
            )
        final = self.report.review_authority.response.action
        out.append(f"- **Finalization:** the exact Report basis {_answered(final)}.")
        roles = sorted(
            {
                payload["instruction_identity"]
                for payload in self.payloads("AttemptStarted")
                if payload.get("instruction_identity")
            }
        )
        if self.explanation is not None:
            roles.append(self.explanation.instruction_identity)
        if roles:
            out.append(
                "- **Agent roles and instructions:** "
                + ", ".join(f"`{role}`" for role in roles)
                + "."
            )
        inputs = [
            PurePath(item["registered_path"]).name
            for payload in self.payloads("RunInputsRegistered")
            for item in payload["inputs"]
        ]
        if inputs:
            out.append(
                "- **Inputs:** " + ", ".join(f"`{name}`" for name in inputs) + "."
            )
        return out

    def _reason_line(self, row: ReadableRow) -> str | None:
        """The L-line holding the target's own ending, whose owner words already
        copy the answer's reason."""
        origin = self.report.collection_view.snps[row.number - 1].state_origin
        if origin is None:
            return None
        return next(
            (
                line.id
                for line in self.lines
                if row in line.targets and line.words == origin.words()
            ),
            None,
        )

    def papers_section(self) -> list[str]:
        out = ["## 7. Papers"]
        if not self.report.papers:
            return out + ["", "No paper was read in this Run."]
        prompts = [
            _one_line(payload["prompt"])
            for payload in self.payloads("ResearchRequestRegistered")
        ]
        for number, paper in enumerate(self.report.papers, start=1):
            out += ["", *self._paper(number, paper, prompts)]
        return out

    def _paper(self, number: int, paper: ReportedPaper, prompts) -> list[str]:
        identity, answer = paper.paper, paper.answer
        title = (
            identity.title or identity.file_name or identity.paper_id or "A Run paper"
        )
        year = f" ({identity.year})" if identity.year else ""
        journal = f", {identity.journal}" if identity.journal else ""
        question = _one_line(paper.question)
        out = [
            f"### P{number} · {title}{year}{journal}",
            "",
            (f"{_ids(identity)}. " if _ids(identity) else "")
            + "Read for: "
            + (
                "the question in section 3."
                if question in prompts
                else _quoted(question)
            ),
        ]
        asserter = asserter_words(paper.asserted_by)
        if asserter is not None:
            out.append(f"Its targets are asserted by {asserter}, not by you.")
        chunks = [chunk for finding in answer.findings for chunk in finding.passages]
        if answer.findings:
            out += [
                "",
                "| # | The paper reports (copied, with where) | This Run found |",
                "|---|---|---|",
            ]
        quoted: dict = {}
        for index, finding in enumerate(answer.findings, start=1):
            values = ", ".join(
                str(value.value) for value in finding.values if value.slot != "gene"
            )
            gene = f" ({finding.gene})" if finding.gene else ""
            quote = finding.passages[0] if finding.passages else None
            span = (
                None if quote is None else (quote.document_id, quote.start, quote.end)
            )
            copied = (
                ""
                if quote is None
                # A quote another finding already printed is named, not repeated.
                else f"the quote of finding {quoted[span]}"
                if span in quoted
                else f"{_quoted(quote.text)} ({_where(quote)}{_within(quote, chunks)})"
            )
            quoted.setdefault(span, index)
            stated = values + gene
            copied = f"{stated}: {copied}" if stated and copied else stated or copied
            # A paper's target links to the findings naming its value (D4 a).
            found = [
                row
                for row in self.rows
                if row.asserted_by == paper.asserted_by
                and row.asserted_by is not None
                and any(
                    str(value.value) == self._requested(row) for value in finding.values
                )
            ]
            result = (
                "; ".join(f"{row.id} {row.label}: {row.result}" for row in found) or "—"
            )
            out.append(f"| {index} | {_cell(copied)} | {_cell(result)} |")
        if answer.findings:
            out.append("")
        if answer.organism is not None:
            out.append(
                f"Organism as written: **{answer.organism.written}** "
                f"({_where(answer.organism.passage)})."
            )
        if answer.reference is not None:
            out.append(
                f"Genome version as written: **{answer.reference.candidate.name}** "
                f"({_where(answer.reference.candidate.sources[0])})."
            )
        elif answer.not_answered:
            out.append(
                "Genome version: none kept; the reader wrote: "
                f"*{_one_line(answer.not_answered)}*"
            )
        for kind, outcomes in (
            ("unresolved", answer.unresolved),
            ("refused", answer.refused),
        ):
            for outcome in outcomes:
                out.append(
                    f"Not reported by the paper ({kind}, "
                    f"`{', '.join(outcome.reasons)}`); the reader wrote: "
                    f"*{_one_line(outcome.stated.reason)}*"
                )
        return out

    def _requested(self, row: ReadableRow) -> str | None:
        return self.report.collection_view.snps[row.number - 1].requested_rsid

    def _answer(self, lookup: RequestLookup | None) -> str:
        source = None if lookup is None else self.sources.get(lookup.source_id)
        return source_answer_cell(
            lookup, None if source is None else self.run.source_answer(source)
        )

    def not_examined(self) -> list[str]:
        out = ["## 8. Not examined", ""]
        skipped = [row for row in self.rows if not row.searched]
        if not skipped:
            return out + ["Every target was examined in the VCF."]
        out += ["| T | Target | Why | Source answer |", "|---|---|---|---|"]
        for row in skipped:
            out.append(
                f"| **{row.id}** | {_cell(row.label)} | {row.not_searched} "
                f"| {_cell(self._answer(row.lookup))} |"
            )
        return out

    def limitations(self) -> list[str]:
        out = [
            "## 9. Limitations",
            "",
            "The report's own qualifications, each once, led by the targets it concerns:",
            "",
        ]
        if not self.lines:
            return out + ["- The report records none."]
        for line in self.lines:
            kind = "" if line.kind == "limitation" else f" ({line.kind})"
            words = line.words.rstrip(".")
            covers = ", ".join(row.id for row in line.targets)
            text = (
                f"{covers}: {words}." if covers else words[:1].upper() + words[1:] + "."
            )
            out.append(f"- **{line.id}**{kind} {text}")
        return out

    def method(self) -> list[str]:
        return ["## 10. Method", "", *_method_lines(self.report)]

    def references(self) -> list[str]:
        report = self.report
        out = ["## 11. References and identifiers", ""]
        for number, paper in enumerate(report.papers, start=1):
            identity = paper.paper
            title = (
                identity.title
                or identity.file_name
                or identity.paper_id
                or "A Run paper"
            )
            where = " ".join(part for part in (identity.journal, identity.year) if part)
            out.append(
                f"- **P{number}.** {title}."
                + (f" {where}." if where else "")
                + (f" {_ids(identity)}." if _ids(identity) else "")
            )
        for row in self.rows:
            lookup = row.lookup
            if lookup is None:
                continue
            out.append(f"- **{row.id} source:** `{lookup.operation}` {lookup.asked}")
            if lookup.model_matched:
                # A contig the written words do not literally name, matched by the
                # model: shown so the researcher can check it.
                out.append(
                    "  The model matched the written sequence name: "
                    + "; ".join(name.audit() for name in lookup.model_matched)
                    + "."
                )
        out += [
            f"- **Grounded report:** `{report.report_digest}` ({report.schema_version})",
            f"- **Report basis:** `{report.report_basis_digest}`",
        ]
        if report.method is not None:
            out.append(f"- **VCF:** `{report.method.vcf_content_id}`")
        if self.explanation is not None:
            out.append(f"- **Explanation:** `{self.explanation.explanation_id}`")
        return out

    def agent(self) -> list[str]:
        """The explainer's own text: its summary, per-target notes and caveats.
        Admitted by citation and qualification membership; meaning unchecked."""
        out = [
            "## 12. The agent's summary and caveats",
            "",
            "*Written by the report explainer. Every row and event it cites is in this "
            "report; its meaning is not checked. Where it differs from a fact above, the "
            "fact is the report's.*",
            "",
        ]
        explanation = self.explanation
        if explanation is None:
            return out + [f"{self.not_written()}."]
        sections = {
            section.request_member_id: section for section in explanation.requests
        }
        out += [
            f"**Summary ✓** {_one_line(explanation.conclusion.text)}",
            "",
            f"**What would change it ✓** {_one_line(explanation.would_change)}",
            "",
        ]
        for row in self.rows:
            section = sections.get(row.request_member_id)
            if isinstance(section, ExplainedRequest):
                out.append(
                    f"- **{row.id} {_cell(row.label)} ✓** {_one_line(section.why.text)}"
                )
            elif section is not None:
                out.append(
                    f"- **{row.id} {_cell(row.label)}:** {_NOT_WRITTEN}: {section.reason.rstrip('.')}."
                )
        out += ["", "**Caveats ✓**", ""]
        out += self._caveats(explanation, sections) or ["- none written"]
        out += ["", f"**Not done ✓** {_one_line(explanation.not_done)}"]
        return out

    def _caveats(self, explanation, sections) -> list[str]:
        """The explainer's limitations, each tagged by the L-lines its typed
        qualification numbers name; a target's caveats on the same L-lines are
        alike (#481 O1): the first target's are shown, the others named beside them."""
        line_of = {number: line.id for line in self.lines for number in line.numbers}
        written = [("the whole Run", item) for item in explanation.limitations]
        written += [
            (row.id, item)
            for row in self.rows
            if isinstance(sections.get(row.request_member_id), ExplainedRequest)
            for item in sections[row.request_member_id].limitations
        ]
        groups: dict[str, list] = {}
        for where, item in written:
            lines = ", ".join(
                dict.fromkeys(
                    line_of[number]
                    for number in item.qualifications
                    if number in line_of
                )
            )
            key = f"{where}|{lines}" if where == "the whole Run" else lines
            groups.setdefault(key, []).append((where, lines, item))
        out = []
        for alike in groups.values():
            # Every caveat of the first target is printed whole; another target's
            # caveats on the same L-lines are named, their words kept in the
            # explanation (M5).
            leader = alike[0][0]
            others = ", ".join(
                dict.fromkeys(
                    other for other, _lines, _item in alike if other != leader
                )
            )
            more = (
                f" (alike for {others}; each one's words: `report-explanation-reopen`)"
                if others
                else ""
            )
            shown = [(lines, item) for where, lines, item in alike if where == leader]
            for index, (lines, item) in enumerate(shown, start=1):
                out.append(
                    f"- *{leader}, on {lines or 'no report line'}:* "
                    f"{_one_line(item.text)}{more if index == len(shown) else ''}"
                )
        return out


def _method_lines(report) -> list[str]:
    """The method the report copies from its verified Execution facts."""
    method = report.method
    if method is None:
        # A failed or stopped Execution may have run: only its verified VCF is absent.
        return ["No verified VCF is recorded for this Run, so no method can be stated."]
    return [
        f"- **Sample:** {method.sample_id} ({method.assay}, {method.read_layout} reads)",
        f"- **Reference:** {method.reference_accession}, from {method.reference_source} "
        f"(`{method.reference_checksum}`)"
        + (
            f" (the researcher asked for {method.requested_reference_accession} and "
            "chose this assembly, whose Galaxy index fits it, for the whole Run; each "
            "target's placement on it is in its own row)"
            if method.requested_reference_accession is not None
            else ""
        ),
        f"- **Index:** {_INDEX[method.index_origin]} ("
        + (
            f"BWA {method.index_bwa_version}"
            if method.index_bwa_version is not None
            else "its publisher states no BWA version"
        )
        + (
            f"; {len(method.index_extra_contigs)} contigs beyond the reference "
            "sequences, as the Execution Review showed"
            if method.index_extra_contigs
            else ""
        )
        + ")",
        f"- **Alignment and calling:** `{' '.join(method.command)}` "
        f"({method.pbrun_version}): {method.aligner}, then {method.caller}"
        + (
            f", calling ploidy {method.calling_ploidy}"
            if method.calling_ploidy is not None
            else ""
        ),
        f"- **Workflow and image:** `{method.workflow_id}` "
        f"(`{method.workflow_version}`), `{method.worker_image}`",
    ]


def source_answer_cell(lookup: RequestLookup | None, body: bytes | None) -> str:
    """A target's source answer: who was asked and its status, then its own body
    as the raw artifact the report names holds it (owner Q3, D1 a), at most
    ``BODY_LIMIT`` bytes inline before its named cut; ``body`` None when the Run
    does not retain it here."""
    if lookup is None:
        return "no source was asked"
    who = _AUTHORITY.get(lookup.authority, lookup.authority)
    lead = f"{who} `{lookup.operation}` {_status(lookup.status_code)}"
    if body is None:
        return lead + ", its answer is not retained in this report"
    if not body:
        return lead + ", empty body"
    text = body.decode("utf-8", "replace")
    if len(body) > BODY_LIMIT:
        text = (
            body[:BODY_LIMIT].decode("utf-8", "ignore")
            + f" … [cut: {len(body):,} bytes]"
        )
    return f"{lead}: `{_one_line(text)}`"


def _ids(identity) -> str:
    return ", ".join(
        f"{label}:{value}"
        for label, value in (
            ("doi", identity.doi),
            ("pmid", identity.pmid),
            ("pmcid", identity.pmcid),
        )
        if value
    )


def render_readable_report(
    report: GroundedReport, state: ExplanationState, run: RunFacts
) -> str:
    """The readable Markdown page, or ``ReadableReportInconsistent`` when a row
    would state what the report does not."""
    rows = readable_rows(report)
    faults = consistency_faults(report, rows) if rows else ()
    if faults:
        raise ReadableReportInconsistent(faults)
    return _Page(report, state, run).render()


__all__ = [
    "ExplanationState",
    "ReadableItem",
    "ReadableReportInconsistent",
    "ReadableRow",
    "RunFacts",
    "consistency_faults",
    "readable_rows",
    "render_readable_report",
]
