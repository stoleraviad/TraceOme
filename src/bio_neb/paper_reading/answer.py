"""A researcher's answer about one paper as code keeps it: the model reads, code checks.

A reading item's model step answers by span ids and written values (``reading``): each
value the paper writes for a finding (contig, position, alleles, rsIDs, change names, the
gene), exactly as the paper writes it, and the span id of the paragraph or table row that
writes it; the model copies no paragraph (#404). Code finds each written text in its span,
reads it into its slot (``judge``), and decides each finding on its own: accepted,
unresolved (the paper itself is unclear) or refused (a check failed). It copies every
passage and chooses each finding's quote from the paper (``kept_finding``), and verifies a
recorded answer against the paper's documents (``verify``). The answer holds the reference
as the paper writes it and every finding the question asks for, with no cap (owner, #395
decisions 1-5; #404 d1-d3, d7).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from bio_neb.content import shown_to_model
from bio_neb.investigation.investigation_request import RSID_RULE
from bio_neb.paper_reading.contracts import (
    MAX_AREAS,
    FindingOutcome,
    FindingPointer,
    KeptFinding,
    KeptValue,
    OrganismPointer,
    PaperAnswer,
    ReadDocument,
    ReferencePointer,
    StatedOrganism,
    ValuePointer,
    Verification,
)
from bio_neb.paper_reading.reader import Paper, Unit, passage
from bio_neb.target_request.paper_targets import (
    Finding,
    PaperIdentity,
    ReferenceCandidate,
    SourceLine,
    StatedReference,
    VariantCandidate,
    candidate_id,
    reference_key,
)
from bio_neb.target_request.paper_text import (
    bare_contig_ok,
    change_bases,
    contig_before,
    directed_pair,
    genomic_change,
    is_bases,
    is_change,
    is_contig,
    pair_after,
    read,
    written_number,
)


def contig_name(contig: str) -> str:
    """The name two spellings of one contig share: ``chr4``, ``4`` and ``Chromosome 4``
    are ``4``; an accession is itself, case aside.

    Lane P's rule, moved with the reader as it ran (owner, #383: a move, not a
    rewrite). ADR 0049 would retire it (#467: no code equates contig spellings; the
    model names the sequence a written contig means); the owner keeps it, a decision
    row of the value guard (#486 Q4 a)."""

    return re.sub(r"^(?:chr(?:omosome)?)\s*", "", contig.strip(), flags=re.I).casefold()


# Each refusal code, as the repair explains it to the model.
REFUSALS = {
    "unknown_span": "no paragraph or table row has this span id; cite a span id that grep_paper, read_paper or filter_paper_rows gave",
    "unknown_column": "this span is no table row with a column of this name; name a column as filter_paper_rows gives it, or leave column out",
    "unknown_view": "no section, table, document or span of the paper has this id; outline_paper lists the sections, tables and documents, and every reading result names its spans",
    "written_not_in_span": "this span does not write this text exactly; give the value character for character as the span writes it (spaces, commas and case included), or cite the span that writes it",
    "not_read": "the span writes this text, but it is no value of this slot as written: a contig such as chr5 or NC_000005.10 (a bare name only alone in a cell or before :position), a position of whole digits, alleles as a directed pair such as C>G, a REF or ALT cell, an rsID, a change name such as c.200C>T or p.Arg67Cys, a gene symbol",
    "number_not_stated": "a position states its number too: the number its written digits spell, in order",
    "number_differs": "the stated number is not the number these written digits spell in order; give the number the written text spells",
    "alleles_without_direction": "REF and ALT need a direction mark (>, ->, →): a slash pair such as A/C is a genotype; state no alleles when the paper writes none",
    "alleles_from_change_name": "a change name's bases are not REF and ALT: c.200C>T names a change on a transcript, not the genome's bases; state the change name as a change, and alleles only from a table's REF and ALT cells or a written genomic pair such as chr5: 1,234,567:G>A",
    "alleles_incomplete": "give REF and ALT together, or neither",
    "alleles_disagree": "the finding states its alleles two ways that differ; state REF and ALT once, as the paper writes them for this variant",
    "locus_disagrees": "a genomic change name the finding states writes another position or allele pair than the finding's own; state one variant's values",
    "located_values_apart": "the contig, position and alleles are each written, but neither in one span nor down one column of one table; cite the one paragraph or row that states this variant's locus, or, in a table that lists each variant as a column, each value at its row's span with that same column",
    "contig_without_position": "a contig belongs to a position; give the position written with it, or leave the contig out",
    "value_not_stated": "the span writes this value with the finding's position, but the finding does not state it; state it as written, with its span",
    "one_value_per_slot": "one finding names one variant: state one contig, position, allele pair and gene (rsIDs and change names may be several)",
    "no_values": "state the values the paper writes for this finding, each with its span",
    "organism_not_written": "this span does not write this organism text exactly; give the organism character for character as the span writes it, and cite the span that writes it, or leave the organism out",
    "reference_not_written": "this span does not write this version name; give one version name exactly as written in name, the others the span writes for the same version in aliases, and cite the span that writes them; the version names the span writes inside it, if any, follow it",
    "nothing_proposed": "propose the areas the question needs, or say why in not_answered",
    "too_many_areas": f"propose at most {MAX_AREAS} areas; join areas the same views or values answer into one",
    "item_unanswered": "give the findings this item holds for the question, or say why in not_answered",
}

# Each reason a finding stays unresolved: the paper itself is unclear, so it is never
# asked again; a person decides it from its chunk.
UNRESOLVED = {
    "ambiguous_number": "the paper writes this number in a form with two readings (a decimal comma or a damaged group), so code keeps no value for it",
}

Outcome = Literal["accepted", "unresolved", "refused"]
_LOCUS_SLOTS = ("contig", "position", "alleles", "ref", "alt")


@dataclass(frozen=True)
class Judged:
    """One finding as code decides it: its outcome and reasons, the values it kept and
    the lines of the spans it cites, in order (each value's span, then ``spans``)."""

    stated: FindingPointer
    outcome: Outcome
    reasons: tuple[str, ...]
    values: tuple[KeptValue, ...]
    lines: tuple[SourceLine, ...]


def _bounded(text: str, at: int, written: str, slot: str) -> bool:
    """``written`` at ``at`` is a whole token of ``text``: no letter or digit runs on
    either side, and a number runs into no decimal part or longer group."""

    before = text[at - 1] if at else ""
    after = text[at + len(written) : at + len(written) + 2]
    if slot == "position":
        # Nor a group of a longer number: "373" of "86,770, 373", "1 234" of "1 234 567".
        group = r"(?:, ?|[ \u00a0\u2009\u202f])"
        return not (
            re.match(r"[\w.,]", before)
            or re.match(r"\w|[.,]\d", after)
            or re.search(rf"\d{group}$", text[:at])
            or re.match(rf"{group}\d{{3}}(?!\d)", text[at + len(written) :])
        )
    if slot == "gene":
        return not re.match(r"[A-Za-z0-9_-]", before) and not re.match(
            r"[A-Za-z0-9_-]", after
        )
    return not re.match(r"[A-Za-z0-9_]", before) and not re.match(
        r"[A-Za-z0-9_]", after
    )


def _found(place: SourceLine, written: str, slot: str) -> list[SourceLine]:
    """Every exact, whole-token occurrence of ``written`` in ``place``, matched as the
    model is shown the text (``shown``); each span is the paper's own text."""

    text, found = shown_to_model(place.text), []
    at = text.find(written)
    while at != -1:
        # Matched as shown, kept only where the paper's own text is the written text.
        raw = place.text[at : at + len(written)] == written
        if raw and _bounded(text, at, written, slot):
            found.append(
                place.model_copy(
                    update={
                        "start": place.start + at,
                        "end": place.start + at + len(written),
                        "text": place.text[at : at + len(written)],
                    }
                )
            )
        at = text.find(written, at + 1)
    return found


def _cells(unit: Unit) -> list[tuple[str, SourceLine]]:
    return [
        (column, cell)
        for column in unit.columns or ()
        if (cell := unit.cell(column)) is not None
    ]


def _places(
    unit: Unit, value: ValuePointer
) -> list[tuple[SourceLine, str | None]] | str:
    """Every place the span writes the value, with its column, or a refusal. In a table
    row the written text lies inside one cell, never across two."""

    if value.column is not None:
        cell = unit.cell(value.column)
        if cell is None:
            return f"unknown_column:{value.span}:{value.column}"
        places = [
            (found, value.column) for found in _found(cell, value.written, value.slot)
        ]
    elif unit.columns is not None:
        places = [
            (found, column)
            for column, cell in _cells(unit)
            for found in _found(cell, value.written, value.slot)
        ]
    else:
        places = [
            (found, None) for found in _found(unit.line, value.written, value.slot)
        ]
    return places or _not_found(unit, value)


def _not_found(unit: Unit, value: ValuePointer) -> str:
    """Why the written text is not where the value cites it: alleles read from a change
    name's bases get their own refusal (live h1 run 3, #403 C)."""

    pair = directed_pair(value.written)
    if value.slot == "alleles" and pair:
        named = [
            name
            for name in sorted(read(unit.line.text).changes, key=lambda n: (len(n), n))
            if change_bases(name) == pair
        ]
        if named:
            return f"alleles_from_change_name:{named[0]}"
    return f"written_not_in_span:{value.slot}={value.written} in {value.span}"


def _read(
    unit: Unit, value: ValuePointer, place: SourceLine, column: str | None
) -> tuple[int | str, str] | str:
    """The value the written text is in its slot and its form, or why it is none: a
    refusal, or ``ambiguous_number`` when the paper's form has two readings."""

    slot, written = value.slot, value.written
    detail = f"{slot}={written}"
    if slot == "position":
        number = written_number(written)
        if number is None or number.form == "scientific":
            return f"not_read:{detail}"
        if number.form == "ambiguous":
            return f"ambiguous_number:{detail}"
        if not isinstance(number.value, int) or number.value < 1:
            return f"not_read:{detail}"
        if value.number is None:
            return f"number_not_stated:{detail}"
        if value.number != number.value:
            return f"number_differs:{detail},number={value.number}"
        return number.value, number.form
    whole_cell = column is not None and unit.cell(column).text.strip() == written
    if slot == "contig":
        holder = unit.line if column is None else unit.cell(column)
        before = holder.text[: place.start - holder.start]
        after = holder.text[place.end - holder.start :]
        colon = re.match(r"\s?:\s?(?=\d)", after)
        bare = whole_cell or (
            colon is not None and bare_contig_ok(before, colon.group())
        )
        return (
            (written, "as_written")
            if is_contig(written, bare=bare)
            else (f"not_read:{detail}")
        )
    if slot == "alleles":
        if directed_pair(written):
            return written, "as_written"
        if re.fullmatch(r"[ACGTN]+\s?/\s?[ACGTN]+", written):
            return f"alleles_without_direction:{written}"
        return f"not_read:{detail}"
    if slot in ("ref", "alt"):
        # The model names which cells are REF and ALT, whatever their labels say; code
        # copies a whole cell of bases.
        if whole_cell and is_bases(written):
            return written, "as_written"
        return f"not_read:{detail}"
    if slot == "rsid":
        return (
            (written, "as_written")
            if RSID_RULE.stated(written) == (written.lower(),)
            else f"not_read:{detail}"
        )
    if slot == "change":
        genomic = genomic_change(written)
        if not is_change(written) or (genomic and genomic[1] < 1):
            return f"not_read:{detail}"
        return written, "as_written"
    return (
        (written, "as_written")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", written)
        else f"not_read:{detail}"
    )


_ROW_SPAN = re.compile(r"(t\d+):r\d+")


def _one_column(values: Sequence[ValuePointer]) -> bool:
    """The values sit in one column of one table, each in its own row: a table that lists
    each variant down a column (#404 L1, owner Q1)."""

    tables = {m[1] for v in values if (m := _ROW_SPAN.fullmatch(v.span))}
    columns = {v.column for v in values}
    return (
        len(tables) == 1
        and all(_ROW_SPAN.fullmatch(v.span) for v in values)
        and len(columns) == 1
        and None not in columns
    )


def _alleles(values: Sequence[KeptValue]) -> tuple[str, str] | None:
    by_slot = {v.slot: v for v in values}
    if "alleles" in by_slot:
        return directed_pair(by_slot["alleles"].written)
    if "ref" in by_slot and "alt" in by_slot:
        return by_slot["ref"].written, by_slot["alt"].written
    return None


def _finding_reasons(values: Sequence[KeptValue], stated: FindingPointer) -> list[str]:
    """The finding-level refusals of its kept values: one value per locus slot, the
    locus in one span, REF with ALT, a contig with its position."""

    reasons = []
    slots = [v.slot for v in stated.values]
    reasons += [
        f"one_value_per_slot:{slot}"
        for slot in dict.fromkeys(slots)
        if slot not in ("rsid", "change") and slots.count(slot) > 1
    ]
    if ("ref" in slots) != ("alt" in slots):
        reasons.append("alleles_incomplete")
    by_slot = {v.slot: v for v in values}
    pair = _alleles(values)
    if "alleles" in by_slot and "ref" in by_slot and "alt" in by_slot:
        two_ways = (by_slot["ref"].written, by_slot["alt"].written)
        if directed_pair(by_slot["alleles"].written) != two_ways:
            reasons.append(
                f"alleles_disagree:{by_slot['alleles'].written},"
                f"ref={two_ways[0]},alt={two_ways[1]}"
            )
    for change in (v for v in values if v.slot == "change"):
        genomic = genomic_change(change.written)
        if genomic is None:
            continue
        position = by_slot.get("position")
        if (position is not None and position.value != genomic[1]) or (
            pair is not None and pair != genomic[2:]
        ):
            reasons.append(f"locus_disagrees:{change.written}")
    locus = [v for v in stated.values if v.slot in _LOCUS_SLOTS]
    if len({v.span for v in locus}) > 1 and not _one_column(locus):
        reasons.append("located_values_apart")
    genomic = any(v.slot == "change" and genomic_change(v.written) for v in values)
    if "contig" in slots and "position" not in slots and not genomic:
        contig = next(v.written for v in stated.values if v.slot == "contig")
        reasons.append(f"contig_without_position:{contig}")
    return reasons


def _unstated(kept: Sequence[KeptValue], paper: Paper) -> list[str]:
    """The locus values the span writes with the kept position that the finding does
    not state: the contig before it (``chr11: 207,469``) or the allele pair after it
    (``207,469:A>C``) (Standards r4 S6, on spans). Code names no table cell REF or ALT
    (owner, #486 Q2 a)."""

    position = next((v for v in kept if v.slot == "position"), None)
    if position is None:
        return []
    slots = {v.slot for v in kept}
    unit = paper.unit(position.span)
    holder = unit.line if position.column is None else unit.cell(position.column)
    before = holder.text[: position.source.start - holder.start]
    after = holder.text[position.source.end - holder.start :]
    problems = []
    if "contig" not in slots and (contig := contig_before(before)):
        problems.append(f"value_not_stated:contig={contig}")
    if not slots & {"alleles", "ref", "alt"} and (pair := pair_after(after)):
        problems.append(f"value_not_stated:alleles={pair}")
    return problems


def judge(stated: FindingPointer, paper: Paper) -> Judged:
    """One finding decided on its own (#404 d2): each value's written text found exactly
    in its span and read into its slot. A failed check refuses the finding; a paper
    form with two readings leaves it unresolved; else it is accepted."""

    refused: list[str] = []
    unresolved: list[str] = []
    lines: list[SourceLine] = []
    if not stated.values:
        refused.append("no_values")
    # Each value's places the span writes it at and reads it in its slot.
    read_at: list[tuple[ValuePointer, list[KeptValue]]] = []
    for value in stated.values:
        unit = paper.unit(value.span)
        if unit is None:
            refused.append(f"unknown_span:{value.span}")
            continue
        lines.append(unit.line)
        places = _places(unit, value)
        if isinstance(places, str):
            refused.append(places)
            continue
        readings = [(p, c, _read(unit, value, p, c)) for p, c in places]
        good = [
            KeptValue(
                slot=value.slot,
                written=value.written,
                span=value.span,
                column=column,
                source=place,
                value=reading[0],
                form=reading[1],
            )
            for place, column, reading in readings
            if not isinstance(reading, str)
        ]
        if not good:
            reason = readings[0][2]
            (unresolved if reason.startswith("ambiguous_") else refused).append(reason)
            continue
        read_at.append((value, good))
    kept = _nearest(read_at)
    for span in stated.spans:
        unit = paper.unit(span)
        if unit is None:
            refused.append(f"unknown_span:{span}")
        else:
            lines.append(unit.line)
    if not refused:
        refused += _finding_reasons(kept, stated) or _unstated(kept, paper)
    outcome: Outcome = (
        "refused" if refused else ("unresolved" if unresolved else "accepted")
    )
    return Judged(
        stated=stated,
        outcome=outcome,
        reasons=tuple(dict.fromkeys(refused or unresolved)),
        values=tuple(kept),
        lines=tuple(dict.fromkeys(lines)),
    )


def _nearest(
    read_at: Sequence[tuple[ValuePointer, list[KeptValue]]],
) -> list[KeptValue]:
    """One place per value: where the span writes it once, there; where it writes it
    more than once (``PBP2 (Y127X) and PBP2 (Q30X)``), the place nearest the finding's
    values written once, else the first."""

    anchors = [places[0].source for _, places in read_at if len(places) == 1]

    def distance(kept: KeptValue) -> int:
        return min(
            (
                abs(kept.source.start - anchor.start)
                for anchor in anchors
                if anchor.document_id == kept.source.document_id
            ),
            default=0,
        )

    return [min(places, key=distance) for _, places in read_at]


def judge_all(stated: Sequence[FindingPointer], paper: Paper) -> tuple[Judged, ...]:
    return tuple(judge(finding, paper) for finding in stated)


def _repair_line(problem: str) -> str:
    """One refusal as sent: the refusal with its detail, then its explanation."""

    code = re.match(r"[a-z_]+", problem).group()
    return f"- {problem} ({REFUSALS[code]})"


def stated_text(finding: FindingPointer) -> str:
    """A finding's stated values as the repair and a re-entered item show them."""

    return ", ".join(
        f'{v.slot} "{shown_to_model(v.written)}" in {v.span}'
        + (f" column {v.column}" if v.column else "")
        for v in finding.values
    )


def repair_text(problems: Sequence[str]) -> str:
    """The repair of a whole output (a plan, or an item that answered nothing): each
    refusal with its explanation, and a request for the whole output again."""

    return "\n".join(
        [
            "Code cannot keep this answer from the paper. Call the output Tool again with "
            "the whole answer and the corrections below. The same call again is refused "
            "again.",
            *(_repair_line(problem) for problem in problems),
        ]
    )


def findings_repair(
    pending: Sequence[Judged], *, kept: int = 0, unclear: int = 0
) -> str:
    """The repair of an item's refused findings only (#404 d2): how many findings code
    keeps or leaves unresolved, never to be sent again, and each refused finding still
    pending, numbered for the correction that names it (``corrects``), with what it
    stated and each refusal. Leaving one out withdraws nothing: it stays refused."""

    lines = [
        (
            "Code keeps 1 finding of this item: do not send it again."
            if kept == 1
            else f"Code keeps {kept} findings of this item: do not send them again."
        )
        if kept
        else "Code keeps no finding of this item yet."
    ]
    if unclear:
        lines.append(
            f"{unclear} finding{'s are' if unclear > 1 else ' is'} unresolved: the paper "
            "itself is unclear; code records them for a person, and they need no repair."
        )
    lines.append(
        "Call the output Tool again with a corrected finding for each refused finding "
        "below, each giving every value it states and corrects, the refused finding's "
        "number. A refused finding you cannot correct, leave out: code records it as "
        "refused with its reasons. The same call again is refused again."
    )
    for number, one in enumerate(pending, start=1):
        lines.append(f"Refused {number} ({stated_text(one.stated)}):")
        lines += [_repair_line(problem) for problem in one.reasons]
    return "\n".join(lines)


def alias_written(alias: str, text: str) -> bool:
    """``text`` writes ``alias`` whole, bounded by no letter or digit: the one rule for an
    alias, at plan admission and in ``verify`` (c3 live run 1: "IRGSP-1.0, release 52";
    #411 LF1, round 3 S1/P5). The recorded name keeps the one-version rule."""

    return re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text) is not None


def reference_problems(pointer: ReferencePointer | None, paper: Paper) -> list[str]:
    """Why the reference pointer names no version its span writes."""

    if pointer is None:
        return []
    unit = paper.unit(pointer.span)
    if unit is None:
        return [f"unknown_span:{pointer.span}"]
    text = unit.line.text
    versions = read(text).versions
    # The name is one version the span writes, and so is each alias the span does not
    # write whole; else the versions the span writes inside it follow it (UCSC
    # sacCer1: sacCer1).
    return [
        ":".join(
            ["reference_not_written", name, *sorted(v for v in versions if v in name)]
        )
        for name, alias in (
            (pointer.name, False),
            *((a, True) for a in pointer.aliases),
        )
        if name not in versions and not (alias and alias_written(name, text))
    ]


def stated_reference(
    pointer: ReferencePointer | None, paper: Paper
) -> StatedReference | None:
    """The version as the paper writes it, with the line that writes it and the reason."""

    if pointer is None:
        return None
    return StatedReference(
        candidate=ReferenceCandidate(
            candidate_id=candidate_id("r", reference_key(pointer.name)),
            name=pointer.name,
            spellings=(pointer.name,),
            sources=(paper.unit(pointer.span).line,),
        ),
        reason=pointer.reason,
        aliases=pointer.aliases,
    )


def organism_problems(pointer: OrganismPointer | None, paper: Paper) -> list[str]:
    """Why code cannot keep the organism the plan states: its span is no paragraph or
    row of the paper, or the span does not write its text as a whole word."""

    if pointer is None:
        return []
    unit = paper.unit(pointer.span)
    if unit is None:
        return [f"unknown_span:{pointer.span}"]
    if not _found(unit.line, pointer.written, "organism"):
        return [f"organism_not_written:{pointer.written} in {pointer.span}"]
    return []


def stated_organism(
    pointer: OrganismPointer | None, paper: Paper
) -> StatedOrganism | None:
    """The organism as the span writes it: its first whole-word occurrence there, and
    the span's paragraph or row, both copied from the paper (no taxon is named)."""

    if pointer is None:
        return None
    line = paper.unit(pointer.span).line
    return StatedOrganism(
        written=pointer.written,
        span=pointer.span,
        source=_found(line, pointer.written, "organism")[0],
        passage=line,
    )


def _quote(
    values: Sequence[KeptValue],
    lines: Sequence[SourceLine],
    others: Sequence[SourceLine] = (),
) -> SourceLine:
    """The paper's own text around the variant's first change name, else its position,
    else its first locus value, else its first value, out to the finding's other values in
    that span (code chooses it, the model copies nothing; #404 d1). It stops before a
    value whose way there crosses one of ``others``, the words other findings keep
    (#411 O-a: c2's SV2 shares SV1's contig). Its change names are what
    ``paper-targets/1``'s change reads."""

    changes = [v for v in values if v.slot == "change"]
    positions = [v for v in values if v.slot == "position"]
    locus = [v for v in values if v.slot in _LOCUS_SLOTS]
    anchor = (changes or positions or locus or values)[0]
    line = max(
        (
            line
            for line in lines
            if line.document_id == anchor.source.document_id
            and line.start <= anchor.source.start <= line.end
        ),
        key=lambda line: line.end - line.start,
    )

    def crosses(low: int, high: int) -> bool:
        return any(
            o.document_id == line.document_id and low <= o.start and o.end <= high
            for o in others
        )

    start, end = anchor.source.start, anchor.source.end
    places = sorted(
        (v.source for v in values if v.span == anchor.span),
        key=lambda p: p.start,
    )
    for place in (p for p in places if p.start >= end):
        if crosses(end, place.start):
            break
        end = max(end, place.end)
    for place in (p for p in reversed(places) if p.end <= start):
        if crosses(place.end, start):
            break
        start = place.start
    end += _closing(
        line.text[start - line.start : end - line.start], line.text[end - line.start :]
    )
    return line.model_copy(
        update={
            "start": start,
            "end": end,
            "text": line.text[start - line.start : end - line.start],
        }
    )


def _closing(quote: str, after: str) -> int:
    """How many of the brackets the quote opens and leaves open the line closes right
    after it (``PBP2 (Y127X`` and ``)``)."""

    open_ = []
    for char in quote:
        if char in "([":
            open_.append(char)
        elif char in ")]" and open_:
            open_.pop()
    closed = 0
    for char in after:
        if not open_ or char != {"(": ")", "[": "]"}[open_[-1]]:
            break
        open_.pop()
        closed += 1
    return closed


def _variant(judged: Judged) -> VariantCandidate | None:
    """The one variant the kept values make, with the lines that write them (the line
    stating the locus first); None when it states no position, change name or rsID. A
    genomic change name gives the locus it writes."""

    values = judged.values
    by_slot = {v.slot: v for v in values}
    contig = by_slot["contig"].written if "contig" in by_slot else None
    position = by_slot["position"].value if "position" in by_slot else None
    alleles = _alleles(values)
    changes = tuple(dict.fromkeys(v.written for v in values if v.slot == "change"))
    # An rsID in main's spelling (RS1800562 is rs1800562); its written text stays kept.
    rsids = tuple(
        dict.fromkeys(
            RSID_RULE.stated(v.written)[0] for v in values if v.slot == "rsid"
        )
    )
    if position is None:
        genomic = next(
            (
                g
                for v in values
                if v.slot == "change"
                if (g := genomic_change(v.written))
            ),
            None,
        )
        if genomic is not None:
            accession, position, ref, alt = genomic
            contig = contig or accession
            alleles = alleles or (ref, alt)
    if position is None and not (changes or rsids):
        return None
    fields = {
        "contig": contig,
        "position": position,
        "ref": alleles[0] if alleles else None,
        "alt": alleles[1] if alleles else None,
        "rsids": rsids,
        "hgvs": changes,
    }

    def line_of(value: KeptValue) -> SourceLine:
        return next(
            line
            for line in judged.lines
            if line.document_id == value.source.document_id
            and line.start <= value.source.start
            and value.source.end <= line.end
        )

    ordered = sorted(values, key=lambda v: v.slot not in _LOCUS_SLOTS)
    return VariantCandidate(
        candidate_id=candidate_id("v", VariantCandidate.model_construct(**fields).key),
        **fields,
        sources=tuple(
            dict.fromkeys(
                line_of(v)
                for v in ordered
                if v.slot in (*_LOCUS_SLOTS, "rsid", "change")
            )
        ),
    )


def kept_finding(judged: Judged) -> KeptFinding:
    """An accepted finding as the paper writes it: its quote (code-chosen), then each
    line it cites, the variant its kept values make, its gene and the model's reason."""

    variant = _variant(judged)
    gene = next((v.written for v in judged.values if v.slot == "gene"), None)
    return KeptFinding(
        variants=(variant,) if variant else (),
        passages=tuple(
            dict.fromkeys((_quote(judged.values, judged.lines), *judged.lines))
        ),
        reason=judged.stated.reason,
        gene=gene,
        values=judged.values,
    )


def finding_outcome(judged: Judged) -> FindingOutcome:
    """A finding code did not accept, kept with its reasons and its chunk for a person."""

    return FindingOutcome(
        outcome=judged.outcome,
        reasons=judged.reasons,
        stated=judged.stated,
        passages=judged.lines,
        values=judged.values,
    )


def _connected(id_sets: Sequence[set[str]]) -> list[list[int]]:
    """The indexes of the sets that share an id, directly or through another set: each
    group in first-seen order, the groups by their first member."""

    groups: list[list[int]] = []
    for index, ids in enumerate(id_sets):
        joined = [g for g in groups if any(ids & id_sets[i] for i in g)]
        for group in joined:
            groups.remove(group)
        groups.append(sorted([*(i for g in joined for i in g), index]))
    return sorted(groups, key=lambda group: group[0])


def _locus_keys(findings: Sequence[Finding]) -> list[set[str]]:
    """Each finding's ids, and the one locus it states: contig (``chr`` aside) and
    position, with its REF and ALT, or with the one allele pair any finding states there
    when it states none. A locus stated without alleles where two pairs are stated is
    joined to neither."""

    def locus(variant: VariantCandidate) -> tuple[str, int] | None:
        if variant.position is None:
            return None
        return contig_name(variant.contig or ""), variant.position

    pairs: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for variant in (v for f in findings for v in f.variants):
        if locus(variant) is not None and variant.ref is not None:
            pairs.setdefault(locus(variant), set()).add((variant.ref, variant.alt))
    keys = []
    for finding in findings:
        own = {v.candidate_id for v in finding.variants}
        for variant in finding.variants:
            at = locus(variant)
            if at is None:
                continue
            stated = pairs.get(at, set())
            if variant.ref is not None:
                own.add(f"locus:{at}:{variant.ref}>{variant.alt}")
            elif len(stated) == 1:
                ref, alt = next(iter(stated))
                own.add(f"locus:{at}:{ref}>{alt}")
            elif not stated:
                own.add(f"locus:{at}")
        keys.append(own)
    return keys


def merged_findings(findings: Sequence[KeptFinding]) -> tuple[KeptFinding, ...]:
    """One variant is one finding: findings whose stated values make the same variant (one
    id, derived from the values), or state one locus where one less fully stated fits the
    one allele pair stated there, become one, in first-seen order, with every variant,
    line, passage, value and reason of each. The first finding's quote stays first."""

    def one(group: list[KeptFinding]) -> KeptFinding:
        if len(group) == 1:
            return group[0]
        variants: dict[str, VariantCandidate] = {}
        for variant in (v for f in group for v in f.variants):
            earlier = variants.get(variant.candidate_id)
            variants[variant.candidate_id] = (
                variant
                if earlier is None
                else earlier.model_copy(
                    # One variant's values agree by its id; its identifiers and lines
                    # are each finding's (Standards r1 S2).
                    update={
                        name: tuple(
                            dict.fromkeys(
                                (*getattr(earlier, name), *getattr(variant, name))
                            )
                        )
                        for name in ("rsids", "hgvs", "sources")
                    }
                )
            )
        return KeptFinding(
            variants=tuple(variants.values()),
            passages=tuple(dict.fromkeys(p for f in group for p in f.passages)),
            reason=" ".join(dict.fromkeys(f.reason for f in group)),
            gene=next((f.gene for f in group if f.gene), None),
            values=tuple(dict.fromkeys(v for f in group for v in f.values)),
        )

    groups = _connected(_locus_keys(findings))
    return tuple(one([findings[i] for i in group]) for group in groups)


def _bounded_quotes(findings: Sequence[KeptFinding]) -> tuple[KeptFinding, ...]:
    """Each finding's quote chosen again where it holds words another finding keeps and
    it does not, so it stops at its own variant's words (#411 O-a). A quote that is no
    whole line it cites is replaced; the lines stay."""

    bounded = []
    for index, finding in enumerate(findings):
        own = {v.source for v in finding.values}
        others = [
            v.source
            for i, other in enumerate(findings)
            if i != index
            for v in other.values
            if v.source not in own
        ]
        quote, *rest = finding.passages
        if not any(
            o.document_id == quote.document_id
            and quote.start <= o.start
            and o.end <= quote.end
            for o in others
        ):
            bounded.append(finding)
            continue
        whole = quote not in rest and not any(
            p.document_id == quote.document_id
            and p.start <= quote.start
            and quote.end <= p.end
            for p in rest
        )
        lines = (quote, *rest) if whole else tuple(rest)
        bounded.append(
            finding.model_copy(
                update={
                    "passages": tuple(
                        dict.fromkeys((_quote(finding.values, lines, others), *lines))
                    )
                }
            )
        )
    return tuple(bounded)


# The value slots a quote may be taken from when two variants share one, in this order.
_QUOTE_SLOTS = ("position", "contig", "ref", "alt", "alleles", "change", "gene", "rsid")


def _distinct_quotes(
    findings: Sequence[KeptFinding],
) -> tuple[tuple[KeptFinding, ...], list[str]]:
    """Main joins targets by their words: where findings' quotes are one text but their
    variants differ (two table columns writing one locus), each takes the first of its
    own kept values whose slot and text no other of them shares and whose text no other
    finding's quote is, still the paper's own text (Standards live2-3 S8, Spec P4).
    Where no exact text tells two variants apart, it says so instead of merging them
    silently (owner, 2026-10-20)."""

    def locus(finding: KeptFinding) -> frozenset[str]:
        return frozenset(v.key for v in finding.variants)

    chosen = list(findings)
    notes: list[str] = []
    for _ in range(len(findings)):
        by_text: dict[str, list[int]] = {}
        for index, finding in enumerate(chosen):
            by_text.setdefault(finding.passages[0].text, []).append(index)
        clash = next(
            (g for g in by_text.values() if len({locus(chosen[i]) for i in g}) > 1),
            None,
        )
        if clash is None:
            return tuple(chosen), notes
        moved = False
        for index in clash:
            finding = chosen[index]
            taken = {f.passages[0].text for i, f in enumerate(chosen) if i != index}
            shared = {
                (v.slot, v.source.text)
                for i in clash
                if i != index
                for v in chosen[i].values
            }
            own = next(
                (
                    v.source
                    for slot in _QUOTE_SLOTS
                    for v in finding.values
                    if v.slot == slot
                    and (v.slot, v.source.text) not in shared
                    and v.source.text not in taken
                ),
                None,
            )
            if own is not None:
                chosen[index] = finding.model_copy(
                    update={"passages": tuple(dict.fromkeys((own, *finding.passages)))}
                )
                moved = True
                break
        if not moved:
            numbers = ", ".join(str(i + 1) for i in clash)
            notes.append(
                f"findings {numbers} state different variants that no exact text of the "
                "paper tells apart, so their targets share one request"
            )
            return tuple(chosen), notes
    return tuple(chosen), notes


def paper_answer(
    question: str,
    paper: Paper,
    reference: StatedReference | None,
    findings: Sequence[KeptFinding],
    not_answered: str | None,
    unresolved: Sequence[FindingOutcome] = (),
    refused: Sequence[FindingOutcome] = (),
    organism: StatedOrganism | None = None,
) -> PaperAnswer:
    """The answer with the documents its reference, organism, findings and outcomes
    cite, each by its stored text's digest."""

    return answer_of(
        question,
        paper.identity,
        paper.listing,
        reference,
        findings,
        not_answered,
        unresolved,
        refused,
        organism=organism,
    )


def answer_of(
    question: str,
    identity: PaperIdentity | None,
    listing: Sequence[ReadDocument],
    reference: StatedReference | None,
    findings: Sequence[KeptFinding],
    not_answered: str | None,
    unresolved: Sequence[FindingOutcome] = (),
    refused: Sequence[FindingOutcome] = (),
    organism: StatedOrganism | None = None,
) -> PaperAnswer:
    """``paper_answer`` from the paper's identity and its documents' listing alone, as
    a reading's fold holds them: no text is read here."""

    cited = {
        line.document_id
        for lines in (
            *([reference.candidate.sources] if reference else []),
            *([(organism.passage,)] if organism else []),
            *(v.sources for f in findings for v in f.variants),
            *(f.passages for f in (*findings, *unresolved, *refused)),
        )
        for line in lines
    }
    distinct, notes = _distinct_quotes(_bounded_quotes(findings))
    not_answered = "; ".join(n for n in (not_answered, *notes) if n)
    return PaperAnswer(
        question=question,
        paper=identity,
        documents=tuple(d for d in listing if d.document_id in cited),
        reference=reference,
        organism=organism,
        findings=distinct,
        unresolved=tuple(unresolved),
        refused=tuple(refused),
        not_answered=not_answered or None,
    )


def _exact(paper: Paper, line: SourceLine) -> bool:
    """The line is exactly ``start:end`` of its document, inside one paragraph or row."""

    holding = passage(paper, line.document_id, line.start)
    return (
        holding is not None
        and paper.documents[line.document_id].text[line.start : line.end] == line.text
        and line.end <= holding.end
    )


def _value_problems(value: KeptValue, paper: Paper) -> list[str]:
    """Why a recorded value is not what its span writes: its place, or its reading."""

    unit = paper.unit(value.span)
    if unit is None:
        return [f"unknown_span:{value.span}"]
    if not _exact(paper, value.source):
        return [f"passage_differs:{value.source.document_id}@{value.source.start}"]
    holder = unit.line if value.column is None else unit.cell(value.column)
    if holder is None or not (
        holder.document_id == value.source.document_id
        and holder.start <= value.source.start
        and value.source.end <= holder.end
    ):
        return [f"value_differs:{value.slot}={value.written}"]
    stated = ValuePointer(
        slot=value.slot,
        written=value.written,
        span=value.span,
        column=value.column,
        number=value.value if value.slot == "position" else None,
    )
    # The place must be one judge finds: a whole token, in its cell (Standards r1 S2).
    places = _places(unit, stated)
    if isinstance(places, str):
        return [places]
    if value.source not in [place for place, _ in places]:
        return [f"written_not_in_span:{value.slot}={value.written} in {value.span}"]
    reading = _read(unit, stated, value.source, value.column)
    if isinstance(reading, str):
        return [reading]
    if reading != (value.value, value.form):
        return [f"value_differs:{value.slot}={value.written}"]
    return []


def _projection_problems(finding: KeptFinding) -> list[str]:
    """Each stored variant field and the gene a kept value of the finding supports:
    the position, contig and allele pair stated or a genomic change's, each rsID in
    main's spelling, each change name (Standards r1 S1)."""

    if not finding.values:
        return ["values_missing"]  # an accepted finding keeps at least one value
    values = finding.values
    genomic = [
        g for v in values if v.slot == "change" if (g := genomic_change(v.written))
    ]
    positions = {v.value for v in values if v.slot == "position"} | {
        g[1] for g in genomic
    }
    contigs = {v.written for v in values if v.slot == "contig"} | {
        g[0] for g in genomic if g[0]
    }
    pairs = {g[2:] for g in genomic}
    pairs |= {
        p for v in values if v.slot == "alleles" if (p := directed_pair(v.written))
    }
    refs = [v.written for v in values if v.slot == "ref"]
    alts = [v.written for v in values if v.slot == "alt"]
    pairs |= {(r, a) for r in refs for a in alts}
    # Only values code can read support a projection (Standards r2 S6).
    rsids = {
        r for v in values if v.slot == "rsid" for r in RSID_RULE.stated(v.written)[:1]
    }
    changes = {v.written for v in values if v.slot == "change"}
    genes = {v.written for v in values if v.slot == "gene"}
    problems = []
    for variant in finding.variants:
        if variant.position is not None and variant.position not in positions:
            problems.append(f"variant_unsupported:position={variant.position}")
        if variant.contig is not None and variant.contig not in contigs:
            problems.append(f"variant_unsupported:contig={variant.contig}")
        if (variant.ref, variant.alt) != (None, None) and (
            variant.ref,
            variant.alt,
        ) not in pairs:
            problems.append(f"variant_unsupported:alleles={variant.ref}>{variant.alt}")
        problems += [
            f"variant_unsupported:rsid={r}" for r in variant.rsids if r not in rsids
        ]
        problems += [
            f"variant_unsupported:change={c}" for c in variant.hgvs if c not in changes
        ]
    if finding.gene is not None and finding.gene not in genes:
        problems.append(f"variant_unsupported:gene={finding.gene}")
    return problems


def verify(answer: PaperAnswer, paper: Paper) -> Verification:
    """Check a recorded answer against the paper's documents, with no model.

    Every passage, source line and kept value must be exactly its document's text at its
    place, inside one paragraph or row; every kept value must still be read where it is,
    as recorded (its value and form); the reference's name must be read in its first
    line; each cited document must be the paper's, the same file and length however it
    was fetched (an answer from the live fetch verifies against its replay). Every
    mismatch is a problem, ``code:finding:pointer`` (the reference is finding
    ``reference``)."""

    # A document is the same stored text by its digest, however a reading reached it.
    listed = {
        d.document_id: d.model_dump(exclude={"read_event_id"}) for d in paper.listing
    }
    problems = []
    if answer.reference:
        candidate = answer.reference.candidate
        for line in candidate.sources:
            if not _exact(paper, line):
                problems.append(
                    f"passage_differs:reference:{line.document_id}@{line.start}"
                )
        if candidate.name not in read(candidate.sources[0].text).versions:
            problems.append(f"reference_not_written:reference:{candidate.name}")
        problems += [
            f"reference_not_written:reference:alias={alias}"
            for alias in answer.reference.aliases
            if not alias_written(alias, candidate.sources[0].text)
        ]
    if answer.organism is not None:
        organism = answer.organism
        problems += [
            f"passage_differs:organism:{line.document_id}@{line.start}"
            for line in (organism.source, organism.passage)
            if not _exact(paper, line)
        ]
        unit = paper.unit(organism.span)
        if (
            unit is None
            or unit.line != organism.passage
            or organism.source not in _found(unit.line, organism.written, "organism")
        ):
            problems.append(f"organism_not_written:organism:{organism.written}")
    for number, finding in enumerate(answer.findings, start=1):
        lines = [*finding.passages, *(s for v in finding.variants for s in v.sources)]
        problems += [
            f"passage_differs:{number}:{line.document_id}@{line.start}"
            for line in dict.fromkeys(lines)
            if not _exact(paper, line)
        ]
        problems += [
            f"{problem}:{number}"
            for value in finding.values
            for problem in _value_problems(value, paper)
        ]
        problems += [f"{problem}:{number}" for problem in _projection_problems(finding)]
    for kind, outcomes in (
        ("unresolved", answer.unresolved),
        ("refused", answer.refused),
    ):
        for number, outcome in enumerate(outcomes, start=1):
            label = f"{kind} {number}"
            problems += [
                f"passage_differs:{label}:{line.document_id}@{line.start}"
                for line in outcome.passages
                if not _exact(paper, line)
            ]
            problems += [
                f"{problem}:{label}"
                for value in outcome.values
                for problem in _value_problems(value, paper)
            ]
    problems += [
        f"document_differs:{d.document_id}"
        for d in answer.documents
        if listed.get(d.document_id) != d.model_dump(exclude={"read_event_id"})
    ]
    return Verification(
        verified=not problems,
        findings=len(answer.findings),
        problems=tuple(dict.fromkeys(problems)),
    )


# ---------------------------------------------------------------------------
# The evidence view: each cited span with every value marked by slot
# ---------------------------------------------------------------------------


def _marked(line: SourceLine, values: Iterable[KeptValue]) -> str:
    """The span's exact text with each kept value marked ``==written==``, line breaks
    shown as spaces; overlapping values keep the first."""

    marks, end = [], line.start
    for value in sorted(values, key=lambda v: (v.source.start, -v.source.end)):
        if value.source.start < end or not (
            line.start <= value.source.start and value.source.end <= line.end
        ):
            continue
        marks.append(value.source)
        end = value.source.end
    out, at = [], line.start
    for source in marks:
        out += [
            line.text[at - line.start : source.start - line.start],
            f"=={source.text}==",
        ]
        at = source.end
    out.append(line.text[at - line.start :])
    return " ".join("".join(out).split())


def _value_text(value: KeptValue) -> str:
    shown_value = f"**{value.value}**"
    written = (
        f' (written "{value.written}"'
        + (f", {value.form}" if value.form != "as_written" else "")
        + ")"
        if str(value.value) != value.written or value.form != "as_written"
        else ""
    )
    column = f" · column {value.column}" if value.column else ""
    return f"{value.slot} {shown_value}{written}{column}"


def _card(
    line: SourceLine,
    span: str,
    citing: Sequence[tuple[str, Sequence[KeptValue]]],
) -> list[str]:
    """One cited span: its id, document and section and the findings citing it; its
    text with every kept value marked; then one line per finding with each slot."""

    values = [v for _, vs in citing for v in vs]
    out = [
        f"#### `{span}` · `{line.document_id}` · {line.section} · "
        + ", ".join(label.removeprefix("accepted ") for label, _ in citing),
        "",
        f"> {_marked(line, values)}",
        "",
    ]
    for label, own in citing:
        here = [v for v in own if v.span == span]
        out.append(
            f"- {label}: "
            + (" · ".join(_value_text(v) for v in here) or "no value in this span")
        )
    return [*out, ""]


def _span_of(line: SourceLine, values: Sequence[KeptValue], paper: Paper | None) -> str:
    """The span id of a cited line: the one the paper issues for it, else the one its
    values cite, else its document and offset."""

    if paper is not None and line.document_id in paper.lines:
        return paper.span_of(
            line.document_id, paper.lines[line.document_id].at(line.start)
        )
    return next(
        (
            v.span
            for v in values
            if v.source.document_id == line.document_id
            and line.start <= v.source.start <= line.end
        ),
        f"{line.document_id}@{line.start}",
    )


def _whole(lines: Sequence[SourceLine]) -> list[SourceLine]:
    """The cited spans among a finding's lines: each line no other of its lines holds (a
    quote lies inside its span's line, unless it is the whole line)."""

    def inside(line: SourceLine, other: SourceLine) -> bool:
        return (
            other != line
            and other.document_id == line.document_id
            and other.start <= line.start
            and line.end <= other.end
        )

    return [line for line in lines if not any(inside(line, o) for o in lines)]


def _cards(answer: PaperAnswer, paper: Paper | None) -> list[str]:
    """Every span the answer cites, once, in first-cited order, as a card."""

    entries = [
        (f"accepted finding {n}", f.values, f.passages)
        for n, f in enumerate(answer.findings, start=1)
    ]
    for kind, outcomes in (
        ("unresolved", answer.unresolved),
        ("refused", answer.refused),
    ):
        entries += [
            (f"{kind} {n}", o.values, o.passages)
            for n, o in enumerate(outcomes, start=1)
        ]
    spans: dict[tuple[str, int], tuple[SourceLine, str, list]] = {}
    for label, values, lines in entries:
        for line in _whole(lines):
            key = (line.document_id, line.start)
            if key not in spans:
                spans[key] = (line, _span_of(line, values, paper), [])
            if not any(c[0] == label for c in spans[key][2]):
                spans[key][2].append((label, values))
    out = []
    for line, span, citing in spans.values():
        out += _card(line, span, citing)
    return out


def answer_markdown(
    answer: PaperAnswer | None, stopped: Sequence[str] = (), paper: Paper | None = None
) -> str:
    """The answer as a researcher reads it: the reference, each finding's values, and
    the evidence: each span the answer cites with every value marked at its place and
    labelled by slot (#404 d8), unresolved and refused findings listed with their
    reasons, and each reading item that stopped without its answer."""

    stops = (
        ["## Stopped", "", *(f"- {item}" for item in stopped), ""] if stopped else []
    )
    if answer is None:
        return "\n".join(["# No answer", "", *stops])
    identity = answer.paper
    out = [f"# {answer.question}", ""]
    if identity:
        out += [
            f"Paper: {identity.title or ''} (doi:{identity.doi}, {identity.pmcid})",
            "",
        ]
    out += ["## Reference as the paper states it", ""]
    if answer.reference:
        source = answer.reference.candidate.sources[0]
        out += [
            f"**{answer.reference.candidate.name}**",
            f"- Reason: {answer.reference.reason}",
            f"- Line: `{source.document_id}` · {source.section} · "
            + " ".join(source.text.split()),
        ]
    else:
        out.append("None given.")
    if answer.organism:
        out += [
            "",
            "## Organism as the paper writes it",
            "",
            f"**{answer.organism.written}** in `{answer.organism.span}` · "
            + " ".join(answer.organism.passage.text.split()),
        ]
    out += ["", f"## Findings ({len(answer.findings)})", ""]
    for number, finding in enumerate(answer.findings, start=1):
        out += [
            f"### {number}. {finding.gene or 'Variant'}",
            f"- Reason: {finding.reason}",
        ]
        out.append(f"- Quote: {' '.join(finding.passages[0].text.split())}")
        out += [f"- {_value_text(v)} in `{v.span}`" for v in finding.values]
        out.append("")
    for name, outcomes in (
        ("Unresolved", answer.unresolved),
        ("Refused", answer.refused),
    ):
        if not outcomes:
            continue
        out += [f"## {name} ({len(outcomes)})", ""]
        for number, outcome in enumerate(outcomes, start=1):
            out.append(f"### {name.lower()} {number}: {outcome.stated.reason}")
            out += [f"- Reason: {reason}" for reason in outcome.reasons]
            out += [f"- Stated: {stated_text(outcome.stated)}", ""]
    cards = _cards(answer, paper)
    if cards:
        out += ["## Evidence", "", *cards]
    if answer.not_answered:
        out += ["## Not answered", "", answer.not_answered, ""]
    return "\n".join([*out, *stops])
