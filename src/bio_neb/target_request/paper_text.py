"""The general parser: one passage read into the typed slots the next readers check.

The model reads the paper and states each value; code keeps a value only where this parser
reads it in a passage the model points at (owner, #395 decisions 1 and 2). The slots are the
ones main's request item and lane Rf's change type, not any paper's layout:

- a position: a whole number as written, with or without thousands separators
  (``150951552``, ``207,469``); a decimal (``54.1 Mb``) is not a position;
- a locus: a contig and a position written together (``chr11: 207,469``, ``4:54108297``);
- a contig: ``chr``/``chromosome`` and a chromosome name, a sequence accession
  (``NC_006592.3``, ``CM009151.1``), a chromosome name before ``:position``, or a table cell
  holding only a chromosome name;
- REF and ALT: a pair of base runs written ``A>G``, ``A->G`` or ``A→G``, or a genomic change's
  pair (``g.54108297G>T``). Base cells alone state no direction, whatever their columns are
  named: the model names a table's REF and ALT cells and code copies them (owner, #486 Q2 a).
  Neither does a slash: ``A/C`` is a genotype (#396). A coding, transcript or protein change
  states no genome bases, so its pair is never read as REF and ALT;
- an rsID, by main's rsID rule;
- a change as written: an HGVS-style name (``c.``, ``g.``, ``n.``, ``m.``, ``r.``, ``p.``,
  with or without its accession) or a protein change in one- or three-letter code
  (``Y127X``, ``Ser875*``);
- a coding transcript id (RefSeq ``NM_``/``XM_``, Ensembl ``ENST``; owner #351 q10);
- a genome version name by shape: a token with a letter and a digit (``Hg38``, ``EquCab3.0``,
  ``GCF_000001405.40``) or a name and a dotted number (``CanFam 3.1``), in any case. Code does
  not decide that a token is a genome version; the model does, and lane Rf resolves it.

A researcher's words for a target are not read here: the model types a prompt or Review
answer's site and code checks only that the quote holds it (#472, ADR 0049); this reader serves
paper passages until #383 brings paper targets to the model.

A table row (the document's tab-separated line) is read cell by cell. Every value keeps the paper's spelling; nothing is normalized
beyond reading a position as a number. No code says two spellings name one contig: a contig
keeps its written form, and the Run's lookup has the model name the sequence it means (#467,
ADR 0049).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from bio_neb.investigation.investigation_request import RSID_RULE

# The form a number is written in: as written, a damaged thousands group ("86,770,
# 373"), scientific notation, or ambiguous (no value).
NumberForm = Literal["as_written", "damaged_thousands_group", "scientific", "ambiguous"]


@dataclass(frozen=True)
class Readings:
    """Every value one passage writes, by slot."""

    positions: frozenset[int] = frozenset()
    contigs: frozenset[str] = frozenset()
    loci: frozenset[tuple[str, int]] = (
        frozenset()
    )  # a contig and position written together
    allele_pairs: frozenset[tuple[str, str]] = frozenset()
    rsids: frozenset[str] = frozenset()
    changes: frozenset[str] = frozenset()
    transcripts: frozenset[str] = frozenset()
    versions: frozenset[str] = frozenset()


@dataclass(frozen=True)
class WrittenNumber:
    """A number as the paper writes it: its value and the form it is written in."""

    value: int | Decimal | None
    form: NumberForm


# A group separator: a comma, or a space of any width; a comma with a space after it is a
# damaged one (c2's "86,770, 373"). A decimal point or an exponent is never a separator.
_GROUPS = re.compile(r"\d{1,3}(?:(?:, ?|[ \u00a0\u2009\u202f])\d{3})+", re.ASCII)
_SEPARATOR = re.compile(r"[, \u00a0\u2009\u202f]+")
_DECIMAL = re.compile(r"\d+\.\d+", re.ASCII)
_SCIENTIFIC = re.compile(r"\d+(?:\.\d+)?[eE][-+]?\d+", re.ASCII)
_LOOSE_GROUPS = re.compile(r"\d+(?:(?:, ?|[ \u00a0\u2009\u202f])\d+)+", re.ASCII)


def written_number(written: str) -> WrittenNumber | None:
    """The number ``written`` is, with its form; None when it is no number.

    Whole digits, or groups of three after a first group of one to three, are read
    ``as_written`` when one separator spells every group (``86,770,373``, ``86 770 373``)
    and as a ``damaged_thousands_group`` when a comma has a stray space after it
    (``86,770, 373``). Other groups (``3,01``) are ``ambiguous``: a decimal comma or a
    damaged group, so no value. A decimal is read as written; nothing is ever read by its
    digits alone, so ``3.01`` is never ``30.1`` or ``301``."""

    if re.match(r"0\d", written) and re.fullmatch(
        r"[\d, \u00a0\u2009\u202f]+", written, re.ASCII
    ):
        return WrittenNumber(None, "ambiguous")  # a leading zero: no number as written
    if re.fullmatch(r"\d+", written, re.ASCII):
        return WrittenNumber(int(written), "as_written")
    if _GROUPS.fullmatch(written):
        separators = set(_SEPARATOR.findall(written))
        form = (
            "as_written"
            if len(separators) == 1 and len(next(iter(separators))) == 1
            else ("damaged_thousands_group")
        )
        return WrittenNumber(int(re.sub(r"\D", "", written)), form)
    if _DECIMAL.fullmatch(written):
        return WrittenNumber(Decimal(written), "as_written")
    if _SCIENTIFIC.fullmatch(written):
        return WrittenNumber(Decimal(written), "scientific")
    if _LOOSE_GROUPS.fullmatch(written):
        return WrittenNumber(None, "ambiguous")
    return None


_BASES = r"[ACGTN]+"
_CHROMOSOME = r"(?:[1-9]\d?|[XYZWM]|MT|Un|[IVX]{1,5}|\d[LR])"
_SEQUENCE = r"(?:N[CTW]_\d+\.\d+|CM\d{6}\.\d+|[A-Z]{2}\d{6}\.\d+)"
_ACCESSION = (
    r"(?:N[CTWGMRP]_\d+(?:\.\d+)?|[NX][MR]_\d+\.\d+|ENS[TGP]\d+(?:\.\d+)?|CM\d{6}\.\d+)"
)

_NUCLEOTIDE_CHANGE = re.compile(
    rf"(?<![\w.])(?:(?P<accession>{_ACCESSION})[:(]?)?"
    r"(?P<kind>[cgnmr])\.(?P<body>[\d*(\-][\w*>+\-?]*)"
)
_GENOMIC_SUBSTITUTION = re.compile(rf"(\d+)({_BASES})>({_BASES})$")
_PROTEIN_NAME = re.compile(
    r"(?<![\w.])(?:NP_\d+\.\d+:)?p\.\s?"
    r"(?:\([^()\s]+\)|(?:[A-Z][a-z]{2}|[A-Z*])\d+[\w*=?]*)"
)
_PROTEIN_CHANGE = re.compile(
    r"(?<![\w.*])(?:[A-Z][a-z]{2}|[A-Z])\d+(?:[A-Z][a-z]{2}|[A-Z*]|fs\w*|del|ins\w*|dup)"
    r"(?![\w*])"
)
_PAIR = re.compile(
    rf"(?<![A-Za-z0-9])({_BASES})\s?(?:>|->|→)\s?({_BASES})(?![A-Za-z0-9])"
)
_POSITION = re.compile(r"(?<![\w.,])(\d{1,3}(?:[,  ]\d{3})+|\d+)(?!\w)(?![.,]\d)")
_NAMED_CONTIG = re.compile(
    rf"(?<!\w)(?P<written>(?:chr|Chr|CHR|chromosome|Chromosome)\s?(?P<name>{_CHROMOSOME}))"
    r"(?!\w)"
)
_SEQUENCE_CONTIG = re.compile(rf"(?<!\w){_SEQUENCE}(?!\w)")
_LOCUS = re.compile(
    rf"(?<![\w.])(?P<contig>(?:chr|Chr|CHR|chromosome|Chromosome)\s?{_CHROMOSOME}"
    rf"|{_SEQUENCE}|{_CHROMOSOME})\s?:\s?"
    r"(?P<position>\d{1,3}(?:[,\u2009\u202f]\d{3})+|\d+)(?!\d)(?![.,]\d)"
)
_LOCUS_CONTIG = re.compile(rf"(?<![\w.:])({_CHROMOSOME})(?=:\s?\d)")
_CELL_CONTIG = re.compile(rf"(?:chr)?{_CHROMOSOME}")
_TRANSCRIPT = re.compile(r"(?<!\w)(?:[NX]M_\d+\.\d+|ENST\d+(?:\.\d+)?)(?!\w)")
_VERSION_TOKEN = re.compile(r"(?<![\w.\-])[A-Za-z][\w.\-]*")
_SPACED_VERSION = re.compile(r"(?<!\w)[A-Za-z]{2,} \d+(?:\.\d+)+(?!\w)")


_CHANGE_BASES = re.compile(rf"({_BASES})>({_BASES})$")


def change_bases(name: str) -> tuple[str, str] | None:
    """The base pair a coding or non-coding transcript change writes (``c.1841C>T``: C
    and T). It states no genome bases, so it is never REF and ALT (#403 C)."""

    change = _NUCLEOTIDE_CHANGE.fullmatch(name.strip())
    if change is None or change.group("kind") not in "cn":
        return None
    bases = _CHANGE_BASES.search(change.group("body"))
    return (bases.group(1), bases.group(2)) if bases else None


def directed_pair(written: str) -> tuple[str, str] | None:
    """REF and ALT when ``written`` is one directed pair of base runs (``C>G``, ``A->G``,
    ``A→G``); a slash pair is a genotype and gives none (#396 D3)."""

    match = _PAIR.fullmatch(written)
    return (match[1], match[2]) if match and match[1] != match[2] else None


def genomic_change(name: str) -> tuple[str | None, int, str, str] | None:
    """The accession, position, REF and ALT a genomic substitution writes
    (``NC_000004.12:g.54108297G>T``); None for any other name."""

    change = _NUCLEOTIDE_CHANGE.fullmatch(name)
    if change is None or change.group("kind") not in "gm":
        return None
    genomic = _GENOMIC_SUBSTITUTION.match(change.group("body"))
    if genomic is None:
        return None
    return (change.group("accession"), int(genomic[1]), genomic[2], genomic[3])


def is_change(written: str) -> bool:
    """``written`` is one change name as a whole: an HGVS-style name or a protein change
    (``c.200C>T``, ``p. (Arg67Cys)``, ``W45X``)."""

    return written in read(written).changes


def is_contig(written: str, *, bare: bool) -> bool:
    """``written`` names a contig: ``chr``/``chromosome`` and a name, or a sequence
    accession; a bare chromosome name only where ``bare`` (a whole cell, or before
    ``:position``)."""

    named = _NAMED_CONTIG.fullmatch(written) or _SEQUENCE_CONTIG.fullmatch(written)
    return bool(named) or (bare and re.fullmatch(_CHROMOSOME, written) is not None)


_CONTIG_BEFORE = re.compile(
    rf"(?P<contig>(?:chr|Chr|CHR|chromosome|Chromosome)\s?{_CHROMOSOME}|{_SEQUENCE}"
    rf"|(?<![\w.]){_CHROMOSOME})(?P<colon>\s?:\s?)$"
)
_PAIR_AFTER = re.compile(
    rf"^\s?:?\s?({_BASES})\s?(?:>|->|→)\s?({_BASES})(?![A-Za-z0-9])"
)


def contig_before(text: str) -> str | None:
    """The contig written right before a position, as in ``chr11: 207,469``: the end of
    ``text`` is a contig and a colon."""

    match = _CONTIG_BEFORE.search(text)
    if match is None:
        return None
    bare = re.fullmatch(_CHROMOSOME, match["contig"]) is not None
    if bare and not bare_contig_ok(text[: match.start()], match["colon"]):
        return None
    return match["contig"]


def pair_after(text: str) -> str | None:
    """The directed allele pair written right after a position, as written
    (``207,469:A>C`` gives ``A>C``)."""

    match = _PAIR_AFTER.match(text)
    if match is None or match[1] == match[2]:
        return None
    return text[match.start(1) : match.end(2)]


def bare_contig_ok(before: str, colon: str) -> bool:
    """A bare chromosome name before ``:position`` is a contig, unless a word stands just
    before it and the colon is spaced as prose (``exon 2: 207,469`` names no chromosome;
    ``Variant 4:54108297`` does)."""

    return colon == ":" or re.search(r"[A-Za-z] $", before) is None


def is_bases(written: str) -> bool:
    return re.fullmatch(_BASES, written) is not None


def read(text: str) -> Readings:
    """Every value ``text`` writes, by slot; a table row is read cell by cell."""

    cells = text.split("\t") if "\t" in text else [text]
    found: dict[str, set] = {name: set() for name in Readings.__dataclass_fields__}
    for cell in cells:
        stripped = cell.strip()
        _read_text(stripped, found)
        if _CELL_CONTIG.fullmatch(stripped):
            found["contigs"].add(stripped)
    return Readings(**{name: frozenset(values) for name, values in found.items()})


def _read_text(text: str, found: dict[str, set]) -> None:
    masked = list(text)
    for match in _NUCLEOTIDE_CHANGE.finditer(text):
        found["changes"].add(match.group(0))
        if match.group("accession"):
            found["changes"].add(match.group(0)[match.start("kind") - match.start() :])
        if match.group("kind") in "gm":
            genomic = _GENOMIC_SUBSTITUTION.match(match.group("body"))
            if genomic:
                found["positions"].add(int(genomic.group(1)))
                found["allele_pairs"].add((genomic.group(2), genomic.group(3)))
                if match.group("accession"):
                    found["contigs"].add(match.group("accession"))
                    found["loci"].add((match.group("accession"), int(genomic.group(1))))
        masked[match.start() : match.end()] = " " * (match.end() - match.start())
    # A bare protein change is read only outside the names already read (``Ala614Val``
    # inside ``p. (Ala614Val)`` is that name, not a second change).
    for pattern in (_PROTEIN_NAME, _PROTEIN_CHANGE):
        for match in pattern.finditer("".join(masked)):
            found["changes"].add(match.group(0))
            masked[match.start() : match.end()] = " " * (match.end() - match.start())
    rest = "".join(masked)
    for match in _PAIR.finditer(rest):
        ref, alt = match.groups()
        if ref != alt:
            found["allele_pairs"].add((ref, alt))
    for match in _POSITION.finditer(rest):
        number = int(re.sub(r"\D", "", match.group(1)))
        if number > 0:
            found["positions"].add(number)
    for match in _NAMED_CONTIG.finditer(text):
        found["contigs"].update((match.group("written"), match.group("name")))
    found["contigs"].update(m.group(0) for m in _SEQUENCE_CONTIG.finditer(text))
    found["contigs"].update(m.group(1) for m in _LOCUS_CONTIG.finditer(text))
    for match in _LOCUS.finditer(rest):
        position = int(re.sub(r"\D", "", match.group("position")))
        if position > 0:
            found["loci"].add((match.group("contig"), position))
    found["rsids"].update(RSID_RULE.stated(text))
    found["transcripts"].update(m.group(0) for m in _TRANSCRIPT.finditer(text))
    for match in _VERSION_TOKEN.finditer(text):
        token = match.group(0).rstrip("._-")
        if re.search(r"\d", token):
            found["versions"].add(token)
    found["versions"].update(m.group(0) for m in _SPACED_VERSION.finditer(text))
