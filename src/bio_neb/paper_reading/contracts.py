"""The paper reader's contracts: the views of a paper's documents, the reading Tools'
results, and the answer the model points to and code keeps (ADR 0046).

Every model is frozen and rejects extra fields, like the product contracts in
``bio_neb.contracts``. The Journey facts that carry them are ``paper_reading.journey``'s.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from bio_neb.contracts import FrozenContract
from bio_neb.target_request.paper_targets import (
    Finding,
    PaperIdentity,
    SourceLine,
    StatedReference,
)
from bio_neb.target_request.paper_text import NumberForm

DocumentKind = Literal["main_text", "supplementary"]
# What a reading Tool refuses before it reads (admission refuses each first).
FailureCode = Literal["unknown_view", "unknown_column"]


class PaperDocument(FrozenContract):
    """One readable document of a paper. ``text`` is the stored text every offset and
    span refers to: in a Run, the text its ``PaperDocumentRead`` retained
    (``read_event_id``)."""

    document_id: str
    kind: DocumentKind
    url: str
    title: str | None = None
    text: str
    read_event_id: str | None = None

    @property
    def char_count(self) -> int:
        return len(self.text)


class ReadDocument(FrozenContract):
    """A document a reading names: its id, its stored text's digest and length, and, in
    a Run, the ``PaperDocumentRead`` that retained that text. A span's ``start:end``
    points into exactly that text."""

    document_id: str
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    characters: int = Field(ge=0)
    read_event_id: str | None = None


# ---------------------------------------------------------------------------
# Reading one paper: the views of its documents and the reading Tools' results
# ---------------------------------------------------------------------------


class TextSection(FrozenContract):
    """A text view: one heading's span of a document, read by ``read``.

    ``characters`` counts its text only; the rows of a table inside it belong to the table.
    """

    section_id: str
    title: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    characters: int = Field(ge=0)


class TableView(FrozenContract):
    """A table view: one sheet of a workbook, or one table of a text, read by ``rows``.

    Its rows are the document's lines ``start:end``, the first being row 1 (of the sheet, or
    of the table). ``location`` is the sheet's name or the table's label; ``columns`` are named
    by the header row (a column with no name by its letter, A, B, ...).
    """

    table_id: str
    location: str
    section: str  # the heading above it
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    header_row: int | None = None
    columns: tuple[
        str, ...
    ] = ()  # one unique name per column, as wide as its widest row
    rows: int = Field(ge=0)  # rows holding a value, the header aside


class DocumentViews(FrozenContract):
    """One retrieved file, the root of its views."""

    document_id: str
    kind: DocumentKind
    url: str
    characters: int = Field(ge=0)
    sections: tuple[TextSection, ...] = ()
    tables: tuple[TableView, ...] = ()


class Outline(FrozenContract):
    """Every document of the paper with its text sections and its tables."""

    documents: tuple[DocumentViews, ...] = ()


class TableCell(FrozenContract):
    """Where a line sits in a table: the table, its location, the row and the column."""

    table_id: str
    location: str
    row: int = Field(ge=1)
    column: str | None = None


class GrepHit(FrozenContract):
    """One match: its document and offset, its line around it, and where the line is.

    ``offset`` points into the document and may be a finding's pointer.
    """

    document_id: str
    offset: int = Field(ge=0)
    span: str  # the id of its paragraph or table row, which a finding cites
    line: str
    section: str | None = None  # the heading above a text line
    table: TableCell | None = None  # the row of a table line


class GrepResult(FrozenContract):
    """``matches`` counts every match in every document, so a hit cap hides no table."""

    pattern: str
    literal: bool  # the pattern was not a regular expression, so its text was searched
    matches: dict[str, int]  # every document, with the matches it holds
    table_matches: dict[str, int]  # each table holding a match
    hits: tuple[GrepHit, ...]  # the first hits, in document order


class ReadPassage(FrozenContract):
    """One line of a read: a paragraph or a table row under its span id, as far as the
    read reaches into it."""

    span: str
    text: str


class ReadResult(FrozenContract):
    document_id: str
    section_id: str | None = None
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    characters: int = Field(ge=0)  # the whole document's
    # The document's text from start to end, line by line under each line's span id.
    passages: tuple[ReadPassage, ...]
    tables: tuple[str, ...] = ()  # tables with rows in this span; rows reads them


class TableRow(FrozenContract):
    """One row by its named columns, with the pointer to its line."""

    table_id: str
    location: str
    row: int = Field(ge=1)
    span: str  # the row's id, which a finding cites
    document_id: str
    offset: int = Field(ge=0)
    cells: dict[str, str]


class RowsResult(FrozenContract):
    tables: tuple[str, ...]  # the tables searched
    matched: int = Field(ge=0)
    rows: tuple[TableRow, ...]  # the first rows matched


IdKind = Literal["rsid", "position", "name"]


class IdLookup(FrozenContract):
    """One id looked up in code: found on ``lines`` lines, or marked missing.

    A position is found where its number and its contig share a line; ``near`` holds lines
    with the number alone.
    """

    id: str
    kind: IdKind
    found: bool
    lines: int = Field(ge=0)
    hits: tuple[GrepHit, ...] = ()
    near: tuple[GrepHit, ...] = ()


# ---------------------------------------------------------------------------
# The answer to a researcher's question: the model points, code fills
# ---------------------------------------------------------------------------


class ReferencePointer(FrozenContract):
    """The genome version the paper's positions are on, as the paper writes it."""

    name: str = Field(
        min_length=1,
        description=(
            "The assembly's own version name exactly as the paper writes it, such as "
            "GRCh37 or TAIR10; another name the paper writes for the same version goes in "
            "aliases"
        ),
    )
    aliases: tuple[str, ...] = Field(
        default=(),
        description=(
            "Other names the same span writes for this version, each exactly as written, "
            "such as hg19 beside GRCh37"
        ),
    )
    span: str = Field(
        min_length=1,
        description=(
            "The span id of the paragraph or table row that writes it, such as main:p12 "
            "or t1:r1, as grep_paper, read_paper or filter_paper_rows gave it"
        ),
    )
    reason: str = Field(
        min_length=1,
        description="Why the paper's positions are on this version, from the paper",
    )


# The slots a stated value fills: main's request item and lane Rf's change type.
Slot = Literal["contig", "position", "alleles", "ref", "alt", "rsid", "change", "gene"]


class ValuePointer(FrozenContract):
    """One value as the paper writes it, and the span that writes it; code finds the
    written text there and reads it. The model copies no paragraph."""

    slot: Slot = Field(
        description=(
            "contig, position, alleles (a directed pair in text, such as C>G), ref or "
            "alt (a table's REF or ALT cell), rsid, change (c., g., p. or a protein "
            "change), or gene"
        )
    )
    written: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "The value exactly as the span writes it, character for character, such as "
            "chr 5, 12,345, 678, C>G, rs12345, c.200C>T or GENE1"
        ),
    )
    span: str = Field(
        min_length=1,
        description=(
            "The span id of the paragraph or table row that writes it, such as main:p41 "
            "or t1:r3"
        ),
    )
    column: str | None = Field(
        default=None,
        description="In a table row, the column that holds it, by its name",
    )
    number: int | None = Field(
        default=None,
        description=(
            "For a position: the number its digits spell, such as 12345678 for "
            "12,345, 678"
        ),
    )


class FindingPointer(FrozenContract):
    """One finding as the model reads it: each value it states, as written and where, and
    its reason. Code keeps each value only where the span writes it (``answer``)."""

    values: tuple[ValuePointer, ...] = Field(
        description="Each value the paper writes for this one variant, with its span"
    )
    spans: tuple[str, ...] = Field(
        default=(),
        description="Further span ids the finding rests on, beyond its values' spans",
    )
    corrects: int | None = Field(
        default=None,
        ge=1,
        description=(
            "In a repair: the number of the refused finding this one corrects, as the "
            "repair numbers it"
        ),
    )
    reason: str = Field(
        min_length=1,
        description="Why this finding answers the question, from the paper",
    )


class OrganismPointer(FrozenContract):
    """The organism the paper's sample or reads come from, as the paper writes it."""

    written: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "The organism exactly as the span writes it, character for character, such "
            "as a species name, a common name or a strain"
        ),
    )
    span: str = Field(
        min_length=1,
        description=(
            "The span id of the paragraph or table row that writes it, such as main:p3, "
            "as grep_paper, read_paper or filter_paper_rows gave it"
        ),
    )


class StatedOrganism(FrozenContract):
    """The organism as code kept it: its written text exactly where the span writes it
    (``source``), and that span's whole paragraph or row (``passage``). Nothing here
    names a taxon: the planning Agent resolves it from these words."""

    written: str
    span: str
    source: SourceLine
    passage: SourceLine

    @model_validator(mode="after")
    def _written_where_it_is(self) -> StatedOrganism:
        if self.source.text != self.written:
            raise ValueError("a kept organism's source is its written text")
        return self


class ReadingArea(FrozenContract):
    """One area the paper item proposes; it becomes its own reading item."""

    area: str = Field(
        min_length=1,
        description=(
            "What to read for, in the paper's terms: a table's variants, the protein "
            "changes of a results paragraph, the rows of one gene"
        ),
    )
    views: tuple[str, ...] = Field(
        default=(),
        description=(
            "The section, table or document ids from outline_paper, or the span ids of "
            "paragraphs or rows, that hold it"
        ),
    )
    reason: str = Field(min_length=1, description="Why the question needs this area")


# The most areas one paper item proposes, each its own reading item with its own calls;
# code refuses a plan above it.
MAX_AREAS = 6


class ReadingPlan(FrozenContract):
    """The paper item's output: the reference as written and the areas to read."""

    reference: ReferencePointer | None = Field(
        description="The reference the paper's positions are on, or null if none is stated"
    )
    organism: OrganismPointer | None = Field(
        default=None,
        description=(
            "The organism the paper's samples or reads come from, as written, with the "
            "span id that writes it; null if the paper does not write it"
        ),
    )
    areas: tuple[ReadingArea, ...] = Field(
        description=(
            f"Every area the question needs, at most {MAX_AREAS}, each read as its own "
            "item; empty only when the ids found in code answer it or nothing in the paper "
            "does"
        )
    )
    not_answered: str | None = Field(
        default=None,
        description="When you give no reference or no area: why, in one sentence",
    )


class ItemAnswer(FrozenContract):
    """An area or id item's output: its findings by pointer, each with its reason."""

    findings: tuple[FindingPointer, ...] = Field(
        description="Every finding of this item the question asks for; empty only when none"
    )
    not_answered: str | None = Field(
        default=None,
        description="When you give no finding: why, in one sentence",
    )


class KeptValue(FrozenContract):
    """One value code kept: its written text exactly where the paper writes it
    (``source``), the span id and column that cited it, its typed value (a position's
    number; else the text as written) and the form it is written in."""

    slot: Slot
    written: str
    span: str
    column: str | None = None
    source: SourceLine
    value: int | str
    form: NumberForm = "as_written"

    @model_validator(mode="after")
    def _written_where_it_is(self) -> KeptValue:
        if self.source.text != self.written:
            raise ValueError("a kept value's source is its written text")
        return self


class KeptFinding(Finding):
    """An accepted finding with every value code kept for it. ``passages`` start with the
    paper's text from its first to its last kept value in one span (code chooses it; the
    model copies nothing), then each span it cites, whole."""

    values: tuple[KeptValue, ...] = ()


class FindingOutcome(FrozenContract):
    """A finding code did not accept (#404 d2): ``unresolved`` when the paper itself is
    unclear (never asked again), ``refused`` when a check failed and the repairs ran out.
    It keeps the finding as the model stated it, each reason, the spans it cites and any
    value code could keep, so a person can decide it (HITL) from the chunk."""

    outcome: Literal["unresolved", "refused"]
    reasons: tuple[str, ...] = Field(min_length=1)
    stated: FindingPointer
    passages: tuple[SourceLine, ...] = ()
    values: tuple[KeptValue, ...] = ()


class PaperAnswer(FrozenContract):
    """A researcher's question about one paper, answered from its verified reading, with
    no cap: the accepted findings, and each finding left unresolved or refused."""

    question: str
    paper: PaperIdentity | None = None
    documents: tuple[ReadDocument, ...] = ()  # the documents the answer cites
    reference: StatedReference | None = None
    organism: StatedOrganism | None = None
    findings: tuple[KeptFinding, ...] = ()
    unresolved: tuple[FindingOutcome, ...] = ()
    refused: tuple[FindingOutcome, ...] = ()
    not_answered: str | None = None


class Verification(FrozenContract):
    """``verify``'s result: every finding rebuilt from its pointers, and each mismatch."""

    verified: bool
    findings: int = Field(ge=0)
    problems: tuple[str, ...] = ()


class ReadingRefused(FrozenContract):
    code: FailureCode
    message: str = Field(max_length=500)


ItemKind = Literal["paper", "area", "id"]


class ReadingItem(FrozenContract):
    """One reading item: the paper, an area the paper item proposed, or an id code missed.

    ``item_id`` names it within its reading (``paper``, ``area-1``, ``id-1``...): code
    derives every item from the paper item's areas and the ids code missed, in order."""

    item_id: str = Field(pattern=r"^(paper|area-[1-9]\d*|id-[1-9]\d*)$")
    kind: ItemKind
    subject: str  # the question, the area, or the id
    views: tuple[str, ...] = ()  # an area's sections, tables, documents or spans
    reason: str | None = None  # why the paper item proposed the area
    near: tuple[GrepHit, ...] = ()  # an id's number on lines without its contig
