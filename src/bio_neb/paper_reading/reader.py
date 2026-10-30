"""One Run paper, read through the views of its documents.

Each readable document is a root: one document (``PaperDocument``) whose text the Run's
intake stored (``PaperDocumentRead``, written by the product converters,
``bio_neb.paper_intake.documents``; ``stored_paper``). Views point back into that text, and
every offset a view or a Tool gives is an offset of it:

- a **text section** is one heading's span (``sections``);
- a **table** is one sheet of a workbook (``documents.workbook_to_text``: line ``r`` is row ``r``
  of the sheet) or one run of tab-separated lines in any other text (a JATS or .docx table, a
  tab-separated file). It keeps its location (the file, the sheet's name or the table's label,
  the row, the column) and names its columns by its header row: the first row with two cells
  that hold a value.

Tables leave the text index: a section counts only its text, the outline lists tables apart
with their columns, and ``rows`` reads them by named column.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Literal

from bio_neb.content import shown_to_model
from bio_neb.paper_intake.documents import sheet_heading
from bio_neb.paper_reading.contracts import (
    DocumentViews,
    FailureCode,
    FrozenContract,
    GrepHit,
    GrepResult,
    IdLookup,
    Outline,
    PaperDocument,
    ReadDocument,
    ReadingRefused,
    ReadPassage,
    ReadResult,
    RowsResult,
    TableCell,
    TableRow,
    TableView,
    TextSection,
)
from bio_neb.target_request.paper_targets import PaperIdentity, SourceLine

MAX_HITS = 50
# The most characters one read returns; the model chooses its length.
MAX_READ = 50_000
MAX_ROWS = 100
HIT_CONTEXT = 200  # characters of a hit's line kept on each side of the match
_LABEL = re.compile(
    r"^\s*((?:supplementary\s+)?(?:table|tab\.)\s*S?\d+[A-Za-z]?)\b", re.IGNORECASE
)


class Section(FrozenContract):
    section_id: str
    title: str
    start: int
    end: int


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


def sections(text: str) -> list[Section]:
    """Markdown headings split the text; text before the first heading is its own section."""

    headings = list(_HEADING.finditer(text))
    starts = [(h.start(), h.group(2).strip()) for h in headings]
    if not starts or starts[0][0] > 0:
        starts.insert(0, (0, "(no heading)"))
    return [
        Section(
            section_id=f"s{index + 1}",
            title=title[:120],
            start=start,
            end=starts[index + 1][0] if index + 1 < len(starts) else len(text),
        )
        for index, (start, title) in enumerate(starts)
    ]


def heading_above(found: Sequence[Section], offset: int) -> str:
    """The title of the section holding ``offset`` (the last section's past the text's end)."""

    return next((s.title for s in found if s.start <= offset < s.end), found[-1].title)


def source_line(
    document: PaperDocument, found: Sequence[Section], unit: tuple[int, int]
) -> SourceLine:
    """The line or sentence ``unit`` of a document, with the heading above it."""

    start, end = unit
    return SourceLine(
        document_id=document.document_id,
        section=heading_above(found, start),
        start=start,
        end=end,
        text=document.text[start:end],
    )


def _refused(code: FailureCode, message: str) -> ReadingRefused:
    """A refusal as a Tool result, its message cut to the contract's 500 characters (a
    query may name anything, at any length)."""

    return ReadingRefused(
        code=code, message=message[:497] + "..." if len(message) > 500 else message
    )


def column_letter(number: int) -> str:
    letters = ""
    while number:
        number, rest = divmod(number - 1, 26)
        letters = chr(65 + rest) + letters
    return letters


@dataclass(frozen=True)
class _Table:
    """A table's rows as line indexes of its document, before its views are numbered."""

    location: str
    section: str
    lines: range  # line indexes, first to last row


@dataclass(frozen=True)
class Unit:
    """The paragraph or table row a span id names: its whole line, and a row's column names."""

    span: str
    line: SourceLine
    columns: tuple[str, ...] | None = None

    def cell(self, column: str) -> SourceLine | None:
        """The exact text of a row's cell under ``column``; None in text, for a column the
        table does not name, or past the row's last cell."""

        if self.columns is None or column not in self.columns:
            return None
        cells = self.line.text.split("\t")
        index = self.columns.index(column)
        if index >= len(cells):
            return None
        start = self.line.start + sum(len(c) + 1 for c in cells[:index])
        return self.line.model_copy(
            update={
                "start": start,
                "end": start + len(cells[index]),
                "text": cells[index],
            }
        )


# A span id: a table's row (``t1:r2``), or a document's paragraph by line (``main:p41``).
_ROW_SPAN = re.compile(r"(?P<table>t\d+):r(?P<row>[1-9]\d*)")
_TEXT_SPAN = re.compile(r"(?P<document>.+):p(?P<line>[1-9]\d*)")


class _Lines:
    """A document's lines by offset: line ``i`` is ``text[starts[i]:ends[i]]``."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.starts = [0, *(m.end() for m in re.finditer("\n", text))]
        self.ends = [*(s - 1 for s in self.starts[1:]), len(text)]

    def __len__(self) -> int:
        return len(self.starts)

    def line(self, index: int) -> str:
        return self.text[self.starts[index] : self.ends[index]]

    def at(self, offset: int) -> int:
        """The index of the line holding ``offset`` (a line's end is its newline)."""

        return bisect.bisect_right(self.starts, offset) - 1


def _sheets(lines: _Lines) -> list[_Table] | None:
    """The sheets of a workbook's text, when the whole text is sheets; else None."""

    tables, index = [], 0
    while index < len(lines):
        heading = sheet_heading(lines.line(index))
        if heading is None or index + heading[1] >= len(lines):
            return None
        name, count = heading
        tables.append(_Table(name, name, range(index + 1, index + 1 + count)))
        index += 1 + count
    return tables or None


def _tab_runs(lines: _Lines, found: list[Section]) -> list[_Table]:
    """Runs of tab-separated lines, each labelled by a "Table N" line just above it."""

    tables, index, count = [], 0, 0
    while index < len(lines):
        if "\t" not in lines.line(index):
            index += 1
            continue
        end = index
        while end + 1 < len(lines) and "\t" in lines.line(end + 1):
            end += 1
        count += 1
        above = [lines.line(i) for i in range(max(0, index - 3), index)]
        label = next(
            (m.group(1) for m in map(_LABEL.match, reversed(above)) if m),
            f"table {count}",
        )
        section = heading_above(found, lines.starts[index])
        tables.append(_Table(label, section, range(index, end + 1)))
        index = end + 1
    return tables


def shown_ids(documents: Sequence[PaperDocument]) -> list[PaperDocument]:
    """The documents, each under the id the model is shown and sends back: its recorded id
    ``shown``, and when that spelling is taken, with ``#2``, ``#3``... after it, as
    ``retrieve`` names a second file of one name (review r8 S9). An id with no control
    character is kept as recorded."""

    taken = {
        d.document_id
        for d in documents
        if shown_to_model(d.document_id) == d.document_id
    }
    kept = []
    for d in documents:
        name = shown_to_model(d.document_id)
        if name != d.document_id:
            base, number = name, 1
            while name in taken:
                number += 1
                name = f"{base}#{number}"
            taken.add(name)
            d = d.model_copy(update={"document_id": name})
        kept.append(d)
    return kept


class Paper:
    """One paper: its identity, its documents and their views.

    ``listing`` names each document by its shown id (``shown_ids``) and its text's
    digest and length: those its intake recorded (``recorded``, in document order),
    else, for a text no intake admitted, computed here."""

    def __init__(
        self,
        identity: PaperIdentity | None,
        documents: Sequence[PaperDocument] = (),
        *,
        recorded: Sequence[tuple[str, int]] | None = None,
    ) -> None:
        self.identity = identity
        documents = shown_ids(documents)
        self.documents = {d.document_id: d for d in documents}
        if recorded is None:
            recorded = [
                (hashlib.sha256(d.text.encode("utf-8")).hexdigest(), d.char_count)
                for d in documents
            ]
        self.listing = tuple(
            ReadDocument(
                document_id=d.document_id,
                text_sha256=digest,
                characters=characters,
                read_event_id=d.read_event_id,
            )
            for d, (digest, characters) in zip(documents, recorded, strict=True)
        )
        self.lines = {d.document_id: _Lines(d.text) for d in documents}
        self.found = {d.document_id: sections(d.text) for d in documents}
        # Each table line, by document and line index: its table and its first line.
        self.line_tables: dict[str, dict[int, tuple[TableView, int]]] = {}
        self.outline = self._views()
        self.word_lines: dict[str, dict[tuple[str, int], None]] | None = None
        self.sections = {
            s.section_id: (d.document_id, s)
            for d in self.outline.documents
            for s in d.sections
        }
        self.tables = {
            t.table_id: (d.document_id, t)
            for d in self.outline.documents
            for t in d.tables
        }

    @classmethod
    def stored(cls, journey: Any, paper_event_id: str) -> Paper:
        """The Run paper its ``PaperRecorded`` names, read from the texts its intake
        stored: each document's ``PaperDocumentRead`` text, opened by digest
        (``Journey.read_artifact``), ``main`` and each ``supp:<name>`` in its order,
        listed by the digest and length that intake recorded (ADR 0029: the admitted
        text is not hashed again)."""
        from bio_neb.paper_intake.journey import (
            PaperDocumentRead,
            PaperDocumentRetained,
            PaperRecorded,
        )
        from bio_neb.recording import evidence_view

        by_id = {event["event_id"]: event for event in evidence_view(journey)}
        event = by_id.get(paper_event_id)
        if event is None or event["event_type"] != "PaperRecorded":
            raise ValueError("a reading names a recorded Run paper")
        recorded = PaperRecorded.model_validate(event["payload"])
        documents, digests = [], []
        for document in recorded.documents:
            read_event = by_id[document.read_event_id]
            read = PaperDocumentRead.model_validate(read_event["payload"])
            retained = PaperDocumentRetained.model_validate(
                by_id[read.retained_event_id]["payload"]
            )
            text = journey.read_artifact(read_event["artifact_references"][0])
            digests.append((read.text_sha256, read.characters))
            documents.append(
                PaperDocument(
                    document_id=document.document_id,
                    kind="main_text"
                    if document.document_id == "main"
                    else "supplementary",
                    url=retained.location,
                    title=None if document.document_id == "main" else document.name,
                    text=text.decode("utf-8"),
                    read_event_id=document.read_event_id,
                )
            )
        return cls(paper_identity(recorded), documents, recorded=digests)

    def columns_at(self, document_id: str, offset: int) -> tuple[str, ...] | None:
        """The column names of the table whose row holds ``offset``; None in text."""

        if document_id not in self.lines:
            return None
        in_table = self.line_tables[document_id].get(self.lines[document_id].at(offset))
        return None if in_table is None else in_table[0].columns

    def span_of(self, document_id: str, index: int) -> str:
        """The id code issues for line ``index`` of a document: its table row's, or its
        paragraph's by line number. The same paper gives the same ids on every read."""

        in_table = self.line_tables[document_id].get(index)
        if in_table is None:
            return f"{document_id}:p{index + 1}"
        table, first = in_table
        return f"{table.table_id}:r{index - first + 1}"

    def unit(self, span: str) -> Unit | None:
        """The paragraph or table row ``span`` names; None for an id code never issued,
        or for a line that holds no text."""

        row = _ROW_SPAN.fullmatch(span)
        text = None if row else _TEXT_SPAN.fullmatch(span)
        if row and row["table"] in self.tables:
            document_id, table = self.tables[row["table"]]
            lines = self.lines[document_id]
            index = lines.at(table.start) + int(row["row"]) - 1
            if self.line_tables[document_id].get(index, (None,))[0] != table:
                return None
        elif text and text["document"] in self.lines:
            document_id, index = text["document"], int(text["line"]) - 1
            lines = self.lines[document_id]
            if index >= len(lines) or self.span_of(document_id, index) != span:
                return None
        else:
            return None
        line = passage(self, document_id, lines.starts[index])
        if line is None:
            return None
        return Unit(span, line, self.columns_at(document_id, line.start))

    def hit(self, document_id: str, start: int, end: int) -> GrepHit:
        """The hit for ``start:end`` of a document: its line cut to ``HIT_CONTEXT`` characters
        each side, and the heading above it or its table row and column."""

        lines = self.lines[document_id]
        index = lines.at(start)
        line_start, line_end = lines.starts[index], lines.ends[index]
        line = lines.text[
            max(line_start, start - HIT_CONTEXT) : min(line_end, end + HIT_CONTEXT)
        ]
        in_table = self.line_tables[document_id].get(index)
        if in_table is None:
            return GrepHit(
                document_id=document_id,
                offset=start,
                span=self.span_of(document_id, index),
                line=line,
                section=heading_above(self.found[document_id], start),
            )
        table, first = in_table
        cell = lines.text[line_start:start].count("\t")
        return GrepHit(
            document_id=document_id,
            offset=start,
            span=self.span_of(document_id, index),
            line=line,
            table=TableCell(
                table_id=table.table_id,
                location=table.location,
                row=index - first + 1,
                column=table.columns[cell],
            ),
        )

    def _views(self) -> Outline:
        documents, section_count, table_count = [], 0, 0
        for document in self.documents.values():
            lines = self.lines[document.document_id]
            sheets = _sheets(lines)
            found = (
                sheets
                if sheets is not None
                else _tab_runs(lines, self.found[document.document_id])
            )
            in_tables = {index for table in found for index in table.lines}
            tables = []
            line_tables = self.line_tables.setdefault(document.document_id, {})
            for table in found:
                table_count += 1
                view = self._table_view(document, f"t{table_count}", table)
                tables.append(view)
                for index in table.lines:
                    line_tables[index] = (view, table.lines.start)
            texts = []
            for section in (
                [] if sheets is not None else self.found[document.document_id]
            ):
                own = range(
                    lines.at(section.start),
                    lines.at(max(section.start, section.end - 1)) + 1,
                )
                section_count += 1
                texts.append(
                    TextSection(
                        section_id=f"s{section_count}",
                        title=section.title,
                        start=section.start,
                        end=section.end,
                        characters=(section.end - section.start)
                        - sum(
                            min(lines.ends[i] + 1, section.end) - lines.starts[i]
                            for i in own
                            if i in in_tables
                        ),
                    )
                )
            documents.append(
                DocumentViews(
                    document_id=document.document_id,
                    kind=document.kind,
                    url=document.url,
                    characters=len(document.text),
                    sections=tuple(texts),
                    tables=tuple(tables),
                )
            )
        return Outline(documents=tuple(documents))

    def _table_view(
        self, document: PaperDocument, table_id: str, table: _Table
    ) -> TableView:
        lines = self.lines[document.document_id]
        cells = [lines.line(i).split("\t") for i in table.lines]
        width = max((len(row) for row in cells), default=0)
        header = next(
            (
                n
                for n, row in enumerate(cells)
                if sum(bool(c.strip()) for c in row) >= 2
            ),
            None,
        )
        first = table.lines.start
        # A sheet with no row sits at the end of its heading line.
        start = lines.starts[first] if table.lines else lines.ends[first - 1]
        return TableView(
            table_id=table_id,
            location=table.location,
            section=table.section,
            start=start,
            end=lines.ends[table.lines[-1]] if table.lines else start,
            header_row=None if header is None else header + 1,
            columns=column_names(cells[header] if header is not None else [], width),
            rows=sum(
                any(c.strip() for c in row)
                for row in cells[(header + 1 if header is not None else 0) :]
            ),
        )


def paper_identity(recorded: Any) -> PaperIdentity:
    """What the paper's source said it is, copied from its ``PaperRecorded``: a cited
    paper's Europe PMC ids; a supplied file with none, by the Run's ``paper_id`` and the
    file's name (owner, #383 A1b), so its targets are asserted by it."""

    if recorded.ids is None:
        return PaperIdentity(
            paper_id=recorded.paper_id,
            file_name=None
            if recorded.paper_file is None
            else PurePath(recorded.paper_file).name,
            resolved_by="user_supplied",
        )
    ids = recorded.ids
    return PaperIdentity(
        doi=ids.doi,
        pmid=ids.pmid,
        pmcid=ids.pmcid,
        title=ids.title,
        journal=ids.journal,
        year=ids.year,
        open_access=ids.open_access,
        resolved_by="europe_pmc" if recorded.origin == "citation" else "user_supplied",
    )


def column_names(header: Sequence[str], width: int) -> tuple[str, ...]:
    """One name per column of a table ``width`` columns wide, each name once: the header
    cell's text, or the column's letter when it has none; a name already given takes the
    column's letter after it (``B``, ``B (B)``)."""

    names: list[str] = []
    for number in range(1, width + 1):
        name = (
            shown_to_model(header[number - 1]).strip() if number <= len(header) else ""
        ) or (column_letter(number))
        while name in names:
            name = f"{name} ({column_letter(number)})"
        names.append(name)
    return tuple(names)


def outline(paper: Paper) -> Outline:
    """Every document with its text sections and its tables (columns and row counts)."""

    return paper.outline


def grep(paper: Paper, pattern: str, max_hits: int = 10) -> GrepResult:
    """Every match of a case-insensitive regular expression in every document, counted per
    document and per table; the first ``max_hits`` (at most ``MAX_HITS``) come back as hits.

    A pattern that is no regular expression is searched as written (``literal``)."""

    try:
        regex, literal = re.compile(pattern, re.IGNORECASE), False
    except re.error:
        regex, literal = re.compile(re.escape(pattern), re.IGNORECASE), True
    limit = max(1, min(max_hits, MAX_HITS))
    matches: dict[str, int] = {}
    table_matches: dict[str, int] = {}
    hits: list[GrepHit] = []
    for document_id, document in paper.documents.items():
        lines, line_tables = paper.lines[document_id], paper.line_tables[document_id]
        matches[document_id] = 0
        for match in regex.finditer(document.text):
            if match.end() == match.start():
                continue
            matches[document_id] += 1
            in_table = line_tables.get(lines.at(match.start()))
            if in_table is not None:
                table_id = in_table[0].table_id
                table_matches[table_id] = table_matches.get(table_id, 0) + 1
            if len(hits) < limit:
                hits.append(paper.hit(document_id, match.start(), match.end()))
    return GrepResult(
        pattern=pattern,
        literal=literal,
        matches=matches,
        table_matches=table_matches,
        hits=tuple(hits),
    )


def read(
    paper: Paper, at: str, offset: int | None = None, length: int = 3000
) -> ReadResult | ReadingRefused:
    """The text of a section (``at`` a section id), a document (``at`` a document id), a
    table (``at`` a table id) or one paragraph or table row (``at`` a span id) from ``offset`` (its start by default),
    ``length`` characters and never more than ``MAX_READ``. ``tables`` names the tables
    with rows in the span."""

    unit = None if at in paper.sections or at in paper.documents else paper.unit(at)
    if unit is not None:
        # A span id reads inside its paragraph or row, within the same limit.
        document_id, section_id = unit.line.document_id, None
        low, high = unit.line.start, unit.line.end
    elif at in paper.sections:
        document_id, section = paper.sections[at]
        low, high, section_id = section.start, section.end, at
    elif at in paper.documents:
        document_id, section_id = at, None
        low, high = 0, len(paper.documents[at].text)
    elif at in paper.tables:
        # A table reads as its rows by span, as a section does.
        document_id, table = paper.tables[at]
        section_id, low, high = None, table.start, table.end
    else:
        return _refused(
            code="unknown_view",
            message=(
                f"no section, document, table or span is named {at!r}; outline lists "
                "the sections, documents and tables, and every reading result names its "
                "spans"
            ),
        )
    begin = min(high, max(low, low if offset is None else offset))
    stop = min(high, begin + max(1, min(length, MAX_READ)))
    text = paper.documents[document_id].text
    lines = paper.lines[document_id]
    indexes = range(lines.at(begin), lines.at(stop - 1) + 1) if stop > begin else ()
    return ReadResult(
        document_id=document_id,
        section_id=section_id,
        start=begin,
        end=stop,
        characters=len(text),
        passages=tuple(
            ReadPassage(
                span=paper.span_of(document_id, index),
                text=text[
                    max(begin, lines.starts[index]) : min(stop, lines.ends[index])
                ],
            )
            for index in indexes
        ),
        tables=tuple(
            table_id
            for table_id, (in_document, table) in paper.tables.items()
            if in_document == document_id and table.start < stop and begin < table.end
        ),
    )


def _same(cell: str, wanted: str) -> bool:
    """A cell holds a filter's value: the same text, case aside, or the same number."""

    cell, wanted = shown_to_model(cell).strip(), wanted.strip()
    if cell.casefold() == wanted.casefold():
        return True
    try:
        return float(cell.replace(",", "")) == float(wanted.replace(",", ""))
    except ValueError:
        return False


RowsRefusal = Literal[
    "unknown_table", "rows_unscoped", "unknown_filter_column", "unknown_column"
]


def _row_tables(
    paper: Paper,
    table_id: str | None,
    where: Mapping[str, str],
    columns: Sequence[str],
) -> list[tuple[str, TableView]] | RowsRefusal:
    """The tables ``rows`` reads, or why there are none: the named table, or every table
    naming the filtered columns; each must name the kept columns too."""

    if table_id is not None:
        if table_id not in paper.tables:
            return "unknown_table"
        tables = [paper.tables[table_id]]
        if any(c not in tables[0][1].columns for c in where):
            return "unknown_filter_column"
    elif not where:
        return "rows_unscoped"
    else:
        tables = [
            (document_id, table)
            for document_id, table in paper.tables.values()
            if all(c in table.columns for c in where)
        ]
        if not tables:
            return "unknown_filter_column"
    chosen = [t for t in tables if all(c in t[1].columns for c in columns)]
    return chosen or "unknown_column"


def rows_refusal(
    paper: Paper,
    table_id: str | None = None,
    where: Mapping[str, str] | None = None,
    columns: Sequence[str] = (),
) -> RowsRefusal | None:
    """Why ``rows`` would refuse this call, before it runs; None when it reads."""

    chosen = _row_tables(paper, table_id, dict(where or {}), columns)
    return chosen if isinstance(chosen, str) else None


def rows(
    paper: Paper,
    table_id: str | None = None,
    where: Mapping[str, str] | None = None,
    columns: Sequence[str] = (),
    max_rows: int = 20,
) -> RowsResult | ReadingRefused:
    """The rows of one table, or of every table naming the filtered columns, whose cells
    equal ``where`` (text case aside, or the same number), by named column; ``columns``
    keeps only those. The first ``max_rows`` (at most ``MAX_ROWS``) come back, each with
    its document offset, the pointer a finding cites."""

    where = dict(where or {})
    chosen = _row_tables(paper, table_id, where, columns)
    if isinstance(chosen, str):
        named = [*where, *columns]
        if chosen == "unknown_table":
            message = f"no table is named {table_id!r}; outline lists the tables"
        elif chosen == "rows_unscoped":
            message = "rows takes a table id, or filters naming the columns to search"
        elif table_id is not None:
            have = paper.tables[table_id][1].columns
            message = (
                f"{table_id} has no column {[c for c in named if c not in have]}; "
                f"its columns are {list(have)}"
            )
        else:
            message = f"no table has the columns {named}; outline lists each table's"
        return _refused(
            code="unknown_column" if chosen.endswith("column") else "unknown_view",
            message=message,
        )
    limit = max(1, min(max_rows, MAX_ROWS))
    matched, found = 0, []
    for document_id, table in chosen:
        if not table.rows:
            continue
        lines = paper.lines[document_id]
        first = lines.at(table.start)
        for index in range(first + (table.header_row or 0), lines.at(table.end) + 1):
            values = lines.line(index).split("\t")
            if not any(v.strip() for v in values):
                continue
            padded = (*values, *[""] * (len(table.columns) - len(values)))
            cells = dict(zip(table.columns, padded, strict=True))
            if not all(_same(cells[c], v) for c, v in where.items()):
                continue
            matched += 1
            if len(found) < limit:
                found.append(
                    TableRow(
                        table_id=table.table_id,
                        location=table.location,
                        row=index - first + 1,
                        span=paper.span_of(document_id, index),
                        document_id=document_id,
                        offset=lines.starts[index],
                        cells={c: cells[c] for c in columns} if columns else cells,
                    )
                )
    return RowsResult(
        tables=tuple(table.table_id for _, table in chosen),
        matched=matched,
        rows=tuple(found),
    )


def passage(paper: Paper, document_id: str, offset: int) -> SourceLine | None:
    """The paragraph or table row holding ``offset`` of a document, copied whole with the
    heading above it: what a finding's passage pointer cites. None when the document is not
    the paper's, the offset is outside it, or its line holds no text."""

    lines = paper.lines.get(document_id)
    if lines is None or not 0 <= offset < len(lines.text):
        return None
    index = lines.at(offset)
    start, end = lines.starts[index], lines.ends[index]
    if not lines.text[start:end].strip():
        return None
    return source_line(
        paper.documents[document_id], paper.found[document_id], (start, end)
    )


# A word of the paper: a number with thousands separators (read without them), or letters,
# digits and underscores with dots or hyphens inside (rs1800562, chr4, NC_009147.3,
# HLA-A, c.2624C).
_TOKEN = re.compile(
    r"\d{1,3}(?:,\d{3})+(?![\d,])|[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?"
)
_RSID = re.compile(r"rs\d+", re.IGNORECASE)
# A position: a contig and a number of three digits or more, as the candidate patterns read
# positions (``4:1`` is no position: "1" sits beside "4" on many a table row).
_POSITION = re.compile(
    r"(?:chr)?([A-Za-z0-9_.]+?)\s*[:\s]\s*(\d{1,3}(?:,\d{3})+|\d{3,})", re.IGNORECASE
)


def _word(text: str) -> str:
    """A word as the index keeps it: a number without its separators, else lower case."""

    return text.replace(",", "") if re.fullmatch(r"[\d,]+", text) else text.lower()


def _words(paper: Paper) -> dict[str, dict[tuple[str, int], None]]:
    """Every word of the paper with the lines holding it, in document and line order."""

    if paper.word_lines is None:
        index: dict[str, dict[tuple[str, int], None]] = {}
        for document_id, lines in paper.lines.items():
            for number in range(len(lines)):
                for match in _TOKEN.finditer(lines.line(number)):
                    index.setdefault(_word(match.group()), {})[
                        (document_id, number)
                    ] = None
        paper.word_lines = index
    return paper.word_lines


def _word_pattern(word: str) -> re.Pattern[str]:
    """``word`` where it stands as a whole word of a line (a number with or without its
    thousands separators)."""

    if word.isdigit():
        groups = []
        rest = word
        while rest:
            groups.insert(0, rest[-3:])
            rest = rest[:-3]
        body = ",?".join(groups)
        return re.compile(rf"(?<![\d,]){body}(?![\d]|,\d)")
    return re.compile(
        rf"(?<![A-Za-z0-9_])(?<![A-Za-z0-9_][.-]){re.escape(word)}"
        rf"(?![A-Za-z0-9_]|[.-][A-Za-z0-9_])",
        re.IGNORECASE,
    )


def _word_hits(
    paper: Paper, pattern: re.Pattern[str], at: Sequence[tuple[str, int]]
) -> tuple[GrepHit, ...]:
    hits = []
    for document_id, number in at:
        lines = paper.lines[document_id]
        match = pattern.search(lines.text, lines.starts[number], lines.ends[number])
        if match is not None:
            hits.append(paper.hit(document_id, match.start(), match.end()))
    return tuple(hits)


def lookup(paper: Paper, ids: Sequence[str], max_hits: int = 5) -> tuple[IdLookup, ...]:
    """Each id of a list (rsIDs, positions such as ``4:54108297`` or ``chr4:54,108,297``,
    gene or other names) looked up in every document in code, with no model: found on the
    lines holding it as a whole word, or marked missing. A position is found where its
    number and its contig (as written, or after ``chr``) share a line; ``near`` holds the
    lines with its number alone. Each id gives at most ``max_hits`` hits."""

    words = _words(paper)
    found = []
    for raw in ids:
        id_ = raw.strip()
        position = None if _RSID.fullmatch(id_) else _POSITION.fullmatch(id_)
        if position is None:
            kind = "rsid" if _RSID.fullmatch(id_) else "name"
            word, near = _word(id_), []
            lines = list(words.get(word, {}))
        else:
            kind, word = "position", _word(position.group(2))
            contig = position.group(1).lower()
            lines, near = [], []
            for at in words.get(word, {}):
                named = {
                    _word(m.group())
                    for m in _TOKEN.finditer(paper.lines[at[0]].line(at[1]))
                }
                (lines if {contig, f"chr{contig}"} & named else near).append(at)
        pattern = _word_pattern(word)
        found.append(
            IdLookup(
                id=raw,
                kind=kind,
                found=bool(lines),
                lines=len(lines),
                hits=_word_hits(paper, pattern, lines[:max_hits]),
                near=_word_hits(paper, pattern, near[:max_hits]),
            )
        )
    return tuple(found)
