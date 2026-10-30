"""Converters: a paper document's bytes as the text the reading Tools read.

A Run paper's documents are downloaded (`bio_neb.providers.paper_sources`) or supplied as
files; these read them: JATS (with section headings; each table as its rows, also one
floated inside a paragraph), HTML, .xlsx (sheets by name, line r = row r, each cell at its
column), .docx (each table once, as rows), .csv, text files, PDFs (each page's text under a
``## Page n`` heading, #420 F4) and zip archives (one section per file, read back by
`archive_files`). A document with no reader here, or no text, reads as None. Nothing here
contacts a host.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import os
import posixpath
import re
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Literal

from pypdf import PdfReader
from pypdf.errors import PyPdfError


class _LinkPreservingText(HTMLParser):
    """HTML to plain text that keeps anchors as markdown links for supplement detection."""

    _BLOCK = {
        "p",
        "div",
        "br",
        "tr",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "section",
    }
    _SKIP = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._href: str | None = None
        self._anchor: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag == "a":
            self._href = dict(attrs).get("href")
            self._anchor = []
        elif tag in ("td", "th"):
            self.parts.append("\t")
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "a" and self._href is not None:
            text = " ".join("".join(self._anchor).split())
            self.parts.append(f"[{text}]({self._href})" if text else "")
            self._href = None

    def handle_data(self, data):
        if self._skip:
            return
        if self._href is not None:
            self._anchor.append(data)
        else:
            self.parts.append(data)


def html_to_text(html: str, base_url: str) -> str:
    parser = _LinkPreservingText()
    parser.feed(html)
    text = "".join(parser.parts)

    def absolute(match: re.Match[str]) -> str:
        return f"[{match.group(1)}]({urllib.parse.urljoin(base_url, match.group(2))})"

    text = re.sub(r"\[([^\]]*)\]\(([^)]+)\)", absolute, text)
    return re.sub(r"\n\s*\n+", "\n\n", text)


def _xml_local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def office_zip_to_text(data: bytes) -> str | None:
    """Plain text of an .xlsx (``workbook_to_text``) or .docx (``docx_to_text``), stdlib only."""

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return None
    names = set(archive.namelist())
    if "word/document.xml" in names:
        return docx_to_text(ET.fromstring(archive.read("word/document.xml")))
    if any(name.startswith("xl/worksheets/") for name in names):
        return workbook_to_text(archive)
    return None


def docx_to_text(root: ET.Element) -> str:
    """One line per paragraph and per table row (its cells tab-separated), in document order.

    A table's cells are read once, as its rows: the paragraphs inside a cell join its text.
    """

    lines: list[str] = []

    def text(node: ET.Element) -> str:
        return _flat(
            "".join(t.text or "" for t in node.iter() if _xml_local(t.tag) == "t")
        )

    def walk(node: ET.Element) -> None:
        tag = _xml_local(node.tag)
        if tag == "tbl":
            for row in node.iter():
                if _xml_local(row.tag) == "tr":
                    cells = [
                        " ".join(
                            filter(
                                None,
                                (text(p) for p in cell if _xml_local(p.tag) == "p"),
                            )
                        )
                        for cell in row
                        if _xml_local(cell.tag) == "tc"
                    ]
                    lines.append("\t".join(cells))
            return
        if tag == "p":
            lines.append(text(node))
            return
        for child in node:
            walk(child)

    walk(root)
    return "\n".join(line for line in lines if line.strip())


_SHEET_HEADING = re.compile(r"^## (.*) \((\d+) rows?\)$")


def _flat(value: str) -> str:
    """A cell or paragraph on one line: its tabs and line breaks become spaces."""

    return re.sub(r"[\t\r\n]", " ", value)


def _column_number(reference: str) -> int | None:
    letters = re.match(r"[A-Z]+", reference or "")
    if letters is None:
        return None
    number = 0
    for letter in letters.group(0):
        number = number * 26 + ord(letter) - 64
    return number


def _worksheets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """(sheet name, worksheet file) in the workbook's order; without a workbook, each
    worksheet file in number order, named by its file."""

    names = set(archive.namelist())
    if {"xl/workbook.xml", "xl/_rels/workbook.xml.rels"} <= names:
        targets = {
            node.get("Id"): node.get("Target", "")
            for node in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        sheets = []
        for node in ET.fromstring(archive.read("xl/workbook.xml")).iter():
            if _xml_local(node.tag) != "sheet":
                continue
            relation = next(
                (v for k, v in node.attrib.items() if _xml_local(k) == "id"), None
            )
            target = targets.get(relation, "").lstrip("/")
            path = target if target.startswith("xl/") else f"xl/{target}"
            if path in names:
                sheets.append((node.get("name") or posixpath.basename(path), path))
        if sheets:
            return sheets
    found = [n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)]
    found.sort(key=lambda n: int(re.sub(r"\D", "", posixpath.basename(n))))
    return [(posixpath.basename(n).removesuffix(".xml"), n) for n in found]


def workbook_to_text(archive: zipfile.ZipFile) -> str:
    """Each sheet, in the workbook's order, as ``## <name> (<n> rows)`` and then exactly one
    line per row of the sheet: line ``r`` is row ``r`` (an empty line for a row with no value)
    and each cell sits at its own column, tab-separated, so a table read back from the text
    keeps the sheet, row and column it came from. A cell's tabs and line breaks become spaces."""

    names = set(archive.namelist())
    shared: list[str] = []
    if "xl/sharedStrings.xml" in names:
        for item in ET.fromstring(archive.read("xl/sharedStrings.xml")):
            shared.append(
                "".join(t.text or "" for t in item.iter() if _xml_local(t.tag) == "t")
            )
    out = []
    for name, path in _worksheets(archive):
        rows: dict[int, dict[int, str]] = {}
        number = 0
        for row in ET.fromstring(archive.read(path)).iter():
            if _xml_local(row.tag) != "row":
                continue
            number = int(row.get("r") or number + 1)
            cells, column = rows.setdefault(number, {}), 0
            for cell in row:
                if _xml_local(cell.tag) != "c":
                    continue
                column = _column_number(cell.get("r", "")) or column + 1
                value = next((v.text for v in cell if _xml_local(v.tag) == "v"), None)
                if cell.get("t") == "s" and value is not None:
                    value = shared[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(
                        t.text or "" for t in cell.iter() if _xml_local(t.tag) == "t"
                    )
                if value:
                    cells[column] = _flat(value)
        last = max(rows, default=0)
        out.append(f"## {name} ({last} row{'' if last == 1 else 's'})")
        for r in range(1, last + 1):
            cells = rows.get(r, {})
            width = max(cells, default=0)
            out.append("\t".join(cells.get(c, "") for c in range(1, width + 1)))
    return "\n".join(out)


def sheet_heading(line: str) -> tuple[str, int] | None:
    """The sheet name and row count of a ``workbook_to_text`` heading line, or None."""

    match = _SHEET_HEADING.match(line)
    return (match.group(1), int(match.group(2))) if match else None


_TEXT_EXTENSIONS = (
    ".txt",
    ".tsv",
    ".tab",
    ".vcf",
    ".bed",
    ".gff",
    ".gff3",
    ".md",
    ".xml",
    ".json",
)
_ARCHIVE_HEADER = "[file] "


def _extension(name: str) -> str:
    return os.path.splitext(name.split("?")[0].split("#")[0])[1].lower()


def _csv_to_text(data: bytes) -> str:
    rows = csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace")))
    return "\n".join("\t".join(row) for row in rows)


def pdf_to_text(data: bytes) -> str | None:
    """Each page's text under its ``## Page n`` heading; None when the PDF does not open
    or writes no text (a scan)."""

    try:
        pages = [
            page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages
        ]
    except (PyPdfError, ValueError, KeyError, TypeError):
        return None
    if not any(text.strip() for text in pages):
        return None
    return "\n".join(
        f"## Page {number}\n{text.strip()}" for number, text in enumerate(pages, 1)
    )


def file_to_text(name: str, data: bytes) -> str | None:
    """Text of one file chosen by its name's extension; None when there is no reader."""

    extension = _extension(name)
    if extension in (".xlsx", ".docx"):
        return office_zip_to_text(data)
    if extension == ".pdf":
        return pdf_to_text(data)
    if extension == ".csv":
        return _csv_to_text(data)
    if extension in _TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="replace")
    return None


@dataclass(frozen=True)
class ArchiveFile:
    """One file of a zip: its text, or the extension that has no reader here."""

    name: str
    text: str | None
    unread: str = ""


def _archive_files(data: bytes, prefix: str = "", depth: int = 0):
    archive = zipfile.ZipFile(io.BytesIO(data))
    for entry in archive.infolist():
        if entry.is_dir():
            continue
        name, content = prefix + entry.filename, archive.read(entry)
        if _extension(name) == ".zip" and depth < 2:
            try:
                yield from _archive_files(content, f"{name}/", depth + 1)
                continue
            except zipfile.BadZipFile:
                pass
        extension = _extension(name) or "no extension"
        try:
            text = file_to_text(name, content)
        except PARSE_ERRORS as error:
            # A supported format that does not parse is unread, with its parser's
            # error.
            yield ArchiveFile(name, None, f"{extension} ({type(error).__name__})")
            continue
        yield ArchiveFile(name, text, "" if text is not None else extension)


def archive_to_text(data: bytes) -> str | None:
    """Each file of a zip (nested zips opened) after a JSON header line with its name and
    the exact length of its text, so no file's content can pass for a header; or None."""

    try:
        files = list(_archive_files(data))
    except zipfile.BadZipFile:
        return None
    parts = []
    for file in files:
        if file.text is None:
            parts.append(
                _ARCHIVE_HEADER + json.dumps({"name": file.name, "unread": file.unread})
            )
        else:
            header = json.dumps({"name": file.name, "characters": len(file.text)})
            parts.append(f"{_ARCHIVE_HEADER}{header}\n{file.text}")
    return "\n".join(parts)


def archive_files(text: str) -> list[ArchiveFile] | None:
    """The files ``archive_to_text`` wrote, or None when ``text`` is not such an archive."""

    files, at = [], 0
    while at < len(text):
        end = text.find("\n", at)
        end = len(text) if end < 0 else end
        if not text.startswith(_ARCHIVE_HEADER, at):
            return None
        try:
            header = json.loads(text[at + len(_ARCHIVE_HEADER) : end])
            if "characters" in header:
                at = end + 1 + header["characters"]
                files.append(ArchiveFile(header["name"], text[end + 1 : at]))
            else:
                at = end
                files.append(ArchiveFile(header["name"], None, header["unread"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        at += 1  # the newline between files
    return files


def document_text(url: str, media_type: str, body: bytes) -> str | None:
    """A document's text, read by its served media type, then by its content and name;
    None when there is no reader here."""

    kind, _, parameters = media_type.partition(";")
    kind = kind.strip().lower()
    charset = re.search(r"charset=([\w.-]+)", parameters, re.IGNORECASE)
    try:
        served = body.decode(charset.group(1) if charset else "utf-8", errors="replace")
    except LookupError:
        served = body.decode("utf-8", errors="replace")
    if kind in ("text/html", "application/xhtml+xml"):
        return html_to_text(served, url)
    if kind == "application/pdf":
        return pdf_to_text(body)
    if kind == "text/csv" or _extension(url) == ".csv":
        return _csv_to_text(body)
    if kind.startswith("text/") or kind in ("application/json", "application/xml"):
        return served
    # Hosts such as S3 serve every file as generic bytes, so read by content, then name.
    # Only a zip served or named as one is an archive of files; a .pptx is a zip too.
    if zipfile.is_zipfile(io.BytesIO(body)):
        office = office_zip_to_text(body)
        if office is not None:
            return office
        archive = ("application/zip", "application/x-zip-compressed")
        if kind in archive or _extension(url) == ".zip":
            return archive_to_text(body)
    return file_to_text(url, body)


def parse_jats(xml_text: str, pmcid: str) -> tuple[str, list[tuple[str, str]]] | None:
    """Text of a JATS article and its supplement hrefs. Tables become tab-separated rows; the
    article title, the abstract and each section title become markdown headings."""

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    lines: list[str] = []

    def paragraph(node: ET.Element, depth: int) -> None:
        # A table floated inside the paragraph keeps its own lines; the words around it stay
        # paragraphs of their own.
        def keep(words: list[str]) -> None:
            text = " ".join("".join(words).split())
            if text:
                lines.append(text)

        words = [node.text or ""]
        for child in node:
            if _xml_local(child.tag) in ("table-wrap", "table-wrap-group"):
                keep(words)
                walk(child, depth)
                words = []
            else:
                words.append("".join(child.itertext()))
            words.append(child.tail or "")
        keep(words)

    def walk(node: ET.Element, depth: int) -> None:
        tag = _xml_local(node.tag)
        if tag == "tr":
            lines.append(
                "\t".join(" ".join("".join(c.itertext()).split()) for c in node)
            )
            return
        if tag == "p" and any(
            _xml_local(c.tag) in ("table-wrap", "table-wrap-group") for c in node
        ):
            paragraph(node, depth)
            return
        if tag in ("p", "title", "article-title", "label", "caption"):
            text = " ".join("".join(node.itertext()).split())
            if tag == "article-title":
                text = f"# {text}"
            elif tag == "title" and depth:
                text = f"{'#' * (depth + 1)} {text}"
            if text.strip("# "):
                lines.append(text)
            return
        if tag in ("sec", "abstract"):
            depth += 1
            titled = any(_xml_local(child.tag) == "title" for child in node)
            if tag == "abstract" and not titled:
                lines.append("## Abstract")
        for child in node:
            walk(child, depth)

    walk(root, 0)
    supplements = [
        (label, f"https://europepmc.org/articles/{pmcid}/bin/{href}")
        for label, href in _jats_supplements(root)
    ]
    return "\n".join(lines), supplements


def _jats_supplements(root: ET.Element) -> list[tuple[str, str]]:
    supplements = []
    for node in root.iter():
        if _xml_local(node.tag) in ("supplementary-material", "media"):
            href = next((v for k, v in node.attrib.items() if k.endswith("href")), None)
            if href:
                label = " ".join("".join(node.itertext()).split())[:300] or href
                supplements.append((label, href))
    return supplements


_MARKDOWN_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!<>|~])")


def normalize_text(text: str) -> str:
    """Undo markdown escaping (XP\\_0139 -> XP_0139) and unify whitespace and dashes:
    the text a Run paper's document is stored and read as."""

    text = _MARKDOWN_ESCAPE.sub(r"\1", text)
    text = text.replace("**", " ")  # bold table cells from HTML-to-markdown conversion
    text = text.replace("\r\n", "\n").replace("\u00a0", " ").replace("\u2009", " ")
    for dash in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2212"):
        text = text.replace(dash, "-")
    text = text.replace("\u2192", ">").replace("-\u203a", ">").replace("\u203a", ">")
    text = re.sub(r"(?<![*\w])\*([ACGTN]{1,20})\*(?![*\w])", r"\1", text)  # *A* italics
    return text


FileRole = Literal["article", "supplement", "supplement_archive", "supplied"]


@dataclass(frozen=True)
class SourceFile:
    """One file of a Run paper as retained: a download's document or a supplied file."""

    name: str
    url: str
    role: FileRole
    media_type: str
    content: bytes = field(repr=False)


@dataclass(frozen=True)
class FileReading:
    """One file, or one member of a zip, as read: its text, or why it has none."""

    source: int  # the index of its file in the paper's files
    name: str  # the file's name, or the member's path inside its zip
    text: str | None = field(default=None, repr=False)
    reason: str = ""


@dataclass(frozen=True)
class PaperDocumentText:
    """One readable document: ``main`` (the article, or a supplied file) or
    ``supp:<name>``, and its normalized text, the text every reading offset refers to."""

    document_id: str
    source: int
    name: str
    text: str = field(repr=False)


@dataclass(frozen=True)
class DocumentTexts:
    documents: tuple[PaperDocumentText, ...]
    readings: tuple[FileReading, ...]


# The errors a supported format's parser raises on a malformed file: such a file is
# unread with the error's type as its reason, never an application fault.
# LookupError covers a missing part (KeyError) and an index out of its table
# (IndexError, a shared string an.xlsx cell names but does not hold; r2).
PARSE_ERRORS = (
    ET.ParseError,
    zipfile.BadZipFile,
    zlib.error,
    csv.Error,
    LookupError,
    ValueError,
)


def _reading(index: int, name: str, text: str | None, reason: str = "") -> FileReading:
    """A reading; text that is only whitespace is no text (#420 SP5)."""
    if text is not None and not text.strip():
        return FileReading(index, name, reason="no text")
    return FileReading(index, name, text, reason)


def _readings(index: int, file: SourceFile) -> list[FileReading]:
    try:
        return _converted(index, file)
    except PARSE_ERRORS as error:
        return [_reading(index, file.name, None, f"unreadable: {type(error).__name__}")]


def _converted(index: int, file: SourceFile) -> list[FileReading]:
    if file.role == "article":
        parsed = parse_jats(file.content.decode("utf-8", errors="replace"), "")
        if parsed is None:
            return [_reading(index, file.name, None, "not JATS")]
        return [_reading(index, file.name, parsed[0])]
    text = (
        archive_to_text(file.content)
        if file.role == "supplement_archive"
        else document_text(file.url, file.media_type, file.content)
    )
    if text is None:
        reason = "no files" if file.role == "supplement_archive" else "no reader"
        return [_reading(index, file.name, None, reason)]
    members = archive_files(text)
    if members is None:
        return [_reading(index, file.name, text)]
    if not members:
        return [_reading(index, file.name, None, "no files")]
    return [
        _reading(
            index,
            member.name,
            member.text,
            "" if member.text is not None else f"no reader for {member.unread}",
        )
        for member in members
    ]


def document_texts(files: Sequence[SourceFile]) -> DocumentTexts:
    """A paper's readable documents. The article, or a supplied file that is no zip,
    is ``main``; every other readable file is ``supp:<file name>``. A file is the same
    source as one already given only when its name and its text are that file's (the
    zip's copy of a bucket file); any other file of a name already given keeps its own
    path (``supp:S2/table.tsv``), or that path numbered (``supp:table.tsv#2``)."""

    readings = [
        reading
        for index, file in enumerate(files)
        for reading in _readings(index, file)
    ]
    documents: dict[str, PaperDocumentText] = {}
    given: set[tuple[str, str]] = set()  # each file given, by its name and text
    for reading in readings:
        if reading.text is None:
            continue
        text = normalize_text(reading.text)
        main = files[reading.source].role in ("article", "supplied") and (
            reading.name == files[reading.source].name
        )
        if main and "main" not in documents:
            documents["main"] = PaperDocumentText(
                "main", reading.source, reading.name, text
            )
            continue
        name = posixpath.basename(reading.name)
        if (name, text) in given:
            continue  # the same file again, through another route
        given.add((name, text))
        key = next(
            key
            for key in itertools.chain(
                (name, reading.name),
                (f"{reading.name}#{n}" for n in itertools.count(2)),
            )
            if f"supp:{key}" not in documents
        )
        documents[f"supp:{key}"] = PaperDocumentText(
            f"supp:{key}", reading.source, reading.name, text
        )
    return DocumentTexts(tuple(documents.values()), tuple(readings))
