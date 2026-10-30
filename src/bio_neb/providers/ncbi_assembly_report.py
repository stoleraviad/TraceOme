"""NCBI's assembly report, copied: the one reader of an assembly's sequence names (#467).

The report (`<assembly>_assembly_report.txt` in the assembly's FTP directory) lists every
sequence of the assembly, assembled molecules, unlocalized and unplaced scaffolds, alt loci
and patches, with ten tab-separated columns. Each row is copied as NCBI writes it, with no
filter, ranking or normalising (ADR 0049); only NCBI's own ``na`` ("no such name") reads as
``None``. Rows are read lazily from the retained bytes, so a large report (wheat's 91,589
rows, a fragmented assembly's millions) is never held as a whole row list: a search or a
membership check copies only the rows it returns.

Planning's genomic FASTA description (`ncbi_datasets.fetch_assembly_sequences`) and the
Investigation's lookups (`ncbi_investigation.fetch_sequence_names`) both read it here.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from dataclasses import dataclass

COLUMNS = (
    "Sequence-Name",
    "Sequence-Role",
    "Assigned-Molecule",
    "Assigned-Molecule-Location/Type",
    "GenBank-Accn",
    "Relationship",
    "RefSeq-Accn",
    "Assembly-Unit",
    "Sequence-Length",
    "UCSC-style-name",
)
"""The report's columns, in its order."""


class ReportUnreadable(ValueError):
    """An answer that is not an assembly report (or names none): the reader's typing of
    what it cannot read, never a verdict on a name it holds."""


@dataclass(frozen=True)
class SequenceRecord:
    """One row of an assembly report, every column copied."""

    sequence_name: str
    sequence_role: str
    assigned_molecule: str | None
    location_type: str | None
    genbank: str | None
    relationship: str
    refseq: str | None
    assembly_unit: str
    length: int
    ucsc: str | None

    @property
    def names(self) -> frozenset[str]:
        """Every name the row gives its sequence: its sequence name, GenBank and RefSeq
        accessions and UCSC-style name."""
        return frozenset(
            name
            for name in (self.sequence_name, self.genbank, self.refseq, self.ucsc)
            if name is not None
        )

    @property
    def accessions(self) -> frozenset[str]:
        """The sequence's record ids: its GenBank and RefSeq accessions."""
        return frozenset(
            accession for accession in (self.genbank, self.refseq) if accession
        )

    def copied(self) -> dict[str, str | int | None]:
        """The row as the Tool shows it: each column under NCBI's name."""
        return dict(
            zip(
                COLUMNS,
                (
                    self.sequence_name,
                    self.sequence_role,
                    self.assigned_molecule,
                    self.location_type,
                    self.genbank,
                    self.relationship,
                    self.refseq,
                    self.assembly_unit,
                    self.length,
                    self.ucsc,
                ),
                strict=True,
            )
        )


def _named(value: str) -> str | None:
    return None if value in {"", "na"} else value


def report_rows(body: bytes) -> Iterator[SequenceRecord]:
    """The report's rows in its order, read one line at a time; a line that is not
    ten columns with a length, or bytes that are not text, raise `ReportUnreadable`."""
    for raw in io.BytesIO(body):
        try:
            line = raw.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as error:
            raise ReportUnreadable("NCBI's assembly report was not text.") from error
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != len(COLUMNS) or not fields[8].isdigit():
            raise ReportUnreadable(
                "NCBI's assembly report has a row that is not ten columns."
            )
        yield SequenceRecord(
            sequence_name=fields[0],
            sequence_role=fields[1],
            assigned_molecule=_named(fields[2]),
            location_type=_named(fields[3]),
            genbank=_named(fields[4]),
            relationship=fields[5],
            refseq=_named(fields[6]),
            assembly_unit=fields[7],
            length=int(fields[8]),
            ucsc=_named(fields[9]),
        )


def report_header(body: bytes) -> tuple[str, ...]:
    """The report's own header lines (assembly name, organism, taxid, ...), as written,
    without their ``#``; the column line is not one."""
    header = []
    for raw in io.BytesIO(body):
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line.startswith("#"):
            break
        text = line.lstrip("#").strip()
        if text and not text.startswith(COLUMNS[0]):
            header.append(text)
    return tuple(header)


def record_of(body: bytes, accession: str) -> SequenceRecord | None:
    """The one row whose GenBank or RefSeq accession is exactly ``accession``: the
    membership check of a model-named sequence. ``None`` when no row has it."""
    for record in report_rows(body):
        if accession in record.accessions:
            return record
    return None


def search_rows(
    body: bytes, text: str, *, offset: int, limit: int
) -> tuple[tuple[SequenceRecord, ...], int]:
    """The rows any of whose columns holds ``text`` (case-insensitive), in the report's
    order: ``limit`` of them from ``offset``, and how many there are in all, so the rest
    stays reachable (ADR 0049, bounded views)."""
    wanted = text.casefold()
    found: list[SequenceRecord] = []
    total = 0
    for record in report_rows(body):
        if wanted and not any(
            value is not None and wanted in str(value).casefold()
            for value in record.copied().values()
        ):
            continue
        if offset <= total < offset + limit:
            found.append(record)
        total += 1
    return tuple(found), total


def report_url(links_body: bytes) -> str:
    """The assembly report's URL in the one FTP directory NCBI Datasets' ``links``
    answer names: ``<directory>/<directory name>_assembly_report.txt``. The directory
    name is NCBI's and is not derived from the assembly's name (`TAIR10.1`)."""
    try:
        document = json.loads(links_body)
        directories = [
            link["resource_link"]
            for link in document.get("assembly_links", [])
            if link.get("assembly_link_type") == "FTP_LINK"
        ]
    except (ValueError, TypeError, AttributeError, KeyError) as error:
        raise ReportUnreadable("NCBI's links answer was not read.") from error
    if len(directories) != 1 or not isinstance(directories[0], str):
        raise ReportUnreadable("NCBI's links answer names no one FTP directory.")
    directory = directories[0].rstrip("/")
    return f"{directory}/{directory.rsplit('/', 1)[-1]}_assembly_report.txt"


__all__ = [
    "COLUMNS",
    "ReportUnreadable",
    "SequenceRecord",
    "record_of",
    "report_header",
    "report_rows",
    "report_url",
    "search_rows",
]
