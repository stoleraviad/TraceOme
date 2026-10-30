"""Test support for #467's copied sequence names and the model's named sequence.

`assembly_report_text` derives an FTP assembly report (the form `ncbi_sequence_names`
answers, `providers/ncbi_assembly_report.py`) from a retained NCBI Datasets
sequence-report body, row for row: its values are the retained ones; the derived columns
are ``Assigned-Molecule-Location/Type`` (``Chromosome``, else ``na``), ``Relationship``
(``=``) and ``Assembly-Unit`` (``Primary Assembly``), none of which a lookup reads. It is a
derived body, not a recorded GET.

`literal_choices` is the scripted model's naming: on each assembly a fake answers the
sequence names of, the one record that literally holds the written contig among its names,
quoted as the contig itself. A test of a non-literal or refused naming passes its own
choices instead.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from bio_neb.investigation.investigation import SequenceChoice
from bio_neb.providers.ncbi_assembly_report import COLUMNS, report_rows

OPERATION = "ncbi_sequence_names"


def assembly_report_text(sequence_report: bytes, *, header: str = "") -> bytes:
    """The FTP assembly report that holds a Datasets sequence report's rows."""
    lines = [f"# Assembly name:  {header}"] if header else []
    lines.append("# " + "\t".join(COLUMNS))
    for row in json.loads(sequence_report)["reports"]:
        lines.append(
            "\t".join(
                (
                    str(row.get("sequence_name") or row.get("chr_name") or "na"),
                    str(row.get("role") or "na"),
                    str(row.get("chr_name") or "na"),
                    "Chromosome" if row.get("chr_name") else "na",
                    str(row.get("genbank_accession") or "na"),
                    "=",
                    str(row.get("refseq_accession") or "na"),
                    "Primary Assembly",
                    str(row.get("length") or 0),
                    str(row.get("ucsc_style_name") or "na"),
                )
            )
        )
    return ("\n".join(lines) + "\n").encode()


def literal_choices(
    answers: Mapping[tuple[str, str], tuple[int, Any]], contig: str
) -> tuple[SequenceChoice, ...]:
    """The scripted model's choices: per answered assembly, the record literally named
    ``contig``, by its RefSeq accession, else its GenBank one."""
    choices = []
    for (operation, accession), (status, body) in answers.items():
        if operation != OPERATION or not 200 <= status < 300:
            continue
        try:
            held = [row for row in report_rows(body) if contig in row.names]
        except ValueError:
            continue
        if len(held) == 1:
            choices.append(
                SequenceChoice(
                    assembly_accession=accession,
                    record=held[0].refseq or held[0].genbank,
                    quote=contig,
                )
            )
    return tuple(choices)


def names_answers(accession: str, sequence_report: bytes) -> dict[str, bytes]:
    """The two GETs `ncbi_sequence_names` makes for ``accession``, by URL: NCBI
    Datasets' ``links`` answer naming a synthetic FTP directory, and the FTP assembly
    report derived from ``sequence_report`` (no GET)."""
    number = accession[4:13]
    directory = (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/"
        f"{accession[:3]}/{number[:3]}/{number[3:6]}/{number[6:]}/{accession}_x"
    )
    links = json.dumps(
        {
            "assembly_links": [
                {
                    "accession": accession,
                    "assembly_link_type": "FTP_LINK",
                    "resource_link": directory,
                }
            ]
        }
    ).encode()
    return {
        "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
        f"{accession}/links": links,
        f"{directory}/{accession}_x_assembly_report.txt": assembly_report_text(
            sequence_report
        ),
    }
