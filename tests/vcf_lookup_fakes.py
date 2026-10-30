"""Scripted outputs for the VCF lookup's commands, shared by offline adapters.

The lookup's preparation never asks for ``--version``, ``--list-samples`` or a named
``--output`` (the pysam dispatcher prints nothing for the first two and sends the
third to stdout), so a fake only needs to answer these four commands.
"""

from __future__ import annotations

import itertools
from typing import Literal

LookupCommand = Literal[
    "bgzf_derivation", "csi_derivation", "header_inventory", "variant_query"
]


def lookup_command(argv: tuple[str, ...]) -> LookupCommand:
    """Classify one argv the lookup compiles."""
    if argv[1:3] == ("view", "--no-version") and "--header-only" in argv:
        return "header_inventory"
    if argv[1:3] == ("view", "--no-version"):
        return "bgzf_derivation"
    if argv[1:3] == ("index", "--csi"):
        return "csi_derivation"
    if argv[1:4] == ("query", "--allow-undef-tags", "--regions-file"):
        return "variant_query"
    raise AssertionError(f"the VCF lookup never compiles {argv!r}")


def query_region(files, argv: tuple[str, ...]) -> tuple[str, int, int]:
    """A ``variant_query``'s region as ``contig, start, end``: the regions file its argv
    names (#467), which the fake adapter kept in ``files`` when the lookup wrote it."""
    contig, start, end = files[argv[4]].decode().rstrip("\n").split("\t")
    return contig, int(start), int(end)


def query_position(files, argv: tuple[str, ...]) -> int:
    """The one-based position of a ``variant_query`` region."""
    return query_region(files, argv)[1]


def vcf_header(
    sample_id: str, contigs: tuple[tuple[str, int], ...] = (("NC_000019.10", 58617616),)
) -> bytes:
    lines = [
        "##fileformat=VCFv4.2",
        *(f"##contig=<ID={name},length={length}>" for name, length in contigs),
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + sample_id,
    ]
    return ("\n".join(lines) + "\n").encode()


def query_row(
    contig: str,
    position: int,
    ref: str,
    alts: str,
    sample_id: str,
    genotype: str,
    *,
    filters: str = ".",
    qual: str = ".",
    gq: str = ".",
    dp: str = ".",
    ad: str = ".",
) -> bytes:
    """One row in the lookup's query format."""
    return (
        "\t".join(
            (
                contig,
                str(position),
                ref,
                alts,
                filters,
                qual,
                sample_id,
                genotype,
                gq,
                dp,
                ad,
            )
        )
        + "\n"
    ).encode()


_TABLE_READING_CALLS = itertools.count(1)


def chunk_rows(messages) -> list[dict]:
    """The rows of the chunk prompt the table reader was sent."""
    import json

    from pydantic_ai.messages import UserPromptPart

    for message in messages:
        for part in getattr(message, "parts", ()):
            if isinstance(part, UserPromptPart) and '"rows":' in str(part.content):
                return json.loads(part.content)["rows"]
    raise AssertionError("the table reader was sent no chunk rows")


def table_reading_reply(info, messages=None, *, readings=None, call_id=None):
    """The table reader's one output whenever its stage offers it: one reading
    per row (``readings`` maps a row to its own reading, else no_concern)."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    from bio_neb.journey import canonical_json

    if not any(tool.name == "record_item_readings" for tool in info.output_tools):
        return None
    rows = chunk_rows(messages)
    chosen = readings or {}
    return ModelResponse(
        parts=[
            ToolCallPart(
                "record_item_readings",
                canonical_json(
                    {
                        "readings": [
                            chosen.get(
                                row["row"],
                                {
                                    "row": row["row"],
                                    "reading": "no_concern",
                                    "concerns": [],
                                    "note": f"Row {row['row']}: {row['outcome']}.",
                                },
                            )
                            for row in rows
                        ]
                    }
                ),
                tool_call_id=call_id or f"table-reading-{next(_TABLE_READING_CALLS)}",
            )
        ]
    )


class TableReaderModel:
    """A scripted table reader: one reading per row of each chunk it is sent."""

    def __init__(self, *, readings: dict | None = None) -> None:
        from tests.model_endpoint_fakes import ScriptedModelEndpoint

        self.readings = readings
        self.calls = 0
        self.chunks: list[list[dict]] = []
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        self.calls += 1
        reply = table_reading_reply(
            info,
            messages,
            readings=self.readings,
            call_id=f"table-reading-{self.calls}",
        )
        if reply is None:
            raise AssertionError("the table reader was asked outside its stage")
        self.chunks.append(chunk_rows(messages))
        return reply


class EmptyWindowAdapter:
    """Answers each item's window with no records; it runs nothing else."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        # Each window's regions file, as the query wrote it.
        self.files: dict[str, bytes] = {}

    def path_exists(self, path: str) -> bool:
        return path in self.files

    def write_fresh_file(self, path: str, data: bytes) -> None:
        if self.path_exists(path):
            raise FileExistsError(path)
        self.files[path] = data

    def run(self, argv, *, timeout_seconds, stdout_path=None):  # noqa: ANN001, ANN201
        from bio_neb.investigation.vcf_observation import CompletedProcess

        del timeout_seconds
        assert stdout_path is None and lookup_command(argv) == "variant_query"
        self.calls.append(tuple(argv))
        return CompletedProcess(0, b"", b"")


def run_table_reading(journey, adapter=None, *, model=None) -> None:
    """Run the post-approval VCF table stage to its fold, as the coordinator would.

    Finalization reads a collection only from its fold (ADR 0047), so a test that
    compiles it directly after the lookups runs this first. A closed stage writes
    nothing. ``journey`` is a path, or an owned Journey that stays current.
    """
    from bio_neb.investigation.vcf_table_reading import advance_table_reading
    from bio_neb.journey import Journey

    selected = model or TableReaderModel()
    owner = journey if isinstance(journey, Journey) else None
    while advance_table_reading(
        owner or Journey.open(journey),
        environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
        model_factory=selected.model,
        adapter=adapter or EmptyWindowAdapter(),
    ):
        pass
