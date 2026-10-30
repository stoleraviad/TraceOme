"""Place a paper's targets on a Run's assembly and write its request file; no Run.

  python -m bio_neb.devtools.target_placement --assembly ACCESSION \\
      --paper-targets TARGETS --request REQUEST [--stated-sequences TABLE] \\
      [--resume RECORD --retry LABEL --accept LABEL] [--variant-cache DIR]

The developer entry over the product's `target_request.placement.place` and its one request
writer (#367): every NCBI GET is a product source operation on its own failure surface, and
outside a Run no contact journal is written (owner, #417 Q4), so the record counts the GETs
sent (``contact_count``), which a live plan watches. The flank move runs behind the
placement port; this entry holds no mover, so a flank step answers why and the next step
runs. ``python -m tools.reference_index place-targets`` runs this same entry with the
reference-index tool's bwa move behind the port (#367 readiness Q6).

The run's record is kept under ``<variant cache>/paper-targets/records/`` and prints as
JSON. Exit 0 when the ``/3`` file is written; 2 when a target waits for you (each listed,
with how to answer it) or a run input is refused before contact; 1 when the run stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from bio_neb.content import canonical_json
from bio_neb.providers.ncbi_datasets import UrllibNcbiTransport
from bio_neb.providers.ncbi_investigation import NcbiInvestigationSourceAdapter
from bio_neb.providers.source_transport import Opener
from bio_neb.target_request.ncbi_sources import NcbiSources
from bio_neb.target_request.placement import (
    PlacementRefused,
    PlacementRun,
    PlacementSources,
    Resume,
    TargetsFile,
    place,
    read_stated_sequences,
)
from bio_neb.target_request.records import (
    FlankMove,
    FlankMoveRequest,
    PaperTargetsPlaced,
    SourceFailure,
)
from bio_neb.target_request.request_document import write_request_file
from bio_neb.target_request.resolver import FlankMover, FlankMoveRefused

VARIANT_CACHE = (
    Path(__file__).resolve().parents[3] / ".bio-neb-local" / "variant-sources"
)


class NoFlankMover:
    """The development-only placement port with no mover: a flank step says why and the
    next one runs. A Run passes no mover at all, and gets the typed failure (#449)."""

    def sequence(self, accession: str) -> Path | None:
        return None

    def missing(self, accession: str) -> str:
        return "which this entry holds no flank mover for"

    def __call__(
        self, snv: FlankMoveRequest, source: Path, run_assembly: str
    ) -> FlankMove:
        raise FlankMoveRefused("no_flank_mover", "this entry holds no flank mover")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The entry's options, shared with the reference-index tool's ``place-targets``."""
    parser.add_argument(
        "--assembly", dest="accession", required=True, help="the Run's assembly"
    )
    parser.add_argument(
        "--paper-targets", required=True, type=Path, help="a paper-targets/1 record"
    )
    parser.add_argument("--request", required=True, type=Path, help="the /3 file")
    parser.add_argument(
        "--stated-sequences",
        type=Path,
        help="the Run's stated-sequence table (bio-neb.stated-sequences/1)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="a waiting run's record, answered by --retry/--accept",
    )
    parser.add_argument("--retry", action="append", default=[], help="LABEL: rerun it")
    parser.add_argument(
        "--accept", action="append", default=[], help="LABEL: write it as unplaced"
    )
    parser.add_argument("--variant-cache", type=Path, default=VARIANT_CACHE)


def run(
    arguments: argparse.Namespace, *, opener: Opener | None, mover: FlankMover
) -> int:
    """One placement run and its request file, with ``mover`` behind the port."""
    try:
        table = used = None
        if arguments.stated_sequences:
            table, used = read_stated_sequences(arguments.stated_sequences.read_bytes())
        resume = _resume(arguments)
    except OSError as error:
        _print({"refused": "input_unreadable", "detail": str(error)})
        return 2
    except PlacementRefused as refused:
        _print({"refused": refused.reason, "detail": refused.detail})
        return 2
    sources = PlacementSources(
        ncbi=NcbiSources(
            NcbiInvestigationSourceAdapter(UrllibNcbiTransport(opener=opener)),
            arguments.variant_cache / "gene-change",
        ),
        mover=mover,
    )
    targets = TargetsFile.read(arguments.paper_targets)
    placement = PlacementRun(
        assembly=arguments.accession, table=table, table_used=used, resume=resume
    )
    try:
        record = place(targets, placement, sources)
    except PlacementRefused as refused:
        _print({"refused": refused.reason, "detail": refused.detail})
        return 2
    record = write_request_file(record, targets, arguments.request)
    kept = canonical_json(record.model_dump(mode="json"))
    path = (
        arguments.variant_cache
        / "paper-targets"
        / "records"
        / f"{record.made_at.replace(':', '')}-{record.assembly_accession}-"
        f"{hashlib.sha256(kept.encode()).hexdigest()[:12]}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(kept)
    _print(record.model_dump(mode="json"))
    if record.stopped is not None:
        sys.stderr.write(
            f"stopped at {record.stopped.claim_label}: {record.stopped.error}\n"
        )
        return 1
    if record.waiting:
        sys.stderr.write(f"waiting for you ({path}):\n")
        for answer in record.answers:
            if answer.label in record.waiting:
                failure = answer.failure
                attempts = (
                    f" after {failure.contact_attempts} attempts"
                    if isinstance(failure, SourceFailure)
                    else ""
                )
                sys.stderr.write(
                    f"{answer.label} asks: {failure.failure_class}{attempts} on "
                    f"{failure.tool}: {failure.detail}\n"
                )
        sys.stderr.write(
            f"answer each with --resume {path} and --retry LABEL or --accept LABEL\n"
        )
        return 2
    return 0


def _resume(arguments: argparse.Namespace) -> Resume | None:
    """The owner's answers to a waiting run, read from its record."""
    if arguments.resume is None:
        if arguments.retry or arguments.accept:
            raise PlacementRefused(
                "resume_answers", "--retry and --accept answer a --resume record"
            )
        return None
    content = arguments.resume.read_bytes()
    try:
        record = PaperTargetsPlaced.model_validate_json(content)
    except ValidationError as error:
        raise PlacementRefused(
            "resume_unreadable",
            f"not a bio-neb.paper-targets-placed/6 record: {error}",
        ) from error
    return Resume(
        record=record,
        record_sha256=hashlib.sha256(content).hexdigest(),
        retry=frozenset(arguments.retry),
        accept=frozenset(arguments.accept),
    )


def _print(value: object) -> None:
    sys.stdout.write(json.dumps(json.loads(canonical_json(value)), indent=2) + "\n")


def main(argv: Sequence[str] | None = None, *, opener: Opener | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bio_neb.devtools.target_placement")
    add_arguments(parser)
    return run(parser.parse_args(argv), opener=opener, mover=NoFlankMover())


if __name__ == "__main__":
    sys.exit(main())
