"""Coverage and GC of every Work item of a Run in one batched pass (#405 spike).

One `samtools view -M -L` reads the reads overlapping every item's window through the
BAM's index, one `samtools depth -b` their depth and one `samtools faidx -r` the windows'
bases, each through the one runner port, so the batch has three run records whatever
the number of items. The result is split into one
measurement per span: an item's span is its REF, or an insertion's anchor base (#368's
`_target_span`), and its window that span with `flank_bases` each side, kept within the
contig, as its VCF window is. Work items at one span share it. An item the reference
cannot place is flagged by itself; the others are still measured. Viewing only: nothing
here changes a Run file or a disposition.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from bio_neb.investigation.investigation_observation import (
    validate_investigation_work_item_started_fact,
)
from bio_neb.investigation.item_evidence import CoverageMeasurement, coverage_span
from bio_neb.investigation.observation_defaults import CoverageDefaults
from bio_neb.journey import Journey
from bio_neb.parabricks.result import ParabricksExecutionOutcome

from .cache import read_cache
from .health import MANIFEST_NAME, HealthManifest, read_run, run_probe
from .local_process import ProbedProcessRunner
from .region import (
    TIMEOUT_SECONDS,
    RangeCoverage,
    RegionFiles,
    RegionSummary,
    contig_length,
    measure_range,
    reference_lengths,
    summarize,
)
from .runner import RunRecord, RunRefused, ToolRunner, run_tool
from .table import StrictModel, depth_windows_entry

ITEMS_SCHEMA = "bio-neb.run-health-items-coverage/2"
# The batch's result, written last into a Run's batch directory (`measure_run_items`).
ITEMS_NAME = "items-coverage.json"


class ItemTarget(StrictModel):
    """One Work item's place on the Run's reference: its contig, its VCF position and
    its REF (empty for an insertion)."""

    work_item_id: str
    contig: str
    one_based_position: int
    ref: str


class SpanCoverage(StrictModel):
    """The measurement of one span and its window.

    `measurement.depth` is the least depth over the span (a deletion is only as
    covered as its least-covered base); `window` is the lane evidence beside it.
    `uncovered_positions` are the span's positions with no read that passes the
    filters, and `uncovered` says how they combine: none, part or the whole span.
    """

    work_item_ids: tuple[str, ...]
    contig: str
    span_start: int
    span_end: int
    flank_bases: int
    measurement: CoverageMeasurement
    window: RegionSummary
    uncovered_positions: tuple[int, ...]
    uncovered: Literal["none", "part", "whole"]


class ItemFlag(StrictModel):
    """An item the batch could not place, or one whose measurement it is unsure of,
    with its place (contig and span) and the reason: `low_depth` (below the minimum
    depth while its window has reads) or `reference_n` (an N base of the reference in
    its span)."""

    work_item_id: str
    contig: str
    span_start: int
    span_end: int
    code: Literal["contig_unknown", "span_beyond_contig", "low_depth", "reference_n"]
    detail: str


class ItemsCoverage(StrictModel):
    """A batch's spans, its flagged items (not placed), its unsure items (measured, to
    be checked each by itself), its three run records (None when no item could be
    placed, so nothing ran), its wall-clock seconds and the read filters it ran under
    (`parameters`)."""

    schema_version: Literal["bio-neb.run-health-items-coverage/2"] = ITEMS_SCHEMA
    parameters: CoverageDefaults
    spans: tuple[SpanCoverage, ...]
    flagged: tuple[ItemFlag, ...]
    unsure: tuple[ItemFlag, ...]
    slice_record: RunRecord | None
    depth_record: RunRecord | None
    reference_record: RunRecord | None
    seconds: float


def measure_run_items(
    journey_file: Path,
    cache_dir: Path,
    health_dir: Path,
    output_dir: Path,
    *,
    flank_bases: int = 100,
    probes: Sequence[ItemTarget] = (),
    runner: ToolRunner | None = None,
    defaults: CoverageDefaults | None = None,
) -> ItemsCoverage:
    """Every Work item of the Run, and each extra `probes` position, measured in one
    batch (`measure_items`) over the Run's own BAM and the lane's cached reference.

    The BAM is the Run's local copy (`health.read_run`); its index is the one the Run's
    health report wrote (`health_dir`, whose manifest must have read that same BAM).
    samtools is the Run's own (`run_probe`) unless `runner` is given. A Run, cache or
    health report that cannot supply these is refused before anything runs. The
    result is written last as `<output_dir>/items-coverage.json`.
    """
    outcome, files = run_region_files(journey_file, cache_dir, health_dir)
    coverage = measure_items(
        files,
        (*work_item_targets(Path(outcome.journey_file)), *probes),
        flank_bases=flank_bases,
        runner=runner or ProbedProcessRunner(run_probe(outcome)),
        output_dir=output_dir,
        defaults=defaults,
    )
    with (output_dir / ITEMS_NAME).open("x") as written:
        written.write(coverage.model_dump_json(indent=2))
    return coverage


def run_region_files(
    journey_file: Path, cache_dir: Path, health_dir: Path
) -> tuple[ParabricksExecutionOutcome, RegionFiles]:
    """The Run's own BAM with the index its health report wrote, and the lane's cached
    reference; a Run, cache or report that cannot supply them is refused."""
    outcome, inputs = read_run(journey_file)
    cache = read_cache(cache_dir, outcome)
    (bam,) = [item for item in inputs if item.role == "aligned_bam"]
    return outcome, RegionFiles(
        bam=Path(bam.path),
        bam_index=_health_index(health_dir, bam.sha256),
        reference=Path(cache.files["reference_fasta"].path),
        reference_index=Path(cache.files["reference_fai"].path),
    )


def _health_index(health_dir: Path, bam_sha256: str) -> Path:
    """The BAM index a health report of this BAM wrote (its `bam_csi`)."""
    manifest_file = health_dir / MANIFEST_NAME
    try:
        manifest = HealthManifest.model_validate_json(manifest_file.read_bytes())
    except FileNotFoundError:
        raise RunRefused(
            "file_missing", f"no health manifest at {manifest_file}"
        ) from None
    except (OSError, ValueError) as error:
        raise RunRefused(
            "health_unreadable",
            f"{manifest_file} is not a readable health manifest: {error}",
        ) from error
    read = {item.role: item.sha256 for item in manifest.inputs}
    if read.get("aligned_bam") != bam_sha256:
        raise RunRefused(
            "health_other_bam",
            f"{manifest_file} read BAM sha256 {read.get('aligned_bam')}; the Run's "
            f"is {bam_sha256}",
        )
    (index,) = [item for item in manifest.files if item.role == "bam_csi"]
    return health_dir / index.path


def work_item_targets(journey_file: Path) -> tuple[ItemTarget, ...]:
    """The Run's Work items, in their selected order, as the Journey started them."""
    facts = [
        validate_investigation_work_item_started_fact(event)
        for event in Journey.open(journey_file).events_view()
    ]
    return tuple(
        ItemTarget(
            work_item_id=fact.work_item_id,
            contig=fact.target.contig,
            one_based_position=fact.target.one_based_position,
            ref=fact.target.ref,
        )
        for fact in sorted(
            (fact for fact in facts if fact is not None),
            key=lambda fact: fact.selected_ordinal,
        )
    )


def measure_items(
    files: RegionFiles,
    items: Sequence[ItemTarget],
    *,
    flank_bases: int,
    runner: ToolRunner,
    output_dir: Path,
    defaults: CoverageDefaults | None = None,
) -> ItemsCoverage:
    """Measure every placeable item's span in one slice, one depth and one reference
    run; the reference run goes beside the slice and depth runs, which follow each
    other.

    `output_dir` must not exist; it is created even when nothing is measured, and
    receives `windows.bed`, `windows.txt` and each run's directory (`slice/`, `depth/`,
    `reference/`). A missing file or index is refused before anything is written. An item is unsure when its span is below the minimum
    depth while its window has reads, or holds an N base of the reference.
    """
    started = time.monotonic()
    defaults = defaults or CoverageDefaults()
    lengths = reference_lengths(files)
    spans: dict[tuple[str, int, int], list[str]] = {}
    flagged = []
    for item in items:
        first, last = span(item)
        if item.contig not in lengths:
            flagged.append(
                ItemFlag(
                    work_item_id=item.work_item_id,
                    contig=item.contig,
                    span_start=first,
                    span_end=last,
                    code="contig_unknown",
                    detail=f"{item.contig} is not a contig of {files.reference}",
                )
            )
        elif last > lengths[item.contig]:
            flagged.append(
                ItemFlag(
                    work_item_id=item.work_item_id,
                    contig=item.contig,
                    span_start=first,
                    span_end=last,
                    code="span_beyond_contig",
                    detail=f"{item.contig}:{first}-{last} ends past the contig's "
                    f"{lengths[item.contig]} bases",
                )
            )
        else:
            spans.setdefault((item.contig, first, last), []).append(item.work_item_id)
    output_dir = fresh_directory(output_dir)
    if not spans:
        return ItemsCoverage(
            parameters=defaults,
            spans=(),
            flagged=tuple(flagged),
            unsure=(),
            slice_record=None,
            depth_record=None,
            reference_record=None,
            seconds=round(time.monotonic() - started, 3),
        )
    windows = {
        key: (max(1, key[1] - flank_bases), min(lengths[key[0]], key[2] + flank_bases))
        for key in spans
    }
    # Distinct spans can share one window (both clamped to a short contig): each window
    # is read once.
    unique = sorted({(key[0], *window) for key, window in windows.items()})
    bed = output_dir / "windows.bed"
    bed.write_text(
        "".join(f"{contig}\t{start - 1}\t{end}\n" for contig, start, end in unique)
    )
    regions = output_dir / "windows.txt"
    regions.write_text(
        "".join(f"{contig}:{start}-{end}\n" for contig, start, end in unique)
    )

    def reads() -> tuple[RunRecord, RunRecord]:
        slice_record = run_tool(
            "samtools_view_windows",
            {"bam": files.bam, "bam_index": files.bam_index, "windows": bed},
            output_dir / "slice",
            runner=runner,
            timeout_seconds=TIMEOUT_SECONDS,
        )
        return slice_record, run_tool(
            "samtools_depth_windows",
            {"bam": output_dir / "slice" / "windows.bam", "windows": bed},
            output_dir / "depth",
            runner=runner,
            tools={"samtools_depth_windows": depth_windows_entry(defaults)},
            timeout_seconds=TIMEOUT_SECONDS,
        )

    def bases() -> RunRecord:
        return run_tool(
            "samtools_faidx_windows",
            {
                "reference": files.reference,
                "reference_index": files.reference_index,
                "regions": regions,
            },
            output_dir / "reference",
            runner=runner,
            timeout_seconds=TIMEOUT_SECONDS,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        read, based = pool.submit(reads), pool.submit(bases)
        (slice_record, depth_record), reference_record = read.result(), based.result()
    depth = _depths(output_dir / "depth" / "depth.tsv")
    sequences = _sequences(output_dir / "reference" / "windows.fa")
    measured = []
    unsure = []
    for key, work_item_ids in spans.items():
        contig, first, last = key
        start, end = windows[key]
        depths = [
            depth.get((contig, position), 0) for position in range(start, end + 1)
        ]
        window_bases = sequences[f"{contig}:{start}-{end}"]
        over_span = depths[first - start : last - start + 1]
        least = min(over_span)
        uncovered = tuple(
            position
            for position, value in zip(range(first, last + 1), over_span, strict=True)
            if value == 0
        )
        measured.append(
            SpanCoverage(
                work_item_ids=tuple(work_item_ids),
                contig=contig,
                span_start=first,
                span_end=last,
                flank_bases=flank_bases,
                measurement=CoverageMeasurement(
                    parameters=defaults,
                    depth=least,
                    mean_depth=sum(over_span) / len(over_span),
                    meets_minimum_depth=least >= defaults.minimum_depth,
                ),
                window=summarize(contig, start, end, depths, window_bases, defaults),
                uncovered_positions=uncovered,
                uncovered="none"
                if not uncovered
                else "whole"
                if len(uncovered) == len(over_span)
                else "part",
            )
        )
        reason = _unsure(
            least,
            max(depths),
            window_bases[first - start : last - start + 1],
            defaults,
        )
        if reason is not None:
            code, detail = reason
            unsure += [
                ItemFlag(
                    work_item_id=work_item_id,
                    contig=contig,
                    span_start=first,
                    span_end=last,
                    code=code,
                    detail=f"{contig}:{first}-{last}: {detail}",
                )
                for work_item_id in work_item_ids
            ]
    return ItemsCoverage(
        parameters=defaults,
        spans=tuple(measured),
        flagged=tuple(flagged),
        unsure=tuple(unsure),
        slice_record=slice_record,
        depth_record=depth_record,
        reference_record=reference_record,
        seconds=round(time.monotonic() - started, 3),
    )


def _unsure(
    least: int, most: int, span_bases: str, defaults: CoverageDefaults
) -> tuple[Literal["low_depth", "reference_n"], str] | None:
    """Why a measured span needs a check by itself, if it does: an N base of the
    reference in it, or a depth below the minimum while its window has reads (no read
    in the whole window is a plain absence, not a doubt)."""
    if any(base not in "ACGT" for base in span_bases.upper()):
        return "reference_n", f"the reference reads {span_bases} over the span"
    if least < defaults.minimum_depth and most > 0:
        return (
            "low_depth",
            f"depth {least} under the minimum {defaults.minimum_depth}; the window "
            f"reaches {most}",
        )
    return None


def measure_items_alone(
    files: RegionFiles,
    items: Sequence[ItemTarget],
    *,
    flank_bases: int,
    runner: ToolRunner,
    output_dir: Path,
) -> tuple[RangeCoverage, ...]:
    """Each distinct span's window read by itself (`measure_range`, one directory
    each, `<output_dir>/<n>/`), the path the batch replaces; an item the reference
    cannot place is left out, as the batch flags it."""
    windows: dict[str, None] = {}
    for item in items:
        first, last = span(item)
        try:
            length = contig_length(files, item.contig, first, last)
        except RunRefused as refused:
            if refused.code not in {"contig_unknown", "range_beyond_contig"}:
                raise
            continue
        start = max(1, first - flank_bases)
        end = min(length, last + flank_bases)
        windows[f"{item.contig}:{start}-{end}"] = None
    output_dir = fresh_directory(output_dir)
    return tuple(
        measure_range(files, window, runner=runner, output_dir=output_dir / str(index))
        for index, window in enumerate(windows)
    )


def span(item: ItemTarget) -> tuple[int, int]:
    """The one-based positions the item's VCF record must reach (#368), as the
    Journey's coverage fact checks them (`coverage_span`)."""
    return coverage_span(item)


def _depths(path: Path) -> dict[tuple[str, int], int]:
    """Depth by contig and position from `samtools depth -b` (absent: zero)."""
    depths = {}
    for line in path.read_text().splitlines():
        contig, position, value = line.split("\t")
        depths[(contig, int(position))] = int(value)
    return depths


def fresh_directory(output_dir: Path) -> Path:
    """Create the output directory, which must not exist (`output_dir_exists`)."""
    output_dir = output_dir.absolute()
    try:
        output_dir.mkdir(parents=True)
    except FileExistsError:
        raise RunRefused(
            "output_dir_exists",
            f"{output_dir} exists; each batch writes a fresh directory",
        ) from None
    return output_dir


def _sequences(path: Path) -> dict[str, str]:
    """Each `faidx -r` record's bases by its `contig:start-end` header; a repeated
    header is its own record, and the first is kept."""
    sequences: dict[str, list[str]] = {}
    current: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            current = []
            sequences.setdefault(line[1:].strip(), current)
        else:
            current.append(line.strip())
    return {name: "".join(lines) for name, lines in sequences.items()}
