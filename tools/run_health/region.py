"""Coverage and GC over one indexed slice of a Run's BAM and reference (#318).

`measure_range` answers a typed range (`contig:start-end`, 1-based and inclusive);
`measure_item` measures one Work item's position and the window around it. Both read
only the slice: the depth through the BAM's index (`samtools_depth_region`) and the
bases through the reference's `.fai` (`samtools_faidx_region`), each run through the
one runner port, so each run has its record. Nothing here changes a Run file or a
disposition.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from bio_neb.investigation.item_evidence import CoverageMeasurement
from bio_neb.investigation.observation_defaults import CoverageDefaults

from .runner import RunRecord, RunRefused, ToolRunner, run_tool
from .table import StrictModel, depth_entry

# A typed range: `NC_000019.10:44908584-44908784` (commas allowed in the numbers).
# Each lane process's time bound; the product's measurement has its own
# (`failure.StageLimits.process_seconds` for stage `run_health`).
TIMEOUT_SECONDS = 120
# The longest range one query reads, so an answer stays within seconds.
MAX_RANGE_BASES = 1_000_000
# Depths whose covered fraction every summary reports, besides `minimum_depth`.
SUMMARY_DEPTHS = (1, 20, 30)


class RegionFiles(StrictModel):
    """The Run's BAM and reference, each with the index the slice is read through."""

    bam: Path
    bam_index: Path
    reference: Path
    reference_index: Path


class RegionSummary(StrictModel):
    """Depth and GC over one range under the recorded read filters.

    `fraction_at_least` maps each reported depth to the fraction of the range's
    positions at or above it; `gc_fraction` is G and C over the A, C, G and T bases
    (None when the range has none), and `n_bases` counts the other bases.
    """

    contig: str
    start: int
    end: int
    bases: int
    mean_depth: float
    median_depth: float
    min_depth: int
    max_depth: int
    fraction_at_least: dict[str, float]
    gc_fraction: float | None
    n_bases: int
    parameters: CoverageDefaults


class RangeCoverage(StrictModel):
    """A typed range's summary and the two runs it was read from."""

    summary: RegionSummary
    depth_record: RunRecord
    reference_record: RunRecord


class ItemCoverage(StrictModel):
    """One position's measurement, for the item's coverage field, and its window.

    The window is lane evidence kept beside the field, never inside it.
    """

    contig: str
    one_based_position: int
    flank_bases: int
    measurement: CoverageMeasurement
    window: RangeCoverage


def measure_range(
    files: RegionFiles,
    text: str,
    *,
    runner: ToolRunner,
    output_dir: Path,
    defaults: CoverageDefaults | None = None,
) -> RangeCoverage:
    """Summarise a typed range; a bad range or a missing index is refused first."""
    coverage, _ = read_range(
        files, text, runner=runner, output_dir=output_dir, defaults=defaults
    )
    return coverage


def read_range(
    files: RegionFiles,
    text: str,
    *,
    runner: ToolRunner,
    output_dir: Path,
    defaults: CoverageDefaults | None = None,
) -> tuple[RangeCoverage, list[int]]:
    """`measure_range`'s summary and the depth at every position of the range."""
    contig, start, end = parse_range(text)
    contig_length(files, contig, start, end)
    return _measure(files, contig, start, end, runner, output_dir, defaults)


def measure_item(
    files: RegionFiles,
    contig: str,
    one_based_position: int,
    *,
    flank_bases: int,
    runner: ToolRunner,
    output_dir: Path,
    defaults: CoverageDefaults | None = None,
) -> ItemCoverage:
    """The depth at an item's position and the summary of the window around it.

    The window is the position with `flank_bases` on each side, kept within the
    contig, as the item's VCF window is.
    """
    defaults = defaults or CoverageDefaults()
    length = contig_length(files, contig, one_based_position, one_based_position)
    start = max(1, one_based_position - flank_bases)
    end = min(length, one_based_position + flank_bases)
    window, depths = _measure(files, contig, start, end, runner, output_dir, defaults)
    depth = depths[one_based_position - start]
    return ItemCoverage(
        contig=contig,
        one_based_position=one_based_position,
        flank_bases=flank_bases,
        measurement=CoverageMeasurement(
            parameters=defaults,
            depth=depth,
            mean_depth=float(depth),
            meets_minimum_depth=depth >= defaults.minimum_depth,
        ),
        window=window,
    )


def parse_range(text: str) -> tuple[str, int, int]:
    """A typed `contig:start-end`, split at its last colon; the contig is held to
    the reference's own names by `contig_length` (ADR 0049), never by its shape."""
    contig, _, bounds = text.strip().rpartition(":")
    first, _, last = bounds.partition("-")
    try:
        start, end = (int(value.replace(",", "")) for value in (first, last))
    except ValueError:
        start = end = None
    if not contig or start is None:
        raise RunRefused(
            "range_invalid", f"{text!r} is not contig:start-end (1-based, inclusive)"
        )
    if start < 1 or end < start:
        raise RunRefused(
            "range_invalid", f"{text!r} needs 1 <= start <= end, got {start}-{end}"
        )
    return contig, start, end


def contig_length(files: RegionFiles, contig: str, start: int, end: int) -> int:
    """Refuse a missing file or index, an unknown contig or a range off the contig;
    the contig's length otherwise."""
    lengths = reference_lengths(files)
    if contig not in lengths:
        raise RunRefused(
            "contig_unknown", f"{contig} is not a contig of {files.reference}"
        )
    if end > lengths[contig]:
        raise RunRefused(
            "range_beyond_contig",
            f"{contig}:{start}-{end} ends past the contig's {lengths[contig]} bases",
        )
    if end - start + 1 > MAX_RANGE_BASES:
        raise RunRefused(
            "range_too_long",
            f"{end - start + 1} bases; one query reads at most {MAX_RANGE_BASES}",
        )
    return lengths[contig]


def reference_lengths(files: RegionFiles) -> dict[str, int]:
    """Each contig's length from the `.fai`; a missing file or index is refused."""
    for name in ("bam_index", "reference_index"):
        path = getattr(files, name)
        if not path.is_file():
            raise RunRefused("index_missing", f"{name} {path} does not exist")
    for name in ("bam", "reference"):
        path = getattr(files, name)
        if not path.is_file():
            raise RunRefused("input_missing", f"{name} {path} does not exist")
    lengths = {}
    for line in files.reference_index.read_text().splitlines():
        name, length, *_ = line.split("\t")
        lengths[name] = int(length)
    return lengths


def _measure(
    files: RegionFiles,
    contig: str,
    start: int,
    end: int,
    runner: ToolRunner,
    output_dir: Path,
    defaults: CoverageDefaults | None,
) -> tuple[RangeCoverage, list[int]]:
    defaults = defaults or CoverageDefaults()
    region = f"{contig}:{start}-{end}"
    output_dir = output_dir.absolute()
    depth_record = run_tool(
        "samtools_depth_region",
        {"bam": files.bam, "bam_index": files.bam_index},
        output_dir / "depth",
        runner=runner,
        tools={"samtools_depth_region": depth_entry(defaults)},
        parameters={"region": region},
        timeout_seconds=TIMEOUT_SECONDS,
    )
    reference_record = run_tool(
        "samtools_faidx_region",
        {"reference": files.reference, "reference_index": files.reference_index},
        output_dir / "reference",
        runner=runner,
        parameters={"region": region},
        timeout_seconds=TIMEOUT_SECONDS,
    )
    depths = _depths(output_dir / "depth" / "depth.tsv", start, end)
    bases = "".join(
        line.strip()
        for line in (output_dir / "reference" / "region.fa").read_text().splitlines()
        if not line.startswith(">")
    )
    coverage = RangeCoverage(
        summary=summarize(contig, start, end, depths, bases, defaults),
        depth_record=depth_record,
        reference_record=reference_record,
    )
    return coverage, depths


def summarize(
    contig: str,
    start: int,
    end: int,
    depths: list[int],
    bases: str,
    defaults: CoverageDefaults,
) -> RegionSummary:
    """The summary of one range from its depth at every position and its bases."""
    bases = bases.upper()
    called = sum(bases.count(base) for base in "ACGT")
    gc = bases.count("G") + bases.count("C")
    count = len(depths)
    reported = sorted({defaults.minimum_depth, *SUMMARY_DEPTHS})
    return RegionSummary(
        contig=contig,
        start=start,
        end=end,
        bases=count,
        mean_depth=round(sum(depths) / count, 6),
        median_depth=float(statistics.median(depths)),
        min_depth=min(depths),
        max_depth=max(depths),
        fraction_at_least={
            str(depth): round(sum(d >= depth for d in depths) / count, 6)
            for depth in reported
        },
        gc_fraction=round(gc / called, 6) if called else None,
        n_bases=len(bases) - called,
        parameters=defaults,
    )


def _depths(path: Path, start: int, end: int) -> list[int]:
    """Depth at every position of the range, from `samtools depth -a` (zero where
    the slice has no line)."""
    depths = [0] * (end - start + 1)
    for line in path.read_text().splitlines():
        _, position, depth = line.split("\t")
        depths[int(position) - start] = int(depth)
    return depths
