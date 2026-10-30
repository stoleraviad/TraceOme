"""The one tool table: pinned bioconda packages and the tool entries that run them."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bio_neb.investigation.observation_defaults import CoverageDefaults
from bio_neb.investigation.observation_tools import ToolName
from bio_neb.investigation.run_measurement import SAM_FLAGS, depth_read_filters

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Pin(StrictModel):
    """One bioconda package version: its BioContainers image and its local executable.

    Each runner checks its own pinned facts (the image digest for Docker; the `--version`
    line for a local process, which records the executable's digest it observed). They
    come from where the tool was pulled or installed; while a runner's facts are unset,
    that runner refuses the run.
    """

    package: str
    version: str
    build: str | None = None
    image_digest: Digest | None = None
    version_line: str | None = None

    @property
    def image_repository(self) -> str:
        return f"quay.io/biocontainers/{self.package}"

    @property
    def image_reference(self) -> str:
        tag = self.version if self.build is None else f"{self.version}--{self.build}"
        return f"{self.image_repository}:{tag}"


class ToolEntry(StrictModel):
    """One tool-table row: what it runs, its argv template, named inputs and outputs.

    What it runs is either a pinned package (`pin`; argv[0] is the package's executable)
    or a tool the Run's probe names (`probed`, main's `ObservationToolsProbe`; argv[0] is
    that tool's name), never both. `{<input name>}` in argv becomes that input's path as
    the tool sees it, `{outputs}` the output directory. `optional_inputs` maps each input
    a run may leave out to the argv value its placeholder takes when it is left out.
    `parameters` names values (not files) each run supplies, such as a region.
    `stdout`, when set, names the output file that receives the tool's stdout.
    """

    pin: Pin | None = None
    probed: ToolName | None = None
    argv: tuple[str, ...]
    inputs: tuple[str, ...] = ()
    optional_inputs: dict[str, str] = Field(default_factory=dict)
    parameters: tuple[str, ...] = ()
    stdout: str | None = None
    outputs: tuple[str, ...]

    @model_validator(mode="after")
    def one_identity(self) -> Self:
        if (self.pin is None) == (self.probed is None):
            raise ValueError("an entry runs either a pinned package or a probed tool")
        if self.probed is not None and self.argv[0] != self.probed:
            raise ValueError("a probed entry's argv[0] is the probed tool's name")
        if set(self.inputs) & set(self.optional_inputs):
            raise ValueError("an input is either required or optional")
        if set(self.parameters) & {*self.inputs, *self.optional_inputs, "outputs"}:
            raise ValueError("a parameter is not an input or the output directory")
        return self


def _installed(package: str, version: str, build: str, line: str) -> Pin:
    return Pin(package=package, version=version, build=build, version_line=line)


# Plan versions as installed under the approved plan (`locks/<package>-linux-64.lock`):
# bioconda build and the first `--version` line. Any install of that version runs
# (Spack, a Galaxy image or the tool's own environment), and each run records the
# executable digest it observed. Image digests stay unpinned until an image route runs.
# samtools and bcftools are never pinned: they are the Run's own, as its tool probe
# names them.
PINS = {
    pin.package: pin
    for pin in (
        _installed(
            "mosdepth",
            "0.3.14",
            "h87be163_2",
            "mosdepth 0.3.14",
        ),
        _installed(
            "goleft",
            "0.2.6",
            "he881be0_1",
            "goleft Version: 0.2.6",
        ),
        _installed(
            "multiqc",
            "1.35",
            "pyhdfd78af_2",
            "multiqc, version 1.35",
        ),
    )
}


# mosdepth's fixed window when no target BED is given (bp), and the depths whose
# covered bases each window or target reports besides `minimum_depth`.
COVERAGE_WINDOW = "500"
COVERAGE_DEPTHS = (1, 20, 30)


def coverage_entry(defaults: CoverageDefaults) -> ToolEntry:
    """mosdepth over the Run's BAM with the product's read-depth parameters.

    Fixed windows by default; a `targets` BED switches the same entry to depth by target.
    `minimum_depth` starts the callable bin (`--quantize 0:1:<depth>:`) and is one of the
    thresholds. mosdepth's default mode counts overlapping mates once and leaves
    deletions out, with no BAQ; a setting it cannot apply is refused. It has no base
    quality filter, so `minimum_base_quality` is not applied.
    """
    unsupported = [
        f"{name}={getattr(defaults, name)!r}"
        for name, supported in (
            ("overlapping_pairs", "count_once"),
            ("deletion_spanning_reads", "excluded"),
            ("applies_baq", False),
        )
        if getattr(defaults, name) != supported
    ]
    unknown = [flag for flag in defaults.excluded_flags if flag not in SAM_FLAGS]
    if unknown:
        unsupported.append(f"excluded_flags {unknown}")
    if unsupported:
        raise ValueError(f"mosdepth cannot apply {', '.join(unsupported)}")
    depths = sorted({defaults.minimum_depth, *COVERAGE_DEPTHS})
    # No coverage, below the minimum depth, callable; a bin collapses when the minimum
    # depth is 1 (no "below" bin) or 0 (every base is callable).
    minimum = defaults.minimum_depth
    bins = sorted({0, min(1, minimum), minimum})
    flags = sum(SAM_FLAGS[flag] for flag in set(defaults.excluded_flags))
    return ToolEntry(
        pin=PINS["mosdepth"],
        argv=(
            "mosdepth",
            "--no-per-base",
            "--threads",
            "4",
            "--by",
            "{targets}",
            "--thresholds",
            ",".join(str(depth) for depth in depths),
            "--quantize",
            "".join(f"{boundary}:" for boundary in bins),
            "--mapq",
            str(defaults.minimum_mapping_quality),
            "--flag",
            str(flags),
            "{outputs}/coverage",
            "{bam}",
        ),
        inputs=("bam", "bam_index"),
        optional_inputs={"targets": COVERAGE_WINDOW},
        outputs=tuple(
            f"coverage.{name}"
            for name in (
                "mosdepth.global.dist.txt",
                "mosdepth.region.dist.txt",
                "mosdepth.summary.txt",
                "quantized.bed.gz",
                "regions.bed.gz",
                "thresholds.bed.gz",
            )
        ),
    )


def depth_entry(defaults: CoverageDefaults) -> ToolEntry:
    """samtools depth over one region of the Run's BAM, through its index.

    Every position of the region is printed (`-a`), and the read filters are exactly
    `CoverageDefaults`: `-g` clears samtools' default excluded flags and `-G` sets the
    product's. samtools depth applies no BAQ, so `applies_baq=True` is refused.
    """
    return ToolEntry(
        probed="samtools",
        argv=(*_depth_argv(defaults), "-r", "{region}", "{bam}"),
        inputs=("bam", "bam_index"),
        parameters=("region",),
        stdout="depth.tsv",
        outputs=("depth.tsv",),
    )


def depth_windows_entry(defaults: CoverageDefaults) -> ToolEntry:
    """samtools depth over every window of a BED (`-b`), with `depth_entry`'s filters
    (#405), streamed over the windows' reads (`samtools_view_windows`).

    `-b` streams its whole input, and with `-a` it walks every position of every
    contig (15 s even over a 200 KB slice of HG001), so `-a` is left out: a position
    the output does not list has no read that passes the filters, depth zero.
    """
    return ToolEntry(
        probed="samtools",
        argv=(*_depth_argv(defaults, every_position=False), "-b", "{windows}", "{bam}"),
        inputs=("bam", "windows"),
        stdout="depth.tsv",
        outputs=("depth.tsv",),
    )


def _depth_argv(
    defaults: CoverageDefaults, *, every_position: bool = True
) -> tuple[str, ...]:
    """The product's samtools depth read filters (`run_measurement`, #414), with
    `-a` for every position of a region."""
    return (
        "samtools",
        "depth",
        *(("-a",) if every_position else ()),
        *depth_read_filters(defaults),
    )


def indexcov_entry(excluded: str) -> ToolEntry:
    """goleft indexcov over the Run's BAM and its `.bai`, with the contigs it leaves out
    (`--excludepatt`, a regular expression). It infers no sex (`--sex ''`): the health
    view reads where the data has coverage (owner, #406 G2)."""
    return ToolEntry(
        pin=PINS["goleft"],
        argv=(
            "goleft",
            "indexcov",
            "--directory",
            "{outputs}/indexcov",
            "--excludepatt",
            excluded,
            "--sex",
            "",
            "{bam}",
        ),
        inputs=("bam", "bam_index"),
        outputs=tuple(
            f"indexcov/{name}"
            for name in (
                "index.html",
                "indexcov-indexcov.bed.gz",
                "indexcov-indexcov.ped",
                "indexcov-indexcov.roc",
            )
        ),
    )


# The summaries MultiQC reads, by the health role of each file (`health.py`).
MULTIQC_SUMMARIES = (
    "samtools_stats",
    "samtools_flagstat",
    "samtools_idxstats",
    "bcftools_stats",
    "mosdepth_global_dist",
    "mosdepth_region_dist",
    "mosdepth_summary",
    "indexcov_ped",
)

# The Run's read and variant health. samtools and bcftools are the Run's own, as its
# tool probe names them; the index is written into its own fresh directory and passed
# back beside the BAM (`/inputs/aligned.bam.csi`), never written next to it.
TOOLS = {
    "samtools_index": ToolEntry(
        probed="samtools",
        argv=("samtools", "index", "-c", "-o", "{outputs}/aligned.bam.csi", "{bam}"),
        inputs=("bam",),
        outputs=("aligned.bam.csi",),
    ),
    "samtools_index_bai": ToolEntry(
        probed="samtools",
        argv=("samtools", "index", "-b", "-o", "{outputs}/aligned.bam.bai", "{bam}"),
        inputs=("bam",),
        outputs=("aligned.bam.bai",),
    ),
    "samtools_stats": ToolEntry(
        probed="samtools",
        argv=("samtools", "stats", "{bam}"),
        inputs=("bam",),
        stdout="samtools-stats.txt",
        outputs=("samtools-stats.txt",),
    ),
    "samtools_flagstat": ToolEntry(
        probed="samtools",
        argv=("samtools", "flagstat", "{bam}"),
        inputs=("bam",),
        stdout="samtools-flagstat.txt",
        outputs=("samtools-flagstat.txt",),
    ),
    "samtools_idxstats": ToolEntry(
        probed="samtools",
        argv=("samtools", "idxstats", "{bam}"),
        inputs=("bam", "bam_index"),
        stdout="samtools-idxstats.tsv",
        outputs=("samtools-idxstats.tsv",),
    ),
    "bcftools_stats": ToolEntry(
        probed="bcftools",
        argv=("bcftools", "stats", "-s", "-", "{vcf}"),
        inputs=("vcf",),
        stdout="bcftools-stats.txt",
        outputs=("bcftools-stats.txt",),
    ),
    # Whole-genome coverage: mosdepth over the BAM and its CSI; indexcov reads only a
    # `.bai` (it cannot read the CSI), built by `samtools_index_bai`.
    "mosdepth": coverage_entry(CoverageDefaults()),
    # One region's depth and reference bases, each read through its index.
    "samtools_depth_region": depth_entry(CoverageDefaults()),
    "samtools_faidx_region": ToolEntry(
        probed="samtools",
        argv=("samtools", "faidx", "{reference}", "{region}"),
        inputs=("reference", "reference_index"),
        parameters=("region",),
        stdout="region.fa",
        outputs=("region.fa",),
    ),
    # Every Work item's window in a fixed number of runs: the reads overlapping
    # any window through the BAM's index (`-M`, the multi-region iterator, each read
    # once), their depth (`samtools_depth_windows`, built per run by
    # `depth_windows_entry`) and the bases through the `.fai`, one `contig:start-end`
    # per line of `regions`.
    "samtools_view_windows": ToolEntry(
        probed="samtools",
        argv=(
            "samtools",
            "view",
            "-M",
            "-L",
            "{windows}",
            "-b",
            "-o",
            "{outputs}/windows.bam",
            "{bam}",
        ),
        inputs=("bam", "bam_index", "windows"),
        outputs=("windows.bam",),
    ),
    "samtools_depth_windows": depth_windows_entry(CoverageDefaults()),
    "samtools_faidx_windows": ToolEntry(
        probed="samtools",
        argv=("samtools", "faidx", "{reference}", "-r", "{regions}"),
        inputs=("reference", "reference_index", "regions"),
        stdout="windows.fa",
        outputs=("windows.fa",),
    ),
    # One report over the Run's read, variant and coverage summaries. Each input is
    # named by the health role of the file it reads; `sample_names` renames each
    # file-named sample to the `run-artifacts` role of the file it summarizes.
    # indexcov's `.roc` is left out: MultiQC 1.35's goleft module breaks on its RefSeq
    # contig names (the `.ped` parses); the ROC stays in indexcov's own report.
    "multiqc": ToolEntry(
        pin=PINS["multiqc"],
        argv=(
            "multiqc",
            "--no-version-check",
            "--no-ansi",
            "--replace-names",
            "{sample_names}",
            "--outdir",
            "{outputs}",
            "--filename",
            "multiqc",
            "--data-format",
            "json",
            *(f"{{{name}}}" for name in MULTIQC_SUMMARIES),
        ),
        inputs=("sample_names", *MULTIQC_SUMMARIES),
        outputs=("multiqc.html", "multiqc_data/multiqc_data.json"),
    ),
}
