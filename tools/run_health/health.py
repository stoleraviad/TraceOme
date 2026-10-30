"""One health report per Run: every whole-Run health tool over the Run's own BAM and VCF
(the local copies its Investigation keeps, #363), MultiQC over their summaries, and one
manifest of the files they wrote."""

from __future__ import annotations

import gzip
import hashlib
import time
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from bio_neb.agentic_journey.runtime_config import RuntimeConfigError
from bio_neb.investigation.bam_materialization import CurrentBamAcquisitionReceipt
from bio_neb.investigation.observation_tools import (
    ObservationToolsProbe,
    probe_observation_tools,
)
from bio_neb.investigation.vcf_materialization import VcfMaterializationReceipt
from bio_neb.parabricks.result import (
    ParabricksExecutionOutcome,
)
from bio_neb.providers.object_locations import object_location

from .cache import read_cache, read_outcome
from .local_process import ProbedProcessRunner
from .runner import RunRecord, RunRefused, ToolRunner, run_tool
from .table import (
    COVERAGE_WINDOW,
    MULTIQC_SUMMARIES,
    TOOLS,
    StrictModel,
    indexcov_entry,
)

MANIFEST_SCHEMA = "bio-neb.run-health-manifest/2"
MANIFEST_NAME = "run-health-manifest.json"
# The MultiQC sample renames (`table.py`, the `multiqc` entry), derived from the Run
# (`derive_arguments`) and written beside the manifest.
SAMPLE_NAMES_NAME = "multiqc-sample-names.tsv"
# The regions the Run's reads cover, as BED, written beside the manifest.
COVERED_NAME = "covered-regions.bed"
# Where a Run keeps its Investigation's local inputs, beside its Journey.
INVESTIGATION = "investigation"
# How MultiQC names the sample of each summary, by the entry that wrote it: by the
# summary's own file name (the default), by the Run file the entry read (bcftools
# stats names its VCF), or kept as MultiQC names it (indexcov's `.ped`, as without
# derived renames). MultiQC's renames match substrings, so only names it uses are
# renamed: a rename of a name inside another (`indexcov-indexcov` in
# `indexcov-indexcov.ped`) would leave part of it (`aligned_bam.ped`, seen in a live run).
_SAMPLE_NAMED_BY = {"bcftools_stats": "input", "goleft_indexcov": "kept"}

# The Run's files the health tools read, by their `run-artifacts` role: the directory
# under `investigation/` that holds the local copy, its receipt and the receipt's
# contract.
_RUN_FILES = {
    "aligned_bam": (
        "bam",
        "current-bam-acquisition-receipt.json",
        CurrentBamAcquisitionReceipt,
    ),
    "VCF": ("vcf", "vcf-materialization-receipt.json", VcfMaterializationReceipt),
}
_RUN_ROLES = tuple(_RUN_FILES)

# Each health run in order: its table entry and, per input, the role of the file it
# reads (a Run role above or a health role an earlier run wrote).
_RUNS = (
    ("samtools_index", {"bam": "aligned_bam"}),
    ("samtools_stats", {"bam": "aligned_bam"}),
    ("samtools_flagstat", {"bam": "aligned_bam"}),
    ("samtools_idxstats", {"bam": "aligned_bam", "bam_index": "bam_csi"}),
    ("bcftools_stats", {"vcf": "VCF"}),
    ("mosdepth", {"bam": "aligned_bam", "bam_index": "bam_csi"}),
    ("samtools_index_bai", {"bam": "aligned_bam"}),
    ("goleft_indexcov", {"bam": "aligned_bam", "bam_index": "bam_bai"}),
    (
        "multiqc",
        {"sample_names": "multiqc_sample_names"}
        | {summary: summary for summary in MULTIQC_SUMMARIES},
    ),
)

# The role of each file a health run writes: by its path under the run's directory,
# else the run's role for its other files (None: an unnamed file is refused).
_ROLES: dict[str, tuple[dict[str, str], str | None]] = {
    "samtools_index": ({"aligned.bam.csi": "bam_csi"}, None),
    "samtools_stats": ({"samtools-stats.txt": "samtools_stats"}, None),
    "samtools_flagstat": ({"samtools-flagstat.txt": "samtools_flagstat"}, None),
    "samtools_idxstats": ({"samtools-idxstats.tsv": "samtools_idxstats"}, None),
    "bcftools_stats": ({"bcftools-stats.txt": "bcftools_stats"}, None),
    "mosdepth": (
        {
            "coverage.mosdepth.global.dist.txt": "mosdepth_global_dist",
            "coverage.mosdepth.region.dist.txt": "mosdepth_region_dist",
            "coverage.mosdepth.summary.txt": "mosdepth_summary",
            "coverage.regions.bed.gz": "mosdepth_regions",
            "coverage.thresholds.bed.gz": "mosdepth_thresholds",
            "coverage.quantized.bed.gz": "mosdepth_quantized",
        },
        "mosdepth_bed_index",
    ),
    "samtools_index_bai": ({"aligned.bam.bai": "bam_bai"}, None),
    "goleft_indexcov": (
        {
            "indexcov/index.html": "indexcov_report",
            "indexcov/indexcov-indexcov.bed.gz": "indexcov_bed",
            "indexcov/indexcov-indexcov.ped": "indexcov_ped",
            "indexcov/indexcov-indexcov.roc": "indexcov_roc",
        },
        "indexcov_chromosome_plot",
    ),
    "multiqc": (
        {
            "multiqc.html": "multiqc_report",
            "multiqc_data/multiqc_data.json": "multiqc_data",
        },
        "multiqc_data_file",
    ),
}


class HealthInput(StrictModel):
    """One of the Run's files the health tools read: its `run-artifacts` output entry
    (role, location, byte count, sha256), the local copy the Run's Investigation
    downloaded (`path`) and that copy's receipt (`receipt`)."""

    role: str
    location: str
    byte_count: int
    sha256: str
    path: str
    receipt: str


class HealthFile(StrictModel):
    """One file a health run wrote. `path` is relative to the manifest's directory;
    `record` names the run (its entry) in the manifest's `records`."""

    role: str
    path: str
    byte_count: int
    sha256: str
    record: str


class HealthArguments(StrictModel):
    """The health tools' arguments, derived from the Run.

    `indexcov_excluded` leaves out no contig (ADR 0049); `sample_names` maps each
    MultiQC sample name to the `run-artifacts` role of the Run file it summarizes. indexcov
    infers no sex (`--sex ''`): the health view asks where the data has coverage, not
    which chromosomes are sex chromosomes (owner, #406 G2).
    """

    reference_index: str
    indexcov_excluded: str
    sample_names: tuple[tuple[str, str], ...]


class CoveredContig(StrictModel):
    """Where one contig's reads lie: its length and mapped reads (`samtools idxstats`)
    and the bases of its covered mosdepth windows, with the first and last covered
    base (1-based; None when no window is covered)."""

    contig: str
    length: int
    mapped_reads: int
    covered_bases: int
    first_covered: int | None
    last_covered: int | None


class CoveredRegions(StrictModel):
    """The regions of the reference the Run's reads cover (owner, #406 G2: the data may
    cover only a region, `<contig>:<start>-<end>`, and the health view finds it).

    A window of mosdepth's fixed `window_bases` is covered when its mean depth is at
    least `minimum_mean_depth`; adjacent covered windows merge into one region of the
    BED at `path` (relative to the manifest's directory). `contigs` lists every contig
    with a mapped read or a covered window.
    """

    window_bases: int
    minimum_mean_depth: float
    path: str
    byte_count: int
    sha256: str
    contigs: tuple[CoveredContig, ...]


class HealthManifest(StrictModel):
    """Everything a reader needs about one Run's health: which Run, the files the tools
    read, every file they wrote and each run's record (`bio-neb.run-health-record/1`)."""

    schema_version: Literal["bio-neb.run-health-manifest/2"] = MANIFEST_SCHEMA
    journey_id: str
    journey_file: str
    run_root: str
    execution_id: str
    outcome_event_id: str
    inputs: tuple[HealthInput, ...]
    files: tuple[HealthFile, ...]
    records: dict[str, RunRecord]
    arguments: HealthArguments
    covered: CoveredRegions
    # Each run's wall-clock seconds, from just before `run_tool` to its record.
    entry_seconds: dict[str, float]


def derive_arguments(
    reference_index: Path, inputs: tuple[HealthInput, ...]
) -> HealthArguments:
    """The health arguments the Run supplies: from its reference and its files.

    indexcov leaves out no contig: no pattern of ours judges a sequence name by its
    shape (ADR 0049, #465). MultiQC's renames map each sample name MultiQC gives a
    summary (its file name, or for bcftools the Run's VCF's) to its Run role;
    indexcov's is kept.
    """
    stems = {item.role: Path(item.path).name.split(".", 1)[0] for item in inputs}
    names = {}
    for entry, reads in _RUNS:
        run_roles = {role for role in reads.values() if role in _RUN_ROLES}
        named_by = _SAMPLE_NAMED_BY.get(entry, "summary")
        if len(run_roles) != 1 or named_by == "kept":
            continue
        (run_role,) = run_roles
        if named_by == "input":
            if run_role in stems:
                names[stems[run_role]] = run_role
            continue
        for path, role in _ROLES[entry][0].items():
            if role in MULTIQC_SUMMARIES:
                names[Path(path).name.split(".", 1)[0]] = run_role
    return HealthArguments(
        reference_index=str(reference_index),
        indexcov_excluded="^$",
        sample_names=tuple(sorted(names.items())),
    )


def covered_regions(
    idxstats: Path, windows: Path, bed: Path, *, minimum_mean_depth: float = 1.0
) -> CoveredRegions:
    """The regions the reads cover, from `samtools idxstats` and mosdepth's fixed
    windows (`coverage.regions.bed.gz`); writes the merged regions to `bed`."""
    lengths, mapped = {}, {}
    for line in idxstats.read_text().splitlines():
        contig, length, reads, _ = line.split("\t")
        if contig != "*":
            lengths[contig], mapped[contig] = int(length), int(reads)
    regions: list[list] = []
    with gzip.open(windows, "rt") as rows:
        for line in rows:
            contig, start, end, mean = line.rstrip("\n").split("\t")
            if float(mean) < minimum_mean_depth:
                continue
            if regions and regions[-1][0] == contig and regions[-1][2] == int(start):
                regions[-1][2] = int(end)
            else:
                regions.append([contig, int(start), int(end)])
    content = "".join(f"{contig}\t{start}\t{end}\n" for contig, start, end in regions)
    bed.write_text(content)
    by_contig: dict[str, list] = {}
    for contig, start, end in regions:
        by_contig.setdefault(contig, []).append((start, end))
    return CoveredRegions(
        window_bases=int(COVERAGE_WINDOW),
        minimum_mean_depth=minimum_mean_depth,
        path=bed.name,
        byte_count=len(content.encode()),
        sha256=hashlib.sha256(content.encode()).hexdigest(),
        contigs=tuple(
            CoveredContig(
                contig=contig,
                length=length,
                mapped_reads=mapped[contig],
                covered_bases=sum(
                    end - start for start, end in by_contig.get(contig, ())
                ),
                first_covered=by_contig[contig][0][0] + 1
                if contig in by_contig
                else None,
                last_covered=by_contig[contig][-1][1] if contig in by_contig else None,
            )
            for contig, length in lengths.items()
            if mapped[contig] or contig in by_contig
        ),
    )


def read_run(
    journey_file: Path,
) -> tuple[ParabricksExecutionOutcome, tuple[HealthInput, ...]]:
    """The Run's verified Execution result and the local copies of its BAM and VCF that
    its Investigation downloaded beside its Journey (#363), each held to the Run's
    verified output of its role by the copy's receipt.

    The receipts are read for their meaning; the copies they name are trusted, never
    reread (ADR 0029). A Journey that cannot be read (`run_unreadable`), a Run without
    a completed Execution (`outcome_unavailable`) and a Run that holds no local copy, as
    one from before #363 (`run_file_missing`), are refused by name.
    """
    outcome = read_outcome(journey_file)
    root = Path(outcome.journey_file).parent / INVESTIGATION
    verified = {item.role: item for item in outcome.verified_objects}
    inputs = []
    for role, (directory, receipt_name, contract) in _RUN_FILES.items():
        if role not in verified:
            raise RunRefused("file_missing", f"the Run lists no verified {role}")
        descriptor = verified[role]
        receipt_file = root / directory / receipt_name
        try:
            content = receipt_file.read_bytes()
        except FileNotFoundError:
            raise RunRefused(
                "run_file_missing",
                f"the Run holds no local {role} ({receipt_file} is absent): a Run from "
                "before #363, or its Investigation has not downloaded it",
            ) from None
        except OSError as error:
            raise RunRefused(
                "run_file_unreadable", f"{receipt_file} cannot be read: {error}"
            ) from error
        try:
            receipt = contract.model_validate_json(content)
        except ValidationError as error:
            raise RunRefused(
                "receipt_invalid", f"{receipt_file} is not a {contract.__name__}"
            ) from error
        if receipt.descriptor != descriptor:
            raise RunRefused(
                "file_changed",
                f"{receipt_file} names {receipt.descriptor.object_key} sha256 "
                f"{receipt.descriptor.sha256}; the Run's {role} is "
                f"{descriptor.object_key} sha256 {descriptor.sha256}",
            )
        copy = (
            receipt.materialized_bam
            if isinstance(receipt, CurrentBamAcquisitionReceipt)
            else receipt.materialized_vcf
        )
        inputs.append(
            HealthInput(
                role=role,
                location=object_location(descriptor.bucket, descriptor.object_key),
                byte_count=descriptor.byte_count,
                sha256=descriptor.sha256,
                path=copy.path,
                receipt=str(receipt_file),
            )
        )
    return outcome, tuple(inputs)


def run_probe(outcome: ParabricksExecutionOutcome) -> ObservationToolsProbe:
    """The tool probe the Run's Investigation kept (`investigation/tools`): its
    samtools and bcftools, read back without running or writing anything."""
    directory = Path(outcome.journey_file).parent / INVESTIGATION / "tools"
    try:
        return probe_observation_tools(None, directory)
    except RuntimeConfigError as error:
        raise RunRefused(
            "not_probed", f"the Run kept no tool probe in {directory}"
        ) from error
    except (OSError, ValueError) as error:
        raise RunRefused(
            "not_probed", f"the Run's tool probe in {directory} cannot be read: {error}"
        ) from error


def run_health(
    journey_file: Path,
    cache_dir: Path,
    output_dir: Path,
    *,
    pinned: ToolRunner,
    probed: ToolRunner | None = None,
    timeout_seconds: int | None = None,
) -> HealthManifest:
    """Run every whole-Run health tool over the Run's own BAM and VCF and MultiQC over
    their summaries; write and return the manifest.

    The Run's files come from `read_run`, its reference `.fai` from the lane's cache
    (`cache_dir`), and the tools' arguments from both (`derive_arguments`); a Run or
    cache that cannot supply them is refused before anything runs. Each run writes `<output_dir>/<entry>/`, MultiQC's
    renames `<output_dir>/multiqc-sample-names.tsv`, and the manifest is written last,
    as `<output_dir>/run-health-manifest.json`. The first refused run stops the rest;
    its refusal names the entry. Probed entries run through `probed` (default: the
    Run's own probe, `run_probe`), pinned ones through `pinned`; `timeout_seconds`
    bounds each process.
    """
    outcome, inputs = read_run(journey_file)
    cache = read_cache(cache_dir, outcome)
    arguments = derive_arguments(Path(cache.files["reference_fai"].path), inputs)
    probed = probed or ProbedProcessRunner(run_probe(outcome))
    output_dir = output_dir.absolute()
    try:
        output_dir.mkdir(parents=True)
    except FileExistsError:
        raise RunRefused(
            "output_dir_exists",
            f"{output_dir} exists; each health run writes a fresh directory",
        ) from None
    tools = dict(TOOLS) | {
        "goleft_indexcov": indexcov_entry(arguments.indexcov_excluded)
    }
    sample_names = output_dir / SAMPLE_NAMES_NAME
    sample_names.write_text(
        "".join(f"{name}\t{role}\n" for name, role in arguments.sample_names)
    )
    by_role = {item.role: Path(item.path) for item in inputs} | {
        "multiqc_sample_names": sample_names
    }
    files = []
    records = {}
    seconds = {}
    for entry, reads in _RUNS:
        started = time.monotonic()
        try:
            record = run_tool(
                entry,
                {name: by_role[role] for name, role in reads.items()},
                output_dir / entry,
                runner=probed if tools[entry].probed else pinned,
                tools=tools,
                timeout_seconds=timeout_seconds,
            )
        except RunRefused as refused:
            raise RunRefused(refused.code, f"{entry}: {refused.detail}") from None
        named, other = _ROLES[entry]
        for path, output in record.outputs.items():
            role = named.get(path, other)
            if role is None:
                raise RunRefused(
                    "output_unnamed",
                    f"{entry} wrote {path}, which has no health role",
                )
            files.append(
                HealthFile(
                    role=role,
                    path=f"{entry}/{path}",
                    byte_count=output.byte_count,
                    sha256=output.sha256,
                    record=entry,
                )
            )
            by_role[role] = output_dir / entry / path
        records[entry] = record
        seconds[entry] = round(time.monotonic() - started, 3)
    manifest = HealthManifest(
        journey_id=outcome.journey_id,
        journey_file=outcome.journey_file,
        run_root=outcome.run_root,
        execution_id=outcome.execution_id,
        outcome_event_id=outcome.outcome_event_id,
        inputs=inputs,
        files=tuple(files),
        records=records,
        arguments=arguments,
        covered=covered_regions(
            by_role["samtools_idxstats"],
            by_role["mosdepth_regions"],
            output_dir / COVERED_NAME,
        ),
        entry_seconds=seconds,
    )
    with (output_dir / MANIFEST_NAME).open("x") as written:
        written.write(manifest.model_dump_json(indent=2))
    return manifest
