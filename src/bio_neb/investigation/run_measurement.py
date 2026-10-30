"""The Investigation's first step: the Run measures its own health and each Work
item's read depth (#414).

Once the Run's BAM and VCF are local (the grant's preparation downloaded them),
before any item's window or reading, the coordinator records the Run's health
once and one coverage fact per started Work item of the collection. It runs the
Run's own probed samtools (``bio-neb-samtools``) over the local BAM, and the VCF's
Ti/Tv comes from the product VCF tool (ADR 0047); no reference FASTA, cache or
other install is read. Each process ends within the stage's own
``failure.StageLimits`` declaration (``STAGE_LIMITS``).

A measuring process that times out, cannot start or exits nonzero is a typed
``MeasurementFault`` (ADR 0031); any other error is ours and holds the Run. A
health measurement's fault asks its Review (``run_health``): ``retry`` measures
again, ``abort`` records it not measured. Item depths are read in one batch; when
the batch fails, each placed item is read alone, and an item whose own read
fails records ``measurement_failed``. Nothing here changes a decision.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

from bio_neb.failure import FailureSurface, limits_for
from bio_neb.investigation.item_evidence import (
    CoverageMeasurement,
    InvestigationItemCoverageRecorded,
    coverage_span,
    validate_item_evidence_fact,
)
from bio_neb.investigation.observation_defaults import CoverageDefaults
from bio_neb.investigation.run_health import (
    STAGE,
    HealthNotMeasured,
    HealthNumbers,
    MeasurementFaultCode,
    RunHealthMeasured,
    ask_health_review,
    project_run_health,
)
from bio_neb.investigation.vcf_observation import (
    DeterministicVcfObservationTool,
    PreparedVcfObservationReceipt,
    VcfProcessAdapter,
)
from bio_neb.journey import Journey

MANIFEST_SCHEMA = "bio-neb.run-health-manifest/3"
MANIFEST_NAME = "run-health-manifest.json"
_INDEX_NAME = "aligned.bam.csi"
_HEALTH = "health"
_NO_BAM = (
    "the Run holds no local BAM: no Execution made one, or its download was aborted"
)

# SAM flag bits by their samtools names, for `CoverageDefaults.excluded_flags`;
# lane H's development reads use the same (`tools/run_health/table.py`).
SAM_FLAGS = {
    "PAIRED": 0x1,
    "PROPER_PAIR": 0x2,
    "UNMAP": 0x4,
    "MUNMAP": 0x8,
    "REVERSE": 0x10,
    "MREVERSE": 0x20,
    "READ1": 0x40,
    "READ2": 0x80,
    "SECONDARY": 0x100,
    "QCFAIL": 0x200,
    "DUP": 0x400,
    "SUPPLEMENTARY": 0x800,
}
# The flags samtools depth leaves out unless told otherwise, cleared by `-g`.
_DEPTH_DEFAULT_EXCLUDED = ("UNMAP", "SECONDARY", "QCFAIL", "DUP")


@dataclass(frozen=True)
class MeasuredFiles:
    """The Run's local files and tools the measurement reads."""

    bam: Path
    bam_sha256: str
    vcf_sha256: str
    samtools: str
    preparation: PreparedVcfObservationReceipt | None
    directory: Path


class MeasurementFault(Exception):
    """A measuring process that timed out, could not start or exited nonzero: an
    operational fault the stage routes. Every other error is ours (ADR 0031)."""

    def __init__(self, code: MeasurementFaultCode, detail: str) -> None:
        assert code in get_args(MeasurementFaultCode)
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ItemSpan:
    work_item_id: str
    contig: str
    start: int
    end: int


def _process_seconds() -> int:
    return limits_for(
        FailureSurface(scope="stage", name=STAGE)
    ).effective.process_seconds


def _process_fault(call, label: str):
    """Run ``call``; its process's timeout, start failure or nonzero exit is a
    ``MeasurementFault``, anything else propagates."""
    try:
        return call()
    except subprocess.TimeoutExpired as error:
        raise MeasurementFault(
            "timeout", f"{label} ran past {error.timeout} s"
        ) from error
    except subprocess.CalledProcessError as error:
        raise MeasurementFault(
            "process_failed", f"{label} exited {error.returncode}"
        ) from error
    except OSError as error:
        raise MeasurementFault("os_error", f"{label}: {error}") from error


class _Run:
    """One stage's processes under its own time limit (``StageLimits``)."""

    def __init__(self, adapter: VcfProcessAdapter) -> None:
        self.adapter = adapter
        self.seconds = _process_seconds()

    def __call__(self, *argv: str) -> bytes:
        label = f"{Path(argv[0]).name} {argv[1]}"
        process = _process_fault(
            lambda: self.adapter.run(tuple(argv), timeout_seconds=self.seconds), label
        )
        if process.return_code != 0:
            raise MeasurementFault(
                "process_failed",
                f"{label} exited {process.return_code}: "
                + process.stderr.decode("utf-8", errors="replace").strip()[-300:],
            )
        return process.stdout


def _excluded(defaults: CoverageDefaults) -> tuple[str, ...]:
    unsupported = []
    unknown = [flag for flag in defaults.excluded_flags if flag not in SAM_FLAGS]
    if unknown:
        unsupported.append(f"excluded_flags {unknown}")
    if defaults.applies_baq:
        unsupported.append("applies_baq=True")
    if unsupported:
        raise ValueError(f"samtools depth cannot apply {', '.join(unsupported)}")
    return tuple(sorted(set(defaults.excluded_flags), key=SAM_FLAGS.__getitem__))


def depth_read_filters(defaults: CoverageDefaults) -> tuple[str, ...]:
    """samtools depth's read filters for exactly `CoverageDefaults`: `-g` clears
    samtools' own excluded flags and `-G` sets the product's. samtools depth
    applies no BAQ, so `applies_baq=True` is refused."""
    excluded = _excluded(defaults)
    return (
        *(("-s",) if defaults.overlapping_pairs == "count_once" else ()),
        *(("-J",) if defaults.deletion_spanning_reads == "included" else ()),
        "-q",
        str(defaults.minimum_base_quality),
        "-Q",
        str(defaults.minimum_mapping_quality),
        "-g",
        ",".join(_DEPTH_DEFAULT_EXCLUDED),
        *(("-G", ",".join(excluded)) if excluded else ()),
    )


def _depth_argv(samtools: str, defaults: CoverageDefaults) -> tuple[str, ...]:
    return (samtools, "depth", *depth_read_filters(defaults))


def _fresh(directory: Path, name: str) -> Path:
    path = directory / f"{name}-{uuid.uuid4().hex[:12]}"
    path.mkdir(mode=0o700, parents=True)
    return path


def _write(path: Path, data: bytes) -> tuple[int, str]:
    with path.open("xb") as written:
        written.write(data)
    return len(data), hashlib.sha256(data).hexdigest()


def flagstat_numbers(text: str) -> tuple[int, float]:
    """Reads (QC-passed) and % mapped, as `samtools flagstat` states them."""
    reads = mapped = None
    for line in text.splitlines():
        if line.endswith("in total (QC-passed reads + QC-failed reads)"):
            reads = int(line.split(" + ")[0])
        elif " mapped (" in line and "primary" not in line:
            stated = line.split(" mapped (")[1].split("%")[0]
            mapped = 0.0 if stated.startswith("N/A") else float(stated)
    if reads is None or mapped is None:
        raise ValueError("samtools flagstat stated no reads or mapped share")
    return reads, mapped


def coverage_rows(text: str) -> tuple[dict[str, Any], ...]:
    """Each contig's length, covered bases and mean depth from `samtools coverage`."""
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, start, end, _reads, covered, _share, mean, *_ = line.split("\t")
        rows.append(
            {
                "contig": name,
                "length": int(end) - int(start) + 1,
                "covered_bases": int(covered),
                "mean_depth": float(mean),
            }
        )
    if not rows or not sum(row["length"] for row in rows):
        raise ValueError("samtools coverage stated no contig")
    return tuple(rows)


def measure_health(
    files: MeasuredFiles,
    adapter: VcfProcessAdapter,
    defaults: CoverageDefaults | None = None,
) -> HealthNumbers:
    """The Run's key numbers, with every tool output and the manifest written to
    a fresh directory; the manifest's digest is the fact's."""
    defaults = defaults or CoverageDefaults()
    run = _Run(adapter)
    output = _fresh(files.directory, "run")
    flagstat_argv = (files.samtools, "flagstat", str(files.bam))
    flagstat = run(*flagstat_argv)
    coverage_argv = (
        files.samtools,
        "coverage",
        "-q",
        str(defaults.minimum_mapping_quality),
        "-Q",
        str(defaults.minimum_base_quality),
        *(("--ff", ",".join(excluded)) if (excluded := _excluded(defaults)) else ()),
        str(files.bam),
    )
    coverage = run(*coverage_argv)
    reads, mapped = flagstat_numbers(flagstat.decode("utf-8"))
    rows = coverage_rows(coverage.decode("utf-8"))
    ti_tv = (
        None
        if files.preparation is None
        else _process_fault(
            lambda: DeterministicVcfObservationTool().transition_transversion(
                files.preparation, adapter, timeout_seconds=run.seconds
            ),
            "bcftools stats",
        )
    )
    length = sum(row["length"] for row in rows)
    covered = sum(row["covered_bases"] for row in rows)
    mean = sum(row["mean_depth"] * row["length"] for row in rows) / length
    outputs = {}
    for name, data in (("flagstat.txt", flagstat), ("coverage.tsv", coverage)):
        byte_count, sha256 = _write(output / name, data)
        outputs[name] = {"byte_count": byte_count, "sha256": sha256}
    numbers = {
        "reads": reads,
        "mapped_percent": mapped,
        "mean_depth": mean,
        "ti_tv": ti_tv,
        "ti_tv_reason": None
        if ti_tv is not None
        else "vcf_unprepared"
        if files.preparation is None
        else "no_transversion",
        "covered_contigs": sum(row["covered_bases"] > 0 for row in rows),
        "covered_bases": covered,
        "covered_fraction": covered / length,
    }
    manifest = json.dumps(
        {
            "schema_version": MANIFEST_SCHEMA,
            "bam": {"path": str(files.bam), "sha256": files.bam_sha256},
            "vcf_sha256": files.vcf_sha256,
            "parameters": defaults.model_dump(mode="json"),
            "argv": {"flagstat": flagstat_argv, "coverage": coverage_argv},
            "outputs": outputs,
            "contigs": rows,
            "numbers": numbers,
        },
        indent=2,
        sort_keys=True,
    ).encode()
    byte_count, sha256 = _write(output / MANIFEST_NAME, manifest)
    return HealthNumbers(
        manifest_sha256=sha256, manifest_byte_count=byte_count, **numbers
    )


def bam_index(files: MeasuredFiles, adapter: VcfProcessAdapter) -> Path:
    """The BAM's one CSI index beside the measurements, written once whole."""
    index = files.directory / _INDEX_NAME
    if index.exists():
        return index
    staged = files.directory / f".{_INDEX_NAME}.{uuid.uuid4().hex[:12]}"
    _Run(adapter)(files.samtools, "index", "-c", "-o", str(staged), str(files.bam))
    os.replace(staged, index)
    return index


def contig_lengths(
    files: MeasuredFiles, index: Path, adapter: VcfProcessAdapter
) -> dict[str, int]:
    """Each BAM header sequence's length, by its exact name (`samtools idxstats`)."""
    stdout = _Run(adapter)(files.samtools, "idxstats", "-X", str(files.bam), str(index))
    lengths = {}
    for line in stdout.decode("utf-8").splitlines():
        name, length, *_ = line.split("\t")
        if name != "*":
            lengths[name] = int(length)
    return lengths


def _summaries(spans: Sequence[ItemSpan], lines) -> dict[str, tuple[int, float, int]]:
    """Each span's least, mean and greatest depth from `samtools depth` lines
    (contig, position, depth); a position no line names has depth zero."""
    by_contig: dict[str, list[ItemSpan]] = {}
    for span in spans:
        by_contig.setdefault(span.contig, []).append(span)
    starts = {
        contig: sorted(items, key=lambda span: span.start)
        for contig, items in by_contig.items()
    }
    keys = {contig: [span.start for span in items] for contig, items in starts.items()}
    sums = {span.work_item_id: 0 for span in spans}
    seen = dict(sums)
    least = {span.work_item_id: None for span in spans}
    most = dict(sums)
    for line in lines:
        contig, position, depth = line.split("\t")
        items = starts.get(contig)
        if items is None:
            continue
        position, depth = int(position), int(depth)
        for span in items[: bisect.bisect_right(keys[contig], position)]:
            if span.end < position:
                continue
            item = span.work_item_id
            sums[item] += depth
            seen[item] += 1
            most[item] = max(most[item], depth)
            least[item] = depth if least[item] is None else min(least[item], depth)
    result = {}
    for span in spans:
        item = span.work_item_id
        length = span.end - span.start + 1
        lowest = least[item] if seen[item] == length else 0
        result[item] = (lowest, sums[item] / length, most[item])
    return result


def measure_items(
    files: MeasuredFiles,
    spans: Sequence[ItemSpan],
    adapter: VcfProcessAdapter,
    defaults: CoverageDefaults | None = None,
) -> dict[str, dict[str, Any]]:
    """Each item's coverage outcome: its measurement and flags, or why none.

    Every placeable span is read in one batch; when the batch fails, each is read
    alone, and an item whose own read fails records ``measurement_failed``.
    """
    defaults = defaults or CoverageDefaults()
    run = _Run(adapter)
    try:
        index = bam_index(files, adapter)
        lengths = contig_lengths(files, index, adapter)
    except MeasurementFault as fault:
        detail = str(fault)
        return {
            span.work_item_id: _not_measured("measurement_failed", detail)
            for span in spans
        }
    outcomes: dict[str, dict[str, Any]] = {}
    placed = []
    for span in spans:
        if span.contig not in lengths:
            outcomes[span.work_item_id] = _not_measured(
                "contig_unknown", f"{span.contig} is no sequence of the Run's BAM"
            )
        elif span.end > lengths[span.contig]:
            outcomes[span.work_item_id] = _not_measured(
                "span_beyond_contig",
                f"{span.contig}:{span.start}-{span.end} ends past its "
                f"{lengths[span.contig]} bases",
            )
        else:
            placed.append(span)
    if not placed:
        return outcomes
    output = _fresh(files.directory, "items")
    argv = _depth_argv(files.samtools, defaults)
    bed = output / "spans.bed"
    _write(
        bed,
        "".join(f"{s.contig}\t{s.start - 1}\t{s.end}\n" for s in placed).encode(),
    )
    try:
        stdout = run(*argv, "-X", "-b", str(bed), str(files.bam), str(index))
        summaries = _summaries(placed, stdout.decode("utf-8").splitlines())
    except MeasurementFault:
        # The batch failed: each placed item is read alone.
        summaries = {}
    for span in placed:
        summary = summaries.get(span.work_item_id)
        if summary is None:
            region = f"{span.contig}:{span.start}-{span.end}"
            try:
                stdout = run(
                    *argv, "-a", "-X", "-r", region, str(files.bam), str(index)
                )
                summary = _summaries((span,), stdout.decode("utf-8").splitlines())[
                    span.work_item_id
                ]
            except MeasurementFault as fault:
                outcomes[span.work_item_id] = _not_measured(
                    "measurement_failed", str(fault)
                )
                continue
        least, mean, most = summary
        outcomes[span.work_item_id] = {
            "measurement": CoverageMeasurement(
                parameters=defaults,
                depth=least,
                mean_depth=mean,
                meets_minimum_depth=least >= defaults.minimum_depth,
            ),
            "batch_flags": ("low_depth",)
            if least < defaults.minimum_depth and most > 0
            else (),
        }
    return outcomes


def _not_measured(code: str, detail: str) -> dict[str, Any]:
    return {"not_measured": {"code": code, "detail": detail}}


def measured_files(journey: Journey, preparation) -> MeasuredFiles | None:
    """The BAM the grant's preparation published and the Run's probed samtools
    (``observation_inputs.published_observation_inputs``); None when the Run
    holds no local BAM: no Execution made one, or its download was aborted."""
    from bio_neb.investigation.observation_inputs import published_observation_inputs
    from bio_neb.parabricks.result import (
        ParabricksResultUnavailable,
        read_completed_parabricks_result,
    )

    try:
        verified = {
            item.role: item
            for item in read_completed_parabricks_result(
                str(journey.path)
            ).verified_objects
        }
    except ParabricksResultUnavailable:
        return None
    published = published_observation_inputs(journey)
    if published is None:
        return None
    directory = published.root / _HEALTH
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return MeasuredFiles(
        bam=published.bam,
        bam_sha256=verified["aligned_bam"].sha256,
        vcf_sha256=verified["VCF"].sha256,
        samtools=published.samtools,
        preparation=preparation,
        directory=directory,
    )


def _record(journey: Journey, fact) -> None:
    journey.record(
        fact.event_type,
        fact.model_dump(mode="json", exclude={"event_type"}),
        [journey.head_event_id],
    )


def advance_run_measurement(
    journey: Journey,
    stage,
    events: Sequence[Mapping[str, Any]],
    preparation: PreparedVcfObservationReceipt | None,
    adapter: VcfProcessAdapter,
) -> bool | None:
    """One step of the measurement: True when it wrote, False while its Review
    is pending, None once the Run's health and every item's coverage are in."""
    from bio_neb.investigation.investigation_observation import (
        validate_investigation_work_item_started_fact,
    )

    state = project_run_health(events)
    if state.pending:
        return False
    targets = {
        fact.work_item_id: fact.target
        for event in events
        if (fact := validate_investigation_work_item_started_fact(event)) is not None
    }
    covered = {
        fact.work_item_id
        for event in events
        if isinstance(
            fact := validate_item_evidence_fact(event),
            InvestigationItemCoverageRecorded,
        )
        and fact.collection_event_id == stage.collection_event_id
    }
    spans = [
        ItemSpan(item.work_item_id, targets[item.work_item_id].contig, *span)
        for item in stage.collection.item_states
        if item.work_item_id in targets and item.work_item_id not in covered
        for span in (coverage_span(targets[item.work_item_id]),)
    ]
    if state.health is not None and not spans:
        return None
    files = measured_files(journey, preparation)
    if state.health is None:
        _measure_health(journey, stage, files, state, adapter)
        return True
    outcomes = (
        {span.work_item_id: _not_measured("bam_unavailable", _NO_BAM) for span in spans}
        if files is None
        else measure_items(files, spans, adapter)
    )
    for span in spans:
        _record(
            journey,
            InvestigationItemCoverageRecorded.model_validate(
                {
                    "authority_id": stage.collection.authority.authority_id,
                    "collection_event_id": stage.collection_event_id,
                    "work_item_id": span.work_item_id,
                    "contig": span.contig,
                    "span_start": span.start,
                    "span_end": span.end,
                    **outcomes[span.work_item_id],
                }
            ),
        )
    return True


def _measure_health(
    journey, stage, files: MeasuredFiles | None, state, adapter
) -> None:
    if files is None:
        fact = RunHealthMeasured(
            collection_event_id=stage.collection_event_id,
            not_measured=HealthNotMeasured(code="bam_unavailable", detail=_NO_BAM),
        )
        _record(journey, fact)
        return
    identity = {
        "collection_event_id": stage.collection_event_id,
        "bam_sha256": files.bam_sha256,
        "vcf_sha256": files.vcf_sha256,
    }
    if state.latest_action == "abort":
        failure = state.last_failure
        fact = RunHealthMeasured(
            **identity,
            not_measured=HealthNotMeasured(
                code="measurement_failed", detail=f"{failure.code}: {failure.detail}"
            ),
        )
    else:
        try:
            numbers = measure_health(files, adapter)
        except MeasurementFault as fault:
            ask_health_review(
                journey, stage.collection_event_id, fault.code, fault.detail
            )
            return
        fact = RunHealthMeasured(**identity, measurement=numbers)
    _record(journey, fact)


__all__ = [
    "MANIFEST_NAME",
    "SAM_FLAGS",
    "ItemSpan",
    "MeasurementFault",
    "MeasuredFiles",
    "advance_run_measurement",
    "bam_index",
    "contig_lengths",
    "coverage_rows",
    "depth_read_filters",
    "flagstat_numbers",
    "measure_health",
    "measure_items",
    "measured_files",
]
