"""Secret-free provider and worker observations with fail-closed ordering.

Provider state is deliberately independent of worker and tool evidence.  Raw
logs are reduced to a bounded digest; only strict worker markers become
semantic observations.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    Field,
    JsonValue,
    StrictInt,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from bio_neb.agentic_journey.failure_cause import cause_identity
from bio_neb.providers.job_observations import (
    PARTIAL_WINDOW_CEILING,
    BoundedProviderLogObserved,
    FetchedProviderEnvelope,
    ProviderCreateObserved,
    ProviderStateObserved,
    _line,
    _Observation,
    _timestamp,
    bounded_log_observation,
    log_window_failure_evidence,
)

_SHA256 = r"^[0-9a-f]{64}$"
# The authoritative vocabulary is JobStatus.State in nebius/ai/v1/job.proto.
# SUCCEEDED, QUEUED, PENDING, TIMED_OUT, TIMEOUT and CANCELED are deliberately absent
# from these sets: no such value exists in the enum.
_TERMINAL_SUCCESS = frozenset({"COMPLETED"})
_TERMINAL_FAILURE = frozenset({"FAILED", "CANCELLED", "ERROR"})
_TERMINAL_STATES = _TERMINAL_SUCCESS | _TERMINAL_FAILURE
# In-flight states have no order: the provider guarantees none, and a live job has
# been seen stepping IMAGE_PULLING back to an earlier stage. An in-flight state may
# recur and is recorded; only a state after a terminal, or a second terminal, fails.
_MARKER = re.compile(r"^\[([^\]]+)\]\s+(BIO_NEB_[A-Z0-9_]+)\s+(\{.*\})$")


class WorkerStartupObserved(_Observation):
    kind: Literal["worker_startup"] = "worker_startup"
    interpreter: StrictStr
    interpreter_resolved: StrictStr
    request_schema: Literal["bio-neb.parabricks-worker-request/7"]

    @field_validator("interpreter", "interpreter_resolved")
    @classmethod
    def absolute_runtime(cls, value: str) -> str:
        if not PurePosixPath(value).is_absolute():
            raise ValueError("worker runtime path must be absolute")
        return value


class WorkerRequestObserved(_Observation):
    kind: Literal["worker_request"] = "worker_request"
    byte_count: StrictInt = Field(gt=0)
    request_sha256: StrictStr = Field(pattern=_SHA256)
    worker_sha256: StrictStr = Field(pattern=_SHA256)


class WorkerToolIdentity(_Observation):
    name: StrictStr
    path: StrictStr
    version: StrictStr
    resolved_path: StrictStr | None = None

    @field_validator("path", "resolved_path")
    @classmethod
    def absolute_paths(cls, value: str | None) -> str | None:
        if value is not None and not PurePosixPath(value).is_absolute():
            raise ValueError("tool path must be absolute")
        return value


class WorkerToolsObserved(_Observation):
    kind: Literal["worker_tools"] = "worker_tools"
    tools: tuple[WorkerToolIdentity, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_tools(self) -> WorkerToolsObserved:
        if len({item.name for item in self.tools}) != len(self.tools):
            raise ValueError("worker tool identity repeated")
        return self


class WorkerInputVerifiedObserved(_Observation):
    kind: Literal["worker_input_verified"] = "worker_input_verified"
    role: StrictStr
    bucket: StrictStr
    object_key: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=_SHA256)


class WorkerSourceAcquiredObserved(_Observation):
    """The worker fetched the approved URL and its md5 and size matched."""

    kind: Literal["worker_source_acquired"] = "worker_source_acquired"
    role: Literal["reference_source"]
    url: StrictStr
    byte_count: StrictInt = Field(gt=0)
    md5: StrictStr = Field(pattern=r"^[0-9a-f]{32}$")
    sha256: StrictStr = Field(pattern=_SHA256)


class WorkerSourceMismatchObserved(_Observation):
    """The fetched bytes differed from the approved md5 or size; nothing was stored."""

    kind: Literal["worker_source_mismatch"] = "worker_source_mismatch"
    role: Literal["reference_source"]
    url: StrictStr
    expected_byte_count: StrictInt = Field(gt=0)
    observed_byte_count: StrictInt = Field(ge=0)
    expected_md5: StrictStr = Field(pattern=r"^[0-9a-f]{32}$")
    observed_md5: StrictStr = Field(pattern=r"^[0-9a-f]{32}$")


class PreparedReferenceObjectObserved(_Observation):
    role: StrictStr
    object_key: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=_SHA256)


class WorkerReferenceSupportObserved(_Observation):
    kind: Literal["worker_reference_support"] = "worker_reference_support"
    mode: Literal["derive", "verify"]
    objects: tuple[PreparedReferenceObjectObserved, ...] = Field(
        min_length=8, max_length=8
    )


class WorkerPreparedReferenceVerifiedObserved(_Observation):
    kind: Literal["worker_prepared_reference_verified"] = (
        "worker_prepared_reference_verified"
    )
    objects: tuple[PreparedReferenceObjectObserved, ...] = Field(
        min_length=8, max_length=8
    )


class ToolStarted(_Observation):
    kind: Literal["tool_started"] = "tool_started"
    command_id: StrictStr
    argv: tuple[StrictStr, ...] = Field(min_length=1)
    shell: Literal[False] = False

    @model_validator(mode="after")
    def absolute_tool(self) -> ToolStarted:
        if not PurePosixPath(self.argv[0]).is_absolute():
            raise ValueError("tool executable must be absolute")
        return self


class ToolExited(_Observation):
    kind: Literal["tool_exited"] = "tool_exited"
    command_id: StrictStr
    returncode: StrictInt


class UploadStarted(_Observation):
    kind: Literal["upload_started"] = "upload_started"
    role: StrictStr
    bucket: StrictStr
    object_key: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=_SHA256)
    multipart_chunk_bytes: StrictInt = Field(gt=0)


class UploadCompleted(_Observation):
    kind: Literal["upload_completed"] = "upload_completed"
    role: StrictStr
    bucket: StrictStr
    object_key: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=_SHA256)


class WorkerReceiptObject(_Observation):
    role: Literal["worker_receipt"]
    bucket: StrictStr
    object_key: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=_SHA256)


class WorkerReceiptObserved(_Observation):
    kind: Literal["worker_receipt"] = "worker_receipt"
    receipt: dict[StrictStr, JsonValue]
    receipt_object: WorkerReceiptObject

    @model_validator(mode="after")
    def non_authoritative_receipt(self) -> WorkerReceiptObserved:
        if self.receipt.get("authority") != "non-authoritative-worker-observation":
            raise ValueError("worker receipt changed its non-authoritative boundary")
        return self


class WorkerFailureObserved(_Observation):
    kind: Literal["worker_failure"] = "worker_failure"
    reason: StrictStr

    @field_validator("reason")
    @classmethod
    def controlled_reason(cls, value: str) -> str:
        return _line(value, "worker_failure_reason")


TruthfulExecutionObservation = Annotated[
    ProviderCreateObserved
    | FetchedProviderEnvelope
    | ProviderStateObserved
    | BoundedProviderLogObserved
    | WorkerStartupObserved
    | WorkerRequestObserved
    | WorkerToolsObserved
    | WorkerInputVerifiedObserved
    | WorkerSourceAcquiredObserved
    | WorkerSourceMismatchObserved
    | WorkerReferenceSupportObserved
    | WorkerPreparedReferenceVerifiedObserved
    | ToolStarted
    | ToolExited
    | UploadStarted
    | UploadCompleted
    | WorkerReceiptObserved
    | WorkerFailureObserved,
    Field(discriminator="kind"),
]
_OBSERVATION = TypeAdapter(TruthfulExecutionObservation)


# The failure evidence keeps at most this many characters of the window that the
# observation failed on, from its head -- the end an overlap check reads.

# How many consecutive marker-less windows shorter than the tail bound
# one observation tolerates before it fails closed.  Every dip the a6 run showed --
# ten on the CPU stage, one on the H100 stage (8 -> 101 -> 1 lines of a RUNNING
# job that later completed) -- recovered on the very next poll, so three gives two
# polls of headroom while a log that really vanished still fails within four.


def worker_marker_lines(output: str) -> tuple[str, ...]:
    """Retain exact timestamped worker marker lines from one bounded log window."""

    markers = []
    for line in output.splitlines():
        if _MARKER.fullmatch(line) is not None:
            markers.append(line)
        elif "BIO_NEB_" in line:
            raise ValueError("worker marker lacks provider timestamp grammar")
    return tuple(markers)


def _tool_identities(value: object) -> tuple[WorkerToolIdentity, ...]:
    if not isinstance(value, Mapping):
        raise TypeError("worker tools marker is invalid")
    identities = []
    for name, raw in sorted(value.items()):
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise TypeError("worker tool identity is invalid")
        identities.append(WorkerToolIdentity(name=name, **dict(raw)))
    return tuple(identities)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [pair[0] for pair in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("worker marker JSON contains duplicate keys")
    return dict(pairs)


def parse_worker_markers(output: str) -> tuple[TruthfulExecutionObservation, ...]:
    observations: list[TruthfulExecutionObservation] = []
    owners: dict[str, type[_Observation]] = {
        "BIO_NEB_PARABRICKS_WORKER_START": WorkerStartupObserved,
        "BIO_NEB_WORKER_REQUEST_ACCEPTED": WorkerRequestObserved,
        "BIO_NEB_WORKER_INPUT_VERIFIED": WorkerInputVerifiedObserved,
        "BIO_NEB_WORKER_SOURCE_ACQUIRED": WorkerSourceAcquiredObserved,
        "BIO_NEB_WORKER_SOURCE_MISMATCH": WorkerSourceMismatchObserved,
        "BIO_NEB_WORKER_REFERENCE_SUPPORT": WorkerReferenceSupportObserved,
        "BIO_NEB_WORKER_PREPARED_REFERENCE_VERIFIED": (
            WorkerPreparedReferenceVerifiedObserved
        ),
        "BIO_NEB_WORKER_TOOL_START": ToolStarted,
        "BIO_NEB_WORKER_TOOL_EXIT": ToolExited,
        "BIO_NEB_WORKER_UPLOAD_START": UploadStarted,
        "BIO_NEB_WORKER_UPLOAD_COMPLETE": UploadCompleted,
        "BIO_NEB_WORKER_RECEIPT": WorkerReceiptObserved,
        "BIO_NEB_PARABRICKS_WORKER_ERROR": WorkerFailureObserved,
    }
    for line in output.splitlines():
        match = _MARKER.search(line)
        if match is None:
            if "BIO_NEB_" in line:
                raise ValueError("worker marker lacks provider timestamp grammar")
            continue
        timestamp, marker, encoded = match.groups()
        _timestamp(timestamp, "provider_log_timestamp")
        if marker == "BIO_NEB_WORKER_TOOL_ERROR":
            continue
        owner = owners.get(marker)
        if owner is None and marker != "BIO_NEB_WORKER_TOOLS":
            continue
        try:
            payload = json.loads(encoded, object_pairs_hook=_unique_json_object)
        except json.JSONDecodeError as error:
            raise ValueError("worker marker JSON is invalid") from error
        if not isinstance(payload, dict):
            raise TypeError("worker marker payload is invalid")
        canonical = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        if canonical != encoded:
            raise ValueError("worker marker JSON is not canonical")
        if marker == "BIO_NEB_WORKER_TOOLS":
            observation = WorkerToolsObserved(
                tools=_tool_identities(payload.get("tools"))
            )
        else:
            assert owner is not None
            observation = owner.model_validate(payload)
        observations.append(_OBSERVATION.validate_python(observation))
    validate_truthful_observation_order(observations, terminal=False)
    return tuple(observations)


# Preserve the existing ordered observation contract during package organization.
def validate_truthful_observation_order(  # noqa: C901, PLR0915
    observations: Iterable[TruthfulExecutionObservation],
    *,
    terminal: bool,
    durable: bool = False,
) -> None:
    create = envelope = startup = request = tools = receipt = failure = False
    logs: list[BoundedProviderLogObserved] = []
    reference_support = prepared_reference = False
    source_acquired = source_mismatch = False
    verified_inputs: list[WorkerInputVerifiedObserved] = []
    states: list[str] = []
    state_observations: list[ProviderStateObserved] = []
    started_tools: dict[str, ToolStarted] = {}
    exited_tools: set[str] = set()
    started_uploads: dict[tuple[str, str, str], UploadStarted] = {}
    completed_uploads: set[tuple[str, str, str]] = set()
    completed_upload_observations: list[UploadCompleted] = []
    active_upload: tuple[str, str, str] | None = None
    receipt_upload_completed = False
    tool_observation: WorkerToolsObserved | None = None
    reference_support_observation: WorkerReferenceSupportObserved | None = None
    job_id: str | None = None
    for raw in observations:
        item = _OBSERVATION.validate_python(raw)
        if (
            receipt_upload_completed
            and not receipt
            and not isinstance(item, WorkerReceiptObserved)
        ):
            raise ValueError("worker receipt upload was not terminal")
        if isinstance(item, ProviderCreateObserved):
            if create or any((envelope, states, logs, startup)):
                raise ValueError("provider create observation repeated or out of order")
            create = True
            job_id = item.provider_job_id
        elif isinstance(item, FetchedProviderEnvelope):
            if not create or envelope or item.provider_job_id != job_id:
                raise ValueError("fetched provider envelope repeated or lacks create")
            envelope = True
        elif isinstance(item, ProviderStateObserved):
            if not envelope or startup or item.provider_job_id != job_id:
                raise ValueError("provider state lacks exact fetched envelope")
            state = item.provider_state.upper()
            if states and states[-1] in _TERMINAL_STATES:
                raise ValueError(
                    f"provider state followed a terminal: {states[-1]} -> {state}"
                )
            if states and state == states[-1]:
                raise ValueError(f"provider state repeated: {state} -> {state}")
            if state in _TERMINAL_STATES and any(
                prior in _TERMINAL_STATES and prior != state for prior in states
            ):
                raise ValueError("provider terminal states conflict")
            if item.finished_at is not None and state not in _TERMINAL_STATES:
                raise ValueError("nonterminal provider state acquired finish time")
            if state_observations:
                prior = state_observations[-1]
                if item.created_at != prior.created_at:
                    raise ValueError("provider created_at changed across states")
                if prior.started_at is not None and item.started_at != prior.started_at:
                    raise ValueError("provider started_at changed or disappeared")
                if (
                    prior.finished_at is not None
                    and item.finished_at != prior.finished_at
                ):
                    raise ValueError("provider finished_at changed or disappeared")
            states.append(state)
            state_observations.append(item)
        elif isinstance(item, BoundedProviderLogObserved):
            if not states or startup or item.provider_job_id != job_id:
                raise ValueError("provider log is premature, late, or changed jobs")
            logs.append(item)
        elif isinstance(item, WorkerStartupObserved):
            provider_trace_present = create or envelope or bool(states) or bool(logs)
            if (
                provider_trace_present
                and (
                    (not logs and not durable)
                    or not states
                    or states[-1] not in _TERMINAL_STATES
                )
            ) or startup:
                raise ValueError("worker startup is absent, repeated, or out of order")
            startup = True
        elif isinstance(item, WorkerRequestObserved):
            if not startup or request:
                raise ValueError("worker request is absent, repeated, or out of order")
            request = True
        elif isinstance(item, WorkerToolsObserved):
            if not request or tools:
                raise ValueError("worker tools are absent, repeated, or out of order")
            tools = True
            tool_observation = item
        elif isinstance(item, WorkerInputVerifiedObserved):
            identity = (item.role, item.bucket, item.object_key)
            if (
                not tools
                or started_tools
                or reference_support
                or any(
                    (prior.role, prior.bucket, prior.object_key) == identity
                    for prior in verified_inputs
                )
            ):
                raise ValueError("worker input verification repeated or out of order")
            verified_inputs.append(item)
        elif isinstance(
            item, (WorkerSourceAcquiredObserved, WorkerSourceMismatchObserved)
        ):
            if (
                not tools
                or started_tools
                or reference_support
                or verified_inputs
                or source_acquired
                or source_mismatch
            ):
                raise ValueError("worker source download repeated or out of order")
            source_acquired = isinstance(item, WorkerSourceAcquiredObserved)
            source_mismatch = not source_acquired
        elif isinstance(item, WorkerReferenceSupportObserved):
            if (
                not tools
                or not (verified_inputs or source_acquired)
                or reference_support
                or (item.mode == "verify" and started_tools)
            ):
                raise ValueError("reference support proof repeated or out of order")
            reference_support = True
            reference_support_observation = item
        elif isinstance(item, WorkerPreparedReferenceVerifiedObserved):
            if (
                not reference_support
                or prepared_reference
                or reference_support_observation is None
                or reference_support_observation.mode != "verify"
                or item.objects != reference_support_observation.objects
            ):
                raise ValueError("prepared-reference proof changed verified support")
            prepared_reference = True
        elif isinstance(item, ToolStarted):
            if not tools or receipt or item.command_id in started_tools:
                raise ValueError("tool startup repeated or lacks runtime identity")
            started_tools[item.command_id] = item
        elif isinstance(item, ToolExited):
            if (
                item.command_id not in started_tools
                or item.command_id in exited_tools
                or receipt
            ):
                raise ValueError("tool exit is missing its unique startup")
            exited_tools.add(item.command_id)
        elif isinstance(item, UploadStarted):
            key = (item.role, item.bucket, item.object_key)
            if (
                not tools
                or receipt
                or source_mismatch
                or active_upload is not None
                or key in started_uploads
                or (item.role == "reference_source" and not source_acquired)
            ):
                raise ValueError("upload startup repeated or lacks worker identity")
            started_uploads[key] = item
            active_upload = key
        elif isinstance(item, UploadCompleted):
            key = (item.role, item.bucket, item.object_key)
            started = started_uploads.get(key)
            if (
                started is None
                or key != active_upload
                or key in completed_uploads
                or receipt
                or (started.byte_count, started.sha256)
                != (item.byte_count, item.sha256)
            ):
                raise ValueError("upload completion changed or lacks its startup")
            completed_uploads.add(key)
            completed_upload_observations.append(item)
            active_upload = None
            if item.role == "worker_receipt":
                receipt_upload_completed = True
        elif isinstance(item, WorkerReceiptObserved):
            if (
                receipt
                or failure
                or not tools
                or not started_tools
                or not started_uploads
                or (not receipt_upload_completed and not durable)
                or active_upload is not None
                or exited_tools != set(started_tools)
                or completed_uploads != set(started_uploads)
            ):
                raise ValueError("worker receipt preceded exact tool/upload closure")
            receipt_object = item.receipt_object
            key = (
                receipt_object.role,
                receipt_object.bucket,
                receipt_object.object_key,
            )
            started = started_uploads.get(key)
            if not durable and (
                key not in completed_uploads
                or started is None
                or (started.byte_count, started.sha256)
                != (receipt_object.byte_count, receipt_object.sha256)
            ):
                raise ValueError("worker receipt object changed upload identity")
            if tool_observation is None:
                raise ValueError("worker receipt lacks tool identity")
            expected_tools = {
                tool.name: tool.model_dump(
                    mode="json", exclude={"name"}, exclude_none=True
                )
                for tool in tool_observation.tools
            }
            if item.receipt.get("tools") != expected_tools:
                raise ValueError("worker receipt changed tool identities")
            receipt_uploads = item.receipt.get("uploads")
            if receipt_uploads is None:
                receipt_uploads = item.receipt.get("prepared_uploads")
            # A downloaded source is the receipt's ``source``, not a product upload.
            expected_uploads = [
                upload.model_dump(mode="json", exclude={"kind"})
                for upload in completed_upload_observations
                if upload.role not in {"worker_receipt", "reference_source"}
            ]
            if receipt_uploads != expected_uploads:
                raise ValueError("worker receipt changed ordered uploads")
            if (
                not isinstance(item.receipt.get("request_id"), str)
                or not item.receipt["request_id"]
            ):
                raise ValueError("worker receipt omitted request identity")
            receipt = True
        elif isinstance(item, WorkerFailureObserved):
            if receipt or failure:
                raise ValueError("worker failure conflicts with terminal receipt")
            failure = True
    if terminal:
        if not (create and envelope and states and (logs or durable)):
            raise ValueError(
                "terminal observation lacks provider create/envelope/state/log facts"
            )
        if started_tools != {key: started_tools[key] for key in exited_tools}:
            raise ValueError("terminal observation omitted a tool exit")
        if set(started_uploads) != completed_uploads:
            raise ValueError("terminal observation omitted an upload completion")
        if states[-1] in {"COMPLETED", "SUCCEEDED"} and not (
            startup and request and tools and receipt
        ):
            raise ValueError("provider completion lacks complete worker facts")


def source_mismatch(output: object) -> WorkerSourceMismatchObserved | None:
    """The worker's typed source mismatch in a failed job's log window, if present."""
    if not isinstance(output, str):
        return None
    try:
        observations = parse_worker_markers(output)
    except (TypeError, ValueError):
        return None
    return next(
        (
            item
            for item in observations
            if isinstance(item, WorkerSourceMismatchObserved)
        ),
        None,
    )


def explicit_worker_exit(
    observations: Iterable[TruthfulExecutionObservation],
) -> int | None:
    """Return only an exit supported by explicit, consistently ordered markers."""

    items = tuple(observations)
    validate_truthful_observation_order(items, terminal=False)
    exits = [item.returncode for item in items if isinstance(item, ToolExited)]
    nonzero = next((code for code in exits if code != 0), None)
    if nonzero is not None:
        return nonzero
    receipt = any(isinstance(item, WorkerReceiptObserved) for item in items)
    starts = {item.command_id for item in items if isinstance(item, ToolStarted)}
    completed = {item.command_id for item in items if isinstance(item, ToolExited)}
    return 0 if receipt and starts and starts == completed else None


__all__ = [
    "BoundedProviderLogObserved",
    "FetchedProviderEnvelope",
    "ProviderCreateObserved",
    "ProviderStateObserved",
    "PreparedReferenceObjectObserved",
    "ToolExited",
    "ToolStarted",
    "TruthfulExecutionObservation",
    "UploadCompleted",
    "UploadStarted",
    "WorkerFailureObserved",
    "WorkerInputVerifiedObserved",
    "WorkerPreparedReferenceVerifiedObserved",
    "WorkerReferenceSupportObserved",
    "WorkerReceiptObject",
    "WorkerSourceAcquiredObserved",
    "WorkerSourceMismatchObserved",
    "WorkerReceiptObserved",
    "WorkerRequestObserved",
    "WorkerStartupObserved",
    "WorkerToolIdentity",
    "WorkerToolsObserved",
    "bounded_log_observation",
    "explicit_worker_exit",
    "parse_worker_markers",
    "source_mismatch",
    "validate_truthful_observation_order",
    "worker_marker_lines",
]


class WorkerMarkerWindowError(ValueError):
    def __init__(self, reason, evidence):
        super().__init__(reason)
        self.reason, self.evidence = reason, evidence


@dataclass
class WorkerMarkerWindow:
    """Keep worker marker continuity across bounded log windows, including partial reads."""

    markers: list[str] = field(default_factory=list)
    partial_windows: int = 0

    def failure(self, reason, output):
        return WorkerMarkerWindowError(
            reason,
            log_window_failure_evidence(
                output,
                retained_marker_count=len(self.markers),
                consecutive_partial_windows=self.partial_windows,
            ),
        )

    def partial_read(self, output):
        self.partial_windows += 1
        if self.partial_windows > PARTIAL_WINDOW_CEILING:
            raise self.failure("marker_window_partial_ceiling_exceeded", output)

    def merge(self, output, log):
        if output is None:
            return
        retained = self.markers
        try:
            current = list(worker_marker_lines(output))
        except Exception as error:
            raise self.failure(cause_identity(error), output) from error
        if not current:
            if not retained:
                return
            if log.limit_exhausted is None:
                raise self.failure("marker_window_aged_out", output)
            self.partial_read(output)
            return
        if not retained:
            retained.extend(current)
            self.partial_windows = 0
            return
        overlap = 0
        for size in range(min(len(retained), len(current)), 0, -1):
            if retained[-size:] == current[:size]:
                overlap = size
                break
        if overlap:
            retained.extend(current[overlap:])
            self.partial_windows = 0
            return
        identities: dict[str, str] = {}
        for line in (*retained, *current):
            key = line.split(" ", 2)[:2]
            identity = " ".join(key)
            prior = identities.setdefault(identity, line)
            if prior != line:
                raise self.failure("marker_window_conflict", output)
        if log.limit_exhausted is None:
            raise self.failure("marker_window_aged_out", output)
        self.partial_read(output)


@dataclass(frozen=True)
class JobObservation:
    """Declared process defaults, shared by the CPU and GPU recipes."""

    max_wait: float = 24 * 60 * 60
    log_lines: int = 1000


def observe_worker(ctx, jobs, tuning: JobObservation):
    """Keep optional diagnostics beside provider state, never as completion authority."""
    from bio_neb.failure import Invariant

    submission = ctx.receipts[-1]
    if len(submission.provider_ids) != 1:
        raise Invariant(
            "worker_submission_invalid", "submission needs one job identity"
        )
    job_id = submission.provider_ids[0]

    def sample():
        value = jobs.observe(job_id, window=tuning.log_lines)
        if not isinstance(value.values.get("state"), str):
            raise Invariant("worker_observation_invalid", "sample lacks provider state")
        output = value.values.get("log")
        diagnostic = log_window_failure_evidence(
            output if isinstance(output, str) else "",
            retained_marker_count=0,
            consecutive_partial_windows=0,
        )
        return value.model_copy(
            update={
                "values": {
                    **value.values,
                    "provider_job_id": job_id,
                    "log_window": diagnostic.model_dump(mode="json"),
                }
            }
        )

    terminal = ctx.observe(sample, max_wait=tuning.max_wait)
    return terminal.model_copy(
        update={
            "values": {
                **terminal.values,
                "create": submission.values.get("create"),
                "envelope": submission.values.get("envelope"),
            }
        }
    )


def durable_worker_observations(terminal, raw, bucket: str, key: str):
    """Join one existing receipt read with independent provider terminal evidence.

    Receipt upload identity comes from the transport receipt. It is not a fictional
    producer upload event (a receipt cannot contain its own final digest).
    """
    from bio_neb.content import canonical_json
    from bio_neb.failure import Invariant
    from bio_neb.parabricks.worker_exchange import RECEIPT_ADAPTER

    try:
        provider = (
            ProviderCreateObserved.model_validate(terminal.values["create"]),
            FetchedProviderEnvelope.model_validate(terminal.values["envelope"]),
            ProviderStateObserved.model_validate(terminal.values["details"]),
        )
    except (KeyError, ValueError) as error:
        raise Invariant(
            "worker_observation_invalid", "invalid internal provider evidence"
        ) from error
    if provider[-1].provider_state != terminal.values["state"]:
        raise Invariant(
            "worker_observation_invalid", "conflicting internal terminal state"
        )
    if (
        raw.content is None
        or type(raw.values.get("bytes")) is not int
        or raw.values.get("bytes") != len(raw.content)
        or not isinstance(raw.values.get("sha256"), str)
        or re.fullmatch(_SHA256, raw.values["sha256"]) is None
    ):
        raise Invariant(
            "worker_receipt_read_invalid",
            "receipt read lacks bytes or transport identity",
        )
    worker = json.loads(raw.content)
    RECEIPT_ADAPTER.validate_python(worker, strict=True)
    if raw.content != canonical_json(worker).encode():
        raise ValueError("worker receipt is not canonical")
    facts = tuple(_OBSERVATION.validate_python(fact) for fact in worker["validation"])
    allowed = (
        WorkerStartupObserved,
        WorkerRequestObserved,
        WorkerToolsObserved,
        WorkerInputVerifiedObserved,
        WorkerSourceAcquiredObserved,
        WorkerReferenceSupportObserved,
        WorkerPreparedReferenceVerifiedObserved,
        ToolStarted,
        ToolExited,
        UploadStarted,
        UploadCompleted,
    )
    if any(
        not isinstance(fact, allowed) or getattr(fact, "role", None) == "worker_receipt"
        for fact in facts
    ):
        raise ValueError("receipt contains non-producer validation facts")
    observed = WorkerReceiptObserved(
        receipt=worker,
        receipt_object=WorkerReceiptObject(
            role="worker_receipt",
            bucket=bucket,
            object_key=key,
            byte_count=raw.values["bytes"],
            sha256=raw.values["sha256"],
        ),
    )
    return (*provider, *facts, observed)


@dataclass(frozen=True, slots=True)
class ProviderTerminalContact:
    """A completed provider job with domain-interpreted worker marker continuity."""

    provider_job_id: str
    provider_state: Literal["COMPLETED"]
    state: ProviderStateObserved
    marker_lines: tuple[str, ...]


def observe_job(
    contact,
    created,
    *,
    heartbeat_interval_seconds,
    max_wait=None,
    progress=None,
    pause=None,
):
    """Diagnose a submitted worker through provider polling and shared observation.

    This is marker continuity proof for diagnostics; production completion additionally
    admits the durable worker receipt through ``observe_worker``. It grants no replay,
    cancellation or resource creation authority.
    """
    from bio_neb.observation import (
        AdapterHeartbeat,
        LiveEmitter,
        ObservationPolicy,
        observe,
    )
    from bio_neb.providers.nebius_serverless_job_contact import NebiusJobContactError

    markers = WorkerMarkerWindow()
    emit = LiveEmitter(progress)
    last_state = None
    first = True

    def sample():
        nonlocal first
        if first:
            first = False
            return created.initial_state
        return contact.poll_state(created.provider_job_id)[0]

    def close(recent):
        nonlocal last_state
        state = recent[-1]
        emit(state if state.provider_state != last_state else AdapterHeartbeat())
        last_state = state.provider_state
        if state.provider_state in _TERMINAL_FAILURE:
            raise NebiusJobContactError(
                "observe",
                "provider_failed",
                reason=state.provider_state.casefold(),
                provider_job_id=created.provider_job_id,
            )
        if state.provider_state in {"RUNNING", "COMPLETED"}:
            output = contact.read_log(created.provider_job_id, window=1000)
            log = bounded_log_observation(created.provider_job_id, output)
            emit(log)
            try:
                markers.merge(output, log)
            except WorkerMarkerWindowError as error:
                raise NebiusJobContactError(
                    "logs",
                    "log_invalid",
                    reason=error.reason,
                    provider_job_id=created.provider_job_id,
                    log_window=error.evidence,
                ) from error
        if state.provider_state == "COMPLETED":
            return ProviderTerminalContact(
                created.provider_job_id, "COMPLETED", state, tuple(markers.markers)
            )
        return None

    return observe(
        sample,
        ObservationPolicy(
            interval=heartbeat_interval_seconds,
            max_wait=JobObservation().max_wait if max_wait is None else max_wait,
            window=1,
            closure=close,
            **({"pause": pause} if pause is not None else {}),
        ),
    )
