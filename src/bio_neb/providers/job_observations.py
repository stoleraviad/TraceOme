"""Secret-free Nebius job envelopes, states and bounded log evidence."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

_SHA256 = r"^[0-9a-f]{64}$"
PROVIDER_LOG_WINDOW_TEXT_BOUND = 65536
PARTIAL_WINDOW_CEILING = 3


class _Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _line(value: str, label: str) -> str:
    if not value or value.strip() != value or any(char in value for char in "\r\n\x00"):
        raise ValueError(f"{label} is not canonical text")
    return value


def _timestamp(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    _line(value, label)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} lacks timezone")
    return value


class ProviderCreateObserved(_Observation):
    kind: Literal["provider_create"] = "provider_create"
    provider_job_id: StrictStr
    provider_job_name: StrictStr
    # What this field distinguishes is only whether the create call itself returned
    # a usable answer.  The name lookup is the identity path, not a recovery from a lost
    # create response: every job id comes through the lookup, and under --async the
    # create response is 33 bytes of prose that is deliberately never parsed.
    disposition: Literal["created", "created_with_ambiguous_create_response", "reused"]
    # Structured facts about the retained create response.  The bytes themselves are an
    # immutable content-addressed Journey artifact; only their identity belongs in the
    # event.
    response_byte_count: StrictInt = Field(default=0, ge=0)
    response_sha256: StrictStr = Field(
        default=hashlib.sha256(b"").hexdigest(), pattern=_SHA256
    )
    invocation_count: Literal[0, 1] = 1


class FetchedProviderEnvelope(_Observation):
    """Exact redacted provider envelope; no selector IDs or secret values."""

    kind: Literal["fetched_envelope"] = "fetched_envelope"
    provider_job_id: StrictStr
    provider_job_name: StrictStr
    project_id: StrictStr
    provider_image_reference: StrictStr
    platform: StrictStr
    preset: StrictStr
    disk_size_bytes: StrictInt = Field(ge=0)
    subnet_id: StrictStr
    timeout: StrictStr
    container_command_requested: StrictStr
    container_command_fetched: StrictStr | None
    normalized_arguments: tuple[StrictStr, ...]
    injected_paths: tuple[StrictStr, ...]
    environment_names: tuple[StrictStr, ...]
    version_bound_secret_environment_names: tuple[StrictStr, ...]
    shared_memory_bytes: StrictInt = Field(ge=0)
    restart_requested: Literal["never"] = "never"
    restart_fetched: Literal["never", "omitted"]

    @field_validator(
        "normalized_arguments",
        "injected_paths",
        "environment_names",
        "version_bound_secret_environment_names",
    )
    @classmethod
    def canonical_tuple(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        for item in value:
            _line(item, info.field_name)
        if info.field_name in {
            "injected_paths",
            "environment_names",
            "version_bound_secret_environment_names",
        } and len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} contains duplicates")
        if info.field_name == "injected_paths" and any(
            not PurePosixPath(item).is_absolute() for item in value
        ):
            raise ValueError("injected paths must be absolute")
        return value

    @model_validator(mode="after")
    def redacted_secret_names(self) -> FetchedProviderEnvelope:
        if not set(self.version_bound_secret_environment_names) <= set(
            self.environment_names
        ):
            raise ValueError(
                "version-bound secret names are absent from the environment"
            )
        return self


class ProviderStateObserved(_Observation):
    kind: Literal["provider_state"] = "provider_state"
    provider_job_id: StrictStr
    provider_state: StrictStr
    state_details: tuple[StrictStr, ...] = ()
    # status.instances[] uses a different enum -- JobInstanceStatus.State, with its own
    # ordinals and its own COMPLETING value -- so it is retained as opaque canonical
    # lines and never classified with the job-level vocabulary above.  Live jobs have
    # shown an instance FAILED under a job-level PROVISIONING, an instance RUNNING under
    # a terminal COMPLETED, and the whole key absent for the first 338 s and gone again
    # by the terminal envelope.  It has disagreed with the job in both directions and its
    # presence signals nothing, so it is diagnostic context only.  Never derive a job
    # outcome from it.
    instances: tuple[StrictStr, ...] = ()
    created_at: StrictStr
    started_at: StrictStr | None = None
    finished_at: StrictStr | None = None

    @model_validator(mode="after")
    def timestamp_order(self) -> ProviderStateObserved:
        values = [
            _timestamp(self.created_at, "created_at"),
            _timestamp(self.started_at, "started_at"),
            _timestamp(self.finished_at, "finished_at"),
        ]
        parsed = [
            datetime.fromisoformat(value) for value in values if value is not None
        ]
        if parsed != sorted(parsed):
            raise ValueError("provider timestamps changed lifecycle order")
        # Timestamps are recorded, never required.  A terminal FAILED can arrive as
        # {state, state_details} with neither started_at nor finished_at, when the job dies
        # before its container ever starts; a terminal COMPLETED carries both.  Both are
        # truthful terminal envelopes, so presence is recorded and never demanded.
        # Ordering above still holds on whatever timestamps do arrive.
        _line(self.provider_state, "provider_state")
        return self


class BoundedProviderLogObserved(_Observation):
    kind: Literal["provider_log"] = "provider_log"
    provider_job_id: StrictStr
    available: StrictBool
    byte_count: StrictInt = Field(ge=0)
    line_count: StrictInt = Field(ge=0, le=1000)
    sha256: StrictStr = Field(pattern=_SHA256)
    tail_line_limit: Literal[1000] = 1000
    limit_exhausted: Literal[False] | None
    unavailable_reason: Literal["provider_log_unavailable"] | None = None

    @model_validator(mode="after")
    def exact_availability(self) -> BoundedProviderLogObserved:
        if self.available == (self.unavailable_reason is not None):
            raise ValueError("provider log availability is contradictory")
        if not self.available and (
            self.byte_count
            or self.line_count
            or self.sha256 != hashlib.sha256(b"").hexdigest()
        ):
            raise ValueError("unavailable provider log retained content")
        return self


def bounded_log_observation(
    provider_job_id: str, output: str | None
) -> BoundedProviderLogObserved:
    if output is None:
        body = b""
        return BoundedProviderLogObserved(
            provider_job_id=provider_job_id,
            available=False,
            byte_count=0,
            line_count=0,
            sha256=hashlib.sha256(body).hexdigest(),
            limit_exhausted=False,
            unavailable_reason="provider_log_unavailable",
        )
    body = output.encode("utf-8")
    lines = output.splitlines()
    if len(lines) > 1000:
        raise ValueError("provider log exceeded the requested tail bound")
    return BoundedProviderLogObserved(
        provider_job_id=provider_job_id,
        available=True,
        byte_count=len(body),
        line_count=len(lines),
        sha256=hashlib.sha256(body).hexdigest(),
        limit_exhausted=None if len(lines) == 1000 else False,
    )


class ProviderLogWindowFailureEvidence(BaseModel):
    """The exact log window an observation failed on, bounded for the Journey.

    Issue 45: the a6 postmortem could read only a sha256 and two counts of the
    one-line window that failed a RUNNING job, so it could not tell a partial
    provider read from a vanished log.  This evidence names the window's shape,
    the markers already retained, how deep into a run of partial reads it was,
    and the window text itself, head first, up to a literal bound.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    line_count: StrictInt = Field(ge=0)
    byte_count: StrictInt = Field(ge=0)
    sha256: StrictStr = Field(pattern=_SHA256)
    retained_marker_count: StrictInt = Field(ge=0)
    consecutive_partial_windows: StrictInt = Field(ge=0)
    partial_window_ceiling: Literal[3] = PARTIAL_WINDOW_CEILING
    window_text: StrictStr = Field(max_length=PROVIDER_LOG_WINDOW_TEXT_BOUND)
    window_text_truncated: StrictBool


def log_window_failure_evidence(
    output: str, *, retained_marker_count: int, consecutive_partial_windows: int
) -> ProviderLogWindowFailureEvidence:
    body = output.encode("utf-8")
    return ProviderLogWindowFailureEvidence(
        line_count=len(output.splitlines()),
        byte_count=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        retained_marker_count=retained_marker_count,
        consecutive_partial_windows=consecutive_partial_windows,
        window_text=output[:PROVIDER_LOG_WINDOW_TEXT_BOUND],
        window_text_truncated=len(output) > PROVIDER_LOG_WINDOW_TEXT_BOUND,
    )
