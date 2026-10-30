"""Typed lifecycle facts and runtime ownership for observed activities."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from bio_neb.agentic_journey.invocation_observation import inspection_event
from bio_neb.content import ContentId
from bio_neb.journey import ArtifactEnrichmentRetained, Journey

ActivityAdapter = Literal[
    "pydantic_ai",
    "pydantic_ai_model",
    "local_files",
    "deterministic",
    "ncbi_datasets",
    "galaxy_datacache",
    "ncbi_investigation",
    "europe_pmc",
    "compiled_capability_snapshot",
    "nebius_cli",
]
ActivityStage = Literal[
    "planning_attempt",
    "investigation_planning_attempt",
    "investigation_table_reading_attempt",
    "report_explanation_attempt",
    "paper_reading_attempt",
    "model_contact",
    "inspect_input_metadata",
    "verify_sample_attestation",
    "resolve_taxon",
    "discover_reference_assemblies",
    "describe_reference",
    "request_companion_asset_selection",
    "request_companion_refinement",
    "describe_reference_companion",
    "search_prebuilt_indexes",
    "read_prebuilt_index",
    "suggest_assembly_with_index",
    "find_paper",
    "retrieve_paper",
    "list_compatible_analysis_recipes",
    "lookup_compatible_resource_profiles",
    "select_recipe_candidate",
    "select_resource_candidate",
    "request_recipe_selection",
    "request_resource_selection",
    "request_resource_recipe_dependency",
    "request_recipe_eligibility_assertion",
    "execute_approved_parabricks",
    "prepare_reference_cpu",
    "execute_prepared_reference_h100",
    "verify_execution_inputs",
]


def _canonical_uuid(value: str, field_name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{field_name} must be canonical")
    return value


class _ActivityFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    event_type: StrictStr
    activity_id: StrictStr
    adapter: ActivityAdapter
    stage: ActivityStage
    attempt: StrictInt = Field(gt=0)
    elapsed_ms: StrictInt = Field(ge=0)

    @field_validator("activity_id")
    @classmethod
    def canonical_identity(cls, value: str) -> str:
        return _canonical_uuid(value, "activity_id")

    @model_validator(mode="after")
    def canonical_responsibility(self) -> "_ActivityFact":
        expected = {
            "planning_attempt": "pydantic_ai",
            "investigation_planning_attempt": "pydantic_ai",
            "investigation_table_reading_attempt": "pydantic_ai",
            "report_explanation_attempt": "pydantic_ai",
            "paper_reading_attempt": "pydantic_ai",
            "model_contact": "pydantic_ai_model",
            "inspect_input_metadata": "local_files",
            "verify_sample_attestation": "deterministic",
            "resolve_taxon": "ncbi_datasets",
            "discover_reference_assemblies": "ncbi_datasets",
            "describe_reference": "ncbi_datasets",
            "request_companion_asset_selection": "deterministic",
            "request_companion_refinement": "deterministic",
            "describe_reference_companion": "ncbi_datasets",
            "search_prebuilt_indexes": "galaxy_datacache",
            "read_prebuilt_index": "galaxy_datacache",
            "suggest_assembly_with_index": "ncbi_investigation",
            "find_paper": "europe_pmc",
            "retrieve_paper": "europe_pmc",
            "list_compatible_analysis_recipes": "compiled_capability_snapshot",
            "lookup_compatible_resource_profiles": "compiled_capability_snapshot",
            "select_recipe_candidate": "deterministic",
            "select_resource_candidate": "deterministic",
            "request_recipe_selection": "deterministic",
            "request_resource_selection": "deterministic",
            "request_resource_recipe_dependency": "deterministic",
            "request_recipe_eligibility_assertion": "deterministic",
            "execute_approved_parabricks": "nebius_cli",
            "prepare_reference_cpu": "nebius_cli",
            "execute_prepared_reference_h100": "nebius_cli",
            "verify_execution_inputs": "nebius_cli",
        }
        if expected[self.stage] != self.adapter:
            raise ValueError(
                "activity adapter and stage do not identify one responsibility"
            )
        return self


class ActivityStarted(_ActivityFact):
    event_type: Literal["activity_started"] = "activity_started"
    elapsed_ms: Literal[0] = 0


class ActivityProgress(_ActivityFact):
    event_type: Literal["activity_progress"] = "activity_progress"
    current_task: StrictStr = Field(min_length=1, max_length=120)
    completed: StrictInt | None = Field(default=None, ge=0)
    total: StrictInt | None = Field(default=None, gt=0)

    @field_validator("current_task")
    @classmethod
    def canonical_task(cls, value: str) -> str:
        if value.strip() != value or any(character in value for character in "\r\n"):
            raise ValueError("current_task must be one short canonical line")
        return value

    @model_validator(mode="after")
    def truthful_counters(self) -> "ActivityProgress":
        if (self.completed is None) != (self.total is None):
            raise ValueError("progress counters must be supplied together")
        if self.completed is not None and self.completed > self.total:
            raise ValueError("completed progress cannot exceed total")
        return self


class ActivityHeartbeat(_ActivityFact):
    event_type: Literal["activity_heartbeat"] = "activity_heartbeat"


class ActivitySucceeded(_ActivityFact):
    event_type: Literal["activity_succeeded"] = "activity_succeeded"
    result_event_id: StrictStr

    @field_validator("result_event_id")
    @classmethod
    def canonical_result_reference(cls, value: str) -> str:
        return _canonical_uuid(value, "result_event_id")


class ActivityFailed(_ActivityFact):
    event_type: Literal["activity_failed"] = "activity_failed"
    failure_event_id: StrictStr

    @field_validator("failure_event_id")
    @classmethod
    def canonical_failure_reference(cls, value: str) -> str:
        return _canonical_uuid(value, "failure_event_id")


class ActivityCancelled(_ActivityFact):
    event_type: Literal["activity_cancelled"] = "activity_cancelled"
    cancellation_event_id: StrictStr

    @field_validator("cancellation_event_id")
    @classmethod
    def canonical_cancellation_reference(cls, value: str) -> str:
        return _canonical_uuid(value, "cancellation_event_id")


ActivityFact = (
    ActivityStarted
    | ActivityProgress
    | ActivityHeartbeat
    | ActivitySucceeded
    | ActivityFailed
    | ActivityCancelled
)

_FACT_TYPES = {
    fact.model_fields["event_type"].default: fact
    for fact in (
        ActivityStarted,
        ActivityProgress,
        ActivityHeartbeat,
        ActivitySucceeded,
        ActivityFailed,
        ActivityCancelled,
    )
}


def validate_activity_fact(event: Mapping[str, Any]) -> ActivityFact | None:
    """Validate one known lifecycle event; return ``None`` for another event family."""

    event_type = event.get("event_type")
    owner = _FACT_TYPES.get(event_type)
    if owner is None:
        if isinstance(event_type, str) and event_type.startswith("activity_"):
            raise ValueError("unknown activity lifecycle event")
        return None
    return owner.model_validate({"event_type": event_type, **event.get("payload", {})})


class ActivityLifecycle:
    """Own one causal activity stack over a Journey append head."""

    def __init__(
        self,
        journey: Journey,
        *,
        causal_head: str,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._journey = journey
        self._causal_head = causal_head
        self._clock = clock
        self._active: list[Activity] = []

    def start(
        self,
        *,
        adapter: ActivityAdapter,
        stage: ActivityStage,
    ) -> "Activity":
        if self._active and self._active[-1].closed:
            raise RuntimeError("closed activity cannot own a child")
        activity = Activity(
            self,
            activity_id=str(uuid.uuid4()),
            adapter=adapter,
            stage=stage,
            attempt=1,
            started_at=self._clock(),
        )
        self._active.append(activity)
        activity._record_lifecycle(
            ActivityStarted(
                activity_id=activity.activity_id,
                adapter=adapter,
                stage=stage,
                attempt=activity.attempt,
            )
        )
        return activity

    def _require_active(self, activity: "Activity") -> None:
        if activity.closed:
            raise RuntimeError("activity is already closed")
        if not self._active or self._active[-1] is not activity:
            raise RuntimeError("activity is not the active causal owner")

    def _record(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = self._journey.record(event_type, payload, [self._causal_head])
        self._causal_head = event["event_id"]
        inspection_event(
            "Journey observation", journey_id=str(self._journey.journey_id), event=event
        )
        return event

    def _close(self, activity: "Activity") -> None:
        self._require_active(activity)
        activity.closed = True
        self._active.pop()


class Activity:
    """Small caller interface for one currently observed activity."""

    def __init__(
        self,
        lifecycle: ActivityLifecycle,
        *,
        activity_id: str,
        adapter: ActivityAdapter,
        stage: ActivityStage,
        attempt: int,
        started_at: float,
    ):
        self._lifecycle = lifecycle
        self.activity_id = activity_id
        self.adapter = adapter
        self.stage = stage
        self.attempt = attempt
        self._started_at = started_at
        self._last_elapsed_ms = 0
        self._owned_event_ids: set[str] = set()
        self.closed = False

    def start_child(
        self,
        *,
        adapter: ActivityAdapter,
        stage: ActivityStage,
    ) -> "Activity":
        self._lifecycle._require_active(self)
        return self._lifecycle.start(
            adapter=adapter,
            stage=stage,
        )

    def require_active_child(self, child: "Activity") -> None:
        """Check a scoped handoff to this owner's current direct child."""
        if (
            child._lifecycle is not self._lifecycle
            or self.closed
            or len(self._lifecycle._active) < 2
            or self._lifecycle._active[-2] is not self
        ):
            raise RuntimeError("activity is not the active direct child")
        self._lifecycle._require_active(child)

    def record(self, fact: BaseModel) -> dict[str, Any]:
        """Append one typed non-lifecycle fact at the owned causal head."""

        self._lifecycle._require_active(self)
        event_type = getattr(fact, "event_type", None)
        if not isinstance(event_type, str) or event_type.startswith("activity_"):
            raise TypeError("activity.record requires a typed non-lifecycle fact")
        payload = fact.model_dump(mode="json")
        payload.pop("event_type")
        self._check_before_append(event_type, payload)
        event = self._lifecycle._record(event_type, payload)
        self._owned_event_ids.add(event["event_id"])
        return event

    def _check_before_append(self, event_type: str, payload: dict) -> None:
        """Run the fact's own reopen checks on the prefix plus it (#476 C8)."""
        from bio_neb.failure import RecordedFactRefused
        from bio_neb.journey_contracts import append_checks
        from bio_neb.recording import evidence_view

        checks = append_checks(event_type)
        if not checks:
            return
        journey = self._lifecycle._journey
        candidate = {
            "event_type": event_type,
            "journey_id": str(journey.journey_id),
            "payload": payload,
        }
        # The owner's shared read-only prefix; a check copies nothing (ADR 0032).
        prefix = evidence_view(journey)
        for check in checks:
            try:
                check.check(prefix, candidate, journey)
            except ValueError as error:
                raise RecordedFactRefused(
                    event_type, check.name, str(error), payload
                ) from error

    @property
    def journey(self):
        """The Journey this activity records into, for its owners' reads."""
        return self._lifecycle._journey

    def record_contact(self, fact: BaseModel) -> dict[str, Any]:
        """Retain mechanical evidence without changing the semantic causal head."""
        from bio_neb.recording import record_contact_evidence

        self._lifecycle._require_active(self)
        event = record_contact_evidence(self._lifecycle._journey, fact)
        self._owned_event_ids.add(event["event_id"])
        inspection_event(
            "Contact observation",
            journey_id=str(self._lifecycle._journey.journey_id),
            event=event,
        )
        return event

    def retained_artifact_sources(
        self, digests: Sequence[str], result_event
    ) -> tuple[str, ...]:
        """Bind only exact already-retained bytes named by this operation result."""
        from bio_neb.recording import read_evidence

        if not digests:
            # Nothing to bind: no history is read.
            return ()
        events = read_evidence(self._lifecycle._journey)
        sources = []
        for digest in digests:
            matches = {
                reference["source_event_id"]
                for event in events
                if event["event_id"] == result_event["payload"].get("source_event_id")
                and event["payload"].get("attempt_id")
                == result_event["payload"].get("attempt_id")
                for reference in event["artifact_references"]
                if reference["artifact_id"] == str(ContentId(digest))
            }
            if len(matches) != 1:
                raise ValueError(
                    "Tool result lacks its unique retained artifact source"
                )
            sources.append(matches.pop())
        return tuple(sources)

    def record_with_artifact(
        self, fact: BaseModel, artifact_bytes: bytes, *, media_type: str
    ) -> dict[str, Any]:
        """Append one owned typed fact and content-addressed output bytes."""

        self._lifecycle._require_active(self)
        event_type = getattr(fact, "event_type", None)
        if not isinstance(event_type, str) or event_type.startswith("activity_"):
            raise TypeError("activity.record_with_artifact requires a typed fact")
        payload = fact.model_dump(mode="json")
        payload.pop("event_type")
        self._check_before_append(event_type, payload)
        event = self._lifecycle._journey.record_with_artifact(
            event_type,
            payload,
            artifact_bytes,
            media_type,
            [self._lifecycle._causal_head],
        )
        self._lifecycle._causal_head = event["event_id"]
        self._owned_event_ids.add(event["event_id"])
        inspection_event(
            "Journey observation",
            journey_id=str(self._lifecycle._journey.journey_id),
            event=event,
        )
        return event

    def enrich_artifact(
        self,
        source_event: dict[str, Any],
        artifact_bytes: Callable[[], bytes],
        *,
        role: str,
        byte_count: int,
        sha256: str,
        media_type: str,
    ) -> dict[str, Any]:
        """Attach fallible local evidence after its semantic identity is durable."""

        self._lifecycle._require_active(self)
        source_event_id = source_event.get("event_id")
        if (
            not isinstance(source_event_id, str)
            or source_event_id not in self._owned_event_ids
            or self._lifecycle._causal_head != source_event_id
        ):
            raise ValueError("artifact enrichment requires the owned causal head")
        content = artifact_bytes()
        if not isinstance(content, bytes):
            raise TypeError("artifact enrichment must return bytes")
        if len(content) != byte_count or hashlib.sha256(content).hexdigest() != sha256:
            raise ValueError("artifact enrichment changed declared byte identity")
        return self.record_with_artifact(
            ArtifactEnrichmentRetained(
                source_event_id=source_event_id,
                role=role,
                byte_count=byte_count,
                sha256=sha256,
            ),
            content,
            media_type=media_type,
        )

    def succeeded(self, result_event: Mapping[str, Any]) -> dict[str, Any]:
        result_event_id = self._exact_owned_reference(result_event, "result")
        terminal = self._record_lifecycle(
            ActivitySucceeded(
                **self._identity(),
                elapsed_ms=self._elapsed_ms(),
                result_event_id=result_event_id,
            )
        )
        self._lifecycle._close(self)
        return terminal

    def failed(self, failure_event: Mapping[str, Any]) -> dict[str, Any]:
        failure_event_id = self._exact_owned_reference(failure_event, "failure")
        terminal = self._record_lifecycle(
            ActivityFailed(
                **self._identity(),
                elapsed_ms=self._elapsed_ms(),
                failure_event_id=failure_event_id,
            )
        )
        self._lifecycle._close(self)
        return terminal

    def cancelled(self, cancellation_event: Mapping[str, Any]) -> dict[str, Any]:
        cancellation_event_id = self._exact_owned_reference(
            cancellation_event, "cancellation"
        )
        terminal = self._record_lifecycle(
            ActivityCancelled(
                **self._identity(),
                elapsed_ms=self._elapsed_ms(),
                cancellation_event_id=cancellation_event_id,
            )
        )
        self._lifecycle._close(self)
        return terminal

    def _identity(self) -> dict[str, Any]:
        return {
            "activity_id": self.activity_id,
            "adapter": self.adapter,
            "stage": self.stage,
            "attempt": self.attempt,
        }

    def _elapsed_ms(self) -> int:
        measured = max(0, int((self._lifecycle._clock() - self._started_at) * 1000))
        self._last_elapsed_ms = max(self._last_elapsed_ms, measured)
        return self._last_elapsed_ms

    def _exact_owned_reference(
        self, event: Mapping[str, Any], reference_name: str
    ) -> str:
        event_id = event.get("event_id")
        if event_id not in self._owned_event_ids:
            raise ValueError(
                f"activity {reference_name} must reference evidence recorded by the activity"
            )
        return event_id

    def _record_lifecycle(self, fact: ActivityFact) -> dict[str, Any]:
        self._lifecycle._require_active(self)
        payload = fact.model_dump(mode="json")
        payload.pop("event_type")
        return self._lifecycle._record(fact.event_type, payload)
