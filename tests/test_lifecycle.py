import hashlib
import tempfile
import unittest
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from bio_neb.agentic_journey.lifecycle import (
    ActivityFailed,
    ActivityHeartbeat,
    ActivityLifecycle,
    ActivityProgress,
    ActivityStarted,
    ActivitySucceeded,
    validate_activity_fact,
)
from bio_neb.journey import Journey


class _ObservedResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    event_type: StrictStr = "ObservedResult"
    value: StrictStr


def identity():
    return {
        "activity_id": str(uuid.uuid4()),
        "adapter": "pydantic_ai_model",
        "stage": "model_contact",
        "attempt": 1,
    }


class _Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class LifecycleFactTests(unittest.TestCase):
    def test_strict_frozen_fact_shapes_reject_noncanonical_values_and_extras(self):
        valid = identity()
        started = ActivityStarted(**valid)
        with self.assertRaises(ValidationError):
            started.attempt = 2

        invalid_values = (
            (ActivityStarted, {**valid, "activity_id": "not-a-uuid"}),
            (ActivityStarted, {**valid, "attempt": 0}),
            (ActivityStarted, {**valid, "attempt": True}),
            (ActivityStarted, {**valid, "elapsed_ms": 1}),
            (ActivityStarted, {**valid, "adapter": "local_files"}),
            (ActivityStarted, {**valid, "unexpected": "value"}),
            (ActivityHeartbeat, {**valid, "elapsed_ms": -1}),
            (
                ActivityHeartbeat,
                {**valid, "elapsed_ms": 1, "provider_state": "waiting"},
            ),
            (
                ActivityHeartbeat,
                {**valid, "elapsed_ms": 1, "current_task": "not liveness"},
            ),
            (
                ActivitySucceeded,
                {**valid, "elapsed_ms": 1, "result_event_id": "not-a-uuid"},
            ),
            (
                ActivityFailed,
                {**valid, "elapsed_ms": 1, "failure_event_id": "not-a-uuid"},
            ),
        )
        for owner, values in invalid_values:
            with self.subTest(owner=owner.__name__, values=values):
                with self.assertRaises(ValidationError):
                    owner(**values)

    def test_progress_requires_a_short_task_and_truthful_paired_counters(self):
        valid = {**identity(), "elapsed_ms": 1, "current_task": "inspect inputs"}
        ActivityProgress(**valid)
        ActivityProgress(**valid, completed=0, total=2)

        invalid_values = (
            {**valid, "current_task": ""},
            {**valid, "current_task": " inspect inputs"},
            {**valid, "current_task": "inspect\ninputs"},
            {**valid, "completed": 1},
            {**valid, "total": 2},
            {**valid, "completed": 0, "total": 0},
            {**valid, "completed": 3, "total": 2},
            {**valid, "completed": True, "total": 2},
        )
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                ActivityProgress(**values)

    def test_activity_facts_chain_linearly_and_elapsed_time_never_goes_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            seed = journey.record("Seed", {})
            clock = _Clock()
            lifecycle = ActivityLifecycle(
                journey, causal_head=seed["event_id"], clock=clock
            )
            activity = lifecycle.start(
                adapter="pydantic_ai",
                stage="planning_attempt",
            )
            clock.now = 10.2
            result = activity.record(_ObservedResult(value="done"))
            clock.now = 9.0
            activity.succeeded(result)

            events = journey.read()[1:]
            self.assertEqual(
                [event["parent_ids"] for event in events],
                [[seed["event_id"]]] + [[event["event_id"]] for event in events[:-1]],
            )
            self.assertEqual(
                [
                    (event["event_type"], event["payload"]["elapsed_ms"])
                    for event in events
                    if event["event_type"].startswith("activity_")
                ],
                [("activity_started", 0), ("activity_succeeded", 0)],
            )

    def test_runtime_closes_once_with_owned_evidence_and_refuses_post_close_records(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            seed = journey.record("Seed", {})
            lifecycle = ActivityLifecycle(journey, causal_head=seed["event_id"])
            activity = lifecycle.start(
                adapter="pydantic_ai",
                stage="planning_attempt",
            )
            with self.assertRaisesRegex(
                ValueError, "activity result must reference evidence recorded by"
            ):
                activity.succeeded(seed)
            result = activity.record(_ObservedResult(value="done"))
            terminal = activity.succeeded(result)

            self.assertEqual(terminal["payload"]["result_event_id"], result["event_id"])
            closed = journey.path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "activity is already closed"):
                activity.record(_ObservedResult(value="after close"))
            with self.assertRaisesRegex(RuntimeError, "activity is already closed"):
                activity.succeeded(result)
            self.assertEqual(journey.path.read_bytes(), closed)

    def test_failed_artifact_enrichment_leaves_causal_head_at_semantic_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            seed = journey.record("Seed", {})
            activity = ActivityLifecycle(journey, causal_head=seed["event_id"]).start(
                adapter="pydantic_ai",
                stage="planning_attempt",
            )
            identity_event = activity.record(_ObservedResult(value="spent identity"))

            def unavailable():
                raise RuntimeError("local enrichment unavailable")

            with self.assertRaisesRegex(RuntimeError, "local enrichment unavailable"):
                activity.enrich_artifact(
                    identity_event,
                    unavailable,
                    role="provider_create_response",
                    byte_count=1,
                    sha256=hashlib.sha256(b"x").hexdigest(),
                    media_type="text/plain",
                )
            failure = activity.record(_ObservedResult(value="failed"))

            self.assertEqual(failure["parent_ids"], [identity_event["event_id"]])
            self.assertFalse(
                any(
                    event["event_type"] == "ArtifactEnrichmentRetained"
                    for event in journey.read()
                )
            )

    def test_nested_activity_is_the_only_active_causal_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            seed = journey.record("Seed", {})
            lifecycle = ActivityLifecycle(journey, causal_head=seed["event_id"])
            parent = lifecycle.start(
                adapter="pydantic_ai",
                stage="planning_attempt",
            )
            child = parent.start_child(
                adapter="pydantic_ai_model",
                stage="model_contact",
            )
            parent.require_active_child(child)
            with self.assertRaises(RuntimeError):
                child.require_active_child(parent)
            with self.assertRaises(RuntimeError):
                parent.require_active_child(parent)
            with self.assertRaises(RuntimeError):
                parent.record(_ObservedResult(value="out of order"))
            child_result = child.record(_ObservedResult(value="response"))
            child.succeeded(child_result)
            with self.assertRaises(RuntimeError):
                parent.require_active_child(child)
            parent_result = parent.record(_ObservedResult(value="attempt"))
            parent.succeeded(parent_result)

    def test_validation_rejects_unknown_lifecycle_events(self):
        with self.assertRaises(ValueError):
            validate_activity_fact({"event_type": "activity_waiting", "payload": {}})


if __name__ == "__main__":
    unittest.main()
