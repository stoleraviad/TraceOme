from __future__ import annotations

import unittest
import uuid
from unittest.mock import patch

from bio_neb.agentic_journey.attempt_terminal import (
    SliceReachedTerminal,
    project_attempt_terminal,
)


def _id() -> str:
    return str(uuid.uuid4())


def _event(
    event_type: str,
    *,
    event_id: str | None = None,
    parent_ids: list[str] | None = None,
    payload: dict | None = None,
) -> dict:
    return {
        "event_id": event_id or _id(),
        "event_type": event_type,
        "parent_ids": parent_ids or [],
        "payload": payload or {},
    }


def _attempt(role: str) -> tuple[str, list[dict]]:
    attempt_id = _id()
    started = _event(
        "AttemptStarted",
        payload={"attempt_id": attempt_id, "agent_role": role},
    )
    return attempt_id, [started]


def _append(
    events: list[dict], event_type: str, *, payload: dict | None = None
) -> dict:
    event = _event(event_type, parent_ids=[events[-1]["event_id"]], payload=payload)
    events.append(event)
    return event


class AttemptTerminalProjectionTests(unittest.TestCase):
    def test_missing_stale_cross_attempt_and_ambiguous_sources_fail_closed(self):
        role = "planning_agent"
        reasons = {
            "missing": "missing from retained Journey facts",
            "stale": "missing from retained Journey facts",
            "cross_attempt": "changed its owning Attempt",
            "ambiguous": "does not have one current causal head",
        }
        for condition, reason in reasons.items():
            with self.subTest(condition=condition):
                attempt_id, events = _attempt(role)
                source_id = _id()
                if condition != "missing":
                    payload_attempt = (
                        _id() if condition == "cross_attempt" else attempt_id
                    )
                    source = _append(
                        events,
                        "InputMetadataInspected",
                        payload={"attempt_id": payload_attempt},
                    )
                    source_id = source["event_id"]
                if condition == "stale":
                    source_id = _id()
                if condition == "ambiguous":
                    events.append(
                        _event(
                            "ModelResponseRecorded",
                            parent_ids=[events[0]["event_id"]],
                            payload={"attempt_id": attempt_id},
                        )
                    )
                # The assignment's own terminal choice cites the source; the
                # shared causal audit must still refuse it.
                disposition = SliceReachedTerminal(
                    checkpoint="registered_inputs_inspected",
                    source_event_id=source_id,
                )
                with patch(
                    "bio_neb.agentic_journey.planning_assignment.assignment_terminal",
                    return_value=disposition,
                ):
                    with self.assertRaisesRegex(ValueError, reason):
                        project_attempt_terminal(role, attempt_id, events)

    def test_completed_attempt_source_is_stale(self):
        attempt_id, events = _attempt("planning_agent")
        source = _append(
            events,
            "InputMetadataInspected",
            payload={"result": {"status": "succeeded"}},
        )
        _append(
            events,
            "AttemptSucceeded",
            payload={
                "attempt_id": attempt_id,
                "checkpoint_event_id": source["event_id"],
            },
        )
        with self.assertRaisesRegex(ValueError, "completed Attempt"):
            project_attempt_terminal("planning_agent", attempt_id, events)

    def test_unknown_or_retired_role_is_refused_before_projection(self):
        for role in ("unregistered_role", "execution_planning_agent"):
            with self.subTest(role=role):
                attempt_id, events = _attempt(role)
                with self.assertRaisesRegex(
                    ValueError, "not an administrative terminal stage"
                ):
                    project_attempt_terminal(role, attempt_id, events)


if __name__ == "__main__":
    unittest.main()
