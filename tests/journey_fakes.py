"""Reusable in-memory Journey fakes for current execution seams."""

from __future__ import annotations

import copy
import uuid


class MemoryJourney:
    def __init__(self, timeline):
        self.journey_id = timeline[0]["journey_id"]
        self.timeline = timeline

    def read(self):
        return copy.deepcopy(self.timeline)

    def record(self, event_type, event_payload, parent_ids, *, event_id=None, **_):
        event = {
            "journey_id": self.journey_id,
            "event_id": event_id or str(uuid.uuid4()),
            "event_type": event_type,
            "parent_ids": list(parent_ids),
            "payload": copy.deepcopy(event_payload),
        }
        self.timeline.append(event)
        return copy.deepcopy(event)


class ArtifactMemoryJourney(MemoryJourney):
    def record_with_artifact(
        self,
        event_type,
        event_payload,
        artifact_bytes,
        media_type,
        parent_ids,
        **kwargs,
    ):
        del artifact_bytes, media_type
        self.record(event_type, event_payload, parent_ids, **kwargs)
        self.timeline[-1]["artifact_references"] = []
        return copy.deepcopy(self.timeline[-1])
