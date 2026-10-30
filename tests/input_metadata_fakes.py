"""Reusable registered-input metadata observations."""

from __future__ import annotations

import copy
from typing import Any, Sequence

from bio_neb.input_inspection.tool import (
    InputMetadataObservation,
    RegisteredInputPath,
)

STARTED = "2026-08-06T10:00:00+00:00"
COMPLETED = "2026-08-06T10:00:01Z"


class ReturningAdapter:
    def __init__(self, value: Any):
        self.value = InputMetadataObservation.model_validate(value)
        self.requests: tuple[RegisteredInputPath, ...] | None = None

    def observe(
        self, requests: Sequence[RegisteredInputPath]
    ) -> InputMetadataObservation:
        self.requests = tuple(requests)
        return copy.deepcopy(self.value)


def metadata(role, path, *, size=100, inode="100", modified=1_000):
    return {
        "role": role,
        "registered_path": str(path),
        "exists": True,
        "is_regular": True,
        "size_bytes": size,
        "filesystem_identity": {"device": "7", "inode": inode},
        "modified_at_ns": modified,
    }


def observation(items, *, started=STARTED, completed=COMPLETED):
    return {
        "started_at": started,
        "completed_at": completed,
        "inputs": copy.deepcopy(items),
    }
