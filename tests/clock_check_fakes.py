"""A clock-check record as `bio_neb.devtools.clock_check` writes it, for readiness tests.

Written here, not by the tool, so readiness tests do not run the check they gate on.
`steady_clock` fixes the clocks readiness reads: the host's own wall clock may step
(#392), and a test owns its time inputs.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from bio_neb.devtools import clock_check

# 2026-10-24T12:00:00Z on the wall clock, 1000 s since boot.
WALL = 1792843200.0
BOOT = 1000.0


def steady_clock() -> ExitStack:
    """Readiness reads `WALL` and `BOOT` for as long as the returned stack is open."""
    stack = ExitStack()
    stack.enter_context(patch.object(clock_check, "wall_clock", lambda: WALL))
    stack.enter_context(patch.object(clock_check, "boot_clock", lambda: BOOT))
    return stack


def write_clock_record(
    home: Path,
    *,
    offset_seconds: float = 0.0,
    age_seconds: float = 0.0,
    largest_step_seconds: float = 0.0,
    stepped_since_seconds: float = 0.0,
) -> Path:
    """`<home>/clock-check.json` taken `age_seconds` before `WALL`, offset as given.

    Its watch saw `largest_step_seconds`; by `WALL` the wall clock has moved
    `stepped_since_seconds` against the boot clock since the check.
    """
    checked_at = datetime.fromtimestamp(WALL - age_seconds, timezone.utc)
    record = {
        "schema_version": "bio-neb.clock-check/1",
        "source": "https://api.github.com/rate_limit",
        "checked_at": checked_at.isoformat().replace("+00:00", "Z"),
        "readings": [],
        "offset_seconds": offset_seconds,
        "spread_seconds": 0.0,
        "watch": {"largest_step_seconds": largest_step_seconds, "steps": []},
        "wall_minus_boot_seconds": WALL - BOOT - stepped_since_seconds,
        "verdict": "recorded verdicts are not read",
        "warn_above_seconds": 2.0,
        "block_above_seconds": 5.0,
    }
    home.mkdir(parents=True, exist_ok=True)
    path = home / "clock-check.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path
