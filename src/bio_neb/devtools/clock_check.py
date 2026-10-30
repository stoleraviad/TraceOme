"""The host wall clock against an external time source, checked before live work (#392).

    uv run --locked --offline python -m bio_neb.devtools.clock_check

Journey `occurred_at`, the watcher and Logfire's spans (the OpenTelemetry SDK stamps them
with `time.time_ns`) all read the host wall clock, so their times are grounded only when
it is. This DEV tool first watches the wall clock against the boot clock for 35 s,
contact-free, and records every step (the 2026-10-24 host stepped back about 111 s for
0.2 s every ~30 s; orchestrator answer (A)). It then takes three readings, each one HTTPS
GET of `https://api.github.com/rate_limit` (#452: it answers `Cache-Control: no-cache` and
counts against no rate limit, where GitHub's edge-cached root once answered a `Date` 20 s
old on a correct clock). A reading counts only from an answer that forbids caching
(`no-cache` or `no-store`), carries no `Age` and is dated no earlier than the reading
before it; any other answer stops the check as stale. A reading's offset is the server's
`Date` minus the local midpoint of its request; `Date` has whole-second resolution. It
prints and writes one record, `<controller-home>/clock-check.json`: the
watch, the readings, their median offset and spread, the wall-minus-boot baseline, and
the verdict on the larger of the offset and the largest step, `ok`, `warn` above 2 s or
`block` above 5 s (owner, #392 Q6). It exits 0 for `ok` or `warn`, 1 for `block`, and 2
when a reading fails or is stale: it then writes no record, removes the earlier one and
reports the watch on stderr. Contact-free `live_readiness` reads the record (`readiness`) and blocks a
live step on a missing, unreadable, stale, out-of-bound or since-stepped one; it never
contacts the source itself.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from time import sleep
from time import time as wall_clock
from urllib.request import Request, urlopen

from bio_neb import locations

SCHEMA_VERSION = "bio-neb.clock-check/1"
SOURCE = "https://api.github.com/rate_limit"
READINGS = 3
TIMEOUT_SECONDS = 10.0
WARN_ABOVE_SECONDS = 2.0
BLOCK_ABOVE_SECONDS = 5.0
# The watch outlasts one cycle of a stepping host clock (observed every ~30 s).
WATCH_SECONDS = 35.0
WATCH_INTERVAL_SECONDS = 0.1
# A change of wall minus boot clock this large between samples is a step, not slew.
STEP_SECONDS = 0.1
# A record older than this no longer speaks for the clock a live step starts on.
FRESH_FOR_SECONDS = 600.0
COMMAND = (
    "uv",
    "run",
    "--locked",
    "--offline",
    "python",
    "-m",
    "bio_neb.devtools.clock_check",
)


def boot_clock() -> float:
    """Seconds since boot: never stepped, so the wall clock is watched against it."""
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def record_path() -> Path:
    return locations.controller_home() / "clock-check.json"


def verdict(offset_seconds: float) -> str:
    """The one bound judgment, applied by the check and again by readiness."""
    size = abs(offset_seconds)
    if size > BLOCK_ABOVE_SECONDS:
        return "block"
    return "warn" if size > WARN_ABOVE_SECONDS else "ok"


class StaleAnswer(Exception):
    """An answer a cache may have served, or dated before the reading before it."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _fresh_cache_control(headers) -> str:
    """The answer's `Cache-Control`, when it forbids serving the answer from a cache.

    `max-age=0` does not: `github.com/` answered it with a `Date` 8 s old (#452).
    """
    cache_control = headers.get("Cache-Control", "")
    directives = {
        part.split("=")[0].strip().lower() for part in cache_control.split(",")
    }
    if not directives & {"no-cache", "no-store"}:
        raise StaleAnswer("answer_cacheable")
    if headers.get("Age") is not None:
        raise StaleAnswer("answer_aged")
    return cache_control


def _iso(seconds: float) -> str:
    moment = datetime.fromtimestamp(seconds, timezone.utc)
    return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _reading() -> tuple[dict[str, object], float, float]:
    """One reading, the local time it came back and the server's time in it."""
    request = Request(SOURCE, method="GET", headers={"Cache-Control": "no-cache"})
    sent = wall_clock()
    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        date = response.headers["Date"]
        cache_control = _fresh_cache_control(response.headers)
    received = wall_clock()
    server = parsedate_to_datetime(date).timestamp()
    return (
        {
            "date": date,
            "cache_control": cache_control,
            "sent_at": _iso(sent),
            "round_trip_seconds": round(received - sent, 3),
            "offset_seconds": round(server - (sent + received) / 2, 3),
        },
        received,
        server,
    )


def _watch() -> tuple[dict[str, object], float]:
    """Wall minus boot clock, sampled across `WATCH_SECONDS`: each step and the last value.

    Contact-free. A host whose wall clock steps for a fraction of a second between
    readings reads right to the source and wrong to every event stamped meanwhile.
    """
    count = round(WATCH_SECONDS / WATCH_INTERVAL_SECONDS) + 1
    steps: list[dict[str, object]] = []
    previous = None
    start = boot_clock()
    for index in range(count):
        if index:
            sleep(WATCH_INTERVAL_SECONDS)
        boot = boot_clock()
        difference = wall_clock() - boot
        if previous is not None and abs(difference - previous) > STEP_SECONDS:
            steps.append(
                {
                    "after_seconds": round(boot - start, 3),
                    "size_seconds": round(difference - previous, 3),
                }
            )
        previous = difference
    largest = max((abs(step["size_seconds"]) for step in steps), default=0.0)
    watch = {
        "seconds": WATCH_SECONDS,
        "interval_seconds": WATCH_INTERVAL_SECONDS,
        "samples": count,
        "steps": steps,
        "largest_step_seconds": largest,
    }
    return watch, round(previous, 3)


def _readings() -> tuple[list[dict[str, object]], float]:
    """The source's readings, and the local time the last one came back.

    Server time cannot go back between readings, and whole-second `Date`s keep that true
    for fresh answers, so a `Date` before the previous one is a cached answer.
    """
    readings = []
    previous = None
    for _ in range(READINGS):
        reading, received, server = _reading()
        if previous is not None and server < previous:
            raise StaleAnswer("date_went_back")
        readings.append(reading)
        previous = server
    return readings, received


def _record(
    watch: dict[str, object],
    wall_minus_boot: float,
    readings: list[dict[str, object]],
    received: float,
) -> dict[str, object]:
    offsets = [reading["offset_seconds"] for reading in readings]
    offset = statistics.median(offsets)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "checked_at": _iso(received),
        "readings": readings,
        "offset_seconds": offset,
        "spread_seconds": round(max(offsets) - min(offsets), 3),
        "watch": watch,
        "wall_minus_boot_seconds": wall_minus_boot,
        "verdict": verdict(max(abs(offset), watch["largest_step_seconds"])),
        "warn_above_seconds": WARN_ABOVE_SECONDS,
        "block_above_seconds": BLOCK_ABOVE_SECONDS,
    }


def _write(record: dict[str, object]) -> None:
    path = record_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, staging = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(staging, path)


def readiness() -> tuple[dict[str, object], list[str]]:
    """What the last record says about the clock now, and the blocker it raises.

    Contact-free: it reads only the record and the two local clocks. The bound is judged
    again here from the recorded median and the watch's largest step, so a record's own
    verdict is never trusted. A record older than `FRESH_FOR_SECONDS`, or dated later
    than now by more than the block bound, is stale; a wall clock that has moved against
    the boot clock by more than the block bound since the check has stepped since it.
    """
    path = record_path()
    section: dict[str, object] = {"record": str(path), "command": list(COMMAND)}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {**section, "present": False}, ["clock_check_missing"]
    except (OSError, ValueError):
        return {**section, "present": True}, ["clock_check_unreadable"]
    try:
        record = json.loads(text)
        checked_at = datetime.fromisoformat(record["checked_at"]).timestamp()
        offset = float(record["offset_seconds"])
        spread = float(record["spread_seconds"])
        largest_step = float(record["watch"]["largest_step_seconds"])
        wall_minus_boot = float(record["wall_minus_boot_seconds"])
    except (OSError, ValueError, TypeError, KeyError):
        return {**section, "present": True}, ["clock_check_unreadable"]
    now = wall_clock()
    age = round(now - checked_at, 3)
    stepped = round(now - boot_clock() - wall_minus_boot, 3)
    section.update(
        present=True,
        checked_at=record["checked_at"],
        age_seconds=age,
        offset_seconds=offset,
        spread_seconds=spread,
        largest_step_seconds=largest_step,
        stepped_since_check_seconds=stepped,
        verdict=verdict(max(abs(offset), largest_step)),
    )
    if not -BLOCK_ABOVE_SECONDS <= age <= FRESH_FOR_SECONDS:
        return section, ["clock_check_stale"]
    if abs(stepped) > BLOCK_ABOVE_SECONDS:
        return section, ["clock_stepped_since_check"]
    if section["verdict"] == "block":
        return section, ["clock_offset_beyond_bound"]
    return section, []


def main(argv: list[str] | None = None) -> int:
    if argv:
        print(f"usage: {' '.join(COMMAND[4:])}", file=sys.stderr)
        return 2
    watch, wall_minus_boot = _watch()
    try:
        readings, received = _readings()
    except StaleAnswer as stale:
        refusal = {"code": "clock_source_stale", "reason": stale.reason}
    except (OSError, KeyError, TypeError, ValueError) as error:
        refusal = {"code": "clock_source_unreadable", "cause": type(error).__name__}
    else:
        record = _record(watch, wall_minus_boot, readings, received)
        _write(record)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 1 if record["verdict"] == "block" else 0
    # No fresh reading is no check, and the earlier record no longer speaks for a clock
    # this watch may just have seen step: remove it, and report the watch.
    record_path().unlink(missing_ok=True)
    print(
        json.dumps({"status": "refused", **refusal, "watch": watch}),
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
