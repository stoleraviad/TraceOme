"""The host clock check against an external time source, and its record (#392).

Rule owners: #392's Prepared answers Q4–Q6 (a GitHub source, three readings, median and
spread, warn above 2 s, block above 5 s), the orchestrator's answer (A) of 2026-10-24 (a
contact-free watch of the wall clock against the boot clock for 35 s, blocking on a step
over 5 s) and #452's owner answers Q1–Q2 (`GET https://api.github.com/rate_limit`; a
reading counts only from an answer that forbids caching, carries no `Age` and is not dated
before the previous one). The source is faked at `urlopen`; the clocks and `sleep` belong
to a scripted host.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from email.message import Message
from email.utils import formatdate
from pathlib import Path
from unittest.mock import patch

from bio_neb.devtools import clock_check

# 2026-10-24T12:00:00Z: real time when the scripted host's boot clock reads 0.
NOON = 1792843200.0


class _Host:
    """A boot clock that `sleep` advances, and a wall clock `offset(boot)` off real time."""

    def __init__(self, offset=lambda boot: 0.0) -> None:
        self.boot = 0.0
        self.offset = offset

    def boot_clock(self) -> float:
        return self.boot

    def wall_clock(self) -> float:
        return NOON + self.boot + self.offset(self.boot)

    def sleep(self, seconds: float) -> None:
        self.boot += seconds


class _Source:
    """GitHub's whole-second `Date` at real time, after a 0.2 s round trip.

    Each answer carries `cache_control` and, when given, `age`, as `/rate_limit` answers
    `Cache-Control: no-cache`; `stale_by(index)` dates answer `index` that many seconds
    before real time, as an edge cache does.
    """

    def __init__(
        self,
        host: _Host,
        cache_control: str | None = "no-cache",
        age: str | None = None,
        stale_by=lambda index: 0,
    ) -> None:
        self.host = host
        self.cache_control = cache_control
        self.age = age
        self.stale_by = stale_by
        self.requests = []

    def __call__(self, request, timeout):
        index = len(self.requests)
        self.requests.append((request, timeout))
        self.host.sleep(0.1)
        served = int(NOON + self.host.boot) - self.stale_by(index)
        self.host.sleep(0.1)
        headers = Message()
        headers["Date"] = formatdate(served, usegmt=True)
        if self.cache_control is not None:
            headers["Cache-Control"] = self.cache_control
        if self.age is not None:
            headers["Age"] = self.age
        return _Response(headers)


class _Response:
    def __init__(self, headers: Message) -> None:
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


class ClockCheckTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(
            patch.dict(os.environ, {"BIO_NEB_CONTROLLER_HOME": str(self.home)})
        )

    def _main(self, host: _Host, source=None) -> tuple[int, dict, _Source]:
        source = source or _Source(host)
        stdout = io.StringIO()
        with (
            patch.object(clock_check, "urlopen", source),
            patch.object(clock_check, "wall_clock", host.wall_clock),
            patch.object(clock_check, "boot_clock", host.boot_clock),
            patch.object(clock_check, "sleep", host.sleep),
            redirect_stdout(stdout),
        ):
            code = clock_check.main([])
        return code, json.loads(stdout.getvalue() or "null"), source

    def test_a_host_three_seconds_behind_reads_warn_from_three_uncached_readings(self):
        # Readings start at real 12:00:35.4 (local 12:00:32.4), 0.1 s out and 0.1 s
        # back, 0.2 s apart; the server's Date floors to 12:00:35 each time, so the
        # offsets are 3 s less the sub-second part of each midpoint.
        host = _Host(lambda boot: -3.0)
        host.boot = 0.4

        code, printed, source = self._main(host)

        self.assertEqual(code, 0)
        self.assertEqual(
            [reading["offset_seconds"] for reading in printed["readings"]],
            [2.5, 2.3, 2.1],
        )
        self.assertEqual(printed["offset_seconds"], 2.3)
        self.assertEqual(printed["spread_seconds"], 0.4)
        self.assertEqual(printed["verdict"], "warn")
        self.assertEqual(printed["source"], "https://api.github.com/rate_limit")
        self.assertEqual(
            [reading["cache_control"] for reading in printed["readings"]],
            ["no-cache"] * 3,
        )
        # The watch ran before the readings and saw no step.
        watch = printed["watch"]
        self.assertEqual((watch["seconds"], watch["samples"]), (35.0, 351))
        self.assertEqual((watch["steps"], watch["largest_step_seconds"]), ([], 0.0))
        self.assertEqual(printed["wall_minus_boot_seconds"], NOON - 3.0)
        # One GET of `/rate_limit` per reading (it answers HEAD with 404), asking for no
        # cached answer.
        self.assertEqual(len(source.requests), 3)
        for request, timeout in source.requests:
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(request.full_url, "https://api.github.com/rate_limit")
            self.assertEqual(request.get_header("Cache-control"), "no-cache")
            self.assertGreater(timeout, 0)
        # The record readiness reads is what was printed.
        self.assertEqual(
            json.loads((self.home / "clock-check.json").read_text(encoding="utf-8")),
            printed,
        )

    def test_a_host_that_steps_back_between_readings_blocks_on_the_watch(self):
        # A recorded trial host: back about 111 s for 0.2 s, then forward again. The
        # readings land between steps and read the clock right; the watch does not.
        host = _Host(lambda boot: -111.0 if 10.05 <= boot < 10.25 else 0.0)

        code, printed, _ = self._main(host)

        self.assertEqual(code, 1)
        self.assertEqual(printed["offset_seconds"], -0.3)
        self.assertEqual(
            [step["size_seconds"] for step in printed["watch"]["steps"]],
            [-111.0, 111.0],
        )
        self.assertEqual(printed["watch"]["largest_step_seconds"], 111.0)
        self.assertEqual(printed["verdict"], "block")
        self.assertTrue((self.home / "clock-check.json").is_file())

    def test_an_offset_beyond_five_seconds_blocks_and_exits_one(self):
        # A host 103 s behind throughout, the backstep held: no step, a far offset.
        host = _Host(lambda boot: -103.0)

        code, printed, _ = self._main(host)

        self.assertEqual(code, 1)
        self.assertEqual(
            (printed["offset_seconds"], printed["verdict"]), (102.7, "block")
        )
        self.assertEqual(printed["watch"]["steps"], [])

    def test_the_bound_is_strictly_above_two_and_five_seconds_either_way(self):
        for offset, expected in (
            (0.0, "ok"),
            (2.0, "ok"),
            (-2.0, "ok"),
            (2.001, "warn"),
            (-4.0, "warn"),
            (5.0, "warn"),
            (5.001, "block"),
            (-103.0, "block"),
        ):
            with self.subTest(offset=offset):
                self.assertEqual(clock_check.verdict(offset), expected)

    def test_an_unreachable_source_removes_the_last_record_and_reports_its_watch(self):
        # A failed check is no reading, and the earlier clean record no longer speaks
        # for the clock: the watch may just have seen it step (Spec r1 SPEC-1).
        clean = _Host(lambda boot: -111.0 if 45.65 <= boot < 45.85 else 0.0)
        code, _, _ = self._main(clean)
        self.assertEqual(code, 0)
        self.assertTrue((self.home / "clock-check.json").is_file())

        def unreachable(request, timeout):
            raise OSError("network unreachable")

        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            code, printed, _ = self._main(clean, unreachable)

        self.assertEqual((code, printed), (2, None))
        refusal = json.loads(stderr.getvalue())
        self.assertEqual(refusal["code"], "clock_source_unreadable")
        self.assertEqual(refusal["cause"], "OSError")
        self.assertEqual(
            [step["size_seconds"] for step in refusal["watch"]["steps"]],
            [-111.0, 111.0],
        )
        self.assertFalse((self.home / "clock-check.json").exists())
        # Readiness, under the same clocks, now finds no record to trust.
        with (
            patch.object(clock_check, "wall_clock", clean.wall_clock),
            patch.object(clock_check, "boot_clock", clean.boot_clock),
        ):
            _, blockers = clock_check.readiness()
        self.assertEqual(blockers, ["clock_check_missing"])

    def test_an_answer_a_cache_may_have_served_refuses_the_check_and_its_record(self):
        # `https://api.github.com/` answered `public, max-age=60, s-maxage=60` and a
        # `Date` 20 s old on a correct clock; `/zen` answered with no cache header 14 s
        # old; `github.com/` answered `max-age=0` 8 s old. Only an answer that forbids
        # caching and carries no `Age` is a reading.
        for name, source_options, reason in (
            (
                "edge_cached",
                {"cache_control": "public, max-age=60, s-maxage=60"},
                "answer_cacheable",
            ),
            ("no_cache_header", {"cache_control": None}, "answer_cacheable"),
            (
                "max_age_zero",
                {"cache_control": "max-age=0, private, must-revalidate"},
                "answer_cacheable",
            ),
            ("aged", {"cache_control": "no-cache", "age": "20"}, "answer_aged"),
        ):
            with self.subTest(case=name):
                host = _Host()
                self._main(host)
                self.assertTrue((self.home / "clock-check.json").is_file())

                stderr = io.StringIO()
                with patch("sys.stderr", stderr):
                    code, printed, source = self._main(
                        host, _Source(host, **source_options)
                    )

                self.assertEqual((code, printed), (2, None))
                refusal = json.loads(stderr.getvalue())
                self.assertEqual(refusal["code"], "clock_source_stale")
                self.assertEqual(refusal["reason"], reason)
                # The first such answer stops the check: no further request is sent.
                self.assertEqual(len(source.requests), 1)
                self.assertFalse((self.home / "clock-check.json").exists())

    def test_a_date_earlier_than_the_previous_readings_refuses_the_check(self):
        # Two GitHub answers milliseconds apart once carried `Date`s 25 s apart. Server
        # time cannot go back between readings, so the later answer is stale even though
        # its headers forbid caching.
        host = _Host()
        source = _Source(host, stale_by=lambda index: 25 if index == 2 else 0)
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            code, printed, _ = self._main(host, source)

        self.assertEqual((code, printed), (2, None))
        refusal = json.loads(stderr.getvalue())
        self.assertEqual(
            (refusal["code"], refusal["reason"]),
            ("clock_source_stale", "date_went_back"),
        )
        self.assertEqual(len(source.requests), 3)
        self.assertFalse((self.home / "clock-check.json").exists())

    def test_a_record_that_cannot_be_read_or_decoded_is_unreadable(self):
        # Standards r1 S1 / Spec r1 SPEC-2: every read failure but absence is one
        # blocker, with the command to take the check again.
        record = self.home / "clock-check.json"
        for name, make in (
            ("truncated_json", lambda: record.write_text("{", encoding="utf-8")),
            ("not_utf8", lambda: record.write_bytes(b"\xff\xfe")),
            ("not_an_object", lambda: record.write_text("[]", encoding="utf-8")),
            ("directory", lambda: record.mkdir()),
        ):
            with self.subTest(case=name):
                make()
                section, blockers = clock_check.readiness()
                self.assertEqual(blockers, ["clock_check_unreadable"])
                self.assertEqual(section["record"], str(record))
                self.assertEqual(section["command"][-1], "bio_neb.devtools.clock_check")
                if record.is_dir():
                    record.rmdir()
                else:
                    record.unlink()


if __name__ == "__main__":
    unittest.main()
