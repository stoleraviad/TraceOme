"""Batched offline runs: minimal ledger, resume, per-module timeout and comparison.

Each test runs the real batch controller against a throwaway Git checkout holding a tiny
synthetic test package, so module children, timeouts and the post-batch checkout check are
real processes and real Git state rather than doubles of the runner's own mechanics.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import ANY, Mock, patch

from bio_neb import locations
from bio_neb.devtools import run_sandbox, test_queue
from bio_neb.devtools import testing as testing_operation


def setUpModule() -> None:
    # The runner commands these tests start take host test slots; a private queue
    # keeps a direct run of this module off the host's slots.
    directory = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(directory.cleanup)
    environment = patch.dict(
        os.environ, {locations.TEST_QUEUE_OVERRIDE: directory.name}
    )
    environment.start()
    unittest.addModuleCleanup(environment.stop)
    os.environ.pop(test_queue.HELD, None)


_RUNS_LOG = "BIO_NEB_BATCH_TEST_RUNS"
_FAKE_MODULES = {
    "test_green.py": """
        import unittest

        class Green(unittest.TestCase):
            def test_passes(self):
                pass
    """,
    "test_red.py": """
        import unittest

        class Red(unittest.TestCase):
            def test_fails(self):
                self.fail("expected red")

            def test_errors(self):
                try:
                    {}["missing"]
                except KeyError:
                    raise RuntimeError(
                        "expected error\\n" + "x" * 2500 + "\\nlast line"
                    )

            def test_passes(self):
                pass
    """,
    "test_slow.py": """
        import os
        import time
        import unittest

        class Slow(unittest.TestCase):
            def test_waits(self):
                started = os.environ.get("BIO_NEB_BATCH_TEST_STARTED")
                if started:
                    open(started, "w").close()
                time.sleep(float(os.environ.get("BIO_NEB_BATCH_TEST_SLEEP", "0")))
    """,
    "test_residue.py": """
        import unittest
        from pathlib import Path

        class Residue(unittest.TestCase):
            def test_writes_into_the_checkout(self):
                Path("left-behind.txt").write_text("residue")
    """,
    "test_hang.py": """
        import time
        import unittest

        class Hang(unittest.TestCase):
            def test_never_finishes_in_time(self):
                time.sleep(60)
    """,
    "test_spawner.py": """
        import os
        import subprocess
        import sys
        import unittest

        class Spawner(unittest.TestCase):
            def test_leaves_a_process_behind(self):
                sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
                with open(os.environ["BIO_NEB_BATCH_TEST_SPAWNED"], "w") as stream:
                    stream.write(str(sleeper.pid))
    """,
    "test_ladder.py": """
        import os
        import time
        import unittest

        class Ladder(unittest.TestCase):
            def setUp(self):
                with open(os.environ["BIO_NEB_BATCH_TEST_EXECUTED"], "a") as stream:
                    stream.write(self.id() + "\\n")

            def test_representative(self):
                if os.environ.get("BIO_NEB_BATCH_TEST_HANG_REPRESENTATIVE"):
                    time.sleep(60)
                if os.environ.get("BIO_NEB_BATCH_TEST_RED_REPRESENTATIVE"):
                    self.fail("forced red representative")

            def test_other_green(self):
                pass

            def test_other_red(self):
                self.fail("red beyond the representative")

        class LadderChild(Ladder):
            def test_other_red(self):
                pass

        def helper():
            return None
    """,
    "test_broken.py": """
        raise ImportError("broken on import")
    """,
    "test_mixin.py": """
        import unittest

        class Checks:
            def test_mixed(self):
                pass

        class Mixed(Checks, unittest.TestCase):
            pass
    """,
    "test_tmp_writer.py": """
        import json
        import os
        import tempfile
        import unittest
        from pathlib import Path

        class TmpWriter(unittest.TestCase):
            def test_leaves_a_temporary_file_and_writes_a_fixed_path(self):
                tempfile.mkstemp()
                try:
                    closed = os.open(os.devnull, os.O_RDONLY)
                    os.close(closed)
                    os.fsync(closed)  # a real fsync refuses a closed descriptor
                    fsync = "no-op"
                except OSError:
                    fsync = "real"
                names = (
                    "GIT_OPTIONAL_LOCKS",
                    "BIO_NEB_LOCAL_STATE",
                    "BIO_NEB_CONTROLLER_HOME",
                    "BIO_NEB_TESTING_RUN_ROOTS",
                    "BIO_NEB_PREREQUISITE_STORE",
                    "PYTHONPYCACHEPREFIX",
                )
                with open(os.environ["BIO_NEB_BATCH_TEST_ENVIRONMENT"], "w") as stream:
                    json.dump(
                        {
                            "tmpdir": tempfile.gettempdir(),
                            "fsync": fsync,
                            **{name: os.environ.get(name) for name in names},
                        },
                        stream,
                    )
                fixed = os.environ.get("BIO_NEB_BATCH_TEST_FIXED")
                if fixed:
                    Path(fixed).write_text("shared")
    """,
    "test_escape.py": """
        import json
        import os
        import unittest
        from pathlib import Path

        from bio_neb import locations

        class Escape(unittest.TestCase):
            def test_writes_home_local_state_and_the_checkout(self):
                written = {
                    "home": Path.home() / "escaped-home.txt",
                    "local_state": locations.local_state_root() / "escaped-state.txt",
                    "checkout": Path("escaped-checkout.txt"),
                }
                for path in written.values():
                    path.write_text("escaped")
                with open(os.environ["BIO_NEB_BATCH_TEST_ENVIRONMENT"], "w") as stream:
                    json.dump(
                        {name: str(path.resolve()) for name, path in written.items()},
                        stream,
                    )
    """,
}
# Two modules that each wait for the other to start: green only when they run together.
_MEET = """
import os
import time
import unittest
from pathlib import Path

class Meet(unittest.TestCase):
    def test_meets_its_partner(self):
        meeting = Path(os.environ["BIO_NEB_BATCH_TEST_MEET"])
        (meeting / __name__).touch()
        partner = __name__[:-1] + ("b" if __name__.endswith("a") else "a")
        deadline = time.monotonic() + float(os.environ["BIO_NEB_BATCH_TEST_MEET_SECONDS"])
        while not (meeting / partner).exists():
            if time.monotonic() > deadline:
                self.fail("partner never ran alongside")
            time.sleep(0.05)
"""
_FAKE_MODULES["test_meet_a.py"] = _MEET
_FAKE_MODULES["test_meet_b.py"] = _MEET
# Every fake module records its own import so a test can count actual module runs.
_RUN_RECORDER = """
import os
from pathlib import Path

_log = os.environ.get("{runs_log}")
if _log:
    with Path(_log).open("a") as stream:
        stream.write(__name__ + "\\n")
"""


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=batch-test",
            "-c",
            "user.email=batch-test@example.invalid",
            *arguments,
        ],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class _FakeCheckoutTestCase(unittest.TestCase):
    """A throwaway Git checkout holding the fake suite, plus batch helpers."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base = Path(directory.name).resolve()
        self.checkout = base / "checkout"
        package = self.checkout / "fakesuite"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        recorder = _RUN_RECORDER.format(runs_log=_RUNS_LOG)
        for name, source in _FAKE_MODULES.items():
            (package / name).write_text(recorder + textwrap.dedent(source))
        (self.checkout / "uv.lock").write_text("locked = true\n")
        (self.checkout / ".gitignore").write_text("__pycache__/\n")
        _git(self.checkout, "init", "-q")
        _git(self.checkout, "add", "-A")
        _git(self.checkout, "commit", "-q", "-m", "fake suite")
        self.root = base / "ledger-root"
        self.runs_log = base / "runs.log"
        self.started = base / "slow-started"
        self.executed = base / "executed.log"
        self.meeting = base / "meeting"
        self.meeting.mkdir()
        self.module_environment = base / "module-environment.json"
        environment = patch.dict(
            os.environ,
            {
                _RUNS_LOG: str(self.runs_log),
                "BIO_NEB_BATCH_TEST_SLEEP": "0",
                "BIO_NEB_BATCH_TEST_EXECUTED": str(self.executed),
                "BIO_NEB_BATCH_TEST_MEET": str(self.meeting),
                "BIO_NEB_BATCH_TEST_MEET_SECONDS": "20",
                "BIO_NEB_BATCH_TEST_ENVIRONMENT": str(self.module_environment),
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def _batch(self, modules: list[str], *, timeout: float = 60.0, **options) -> int:
        with patch("sys.stdout"):
            return testing_operation._run_batch(
                modules, self.root, timeout=timeout, checkout=self.checkout, **options
            )

    def _start_controller(
        self, modules: list[str], *, timeout: float, sleep: str, concurrency: int = 1
    ) -> subprocess.Popen[bytes]:
        """Run the controller in its own process so a test can interrupt or kill it."""
        controller = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "from pathlib import Path\n"
                "from bio_neb.devtools.testing import _run_batch\n"
                "concurrency = int(sys.argv[4])\n"
                "_run_batch(sys.argv[5:], Path(sys.argv[1]), timeout=float(sys.argv[3]),"
                " checkout=Path(sys.argv[2]), concurrency=concurrency)\n",
                str(self.root),
                str(self.checkout),
                str(timeout),
                str(concurrency),
                *modules,
            ],
            env={
                **os.environ,
                "BIO_NEB_BATCH_TEST_STARTED": str(self.started),
                "BIO_NEB_BATCH_TEST_SLEEP": sleep,
            },
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(controller.kill)
        deadline = time.monotonic() + 30
        while not self.started.exists():
            self.assertLess(time.monotonic(), deadline, "slow module never started")
            time.sleep(0.05)
        return controller

    def _ledger(self) -> list[dict]:
        lines = (self.root / "ledger.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]

    def _runs(self) -> list[str]:
        if not self.runs_log.exists():
            return []
        return self.runs_log.read_text().splitlines()


class BatchLedgerTests(_FakeCheckoutTestCase):
    def test_fresh_batch_writes_a_header_and_one_finish_line_per_module(self) -> None:
        result = self._batch(["fakesuite.test_green", "fakesuite.test_red"])

        self.assertEqual(result, 1)
        header, green, red, after = self._ledger()
        self.assertEqual(
            header,
            {
                "schema": "bio-neb.testing-batch-ledger/5",
                "commit": _git(self.checkout, "rev-parse", "HEAD"),
                "clean_tree": True,
                "dirty_paths": [],
                "lock_sha256": hashlib.sha256(b"locked = true\n").hexdigest(),
                "interpreter": sys.executable,
                "selection": ["fakesuite.test_green", "fakesuite.test_red"],
                "rung": "full",
                "timeout_seconds": 60.0,
                "concurrency": 1,
                "order": "sorted",
                "detect_writes": False,
                "shards": {},
                "estimated_seconds": {},
                "timing_ledgers": [],
            },
        )
        self.assertEqual(
            {key: green[key] for key in ("module", "exit", "failed", "errored")},
            {
                "module": "fakesuite.test_green",
                "exit": 0,
                "failed": [],
                "errored": [],
            },
        )
        self.assertEqual(
            {key: red[key] for key in ("module", "exit", "failed", "errored")},
            {
                "module": "fakesuite.test_red",
                "exit": 1,
                "failed": ["fakesuite.test_red.Red.test_fails"],
                "errored": ["fakesuite.test_red.Red.test_errors"],
            },
        )
        for finish in (green, red):
            with self.subTest(module=finish["module"]):
                self.assertEqual(
                    set(finish),
                    {
                        "module",
                        "exit",
                        "seconds",
                        "failed",
                        "errored",
                        "reasons",
                        "log",
                        "test_seconds",
                        "peak_rss_kib",
                        "descendant_peak_rss_kib",
                    },
                )
                self.assertGreater(finish["seconds"], 0)
                self.assertGreater(finish["peak_rss_kib"], 1024)
                self.assertGreaterEqual(finish["descendant_peak_rss_kib"], 0)
                log = self.root / finish["log"]
                self.assertTrue(log.is_file())
                self.assertIn("Ran ", log.read_text())
        self.assertIn("expected red", (self.root / red["log"]).read_text())
        self.assertEqual(green["reasons"], {})
        self.assertEqual(
            red["reasons"],
            {
                "fakesuite.test_red.Red.test_fails": "AssertionError: expected red",
                "fakesuite.test_red.Red.test_errors": (
                    "RuntimeError: expected error\n" + "x" * 2500 + "\nlast line"
                ),
            },
        )
        # A batch run without the command took no slots of its own to wait for.
        self.assertEqual(
            after,
            {
                "clean_tree_after": True,
                "dirty_paths_after": [],
                "slot_wait_seconds": None,
            },
        )

    def test_each_finish_line_times_every_test_the_module_ran(self) -> None:
        with patch.dict(os.environ, {"BIO_NEB_BATCH_TEST_SLEEP": "0.3"}):
            self._batch(["fakesuite.test_red", "fakesuite.test_slow"])

        _, red, slow, _ = self._ledger()
        self.assertEqual(
            sorted(red["test_seconds"]),
            [
                "fakesuite.test_red.Red.test_errors",
                "fakesuite.test_red.Red.test_fails",
                "fakesuite.test_red.Red.test_passes",
            ],
        )
        self.assertTrue(all(seconds >= 0 for seconds in red["test_seconds"].values()))
        (waited,) = slow["test_seconds"].values()
        self.assertGreaterEqual(waited, 0.3)
        self.assertLessEqual(waited, slow["seconds"])

    def test_resume_skips_finished_reds_and_timeouts_and_reruns_the_interrupted_one(
        self,
    ) -> None:
        modules = [
            "fakesuite.test_green",
            "fakesuite.test_hang",
            "fakesuite.test_red",
            "fakesuite.test_slow",
        ]
        controller = self._start_controller(modules, timeout=3.0, sleep="60")
        controller.send_signal(signal.SIGINT)
        controller.wait(timeout=30)

        interrupted = self._ledger()
        self.assertEqual(
            [(line.get("module"), line.get("exit")) for line in interrupted[1:]],
            [
                ("fakesuite.test_green", 0),
                ("fakesuite.test_hang", "timeout"),
                ("fakesuite.test_red", 1),
            ],
        )
        self.assertTrue((self.root / "logs/fakesuite.test_slow.log").is_file())
        finished_logs = {
            line["log"]: (self.root / line["log"]).read_bytes()
            for line in interrupted[1:]
        }

        result = self._batch(modules, timeout=3.0)

        self.assertEqual(result, 1)
        resumed = self._ledger()
        self.assertEqual(resumed[: len(interrupted)], interrupted)
        self.assertEqual(
            [line.get("module") for line in resumed[1:]],
            [*modules[:3], "fakesuite.test_slow", None],
        )
        self.assertEqual(
            (resumed[4]["exit"], resumed[4]["log"]),
            (0, "logs/fakesuite.test_slow.2.log"),
        )
        self.assertEqual(
            sorted(self._runs()),
            [*modules[:3], "fakesuite.test_slow", "fakesuite.test_slow"],
        )
        for log, content in finished_logs.items():
            with self.subTest(log=log):
                self.assertEqual((self.root / log).read_bytes(), content)

    def test_a_killed_controller_blocks_resume_until_its_orphaned_module_stops(
        self,
    ) -> None:
        modules = ["fakesuite.test_slow"]
        controller = self._start_controller(modules, timeout=60.0, sleep="10")
        controller.kill()
        controller.wait(timeout=30)

        with self.assertRaisesRegex(
            testing_operation.ControlSurfaceError, "still running"
        ):
            self._batch(modules)

        deadline = time.monotonic() + 30
        while True:
            try:
                result = self._batch(modules)
                break
            except testing_operation.ControlSurfaceError:
                self.assertLess(time.monotonic(), deadline, "orphan never stopped")
                time.sleep(0.2)
        self.assertEqual(result, 0)
        self.assertEqual(self._runs(), ["fakesuite.test_slow"] * 2)

    def test_timeout_is_recorded_as_timeout(self) -> None:
        result = self._batch(["fakesuite.test_hang"], timeout=1.0)

        self.assertEqual(result, 1)
        _, timed_out, after = self._ledger()
        self.assertEqual(timed_out["exit"], "timeout")
        self.assertIsNone(timed_out["failed"])
        self.assertIsNone(timed_out["errored"])
        self.assertLess(timed_out["seconds"], 30)
        self.assertTrue(after["clean_tree_after"])

    def test_a_module_limit_stops_only_its_module(self) -> None:
        """Ticket 234: a hanging fast module stops at its own limit, not the batch's."""
        with patch.dict(os.environ, {"BIO_NEB_BATCH_TEST_SLEEP": "3"}):
            result = self._batch(
                ["fakesuite.test_hang", "fakesuite.test_slow"],
                timeout=60.0,
                module_timeouts={"fakesuite.test_hang": 1.0},
            )

        self.assertEqual(result, 1)
        header, hang, slow, _ = self._ledger()
        self.assertEqual(header["timeout_seconds"], 60.0)
        self.assertEqual(header["module_timeout_seconds"], {"fakesuite.test_hang": 1.0})
        self.assertEqual(hang["exit"], "timeout")
        self.assertLess(hang["seconds"], 30)
        # The slow module outlives the hanging one's limit under the batch's own.
        self.assertEqual(slow["exit"], 0)
        self.assertGreater(slow["seconds"], 3)

    def test_resume_refuses_other_module_limits(self) -> None:
        self.assertEqual(
            self._batch(
                ["fakesuite.test_green"], module_timeouts={"fakesuite.test_green": 30.0}
            ),
            0,
        )

        with self.assertRaisesRegex(
            testing_operation.ControlSurfaceError, "module_timeout_seconds"
        ):
            self._batch(
                ["fakesuite.test_green"], module_timeouts={"fakesuite.test_green": 40.0}
            )

    def test_no_module_process_outlives_its_finish_line(self) -> None:
        spawned = self.root.parent / "spawned-pid"
        with patch.dict(os.environ, {"BIO_NEB_BATCH_TEST_SPAWNED": str(spawned)}):
            self.assertEqual(self._batch(["fakesuite.test_spawner"]), 0)

        pid = int(spawned.read_text())
        deadline = time.monotonic() + 10
        while Path(f"/proc/{pid}").exists() and (
            Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
        ):
            self.assertLess(time.monotonic(), deadline, "spawned process survived")
            time.sleep(0.05)

    def test_checkout_residue_fails_the_post_batch_clean_check(self) -> None:
        result = self._batch(["fakesuite.test_residue", "fakesuite.test_green"])

        self.assertEqual(result, 1)
        header, residue, green, after = self._ledger()
        self.assertTrue(header["clean_tree"])
        self.assertEqual((residue["exit"], green["exit"]), (0, 0))
        self.assertEqual(
            after,
            {
                "clean_tree_after": False,
                "dirty_paths_after": ["?? left-behind.txt"],
                "slot_wait_seconds": None,
            },
        )

    def test_residue_from_a_finished_module_does_not_block_the_resume(self) -> None:
        modules = ["fakesuite.test_residue", "fakesuite.test_slow"]
        controller = self._start_controller(modules, timeout=60.0, sleep="60")
        controller.send_signal(signal.SIGINT)
        controller.wait(timeout=30)

        result = self._batch(modules)

        self.assertEqual(result, 1)
        self.assertEqual(
            self._runs(),
            ["fakesuite.test_residue", "fakesuite.test_slow", "fakesuite.test_slow"],
        )
        self.assertEqual(
            self._ledger()[-1],
            {
                "clean_tree_after": False,
                "dirty_paths_after": ["?? left-behind.txt"],
                "slot_wait_seconds": None,
            },
        )

    def test_resume_refuses_a_batch_bound_to_other_source(self) -> None:
        self.assertEqual(self._batch(["fakesuite.test_green"]), 0)
        before = (self.root / "ledger.jsonl").read_bytes()
        green = self.checkout / "fakesuite" / "test_green.py"
        green.write_text("import unittest\n")
        changes = {
            "tracked checkout changes": lambda: None,
            "commit": lambda: _git(self.checkout, "commit", "-q", "-am", "changed"),
        }
        for refusal, change in changes.items():
            with self.subTest(refusal):
                change()

                with self.assertRaisesRegex(
                    testing_operation.ControlSurfaceError, refusal
                ):
                    self._batch(["fakesuite.test_green"])

                self.assertEqual((self.root / "ledger.jsonl").read_bytes(), before)
                self.assertEqual(self._runs(), ["fakesuite.test_green"])

    def test_resume_drops_an_interrupted_last_line_before_appending(self) -> None:
        modules = ["fakesuite.test_green", "fakesuite.test_slow"]
        controller = self._start_controller(modules, timeout=60.0, sleep="60")
        controller.send_signal(signal.SIGINT)
        controller.wait(timeout=30)
        with (self.root / "ledger.jsonl").open("a") as stream:
            stream.write('{"module": "fakesuite.te')

        self.assertEqual(self._batch(modules), 0)

        self.assertEqual(
            [line.get("module") for line in self._ledger()[1:]], [*modules, None]
        )

    def test_refused_resume_leaves_an_interrupted_last_line_in_place(self) -> None:
        self.assertEqual(self._batch(["fakesuite.test_green"]), 0)
        ledger = self.root / "ledger.jsonl"
        with ledger.open("a") as stream:
            stream.write('{"module": "fakesuite.te')
        before = ledger.read_bytes()

        with self.assertRaisesRegex(testing_operation.ControlSurfaceError, "selection"):
            self._batch(["fakesuite.test_red"])

        self.assertEqual(ledger.read_bytes(), before)

    def test_batch_refuses_a_root_another_batch_process_holds(self) -> None:
        self.root.mkdir()
        with (self.root / "ledger.lock").open("a") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX)

            with self.assertRaisesRegex(
                testing_operation.ControlSurfaceError, "still running"
            ):
                self._batch(["fakesuite.test_green"])

        self.assertFalse((self.root / "ledger.jsonl").exists())
        self.assertEqual(self._runs(), [])


_LADDER = "fakesuite.test_ladder"
_REPRESENTATIVE = f"{_LADDER}.Ladder.test_representative"
_LADDER_IDS = [
    f"{_LADDER}.{cls}.{method}"
    for cls in ("Ladder", "LadderChild")
    for method in ("test_representative", "test_other_green", "test_other_red")
]


class ParallelBatchTests(_FakeCheckoutTestCase):
    def test_concurrent_slots_run_modules_together_and_record_the_experiment(
        self,
    ) -> None:
        modules = [
            "fakesuite.test_green",
            "fakesuite.test_meet_a",
            "fakesuite.test_meet_b",
        ]

        result = self._batch(modules, concurrency=2, parallel_experiment=True)

        self.assertEqual(result, 0)
        header, *finishes, after = self._ledger()
        self.assertEqual(
            {
                key: header[key]
                for key in (
                    "selection",
                    "concurrency",
                    "parallel_experiment",
                    "serial_only",
                    "supplemental_only",
                )
            },
            {
                "selection": modules,
                "concurrency": 2,
                "parallel_experiment": True,
                "serial_only": [],
                "supplemental_only": True,
            },
        )
        self.assertEqual(
            sorted((line["module"], line["exit"]) for line in finishes),
            [(module, 0) for module in modules],
        )
        self.assertTrue(after["clean_tree_after"])

    def test_an_ordinary_concurrent_batch_needs_no_basis_and_is_not_supplemental(
        self,
    ) -> None:
        result = self._batch(
            ["fakesuite.test_meet_a", "fakesuite.test_meet_b"], concurrency=2
        )

        self.assertEqual(result, 0)
        header = self._ledger()[0]
        self.assertEqual((header["concurrency"], header["serial_only"]), (2, []))
        for absent in ("parallel_experiment", "supplemental_only", "parallel_basis"):
            self.assertNotIn(absent, header)

    def test_serial_only_modules_run_alone_after_the_concurrent_ones(self) -> None:
        with patch.dict(os.environ, {"BIO_NEB_BATCH_TEST_MEET_SECONDS": "1"}):
            result = self._batch(
                ["fakesuite.test_meet_a", "fakesuite.test_meet_b"],
                concurrency=2,
                serial_only=["fakesuite.test_meet_a", "fakesuite.test_other"],
            )

        self.assertEqual(result, 1)
        header, meet_b, meet_a, _ = self._ledger()
        self.assertEqual(header["serial_only"], ["fakesuite.test_meet_a"])
        self.assertEqual(
            [(line["module"], line["exit"]) for line in (meet_b, meet_a)],
            [("fakesuite.test_meet_b", 1), ("fakesuite.test_meet_a", 0)],
        )

    def test_an_interrupt_stops_every_slot_and_the_resume_reruns_them(self) -> None:
        modules = ["fakesuite.test_meet_a", "fakesuite.test_slow"]
        with patch.dict(os.environ, {"BIO_NEB_BATCH_TEST_MEET_SECONDS": "60"}):
            controller = self._start_controller(
                modules, timeout=120.0, sleep="60", concurrency=2
            )
            deadline = time.monotonic() + 30
            while not (self.meeting / "fakesuite.test_meet_a").exists():
                self.assertLess(time.monotonic(), deadline, "meet_a never started")
                time.sleep(0.05)
            controller.send_signal(signal.SIGINT)
            controller.wait(timeout=60)

        self.assertEqual(len(self._ledger()), 1)
        with (self.root / "ledger.lock").open("a") as lock:
            # No module child still holds the root's lock.
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        with patch.dict(os.environ, {"BIO_NEB_BATCH_TEST_MEET_SECONDS": "0.2"}):
            result = self._batch(modules, timeout=120.0, concurrency=2)

        # Both interrupted modules ran again and now have finish lines; meet_a's partner
        # is not selected, so its own result on the resume is irrelevant here.
        self.assertEqual(sorted(self._runs()), sorted(modules * 2))
        self.assertEqual(
            sorted(line["module"] for line in self._ledger()[1:-1]), sorted(modules)
        )
        self.assertNotEqual(result, 130)

    def test_a_laddered_module_expands_in_its_own_slot(self) -> None:
        with patch.dict(os.environ, {"BIO_NEB_BATCH_TEST_RED_REPRESENTATIVE": "1"}):
            result = self._batch(
                [_LADDER, "fakesuite.test_green"],
                concurrency=2,
                representatives={_LADDER: [_REPRESENTATIVE]},
                base_commit=_git(self.checkout, "rev-parse", "HEAD"),
            )

        self.assertEqual(result, 1)
        ladder = [line for line in self._ledger()[1:] if line.get("module") == _LADDER]
        self.assertEqual(
            [line["phase"] for line in ladder], ["representative", "expansion"]
        )
        self.assertEqual(ladder[1]["tests_run"], 5)


class ShardBatchTests(_FakeCheckoutTestCase):
    """Ticket 293: a heavy module runs as shards of its tests, one process each."""

    _FIRST = [f"{_LADDER}.Ladder.test_other_green", f"{_LADDER}.Ladder.test_other_red"]

    def test_each_shard_runs_in_its_own_process_and_the_last_runs_the_rest(
        self,
    ) -> None:
        result = self._batch(
            [_LADDER, "fakesuite.test_green"], shards={_LADDER: [self._FIRST]}
        )

        self.assertEqual(result, 1)
        header, *finishes, _ = self._ledger()
        self.assertEqual(header["schema"], "bio-neb.testing-batch-ledger/5")
        self.assertEqual(header["shards"], {_LADDER: [self._FIRST]})
        lines = {(line["module"], line.get("shard")): line for line in finishes}
        self.assertEqual(
            sorted(lines, key=str),
            sorted(
                [("fakesuite.test_green", None), (_LADDER, 1), (_LADDER, 2)], key=str
            ),
        )
        first, rest = lines[(_LADDER, 1)], lines[(_LADDER, 2)]
        self.assertEqual(
            [
                (line["shards"], line["tests_run"], line["module_tests"])
                for line in (first, rest)
            ],
            [(2, 2, 6), (2, 4, 6)],
        )
        self.assertEqual(first["failed"], [f"{_LADDER}.Ladder.test_other_red"])
        self.assertEqual(rest["failed"], [])
        self.assertNotEqual(first["log"], rest["log"])
        self.assertEqual(
            sorted(self.executed.read_text().splitlines()), sorted(_LADDER_IDS)
        )
        self.assertEqual(self._runs().count(_LADDER), 2, "one process per shard")

    def test_a_resume_reruns_only_the_shard_without_a_finish_line(self) -> None:
        self._batch([_LADDER], shards={_LADDER: [self._FIRST]})
        header, first = (self.root / "ledger.jsonl").read_text().splitlines()[:2]
        # The controller stopped after the first shard's finish line.
        (self.root / "ledger.jsonl").write_text(f"{header}\n{first}\n")
        self.runs_log.unlink()
        self.executed.unlink()

        self._batch([_LADDER], shards={_LADDER: [self._FIRST]})

        self.assertEqual(self._runs(), [_LADDER])
        self.assertEqual(
            sorted(self.executed.read_text().splitlines()),
            sorted(set(_LADDER_IDS) - set(self._FIRST)),
        )
        _, kept, rerun, _ = self._ledger()
        self.assertEqual((kept["shard"], rerun["shard"]), (1, 2))
        with self.assertRaisesRegex(testing_operation.ControlSurfaceError, "shards"):
            self._batch([_LADDER], shards={_LADDER: [self._FIRST[:1]]})

    def test_a_shard_limit_stops_only_its_shard(self) -> None:
        hanging = [
            f"{_LADDER}.Ladder.test_representative",
            f"{_LADDER}.LadderChild.test_representative",
        ]
        with patch.dict(os.environ, {"BIO_NEB_BATCH_TEST_HANG_REPRESENTATIVE": "1"}):
            self._batch(
                [_LADDER],
                shards={_LADDER: [hanging]},
                module_timeouts={f"{_LADDER}#1": 1.0},
            )

        header, first, rest, _ = self._ledger()
        self.assertEqual(header["module_timeout_seconds"], {f"{_LADDER}#1": 1.0})
        self.assertEqual((first["shard"], first["exit"]), (1, "timeout"))
        self.assertEqual((rest["shard"], rest["exit"]), (2, 1))
        self.assertEqual(rest["failed"], [f"{_LADDER}.Ladder.test_other_red"])
        self.assertEqual(
            testing_operation._compare_ledgers(self.root, self.root)["limits"],
            {_LADDER: {"left": "timeout", "right": "timeout"}},
        )

    def test_shard_layout_leaves_the_comparison_unchanged(self) -> None:
        modules = ["fakesuite.test_broken", _LADDER, "fakesuite.test_red"]
        whole = self.root
        self._batch(modules)
        self.root = whole.with_name("sharded")

        self._batch(
            modules,
            shards={
                # A module that fails to load fails the same way in every shard.
                "fakesuite.test_broken": [["fakesuite.test_broken.Broken.test_x"]],
                _LADDER: [self._FIRST],
                "fakesuite.test_red": [
                    ["fakesuite.test_red.Red.test_errors"],
                    ["fakesuite.test_red.Red.test_fails"],
                ],
            },
        )

        comparison = testing_operation._compare_ledgers(whole, self.root)
        self.assertEqual(comparison["compared_modules"], modules)
        self.assertEqual(
            {
                key: comparison[key]
                for key in ("added", "removed", "reason_changed", "limits")
            },
            {"added": [], "removed": [], "reason_changed": [], "limits": {}},
        )
        nothing_red = self.root.with_name("no-red-ids.txt")
        nothing_red.write_text("")
        self.assertEqual(
            testing_operation._compare_against_ids(self.root, nothing_red)["added"],
            [
                f"{_LADDER}.Ladder.test_other_red",
                "fakesuite.test_red.Red.test_errors",
                "fakesuite.test_red.Red.test_fails",
                "unittest.loader._FailedTest.test_broken",
            ],
        )


class BatchOrderTests(_FakeCheckoutTestCase):
    _MODULES = [
        "fakesuite.test_green",
        "fakesuite.test_ladder",
        "fakesuite.test_mixin",
        "fakesuite.test_red",
        "fakesuite.test_slow",
    ]

    def test_reversed_order_runs_modules_in_reverse_and_binds_the_resume(
        self,
    ) -> None:
        self._batch(self._MODULES, order="reversed")

        header = self._ledger()[0]
        self.assertEqual(
            (header["selection"], header["order"]), (self._MODULES, "reversed")
        )
        self.assertEqual(self._runs(), self._MODULES[::-1])
        with self.assertRaisesRegex(testing_operation.ControlSurfaceError, "order"):
            self._batch(self._MODULES)

    def test_longest_order_starts_the_slowest_estimated_units_first(self) -> None:
        estimates = {
            "fakesuite.test_green": 4.0,
            "fakesuite.test_red": 9.0,
            "fakesuite.test_ladder#1": 6.0,
            "fakesuite.test_ladder#2": 4.0,
        }
        ladder = [f"{_LADDER}.Ladder.test_other_green"]

        self._batch(
            self._MODULES,
            order="longest",
            estimates=estimates,
            shards={_LADDER: [ladder]},
            timing_ledgers=["/previous/ledger.jsonl"],
        )

        header = self._ledger()[0]
        self.assertEqual(
            (header["order"], header["estimated_seconds"], header["timing_ledgers"]),
            ("longest", estimates, ["/previous/ledger.jsonl"]),
        )
        # Ties keep sorted order; units without an estimate start last, sorted.
        self.assertEqual(
            self._runs(),
            [
                "fakesuite.test_red",
                "fakesuite.test_ladder",
                "fakesuite.test_green",
                "fakesuite.test_ladder",
                "fakesuite.test_mixin",
                "fakesuite.test_slow",
            ],
        )
        self.assertEqual(
            [(line["module"], line.get("shard")) for line in self._ledger()[1:-1]],
            [
                ("fakesuite.test_red", None),
                (_LADDER, 1),
                ("fakesuite.test_green", None),
                (_LADDER, 2),
                ("fakesuite.test_mixin", None),
                ("fakesuite.test_slow", None),
            ],
        )

    def test_a_seeded_shuffle_reorders_modules_and_methods_reproducibly(self) -> None:
        orders = []
        for attempt in ("first", "second"):
            self.root = self.root.with_name(f"shuffle-{attempt}")
            self.runs_log.unlink(missing_ok=True)
            self.executed.unlink(missing_ok=True)

            self._batch(self._MODULES, order="shuffle:7")

            self.assertEqual(self._ledger()[0]["order"], "shuffle:7")
            orders.append((self._runs(), self.executed.read_text().splitlines()))

        self.assertEqual(orders[0], orders[1])
        modules, methods = orders[0]
        self.assertEqual(sorted(modules), self._MODULES)
        self.assertNotEqual(modules, self._MODULES)
        self.assertEqual(sorted(methods), sorted(_LADDER_IDS))
        self.assertNotEqual(methods, _LADDER_IDS)
        classes = [test_id.split(".")[2] for test_id in methods]
        self.assertEqual(
            len(
                [
                    1
                    for left, right in zip(classes, classes[1:], strict=False)
                    if left != right
                ]
            ),
            1,
            "each class's methods stay together",
        )

    def test_longest_first_with_a_method_shuffle_keeps_module_order(self) -> None:
        # The checkpoint's order: modules start slowest first and each module's
        # methods run in a seeded order.
        order = "longest+shuffle:7"
        estimates = {"fakesuite.test_ladder": 9.0, "fakesuite.test_green": 4.0}
        executed = []
        for attempt in ("first", "second"):
            self.root = self.root.with_name(f"longest-shuffle-{attempt}")
            self.runs_log.unlink(missing_ok=True)
            self.executed.unlink(missing_ok=True)

            self._batch(self._MODULES, order=order, estimates=estimates)

            self.assertEqual(self._ledger()[0]["order"], order)
            self.assertEqual(
                self._runs(),
                [
                    "fakesuite.test_ladder",
                    "fakesuite.test_green",
                    "fakesuite.test_mixin",
                    "fakesuite.test_red",
                    "fakesuite.test_slow",
                ],
            )
            executed.append(self.executed.read_text().splitlines())

        self.assertEqual(executed[0], executed[1])
        unshuffled = [
            f"{_LADDER}.{cls}.{method}"
            for cls in ("Ladder", "LadderChild")
            for method in ("test_other_green", "test_other_red", "test_representative")
        ]
        self.assertEqual(sorted(executed[0]), sorted(unshuffled))
        self.assertNotEqual(executed[0], unshuffled)


class BatchIsolationTests(_FakeCheckoutTestCase):
    def test_a_module_runs_in_its_own_run_sandbox(self) -> None:
        store = self.root.parent / "store"
        with patch.dict(
            os.environ, {"BIO_NEB_PREREQUISITE_STORE": "/controller/store"}
        ):
            self.assertEqual(self._batch(["fakesuite.test_tmp_writer"], store=store), 0)

        environment = json.loads(self.module_environment.read_text())
        sandbox = Path(environment["tmpdir"]).parent
        self.assertEqual(sandbox.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(sandbox.name.startswith("bio-neb-batch-"))
        self.assertFalse(sandbox.exists(), "the sandbox is removed after the module")
        private_roots = [
            environment[name]
            for name in (
                "BIO_NEB_LOCAL_STATE",
                "BIO_NEB_CONTROLLER_HOME",
                "BIO_NEB_TESTING_RUN_ROOTS",
            )
        ]
        self.assertEqual([Path(root).parent for root in private_roots], [sandbox] * 3)
        self.assertEqual(len(set(private_roots)), 3)
        self.assertEqual(environment["GIT_OPTIONAL_LOCKS"], "0")
        self.assertEqual(environment["fsync"], "no-op")
        self.assertEqual(environment["BIO_NEB_PREREQUISITE_STORE"], str(store))
        self.assertEqual(
            environment["PYTHONPYCACHEPREFIX"], str(self.root / "python-cache")
        )
        finish = self._ledger()[1]
        self.assertEqual(finish["tmp_residue_entries"], 1)
        self.assertNotIn("writes_outside_private", finish)

    def test_without_a_store_a_module_sees_none_whatever_the_controller_has(
        self,
    ) -> None:
        with patch.dict(
            os.environ, {"BIO_NEB_PREREQUISITE_STORE": "/controller/store"}
        ):
            self.assertEqual(self._batch(["fakesuite.test_tmp_writer"]), 0)

        environment = json.loads(self.module_environment.read_text())
        self.assertIsNone(environment["BIO_NEB_PREREQUISITE_STORE"])

    def test_a_direct_unittest_run_keeps_the_real_fsync(self) -> None:
        # A developer's shell: no search path the runner set, whether or not this
        # test itself runs inside a batch.
        shell = {
            name: value for name, value in os.environ.items() if name != "PYTHONPATH"
        }
        shell["TMPDIR"] = str(self.root.parent)  # its temporary file stays in this test
        subprocess.run(
            [sys.executable, "-m", "unittest", "-q", "fakesuite.test_tmp_writer"],
            cwd=self.checkout,
            env=shell,
            check=True,
            capture_output=True,
        )

        environment = json.loads(self.module_environment.read_text())
        self.assertEqual(environment["fsync"], "real")

    def test_write_detection_reports_writes_outside_private_roots(self) -> None:
        # Every watched location is this test's own, so no other host process writes it.
        base = self.root.parent
        watched = {
            name: base / name
            for name in ("home", "state", "controller", "testing-runs", "tmp")
        }
        for directory in watched.values():
            directory.mkdir()
        fixed = watched["tmp"] / "bio-neb-fixed"
        with (
            patch.dict(
                os.environ,
                {
                    "BIO_NEB_BATCH_TEST_FIXED": str(fixed),
                    "HOME": str(watched["home"]),
                    "BIO_NEB_LOCAL_STATE": str(watched["state"]),
                    "BIO_NEB_CONTROLLER_HOME": str(watched["controller"]),
                    "BIO_NEB_TESTING_RUN_ROOTS": str(watched["testing-runs"]),
                },
            ),
            patch.object(tempfile, "tempdir", str(watched["tmp"])),
        ):
            result = self._batch(
                [
                    "fakesuite.test_green",
                    "fakesuite.test_residue",
                    "fakesuite.test_tmp_writer",
                ],
                detect_writes=True,
            )

        self.assertEqual(result, 1)
        header, green, residue, writer, _ = self._ledger()
        self.assertTrue(header["detect_writes"])
        self.assertEqual(
            {
                line["module"]: line["writes_outside_private"]
                for line in (green, residue, writer)
            },
            {
                "fakesuite.test_green": [],
                "fakesuite.test_residue": [
                    f"created {self.checkout / 'left-behind.txt'}"
                ],
                "fakesuite.test_tmp_writer": [f"created {fixed}"],
            },
        )


class BatchIsolationContractTests(_FakeCheckoutTestCase):
    """ADR 0041: a child's writes to HOME and local state stay in its Run sandbox, and a
    write to the shared checkout fails the batch."""

    def test_a_child_cannot_write_the_real_home_or_local_state_and_a_checkout_write_fails(
        self,
    ) -> None:
        base = self.root.parent
        real = {"home": base / "real-home", "local_state": base / "real-state"}
        for directory in real.values():
            directory.mkdir()
        (real["home"] / ".gitconfig").write_text("[user]\n\tname = real\n")
        with patch.dict(
            os.environ,
            {
                "HOME": str(real["home"]),
                locations.LOCAL_STATE_OVERRIDE: str(real["local_state"]),
            },
        ):
            os.environ.pop("GIT_CONFIG_GLOBAL", None)
            result = self._batch(["fakesuite.test_escape", "fakesuite.test_green"])

        self.assertEqual(result, 1)
        header, escape, green, after = self._ledger()
        self.assertEqual((escape["exit"], green["exit"]), (0, 0))
        self.assertEqual(
            {name: sorted(os.listdir(path)) for name, path in real.items()},
            {"home": [".gitconfig"], "local_state": []},
        )
        landed = json.loads(self.module_environment.read_text())
        sandbox = Path(tempfile.gettempdir()).resolve()
        for name in ("home", "local_state"):
            with self.subTest(name):
                self.assertTrue(Path(landed[name]).is_relative_to(sandbox))
                self.assertFalse(
                    Path(landed[name]).exists(), "removed with the sandbox"
                )
        self.assertEqual(
            after,
            {
                "clean_tree_after": False,
                "dirty_paths_after": ["?? escaped-checkout.txt"],
                "slot_wait_seconds": None,
            },
        )


class RepresentativeLadderTests(_FakeCheckoutTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.base = _git(self.checkout, "rev-parse", "HEAD")

    def _ladder_batch(
        self,
        modules: list[str],
        representatives: dict[str, list[str]] | None,
        *,
        timeout: float = 60.0,
    ) -> tuple[int, list[dict]]:
        with patch("sys.stdout") as stdout:
            result = testing_operation._run_batch(
                modules,
                self.root,
                timeout=timeout,
                checkout=self.checkout,
                representatives=representatives,
                base_commit=None if representatives is None else self.base,
            )
        written = "".join(call.args[0] for call in stdout.write.call_args_list)
        return result, [json.loads(line) for line in written.splitlines()]

    def _executed(self) -> list[str]:
        if not self.executed.exists():
            return []
        return self.executed.read_text().splitlines()

    def test_green_representative_runs_only_its_ids_and_is_labelled_supplemental(
        self,
    ) -> None:
        result, printed = self._ladder_batch(
            [_LADDER, "fakesuite.test_green"], {_LADDER: [_REPRESENTATIVE]}
        )

        self.assertEqual(result, 0)
        header, green, ladder, after = self._ledger()
        self.assertEqual(
            {
                key: header[key]
                for key in ("rung", "supplemental_only", "representatives")
            },
            {
                "rung": "representative",
                "supplemental_only": True,
                "representatives": {_LADDER: [_REPRESENTATIVE]},
            },
        )
        self.assertEqual(header["base_commit"], self.base)
        self.assertEqual(
            set(ladder),
            {
                "module",
                "phase",
                "exit",
                "seconds",
                "failed",
                "errored",
                "reasons",
                "log",
                "tests_run",
                "module_tests",
                "test_seconds",
                "peak_rss_kib",
                "descendant_peak_rss_kib",
            },
        )
        self.assertEqual(
            {key: ladder[key] for key in ("phase", "exit", "failed", "errored")},
            {"phase": "representative", "exit": 0, "failed": [], "errored": []},
        )
        self.assertEqual((ladder["tests_run"], ladder["module_tests"]), (1, 6))
        self.assertEqual((green["depth"], green["exit"]), ("full", 0))
        self.assertEqual(self._executed(), [_REPRESENTATIVE])
        self.assertTrue(after["clean_tree_after"])
        self.assertEqual(
            (printed[-1]["rung"], printed[-1]["supplemental_only"]),
            ("representative", True),
        )

    def test_red_representative_expands_once_and_keeps_the_original_red(self) -> None:
        with patch.dict(os.environ, {"BIO_NEB_BATCH_TEST_RED_REPRESENTATIVE": "1"}):
            result, _ = self._ladder_batch([_LADDER], {_LADDER: [_REPRESENTATIVE]})

        self.assertEqual(result, 1)
        _, representative, expansion, _ = self._ledger()
        self.assertEqual(
            {
                key: representative[key]
                for key in ("phase", "exit", "failed", "errored", "tests_run", "log")
            },
            {
                "phase": "representative",
                "exit": 1,
                "failed": [_REPRESENTATIVE],
                "errored": [],
                "tests_run": 1,
                "log": "logs/fakesuite.test_ladder.log",
            },
        )
        self.assertEqual(
            {
                key: expansion[key]
                for key in (
                    "module",
                    "phase",
                    "exit",
                    "failed",
                    "errored",
                    "tests_run",
                    "log",
                )
            },
            {
                "module": _LADDER,
                "phase": "expansion",
                "exit": 1,
                "failed": [
                    f"{_LADDER}.Ladder.test_other_red",
                    f"{_LADDER}.LadderChild.test_representative",
                ],
                "errored": [],
                "tests_run": 5,
                "log": "logs/fakesuite.test_ladder.expansion.log",
            },
        )
        executed = self._executed()
        self.assertEqual(executed[0], _REPRESENTATIVE)
        self.assertEqual(sorted(executed), sorted(_LADDER_IDS))
        self.assertIn(
            "forced red representative",
            (self.root / representative["log"]).read_text(),
        )

    def test_an_interrupted_expansion_resumes_without_rerunning_the_red(self) -> None:
        representatives = {_LADDER: [_REPRESENTATIVE]}
        with patch.dict(os.environ, {"BIO_NEB_BATCH_TEST_RED_REPRESENTATIVE": "1"}):
            self._ladder_batch([_LADDER], representatives)
        ledger = self.root / "ledger.jsonl"
        header, representative = ledger.read_text().splitlines(keepends=True)[:2]
        # The controller stopped after the representative line, during the expansion.
        ledger.write_text(header + representative)
        self.executed.unlink()

        result, printed = self._ladder_batch([_LADDER], representatives)

        self.assertEqual(result, 1)
        self.assertEqual(
            ledger.read_text().splitlines(keepends=True)[1], representative
        )
        self.assertEqual(
            sorted(self._executed()),
            sorted(set(_LADDER_IDS) - {_REPRESENTATIVE}),
        )
        self.assertEqual(
            (printed[0]["phase"], printed[0]["log"]),
            ("expansion", "logs/fakesuite.test_ladder.expansion.2.log"),
        )

    def test_a_representative_timeout_does_not_expand(self) -> None:
        with patch.dict(os.environ, {"BIO_NEB_BATCH_TEST_HANG_REPRESENTATIVE": "1"}):
            result, _ = self._ladder_batch(
                [_LADDER], {_LADDER: [_REPRESENTATIVE]}, timeout=1.0
            )

        self.assertEqual(result, 1)
        _, representative, after = self._ledger()
        self.assertEqual(
            (representative["phase"], representative["exit"]),
            ("representative", "timeout"),
        )
        self.assertIn("clean_tree_after", after)
        self.assertEqual(self._executed(), [_REPRESENTATIVE])

    def test_a_listed_id_the_module_does_not_load_is_errored_and_expands(
        self,
    ) -> None:
        mixin = "fakesuite.test_mixin.Checks.test_mixed"

        result, _ = self._ladder_batch(
            ["fakesuite.test_mixin"], {"fakesuite.test_mixin": [mixin]}
        )

        self.assertEqual(result, 1)
        _, representative, expansion, _ = self._ledger()
        self.assertEqual(
            {
                key: representative[key]
                for key in ("exit", "failed", "errored", "tests_run", "module_tests")
            },
            {
                "exit": 1,
                "failed": [],
                "errored": [mixin],
                "tests_run": 0,
                "module_tests": 1,
            },
        )
        self.assertEqual(
            (expansion["exit"], expansion["failed"], expansion["tests_run"]),
            (0, [], 1),
        )

    def test_unlisted_modules_in_a_representative_batch_run_whole(self) -> None:
        result, _ = self._ladder_batch([_LADDER], {})

        self.assertEqual(result, 1)
        _, ladder, _ = self._ledger()
        self.assertEqual(
            (ladder["depth"], ladder["failed"]),
            ("full", [f"{_LADDER}.Ladder.test_other_red"]),
        )
        self.assertNotIn("phase", ladder)
        self.assertEqual(sorted(self._executed()), sorted(_LADDER_IDS))

    def test_resume_is_bound_to_the_rung_and_representative_ids(self) -> None:
        representatives = {_LADDER: [_REPRESENTATIVE]}
        self.assertEqual(self._ladder_batch([_LADDER], representatives)[0], 0)
        before = (self.root / "ledger.jsonl").read_bytes()
        refusals = {
            "representatives": {
                _LADDER: [_REPRESENTATIVE, f"{_LADDER}.Ladder.test_other_green"]
            },
            "rung": None,
        }
        for refusal, changed in refusals.items():
            with self.subTest(refusal):
                with self.assertRaisesRegex(
                    testing_operation.ControlSurfaceError, refusal
                ):
                    self._ladder_batch([_LADDER], changed)

                self.assertEqual((self.root / "ledger.jsonl").read_bytes(), before)

        result, printed = self._ladder_batch([_LADDER], representatives)

        self.assertEqual(result, 0)
        self.assertEqual(printed[0], {"skipped_finished": _LADDER})
        self.assertEqual(self._executed(), [_REPRESENTATIVE])
        self.assertEqual((self.root / "ledger.jsonl").read_bytes(), before)

    def test_changed_test_methods_cover_commits_working_edits_and_inheritors(
        self,
    ) -> None:
        ladder = self.checkout / "fakesuite" / "test_ladder.py"
        source = ladder.read_text()
        ladder.write_text(
            source.replace(
                "    def test_other_green(self):\n        pass\n",
                "    def test_other_green(self):\n        self.assertTrue(True)\n",
            )
        )
        _git(self.checkout, "commit", "-q", "-am", "edit an inherited test")
        ladder.write_text(
            ladder.read_text()
            .replace(
                "    def test_other_red(self):\n        pass\n",
                "    def test_other_red(self):\n        self.assertTrue(True)\n",
            )
            .replace("    return None\n", "    return 1\n")
        )
        mixin = self.checkout / "fakesuite" / "test_mixin.py"
        mixin.write_text(
            mixin.read_text().replace("        pass\n", "        assert True\n", 1)
        )

        changed = testing_operation._changed_test_methods(
            self.base,
            [
                "fakesuite/test_ladder.py",
                "fakesuite/test_green.py",
                "fakesuite/test_mixin.py",
            ],
            self.checkout,
        )

        self.assertEqual(
            changed,
            {
                f"{_LADDER}.Ladder.test_other_green",
                f"{_LADDER}.LadderChild.test_other_green",
                f"{_LADDER}.LadderChild.test_other_red",
                "fakesuite.test_mixin.Mixed.test_mixed",
            },
        )


class RepresentativeSelectionTests(unittest.TestCase):
    def test_representative_lists_refuse_stale_unsorted_or_foreign_ids(self) -> None:
        method_ids = {
            "tests.test_a.T.test_one",
            "tests.test_a.T.test_two",
            "tests.test_b.T.test_one",
        }
        valid = {
            "tests.test_a": {
                "description": "one flow",
                "test_ids": ["tests.test_a.T.test_one", "tests.test_a.T.test_two"],
            }
        }

        self.assertEqual(
            testing_operation._representative_selections(valid, method_ids),
            {"tests.test_a": ["tests.test_a.T.test_one", "tests.test_a.T.test_two"]},
        )
        cases = {
            "stale": (["tests.test_a.T.test_gone"], "stale test_ids"),
            "unsorted": (
                ["tests.test_a.T.test_two", "tests.test_a.T.test_one"],
                "sorted unique",
            ),
            "duplicate": (
                ["tests.test_a.T.test_one", "tests.test_a.T.test_one"],
                "sorted unique",
            ),
            "empty": ([], "sorted unique"),
            "another module": (["tests.test_b.T.test_one"], "outside its module"),
        }
        for name, (test_ids, refusal) in cases.items():
            with self.subTest(name):
                selections = {
                    "tests.test_a": {"description": "one flow", "test_ids": test_ids}
                }
                with self.assertRaisesRegex(
                    testing_operation.ControlSurfaceError, refusal
                ):
                    testing_operation._representative_selections(selections, method_ids)
        with self.assertRaisesRegex(
            testing_operation.ControlSurfaceError, "must be an object"
        ):
            testing_operation._representative_selections([], method_ids)

    def test_regression_known_red_members_are_named_members_with_a_reason(
        self,
    ) -> None:
        """#272 (LEFT-19): a red member is declared with its owner, never unexplained."""
        method_ids = {"tests.test_a.T.test_one", "tests.test_a.T.test_two"}

        def selections(known_red):
            return {
                "known": {
                    "description": "two reproducers",
                    "test_ids": ["tests.test_a.T.test_one", "tests.test_a.T.test_two"],
                    "known_red": known_red,
                }
            }

        valid = {"tests.test_a.T.test_two": "RED-5, owner #276"}
        testing_operation._regression_selections(selections(valid), method_ids)
        cases = {
            "not a member": ({"tests.test_b.T.test_x": "RED-5"}, "known_red"),
            "blank reason": ({"tests.test_a.T.test_one": " "}, "known_red"),
            "not a mapping": (["tests.test_a.T.test_one"], "known_red"),
        }
        for name, (known_red, refusal) in cases.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(testing_operation.ControlSurfaceError, refusal),
            ):
                testing_operation._regression_selections(
                    selections(known_red), method_ids
                )

    def test_laddered_modules_declare_current_representatives(self) -> None:
        """The #201 pilots and the slow modules #278 laddered (COST-14)."""
        laddered = [
            "tests.test_cli",
            "tests.test_default_agentic_loop",
            "tests.test_execution_application_integration",
            "tests.test_execution_planning_agent",
            "tests.test_finalization",
            "tests.test_grounded_report",
            "tests.test_h100_stage",
            "tests.test_operation",
            "tests.test_planning_assignment",
            "tests.test_reference_stages",
            "tests.test_run_observation_application",
            "tests.test_sequential_investigation",
            "tests.test_shared_review_responses",
        ]
        data = testing_operation._load_ownership()
        method_ids = testing_operation._test_method_ids(
            {f"{module.replace('.', '/')}.py" for module in laddered}
        )

        selections = testing_operation._representative_selections(
            data["representative_selections"], method_ids
        )

        self.assertEqual(sorted(selections), laddered)


def _write_ledger(
    root: Path,
    commit: str,
    selection: list[str],
    *finishes: dict,
    ladder: dict[str, list[str]] | None = None,
    schema: str = "bio-neb.testing-batch-ledger/3",
    shards: dict[str, list[list[str]]] | None = None,
    shard_schema: str = "bio-neb.testing-batch-ledger/4",
) -> None:
    root.mkdir()
    header = {
        "schema": schema,
        "commit": commit,
        "clean_tree": True,
        "dirty_paths": [],
        "lock_sha256": "0" * 64,
        "interpreter": "/venv/bin/python",
        "selection": selection,
        "rung": "full",
        "timeout_seconds": 3600.0,
    }
    if ladder is not None:
        header.update(
            rung="representative",
            supplemental_only=True,
            base_commit=commit,
            representatives=ladder,
        )
    if shards is not None:
        header.update(
            schema=shard_schema,
            concurrency=1,
            order="sorted",
            detect_writes=False,
            shards=shards,
            estimated_seconds={},
            timing_ledgers=[],
        )
    lines = [header, *finishes]
    (root / "ledger.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )


def _finish(
    module: str, exit_status, failed=(), errored=(), reasons=None, **extra
) -> dict:
    """A finish line; each red ID's reason defaults to one shared assertion line."""
    unknown = exit_status == "timeout"
    return {
        "module": module,
        "exit": exit_status,
        "seconds": 1.0,
        "failed": None if unknown else list(failed),
        "errored": None if unknown else list(errored),
        "reasons": None
        if unknown
        else reasons or dict.fromkeys((*failed, *errored), "AssertionError: red"),
        "log": f"logs/{module}.log",
        **extra,
    }


class LedgerComparisonTests(unittest.TestCase):
    def test_comparison_reports_added_and_removed_red_ids_and_keeps_limits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            _write_ledger(
                left,
                "a" * 40,
                ["tests.a", "tests.b", "tests.c", "tests.d", "tests.f"],
                _finish(
                    "tests.a", 1, failed=["tests.a.T.test_old", "tests.a.T.test_kept"]
                ),
                _finish("tests.b", "timeout"),
                _finish("tests.c", 1, errored=["tests.c.T.test_hidden"]),
                _finish("tests.d", 1, failed=["tests.d.T.test_only_left"]),
                _finish("tests.f", 5),
            )
            _write_ledger(
                right,
                "b" * 40,
                ["tests.a", "tests.b", "tests.c", "tests.e", "tests.f"],
                _finish(
                    "tests.a",
                    1,
                    failed=["tests.a.T.test_kept"],
                    errored=["tests.a.T.test_new"],
                ),
                _finish("tests.b", 0),
                _finish("tests.e", 1, failed=["tests.e.T.test_only_right"]),
                _finish("tests.f", 1, failed=["tests.f.T.test_seen_once"]),
            )

            comparison = testing_operation._compare_ledgers(left, right)

        self.assertEqual(comparison["added"], ["tests.a.T.test_new"])
        self.assertEqual(comparison["removed"], ["tests.a.T.test_old"])
        self.assertEqual(comparison["compared_modules"], ["tests.a"])
        self.assertEqual(
            comparison["limits"],
            {
                "tests.b": {"left": "timeout", "right": "finished"},
                "tests.c": {"left": "finished", "right": "unfinished"},
                "tests.f": {"left": "no_result", "right": "finished"},
            },
        )
        self.assertEqual(comparison["only_left"], ["tests.d"])
        self.assertEqual(comparison["only_right"], ["tests.e"])
        self.assertEqual(
            (comparison["left"]["commit"], comparison["right"]["commit"]),
            ("a" * 40, "b" * 40),
        )
        for side in ("left", "right"):
            with self.subTest(side=side):
                self.assertEqual(comparison[side]["rung"], "full")
                self.assertIn("tests.a", comparison[side]["selection"])


class ReasonComparisonTests(unittest.TestCase):
    def test_same_red_ids_with_a_new_reason_are_reported_as_reason_changed(
        self,
    ) -> None:
        long_preamble = "AssertionError: known failures\n" + "a" * 3000 + "\n"
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            _write_ledger(
                left,
                "a" * 40,
                ["tests.a"],
                _finish(
                    "tests.a",
                    1,
                    failed=[
                        "tests.a.T.test_long_tail",
                        "tests.a.T.test_new_reason",
                        "tests.a.T.test_noisy (case=1)",
                        "tests.a.T.test_truncated",
                    ],
                    reasons={
                        "tests.a.T.test_long_tail": long_preamble + "old mismatch",
                        "tests.a.T.test_new_reason": "AssertionError: 2 != 3",
                        "tests.a.T.test_noisy (case=1)": (
                            "KeyError: 'run 0b6e2c1a-5f4e-4c2d-9a1b-3c4d5e6f7a8b at "
                            "/tmp/bio-neb-batch-x1y2/run sha 9f86d081884c7d659a2feaa0c55a"
                            "d015a3bf4f1b2b0b822cd15d6c15b0f00a08 took 1.25 s'"
                        ),
                        "tests.a.T.test_truncated": (
                            "AssertionError: {'attempt_id': '5eae3f9-1d2c[23 chars]4a5'}"
                        ),
                    },
                ),
            )
            _write_ledger(
                right,
                "b" * 40,
                ["tests.a"],
                _finish(
                    "tests.a",
                    1,
                    failed=[
                        "tests.a.T.test_long_tail",
                        "tests.a.T.test_new_reason",
                        "tests.a.T.test_noisy (case=2)",
                        "tests.a.T.test_truncated",
                    ],
                    reasons={
                        "tests.a.T.test_long_tail": long_preamble + "new regression",
                        "tests.a.T.test_new_reason": (
                            "ValueError: allowlist entry missing"
                        ),
                        "tests.a.T.test_noisy (case=2)": (
                            "KeyError: 'run 7c1d9e2f-0a3b-4c5d-8e9f-a0b1c2d3e4f5 at "
                            "/tmp/bio-neb-batch-q9r8/run sha 2c26b46b68ffc68ff99b453c1d304"
                            "13413422d706483bfa0f98a5e886266e7ae took 3 s'"
                        ),
                        "tests.a.T.test_truncated": (
                            "AssertionError: {'attempt_id': '8ccd1-2d4[19 chars]7'}"
                        ),
                    },
                ),
            )

            comparison = testing_operation._compare_ledgers(left, right)

        self.assertEqual(comparison["added"], [])
        self.assertEqual(comparison["removed"], [])
        self.assertEqual(
            comparison["reasons"], {"left": "recorded", "right": "recorded"}
        )
        self.assertEqual(
            comparison["reason_changed"],
            [
                {
                    "id": "tests.a.T.test_long_tail",
                    "left": [long_preamble + "old mismatch"],
                    "right": [long_preamble + "new regression"],
                },
                {
                    "id": "tests.a.T.test_new_reason",
                    "left": ["AssertionError: <n> != <n>"],
                    "right": ["ValueError: allowlist entry missing"],
                },
            ],
        )

    def test_a_pydantic_truncated_uuid_is_masked_whatever_its_groups_hold(
        self,
    ) -> None:
        """Ticket 234 (from #226): the two checkpoint reasons differ only in the UUID."""
        test_id = (
            "tests.test_planning_hitl_application.PlanningHitlApplicationTests."
            "test_response_command_appends_once_and_is_byte_idempotent"
        )

        def reason(attempt_id: str) -> str:
            return (
                "pydantic_core._pydantic_core.ValidationError: 1 validation error for "
                "AttemptStarted\ninput_set_id\n  Field required [type=missing, "
                f"input_value={{'attempt_id': '{attempt_id}'}}, "
                "input_type=ReadOnlyDict]\n    For further information visit "
                "https://errors.pydantic.dev/2.13/v/missing"
            )

        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            for root, sha, attempt_id in (
                (left, "a" * 40, "59f6a09a-...5d01-beaf-6a32dc77bff5"),
                (right, "b" * 40, "f2ac3ff6-...58b1-9a33-8d13d6f658b6"),
            ):
                _write_ledger(
                    root,
                    sha,
                    ["tests.test_planning_hitl_application"],
                    _finish(
                        "tests.test_planning_hitl_application",
                        1,
                        failed=[test_id],
                        reasons={test_id: reason(attempt_id)},
                    ),
                )

            comparison = testing_operation._compare_ledgers(left, right)

        self.assertEqual(comparison["added"], [])
        self.assertEqual(comparison["reason_changed"], [])

    def test_a_uuid_cut_by_unittest_truncation_is_masked_whatever_its_groups_hold(
        self,
    ) -> None:
        """Errors #3 entry 18875748: two #231 reasons differed only in `adea`."""
        test_id = "tests.test_investigation.InvestigationContractTests.test_observed"

        def reason(head: str, tail: str) -> str:
            return (
                "AssertionError: {'attempt_id': 'x'} != "
                f"{{'attempt_id': '{head}[235 chars]ion'}}\n"
                f"  {{'adapter_identity': 'ncbi_investigation', 'next': '[9 chars]{tail}'}}"
            )

        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            for root, sha, (head, tail) in (
                (
                    left,
                    "a" * 40,
                    ("cad1cb8f-fd24-484c-a40d-3191", "4c-a40d-3191bb06195f"),
                ),
                (
                    right,
                    "b" * 40,
                    ("801ee2e8-adea-4c2a-85c6-9637", "ea-beef-96378c308b7c"),
                ),
            ):
                _write_ledger(
                    root,
                    sha,
                    ["tests.test_investigation"],
                    _finish(
                        "tests.test_investigation",
                        1,
                        failed=[test_id],
                        reasons={test_id: reason(head, tail)},
                    ),
                )

            comparison = testing_operation._compare_ledgers(left, right)

        self.assertEqual(comparison["added"], [])
        self.assertEqual(comparison["reason_changed"], [])

    def test_uuid_fragment_masks_need_unittest_truncation_beside_them(self) -> None:
        """#272 Standards r1: only a UUID head or tail cut by `[N chars]` is noise."""
        for left, right in (
            ("deadbeef-deadlock", "deadbeef-beadlock"),
            ("version abcdefab-cafe", "version abcdefab-feed"),
            ("label cafe-0123456789ab", "label beef-0123456789ab"),
            ("commit befcabbb", "commit f2454275f"),
        ):
            with self.subTest(left=left):
                self.assertNotEqual(
                    testing_operation._reason_key(left),
                    testing_operation._reason_key(right),
                )
        for left, right in (
            (
                "'cad1cb8f-fd24-484c-a40d-3191[235 chars]'",
                "'801ee2e8-adea-4c2a-85c6-9637[235 chars]'",
            ),
            ("'[9 chars]4c-a40d-3191bb06195f'", "'[9 chars]ea-beef-96378c308b7c'"),
            ("'cad1cb8f-[30 chars]'", "'adeadead-[30 chars]'"),
        ):
            with self.subTest(left=left):
                self.assertEqual(
                    testing_operation._reason_key(left),
                    testing_operation._reason_key(right),
                )

    def test_a_new_offender_subtest_is_added_beside_an_unchanged_one(self) -> None:
        """#272 Standards r1: `_each_offender` reds keep their identity in compare,
        while other subtests still fold into their test."""
        test = "tests.test_repository_contract.RepositoryContractTests.test_links"
        one, two = (
            f"{test} (offender='docs/agents/coding.md -> missing-{n}.md')"
            for n in ("one", "two")
        )
        other = "tests.test_cli.CliTests.test_kinds"
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            for root, sha, failed in (
                (left, "a" * 40, [one, f"{other} (kind='a')"]),
                (
                    right,
                    "b" * 40,
                    [one, two, f"{other} (kind='a')", f"{other} (kind='b')"],
                ),
            ):
                _write_ledger(
                    root,
                    sha,
                    ["tests.test_repository_contract"],
                    _finish(
                        "tests.test_repository_contract",
                        1,
                        failed=failed,
                        reasons={
                            test_id: f"AssertionError: {test_id[-12:]}"
                            if "offender" in test_id
                            else "AssertionError: kind"
                            for test_id in failed
                        },
                    ),
                )

            comparison = testing_operation._compare_ledgers(left, right)

        self.assertEqual(comparison["added"], [two])
        self.assertEqual(comparison["removed"], [])
        self.assertEqual(comparison["reason_changed"], [])

    def test_a_ledger_without_recorded_reasons_compares_its_ids_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / "old", Path(directory) / "new"
            red = _finish("tests.a", 1, failed=["tests.a.T.test_red"])
            del red["reasons"]
            _write_ledger(
                old,
                "a" * 40,
                ["tests.a"],
                red,
                schema="bio-neb.testing-batch-ledger/1",
            )
            _write_ledger(
                new,
                "b" * 40,
                ["tests.a"],
                _finish(
                    "tests.a",
                    1,
                    failed=["tests.a.T.test_red", "tests.a.T.test_added"],
                ),
            )

            comparison = testing_operation._compare_ledgers(old, new)

        self.assertEqual(comparison["added"], ["tests.a.T.test_added"])
        self.assertEqual(
            comparison["reasons"], {"left": "unavailable", "right": "recorded"}
        )
        self.assertIsNone(comparison["reason_changed"])

    def test_a_ledger_from_before_test_seconds_compares_its_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / "old", Path(directory) / "new"
            _write_ledger(
                old,
                "a" * 40,
                ["tests.a"],
                _finish("tests.a", 1, failed=["tests.a.T.test_red"]),
                schema="bio-neb.testing-batch-ledger/2",
            )
            _write_ledger(
                new,
                "b" * 40,
                ["tests.a"],
                _finish(
                    "tests.a",
                    1,
                    failed=["tests.a.T.test_red"],
                    reasons={"tests.a.T.test_red": "AssertionError: other"},
                    test_seconds={"tests.a.T.test_red": 0.5},
                    peak_rss_kib=50000,
                    descendant_peak_rss_kib=0,
                ),
            )

            comparison = testing_operation._compare_ledgers(old, new)

        self.assertEqual(
            comparison["reasons"], {"left": "recorded", "right": "recorded"}
        )
        self.assertEqual(
            comparison["reason_changed"],
            [
                {
                    "id": "tests.a.T.test_red",
                    "left": ["AssertionError: red"],
                    "right": ["AssertionError: other"],
                }
            ],
        )

    def test_a_recorded_ledger_refuses_a_red_id_without_its_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "batch"
            _write_ledger(
                root,
                "a" * 40,
                ["tests.a"],
                _finish(
                    "tests.a",
                    1,
                    failed=["tests.a.T.test_red", "tests.a.T.test_unexplained"],
                    reasons={"tests.a.T.test_red": "AssertionError: red"},
                ),
            )

            with self.assertRaisesRegex(
                testing_operation.ControlSurfaceError,
                "tests.a.T.test_unexplained has no recorded reason",
            ):
                testing_operation._compare_ledgers(root, root)


class LadderComparisonTests(unittest.TestCase):
    def test_representative_depth_compares_only_with_the_same_representatives(
        self,
    ) -> None:
        rep_a, rep_c = ["tests.a.T.test_rep"], ["tests.c.T.test_rep"]
        with tempfile.TemporaryDirectory() as directory:
            full = Path(directory) / "full"
            _write_ledger(
                full,
                "a" * 40,
                ["tests.a", "tests.b", "tests.c"],
                _finish("tests.a", 0),
                _finish("tests.b", 0),
                _finish("tests.c", 1, failed=["tests.c.T.test_other"]),
            )
            left, right = Path(directory) / "left", Path(directory) / "right"
            _write_ledger(
                left,
                "b" * 40,
                ["tests.a", "tests.b", "tests.c", "tests.d"],
                _finish("tests.a", 0, phase="representative"),
                _finish("tests.b", 0, depth="full"),
                _finish("tests.c", 1, failed=rep_c, phase="representative"),
                _finish(
                    "tests.c", 1, failed=["tests.c.T.test_other"], phase="expansion"
                ),
                _finish("tests.d", 0, phase="representative"),
                ladder={
                    "tests.a": rep_a,
                    "tests.c": rep_c,
                    "tests.d": ["tests.d.T.test_rep"],
                },
            )
            _write_ledger(
                right,
                "c" * 40,
                ["tests.a", "tests.d"],
                _finish("tests.a", 1, failed=rep_a, phase="representative"),
                _finish(
                    "tests.a", 1, failed=["tests.a.T.test_other"], phase="expansion"
                ),
                _finish("tests.d", 0, phase="representative"),
                ladder={"tests.a": rep_a, "tests.d": ["tests.d.T.test_changed"]},
            )

            against_full = testing_operation._compare_ledgers(full, left)
            between_ladders = testing_operation._compare_ledgers(left, right)

        self.assertEqual(against_full["compared_modules"], ["tests.b", "tests.c"])
        self.assertEqual(against_full["representative_only"], [])
        self.assertEqual(against_full["added"], rep_c)
        self.assertEqual(against_full["removed"], [])
        self.assertEqual(
            against_full["limits"],
            {"tests.a": {"left": "finished", "right": "representative"}},
        )
        self.assertEqual(
            (against_full["left"]["rung"], against_full["right"]["rung"]),
            ("full", "representative"),
        )
        self.assertEqual(
            (
                against_full["left"]["supplemental_only"],
                against_full["right"]["supplemental_only"],
            ),
            (False, True),
        )
        self.assertEqual(between_ladders["compared_modules"], ["tests.a"])
        self.assertEqual(between_ladders["representative_only"], ["tests.a"])
        self.assertEqual(between_ladders["added"], rep_a)
        self.assertEqual(
            between_ladders["limits"],
            {"tests.d": {"left": "representative", "right": "representative"}},
        )

    def test_a_ledger_refuses_an_expansion_without_a_red_representative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ladder"
            _write_ledger(
                root,
                "a" * 40,
                ["tests.a"],
                _finish("tests.a", 0, phase="representative"),
                _finish("tests.a", 1, failed=["tests.a.T.test_x"], phase="expansion"),
                ladder={"tests.a": ["tests.a.T.test_rep"]},
            )

            with self.assertRaisesRegex(
                testing_operation.ControlSurfaceError, "inconsistent batch ledger"
            ):
                testing_operation._compare_ledgers(root, root)


class ShardComparisonTests(unittest.TestCase):
    """Ticket 293: a sharded module compares as one module once every shard finished."""

    _SHARDS = {"tests.a": [["tests.a.T.test_first"]]}

    def test_a_module_compares_only_once_every_shard_finished(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            _write_ledger(
                left,
                "a" * 40,
                ["tests.a"],
                _finish("tests.a", 1, failed=["tests.a.T.test_rest"]),
            )
            _write_ledger(
                right,
                "b" * 40,
                ["tests.a"],
                _finish("tests.a", 0, shard=1, shards=2),
                shards=self._SHARDS,
            )

            unfinished = testing_operation._compare_ledgers(left, right)
            with (right / "ledger.jsonl").open("a") as stream:
                rest = _finish("tests.a", 1, ["tests.a.T.test_rest"], shard=2, shards=2)
                stream.write(json.dumps(rest) + "\n")
            finished = testing_operation._compare_ledgers(left, right)

        self.assertEqual(unfinished["compared_modules"], [])
        self.assertEqual(
            unfinished["limits"],
            {"tests.a": {"left": "finished", "right": "unfinished"}},
        )
        self.assertEqual(
            {key: finished[key] for key in ("compared_modules", "added", "removed")},
            {"compared_modules": ["tests.a"], "added": [], "removed": []},
        )
        self.assertEqual(finished["reason_changed"], [])

    def test_a_ledger_refuses_a_shard_its_header_did_not_plan(self) -> None:
        cases = {
            "unplanned module": _finish("tests.b", 0, shard=1, shards=2),
            "shard beyond the plan": _finish("tests.a", 0, shard=3, shards=3),
            "count that differs from the plan": _finish(
                "tests.a", 0, shard=1, shards=3
            ),
            "second line for one shard": None,
        }
        for name, line in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "root"
                lines = (
                    [line]
                    if line is not None
                    else [_finish("tests.a", 0, shard=1, shards=2)] * 2
                )
                _write_ledger(
                    root, "a" * 40, ["tests.a", "tests.b"], *lines, shards=self._SHARDS
                )

                with self.assertRaisesRegex(
                    testing_operation.ControlSurfaceError, "inconsistent batch ledger"
                ):
                    testing_operation._compare_ledgers(root, root)


def _schedule(**overrides) -> dict:
    """The scheduling keywords `batch` passes to `_run_batch`, defaults first; each
    case's host queue is its own, so a default batch is the sole runner (#455)."""
    return {
        "concurrency": min(8, os.cpu_count() or 1),
        "order": "sorted",
        "detect_writes": False,
        "parallel_experiment": False,
        "serial_only": testing_operation._load_ownership()["serial_only"],
        **overrides,
    }


class DerivedTimeoutTests(unittest.TestCase):
    """Ticket 234: each module's default limit comes from its last recorded seconds."""

    _MODULES = ["tests.test_lifecycle", "tests.test_testing_batch"]

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.runs = Path(directory.name).resolve() / "testing-runs"
        self.runs.mkdir()
        environment = patch.dict(
            os.environ, {locations.TESTING_RUN_ROOTS_OVERRIDE: str(self.runs)}
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.root = self.runs / "234-batch"

    def _ledger(self, name: str, age: int, *finishes: dict) -> Path:
        """A previous batch under the run roots, `age` seconds old."""
        root = self.runs / name
        _write_ledger(root, "a" * 40, self._MODULES, *finishes)
        ledger = root / "ledger.jsonl"
        stamp = time.time() - age
        os.utime(ledger, (stamp, stamp))
        return ledger

    def _timeouts(self, *arguments: str) -> tuple[float, dict, dict]:
        """The batch's fallback limit, its per-module limits and the reported basis."""
        output = io.StringIO()
        with (
            patch.object(testing_operation, "_run_batch", return_value=0) as run,
            redirect_stdout(output),
        ):
            result = testing_operation.main(
                ["batch", *self._MODULES, "--root", str(self.root), *arguments]
            )
        self.assertEqual(result, 0)
        reported = json.loads(output.getvalue().splitlines()[0])["timeout"]
        keywords = run.call_args.kwargs
        self.assertIsNone(keywords["store"], "no selected module uses the store")
        return keywords["timeout"], keywords["module_timeouts"], reported

    def test_each_module_gets_five_times_its_own_last_seconds(self) -> None:
        older = self._ledger(
            "older",
            300,
            _finish("tests.test_lifecycle", 0, seconds=50.0),
            _finish("tests.test_testing_batch", 0, seconds=200.0),
        )
        newer = self._ledger(
            "newer",
            200,
            _finish("tests.test_lifecycle", 0, seconds=30.0),
            _finish("tests.test_testing_batch", 1, ["tests.x.T.t"], seconds=40.0),
        )
        # A timeout records how long the module was allowed, not how long it takes.
        self._ledger(
            "newest", 100, _finish("tests.test_lifecycle", "timeout", seconds=1355.0)
        )

        timeout, module_timeouts, reported = self._timeouts()

        self.assertEqual(timeout, 3600.0)
        self.assertEqual(
            module_timeouts,
            {"tests.test_lifecycle": 150.0, "tests.test_testing_batch": 200.0},
        )
        self.assertEqual(
            reported,
            {
                "basis": "last_ledger_seconds",
                "seconds": 3600.0,
                "modules": module_timeouts,
                "unrecorded": [],
                "ledgers": [str(newer)],
            },
        )
        self.assertNotIn(str(older), reported["ledgers"])

    def test_an_interrupted_finish_line_is_not_a_duration(self) -> None:
        self._ledger(
            "older",
            200,
            _finish("tests.test_lifecycle", 0, seconds=100.0),
            _finish("tests.test_testing_batch", 0, seconds=2.0),
        )
        interrupted = self._ledger("newer", 100)
        with interrupted.open("a") as stream:
            # Written without its newline: the controller stopped mid-record.
            stream.write(json.dumps(_finish("tests.test_lifecycle", 0, seconds=3.0)))
        stamp = time.time() - 100
        os.utime(interrupted, (stamp, stamp))

        _, module_timeouts, _ = self._timeouts()

        self.assertEqual(module_timeouts["tests.test_lifecycle"], 500.0)

    def test_a_fast_module_that_hangs_stops_after_about_two_minutes(self) -> None:
        # A module that ran 3 s at base hung for 1355 s beside slower modules.
        self._ledger(
            "scoped",
            100,
            _finish("tests.test_lifecycle", 0, seconds=3.0),
            _finish("tests.test_testing_batch", 0, seconds=312.13),
        )

        _, module_timeouts, _ = self._timeouts()

        self.assertEqual(module_timeouts["tests.test_lifecycle"], 120.0)

    def test_a_derived_limit_never_exceeds_the_flat_default(self) -> None:
        # The f702e86d checkpoint's slowest module took 747 s; five times is 3735 s.
        self._ledger(
            "checkpoint",
            100,
            _finish("tests.test_lifecycle", 0, seconds=747.042),
            _finish("tests.test_testing_batch", 0, seconds=2.0),
        )

        _, module_timeouts, _ = self._timeouts()

        self.assertEqual(module_timeouts["tests.test_lifecycle"], 3600.0)

    def test_a_module_without_recorded_seconds_keeps_the_flat_default(self) -> None:
        self._ledger("partial", 100, _finish("tests.test_lifecycle", 0, seconds=3.0))

        timeout, module_timeouts, reported = self._timeouts()

        self.assertEqual(timeout, 3600.0)
        self.assertEqual(module_timeouts, {"tests.test_lifecycle": 120.0})
        self.assertEqual(reported["unrecorded"], ["tests.test_testing_batch"])

    def test_a_resumed_root_keeps_the_limits_its_header_bound(self) -> None:
        self._ledger(
            "fast",
            100,
            _finish("tests.test_lifecycle", 0, seconds=3.0),
            _finish("tests.test_testing_batch", 0, seconds=2.0),
        )
        _write_ledger(
            self.root,
            "a" * 40,
            self._MODULES,
            shards={},
            shard_schema="bio-neb.testing-batch-ledger/5",
        )
        header, *rest = (self.root / "ledger.jsonl").read_text().splitlines()
        bound = {**json.loads(header), "module_timeout_seconds": {"tests.x": 300.0}}
        (self.root / "ledger.jsonl").write_text(
            "\n".join([json.dumps(bound), *rest]) + "\n"
        )

        timeout, module_timeouts, reported = self._timeouts()

        self.assertEqual(timeout, 3600.0)
        self.assertEqual(module_timeouts, {"tests.x": 300.0})
        self.assertEqual(reported["basis"], "resumed_root")

    def test_an_explicit_timeout_applies_to_every_module(self) -> None:
        self._ledger("fast", 100, _finish("tests.test_lifecycle", 0, seconds=3.0))

        timeout, module_timeouts, reported = self._timeouts("--timeout", "45")

        self.assertEqual((timeout, module_timeouts), (45.0, {}))
        self.assertEqual(reported, {"seconds": 45.0, "basis": "explicit"})


class _PreviousLedgersTestCase(unittest.TestCase):
    """A batch command over real modules, with previous ledgers under the run roots."""

    _MODULES = ["tests.test_cli", "tests.test_lifecycle", "tests.test_testing_batch"]

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.runs = Path(directory.name).resolve() / "testing-runs"
        self.runs.mkdir()
        environment = patch.dict(
            os.environ, {locations.TESTING_RUN_ROOTS_OVERRIDE: str(self.runs)}
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.root = self.runs / "272-batch"
        # Scheduling only; the builder phase has its own cases.
        builder = patch.object(testing_operation, "_build_prerequisites")
        builder.start().return_value = None
        self.addCleanup(builder.stop)

    def _ledger(self, name: str, age: int, *finishes: dict, **options) -> Path:
        root = self.runs / name
        _write_ledger(root, "a" * 40, self._MODULES, *finishes, **options)
        ledger = root / "ledger.jsonl"
        stamp = time.time() - age
        os.utime(ledger, (stamp, stamp))
        return ledger

    def _batch_keywords(self, *arguments: str) -> dict:
        with (
            patch.object(testing_operation, "_run_batch", return_value=0) as run,
            patch("sys.stdout"),
        ):
            result = testing_operation.main(
                ["batch", *self._MODULES, "--root", str(self.root), *arguments]
            )
        self.assertEqual(result, 0)
        return run.call_args.kwargs

    def _order(self, *arguments: str) -> str:
        return self._batch_keywords(*arguments)["order"]


class LongestFirstOrderTests(_PreviousLedgersTestCase):
    """Batches start longest-first by each module's newest recorded seconds."""

    def test_each_module_is_timed_by_the_newest_ledger_that_finished_it_whole(
        self,
    ) -> None:
        older = self._ledger(
            "older-wide",
            300,
            _finish("tests.test_cli", 0, seconds=5.0),
            _finish("tests.test_lifecycle", 0, seconds=50.0),
            _finish("tests.test_testing_batch", 0, seconds=7.0),
        )
        newest = self._ledger(
            "newest-narrow", 100, _finish("tests.test_cli", 0, seconds=9.0)
        )
        # A timeout and a ladder phase record no whole-module duration.
        self._ledger(
            "newer-untimed",
            50,
            _finish("tests.test_cli", "timeout", seconds=500.0),
            _finish("tests.test_lifecycle", 0, seconds=99.0, phase="representative"),
        )
        # An unreadable ledger times nothing.
        self._ledger("newest-unreadable", 10).write_bytes(b"\xff\xfe not a ledger\n")

        keywords = self._batch_keywords()

        self.assertEqual(keywords["order"], "longest")
        self.assertEqual(
            keywords["estimates"],
            {
                "tests.test_cli": 9.0,
                "tests.test_lifecycle": 50.0,
                "tests.test_testing_batch": 7.0,
            },
        )
        self.assertEqual(keywords["timing_ledgers"], sorted([str(older), str(newest)]))
        self.assertEqual(
            testing_operation._run_order(
                self._MODULES, keywords["order"], keywords["estimates"]
            ),
            ["tests.test_lifecycle", "tests.test_cli", "tests.test_testing_batch"],
        )

    def test_without_a_timed_previous_ledger_the_batch_starts_sorted(self) -> None:
        self._ledger("timeouts", 100, _finish("tests.test_cli", "timeout"))

        self.assertEqual(self._order(), "sorted")

    def test_a_resumed_root_keeps_the_order_its_header_bound(self) -> None:
        self._ledger("newer", 100, _finish("tests.test_cli", 0, seconds=5.0))
        _write_ledger(
            self.root,
            "a" * 40,
            self._MODULES,
            shards={},
            shard_schema="bio-neb.testing-batch-ledger/5",
        )
        header, *rest = (self.root / "ledger.jsonl").read_text().splitlines()
        bound = {**json.loads(header), "order": "sorted"}
        (self.root / "ledger.jsonl").write_text(
            "\n".join([json.dumps(bound), *rest]) + "\n"
        )

        self.assertEqual(self._order(), "sorted")

    def test_an_explicit_order_overrides_longest_first(self) -> None:
        self._ledger("timed", 100, _finish("tests.test_cli", 0, seconds=5.0))

        self.assertEqual(self._order("--order", "sorted"), "sorted")

    def test_units_without_an_estimate_start_after_the_timed_ones_sorted(
        self,
    ) -> None:
        self.assertEqual(
            testing_operation._run_order(
                [
                    "tests.test_testing_batch#2",
                    "tests.test_lifecycle",
                    "tests.test_cli",
                ],
                "longest",
                {"tests.test_testing_batch#2": 3.0, "tests.test_lifecycle": 5.0},
            ),
            ["tests.test_lifecycle", "tests.test_testing_batch#2", "tests.test_cli"],
        )


class ShardPlanTests(_PreviousLedgersTestCase):
    """Ticket 293: a heavy module splits into shards by its tests' recorded seconds.

    The batch's estimated seconds over its slots, halved, is the most one unit should
    take; a module above it splits into contiguous runs of its sorted test IDs, each at
    least 30 s of recorded test time, the last running whatever the others do not.
    """

    _HEAVY = "tests.test_testing_batch"

    def setUp(self) -> None:
        super().setUp()
        self.ids = sorted(
            testing_operation._test_method_ids({"tests/test_testing_batch.py"})
        )[:4]

    def _heavy_seconds(self, *ids: str, seconds: float = 20.0) -> dict[str, float]:
        return dict.fromkeys(ids, seconds)

    def test_a_heavy_module_splits_by_its_recorded_test_seconds(self) -> None:
        gone = f"{self._HEAVY}.RetiredTests.test_no_longer_exists"
        ledger = self._ledger(
            "timed",
            100,
            # 80 s of current tests, 20 s of a retired one and 4 s of setup.
            _finish(
                self._HEAVY,
                0,
                seconds=104.0,
                test_seconds=self._heavy_seconds(*self.ids, gone),
            ),
            _finish("tests.test_cli", 0, seconds=10.0, test_seconds={}),
            # Above the share too, but without per-test seconds it cannot split.
            _finish("tests.test_lifecycle", 0, seconds=100.0),
        )

        keywords = self._batch_keywords("--concurrency", "4")

        # 214 s over 4 slots, halved: about 27 s per unit; 80 s of tests at 30 s or
        # more per shard is two shards; the retired test is no shard's.
        self.assertEqual(keywords["shards"], {self._HEAVY: [self.ids[:2]]})
        self.assertEqual(
            keywords["estimates"],
            {
                f"{self._HEAVY}#1": 44.0,
                f"{self._HEAVY}#2": 44.0,
                "tests.test_cli": 10.0,
                "tests.test_lifecycle": 100.0,
            },
        )
        self.assertEqual(
            keywords["module_timeouts"],
            {
                f"{self._HEAVY}#1": 220.0,
                f"{self._HEAVY}#2": 220.0,
                "tests.test_cli": 120.0,
                "tests.test_lifecycle": 500.0,
            },
        )
        self.assertEqual(keywords["timing_ledgers"], [str(ledger)])
        self.assertEqual(
            testing_operation._run_order(
                [*self._MODULES[:2], f"{self._HEAVY}#1", f"{self._HEAVY}#2"],
                keywords["order"],
                keywords["estimates"],
            ),
            [
                "tests.test_lifecycle",
                f"{self._HEAVY}#1",
                f"{self._HEAVY}#2",
                "tests.test_cli",
            ],
        )

    def test_a_module_above_the_share_with_under_30_s_of_tests_stays_whole(
        self,
    ) -> None:
        # Spec review r1 S1: 20 s of tests and 1 s of setup, far above a 2.9 s share,
        # is still too little for even one 30 s shard.
        self._ledger(
            "timed",
            100,
            _finish(
                self._HEAVY,
                0,
                seconds=21.0,
                test_seconds=self._heavy_seconds(*self.ids[:2], seconds=10.0),
            ),
            _finish("tests.test_cli", 0, seconds=1.0),
            _finish("tests.test_lifecycle", 0, seconds=1.0),
        )

        keywords = self._batch_keywords("--concurrency", "4")

        self.assertEqual(keywords["shards"], {})
        self.assertEqual(keywords["estimates"][self._HEAVY], 21.0)
        self.assertEqual(keywords["module_timeouts"][self._HEAVY], 120.0)

    def test_a_sharded_ledger_times_its_module_whole(self) -> None:
        self._ledger(
            "sharded",
            100,
            _finish(
                self._HEAVY,
                0,
                seconds=45.0,
                test_seconds=self._heavy_seconds(*self.ids[:2]),
                shard=1,
                shards=2,
            ),
            _finish(
                self._HEAVY,
                0,
                seconds=43.0,
                test_seconds=self._heavy_seconds(*self.ids[2:]),
                shard=2,
                shards=2,
            ),
            _finish("tests.test_cli", 0, seconds=10.0),
            _finish("tests.test_lifecycle", 0, seconds=10.0),
            shards={self._HEAVY: [self.ids[:2]]},
        )
        # A newer ledger missing one shard does not time the module.
        self._ledger(
            "half",
            50,
            _finish(self._HEAVY, 0, seconds=5.0, test_seconds={}, shard=1, shards=2),
            shards={self._HEAVY: [self.ids[:2]]},
        )

        keywords = self._batch_keywords("--concurrency", "4")

        # The shards' mean setup, 4 s, plus all 80 s of tests.
        self.assertEqual(keywords["shards"], {self._HEAVY: [self.ids[:2]]})
        self.assertEqual(
            (keywords["estimates"][f"{self._HEAVY}#1"], keywords["timing_ledgers"]),
            (44.0, [str(self.runs / "sharded" / "ledger.jsonl")]),
        )

    def test_one_slot_and_laddered_modules_run_whole(self) -> None:
        self._ledger(
            "timed",
            100,
            _finish(
                self._HEAVY,
                0,
                seconds=84.0,
                test_seconds=self._heavy_seconds(*self.ids),
            ),
            _finish("tests.test_cli", 0, seconds=10.0),
            _finish("tests.test_lifecycle", 0, seconds=10.0),
        )

        one_slot = self._batch_keywords("--concurrency", "1")
        declared = {self._HEAVY: self.ids[:1]}
        with (
            patch.object(testing_operation, "_run_batch", return_value=0) as run,
            patch.object(
                testing_operation, "_changed_test_methods", return_value=set()
            ),
            patch("sys.stdout"),
        ):
            testing_operation._batch_command(
                self._MODULES,
                False,
                self.root,
                None,
                "representative",
                _git(Path(testing_operation.__file__).parents[3], "rev-parse", "HEAD"),
                declared,
                _schedule(concurrency=4, order="longest"),
            )

        self.assertEqual(one_slot["shards"], {})
        self.assertEqual(one_slot["estimates"][self._HEAVY], 84.0)
        self.assertEqual(run.call_args.kwargs["shards"], {})
        self.assertEqual(run.call_args.kwargs["representatives"], declared)

    def test_a_resumed_root_keeps_the_plan_its_header_bound(self) -> None:
        self._ledger(
            "newer",
            100,
            _finish(
                self._HEAVY,
                0,
                seconds=84.0,
                test_seconds=self._heavy_seconds(*self.ids),
            ),
            _finish("tests.test_cli", 0, seconds=10.0),
        )
        _write_ledger(
            self.root,
            "a" * 40,
            self._MODULES,
            shards={self._HEAVY: [self.ids[:1]]},
            shard_schema="bio-neb.testing-batch-ledger/5",
        )
        header, *rest = (self.root / "ledger.jsonl").read_text().splitlines()
        bound = {
            **json.loads(header),
            "order": "longest",
            "estimated_seconds": {"tests.test_cli": 1.0},
            "timing_ledgers": ["/elsewhere/ledger.jsonl"],
        }
        (self.root / "ledger.jsonl").write_text(
            "\n".join([json.dumps(bound), *rest]) + "\n"
        )

        keywords = self._batch_keywords("--concurrency", "4")

        self.assertEqual(
            (
                keywords["shards"],
                keywords["estimates"],
                keywords["timing_ledgers"],
                keywords["order"],
            ),
            (
                {self._HEAVY: [self.ids[:1]]},
                {"tests.test_cli": 1.0},
                ["/elsewhere/ledger.jsonl"],
                "longest",
            ),
        )


# The runner a snapshot executes: it records what it saw instead of running a suite, so
# a test observes the snapshot's commit, files, import path and forwarded arguments.
_SNAPSHOT_RUNNER = """
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

record = {
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "runner": __file__,
    "head": subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip(),
    "marker": Path("marker.txt").read_text(),
    "pid": os.getpid(),
}
Path(os.environ["SNAPSHOT_TEST_RECORD"]).write_text(json.dumps(record))
if "--root" in sys.argv:
    root = Path(sys.argv[sys.argv.index("--root") + 1])
    module = sys.argv[2]
    (root / "ledger.jsonl").write_text(
        json.dumps({"schema": "bio-neb.testing-batch-ledger/2", "commit": record["head"]})
        + "\\n"
        + json.dumps({"module": module, "exit": 1, "seconds": 1.0,
                      "failed": [module + ".T.test_red"], "errored": [],
                      "reasons": {module + ".T.test_red": "AssertionError: base red"},
                      "log": "logs/x.log"})
        + "\\n"
    )
    if os.environ.get("SNAPSHOT_TEST_SHARDED"):
        # A base runner that splits the module: a red shard, then the green rest.
        red = {"module": module, "exit": 1, "seconds": 1.0,
               "failed": [module + ".T.test_red"], "errored": [],
               "reasons": {module + ".T.test_red": "AssertionError: base red"},
               "log": "logs/x#1.log", "shard": 1, "shards": 2}
        green = {"module": module, "exit": 0, "seconds": 1.0, "failed": [],
                 "errored": [], "reasons": {}, "log": "logs/x#2.log",
                 "shard": 2, "shards": 2}
        (root / "ledger.jsonl").write_text(
            "".join(json.dumps(line) + "\\n" for line in (
                {"schema": "bio-neb.testing-batch-ledger/4", "commit": record["head"],
                 "selection": [module], "shards": {module: [[module + ".T.test_red"]]}},
                red,
                green,
            ))
        )
if os.environ.get("SNAPSHOT_TEST_ORPHAN_MODULE"):
    # A batch controller killed while its module, holding the batch lock, runs on.
    import fcntl

    lock = open(root / "ledger.lock", "a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    module = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        pass_fds=(lock.fileno(),),
        start_new_session=True,
    )
    Path(os.environ["SNAPSHOT_TEST_ORPHAN_MODULE"]).write_text(str(module.pid))
    os.kill(os.getpid(), signal.SIGKILL)
started = os.environ.get("SNAPSHOT_TEST_STARTED")
if started:
    # Like the real runner, a stop request ends the run with 130.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(130))
    Path(started).touch()
    time.sleep(60)
raise SystemExit(int(os.environ.get("SNAPSHOT_TEST_EXIT", "0")))
"""


class SnapshotRunTests(unittest.TestCase):
    """Ticket 234: `--at <sha>` and `base-repro` run from a worktree of one commit."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base = Path(directory.name).resolve()
        self.repository = base / "repository"
        runner = self.repository / "src" / "bio_neb" / "devtools"
        runner.mkdir(parents=True)
        for package in (runner.parent, runner):
            (package / "__init__.py").write_text("")
        (runner / "testing.py").write_text(_SNAPSHOT_RUNNER)
        (self.repository / "uv.lock").write_text("locked = true\n")
        (self.repository / "marker.txt").write_text("base\n")
        _git(self.repository, "init", "-q")
        _git(self.repository, "add", "-A")
        _git(self.repository, "commit", "-q", "-m", "base")
        self.base = _git(self.repository, "rev-parse", "HEAD")
        (self.repository / "marker.txt").write_text("committed\n")
        _git(self.repository, "commit", "-q", "-am", "candidate")
        self.candidate = _git(self.repository, "rev-parse", "HEAD")
        # The implementer keeps editing while the snapshot runs.
        (self.repository / "marker.txt").write_text("edited, uncommitted\n")
        self.runs = base / "testing-runs"
        self.runs.mkdir()
        self.root = self.runs / "234-at"
        self.record = base / "record.json"
        environment = patch.dict(
            os.environ,
            {
                locations.TESTING_RUN_ROOTS_OVERRIDE: str(self.runs),
                "SNAPSHOT_TEST_RECORD": str(self.record),
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def _main(self, *arguments: str) -> tuple[int, list[dict], str]:
        output, errors = io.StringIO(), io.StringIO()
        with (
            patch.object(testing_operation, "_REPOSITORY", self.repository),
            patch.object(testing_operation, "_import_graph"),
            patch.object(testing_operation, "_validate_intent"),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            result = testing_operation.main(list(arguments))
        lines = [json.loads(line) for line in output.getvalue().splitlines()]
        return result, lines, errors.getvalue()

    def _seen(self) -> dict:
        return json.loads(self.record.read_text())

    def _worktrees(self) -> str:
        return _git(self.repository, "worktree", "list", "--porcelain")

    def test_batch_at_a_commit_runs_that_commit_while_the_tree_changes(self) -> None:
        result, lines, _ = self._main(
            "batch", "tests.test_x", "--root", str(self.root), "--at", self.candidate
        )

        self.assertEqual(result, 0)
        seen = self._seen()
        checkout = self.root / "checkout"
        self.assertEqual(seen["head"], self.candidate)
        self.assertEqual(seen["marker"], "committed\n")
        self.assertEqual(seen["cwd"], str(checkout))
        # The snapshot's own runner and product code, not the working checkout's.
        self.assertEqual(
            seen["runner"], str(checkout / "src/bio_neb/devtools/testing.py")
        )
        self.assertEqual(
            seen["argv"], ["batch", "tests.test_x", "--root", str(self.root)]
        )
        self.assertEqual(
            lines[0],
            {"snapshot": {"commit": self.candidate, "checkout": str(checkout)}},
        )
        # A checkout, not a retained resource: removed once the run ends.
        self.assertFalse(checkout.exists())
        self.assertNotIn(str(checkout), self._worktrees())
        self.assertEqual(
            (self.repository / "marker.txt").read_text(), "edited, uncommitted\n"
        )

    def test_the_snapshot_runs_exit_status_is_the_commands(self) -> None:
        with patch.dict(os.environ, {"SNAPSHOT_TEST_EXIT": "1"}):
            result, _, _ = self._main(
                "scoped",
                "--ticket-id",
                "234",
                "--base-sha",
                self.base,
                "--root",
                str(self.root),
                "--at",
                self.candidate,
            )

        self.assertEqual(result, 1)
        self.assertEqual(
            self._seen()["argv"],
            [
                "scoped",
                "--ticket-id",
                "234",
                "--base-sha",
                self.base,
                "--root",
                str(self.root),
            ],
        )

    def test_a_snapshot_needs_the_lock_the_environment_was_built_from(self) -> None:
        (self.repository / "uv.lock").write_text("locked = differently\n")
        _git(self.repository, "commit", "-q", "-am", "lock change")

        result, _, errors = self._main(
            "batch", "tests.test_x", "--root", str(self.root), "--at", self.base
        )

        self.assertEqual(result, 2)
        self.assertIn("uv.lock", errors)
        self.assertFalse(self.record.exists())
        self.assertFalse((self.root / "checkout").exists())
        self.assertNotIn(str(self.root / "checkout"), self._worktrees())

    def test_a_module_outliving_its_batch_controller_keeps_the_checkout(self) -> None:
        orphan = self.record.parent / "orphan-module-pid"
        self.addCleanup(
            lambda: orphan.exists() and _kill_quietly(int(orphan.read_text()))
        )

        arguments = (
            "batch",
            "tests.test_x",
            "--root",
            str(self.root),
            "--at",
            self.candidate,
        )
        with patch.dict(os.environ, {"SNAPSHOT_TEST_ORPHAN_MODULE": str(orphan)}):
            result, _, errors = self._main(*arguments)

        self.assertEqual(result, 128 + signal.SIGKILL)
        self.assertIn("testing snapshot left in place", errors)
        self.assertEqual(
            (self.root / "checkout" / "marker.txt").read_text(), "committed\n"
        )
        # Once the module has ended, the next run reclaims the snapshot it left.
        _kill_quietly(int(orphan.read_text()))
        deadline = time.monotonic() + 30
        while (result := self._main(*arguments)[0]) == 2:
            self.assertLess(time.monotonic(), deadline, "module never released")
            time.sleep(0.1)
        self.assertEqual(result, 0)
        self.assertFalse((self.root / "checkout").exists())

    def test_a_checkout_directory_the_runner_did_not_make_is_kept(self) -> None:
        unrelated = self.root / "checkout" / "unrelated-work.txt"
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text("keep me\n")

        result, _, errors = self._main(
            "batch", "tests.test_x", "--root", str(self.root), "--at", self.candidate
        )

        self.assertEqual(result, 2)
        self.assertIn("is not a snapshot worktree", errors)
        self.assertEqual(unrelated.read_text(), "keep me\n")
        self.assertFalse(self.record.exists())

    def test_a_detached_worktree_the_runner_did_not_make_is_kept(self) -> None:
        checkout = self.root / "checkout"
        self.root.mkdir()
        _git(self.repository, "worktree", "add", "--detach", str(checkout), self.base)
        (checkout / "unrelated-work.txt").write_text("keep me\n")

        result, _, errors = self._main(
            "batch", "tests.test_x", "--root", str(self.root), "--at", self.candidate
        )

        self.assertEqual(result, 2)
        self.assertIn("is not a snapshot worktree", errors)
        self.assertEqual((checkout / "unrelated-work.txt").read_text(), "keep me\n")
        self.assertFalse(self.record.exists())

    def test_a_snapshot_git_cannot_remove_stays_in_place(self) -> None:
        checkout = self.root / "checkout"
        self.root.mkdir()
        # A snapshot an earlier run left: locked with the runner's reason for this root.
        _git(
            self.repository,
            "worktree",
            "add",
            "--detach",
            "--lock",
            "--reason",
            f"bio-neb testing snapshot {self.root}",
            str(checkout),
            self.base,
        )
        stuck = checkout / "read-only"
        stuck.mkdir()
        (stuck / "file.txt").write_text("cannot be removed\n")
        stuck.chmod(0o500)
        self.addCleanup(stuck.chmod, 0o700)

        result, _, errors = self._main(
            "batch", "tests.test_x", "--root", str(self.root), "--at", self.candidate
        )

        self.assertEqual(result, 2)
        self.assertIn("cannot remove snapshot", errors)
        # Git removes what it can; the runner deletes nothing more and runs nothing.
        self.assertEqual((stuck / "file.txt").read_text(), "cannot be removed\n")
        self.assertFalse(self.record.exists())

    def test_a_snapshot_needs_a_batch_root(self) -> None:
        result, _, errors = self._main(
            "scoped",
            "--ticket-id",
            "234",
            "--base-sha",
            self.base,
            "--at",
            self.candidate,
        )

        self.assertEqual(result, 2)
        self.assertIn("--at needs --root", errors)

    def test_base_repro_runs_the_module_at_the_base_and_reports_its_result(
        self,
    ) -> None:
        result, lines, _ = self._main(
            "base-repro", "tests/test_x.py", "--base-sha", self.base
        )

        self.assertEqual(result, 0)
        seen = self._seen()
        self.assertEqual(seen["head"], self.base)
        self.assertEqual(seen["marker"], "base\n")
        # The derived timeout is passed on, so a base without derivation still binds it.
        root = Path(seen["argv"][3])
        self.assertEqual(
            seen["argv"],
            ["batch", "tests.test_x", "--root", str(root), "--timeout", "3600.0"],
        )
        self.assertEqual(root.parent, self.runs)
        self.assertTrue(root.name.startswith("base-repro-"))
        self.assertTrue(root.name.endswith(f"-{self.base[:8]}"))
        self.assertEqual(
            lines[-1],
            {
                "base_repro": {
                    "commit": self.base,
                    "ledger": str(root / "ledger.jsonl"),
                    "modules": {
                        "tests.test_x": {
                            "exit": 1,
                            "failed": ["tests.test_x.T.test_red"],
                            "errored": [],
                            "reasons": {
                                "tests.test_x.T.test_red": "AssertionError: base red"
                            },
                        }
                    },
                }
            },
        )
        self.assertFalse((root / "checkout").exists())

    def test_base_repro_reports_a_sharded_module_as_one_result(self) -> None:
        with patch.dict(os.environ, {"SNAPSHOT_TEST_SHARDED": "1"}):
            result, lines, _ = self._main(
                "base-repro", "tests/test_x.py", "--base-sha", self.base
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            lines[-1]["base_repro"]["modules"],
            {
                "tests.test_x": {
                    "exit": 1,
                    "failed": ["tests.test_x.T.test_red"],
                    "errored": [],
                    "reasons": {"tests.test_x.T.test_red": "AssertionError: base red"},
                }
            },
        )

    def _start_controller(self) -> subprocess.Popen[bytes]:
        """A `batch --at` controller in its own process, once its snapshot run started."""
        started = self.record.parent / "started"
        program = (
            "import sys\n"
            "from pathlib import Path\n"
            "from unittest.mock import patch\n"
            "from bio_neb.devtools import testing\n"
            "testing._REPOSITORY = Path(sys.argv[1])\n"
            "with patch.object(testing, '_import_graph'),"
            " patch.object(testing, '_validate_intent'):\n"
            "    raise SystemExit(testing.main(sys.argv[2:]))\n"
        )
        controller = subprocess.Popen(
            [
                sys.executable,
                "-c",
                program,
                str(self.repository),
                "batch",
                "tests.test_x",
                "--root",
                str(self.root),
                "--at",
                self.candidate,
            ],
            env={**os.environ, "SNAPSHOT_TEST_STARTED": str(started)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(controller.kill)
        deadline = time.monotonic() + 30
        while not started.exists():
            self.assertLess(time.monotonic(), deadline, "snapshot run never started")
            time.sleep(0.05)
        return controller

    def test_a_stop_signal_reaches_the_snapshot_run_before_its_checkout_goes(
        self,
    ) -> None:
        controller = self._start_controller()

        controller.send_signal(signal.SIGTERM)

        self.assertEqual(controller.wait(timeout=30), 130)
        self.assertFalse((self.root / "checkout").exists())
        self.assertNotIn(str(self.root / "checkout"), self._worktrees())

    def test_a_second_snapshot_run_on_a_live_root_is_refused(self) -> None:
        controller = self._start_controller()

        result, _, errors = self._main(
            "batch", "tests.test_x", "--root", str(self.root), "--at", self.candidate
        )

        self.assertEqual(result, 2)
        self.assertIn("still running", errors)
        self.assertEqual(
            (self.root / "checkout" / "marker.txt").read_text(), "committed\n"
        )
        controller.send_signal(signal.SIGTERM)
        self.assertEqual(controller.wait(timeout=30), 130)

    def test_an_orphaned_snapshot_run_keeps_its_checkout(self) -> None:
        """A killed controller's snapshot run still owns the root until it ends."""
        controller = self._start_controller()
        orphan = self._seen()["pid"]
        self.addCleanup(lambda: _kill_quietly(orphan))
        controller.kill()
        controller.wait(timeout=30)

        result, _, errors = self._main(
            "batch", "tests.test_x", "--root", str(self.root), "--at", self.candidate
        )

        self.assertEqual(result, 2)
        self.assertIn("still running", errors)
        self.assertEqual(
            (self.root / "checkout" / "marker.txt").read_text(), "committed\n"
        )


def _kill_quietly(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class BatchCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        # No previous ledger, so an unset --timeout keeps the flat default.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        environment = patch.dict(
            os.environ, {locations.TESTING_RUN_ROOTS_OVERRIDE: directory.name}
        )
        environment.start()
        self.addCleanup(environment.stop)
        # These cases check the arguments `batch` passes on; the builder phase before
        # the batch has its own cases in PrerequisiteBuilderPhaseTests.
        builder = patch.object(testing_operation, "_build_prerequisites")
        builder.start().return_value = None
        self.addCleanup(builder.stop)

    def _batch_main(self, *arguments: str) -> tuple[int, str, object]:
        with (
            patch.object(testing_operation, "_run_batch", return_value=0) as run,
            patch("sys.stdout"),
            patch("sys.stderr") as stderr,
        ):
            result = testing_operation.main(
                ["batch", "tests.test_lifecycle", "--root", "/tmp/batch-x", *arguments]
            )
        written = "".join(call.args[0] for call in stderr.write.call_args_list)
        return result, written, run

    def _assert_refused(self, cases: dict[str, tuple[list[str], str]]) -> None:
        for name, (arguments, refusal) in cases.items():
            with self.subTest(name):
                result, written, run = self._batch_main(*arguments)

                self.assertEqual(result, 2)
                self.assertIn(refusal, written)
                run.assert_not_called()

    def test_a_sole_runner_batch_runs_eight_modules_at_once_unless_told_otherwise(
        self,
    ) -> None:
        # "sole runner we do 8 concerany and raid to 10" (4 beside another runner);
        # isolation is the Run sandbox's (ADR 0041), so no approval record or digest
        # is consulted. This case's host queue is its own, so the batch is the sole
        # runner.
        root = Path("/tmp/batch-x")
        cases = {
            "default": ([], 24, 8),
            "default on a two-CPU machine": ([], 2, 2),
            "default on a six-CPU machine": ([], 6, 6),
            "override below the default": (["--concurrency", "2"], 24, 2),
            "override at the ceiling": (["--concurrency", "10"], 24, 10),
        }
        for name, (arguments, cpus, concurrency) in cases.items():
            with self.subTest(name), patch("os.cpu_count", return_value=cpus):
                result, _, run = self._batch_main(*arguments)

                self.assertEqual(result, 0)
                run.assert_called_once_with(
                    ["tests.test_lifecycle"],
                    root,
                    timeout=3600.0,
                    module_timeouts={},
                    store=None,
                    shards={},
                    estimates={},
                    timing_ledgers=[],
                    slot_wait_seconds=ANY,
                    **_schedule(concurrency=concurrency),
                )
        with patch("os.cpu_count", return_value=24):
            self._assert_refused(
                {
                    "above the ceiling": (
                        ["--concurrency", "11"],
                        "--concurrency must be from 1 to 10",
                    )
                }
            )

    def test_concurrency_order_and_experiment_options_are_validated(self) -> None:
        cpus = os.cpu_count() or 1
        experiment = ["--parallel-experiment", "--detect-writes"]
        self._assert_refused(
            {
                "zero": (["--concurrency", "0"], "--concurrency must be from 1 to"),
                "experiment above the CPU count": (
                    ["--concurrency", str(cpus + 1), *experiment],
                    f"--concurrency must be from 1 to {cpus}",
                ),
                "experiment without write detection": (
                    ["--concurrency", "2", "--parallel-experiment"],
                    "--parallel-experiment needs --detect-writes",
                ),
                "shuffle without seed": (
                    ["--order", "shuffle"],
                    "--order shuffle needs --seed",
                ),
                "longest+shuffle without seed": (
                    ["--order", "longest+shuffle"],
                    "--order longest+shuffle needs --seed",
                ),
                "seed without shuffle": (
                    ["--seed", "7"],
                    "--seed applies only to a shuffle order",
                ),
            }
        )

    def test_an_experiment_root_may_run_above_the_ceiling_up_to_the_cpu_count(
        self,
    ) -> None:
        with patch("os.cpu_count", return_value=24):
            result, _, run = self._batch_main(
                "--concurrency",
                "16",
                "--parallel-experiment",
                "--order",
                "shuffle",
                "--seed",
                "7",
                "--detect-writes",
            )
        self.assertEqual(result, 0)
        run.assert_called_once_with(
            ["tests.test_lifecycle"],
            Path("/tmp/batch-x"),
            timeout=3600.0,
            module_timeouts={},
            store=None,
            shards={},
            estimates={},
            timing_ledgers=[],
            slot_wait_seconds=ANY,
            **_schedule(
                concurrency=16,
                order="shuffle:7",
                detect_writes=True,
                parallel_experiment=True,
            ),
        )

    def test_a_longest_shuffle_order_binds_its_seed(self) -> None:
        # No previous ledger timed the module, so no ledger is named.
        result, _, run = self._batch_main("--order", "longest+shuffle", "--seed", "7")

        self.assertEqual(result, 0)
        self.assertEqual(run.call_args.kwargs["order"], "longest+shuffle:7")

    def test_batch_command_passes_named_modules_sorted_and_dotted_to_the_batch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "batch"
            with (
                patch.object(testing_operation, "_run_batch", return_value=0) as run,
                patch("sys.stdout"),
            ):
                result = testing_operation.main(
                    [
                        "batch",
                        "tests/test_testing_batch.py",
                        "tests.test_lifecycle",
                        "--root",
                        str(root),
                    ]
                )

        self.assertEqual(result, 0)
        run.assert_called_once_with(
            ["tests.test_lifecycle", "tests.test_testing_batch"],
            root,
            timeout=3600.0,
            module_timeouts={},
            store=None,
            shards={},
            estimates={},
            timing_ledgers=[],
            slot_wait_seconds=ANY,
            **_schedule(),
        )

    def test_representative_batch_adds_changed_methods_to_declared_lists(
        self,
    ) -> None:
        repository = Path(testing_operation.__file__).resolve().parents[3]
        head = _git(repository, "rev-parse", "HEAD")
        declared = testing_operation._load_ownership()["representative_selections"]
        changed = (
            "tests.test_operation.OperationTests."
            "test_result_contract_import_and_read_work_without_execution_modules"
        )
        self.assertNotIn(changed, declared["tests.test_operation"]["test_ids"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "batch"
            with (
                patch.object(testing_operation, "_run_batch", return_value=0) as run,
                patch.object(
                    testing_operation,
                    "_changed_test_methods",
                    return_value={changed, "tests.test_lifecycle.T.test_whole"},
                ) as changed_methods,
                patch("sys.stdout"),
            ):
                result = testing_operation.main(
                    [
                        "batch",
                        "tests.test_operation",
                        "tests.test_lifecycle",
                        "--rung",
                        "representative",
                        "--base-sha",
                        head,
                        "--root",
                        str(root),
                    ]
                )

        self.assertEqual(result, 0)
        changed_methods.assert_called_once_with(head, ["tests/test_operation.py"])
        run.assert_called_once_with(
            ["tests.test_lifecycle", "tests.test_operation"],
            root,
            timeout=3600.0,
            module_timeouts={},
            store=None,
            shards={},
            estimates={},
            timing_ledgers=[],
            slot_wait_seconds=ANY,
            **_schedule(),
            representatives={
                "tests.test_operation": sorted(
                    [*declared["tests.test_operation"]["test_ids"], changed]
                )
            },
            base_commit=head,
        )

    def test_batch_command_all_selects_every_discovered_module(self) -> None:
        with (
            patch.object(testing_operation, "_run_batch", return_value=0) as run,
            patch("sys.stdout"),
        ):
            testing_operation.main(["batch", "--all", "--root", "/tmp/batch-all"])

        tests = Path(testing_operation.__file__).resolve().parents[3] / "tests"
        self.assertEqual(
            run.call_args.args[0],
            sorted(f"tests.{path.stem}" for path in tests.glob("test_*.py")),
        )

    def test_batch_command_refuses_unknown_modules_and_in_repository_roots(
        self,
    ) -> None:
        repository = Path(testing_operation.__file__).resolve().parents[3]
        cases = {
            "unknown module": (
                ["tests.test_not_a_module", "--root", "/tmp/batch-x"],
                "not a discovered test module",
            ),
            "in-repository root": (
                ["tests.test_lifecycle", "--root", str(repository / "batch-root")],
                "outside the repository",
            ),
            "no selection": (
                ["--root", "/tmp/batch-x"],
                "name test modules or --all (not both)",
            ),
            "both selections": (
                ["tests.test_lifecycle", "--all", "--root", "/tmp/batch-x"],
                "name test modules or --all (not both)",
            ),
            "relative root": (
                ["tests.test_lifecycle", "--root", "batch-x"],
                "--root must be an absolute path",
            ),
            "missing root parent": (
                ["tests.test_lifecycle", "--root", "/nonexistent-parent/batch-x"],
                "cannot create --root",
            ),
            "non-finite timeout": (
                ["tests.test_lifecycle", "--root", "/tmp/batch-x", "--timeout", "nan"],
                "--timeout must be a positive number of seconds",
            ),
            "representative without base": (
                [
                    "tests.test_lifecycle",
                    "--root",
                    "/tmp/batch-x",
                    "--rung",
                    "representative",
                ],
                "--rung representative requires --base-sha",
            ),
            "base without representative": (
                [
                    "tests.test_lifecycle",
                    "--root",
                    "/tmp/batch-x",
                    "--base-sha",
                    "a" * 40,
                ],
                "--base-sha applies only to --rung representative",
            ),
        }
        for name, (arguments, refusal) in cases.items():
            with self.subTest(name), patch("sys.stdout"), patch("sys.stderr") as stderr:
                result = testing_operation.main(["batch", *arguments])

                self.assertEqual(result, 2)
                written = "".join(call.args[0] for call in stderr.write.call_args_list)
                self.assertIn(refusal, written)
        self.assertFalse((repository / "batch-root").exists())

    def test_sigterm_interrupts_the_batch_command_with_130_and_restores_handling(
        self,
    ) -> None:
        def receive_sigterm(*arguments, **keywords):
            os.kill(os.getpid(), signal.SIGTERM)
            self.fail("SIGTERM did not interrupt the batch")

        before = signal.getsignal(signal.SIGTERM)
        with (
            patch.object(testing_operation, "_run_batch", side_effect=receive_sigterm),
            patch("sys.stdout"),
            patch("sys.stderr") as stderr,
        ):
            result = testing_operation.main(
                ["batch", "tests.test_lifecycle", "--root", "/tmp/batch-int"]
            )

        self.assertEqual(result, 130)
        written = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("rerun the same command to resume", written)
        self.assertIs(signal.getsignal(signal.SIGTERM), before)

    def test_compare_command_prints_the_ledger_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            _write_ledger(left, "a" * 40, ["tests.a"], _finish("tests.a", 0))
            _write_ledger(
                right,
                "b" * 40,
                ["tests.a"],
                _finish("tests.a", 1, failed=["tests.a.T.t"]),
            )
            with patch("sys.stdout") as stdout:
                result = testing_operation.main(["compare", str(left), str(right)])

        self.assertEqual(result, 0)
        printed = json.loads(
            "".join(call.args[0] for call in stdout.write.call_args_list)
        )
        self.assertEqual(printed["added"], ["tests.a.T.t"])
        self.assertEqual(printed["schema"], "bio-neb.testing-batch-comparison/2")

    def test_compare_against_ids_matches_both_id_forms_to_the_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, ids = Path(directory) / "batch", Path(directory) / "reds.txt"
            _write_ledger(
                root,
                "a" * 40,
                ["tests.a", "tests.b"],
                _finish(
                    "tests.a",
                    1,
                    failed=["tests.a.T.test_kept", "tests.a.T.test_sub (case=1)"],
                    errored=["tests.a.T.test_new"],
                ),
                _finish("tests.b", 0),
            )
            ids.write_text(
                "# expected reds\n"
                "tests/a.py::T::test_kept\n"
                "test_sub (tests.a.T.test_sub) (case=2)\n"
                "\n"
                "tests/a.py::T::test_fixed[param]\n"
                "tests/z.py::T::test_unselected\n"
            )
            with patch("sys.stdout") as stdout:
                result = testing_operation.main(
                    ["compare", str(root), "--against-ids", str(ids)]
                )

        self.assertEqual(result, 0)
        printed = json.loads(
            "".join(call.args[0] for call in stdout.write.call_args_list)
        )
        self.assertEqual(printed["left"], {"ids": str(ids), "count": 4})
        self.assertEqual(printed["right"]["commit"], "a" * 40)
        self.assertEqual(printed["compared_modules"], ["tests.a", "tests.b"])
        self.assertEqual(printed["added"], ["tests.a.T.test_new"])
        self.assertEqual(printed["removed"], ["tests.a.T.test_fixed"])
        self.assertEqual(printed["only_left"], ["tests.z"])
        self.assertEqual(
            printed["reasons"], {"left": "unavailable", "right": "recorded"}
        )
        self.assertIsNone(printed["reason_changed"])

    def test_compare_takes_two_roots_or_one_root_and_an_id_list(self) -> None:
        for argv in (["compare", "/a"], ["compare", "/a", "/b", "--against-ids", "/c"]):
            with self.subTest(argv=argv), patch("sys.stderr") as stderr:
                self.assertEqual(testing_operation.main(argv), 2)
                written = "".join(call.args[0] for call in stderr.write.call_args_list)
                self.assertIn(
                    "compare takes two batch roots, or one with --against-ids", written
                )


# A builder program standing in for the prerequisite builder entry: it writes the
# receipt the runner reads, and records the builder module it was named and the
# environment it ran with.
_FAKE_BUILDER = """
import json, os, sys
from pathlib import Path

store = Path(sys.argv[2])
store.mkdir(parents=True, exist_ok=True)
try:
    closed = os.open(os.devnull, os.O_RDONLY)
    os.close(closed)
    os.fsync(closed)  # a real fsync refuses a closed descriptor
    fsync = "no-op"
except OSError:
    fsync = "real"
Path(os.environ["BIO_NEB_TEST_BUILDER_ENV"]).write_text(
    json.dumps({**os.environ, "fsync": fsync, "builder": sys.argv[1]})
)
fail = os.environ.get("BIO_NEB_TEST_BUILDER_FAIL")
receipt = {
    "key": "k" * 64,
    "states": {"executed-default": {"status": "built"}},
    "failed": None,
}
if fail:
    receipt["failed"] = {"state": fail, "reason": "RuntimeError('builder broke')"}
(store / "receipt.json").write_text(json.dumps(receipt))
raise SystemExit(1 if fail else 0)
"""


def _kill_if_alive(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


class PrerequisiteBuilderPhaseTests(unittest.TestCase):
    """The batch builds its catalog states once, before any module, and names the store."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base = Path(directory.name).resolve()
        self.root = base / "batch"
        self.builder_environment = base / "builder-env.json"
        environment = patch.dict(
            os.environ,
            {
                locations.TESTING_RUN_ROOTS_OVERRIDE: str(base),
                "BIO_NEB_TEST_BUILDER_ENV": str(self.builder_environment),
                run_sandbox.PREREQUISITE_STORE: "/earlier/store",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        program = patch.object(
            testing_operation, "_PREREQUISITE_PROGRAM", _FAKE_BUILDER
        )
        program.start()
        self.addCleanup(program.stop)

    def _main(self, module: str, run) -> tuple[int, str, str]:
        with (
            patch.object(testing_operation, "_run_batch", side_effect=run),
            patch("sys.stdout") as stdout,
            patch("sys.stderr") as stderr,
        ):
            result = testing_operation.main(["batch", module, "--root", str(self.root)])
        return (
            result,
            "".join(call.args[0] for call in stdout.write.call_args_list),
            "".join(call.args[0] for call in stderr.write.call_args_list),
        )

    def test_a_store_user_gets_the_store_built_first_and_named_while_modules_run(
        self,
    ) -> None:
        seen: dict[str, object] = {}

        def run(*arguments, **keywords) -> int:
            seen["built_first"] = (self.root / "prerequisites/receipt.json").is_file()
            seen["store"] = keywords["store"]
            seen["controller"] = os.environ.get(run_sandbox.PREREQUISITE_STORE)
            return 0

        result, stdout, _ = self._main("tests.test_run_prerequisites", run)

        self.assertEqual(result, 0)
        self.assertEqual(
            seen,
            {
                "built_first": True,
                "store": self.root / "prerequisites",
                "controller": "/earlier/store",
            },
        )
        printed = json.loads(
            next(
                line
                for line in stdout.splitlines()
                if line.startswith('{"prerequisites"')
            )
        )
        self.assertEqual(
            printed["prerequisites"],
            {
                "store": str(self.root / "prerequisites"),
                "key": "k" * 64,
                "states": {"executed-default": "built"},
            },
        )
        builder = json.loads(self.builder_environment.read_text())
        # The module name comes from the intent, never a literal: a dotted literal
        # here would make this module a store user in the derived import graph.
        named = testing_operation._load_ownership()["prerequisite_builder"]
        self.assertEqual(builder["builder"], testing_operation.module_name(named))
        self.assertNotIn(run_sandbox.PREREQUISITE_STORE, builder)
        sandbox = Path(builder["TMPDIR"]).parent
        self.assertTrue(sandbox.name.startswith(run_sandbox.PRIVATE_PREFIX))
        self.assertEqual(Path(builder["BIO_NEB_LOCAL_STATE"]).parent, sandbox)
        self.assertEqual(
            builder["PYTHONPYCACHEPREFIX"], str(self.root / "python-cache")
        )
        self.assertEqual(builder["fsync"], "no-op")
        self.assertFalse(sandbox.exists(), "the builder's sandbox is removed")

    def test_an_orphaned_builder_keeps_the_root_until_it_ends(self) -> None:
        started = self.root.parent / "builder-started"
        sleeper = (
            "import os, sys, time\n"
            "from pathlib import Path\n"
            f"Path({str(started)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(60)\n"
        )
        controller = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "from pathlib import Path\n"
                "from types import SimpleNamespace\n"
                "from bio_neb.devtools import testing\n"
                "testing._PREREQUISITE_PROGRAM = sys.argv[2]\n"
                "users = {'tests/run_prerequisites.py': {'tests/test_users.py'}}\n"
                "testing._import_graph = lambda: SimpleNamespace(\n"
                "    users=SimpleNamespace(reach=users))\n"
                "testing._build_prerequisites(['tests.test_users'], Path(sys.argv[1]))\n",
                str(self.root),
                sleeper,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(controller.kill)
        deadline = time.monotonic() + 30
        while not started.exists() or not started.read_text():
            self.assertLess(time.monotonic(), deadline, "builder never started")
            time.sleep(0.05)
        builder = int(started.read_text())
        self.addCleanup(_kill_if_alive, builder)
        controller.kill()
        controller.wait()

        with (self.root / "ledger.lock").open("a") as lock:
            with self.assertRaises(
                BlockingIOError, msg="orphaned builder lost the root"
            ):
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.kill(builder, signal.SIGKILL)
            deadline = time.monotonic() + 30
            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self.assertLess(time.monotonic(), deadline, "root never released")
                    time.sleep(0.05)

    def test_a_failed_builder_refuses_the_batch_with_one_named_failure(self) -> None:
        os.environ["BIO_NEB_TEST_BUILDER_FAIL"] = "executed-vcf"
        run = Mock(return_value=0)

        result, _, stderr = self._main("tests.test_run_prerequisites", run)

        self.assertEqual(result, 2)
        self.assertIn(
            "prerequisite builder failed (executed-vcf: RuntimeError('builder broke'));"
            " no module ran",
            stderr,
        )
        run.assert_not_called()

    def test_the_builder_is_the_module_the_ownership_intent_names(self) -> None:
        ownership = testing_operation._load_ownership()
        support = "tests/terminal_journey_fakes.py"
        ownership["prerequisite_builder"] = support
        seen: dict[str, object] = {}

        def run(*arguments, **keywords) -> int:
            seen["store"] = keywords["store"]
            return 0

        with patch.object(testing_operation, "_load_ownership", return_value=ownership):
            result, _, _ = self._main("tests.test_cli", run)

        self.assertEqual(result, 0)
        self.assertEqual(seen["store"], self.root / "prerequisites")
        builder = json.loads(self.builder_environment.read_text())
        self.assertEqual(builder["builder"], testing_operation.module_name(support))

    def test_an_ownership_builder_that_is_no_support_module_is_refused(
        self,
    ) -> None:
        for named in ("tests/test_cli.py", "tests/absent.py", None):
            with self.subTest(named=named):
                ownership = testing_operation._load_ownership()
                ownership["prerequisite_builder"] = named
                run = Mock(return_value=0)

                with patch.object(
                    testing_operation, "_load_ownership", return_value=ownership
                ):
                    result, _, stderr = self._main("tests.test_cli", run)

                self.assertEqual(result, 2)
                self.assertIn("prerequisite_builder must name", stderr)
                run.assert_not_called()

    def test_a_selection_no_module_of_which_uses_the_store_builds_nothing(
        self,
    ) -> None:
        seen: dict[str, object] = {}

        def run(*arguments, **keywords) -> int:
            seen["store"] = keywords["store"]
            return 0

        result, stdout, _ = self._main("tests.test_lifecycle", run)

        self.assertEqual((result, seen), (0, {"store": None}))
        self.assertFalse(self.builder_environment.exists())
        self.assertNotIn("prerequisites", stdout)


_ENVIRONMENT_PROBE = """
import json, os, sys
try:
    closed = os.open(os.devnull, os.O_RDONLY)
    os.close(closed)
    os.fsync(closed)  # a real fsync refuses a closed descriptor
    fsync = "no-op"
except OSError:
    fsync = "real"
json.dump({**os.environ, "fsync": fsync}, open(sys.argv[1], "w"))
"""


class CommandSandboxTests(unittest.TestCase):
    """Closure, checkpoint, lint and measure commands start through `_run_command`."""

    def _probe(self, phase: str) -> dict[str, str]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        seen = Path(directory.name) / "environment.json"
        result = testing_operation._run_command(
            [sys.executable, "-c", _ENVIRONMENT_PROBE, str(seen)], phase=phase
        )
        self.assertEqual(result["returncode"], 0)
        return json.loads(seen.read_text())

    def test_a_test_command_runs_in_a_fresh_run_sandbox(self) -> None:
        for phase in ("tests.scoped", "tests.regression", "measure.tests.x.0"):
            with self.subTest(phase=phase):
                environment = self._probe(phase)

                sandbox = Path(environment["TMPDIR"]).parent
                self.assertTrue(sandbox.name.startswith(run_sandbox.PRIVATE_PREFIX))
                self.assertFalse(sandbox.exists())
                self.assertEqual(
                    Path(environment["BIO_NEB_TESTING_RUN_ROOTS"]).parent, sandbox
                )
                self.assertEqual(Path(environment["HOME"]).parent, sandbox)
                self.assertEqual(environment["fsync"], "no-op")
                self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
                self.assertEqual(environment["UV_NO_CACHE"], "1")

    def test_a_test_child_queues_privately_inside_its_controllers_slots(self) -> None:
        # A runner a test starts (the runner's own tests) must never wait on the host
        # slots its controller holds, and a re-execution must still see them.
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    locations.TEST_QUEUE_OVERRIDE: directory,
                    test_queue.HELD: str(test_queue.CHECKPOINT_WIDTH),
                },
            ),
        ):
            child = self._probe("tests.scoped")
            suite = self._probe(testing_operation._CHECKPOINT_SUITE)

        sandbox = Path(child["TMPDIR"]).parent
        self.assertEqual(Path(child[locations.TEST_QUEUE_OVERRIDE]).parent, sandbox)
        self.assertNotIn(test_queue.HELD, child)
        self.assertEqual(suite[locations.TEST_QUEUE_OVERRIDE], directory)
        self.assertEqual(suite[test_queue.HELD], str(test_queue.CHECKPOINT_WIDTH))

    def test_the_pinned_ruff_reads_the_developers_uv_cache_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            with patch.dict(os.environ, {"HOME": str(home)}):
                for name in ("XDG_CACHE_HOME", "UV_CACHE_DIR", "UV_NO_CACHE"):
                    os.environ.pop(name, None)
                with run_sandbox.run_sandbox(uv_cache=True) as sandbox:
                    environment = sandbox.environment

        self.assertEqual(environment["UV_CACHE_DIR"], str(home / ".cache" / "uv"))
        self.assertNotIn("UV_NO_CACHE", environment)
        self.assertNotEqual(environment["HOME"], str(home))

    def test_a_sandbox_and_a_sandbox_inside_it_keep_the_developers_home_reads(
        self,
    ) -> None:
        # What a child reads from the developer's home survives the private HOME: both
        # global git configuration files, uv's configuration and the Nebius location. A
        # batch module running the runner (its own tests) makes a sandbox inside one.
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            (home / ".config" / "git").mkdir(parents=True)
            (home / ".config" / "uv").mkdir()
            (home / ".nebius" / "bin").mkdir(parents=True)
            (home / ".gitconfig").write_text("[probe]\n\thome = present\n")
            (home / ".config" / "git" / "config").write_text(
                "[probe]\n\txdg = present\n"
            )
            # A setting uv refuses, so a uv that read this file exits 2.
            (home / ".config" / "uv" / "uv.toml").write_text("probe-setting = 1\n")
            nebius = home / ".nebius" / "bin" / "nebius"
            nebius.write_text("")
            uv = str(locations.uv_executable())
            with patch.dict(os.environ, {"HOME": str(home)}):
                # Every override of these reads goes, so only the scratch home answers.
                for name in (
                    "XDG_CONFIG_HOME",
                    "XDG_CACHE_HOME",
                    "UV_CACHE_DIR",
                    "UV_CONFIG_FILE",
                    "UV_NO_CONFIG",
                    "GIT_CONFIG_GLOBAL",
                    locations.NEBIUS_EXECUTABLE_OVERRIDE,
                ):
                    os.environ.pop(name, None)
                with run_sandbox.run_sandbox() as outer:
                    with patch.dict(os.environ, outer.environment, clear=True):
                        with run_sandbox.run_sandbox(uv_cache=True) as inner:
                            for name, sandbox in (("outer", outer), ("inner", inner)):
                                with self.subTest(name):
                                    self._assert_home_reads(sandbox, home, nebius, uv)

    def _assert_home_reads(
        self, sandbox: run_sandbox.RunSandbox, home: Path, nebius: Path, uv: str
    ) -> None:
        environment = sandbox.environment
        self.assertNotEqual(environment["HOME"], str(home))

        def read(*argv: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                argv,
                cwd=sandbox.tmp,  # outside any repository
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        # An ordinary lookup, as git's own reads do (`--global` ignores includes).
        self.assertEqual(
            [
                read("git", "config", "--get", key).stdout.strip()
                for key in ("probe.home", "probe.xdg")
            ],
            ["present", "present"],
        )
        configured = read(uv, "--offline", "cache", "dir")
        self.assertEqual(configured.returncode, 2, configured.stderr)
        self.assertIn("probe-setting", configured.stderr)
        located = read(
            sys.executable,
            "-c",
            "from bio_neb import locations; print(locations.nebius_executable())",
        )
        self.assertEqual(located.stdout.strip(), str(nebius))
        self.assertEqual(environment["UV_CACHE_DIR"], str(home / ".cache" / "uv"))

    def test_the_checkpoint_suite_is_a_controller_with_the_real_locations(
        self,
    ) -> None:
        # The runner started from a developer's shell, even when this test itself runs
        # inside a batch module's sandbox.
        sandboxed = (
            "PYTHONPATH",
            "TMPDIR",
            locations.LOCAL_STATE_OVERRIDE,
            locations.CONTROLLER_HOME_OVERRIDE,
            locations.TESTING_RUN_ROOTS_OVERRIDE,
        )
        with patch.dict(os.environ):
            for name in sandboxed:
                os.environ.pop(name, None)
            environment = self._probe(testing_operation._CHECKPOINT_SUITE)

        self.assertEqual(environment["fsync"], "real")
        self.assertEqual(
            [name for name in sandboxed if name in environment], [], "no sandbox"
        )


class MeasureCommandTests(unittest.TestCase):
    """`testing measure` through `main`: real discovery, statistics and receipt.

    Only each module's child process is scripted, so a sample's return code and
    elapsed time are known inputs rather than wall-clock observations.
    """

    _ELAPSED = (4.0, 1.0, 2.0, 3.0)

    def _measure(self, failing: str | None = None) -> tuple[int, dict, list[dict]]:
        calls: list[list[str]] = []

        def scripted(argv, *, phase):
            index = sum(call == list(argv) for call in calls)
            calls.append(list(argv))
            return {
                "phase": phase,
                "argv": list(argv),
                "returncode": int(argv[-1] == failing and index == 2),
                "elapsed_seconds": self._ELAPSED[index],
            }

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        run_root = Path(directory.name).resolve()
        receipt = run_root / "timings.json"
        output = io.StringIO()
        with (
            patch.object(testing_operation, "_run_command", scripted),
            redirect_stdout(output),
        ):
            result = testing_operation.main(
                ["measure", "--receipt", str(receipt), "--run-root", str(run_root)]
            )
        lines = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(
            lines[-1]["timing_receipt_sha256"],
            hashlib.sha256(receipt.read_bytes()).hexdigest(),
        )
        return result, json.loads(receipt.read_bytes()), lines

    def test_measure_times_every_module_cold_then_warm_and_fails_on_any_red_sample(
        self,
    ) -> None:
        repository = Path(testing_operation.__file__).resolve().parents[3]
        inventory = sorted(
            f"tests/{path.name}" for path in (repository / "tests").glob("test_*.py")
        )
        result, document, lines = self._measure()
        self.assertEqual(result, 0)
        self.assertEqual(document["result"], "passed")
        self.assertEqual(document["module_inventory"], inventory)
        self.assertEqual([item["path"] for item in document["modules"]], inventory)
        self.assertEqual(lines[-2]["completed_modules"], len(inventory))
        first = document["modules"][0]
        module = first["path"].removesuffix(".py").replace("/", ".")
        self.assertEqual(
            [(sample["sample"], sample["argv"]) for sample in first["samples"]],
            [
                (label, [sys.executable, "-m", "unittest", "-q", module])
                for label in ("cold", "warm-1", "warm-2", "warm-3")
            ],
        )
        # Warm samples 1.0, 2.0 and 3.0 against a 4.0 cold sample.
        self.assertEqual(
            {key: first["warm"][key] for key in ("minimum_seconds", "maximum_seconds")},
            {"minimum_seconds": 1.0, "maximum_seconds": 3.0},
        )
        self.assertEqual(first["warm"]["median_seconds"], 2.0)
        self.assertEqual(first["warm"]["mean_seconds"], 2.0)
        self.assertEqual(first["cold_to_warm_median_ratio"], 2.0)

        last = inventory[-1].removesuffix(".py").replace("/", ".")
        result, document, _ = self._measure(failing=last)
        self.assertEqual(result, 1)
        self.assertEqual(document["result"], "failed")
        self.assertEqual(
            [sample["returncode"] for sample in document["modules"][-1]["samples"]],
            [0, 0, 1, 0],
        )


if __name__ == "__main__":
    unittest.main()
