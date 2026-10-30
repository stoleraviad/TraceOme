"""The host test queue: slots taken before any test child starts, shared by every checkout.

Each test uses its own queue directory and real holder processes, so the kernel's lock
release on exit, the wait and the reclaim are the operating system's own behaviour rather
than doubles of the queue's mechanics.
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from bio_neb import locations
from bio_neb.devtools import test_queue
from bio_neb.devtools import testing as testing_operation

_REPOSITORY = Path(__file__).resolve().parents[1]

_HOLDER = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from bio_neb.devtools import test_queue

    with test_queue.slots(int(sys.argv[2]), queue=Path(sys.argv[1]), holder="holder"):
        print("held", flush=True)
        sys.stdin.readline()
    """
)


_PROBE_CASE = textwrap.dedent(
    """
    import unittest

    class Probe(unittest.TestCase):
        def test_probe(self):
            pass
    """
)


class _QueueTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.queue = Path(directory.name) / "test-queue"
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop(test_queue.HELD, None)

    def hold(self, count: int) -> subprocess.Popen[str]:
        """A separate process holding `count` slots until its stdin gets a line."""
        holder = subprocess.Popen(
            [sys.executable, "-c", _HOLDER, str(self.queue), str(count)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(self.end, holder)
        self.assertEqual(holder.stdout.readline(), "held\n")
        return holder

    def end(self, holder: subprocess.Popen[str]) -> None:
        if holder.poll() is None:
            holder.kill()
        holder.wait()
        for stream in (holder.stdin, holder.stdout):
            stream.close()

    def request(self, count: int) -> tuple[threading.Event, dict[str, object]]:
        """Request `count` slots in a thread; the event is set once they are held.

        `result["stream"]` collects its heartbeats, so a test can wait until it waits.
        """
        acquired = threading.Event()
        result: dict[str, object] = {"stream": io.StringIO()}

        def take() -> None:
            with test_queue.slots(
                count,
                queue=self.queue,
                holder="request",
                stream=result["stream"],
                heartbeat=0.1,
            ) as held:
                result["waited"] = held.waited_seconds
                acquired.set()

        thread = threading.Thread(target=take, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 10)
        return acquired, result

    def beats(self, stream: io.StringIO, at_least: int) -> list[dict[str, object]]:
        """The heartbeats once `at_least` were written: the request is waiting."""
        deadline = time.monotonic() + 30
        while len(stream.getvalue().splitlines()) < at_least:
            self.assertLess(time.monotonic(), deadline, "the request never waited")
            time.sleep(0.05)
        return [
            json.loads(line)["test_queue"] for line in stream.getvalue().splitlines()
        ]


class HostQueueTests(_QueueTestCase):
    def test_a_request_waits_while_the_host_is_full_and_starts_when_the_holder_releases(
        self,
    ):
        holder = self.hold(test_queue.CAPACITY)
        acquired, result = self.request(1)
        # Its first heartbeat proves the request is waiting; its clock started before.
        self.beats(result["stream"], 1)
        self.assertFalse(acquired.wait(0.5), "a slot was taken from a full host")
        holder.stdin.write("release\n")
        holder.stdin.flush()
        self.assertTrue(acquired.wait(10), "the released slot was never taken")
        self.assertGreaterEqual(result["waited"], 0.5)

    def test_a_killed_holders_slots_are_reclaimed_and_the_waiting_request_proceeds(
        self,
    ):
        holder = self.hold(test_queue.CAPACITY)
        acquired, _ = self.request(test_queue.CAPACITY)
        self.assertFalse(acquired.wait(0.5))
        holder.send_signal(signal.SIGKILL)
        self.assertTrue(acquired.wait(10), "a killed holder's slots stayed taken")

    def test_a_request_waits_for_all_its_slots_rather_than_starting_smaller(self):
        holder = self.hold(1)
        acquired, _ = self.request(test_queue.CAPACITY)
        self.assertFalse(
            acquired.wait(1.0), "a request started on fewer slots than it asked for"
        )
        holder.stdin.write("release\n")
        holder.stdin.flush()
        self.assertTrue(acquired.wait(10))


class SoleRunnerTests(_QueueTestCase):
    """#455 (owner, 2026-10-24): a request that is the host's sole runner runs wider."""

    def test_a_request_alone_on_the_host_takes_its_sole_runner_width(self):
        with test_queue.slots(4, queue=self.queue, holder="r", sole=8) as held:
            self.assertEqual((held.count, len(held.fds)), (8, 8))
            self.assertEqual(os.environ[test_queue.HELD], "8")

    def test_a_request_beside_another_holder_takes_its_own_count(self):
        self.hold(1)
        with test_queue.slots(4, queue=self.queue, holder="r", sole=8) as held:
            self.assertEqual((held.count, len(held.fds)), (4, 4))

    def test_a_re_executed_runner_takes_the_width_its_controller_holds(self):
        for inherited, width in (("8", 8), ("4", 4)):
            os.environ[test_queue.HELD] = inherited
            with (
                self.subTest(inherited=inherited),
                test_queue.slots(4, queue=self.queue, holder="r", sole=8) as held,
            ):
                self.assertEqual((held.count, held.inherited), (width, True))


class HeartbeatTests(_QueueTestCase):
    def test_waiting_prints_a_heartbeat_naming_who_holds_the_slots(self):
        holder = self.hold(test_queue.CAPACITY)
        acquired, result = self.request(2)
        beats = self.beats(result["stream"], 3)
        holder.stdin.write("release\n")
        holder.stdin.flush()
        self.assertTrue(acquired.wait(10))
        self.assertEqual(beats[0]["waiting_for"], 2)
        self.assertEqual(
            beats[0]["holders"],
            [{"pid": holder.pid, "holder": "holder", "slots": test_queue.CAPACITY}],
        )
        self.assertGreater(beats[-1]["waited_seconds"], beats[0]["waited_seconds"])

    def test_a_request_that_finds_free_slots_prints_nothing(self):
        stream = io.StringIO()
        with test_queue.slots(
            test_queue.CAPACITY, queue=self.queue, holder="request", stream=stream
        ) as held:
            self.assertEqual(len(held.fds), test_queue.CAPACITY)
        self.assertEqual(stream.getvalue(), "")


class _Interrupting(io.StringIO):
    """A heartbeat stream that delivers Ctrl-C at the first heartbeat."""

    def write(self, text: str) -> int:
        raise KeyboardInterrupt


class _RefusingWait(io.StringIO):
    """A stderr that delivers Ctrl-C at a queue heartbeat, so a wait cannot hang a test."""

    def write(self, text: str) -> int:
        if '"test_queue"' in text:
            raise KeyboardInterrupt
        return super().write(text)


class ReleaseTests(_QueueTestCase):
    def test_an_interrupt_while_waiting_returns_the_gate_and_the_partial_slots(self):
        holder = self.hold(2)
        with (
            self.assertRaises(KeyboardInterrupt),
            test_queue.slots(
                test_queue.CAPACITY,
                queue=self.queue,
                holder="r",
                stream=_Interrupting(),
            ),
        ):
            self.fail("slots were granted on a host holding two")
        self.assertNotIn(test_queue.HELD, os.environ)
        started = time.monotonic()
        with test_queue.slots(2, queue=self.queue, holder="next") as held:
            self.assertEqual(len(held.fds), 2)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertIsNone(holder.poll())

    def test_a_request_beyond_the_host_capacity_is_refused_before_it_waits(self):
        for count in (0, test_queue.CAPACITY + 1):
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(
                    test_queue.QueueRefused, f"from 1 to {test_queue.CAPACITY}"
                ),
                test_queue.slots(count, queue=self.queue, holder="r"),
            ):
                pass
        self.assertFalse(self.queue.exists())


class InheritedSlotsTests(_QueueTestCase):
    def test_a_holder_names_its_slots_to_the_runners_it_re_executes(self):
        with test_queue.slots(3, queue=self.queue, holder="controller") as held:
            self.assertEqual(os.environ[test_queue.HELD], "3")
            self.assertFalse(held.inherited)
        self.assertNotIn(test_queue.HELD, os.environ)

    def test_a_re_executed_runner_takes_no_second_slots_on_a_full_host(self):
        self.hold(test_queue.CAPACITY)
        os.environ[test_queue.HELD] = str(test_queue.CAPACITY)
        started = time.monotonic()
        with test_queue.slots(2, queue=self.queue, holder="suite batch") as held:
            self.assertEqual(held.fds, ())
            self.assertTrue(held.inherited)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(os.environ[test_queue.HELD], str(test_queue.CAPACITY))

    def test_a_re_execution_asking_beyond_its_inherited_slots_is_refused(self):
        os.environ[test_queue.HELD] = "1"
        with (
            self.assertRaisesRegex(
                test_queue.QueueRefused, "inherits 1 slot.*asks for 2"
            ),
            test_queue.slots(2, queue=self.queue, holder="suite batch"),
        ):
            pass


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class RunnerQueueTests(_QueueTestCase):
    """The testing runner takes host slots before it starts any child (#391)."""

    def setUp(self):
        super().setUp()
        runs = tempfile.TemporaryDirectory()
        self.addCleanup(runs.cleanup)
        self.runs = Path(runs.name).resolve()
        os.environ[locations.TEST_QUEUE_OVERRIDE] = str(self.queue)
        os.environ[locations.TESTING_RUN_ROOTS_OVERRIDE] = str(self.runs)

    def runner(self, *arguments: str) -> subprocess.Popen[str]:
        runner = subprocess.Popen(
            [sys.executable, "-m", "bio_neb.devtools.testing", *arguments],
            cwd=_REPOSITORY,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(runner.kill)
        return runner

    def release(self, holder: subprocess.Popen[str]) -> None:
        holder.stdin.write("release\n")
        holder.stdin.flush()

    def until_waiting(self, read) -> str:
        """Read the runner's stderr until its first queue heartbeat: it is waiting."""
        deadline, seen = time.monotonic() + 60, ""
        while '"test_queue"' not in seen:
            self.assertLess(time.monotonic(), deadline, f"never waited: {seen}")
            seen += read()
        return seen

    def in_thread(self, call) -> tuple[threading.Event, dict[str, object]]:
        """Run `call` in a thread; `result["stderr"]` collects what it writes there."""
        done, result = threading.Event(), {"stderr": io.StringIO()}

        def run() -> None:
            with patch("sys.stdout"), patch("sys.stderr", result["stderr"]):
                result["returned"] = call()
            done.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 30)
        return done, result

    def test_a_batch_starts_no_module_until_the_host_has_all_its_slots(self):
        holder = self.hold(test_queue.CAPACITY - 1)
        root = self.runs / "batch"
        runner = self.runner(
            "batch", "tests.test_locations", "--root", str(root), "--concurrency", "2"
        )
        self.until_waiting(runner.stderr.readline)
        time.sleep(1.0)
        self.assertIsNone(runner.poll(), "the batch ran on one free slot of two")
        self.assertFalse((root / "logs").exists(), "a child started before its slots")
        self.release(holder)
        _, stderr = runner.communicate(timeout=120)

        self.assertEqual(runner.returncode, 0, stderr)
        ledger = [
            json.loads(line)
            for line in (root / "ledger.jsonl").read_text().splitlines()
        ]
        self.assertEqual(ledger[0]["schema"], "bio-neb.testing-batch-ledger/5")
        self.assertEqual(ledger[0]["concurrency"], 2)
        self.assertGreaterEqual(ledger[-1]["slot_wait_seconds"], 1.0)

    def test_a_focused_run_waits_for_one_slot_and_returns_unittests_result(self):
        holder = self.hold(test_queue.CAPACITY)
        passing = self.runner(
            "focused",
            "tests.test_locations.LocationDefaultsTests"
            ".test_symlinked_home_resolves_to_canonical_paths",
        )
        self.until_waiting(passing.stderr.readline)
        self.assertIsNone(passing.poll(), "a focused run took a slot from a full host")
        self.release(holder)
        _, stderr = passing.communicate(timeout=60)
        self.assertEqual(passing.returncode, 0, stderr)
        self.assertIn("Ran 1 test", stderr)

        failing = self.runner("focused", "tests.test_no_such_module")
        _, stderr = failing.communicate(timeout=60)
        self.assertEqual(failing.returncode, 1, stderr)

    def _probe_module(self, body: str) -> tuple[Path, Path]:
        """A test module on the runner's search path; returns its directory and record."""
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, True))
        record = directory / "record.json"
        (directory / "focused_probe_test.py").write_text(
            textwrap.dedent(
                """
                import json, os, sys, time, unittest

                class Probe(unittest.TestCase):
                    def test_probe(self):
                """
            )
            + textwrap.indent(textwrap.dedent(body), " " * 8)
        )
        os.environ["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(directory), os.environ.get("PYTHONPATH")])
        )
        os.environ["FOCUSED_PROBE_RECORD"] = str(record)
        return directory, record

    def test_a_focused_test_runs_in_its_own_run_sandbox(self):
        _, record = self._probe_module(
            """
            json.dump(
                {
                    "HOME": os.environ["HOME"],
                    "TMPDIR": os.environ["TMPDIR"],
                    "queue": os.environ.get("BIO_NEB_TEST_QUEUE"),
                    "held": os.environ.get("BIO_NEB_TEST_QUEUE_HELD"),
                },
                open(os.environ["FOCUSED_PROBE_RECORD"], "w"),
            )
            """
        )
        focused = self.runner("focused", "focused_probe_test")
        _, stderr = focused.communicate(timeout=60)

        self.assertEqual(focused.returncode, 0, stderr)
        seen = json.loads(record.read_text())
        sandbox = Path(seen["TMPDIR"]).parent
        self.assertTrue(sandbox.name.startswith("bio-neb-batch-"), seen)
        self.assertEqual(Path(seen["HOME"]).parent, sandbox)
        self.assertEqual(Path(seen["queue"]).parent, sandbox)
        self.assertIsNone(seen["held"])

    def test_a_stopped_focused_run_ends_its_test_before_its_slot_returns(self):
        _, record = self._probe_module(
            """
            open(os.environ["FOCUSED_PROBE_RECORD"], "w").write(str(os.getpid()))
            time.sleep(120)
            """
        )
        focused = self.runner("focused", "focused_probe_test")
        deadline = time.monotonic() + 60
        while not record.exists() or not record.read_text():
            self.assertLess(
                time.monotonic(), deadline, "the focused test never started"
            )
            time.sleep(0.1)
        test_pid = int(record.read_text())

        focused.send_signal(signal.SIGTERM)
        focused.communicate(timeout=60)
        with test_queue.slots(
            test_queue.CAPACITY, queue=self.queue, holder="next", stream=io.StringIO()
        ):
            with self.assertRaises(
                ProcessLookupError, msg="the test outlived its slot"
            ):
                os.kill(test_pid, 0)
        self.assertNotEqual(focused.returncode, 0)

    def test_a_focused_test_that_ignores_sigterm_is_killed_within_the_grace(self):
        _, record = self._probe_module(
            """
            import signal
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            open(os.environ["FOCUSED_PROBE_RECORD"], "w").write(str(os.getpid()))
            time.sleep(120)
            """
        )
        focused = self.runner("focused", "focused_probe_test")
        deadline = time.monotonic() + 60
        while not record.exists() or not record.read_text():
            self.assertLess(
                time.monotonic(), deadline, "the focused test never started"
            )
            time.sleep(0.1)
        test_pid = int(record.read_text())

        stopped = time.monotonic()
        focused.send_signal(signal.SIGTERM)
        focused.communicate(timeout=60)

        self.assertLess(
            time.monotonic() - stopped,
            testing_operation._TIMEOUT_GRACE_SECONDS + 15,
            "the stop waited on a test that ignores SIGTERM",
        )
        with self.assertRaises(ProcessLookupError, msg="the test outlived its slot"):
            os.kill(test_pid, 0)
        with test_queue.slots(
            test_queue.CAPACITY, queue=self.queue, holder="next", stream=io.StringIO()
        ) as held:
            self.assertLess(held.waited_seconds, 1.0)

    def test_a_second_stop_during_the_grace_still_ends_the_test_first(self):
        _, record = self._probe_module(
            """
            import signal
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            open(os.environ["FOCUSED_PROBE_RECORD"], "w").write(str(os.getpid()))
            time.sleep(120)
            """
        )
        focused = self.runner("focused", "focused_probe_test")
        deadline = time.monotonic() + 60
        while not record.exists() or not record.read_text():
            self.assertLess(
                time.monotonic(), deadline, "the focused test never started"
            )
            time.sleep(0.1)
        test_pid = int(record.read_text())

        focused.send_signal(signal.SIGTERM)
        time.sleep(1.0)  # inside the grace: the test ignores the first stop
        focused.send_signal(signal.SIGTERM)
        focused.communicate(timeout=60)

        with self.assertRaises(ProcessLookupError, msg="the test outlived its slot"):
            os.kill(test_pid, 0)

    def test_a_stop_before_the_focused_test_starts_still_reaches_it(self):
        # A SIGTERM between taking the slot and starting the test is held and replayed.
        _, record = self._probe_module(
            """
            open(os.environ["FOCUSED_PROBE_RECORD"], "w").write(str(os.getpid()))
            time.sleep(120)
            """
        )
        launch = subprocess.Popen

        def stopped_launch(*arguments, **keywords):
            os.kill(os.getpid(), signal.SIGTERM)
            return launch(*arguments, **keywords)

        started = time.monotonic()
        with patch.object(
            testing_operation.subprocess, "Popen", side_effect=stopped_launch
        ):
            returned = testing_operation._focused(["focused_probe_test"], False)

        self.assertEqual(returned, 128 + signal.SIGTERM)
        self.assertLess(time.monotonic() - started, 60, "the early stop was lost")
        if record.exists() and record.read_text():
            with self.assertRaises(ProcessLookupError):
                os.kill(int(record.read_text()), 0)

    def test_a_focused_run_names_its_tests(self):
        refused = self.runner("focused")
        _, stderr = refused.communicate(timeout=60)
        self.assertEqual(refused.returncode, 2, stderr)
        self.assertIn("name the tests to run", stderr)

    def test_a_whole_suite_focused_run_is_refused_toward_the_batch(self):
        # Every slot is held: a run that queued, or started unittest after its slot,
        # would still be waiting when the communicate below times out.
        holder = self.hold(test_queue.CAPACITY)
        self.addCleanup(self.release, holder)
        for arguments in (["--all"], ["--all", "tests.test_locations"]):
            with self.subTest(arguments=arguments):
                refused = self.runner("focused", *arguments)
                _, stderr = refused.communicate(timeout=60)
                self.assertEqual(refused.returncode, 2, stderr)
                self.assertIn("testing batch --all --root <batch-root>", stderr)
                self.assertNotIn('"test_queue"', stderr)

    def _probe_modules(self, *bodies: str) -> tuple[Path, list[str]]:
        """Test modules beside each other on the runner's search path, each recording
        its pid and when it ran into its own record; returns their directory and names."""
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, True))
        names = []
        for index, body in enumerate(bodies):
            name = f"focused_probe_{index}_test"
            (directory / f"{name}.py").write_text(
                textwrap.dedent(
                    f"""
                    import json, os, time, unittest

                    class Probe(unittest.TestCase):
                        def test_probe(self):
                            started = time.monotonic()
                            time.sleep(1.5)
                            record = os.path.join({str(directory)!r}, "{name}.json")
                            json.dump(
                                {{"pid": os.getpid(), "started": started,
                                  "ended": time.monotonic()}},
                                open(record, "w"),
                            )
                    """
                )
                + textwrap.indent(textwrap.dedent(body), " " * 8)
            )
            names.append(name)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(directory), os.environ.get("PYTHONPATH")])
        )
        return directory, names

    def test_a_focused_run_across_modules_runs_one_process_per_module_at_once(self):
        # Several modules run in parallel, one process per module at the batch
        # width; one module stays one process.
        directory, names = self._probe_modules("pass", "pass")
        focused = self.runner(
            "focused", f"{names[0]}.Probe.test_probe", f"{names[1]}.Probe"
        )
        _, stderr = focused.communicate(timeout=120)

        self.assertEqual(focused.returncode, 0, stderr)
        records = [
            json.loads((directory / f"{name}.json").read_text()) for name in names
        ]
        self.assertNotEqual(records[0]["pid"], records[1]["pid"])
        # Both modules' tests overlapped in time: they ran at once.
        self.assertLess(
            max(item["started"] for item in records),
            min(item["ended"] for item in records),
        )
        for name in names:
            self.assertIn(f"== {name}: exit 0", stderr)
        self.assertEqual(stderr.count("Ran 1 test"), 2)

    def test_a_red_module_fails_the_run_and_its_sibling_still_reports(self):
        _, names = self._probe_modules("pass", "self.fail('red on purpose')")
        focused = self.runner("focused", names[0], names[1])
        _, stderr = focused.communicate(timeout=120)

        self.assertEqual(focused.returncode, 1, stderr)
        self.assertIn(f"== {names[0]}: exit 0", stderr)
        self.assertIn(f"== {names[1]}: exit 1", stderr)
        self.assertIn("red on purpose", stderr)

    def test_a_focused_run_groups_spelled_paths_and_sibling_modules_by_module(self):
        # A name's module is the one the child imports, however it is spelled: here a
        # regular package, which Python imports before the same-named file beside it,
        # so its two submodules are two processes and the file beside it none.
        directory, _ = self._probe_module("pass")
        package = directory / "focused_probe_package"
        package.mkdir()
        for path in (
            directory / "focused_probe_package.py",
            package / "__init__.py",
            package / "left.py",
            package / "right.py",
        ):
            path.write_text(_PROBE_CASE)
        focused = self.runner(
            "focused",
            "focused_probe_package.left.Probe",
            "focused_probe_package.right.Probe.test_probe",
        )
        _, stderr = focused.communicate(timeout=120)
        self.assertEqual(focused.returncode, 0, stderr)
        self.assertIn("== focused_probe_package.left: exit 0", stderr)
        self.assertIn("== focused_probe_package.right: exit 0", stderr)

    def test_a_focused_run_takes_ids_of_a_package_module_or_a_split_namespace(self):
        # Tests in a package's `__init__`, and a module a later root adds to a namespace
        # package, nested or not, are one module each, as the child imports them.
        directory, _ = self._probe_module("pass")
        package = directory / "focused_probe_package"
        package.mkdir()
        (package / "__init__.py").write_text(_PROBE_CASE)
        later = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(later, True))
        for root in (directory, later):
            (root / "focused_probe_namespace" / "nested").mkdir(parents=True)
        (later / "focused_probe_namespace" / "selected.py").write_text(_PROBE_CASE)
        (later / "focused_probe_namespace" / "nested" / "inner.py").write_text(
            _PROBE_CASE
        )
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [os.environ["PYTHONPATH"], str(later)]
        )
        for arguments in (
            ["focused_probe_package.Probe", "focused_probe_package.Probe.test_probe"],
            [
                "focused_probe_namespace.selected.Probe",
                "focused_probe_namespace.selected.Probe.test_probe",
            ],
            [
                "focused_probe_namespace.nested.inner.Probe",
                "focused_probe_namespace.nested.inner.Probe.test_probe",
            ],
        ):
            with self.subTest(arguments=arguments):
                focused = self.runner("focused", *arguments)
                _, stderr = focused.communicate(timeout=60)
                self.assertEqual(focused.returncode, 0, stderr)
                self.assertIn("Ran 2 tests", stderr)

    def test_a_focused_run_refuses_unittests_discovery_spellings(self):
        # `discover`, or options with no test name, make unittest discover the suite.
        holder = self.hold(test_queue.CAPACITY)
        self.addCleanup(self.release, holder)
        for arguments in (["discover"], ["DISCOVER"], ["--", "-v"]):
            with self.subTest(arguments=arguments):
                refused = self.runner("focused", *arguments)
                _, stderr = refused.communicate(timeout=60)
                self.assertEqual(refused.returncode, 2, stderr)
                self.assertIn("testing batch --all --root <batch-root>", stderr)
                self.assertNotIn('"test_queue"', stderr)

    def test_a_focused_run_may_name_several_tests_of_one_module(self):
        _, record = self._probe_module(
            """
            open(os.environ["FOCUSED_PROBE_RECORD"], "a").write("ran\\n")
            """
        )
        focused = self.runner(
            "focused", "focused_probe_test.Probe.test_probe", "focused_probe_test.Probe"
        )
        _, stderr = focused.communicate(timeout=60)
        self.assertEqual(focused.returncode, 0, stderr)
        self.assertEqual(record.read_text(), "ran\nran\n")

    def _closure(self, *arguments: str):
        """`main` with each command recording the slots its controller holds."""
        seen: list[str | None] = []

        def command(argv, *, phase):
            seen.append(os.environ.get(test_queue.HELD))
            return {
                "phase": phase,
                "argv": list(argv),
                "returncode": 0,
                "elapsed_seconds": 0,
            }

        receipt = self.runs / "receipt.json"
        commands = patch.object(testing_operation, "_run_command", side_effect=command)
        clean = patch.object(
            testing_operation,
            "_run_clean_gate",
            return_value={
                "phase": "repository.clean",
                "returncode": 0,
                "elapsed_seconds": 0,
            },
        )
        commands.start()
        clean.start()
        self.addCleanup(commands.stop)
        self.addCleanup(clean.stop)
        done, result = self.in_thread(
            lambda: testing_operation.main(
                [*arguments, "--run-root", str(self.runs), "--receipt", str(receipt)]
            )
        )
        return done, result, seen, receipt

    def test_a_checkpoint_waits_for_its_eight_slots_and_records_the_wait(self):
        # A checkpoint takes a fixed width, not the whole host.
        self.assertEqual(test_queue.CHECKPOINT_WIDTH, 8)
        holder = self.hold(test_queue.CAPACITY - test_queue.CHECKPOINT_WIDTH + 1)
        done, result, seen, receipt = self._closure(
            "checkpoint", "--base-sha", _head(), "--root", str(self.runs / "suite")
        )
        self.until_waiting(lambda: time.sleep(0.1) or result["stderr"].getvalue())
        time.sleep(1.0)
        self.assertFalse(done.is_set(), "a checkpoint started short of its width")
        self.assertEqual(seen, [])
        self.release(holder)
        self.assertTrue(done.wait(60))

        self.assertEqual(result["returned"], 0)
        self.assertEqual(set(seen), {str(test_queue.CHECKPOINT_WIDTH)})
        recorded = json.loads(receipt.read_text())
        self.assertEqual(recorded["schema_version"], "bio-neb.testing-result/3")
        self.assertGreaterEqual(recorded["slot_wait_seconds"], 1.0)

    def test_a_checkpoint_starts_beside_the_slots_it_leaves_open(self):
        holder = self.hold(test_queue.CAPACITY - test_queue.CHECKPOINT_WIDTH)
        done, result, seen, receipt = self._closure(
            "checkpoint", "--base-sha", _head(), "--root", str(self.runs / "suite")
        )
        self.assertTrue(done.wait(60), "a checkpoint waited on a host with its width")
        self.assertIsNone(holder.poll())

        self.assertEqual(result["returned"], 0)
        self.assertEqual(set(seen), {str(test_queue.CHECKPOINT_WIDTH)})
        recorded = json.loads(receipt.read_text())
        self.assertLess(recorded["slot_wait_seconds"], 1.0)

    def test_a_closure_receipt_takes_one_slot(self):
        holder = self.hold(test_queue.CAPACITY - 1)
        done, result, seen, receipt = self._closure(
            "core", "--ticket-id", "391", "--base-sha", _head()
        )
        self.assertTrue(done.wait(60), "a one-slot closure waited on a free slot")
        self.assertIsNone(holder.poll())

        self.assertEqual(result["returned"], 0)
        self.assertEqual(seen, ["1"])
        recorded = json.loads(receipt.read_text())
        self.assertEqual(recorded["schema_version"], "bio-neb.testing-result/3")
        self.assertLess(recorded["slot_wait_seconds"], 1.0)

    def test_a_capacity_experiment_above_the_host_slots_waits_for_an_empty_host(self):
        # `batch` installs its stop handlers, so it runs on the main thread here and
        # a helper releases the holder once the batch reports that it waits.
        seen: list[str | None] = []
        stderr = io.StringIO()

        def batch(*arguments, **keywords):
            seen.append(os.environ.get(test_queue.HELD))
            return 0

        holder = self.hold(1)
        released = threading.Event()

        def release_when_waiting() -> None:
            deadline = time.monotonic() + 60
            while '"test_queue"' not in stderr.getvalue():
                if time.monotonic() > deadline:
                    return
                time.sleep(0.1)
            released.set()
            self.release(holder)

        helper = threading.Thread(target=release_when_waiting, daemon=True)
        helper.start()
        self.addCleanup(helper.join, 70)
        concurrency = test_queue.CAPACITY + 2
        with (
            patch("os.cpu_count", return_value=concurrency),
            patch.object(testing_operation, "_run_batch", side_effect=batch),
            patch("sys.stdout"),
            patch("sys.stderr", stderr),
        ):
            returned = testing_operation.main(
                [
                    "batch",
                    "tests.test_locations",
                    "--root",
                    str(self.runs / "x"),
                    "--concurrency",
                    str(concurrency),
                    "--parallel-experiment",
                    "--detect-writes",
                ]
            )

        self.assertTrue(released.is_set(), "the experiment never waited")
        self.assertEqual(returned, 0, stderr.getvalue())
        self.assertEqual(seen, [str(test_queue.CAPACITY)])

    def test_a_default_batch_starts_beside_another_on_the_sixteen_slots(self):
        # Sixteen host slots; a batch's default width stays 4, so a second default
        # batch does not wait.
        self.assertEqual(test_queue.CAPACITY, 16)
        seen: list[str | None] = []

        def batch(*arguments, **keywords):
            seen.append(os.environ.get(test_queue.HELD))
            return 0

        self.hold(4)
        stderr = _RefusingWait()
        with (
            patch("os.cpu_count", return_value=24),
            patch.object(testing_operation, "_run_batch", side_effect=batch),
            patch("sys.stdout"),
            patch("sys.stderr", stderr),
        ):
            returned = testing_operation.main(
                ["batch", "tests.test_locations", "--root", str(self.runs / "second")]
            )

        self.assertEqual(returned, 0, stderr.getvalue())
        self.assertEqual(seen, ["4"])

    def _default_batch(self, root: Path, stderr: io.StringIO) -> tuple[int, list]:
        """`batch` without --concurrency; each `_run_batch` call's HELD and width."""
        seen: list = []

        def batch(*arguments, **keywords):
            seen.append((os.environ.get(test_queue.HELD), keywords["concurrency"]))
            return 0

        with (
            patch("os.cpu_count", return_value=24),
            patch.object(testing_operation, "_run_batch", side_effect=batch),
            patch("sys.stdout"),
            patch("sys.stderr", stderr),
        ):
            returned = testing_operation.main(
                ["batch", "tests.test_locations", "--root", str(root)]
            )
        return returned, seen

    def test_a_default_batch_alone_on_the_host_runs_eight_wide(self):
        # "sole runner we do 8 concerany".
        stderr = _RefusingWait()
        returned, seen = self._default_batch(self.runs / "alone", stderr)

        self.assertEqual(returned, 0, stderr.getvalue())
        self.assertEqual(seen, [("8", 8)])

    def test_a_resumed_batch_waits_for_the_width_its_root_bound(self):
        # Its header binds its width and shards, so a resume beside another runner
        # waits for the eight slots rather than being refused at four.
        root = self.runs / "resumed"
        root.mkdir()
        (root / "ledger.jsonl").write_text(
            json.dumps({"schema": "bio-neb.testing-batch-ledger/5", "concurrency": 8})
            + "\n"
        )
        holder = self.hold(test_queue.CAPACITY - 7)
        stderr = io.StringIO()

        def release_when_waiting() -> None:
            deadline = time.monotonic() + 60
            while '"test_queue"' not in stderr.getvalue():
                if time.monotonic() > deadline:
                    return
                time.sleep(0.1)
            self.release(holder)

        helper = threading.Thread(target=release_when_waiting, daemon=True)
        helper.start()
        self.addCleanup(helper.join, 70)
        returned, seen = self._default_batch(root, stderr)

        self.assertIn('"waiting_for": 8', stderr.getvalue())
        self.assertEqual(returned, 0, stderr.getvalue())
        self.assertEqual(seen, [("8", 8)])

    def test_a_snapshot_alone_on_the_host_holds_the_sole_width_for_its_runner(self):
        seen: list[str | None] = []
        forwarded: list[list[str]] = []

        def snapshot(commit, root, checkout, argv, owner_fd):
            seen.append(os.environ.get(test_queue.HELD))
            forwarded.append(list(argv))
            return 0

        with (
            patch("os.cpu_count", return_value=24),
            patch.object(testing_operation, "_run_snapshot", side_effect=snapshot),
            patch("sys.stdout"),
            patch("sys.stderr", _RefusingWait()),
        ):
            returned = testing_operation.main(
                ["batch", "tests.test_locations", "--root", str(self.runs / "s8")]
                + ["--at", "HEAD"]
            )

        self.assertEqual(returned, 0)
        self.assertEqual(seen, ["8"])
        # The snapshot's runner chooses no width of its own: it runs at what it inherits.
        self.assertNotIn("--concurrency", forwarded[0])

    def test_a_snapshot_run_holds_its_batchs_slots_for_the_runner_it_starts(self):
        seen: list[str | None] = []

        def snapshot(commit, root, checkout, argv, owner_fd):
            seen.append(os.environ.get(test_queue.HELD))
            return 0

        holder = self.hold(test_queue.CAPACITY - 1)
        with patch.object(testing_operation, "_run_snapshot", side_effect=snapshot):
            done, result = self.in_thread(
                lambda: testing_operation.main(
                    [
                        "batch",
                        "tests.test_locations",
                        "--root",
                        str(self.runs / "s"),
                        "--at",
                        "HEAD",
                        "--concurrency",
                        "2",
                    ]
                )
            )
            self.until_waiting(lambda: time.sleep(0.1) or result["stderr"].getvalue())
            self.assertFalse(done.is_set(), "the snapshot ran on one slot of two")
            self.release(holder)
            self.assertTrue(done.wait(60))

        self.assertEqual(result["returned"], 0)
        self.assertEqual(seen, ["2"])
        # The snapshot's runner inherits the slots and may predate the queue, so the
        # wait is retained beside its ledger by the command that waited.
        (waited,) = [
            json.loads(line)
            for line in (self.runs / "s" / "slot-waits.jsonl").read_text().splitlines()
        ]
        self.assertEqual(waited["slots"], 2)
        self.assertGreater(waited["slot_wait_seconds"], 0.0)

    def test_a_snapshot_experiment_above_the_host_slots_takes_all_of_them(self):
        seen: list[str | None] = []
        forwarded: list[list[str]] = []

        def snapshot(commit, root, checkout, argv, owner_fd):
            seen.append(os.environ.get(test_queue.HELD))
            forwarded.append(list(argv))
            return 0

        concurrency = str(test_queue.CAPACITY + 2)
        with (
            patch("os.cpu_count", return_value=test_queue.CAPACITY + 2),
            patch.object(testing_operation, "_run_snapshot", side_effect=snapshot),
            patch("sys.stdout"),
            patch("sys.stderr"),
        ):
            returned = testing_operation.main(
                [
                    "batch",
                    "tests.test_locations",
                    "--root",
                    str(self.runs / "x"),
                    "--at",
                    "HEAD",
                    "--concurrency",
                    concurrency,
                    "--parallel-experiment",
                    "--detect-writes",
                ]
            )

        self.assertEqual(returned, 0)
        self.assertEqual(seen, [str(test_queue.CAPACITY)])
        self.assertIn(concurrency, forwarded[0], "the experiment's own schedule")


if __name__ == "__main__":
    unittest.main()
