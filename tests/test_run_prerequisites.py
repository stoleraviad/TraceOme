"""The prerequisite interface: lazy builds, pristine reuse, shared reads, leaks fail.

Most cases replace the builder with a small registered Journey so they check the
sharing, restore and leak contract of `prerequisite(state, **choices)`, not
execution; modules that consume the real prerequisites own the execution claims.
`SharedPrerequisiteTests` and `ModuleScopeTests` cover read-only states owned by a
class or a module, `RestoredPrerequisiteTests` states restored in place,
`ForkedPrerequisiteTests` states forked from the shared Run while its registered
inputs stay outside their root, and `PrerequisiteStoreTests` the batch catalog
adapter. `RestoredRunTests` checks the real review-pending, approved, executed
and planned-evidence states survive a restore after their bytes were changed.
"""

import functools
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from bio_neb.application.advancement import use_run_advancement_composition
from bio_neb.journey import Journey
from tests import prerequisite_store, run_prerequisites
from tests.run_prerequisites import (
    RunPrerequisites,
    State,
    _tree,
    prerequisite,
)


class _Builder:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, object]] = []
        self.journeys: dict[Path, Journey] = {}

    def __call__(self, root: Path, *, output_payloads=None) -> Path:
        self.calls.append((root, output_payloads))
        journey = Journey.create(root / "runs")
        self.journeys[journey.path] = journey
        return journey.path


def _without_store(test: unittest.TestCase) -> None:
    """Run the case as an exact test run does: no batch store is named."""
    environment = patch.dict(os.environ)
    environment.start()
    test.addCleanup(environment.stop)
    os.environ.pop(prerequisite_store.STORE, None)
    prerequisite_store.forget()
    test.addCleanup(prerequisite_store.forget)


def _run(*tests: unittest.TestCase) -> unittest.TestResult:
    result = unittest.TextTestResult(io.StringIO(), descriptions=False, verbosity=0)
    unittest.TestSuite(tests).run(result)
    return result


def _problems(result: unittest.TestResult) -> dict[str, str]:
    return {
        test.id().rsplit(".", 1)[-1]: trace
        for test, trace in (*result.failures, *result.errors)
    }


class SharedPrerequisiteTests(unittest.TestCase):
    def setUp(self) -> None:
        _without_store(self)
        self.builder = _Builder()
        builder = patch("tests.run_prerequisites.execute_terminal", self.builder)
        builder.start()
        self.addCleanup(builder.stop)

    def test_class_builds_each_variant_once_and_removes_it_after_the_class(
        self,
    ) -> None:
        seen: dict[str, Path] = {}

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_a(self) -> None:
                seen["a"] = self.prerequisite("executed").journey_file

            def test_b(self) -> None:
                seen["b"] = self.prerequisite("executed").journey_file
                seen["b-again"] = self.prerequisite("executed").journey_file

            def test_payload(self) -> None:
                seen["payload"] = self.prerequisite(
                    "executed", output_payloads={"VCF": b"x"}
                ).journey_file

            def test_slot(self) -> None:
                seen["slot"] = self.prerequisite(
                    "executed", slot="independent"
                ).journey_file

        result = _run(
            *(
                Consumer(name)
                for name in sorted(Consumer.__dict__)
                if name.startswith("test_")
            )
        )

        self.assertEqual(_problems(result), {})
        self.assertEqual(seen["a"], seen["b"])
        self.assertEqual(seen["b"], seen["b-again"])
        self.assertEqual(
            len({seen["a"], seen["payload"], seen["slot"]}), 3, "variants share"
        )
        self.assertEqual(
            sorted(payloads is not None for _, payloads in self.builder.calls),
            [False, False, True],
        )
        self.assertFalse(any(path.exists() for path in seen.values()))
        self.assertNotIn("_shared_prerequisites", Consumer.__dict__)

        _run(Consumer("test_a"))

        self.assertEqual(len(self.builder.calls), 4, "a later run must rebuild")

    def test_exact_test_builds_its_own_parent(self) -> None:
        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_only(self) -> None:
                self.assertTrue(self.prerequisite("executed").journey_file.is_file())

            def test_other(self) -> None:
                raise AssertionError("not selected")

        result = _run(Consumer("test_only"))

        self.assertEqual((result.testsRun, _problems(result)), (1, {}))
        self.assertEqual(len(self.builder.calls), 1)

    def test_case_that_changes_the_parent_fails_and_the_next_gets_a_fresh_one(
        self,
    ) -> None:
        seen: dict[str, Path] = {}

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_1_appends(self) -> None:
                seen["1"] = self.prerequisite("executed").journey_file
                with seen["1"].open("ab") as handle:
                    handle.write(b"{}\n")

            def test_2_adds_empty_file(self) -> None:
                seen["2"] = self.prerequisite("executed").journey_file
                (seen["2"].parent / "added-empty").write_bytes(b"")

            def test_3_reads(self) -> None:
                seen["3"] = self.prerequisite("executed").journey_file

        result = _run(
            Consumer("test_1_appends"),
            Consumer("test_2_adds_empty_file"),
            Consumer("test_3_reads"),
        )

        problems = _problems(result)
        self.assertEqual(sorted(problems), ["test_1_appends", "test_2_adds_empty_file"])
        self.assertIn(
            "shared executed prerequisite changed", problems["test_1_appends"]
        )
        self.assertIn("/journey.jsonl'", problems["test_1_appends"])
        self.assertIn("added-empty'", problems["test_2_adds_empty_file"])
        self.assertEqual(len(self.builder.calls), 3, "a changed parent is rebuilt")
        self.assertEqual(len(set(seen.values())), 3)

    def test_live_writer_bound_composition_and_leftover_thread_fail_the_case(
        self,
    ) -> None:
        release = threading.Event()
        self.addCleanup(release.set)
        journeys = self.builder.journeys

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_writer(self) -> None:
                held = []
                self.addCleanup(lambda: held and held[0].__exit__(None, None, None))
                writer = journeys[
                    self.prerequisite("executed").journey_file
                ].processing()
                writer.__enter__()
                held.append(writer)

            def test_composition(self) -> None:
                bound = use_run_advancement_composition(object())
                self.addCleanup(bound.__exit__, None, None, None)
                bound.__enter__()
                self.prerequisite("executed")

            def test_thread(self) -> None:
                worker = threading.Thread(target=release.wait, name="left-running")
                self.addCleanup(worker.join)
                self.addCleanup(release.set)
                self.prerequisite("executed")
                worker.start()

        result = _run(
            Consumer("test_writer"),
            Consumer("test_composition"),
            Consumer("test_thread"),
        )

        problems = _problems(result)
        self.assertIn("live writer", problems["test_writer"])
        self.assertIn("composition left bound", problems["test_composition"])
        self.assertIn("left-running", problems["test_thread"])


class _Registration:
    """A registered Run: the Journey records its prompt, runtime file and FASTQs."""

    def __init__(self) -> None:
        self.roots: list[Path] = []

    def __call__(self, root: Path) -> tuple[Path, str]:
        from tests.planning_assignment_fakes import configured_planning_run

        self.roots.append(root)
        receipt, _, _ = configured_planning_run(root)
        journey_file = Path(receipt.journey_file)
        return journey_file, Journey.open(journey_file).journey_id


def _restored(test: RunPrerequisites, build: _Registration, slot: str = "default"):
    return test.prerequisite(State("registered", build), slot=slot)


def _corrupt(root: Path, journey_file: Path) -> None:
    """Change Journey, artifact and contact bytes and add work-directory files."""
    with journey_file.open("ab") as handle:
        handle.write(b"{}\n")
    for path in journey_file.parent.rglob("*"):
        if path.is_file() and path != journey_file:
            path.write_bytes(b"corrupted")
    obstruction = root / "execution" / "cpu" / "request.json"
    obstruction.parent.mkdir(parents=True)
    obstruction.write_bytes(b"obstruction")
    (root / "scratch.json").write_bytes(b"{}")
    (journey_file.parent / "added-empty").write_bytes(b"")


def _canary_pair(order: tuple[int, int]) -> tuple[unittest.TestResult, _Registration]:
    """Two cases that each check pristine state and then corrupt it."""
    build = _Registration()
    seen: dict[str, object] = {}

    class Canary(RunPrerequisites, unittest.TestCase):
        def check_pristine(self):
            baseline = _restored(self, build)
            state = (
                baseline.root,
                baseline.journey_file,
                _tree(baseline.root),
                (baseline.root / "r1.fastq").stat().st_ino,
            )
            self.assertEqual(seen.setdefault("state", state), state)
            return baseline

        def test_1_corrupts(self) -> None:
            baseline = self.check_pristine()
            _corrupt(baseline.root, baseline.journey_file)

        def test_2_corrupts_too(self) -> None:
            baseline = self.check_pristine()
            _corrupt(baseline.root, baseline.journey_file)

    names = {1: "test_1_corrupts", 2: "test_2_corrupts_too"}
    return _run(*(Canary(names[index]) for index in order)), build


def _changed_input(change) -> tuple[unittest.TestResult, _Registration, list[str]]:
    """One case changes a registered FASTQ; the next requests the baseline."""
    build = _Registration()
    ids: list[str] = []

    class Inputs(RunPrerequisites, unittest.TestCase):
        def test_1_changes_input(self) -> None:
            baseline = _restored(self, build)
            ids.append(baseline.value)
            change(baseline.root / "r1.fastq")

        def test_2_reads(self) -> None:
            baseline = _restored(self, build)
            ids.append(baseline.value)
            self.assertEqual(
                Journey.open(baseline.journey_file).journey_id, baseline.value
            )
            self.assertEqual(
                sorted(path.name for path in baseline.root.iterdir()),
                ["prompt.txt", "r1.fastq", "r2.fastq", "runs", "runtime.json"],
            )

    result = _run(Inputs("test_1_changes_input"), Inputs("test_2_reads"))
    return result, build, ids


class RestoredPrerequisiteTests(unittest.TestCase):
    def test_canary_pair_sees_pristine_bytes_at_the_same_path_in_either_order(
        self,
    ) -> None:
        for order in ((1, 2), (2, 1)):
            with self.subTest(order=order):
                result, build = _canary_pair(order)

                self.assertEqual(_problems(result), {})
                self.assertEqual(len(build.roots), 1, "a restore must not rebuild")

    def test_each_scenario_of_one_test_gets_a_restored_baseline(self) -> None:
        build = _Registration()

        class Scenarios(RunPrerequisites, unittest.TestCase):
            def test_scenarios(self) -> None:
                pristine = None
                for _ in range(3):
                    baseline = _restored(self, build)
                    pristine = pristine or _tree(baseline.root)
                    self.assertEqual(_tree(baseline.root), pristine)
                    _corrupt(baseline.root, baseline.journey_file)

        self.assertEqual(_problems(_run(Scenarios("test_scenarios"))), {})
        self.assertEqual(len(build.roots), 1)

    def test_changed_registered_input_forces_a_rebuild_at_the_same_path(
        self,
    ) -> None:
        def replace(path: Path) -> None:
            path.with_suffix(".new").write_bytes(path.read_bytes())
            path.with_suffix(".new").replace(path)

        def rewrite(path: Path) -> None:
            path.write_bytes(path.read_bytes() + b"\n")

        def chmod(path: Path) -> None:
            os.chmod(path, 0o400)

        def remove(path: Path) -> None:
            path.unlink()

        for change in (replace, rewrite, chmod, remove):
            with self.subTest(change=change.__name__):
                result, build, ids = _changed_input(change)

                self.assertEqual(_problems(result), {})
                self.assertEqual(len(build.roots), 2, "a changed input must rebuild")
                self.assertEqual(build.roots[0], build.roots[1])
                self.assertNotEqual(ids[0], ids[1])

    def test_exact_test_builds_its_own_and_class_cleanup_removes_it(self) -> None:
        build = _Registration()

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_only(self) -> None:
                self.assertTrue(_restored(self, build).journey_file.is_file())

            def test_other(self) -> None:
                raise AssertionError("not selected")

        result = _run(Consumer("test_only"))

        self.assertEqual((result.testsRun, _problems(result)), (1, {}))
        self.assertEqual(len(build.roots), 1)
        self.assertFalse(build.roots[0].exists())
        self.assertNotIn("_restored_prerequisites", Consumer.__dict__)

        _run(Consumer("test_only"))

        self.assertEqual(len(build.roots), 2, "a later run must rebuild")

    def test_slots_are_independent_runs(self) -> None:
        build = _Registration()
        seen = []

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_two_runs(self) -> None:
                seen.append(_restored(self, build))
                seen.append(_restored(self, build, slot="newer"))

        self.assertEqual(_problems(_run(Consumer("test_two_runs"))), {})
        self.assertNotEqual(seen[0].root, seen[1].root)
        self.assertNotEqual(seen[0].value, seen[1].value)

    def test_choices_of_different_types_are_different_runs(self) -> None:
        build = _Registration()
        seen: list[str] = []

        def typed(root: Path, *, choice) -> tuple[Path, str]:
            return build(root)[0], type(choice).__name__

        state = State("typed-choice", typed)
        choices = (
            {"key": "value"},
            [("key", "value")],
            (("key", "value"),),
            1,
            True,
            {1: "value"},
            {True: "value"},
        )

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_each_choice(self) -> None:
                for choice in choices:
                    seen.append(self.prerequisite(state, choice=choice).value)
                    seen.append(self.prerequisite(state, choice=choice).value)

        self.assertEqual(_problems(_run(Consumer("test_each_choice"))), {})
        self.assertEqual(
            seen,
            ["dict", "dict", "list", "list", "tuple", "tuple", "int", "int"]
            + ["bool", "bool", "dict", "dict", "dict", "dict"],
        )
        self.assertEqual(len(build.roots), 7, "one build per distinct choice")

    def test_a_choice_equal_to_the_builders_default_is_the_same_run(self) -> None:
        build = _Registration()
        seen: list[Path] = []

        def defaulted(root: Path, *, choice=None, **overrides) -> tuple[Path, None]:
            return build(root)[0], None

        state = State("defaulted", defaulted)

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_spellings(self) -> None:
                seen.append(self.prerequisite(state).journey_file)
                seen.append(self.prerequisite(state, choice=None).journey_file)

        self.assertEqual(_problems(_run(Consumer("test_spellings"))), {})
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(len(build.roots), 1, "a defaulted choice must not rebuild")

    def test_a_choice_the_builder_does_not_take_is_refused(self) -> None:
        build = _Registration()

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_unknown(self) -> None:
                self.prerequisite(State("registered", build), unknown=1)

        problems = _problems(_run(Consumer("test_unknown")))

        self.assertIn("unexpected keyword argument 'unknown'", problems["test_unknown"])
        self.assertEqual(build.roots, [])

    def test_an_interface_owned_argument_is_refused_even_after_a_build(self) -> None:
        build = _Registration()
        refusals: list[str] = []

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_root_choice(self) -> None:
                state = State("registered", build)
                self.prerequisite(state)
                with self.assertRaises(TypeError) as refused:
                    self.prerequisite(state, root=Path("/invalid-choice"))
                refusals.append(str(refused.exception))

        self.assertEqual(_problems(_run(Consumer("test_root_choice"))), {})
        self.assertIn("multiple values for argument 'root'", refusals[0])
        self.assertEqual(len(build.roots), 1)

    def test_an_unhashable_choice_is_refused(self) -> None:
        build = _Registration()

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_set(self) -> None:
                self.prerequisite(
                    State("set-choice", lambda root, *, choice: build(root)),
                    choice={1},
                )

        problems = _problems(_run(Consumer("test_set")))

        self.assertIn("unhashable", problems["test_set"])
        self.assertEqual(build.roots, [])

    def test_failed_build_is_not_served_and_the_next_request_builds_again(
        self,
    ) -> None:
        build = _Registration()
        attempts = []

        def flaky(root: Path):
            attempts.append(root)
            if len(attempts) == 1:
                raise RuntimeError("build interrupted")
            return build(root)

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_1_build_fails(self) -> None:
                self.prerequisite(State("flaky", flaky))

            def test_2_builds(self) -> None:
                self.assertTrue(
                    self.prerequisite(State("flaky", flaky)).journey_file.is_file()
                )

        problems = _problems(
            _run(Consumer("test_1_build_fails"), Consumer("test_2_builds"))
        )

        self.assertEqual(sorted(problems), ["test_1_build_fails"])
        self.assertIn("build interrupted", problems["test_1_build_fails"])
        self.assertEqual(len(attempts), 2)

    def test_live_writer_or_leftover_thread_is_refused(self) -> None:
        build = _Registration()
        release = threading.Event()
        self.addCleanup(release.set)

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_writer(self) -> None:
                held = []
                self.addCleanup(lambda: held and held[0].__exit__(None, None, None))
                writer = Journey.open(_restored(self, build).journey_file).processing()
                writer.__enter__()
                held.append(writer)
                _restored(self, build)

            def test_thread(self) -> None:
                worker = threading.Thread(target=release.wait, name="left-running")
                self.addCleanup(worker.join)
                self.addCleanup(release.set)
                _restored(self, build)
                worker.start()
                _restored(self, build)

            def test_composition(self) -> None:
                bound = use_run_advancement_composition(object())
                self.addCleanup(bound.__exit__, None, None, None)
                bound.__enter__()
                _restored(self, build)

        result = _run(
            Consumer("test_writer"),
            Consumer("test_thread"),
            Consumer("test_composition"),
        )

        problems = _problems(result)
        self.assertIn(
            "live writer still owns the restored baseline", problems["test_writer"]
        )
        self.assertIn("left-running", problems["test_thread"])
        self.assertIn("composition left bound", problems["test_composition"])


class _ParentRun:
    """A registered Run standing in for the executed parent."""

    def __init__(self) -> None:
        self.roots: list[Path] = []

    def __call__(self, root: Path, *, output_payloads=None) -> Path:
        from tests.planning_assignment_fakes import configured_planning_run

        self.roots.append(root)
        receipt, _, _ = configured_planning_run(root)
        return Path(receipt.journey_file)


class _Fork:
    """A child under root that records a Run's registered inputs where they are.

    Copying the Run directory keeps the absolute input paths of its Journey, as
    a product Fork inherits the parent's registration.
    """

    def __init__(self, source: Path | None = None) -> None:
        self.source = source
        self.roots: list[Path] = []

    def __call__(self, root: Path, parent_file: Path) -> tuple[Path, Path]:
        import shutil

        self.roots.append(root)
        run = (self.source or parent_file).parent
        shutil.copytree(run, root / "runs" / run.name)
        return root / "runs" / run.name / parent_file.name, parent_file


def _forked_pair(order: tuple[int, int]) -> tuple[unittest.TestResult, _Fork]:
    """Two cases that each check a pristine forked child and then corrupt it.

    The shared parent's byte check runs after each case, so a restore that
    touched the parent fails it.
    """
    fork = _Fork()
    seen: dict[str, object] = {}

    class Consumer(RunPrerequisites, unittest.TestCase):
        def check_pristine(self):
            baseline = self.prerequisite(State("child", fork, forks_from="executed"))
            state = (baseline.root, _tree(baseline.root))
            self.assertEqual(seen.setdefault("state", state), state)
            return baseline

        def test_1_corrupts(self) -> None:
            baseline = self.check_pristine()
            _corrupt(baseline.root, baseline.journey_file)

        def test_2_corrupts_too(self) -> None:
            baseline = self.check_pristine()
            _corrupt(baseline.root, baseline.journey_file)

    names = {1: "test_1_corrupts", 2: "test_2_corrupts_too"}
    return _run(*(Consumer(names[index]) for index in order)), fork


class ForkedPrerequisiteTests(unittest.TestCase):
    def setUp(self) -> None:
        _without_store(self)
        self.parent = _ParentRun()
        builder = patch("tests.run_prerequisites.execute_terminal", self.parent)
        builder.start()
        self.addCleanup(builder.stop)

    def test_forked_baseline_restores_in_either_order_and_leaves_the_parent(
        self,
    ) -> None:
        for order in ((1, 2), (2, 1)):
            with self.subTest(order=order):
                result, fork = _forked_pair(order)

                self.assertEqual(_problems(result), {})
                self.assertEqual(len(fork.roots), 1, "a restore must not rebuild")

    def test_inputs_outside_the_baseline_and_the_parent_are_refused(self) -> None:
        other = _ParentRun()

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_foreign_inputs(self) -> None:
                import tempfile

                with tempfile.TemporaryDirectory() as foreign:
                    source = other(Path(foreign).resolve())
                    self.prerequisite(
                        State("foreign", _Fork(source), forks_from="executed")
                    )

        problems = _problems(_run(Consumer("test_foreign_inputs")))

        self.assertIn(
            "registered inputs outside the baseline", problems["test_foreign_inputs"]
        )

    def test_a_replaced_parent_gives_a_new_baseline(self) -> None:
        fork = _Fork()
        parents: list[Path] = []

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_1_changes_the_parent(self) -> None:
                baseline = self.prerequisite(
                    State("child", fork, forks_from="executed")
                )
                parents.append(baseline.value)
                with baseline.value.open("ab") as handle:
                    handle.write(b"{}\n")

            def test_2_forks_again(self) -> None:
                parents.append(
                    self.prerequisite(State("child", fork, forks_from="executed")).value
                )

        problems = _problems(
            _run(Consumer("test_1_changes_the_parent"), Consumer("test_2_forks_again"))
        )

        self.assertEqual(list(problems), ["test_1_changes_the_parent"])
        self.assertIn(
            "shared executed prerequisite changed",
            problems["test_1_changes_the_parent"],
        )
        self.assertEqual(len(self.parent.roots), 2)
        self.assertNotEqual(parents[0], parents[1])
        self.assertEqual(len(fork.roots), 2, "a new parent must build a new child")


class _StateBuilder:
    """A small registered Journey per catalog state, standing in for Execution."""

    def __init__(self, fail: str | None = None) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def __call__(self, state: str, root: Path) -> Path:
        self.calls.append(state)
        if state == self.fail:
            raise RuntimeError("builder broke")
        return Journey.create(root / "runs").path

    def builders(self) -> dict:
        return {
            state: functools.partial(self, state)
            for state in ("executed-default", "executed-vcf")
        }


class PrerequisiteStoreTests(unittest.TestCase):
    """The batch catalog: built once, read-only, admitted once per process."""

    def setUp(self) -> None:
        _without_store(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Path(directory.name).resolve() / "prerequisites"
        self.addCleanup(self._writable)
        self.builder = _StateBuilder()

    def _build(self, builder: _StateBuilder | None = None) -> int:
        builder = builder or self.builder
        return prerequisite_store.build_store(str(self.store), builder.builders())

    def _writable(self) -> None:
        for path in (self.store, *self.store.rglob("*")) if self.store.exists() else ():
            if path.is_dir():
                path.chmod(0o755)

    def _receipt(self) -> dict:
        return json.loads((self.store / "receipt.json").read_text(encoding="utf-8"))

    def _name_store(self) -> None:
        os.environ[prerequisite_store.STORE] = str(self.store)

    def _consumer_builds(self) -> _Builder:
        local = _Builder()
        builder = patch("tests.run_prerequisites.execute_terminal", local)
        builder.start()
        self.addCleanup(builder.stop)
        return local

    def test_each_state_is_built_once_read_only_and_a_resume_reuses_it(self) -> None:
        self.assertEqual(self._build(), 0)
        first = self._receipt()

        self.assertEqual(self.builder.calls, ["executed-default", "executed-vcf"])
        self.assertEqual(first["key"], prerequisite_store.source_key())
        self.assertEqual(
            {name: state["status"] for name, state in first["states"].items()},
            {"executed-default": "built", "executed-vcf": "built"},
        )
        journey_file = Path(first["states"]["executed-default"]["journey_file"])
        self.assertTrue(journey_file.is_relative_to(self.store / "executed-default"))
        with self.assertRaises(PermissionError):
            journey_file.open("ab")
        with self.assertRaises(PermissionError):
            (journey_file.parent / "added").write_bytes(b"")

        self.assertEqual(self._build(), 0)

        self.assertEqual(len(self.builder.calls), 2, "a resume must not rebuild")
        self.assertEqual(
            {
                name: state["status"]
                for name, state in self._receipt()["states"].items()
            },
            {"executed-default": "reused", "executed-vcf": "reused"},
        )

    def test_a_store_built_from_other_sources_is_refused(self) -> None:
        self._build()
        with (
            patch.object(prerequisite_store, "source_key", return_value="0" * 64),
            patch("sys.stderr"),
        ):
            self.assertEqual(self._build(), 1)

        self.assertEqual(
            self._receipt()["failed"], {"state": None, "reason": "store_key_changed"}
        )
        self.assertEqual(len(self.builder.calls), 2, "a stale store is never rebuilt")

    def test_a_state_left_by_an_interrupted_build_is_built_again(self) -> None:
        (self.store / "executed-default" / "runs").mkdir(parents=True)
        (self.store / "executed-default" / "partial").write_bytes(b"half")

        self.assertEqual(self._build(), 0)

        self.assertFalse((self.store / "executed-default" / "partial").exists())
        self.assertEqual(
            self._receipt()["states"]["executed-default"]["status"], "built"
        )

    def test_a_failed_builder_names_its_state_and_serves_nothing(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "builder broke"):
            self._build(_StateBuilder(fail="executed-vcf"))

        self.assertEqual(self._receipt()["failed"]["state"], "executed-vcf")
        self._name_store()
        self.assertIsNone(prerequisite_store.admitted("executed-vcf"))

    def test_a_state_is_admitted_only_while_the_key_and_its_bytes_match(self) -> None:
        self._build()
        known = self._receipt()["states"]["executed-default"]

        self.assertIsNone(prerequisite_store.admitted("executed-default"), "no store")
        self._name_store()
        with patch("unittest.addModuleCleanup"):
            served = prerequisite_store.admitted("executed-default")
            self.assertEqual(
                served,
                (self.store / "executed-default", Path(known["journey_file"])),
            )
            self.assertIsNone(prerequisite_store.admitted("executed-other"))
            prerequisite_store.forget()
            with patch.object(prerequisite_store, "source_key", return_value="0" * 64):
                self.assertIsNone(prerequisite_store.admitted("executed-default"))
            prerequisite_store.forget()
            self._writable()
            (served.journey_file.parent / "added").write_bytes(b"")
            self.assertIsNone(prerequisite_store.admitted("executed-default"))

    def test_the_store_is_validated_once_per_process_not_per_call(self) -> None:
        self._build()
        self._name_store()
        key = patch.object(
            prerequisite_store, "source_key", wraps=prerequisite_store.source_key
        )
        manifest = patch.object(
            prerequisite_store, "_manifest", wraps=prerequisite_store._manifest
        )

        with (
            key as keyed,
            manifest as hashed,
            patch("unittest.addModuleCleanup") as checks,
        ):
            for _ in range(3):
                self.assertIsNotNone(prerequisite_store.admitted("executed-default"))
                self.assertIsNotNone(prerequisite_store.admitted("executed-vcf"))

        self.assertEqual(keyed.call_count, 1, "the source key is hashed once")
        self.assertEqual(hashed.call_count, 2, "each served state is hashed once")
        self.assertEqual(checks.call_count, 2, "one module-end check per state")

    def test_a_module_that_changed_a_served_state_fails_when_it_ends(self) -> None:
        self._build()
        self._name_store()
        cleanups: list = []
        with patch(
            "unittest.addModuleCleanup",
            lambda function, *args: cleanups.append((function, args)),
        ):
            served = prerequisite_store.admitted("executed-default")
        [(check, arguments)] = cleanups
        check(*arguments)

        self._writable()
        (served.journey_file.parent / "added").write_bytes(b"")

        with self.assertRaisesRegex(AssertionError, "executed-default changed"):
            check(*arguments)

    def test_a_store_from_other_sources_serves_nothing_and_the_class_builds(
        self,
    ) -> None:
        self._build()
        self._name_store()
        local = self._consumer_builds()
        stored = Path(self._receipt()["states"]["executed-default"]["journey_file"])
        seen: list[Path] = []

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_1(self) -> None:
                seen.append(self.prerequisite("executed").journey_file)

            def test_2(self) -> None:
                seen.append(self.prerequisite("executed").journey_file)

        with (
            patch.object(prerequisite_store, "source_key", return_value="0" * 64),
            patch("unittest.addModuleCleanup"),
        ):
            result = _run(Consumer("test_1"), Consumer("test_2"))

        self.assertEqual(_problems(result), {})
        self.assertNotIn(stored, seen, "a stale state was served")
        self.assertEqual(seen[0], seen[1], "the local parent is shared")
        self.assertEqual(len(local.calls), 1)

    def test_a_test_that_changed_a_stored_state_fails_and_the_store_is_dropped(
        self,
    ) -> None:
        self._build()
        self._name_store()
        local = self._consumer_builds()
        stored = Path(self._receipt()["states"]["executed-default"]["journey_file"])
        seen: list[Path] = []

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_1_changes(self) -> None:
                seen.append(self.prerequisite("executed").journey_file)
                seen[0].parent.chmod(0o755)
                (seen[0].parent / "added").write_bytes(b"")

            def test_2_reads(self) -> None:
                seen.append(self.prerequisite("executed").journey_file)

        with patch("unittest.addModuleCleanup"):
            result = _run(Consumer("test_1_changes"), Consumer("test_2_reads"))

        problems = _problems(result)
        self.assertEqual(list(problems), ["test_1_changes"])
        self.assertIn(
            "shared executed prerequisite changed", problems["test_1_changes"]
        )
        self.assertEqual(seen[0], stored)
        self.assertNotEqual(seen[1], stored, "a changed stored state was served")
        self.assertEqual(len(local.calls), 1)
        self.assertIsNone(prerequisite_store.admitted("executed-default"))

    def test_the_interface_forks_from_the_store_without_building(self) -> None:
        self._build()
        self._name_store()
        local = _ParentRun()
        builder = patch("tests.run_prerequisites.execute_terminal", local)
        builder.start()
        self.addCleanup(builder.stop)
        seen: dict[str, Path] = {}

        class Consumer(RunPrerequisites, unittest.TestCase):
            def test_default(self) -> None:
                seen["default"] = self.prerequisite("executed").journey_file

            def test_slot(self) -> None:
                seen["slot"] = self.prerequisite(
                    "executed", slot="independent"
                ).journey_file

            def test_writable(self) -> None:
                seen["writable"] = self.prerequisite(
                    "executed", writable=True
                ).journey_file

        with patch("unittest.addModuleCleanup"):
            result = _run(
                Consumer("test_default"),
                Consumer("test_slot"),
                Consumer("test_writable"),
            )

        self.assertEqual(_problems(result), {})
        self.assertEqual(
            seen["default"],
            Path(self._receipt()["states"]["executed-default"]["journey_file"]),
        )
        self.assertEqual(
            len(local.roots), 2, "only the independent slot and writable Run build"
        )
        self.assertTrue(seen["default"].is_file(), "class cleanup kept the store")

    def test_the_builder_entry_builds_every_stored_choice_set_of_the_catalog(
        self,
    ) -> None:
        local = self._consumer_builds()

        with patch.dict(
            run_prerequisites.CATALOG["executed"].stored,
            {"vcf": {"output_payloads": {"VCF": b"v"}}},
            clear=True,
        ):
            self.assertEqual(run_prerequisites.build_store(str(self.store)), 0)

        self.assertEqual(list(self._receipt()["states"]), ["executed-vcf"])
        self.assertEqual([payloads for _, payloads in local.calls], [{"VCF": b"v"}])
        self.assertEqual(local.calls[0][0], self.store / "executed-vcf")


class ModuleScopeTests(unittest.TestCase):
    """A helper called with no test gets a read-only Run the module owns."""

    def setUp(self) -> None:
        _without_store(self)
        self.builder = _Builder()
        builder = patch("tests.run_prerequisites.execute_terminal", self.builder)
        builder.start()
        self.addCleanup(builder.stop)
        self.cleanups: list = []
        module_cleanup = patch(
            "unittest.addModuleCleanup",
            lambda function, *args: self.cleanups.append((function, args)),
        )
        module_cleanup.start()
        self.addCleanup(module_cleanup.stop)
        self.addCleanup(self._end_module)

    def _end_module(self) -> list[str]:
        """Run the module cleanups as unittest does: last registered first."""
        problems = []
        while self.cleanups:
            function, arguments = self.cleanups.pop()
            try:
                function(*arguments)
            except AssertionError as error:
                problems.append(str(error))
        return problems

    def test_the_module_builds_once_checks_at_its_end_and_removes_it(self) -> None:
        first = prerequisite("executed").journey_file
        second = prerequisite("executed").journey_file

        self.assertEqual(first, second)
        self.assertEqual(len(self.builder.calls), 1)
        self.assertEqual(self._end_module(), [])
        self.assertFalse(first.exists(), "the module's Run outlived the module")

        prerequisite("executed")

        self.assertEqual(len(self.builder.calls), 2, "the next module builds its own")

    def test_a_changed_module_run_fails_when_the_module_ends(self) -> None:
        journey_file = prerequisite("executed").journey_file
        with journey_file.open("ab") as handle:
            handle.write(b"{}\n")

        [problem] = self._end_module()

        self.assertIn("shared executed prerequisite changed", problem)
        self.assertIn("journey.jsonl", problem)

    def test_a_failed_module_build_is_removed_when_the_module_ends(self) -> None:
        roots: list[Path] = []

        def broken(root: Path, *, output_payloads=None) -> Path:
            roots.append(root)
            (root / "partial").mkdir(parents=True)
            raise RuntimeError("build interrupted")

        with (
            patch("tests.run_prerequisites.execute_terminal", broken),
            self.assertRaisesRegex(RuntimeError, "build interrupted"),
        ):
            prerequisite("executed")

        self.assertTrue(roots[0].exists())
        self.assertEqual(self._end_module(), [])
        self.assertFalse(roots[0].exists(), "a failed build outlived the module")

        journey_file = prerequisite("executed").journey_file

        self.assertTrue(journey_file.is_file(), "the next module builds afresh")
        self.assertEqual(len(self.builder.calls), 1)

    def test_a_live_writer_bound_composition_or_leftover_thread_fails_the_module(
        self,
    ) -> None:
        release = threading.Event()
        self.addCleanup(release.set)
        journey_file = prerequisite("executed").journey_file
        writer = self.builder.journeys[journey_file].processing()
        writer.__enter__()
        self.addCleanup(writer.__exit__, None, None, None)
        bound = use_run_advancement_composition(object())
        bound.__enter__()
        self.addCleanup(bound.__exit__, None, None, None)
        worker = threading.Thread(target=release.wait, name="left-running")
        self.addCleanup(worker.join)
        self.addCleanup(release.set)
        worker.start()

        [problem] = self._end_module()

        self.assertIn("a live writer still owns the shared executed", problem)
        self.assertIn("run_advancement_composition", problem)
        self.assertIn("left-running", problem)

    def test_a_writable_run_needs_its_test(self) -> None:
        with self.assertRaisesRegex(TypeError, "needs the test that owns it"):
            prerequisite("approved")


class RestoredRunTests(RunPrerequisites, unittest.TestCase):
    """Real approved and executed baselines reopen identically after a restore."""

    def test_restored_approval_keeps_its_bound_inputs_and_reopens_identically(
        self,
    ) -> None:
        from bio_neb.journey_path import JourneyPath

        first = self.prerequisite("approved")
        approved = first.value
        inodes = [
            Path(item.registered_path).stat().st_ino
            for item in approved.proposal.execution_package.inputs
        ]
        before = JourneyPath.open(first.journey_file).state
        _corrupt(first.root, first.journey_file)

        second = self.prerequisite("approved")

        self.assertEqual(
            (second.root, second.journey_file), (first.root, first.journey_file)
        )
        self.assertEqual(
            [
                Path(item.registered_path).stat().st_ino
                for item in approved.proposal.execution_package.inputs
            ],
            inodes,
        )
        self.assertEqual(JourneyPath.open(second.journey_file).state, before)

    def test_restored_review_pending_run_reopens_at_the_same_review(self) -> None:
        from bio_neb.journey_path import JourneyPath

        first = self.prerequisite("review-pending")
        request = first.value.current_review.request
        before = JourneyPath.open(first.journey_file).state
        _corrupt(first.root, first.journey_file)

        second = self.prerequisite("review-pending")

        self.assertEqual(second.journey_file, first.journey_file)
        self.assertEqual(JourneyPath.open(second.journey_file).state, before)
        self.assertEqual(
            second.value.current_review.request.request_event_id,
            request.request_event_id,
        )

    def test_restored_executed_journey_yields_the_same_contact_review(self) -> None:
        from bio_neb.journey_path import JourneyPath
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )

        first = self.prerequisite("executed", writable=True)
        review = review_execution_contacts(str(first.journey_file))
        state = JourneyPath.open(first.journey_file).state
        _corrupt(first.root, first.journey_file)

        second = self.prerequisite("executed", writable=True)

        self.assertEqual(second.journey_file, first.journey_file)
        self.assertEqual(review.status, "complete", review.review.discrepancies)
        self.assertEqual(review_execution_contacts(str(second.journey_file)), review)
        self.assertEqual(JourneyPath.open(second.journey_file).state, state)

    def test_restored_planned_evidence_compiles_the_same_finalization_basis(
        self,
    ) -> None:
        from bio_neb.reporting.finalization_commands import (
            CompileFinalizationCommand,
            compile_finalization,
        )

        def basis(journey_file: Path):
            pending = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey_file))
            ).pending_request
            assert pending is not None
            return pending.basis

        first = self.prerequisite("planned-evidence")
        compiled = basis(first.journey_file)
        _corrupt(first.root, first.journey_file)

        second = self.prerequisite("planned-evidence")

        self.assertEqual(second.journey_file, first.journey_file)
        self.assertEqual(basis(second.journey_file), compiled)


if __name__ == "__main__":
    unittest.main()
