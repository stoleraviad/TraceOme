"""Change-impact selection derived from source and test imports.

Most tests derive a selection over a throwaway repository whose imports are chosen to show one
derivation rule each; the last class checks the rule against this repository's own imports.
"""

from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import ANY, patch

from bio_neb import locations
from bio_neb.devtools import test_queue
from bio_neb.devtools import testing as testing_operation
from bio_neb.devtools.testing_selection import ImportGraph, derive


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


_REPOSITORY = Path(testing_operation.__file__).resolve().parents[3]

# One file per derivation rule: `pkg.core` is a hub imported directly by one test and
# through `pkg.api` by another; `pkg.hidden` is reached only through `pkg.api2`; `pkg.cli`
# runs only as `python -m pkg.cli`; `pkg.orphan` is reached by nothing.
_FAKE_REPOSITORY = {
    "src/pkg/__init__.py": "",
    "src/pkg/core.py": "VALUE = 1\n",
    "src/pkg/api.py": "from pkg.core import VALUE\n",
    "src/pkg/hidden.py": "X = 1\n",
    "src/pkg/api2.py": "from . import hidden\n",
    "src/pkg/outer.py": "from pkg import api\n",
    "src/pkg/orphan.py": "Y = 1\n",
    "src/pkg/cli/__init__.py": "",
    "src/pkg/cli/__main__.py": "print('run')\n",
    "src/pkg/data.txt": "data\n",
    "tests/support_fakes.py": "from pkg import core\n",
    "tests/test_core.py": "import pkg.core\n",
    "tests/test_api.py": "from pkg.api import VALUE\n",
    "tests/test_api2.py": "from pkg import api2\n",
    "tests/test_outer.py": "import pkg.outer\n",
    "tests/test_fakes_user.py": "from tests.support_fakes import core\n",
    "tests/test_cli.py": "COMMAND = ['python', '-m', 'pkg.cli']\n",
    "tests/test_docs.py": "GUIDE = 'docs/guide.md'\n",
    "docs/guide.md": "# Guide\n",
    "docs/notes.txt": "notes\n",
    "pyproject.toml": "[project]\n",
}


class _FakeRepositoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repository = Path(directory.name).resolve()
        for path, source in _FAKE_REPOSITORY.items():
            target = self.repository / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
        self.graph = ImportGraph.build(self.repository)

    def _derive(self, *changed: str, **options):
        return derive(self.graph, changed, **options)


class DerivedSelectionTests(_FakeRepositoryTestCase):
    def test_a_changed_source_selects_its_direct_users_and_reports_wider_importers(
        self,
    ) -> None:
        derivation = self._derive("src/pkg/core.py")

        # A test uses a file when the test module or the support it imports imports it.
        self.assertEqual(
            derivation.test_paths, ("tests/test_core.py", "tests/test_fakes_user.py")
        )
        self.assertEqual(
            derivation.reasons["tests/test_fakes_user.py"], ("uses src/pkg/core.py",)
        )
        self.assertEqual(derivation.changed_path_users, {"src/pkg/core.py": 2})
        self.assertEqual(derivation.fallback_reasons, ())
        # Users of importers are reported as risk, not selected.
        self.assertEqual(
            derivation.wider,
            {
                "depth_1": (
                    "tests/test_api.py",
                    "tests/test_core.py",
                    "tests/test_fakes_user.py",
                ),
                "depth_2": (
                    "tests/test_api.py",
                    "tests/test_core.py",
                    "tests/test_fakes_user.py",
                    "tests/test_outer.py",
                ),
                "transitive": (
                    "tests/test_api.py",
                    "tests/test_core.py",
                    "tests/test_fakes_user.py",
                    "tests/test_outer.py",
                ),
            },
        )

    def test_depth_selects_users_of_importers_up_to_that_distance(self) -> None:
        derivation = self._derive("src/pkg/core.py", depth=1)

        self.assertIn("tests/test_api.py", derivation.test_paths)
        self.assertNotIn("tests/test_outer.py", derivation.test_paths)
        self.assertEqual(
            derivation.reasons["tests/test_api.py"],
            ("uses a depth-1 importer of src/pkg/core.py",),
        )

    def test_a_source_no_test_imports_selects_the_users_of_its_nearest_importers(
        self,
    ) -> None:
        derivation = self._derive("src/pkg/hidden.py")

        self.assertEqual(derivation.test_paths, ("tests/test_api2.py",))
        self.assertEqual(
            derivation.reasons["tests/test_api2.py"],
            ("uses an importer of src/pkg/hidden.py",),
        )

    def test_changed_tests_and_support_select_themselves_and_their_consumers(
        self,
    ) -> None:
        derivation = self._derive("tests/test_api.py", "tests/support_fakes.py")

        self.assertEqual(
            derivation.test_paths, ("tests/test_api.py", "tests/test_fakes_user.py")
        )
        self.assertEqual(derivation.reasons["tests/test_api.py"], ("changed",))
        self.assertEqual(
            derivation.reasons["tests/test_fakes_user.py"],
            ("uses tests/support_fakes.py",),
        )

    def test_a_module_run_with_python_dash_m_is_used_by_the_test_naming_it(
        self,
    ) -> None:
        self.assertEqual(
            self._derive("src/pkg/cli/__main__.py").test_paths, ("tests/test_cli.py",)
        )

    def test_intent_readers_select_what_imports_cannot_show(self) -> None:
        readers = {
            "docs/*.md": ["tests/test_docs.py"],
            "src/pkg/*.txt": ["src/pkg/core.py"],
            "pyproject.toml": "checkpoint",
        }
        derivation = self._derive(
            "docs/guide.md", "src/pkg/data.txt", "pyproject.toml", readers=readers
        )

        self.assertEqual(
            derivation.test_paths,
            ("tests/test_core.py", "tests/test_docs.py", "tests/test_fakes_user.py"),
        )
        self.assertEqual(
            derivation.reasons["tests/test_docs.py"], ("reads docs/guide.md",)
        )
        self.assertEqual(
            derivation.reasons["tests/test_core.py"],
            ("uses src/pkg/core.py, which reads src/pkg/data.txt",),
        )
        self.assertEqual(
            derivation.fallback_reasons,
            ("checkpoint-only changed path: pyproject.toml",),
        )

    def test_a_change_nothing_reaches_is_an_unowned_coverage_warning(self) -> None:
        derivation = self._derive("src/pkg/orphan.py", "docs/notes.txt")

        self.assertEqual(derivation.test_paths, ())
        self.assertEqual(
            derivation.fallback_reasons,
            (
                "unowned changed path: src/pkg/orphan.py",
                "unowned changed path: docs/notes.txt",
            ),
        )

    def test_required_modules_keep_their_reason_beside_derived_ones(self) -> None:
        derivation = self._derive(
            "src/pkg/core.py",
            required=[
                ("tests/test_core.py", "core"),
                ("tests/test_docs.py", "included"),
            ],
        )

        self.assertEqual(
            derivation.reasons["tests/test_core.py"], ("core", "uses src/pkg/core.py")
        )
        self.assertEqual(derivation.reasons["tests/test_docs.py"], ("included",))
        self.assertEqual(derivation.derived_by, {"core": 1, "direct": 2, "included": 1})


class IntentValidationTests(_FakeRepositoryTestCase):
    def _readers(self, paths: dict) -> dict:
        files = [path for path in _FAKE_REPOSITORY if not path.endswith("__init__.py")]
        return testing_operation._intent_readers(paths, self.graph, files)

    def test_intent_paths_accept_readers_and_checkpoint(self) -> None:
        paths = {
            "docs/*.md": ["tests/test_docs.py"],
            "src/pkg/*.txt": ["src/pkg/core.py"],
            "pyproject.toml": "checkpoint",
        }
        self._readers(paths)

    def test_intent_paths_refuse_drift_and_derivable_entries(self) -> None:
        for paths, refusal in (
            ({"docs/*.rst": ["tests/test_docs.py"]}, "matches no repository file"),
            ({"docs/*.md": []}, "must name reader modules or checkpoint"),
            ({"docs/*.md": "full"}, "must name reader modules or checkpoint"),
            ({"docs/*.md": ["docs/notes.txt"]}, "not a Python source"),
            ({"docs/*.md": ["src/pkg/orphan.py"]}, "reaches no test"),
            ({"src/pkg/core.py": ["tests/test_docs.py"]}, "imports already derive"),
        ):
            with self.subTest(paths=paths):
                with self.assertRaisesRegex(
                    testing_operation.ControlSurfaceError, refusal
                ):
                    self._readers(paths)

    def test_a_module_only_a_subprocess_reaches_may_name_its_readers(self) -> None:
        self._readers({"src/pkg/orphan.py": ["tests/test_docs.py"]})


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _main(*arguments: str, changed: tuple[str, ...] = ()) -> tuple[int, str, str]:
    """Run a derived mode for ticket 202 over a fixed change set, not the working tree."""
    output, errors = io.StringIO(), io.StringIO()
    with (
        patch.object(testing_operation, "_changes_since", return_value=changed),
        redirect_stdout(output),
        redirect_stderr(errors),
    ):
        result = testing_operation.main(
            [*arguments, "--base-sha", _head(), "--ticket-id", "202"]
        )
    return result, output.getvalue(), errors.getvalue()


def _explain(*arguments: str, changed: tuple[str, ...] = ()) -> dict:
    result, output, errors = _main(*arguments, "--explain", changed=changed)
    if result:
        raise AssertionError(f"explain exited {result}: {errors}")
    return json.loads(output)


def _direct_test_importers(module: str) -> set[str]:
    """Test modules whose own source imports `module`, read independently of the graph."""
    importers = set()
    for path in sorted((_REPOSITORY / "tests").glob("test_*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom) and not node.level
                else []
            )
            if module in names:
                importers.add(f"tests/{path.name}")
    return importers


class RepositorySelectionTests(unittest.TestCase):
    def test_current_fixture_mode_adds_every_registered_fixture_consumer(
        self,
    ) -> None:
        ownership = json.loads(
            (_REPOSITORY / "tests" / "testing_ownership.json").read_text()
        )
        registrations = [
            ownership["execution_fixture"],
            *ownership["additional_execution_fixtures"].values(),
        ]
        consumers = {test for item in registrations for test in item["tests"]}
        changed = ("src/bio_neb/devtools/stage_harness.py",)

        scoped = _explain("scoped", changed=changed)
        with_fixture = _explain("scoped-with-current-fixture", changed=changed)

        # The change alone reaches only some consumers, so the mode is what adds them.
        self.assertTrue(consumers - set(scoped["test_paths"]))
        self.assertEqual(
            set(with_fixture["test_paths"]), {*scoped["test_paths"], *consumers}
        )
        for test in consumers:
            self.assertIn("fixture consumer", with_fixture["derivation"][test], test)

    def test_a_hub_change_selects_every_direct_importer_and_reports_its_share(
        self,
    ) -> None:
        explained = _explain("core", changed=("src/bio_neb/journey.py",))

        importers = _direct_test_importers("bio_neb.journey")
        # The retired hand map selected 11 journey-truth modules for this hub.
        self.assertGreater(len(importers), 11)
        self.assertLessEqual(importers, set(explained["test_paths"]))
        discovered = len(list((_REPOSITORY / "tests").glob("test_*.py")))
        self.assertEqual(explained["discovered_test_module_count"], discovered)
        self.assertEqual(
            explained["selected_module_share"],
            round(explained["test_module_count"] / discovered, 4),
        )
        for label in ("depth_1", "depth_2", "transitive"):
            self.assertGreaterEqual(
                explained["wider_selection"][label]["test_module_count"],
                explained["test_module_count"],
            )

    def test_scoped_expansion_needs_a_reason_and_is_retained(self) -> None:
        explained = _explain(
            "scoped",
            "--include",
            "tests.test_cli",
            "--depth",
            "1",
            "--expansion-reason",
            "CLI wording depends on the changed view",
            changed=("src/bio_neb/devtools/testing.py",),
        )

        self.assertIn("tests/test_cli.py", explained["test_paths"])
        self.assertEqual(explained["derivation"]["tests/test_cli.py"][0], "included")
        self.assertEqual(explained["includes"], ["tests/test_cli.py"])
        self.assertEqual(explained["depth"], 1)
        self.assertEqual(
            explained["expansion_reason"], "CLI wording depends on the changed view"
        )
        for arguments in (("--include", "tests.test_cli"), ("--depth", "1")):
            with self.subTest(arguments=arguments):
                result, _, errors = _main("scoped", *arguments, "--explain")
                self.assertEqual(result, 2)
                self.assertIn(
                    "--include and --depth require one nonblank --expansion-reason",
                    errors,
                )


class DerivedBatchTests(unittest.TestCase):
    _CHANGED = ("src/bio_neb/devtools/testing.py",)

    def setUp(self) -> None:
        # No previous ledger, so an unset --timeout keeps the flat default.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        environment = patch.dict(
            os.environ, {locations.TESTING_RUN_ROOTS_OVERRIDE: directory.name}
        )
        environment.start()
        self.addCleanup(environment.stop)

    def _batch(self, *arguments: str) -> tuple[int, str, object]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "batch"
            # The selection reaching the batch only; the builder phase has its own
            # cases in tests/test_testing_batch.py.
            with (
                patch.object(
                    testing_operation, "_build_prerequisites", return_value=None
                ),
                patch.object(testing_operation, "_run_batch", return_value=0) as run,
            ):
                result, _, errors = _main(
                    "core", "--root", str(root), *arguments, changed=self._CHANGED
                )
        return result, errors, (run, root)

    def test_a_derived_selection_feeds_the_batch_with_its_modules_and_rung(
        self,
    ) -> None:
        modules = [
            path.removesuffix(".py").replace("/", ".")
            for path in _explain("core", changed=self._CHANGED)["test_paths"]
        ]
        declared = testing_operation._load_ownership()["representative_selections"]

        result, _, (run, root) = self._batch()
        self.assertEqual(result, 0)
        run.assert_called_once_with(
            modules,
            root,
            timeout=3600.0,
            module_timeouts={},
            store=None,
            shards={},
            estimates={},
            timing_ledgers=[],
            slot_wait_seconds=ANY,
            # The case's host queue is its own: the batch is the sole runner.
            concurrency=min(8, os.cpu_count() or 1),
            order="sorted",
            detect_writes=False,
            parallel_experiment=False,
            serial_only=testing_operation._load_ownership()["serial_only"],
        )

        result, _, (run, root) = self._batch(
            "--rung", "representative", "--timeout", "60"
        )
        self.assertEqual(result, 0)
        laddered = sorted(module for module in declared if module in modules)
        self.assertTrue(laddered)
        # The selection's --base-sha is also the representative rung's base.
        self.assertEqual(run.call_args.kwargs["base_commit"], _head())
        self.assertEqual(run.call_args.kwargs["timeout"], 60.0)
        self.assertEqual(sorted(run.call_args.kwargs["representatives"]), laddered)
        self.assertEqual(run.call_args.args, (modules, root))

    def test_batch_options_conflicting_with_the_receipt_path_are_refused(
        self,
    ) -> None:
        for arguments, refusal in (
            (
                ("--rung", "representative", "--explain"),
                "--rung, --timeout and --concurrency need --root",
            ),
            (
                ("--timeout", "60", "--explain"),
                "--rung, --timeout and --concurrency need --root",
            ),
            (
                ("--concurrency", "2", "--explain"),
                "--rung, --timeout and --concurrency need --root",
            ),
        ):
            with self.subTest(arguments=arguments):
                result, _, errors = _main("core", *arguments)
                self.assertEqual(result, 2)
                self.assertIn(refusal, errors)
        result, errors, (run, _) = self._batch(
            "--run-root", "/tmp", "--receipt", "/tmp/receipt.json"
        )
        self.assertEqual(result, 2)
        self.assertIn("--root runs a batch", errors)
        run.assert_not_called()

    def test_a_derived_batch_runs_eight_modules_as_sole_runner_and_never_above_ten(
        self,
    ) -> None:
        # "sole runner we do 8 concerany and raid to 10"; the case's host queue is its
        # own, so the batch is the sole runner.
        with patch("os.cpu_count", return_value=24):
            result, _, (run, _) = self._batch()
            self.assertEqual(result, 0)
            self.assertEqual(run.call_args.kwargs["concurrency"], 8)

            result, _, (run, _) = self._batch("--concurrency", "10")
            self.assertEqual(result, 0)
            self.assertEqual(run.call_args.kwargs["concurrency"], 10)

            result, errors, (run, _) = self._batch("--concurrency", "11")
        self.assertEqual(result, 2)
        self.assertIn("--concurrency must be from 1 to 10", errors)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
