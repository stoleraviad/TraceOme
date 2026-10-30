"""The single location policy: defaults, one override each, and the printed report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bio_neb import locations


class LocationDefaultsTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in locations.OVERRIDES:
            os.environ.pop(name, None)

    def _checkout(self, git: str) -> Path:
        """A checkout whose `.git` is a directory (main or clone) or a file (linked)."""
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        if git == "directory":
            (root / ".git").mkdir()
        else:
            (root / ".git").write_text(
                "gitdir: /elsewhere/.git/worktrees/lane\n", encoding="utf-8"
            )
        self.enterContext(patch.object(locations, "REPOSITORY_ROOT", root))
        return root

    def test_a_main_checkout_keeps_its_self_contained_defaults(self):
        repository = self._checkout("directory")
        local_state = repository / ".bio-neb-local"
        self.assertEqual(locations.local_state_root(), local_state)
        self.assertEqual(locations.controller_home(), local_state)
        self.assertEqual(locations.run_roots(), local_state / "live-runs")
        self.assertEqual(locations.checkouts_root(), local_state / "checkouts")
        self.assertEqual(locations.bare_cache(), local_state / "repo.git")
        self.assertEqual(locations.controller_env(), local_state / "controller.env")
        self.assertEqual(locations.model_env(), local_state / "model.env")
        self.assertEqual(locations.archive_root(), repository / ".archive")
        self.assertEqual(
            locations.testing_run_roots(),
            Path.home().resolve() / ".local" / "state" / "bio-neb" / "testing-runs",
        )
        self.assertEqual(
            locations.nebius_executable(),
            str((Path.home() / ".nebius" / "bin" / "nebius").resolve()),
        )
        self.assertTrue(locations.uv_executable().is_absolute())

    def test_a_linked_worktree_without_the_local_state_override_refuses(self):
        # Its own `.bio-neb-local` is an empty home no runner means; the
        # host-level variables are the fix, the refusal names both.
        self._checkout("file")
        for name, call in (
            ("local_state_root", locations.local_state_root),
            ("controller_home", locations.controller_home),
            ("run_roots", locations.run_roots),
        ):
            with self.subTest(location=name):
                with self.assertRaises(locations.LinkedWorktreeUnconfigured) as raised:
                    call()
                message = str(raised.exception)
                self.assertIn("BIO_NEB_LOCAL_STATE", message)
                self.assertIn("BIO_NEB_CONTROLLER_HOME", message)
        # Only the local state is required: the Controller home still follows it.
        with patch.dict(os.environ, {locations.LOCAL_STATE_OVERRIDE: "/srv/home"}):
            self.assertEqual(locations.local_state_root(), Path("/srv/home"))
            self.assertEqual(locations.controller_home(), Path("/srv/home"))
        with (
            patch.dict(os.environ, {locations.CONTROLLER_HOME_OVERRIDE: "/srv/home"}),
            self.assertRaises(locations.LinkedWorktreeUnconfigured),
        ):
            locations.local_state_root()

    def test_each_override_selects_exactly_its_location(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            with patch.dict(
                os.environ,
                {
                    locations.LOCAL_STATE_OVERRIDE: str(base / "state"),
                    locations.NEBIUS_EXECUTABLE_OVERRIDE: str(base / "nebius"),
                    locations.UV_EXECUTABLE_OVERRIDE: str(base / "uv"),
                },
            ):
                self.assertEqual(locations.local_state_root(), base / "state")
                # The Controller home follows the local state unless set itself.
                self.assertEqual(locations.run_roots(), base / "state" / "live-runs")
                self.assertEqual(locations.nebius_executable(), str(base / "nebius"))
                self.assertEqual(locations.uv_executable(), base / "uv")
                with patch.dict(
                    os.environ,
                    {locations.CONTROLLER_HOME_OVERRIDE: str(base / "controller")},
                ):
                    self.assertEqual(locations.local_state_root(), base / "state")
                    self.assertEqual(
                        locations.run_roots(), base / "controller" / "live-runs"
                    )
                    self.assertEqual(
                        locations.controller_env(),
                        base / "controller" / "controller.env",
                    )
                    self.assertEqual(
                        locations.model_env(), base / "controller" / "model.env"
                    )

    def test_testing_run_roots_default_outside_the_repository_with_own_override(
        self,
    ):
        # Receipts prove a committed candidate from outside the tree they prove, so
        # the default never lives in the checkout, however the local state is placed.
        self._checkout("directory")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            with patch.dict(
                os.environ, {locations.LOCAL_STATE_OVERRIDE: str(base / "state")}
            ):
                default = locations.testing_run_roots()
                self.assertNotIn(locations.REPOSITORY_ROOT, default.parents)
                self.assertNotEqual(default, locations.REPOSITORY_ROOT)
                self.assertNotEqual(default, locations.run_roots())
                self.assertEqual(locations.run_roots(), base / "state" / "live-runs")
            with patch.dict(
                os.environ,
                {locations.TESTING_RUN_ROOTS_OVERRIDE: str(base / "receipts")},
            ):
                self.assertEqual(locations.testing_run_roots(), base / "receipts")
                # The override selects exactly its location; live runs are untouched.
                self.assertEqual(
                    locations.run_roots(),
                    locations.REPOSITORY_ROOT / ".bio-neb-local" / "live-runs",
                )

    def test_test_queue_is_one_host_location_for_every_checkout_with_own_override(
        self,
    ):
        # Every worktree's runner must count against the same slots, so the
        # default follows the user's home, never the checkout or its local state.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            with patch.dict(
                os.environ,
                {
                    locations.LOCAL_STATE_OVERRIDE: str(base / "state"),
                    locations.CONTROLLER_HOME_OVERRIDE: str(base / "controller"),
                },
            ):
                self.assertEqual(
                    locations.test_queue(),
                    Path.home().resolve()
                    / ".local"
                    / "state"
                    / "bio-neb"
                    / "test-queue",
                )
            with patch.dict(
                os.environ, {locations.TEST_QUEUE_OVERRIDE: str(base / "queue")}
            ):
                self.assertEqual(locations.test_queue(), base / "queue")
                self.assertNotEqual(locations.testing_run_roots(), base / "queue")

    def test_symlinked_home_resolves_to_canonical_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            real = base / "real"
            real.mkdir()
            link = base / "link"
            link.symlink_to(real, target_is_directory=True)
            with patch.dict(
                os.environ, {locations.CONTROLLER_HOME_OVERRIDE: str(link)}
            ):
                self.assertEqual(locations.run_roots(), real / "live-runs")

    def test_report_lists_every_location_and_override(self):
        # The printed report runs from this checkout, linked or not, so it names a home.
        os.environ[locations.LOCAL_STATE_OVERRIDE] = str(
            Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        )
        report = locations.describe()
        self.assertEqual(
            set(report),
            {
                "repository_root",
                "local_state_root",
                "controller_home",
                "run_roots",
                "testing_run_roots",
                "test_queue",
                "archive_root",
                "checkouts_root",
                "bare_cache",
                "controller_env",
                "model_env",
                "parabricks_account",
                "nebius_executable",
                "uv_executable",
                "overrides",
            },
        )
        self.assertEqual(set(report["overrides"]), set(locations.OVERRIDES))
        printed = subprocess.run(
            [sys.executable, "-m", "bio_neb.locations"],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONPATH": str(locations.REPOSITORY_ROOT / "src")},
        )
        self.assertEqual(json.loads(printed.stdout), report)


if __name__ == "__main__":
    unittest.main()
