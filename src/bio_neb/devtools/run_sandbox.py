"""The Run sandbox: the one environment every testing-runner child process sees.

Each child the testing runner starts (a batch module, the prerequisite builder, a closure
or checkpoint command, a `measure` sample) runs inside a fresh private directory made
under the controller's temporary directory and removed when the child ends. The child's
`TMPDIR`, `HOME`, local state, controller home, testing run roots and test queue point
there, so a child's writes to those locations never reach the developer's (ADR 0041), and a
runner a test starts queues inside its own sandbox, never against the host slots its
controller already holds for it (#391); what it reads
from the developer's home (configuration directory, uv's cache, the Nebius location and
`~/.gitconfig`) is named to it explicitly. Its git reads take no optional locks; its
bytecode goes to the cache the caller names; and the prerequisite store reaches it only
when the caller names one, never through the controller's own environment.

Inside the sandbox `os.fsync` is a no-op (owner, #291 Q1/Q7): no offline test can observe
durability, and fsync through the container's overlay filesystem cost about a quarter of
suite time. A start-up file on the child's search path installs it, so a child keeps
its own argv and a direct `python -m unittest` outside the runner keeps the real fsync.
The controller itself never runs in a sandbox: its ledger lines are the resume record.

A runner re-execution (a snapshot's own runner, the checkpoint's suite batch) is a
controller, not a sandboxed child: `controller_environment` keeps the real locations so
it finds previous ledgers, and its own children get their sandboxes from it.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from bio_neb.devtools import test_queue
from bio_neb.locations import (
    CONTROLLER_HOME_OVERRIDE,
    LOCAL_STATE_OVERRIDE,
    NEBIUS_EXECUTABLE_OVERRIDE,
    TEST_QUEUE_OVERRIDE,
    TESTING_RUN_ROOTS_OVERRIDE,
    nebius_executable,
)

# Short, so socket paths under a sandbox's TMPDIR still fit.
PRIVATE_PREFIX = "bio-neb-batch-"
PREREQUISITE_STORE = "BIO_NEB_PREREQUISITE_STORE"
_STARTUP = '''"""Run sandbox start-up (bio_neb.devtools.run_sandbox): fsync is a no-op here."""
import os


def _sandbox_fsync(fd):
    """Skip the flush; the Run sandbox's files are thrown away with it."""


os.fsync = _sandbox_fsync
'''


@dataclass(frozen=True)
class RunSandbox:
    """One child's private directory and the environment that points into it."""

    root: Path
    environment: dict[str, str]

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    def tmp_residue(self) -> int:
        """Entries the child left in its TMPDIR; 0 when it removed or locked it."""
        try:
            return len(os.listdir(self.tmp))
        except OSError:
            return 0


def _keep_home_reads(environment: dict[str, str], home: Path) -> None:
    """Name what a child reads from the developer's home before `home` replaces it.

    The configuration directory (git's and uv's configuration), uv's cache and the
    Nebius location keep the controller's values; the private home's `.gitconfig`
    includes the developer's, which git reads after the configuration directory's, as it
    does without a sandbox. A value the controller sets is kept, so a sandbox made inside
    a sandbox (a runner test running `lint`) reads what its controller read. Every other
    read of the home sees the empty private one.
    """
    developer = Path.home()
    cache = os.environ.get("XDG_CACHE_HOME")
    environment.setdefault("XDG_CONFIG_HOME", str(developer / ".config"))
    environment.setdefault(
        "UV_CACHE_DIR", str((Path(cache) if cache else developer / ".cache") / "uv")
    )
    environment.setdefault(NEBIUS_EXECUTABLE_OVERRIDE, nebius_executable())
    (home / ".gitconfig").write_text(
        f'[include]\n\tpath = "{developer / ".gitconfig"}"\n', encoding="utf-8"
    )


def _search_path(first: Path, inherited: str | None) -> str:
    return os.pathsep.join(filter(None, [str(first), inherited]))


@contextmanager
def run_sandbox(
    *, cache: Path | None = None, store: Path | None = None, uv_cache: bool = False
) -> Iterator[RunSandbox]:
    """A fresh sandbox for one child, removed on exit.

    `cache` is a bytecode cache the caller shares between its children; without it the
    child writes no bytecode and reads none from the checkout. `store` names the
    prerequisite store the child may read. `uv_cache` keeps uv's own cache for a tool
    uv runs offline from it (the pinned Ruff); every other child runs uv without one.
    """
    root = Path(tempfile.mkdtemp(prefix=PRIVATE_PREFIX))
    try:
        private = {
            LOCAL_STATE_OVERRIDE: root / "state",
            CONTROLLER_HOME_OVERRIDE: root / "controller",
            TESTING_RUN_ROOTS_OVERRIDE: root / "testing-runs",
            TEST_QUEUE_OVERRIDE: root / "test-queue",
        }
        startup = root / "startup"
        for directory in (root / "tmp", root / "home", startup, *private.values()):
            directory.mkdir()
        (startup / "sitecustomize.py").write_text(_STARTUP, encoding="utf-8")
        environment = dict(os.environ)
        tmp = str(root / "tmp")
        environment.update(TMPDIR=tmp, TEMP=tmp, TMP=tmp)
        _keep_home_reads(environment, root / "home")
        environment["HOME"] = str(root / "home")
        environment.update({name: str(path) for name, path in private.items()})
        environment["PYTHONPATH"] = _search_path(startup, environment.get("PYTHONPATH"))
        environment["PYTHONPYCACHEPREFIX"] = str(cache or root / "python-cache")
        if cache is None:
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
        # A child's git reads must not take the index lock another child's git may need.
        environment["GIT_OPTIONAL_LOCKS"] = "0"
        if uv_cache:
            environment.pop("UV_NO_CACHE", None)
        else:
            environment["UV_NO_CACHE"] = "1"
        environment.pop(test_queue.HELD, None)
        environment.pop(PREREQUISITE_STORE, None)
        if store is not None:
            environment[PREREQUISITE_STORE] = str(store)
        yield RunSandbox(root, environment)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def controller_environment(
    *, cache: Path | None = None, source: Path | None = None
) -> dict[str, str]:
    """The environment of a runner re-execution or of the controller's own git reads.

    It keeps the real locations and fsync. `cache` is its bytecode cache and `source`
    a snapshot's `src` directory, put first on its search path.
    """
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    if cache is not None:
        environment["PYTHONPYCACHEPREFIX"] = str(cache)
    if source is not None:
        environment["PYTHONPATH"] = _search_path(source, environment.get("PYTHONPATH"))
    return environment
