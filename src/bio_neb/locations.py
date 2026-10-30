"""One place for every machine-specific location Bio Neb uses.

The defaults describe a self-contained checkout: local state lives in the gitignored
``.bio-neb-local`` directory beside the source, testing receipts live outside the
checkout under the user's XDG state directory, and external tools resolve from the
current user's home or ``PATH``.  Each location has exactly one environment override,
named beside it below, so a machine is configured by exporting ``BIO_NEB_*`` variables
rather than by editing modules.  A linked worktree has no local state of its own: it
refuses until ``BIO_NEB_LOCAL_STATE`` names the operator home.  ``python -m
bio_neb.locations`` prints the resolved set for the current environment.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

LOCAL_STATE_OVERRIDE = "BIO_NEB_LOCAL_STATE"
CONTROLLER_HOME_OVERRIDE = "BIO_NEB_CONTROLLER_HOME"
TESTING_RUN_ROOTS_OVERRIDE = "BIO_NEB_TESTING_RUN_ROOTS"
TEST_QUEUE_OVERRIDE = "BIO_NEB_TEST_QUEUE"
NEBIUS_EXECUTABLE_OVERRIDE = "BIO_NEB_NEBIUS_EXECUTABLE"
UV_EXECUTABLE_OVERRIDE = "BIO_NEB_UV_EXECUTABLE"
ARCHIVE_ROOT_OVERRIDE = "BIO_NEB_ARCHIVE_ROOT"
OVERRIDES = (
    LOCAL_STATE_OVERRIDE,
    CONTROLLER_HOME_OVERRIDE,
    TESTING_RUN_ROOTS_OVERRIDE,
    TEST_QUEUE_OVERRIDE,
    NEBIUS_EXECUTABLE_OVERRIDE,
    UV_EXECUTABLE_OVERRIDE,
    ARCHIVE_ROOT_OVERRIDE,
)


class LinkedWorktreeUnconfigured(RuntimeError):
    """A linked worktree with no local-state override: its own default is a stray home."""


def _path(override: str, default: Path) -> Path:
    configured = os.environ.get(override)
    return (Path(configured) if configured else default).resolve()


def local_state_root() -> Path:
    """Gitignored directory holding fixtures, run roots and other local state.

    A linked worktree (its ``.git`` is a file) refuses without the override: its own
    ``.bio-neb-local`` is an empty home, while the operator home is the main checkout's,
    exported once for the host by ``tools/devbox/bootstrap.sh`` (#392).
    """

    if (
        not os.environ.get(LOCAL_STATE_OVERRIDE)
        and (REPOSITORY_ROOT / ".git").is_file()
    ):
        raise LinkedWorktreeUnconfigured(
            f"{REPOSITORY_ROOT} is a linked worktree and {LOCAL_STATE_OVERRIDE} is "
            f"unset; export {LOCAL_STATE_OVERRIDE} and {CONTROLLER_HOME_OVERRIDE} as "
            "the operator home (tools/devbox/bootstrap.sh writes them)"
        )
    return _path(LOCAL_STATE_OVERRIDE, REPOSITORY_ROOT / ".bio-neb-local")


def controller_home() -> Path:
    """Directory that owns the native Controller's fixed roots."""

    configured = os.environ.get(CONTROLLER_HOME_OVERRIDE)
    return Path(configured).resolve() if configured else local_state_root()


def run_roots() -> Path:
    return controller_home() / "live-runs"


def testing_run_roots() -> Path:
    """Directory holding the fresh roots that testing receipts are written under.

    A testing receipt proves a committed candidate from outside the tree it proves,
    so this default is never inside the repository, whatever the local state is set
    to.  It is container-local (the XDG state directory), not a bind mount.
    """

    return _path(
        TESTING_RUN_ROOTS_OVERRIDE,
        Path.home() / ".local" / "state" / "bio-neb" / "testing-runs",
    )


def test_queue() -> Path:
    """The host test queue's slot locks, one directory for every checkout and lane.

    The testing runner takes its slots here before it starts a test child (#391), so the
    default follows the user's home rather than the checkout or its local state.
    """

    return _path(
        TEST_QUEUE_OVERRIDE,
        Path.home() / ".local" / "state" / "bio-neb" / "test-queue",
    )


def archive_root() -> Path:
    """The repository's ``.archive``: owner-approved run roots move to ``<batch>/run-roots``.

    A run-ledger row keyed ``archive:<batch>`` names ``archive_root()/<batch>/run-roots``.
    """

    return _path(ARCHIVE_ROOT_OVERRIDE, REPOSITORY_ROOT / ".archive")


def checkouts_root() -> Path:
    return controller_home() / "checkouts"


def bare_cache() -> Path:
    return controller_home() / "repo.git"


def controller_env() -> Path:
    """Credential file for Runs that reach storage or execution stages."""

    return controller_home() / "controller.env"


def model_env() -> Path:
    """Model-only credential file for planning-only live Runs; it holds no storage key."""

    return controller_home() / "model.env"


def parabricks_account() -> Path:
    """This account's Parabricks values; a Run given no section registers from it (#454)."""

    return controller_home() / "parabricks-account.json"


def nebius_executable() -> str:
    """Path of the accepted Nebius CLI payload; its identity is pinned elsewhere."""

    return str(
        _path(NEBIUS_EXECUTABLE_OVERRIDE, Path.home() / ".nebius" / "bin" / "nebius")
    )


def uv_executable() -> Path:
    found = shutil.which("uv")
    default = Path(found) if found else Path.home() / ".local" / "bin" / "uv"
    return _path(UV_EXECUTABLE_OVERRIDE, default)


def describe() -> dict[str, object]:
    """Every resolved location with the override that selects it."""

    return {
        "repository_root": str(REPOSITORY_ROOT),
        "local_state_root": str(local_state_root()),
        "controller_home": str(controller_home()),
        "run_roots": str(run_roots()),
        "testing_run_roots": str(testing_run_roots()),
        "test_queue": str(test_queue()),
        "archive_root": str(archive_root()),
        "checkouts_root": str(checkouts_root()),
        "bare_cache": str(bare_cache()),
        "controller_env": str(controller_env()),
        "model_env": str(model_env()),
        "parabricks_account": str(parabricks_account()),
        "nebius_executable": nebius_executable(),
        "uv_executable": str(uv_executable()),
        "overrides": {name: os.environ.get(name) for name in OVERRIDES},
    }


if __name__ == "__main__":
    print(json.dumps(describe(), indent=2, sort_keys=True))
