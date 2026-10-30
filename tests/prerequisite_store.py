"""The batch catalog adapter: Run states a batch builds once per source key, read-only.

The testing runner's builder phase calls the prerequisite builder entry
(`build_store` in the module `tests/testing_ownership.json` names as
`prerequisite_builder`) with `<batch-root>/prerequisites` before any module
starts, and each module process's Run sandbox names that store in
`BIO_NEB_PREREQUISITE_STORE`. Each catalog state is built once, through the
same coordinator-path builder a test would call, at its fixed final path
`<store>/<state>/` (path- and inode-bound state is never relocated), then made
read-only. `receipt.json` records the source key, each state's Journey file, its
byte manifest and whether this call built or reused it.

The key is the SHA-256 of every file under `src/`, every non-test file under
`tests/` (the builders and everything they read), `uv.lock` and the interpreter
version. `admitted()` checks the key and a state's bytes once per process, at
its first request; a mismatch serves nothing, so the caller builds its own.
Without a store, as in an exact test run, every caller builds its own.

A module that used a stored state checks its bytes again when the module ends
and fails if they changed; a test that changed one rejects it for the rest of
the process.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import unittest
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple

from bio_neb.devtools.run_sandbox import PREREQUISITE_STORE as STORE

_RECEIPT = "receipt.json"
_SCHEMA = "bio-neb.prerequisite-store/1"
_REPOSITORY = Path(__file__).resolve().parents[1]
# This process's admission results: receipts by store, states by (store, name).
_RECEIPTS: dict[str, dict | None] = {}
_ADMITTED: dict[tuple[str, str], StoredState | None] = {}


def source_key(repository: Path = _REPOSITORY) -> str:
    """SHA-256 of the sources, test support, lock and interpreter a state depends on."""
    digest = hashlib.sha256()
    files = [
        *(path for path in (repository / "src").rglob("*") if path.is_file()),
        *(
            path
            for path in (repository / "tests").rglob("*")
            if path.is_file() and not path.name.startswith("test_")
        ),
        repository / "uv.lock",
    ]
    for path in sorted(
        path
        for path in files
        if "__pycache__" not in path.relative_to(repository).parts
    ):
        digest.update(path.relative_to(repository).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    digest.update(sys.version.encode())
    return digest.hexdigest()


def _manifest(root: Path) -> dict[str, str | None]:
    """Every path below root: file SHA-256, or None for a directory."""
    return {
        path.relative_to(root).as_posix(): (
            None if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in sorted(root.rglob("*"))
    }


def _read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _writable(root: Path) -> None:
    root.chmod(0o755)
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            path.chmod(0o755)


def _remove(root: Path) -> None:
    _writable(root)
    shutil.rmtree(root)


def build_store(store: str, builders: Mapping[str, Callable[[Path], Path]]) -> int:
    """Build or reuse every named state in store; 0 when all are ready.

    `builders[name](root)` builds the state at root and returns its Journey file.
    A store already holding this key (a resumed batch) is reused; a store holding
    another key is refused, because a changed source needs a fresh batch root. A
    state left by an interrupted build is removed and built again.
    """
    directory = Path(store)
    directory.mkdir(parents=True, exist_ok=True)
    key = source_key()
    receipt_file = directory / _RECEIPT
    receipt = (
        json.loads(receipt_file.read_text(encoding="utf-8"))
        if receipt_file.is_file()
        else {"schema": _SCHEMA, "key": key, "states": {}, "failed": None}
    )
    if receipt["key"] != key:
        receipt["failed"] = {"state": None, "reason": "store_key_changed"}
        receipt_file.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
        print(json.dumps(receipt["failed"]), file=sys.stderr)
        return 1
    for state, build in builders.items():
        root = directory / state
        if (
            state in receipt["states"]
            and _manifest(root) == receipt["states"][state]["manifest"]
        ):
            receipt["states"][state]["status"] = "reused"
            continue
        if root.exists():
            _remove(root)
        started = time.monotonic()
        try:
            journey_file = build(root)
        except BaseException as error:
            receipt["failed"] = {"state": state, "reason": repr(error)}
            receipt_file.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
            raise
        receipt["states"][state] = {
            "journey_file": str(journey_file),
            "status": "built",
            "seconds": round(time.monotonic() - started, 3),
            "manifest": _manifest(root),
        }
        _read_only(root)
        receipt_file.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    receipt["failed"] = None
    receipt_file.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    return 0


def _module_check(root: Path, manifest: dict[str, str | None], state: str) -> None:
    if _manifest(root) != manifest:
        raise AssertionError(f"shared prerequisite state {state} changed: {root}")


class StoredState(NamedTuple):
    """A catalog state's fixed root in the store and its Journey file."""

    root: Path
    journey_file: Path


def _receipt(store: str) -> dict | None:
    """The store's receipt while it holds this process's source key, read once."""
    if store not in _RECEIPTS:
        try:
            receipt = json.loads((Path(store) / _RECEIPT).read_text(encoding="utf-8"))
        except FileNotFoundError:
            receipt = None
        if receipt is not None and receipt["key"] != source_key():
            receipt = None
        _RECEIPTS[store] = receipt
    return _RECEIPTS[store]


def admitted(state: str) -> StoredState | None:
    """The batch's read-only state, or None to build one.

    None when no store is named, the store has no such state, or its source key
    or the state's bytes did not match at this process's first request. The
    first admission adds the module-end byte check.
    """
    store = os.environ.get(STORE)
    if not store:
        return None
    if (store, state) not in _ADMITTED:
        receipt = _receipt(store)
        known = None if receipt is None else receipt["states"].get(state)
        served = None
        if known is not None and _manifest(Path(store) / state) == known["manifest"]:
            served = StoredState(Path(store) / state, Path(known["journey_file"]))
            unittest.addModuleCleanup(
                _module_check, served.root, known["manifest"], state
            )
        _ADMITTED[store, state] = served
    return _ADMITTED[store, state]


def reject(state: str) -> None:
    """Serve nothing more of a state a test changed; callers build their own."""
    store = os.environ.get(STORE)
    if store:
        _ADMITTED[store, state] = None


def forget() -> None:
    """Drop this process's admission results, as a new process starts without them."""
    _RECEIPTS.clear()
    _ADMITTED.clear()


__all__ = [
    "STORE",
    "StoredState",
    "admitted",
    "build_store",
    "forget",
    "reject",
    "source_key",
]
