"""Dependency-free control surface for serial offline repository testing."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TextIO

from bio_neb.devtools import review_flag, run_sandbox, test_queue
from bio_neb.devtools.run_sandbox import PRIVATE_PREFIX, controller_environment
from bio_neb.devtools.testing_selection import (
    CHECKPOINT,
    Derivation,
    ImportGraph,
    derive,
    module_name,
)
from bio_neb.locations import (
    TESTING_RUN_ROOTS_OVERRIDE,
    controller_home,
    local_state_root,
    testing_run_roots,
    uv_executable,
)
from bio_neb.locations import test_queue as test_queue_root

_REPOSITORY = Path(__file__).resolve().parents[3]
_OWNERSHIP_PATH = _REPOSITORY / "tests" / "testing_ownership.json"
_ROUTER_PATH = _REPOSITORY / "docs" / "agents" / "issue-tracker.md"
_TESTING_GUIDE_PATH = _REPOSITORY / "docs" / "agents" / "testing.md"
_OPERATOR_GUIDE_PATH = _REPOSITORY / "docs" / "agents" / "native-ubuntu-operator.md"
_TEST_PATTERN = "test_*.py"
_RUFF = [
    str(uv_executable().with_name("uvx")),
    "--offline",
    "--from",
    "ruff==0.12.8",
    "ruff",
]


class ControlSurfaceError(RuntimeError):
    """The command or checked-in control data cannot select truthful proof."""


@contextlib.contextmanager
def _host_slots(
    count: int, holder: str, sole: int | None = None
) -> Iterator[test_queue.Held]:
    """Take `count` of the host's test slots before any child starts (#391), or `sole`
    of them when no other runner holds any (#455)."""
    try:
        with test_queue.slots(
            count,
            queue=test_queue_root(),
            holder=f"{_REPOSITORY}: {holder}",
            sole=sole,
        ) as held:
            yield held
    except test_queue.QueueRefused as error:
        raise ControlSurfaceError(str(error)) from error


def _batch_slots(concurrency: int) -> int:
    """A batch's host slots: its concurrency, all of them for an experiment above it."""
    return min(concurrency, test_queue.CAPACITY)


def _slot_wait(held: test_queue.Held) -> float | None:
    """The wait a record keeps; an inherited request's wait is its controller's."""
    return None if held.inherited else held.waited_seconds


_DERIVED_MODES = ("core", "scoped", "scoped-with-current-fixture")


def _share(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


@dataclass(frozen=True)
class Selection:
    mode: str
    base_sha: str | None
    ticket_id: int | None
    test_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]
    fallback_reasons: tuple[str, ...]
    checkpoint_recommended: bool
    recommendation_reasons: tuple[str, ...]
    includes: tuple[str, ...] = ()
    depth: int = 0
    expansion_reason: str | None = None
    derivation: Derivation | None = None
    regression_selection: str | None = None
    test_ids: tuple[str, ...] = ()
    known_red: tuple[tuple[str, str], ...] = ()

    def as_json(self) -> dict[str, Any]:
        discovered = _discovered_tests()
        selected = set(self.test_paths)
        regression = self.mode == "regression"
        method_count = (
            len(self.test_ids) if regression else len(_test_method_ids(selected))
        )
        discovered_methods = len(_test_method_ids(discovered))
        derivation = self.derivation
        return {
            "schema_version": "bio-neb.testing-selection/5",
            "mode": self.mode,
            "base_sha": self.base_sha,
            "ticket_id": self.ticket_id,
            "regression_selection": self.regression_selection,
            "known_red": dict(self.known_red),
            "test_paths": list(self.test_paths),
            "test_module_count": len(self.test_paths),
            "discovered_test_module_count": len(discovered),
            "selected_module_share": _share(len(self.test_paths), len(discovered)),
            "test_ids": list(self.test_ids),
            "test_method_count": method_count,
            "discovered_test_method_count": discovered_methods,
            "selected_method_share": _share(method_count, discovered_methods),
            "excluded_test_paths": sorted(discovered - selected),
            "excluded_test_module_count": len(discovered - selected),
            "changed_paths": list(self.changed_paths),
            "derivation": (
                {test: list(why) for test, why in derivation.reasons.items()}
                if derivation
                else {}
            ),
            "derived_by": dict(derivation.derived_by) if derivation else {},
            "changed_path_users": (
                dict(derivation.changed_path_users) if derivation else {}
            ),
            "wider_selection": (
                {
                    label: {
                        "test_module_count": len(tests),
                        "selected_module_share": _share(len(tests), len(discovered)),
                    }
                    for label, tests in derivation.wider.items()
                }
                if derivation
                else {}
            ),
            "includes": list(self.includes),
            "depth": self.depth,
            "expansion_reason": self.expansion_reason,
            "fallback_reasons": list(self.fallback_reasons),
            "checkpoint_recommended": self.checkpoint_recommended,
            "recommendation_reasons": list(self.recommendation_reasons),
            "decision_authority": "ticket_then_root",
            "supplemental_only": regression,
            "acceptance_bypass": False,
            "proof_sequence": (
                ["supplemental_regression"]
                if regression
                else ["checkpoint"]
                if self.mode == "checkpoint"
                else ["closure"]
            ),
        }


def _relative(path: Path) -> str:
    return path.relative_to(_REPOSITORY).as_posix()


def _load_ownership() -> dict[str, Any]:
    try:
        value = json.loads(_OWNERSHIP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlSurfaceError(f"cannot load {_OWNERSHIP_PATH}: {error}") from error
    if not isinstance(value, dict):
        raise ControlSurfaceError("testing ownership must contain one JSON object")
    return value


def _discovered_tests() -> set[str]:
    return {
        _relative(path)
        for path in (_REPOSITORY / "tests").glob(_TEST_PATTERN)
        if path.is_file()
    }


def _class_test_methods(
    classes: dict[str, ast.ClassDef], name: str, visiting: frozenset[str] = frozenset()
) -> set[str]:
    if name not in classes or name in visiting:
        return set()
    node = classes[name]
    inherited = set().union(
        *(
            _class_test_methods(classes, base.id, visiting | {name})
            for base in node.bases
            if isinstance(base, ast.Name)
        )
    )
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name.startswith("test_"):
                inherited.add(item.name)
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    inherited.discard(target.id)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if item.value is not None:
                inherited.discard(item.target.id)
    return inherited


def _test_method_ids(test_paths: set[str]) -> set[str]:
    method_ids: set[str] = set()
    for path in sorted(test_paths):
        try:
            tree = ast.parse(
                (_REPOSITORY / path).read_text(encoding="utf-8"), filename=path
            )
        except (OSError, SyntaxError) as error:
            raise ControlSurfaceError(
                f"cannot inspect test methods in {path}: {error}"
            ) from error
        module = module_name(path)
        classes = {
            node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
        }

        for name in classes:
            method_ids.update(
                f"{module}.{name}.{method}"
                for method in _class_test_methods(classes, name)
            )
    return method_ids


def _test_paths_for_ids(test_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                f"{test_id.rsplit('.', 2)[0].replace('.', '/')}.py"
                for test_id in test_ids
            }
        )
    )


def _import_graph() -> ImportGraph:
    """The graph over the repository's unignored sources, as intent paths see them."""
    try:
        return ImportGraph.build(_REPOSITORY, _repository_files())
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise ControlSurfaceError(f"cannot derive the import graph: {error}") from error


def _repository_files() -> list[str]:
    """Tracked and untracked, unignored files that intent path patterns match against."""
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=_REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise ControlSurfaceError("cannot list repository files for intent paths")
    return sorted(
        {
            path
            for path in completed.stdout.split("\0")
            if path and (_REPOSITORY / path).is_file()
        }
    )


def _intent_readers(paths: Any, graph: ImportGraph, files: Sequence[str]) -> None:
    """Validate the intent's path patterns: pattern -> reader modules or checkpoint.

    A pattern names what imports cannot show, so it must match a current file that no test
    already uses through imports, and each reader must be a source some test uses.
    """
    if not isinstance(paths, dict) or not paths:
        raise ControlSurfaceError("intent paths must be a non-empty object")
    for pattern, readers in paths.items():
        matched = [path for path in files if fnmatchcase(path, pattern)]
        if not matched:
            raise ControlSurfaceError(
                f"intent path {pattern!r} matches no repository file"
            )
        if readers != CHECKPOINT and (
            not isinstance(readers, list)
            or not readers
            or any(not isinstance(reader, str) for reader in readers)
        ):
            raise ControlSurfaceError(
                f"intent path {pattern!r} must name reader modules or checkpoint"
            )
        derivable = [path for path in matched if graph.users.of(path)[0]]
        if derivable:
            raise ControlSurfaceError(
                f"intent path {pattern!r} restates what imports already derive for "
                f"{derivable[0]}"
            )
        for reader in readers if readers != CHECKPOINT else ():
            if reader not in graph.imports:
                raise ControlSurfaceError(
                    f"intent reader {reader!r} is not a Python source"
                )
            if not graph.users.of(reader)[0]:
                raise ControlSurfaceError(f"intent reader {reader!r} reaches no test")


def _fixture_registrations(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        data["execution_fixture"],
        *_additional_execution_fixtures(data).values(),
    ]


def _readers(data: dict[str, Any]) -> dict[str, Any]:
    """Intent path readers plus each registered fixture document's loader."""
    readers = dict(data["paths"])
    for registration in _fixture_registrations(data):
        listed = readers.get(registration["path"], [])
        if listed != CHECKPOINT:
            readers[registration["path"]] = sorted({*listed, registration["loader"]})
    return readers


def _registered_fixture_kind(content):
    """The loader and the one purpose each exported document kind serves."""
    from bio_neb.devtools.parabricks_fixture_export import (
        load_execution_fixture,
        load_pre_execution_fixture,
    )
    from bio_neb.devtools.reached_fixture_export import load_reached_fixture

    kinds = {
        "bio-neb.execution-fixture/1": (load_execution_fixture, "artifact_and_flow"),
        "bio-neb.pre-execution-fixture/1": (
            load_pre_execution_fixture,
            "planning_flow",
        ),
        "bio-neb.reached-fixture/1": (load_reached_fixture, "reached_prefix"),
    }
    try:
        return kinds[json.loads(content)["schema_version"]]
    except (ValueError, KeyError, TypeError) as error:
        raise ControlSurfaceError("execution_fixture document invalid") from error


def _validate_execution_fixture(registration, discovered, graph):
    if not isinstance(registration, dict):
        raise ControlSurfaceError("execution_fixture must be registered")
    path, loader, tests = (registration.get(key) for key in ("path", "loader", "tests"))
    if (
        not isinstance(path, str)
        or not isinstance(loader, str)
        or not isinstance(tests, list)
        or not tests
        or not set(tests) <= discovered
        or not (_REPOSITORY / path).is_file()
        or not (_REPOSITORY / loader).is_file()
    ):
        raise ControlSurfaceError("execution_fixture paths/consumers invalid")
    content = (_REPOSITORY / path).read_bytes()
    load_fixture, purpose = _registered_fixture_kind(content)
    if (
        (registration.get("byte_count"), registration.get("sha256"))
        != (len(content), hashlib.sha256(content).hexdigest())
        or registration.get("grants_live_authority") is not False
        or registration.get("purpose") != purpose
    ):
        raise ControlSurfaceError("execution_fixture identity/purpose drift")
    try:
        fixture = load_fixture(content)
    except ValueError as error:
        raise ControlSurfaceError("execution_fixture document invalid") from error
    if (
        registration.get("scenario_count") != len(fixture.scenarios)
        or registration.get("source_tickets")
        != [scenario.source_ticket for scenario in fixture.scenarios]
        or any(
            scenario.source_commit != registration.get("source_commit")
            or scenario.evidence_kind != registration.get("evidence_kind")
            for scenario in fixture.scenarios
        )
    ):
        raise ControlSurfaceError("execution_fixture provenance drift")
    consumers = sorted(
        consumer
        for consumer, dependencies in graph.items()
        if loader in dependencies and consumer.startswith("tests/test_")
    )
    if consumers != sorted(tests):
        raise ControlSurfaceError("execution_fixture consumer drift")


def _additional_execution_fixtures(data):
    registrations = data.get("additional_execution_fixtures", {})
    if not isinstance(registrations, dict) or any(
        not isinstance(name, str) or not name for name in registrations
    ):
        raise ControlSurfaceError(
            "additional execution fixtures must be a named mapping"
        )
    return registrations


def _regression_selections(selections: Any, method_ids: set[str]) -> None:
    """Validate the named regression selections.

    `known_red`, when present, maps member IDs already red for a recorded reason (its
    owning ticket) to that reason, so a supplemental run names its inherited reds.
    """
    if not isinstance(selections, dict) or not selections:
        raise ControlSurfaceError("regression_selections must be a non-empty object")
    for name, selection in selections.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(selection, dict)
        ):
            raise ControlSurfaceError(
                "regression selection names and values are invalid"
            )
        description = selection.get("description")
        test_ids = selection.get("test_ids")
        if (
            not isinstance(description, str)
            or not description.strip()
            or not isinstance(test_ids, list)
            or not test_ids
            or any(not isinstance(test_id, str) for test_id in test_ids)
            or test_ids != sorted(test_ids)
            or len(set(test_ids)) != len(test_ids)
        ):
            raise ControlSurfaceError(
                f"regression selection {name!r} must have a description and "
                "sorted unique test_ids"
            )
        stale_test_ids = sorted(set(test_ids) - method_ids)
        if stale_test_ids:
            raise ControlSurfaceError(
                f"regression selection {name!r} has stale test_ids {stale_test_ids!r}"
            )
        known_red = selection.get("known_red", {})
        if (
            not isinstance(known_red, dict)
            or not set(known_red) <= set(test_ids)
            or any(
                not isinstance(reason, str) or not reason.strip()
                for reason in known_red.values()
            )
        ):
            raise ControlSurfaceError(
                f"regression selection {name!r} known_red must map its own "
                "test_ids to a nonblank reason"
            )


def _representative_selections(
    selections: Any, method_ids: set[str]
) -> dict[str, list[str]]:
    """Validate the laddered modules' representative lists; return module -> IDs.

    Each list names current test methods of its own module, sorted and unique, like a
    regression selection; a stale ID refuses selection.
    """
    if not isinstance(selections, dict):
        raise ControlSurfaceError("representative_selections must be an object")
    representatives: dict[str, list[str]] = {}
    for module, selection in selections.items():
        description = (
            selection.get("description") if isinstance(selection, dict) else None
        )
        test_ids = selection.get("test_ids") if isinstance(selection, dict) else None
        if (
            not isinstance(description, str)
            or not description.strip()
            or not isinstance(test_ids, list)
            or not test_ids
            or any(not isinstance(test_id, str) for test_id in test_ids)
            or test_ids != sorted(test_ids)
            or len(set(test_ids)) != len(test_ids)
        ):
            raise ControlSurfaceError(
                f"representative selection {module!r} must have a description and "
                "sorted unique test_ids"
            )
        foreign = [
            test_id for test_id in test_ids if not test_id.startswith(f"{module}.")
        ]
        if foreign:
            raise ControlSurfaceError(
                f"representative selection {module!r} names IDs outside its module "
                f"{foreign!r}"
            )
        stale_test_ids = sorted(set(test_ids) - method_ids)
        if stale_test_ids:
            raise ControlSurfaceError(
                f"representative selection {module!r} has stale test_ids "
                f"{stale_test_ids!r}"
            )
        representatives[module] = test_ids
    return representatives


def _validate_intent(data: dict[str, Any], graph: ImportGraph) -> None:
    if data.get("schema_version") != "bio-neb.testing-ownership/2":
        raise ControlSurfaceError("unsupported testing ownership schema")
    if data.get("fallback") != CHECKPOINT:
        raise ControlSurfaceError("testing ownership fallback must be checkpoint")
    core_tests = data.get("core_tests")
    if not isinstance(core_tests, list) or not core_tests:
        raise ControlSurfaceError("core_tests must be a non-empty list")

    discovered = _discovered_tests()
    method_ids = _test_method_ids(discovered)
    _regression_selections(data.get("regression_selections"), method_ids)
    _representative_selections(data.get("representative_selections"), method_ids)
    serial_only = data.get("serial_only")
    if (
        not isinstance(serial_only, list)
        or any(not isinstance(test, str) for test in serial_only)
        or serial_only != sorted(set(serial_only))
        or not set(serial_only) <= discovered
    ):
        raise ControlSurfaceError(
            "serial_only must list sorted unique discovered test modules"
        )
    if not set(core_tests) <= discovered:
        raise ControlSurfaceError("core_tests contains a non-discovered module")

    _intent_readers(data.get("paths"), graph, _repository_files())
    imports = graph.imports
    executable_imports = sorted(
        (consumer, dependency)
        for consumer, dependencies in imports.items()
        if consumer.startswith("tests/test_")
        for dependency in dependencies
        if dependency.startswith("tests/test_")
    )
    if executable_imports:
        raise ControlSurfaceError(
            f"executable tests import executable tests: {executable_imports!r}"
        )
    _prerequisite_builder(data)
    _validate_execution_fixture(data.get("execution_fixture"), discovered, imports)
    for registration in _additional_execution_fixtures(data).values():
        _validate_execution_fixture(registration, discovered, imports)


def _prerequisite_builder(data: dict[str, Any]) -> str:
    """The support module whose `build_store(store)` builds a batch's catalog states.

    The ownership intent names it as `prerequisite_builder`; imports cannot show it.
    """
    builder = data.get("prerequisite_builder")
    if (
        not isinstance(builder, str)
        or not builder.endswith(".py")
        or builder not in _repository_files()
        or builder in _discovered_tests()
    ):
        raise ControlSurfaceError(
            f"prerequisite_builder must name a test-support module, not {builder!r}"
        )
    return builder


def _full_sha(value: str) -> str:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ControlSurfaceError(
            "--base-sha must be 40 lowercase hexadecimal characters"
        )
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{value}^{{commit}}"],
        cwd=_REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode or completed.stdout.strip() != value:
        raise ControlSurfaceError("--base-sha is not an exact reachable commit")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", value, "HEAD"],
        cwd=_REPOSITORY,
        check=False,
    )
    if ancestor.returncode:
        raise ControlSurfaceError("--base-sha must be an ancestor of candidate HEAD")
    return value


def _candidate_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode or len(value) != 40:
        raise ControlSurfaceError("cannot resolve candidate HEAD")
    return value


def _changed_paths(base_sha: str) -> tuple[str, ...]:
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            "--diff-filter=ACDMRTUXB",
            f"{base_sha}...HEAD",
        ],
        cwd=_REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise ControlSurfaceError("cannot derive the exact base...HEAD change set")
    return tuple(
        dict.fromkeys(
            line.strip().replace("\\", "/")
            for line in completed.stdout.splitlines()
            if line.strip()
        )
    )


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _method_owner(
    classes: dict[str, ast.ClassDef],
    name: str,
    method: str,
    visiting: frozenset[str] = frozenset(),
) -> str | None:
    """The class whose body defines `method` for class `name`, searching bases in order."""
    if name not in classes or name in visiting:
        return None
    node = classes[name]
    if any(
        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == method
        for item in node.body
    ):
        return name
    for base in node.bases:
        if isinstance(base, ast.Name):
            owner = _method_owner(classes, base.id, method, visiting | {name})
            if owner is not None:
                return owner
    return None


def _has_outside_base(
    classes: dict[str, ast.ClassDef], name: str, visiting: frozenset[str] = frozenset()
) -> bool:
    """Whether a class reaches a base defined outside its module, such as `TestCase`.

    A class built only from in-module mixins is never loaded as a test case.
    """
    if name not in classes or name in visiting:
        return False
    return any(
        not (isinstance(base, ast.Name) and base.id in (*classes, "object"))
        or _has_outside_base(classes, base.id, visiting | {name})
        for base in classes[name].bases
    )


def _changed_test_methods(
    base_sha: str, test_paths: Sequence[str], checkout: Path = _REPOSITORY
) -> set[str]:
    """Test method IDs whose definition differs between `base_sha` and the working tree.

    Committed and uncommitted edits both count. A changed method also selects the same
    method in every test-case class of its module that inherits it; an in-module mixin
    itself is never selected.
    """
    if not test_paths:
        return set()
    completed = subprocess.run(
        [
            "git",
            "diff",
            "-U0",
            "--no-renames",
            "--no-color",
            "--no-ext-diff",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            base_sha,
            "--",
            *test_paths,
        ],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise ControlSurfaceError("cannot derive test methods changed since --base-sha")
    lines: dict[str, set[int]] = defaultdict(set)
    path = None
    for line in completed.stdout.splitlines():
        if line.startswith("+++ "):
            path = line.removeprefix("+++ b/") if line.startswith("+++ b/") else None
        elif path is not None and (hunk := _HUNK.match(line)):
            start, count = int(hunk[1]), int(hunk[2] or 1)
            # A pure deletion touches the lines on either side of it.
            lines[path].update(
                range(start, start + count) if count else (start, start + 1)
            )
    changed: set[str] = set()
    for path, touched in lines.items():
        tree = ast.parse((checkout / path).read_text(encoding="utf-8"), filename=path)
        classes = {
            node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
        }
        edited = {
            (name, item.name)
            for name, node in classes.items()
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name.startswith("test_")
            and any(
                min([item.lineno, *(d.lineno for d in item.decorator_list)])
                <= number
                <= (item.end_lineno or item.lineno)
                for number in touched
            )
        }
        module = module_name(path)
        for name in classes:
            if _has_outside_base(classes, name):
                changed.update(
                    f"{module}.{name}.{method}"
                    for method in _class_test_methods(classes, name)
                    if (_method_owner(classes, name, method), method) in edited
                )
    return changed


def _checkpoint_selection(
    *, base_sha: str, discovered: set[str], changed: tuple[str, ...]
) -> Selection:
    return Selection(
        mode="checkpoint",
        base_sha=base_sha,
        ticket_id=None,
        test_paths=tuple(sorted(discovered)),
        changed_paths=changed,
        fallback_reasons=(),
        checkpoint_recommended=True,
        recommendation_reasons=("explicit checkpoint mode",),
    )


def _risk_fallback(reason: str | None) -> tuple[str, ...]:
    if reason is None:
        return ()
    if not reason.strip():
        raise ControlSurfaceError("--risk-reason must be nonblank")
    return (f"material risk: {reason.strip()}",)


def _changes_since(base_sha: str) -> tuple[str, ...]:
    """base...HEAD plus the working tree, so a check before commit sees its own edits."""
    return tuple(dict.fromkeys(_changed_paths(base_sha) + _working_tree_paths()))


def _test_path(name: str, discovered: set[str]) -> str:
    path = name if name.endswith(".py") else f"{name.replace('.', '/')}.py"
    if path not in discovered:
        raise ControlSurfaceError(f"not a discovered test module: {name}")
    return path


def _selection(
    args: argparse.Namespace, data: dict[str, Any], graph: ImportGraph
) -> Selection:
    discovered = _discovered_tests()
    if args.mode == "regression":
        selection = data["regression_selections"][args.regression_selection]
        test_ids = tuple(selection["test_ids"])
        return Selection(
            mode=args.mode,
            base_sha=None,
            ticket_id=None,
            test_paths=_test_paths_for_ids(test_ids),
            changed_paths=(),
            fallback_reasons=(),
            checkpoint_recommended=False,
            recommendation_reasons=(),
            regression_selection=args.regression_selection,
            test_ids=test_ids,
            known_red=tuple(sorted(selection.get("known_red", {}).items())),
        )
    base = _full_sha(args.base_sha)
    if args.mode == "checkpoint":
        return _checkpoint_selection(
            base_sha=base, discovered=discovered, changed=_changed_paths(base)
        )
    includes = tuple(
        dict.fromkeys(
            _test_path(name, discovered) for name in getattr(args, "include", ())
        )
    )
    depth = getattr(args, "depth", 0)
    reason = getattr(args, "expansion_reason", None)
    if depth < 0:
        raise ControlSurfaceError("--depth must not be negative")
    if (includes or depth) and not (reason and reason.strip()):
        raise ControlSurfaceError(
            "--include and --depth require one nonblank --expansion-reason"
        )
    if reason and not (includes or depth):
        raise ControlSurfaceError("--expansion-reason requires --include or --depth")
    required = [(test, "core") for test in data["core_tests"]]
    required += [(test, "included") for test in includes]
    if args.mode == "scoped-with-current-fixture":
        required += [
            (test, "fixture consumer")
            for registration in _fixture_registrations(data)
            for test in registration["tests"]
        ]
    changed = _changes_since(base)
    derivation = derive(
        graph, changed, readers=_readers(data), required=required, depth=depth
    )
    if args.ticket_id <= 0:
        raise ControlSurfaceError("--ticket-id must be a positive integer")
    fallback = (*derivation.fallback_reasons, *_risk_fallback(args.risk_reason))
    return Selection(
        mode=args.mode,
        base_sha=base,
        ticket_id=args.ticket_id,
        test_paths=derivation.test_paths,
        changed_paths=changed,
        fallback_reasons=fallback,
        checkpoint_recommended=bool(fallback),
        recommendation_reasons=fallback,
        includes=includes,
        depth=depth,
        expansion_reason=reason.strip() if reason else None,
        derivation=derivation,
    )


def _file_identity(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": _relative(path),
        "byte_count": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _control_surface_identity(data: dict[str, Any]) -> dict[str, Any]:
    execution = data["execution_fixture"]
    discovered = _discovered_tests()
    return {
        "schema_version": "bio-neb.testing-control-surface/4",
        "discovered_test_modules": len(discovered),
        "discovered_test_methods": len(_test_method_ids(discovered)),
        "test_module_inventory": sorted(discovered),
        "ownership": _file_identity(_OWNERSHIP_PATH),
        "router": _file_identity(_ROUTER_PATH),
        "guides": [
            _file_identity(_TESTING_GUIDE_PATH),
            _file_identity(_OPERATOR_GUIDE_PATH),
        ],
        "execution_fixture": {
            **_file_identity(_REPOSITORY / execution["path"]),
            "loader": _file_identity(_REPOSITORY / execution["loader"]),
            "test_consumers": list(execution["tests"]),
            "scenario_count": execution["scenario_count"],
            "source_tickets": list(execution["source_tickets"]),
            "source_commit": execution["source_commit"],
            "evidence_kind": execution["evidence_kind"],
            "purpose": execution["purpose"],
            "grants_live_authority": execution["grants_live_authority"],
        },
        "additional_execution_fixtures": {
            name: {
                **_file_identity(_REPOSITORY / registration["path"]),
                "loader": _file_identity(_REPOSITORY / registration["loader"]),
                "test_consumers": list(registration["tests"]),
                "scenario_count": registration["scenario_count"],
                "source_tickets": list(registration["source_tickets"]),
                "source_commit": registration["source_commit"],
                "evidence_kind": registration["evidence_kind"],
                "purpose": registration["purpose"],
                "grants_live_authority": registration["grants_live_authority"],
            }
            for name, registration in _additional_execution_fixtures(data).items()
        },
    }


def _run_command(argv: Sequence[str], *, phase: str) -> dict[str, Any]:
    """Run one closure, checkpoint, lint or measure command in its own Run sandbox.

    The checkpoint's suite phase is the runner itself running a batch: a controller,
    whose module children get their sandboxes from it.
    """
    started = time.perf_counter()
    ruff = tuple(argv[: len(_RUFF)]) == tuple(_RUFF)
    # Ruff's human text goes to the process's stderr (fd 2), so the runner's
    # stdout stays one JSON document a caller can `json.load`.
    output = 2 if ruff else None
    try:
        with (
            contextlib.nullcontext()
            if phase == _CHECKPOINT_SUITE
            else run_sandbox.run_sandbox(uv_cache=ruff)
        ) as sandbox:
            completed = subprocess.run(
                argv,
                cwd=_REPOSITORY,
                env=controller_environment()
                if sandbox is None
                else sandbox.environment,
                check=False,
                stdout=output,
            )
    except OSError as error:
        return {
            "phase": phase,
            "owner": f"bio_neb.devtools.testing.{phase}",
            "argv": list(argv),
            "returncode": 127,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "launch_error": f"{type(error).__name__}: {error}",
        }
    return {
        "phase": phase,
        "owner": f"bio_neb.devtools.testing.{phase}",
        "argv": list(argv),
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def _run_clean_gate() -> dict[str, Any]:
    argv = ["git", "status", "--porcelain=v1", "--untracked-files=all"]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            cwd=_REPOSITORY,
            env=controller_environment(),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return {
            "phase": "repository.clean",
            "owner": "bio_neb.devtools.testing.repository.clean",
            "argv": argv,
            "returncode": 127,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "launch_error": f"{type(error).__name__}: {error}",
        }
    output = completed.stdout.strip()
    return {
        "phase": "repository.clean",
        "owner": "bio_neb.devtools.testing.repository.clean",
        "argv": argv,
        "returncode": completed.returncode or (1 if output else 0),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "porcelain": output.splitlines(),
    }


def _owned_receipt_location() -> str:
    """Name the location testing receipts are retained under, and how it is set."""

    return (
        f"{testing_run_roots()} (`testing_run_roots` in "
        f"`python -m bio_neb.locations`, override {TESTING_RUN_ROOTS_OVERRIDE})"
    )


def _receipt_help(prefix: str = "") -> str:
    return (
        f"{prefix}fresh absolute JSON receipt path beneath --run-root, outside the "
        "repository"
    )


def _run_root_help(prefix: str = "") -> str:
    # argparse formats help with %-interpolation, so the resolved path is escaped.
    location = _owned_receipt_location().replace("%", "%%")
    return (
        f"{prefix}fresh native operator Run root that will contain the receipt: a "
        f"fresh directory under the owned testing location {location}"
    )


def _preflight_receipt(path: Path, run_root: Path) -> None:
    if not path.is_absolute() or not run_root.is_absolute():
        raise ControlSurfaceError("--run-root and --receipt must be absolute paths")
    try:
        resolved_root = run_root.resolve(strict=True)
    except OSError as error:
        raise ControlSurfaceError("--run-root must be an existing directory") from error
    resolved_path = path.resolve(strict=False)
    try:
        resolved_root.relative_to(_REPOSITORY)
    except ValueError:
        pass
    else:
        raise ControlSurfaceError(
            "--run-root must stay outside the repository; use a fresh root under "
            f"the owned testing location {_owned_receipt_location()}"
        )
    if not resolved_root.is_dir():
        raise ControlSurfaceError("--run-root must be an existing directory")
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ControlSurfaceError("--receipt must stay beneath --run-root") from error
    if path.exists():
        raise ControlSurfaceError("--receipt already exists")
    if not resolved_path.parent.is_dir():
        raise ControlSurfaceError("--receipt parent must already exist")
    if path.with_name(f".{path.name}.new").exists():
        raise ControlSurfaceError("receipt staging collision")


def _write_receipt(path: Path, document: dict[str, Any]) -> str:
    encoded = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    staging = path.with_name(f".{path.name}.new")
    staging.write_bytes(encoded)
    os.chmod(staging, 0o600)
    os.replace(staging, path)
    return hashlib.sha256(encoded).hexdigest()


def _changed_python(changed_paths: Sequence[str]) -> list[str]:
    return [
        path
        for path in changed_paths
        if path.endswith(".py") and (_REPOSITORY / path).is_file()
    ]


def _quality_commands(
    changed_python: Sequence[str], *, fix: bool = False
) -> list[tuple[str, list[str]]]:
    """Lint and format only the changed files, under the rules pyproject selects.

    Ticket 251 narrowed the lint phase from the whole repository to the changed set. The
    format phase already worked this way; running both from one changed list keeps a rule
    addition from turning untouched files red, which is the whole-repository baseline the
    route rules out. With `fix` the same two phases repair the changed files instead
    (ticket 234): `check --fix`, then `format`.
    """
    commands: list[tuple[str, list[str]]] = [
        ("quality.ruff_version", [*_RUFF, "--version"]),
    ]
    if changed_python and fix:
        commands.extend(
            [
                (
                    "quality.ruff_check_fix",
                    [*_RUFF, "check", "--fix", "--no-cache", *changed_python],
                ),
                (
                    "quality.ruff_format_fix",
                    [*_RUFF, "format", "--no-cache", *changed_python],
                ),
            ]
        )
    elif changed_python:
        commands.extend(
            [
                (
                    "quality.ruff_check",
                    [*_RUFF, "check", "--no-cache", *changed_python],
                ),
                (
                    "quality.ruff_format",
                    [*_RUFF, "format", "--check", "--no-cache", *changed_python],
                ),
            ]
        )
    return commands


_CHECKPOINT_SUITE = "tests.checkpoint"


def _checkpoint_commands(
    selection: Selection, batch_root: Path | None, concurrency: int
) -> list[tuple[str, list[str]]]:
    """The suite runs once, as the whole-suite batch whose ledger is the next compare
    baseline (ticket 226, owner option A); the static gates follow."""
    if selection.base_sha is None:
        raise ControlSurfaceError("checkpoint execution requires --base-sha")
    commands = [
        (
            _CHECKPOINT_SUITE,
            [
                sys.executable,
                "-m",
                "bio_neb.devtools.testing",
                "batch",
                "--all",
                "--root",
                "<batch-root>" if batch_root is None else str(batch_root),
                "--concurrency",
                str(concurrency),
                # A dated seed: each day's checkpoints shuffle methods the same way.
                "--order",
                _LONGEST_SHUFFLE,
                "--seed",
                datetime.now(UTC).strftime("%Y%m%d"),
            ],
        ),
        (
            "static.compile",
            [sys.executable, "-m", "compileall", "-q", "src", "tests"],
        ),
        (
            "static.adr_pages",
            [sys.executable, "tools/check_adr_pages.py", "docs/adr"],
        ),
        *_quality_commands(_changed_python(selection.changed_paths)),
    ]
    commands.extend(
        [
            (
                "dependency.lock",
                [str(uv_executable()), "lock", "--check", "--offline"],
            ),
            (
                "repository.diff",
                ["git", "diff", "--check", f"{selection.base_sha}...HEAD"],
            ),
        ]
    )
    return commands


def _execute_selection(
    selection: Selection,
    receipt: Path,
    run_root: Path,
    batch_root: Path | None = None,
    concurrency: int = 1,
) -> int:
    _preflight_receipt(receipt, run_root)
    checkpoint = selection.mode == "checkpoint"
    if checkpoint:
        if batch_root is None:
            raise ControlSurfaceError(
                "checkpoint runs its suite as a batch: --root <batch-root> is required"
            )
        commands = _checkpoint_commands(selection, batch_root, concurrency)
    elif selection.mode == "regression":
        commands = [
            (
                "tests.regression",
                [sys.executable, "-m", "unittest", "-q", *selection.test_ids],
            )
        ]
    else:
        phase = {
            "core": "tests.core",
            "scoped": "tests.scoped",
            "scoped-with-current-fixture": "tests.scoped_current_fixture",
        }[selection.mode]
        commands = [
            (
                phase,
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "-q",
                    *[module_name(path) for path in selection.test_paths],
                ],
            )
        ]
    # A checkpoint takes its fixed width, which its suite batch inherits as its ceiling,
    # so the host's other slots stay open for focused and reviewer checks.
    with _host_slots(
        test_queue.CHECKPOINT_WIDTH if checkpoint else 1, f"{selection.mode} {receipt}"
    ) as held:
        results = [_run_command(command, phase=phase) for phase, command in commands]
    if checkpoint:
        results.append(_run_clean_gate())
    returncode = next(
        (result["returncode"] for result in results if result["returncode"]), 0
    )
    failure = next(
        (
            {
                "phase": result["phase"],
                "owner": result["owner"],
                "returncode": result["returncode"],
            }
            for result in results
            if result["returncode"]
        ),
        None,
    )
    document = {
        # `/3` adds the wait for host test slots.
        "schema_version": "bio-neb.testing-result/3",
        "candidate_sha": _candidate_sha(),
        "run_root": str(run_root.resolve()),
        "control_surface": _control_surface_identity(_load_ownership()),
        "selection": selection.as_json(),
        "commands": results,
        "elapsed_seconds": round(
            sum(result["elapsed_seconds"] for result in results), 6
        ),
        "slot_wait_seconds": _slot_wait(held),
        "result": "passed" if returncode == 0 else "failed",
        "failure": failure,
        **({"batch_root": str(batch_root)} if checkpoint else {}),
        "structural_boundary": {
            "credential_file_requested": False,
            "live_command_authority": False,
            "automatic_rerun": False,
            # Only a checkpoint's suite batch may run modules at once.
            "parallel_execution": checkpoint and concurrency > 1,
            "result_overwrite": False,
            "cleanup_command_available": False,
        },
    }
    digest = _write_receipt(receipt, document)
    print(
        json.dumps(
            {
                "testing_receipt": str(receipt),
                "testing_receipt_sha256": digest,
                "result": document["result"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return returncode


def _focused_module(name: str) -> str:
    """The module a focused name runs in, as the unittest child imports it.

    A `.py` file in the checkout is first converted as unittest converts it. Each
    dotted component is then placed by the finders Python's import system uses for the
    child's search path (the checkout, `PYTHONPATH`, then this interpreter's path),
    with its precedence: a package or module in the first entry that has one, else a
    namespace package across every entry. A component is a submodule only inside a
    package; the rest are classes and methods. A name nothing places is kept whole,
    and unittest reports it.
    """
    file = _REPOSITORY / name
    relative = os.path.normpath(os.path.relpath(file, _REPOSITORY))
    if (
        name.lower().endswith(".py")
        and file.is_file()
        and not relative.startswith("..")
    ):
        name = relative[:-3].replace(os.sep, ".")
    parts = name.split(".")
    search = [
        str(_REPOSITORY),
        *(
            str(_REPOSITORY / entry)
            for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if entry
        ),
        *(entry for entry in sys.path if entry),
    ]
    locations, module = search, None
    for index in range(1, len(parts) + 1):
        fullname = ".".join(parts[:index])
        found, portions = None, []
        for entry in locations:
            finder = _path_entry_finder(entry)
            spec = None if finder is None else finder.find_spec(fullname)
            if spec is not None and spec.loader is not None:
                found = spec
                break
            if spec is not None:
                portions += spec.submodule_search_locations or []
        if found is None and not portions:
            break
        module = fullname
        if found is None:
            locations = portions  # a namespace package, across every root that has it
        elif found.submodule_search_locations is None:
            break  # a module: the rest are classes and methods
        else:
            locations = list(found.submodule_search_locations)
    return name if module is None else module


def _path_entry_finder(entry: str) -> Any:
    """The finder Python's import system uses for one search-path entry; it finds a
    module's file without running it or importing its parent package."""
    for hook in sys.path_hooks:
        try:
            return hook(entry)
        except ImportError:
            continue
    return None


def _focused(names: Sequence[str], select_all: bool) -> int:
    """Run `unittest` on named tests: one module under one host slot, or several,
    one process per module at the batch width (owner, 2026-10-26, #463 follow-up).

    The reviewer's and owner's quick check (#391): one `unittest` process from this
    checkout, in its own Run sandbox and session. A stop (SIGINT, SIGTERM, a lost
    terminal) is forwarded to its process group, and the group is gone before the
    slot returns to the queue. The whole suite runs as a batch (owner, #463).
    """
    # unittest discovers the whole suite for `discover`, or for options without a name.
    if (
        select_all
        or (bool(names) and names[0].lower() == "discover")
        or any(name.startswith("-") for name in names)
    ):
        raise ControlSurfaceError(
            "focused runs named tests; run the whole suite as "
            "`testing batch --all --root <batch-root>`"
        )
    if not names:
        raise ControlSurfaceError("name the tests to run")
    modules: dict[str, list[str]] = {}
    for name in names:
        modules.setdefault(_focused_module(name), []).append(name)
    if len(modules) > 1:
        return _focused_modules(modules)
    argv = [sys.executable, "-m", "unittest", *names]
    stops = [signal.SIGINT, signal.SIGTERM]
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        stops.append(signal.SIGHUP)
    with (
        _host_slots(1, f"focused {' '.join(names)}"),
        run_sandbox.run_sandbox() as sandbox,
    ):
        # A stop from here on is held until the test runs, then forwarded to it.
        children: list[subprocess.Popen[bytes]] = []
        received: list[int] = []

        def forward(signum: int, frame: Any) -> None:
            received.append(signum)
            for child in children:
                _signal_group(child, signum)

        previous = {stop: signal.signal(stop, forward) for stop in stops}
        try:
            try:
                children.append(
                    subprocess.Popen(
                        argv,
                        cwd=_REPOSITORY,
                        env=sandbox.environment,
                        start_new_session=True,
                    )
                )
                for signum in received:
                    _signal_group(children[0], signum)
                # Unreaped, the test keeps its group ID reserved for the signals below.
                while not _wait_unreaped(children[0], 1.0) and not received:
                    pass
            finally:
                if children and received:
                    # A stopped test gets the batch's grace, then its group is killed.
                    _stop_module(children[0])
                elif children:
                    # No process of the test outlives its slot.
                    _signal_group(children[0], signal.SIGKILL)
                    children[0].wait()
        finally:
            # Only now may a further stop end the controller: the test is gone.
            for stop, handler in previous.items():
                signal.signal(stop, handler)
        returncode = children[0].returncode
    return returncode if returncode >= 0 else 128 - returncode


def _focused_modules(modules: Mapping[str, Sequence[str]]) -> int:
    """Run each module's named tests in its own `unittest` process, at most the
    batch width at once (4, or 8 as the host's sole runner) under the host queue.

    Each process has its own Run sandbox and session; its output is printed whole,
    after a `== <module>: exit <code>` line, as it ends. The run's exit is the first
    module's nonzero exit in name order, else 0. A stop is forwarded to every running
    process group, and none outlives the slots.
    """
    width = min(len(modules), _default_concurrency())
    sole = min(len(modules), _SOLE_RUNNER_CONCURRENCY, os.cpu_count() or 1)
    stops = [signal.SIGINT, signal.SIGTERM]
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        stops.append(signal.SIGHUP)
    running: list[subprocess.Popen[bytes]] = []
    received: list[int] = []
    lock = threading.Lock()
    exits: dict[str, int] = {}

    def forward(signum: int, frame: Any) -> None:
        received.append(signum)
        with lock:
            for child in running:
                _signal_group(child, signum)

    def run(module: str) -> None:
        if received:
            return
        with (
            run_sandbox.run_sandbox() as sandbox,
            tempfile.TemporaryFile() as output,
        ):
            child = subprocess.Popen(
                [sys.executable, "-m", "unittest", *modules[module]],
                cwd=_REPOSITORY,
                env=sandbox.environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with lock:
                running.append(child)
            try:
                while not _wait_unreaped(child, 1.0) and not received:
                    pass
            finally:
                if received:
                    _stop_module(child)
                else:
                    # No process of the module outlives its slot.
                    _signal_group(child, signal.SIGKILL)
                    child.wait()
                with lock:
                    running.remove(child)
            code = child.returncode if child.returncode >= 0 else 128 - child.returncode
            output.seek(0)
            with lock:
                exits[module] = code
                sys.stderr.write(f"== {module}: exit {code}\n")
                sys.stderr.write(output.read().decode("utf-8", "replace"))
                sys.stderr.flush()

    with _host_slots(
        width, f"focused {' '.join(modules)}", sole=sole if sole > width else None
    ) as held:
        previous = {stop: signal.signal(stop, forward) for stop in stops}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=held.count) as pool:
                for future in [pool.submit(run, module) for module in modules]:
                    future.result()
        finally:
            for stop, handler in previous.items():
                signal.signal(stop, handler)
    if received:
        return 128 + received[0]
    return next((exits[module] for module in modules if exits.get(module)), 0)


def _review_flag(args: argparse.Namespace) -> int:
    """Print the review flag's state after the requested action; return its status."""
    queue = test_queue_root()
    if args.action == "claim":
        status, state = review_flag.claim(
            queue,
            ticket=args.ticket,
            map_id=args.map,
            worker=args.worker,
            orchestrator=args.orchestrator,
        )
    elif args.action == "release":
        status, state = review_flag.release(queue, ticket=args.ticket, stale=args.stale)
    else:
        status, state = review_flag.show(queue)
    print(json.dumps({"review_flag": state}, sort_keys=True), flush=True)
    return status


def _working_tree_paths() -> tuple[str, ...]:
    """Paths the working tree changes, so the fast loop sees uncommitted work."""
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=_REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise ControlSurfaceError("cannot derive the working-tree change set")
    paths = []
    for line in completed.stdout.splitlines():
        entry = line[3:].strip()
        if not entry:
            continue
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        paths.append(entry.strip('"').replace("\\", "/"))
    return tuple(dict.fromkeys(paths))


def _lint(
    base_sha: str, receipt: Path | None, run_root: Path | None, *, fix: bool = False
) -> int:
    """Run the changed-file quality gate alone, for the inner iteration loop.

    Checkpoint binds its change set to base...HEAD because it also gates on a clean tree.
    This mode adds the working tree, so a ticket lints what it is about to commit rather
    than only what it already committed. A receipt is optional here: retained proof comes
    from the checkpoint phase, and requiring an operator root would defeat a fast loop.
    `fix` repairs the same changed files instead of only checking them.
    """
    if (receipt is None) != (run_root is None):
        raise ControlSurfaceError("--run-root and --receipt are required together")
    changed = _changed_python(_changes_since(base_sha))
    commands = _quality_commands(changed, fix=fix)
    if receipt is not None and run_root is not None:
        _preflight_receipt(receipt, run_root)
    results = [_run_command(argv, phase=phase) for phase, argv in commands]
    returncode = next(
        (result["returncode"] for result in results if result["returncode"]), 0
    )
    document = {
        "schema_version": "bio-neb.testing-lint/1",
        "candidate_sha": _candidate_sha(),
        "base_sha": base_sha,
        "changed_python": list(changed),
        "fix": fix,
        "commands": results,
        "elapsed_seconds": round(
            sum(result["elapsed_seconds"] for result in results), 6
        ),
        "result": "passed" if returncode == 0 else "failed",
        "failure": next(
            (
                {
                    "phase": result["phase"],
                    "owner": result["owner"],
                    "returncode": result["returncode"],
                }
                for result in results
                if result["returncode"]
            ),
            None,
        ),
        "proof_limits": (
            "changed-file lint and format only; proves no test, contract or "
            "repository behavior"
        ),
    }
    if receipt is not None:
        document["run_root"] = str(run_root.resolve()) if run_root else None
        digest = _write_receipt(receipt, document)
        print(
            json.dumps(
                {
                    "testing_receipt": str(receipt),
                    "testing_receipt_sha256": digest,
                    "result": document["result"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(json.dumps(document, indent=2, sort_keys=True), flush=True)
    return returncode


def _measure(data: dict[str, Any], receipt: Path, run_root: Path) -> int:
    _preflight_receipt(receipt, run_root)
    discovered = sorted(_discovered_tests())
    modules: list[dict[str, Any]] = []
    failed = False
    with _host_slots(1, f"measure {receipt}"):
        for path in discovered:
            samples = []
            for index in range(4):
                result = _run_command(
                    [sys.executable, "-m", "unittest", "-q", module_name(path)],
                    phase=f"measure.{module_name(path)}.{index}",
                )
                result["sample"] = "cold" if index == 0 else f"warm-{index}"
                samples.append(result)
                failed = failed or result["returncode"] != 0
            warm = [sample["elapsed_seconds"] for sample in samples[1:]]
            warm_mean = statistics.fmean(warm)
            warm_median = statistics.median(warm)
            warm_deviation = statistics.pstdev(warm)
            modules.append(
                {
                    "path": path,
                    "samples": samples,
                    "warm": {
                        "minimum_seconds": min(warm),
                        "median_seconds": warm_median,
                        "maximum_seconds": max(warm),
                        "mean_seconds": warm_mean,
                        "population_stdev_seconds": warm_deviation,
                        "coefficient_of_variation": (
                            warm_deviation / warm_mean if warm_mean else 0.0
                        ),
                    },
                    "cold_to_warm_median_ratio": (
                        samples[0]["elapsed_seconds"] / warm_median
                        if warm_median
                        else None
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "timed_module": path,
                        "completed_modules": len(modules),
                        "total_modules": len(discovered),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    ownership_bytes = _OWNERSHIP_PATH.read_bytes()
    document = {
        "schema_version": "bio-neb.testing-module-timings/2",
        "measured_candidate_sha": _candidate_sha(),
        "run_root": str(run_root.resolve()),
        "ownership_path": _relative(_OWNERSHIP_PATH),
        "ownership_sha256": hashlib.sha256(ownership_bytes).hexdigest(),
        "control_surface": _control_surface_identity(data),
        "module_inventory": discovered,
        "method": {
            "ordering": "sorted module paths, serial",
            "cold": (
                "first fresh-interpreter observation in one freshly provisioned "
                "native capsule; host filesystem caches are not evicted"
            ),
            "warm": "three immediate fresh-interpreter observations",
            "failure_meaning": "any nonzero sample makes the measurement red",
            "cost_proxies": "raw elapsed observations, not file size or test count",
        },
        "modules": modules,
        "result": "failed" if failed else "passed",
        "structural_boundary": {
            "credential_file_requested": False,
            "live_command_authority": False,
            "automatic_rerun": False,
            "parallel_execution": False,
            "result_overwrite": False,
            "cleanup_command_available": False,
        },
    }
    digest = _write_receipt(receipt, document)
    print(
        json.dumps(
            {
                "timing_receipt": str(receipt),
                "timing_receipt_sha256": digest,
                "result": document["result"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if failed else 0


# `/5` adds the wait for host test slots to the post-batch line; `/4` runs a
# heavy module as shards of its tests, one finish line each; `/3` finish lines
# added each test's seconds and the module process's peak RSS.
_LEDGER_SCHEMA = "bio-neb.testing-batch-ledger/5"
_SHARDED_LEDGER_SCHEMAS = (_LEDGER_SCHEMA, "bio-neb.testing-batch-ledger/4")
# A `/1` ledger has no failure reasons; it still compares, with its reasons unavailable.
_REASONLESS_LEDGER_SCHEMA = "bio-neb.testing-batch-ledger/1"
_READABLE_LEDGER_SCHEMAS = (
    *_SHARDED_LEDGER_SCHEMAS,
    "bio-neb.testing-batch-ledger/3",
    "bio-neb.testing-batch-ledger/2",
    _REASONLESS_LEDGER_SCHEMA,
)
_LEDGER_NAME = "ledger.jsonl"
# A snapshot run's wait for its host slots, one line per run.
_SLOT_WAITS_NAME = "slot-waits.jsonl"
# A full batch runs whole modules. A representative batch runs each laddered module's
# representative IDs and expands a red one to the rest of its module; its result is
# supplemental and never acceptance proof.
_FULL_RUNG = "full"
_REPRESENTATIVE_RUNG = "representative"
_EXPANSION_PHASE = "expansion"
_DEFAULT_MODULE_TIMEOUT_SECONDS = 3600.0
# Without --timeout each module's limit derives from the previous ledgers: five times the
# module's own last recorded seconds, at least two minutes and never above the flat
# default, which a module no ledger finished keeps.
_DERIVED_TIMEOUT_MULTIPLE = 5.0
_DERIVED_TIMEOUT_MINIMUM_SECONDS = 120.0
_TIMEOUT_GRACE_SECONDS = 10.0
# Batches run 4 modules at once unless told otherwise, and 8 when the batch is the host's
# sole runner; an ordinary batch asked for more runs at most 10. A machine with fewer
# CPUs runs one module per CPU. Isolation is the Run sandbox's by construction (ADR
# 0041), so these widths are capacity choices the owner makes, not an approval of the
# runner's source. An experiment root (`--parallel-experiment`) may run up to the CPU
# count; it is supplemental. A checkpoint's suite batch keeps the default as its ceiling.
_DEFAULT_CONCURRENCY = 4
_SOLE_RUNNER_CONCURRENCY = 8
_CONCURRENCY_CEILING = 10
_LONGEST = "longest"
_SHUFFLE = "shuffle"
# Slowest modules first, each module's methods shuffled: the checkpoint's order, so the
# seed reorders tests without moving the slowest module last.
_LONGEST_SHUFFLE = f"{_LONGEST}+{_SHUFFLE}"
_LONGEST_SHUFFLE_PREFIX = f"{_LONGEST_SHUFFLE}:"
_ORDERS = (_LONGEST, "sorted", "reversed", _SHUFFLE, _LONGEST_SHUFFLE)
# A batch runs a heavy module as shards of its tests, each one fresh process like a
# whole module. A module splits when its estimated seconds exceed the
# batch's estimated seconds over its slots, halved: two units per slot keep the last
# ones short whatever the estimates miss. Each shard holds at least
# `_MINIMUM_SHARD_SECONDS` of recorded test time, so its own process start and class
# setup stay small beside it. Shard `n` of a module is the unit `<module>#<n>`.
_UNITS_PER_SLOT = 2
_MINIMUM_SHARD_SECONDS = 30.0
_SHARD_MARK = "#"
# Catalog Run states a batch builds once before its modules start; module processes Fork
# from them read-only, and their Run sandboxes name the store.
# The builder entry is `build_store(store)` of the module the ownership intent names as
# `prerequisite_builder`.
_PREREQUISITE_DIRECTORY = "prerequisites"
_PREREQUISITE_PROGRAM = (
    "import importlib, sys\n"
    "builder = importlib.import_module(sys.argv[1])\n"
    "raise SystemExit(builder.build_store(sys.argv[2]))\n"
)
# What a module child measures of itself: each test's seconds, setUp to tearDown, by
# test ID; its own peak RSS; and the largest peak of the processes it started and
# waited for, in KiB.
_MODULE_MEASURES = ("test_seconds", "peak_rss_kib", "descendant_peak_rss_kib")
# Each module child runs this program so its red test IDs are recorded beside its log.
_MODULE_CHILD_PROGRAM = (
    "import sys\n"
    "from bio_neb.devtools.testing import _record_module\n"
    "raise SystemExit(_record_module(*sys.argv[1:]))\n"
)


def _failure_reason(traceback_text: str) -> str:
    """A formatted traceback's final exception, header line to last message line.

    The whole message is kept because a multi-line assertion (a set or list difference)
    names what differs only after its first line.
    """
    lines = traceback_text.rstrip().splitlines()
    frames = [index for index, line in enumerate(lines) if line.startswith("  File ")]
    tail = lines[frames[-1] + 1 if frames else 0 :]
    start = next(
        (index for index, line in enumerate(tail) if line and not line[0].isspace()),
        0,
    )
    return "\n".join(tail[start:])


def _record_module(module: str, result_path: str, phase_json: str = "{}") -> int:
    """Run one module like `python -m unittest -q <module>` and record its red test IDs.

    Each red ID's reason is its final exception's message. The result also records
    `_MODULE_MEASURES`.

    `phase_json` may name `include` IDs (run only those) or `exclude` IDs (run the rest of
    the module). A filtered run also records how many tests ran of the module's
    total; an included ID the module does not load is recorded as errored. A module
    that fails to load runs its load failure under any filter instead, so every part of
    it fails as the whole module would. A `shuffle` seed reorders the classes and each
    class's methods, keeping a class's methods together so its class-level setup still
    runs once.
    """
    import resource
    import unittest

    test_seconds: dict[str, float] = {}

    class TimedResult(unittest.TextTestResult):
        def addDuration(self, test, elapsed):  # noqa: N802 - unittest API
            super().addDuration(test, elapsed)
            test_seconds[test.id()] = round(elapsed, 6)

    class TimedRunner(unittest.TextTestRunner):
        resultclass = TimedResult

    phase = json.loads(phase_json)
    include, exclude = phase.get("include"), phase.get("exclude")
    shuffle = phase.get("shuffle")
    loaded: dict[str, Any] = {}

    def flatten(suite: unittest.TestSuite) -> list[unittest.TestCase]:
        tests = []
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                tests.extend(flatten(test))
            else:
                tests.append(test)
        return tests

    def load_failure(test: unittest.TestCase) -> bool:
        return test.id().startswith("unittest.loader._FailedTest.")

    class PhaseLoader(unittest.TestLoader):
        def loadTestsFromNames(self, names, module=None):  # noqa: N802 - unittest API
            tests = flatten(super().loadTestsFromNames(names, module))
            loaded["ids"] = {test.id() for test in tests}
            loaded["failed"] = any(map(load_failure, tests))
            if include is not None:
                tests = [
                    test for test in tests if test.id() in include or load_failure(test)
                ]
            elif exclude is not None:
                tests = [test for test in tests if test.id() not in exclude]
            if shuffle is not None:
                generator = random.Random(shuffle)
                classes: dict[type, list[unittest.TestCase]] = {}
                for test in tests:
                    classes.setdefault(type(test), []).append(test)
                groups = list(classes.values())
                generator.shuffle(groups)
                for group in groups:
                    generator.shuffle(group)
                tests = [test for group in groups for test in group]
            return unittest.TestSuite(tests)

    filtered = include is not None or exclude is not None
    program = unittest.main(
        module=None,
        argv=["python -m unittest", "-q", module],
        testLoader=(
            PhaseLoader()
            if filtered or shuffle is not None
            else unittest.defaultTestLoader
        ),
        testRunner=TimedRunner,
        exit=False,
    )
    result = program.result
    missing = (
        []
        if loaded.get("failed")
        else sorted(set(include or ()) - loaded.get("ids", set()))
    )
    reasons = {
        **{test.id(): _failure_reason(text) for test, text in result.failures},
        **{test.id(): "unexpected success" for test in result.unexpectedSuccesses},
        **{test.id(): _failure_reason(text) for test, text in result.errors},
        **{test_id: "not loaded by the module" for test_id in missing},
    }
    failed = {test.id() for test, _ in result.failures}
    failed.update(test.id() for test in result.unexpectedSuccesses)
    errored = {test.id() for test, _ in result.errors}
    errored.update(missing)
    document: dict[str, Any] = {
        "failed": sorted(failed),
        "errored": sorted(errored),
        "reasons": reasons,
        "test_seconds": dict(sorted(test_seconds.items())),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "descendant_peak_rss_kib": resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss,
    }
    if filtered:
        document["tests_run"] = result.testsRun
        document["module_tests"] = len(loaded.get("ids", ()))
    Path(result_path).write_text(json.dumps(document) + "\n", encoding="utf-8")
    if missing:
        return 1
    if result.testsRun == 0 and not result.skipped and exclude is None:
        return 5  # unittest's own exit status when no tests ran
    return 0 if result.wasSuccessful() else 1


def _checkout_status(checkout: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
        # A status read must not take the index lock a running module's git may need.
        env=controller_environment(),
    )
    if completed.returncode:
        raise ControlSurfaceError(f"cannot read checkout status of {checkout}")
    return completed.stdout.splitlines()


def _batch_header(  # noqa: PLR0913 - one argument per bound batch input
    modules: Sequence[str],
    timeout: float,
    checkout: Path,
    schedule: dict[str, Any],
    representatives: dict[str, list[str]] | None = None,
    base_commit: str | None = None,
    module_timeouts: dict[str, float] | None = None,
) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode:
        raise ControlSurfaceError(f"cannot resolve HEAD of {checkout}")
    dirty = _checkout_status(checkout)
    lock = checkout / "uv.lock"
    header = {
        "schema": _LEDGER_SCHEMA,
        "commit": head.stdout.strip(),
        "clean_tree": not dirty,
        "dirty_paths": dirty,
        "lock_sha256": (
            hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None
        ),
        "interpreter": sys.executable,
        "selection": list(modules),
        "rung": _FULL_RUNG,
        "timeout_seconds": float(timeout),
        **schedule,
    }
    if module_timeouts:
        header["module_timeout_seconds"] = dict(sorted(module_timeouts.items()))
    if representatives is not None:
        header.update(
            rung=_REPRESENTATIVE_RUNG,
            supplemental_only=True,
            base_commit=base_commit,
            representatives=representatives,
        )
    return header


def _complete_ledger_bytes(ledger: Path) -> tuple[bytes, int]:
    """Return a ledger's complete lines and its full size.

    A final line without its newline is a record interrupted mid-write and is not part of
    the ledger.
    """
    try:
        content = ledger.read_bytes()
    except OSError as error:
        raise ControlSurfaceError(f"cannot read batch ledger {ledger}") from error
    return content[: content.rfind(b"\n") + 1], len(content)


@dataclass(frozen=True)
class _Unit:
    """One process of a batch: a whole module, or shard `shard` of its `shards`."""

    module: str
    shard: int | None = None
    shards: int | None = None

    @property
    def label(self) -> str:
        if self.shard is None:
            return self.module
        return f"{self.module}{_SHARD_MARK}{self.shard}"


def _units(modules: Sequence[str], shards: dict[str, list[list[str]]]) -> list[_Unit]:
    """A batch's units: each planned module's shards, the last running its rest, and
    every other module whole."""
    units = []
    for module in modules:
        count = len(shards[module]) + 1 if module in shards else 0
        units.extend(
            [_Unit(module, shard, count) for shard in range(1, count + 1)]
            or [_Unit(module)]
        )
    return units


def _unit_phase(
    unit: _Unit, shards: dict[str, list[list[str]]]
) -> dict[str, list[str]] | None:
    """The tests a shard runs: its planned IDs, or for the last, every other test."""
    if unit.shard is None:
        return None
    planned = shards[unit.module]
    if unit.shard <= len(planned):
        return {"include": planned[unit.shard - 1]}
    return {"exclude": sorted({test_id for ids in planned for test_id in ids})}


def _unit_label(record: dict[str, Any]) -> str:
    return _Unit(record["module"], record.get("shard")).label


def _unit_key(label: str) -> tuple[str, int]:
    """Sorts a module's shards by number after the units before them."""
    module, _, shard = label.partition(_SHARD_MARK)
    return module, int(shard or 0)


def _planned_shards(header: dict[str, Any]) -> dict[str, list[list[str]]]:
    """The shards a ledger planned; a ledger before `/4` ran every module whole."""
    return header["shards"] if header["schema"] in _SHARDED_LEDGER_SCHEMAS else {}


def _read_ledger(
    ledger: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Return a ledger's header, finish lines by unit label and post-batch check.

    A laddered module's representative and expansion lines fold into one finish whose
    `expansion` holds the second phase. A shard's line must be one its header planned.
    """
    complete, _ = _complete_ledger_bytes(ledger)
    try:
        records = [json.loads(line) for line in complete.splitlines()]
    except json.JSONDecodeError as error:
        raise ControlSurfaceError(f"unreadable batch ledger {ledger}") from error
    if not records or records[0].get("schema") not in _READABLE_LEDGER_SCHEMAS:
        raise ControlSurfaceError(f"{ledger} has no {_LEDGER_SCHEMA} header")
    header, finishes, after = records[0], {}, None
    shards = _planned_shards(header)
    for record in records[1:]:
        module = record.get("module")
        if after is not None or (
            module is not None and module not in header["selection"]
        ):
            raise ControlSurfaceError(f"inconsistent batch ledger {ledger}")
        if module is None:
            after = record
            continue
        if (module in shards) != ("shard" in record) or (
            "shard" in record
            and not (
                record["shards"] == len(shards[module]) + 1
                and 1 <= record["shard"] <= record["shards"]
            )
        ):
            raise ControlSurfaceError(f"inconsistent batch ledger {ledger}")
        folded = _fold_finish(finishes, record)
        if not folded:
            raise ControlSurfaceError(f"inconsistent batch ledger {ledger}")
    return header, finishes, after


def _fold_finish(finishes: dict[str, dict[str, Any]], record: dict[str, Any]) -> bool:
    """Add one ledger line to its unit's finish; False when it does not fit there."""
    unit, phase = _unit_label(record), record.get("phase")
    finish = finishes.get(unit)
    if phase == _EXPANSION_PHASE:
        if finish is None or not _awaits_expansion(finish):
            return False
        finish["expansion"] = {
            key: value
            for key, value in record.items()
            if key not in ("module", "phase")
        }
        return True
    if finish is not None:
        return False
    finishes[unit] = (
        {**record, "expansion": None} if phase == _REPRESENTATIVE_RUNG else dict(record)
    )
    return True


def _awaits_expansion(finish: dict[str, Any]) -> bool:
    """A recorded red representative phase whose expansion has not finished.

    Only a nonzero exit that names its red test IDs expands. A timeout, a missing result
    or a nonzero exit naming no red test cannot be told apart from an incomplete run, so it
    does not expand; reproduce it in isolation instead.
    """
    return (
        finish.get("phase") == _REPRESENTATIVE_RUNG
        and finish["expansion"] is None
        and finish["exit"] != 0
        and _phase_state(finish) == "finished"
    )


def _write_line(stream: Any, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _append_line(ledger: Path, record: dict[str, Any]) -> None:
    """Append one ledger line; a finish line first gains its red IDs' reasons.

    The reasons come from the module's result file beside its log.
    """
    if "log" in record:
        record["reasons"] = _recorded_reasons(ledger.parent, record)
    with ledger.open("a", encoding="utf-8") as stream:
        _write_line(stream, record)


def _recorded_reasons(root: Path, finish: dict[str, Any]) -> dict[str, str] | None:
    """The reasons of a finish line's red IDs, or None when it recorded no result."""
    if finish["failed"] is None:
        return None
    result = (root / finish["log"]).with_suffix(".result.json")
    try:
        reasons = json.loads(result.read_text(encoding="utf-8"))["reasons"]
        return {
            test_id: reasons[test_id]
            for test_id in (*finish["failed"], *finish["errored"])
        }
    except (OSError, ValueError, KeyError) as error:
        raise ControlSurfaceError(f"{result} records no reason for {error}") from error


def _start_ledger(ledger: Path, header: dict[str, Any]) -> None:
    """Publish the header whole, so an interrupted start never leaves a headerless ledger."""
    staging = ledger.with_name(f".{ledger.name}.new")
    with staging.open("w", encoding="utf-8") as stream:
        _write_line(stream, header)
    try:
        os.link(staging, ledger)
    except OSError as error:
        raise ControlSurfaceError(f"cannot publish batch ledger {ledger}") from error
    finally:
        staging.unlink()


def _tracked_changes(dirty_paths: Sequence[str]) -> list[str]:
    """Checkout status entries other than untracked files."""
    return [entry for entry in dirty_paths if not entry.startswith("?? ")]


def _binding_changes(recorded: dict[str, Any], header: dict[str, Any]) -> list[str]:
    """Header fields whose change means the resume would test other inputs.

    Untracked files that appeared since the batch started are module residue, which the
    post-batch check reports; they do not stop the resume. Tracked edits do.
    """
    changed = [
        field
        for field in header
        if field not in ("clean_tree", "dirty_paths")
        and recorded.get(field) != header[field]
    ]
    if _tracked_changes(recorded["dirty_paths"]) != _tracked_changes(
        header["dirty_paths"]
    ):
        changed.append("tracked checkout changes")
    return sorted(changed)


def _signal_group(child: subprocess.Popen[bytes], stop: int) -> None:
    try:
        os.killpg(child.pid, stop)
    except ProcessLookupError:
        pass


class _BatchStopped(Exception):
    """Another slot of a concurrent batch was interrupted or failed."""


def _wait_unreaped(
    child: subprocess.Popen[bytes],
    timeout: float,
    stop: threading.Event | None = None,
) -> bool:
    """Wait for the module to exit without reaping it.

    The unreaped module keeps its process-group ID reserved, so the group can still be
    signalled without reaching an unrelated process that reused the ID. A set `stop`
    raises `_BatchStopped`.
    """
    deadline = time.monotonic() + timeout
    while True:
        if stop is not None and stop.is_set():
            raise _BatchStopped
        try:
            exited = os.waitid(
                os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
            )
        except ChildProcessError:
            return True  # already reaped, for example under an ignored SIGCHLD
        if exited is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _stop_module(child: subprocess.Popen[bytes]) -> None:
    """Stop a module's whole process group: SIGTERM, then SIGKILL after a grace.

    The SIGKILL is sent even when a second interrupt cuts the grace short.
    """
    try:
        _signal_group(child, signal.SIGTERM)
        _wait_unreaped(child, _TIMEOUT_GRACE_SECONDS)
    finally:
        _signal_group(child, signal.SIGKILL)
        child.wait()


@dataclass(frozen=True)
class _WriteWatch:
    """Before/after snapshots of shared locations outside a module's private roots.

    `trees` are walked whole and report created, modified and removed paths.
    `shallow` locations are shared with other host processes and too large to walk whole,
    so only entries created or removed in them or in their immediate subdirectories are
    reported; a module rewriting an existing file there is not seen. The batch root and the
    modules' private TMPDIRs are excluded. Snapshots taken while other slots run can
    include their writes; a concurrent report names candidates, not attribution.
    """

    trees: tuple[Path, ...]
    shallow: tuple[Path, ...]
    excluded: Path

    @classmethod
    def for_batch(cls, checkout: Path, root: Path) -> _WriteWatch:
        git_dir = subprocess.run(
            ["git", "rev-parse", "--absolute-git-dir"],
            cwd=checkout,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        trees = [checkout, local_state_root(), controller_home()]
        if git_dir:
            trees.append(Path(git_dir))
        shallow = [
            Path(tempfile.gettempdir()),
            Path("/dev/shm"),
            Path.home(),
            testing_run_roots(),
        ]
        return cls(
            tuple(dict.fromkeys(path.resolve() for path in trees)),
            tuple(dict.fromkeys(path.resolve() for path in shallow)),
            root.resolve(),
        )

    def _entries(self, directory: Path, depth: int | None) -> dict[str, tuple]:
        """Entries under `directory`, `depth` levels down (None: the whole tree)."""
        entries: dict[str, tuple] = {}
        pending = [(str(directory), 1)]
        try:
            entries[str(directory)] = ("d", directory.lstat().st_ino)
        except OSError:
            return entries
        temporary = str(Path(tempfile.gettempdir()).resolve())
        while pending:
            current, level = pending.pop()
            try:
                with os.scandir(current) as scan:
                    found = list(scan)
            except OSError:
                continue
            for entry in found:
                if entry.path == str(self.excluded) or (
                    current == temporary and entry.name.startswith(PRIVATE_PREFIX)
                ):
                    continue
                try:
                    status = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    entries[entry.path] = ("d", status.st_ino)
                    if depth is None or level < depth:
                        pending.append((entry.path, level + 1))
                else:
                    entries[entry.path] = (
                        "f",
                        status.st_ino,
                        status.st_size,
                        status.st_mtime_ns,
                    )
        return entries

    def snapshot(self) -> tuple[dict[str, tuple], set[str]]:
        trees: dict[str, tuple] = {}
        for tree in self.trees:
            trees.update(self._entries(tree, depth=None))
        shallow: set[str] = set()
        for directory in self.shallow:
            shallow.update(self._entries(directory, depth=2))
        return trees, shallow

    @staticmethod
    def writes(
        before: tuple[dict[str, tuple], set[str]],
        after: tuple[dict[str, tuple], set[str]],
    ) -> list[str]:
        """`created`, `modified` and `removed` paths; a created tree is named once."""
        (trees_before, top_before), (trees_after, top_after) = before, after
        created = (trees_after.keys() - trees_before.keys()) | (top_after - top_before)
        removed = (trees_before.keys() - trees_after.keys()) | (top_before - top_after)
        modified = {
            path
            for path in trees_before.keys() & trees_after.keys()
            if trees_before[path] != trees_after[path]
        }
        writes = []
        for kind, paths in (
            ("created", created),
            ("modified", modified),
            ("removed", removed),
        ):
            writes.extend(
                f"{kind} {path}"
                for path in paths
                if kind == "modified" or os.path.dirname(path) not in paths
            )
        return sorted(writes)


def _run_module(  # noqa: PLR0913 - one argument per module run input
    module: str,
    root: Path,
    *,
    timeout: float,
    checkout: Path,
    lock_fd: int,
    phase: dict[str, list[str]] | None = None,
    label: str | None = None,
    shuffle: str | None = None,
    stop: threading.Event | None = None,
    watch: _WriteWatch | None = None,
    store: Path | None = None,
) -> dict[str, Any]:
    """Run one module, or the `include`/`exclude` phase of it, in its own process
    inside its own Run sandbox, which names the prerequisite `store` when one is built."""
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    label = label or module
    attempt = 1
    stem = label
    while (logs / f"{stem}.log").exists():
        attempt += 1
        stem = f"{label}.{attempt}"
    log = logs / f"{stem}.log"
    result_path = logs / f"{stem}.result.json"
    child_phase = {**(phase or {}), **({"shuffle": shuffle} if shuffle else {})}
    before = watch.snapshot() if watch is not None else None
    started = time.monotonic()
    with (
        run_sandbox.run_sandbox(cache=root / "python-cache", store=store) as sandbox,
        log.open("xb") as stream,
    ):
        # The child inherits the ledger lock, so a resume cannot start while an
        # orphaned module from an earlier controller is still running.
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _MODULE_CHILD_PROGRAM,
                module,
                str(result_path),
                json.dumps(child_phase),
            ],
            cwd=checkout,
            env=sandbox.environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(lock_fd,),
        )
        try:
            exited = _wait_unreaped(child, timeout, stop)
        except BaseException:
            _stop_module(child)
            raise
        if exited:
            # No process of the module outlives its finish line.
            _signal_group(child, signal.SIGKILL)
            exit_status: int | str = child.wait()
        else:
            _stop_module(child)
            exit_status = "timeout"
        residue = sandbox.tmp_residue()
    seconds = round(time.monotonic() - started, 3)
    module_result = None
    if exit_status != "timeout" and result_path.is_file():
        try:
            module_result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            module_result = None
    finish = {
        "module": module,
        "exit": exit_status,
        "seconds": seconds,
        "log": log.relative_to(root).as_posix(),
    }
    recorded = ["failed", "errored", *_MODULE_MEASURES]
    if phase is not None:
        recorded += ["tests_run", "module_tests"]
    for key in recorded:
        finish[key] = module_result[key] if module_result else None
    if residue:
        finish["tmp_residue_entries"] = residue
    if watch is not None and before is not None:
        finish["writes_outside_private"] = watch.writes(before, watch.snapshot())
    return finish


def _run_order(
    units: Sequence[str], order: str, estimates: dict[str, float] | None = None
) -> list[str]:
    """The order units start in: sorted, reversed, seeded `shuffle:<seed>`, or for
    `longest` and `longest+shuffle:<seed>` the largest of `estimates` first, then the
    units without one, sorted."""
    ordered = sorted(units, key=_unit_key)
    if order == "reversed":
        ordered.reverse()
    elif _is_shuffle(order):
        random.Random(order).shuffle(ordered)
    elif order == _LONGEST or order.startswith(_LONGEST_SHUFFLE_PREFIX):
        seconds = estimates or {}
        ordered.sort(key=lambda unit: -seconds.get(unit, -1.0))
    return ordered


def _is_shuffle(order: str) -> bool:
    return order.startswith(f"{_SHUFFLE}:")


def _method_shuffle(order: str, module: str) -> str | None:
    """The seed that shuffles `module`'s methods under `order`, or None to keep them.

    Every shard of a module takes the module's seed, so a shard's methods run in the
    order they would keep within the whole module's shuffle.
    """
    if _is_shuffle(order) or order.startswith(_LONGEST_SHUFFLE_PREFIX):
        return f"{order}:{module}"
    return None


def _run_slots(
    units: Sequence[str],
    run: Callable[[str], None],
    slots: int,
    stop: threading.Event,
) -> None:
    """Run `run(unit)` for each unit in up to `slots` threads, in list order.

    An interrupt or a slot's failure sets `stop`: every running unit is stopped, no
    new one starts, and the first cause is raised once every slot has ended.
    """
    pending = iter(units)
    take = threading.Lock()
    errors: list[BaseException] = []

    def slot() -> None:
        while not stop.is_set():
            with take:
                unit = next(pending, None)
            if unit is None:
                return
            try:
                run(unit)
            except BaseException as error:  # noqa: BLE001 - re-raised by the controller
                errors.append(error)
                stop.set()
                return

    threads = [threading.Thread(target=slot) for _ in range(min(slots, len(units)))]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            while thread.is_alive():
                thread.join(0.1)
    except BaseException:
        stop.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join()
        raise
    causes = [error for error in errors if not isinstance(error, _BatchStopped)]
    if causes:
        raise causes[0]


def _run_batch(  # noqa: C901, PLR0913, PLR0915 - one argument per batch option
    modules: Sequence[str],
    root: Path,
    *,
    timeout: float = _DEFAULT_MODULE_TIMEOUT_SECONDS,
    checkout: Path = _REPOSITORY,
    representatives: dict[str, list[str]] | None = None,
    base_commit: str | None = None,
    concurrency: int = 1,
    order: str = "sorted",
    detect_writes: bool = False,
    parallel_experiment: bool = False,
    serial_only: Sequence[str] = (),
    module_timeouts: dict[str, float] | None = None,
    store: Path | None = None,
    shards: dict[str, list[list[str]]] | None = None,
    estimates: dict[str, float] | None = None,
    timing_ledgers: Sequence[str] = (),
    slot_wait_seconds: float | None = None,
) -> int:
    """Run each unit, a whole module or a shard of one, in its own process and record
    the batch in one ledger.

    `shards` splits each listed module into its listed ID groups plus a last shard
    running the module's other tests. `module_timeouts` gives listed units their own
    limit; the rest use `timeout`. `estimates` are the units' expected seconds, which
    a longest order starts largest first, and `timing_ledgers` the ledgers they came
    from. `store` is the prerequisite store each unit's Run sandbox names.

    With `representatives` the batch runs at the representative rung: each listed module
    runs its IDs and expands on red, and every other module runs whole.

    With `concurrency` above 1 that many units run at once; a laddered module's two
    phases stay in one slot, and `serial_only` modules run alone after the others. A
    `parallel_experiment` batch is evidence for a capacity decision and supplemental.
    `order` sets the start order; a `shuffle:<seed>` order also shuffles each module's
    test methods. `detect_writes` records each module's writes outside its private roots.

    An existing ledger in `root` resumes that batch: finished units, including reds and
    timeouts, are skipped and every unit without a finish line runs again. The
    post-batch line keeps `slot_wait_seconds`, the run's wait for its host test slots
    (None when a controller holds them for it).
    """
    if not root.is_absolute():
        raise ControlSurfaceError("--root must be an absolute path")
    resolved = root.resolve()
    if resolved == checkout or checkout in resolved.parents:
        raise ControlSurfaceError(
            "--root must stay outside the repository; use a root under the owned "
            f"testing location {_owned_receipt_location()}"
        )
    try:
        root.mkdir(mode=0o700, exist_ok=True)
    except OSError as error:
        raise ControlSurfaceError(f"cannot create --root {root}: {error}") from error
    shards = shards or {}
    bound_schedule: dict[str, Any] = {
        "concurrency": concurrency,
        "order": order,
        "detect_writes": detect_writes,
        "shards": shards,
        "estimated_seconds": estimates or {},
        "timing_ledgers": list(timing_ledgers),
    }
    alone = [module for module in modules if module in serial_only]
    if concurrency > 1:
        bound_schedule["serial_only"] = alone
    if parallel_experiment:
        bound_schedule.update(parallel_experiment=True, supplemental_only=True)
    with (root / "ledger.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ControlSurfaceError(
                f"a batch controller or module process for {root} is still running; "
                "confirm it has stopped before resuming"
            ) from error
        header = _batch_header(
            modules,
            timeout,
            checkout,
            bound_schedule,
            representatives,
            base_commit,
            module_timeouts,
        )
        ledger = root / _LEDGER_NAME
        if ledger.exists():
            recorded, finishes, after = _read_ledger(ledger)
            changed = _binding_changes(recorded, header)
            if changed:
                raise ControlSurfaceError(
                    f"{ledger} binds another batch (differs in {', '.join(changed)}); "
                    "use a fresh root for changed inputs"
                )
            complete, size = _complete_ledger_bytes(ledger)
            if len(complete) != size:
                with ledger.open("r+b") as stream:
                    stream.truncate(len(complete))
        else:
            _start_ledger(ledger, header)
            recorded, finishes, after = header, {}, None
        recording = threading.Lock()
        stop = threading.Event()
        watch = _WriteWatch.for_batch(checkout, root) if detect_writes else None

        def record(line: dict[str, Any]) -> bool:
            """Append one finish line; True when the module now awaits its expansion."""
            with recording:
                _append_line(ledger, line)
                _fold_finish(finishes, line)
                print(json.dumps(line, sort_keys=True), flush=True)
                return _awaits_expansion(finishes[_unit_label(line)])

        units = {unit.label: unit for unit in _units(modules, shards)}

        def run(label: str) -> None:
            unit = units[label]
            module = unit.module
            laddered = representatives is not None and module in representatives
            with recording:
                finish = finishes.get(label)
                expand = finish is not None and _awaits_expansion(finish)
                if finish is not None and not expand:
                    print(json.dumps({"skipped_finished": label}), flush=True)
                    return
            options = {
                "timeout": (module_timeouts or {}).get(label, timeout),
                "checkout": checkout,
                "lock_fd": lock.fileno(),
                "shuffle": _method_shuffle(order, module),
                "stop": stop,
                "watch": watch,
                "store": store,
            }
            if finish is None:
                line = _run_module(
                    module,
                    root,
                    phase=(
                        {"include": representatives[module]}
                        if laddered
                        else _unit_phase(unit, shards)
                    ),
                    label=label,
                    **options,
                )
                if unit.shard is not None:
                    line.update(shard=unit.shard, shards=unit.shards)
                if laddered:
                    line["phase"] = _REPRESENTATIVE_RUNG
                elif representatives is not None:
                    line["depth"] = _FULL_RUNG
                expand = record(line)
            if laddered and expand:
                # The rest of the module runs without rerunning a representative ID, so
                # the representative red stays the module's original result.
                expansion = _run_module(
                    module,
                    root,
                    phase={"exclude": representatives[module]},
                    label=f"{module}.{_EXPANSION_PHASE}",
                    **options,
                )
                del expansion["module_tests"]
                record({**expansion, "phase": _EXPANSION_PHASE})

        ordered = _run_order(list(units), order, estimates)
        if concurrency == 1:
            for label in ordered:
                run(label)
        else:
            _run_slots(
                [label for label in ordered if units[label].module not in alone],
                run,
                concurrency,
                stop,
            )
            for label in ordered:
                if units[label].module in alone:
                    run(label)
        if after is None:
            dirty_after = _checkout_status(checkout)
            after = {
                "clean_tree_after": not dirty_after,
                "dirty_paths_after": dirty_after,
                "slot_wait_seconds": slot_wait_seconds,
            }
            _append_line(ledger, after)
    green = all(finish["exit"] == 0 for finish in finishes.values())
    unchanged = after["dirty_paths_after"] == recorded["dirty_paths"]
    print(
        json.dumps(
            {
                "ledger": str(ledger),
                "rung": recorded["rung"],
                "supplemental_only": recorded.get("supplemental_only", False),
                **after,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if green and unchanged else 1


def _owned_root(root: Path, checkout: Path) -> None:
    """Create `root`, refusing a relative path or one inside the repository."""
    if not root.is_absolute():
        raise ControlSurfaceError("--root must be an absolute path")
    resolved = root.resolve()
    if resolved == checkout or checkout in resolved.parents:
        raise ControlSurfaceError(
            "--root must stay outside the repository; use a root under the owned "
            f"testing location {_owned_receipt_location()}"
        )
    try:
        root.mkdir(mode=0o700, exist_ok=True)
    except OSError as error:
        raise ControlSurfaceError(f"cannot create --root {root}: {error}") from error


def _build_prerequisites(
    modules: Sequence[str], root: Path, checkout: Path = _REPOSITORY
) -> Path | None:
    """Build the catalog Run states once in `<root>/prerequisites`, before any module.

    Runs only when a selected module's test-support surface reaches the store, in its
    own process from the checkout. The store's receipt records the source key and each
    state built or reused; a failed builder stops the batch with one named refusal and
    no module runs. Returns the store, or None when no selected module can use it.
    """
    builder = _prerequisite_builder(_load_ownership())
    users = _import_graph().users.reach.get(builder, set())
    if not {module_name(path) for path in users} & set(modules):
        return None
    _owned_root(root, checkout)
    store = root / _PREREQUISITE_DIRECTORY
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    with (
        run_sandbox.run_sandbox(cache=root / "python-cache") as sandbox,
        _root_lock(root, "ledger.lock") as lock,
        (logs / "prerequisites.log").open("ab") as log,
    ):
        # The builder inherits the root lock, so a resume cannot start while a
        # builder orphaned by a killed controller still writes the store.
        built = subprocess.run(
            [
                sys.executable,
                "-c",
                _PREREQUISITE_PROGRAM,
                module_name(builder),
                str(store),
            ],
            cwd=checkout,
            env=sandbox.environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            pass_fds=(lock.fileno(),),
        )
    try:
        receipt = json.loads((store / "receipt.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        receipt = None  # the builder ended before it wrote a receipt
    failed = None if receipt is None else receipt["failed"]
    if built.returncode or receipt is None or failed is not None:
        named = (
            f"{failed['state'] or 'store'}: {failed['reason']}"
            if failed is not None
            else f"store: exit {built.returncode}"
        )
        raise ControlSurfaceError(
            f"prerequisite builder failed ({named}); no module ran; "
            f"see {logs / 'prerequisites.log'}"
        )
    print(
        json.dumps(
            {
                "prerequisites": {
                    "store": str(store),
                    "key": receipt["key"],
                    "states": {
                        name: state["status"]
                        for name, state in sorted(receipt["states"].items())
                    },
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return store


def _interrupt_batch(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def _concurrency(
    requested: int | None, experiment: bool, ceiling: int = _CONCURRENCY_CEILING
) -> int | None:
    """An explicit width, never above `ceiling` for an ordinary batch nor above the CPU
    count; None leaves the width to `_width`."""
    cpus = os.cpu_count() or 1
    top = cpus if experiment else min(ceiling, cpus)
    if requested is not None and not 1 <= requested <= top:
        raise ControlSurfaceError(f"--concurrency must be from 1 to {top}")
    return requested


def _default_concurrency() -> int:
    return min(_DEFAULT_CONCURRENCY, os.cpu_count() or 1)


def _checkpoint_concurrency(requested: int | None) -> int:
    """A checkpoint's suite batch width: the default, which is also its ceiling."""
    explicit = _concurrency(requested, False, _DEFAULT_CONCURRENCY)
    return _default_concurrency() if explicit is None else explicit


def _width(requested: int | None, root: Path) -> tuple[int, int | None]:
    """The slots a batch asks for, and the wider count it takes as the sole runner.

    An explicit width is taken as asked. A resumed root asks for the width its header
    bound, since the header also binds the shards made for it. Otherwise the batch asks
    for the default and takes the sole-runner width when no other runner holds a slot.
    """
    if requested is not None:
        return requested, None
    try:
        return int(_bound_header(root)["concurrency"]), None
    except (KeyError, TypeError, ValueError):
        pass  # a fresh root
    default = _default_concurrency()
    sole = min(_SOLE_RUNNER_CONCURRENCY, os.cpu_count() or 1)
    return default, sole if sole > default else None


def _batch_order(order: str, seed: int | None) -> str:
    if order in (_SHUFFLE, _LONGEST_SHUFFLE):
        if seed is None:
            raise ControlSurfaceError(f"--order {order} needs --seed")
        return f"{order}:{seed}"
    if seed is not None:
        raise ControlSurfaceError("--seed applies only to a shuffle order")
    return order


def _previous_ledgers(root: Path) -> list[Path]:
    """Batch ledgers under the testing run roots, newest first, other than `root`'s."""
    runs = testing_run_roots()
    ledgers = [*runs.glob(f"*/{_LEDGER_NAME}"), *runs.glob(f"*/*/{_LEDGER_NAME}")]
    own = (root / _LEDGER_NAME).resolve()
    dated = []
    for ledger in ledgers:
        try:
            dated.append((ledger.stat().st_mtime, ledger))
        except OSError:
            continue
    return [
        ledger for _, ledger in sorted(dated, reverse=True) if ledger.resolve() != own
    ]


def _ledger_timings(ledger: Path) -> dict[str, dict[str, Any]]:
    """Each module one ledger finished at full depth: its `seconds` and `tests`, each
    test's recorded seconds (None before `/3` or without a result).

    A timeout line records the limit, not the module's duration, and a ladder phase ran
    only part of the module, so neither counts. A sharded module counts once all its
    shards finished: its seconds are their mean setup, a shard's seconds beyond its
    tests', plus all its tests' seconds. An unreadable ledger times nothing.
    """
    try:
        lines = _complete_ledger_bytes(ledger)[0].decode().splitlines()[1:]
    except (ControlSurfaceError, OSError, UnicodeDecodeError):
        return {}
    timings: dict[str, dict[str, Any]] = {}
    shard_lines: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not (
            isinstance(record, dict)
            and isinstance(record.get("module"), str)
            and "phase" not in record
            and type(record.get("exit")) is int
            and isinstance(record.get("seconds"), (int, float))
        ):
            continue
        if "shard" in record:
            shard_lines[record["module"]][record["shard"]] = record
            continue
        tests = record.get("test_seconds")
        timings[record["module"]] = {
            "seconds": float(record["seconds"]),
            "tests": tests if isinstance(tests, dict) else None,
        }
    for module, shards in shard_lines.items():
        counts = {line.get("shards") for line in shards.values()}
        tests = [line.get("test_seconds") for line in shards.values()]
        if counts != {len(shards)} or not all(isinstance(each, dict) for each in tests):
            continue
        setup = statistics.mean(
            line["seconds"] - sum(each.values())
            for line, each in zip(shards.values(), tests, strict=True)
        )
        merged = {
            test_id: seconds for each in tests for test_id, seconds in each.items()
        }
        timings[module] = {
            "seconds": max(0.0, setup) + sum(merged.values()),
            "tests": merged,
        }
    return timings


def _last_timings(
    modules: Sequence[str], root: Path
) -> dict[str, tuple[dict[str, Any], Path]]:
    """Each module's timing from the newest ledger that finished it at full depth."""
    last: dict[str, tuple[dict[str, Any], Path]] = {}
    for ledger in _previous_ledgers(root):
        wanted = set(modules) - set(last)
        if not wanted:
            break
        for module, timing in _ledger_timings(ledger).items():
            if module in wanted:
                last[module] = (timing, ledger)
    return last


def _shard_schedule(
    timings: dict[str, dict[str, Any]], splittable: Sequence[str], slots: int
) -> tuple[dict[str, list[list[str]]], dict[str, float]]:
    """The shards of each heavy module and every timed unit's estimated seconds.

    A module of `splittable` whose seconds exceed the batch's share, its timed seconds
    over twice its `slots`, splits into contiguous runs of its sorted test IDs, by
    their recorded seconds, as many as keep each run's setup plus tests within the
    share, each holding at least `_MINIMUM_SHARD_SECONDS` of tests. Only IDs the
    module still defines are planned; the last shard runs every other test, so a test
    the timing never saw still runs. One slot never splits.
    """
    share = sum(timing["seconds"] for timing in timings.values()) / (
        slots * _UNITS_PER_SLOT
    )
    shards: dict[str, list[list[str]]] = {}
    estimates: dict[str, float] = {}
    for module, timing in sorted(timings.items()):
        estimates[module] = timing["seconds"]
        if (
            slots == 1
            or module not in splittable
            or timing["tests"] is None
            or timing["seconds"] <= share
        ):
            continue
        setup = max(0.0, timing["seconds"] - sum(timing["tests"].values()))
        defined = _test_method_ids({f"{module.replace('.', '/')}.py"})
        tests = {
            test_id: seconds
            for test_id, seconds in sorted(timing["tests"].items())
            if test_id in defined
        }
        work = sum(tests.values())
        if share <= setup or not work:
            continue
        count = min(
            math.ceil(work / (share - setup)),
            int(work // _MINIMUM_SHARD_SECONDS),
            len(tests),
        )
        if count < 2:
            continue
        runs: list[list[str]] = [[] for _ in range(count)]
        elapsed = 0.0
        for test_id, seconds in tests.items():
            # Each test joins the run its midpoint falls in, so runs stay contiguous
            # and within one test of an equal split.
            runs[min(count - 1, int((elapsed + seconds / 2) / (work / count)))].append(
                test_id
            )
            elapsed += seconds
        runs = [ids for ids in runs if ids]
        if len(runs) < 2:
            continue
        del estimates[module]
        shards[module] = runs[:-1]
        for shard, ids in enumerate(runs, 1):
            estimates[_Unit(module, shard).label] = round(
                setup + sum(tests[test_id] for test_id in ids), 3
            )
    return shards, estimates


def _derived_limit(seconds: float) -> float:
    return min(
        _DEFAULT_MODULE_TIMEOUT_SECONDS,
        max(_DERIVED_TIMEOUT_MINIMUM_SECONDS, _DERIVED_TIMEOUT_MULTIPLE * seconds),
    )


def _bound_header(root: Path) -> dict[str, Any]:
    """The header of the ledger already in `root`, or {} for a fresh or unreadable one."""
    try:
        header = json.loads(
            (root / _LEDGER_NAME).read_text(encoding="utf-8").splitlines()[0]
        )
    except (OSError, IndexError, UnicodeDecodeError, ValueError):
        return {}
    return header if isinstance(header, dict) else {}


def _batch_plan(  # noqa: PLR0913 - one argument per bound batch input
    modules: Sequence[str],
    root: Path,
    explicit: float | None,
    order: str,
    slots: int,
    whole: Sequence[str],
) -> dict[str, Any]:
    """What a batch binds from the previous ledgers: `_run_batch`'s order, shards,
    estimates, timing ledgers and limits, plus the `timeout` line it prints.

    `whole` modules never split. A `longest` order without any timing starts sorted.
    Limits are five times each unit's estimate, at least two minutes and at most the
    flat default, which an untimed unit keeps; an explicit limit applies to every
    unit. A resumed root keeps what its header bound, so a resume is never refused
    because newer ledgers moved the derivation.
    """
    bound = _bound_header(root)
    try:
        if bound["schema"] != _LEDGER_SCHEMA:
            raise KeyError("schema")
        plan: dict[str, Any] = {
            "order": bound["order"] if order == _LONGEST else order,
            "shards": bound["shards"],
            "estimates": bound["estimated_seconds"],
            "timing_ledgers": bound["timing_ledgers"],
        }
        limits = {
            "seconds": float(bound["timeout_seconds"]),
            "modules": bound.get("module_timeout_seconds", {}),
            "basis": "resumed_root",
        }
    except (KeyError, TypeError, ValueError):
        # A fresh root; `_run_batch` refuses an unreadable header with its own reason.
        last = _last_timings(modules, root)
        shards, estimates = _shard_schedule(
            {module: timing for module, (timing, _) in last.items()},
            [module for module in modules if module not in whole],
            slots,
        )
        ledgers = sorted({str(ledger) for _, ledger in last.values()})
        plan = {
            "order": "sorted" if order == _LONGEST and not estimates else order,
            "shards": shards,
            "estimates": estimates,
            "timing_ledgers": ledgers,
        }
        limits = {
            "basis": "last_ledger_seconds",
            "seconds": _DEFAULT_MODULE_TIMEOUT_SECONDS,
            "modules": {
                unit: _derived_limit(seconds)
                for unit, seconds in sorted(estimates.items())
            },
            "unrecorded": sorted(set(modules) - set(last)),
            "ledgers": ledgers,
        }
    if explicit is not None:
        limits = {"seconds": float(explicit), "basis": "explicit"}
    return {
        **plan,
        "timeout": limits["seconds"],
        "module_timeouts": limits.get("modules", {}),
        "limits": limits,
    }


def _batch_command(  # noqa: PLR0913 - one argument per batch option
    names: Sequence[str],
    select_all: bool,
    root: Path,
    timeout: float | None,
    rung: str,
    base_sha: str | None,
    declared: dict[str, list[str]],
    schedule: dict[str, Any],
) -> int:
    """Run the named modules as one batch; `declared` maps laddered modules to their IDs.

    `schedule` holds `_run_batch`'s concurrency, order, write-detection, experiment and
    serial-only keywords. The previous ledgers give its shards, start estimates and,
    without `timeout`, its limits (`_batch_plan`); laddered and serial-only modules run
    whole.
    """
    discovered = {module_name(path) for path in _discovered_tests()}
    if select_all == bool(names):
        raise ControlSurfaceError("name test modules or --all (not both)")
    modules = discovered if select_all else {module_name(name) for name in names}
    unknown = sorted(modules - discovered)
    if unknown:
        raise ControlSurfaceError(f"not a discovered test module: {', '.join(unknown)}")
    if timeout is not None and not 0 < timeout < float("inf"):
        raise ControlSurfaceError("--timeout must be a positive number of seconds")
    ladder: dict[str, Any] = {}
    if rung == _REPRESENTATIVE_RUNG:
        if base_sha is None:
            raise ControlSurfaceError("--rung representative requires --base-sha")
        base_sha = _full_sha(base_sha)
        laddered = sorted(module for module in declared if module in modules)
        changed = _changed_test_methods(
            base_sha, [f"{module.replace('.', '/')}.py" for module in laddered]
        )
        ladder = {
            "representatives": {
                module: sorted(
                    {
                        *declared[module],
                        *(
                            test_id
                            for test_id in changed
                            if test_id.startswith(f"{module}.")
                        ),
                    }
                )
                for module in laddered
            },
            "base_commit": base_sha,
        }
    elif base_sha is not None:
        raise ControlSurfaceError("--base-sha applies only to --rung representative")
    # A capsule interrupt (SIGINT), a stop request (SIGTERM) and a lost terminal (SIGHUP)
    # end the batch the same way: the running module is stopped and stays unfinished for
    # the resume. An ignored SIGHUP stays ignored so `nohup` keeps a batch running.
    stops = [signal.SIGTERM]
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        stops.append(signal.SIGHUP)
    count, sole = _width(schedule["concurrency"], root)
    previous = {stop: signal.signal(stop, _interrupt_batch) for stop in stops}
    try:
        # The batch waits for all its slots rather than running smaller; a capacity
        # experiment above the host's slots runs only on an empty host. Its shards are
        # made for the width the queue grants.
        with _host_slots(_batch_slots(count), f"batch {root}", sole) as held:
            concurrency = count if sole is None else held.count
            plan = _batch_plan(
                sorted(modules),
                root,
                timeout,
                schedule["order"],
                concurrency,
                [*ladder.get("representatives", ()), *schedule["serial_only"]],
            )
            print(
                json.dumps({"timeout": plan.pop("limits")}, sort_keys=True), flush=True
            )
            return _run_batch(
                sorted(modules),
                root,
                store=_build_prerequisites(sorted(modules), root),
                slot_wait_seconds=_slot_wait(held),
                **{**schedule, **plan, "concurrency": concurrency},
                **ladder,
            )
    except KeyboardInterrupt:
        print(
            "testing batch interrupted; the running module has no finish line. "
            "Confirm no module process remains, then rerun the same command to "
            "resume; finished modules are skipped.",
            file=sys.stderr,
        )
        return 130
    finally:
        for stop, handler in previous.items():
            signal.signal(stop, handler)


def _commit(ref: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=_REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise ControlSurfaceError(f"--at {ref} is not a commit")
    return completed.stdout.strip()


def _git_worktree(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "worktree", *arguments],
        cwd=_REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )


def _snapshot_reason(root: Path) -> str:
    """The lock reason a snapshot worktree carries: the evidence the runner made it."""
    return f"bio-neb testing snapshot {root.resolve()}"


def _remove_snapshot(root: Path, checkout: Path) -> None:
    """Remove the snapshot this runner made for `root` at `checkout`, and nothing else.

    A snapshot is a detached worktree locked with `_snapshot_reason(root)`; any other
    directory there is refused, and a failed removal leaves the worktree in place.
    """
    _git_worktree("prune")
    listed = _git_worktree("list", "--porcelain").stdout.split("\n\n")
    owned = any(
        entry.startswith(f"worktree {checkout.resolve()}\n")
        and "\ndetached\n" in entry
        and f"\nlocked {_snapshot_reason(root)}" in entry
        for entry in listed
    )
    if not owned:
        if checkout.exists():
            raise ControlSurfaceError(
                f"{checkout} is not a snapshot worktree of this repository; use a "
                "fresh --root"
            )
        return
    # Removing a locked worktree takes a second --force.
    removed = _git_worktree("remove", "--force", "--force", str(checkout))
    if removed.returncode:
        raise ControlSurfaceError(
            f"cannot remove snapshot {checkout}: {removed.stderr.strip()}"
        )


@contextlib.contextmanager
def _root_lock(root: Path, name: str) -> Iterator[TextIO]:
    """Hold `<root>/<name>` exclusively, or refuse because its holder still runs.

    Yields the open lock file, so a child that must keep the root can inherit it.
    """
    with (root / name).open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ControlSurfaceError(
                f"a batch controller or module process for {root} is still running; "
                "confirm it has stopped before resuming"
            ) from error
        yield lock


def _snapshot_command(
    ref: str, root: Path, argv: Sequence[str], slots: int, sole: int | None = None
) -> int:
    """Run `testing <argv>` from a detached worktree of one commit (ticket 234).

    The snapshot's own runner and sources run, with the current environment, so the
    working checkout can change meanwhile and the batch header names the snapshot
    commit. The worktree is `<root>/checkout`, a checkout rather than a retained
    resource: it is removed when the run ends, and a resume makes it again.

    `<root>/snapshot.lock` is held from the first removal to the last, and the snapshot
    run inherits it, so a second snapshot command on the root is refused before it
    changes the checkout, even while a killed controller's run still goes on. Removal
    also holds the batch lock, so a module that outlived its batch controller keeps
    the checkout until it ends.

    The command holds the run's `slots` host test slots itself, or `sole` of them as the
    host's sole runner: the snapshot's runner inherits them, and a base commit's runner may predate the queue, so the command
    appends its own wait to `<root>/slot-waits.jsonl`.
    """
    commit = _commit(ref)
    _owned_root(root, _REPOSITORY)
    checkout = root / "checkout"
    with (root / "snapshot.lock").open("a") as owner:
        try:
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ControlSurfaceError(
                f"a snapshot run for {root} is still running; confirm it has stopped "
                "before resuming"
            ) from error
        with _root_lock(root, "ledger.lock"):
            _remove_snapshot(root, checkout)
            added = _git_worktree(
                "add",
                "--detach",
                "--lock",
                "--reason",
                _snapshot_reason(root),
                str(checkout),
                commit,
            )
            if added.returncode:
                raise ControlSurfaceError(
                    f"cannot make a snapshot of {commit}: {added.stderr.strip()}"
                )
        try:
            with _host_slots(slots, f"snapshot {commit[:12]} {root}", sole) as held:
                _append_line(
                    root / _SLOT_WAITS_NAME,
                    {
                        "commit": commit,
                        "slots": held.count,
                        "slot_wait_seconds": _slot_wait(held),
                        "granted_at": datetime.now(UTC).isoformat(),
                    },
                )
                return _run_snapshot(commit, root, checkout, argv, owner.fileno())
        finally:
            try:
                with _root_lock(root, "ledger.lock"):
                    _remove_snapshot(root, checkout)
            except ControlSurfaceError as error:
                print(f"testing snapshot left in place: {error}", file=sys.stderr)


def _run_snapshot(
    commit: str, root: Path, checkout: Path, argv: Sequence[str], owner_fd: int
) -> int:
    current, snapshot = _REPOSITORY / "uv.lock", checkout / "uv.lock"
    if (current.read_bytes() if current.is_file() else None) != (
        snapshot.read_bytes() if snapshot.is_file() else None
    ):
        raise ControlSurfaceError(
            f"{commit} has another uv.lock than the environment this runner uses; "
            "run that commit from its own checkout"
        )
    print(
        json.dumps({"snapshot": {"commit": commit, "checkout": str(checkout)}}),
        flush=True,
    )
    environment = controller_environment(
        cache=root / "python-cache", source=checkout / "src"
    )
    # The run has its own session, so a stop reaches it once, forwarded from here;
    # the checkout stays until it has ended.
    stops = [signal.SIGINT, signal.SIGTERM]
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        stops.append(signal.SIGHUP)
    children: list[subprocess.Popen[bytes]] = []
    received: list[int] = []

    def forward(signum: int, frame: Any) -> None:
        received.append(signum)
        for child in children:
            child.send_signal(signum)

    previous = {stop: signal.signal(stop, forward) for stop in stops}
    try:
        child = subprocess.Popen(
            [sys.executable, "-m", "bio_neb.devtools.testing", *argv],
            cwd=checkout,
            env=environment,
            start_new_session=True,
            pass_fds=(owner_fd,),
        )
        children.append(child)
        for signum in received:
            child.send_signal(signum)
        returncode = child.wait()
    finally:
        for stop, handler in previous.items():
            signal.signal(stop, handler)
    return returncode if returncode >= 0 else 128 - returncode


def _without_at(argv: Sequence[str]) -> list[str]:
    forwarded, skip = [], False
    for argument in argv:
        if skip:
            skip = False
        elif argument == "--at":
            skip = True
        elif not argument.startswith("--at="):
            forwarded.append(argument)
    return forwarded


def _module_result(lines: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """A module's exit, red IDs and reasons over its finish lines, one per shard: the
    first nonzero exit, and no red set when any line recorded none."""
    reds = {
        key: None
        if any(line.get(key) is None for line in lines)
        else sorted({test_id for line in lines for test_id in line[key]})
        for key in ("failed", "errored")
    }
    return {
        "exit": next((line.get("exit") for line in lines if line.get("exit") != 0), 0),
        **reds,
        "reasons": None
        if any(line.get("reasons") is None for line in lines)
        else {
            test_id: reason
            for line in lines
            for test_id, reason in line["reasons"].items()
        },
    }


def _base_repro(
    names: Sequence[str], base_sha: str, root: Path | None, timeout: float | None
) -> int:
    """Run modules at the ticket base through a snapshot and report the base result."""
    base = _full_sha(base_sha)
    modules = sorted({module_name(name) for name in names})
    if root is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        root = testing_run_roots() / f"base-repro-{stamp}-{base[:8]}"
    # The base's runner may predate derived limits and takes one --timeout, so the
    # longest of the modules' whole limits is bound here.
    plan = _batch_plan(modules, root, timeout, "sorted", 1, ())
    print(json.dumps({"timeout": plan["limits"]}, sort_keys=True), flush=True)
    limits = plan["module_timeouts"]
    seconds = max(limits.get(module, plan["timeout"]) for module in modules)
    returncode = _snapshot_command(
        base,
        root,
        ["batch", *modules, "--root", str(root), "--timeout", str(seconds)],
        _default_concurrency(),  # the base runner's own default
    )
    ledger = root / _LEDGER_NAME
    if ledger.is_file():
        lines = [
            json.loads(line)
            for line in _complete_ledger_bytes(ledger)[0].decode().splitlines()
        ]
        finishes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for line in lines[1:]:
            if "module" in line:
                finishes[line["module"]].append(line)
        result = {
            module: _module_result(module_lines)
            for module, module_lines in finishes.items()
        }
        print(
            json.dumps(
                {
                    "base_repro": {
                        "commit": lines[0].get("commit"),
                        "ledger": str(ledger),
                        "modules": result,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return returncode


def _phase_state(phase: dict[str, Any]) -> str:
    if phase["exit"] == "timeout":
        return "timeout"
    if phase["failed"] is None:
        return "no_result"
    if phase["exit"] != 0 and not (phase["failed"] or phase["errored"]):
        # A nonzero exit that names no red test (no tests ran, or a crash after the
        # result was written) cannot be read as an empty red set.
        return "no_result"
    return "finished"


def _phases(finish: dict[str, Any]) -> list[dict[str, Any]]:
    if "shard_finishes" in finish:
        return finish["shard_finishes"]
    return [finish, *([finish["expansion"]] if finish.get("expansion") else [])]


def _module_finish(
    header: dict[str, Any], finishes: dict[str, dict[str, Any]], module: str
) -> dict[str, Any] | None:
    """A module's finish: its own line, or its shards' lines joined into one, None
    until every shard has a line."""
    planned = _planned_shards(header).get(module)
    if planned is None:
        return finishes.get(module)
    lines = [
        finishes.get(_Unit(module, shard).label) for shard in range(1, len(planned) + 2)
    ]
    if None in lines:
        return None
    return {"module": module, "shard_finishes": lines}


def _finish_state(finish: dict[str, Any] | None) -> str:
    """A module's comparable state; `representative` means only its listed IDs ran."""
    if finish is None or _awaits_expansion(finish):
        return "unfinished"
    states = {_phase_state(phase) for phase in _phases(finish)}
    for limit in ("timeout", "no_result"):
        if limit in states:
            return limit
    if finish.get("phase") == _REPRESENTATIVE_RUNG and not finish["expansion"]:
        return _REPRESENTATIVE_RUNG
    return "finished"


# Unittest's dotted description (`m (tests.x.C.m)`) and a dotted ID's leading part, which
# drops a subTest suffix such as ` (case=1)` or ` [message]`.
_UNITTEST_DESCRIPTION = re.compile(r"\w+ \(([\w.]+)\)")
# The one subTest form compare keeps apart: a repository scan's offender, so a new
# offender is a new red ID rather than a changed reason on its test.
_OFFENDER_SUBTEST = re.compile(r"([\w.]+) (\(offender=.*\))\Z", re.S)
_DOTTED_TEST_ID = re.compile(r"[\w.]+")


def _canonical_test_id(text: str) -> str:
    """One ID for a test in any form: `tests/x.py::C::m[p]`, `m (tests.x.C.m)` or
    `tests.x.C.m (case=1)` are all `tests.x.C.m`; a subTest or parameter folds into it,
    except an `(offender=...)` subTest, which stays its own ID.
    """
    text = text.strip()
    if "::" in text:
        path, *names = text.split("::")
        names[-1] = names[-1].split("[", 1)[0]
        module = path.removeprefix("./").removesuffix(".py").replace("/", ".")
        return ".".join([module, *names])
    offender = _OFFENDER_SUBTEST.match(text)
    if offender:
        return f"{offender.group(1)} {offender.group(2)}"
    described = _UNITTEST_DESCRIPTION.match(text)
    if described:
        return described.group(1)
    dotted = _DOTTED_TEST_ID.match(text)
    if dotted is None:
        raise ControlSurfaceError(f"not a test ID: {text!r}")
    return dotted.group(0)


_UUID_SHAPE = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"


def _shape(text: str) -> str:
    return "".join("-" if char == "-" else "x" for char in text)


def _uuid_head(match: re.Match[str]) -> str:
    """A UUID's head cut by unittest's `[N chars]`, or the text unchanged."""
    head = match.group(0)
    return "<uuid>" if _UUID_SHAPE.startswith(_shape(head)) else head


def _uuid_tail(match: re.Match[str]) -> str:
    """A UUID's tail resumed after unittest's `[N chars]`, or the text unchanged."""
    tail = match.group(0)
    return "<uuid>" if _UUID_SHAPE.endswith(_shape(tail)) else tail


# Reason parts that vary between runs of the same failure: temporary paths, UUIDs
# (whole, with pydantic's `...` truncation, or a UUID-shaped head or tail directly beside
# unittest's `[N chars]`; kept groups may be all letters), and numbers (integer or decimal)
# or hex runs, including the fragments unittest's truncation leaves.
_REASON_MASKS = (
    (re.compile(r"/tmp/[^\s'\"]*"), "<tmp>"),
    (
        re.compile(
            r"\b[0-9a-f]+(?:-[0-9a-f]+)*-?\.\.\.(?:[0-9a-f]+-)*[0-9a-f]+\b", re.I
        ),
        "<uuid>",
    ),
    (
        re.compile(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
        ),
        "<uuid>",
    ),
    (re.compile(r"\b[0-9a-f-]{9,35}(?=\[\d+ chars\])", re.I), _uuid_head),
    (re.compile(r"(?<= chars\])[0-9a-f-]{13,35}\b", re.I), _uuid_tail),
    (re.compile(r"\b[0-9a-f]*\d[0-9a-f]*(?:\.\d+)?\b", re.I), "<n>"),
)


def _reason_key(reason: str) -> str:
    """A failure reason with its run-to-run noise masked, for comparison."""
    for pattern, mask in _REASON_MASKS:
        reason = pattern.sub(mask, reason)
    return reason


def _red(*phases: dict[str, Any], recorded: bool) -> dict[str, set[str]]:
    """Each red test's canonical ID with its masked reasons, when the side recorded them."""
    red: dict[str, set[str]] = {}
    for phase in phases:
        for test_id in (*phase["failed"], *phase["errored"]):
            keys = red.setdefault(_canonical_test_id(test_id), set())
            if not recorded:
                continue
            if test_id not in (phase["reasons"] or {}):
                raise ControlSurfaceError(f"{test_id} has no recorded reason")
            keys.add(_reason_key(phase["reasons"][test_id]))
    return red


def _reason_changes(
    recorded: dict[str, bool], left: dict[str, set[str]], right: dict[str, set[str]]
) -> list[dict[str, Any]] | None:
    """IDs red on both sides whose masked reasons differ; None when a side has none."""
    if not all(recorded.values()):
        return None
    return [
        {"id": test_id, "left": sorted(left[test_id]), "right": sorted(right[test_id])}
        for test_id in sorted(left.keys() & right.keys())
        if left[test_id] != right[test_id]
    ]


# One side of a comparison: a ledger header, its finishes by module, and how the
# comparison describes the side.
_ComparisonSide = tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]


def _ledger_side(root: Path) -> _ComparisonSide:
    header, finishes = _read_ledger(root / _LEDGER_NAME)[:2]
    return (
        header,
        finishes,
        {
            "ledger": str(root / _LEDGER_NAME),
            "commit": header["commit"],
            "clean_tree": header["clean_tree"],
            "selection": header["selection"],
            "rung": header["rung"],
            "supplemental_only": header.get("supplemental_only", False),
        },
    )


def _id_list_side(path: Path, selection: Sequence[str]) -> _ComparisonSide:
    """A list of red test IDs read as a full batch with no recorded reasons.

    The list counts as the whole red set of `selection` and of its own IDs' modules:
    a module with no listed ID is green. Blank lines and `#` comments are skipped.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ControlSurfaceError(f"cannot read test ID list {path}") from error
    test_ids = sorted(
        {
            _canonical_test_id(line)
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        }
    )
    red: dict[str, list[str]] = {module: [] for module in selection}
    for test_id in test_ids:
        module = next(
            (name for name in selection if test_id.startswith(f"{name}.")),
            test_id.rsplit(".", 2)[0],
        )
        red.setdefault(module, []).append(test_id)
    finishes = {
        module: {
            "module": module,
            "exit": 1 if ids else 0,
            "failed": ids,
            "errored": [],
            "log": None,
        }
        for module, ids in red.items()
    }
    header = {"schema": "test-id-list", "selection": sorted(red)}
    return header, finishes, {"ids": str(path), "count": len(test_ids)}


def _compare_ledgers(left_root: Path, right_root: Path) -> dict[str, Any]:
    """Compare two batches by the failing/erroring IDs of modules both finished."""
    return _compare(_ledger_side(left_root), _ledger_side(right_root))


def _compare_against_ids(root: Path, ids: Path) -> dict[str, Any]:
    """Compare a batch (right) with a list of red test IDs (left)."""
    right = _ledger_side(root)
    return _compare(_id_list_side(ids, right[0]["selection"]), right)


def _compare(left_side: _ComparisonSide, right_side: _ComparisonSide) -> dict[str, Any]:
    """Red IDs added and removed, and reasons changed, over modules both sides finished."""
    sides = {"left": left_side, "right": right_side}
    (left, left_finishes, _), (right, right_finishes, _) = left_side, right_side
    recorded = {
        side: header["schema"] not in (_REASONLESS_LEDGER_SCHEMA, "test-id-list")
        for side, (header, _, _) in sides.items()
    }
    common = [module for module in left["selection"] if module in right["selection"]]
    compared: list[str] = []
    representative_only: list[str] = []
    limits: dict[str, dict[str, str]] = {}
    left_red: dict[str, set[str]] = {}
    right_red: dict[str, set[str]] = {}
    for module in common:
        left_finish, right_finish = (
            _module_finish(left, left_finishes, module),
            _module_finish(right, right_finishes, module),
        )
        states = {
            "left": _finish_state(left_finish),
            "right": _finish_state(right_finish),
        }
        same_representatives = module in left.get("representatives", {}) and left[
            "representatives"
        ][module] == right.get("representatives", {}).get(module)
        if set(states.values()) == {"finished"}:
            compared.append(module)
            left_red.update(_red(*_phases(left_finish), recorded=recorded["left"]))
            right_red.update(_red(*_phases(right_finish), recorded=recorded["right"]))
        elif same_representatives and set(states.values()) <= {
            "finished",
            _REPRESENTATIVE_RUNG,
        }:
            # Both sides ran the same representative IDs; only that phase compares.
            compared.append(module)
            representative_only.append(module)
            left_red.update(_red(left_finish, recorded=recorded["left"]))
            right_red.update(_red(right_finish, recorded=recorded["right"]))
        else:
            limits[module] = states
    return {
        "schema": "bio-neb.testing-batch-comparison/2",
        **{side: description for side, (_, _, description) in sides.items()},
        "compared_modules": compared,
        "representative_only": representative_only,
        "added": sorted(right_red.keys() - left_red.keys()),
        "removed": sorted(left_red.keys() - right_red.keys()),
        "reasons": {
            side: "recorded" if known else "unavailable"
            for side, known in recorded.items()
        },
        "reason_changed": _reason_changes(recorded, left_red, right_red),
        "limits": limits,
        "only_left": [m for m in left["selection"] if m not in right["selection"]],
        "only_right": [m for m in right["selection"] if m not in left["selection"]],
    }


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print the additive selection without executing it",
    )
    parser.add_argument("--receipt", type=Path, help=_receipt_help())
    parser.add_argument("--run-root", type=Path, help=_run_root_help())


def _add_batch_options(parser: argparse.ArgumentParser, *, derived: bool) -> None:
    location = _owned_receipt_location().replace("%", "%%")
    parser.add_argument(
        "--root",
        required=not derived,
        type=Path,
        help=(
            ("run the selection as a batch: " if derived else "")
            + "absolute ledger directory outside the repository; reuse it to resume. "
            f"Retained batches go under {location}"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help=(
            "per-unit (module or shard) wall-clock limit in seconds, recorded as "
            f"`timeout`. Default: {_DERIVED_TIMEOUT_MULTIPLE:g} x each unit's "
            "estimate from the previous ledgers, at least "
            f"{_DERIVED_TIMEOUT_MINIMUM_SECONDS:g} and at most "
            f"{_DEFAULT_MODULE_TIMEOUT_SECONDS:g}, which a module with none keeps"
        ),
    )
    parser.add_argument(
        "--at",
        metavar="COMMIT",
        help=(
            "run from a detached worktree of this commit under --root, so the "
            "checkout can change meanwhile; the ledger names the commit"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help=(
            f"modules run at once (by default {_DEFAULT_CONCURRENCY}, or "
            f"{_SOLE_RUNNER_CONCURRENCY} when the batch is the host's sole runner; at "
            f"most {_CONCURRENCY_CEILING} and the CPU count; a resumed root keeps its "
            "width); on `batch`, --parallel-experiment may go up to the CPU count"
        ),
    )
    parser.add_argument(
        "--rung",
        choices=(_FULL_RUNG, _REPRESENTATIVE_RUNG),
        default=None if derived else _FULL_RUNG,
        help=(
            "full runs whole modules; representative runs each laddered module's "
            "representative IDs plus test methods changed since --base-sha, expands a "
            "red one to the rest of its module and runs other modules whole. A "
            "representative result is supplemental, never acceptance proof"
        ),
    )


def _add_derived_arguments(
    parser: argparse.ArgumentParser, *, expandable: bool
) -> None:
    parser.add_argument(
        "--base-sha",
        required=True,
        help=(
            "exact ticket-base commit; base...HEAD plus the working tree is the "
            "change set"
        ),
    )
    parser.add_argument(
        "--ticket-id",
        required=True,
        type=int,
        help="positive ticket identifier recorded with the selection",
    )
    parser.add_argument(
        "--risk-reason",
        help="nonblank material-risk reason retained for root's checkpoint decision",
    )
    if expandable:
        parser.add_argument(
            "--include",
            action="append",
            default=[],
            metavar="MODULE",
            help="add a test module (tests.test_x or tests/test_x.py); repeatable",
        )
        parser.add_argument(
            "--depth",
            type=int,
            default=0,
            help="also select users of importers up to this many import hops away",
        )
        parser.add_argument(
            "--expansion-reason",
            help="retained nonblank reason for --include and --depth",
        )
    _add_execution_arguments(parser)
    _add_batch_options(parser, derived=True)


def _parser(data: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bio_neb.devtools.testing",
        description=(
            "Serial offline proof. Selection is additive and derived from the "
            "base...HEAD and working-tree changes; "
            "coverage and risk warnings advise root without adding execution phases. "
            "Ticket instructions govern first, then root chooses the mode. "
            "Live acceptance is separate."
        ),
    )
    commands = parser.add_subparsers(dest="mode", required=True)
    core = commands.add_parser(
        "core",
        help="fixture-free invariants plus the tests the change set reaches",
        description=(
            "Start with the fixture-free core, then add every test the changed paths "
            "reach through imports or checked-in intent readers. Unreached paths are "
            "reported."
        ),
    )
    _add_derived_arguments(core, expandable=False)
    scoped = commands.add_parser(
        "scoped",
        help="core plus explicit --include and --depth expansions",
    )
    _add_derived_arguments(scoped, expandable=True)
    fixture = commands.add_parser(
        "scoped-with-current-fixture",
        help="scoped proof plus all current inert fixture consumer modules",
    )
    _add_derived_arguments(fixture, expandable=True)
    regression = commands.add_parser(
        "regression",
        help="run one exact supplemental known-bug regression selection",
        description=(
            "Run one checked-in deterministic exact-test-ID selection. This is "
            "supplemental proof; the ticket and root decide the required verification."
        ),
    )
    regression.add_argument(
        "regression_selection",
        choices=sorted(data["regression_selections"]),
        help="checked-in supplemental regression selection",
    )
    _add_execution_arguments(regression)
    checkpoint = commands.add_parser(
        "checkpoint",
        help=(
            "all currently discovered tests once plus offline "
            "static and repository gates"
        ),
    )
    checkpoint.add_argument("--base-sha", required=True)
    _add_execution_arguments(checkpoint)
    checkpoint.add_argument(
        "--root",
        type=Path,
        help=(
            "required to execute: the suite runs as the whole-suite batch in this "
            "absolute ledger directory, the baseline for later `compare` runs"
        ),
    )
    checkpoint.add_argument(
        "--concurrency",
        type=int,
        help=(
            f"modules the suite batch runs at once (default and ceiling "
            f"{_DEFAULT_CONCURRENCY})"
        ),
    )
    lint = commands.add_parser(
        "lint",
        help="changed-file lint and format only; the fast inner-iteration loop",
        description=(
            "Run the changed-file quality gate without the test suite. The change set is "
            "base...HEAD plus the working tree, so uncommitted work is included. Proves "
            "lint and format only; checkpoint remains the acceptance gate."
        ),
    )
    lint.add_argument(
        "--base-sha",
        required=True,
        help="exact ticket-base commit; the runner derives every changed Python file",
    )
    lint.add_argument(
        "--fix",
        action="store_true",
        help=(
            "repair the same changed files: `ruff check --fix`, then `ruff format`; "
            "never the whole repository"
        ),
    )
    lint.add_argument("--receipt", type=Path, help=_receipt_help("optional "))
    lint.add_argument("--run-root", type=Path, help=_run_root_help("optional "))
    batch = commands.add_parser(
        "batch",
        help="run modules one per process with a resumable ledger and timeout",
        description=(
            "Run each selected test module in its own process, recording one ledger "
            "under --root: a header binding commit, clean tree, lock, interpreter, "
            "selection and rung, then one finish line per module. Rerunning the same "
            "command on the same root resumes: finished modules (including reds and "
            "timeouts) are skipped and unfinished modules run again. "
            f"{_DEFAULT_CONCURRENCY} modules run at once, or "
            f"{_SOLE_RUNNER_CONCURRENCY} when the batch is the host's sole runner, unless "
            "--concurrency says otherwise."
        ),
    )
    batch.add_argument(
        "modules",
        nargs="*",
        help="test modules as tests.test_x or tests/test_x.py",
    )
    batch.add_argument(
        "--all", action="store_true", help="every discovered test module"
    )
    _add_batch_options(batch, derived=False)
    batch.add_argument(
        "--base-sha",
        help="exact ticket-base commit for --rung representative's changed methods",
    )
    batch.add_argument(
        "--parallel-experiment",
        action="store_true",
        help=(
            "capacity-evidence root: allows --concurrency above the ceiling, up to "
            "the CPU count, and marks the batch supplemental_only"
        ),
    )
    batch.add_argument(
        "--order",
        choices=_ORDERS,
        default=_LONGEST,
        help=(
            "unit start order (default: largest first by each module's newest "
            "recorded seconds); shuffle also shuffles each module's test methods, "
            "and longest+shuffle shuffles them while starting the largest first"
        ),
    )
    batch.add_argument(
        "--seed", type=int, help="seed for --order shuffle or longest+shuffle"
    )
    batch.add_argument(
        "--detect-writes",
        action="store_true",
        help="record each module's writes outside its private roots",
    )
    base_repro = commands.add_parser(
        "base-repro",
        help="run test modules at the ticket base through a snapshot",
        description=(
            "Run the named modules as a batch from a detached worktree of --base-sha, "
            "then report each module's base result with its failure reasons."
        ),
    )
    base_repro.add_argument(
        "modules", nargs="+", help="test modules as tests.test_x or tests/test_x.py"
    )
    base_repro.add_argument(
        "--base-sha", required=True, help="exact ticket-base commit to run"
    )
    base_repro.add_argument(
        "--root",
        type=Path,
        help=(
            "absolute batch root (default: a fresh base-repro root under "
            f"{_owned_receipt_location().replace('%', '%%')})"
        ),
    )
    base_repro.add_argument(
        "--timeout", type=float, help="per-module limit in seconds, as for `batch`"
    )
    compare = commands.add_parser(
        "compare",
        help=(
            "added/removed red test IDs and changed failure reasons between two batch "
            "ledgers, or between a ledger and a test ID list"
        ),
    )
    compare.add_argument(
        "roots",
        nargs="+",
        type=Path,
        metavar="root",
        help=(
            "earlier or reference batch root, then the later or candidate one; only "
            "the candidate with --against-ids"
        ),
    )
    compare.add_argument(
        "--against-ids",
        type=Path,
        help="reference file of red test IDs, one per line, in either ID form",
    )
    focused = commands.add_parser(
        "focused",
        help="direct unittest of named tests, one process per module",
        description=(
            "Run `python -m unittest <test>...` on named tests from this checkout: one "
            "module under one slot of the host test queue, several modules one process "
            "each at the batch width. No receipt; the exit status is unittest's, or "
            "the first module's nonzero one. The whole suite runs as `batch --all`."
        ),
    )
    focused.add_argument(
        "tests", nargs="*", help="exact test IDs, classes or a module, as for unittest"
    )
    focused.add_argument(
        "--all",
        action="store_true",
        help="refused: the whole suite runs as `batch --all`",
    )
    flag = commands.add_parser(
        "review-flag",
        help="claim, show or release the host-wide review flag (eight tickets in review)",
        description=(
            "A ticket worker claims the flag before asking for its reviewers and releases "
            "it when its review is final. A taken flag exits "
            f"{review_flag.TAKEN_EXIT} and names its holder to ping; only the "
            "orchestrator or the owner releases a stale one, with --stale <reason>."
        ),
    )
    actions = flag.add_subparsers(dest="action", required=True)
    flag_claim = actions.add_parser("claim", help="claim the flag for a ticket")
    flag_claim.add_argument("--ticket", type=int, required=True)
    flag_claim.add_argument("--map", type=int, required=True, help="the ticket's map")
    flag_claim.add_argument(
        "--worker", required=True, help="the ticket worker's dispatch id, to ping"
    )
    flag_claim.add_argument(
        "--orchestrator", required=True, help="its orchestrator's dispatch id or name"
    )
    flag_release = actions.add_parser("release", help="release a ticket's flag")
    flag_release.add_argument("--ticket", type=int, required=True)
    flag_release.add_argument(
        "--stale",
        metavar="REASON",
        help="release another holder's claim after an unanswered ping",
    )
    actions.add_parser("show", help="the current holders")
    measure = commands.add_parser(
        "measure",
        help="diagnostic per-module cold/warm timing; never checkpoint acceptance",
    )
    measure.add_argument("--receipt", required=True, type=Path, help=_receipt_help())
    measure.add_argument("--run-root", required=True, type=Path, help=_run_root_help())
    return parser


def _declared_representatives(data: dict[str, Any]) -> dict[str, list[str]]:
    return {
        module: selection["test_ids"]
        for module, selection in data["representative_selections"].items()
    }


def _schedule(
    args: argparse.Namespace, data: dict[str, Any], *, experimental: bool
) -> dict[str, Any]:
    """`_run_batch`'s scheduling keywords; derived modes have no experiment options."""
    experiment = experimental and args.parallel_experiment
    if experiment and not args.detect_writes:
        raise ControlSurfaceError("--parallel-experiment needs --detect-writes")
    return {
        "concurrency": _concurrency(args.concurrency, experiment),
        "order": _batch_order(args.order, args.seed) if experimental else _LONGEST,
        "detect_writes": experimental and args.detect_writes,
        "parallel_experiment": experiment,
        "serial_only": data["serial_only"],
    }


def _run_derived_batch(
    selection: Selection, args: argparse.Namespace, data: dict[str, Any]
) -> int:
    """Run a derived selection as a batch; the rung's base is the selection's base."""
    if args.receipt is not None or args.run_root is not None:
        raise ControlSurfaceError(
            "--root runs a batch; --receipt and --run-root run a closure receipt"
        )
    rung = args.rung or _FULL_RUNG
    return _batch_command(
        [module_name(path) for path in selection.test_paths],
        False,
        args.root,
        args.timeout,
        rung,
        selection.base_sha if rung == _REPRESENTATIVE_RUNG else None,
        _declared_representatives(data),
        _schedule(args, data, experimental=False),
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        data = _load_ownership()
        graph = _import_graph()
        _validate_intent(data, graph)
        arguments = list(sys.argv[1:] if argv is None else argv)
        args = _parser(data).parse_args(arguments)
        if getattr(args, "at", None) is not None:
            if args.root is None:
                raise ControlSurfaceError("--at needs --root")
            schedule = _schedule(args, data, experimental=args.mode == "batch")
            count, sole = _width(schedule["concurrency"], args.root)
            return _snapshot_command(
                args.at, args.root, _without_at(arguments), _batch_slots(count), sole
            )
        if args.mode == "base-repro":
            return _base_repro(args.modules, args.base_sha, args.root, args.timeout)
        if args.mode == "measure":
            return _measure(data, args.receipt, args.run_root)
        if args.mode == "focused":
            return _focused(args.tests, args.all)
        if args.mode == "review-flag":
            return _review_flag(args)
        if args.mode == "lint":
            return _lint(args.base_sha, args.receipt, args.run_root, fix=args.fix)
        if args.mode == "batch":
            return _batch_command(
                args.modules,
                args.all,
                args.root,
                args.timeout,
                args.rung,
                args.base_sha,
                _declared_representatives(data),
                _schedule(args, data, experimental=True),
            )
        if args.mode == "compare":
            if len(args.roots) != (1 if args.against_ids else 2):
                raise ControlSurfaceError(
                    "compare takes two batch roots, or one with --against-ids"
                )
            comparison = (
                _compare_against_ids(args.roots[0], args.against_ids)
                if args.against_ids
                else _compare_ledgers(*args.roots)
            )
            print(json.dumps(comparison, indent=2, sort_keys=True))
            return 0
        derived = args.mode in _DERIVED_MODES
        if (
            derived
            and args.root is None
            and (args.rung or args.timeout is not None or args.concurrency is not None)
        ):
            raise ControlSurfaceError("--rung, --timeout and --concurrency need --root")
        selection = _selection(args, data, graph)
        if args.explain:
            explained = selection.as_json()
            if args.mode == "checkpoint":
                explained["commands"] = [
                    {"phase": phase, "argv": argv}
                    for phase, argv in _checkpoint_commands(
                        selection, args.root, _checkpoint_concurrency(args.concurrency)
                    )
                ]
            print(json.dumps(explained, indent=2, sort_keys=True))
            return 0
        if derived and args.root is not None:
            return _run_derived_batch(selection, args, data)
        if args.receipt is None or args.run_root is None:
            raise ControlSurfaceError(
                "--run-root and --receipt are required unless --explain or --root "
                "is used"
            )
        if args.mode == "checkpoint":
            return _execute_selection(
                selection,
                args.receipt,
                args.run_root,
                args.root,
                _checkpoint_concurrency(args.concurrency),
            )
        return _execute_selection(selection, args.receipt, args.run_root)
    except ControlSurfaceError as error:
        print(f"testing control refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
