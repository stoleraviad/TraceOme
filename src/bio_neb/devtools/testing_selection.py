"""Change-impact test selection derived from source and test imports.

A test module's surface is the module plus the `tests/` support it imports, transitively. A
test uses a file when anything on its surface imports that file directly; `python -m <module>`
and `import_module("<module>")` name a module as a string, which counts as an import. A changed
path selects the tests that use it. A source file no test imports selects the users of its
nearest importers instead. Paths imports cannot show (documents, data, fixtures, modules only a
subprocess reaches) name their reader modules in the checked-in intent, and a changed path
selects the tests that use its readers.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import cached_property
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

SOURCE_ROOTS = ("src", "tests", "tools")
CHECKPOINT = "checkpoint"
_WIDER = (("depth_1", 1), ("depth_2", 2), ("transitive", None))


def module_name(path: str) -> str:
    """The dotted module a repository-relative Python path imports as."""
    name = path.removesuffix(".py").removesuffix("/__init__")
    return name.removeprefix("src/").replace("/", ".")


def is_test_module(path: str) -> bool:
    """Whether discovery loads the path: `tests/test_*.py`, not nested."""
    return path.startswith("tests/test_") and "/" not in path[len("tests/") :]


def _imported_names(path: str, tree: ast.AST) -> Iterator[tuple[str, bool]]:
    """Each imported dotted name, flagged when only a string names it."""
    package = module_name(path).split(".")
    if not path.endswith("/__init__.py"):
        package = package[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from ((alias.name, False) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package[: len(package) - node.level + 1]
                base = ".".join([*anchor, *([base] if base else [])])
            yield base, False
            yield from ((f"{base}.{alias.name}", False) for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value, True


@dataclass(frozen=True)
class ImportGraph:
    """Each Python source under `src/`, `tests/` and `tools/` and the sources it imports."""

    imports: Mapping[str, frozenset[str]]

    @classmethod
    def build(cls, repository: Path, files: Iterable[str] | None = None) -> ImportGraph:
        """Parse the sources among `files` (default: every `*.py` under the roots).

        Raises `OSError` or `SyntaxError` for an unreadable source.
        """
        if files is None:
            files = (
                path.relative_to(repository).as_posix()
                for root in SOURCE_ROOTS
                for path in (repository / root).rglob("*.py")
                if "__pycache__" not in path.parts
            )
        paths = sorted(
            path
            for path in files
            if path.endswith(".py")
            and path.split("/", 1)[0] in SOURCE_ROOTS
            and (repository / path).is_file()
        )
        modules = {module_name(path): path for path in paths}
        imports: dict[str, frozenset[str]] = {}
        for path in paths:
            tree = ast.parse(
                (repository / path).read_text(encoding="utf-8"), filename=path
            )
            found: set[str] = set()
            for name, by_string in _imported_names(path, tree):
                # A string naming a test module is a runner argument, not an import.
                if by_string and (name not in modules or is_test_module(modules[name])):
                    continue
                parts = name.split(".")
                # Importing a module first executes every enclosing package.
                found.update(
                    modules[prefix]
                    for prefix in (
                        ".".join(parts[:end]) for end in range(1, len(parts) + 1)
                    )
                    if prefix in modules
                )
                if by_string and f"{name}.__main__" in modules:
                    found.add(modules[f"{name}.__main__"])
            imports[path] = frozenset(found - {path})
        return cls(imports)

    @property
    def tests(self) -> frozenset[str]:
        return frozenset(path for path in self.imports if is_test_module(path))

    @cached_property
    def users(self) -> _Users:
        return _Users(self)


@dataclass(frozen=True)
class Derivation:
    """Selected test modules and why, plus the wider sets reported as unselected risk."""

    test_paths: tuple[str, ...]
    reasons: Mapping[str, tuple[str, ...]]
    derived_by: Mapping[str, int]
    changed_path_users: Mapping[str, int]
    fallback_reasons: tuple[str, ...]
    wider: Mapping[str, tuple[str, ...]]


class _Users:
    """Which test modules use each source, directly or through their support surface."""

    def __init__(self, graph: ImportGraph) -> None:
        self.importers: dict[str, set[str]] = defaultdict(set)
        for path, imported in graph.imports.items():
            if not path.startswith("tests/"):
                for dependency in imported:
                    self.importers[dependency].add(path)
        self.direct: dict[str, set[str]] = defaultdict(set)
        self.reach: dict[str, set[str]] = defaultdict(set)
        for test in graph.tests:
            surface, pending = {test}, [test]
            while pending:
                for dependency in graph.imports[pending.pop()]:
                    if dependency.startswith("tests/") and dependency not in surface:
                        surface.add(dependency)
                        pending.append(dependency)
            for path in surface:
                self.reach[path].add(test)
                for dependency in graph.imports[path]:
                    if not dependency.startswith("tests/"):
                        self.direct[dependency].add(test)

    def of(self, path: str) -> tuple[set[str], bool]:
        """The tests using `path`, and whether they use it only through importers.

        A test module uses itself; a source no test uses directly falls back to the users
        of its nearest importers.
        """
        if path.startswith("tests/"):
            return set(self.reach.get(path, ())), False
        if self.direct.get(path):
            return set(self.direct[path]), False
        for level in self.levels(path, None):
            found = set().union(*(self.direct.get(item, ()) for item in level))
            if found:
                return found, True
        return set(), False

    def levels(self, path: str, limit: int | None) -> Iterator[set[str]]:
        """Non-test importers of `path`, one reverse import hop per level."""
        seen, level, depth = {path}, {path}, 0
        while level and (limit is None or depth < limit):
            level = {
                importer
                for item in level
                for importer in self.importers.get(item, ())
                if importer not in seen
            }
            seen |= level
            depth += 1
            if level:
                yield level


def derive(
    graph: ImportGraph,
    changed: Sequence[str],
    *,
    readers: Mapping[str, Sequence[str] | str] | None = None,
    required: Iterable[tuple[str, str]] = (),
    depth: int = 0,
) -> Derivation:
    """Select the tests a change set affects.

    `readers` maps an fnmatch pattern (`*` also matches `/`) to the Python modules that read
    the matching files, or to `checkpoint`. `required` pairs test modules selected regardless
    of the change, such as the core set, with their reason. `depth` also selects the users of
    importers up to that many reverse import hops from each changed source.
    """
    users = graph.users
    reasons: dict[str, list[str]] = defaultdict(list)
    kinds: dict[str, set[str]] = defaultdict(set)

    def select(tests: set[str], reason: str, kind: str) -> None:
        for test in tests:
            if reason not in reasons[test]:
                reasons[test].append(reason)
            kinds[kind].add(test)

    for test, reason in required:
        select({test}, reason, reason)
    fallback: list[str] = []
    changed_users: dict[str, int] = {}
    sources: set[str] = set()
    for path in changed:
        matched = [
            value
            for pattern, value in (readers or {}).items()
            if fnmatchcase(path, pattern)
        ]
        targets = [path] if path in graph.imports else []
        targets += sorted(
            {reader for value in matched if value != CHECKPOINT for reader in value}
        )
        found: set[str] = set()
        for target in targets:
            tests, nearest = users.of(target)
            if is_test_module(target):
                reason, kind = (
                    ("changed", "changed")
                    if target == path
                    else (f"reads {path}", "intent")
                )
            else:
                reason = f"uses {'an importer of ' if nearest else ''}{target}"
                kind = "nearest" if nearest else "direct"
                if target != path:
                    reason, kind = f"{reason}, which reads {path}", "intent"
                if not target.startswith("tests/"):
                    sources.add(target)
            select(tests, reason, kind)
            found |= tests
        changed_users[path] = len(found)
        if CHECKPOINT in matched:
            fallback.append(f"checkpoint-only changed path: {path}")
        elif not found:
            fallback.append(f"unowned changed path: {path}")

    def within(limit: int | None) -> dict[str, set[str]]:
        reached: dict[str, set[str]] = defaultdict(set)
        for source in sorted(sources):
            for distance, level in enumerate(users.levels(source, limit), start=1):
                for importer in level:
                    for test in users.direct.get(importer, ()):
                        reached[test].add(
                            f"uses a depth-{distance} importer of {source}"
                        )
        return reached

    for test, found_reasons in within(depth).items() if depth else ():
        for reason in sorted(found_reasons):
            select({test}, reason, "depth")
    selected = frozenset(reasons)
    wider = {
        label: tuple(sorted(selected | set(within(limit)))) for label, limit in _WIDER
    }
    return Derivation(
        test_paths=tuple(sorted(selected)),
        reasons={test: tuple(reasons[test]) for test in sorted(selected)},
        derived_by={kind: len(tests) for kind, tests in sorted(kinds.items())},
        changed_path_users=changed_users,
        fallback_reasons=tuple(fallback),
        wider=wider,
    )
