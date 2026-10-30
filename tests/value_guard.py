"""The value guard (#375 D3; ADR 0010's value scan, ADR 0049's word-list guard).

Two scans over `src/bio_neb`, each held to the reviewed allowlist `tests/value_guard.json`:

- **Values.** String values (code strings, not docstrings, and every string of an instruction
  `.toml`) are searched for one-case values: rsIDs, assembly and sequence accessions, sample
  names, ticket tokens and digests. Each (file, value) needs a row.
- **Rules.** In ADR 0049's `applies_to`, every regular expression and fixed string set or mapping
  that a module or class names, and every regex written inline, needs a row with its class.

A row whose value or rule is gone is stale and fails too, so the list only shrinks with the code.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PACKAGE = Path("src") / "bio_neb"

# A one-case value: the kinds ADR 0010 names.
VALUE_KINDS = {
    "rsid": re.compile(r"\b[Rr][Ss][0-9]{3,}\b"),
    "assembly": re.compile(r"\bGC[AF]_[0-9]{9}\.[0-9]+\b"),
    "sequence": re.compile(
        r"\b(?:(?:NC|NG|NT|NW|NZ|NM|NR|NP|XM|XR|XP)_[A-Z]{0,4}[0-9]{6,9}"
        r"|[A-Z]{1,2}[0-9]{5,6}|[A-Z]{4}[0-9]{8,})\.[0-9]+\b"
    ),
    "contig": re.compile(r"(?<![\w.])[Cc]hr(?:[0-9]+|[XYMWZ]|MT|Un|[IVX]{1,5})\b"),
    "sample": re.compile(r"\b(?:HG|NA)[0-9]{5}\b|\bHG[0-9]{3}\b"),
    "ticket": re.compile(
        r"(?i)\b(?:ticket|issue)[ _-]?#?[0-9]+\b|\bt[0-9]{3}\b|(?<![\w&])#[0-9]{2,4}\b"
    ),
    "digest": re.compile(r"(?<![0-9a-f])(?:[0-9a-f]{64}|[0-9a-f]{40})(?![0-9a-f])"),
}


# ADR 0049's applies_to, under src/bio_neb: where a rule may judge a scientific value.
RULE_SCOPE = (
    "investigation/",
    "reference_discovery/",
    "target_request/",
    "paper_intake/",
    "paper_reading/",
    "providers/",
    "prebuilt_index/",
    "parabricks/",
    "agentic_journey/",
    "reporting/",
    "identifier_rule.py",
)
_REGEX_CALLS = frozenset(
    {
        "compile",
        "match",
        "search",
        "fullmatch",
        "findall",
        "finditer",
        "sub",
        "subn",
        "split",
    }
)
_NOT_RULES = frozenset({"__all__", "__slots__", "__match_args__"})


# Each class and the field its row must carry (None: the class word is the whole reason).
VALUE_CLASSES = {
    "example": "partner",  # a teaching example; its file also shows this non-default case
    "pinned": "reason",  # pinned on purpose (an image digest, a policy value)
    "debt": "owner",  # a one-case value the owning ticket removes
}
RULE_CLASSES = {
    "vocabulary": None,  # our own words: event kinds, codes, statuses, roles, field names
    "safety": None,  # injection, URL, path or command safety
    "membership": None,  # reads the researcher's or a source's words for a membership check
    "minted_shape": None,  # the shape of an id or digest this system mints
    "format": "note",  # the syntax a named standard or a consuming tool defines
    "decision": "cite",  # a recorded decision or ADR that limits the scope
    "debt": "owner",  # a rule ADR 0049 retires; the owning ticket deletes it and its row
}
ALLOWLIST = Path(__file__).with_name("value_guard.json")


@dataclass(frozen=True)
class Allowlist:
    """Reviewed rows by file: `values` {path: {value: row}}, `rules` {path: {key: row}}.

    A row is its class word, or an object with `class` and the class's field."""

    values: dict[str, dict[str, object]]
    rules: dict[str, dict[str, object]]

    @classmethod
    def empty(cls) -> Allowlist:
        return cls({}, {})

    @classmethod
    def from_rows(cls, rows: dict) -> Allowlist:
        return cls(dict(rows.get("values", {})), dict(rows.get("rules", {})))

    @classmethod
    def load(cls, path: Path = ALLOWLIST) -> Allowlist:
        with path.open(encoding="utf-8") as handle:
            return cls.from_rows(json.load(handle))


def offenders(root: Path, allowlist: Allowlist) -> list[str]:
    """Each value or rule under `root` the allowlist does not admit, and each stale row."""
    package = root / PACKAGE
    found_values = sorted(set(_values(package)))
    values: dict[str, set[str]] = {}
    for path, _, value in found_values:
        values.setdefault(path, set()).add(value)
    rules: dict[str, set[str]] = {}
    for path, key in _rules(package):
        rules.setdefault(path, set()).add(key)
    found = [
        *(
            f"{path}: {kind} {value} is not allowlisted"
            for path, kind, value in found_values
            if value not in allowlist.values.get(path, {})
        ),
        *(
            f"{path}: rule {key} is not allowlisted"
            for path, keys in sorted(rules.items())
            for key in sorted(keys)
            if key not in allowlist.rules.get(path, {})
        ),
    ]
    for side, rows_by_path, present, classes in (
        ("value", allowlist.values, values, VALUE_CLASSES),
        ("rule", allowlist.rules, rules, RULE_CLASSES),
    ):
        for path, rows in sorted(rows_by_path.items()):
            for key, row in sorted(rows.items()):
                found.extend(
                    _row_problems(package, path, side, key, row, present, classes)
                )
    return found


def _row_problems(
    package: Path,
    path: str,
    side: str,
    key: str,
    row: object,
    present: dict[str, set[str]],
    classes: dict[str, str | None],
) -> Iterator[str]:
    fields = row if isinstance(row, dict) else {"class": row}
    kind = fields.get("class")
    if key not in present.get(path, set()):
        yield f"{path}: row {side} {key} is stale: the file has no such {side}"
    elif kind not in classes:
        yield f"{path}: row {side} {key} has an unknown class {kind!r}"
    elif (field := classes[kind]) and not fields.get(field):
        yield f"{path}: row {side} {key} ({kind}) needs {field!r}"
    elif kind == "example":
        partner = str(fields["partner"])
        if not any(partner in text for text in _scanned_strings(package / path)):
            yield f"{path}: example {key} has no partner {partner!r} in its file"


def _rules(package: Path) -> list[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package).as_posix()
        if relative.startswith(RULE_SCOPE):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            found.update((relative, key) for key in _rule_keys(tree))
    return sorted(found)


def _rule_keys(tree: ast.Module) -> Iterator[str]:
    """A named rule is keyed by its (Class.)NAME; an inline regex by where it is and its text."""
    named: set[int] = set()
    used_as_pattern = {
        node.id
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        for node in _pattern_arguments(call)
        if isinstance(node, ast.Name)
    }
    for owner, statement in _definitions(tree):
        target = _single_target(statement)
        value = statement.value
        if target is None or value is None or target in _NOT_RULES:
            continue
        if (
            (isinstance(value, ast.Call) and _is_regex_call(value))
            or _is_string_collection(value)
            or _is_computed_table(value)
            or (
                (
                    isinstance(value, ast.JoinedStr)
                    or (
                        isinstance(value, ast.Constant) and isinstance(value.value, str)
                    )
                )
                and (target.endswith("PATTERN") or target in used_as_pattern)
            )
        ):
            named.add(id(value))
            yield f"{owner}{target}"
    yield from _inline_patterns(tree, (), named)


def _definitions(tree: ast.Module) -> Iterator[tuple[str, ast.Assign | ast.AnnAssign]]:
    for statement in tree.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            yield "", statement
        elif isinstance(statement, ast.ClassDef):
            for member in statement.body:
                if isinstance(member, (ast.Assign, ast.AnnAssign)):
                    yield f"{statement.name}.", member


def _single_target(statement: ast.Assign | ast.AnnAssign) -> str | None:
    targets = (
        statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    )
    if len(targets) == 1 and isinstance(targets[0], ast.Name):
        return targets[0].id
    return None


def _inline_patterns(
    node: ast.AST, where: tuple[str, ...], named: set[int]
) -> Iterator[str]:
    for child in ast.iter_child_nodes(node):
        inner = where
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            inner = (*where, child.name)
        elif isinstance(child, (ast.Assign, ast.AnnAssign)) and (
            where and isinstance(node, ast.ClassDef)
        ):
            target = _single_target(child)
            inner = (*where, target) if target else where
        if isinstance(child, ast.Call) and id(child) not in named:
            for pattern in _literal_patterns(child):
                yield f"{'.'.join(where) or '<module>'} /{pattern}/"
        yield from _inline_patterns(child, inner, named)


def _pattern_arguments(call: ast.Call) -> list[ast.expr]:
    """A regex call's pattern, or any call's `pattern=` (pydantic `Field`)."""
    arguments = [keyword.value for keyword in call.keywords if keyword.arg == "pattern"]
    if _is_regex_call(call) and call.args:
        arguments.append(call.args[0])
    return arguments


def _literal_patterns(call: ast.Call) -> Iterator[str]:
    for candidate in _pattern_arguments(call):
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
            yield candidate.value
        elif isinstance(candidate, ast.JoinedStr):
            yield ast.unparse(candidate)


def _is_regex_call(call: ast.Call) -> bool:
    function = call.func
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "re"
        and function.attr in _REGEX_CALLS
    )


def _is_computed_table(value: ast.expr) -> bool:
    """A fixed set or mapping a comprehension builds (the codon table from two strings)."""
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"frozenset", "set", "tuple", "list", "dict"}
        and len(value.args) == 1
    ):
        value = value.args[0]
    return isinstance(
        value, (ast.DictComp, ast.SetComp, ast.ListComp, ast.GeneratorExp)
    )


def _is_string_collection(value: ast.expr) -> bool:
    """A fixed set, tuple, list or mapping of literal strings, one or more."""
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"frozenset", "set", "tuple", "list", "dict"}
        and len(value.args) == 1
    ):
        value = value.args[0]
    members = (
        value.keys if isinstance(value, ast.Dict) else getattr(value, "elts", None)
    )
    if (
        not isinstance(value, (ast.Set, ast.Tuple, ast.List, ast.Dict))
        or members is None
    ):
        return False
    return len(members) >= 1 and all(
        isinstance(member, ast.Constant) and isinstance(member.value, str)
        for member in members
    )


def _values(package: Path) -> Iterator[tuple[str, str, str]]:
    for path in sorted([*package.rglob("*.py"), *package.rglob("*.toml")]):
        relative = path.relative_to(package).as_posix()
        for text in _scanned_strings(path):
            yield from _matches(relative, text)


def _scanned_strings(path: Path) -> Iterator[str]:
    """A file's code strings (not docstrings), or its instruction-TOML strings (not comments)."""
    if path.suffix == ".toml":
        with path.open("rb") as handle:
            yield from _toml_strings(tomllib.load(handle))
    else:
        yield from _code_strings(ast.parse(path.read_text(encoding="utf-8")))


def _matches(relative: str, text: str) -> Iterator[tuple[str, str, str]]:
    for kind, pattern in VALUE_KINDS.items():
        for match in pattern.finditer(text):
            yield relative, kind, match.group(0)


def _code_strings(tree: ast.Module) -> Iterator[str]:
    """Every string constant except docstrings: a bare string statement documents."""
    documenting = {
        id(statement.value)
        for node in ast.walk(tree)
        if isinstance(getattr(node, "body", None), list)
        for statement in node.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in documenting
        ):
            yield node.value


def _toml_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _toml_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _toml_strings(item)
