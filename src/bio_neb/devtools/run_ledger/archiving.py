"""``archive``: move owner-approved run roots into an ``.archive`` batch (#228).

Owner decision 2026-10-11: obsolete local run roots move, on the owner's approval of the list,
to ``<archive_root>/<batch>/run-roots/`` with a manifest. Every row stays; a moved root's row
takes the ``archive:<batch>`` location key and notes where it went, and a root that had no row
gets one. Nothing is deleted, and a root whose row feeds a fixture is refused.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ledger import (
    ARCHIVED_RUN_ROOTS,
    LedgerRow,
    archive_key,
    derive_run_id,
    directory_inventory,
    name_tokens,
    read_ledger,
    write_ledger,
)

_TICKET = re.compile(r"^(?:issue|i|t)?(\d+)")


class ArchiveError(ValueError):
    """The move cannot be done as asked; no root moved without its record."""


@dataclass(frozen=True, slots=True)
class ArchivedRoot:
    identity: str
    run_ids: tuple[str, ...]
    original: str
    archived: str
    size_bytes: str
    inventory_sha256: str
    ledger: str

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ArchivedRoot:
        return cls(
            record["identity"],
            tuple(record["run_ids"]),
            record["original"],
            record["archived"],
            record["size_bytes"],
            record["inventory_sha256"],
            record["ledger"],
        )

    def as_record(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "run_ids": list(self.run_ids),
            "original": self.original,
            "archived": self.archived,
            "size_bytes": self.size_bytes,
            "inventory_sha256": self.inventory_sha256,
            "ledger": self.ledger,
        }


def read_approved(path: Path) -> tuple[str, ...]:
    names = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(set(names)) != len(names):
        raise ArchiveError("the approved list names a run root twice")
    for name in names:
        if "/" in name or name in (".", ".."):
            raise ArchiveError(f"approved entries are run root names, not {name!r}")
    return names


def _inventory(root: Path) -> tuple[str, str, str]:
    try:
        size, digest = directory_inventory(root)
    except PermissionError:
        return "", "", "unreadable at archive (mode kept)"
    return str(size), digest, ""


def _new_row(name: str, key: str, ticket: str, date: str, notes: str) -> LedgerRow:
    """A row for a root nobody ledgered: ticket and commit from its name."""

    match = _TICKET.match(name)
    _, sha = name_tokens(name)
    return LedgerRow(
        run_id=derive_run_id(name),
        ticket=match.group(1) if match else ticket,
        date=date,
        source_commit=sha,
        kind="run_root",
        identity=name,
        location_key=key,
        notes=notes,
    )


def archive_run_roots(
    ledger_path: Path,
    run_roots: Path,
    archive_root: Path,
    *,
    batch: str,
    approved: tuple[str, ...],
    ticket: str,
    authority: str,
    dry_run: bool = False,
) -> tuple[ArchivedRoot, ...]:
    """Validate the whole list; for each root, record where it goes, then move it.

    A root is never moved before its rows and manifest entry are written, and a failed move
    rolls that record back. A run interrupted between the two leaves the record; rerunning
    the same list completes the move.
    """

    key = archive_key(batch)
    rows = list(read_ledger(ledger_path))
    batch_root = archive_root / batch
    destination = batch_root / ARCHIVED_RUN_ROOTS
    shown_root = Path(archive_root.name) / batch / ARCHIVED_RUN_ROOTS
    document = _read_manifest(batch_root)
    live = _run_root_rows(rows, "run_roots")
    recorded = _run_root_rows(rows, key)

    plan: list[tuple[str, Path, Path, bool]] = []
    for name in approved:
        source, target = run_roots / name, destination / name
        if name in recorded:
            if name not in document["run_roots"]:
                raise ArchiveError(f"{name} has {key} rows but no manifest entry")
            if target.is_dir() and not source.exists():
                continue
            if not (source.is_dir() and not target.exists()):
                raise ArchiveError(f"{name} is recorded in {key} but cannot be moved")
            plan.append((name, source, target, True))
            continue
        if any(rows[index].fixture_use for index in live.get(name, ())):
            raise ArchiveError(f"{name} feeds a fixture and is never archived")
        if target.exists() or target.is_symlink():
            raise ArchiveError(f"{target} already exists")
        if source.is_symlink() or not source.is_dir():
            raise ArchiveError(f"{source} is not a run root directory")
        plan.append((name, source, target, False))
    if dry_run:
        return tuple(
            ArchivedRoot(
                name,
                (),
                f"run_roots/{name}",
                str(shown_root / name),
                "",
                "",
                "existing" if name in live else "added",
            )
            for name, _, _, _ in plan
        )

    destination.mkdir(parents=True, exist_ok=True)
    archived: list[ArchivedRoot] = []
    for name, source, target, resumed in plan:
        before = (list(rows), _snapshot(batch_root))
        if resumed:
            item = ArchivedRoot.from_record(document["run_roots"][name])
        else:
            rows, item = _archived_rows(
                rows, live.get(name, []), name, source, key, ticket, shown_root
            )
            document["run_roots"][name] = item.as_record()
            document.update(ticket=ticket, authority=authority, location_key=key)
            _record(ledger_path, rows, batch_root, document, before, name)
        try:
            os.rename(source, target)
        except OSError as error:
            if not resumed:
                _roll_back(ledger_path, batch_root, before)
            raise ArchiveError(
                f"{name}: move failed ({error}); nothing moved"
            ) from error
        archived.append(item)
    return tuple(archived)


def _run_root_rows(rows: list[LedgerRow], location_key: str) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row.kind == "run_root" and row.location_key == location_key:
            found.setdefault(row.identity, []).append(index)
    return found


def _archived_rows(
    rows: list[LedgerRow],
    indexes: list[int],
    name: str,
    source: Path,
    key: str,
    ticket: str,
    shown_root: Path,
) -> tuple[list[LedgerRow], ArchivedRoot]:
    """The ledger with ``name`` rekeyed (or added) and the manifest entry for it."""

    size, digest, caveat = _inventory(source)
    where = f"archived_to {shown_root / name} (#{ticket})"
    if caveat:
        where = f"{where}; {caveat}"
    rows = list(rows)
    if indexes:
        for index in indexes:
            row = rows[index]
            rows[index] = row.replace(
                location_key=key,
                notes=f"{row.notes}; {where}" if row.notes else where,
                size_bytes=row.size_bytes or size,
                inventory_sha256=row.inventory_sha256 or digest,
            )
    else:
        modified = _dt.datetime.fromtimestamp(source.lstat().st_mtime, _dt.UTC)
        row = _new_row(
            name,
            key,
            ticket,
            modified.date().isoformat(),
            f"ledgered at archive (#{ticket}); {where}",
        )
        rows.append(row.replace(size_bytes=size, inventory_sha256=digest))
    run_ids = tuple(
        sorted(
            {
                row.run_id
                for row in rows
                if row.identity == name and row.kind == "run_root"
            }
        )
    )
    item = ArchivedRoot(
        name,
        run_ids,
        f"run_roots/{name}",
        str(shown_root / name),
        size,
        digest,
        "existing" if indexes else "added",
    )
    return rows, item


def restore_run_root(
    ledger_path: Path,
    run_roots: Path,
    archive_root: Path,
    *,
    batch: str,
    identity: str,
    ticket: str,
) -> None:
    """Record one archived root as back in ``run_roots``, then move it there."""

    key = archive_key(batch)
    batch_root = archive_root / batch
    source = batch_root / ARCHIVED_RUN_ROOTS / identity
    target = run_roots / identity
    rows = list(read_ledger(ledger_path))
    document = _read_manifest(batch_root)
    indexes = _run_root_rows(rows, key).get(identity, [])
    resumed = not indexes and identity in document["restored"]
    if not indexes and not resumed:
        raise ArchiveError(f"{identity} has no row archived in {key}")
    if source.is_symlink() or not source.is_dir():
        raise ArchiveError(f"{source} is not an archived run root directory")
    if target.exists() or target.is_symlink():
        raise ArchiveError(f"{target} already exists")
    before = (list(rows), _snapshot(batch_root))
    if not resumed:
        note = f"restored to run_roots (#{ticket})"
        for index in indexes:
            row = rows[index]
            rows[index] = row.replace(
                location_key="run_roots", notes=f"{row.notes}; {note}"
            )
        entry = (
            document["run_roots"].pop(identity, None) or document["restored"][identity]
        )
        document["restored"][identity] = {
            **entry,
            "restored_to": f"run_roots/{identity}",
        }
        _record(ledger_path, rows, batch_root, document, before, identity)
    try:
        os.rename(source, target)
    except OSError as error:
        if not resumed:
            _roll_back(ledger_path, batch_root, before)
        raise ArchiveError(
            f"{identity}: move failed ({error}); nothing moved"
        ) from error


_MANIFEST_FILES = ("manifest.json", "MANIFEST.md")
Snapshot = dict[str, bytes | None]


def _snapshot(batch_root: Path) -> Snapshot:
    return {
        name: (batch_root / name).read_bytes()
        if (batch_root / name).is_file()
        else None
        for name in _MANIFEST_FILES
    }


def _record(
    ledger_path: Path,
    rows: list[LedgerRow],
    batch_root: Path,
    document: dict[str, Any],
    before: tuple[list[LedgerRow], Snapshot],
    name: str,
) -> None:
    """Write the manifest, then the ledger; a failure leaves both as they were.

    The manifest goes first, so every interrupted state is one a rerun recognises: a manifest
    entry whose rows were not yet changed is recorded again, and changed rows always have
    their manifest entry.
    """

    try:
        _render_manifest(batch_root, document)
    except OSError as error:
        raise ArchiveError(
            f"{name}: manifest not written ({error}); nothing moved"
        ) from error
    try:
        write_ledger(ledger_path, rows)
    except OSError as error:
        _restore_manifest(
            batch_root, before[1]
        )  # the atomic ledger write left the old file
        raise ArchiveError(
            f"{name}: ledger not written ({error}); nothing moved"
        ) from error


def _roll_back(
    ledger_path: Path, batch_root: Path, before: tuple[list[LedgerRow], Snapshot]
) -> None:
    rows, snapshot = before
    write_ledger(ledger_path, rows)
    _restore_manifest(batch_root, snapshot)


def _restore_manifest(batch_root: Path, snapshot: Snapshot) -> None:
    for name, content in snapshot.items():
        path = batch_root / name
        if content is None:
            path.unlink(missing_ok=True)
        else:
            _atomic_write(path, content)


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _read_manifest(batch_root: Path) -> dict[str, Any]:
    path = batch_root / "manifest.json"
    if not path.is_file():
        return {"run_roots": {}, "restored": {}}
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        return {
            **previous,
            "run_roots": {entry["identity"]: entry for entry in previous["run_roots"]},
            "restored": {
                entry["identity"]: entry for entry in previous.get("restored", ())
            },
        }
    except (ValueError, KeyError, TypeError) as error:
        raise ArchiveError(f"{path} is not a readable batch manifest") from error


def _render_manifest(batch_root: Path, document: dict[str, Any]) -> None:
    entries, restored = document["run_roots"], document["restored"]
    key, ticket, authority = (
        document["location_key"],
        document["ticket"],
        document["authority"],
    )
    rendered: dict[str, Any] = {
        "ticket": ticket,
        "authority": authority,
        "location_key": key,
        "run_roots": [entries[name] for name in sorted(entries)],
    }
    if restored:
        rendered["restored"] = [restored[name] for name in sorted(restored)]
    lines = [
        f"# {batch_root.name}",
        "",
        f"Archived run roots of #{ticket}. Authority: {authority}.",
        "",
        "The bytes stay local under `run-roots/` (gitignored). Each root's ledger rows",
        f"(`docs/runs/ledger.csv`) carry `location_key` `{key}` and an `archived_to` note;",
        "`python -m bio_neb.locations` names `archive_root`. `manifest.json` lists each root's",
        "original location, archived path, size and inventory hash. Nothing was deleted.",
        "",
        "| Run root | Ledger |",
        "| --- | --- |",
        *(f"| `{name}` | {entries[name]['ledger']} |" for name in sorted(entries)),
        "",
    ]
    if restored:
        lines += [
            "Restored to `run_roots` (rows say `restored to run_roots`):",
            "",
            *(f"- `{name}`" for name in sorted(restored)),
            "",
        ]
    batch_root.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        batch_root / "manifest.json", (json.dumps(rendered, indent=2) + "\n").encode()
    )
    _atomic_write(batch_root / "MANIFEST.md", "\n".join(lines).encode())
