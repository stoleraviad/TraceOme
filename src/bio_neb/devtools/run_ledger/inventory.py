"""``inventory``: list, diff, highlight and (on request) fill empty cells.

Read-only by construction: the provider side is ``bio_neb.devtools.run_ledger.provider`` (list
operations only); the local side walks the ``locations`` directories. ``--update``
fills only empty ``size_bytes``, ``inventory_sha256`` and ``storage_class`` cells and
never touches an owner-only column (#39, policy 5 and the tool spec).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ledger import (
    ARCHIVE_KEY_PREFIX,
    ARCHIVED_RUN_ROOTS,
    OWNER_ONLY_COLUMNS,
    LedgerRow,
    archive_batch,
    archive_key,
    attempt_number,
    bucket_listing_sha256,
    directory_inventory,
    read_ledger,
    resource_key,
    resource_record,
    run_purpose,
    write_ledger,
)

__all__ = [
    "Candidate",
    "FilledCell",
    "InventoryDiff",
    "ListedBucket",
    "ListedObject",
    "Listing",
    "LocalResource",
    "Warning",
    "deprecation_candidates",
    "diff_ledger",
    "effective_storage_class",
    "fill_empty_cells",
    "local_listing",
    "run_inventory",
    "run_purpose",
    "safe_to_remove_warnings",
]

_LOCAL_KINDS = ("run_root", "receipt", "checkout")
# Location key → (kind, hash the content). Checkouts are recorded by SHA only; hashing a
# working tree with its virtual environment proves nothing the SHA does not.
_LOCATION_KINDS: Mapping[str, tuple[str, bool]] = {
    "run_roots": ("run_root", True),
    "checkouts_root": ("checkout", False),
    "testing_run_roots": ("receipt", True),
    "legacy_receipts": ("receipt", True),
}


@dataclass(frozen=True, slots=True)
class ListedObject:
    key: str
    size: int
    etag: str = ""
    checksum: str = ""
    storage_class: str = ""


@dataclass(frozen=True, slots=True)
class ListedBucket:
    name: str
    bucket_id: str
    storage_class: str
    objects: tuple[ListedObject, ...] | None
    region: str = ""
    domain: str = ""

    @property
    def location_key(self) -> str:
        return ""

    @property
    def identity(self) -> str:
        return self.name

    @property
    def kind(self) -> str:
        return "bucket"

    @property
    def size_bytes(self) -> int | None:
        if self.objects is None:
            return None
        return sum(item.size for item in self.objects)

    @property
    def inventory_sha256(self) -> str:
        if self.objects is None:
            return ""
        return bucket_listing_sha256(self.objects)

    def as_record(self) -> dict[str, Any]:
        return {
            "kind": "bucket",
            "location_key": "",
            "identity": self.name,
            "bucket_id": self.bucket_id,
            "storage_class": self.storage_class,
            "region": self.region,
            "domain": self.domain,
            "object_count": None if self.objects is None else len(self.objects),
            "size_bytes": self.size_bytes,
            "inventory_sha256": self.inventory_sha256,
            "objects": (
                None if self.objects is None else [asdict(o) for o in self.objects]
            ),
        }


@dataclass(frozen=True, slots=True)
class LocalResource:
    kind: str
    location_key: str
    identity: str
    size_bytes: int
    inventory_sha256: str

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Listing:
    buckets: tuple[ListedBucket, ...]
    local: tuple[LocalResource, ...]
    provider_listed: bool
    contact: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InventoryDiff:
    unledgered: tuple[ListedBucket | LocalResource, ...]
    missing: tuple[LedgerRow, ...]
    removed_but_present: tuple[LedgerRow, ...]
    matched: tuple[LedgerRow, ...]


@dataclass(frozen=True, slots=True)
class Candidate:
    row: LedgerRow
    reason: str


@dataclass(frozen=True, slots=True)
class Warning:
    kind: str
    location_key: str
    identity: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class FilledCell:
    kind: str
    location_key: str
    identity: str
    column: str
    value: str


def effective_storage_class(
    default_class: str, objects: tuple[ListedObject, ...] | None
) -> str:
    """The bucket's default class, else the one class every listed object reports."""

    if default_class:
        return default_class.casefold()
    if not objects:
        return ""
    classes = {item.storage_class.casefold() for item in objects}
    if len(classes) == 1 and "" not in classes:
        return classes.pop()
    return ""


def _directories(root: Path | None) -> list[Path]:
    if root is None or not Path(root).is_dir():
        return []
    return [child for child in sorted(Path(root).iterdir()) if child.is_dir()]


def local_listing(locations: Mapping[str, Path]) -> tuple[LocalResource, ...]:
    """One resource per top-level directory under each known location key.

    ``archive_root`` adds every run root under ``<batch>/run-roots`` keyed ``archive:<batch>``.
    """

    resources: list[LocalResource] = []
    for location_key, (kind, hash_content) in _LOCATION_KINDS.items():
        root = locations.get(location_key)
        if root is None or not Path(root).is_dir():
            continue
        for child in sorted(Path(root).iterdir()):
            if not child.is_dir():
                continue
            size, digest = directory_inventory(child, hash_content=hash_content)
            resources.append(
                LocalResource(kind, location_key, child.name, size, digest)
            )
    for batch in _directories(locations.get("archive_root")):
        if not archive_batch(ARCHIVE_KEY_PREFIX + batch.name):
            continue
        for child in _directories(batch / ARCHIVED_RUN_ROOTS):
            size, digest = directory_inventory(child)
            resources.append(
                LocalResource(
                    "run_root", archive_key(batch.name), child.name, size, digest
                )
            )
    return tuple(resources)


def diff_ledger(rows: Iterable[LedgerRow], listing: Listing) -> InventoryDiff:
    rows = tuple(rows)
    by_identity = {resource_key(row): row for row in rows}
    listed: dict[tuple[str, str, str], ListedBucket | LocalResource] = {}
    if listing.provider_listed:
        listed.update({resource_key(bucket): bucket for bucket in listing.buckets})
    listed.update({resource_key(item): item for item in listing.local})
    unledgered = tuple(item for name, item in listed.items() if name not in by_identity)
    missing: list[LedgerRow] = []
    removed_but_present: list[LedgerRow] = []
    matched: list[LedgerRow] = []
    for row in rows:
        covered = row.kind in _LOCAL_KINDS or (
            row.kind == "bucket" and listing.provider_listed
        )
        if not covered:
            continue
        present = resource_key(row) in listed
        if row.status == "removed":
            if present:
                removed_but_present.append(row)
        elif present:
            matched.append(row)
        else:
            missing.append(row)
    return InventoryDiff(
        unledgered, tuple(missing), tuple(removed_but_present), tuple(matched)
    )


def deprecation_candidates(rows: Iterable[LedgerRow]) -> tuple[Candidate, ...]:
    """Current bucket/run_root rows outrun by a later current row of the same purpose."""

    eligible = [
        row
        for row in rows
        if row.kind in ("bucket", "run_root") and row.status == "current"
    ]
    groups: dict[tuple[str, str], list[LedgerRow]] = {}
    for row in eligible:
        groups.setdefault((row.kind, run_purpose(row.run_id)), []).append(row)
    candidates: list[Candidate] = []
    for group in groups.values():
        newest = max(group, key=lambda row: (row.date, attempt_number(row.run_id)))
        for row in group:
            if row is newest or row.fixture_use:
                continue
            if row.source_commit != newest.source_commit:
                reason = f"superseded_by {newest.source_commit}"
            else:
                reason = f"duplicate_of {newest.run_id}"
            candidates.append(Candidate(row, reason))
    return tuple(candidates)


def safe_to_remove_warnings(
    rows: Iterable[LedgerRow], *, open_tickets: frozenset[str] | None
) -> tuple[Warning, ...]:
    """The five #39 warnings for rows the owner marked ``safe_to_remove``; never a block."""

    warnings: list[Warning] = []
    for row in rows:
        if row.safe_to_remove != "yes" or row.status == "removed":
            continue
        if row.status != "deprecated":
            warnings.append(
                Warning(
                    *resource_key(row),
                    "not_deprecated",
                    "safe_to_remove without status=deprecated",
                )
            )
        if row.fixture_use:
            warnings.append(
                Warning(
                    *resource_key(row),
                    "has_fixture_use",
                    f"feeds fixture {row.fixture_use}",
                )
            )
        if row.kind in _LOCAL_KINDS and not row.backup:
            warnings.append(
                Warning(
                    *resource_key(row),
                    "local_without_backup",
                    "local kind with no recorded backup",
                )
            )
        if row.kind == "bucket" and not row.inventory_sha256:
            warnings.append(
                Warning(
                    *resource_key(row),
                    "bucket_without_inventory_sha256",
                    "bucket contents were never inventoried",
                )
            )
        if open_tickets is not None and row.ticket in open_tickets:
            warnings.append(
                Warning(
                    *resource_key(row),
                    "citing_ticket_open",
                    f"ticket #{row.ticket} is open",
                )
            )
    return tuple(warnings)


def fill_empty_cells(
    rows: Iterable[LedgerRow], listing: Listing
) -> tuple[tuple[LedgerRow, ...], tuple[FilledCell, ...]]:
    """Fill only empty size/hash/class cells from the listing; owner columns untouched."""

    buckets = {resource_key(bucket): bucket for bucket in listing.buckets}
    local = {resource_key(item): item for item in listing.local}
    filled: list[FilledCell] = []
    result: list[LedgerRow] = []
    for row in rows:
        changes: dict[str, str] = {}
        if row.kind == "bucket" and resource_key(row) in buckets:
            bucket = buckets[resource_key(row)]
            if not row.size_bytes and bucket.size_bytes is not None:
                changes["size_bytes"] = str(bucket.size_bytes)
            if not row.inventory_sha256 and bucket.inventory_sha256:
                changes["inventory_sha256"] = bucket.inventory_sha256
            if not row.storage_class and bucket.storage_class:
                changes["storage_class"] = bucket.storage_class
        elif row.kind in _LOCAL_KINDS and resource_key(row) in local:
            item = local[resource_key(row)]
            if not row.size_bytes:
                changes["size_bytes"] = str(item.size_bytes)
            if not row.inventory_sha256 and item.inventory_sha256:
                changes["inventory_sha256"] = item.inventory_sha256
        assert not set(changes) & set(OWNER_ONLY_COLUMNS)
        filled.extend(
            FilledCell(*resource_key(row), column, value)
            for column, value in changes.items()
        )
        result.append(row.replace(**changes) if changes else row)
    return tuple(result), tuple(filled)


def run_inventory(
    ledger_path: Path,
    listing: Listing,
    *,
    update: bool,
    open_tickets: frozenset[str] | None,
) -> dict[str, Any]:
    """Diff, highlight, optionally fill, and return the JSON-ready report."""

    rows = read_ledger(ledger_path)
    filled: tuple[FilledCell, ...] = ()
    if update:
        rows, filled = fill_empty_cells(rows, listing)
        if filled:
            write_ledger(ledger_path, rows)
    diff = diff_ledger(rows, listing)
    candidates = deprecation_candidates(rows)
    warnings = safe_to_remove_warnings(rows, open_tickets=open_tickets)
    return {
        "schema_version": "bio-neb.run-ledger-inventory/2",
        "ledger": str(ledger_path),
        "update": update,
        "provider_listed": listing.provider_listed,
        "contact": dict(listing.contact),
        "open_ticket_check": (
            "not_performed" if open_tickets is None else sorted(open_tickets)
        ),
        "buckets": [bucket.as_record() for bucket in listing.buckets],
        "local": [item.as_record() for item in listing.local],
        "unledgered": [resource_record(item) for item in diff.unledgered],
        "missing": [row.as_record() for row in diff.missing],
        "removed_but_present": [row.as_record() for row in diff.removed_but_present],
        "matched": [resource_record(row) for row in diff.matched],
        "candidates": [
            {
                **resource_record(item.row),
                "run_id": item.row.run_id,
                "reason": item.reason,
            }
            for item in candidates
        ],
        "warnings": [asdict(item) for item in warnings],
        "filled": [asdict(item) for item in filled],
    }
