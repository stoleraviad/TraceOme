"""The run ledger's schema, stable CSV form and listing hashes.

Decided on issue #39 (owner, 2026-09-20): one row per retained resource, grouped by
``run_id``; the header order below is exact; ``backup``, ``safe_to_remove``,
``removed_at`` and ``removed_by`` are owner-only columns that no tool path writes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Protocol

HEADER: tuple[str, ...] = (
    "run_id",
    "ticket",
    "date",
    "source_commit",
    "kind",
    "identity",
    "location_key",
    "size_bytes",
    "inventory_sha256",
    "storage_class",
    "status",
    "deprecated_reason",
    "fixture_use",
    "backup",
    "safe_to_remove",
    "removed_at",
    "removed_by",
    "notes",
)
KINDS: tuple[str, ...] = (
    "bucket",
    "run_root",
    "receipt",
    "checkout",
    "host_evidence",
    "key",
)
STATUSES: tuple[str, ...] = ("current", "deprecated", "removed")
OWNER_ONLY_COLUMNS: tuple[str, ...] = (
    "backup",
    "safe_to_remove",
    "removed_at",
    "removed_by",
)
# The location keys the ``inventory`` command walks; ``host`` is citation only.
LOCATION_KEYS: Mapping[str, tuple[str, ...]] = {
    "run_root": ("run_roots",),
    "receipt": ("testing_run_roots", "legacy_receipts"),
    "checkout": ("checkouts_root",),
    "host_evidence": ("host",),
    "bucket": ("",),
    "key": ("",),
}
LEGACY_RECEIPTS = Path("/home/node/bio-neb-runs")
# A run root approved for archiving lives under
# ``<archive_root>/<batch>/run-roots/<identity>``; its rows carry ``archive:<batch>``.
ARCHIVE_KEY_PREFIX = "archive:"
ARCHIVED_RUN_ROOTS = "run-roots"
_ARCHIVE_BATCH = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*$")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA_TOKEN = re.compile(r"^[0-9a-f]{7,40}$")
_DATE_TOKEN = re.compile(r"^\d{8}(T\d{4,6}Z)?$")
_ATTEMPT_TOKEN = re.compile(r"^a\d+$")


def archive_key(batch: str) -> str:
    if not _ARCHIVE_BATCH.match(batch):
        raise LedgerError(f"archive batch is <yyyy-mm-dd>-<name>, not {batch!r}")
    return ARCHIVE_KEY_PREFIX + batch


def archive_batch(location_key: str) -> str:
    """The batch of an ``archive:<batch>`` location key, else ``""``."""

    if not location_key.startswith(ARCHIVE_KEY_PREFIX):
        return ""
    batch = location_key.removeprefix(ARCHIVE_KEY_PREFIX)
    return batch if _ARCHIVE_BATCH.match(batch) else ""


class ResourceIdentity(Protocol):
    @property
    def kind(self) -> str: ...

    @property
    def location_key(self) -> str: ...

    @property
    def identity(self) -> str: ...


def resource_key(resource: ResourceIdentity) -> tuple[str, str, str]:
    """Existing resource scope; Run citations do not change resource identity."""

    return resource.kind, resource.location_key, resource.identity


def resource_record(resource: ResourceIdentity) -> dict[str, str]:
    return dict(
        zip(("kind", "location_key", "identity"), resource_key(resource), strict=True)
    )


class LedgerError(ValueError):
    """A ledger file or row breaks the #39 schema."""


@dataclass(frozen=True, slots=True)
class LedgerRow:
    run_id: str
    ticket: str
    date: str
    source_commit: str
    kind: str
    identity: str
    location_key: str = ""
    size_bytes: str = ""
    inventory_sha256: str = ""
    storage_class: str = ""
    status: str = "current"
    deprecated_reason: str = ""
    fixture_use: str = ""
    backup: str = ""
    safe_to_remove: str = ""
    removed_at: str = ""
    removed_by: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        for name in HEADER:
            value = getattr(self, name)
            if not isinstance(value, str) or "\n" in value or "\r" in value:
                raise LedgerError(f"{name} must be one line of text")
        if not self.run_id or not self.identity:
            raise LedgerError("run_id and identity are required")
        if self.kind not in KINDS:
            raise LedgerError(f"kind must be one of {KINDS}, not {self.kind!r}")
        if self.status not in STATUSES:
            raise LedgerError(f"status must be one of {STATUSES}, not {self.status!r}")
        if self.safe_to_remove not in ("", "yes"):
            raise LedgerError("safe_to_remove is empty or 'yes'")
        if self.location_key not in LOCATION_KEYS[self.kind] and not (
            self.kind == "run_root" and archive_batch(self.location_key)
        ):
            raise LedgerError(
                f"{self.kind} rows take location_key in {LOCATION_KEYS[self.kind]}"
            )
        if self.size_bytes and not self.size_bytes.isdigit():
            raise LedgerError("size_bytes is a byte count or empty")
        if self.inventory_sha256 and not _SHA256.match(self.inventory_sha256):
            raise LedgerError("inventory_sha256 is a lowercase SHA-256 or empty")

    def replace(self, **changes: str) -> LedgerRow:
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        values.update(changes)
        return LedgerRow(**values)

    def as_record(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in HEADER}


def _sort_key(row: LedgerRow) -> tuple[str, int, str, str]:
    return (row.run_id, KINDS.index(row.kind), row.identity, row.location_key)


def sorted_rows(rows: Iterable[LedgerRow]) -> tuple[LedgerRow, ...]:
    """Stable order: by run, kind, identity, then location to break basename ties."""

    return tuple(sorted(rows, key=_sort_key))


def check_unique(rows: Iterable[LedgerRow]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if resource_key(row) in seen:
            raise LedgerError(f"duplicate resource {resource_key(row)!r}")
        seen.add(resource_key(row))


def read_ledger(path: Path) -> tuple[LedgerRow, ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise LedgerError(f"{path} is empty") from error
        if header != HEADER:
            raise LedgerError(f"{path} header is not the #39 header: {header!r}")
        rows: list[LedgerRow] = []
        for index, record in enumerate(reader, start=2):
            if len(record) != len(HEADER):
                raise LedgerError(f"{path}:{index} has {len(record)} cells")
            try:
                rows.append(LedgerRow(**dict(zip(HEADER, record, strict=True))))
            except LedgerError as error:
                raise LedgerError(f"{path}:{index}: {error}") from error
    check_unique(rows)
    return tuple(rows)


def render_ledger(rows: Iterable[LedgerRow]) -> str:
    ordered = sorted_rows(rows)
    check_unique(ordered)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(HEADER)
    for row in ordered:
        writer.writerow([getattr(row, name) for name in HEADER])
    return buffer.getvalue()


def write_ledger(path: Path, rows: Iterable[LedgerRow]) -> None:
    """Replace the ledger atomically: a failed write leaves the previous file whole."""

    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(render_ledger(rows), encoding="utf-8", newline="")
    os.replace(temporary, path)


def bucket_listing_sha256(objects: Iterable[Mapping[str, object] | object]) -> str:
    """SHA-256 over sorted ``key\\tsize\\tetag\\tchecksum-sha256\\n`` listing lines.

    Never a re-read: every field comes from a bucket listing (#38). ETag quotes are
    stripped; a checksum the listing could not see stays empty.
    """

    lines: list[str] = []
    for item in objects:
        if isinstance(item, Mapping):
            key, size, etag, checksum = (
                item["key"],
                item["size"],
                item.get("etag", ""),
                item.get("checksum", ""),
            )
        else:
            key, size, etag, checksum = (
                item.key,
                item.size,
                item.etag,
                item.checksum,
            )
        lines.append(f"{key}\t{size}\t{str(etag).strip(chr(34))}\t{checksum or ''}\n")
    return hashlib.sha256("".join(sorted(lines)).encode("utf-8")).hexdigest()


def directory_inventory(root: Path, *, hash_content: bool = True) -> tuple[int, str]:
    """Total size and SHA-256 over sorted ``relpath\\tsize\\tsha256\\n`` lines."""

    root = Path(root)
    lines: list[str] = []
    total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        size = path.stat().st_size
        total += size
        if not hash_content:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        lines.append(
            f"{path.relative_to(root).as_posix()}\t{size}\t{digest.hexdigest()}\n"
        )
    if not hash_content:
        return total, ""
    return total, hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def derive_run_id(name: str) -> str:
    """A run id from a bucket, root or receipt name: drop ``bio-neb`` and date tokens."""

    tokens = [token for token in name.split("-") if token]
    if tokens[:2] == ["bio", "neb"]:
        tokens = tokens[2:]
    return "-".join(token for token in tokens if not _DATE_TOKEN.match(token))


def name_tokens(name: str) -> tuple[str, str]:
    """The date token and the commit token of a run, bucket or receipt name (``""`` if absent)."""

    date_token = ""
    sha_token = ""
    for token in name.split("-"):
        if not date_token and _DATE_TOKEN.match(token):
            date_token = token
        elif not sha_token and _SHA_TOKEN.match(token) and not token.isdigit():
            sha_token = token
    return date_token, sha_token


def run_purpose(run_id: str) -> str:
    """The run id without its commit and attempt tokens (``t19-hit-cc52cd6a-a1`` → ``t19-hit``)."""

    tokens = [
        token
        for token in run_id.split("-")
        if not _SHA_TOKEN.match(token) and not _ATTEMPT_TOKEN.match(token)
    ]
    return "-".join(tokens)


def attempt_number(run_id: str) -> int:
    for token in reversed(run_id.split("-")):
        if _ATTEMPT_TOKEN.match(token):
            return int(token[1:])
    return 0


_TICKET_TOKEN = re.compile(r"^(?:issue|i|t)?(\d+)$")
_COMMIT_TOKEN = re.compile(r"^[0-9a-f]{3,40}$")


def numbering_key(name: str) -> tuple[str, str]:
    """``(ticket, purpose)`` of ``<ticket>-<purpose>[-<UTC>][-<commit>]-a<N>``, read by position.

    Only the trailing attempt, commit and date tokens are metadata, so a purpose token that
    merely looks like a commit, date or attempt (``deadbeef``, ``live-a1``) stays in the purpose.
    """

    tokens = [token for token in name.split("-") if token]
    if tokens[:2] == ["bio", "neb"]:
        tokens = tokens[2:]
    ticket = _TICKET_TOKEN.match(tokens[0]) if tokens else None
    if ticket is None:
        return "", ""
    rest = tokens[1:]
    if rest and _ATTEMPT_TOKEN.match(rest[-1]):
        rest = rest[:-1]
    if len(rest) > 2 and _DATE_TOKEN.match(rest[-2]) and _COMMIT_TOKEN.match(rest[-1]):
        rest = rest[:-2]  # <UTC>-<commit>, even when the commit is all digits
    elif len(rest) > 1 and (
        _DATE_TOKEN.match(rest[-1]) or _COMMIT_TOKEN.match(rest[-1])
    ):
        rest = rest[:-1]
    return ticket.group(1), "-".join(rest)


def next_attempt(names: Iterable[str], *, ticket: str, purpose: str) -> int:
    """One past the highest ``a<N>`` any known run of this ticket and purpose used."""

    used = [
        attempt_number(name)
        for name in names
        if numbering_key(name) == (ticket, purpose)
    ]
    return max(used, default=0) + 1
