"""Command line for the run ledger: ``add`` (offline) and ``inventory`` (list-only)."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bio_neb import locations

from .add import AddError, add_run
from .archiving import (
    ArchiveError,
    archive_run_roots,
    read_approved,
    restore_run_root,
)
from .inventory import (
    ListedBucket,
    ListedObject,
    Listing,
    effective_storage_class,
    local_listing,
    run_inventory,
)
from .ledger import (
    ARCHIVED_RUN_ROOTS,
    LEGACY_RECEIPTS,
    LedgerError,
    next_attempt,
    read_ledger,
)

REPOSITORY = Path(__file__).resolve().parents[4]
DEFAULT_LEDGER = REPOSITORY / "docs" / "runs" / "ledger.csv"
_CREDENTIAL_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "BIO_NEB_S3_CREDENTIALS_PROFILE",
)


def _receipt(value: str) -> tuple[str, str]:
    location_key, separator, name = value.partition(":")
    if (
        not separator
        or not name
        or location_key not in ("testing_run_roots", "legacy_receipts")
    ):
        raise argparse.ArgumentTypeError(
            "receipts are <testing_run_roots|legacy_receipts>:<directory-name>"
        )
    return location_key, name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bio_neb.devtools.run_ledger")
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser(
        "add", help="append rows from a finished run root (offline)"
    )
    add.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    add.add_argument(
        "--run-root", type=Path, help="finished run root; omit to ledger receipts alone"
    )
    add.add_argument("--ticket", required=True)
    add.add_argument(
        "--run-id", help="run id for every row; receipt-only rows: one commit only"
    )
    add.add_argument(
        "--source-commit", help="commit for receipt-only rows of one commit"
    )
    add.add_argument(
        "--receipt",
        action="append",
        default=[],
        type=_receipt,
        help="receipt directory to ledger with this run, as <location_key>:<name>",
    )

    next_run_id = commands.add_parser(
        "next-run-id",
        help="name a fresh run with the next a<N> for its ticket and purpose (offline)",
    )
    next_run_id.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    next_run_id.add_argument("--run-roots", type=Path)
    next_run_id.add_argument("--archive-root", type=Path)
    next_run_id.add_argument("--ticket", required=True)
    next_run_id.add_argument("--purpose", required=True)
    next_run_id.add_argument("--commit", required=True)
    next_run_id.add_argument("--utc", help="UTC stamp yyyymmddThhmmssZ; default now")

    archive = commands.add_parser(
        "archive",
        help="move owner-approved run roots into an .archive batch and record where",
    )
    archive.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    archive.add_argument("--run-roots", type=Path)
    archive.add_argument("--archive-root", type=Path)
    archive.add_argument("--batch", required=True, help="<yyyy-mm-dd>-<name>")
    archive.add_argument(
        "--approved", type=Path, required=True, help="the owner-approved run root names"
    )
    archive.add_argument("--ticket", required=True)
    archive.add_argument(
        "--authority", required=True, help="the owner approval locator"
    )
    archive.add_argument(
        "--dry-run", action="store_true", help="validate and list only"
    )

    restore = commands.add_parser(
        "restore", help="move one archived run root back to run_roots and record it"
    )
    restore.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    restore.add_argument("--run-roots", type=Path)
    restore.add_argument("--archive-root", type=Path)
    restore.add_argument("--batch", required=True)
    restore.add_argument("--identity", required=True)
    restore.add_argument("--ticket", required=True)

    inventory = commands.add_parser(
        "inventory", help="list provider and local resources, diff, warn (read-only)"
    )
    inventory.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    inventory.add_argument(
        "--update", action="store_true", help="fill empty cells only"
    )
    inventory.add_argument(
        "--no-contact", action="store_true", help="local walk only; no provider listing"
    )
    inventory.add_argument(
        "--project-id", help="Nebius project whose buckets are listed"
    )
    inventory.add_argument("--profile", default="bio-neb-sa")
    inventory.add_argument("--bucket-prefix", default="bio-neb-")
    inventory.add_argument("--open-tickets", help="comma-separated open ticket numbers")
    inventory.add_argument("--legacy-receipts", type=Path, default=LEGACY_RECEIPTS)
    inventory.add_argument(
        "--from-receipt",
        type=Path,
        help="reuse the bucket listing of an earlier inventory.json instead of contact",
    )
    inventory.add_argument(
        "--ticket", help="names the receipt root (issue<N>-inventory-...)"
    )
    inventory.add_argument(
        "--receipt-root",
        type=Path,
        help="exact fresh receipt directory (default: a fresh root under testing_run_roots)",
    )
    return parser


def _directory_names(root: Path) -> list[str]:
    try:
        return [child.name for child in root.iterdir() if child.is_dir()]
    except FileNotFoundError:
        return []


def _known_run_names(ledger: Path, run_roots: Path, archive_root: Path) -> list[str]:
    """Every run name the ledger, ``run_roots`` or an archive batch already used."""

    names = [
        name
        for row in read_ledger(ledger)
        if row.kind == "run_root"
        for name in (row.run_id, row.identity)
    ]
    names.extend(_directory_names(run_roots))
    for batch in _directory_names(archive_root):
        names.extend(_directory_names(archive_root / batch / ARCHIVED_RUN_ROOTS))
    return names


def _next_run_id(args: argparse.Namespace) -> int:
    names = _known_run_names(
        args.ledger,
        args.run_roots or locations.run_roots(),
        args.archive_root or locations.archive_root(),
    )
    attempt = next_attempt(names, ticket=args.ticket, purpose=args.purpose)
    stamp = args.utc or _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    sys.stdout.write(
        f"{args.ticket}-{args.purpose}-{stamp}-{args.commit[:8]}-a{attempt}\n"
    )
    return 0


def _short_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPOSITORY), "rev-parse", "--short=8", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "nogit"
    return completed.stdout.strip() or "nogit"


def _fresh_receipt_root(ticket: str | None, explicit: Path | None) -> Path:
    if explicit is not None:
        explicit.mkdir(parents=True, exist_ok=False)
        return explicit
    base = locations.testing_run_roots()
    base.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    label = f"issue{ticket}" if ticket else "ledger"
    for attempt in range(1, 100):
        root = base / f"{label}-inventory-{stamp}-{_short_sha()}-a{attempt}"
        if not root.exists():
            root.mkdir(parents=True, exist_ok=False)
            return root
    raise RuntimeError("no free receipt root")


def _buckets_from_receipt(path: Path) -> tuple[ListedBucket, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    buckets = []
    for record in document.get("buckets", ()):
        objects = record.get("objects")
        listed = (
            None if objects is None else tuple(ListedObject(**item) for item in objects)
        )
        buckets.append(
            ListedBucket(
                name=record["identity"],
                bucket_id=record.get("bucket_id", ""),
                storage_class=effective_storage_class(
                    record.get("storage_class", ""), listed
                ),
                objects=listed,
                region=record.get("region", ""),
                domain=record.get("domain", ""),
            )
        )
    return tuple(buckets)


def _provider_listing(
    args: argparse.Namespace, journal: Any
) -> tuple[tuple[ListedBucket, ...], bool]:
    if args.from_receipt is not None:
        return _buckets_from_receipt(args.from_receipt), True
    if args.no_contact:
        return (), False
    if not args.project_id:
        raise SystemExit("inventory: --project-id is required for a provider listing")
    if any(not os.environ.get(name) for name in _CREDENTIAL_NAMES):
        raise SystemExit(
            "inventory: credentials are not in the environment; run through "
            "`uv run --env-file <controller.env>` or pass --no-contact"
        )
    from bio_neb.providers.nebius_common import EnvironmentAwsCredentialResolver
    from bio_neb.providers.nebius_controller_runtime import (
        accepted_nebius_cli_identity,
        make_accepted_nebius_cli_runner,
    )
    from bio_neb.providers.nebius_object_storage import nebius_object_storage_client

    from .provider import list_buckets

    identity = accepted_nebius_cli_identity(str(locations.nebius_executable()))
    runner = make_accepted_nebius_cli_runner(
        identity.executable, execution_contact_observer=journal
    )
    buckets = list_buckets(
        runner,
        nebius_object_storage_client,
        credential_resolver=EnvironmentAwsCredentialResolver(),
        project_id=args.project_id,
        profile=args.profile,
        journal=journal,
        prefix=args.bucket_prefix,
    )
    return buckets, True


def _journal_operations(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.is_file():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("row_kind") == "terminal":
            counts[row["operation"]] = counts.get(row["operation"], 0) + 1
    return counts


def _run_inventory(args: argparse.Namespace) -> int:
    from bio_neb.parabricks.execution_contact_accounting import ExecutionContactJournal

    receipt_root = _fresh_receipt_root(args.ticket, args.receipt_root)
    journal_path = receipt_root / "journal.jsonl"
    digest = hashlib.sha256(receipt_root.name.encode()).hexdigest()
    journal = ExecutionContactJournal(journal_path, f"parabricks-execution:{digest}")
    try:
        buckets, provider_listed = _provider_listing(args, journal)
    finally:
        journal.finalize()
    local = local_listing(
        {
            "run_roots": locations.run_roots(),
            "checkouts_root": locations.checkouts_root(),
            "testing_run_roots": locations.testing_run_roots(),
            "legacy_receipts": args.legacy_receipts,
            "archive_root": locations.archive_root(),
        }
    )
    listing = Listing(
        buckets=buckets,
        local=local,
        provider_listed=provider_listed,
        contact=_journal_operations(journal_path),
    )
    open_tickets = (
        None
        if args.open_tickets is None
        else frozenset(
            token.strip() for token in args.open_tickets.split(",") if token.strip()
        )
    )
    report = run_inventory(
        args.ledger, listing, update=args.update, open_tickets=open_tickets
    )
    report["receipt_root"] = str(receipt_root)
    report["journal"] = str(journal_path)
    (receipt_root / "inventory.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _print_summary(report)
    return 0


def _print_summary(report: dict[str, Any]) -> None:
    out = sys.stdout
    out.write(f"receipt: {report['receipt_root']}\n")
    out.write(
        f"provider listed: {report['provider_listed']}; contact: {report['contact']}\n"
    )
    out.write(
        f"buckets listed: {len(report['buckets'])}; local: {len(report['local'])}\n"
    )
    for title in ("unledgered", "missing", "removed_but_present"):
        items = report[title]
        out.write(f"{title}: {len(items)}\n")
        for item in items:
            out.write(f"  - {item['kind']} {item['location_key']}:{item['identity']}\n")
    out.write(f"deprecation candidates: {len(report['candidates'])}\n")
    for item in report["candidates"]:
        out.write(
            f"  - {item['kind']} {item['location_key']}:{item['identity']}: {item['reason']}\n"
        )
    out.write(f"warnings: {len(report['warnings'])}\n")
    for item in report["warnings"]:
        out.write(
            f"  - {item['kind']} {item['location_key']}:{item['identity']}: {item['code']} ({item['message']})\n"
        )
    out.write(f"filled cells: {len(report['filled'])}\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "add":
            rows = add_run(
                args.ledger,
                args.run_root,
                ticket=args.ticket,
                run_id=args.run_id,
                source_commit=args.source_commit,
                receipts=tuple(args.receipt),
            )
            for row in rows:
                sys.stdout.write(
                    f"added {row.kind} {row.location_key}:{row.identity} ({row.run_id})\n"
                )
            return 0
        if args.command == "archive":
            moved = archive_run_roots(
                args.ledger,
                args.run_roots or locations.run_roots(),
                args.archive_root or locations.archive_root(),
                batch=args.batch,
                approved=read_approved(args.approved),
                ticket=args.ticket,
                authority=args.authority,
                dry_run=args.dry_run,
            )
            verb = "would archive" if args.dry_run else "archived"
            for item in moved:
                sys.stdout.write(
                    f"{verb} {item.identity} -> {item.archived} ({item.ledger})\n"
                )
            return 0
        if args.command == "restore":
            restore_run_root(
                args.ledger,
                args.run_roots or locations.run_roots(),
                args.archive_root or locations.archive_root(),
                batch=args.batch,
                identity=args.identity,
                ticket=args.ticket,
            )
            sys.stdout.write(f"restored {args.identity} -> run_roots\n")
            return 0
        if args.command == "next-run-id":
            return _next_run_id(args)
        return _run_inventory(args)
    except (AddError, ArchiveError, LedgerError) as error:
        sys.stderr.write(f"run_ledger: {error}\n")
        return 1
