"""``add``: derive ledger rows from one finished run root, offline.

Reads current execution results and contact reviews through their public readers,
with a fallback for historical harness roots. Operator metadata supplies source and
exit status. Bucket size, hash and class stay unknown until a separate inventory.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ledger import (
    LedgerRow,
    archive_batch,
    derive_run_id,
    directory_inventory,
    name_tokens,
    read_ledger,
    resource_key,
    write_ledger,
)


class AddError(ValueError):
    """The run root cannot yield rows, or a row would duplicate an identity."""


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AddError(f"{path} is not readable JSON: {error}") from error
    if not isinstance(value, dict):
        raise AddError(f"{path} must contain a JSON object")
    return value


def _journal_operations(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.is_file():
        return counts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("row_kind") == "terminal":
                operation = row.get("operation")
                if isinstance(operation, str):
                    counts[operation] += 1
    return counts


def _date_from(finished: dict[str, Any] | None, run_root: Path) -> str:
    if finished is not None:
        for key in ("finished_at", "started_at"):
            value = finished.get(key)
            if isinstance(value, str) and len(value) >= 10:
                return value[:10]
    for token in run_root.name.split("-"):
        if len(token) >= 8 and token[:8].isdigit():
            return f"{token[:4]}-{token[4:6]}-{token[6:8]}"
    raise AddError(f"{run_root}: no finished.json date and no date token in the name")


def _notes(parts: Iterable[tuple[str, str]]) -> str:
    return "; ".join(f"{key}={value}" for key, value in parts if value)


def _receipt_rows(
    receipts: Iterable[tuple[str, str]],
    *,
    ticket: str,
    run_id: str,
    date: str,
    source_commit: str,
) -> list[LedgerRow]:
    return [
        LedgerRow(
            run_id=run_id,
            ticket=ticket,
            date=date,
            source_commit=source_commit,
            kind="receipt",
            identity=receipt,
            location_key=location_key,
        )
        for location_key, receipt in receipts
    ]


@dataclass(frozen=True)
class _RunEvidence:
    summary: dict[str, str]
    bucket_name: str = ""
    bucket_notes: dict[str, str] | None = None


def _selected_storage(approved: dict) -> tuple[str, str]:
    launch = approved.get("launch")
    storage = launch.get("object_storage") if isinstance(launch, dict) else None
    if not isinstance(storage, dict):
        raise AddError("approved input lacks launch.object_storage")
    bucket, scope = storage.get("bucket"), launch.get("object_scope")
    if not isinstance(bucket, str) or not bucket or not isinstance(scope, str):
        raise AddError("approved input lacks bucket name or object scope")
    return bucket, scope


def _current_execution(run_root: Path) -> _RunEvidence | None:
    """Discover a single execution, leaving result and contact meaning with APP readers."""
    from bio_neb.parabricks.execution_contact_accounting import (
        review_execution_contacts,
    )
    from bio_neb.parabricks.result import (
        ParabricksResultUnavailable,
        read_parabricks_result,
    )

    approved = _load_json(run_root / "approved-input.json")
    candidates = set()
    if approved is not None:
        _selected_storage(approved)
        locator = approved.get("journey_file")
        if not isinstance(locator, str) or not Path(locator).is_file():
            raise AddError("approved-input.json must name an existing Journey")
        candidates.add(Path(locator).resolve())
    for pattern in (
        "journey.jsonl",
        "*/journey.jsonl",
        "app/*/journey.jsonl",
        "run/*/journey.jsonl",
        "runs/*/journey.jsonl",
    ):
        candidates.update(p.resolve() for p in run_root.glob(pattern))
    executions = []
    for path in sorted(candidates):
        if not path.is_relative_to(run_root.resolve()):
            raise AddError(f"Journey is outside the selected run root: {path}")
        try:
            result = read_parabricks_result(str(path))
        except ParabricksResultUnavailable:
            continue
        except (OSError, ValueError) as error:
            raise AddError(f"cannot read execution Journey {path}: {error}") from error
        executions.append((path, result))
    if len(executions) > 1:
        raise AddError("ambiguous run root: multiple execution Journeys")
    if not executions:
        if approved is None:
            return None
        bucket, scope = _selected_storage(approved)
        return _RunEvidence(
            {"outcome": "unknown", "contact_review": "unavailable"},
            bucket,
            {
                "object_scope": scope,
                "resource_evidence": "selected_only",
            },
        )
    path, result = executions[0]
    bucket = result.storage.bucket
    scope = result.object_scope
    try:
        public = review_execution_contacts(str(path))
    except (OSError, ValueError) as error:
        raise AddError(f"cannot review execution contacts {path}: {error}") from error
    review = public.review
    summary = {
        "outcome": result.disposition,
        "status": result.status,
        "journey": str(path),
        "contact_review": public.status,
        "journals": " ".join(p.journal_locator for p in review.retained_passes)
        or public.journal_locator,
        "passes": " ".join(str(p.execution_pass) for p in review.retained_passes),
        "contact_discrepancies": " ".join(review.discrepancies),
        "observed_ops": " ".join(
            f"{operation}={count}"
            for operation, count in sorted(
                Counter(u.operation for u in review.units).items()
            )
        ),
        "logical_attempts": str(len(review.logical_attempts)),
        "logical_results": str(len(review.logical_results)),
    }
    for name in (
        "credential_access",
        "model_contact",
        "source_contact",
        "storage_contact",
        "provider_contact",
        "network_contact",
        "fresh_bucket",
        "cpu_preparation_create",
        "h100_create",
        "post_execution_vcf_query",
    ):
        metric = getattr(review.ledger, name)
        summary[name] = str(metric.count) if metric.count is not None else "unknown"
        if metric.count is None:
            summary[name + "_observed"] = str(metric.observed_count)
            summary[name + "_reason"] = metric.reason
            ids = metric.observed_unit_ids
        else:
            ids = metric.unit_ids
        if name in ("fresh_bucket", "cpu_preparation_create", "h100_create"):
            summary[name + "_ids"] = " ".join(ids)
    bucket_ids = sorted(
        {
            identity
            for attempt in (*review.logical_attempts, *review.logical_results)
            if attempt.resource_kind == "bucket" and attempt.resource_intent == bucket
            for identity in attempt.provider_ids
        }
    )
    return _RunEvidence(
        summary,
        bucket,
        {
            "object_scope": scope,
            "bucket_id": " ".join(bucket_ids),
            "resource_evidence": "observed" if bucket_ids else "selected_only",
        },
    )


def _legacy_execution(run_root: Path) -> _RunEvidence | None:
    result = _load_json(run_root / "harness" / "result.json")
    launch: dict[str, Any] | None = None
    for candidate in (
        run_root / "harness" / "input.json",
        run_root / "evidence" / "input.json",
    ):
        document = _load_json(candidate)
        if document is not None and isinstance(document.get("launch"), dict):
            launch = document["launch"]
            break
    if result is None and launch is None:
        return None
    outcome = str(result.get("outcome") or "") if result else ""
    operations = _journal_operations(run_root / "harness" / "journal.jsonl")
    bucket = result.get("run_bucket", {}) if result else {}
    if not isinstance(bucket, dict):
        bucket = {}
    return _RunEvidence(
        summary={
            "outcome": outcome,
            "ops": " ".join(
                f"{name}={count}" for name, count in sorted(operations.items())
            ),
        },
        bucket_name=str(bucket.get("name") or (launch or {}).get("run_bucket") or ""),
        bucket_notes={
            "bucket_id": str(bucket.get("bucket_id") or ""),
            "object_scope": str((launch or {}).get("object_scope") or ""),
            "created_by": "run " + run_root.name,
        },
    )


def derive_rows(
    run_root: Path | None,
    *,
    ticket: str,
    run_id: str | None = None,
    source_commit: str | None = None,
    receipts: Iterable[tuple[str, str]] = (),
) -> tuple[LedgerRow, ...]:
    """Rows for one run root: ``run_root`` (+ ``bucket``, ``checkout``, ``receipt``).

    Without a run root (an offline ticket that left only testing receipts) the receipt
    rows alone are derived; each row's run id, date and commit come from its own
    receipt name unless given.
    """

    receipts = tuple(receipts)
    if run_root is None:
        if not receipts:
            raise AddError("a run root or at least one receipt is required")
        commits = sorted({name_tokens(name)[1] for _, name in receipts})
        if (run_id or source_commit) and len(commits) > 1:
            raise AddError(
                "--run-id or --source-commit would apply to receipts of several "
                f"commits ({', '.join(commits)}); add each commit's receipts separately"
            )
        rows = []
        for receipt in receipts:
            name = receipt[1]
            date_token, sha_token = name_tokens(name)
            if not date_token:
                raise AddError(
                    f"{name}: no date token in the receipt name; pass --run-root"
                )
            rows.extend(
                _receipt_rows(
                    (receipt,),
                    ticket=ticket,
                    run_id=run_id or derive_run_id(name),
                    date=f"{date_token[:4]}-{date_token[4:6]}-{date_token[6:8]}",
                    source_commit=source_commit or sha_token,
                )
            )
        return tuple(rows)
    run_root = Path(run_root)
    if not run_root.is_dir():
        raise AddError(f"{run_root} is not a directory")
    finished = _load_json(run_root / "operator" / "finished.json")
    current = _current_execution(run_root)
    evidence = current or _legacy_execution(run_root)
    if finished is None and evidence is None:
        raise AddError(f"{run_root}: no operator or execution evidence")
    evidence = evidence or _RunEvidence({})
    identity = run_id or derive_run_id(run_root.name)
    date = _date_from(finished, run_root)
    if not source_commit:
        source_commit = ""
        if finished is not None and isinstance(finished.get("candidate_sha"), str):
            source_commit = finished["candidate_sha"]
    size, digest = directory_inventory(run_root)
    summary = dict(evidence.summary)
    if finished is not None and finished.get("exit_code") is not None:
        summary["exit"] = str(finished["exit_code"])
    rows = [
        LedgerRow(
            run_id=identity,
            ticket=ticket,
            date=date,
            source_commit=source_commit,
            kind="run_root",
            identity=run_root.name,
            location_key="run_roots",
            size_bytes=str(size),
            inventory_sha256=digest,
            notes=_notes(summary.items()),
        )
    ]
    if evidence.bucket_name:
        rows.append(
            LedgerRow(
                run_id=identity,
                ticket=ticket,
                date=date,
                source_commit=source_commit,
                kind="bucket",
                identity=evidence.bucket_name,
                notes=_notes(
                    {
                        **(evidence.bucket_notes or {}),
                        "outcome": summary.get("outcome", ""),
                    }.items()
                ),
            )
        )
    if source_commit:
        rows.append(
            LedgerRow(
                run_id=identity,
                ticket=ticket,
                date=date,
                source_commit=source_commit,
                kind="checkout",
                identity=source_commit,
                location_key="checkouts_root",
                notes="operator --checkout of run " + run_root.name,
            )
        )
    rows.extend(
        _receipt_rows(
            receipts,
            ticket=ticket,
            run_id=identity,
            date=date,
            source_commit=source_commit,
        )
    )
    return tuple(rows)


def add_run(
    ledger_path: Path,
    run_root: Path | None,
    *,
    ticket: str,
    run_id: str | None = None,
    source_commit: str | None = None,
    receipts: Iterable[tuple[str, str]] = (),
) -> tuple[LedgerRow, ...]:
    """Append resources, reusing a cited checkout; refuse other duplicate keys."""

    existing = read_ledger(ledger_path)
    if run_root is not None:
        archived = [
            row.location_key
            for row in existing
            if row.kind == "run_root"
            and row.identity == Path(run_root).name
            and archive_batch(row.location_key)
        ]
        if archived:
            raise AddError(
                f"{Path(run_root).name} names a run root archived in {archived[0]}; "
                "name a fresh run with `next-run-id`"
            )
    rows = derive_rows(
        run_root,
        ticket=ticket,
        run_id=run_id,
        source_commit=source_commit,
        receipts=receipts,
    )
    known = {resource_key(row) for row in existing}
    rows = tuple(
        row for row in rows if row.kind != "checkout" or resource_key(row) not in known
    )
    duplicates = [resource_key(row) for row in rows if resource_key(row) in known]
    if duplicates:
        raise AddError(f"identity already ledgered: {duplicates!r}")
    write_ledger(ledger_path, (*existing, *rows))
    return rows
