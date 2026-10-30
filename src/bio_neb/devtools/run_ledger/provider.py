"""The provider side of ``inventory``: bucket and object listings, nothing else.

Every call goes through the product's contact observer so the receipt journals it.
The allowed contact set is list operations only, asserted by test and checked here
before each call. There is no other operation in this module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bio_neb.parabricks.execution_contact_accounting import ExecutionContactJournal
from bio_neb.providers.nebius_common import (
    observe_execution_contact_finish,
    observe_execution_contact_start,
)

from .inventory import ListedBucket, ListedObject, effective_storage_class

ALLOWED_CONTACT_OPERATIONS: frozenset[str] = frozenset(
    {"nebius.version", "storage.bucket.list", "s3.list-objects-v2"}
)
_COMMON_CLI_ARGUMENTS = (
    "--retries",
    "1",
    "--no-check-update",
    "--no-progress",
    "--format",
    "json",
)


class ProviderContactRefused(RuntimeError):
    """An operation outside the list-only allowlist was about to be made."""


def assert_allowed(operation: str) -> None:
    if operation not in ALLOWED_CONTACT_OPERATIONS:
        raise ProviderContactRefused(operation)


def _nested(value: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _list_bucket_objects(
    client: Any, bucket: str, journal: ExecutionContactJournal | None
) -> tuple[ListedObject, ...]:
    objects: list[ListedObject] = []
    token: str | None = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket}
        if token is not None:
            request["ContinuationToken"] = token
        assert_allowed("s3.list-objects-v2")
        unit = observe_execution_contact_start(
            journal,
            category="storage",
            operation="s3.list-objects-v2",
            executable_identity="s3-sdk",
        )
        try:
            page = client.list_objects_v2(**request)
        except BaseException:
            observe_execution_contact_finish(journal, unit, disposition="failed")
            raise
        observe_execution_contact_finish(
            journal,
            unit,
            disposition="succeeded",
            exit_status=0,
            parse_classification="response_received",
        )
        for item in page.get("Contents", ()) or ():
            objects.append(
                ListedObject(
                    key=str(item["Key"]),
                    size=int(item["Size"]),
                    etag=str(item.get("ETag", "")).strip('"'),
                    checksum="",
                    storage_class=str(item.get("StorageClass", "")).casefold(),
                )
            )
        if not page.get("IsTruncated", False):
            break
        token = page.get("NextContinuationToken")
        if not isinstance(token, str) or not token:
            raise ValueError(
                f"{bucket}: truncated listing without a continuation token"
            )
    return tuple(sorted(objects, key=lambda item: item.key))


def list_buckets(
    runner: Any,
    client_factory: Callable[[str, str, Any], Any],
    *,
    credential_resolver: Callable[[], Any],
    project_id: str,
    profile: str,
    journal: ExecutionContactJournal | None,
    prefix: str = "bio-neb-",
    list_objects: bool = True,
) -> tuple[ListedBucket, ...]:
    """List the project's ``prefix`` buckets, their default class and their objects."""

    assert_allowed("storage.bucket.list")
    payload = runner.json(
        "storage",
        "bucket",
        "list",
        "--parent-id",
        project_id,
        "--all",
        "--profile",
        profile,
        *_COMMON_CLI_ARGUMENTS,
    )
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("bucket listing has no items list")
    buckets: list[ListedBucket] = []
    for item in items:
        name = _nested(item, "metadata", "name")
        if not isinstance(name, str) or not name.startswith(prefix):
            continue
        domain = str(_nested(item, "status", "domain_name") or "")
        region = str(_nested(item, "status", "region") or "")
        storage_class = str(
            _nested(item, "spec", "default_storage_class") or ""
        ).casefold()
        objects: tuple[ListedObject, ...] | None = None
        if list_objects and domain and region:
            client = client_factory(f"https://{domain}", region, credential_resolver())
            objects = _list_bucket_objects(client, name, journal)
        buckets.append(
            ListedBucket(
                name=name,
                bucket_id=str(_nested(item, "metadata", "id") or ""),
                storage_class=effective_storage_class(storage_class, objects),
                objects=objects,
                region=region,
                domain=domain,
            )
        )
    return tuple(sorted(buckets, key=lambda bucket: bucket.name))
