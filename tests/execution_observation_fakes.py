"""Canonical offline worker observation fixtures shared by tests."""

from __future__ import annotations

import hashlib
import json

from bio_neb.journey import canonical_json
from bio_neb.parabricks.execution_observations import (
    UploadCompleted,
    UploadStarted,
    WorkerReceiptObserved,
)


def with_receipt_schema(observations, schema):
    """Change only receipt schema meaning, rebinding its byte-level envelopes."""
    observed = next(
        item for item in observations if isinstance(item, WorkerReceiptObserved)
    )
    receipt = {**observed.receipt, "schema_version": schema}
    content = canonical_json(receipt).encode()
    identity = {
        "byte_count": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    return [
        item.model_copy(
            update={
                "receipt": receipt,
                "receipt_object": item.receipt_object.model_copy(update=identity),
            }
        )
        if isinstance(item, WorkerReceiptObserved)
        else item.model_copy(update=identity)
        if isinstance(item, (UploadStarted, UploadCompleted))
        and item.role == "worker_receipt"
        else item
        for item in observations
    ]


def worker_marker(
    name: str,
    payload: dict[str, object],
    second: int,
    *,
    timestamp_prefix: str = "2026-08-10T02:50:",
    timezone: str = "+03:00",
) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return f"[{timestamp_prefix}{second:02d}{timezone}] {name} {encoded}"


_marker = worker_marker


def _successful_log(
    *,
    request_byte_count: int = 123,
    request_sha256: str = "a" * 64,
) -> str:
    digest = "a" * 64
    lines = [
        _marker(
            "BIO_NEB_PARABRICKS_WORKER_START",
            {
                "interpreter": "/opt/bio-neb/venv/bin/python",
                "interpreter_resolved": "/usr/bin/python3.10",
                "request_schema": "bio-neb.parabricks-worker-request/7",
            },
            1,
        ),
        _marker(
            "BIO_NEB_WORKER_REQUEST_ACCEPTED",
            {
                "byte_count": request_byte_count,
                "request_sha256": request_sha256,
                "worker_sha256": digest,
            },
            2,
        ),
        _marker(
            "BIO_NEB_WORKER_TOOLS",
            {
                "tools": {
                    "pbrun": {
                        "path": "/usr/local/parabricks/pbrun",
                        "version": "pbrun: 4.7.0-1",
                    },
                    "python": {
                        "path": "/opt/bio-neb/venv/bin/python",
                        "resolved_path": "/usr/bin/python3.10",
                        "version": "3.10.12",
                    },
                }
            },
            3,
        ),
        _marker(
            "BIO_NEB_WORKER_TOOL_START",
            {
                "command_id": "pbrun-germline",
                "argv": ["/usr/local/parabricks/pbrun", "germline"],
                "shell": False,
            },
            4,
        ),
        _marker(
            "BIO_NEB_WORKER_TOOL_EXIT",
            {"command_id": "pbrun-germline", "returncode": 0},
            5,
        ),
        _marker(
            "BIO_NEB_WORKER_UPLOAD_START",
            {
                "role": "aligned_bam",
                "bucket": "run-bucket",
                "object_key": "outputs/aligned.bam",
                "byte_count": 100,
                "sha256": digest,
                "multipart_chunk_bytes": 8388608,
            },
            6,
        ),
        _marker(
            "BIO_NEB_WORKER_UPLOAD_COMPLETE",
            {
                "role": "aligned_bam",
                "bucket": "run-bucket",
                "object_key": "outputs/aligned.bam",
                "byte_count": 100,
                "sha256": digest,
            },
            7,
        ),
        _marker(
            "BIO_NEB_WORKER_UPLOAD_START",
            {
                "role": "VCF",
                "bucket": "run-bucket",
                "object_key": "outputs/variants.vcf",
                "byte_count": 25,
                "sha256": digest,
                "multipart_chunk_bytes": 8388608,
            },
            8,
        ),
        _marker(
            "BIO_NEB_WORKER_UPLOAD_COMPLETE",
            {
                "role": "VCF",
                "bucket": "run-bucket",
                "object_key": "outputs/variants.vcf",
                "byte_count": 25,
                "sha256": digest,
            },
            9,
        ),
        _marker(
            "BIO_NEB_WORKER_UPLOAD_START",
            {
                "role": "worker_receipt",
                "bucket": "run-bucket",
                "object_key": "receipts/worker.json",
                "byte_count": 50,
                "sha256": digest,
                "multipart_chunk_bytes": 8388608,
            },
            10,
        ),
        _marker(
            "BIO_NEB_WORKER_UPLOAD_COMPLETE",
            {
                "role": "worker_receipt",
                "bucket": "run-bucket",
                "object_key": "receipts/worker.json",
                "byte_count": 50,
                "sha256": digest,
            },
            11,
        ),
        _marker(
            "BIO_NEB_WORKER_RECEIPT",
            {
                "receipt": {
                    "authority": "non-authoritative-worker-observation",
                    "operation": "consume_prepared_reference",
                    "request_id": "request-1",
                    "tools": {
                        "pbrun": {
                            "path": "/usr/local/parabricks/pbrun",
                            "version": "pbrun: 4.7.0-1",
                        },
                        "python": {
                            "path": "/opt/bio-neb/venv/bin/python",
                            "resolved_path": "/usr/bin/python3.10",
                            "version": "3.10.12",
                        },
                    },
                    "uploads": [
                        {
                            "role": "aligned_bam",
                            "bucket": "run-bucket",
                            "object_key": "outputs/aligned.bam",
                            "byte_count": 100,
                            "sha256": digest,
                        },
                        {
                            "role": "VCF",
                            "bucket": "run-bucket",
                            "object_key": "outputs/variants.vcf",
                            "byte_count": 25,
                            "sha256": digest,
                        },
                    ],
                },
                "receipt_object": {
                    "role": "worker_receipt",
                    "bucket": "run-bucket",
                    "object_key": "receipts/worker.json",
                    "byte_count": 50,
                    "sha256": digest,
                },
            },
            12,
        ),
    ]
    return "\n".join(lines) + "\n"


def durable_receipt(observations):
    """Derive current synthetic producer facts; old marker vectors remain historical.

    Facts are supplied by scripted effects, not observed execution of the worker.
    CPU callers add their newly required input/support facts explicitly.
    """
    receipt = next(
        item.receipt for item in observations if item.kind == "worker_receipt"
    )
    validation = [
        item.model_dump(mode="json", exclude_none=True)
        for item in observations
        if item.kind
        not in {
            "provider_create",
            "fetched_envelope",
            "provider_state",
            "provider_log",
            "worker_receipt",
            "heartbeat",
            "terminal_snapshot",
        }
        and getattr(item, "role", None) != "worker_receipt"
    ]
    return {
        **receipt,
        "schema_version": "bio-neb.parabricks-worker-receipt/5",
        "validation": validation,
    }
