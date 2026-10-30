"""Strict baked worker for prepared-reference production and consumption.

The worker accepts one canonical, secret-free request file.  Version 7 has
three deliberately separate operations: CPU reference preparation (from a stored
object or from the approved publisher URL, indexing the Run's reference sequences
only), GPU prepared-reference consumption, and a credential-free runtime smoke.  Worker receipts are observations only;
host verification remains authoritative.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import urlopen

import boto3
from botocore.config import Config
from pydantic import ValidationError

from bio_neb import contracts, fasta_selection
from bio_neb.fasta_selection import FastaRecordSelection
from bio_neb.parabricks import worker_exchange
from bio_neb.parabricks.worker_exchange import (
    GPU_INPUT_ROLES,
    MAX_REQUEST_BYTES,
    PREPARATION_COMMANDS,
    PREPARED_ROLES,
    RECEIPT_ADAPTER,
    RECEIPT_SCHEMA_VERSION,
    REQUEST_ADAPTER,
    REQUEST_SCHEMA_VERSION,
    canonical_bytes,
)

CHUNK_BYTES = 8 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 300
WORK_ROOT = Path("/work")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_REQUEST_KEYS = {
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}


class WorkerError(RuntimeError):
    """Bounded failure whose message excludes provider responses and secrets."""


# Dependency pins preserve the executable-file SHA contract. Updating any
# shared source requires updating its pin here and publishing a matching image.
_EXCHANGE_SHA256 = "89c4a8cdf0765fd5a641d6e75045d240f9e55cfdd1bd009fa0e299ea31cb9722"
_CONTRACTS_SHA256 = "5b150781a9309ed7da310bc14d05ef65152438d71d6f86203758343d3f6e5f09"
_SELECTION_SHA256 = "21a43680514d1dbce941c91f0913b84be27cc5bfefa119328b0cc270a404ff32"


def worker_source_bytes() -> bytes:
    for module, expected in (
        (worker_exchange, _EXCHANGE_SHA256),
        (contracts, _CONTRACTS_SHA256),
        (fasta_selection, _SELECTION_SHA256),
    ):
        try:
            actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        except OSError as error:
            raise WorkerError("worker exchange dependency unavailable") from error
        if actual != expected:
            raise WorkerError("worker exchange dependency changed")
    return Path(__file__).read_bytes()


# These already-established facts are retained before disposable output is attempted.
# A context-local list keeps direct execute callers and sequential jobs isolated.
_VALIDATION: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "worker_validation", default=None
)
_VALIDATION_KINDS = {
    "BIO_NEB_PARABRICKS_WORKER_START": "worker_startup",
    "BIO_NEB_WORKER_REQUEST_ACCEPTED": "worker_request",
    "BIO_NEB_WORKER_TOOLS": "worker_tools",
    "BIO_NEB_WORKER_INPUT_VERIFIED": "worker_input_verified",
    "BIO_NEB_WORKER_SOURCE_ACQUIRED": "worker_source_acquired",
    "BIO_NEB_WORKER_REFERENCE_SUPPORT": "worker_reference_support",
    "BIO_NEB_WORKER_PREPARED_REFERENCE_VERIFIED": "worker_prepared_reference_verified",
    "BIO_NEB_WORKER_TOOL_START": "tool_started",
    "BIO_NEB_WORKER_TOOL_EXIT": "tool_exited",
    "BIO_NEB_WORKER_UPLOAD_START": "upload_started",
    "BIO_NEB_WORKER_UPLOAD_COMPLETE": "upload_completed",
}


def _emit(marker: str, **fields: object) -> None:
    validation = _VALIDATION.get()
    kind = _VALIDATION_KINDS.get(marker)
    if validation is not None and kind is not None:
        fact = {"kind": kind, **fields}
        if kind == "worker_tools":
            fact["tools"] = [
                {"name": name, **tool} for name, tool in sorted(fields["tools"].items())
            ]
        validation.append(json.loads(canonical_bytes(fact)))
    try:
        print(marker + " " + canonical_bytes(fields).decode(), flush=True)
    except (OSError, ValueError):
        # A closed diagnostic stream cannot revoke established producer facts.
        pass


def _announce_request(payload: bytes, tools: Mapping[str, Mapping[str, str]]) -> None:
    _emit(
        "BIO_NEB_PARABRICKS_WORKER_START",
        interpreter=str(sys.executable),
        interpreter_resolved=str(Path(sys.executable).resolve()),
        request_schema=REQUEST_SCHEMA_VERSION,
    )
    _emit(
        "BIO_NEB_WORKER_REQUEST_ACCEPTED",
        byte_count=len(payload),
        request_sha256=hashlib.sha256(payload).hexdigest(),
        worker_sha256=hashlib.sha256(worker_source_bytes()).hexdigest(),
    )
    _emit("BIO_NEB_WORKER_TOOLS", tools=tools)


def _line(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(c in value for c in "\r\n\x00")
    ):
        raise WorkerError(f"{label} is invalid")
    return value


def _safe_key(value: object, label: str, *, prefix: bool = False) -> str:
    key = _line(value, label)
    if key.startswith("/") or "\\" in key or "//" in key or key.endswith("/") != prefix:
        raise WorkerError(f"{label} is invalid")
    key_path = key[:-1] if prefix else key
    path = PurePosixPath(key_path)
    if str(path) != key_path or any(part in {"", ".", ".."} for part in path.parts):
        raise WorkerError(f"{label} is invalid")
    return key


def _local_target(value: object, label: str) -> str:
    raw = _line(value, label)
    path = Path(raw)
    root = WORK_ROOT.resolve()
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise WorkerError(f"{label} escaped worker scratch") from error
    if not path.is_absolute() or str(resolved) != raw or resolved == root:
        raise WorkerError(f"{label} is not one canonical absolute path")
    return raw


def _reject_secret_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_REQUEST_KEYS:
                raise WorkerError("worker request contained credential material")
            _reject_secret_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_secret_fields(child)


def _validate_exact_object(item: dict[str, Any], label: str) -> dict[str, Any]:
    item["role"] = _line(item["role"], f"{label} role")
    if item.get("bucket") is not None:
        item["bucket"] = _line(item["bucket"], f"{label} bucket")
    item["object_key"] = _safe_key(item["object_key"], f"{label} object key")
    item["local_target"] = _local_target(item["local_target"], f"{label} local target")
    if _SHA256.fullmatch(item["sha256"]) is None:
        raise WorkerError(f"{label} SHA-256 is invalid")
    return item


def _validate_target(item: dict[str, Any], label: str, prefix: str) -> dict[str, Any]:
    item["role"] = _line(item["role"], f"{label} role")
    item["object_key"] = _safe_key(item["object_key"], f"{label} object key")
    item["local_target"] = _local_target(item["local_target"], f"{label} local target")
    if not item["object_key"].startswith(prefix):
        raise WorkerError(f"{label} escaped output prefix")
    return item


def _prepared_paths(fasta: str) -> dict[str, str]:
    return {
        "genomic_fasta": fasta,
        "fasta_fai": fasta + ".fai",
        "reference_dict": str(Path(fasta).with_suffix(".dict")),
        **{
            f"bwa_{ext}": fasta + f".{ext}"
            for ext in ("amb", "ann", "bwt", "pac", "sa")
        },
    }


def _option_values(argv: tuple[str, ...], option: str, count: int) -> tuple[str, ...]:
    if argv.count(option) != 1:
        raise WorkerError(f"approved Parabricks {option} binding changed")
    index = argv.index(option)
    values = argv[index + 1 : index + 1 + count]
    if len(values) != count or any(value.startswith("--") for value in values):
        raise WorkerError(f"approved Parabricks {option} binding changed")
    return values


def _validate_common(request: dict[str, Any]) -> None:
    for field in ("request_id", "bucket", "endpoint_url", "region"):
        request[field] = _line(request[field], field)
    if not request["endpoint_url"].startswith("https://"):
        raise WorkerError("endpoint_url is invalid")
    request["output_prefix"] = _safe_key(
        request["output_prefix"], "output_prefix", prefix=True
    )


def _validate_download(item: dict[str, Any]) -> dict[str, Any]:
    item["url"] = _line(item["url"], "source url")
    if not item["url"].startswith("https://"):
        raise WorkerError("source url is invalid")
    item["object_key"] = _safe_key(item["object_key"], "source object_key")
    item["local_target"] = _local_target(item["local_target"], "source local target")
    return item


def _validate_prepare(request: dict[str, Any]) -> dict[str, Any]:
    _validate_common(request)
    reference = request["reference"]
    if request["download"] is not None:
        source = _validate_download(request["download"])
        plain = reference["plain_fasta"]
    else:
        source = _validate_exact_object(request["inputs"][0], "input[0]")
        if source["role"] != "reference_source":
            raise WorkerError("CPU preparation input inventory changed")
        plain = reference["accepted_plain_fasta"]
        if _SHA256.fullmatch(plain["sha256"]) is None:
            raise WorkerError("accepted plain FASTA SHA-256 is invalid")
        request["inputs"] = [source]
    plain["local_target"] = _local_target(
        plain["local_target"], "plain FASTA local target"
    )
    if source["local_target"] == plain["local_target"]:
        raise WorkerError("compressed and plain reference targets collided")
    if tuple(reference["commands"]) != PREPARATION_COMMANDS:
        raise WorkerError("fixed reference preparation commands changed")
    excluded = reference["excluded_sequences"]
    if len(set(excluded)) != len(excluded) or any(
        not name or name.split() != [name] for name in excluded
    ):
        raise WorkerError("excluded reference sequence names are invalid")
    outputs = [
        _validate_target(item, f"prepared_output[{i}]", request["output_prefix"])
        for i, item in enumerate(request["prepared_outputs"])
    ]
    if tuple(item["role"] for item in outputs) != PREPARED_ROLES:
        raise WorkerError("prepared reference role inventory changed")
    expected_paths = _prepared_paths(plain["local_target"])
    if any(item["local_target"] != expected_paths[item["role"]] for item in outputs):
        raise WorkerError("prepared reference target inventory changed")
    receipt = _validate_target(
        request["receipt_output"], "receipt output", request["output_prefix"]
    )
    if receipt["role"] != "worker_receipt":
        raise WorkerError("worker receipt role changed")
    all_targets = [
        source["local_target"],
        *(item["local_target"] for item in outputs),
        receipt["local_target"],
    ]
    all_keys = [*(item["object_key"] for item in outputs), receipt["object_key"]]
    if request["download"] is not None:
        all_keys.append(source["object_key"])
    if len(all_targets) != len(set(all_targets)) or len(all_keys) != len(set(all_keys)):
        raise WorkerError("CPU preparation identities collided")
    request["prepared_outputs"], request["receipt_output"] = outputs, receipt
    return request


def _validate_consume(request: dict[str, Any]) -> dict[str, Any]:
    _validate_common(request)
    inputs = [
        _validate_exact_object(item, f"input[{i}]")
        for i, item in enumerate(request["inputs"])
    ]
    if tuple(item["role"] for item in inputs) != GPU_INPUT_ROLES:
        raise WorkerError("GPU prepared input role inventory changed")
    by_role = {item["role"]: item for item in inputs}
    fasta = by_role["genomic_fasta"]["local_target"]
    expected_paths = _prepared_paths(fasta)
    if any(
        by_role[role]["local_target"] != path for role, path in expected_paths.items()
    ):
        raise WorkerError("GPU prepared target inventory changed")
    outputs = [
        _validate_target(item, f"output[{i}]", request["output_prefix"])
        for i, item in enumerate(request["outputs"])
    ]
    if [item["role"] for item in outputs] != ["aligned_bam", "VCF"]:
        raise WorkerError("worker must upload BAM before VCF")
    receipt = _validate_target(
        request["receipt_output"], "receipt output", request["output_prefix"]
    )
    if receipt["role"] != "worker_receipt":
        raise WorkerError("worker receipt role changed")
    argv = tuple(
        _line(arg, "Parabricks argument") for arg in request["parabricks_argv"]
    )
    resources = request["resources"]
    if argv[:2] != ("pbrun", "germline"):
        raise WorkerError("worker requires direct pbrun germline")
    bindings = {
        "--ref": (fasta,),
        "--in-fq": (
            by_role["read_1"]["local_target"],
            by_role["read_2"]["local_target"],
        ),
        "--out-bam": (outputs[0]["local_target"],),
        "--out-variants": (outputs[1]["local_target"],),
        "--num-gpus": (str(resources["gpu_count"]),),
    }
    if any(
        _option_values(argv, option, len(values)) != values
        for option, values in bindings.items()
    ):
        raise WorkerError("approved Parabricks binding changed")
    targets = [
        *(item["local_target"] for item in inputs),
        *(item["local_target"] for item in outputs),
        receipt["local_target"],
    ]
    keys = [
        *(item["object_key"] for item in inputs),
        *(item["object_key"] for item in outputs),
        receipt["object_key"],
    ]
    if len(targets) != len(set(targets)) or len(keys) != len(set(keys)):
        raise WorkerError("GPU consumption identities collided")
    (
        request["inputs"],
        request["outputs"],
        request["receipt_output"],
        request["parabricks_argv"],
    ) = inputs, outputs, receipt, argv
    return request


def validate_request(payload: bytes) -> dict[str, Any]:
    worker_source_bytes()
    if not payload or len(payload) > MAX_REQUEST_BYTES:
        raise WorkerError("worker request exceeded its injected-file byte bound")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerError("worker request is not JSON") from error
    _reject_secret_fields(raw)
    if not isinstance(raw, dict) or raw.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise WorkerError("worker request schema changed")
    if canonical_bytes(raw) != payload:
        raise WorkerError("worker request is not canonical JSON")
    try:
        request = REQUEST_ADAPTER.validate_python(raw, strict=True).model_dump(
            mode="json"
        )
    except ValidationError as error:
        raise WorkerError("worker request type contract changed") from error
    if request["operation"] == "runtime_smoke":
        request["request_id"] = _line(request["request_id"], "request_id")
        return request
    if request["operation"] == "prepare_reference":
        return _validate_prepare(request)
    return _validate_consume(request)


def load_request(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file():
        raise WorkerError("worker requires one absolute request-file path")
    with path.open("rb") as stream:
        return validate_request(stream.read(MAX_REQUEST_BYTES + 1))


def _file_digest(path: Path) -> tuple[int, str]:
    digest, byte_count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def _download_verified(client: Any, *, bucket: str, item: Mapping[str, Any]) -> str:
    """Download one exact stored object; return its md5 for the source receipt."""
    destination = Path(item["local_target"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = client.get_object(Bucket=bucket, Key=item["object_key"])["Body"]
    digest, md5, byte_count = hashlib.sha256(), hashlib.md5(), 0
    try:
        with destination.open("xb") as stream:
            while chunk := body.read(CHUNK_BYTES):
                if not isinstance(chunk, bytes):
                    raise WorkerError("S3 input stream returned non-bytes")
                byte_count += len(chunk)
                if byte_count > item["byte_count"]:
                    raise WorkerError("S3 input exceeded its byte bound")
                digest.update(chunk)
                md5.update(chunk)
                stream.write(chunk)
    finally:
        close = getattr(body, "close", None)
        if close is not None:
            close()
    if (byte_count, digest.hexdigest()) != (item["byte_count"], item["sha256"]):
        raise WorkerError("worker input verification failed")
    _emit(
        "BIO_NEB_WORKER_INPUT_VERIFIED",
        role=item["role"],
        bucket=bucket,
        object_key=item["object_key"],
        byte_count=byte_count,
        sha256=digest.hexdigest(),
    )
    return md5.hexdigest()


def _fetch_public_source(
    client: Any,
    request: Mapping[str, Any],
    fetch: Callable[..., Any],
) -> dict[str, Any]:
    """Fetch the approved URL once, check its md5 and size, then store the bytes."""
    item = request["download"]
    destination = Path(item["local_target"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest, md5, byte_count = hashlib.sha256(), hashlib.md5(), 0
    try:
        with (
            fetch(item["url"], timeout=FETCH_TIMEOUT_SECONDS) as body,
            destination.open("xb") as stream,
        ):
            while chunk := body.read(CHUNK_BYTES):
                # Every counted byte is hashed, so a refusal's evidence is one prefix.
                byte_count += len(chunk)
                digest.update(chunk)
                md5.update(chunk)
                if byte_count > item["byte_count"]:
                    break
                stream.write(chunk)
    except OSError as error:
        raise WorkerError("reference source download failed") from error
    if (byte_count, md5.hexdigest()) != (item["byte_count"], item["md5"]):
        _emit(
            "BIO_NEB_WORKER_SOURCE_MISMATCH",
            role=item["role"],
            url=item["url"],
            expected_byte_count=item["byte_count"],
            observed_byte_count=byte_count,
            expected_md5=item["md5"],
            observed_md5=md5.hexdigest(),
        )
        raise WorkerError("reference source identity mismatch")
    _emit(
        "BIO_NEB_WORKER_SOURCE_ACQUIRED",
        role=item["role"],
        url=item["url"],
        byte_count=byte_count,
        md5=md5.hexdigest(),
        sha256=digest.hexdigest(),
    )
    _upload_multipart_immutable(
        client,
        bucket=request["bucket"],
        item=item,
        established_identity=(byte_count, digest.hexdigest()),
    )
    return {
        "role": item["role"],
        "object_key": item["object_key"],
        "byte_count": byte_count,
        "sha256": digest.hexdigest(),
        "md5": md5.hexdigest(),
        "origin": item["url"],
    }


def _acquire_source(
    client: Any, request: Mapping[str, Any], fetch: Callable[..., Any]
) -> dict[str, Any]:
    if request["download"] is not None:
        return _fetch_public_source(client, request, fetch)
    item = request["inputs"][0]
    md5 = _download_verified(client, bucket=request["bucket"], item=item)
    return {
        **{key: item[key] for key in ("role", "object_key", "byte_count", "sha256")},
        "md5": md5,
        "origin": f"s3://{request['bucket']}/{item['object_key']}",
    }


def _plain_fasta(request: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = request["reference"]
    return reference["accepted_plain_fasta"] or reference["plain_fasta"]


def _derive_plain_reference(request: Mapping[str, Any], source: str) -> tuple[int, str]:
    """Decompress the source and keep the Run's reference sequences (its excluded
    records dropped); a stored source must reproduce its accepted identity."""
    accepted = request["reference"]["accepted_plain_fasta"]
    selection = FastaRecordSelection(request["reference"]["excluded_sequences"])
    destination = Path(_plain_fasta(request)["local_target"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest, byte_count = hashlib.sha256(), 0
    try:
        with (
            gzip.open(source, "rb") as compressed,
            destination.open("xb") as plain,
        ):
            finished = False
            while not finished:
                data = compressed.read(CHUNK_BYTES)
                finished = not data
                chunk = selection.feed(data) if data else selection.finish()
                if (
                    accepted is not None
                    and byte_count + len(chunk) > accepted["byte_count"]
                ):
                    raise WorkerError(
                        "compressed reference exceeded accepted plain byte bound"
                    )
                byte_count += len(chunk)
                digest.update(chunk)
                plain.write(chunk)
    except (OSError, EOFError) as error:
        raise WorkerError("compressed reference derivation failed") from error
    if accepted is not None and (byte_count, digest.hexdigest()) != (
        accepted["byte_count"],
        accepted["sha256"],
    ):
        raise WorkerError("accepted plain FASTA derivation changed")
    return byte_count, digest.hexdigest()


def _tool_identity(
    operation: str,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, dict[str, str]]:
    interpreter = str(sys.executable)
    identities = {
        "python": {
            "path": interpreter,
            "resolved_path": str(Path(interpreter).resolve()),
            "version": sys.version.split()[0],
        }
    }
    operation_tools = {
        "prepare_reference": (("bwa", ()), ("samtools", ("--version",))),
        "consume_prepared_reference": (("pbrun", ("--version",)),),
        "runtime_smoke": (
            ("bwa", ()),
            ("samtools", ("--version",)),
            ("pbrun", ("--version",)),
        ),
    }
    selected = operation_tools.get(operation)
    if selected is None:
        raise WorkerError("worker operation tool boundary changed")
    for name, version_args in selected:
        resolved = shutil.which(name)
        if resolved is None or not Path(resolved).is_absolute():
            raise WorkerError(f"required {name} executable is unavailable")
        path = str(Path(resolved).resolve())
        completed = run(
            (path, *version_args),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            shell=False,
            timeout=30,
        )
        lines = [
            line.strip()
            for line in (completed.stdout + b"\n" + completed.stderr)
            .decode(errors="replace")
            .splitlines()
            if line.strip()
        ]
        if name == "bwa":
            version = next((line for line in lines if line.startswith("Version:")), "")
        elif name == "pbrun":
            version = next(
                (line for line in lines if line.casefold().startswith("pbrun:")), ""
            )
        else:
            version = lines[0] if lines else ""
        if not version or (name != "bwa" and completed.returncode != 0):
            raise WorkerError(f"required {name} version proof failed")
        identities[name] = {"path": path, "version": version[:200]}
    return identities


def _select_tools(
    tools: Mapping[str, Mapping[str, str]], *required: str
) -> dict[str, Mapping[str, str]]:
    selected: dict[str, Mapping[str, str]] = {}
    for name in ("python", *required):
        identity = tools.get(name)
        if identity is None:
            raise WorkerError(f"required {name} executable is unavailable")
        selected[name] = identity
    return selected


def _run_vector(
    argv: Sequence[str],
    tools: Mapping[str, Mapping[str, str]],
    *,
    command_id: str,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> None:
    if argv[0] not in tools:
        raise WorkerError("worker command executable changed")
    actual = (tools[argv[0]]["path"], *argv[1:])
    _emit("BIO_NEB_WORKER_TOOL_START", command_id=command_id, argv=actual, shell=False)
    completed = run(
        actual, cwd=WORK_ROOT, stdin=subprocess.DEVNULL, check=False, shell=False
    )
    _emit(
        "BIO_NEB_WORKER_TOOL_EXIT",
        command_id=command_id,
        returncode=completed.returncode,
    )
    if completed.returncode != 0:
        _emit(
            "BIO_NEB_WORKER_TOOL_ERROR",
            command_id=command_id,
            returncode=completed.returncode,
        )
        raise WorkerError(f"{command_id} failed")


def _upload_multipart_immutable(
    client: Any,
    *,
    bucket: str,
    item: Mapping[str, Any],
    established_identity: tuple[int, str],
) -> dict[str, Any]:
    source = Path(item["local_target"])
    if not source.is_file():
        raise WorkerError(f"{item['role']} output is missing")
    byte_count, sha256 = established_identity
    if byte_count == 0:
        raise WorkerError(f"{item['role']} output is empty")
    _emit(
        "BIO_NEB_WORKER_UPLOAD_START",
        role=item["role"],
        bucket=bucket,
        object_key=item["object_key"],
        byte_count=byte_count,
        sha256=sha256,
        multipart_chunk_bytes=CHUNK_BYTES,
    )
    created = client.create_multipart_upload(
        Bucket=bucket, Key=item["object_key"], Metadata={"checksum-sha256": sha256}
    )
    upload_id = created.get("UploadId")
    if not isinstance(upload_id, str) or not upload_id:
        raise WorkerError("S3 multipart upload identity changed")
    parts: list[dict[str, object]] = []
    transferred_digest, transferred_bytes = hashlib.sha256(), 0
    try:
        with source.open("rb") as stream:
            part_number = 1
            while chunk := stream.read(CHUNK_BYTES):
                transferred_bytes += len(chunk)
                transferred_digest.update(chunk)
                uploaded = client.upload_part(
                    Bucket=bucket,
                    Key=item["object_key"],
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=chunk,
                )
                etag = uploaded.get("ETag")
                if not isinstance(etag, str) or not etag:
                    raise WorkerError("S3 multipart part receipt changed")
                parts.append({"ETag": etag, "PartNumber": part_number})
                part_number += 1
        if (transferred_bytes, transferred_digest.hexdigest()) != (
            byte_count,
            sha256,
        ):
            raise WorkerError(f"{item['role']} changed during upload")
        client.complete_multipart_upload(
            Bucket=bucket,
            Key=item["object_key"],
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
            IfNoneMatch="*",
        )
    except Exception:
        # Fail-stop policy retains the exact multipart residue for application
        # evidence. Recovery/cleanup is a later, separately authorized action.
        raise
    _emit(
        "BIO_NEB_WORKER_UPLOAD_COMPLETE",
        role=item["role"],
        bucket=bucket,
        object_key=item["object_key"],
        byte_count=byte_count,
        sha256=sha256,
    )
    return {
        "role": item["role"],
        "bucket": bucket,
        "object_key": item["object_key"],
        "byte_count": byte_count,
        "sha256": sha256,
    }


def _write_and_upload_receipt(
    client: Any, request: Mapping[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    target = request["receipt_output"]
    path = Path(target["local_target"])
    path.parent.mkdir(parents=True, exist_ok=True)
    # The producer and backend parser share structure, not admission authority.
    receipt = {**receipt, "validation": list(_VALIDATION.get() or ())}
    receipt = RECEIPT_ADAPTER.validate_python(receipt, strict=True).model_dump(
        mode="json", exclude_none=True
    )
    receipt_bytes = canonical_bytes(receipt)
    path.write_bytes(receipt_bytes)
    receipt_upload = _upload_multipart_immutable(
        client,
        bucket=request["bucket"],
        item=target,
        established_identity=(
            len(receipt_bytes),
            hashlib.sha256(receipt_bytes).hexdigest(),
        ),
    )
    _emit("BIO_NEB_WORKER_RECEIPT", receipt=receipt, receipt_object=receipt_upload)
    return {**receipt, "receipt_object": receipt_upload}


def _inventory_targets(
    items: Sequence[Mapping[str, Any]],
    *,
    established: Mapping[str, tuple[int, str]] | None = None,
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for item in items:
        path = Path(item["local_target"])
        if not path.is_file() or path.stat().st_size <= 0:
            raise WorkerError("prepared reference inventory is incomplete")
        retained = (established or {}).get(item["role"])
        byte_count, sha256 = retained if retained is not None else _file_digest(path)
        inventory.append(
            {
                "role": item["role"],
                "local_target": str(path),
                "byte_count": byte_count,
                "sha256": sha256,
            }
        )
    _emit("BIO_NEB_WORKER_PREPARED_REFERENCE_INVENTORY", objects=inventory)
    return inventory


def _ensure_clean_prefix(request: Mapping[str, Any], client: Any) -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    if any(WORK_ROOT.iterdir()):
        raise WorkerError("worker scratch root was not empty")
    existing = client.list_objects_v2(
        Bucket=request["bucket"], Prefix=request["output_prefix"], MaxKeys=1
    )
    if existing.get("KeyCount", len(existing.get("Contents", ()))) != 0:
        raise WorkerError("worker output prefix was not empty")


def _execute_prepare(
    request: dict[str, Any],
    client: Any,
    tools: Mapping[str, Mapping[str, str]],
    run_command: Callable[..., subprocess.CompletedProcess[bytes]],
    fetch: Callable[..., Any],
) -> dict[str, Any]:
    tools = _select_tools(tools, "bwa", "samtools")
    source = _acquire_source(client, request, fetch)
    compressed = (request["download"] or request["inputs"][0])["local_target"]
    plain_identity = _derive_plain_reference(request, compressed)
    fasta = _plain_fasta(request)["local_target"]
    vectors = (
        ("bwa", "index", fasta),
        ("samtools", "faidx", fasta),
        ("samtools", "dict", "-o", str(Path(fasta).with_suffix(".dict")), fasta),
    )
    for command_id, vector in zip(PREPARATION_COMMANDS, vectors, strict=True):
        _run_vector(vector, tools, command_id=command_id, run=run_command)
    inventory = _inventory_targets(
        request["prepared_outputs"],
        established={"genomic_fasta": plain_identity},
    )
    support = [
        {
            "role": target["role"],
            "object_key": target["object_key"],
            "byte_count": observed["byte_count"],
            "sha256": observed["sha256"],
        }
        for target, observed in zip(request["prepared_outputs"], inventory, strict=True)
    ]
    _emit("BIO_NEB_WORKER_REFERENCE_SUPPORT", mode="derive", objects=support)
    if (inventory[0]["byte_count"], inventory[0]["sha256"]) != plain_identity:
        raise WorkerError("plain FASTA changed during preparation")
    uploads = [
        _upload_multipart_immutable(
            client,
            bucket=request["bucket"],
            item=item,
            established_identity=(observed["byte_count"], observed["sha256"]),
        )
        for item, observed in zip(request["prepared_outputs"], inventory, strict=True)
    ]
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "authority": "non-authoritative-worker-observation",
        "operation": request["operation"],
        "request_id": request["request_id"],
        "workflow": request["workflow"],
        "source": source,
        "prepared_uploads": uploads,
        "tools": tools,
    }
    return _write_and_upload_receipt(client, request, receipt)


def _execute_consume(
    request: dict[str, Any],
    client: Any,
    tools: Mapping[str, Mapping[str, str]],
    run_command: Callable[..., subprocess.CompletedProcess[bytes]],
) -> dict[str, Any]:
    tools = _select_tools(tools, "pbrun")
    for item in request["inputs"]:
        _download_verified(
            client, bucket=item.get("bucket") or request["bucket"], item=item
        )
    prepared_by_role = {
        item["role"]: item
        for item in request["inputs"]
        if item["role"] in PREPARED_ROLES
    }
    prepared = [
        {key: item[key] for key in ("role", "object_key", "byte_count", "sha256")}
        for item in (prepared_by_role[role] for role in PREPARED_ROLES)
    ]
    _emit("BIO_NEB_WORKER_REFERENCE_SUPPORT", mode="verify", objects=prepared)
    _emit("BIO_NEB_WORKER_PREPARED_REFERENCE_VERIFIED", objects=prepared)
    for output in request["outputs"]:
        Path(output["local_target"]).parent.mkdir(parents=True, exist_ok=True)
    _run_vector(
        request["parabricks_argv"], tools, command_id="pbrun-germline", run=run_command
    )
    output_inventory = _inventory_targets(request["outputs"])
    uploads = [
        _upload_multipart_immutable(
            client,
            bucket=request["bucket"],
            item=item,
            established_identity=(observed["byte_count"], observed["sha256"]),
        )
        for item, observed in zip(request["outputs"], output_inventory, strict=True)
    ]
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "authority": "non-authoritative-worker-observation",
        "operation": request["operation"],
        "request_id": request["request_id"],
        "workflow": request["workflow"],
        "prepared_inputs": prepared,
        "uploads": uploads,
        "tools": tools,
    }
    return _write_and_upload_receipt(client, request, receipt)


def execute(
    request: dict[str, Any] | bytes,
    client: Any,
    tools: Mapping[str, Mapping[str, str]],
    *,
    run_command: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    fetch: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    payload = request if isinstance(request, bytes) else canonical_bytes(request)
    request = validate_request(payload)
    if request["operation"] == "runtime_smoke":
        raise WorkerError("smoke request cannot execute a workload")
    token = _VALIDATION.set([])
    try:
        _announce_request(payload, tools)
        _ensure_clean_prefix(request, client)
        if request["operation"] == "prepare_reference":
            return _execute_prepare(request, client, tools, run_command, fetch)
        return _execute_consume(request, client, tools, run_command)
    finally:
        _VALIDATION.reset(token)


def main(argv: Sequence[str]) -> int:
    if len(argv) != 1:
        raise WorkerError("worker accepts exactly one request-file path")
    request_path = Path(argv[0])
    if not request_path.is_absolute() or not request_path.is_file():
        raise WorkerError("worker requires one absolute request-file path")
    with request_path.open("rb") as stream:
        payload = stream.read(MAX_REQUEST_BYTES + 1)
    request = validate_request(payload)
    tools = _tool_identity(request["operation"])
    if request["operation"] == "runtime_smoke":
        _announce_request(payload, tools)
        _emit(
            "BIO_NEB_PARABRICKS_WORKER_SMOKE_COMPLETE",
            request_id=request["request_id"],
            tools=tools,
        )
        return 0
    # Nebius Object Storage is the provider; compatible S3/AWS names imply no Amazon resource.
    client = boto3.client(
        "s3",
        endpoint_url=request["endpoint_url"],
        region_name=request["region"],
        config=Config(
            connect_timeout=30,
            read_timeout=300,
            retries={"total_max_attempts": 1, "mode": "standard"},
        ),
    )
    execute(payload, client, tools, run_command=subprocess.run)
    return 0


def _entrypoint(argv: Sequence[str]) -> int:
    try:
        return main(argv)
    except WorkerError as error:
        _emit("BIO_NEB_PARABRICKS_WORKER_ERROR", reason=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(_entrypoint(sys.argv[1:]))


__all__ = [
    "CHUNK_BYTES",
    "MAX_REQUEST_BYTES",
    "GPU_INPUT_ROLES",
    "PREPARATION_COMMANDS",
    "PREPARED_ROLES",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "WorkerError",
    "canonical_bytes",
    "execute",
    "load_request",
    "main",
    "validate_request",
    "worker_source_bytes",
]
