"""Credential-free native smoke for the production worker-request Adapter seams.

This operation exists only to prove the compiler -> placement -> Serverless contact ->
worker-observation chain.  It owns no Journey, Object Storage, biological, retry,
recovery, cancellation, or cleanup authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bio_neb.observation import AdapterHeartbeat
from bio_neb.parabricks.execution_observations import (
    ProviderTerminalContact,
    WorkerRequestObserved,
    WorkerStartupObserved,
    WorkerToolsObserved,
    observe_job,
    parse_worker_markers,
)
from bio_neb.parabricks.execution_runtime import (
    ATTESTED_BWA_VERSION,
    ATTESTED_IMAGE_CHILD,
    ATTESTED_IMAGE_INDEX,
    ATTESTED_PBRUN_VERSION,
    ATTESTED_PROVIDER_IMAGE,
    ATTESTED_PYTHON_VERSION,
    ATTESTED_SAMTOOLS_VERSION,
    ATTESTED_WORKER_SHA256,
)
from bio_neb.parabricks.worker_request import (
    RuntimeSmokeParameters,
    compile_worker_request,
)
from bio_neb.parabricks.worker_request_placement import (
    NativeWorkerRequestPlacement,
    PlacedWorkerRequest,
)
from bio_neb.providers.job_observations import (
    BoundedProviderLogObserved,
    ProviderStateObserved,
)
from bio_neb.providers.nebius_controller_runtime import (
    ACCEPTED_NEBIUS_CLI_VERSION,
    ACCEPTED_NEBIUS_EXECUTABLE,
    ACCEPTED_NEBIUS_EXECUTABLE_BYTE_COUNT,
    ACCEPTED_NEBIUS_EXECUTABLE_SHA256,
    make_accepted_nebius_cli_runner,
    run_native_controller_preflight,
)
from bio_neb.providers.nebius_registry_v2 import (
    NebiusRegistryV2Resolver,
)
from bio_neb.providers.nebius_serverless_job_contact import (
    CreatedJobContact,
    NebiusJobContactError,
    NebiusServerlessJobContact,
    ProviderReadyJobRequest,
)
from bio_neb.providers.runtime import NebiusCliTarget

_CONTAINER_PATH = "/opt/bio-neb/request.json"
_EXPECTED_MARKERS = (
    "BIO_NEB_PARABRICKS_WORKER_START",
    "BIO_NEB_WORKER_REQUEST_ACCEPTED",
    "BIO_NEB_WORKER_TOOLS",
    "BIO_NEB_PARABRICKS_WORKER_SMOKE_COMPLETE",
)
_EXPECTED_TOOLS = {
    "bwa": ("/usr/bin/bwa", ATTESTED_BWA_VERSION),
    "pbrun": ("/usr/local/parabricks/pbrun", ATTESTED_PBRUN_VERSION),
    "python": ("/opt/bio-neb/venv/bin/python", ATTESTED_PYTHON_VERSION),
    "samtools": ("/usr/bin/samtools", ATTESTED_SAMTOOLS_VERSION),
}


class RuntimeSmokeRefused(RuntimeError):
    """A fail-stop refusal in the bounded runtime-smoke operation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _exclusive_publish(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    parent_descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("evidence write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
            raise OSError("evidence file mode changed")
        os.close(descriptor)
        descriptor = None
        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _model(value: object) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class RuntimeSmokeConfig:
    run_root: Path
    request_path: Path
    preflight_receipt_path: Path
    result_path: Path
    failure_path: Path
    job_name: str
    request_id: str
    target: NebiusCliTarget
    heartbeat_interval_seconds: float = 1.0


class _CountingRunner:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, str]] = []
        self.preflight_calls = 0

    def preflight(self) -> Any:
        self.preflight_calls += 1
        return self.inner.preflight()

    def json(self, *arguments: str) -> Any:
        self.calls.append({"kind": "json", "family": " ".join(arguments[:3])})
        return self.inner.json(*arguments)

    def json_from_stdin(self, input_text: str, *arguments: str) -> Any:
        self.calls.append(
            {"kind": "json_from_stdin", "family": " ".join(arguments[:3])}
        )
        return self.inner.json_from_stdin(input_text, *arguments)

    def __call__(self, *arguments: str) -> Any:
        self.calls.append({"kind": "text", "family": " ".join(arguments[:3])})
        return self.inner(*arguments)


def _default_contact(runner: Any) -> NebiusServerlessJobContact:
    return NebiusServerlessJobContact(cli_runner=runner)


def _default_registry(runner: Any, profile: str) -> Callable[[str], Any]:
    return NebiusRegistryV2Resolver(cli_runner=runner, profile=profile)


@dataclass(frozen=True, slots=True)
class RuntimeSmokeDependencies:
    placement: NativeWorkerRequestPlacement = field(
        default_factory=NativeWorkerRequestPlacement
    )
    runner_factory: Callable[[str], Any] = make_accepted_nebius_cli_runner
    contact_factory: Callable[[Any], Any] = _default_contact
    registry_factory: Callable[[Any, str], Callable[[str], Any]] = _default_registry
    native_preflight: Callable[..., Any] = run_native_controller_preflight


def _validate_paths(config: RuntimeSmokeConfig) -> None:
    root = config.run_root
    if (
        not root.is_absolute()
        or not root.is_dir()
        or root.resolve(strict=True) != root
        or stat.S_IMODE(root.stat().st_mode) != 0o700
    ):
        raise RuntimeSmokeRefused("runtime_smoke_root_invalid")
    owned = (
        config.request_path,
        config.preflight_receipt_path,
        config.result_path,
        config.failure_path,
    )
    if any(not path.is_absolute() or path.parent != root for path in owned):
        raise RuntimeSmokeRefused("runtime_smoke_path_invalid")
    if len(set(owned)) != len(owned):
        raise RuntimeSmokeRefused("runtime_smoke_path_collision")
    if any(
        path.exists()
        for path in (
            config.preflight_receipt_path,
            config.result_path,
            config.failure_path,
        )
    ):
        raise RuntimeSmokeRefused("runtime_smoke_evidence_collision")
    if (
        not config.job_name
        or config.job_name.strip() != config.job_name
        or not config.request_id
        or config.request_id.strip() != config.request_id
    ):
        raise RuntimeSmokeRefused("runtime_smoke_identity_invalid")


def _file_identity(path: Path) -> dict[str, Any]:
    observed = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(observed.st_mode):
        raise RuntimeSmokeRefused("runtime_smoke_request_not_regular")
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "byte_count": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
        "mode": oct(stat.S_IMODE(observed.st_mode)),
    }


def _marker_name(line: str) -> str:
    parts = line.split(" ", 2)
    if len(parts) != 3:
        raise RuntimeSmokeRefused("runtime_smoke_marker_invalid")
    return parts[1]


def _validate_worker(
    placed: PlacedWorkerRequest,
    terminal: ProviderTerminalContact,
    placement: NativeWorkerRequestPlacement,
) -> tuple[WorkerStartupObserved, WorkerRequestObserved, WorkerToolsObserved]:
    if tuple(_marker_name(line) for line in terminal.marker_lines) != _EXPECTED_MARKERS:
        raise RuntimeSmokeRefused("runtime_smoke_markers_changed")
    parsed = parse_worker_markers("\n".join(terminal.marker_lines))
    startups = tuple(item for item in parsed if isinstance(item, WorkerStartupObserved))
    requests = tuple(item for item in parsed if isinstance(item, WorkerRequestObserved))
    tools = tuple(item for item in parsed if isinstance(item, WorkerToolsObserved))
    if len(startups) != 1 or len(requests) != 1 or len(tools) != 1:
        raise RuntimeSmokeRefused("runtime_smoke_worker_observation_incomplete")
    placement.reconcile_worker_identity(
        placed,
        request_schema=startups[0].request_schema,
        byte_count=requests[0].byte_count,
        request_sha256=requests[0].request_sha256,
    )
    if requests[0].worker_sha256 != ATTESTED_WORKER_SHA256:
        raise RuntimeSmokeRefused("runtime_smoke_worker_sha_changed")
    if (
        startups[0].interpreter != "/opt/bio-neb/venv/bin/python"
        or startups[0].interpreter_resolved != "/usr/bin/python3.10"
    ):
        raise RuntimeSmokeRefused("runtime_smoke_interpreter_changed")
    actual_tools = {item.name: (item.path, item.version) for item in tools[0].tools}
    resolved_tools = {item.name: item.resolved_path for item in tools[0].tools}
    if actual_tools != _EXPECTED_TOOLS or resolved_tools != {
        "bwa": None,
        "pbrun": None,
        "python": "/usr/bin/python3.10",
        "samtools": None,
    }:
        raise RuntimeSmokeRefused("runtime_smoke_tools_changed")
    return startups[0], requests[0], tools[0]


def _observation_payload(value: object) -> dict[str, Any]:
    if isinstance(value, ProviderStateObserved):
        return {"kind": "provider_state", **_model(value)}
    if isinstance(value, AdapterHeartbeat):
        return {"kind": "adapter_heartbeat"}
    if isinstance(value, BoundedProviderLogObserved):
        return {"kind": "bounded_provider_log", **_model(value)}
    if isinstance(value, ProviderTerminalContact):
        return {
            "kind": "provider_terminal",
            "provider_job_id": value.provider_job_id,
            "provider_state": value.provider_state,
            "marker_lines": list(value.marker_lines),
        }
    return {"kind": type(value).__name__}


def _lineage_payload(value: object) -> dict[str, Any]:
    index = getattr(value, "index_digest", None)
    child = getattr(value, "linux_amd64_child_digest", None)
    if index != ATTESTED_IMAGE_INDEX or child != ATTESTED_IMAGE_CHILD:
        raise RuntimeSmokeRefused("runtime_smoke_image_lineage_changed")
    return {"index_digest": index, "linux_amd64_child_digest": child}


def _failure_payload(
    config: RuntimeSmokeConfig,
    error: Exception,
    *,
    claim_count: int,
    callback_counts: Mapping[str, int],
    runner: _CountingRunner | None,
    placement_disposition: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "bio-neb.worker-request-runtime-smoke-failure/1",
        "status": "frozen_failure",
        "frozen_at": _utc_now(),
        "job_name": config.job_name,
        "request_path": config.request_path.as_posix(),
        "error_type": type(error).__name__,
        "reason": getattr(error, "reason", str(error)),
        "counts": {
            "adapter_create_claim": claim_count,
            "create_callback": callback_counts["create_invoked"],
            "provider_create": callback_counts["create_invoked"],
            "lineage_reattest": callback_counts["lineage_reattest"],
        },
        "placement_disposition": placement_disposition,
        "command_calls": [] if runner is None else runner.calls,
        "residue_retained": True,
        "retry_recreate_resume_adoption_cleanup": 0,
    }
    if isinstance(error, NebiusJobContactError):
        payload["provider_error"] = {
            "operation": error.operation,
            "category": error.category,
            "reason": error.reason,
            "provider_reason": error.provider_reason,
            "provider_job_id": error.provider_job_id,
            "reconciliation_match_count": error.reconciliation_match_count,
        }
    return payload


def run_runtime_smoke(
    config: RuntimeSmokeConfig,
    *,
    dependencies: RuntimeSmokeDependencies | None = None,
) -> dict[str, Any]:
    """Run one fail-stop production-seam smoke and publish one terminal record."""

    deps = dependencies or RuntimeSmokeDependencies()
    claim_count = 0
    callback_counts = {"lineage_reattest": 0, "create_invoked": 0}
    runner: _CountingRunner | None = None
    placement_disposition: str | None = None
    try:
        _validate_paths(config)
        preflight = deps.native_preflight(
            executable=config.target.executable,
            run_root=config.run_root.as_posix(),
            receipt_file=config.preflight_receipt_path.as_posix(),
        )
        compiled = compile_worker_request(
            RuntimeSmokeParameters(request_id=config.request_id)
        )
        existed = config.request_path.exists()
        placed = deps.placement.place(
            compiled,
            source_path=config.request_path.as_posix(),
            container_path=_CONTAINER_PATH,
        )
        placement_disposition = "reused_unchanged" if existed else "created"
        placed_identity = _file_identity(config.request_path)
        if (
            placed_identity["mode"] != "0o600"
            or config.request_path.read_bytes() != compiled.request_bytes
        ):
            raise RuntimeSmokeRefused("runtime_smoke_placed_bytes_changed")

        claim_count += 1
        inner_runner = deps.runner_factory(config.target.executable)
        runner = _CountingRunner(inner_runner)
        contact_engine = deps.contact_factory(runner)
        contact_engine.preflight(config.target)
        provider_request = ProviderReadyJobRequest(
            target=config.target,
            provider_job_name=config.job_name,
            provider_image_reference=ATTESTED_PROVIDER_IMAGE,
            platform="cpu-e2",
            preset="2vcpu-8gb",
            disk_size="250Gi",
            disk_size_bytes=250 * 1024**3,
            shared_memory_size=None,
            shared_memory_bytes=0,
            # No deadline is requested: the APP authors none and the server
            # injects its own spec.timeout.
            restart_policy="never",
            normalized_arguments=placed.normalized_arguments,
            injection_provider_values=placed.injection_provider_values,
            injected_paths=placed.injected_paths,
            environment_names=(),
            version_bound_secret_environment_names=(),
        )
        lineage: dict[str, Any] | None = None

        def reattest() -> None:
            nonlocal lineage
            callback_counts["lineage_reattest"] += 1
            lineage = _lineage_payload(
                deps.registry_factory(runner, config.target.profile)(
                    ATTESTED_PROVIDER_IMAGE
                )
            )

        def record_create() -> None:
            callback_counts["create_invoked"] += 1

        created: CreatedJobContact = contact_engine.create(
            provider_request,
            environment_bindings=(),
            secret_environment_bindings=(),
            reattest_before_create=reattest,
            record_create_invoked=record_create,
        )
        deps.placement.reconcile_fetched_route(
            placed,
            normalized_arguments=created.envelope.normalized_arguments,
            injected_paths=created.envelope.injected_paths,
        )
        observations: list[dict[str, Any]] = [
            {"kind": "provider_create", **_model(created.create)},
            {"kind": "fetched_envelope", **_model(created.envelope)},
        ]
        terminal = observe_job(
            contact_engine,
            created,
            heartbeat_interval_seconds=config.heartbeat_interval_seconds,
            progress=lambda value: observations.append(_observation_payload(value)),
        )
        observations.append(_observation_payload(terminal))
        if terminal is None or terminal.provider_state != "COMPLETED":
            raise RuntimeSmokeRefused("runtime_smoke_terminal_missing")
        startup, worker_request, tools = _validate_worker(
            placed, terminal, deps.placement
        )
        terminal_identity = _file_identity(config.request_path)
        if (
            terminal_identity != placed_identity
            or config.request_path.read_bytes() != compiled.request_bytes
        ):
            raise RuntimeSmokeRefused("runtime_smoke_source_changed_after_placement")
        if claim_count != 1 or callback_counts != {
            "lineage_reattest": 1,
            "create_invoked": 1,
        }:
            raise RuntimeSmokeRefused("runtime_smoke_contact_ceiling_changed")
        family_counts: dict[str, int] = {}
        for call in runner.calls:
            family = call["family"]
            family_counts[family] = family_counts.get(family, 0) + 1
        if family_counts.get("ai job create") != 1:
            raise RuntimeSmokeRefused("runtime_smoke_create_count_changed")
        result = {
            "schema_version": "bio-neb.worker-request-runtime-smoke/1",
            "status": "accepted",
            "finished_at": _utc_now(),
            "preflight": _model(preflight),
            "request": {
                "request_id": config.request_id,
                "operation": compiled.identity.operation,
                "schema_version": compiled.identity.schema_version,
                "byte_count": compiled.identity.byte_count,
                "sha256": compiled.identity.sha256,
                "source_path": placed.source_path,
                "container_path": placed.container_path,
                "placement_disposition": placement_disposition,
                "placed_identity": placed_identity,
                "terminal_identity": terminal_identity,
                "retained_unchanged": True,
            },
            "projection": {
                "requested_arguments": list(placed.normalized_arguments),
                "requested_injections": list(placed.injection_provider_values),
                "fetched_arguments": list(created.envelope.normalized_arguments),
                "fetched_injected_paths": list(created.envelope.injected_paths),
                "host_source_returned_by_provider": False,
            },
            "job": {
                "provider_job_id": created.provider_job_id,
                "provider_job_name": config.job_name,
                "provider_state": terminal.provider_state,
                "create_disposition": created.create.disposition,
                "image": ATTESTED_PROVIDER_IMAGE,
                "platform": provider_request.platform,
                "preset": provider_request.preset,
                "disk_size_bytes": provider_request.disk_size_bytes,
                "shared_memory_bytes": provider_request.shared_memory_bytes,
                # The provider's own injected deadline, observed off the envelope.
                "timeout": created.envelope.timeout,
                "restart_policy": provider_request.restart_policy,
                "lineage": lineage,
            },
            "worker": {
                "startup": _model(startup),
                "request": _model(worker_request),
                "tools": _model(tools),
                "marker_names": list(_EXPECTED_MARKERS),
                "marker_lines": list(terminal.marker_lines),
                "marker_lines_sha256": hashlib.sha256(
                    ("\n".join(terminal.marker_lines) + "\n").encode("utf-8")
                ).hexdigest(),
            },
            "counts": {
                "adapter_create_claim": claim_count,
                "lineage_reattest": callback_counts["lineage_reattest"],
                "create_callback": callback_counts["create_invoked"],
                "provider_create": callback_counts["create_invoked"],
                "runner_preflight": runner.preflight_calls,
                "command_families": family_counts,
                "heartbeats": sum(
                    item["kind"] == "adapter_heartbeat" for item in observations
                ),
            },
            "observations": observations,
            "zero_authority": {
                "object_storage_credentials": 0,
                "object_storage_contacts": 0,
                "object_storage_resources": 0,
                "biological_work": 0,
                "journey_writes": 0,
                "model_calls": 0,
                "agent_calls": 0,
                "hitl_calls": 0,
                "image_builds": 0,
                "image_pushes": 0,
                "retry_recreate_resume_adoption": 0,
                "cancellation": 0,
                "cleanup": 0,
            },
        }
        _exclusive_publish(config.result_path, result)
        return result
    except Exception as error:
        failure = _failure_payload(
            config,
            error,
            claim_count=claim_count,
            callback_counts=callback_counts,
            runner=runner,
            placement_disposition=placement_disposition,
        )
        if not config.failure_path.exists():
            _exclusive_publish(config.failure_path, failure)
        raise


def review_receipt(payload: bytes, *, expected_sha256: str) -> bytes:
    """Return a byte-stable contact-free review from retained receipt bytes only."""

    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise RuntimeSmokeRefused("runtime_smoke_review_hash_mismatch")
    try:
        retained = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeSmokeRefused("runtime_smoke_review_invalid") from error
    if (
        not isinstance(retained, dict)
        or retained.get("schema_version") != "bio-neb.worker-request-runtime-smoke/1"
        or retained.get("status") != "accepted"
        or retained.get("request", {}).get("operation") != "runtime_smoke"
        or retained.get("job", {}).get("provider_state") != "COMPLETED"
        or retained.get("worker", {}).get("marker_names") != list(_EXPECTED_MARKERS)
        or retained.get("counts", {}).get("adapter_create_claim") != 1
        or retained.get("counts", {}).get("create_callback") != 1
        or retained.get("counts", {}).get("provider_create") != 1
        or any(retained.get("zero_authority", {}).values())
    ):
        raise RuntimeSmokeRefused("runtime_smoke_review_refused")
    review = {
        "schema_version": "bio-neb.worker-request-runtime-smoke-review/1",
        "result": "accepted_contact_free",
        "source_sha256": observed_sha256,
        "provider_job_id": retained["job"]["provider_job_id"],
        "request_sha256": retained["request"]["sha256"],
        "terminal_state": retained["job"]["provider_state"],
        "marker_names": retained["worker"]["marker_names"],
        "provider_contacts": 0,
        "model_contacts": 0,
        "storage_contacts": 0,
        "registry_contacts": 0,
        "worker_processes": 0,
        "filesystem_writes": 0,
        "cleanup_mutations": 0,
    }
    return _canonical(review) + b"\n"


def _target(args: argparse.Namespace) -> NebiusCliTarget:
    return NebiusCliTarget(
        controller_runtime="native_ubuntu",
        executable=ACCEPTED_NEBIUS_EXECUTABLE,
        resolved_executable=ACCEPTED_NEBIUS_EXECUTABLE,
        cli_version=ACCEPTED_NEBIUS_CLI_VERSION,
        executable_byte_count=ACCEPTED_NEBIUS_EXECUTABLE_BYTE_COUNT,
        executable_sha256=ACCEPTED_NEBIUS_EXECUTABLE_SHA256,
        profile=args.profile,
        tenant_id=args.tenant_id,
        project_id=args.project_id,
        subnet_id=args.subnet_id,
        registry_id=args.registry_id,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    run = subparsers.add_parser("run")
    for name in (
        "run-root",
        "request-path",
        "preflight-receipt",
        "result-file",
        "failure-file",
        "job-name",
        "request-id",
        "profile",
        "tenant-id",
        "project-id",
        "subnet-id",
        "registry-id",
    ):
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--heartbeat-interval-seconds", type=float, default=1.0)
    review = subparsers.add_parser("review")
    review.add_argument("--result-file", required=True)
    review.add_argument("--expected-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "review":
        output = review_receipt(
            Path(args.result_file).read_bytes(),
            expected_sha256=args.expected_sha256,
        )
        os.write(1, output)
        return 0
    config = RuntimeSmokeConfig(
        run_root=Path(args.run_root),
        request_path=Path(args.request_path),
        preflight_receipt_path=Path(args.preflight_receipt),
        result_path=Path(args.result_file),
        failure_path=Path(args.failure_file),
        job_name=args.job_name,
        request_id=args.request_id,
        target=_target(args),
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
    )
    result = run_runtime_smoke(config)
    print(
        json.dumps(
            {
                "result": result["status"],
                "provider_job_id": result["job"]["provider_job_id"],
                "receipt": config.result_path.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RuntimeSmokeConfig",
    "RuntimeSmokeDependencies",
    "RuntimeSmokeRefused",
    "main",
    "review_receipt",
    "run_runtime_smoke",
]
