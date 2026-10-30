from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from bio_neb.devtools.parabricks.runtime_smoke import (
    RuntimeSmokeConfig,
    RuntimeSmokeDependencies,
    RuntimeSmokeRefused,
    review_receipt,
    run_runtime_smoke,
)
from bio_neb.parabricks.execution_runtime import (
    ATTESTED_BWA_VERSION,
    ATTESTED_IMAGE_CHILD,
    ATTESTED_IMAGE_INDEX,
    ATTESTED_PBRUN_VERSION,
    ATTESTED_SAMTOOLS_VERSION,
    ATTESTED_WORKER_SHA256,
)
from bio_neb.parabricks.worker_request_placement import WorkerRequestPlacementError
from bio_neb.providers.job_observations import (
    FetchedProviderEnvelope,
    ProviderCreateObserved,
    ProviderStateObserved,
)
from bio_neb.providers.nebius_controller_runtime import (
    ACCEPTED_NEBIUS_CLI_VERSION,
    ACCEPTED_NEBIUS_EXECUTABLE,
    ACCEPTED_NEBIUS_EXECUTABLE_BYTE_COUNT,
    ACCEPTED_NEBIUS_EXECUTABLE_SHA256,
    NativeControllerPreflightReceipt,
)
from bio_neb.providers.nebius_serverless_job_contact import (
    CreatedJobContact,
)
from bio_neb.providers.runtime import NebiusCliTarget
from tests.execution_observation_fakes import worker_marker

_marker = partial(
    worker_marker,
    timestamp_prefix="2026-08-15T06:00:",
    timezone="+00:00",
)


class _Runner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def preflight(self) -> object:
        return object()

    def json(self, *arguments: str) -> dict[str, object]:
        self.calls.append(arguments)
        return {}

    def json_from_stdin(self, input_text: str, *arguments: str) -> dict[str, object]:
        self.calls.append(arguments)
        return {}

    def __call__(self, *arguments: str) -> str:
        self.calls.append(arguments)
        return ""


@dataclass(frozen=True)
class _Lineage:
    index_digest: str = ATTESTED_IMAGE_INDEX
    linux_amd64_child_digest: str = ATTESTED_IMAGE_CHILD


class _Contact:
    def __init__(self, runner: object, *, route_drift: bool = False) -> None:
        self.runner = runner
        self.route_drift = route_drift
        self.request = None

    def preflight(self, target: NebiusCliTarget) -> None:
        self.runner.preflight()

    def create(
        self,
        request: object,
        *,
        environment_bindings: tuple[str, ...],
        secret_environment_bindings: tuple[str, ...],
        reattest_before_create: object,
        record_create_invoked: object,
    ) -> CreatedJobContact:
        self.request = request
        self.assert_empty(environment_bindings, secret_environment_bindings)
        reattest_before_create()
        record_create_invoked()
        self.runner.json("ai", "job", "create")
        job_id = "aijob-e00offlinefake"
        create = ProviderCreateObserved(
            provider_job_id=job_id,
            provider_job_name=request.provider_job_name,
            disposition="created",
        )
        injected_paths = (
            ("/opt/bio-neb/changed.json",)
            if self.route_drift
            else request.injected_paths
        )
        envelope = FetchedProviderEnvelope(
            provider_job_id=job_id,
            provider_job_name=request.provider_job_name,
            project_id=request.target.project_id,
            provider_image_reference=request.provider_image_reference,
            platform=request.platform,
            preset=request.preset,
            disk_size_bytes=request.disk_size_bytes,
            subnet_id=request.target.subnet_id,
            # Server-injected: the create request sets no
            # deadline.
            timeout="86400s",
            container_command_requested="image-entrypoint",
            container_command_fetched=None,
            normalized_arguments=request.normalized_arguments,
            injected_paths=injected_paths,
            environment_names=(),
            version_bound_secret_environment_names=(),
            shared_memory_bytes=0,
            restart_fetched="omitted",
        )
        state = ProviderStateObserved(
            provider_job_id=job_id,
            provider_state="STARTING",
            created_at="2026-08-15T06:00:00+00:00",
        )
        return CreatedJobContact(
            create=create,
            # The verbatim create response, retained as Run proof and never read.
            create_response="Job ID: " + job_id + chr(10),
            envelope=envelope,
            initial_state=state,
        )

    @staticmethod
    def assert_empty(
        environment_bindings: tuple[str, ...],
        secret_environment_bindings: tuple[str, ...],
    ) -> None:
        if environment_bindings or secret_environment_bindings:
            raise AssertionError("runtime smoke gained an environment binding")

    def poll_state(self, job_id):
        return ProviderStateObserved(
            provider_job_id=job_id,
            provider_state="COMPLETED",
            created_at="2026-08-15T06:00:00+00:00",
            started_at="2026-08-15T06:00:01+00:00",
            finished_at="2026-08-15T06:00:04+00:00",
        ), True

    def read_log(self, job_id: str, *, window: int):
        request = self.request
        source = Path(request.injection_provider_values[0].split(":", 1)[0])
        payload = source.read_bytes()
        tools = {
            "bwa": {
                "path": "/usr/bin/bwa",
                "version": ATTESTED_BWA_VERSION,
            },
            "pbrun": {
                "path": "/usr/local/parabricks/pbrun",
                "version": ATTESTED_PBRUN_VERSION,
            },
            "python": {
                "path": "/opt/bio-neb/venv/bin/python",
                "resolved_path": "/usr/bin/python3.10",
                "version": "3.10.12",
            },
            "samtools": {
                "path": "/usr/bin/samtools",
                "version": ATTESTED_SAMTOOLS_VERSION,
            },
        }
        markers = (
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
                    "byte_count": len(payload),
                    "request_sha256": hashlib.sha256(payload).hexdigest(),
                    "worker_sha256": ATTESTED_WORKER_SHA256,
                },
                2,
            ),
            _marker("BIO_NEB_WORKER_TOOLS", {"tools": tools}, 3),
            _marker(
                "BIO_NEB_PARABRICKS_WORKER_SMOKE_COMPLETE",
                {"request_id": "fake", "tools": tools},
                4,
            ),
        )
        return "\n".join(markers)


class _PlacementFailure:
    def place(self, *args: object, **kwargs: object) -> object:
        raise WorkerRequestPlacementError("fixture_placement_refused")


class RuntimeSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.root.chmod(0o700)
        self.target = NebiusCliTarget(
            controller_runtime="native_ubuntu",
            executable=ACCEPTED_NEBIUS_EXECUTABLE,
            resolved_executable=ACCEPTED_NEBIUS_EXECUTABLE,
            cli_version=ACCEPTED_NEBIUS_CLI_VERSION,
            executable_byte_count=ACCEPTED_NEBIUS_EXECUTABLE_BYTE_COUNT,
            executable_sha256=ACCEPTED_NEBIUS_EXECUTABLE_SHA256,
            profile="fixture-profile",
            tenant_id="tenant-fixture",
            project_id="project-fixture",
            subnet_id="vpcsubnet-fixture",
            registry_id="registry-fixture",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(self, suffix: str = "one") -> RuntimeSmokeConfig:
        return RuntimeSmokeConfig(
            run_root=self.root,
            request_path=self.root / "request.json",
            preflight_receipt_path=self.root / f"preflight-{suffix}.json",
            result_path=self.root / f"result-{suffix}.json",
            failure_path=self.root / f"failure-{suffix}.json",
            job_name=f"bio-neb-runtime-smoke-offline-{suffix}",
            request_id="runtime-smoke:offline",
            target=self.target,
        )

    def preflight(self, **kwargs: str) -> NativeControllerPreflightReceipt:
        return NativeControllerPreflightReceipt(
            controller_runtime="native_ubuntu",
            executable=ACCEPTED_NEBIUS_EXECUTABLE,
            resolved_executable=ACCEPTED_NEBIUS_EXECUTABLE,
            nebius_cli_version=ACCEPTED_NEBIUS_CLI_VERSION,
            executable_byte_count=ACCEPTED_NEBIUS_EXECUTABLE_BYTE_COUNT,
            executable_sha256=ACCEPTED_NEBIUS_EXECUTABLE_SHA256,
            run_root=kwargs["run_root"],
        )

    def dependencies(self, *, route_drift: bool = False) -> RuntimeSmokeDependencies:
        return RuntimeSmokeDependencies(
            runner_factory=lambda executable: _Runner(),
            contact_factory=lambda runner: _Contact(runner, route_drift=route_drift),
            registry_factory=lambda runner, profile: lambda image: _Lineage(),
            native_preflight=self.preflight,
        )

    def test_offline_fake_run_places_contacts_and_reviews_byte_stably(self) -> None:
        config = self.config()
        result = run_runtime_smoke(config, dependencies=self.dependencies())

        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["request"]["placement_disposition"], "created")
        self.assertEqual(result["counts"]["adapter_create_claim"], 1)
        self.assertEqual(result["counts"]["create_callback"], 1)
        self.assertEqual(result["counts"]["provider_create"], 1)
        self.assertEqual(result["zero_authority"]["object_storage_contacts"], 0)
        self.assertEqual(stat.S_IMODE(config.request_path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(config.result_path.stat().st_mode), 0o600)
        payload = config.result_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        first = review_receipt(payload, expected_sha256=digest)
        second = review_receipt(payload, expected_sha256=digest)
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)["provider_contacts"], 0)

    def test_same_canonical_request_is_reused_without_rewrite(self) -> None:
        first = self.config("first")
        run_runtime_smoke(first, dependencies=self.dependencies())
        before = first.request_path.stat()
        second = self.config("second")

        result = run_runtime_smoke(second, dependencies=self.dependencies())

        after = second.request_path.stat()
        self.assertEqual(result["request"]["placement_disposition"], "reused_unchanged")
        self.assertEqual((before.st_dev, before.st_ino), (after.st_dev, after.st_ino))
        self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)

    def test_placement_failure_freezes_before_claim_callback_or_contact(self) -> None:
        config = self.config()
        runner_factory_called = False

        def runner_factory(executable: str) -> _Runner:
            nonlocal runner_factory_called
            runner_factory_called = True
            return _Runner()

        dependencies = RuntimeSmokeDependencies(
            placement=_PlacementFailure(),  # type: ignore[arg-type]
            runner_factory=runner_factory,
            native_preflight=self.preflight,
        )

        with self.assertRaises(WorkerRequestPlacementError) as refused:
            run_runtime_smoke(config, dependencies=dependencies)
        self.assertEqual(refused.exception.reason, "fixture_placement_refused")

        failure = json.loads(config.failure_path.read_bytes())
        self.assertFalse(runner_factory_called)
        self.assertEqual(
            failure["counts"],
            {
                "adapter_create_claim": 0,
                "create_callback": 0,
                "lineage_reattest": 0,
                "provider_create": 0,
            },
        )
        self.assertFalse(config.result_path.exists())

    def test_fetched_route_drift_freezes_after_the_single_create(self) -> None:
        config = self.config()

        with self.assertRaises(WorkerRequestPlacementError) as refused:
            run_runtime_smoke(config, dependencies=self.dependencies(route_drift=True))
        self.assertEqual(refused.exception.reason, "provider_request_route_drift")

        failure = json.loads(config.failure_path.read_bytes())
        self.assertEqual(failure["counts"]["adapter_create_claim"], 1)
        self.assertEqual(failure["counts"]["create_callback"], 1)
        self.assertEqual(failure["counts"]["provider_create"], 1)
        self.assertFalse(config.result_path.exists())

    def test_review_refuses_hash_or_semantic_drift(self) -> None:
        config = self.config()
        run_runtime_smoke(config, dependencies=self.dependencies())
        payload = config.result_path.read_bytes()

        with self.assertRaisesRegex(RuntimeSmokeRefused, "hash_mismatch"):
            review_receipt(payload, expected_sha256="0" * 64)
        changed = json.loads(payload)
        changed["counts"]["provider_create"] = 2
        tampered = json.dumps(changed, sort_keys=True).encode()
        with self.assertRaisesRegex(RuntimeSmokeRefused, "review_refused"):
            review_receipt(
                tampered, expected_sha256=hashlib.sha256(tampered).hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
