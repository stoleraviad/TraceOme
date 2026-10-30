import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bio_neb.parabricks import (
    reusable_object_storage_parabricks as worker,
)
from tests.reusable_parabricks_worker_fakes import (
    fixed_tools,
    prepared_reference_consumption_request,
    reference_preparation_request,
)


def _runtime_smoke_request() -> dict[str, str]:
    return {
        "schema_version": worker.REQUEST_SCHEMA_VERSION,
        "operation": "runtime_smoke",
        "request_id": "synthetic-runtime-smoke",
        "workflow": "nvidia-parabricks-germline",
    }


class ReusableWorkerRequestContractTests(unittest.TestCase):
    def test_request_schema_accepts_reference_preparation_and_consumption_operations(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            prepare, _ = reference_preparation_request(root)
            consume, _ = prepared_reference_consumption_request(root)
            with patch.object(worker, "WORK_ROOT", root):
                self.assertEqual(
                    worker.validate_request(worker.canonical_bytes(prepare))[
                        "operation"
                    ],
                    "prepare_reference",
                )
                self.assertEqual(
                    worker.validate_request(worker.canonical_bytes(consume))[
                        "operation"
                    ],
                    "consume_prepared_reference",
                )

    def test_worker_entrypoint_requires_one_absolute_request_path(self):
        with self.assertRaisesRegex(worker.WorkerError, "exactly one"):
            worker.main([])
        with self.assertRaisesRegex(worker.WorkerError, "absolute"):
            worker.main(["relative.json"])

    # These offline survivors preserve runtime-smoke schema and zero-storage
    # behavior. They are not live provider, storage, or worker proof.
    def test_runtime_smoke_schema_is_accepted(self):
        request = _runtime_smoke_request()
        self.assertEqual(
            worker.validate_request(worker.canonical_bytes(request))["operation"],
            "runtime_smoke",
        )

    def test_runtime_smoke_accepts_canonical_request_without_object_storage_contact(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory).resolve() / "smoke.json"
            request_path.write_bytes(worker.canonical_bytes(_runtime_smoke_request()))
            output = io.StringIO()
            with (
                patch.object(worker, "_tool_identity", return_value=fixed_tools()),
                patch.object(worker.boto3, "client") as client,
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(worker.main([str(request_path)]), 0)
            client.assert_not_called()
        self.assertIn("BIO_NEB_WORKER_REQUEST_ACCEPTED", output.getvalue())
        self.assertIn("BIO_NEB_PARABRICKS_WORKER_SMOKE_COMPLETE", output.getvalue())

    def test_worker_request_rejects_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, _ = reference_preparation_request(root)
            request["unexpected"] = True
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(worker.WorkerError, "type contract"),
            ):
                worker.validate_request(worker.canonical_bytes(request))

    def test_worker_request_rejects_credential_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, _ = reference_preparation_request(root)
            request["aws_secret_access_key"] = "forbidden"
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(worker.WorkerError, "credential"),
            ):
                worker.validate_request(worker.canonical_bytes(request))

    def test_worker_request_rejects_noncanonical_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, _ = reference_preparation_request(root)
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(worker.WorkerError, "canonical"),
            ):
                worker.validate_request(json.dumps(request).encode())

    def test_worker_request_rejects_oversize_bytes(self):
        with self.assertRaisesRegex(worker.WorkerError, "byte bound"):
            worker.validate_request(b" " * (worker.MAX_REQUEST_BYTES + 1))

    def test_reference_preparation_request_requires_fixed_command_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, _ = reference_preparation_request(root)
            request["reference"]["commands"][0] = "arbitrary-command"
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(worker.WorkerError, "fixed reference"),
            ):
                worker.validate_request(worker.canonical_bytes(request))

    def test_reference_preparation_request_requires_exact_eight_role_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, _ = reference_preparation_request(root)
            request["prepared_outputs"] = request["prepared_outputs"][:-1]
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(worker.WorkerError, "type contract"),
            ):
                worker.validate_request(worker.canonical_bytes(request))

    def test_prepared_reference_consumption_requires_exact_ordered_input_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, _ = prepared_reference_consumption_request(root)
            request["inputs"] = [
                item for item in request["inputs"] if item["role"] != "bwa_sa"
            ]
            request["inputs"].append(dict(request["inputs"][-1]))
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(worker.WorkerError, "role inventory"),
            ):
                worker.validate_request(worker.canonical_bytes(request))

    def test_prepared_reference_consumption_forbids_derivation_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, _ = prepared_reference_consumption_request(root)
            request["preparation_commands"] = [["bwa", "index"]]
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(worker.WorkerError, "type contract"),
            ):
                worker.validate_request(worker.canonical_bytes(request))

    def test_prepared_reference_consumption_preserves_read_role_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, _ = prepared_reference_consumption_request(root)
            request["inputs"][0], request["inputs"][1] = (
                request["inputs"][1],
                request["inputs"][0],
            )
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(worker.WorkerError, "role inventory"),
            ):
                worker.validate_request(worker.canonical_bytes(request))

    def test_prepared_reference_consumption_discovers_only_python_and_pbrun(self):
        discovered: list[str] = []

        def which(name: str) -> str:
            discovered.append(name)
            return str(Path.cwd().resolve() / "tools" / name)

        def run(argv, **kwargs):
            del kwargs
            return subprocess.CompletedProcess(argv, 0, b"pbrun: 4.7.0-1\n", b"")

        output = io.StringIO()
        with (
            patch.object(worker.shutil, "which", side_effect=which),
            contextlib.redirect_stdout(output),
        ):
            tools = worker._tool_identity("consume_prepared_reference", run=run)
        self.assertEqual(discovered, ["pbrun"])
        self.assertEqual(set(tools), {"python", "pbrun"})
        marker = output.getvalue()
        self.assertNotIn('"bwa"', marker)
        self.assertNotIn('"samtools"', marker)


if __name__ == "__main__":
    unittest.main()
