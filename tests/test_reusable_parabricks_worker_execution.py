import contextlib
import hashlib
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
    PUBLIC_SOURCE_URL,
    OfflineObjectStorage,
    OfflinePublisher,
    counting_path_open,
    fake_tool_runner,
    fixed_tools,
    prepared_paths,
    prepared_reference_consumption_request,
    public_reference_preparation_request,
    reference_preparation_request,
    sha256,
    target,
)


class ReusableWorkerExecutionTests(unittest.TestCase):
    def _execute_preparation(self, root: Path):
        request, bodies = reference_preparation_request(root)
        client, calls, output = OfflineObjectStorage(dict(bodies)), [], io.StringIO()
        read_bytes: dict[Path, int] = {}
        with (
            patch.object(worker, "WORK_ROOT", root),
            patch.object(worker, "CHUNK_BYTES", 4),
            patch.object(Path, "open", counting_path_open(read_bytes)),
            contextlib.redirect_stdout(output),
        ):
            receipt = worker.execute(
                request,
                client,
                fixed_tools(),
                run_command=fake_tool_runner(request, calls),
            )
        return request, client, calls, output.getvalue(), read_bytes, receipt

    def _execute_consumption(self, root: Path):
        request, bodies = prepared_reference_consumption_request(root)
        client, calls, output = OfflineObjectStorage(dict(bodies)), [], io.StringIO()
        read_bytes: dict[Path, int] = {}
        with (
            patch.object(worker, "WORK_ROOT", root),
            patch.object(worker, "CHUNK_BYTES", 4),
            patch.object(Path, "open", counting_path_open(read_bytes)),
            contextlib.redirect_stdout(output),
        ):
            receipt = worker.execute(
                request,
                client,
                fixed_tools(),
                run_command=fake_tool_runner(request, calls),
            )
        return request, client, calls, output.getvalue(), read_bytes, receipt

    def test_durable_validation_survives_broken_diagnostic_output(self):
        class BrokenOutput:
            def write(self, value):
                raise BrokenPipeError("diagnostic pipe closed")

            def flush(self):
                pass

        for operation in ("prepare", "consume"):
            with (
                self.subTest(operation=operation),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve() / "work"
                build = (
                    reference_preparation_request
                    if operation == "prepare"
                    else prepared_reference_consumption_request
                )
                request, bodies = build(root)
                client, calls = OfflineObjectStorage(dict(bodies)), []
                with (
                    patch.object(worker, "WORK_ROOT", root),
                    contextlib.redirect_stdout(BrokenOutput()),
                ):
                    result = worker.execute(
                        request,
                        client,
                        fixed_tools(),
                        run_command=fake_tool_runner(request, calls),
                    )
                receipt_path = Path(request["receipt_output"]["local_target"])
                retained = json.loads(receipt_path.read_bytes())
                self.assertEqual(retained["request_id"], request["request_id"])
                kinds = [fact["kind"] for fact in retained["validation"]]
                self.assertEqual(
                    kinds[:3], ["worker_startup", "worker_request", "worker_tools"]
                )
                exits = [
                    fact["returncode"]
                    for fact in retained["validation"]
                    if fact["kind"] == "tool_exited"
                ]
                self.assertEqual(exits, [0, 0, 0] if operation == "prepare" else [0])
                self.assertTrue(
                    any(
                        fact["kind"] == "worker_input_verified"
                        for fact in retained["validation"]
                    )
                )
                self.assertEqual(
                    client.uploads[-1][0], request["receipt_output"]["object_key"]
                )
                self.assertEqual(result["validation"], retained["validation"])

    def test_entrypoint_retains_exact_injected_cpu_and_gpu_request_identity(self):
        import hashlib

        for build in (
            reference_preparation_request,
            prepared_reference_consumption_request,
        ):
            with (
                self.subTest(operation=build.__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve() / "work"
                request, bodies = build(root)
                # Same-bucket requests omit optional input bucket fields; admission
                # normalizes them, but producer identity must still name these bytes.
                for item in request["inputs"]:
                    item.pop("bucket", None)
                payload = worker.canonical_bytes(request)
                request_path = Path(directory).resolve() / "request.json"
                request_path.write_bytes(payload)
                client, calls = OfflineObjectStorage(dict(bodies)), []
                with (
                    patch.object(worker, "WORK_ROOT", root),
                    patch.object(worker, "_tool_identity", return_value=fixed_tools()),
                    patch.object(worker.boto3, "client", return_value=client),
                    patch.object(
                        worker.subprocess,
                        "run",
                        side_effect=fake_tool_runner(request, calls),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(worker.main([str(request_path)]), 0)
                retained = json.loads(
                    client.objects[request["receipt_output"]["object_key"]]
                )
                identity = next(
                    fact
                    for fact in retained["validation"]
                    if fact["kind"] == "worker_request"
                )
                self.assertEqual(identity["byte_count"], len(payload))
                self.assertEqual(
                    identity["request_sha256"], hashlib.sha256(payload).hexdigest()
                )
                self.assertEqual(
                    len(calls), 3 if request["operation"] == "prepare_reference" else 1
                )

    def test_entrypoint_emits_exact_parseable_failure_reason(self):
        output = io.StringIO()
        with (
            patch.object(
                worker,
                "main",
                side_effect=worker.WorkerError("retained exact offline reason"),
            ),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(worker._entrypoint([]), 1)
        marker, encoded = output.getvalue().strip().split(" ", 1)
        self.assertEqual(marker, "BIO_NEB_PARABRICKS_WORKER_ERROR")
        self.assertEqual(
            json.loads(encoded), {"reason": "retained exact offline reason"}
        )

    def test_reference_preparation_derives_fasta_and_runs_fixed_absolute_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            _, _, calls, retained, _, _ = self._execute_preparation(root)
            fasta_path = root / "prepared/reference.fa"
            self.assertEqual(fasta_path.read_bytes(), b">synthetic-reference\nACGT\n")
            fasta = str(fasta_path)
            self.assertEqual(
                [call[0] for call in calls],
                [
                    ("/usr/bin/bwa", "index", fasta),
                    ("/usr/bin/samtools", "faidx", fasta),
                    (
                        "/usr/bin/samtools",
                        "dict",
                        "-o",
                        str(root / "prepared/reference.dict"),
                        fasta,
                    ),
                ],
            )
            self.assertTrue(all(call[1]["shell"] is False for call in calls))
            self.assertIn('BIO_NEB_WORKER_REFERENCE_SUPPORT {"mode":"derive"', retained)

    def test_reference_preparation_inventories_exact_eight_roles_before_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            _, _, _, retained, _, _ = self._execute_preparation(root)
            support_lines = [
                line
                for line in retained.splitlines()
                if line.startswith("BIO_NEB_WORKER_REFERENCE_SUPPORT ")
            ]
            self.assertEqual(len(support_lines), 1)
            support = json.loads(support_lines[0].split(" ", 1)[1])
            self.assertEqual(support["mode"], "derive")
            self.assertEqual(
                [item["role"] for item in support["objects"]],
                list(worker.PREPARED_ROLES),
            )
            self.assertTrue(
                all(
                    set(item) == {"role", "object_key", "byte_count", "sha256"}
                    for item in support["objects"]
                )
            )
            self.assertLess(
                retained.index("BIO_NEB_WORKER_PREPARED_REFERENCE_INVENTORY"),
                retained.index("BIO_NEB_WORKER_UPLOAD_START"),
            )

    def test_reference_preparation_streams_role_uploads_before_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            _, client, _, retained, read_bytes, _ = self._execute_preparation(root)
            expected = [
                f"synthetic/prepared/{role}" for role in worker.PREPARED_ROLES
            ] + ["synthetic/prepared/worker-receipt.json"]
            self.assertEqual([item[0] for item in client.uploads], expected)
            for role, path in prepared_paths(root / "prepared/reference.fa").items():
                expected_passes = 1 if role == "genomic_fasta" else 2
                self.assertEqual(
                    read_bytes[path], path.stat().st_size * expected_passes, role
                )
            self.assertTrue(all(max(parts) <= 4 for _, parts, _ in client.uploads))
            self.assertLess(
                retained.rindex("BIO_NEB_WORKER_UPLOAD_COMPLETE"),
                retained.index("BIO_NEB_WORKER_RECEIPT"),
            )

    def test_reference_preparation_receipt_is_non_authoritative_and_secret_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            _, client, _, retained, _, receipt = self._execute_preparation(root)
            self.assertEqual(
                receipt["authority"], "non-authoritative-worker-observation"
            )
            verified = [
                line
                for line in retained.splitlines()
                if line.startswith("BIO_NEB_WORKER_INPUT_VERIFIED ")
            ]
            self.assertEqual(len(verified), 1)
            self.assertEqual(
                json.loads(verified[0].split(" ", 1)[1])["role"],
                "reference_source",
            )
            persisted = json.loads(
                client.objects["synthetic/prepared/worker-receipt.json"]
            )
            self.assertNotIn("endpoint_url", persisted)
            self.assertNotIn("secret", json.dumps(persisted).casefold())

    def _execute_public_preparation(self, root: Path, served: bytes | None = None):
        request, compressed = public_reference_preparation_request(root)
        publisher = OfflinePublisher(compressed if served is None else served)
        client, calls, output = OfflineObjectStorage({}), [], io.StringIO()
        with (
            patch.object(worker, "WORK_ROOT", root),
            patch.object(worker, "CHUNK_BYTES", 4),
            contextlib.redirect_stdout(output),
        ):
            try:
                receipt = worker.execute(
                    request,
                    client,
                    fixed_tools(),
                    run_command=fake_tool_runner(request, calls),
                    fetch=publisher,
                )
            except worker.WorkerError as error:
                receipt = error
        return request, compressed, publisher, client, output.getvalue(), receipt

    def test_public_source_is_fetched_md5_checked_and_written_before_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, compressed, publisher, client, retained, receipt = (
                self._execute_public_preparation(root)
            )
            self.assertEqual(
                [url for url, _ in publisher.requests], [PUBLIC_SOURCE_URL]
            )
            source_key = request["download"]["object_key"]
            self.assertEqual(client.objects[source_key], compressed)
            self.assertEqual(
                [item[0] for item in client.uploads],
                [source_key]
                + [f"synthetic/prepared/{role}" for role in worker.PREPARED_ROLES]
                + ["synthetic/prepared/worker-receipt.json"],
            )
            plain = b">synthetic-reference\nACGT\n"
            self.assertEqual((root / "prepared/reference.fa").read_bytes(), plain)
            self.assertEqual(
                receipt["source"],
                {
                    "role": "reference_source",
                    "object_key": source_key,
                    "byte_count": len(compressed),
                    "sha256": sha256(compressed),
                    "md5": request["download"]["md5"],
                    "origin": PUBLIC_SOURCE_URL,
                },
            )
            self.assertEqual(
                (
                    receipt["prepared_uploads"][0]["byte_count"],
                    receipt["prepared_uploads"][0]["sha256"],
                ),
                (len(plain), sha256(plain)),
            )
            acquired = [
                json.loads(line.split(" ", 1)[1])
                for line in retained.splitlines()
                if line.startswith("BIO_NEB_WORKER_SOURCE_ACQUIRED ")
            ]
            self.assertEqual(
                acquired,
                [
                    {
                        "role": "reference_source",
                        "url": PUBLIC_SOURCE_URL,
                        "byte_count": len(compressed),
                        "md5": request["download"]["md5"],
                        "sha256": sha256(compressed),
                    }
                ],
            )
            self.assertNotIn("BIO_NEB_WORKER_INPUT_VERIFIED", retained)
            self.assertLess(
                retained.index("BIO_NEB_WORKER_SOURCE_ACQUIRED"),
                retained.index("BIO_NEB_WORKER_TOOL_START"),
            )

    def test_public_source_md5_mismatch_stops_with_evidence_before_any_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, compressed = public_reference_preparation_request(root)
            served = bytes([compressed[0] ^ 1]) + compressed[1:]
            _, _, _, client, retained, error = self._execute_public_preparation(
                root, served
            )
            self.assertIsInstance(error, worker.WorkerError)
            self.assertEqual(str(error), "reference source identity mismatch")
            mismatch = [
                json.loads(line.split(" ", 1)[1])
                for line in retained.splitlines()
                if line.startswith("BIO_NEB_WORKER_SOURCE_MISMATCH ")
            ]
            self.assertEqual(
                mismatch,
                [
                    {
                        "role": "reference_source",
                        "url": PUBLIC_SOURCE_URL,
                        "expected_byte_count": len(compressed),
                        "observed_byte_count": len(served),
                        "expected_md5": request["download"]["md5"],
                        "observed_md5": hashlib.md5(served).hexdigest(),
                    }
                ],
            )
            self.assertEqual((client.objects, client.uploads), ({}, []))
            self.assertNotIn("BIO_NEB_WORKER_SOURCE_ACQUIRED", retained)
            self.assertNotIn("BIO_NEB_WORKER_TOOL_START", retained)

    def test_public_source_beyond_its_approved_size_stops_while_streaming(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            _, compressed = public_reference_preparation_request(root)
            served = compressed + b"x" * 64
            _, _, _, client, retained, error = self._execute_public_preparation(
                root, served
            )
            self.assertEqual(str(error), "reference source identity mismatch")
            mismatch = json.loads(
                retained.split("BIO_NEB_WORKER_SOURCE_MISMATCH ", 1)[1].splitlines()[0]
            )
            # The stream stops at the first chunk past the approved size, and the
            # reported md5 describes exactly the bytes it counted.
            counted = (len(compressed) // 4 + 1) * 4
            self.assertEqual(mismatch["observed_byte_count"], counted)
            self.assertEqual(
                mismatch["observed_md5"], hashlib.md5(served[:counted]).hexdigest()
            )
            self.assertEqual(client.objects, {})

    def test_stored_source_receipt_records_its_md5_and_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, _, _, _, _, receipt = self._execute_preparation(root)
            stored = request["inputs"][0]
            body = Path(stored["local_target"]).read_bytes()
            self.assertEqual(
                receipt["source"],
                {
                    "role": "reference_source",
                    "object_key": stored["object_key"],
                    "byte_count": stored["byte_count"],
                    "sha256": stored["sha256"],
                    "md5": hashlib.md5(body).hexdigest(),
                    "origin": "s3://synthetic-offline-bucket/" + stored["object_key"],
                },
            )

    def test_reference_source_hash_drift_prevents_verified_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, bodies = reference_preparation_request(root)
            key = "synthetic/source/reference.fa.gz"
            body = bodies[key]
            bodies[key] = bytes([body[0] ^ 1]) + body[1:]
            output = io.StringIO()
            with (
                patch.object(worker, "WORK_ROOT", root),
                contextlib.redirect_stdout(output),
                self.assertRaisesRegex(worker.WorkerError, "input verification"),
            ):
                worker.execute(request, OfflineObjectStorage(bodies), fixed_tools())
            self.assertNotIn("BIO_NEB_WORKER_INPUT_VERIFIED", output.getvalue())

    def test_reference_source_oversize_prevents_verified_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, bodies = reference_preparation_request(root)
            bodies["synthetic/source/reference.fa.gz"] += b"oversize"
            output = io.StringIO()
            with (
                patch.object(worker, "WORK_ROOT", root),
                contextlib.redirect_stdout(output),
                self.assertRaisesRegex(worker.WorkerError, "exceeded its byte bound"),
            ):
                worker.execute(request, OfflineObjectStorage(bodies), fixed_tools())
            self.assertNotIn("BIO_NEB_WORKER_INPUT_VERIFIED", output.getvalue())

    def test_reference_source_drift_prevents_any_prepared_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, bodies = reference_preparation_request(root)
            bodies[next(iter(bodies))] += b"drift"
            client = OfflineObjectStorage(dict(bodies))
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaises(worker.WorkerError),
            ):
                worker.execute(request, client, fixed_tools())
            self.assertEqual(client.uploads, [])
            self.assertEqual(client._multipart, {})
            self.assertNotIn("synthetic/prepared/worker-receipt.json", client.objects)

    def test_reference_preparation_tool_failure_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, bodies = reference_preparation_request(root)
            client = OfflineObjectStorage(dict(bodies))

            def fail_tool(argv, **kwargs):
                return subprocess.CompletedProcess(argv, 7, b"", b"")

            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(worker.WorkerError, "failed"),
            ):
                worker.execute(request, client, fixed_tools(), run_command=fail_tool)
            self.assertEqual(client.uploads, [])
            self.assertEqual(client._multipart, {})
            self.assertNotIn("synthetic/prepared/worker-receipt.json", client.objects)

    def test_incomplete_prepared_inventory_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, bodies = reference_preparation_request(root)
            client, calls = OfflineObjectStorage(dict(bodies)), []
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(worker.WorkerError, "inventory is incomplete"),
            ):
                worker.execute(
                    request,
                    client,
                    fixed_tools(),
                    run_command=fake_tool_runner(request, calls, omit="bwa_sa"),
                )
            self.assertEqual(client.uploads, [])
            self.assertEqual(client._multipart, {})
            self.assertNotIn("synthetic/prepared/worker-receipt.json", client.objects)

    def test_prepared_object_upload_failure_retains_residue_and_prevents_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, bodies = reference_preparation_request(root)
            failed_key = "synthetic/prepared/bwa_ann"
            client, calls = OfflineObjectStorage(dict(bodies), fail_key=failed_key), []
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(RuntimeError, "offline upload failure"),
            ):
                worker.execute(
                    request,
                    client,
                    fixed_tools(),
                    run_command=fake_tool_runner(request, calls),
                )
            self.assertNotIn("synthetic/prepared/worker-receipt.json", client.objects)
            self.assertEqual(len(client._multipart), 1)
            self.assertEqual(next(iter(client._multipart.values()))["key"], failed_key)

    def test_reference_derivation_is_bounded_by_accepted_plain_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, bodies = reference_preparation_request(root, plain=b"A" * 32)
            request["reference"]["accepted_plain_fasta"]["byte_count"] = 8
            request["reference"]["accepted_plain_fasta"]["sha256"] = sha256(b"A" * 8)
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(worker.WorkerError, "plain byte bound"),
            ):
                worker.execute(request, OfflineObjectStorage(bodies), fixed_tools())

    # The CPU build indexes the Run's reference sequences only; the request names
    # the records to drop.
    WITH_ALT = (
        b">NC_1.1 chromosome 1\nACGT\nAC\n>NT_9.1 alt locus\nGGGG\n>NC_M.1 mito\nTT\n"
    )
    SELECTED = b">NC_1.1 chromosome 1\nACGT\nAC\n>NC_M.1 mito\nTT\n"

    def test_a_stored_source_is_indexed_without_its_excluded_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, bodies = reference_preparation_request(
                root, plain=self.WITH_ALT, excluded=("NT_9.1",), selected=self.SELECTED
            )
            calls = []
            with (
                patch.object(worker, "WORK_ROOT", root),
                patch.object(worker, "CHUNK_BYTES", 4),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                receipt = worker.execute(
                    request,
                    OfflineObjectStorage(dict(bodies)),
                    fixed_tools(),
                    run_command=fake_tool_runner(request, calls),
                )
            self.assertEqual(
                (root / "prepared/reference.fa").read_bytes(), self.SELECTED
            )
            self.assertEqual(
                (
                    receipt["prepared_uploads"][0]["byte_count"],
                    receipt["prepared_uploads"][0]["sha256"],
                ),
                (len(self.SELECTED), sha256(self.SELECTED)),
            )

    def test_a_downloaded_source_is_indexed_without_its_excluded_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, compressed = public_reference_preparation_request(
                root, plain=self.WITH_ALT, excluded=("NT_9.1",)
            )
            calls = []
            with (
                patch.object(worker, "WORK_ROOT", root),
                patch.object(worker, "CHUNK_BYTES", 4),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                receipt = worker.execute(
                    request,
                    OfflineObjectStorage({}),
                    fixed_tools(),
                    run_command=fake_tool_runner(request, calls),
                    fetch=OfflinePublisher(compressed),
                )
            self.assertEqual(
                (root / "prepared/reference.fa").read_bytes(), self.SELECTED
            )
            self.assertEqual(
                receipt["prepared_uploads"][0]["sha256"], sha256(self.SELECTED)
            )

    def test_a_stored_source_accepted_as_the_whole_fasta_is_refused(self):
        """Owner, #464: the accepted plain identity is the selection's; a source whose
        excluded records would have to stay is not indexed whole."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, bodies = reference_preparation_request(
                root, plain=self.WITH_ALT, excluded=("NT_9.1",), selected=self.WITH_ALT
            )
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(worker.WorkerError, "derivation changed"),
            ):
                worker.execute(
                    request, OfflineObjectStorage(dict(bodies)), fixed_tools()
                )

    def test_prepared_reference_consumption_verifies_all_ten_inputs_before_pbrun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            _, _, calls, retained, _, receipt = self._execute_consumption(root)
            verified = [
                line
                for line in retained.splitlines()
                if line.startswith("BIO_NEB_WORKER_INPUT_VERIFIED ")
            ]
            self.assertEqual(len(verified), 10)
            self.assertEqual(
                [json.loads(line.split(" ", 1)[1])["role"] for line in verified],
                list(worker.GPU_INPUT_ROLES),
            )
            self.assertEqual(
                [item["role"] for item in receipt["prepared_inputs"]],
                list(worker.PREPARED_ROLES),
            )
            self.assertIn('BIO_NEB_WORKER_REFERENCE_SUPPORT {"mode":"verify"', retained)
            self.assertLess(
                retained.index("BIO_NEB_WORKER_PREPARED_REFERENCE_VERIFIED"),
                retained.index("pbrun-germline"),
            )
            self.assertEqual(len(calls), 1)

    def test_prepared_reference_consumption_runs_only_fixed_pbrun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            _, _, calls, retained, _, receipt = self._execute_consumption(root)
            self.assertEqual(
                [call[0][0] for call in calls], ["/usr/local/parabricks/pbrun"]
            )
            self.assertNotIn('"command_id":"bwa-', retained)
            self.assertNotIn('"command_id":"samtools-', retained)
            self.assertEqual(set(receipt["tools"]), {"python", "pbrun"})

    def test_consumption_hashes_outputs_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, _, _, _, read_bytes, _ = self._execute_consumption(root)
            for item in request["outputs"]:
                path = Path(item["local_target"])
                self.assertEqual(read_bytes[path], path.stat().st_size * 2)

    def test_consumption_uploads_bam_then_vcf_then_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            _, client, _, _, _, _ = self._execute_consumption(root)
            self.assertEqual(
                [item[0] for item in client.uploads],
                [
                    "synthetic/results/aligned.bam",
                    "synthetic/results/variants.vcf",
                    "synthetic/results/worker-receipt.json",
                ],
            )

    def test_immutable_upload_detects_local_drift_and_retains_multipart_residue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.bam"
            original = b"a" * (12 * 1024)
            path.write_bytes(original)

            class MutatingOfflineStorage(OfflineObjectStorage):
                def upload_part(self, **request):
                    result = super().upload_part(**request)
                    if request["PartNumber"] == 1:
                        with path.open("r+b") as stream:
                            stream.seek(9 * 1024)
                            stream.write(b"b")
                    return result

            client = MutatingOfflineStorage({})
            with (
                patch.object(worker, "CHUNK_BYTES", 4 * 1024),
                self.assertRaisesRegex(worker.WorkerError, "changed during upload"),
            ):
                worker._upload_multipart_immutable(
                    client,
                    bucket="synthetic-offline-bucket",
                    item=target("aligned_bam", "synthetic/outputs/aligned.bam", path),
                    established_identity=(len(original), sha256(original)),
                )
            self.assertEqual(client.uploads, [])
            self.assertEqual(len(client._multipart), 1)

    def test_pbrun_failure_retains_local_outputs_and_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, bodies = prepared_reference_consumption_request(root)
            client, calls, output = (
                OfflineObjectStorage(dict(bodies)),
                [],
                io.StringIO(),
            )

            def fail_after_writing(argv, **kwargs):
                calls.append((tuple(argv), kwargs))
                Path(request["outputs"][0]["local_target"]).write_bytes(b"partial-bam")
                Path(request["outputs"][1]["local_target"]).write_bytes(b"partial-vcf")
                return subprocess.CompletedProcess(argv, 17, b"", b"")

            with (
                patch.object(worker, "WORK_ROOT", root),
                contextlib.redirect_stdout(output),
                self.assertRaisesRegex(worker.WorkerError, "pbrun-germline failed"),
            ):
                worker.execute(
                    request, client, fixed_tools(), run_command=fail_after_writing
                )
            self.assertEqual(len(calls), 1)
            self.assertFalse(calls[0][1]["shell"])
            self.assertTrue(Path(request["outputs"][0]["local_target"]).is_file())
            self.assertTrue(Path(request["outputs"][1]["local_target"]).is_file())
            self.assertEqual(client.uploads, [])
            self.assertEqual(client._multipart, {})
            self.assertNotIn("synthetic/results/worker-receipt.json", client.objects)
            self.assertIn(
                "BIO_NEB_WORKER_TOOL_EXIT "
                '{"command_id":"pbrun-germline","returncode":17}',
                output.getvalue(),
            )
            self.assertIn("BIO_NEB_WORKER_TOOL_ERROR", output.getvalue())
            self.assertNotIn("BIO_NEB_WORKER_UPLOAD_START", output.getvalue())

    def test_prepared_input_drift_stops_before_pbrun_and_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, bodies = prepared_reference_consumption_request(root)
            drift_key = "synthetic/prepared/bwa_bwt"
            bodies[drift_key] += b"drift"
            client, calls, output = (
                OfflineObjectStorage(dict(bodies)),
                [],
                io.StringIO(),
            )
            with (
                patch.object(worker, "WORK_ROOT", root),
                contextlib.redirect_stdout(output),
                self.assertRaises(worker.WorkerError),
            ):
                worker.execute(
                    request,
                    client,
                    fixed_tools(),
                    run_command=fake_tool_runner(request, calls),
                )
            self.assertEqual(calls, [])
            self.assertEqual(client.uploads, [])
            self.assertNotIn(drift_key, output.getvalue())
            self.assertNotIn("synthetic/results/worker-receipt.json", client.objects)

    def test_bam_upload_failure_retains_residue_and_prevents_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, bodies = prepared_reference_consumption_request(root)
            failed_key = "synthetic/results/aligned.bam"
            client, calls = OfflineObjectStorage(dict(bodies), fail_key=failed_key), []
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(RuntimeError, "offline upload failure"),
            ):
                worker.execute(
                    request,
                    client,
                    fixed_tools(),
                    run_command=fake_tool_runner(request, calls),
                )
            self.assertEqual(client.uploads, [])
            self.assertEqual(len(client._multipart), 1)
            self.assertEqual(next(iter(client._multipart.values()))["key"], failed_key)
            self.assertNotIn("synthetic/results/worker-receipt.json", client.objects)

    def test_vcf_upload_failure_preserves_completed_bam_and_prevents_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "work"
            request, bodies = prepared_reference_consumption_request(root)
            failed_key = "synthetic/results/variants.vcf"
            client, calls = OfflineObjectStorage(dict(bodies), fail_key=failed_key), []
            with (
                patch.object(worker, "WORK_ROOT", root),
                self.assertRaisesRegex(RuntimeError, "offline upload failure"),
            ):
                worker.execute(
                    request,
                    client,
                    fixed_tools(),
                    run_command=fake_tool_runner(request, calls),
                )
            self.assertEqual(
                [item[0] for item in client.uploads],
                ["synthetic/results/aligned.bam"],
            )
            self.assertEqual(len(client._multipart), 1)
            self.assertEqual(next(iter(client._multipart.values()))["key"], failed_key)
            self.assertNotIn("synthetic/results/worker-receipt.json", client.objects)


if __name__ == "__main__":
    unittest.main()
