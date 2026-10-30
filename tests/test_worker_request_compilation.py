import hashlib
import json
import subprocess
import sys
import unittest

from pydantic import ValidationError

from bio_neb.parabricks import (
    reusable_object_storage_parabricks as worker,
)
from bio_neb.parabricks.worker_request import (
    PREPARED_ROLES,
    CompiledWorkerRequest,
    ConsumePreparedReferenceParameters,
    PlainFastaTarget,
    PrepareReferenceParameters,
    PublicReferenceDownload,
    RuntimeSmokeParameters,
    WorkerRequestContractError,
    compile_worker_request,
)
from tests.approved_execution_facts import (
    _consume_parameters,
    _prepare_parameters,
)


class WorkerRequestCompilationTests(unittest.TestCase):
    def _assert_canonical_compilation(
        self,
        parameters: (
            PrepareReferenceParameters
            | ConsumePreparedReferenceParameters
            | RuntimeSmokeParameters
        ),
        *,
        operation: str,
    ) -> dict[str, object]:
        first = compile_worker_request(parameters)
        second = compile_worker_request(parameters.model_copy(deep=True))
        self.assertEqual(first, second)
        self.assertEqual(first.identity.operation, operation)
        self.assertEqual(first.identity.schema_version, worker.REQUEST_SCHEMA_VERSION)
        self.assertLess(first.identity.byte_count, 64 * 1024)
        self.assertEqual(
            first.identity.sha256,
            hashlib.sha256(first.request_bytes).hexdigest(),
        )
        document = json.loads(first.request_bytes)
        self.assertEqual(worker.canonical_bytes(document), first.request_bytes)
        self.assertNotIn("secret", first.request_bytes.decode().casefold())
        return document

    def test_contract_imports_do_not_load_worker_or_provider_mechanics(self):
        code = """
import importlib.abc
import sys
class RejectMechanics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('boto3', 'botocore', 'bio_neb.parabricks.reusable_object_storage_parabricks')):
            raise AssertionError('contract imported executable/provider mechanics: ' + fullname)
sys.meta_path.insert(0, RejectMechanics())
from bio_neb.parabricks.worker_request import compile_worker_request, RuntimeSmokeParameters
from bio_neb.parabricks.worker_exchange import REQUEST_ADAPTER, RECEIPT_ADAPTER
from bio_neb.parabricks.reference_recipe import ReferencePreparationRequest
compiled = compile_worker_request(RuntimeSmokeParameters(request_id='isolated'))
assert REQUEST_ADAPTER.validate_json(compiled.request_bytes).operation == 'runtime_smoke'
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_connected_ingress_preserves_pre_extraction_request_identities(self):
        # Captured from the real compiler at the accepted baseline, before extraction;
        # schema 6 changed only the version label (identical bytes with /5 restored).
        # Schema 7 changed the label and added the preparation lineage's
        # `excluded_sequences`: with both undone the schema-6 bytes come back.
        cases = (
            (
                _prepare_parameters(),
                1919,
                "9a84a931544ebe5d640079c1d48be521cfe4cedbb9a18cf013316d1ab661c73e",
            ),
            (
                _consume_parameters(),
                3111,
                "4f54b638a0f4f393a1ee83d1892a85152cb92127fd4bb1aba160217e5cdea6c5",
            ),
            (
                RuntimeSmokeParameters(request_id="smoke-neutral"),
                153,
                "2dc23bfbca89d0e57ed852fee4b402ab9d4aa13fac7e633300421c866e485213",
            ),
        )
        for parameters, byte_count, digest in cases:
            with self.subTest(operation=type(parameters).__name__):
                compiled = compile_worker_request(parameters)
                schema_6 = compiled.request_bytes.replace(
                    b'"excluded_sequences":["NT_187515.1"],', b""
                ).replace(b"worker-request/7", b"worker-request/6")
                self.assertEqual(
                    (len(schema_6), hashlib.sha256(schema_6).hexdigest()),
                    (byte_count, digest),
                )
                parsed = worker.validate_request(compiled.request_bytes)
                self.assertEqual(parsed["request_id"], parameters.request_id)
                self.assertEqual(parsed["operation"], compiled.identity.operation)

    def test_prepare_reference_request_compiles_canonical_bytes_with_exact_eight_role_outputs(
        self,
    ) -> None:
        parameters = _prepare_parameters()
        request = self._assert_canonical_compilation(
            parameters, operation="prepare_reference"
        )
        self.assertEqual(
            [item["role"] for item in request["prepared_outputs"]],
            list(PREPARED_ROLES),
        )
        self.assertEqual(
            [item["role"] for item in request["inputs"]],
            ["reference_source"],
        )
        self.assertEqual(
            request["reference"]["source_role"],  # type: ignore[index]
            "reference_source",
        )
        self.assertEqual(
            request["reference"]["compression"],  # type: ignore[index]
            "gzip",
        )
        self.assertEqual(
            request["reference"]["accepted_plain_fasta"][  # type: ignore[index]
                "role"
            ],
            "genomic_fasta",
        )
        self.assertEqual(
            worker.PREPARATION_COMMANDS,
            ("bwa-index", "samtools-faidx", "samtools-dict"),
        )
        self.assertEqual(
            request["reference"]["commands"],  # type: ignore[index]
            list(worker.PREPARATION_COMMANDS),
        )
        self.assertEqual(
            request["receipt_output"]["role"],  # type: ignore[index]
            "worker_receipt",
        )

    def test_the_preparation_lineage_names_the_records_to_drop(self):
        """Owner, #464 ("primey for the ref we are running"): the CPU build indexes the
        Run's reference sequences; the request carries the names it drops, an
        exclusion list that stays under the injected-file bound for a fragmented
        assembly. A blank, spaced or repeated name names no FASTA record."""
        request = self._assert_canonical_compilation(
            _prepare_parameters(), operation="prepare_reference"
        )
        self.assertEqual(
            request["reference"]["excluded_sequences"],  # type: ignore[index]
            ["NT_187515.1"],
        )
        # GRCh38.p14 drops about 500 alt loci and patches; 2,000 such names fit.
        many = tuple(f"NW_{index:09d}.1" for index in range(2000))
        compiled = compile_worker_request(_prepare_parameters(excluded_sequences=many))
        self.assertLess(compiled.identity.byte_count, 64 * 1024)
        self.assertEqual(
            worker.validate_request(compiled.request_bytes)["reference"][
                "excluded_sequences"
            ],
            list(many),
        )
        for names in (("",), ("NT_1.1 alt",), ("NT_1.1", "NT_1.1")):
            with self.subTest(names=names):
                with self.assertRaisesRegex(
                    WorkerRequestContractError,
                    "worker_request_excluded_sequences_invalid",
                ):
                    compile_worker_request(
                        _prepare_parameters(excluded_sequences=names)
                    )
                wire = json.loads(compiled.request_bytes)
                wire["reference"]["excluded_sequences"] = list(names)
                with self.assertRaisesRegex(worker.WorkerError, "excluded reference"):
                    worker.validate_request(worker.canonical_bytes(wire))

    def test_public_source_request_names_the_download_and_no_plain_identity(self):
        stored = _prepare_parameters()
        download = PublicReferenceDownload(
            role="reference_source",
            url="https://publisher.example.test/reference.fna.gz",
            byte_count=101,
            md5="0" * 32,
            object_key="raw/neutral/reference.fna.gz",
            local_target="/work/source/reference.fa.gz",
        )
        parameters = stored.model_copy(
            update={
                "source": download,
                "plain_fasta": PlainFastaTarget(
                    local_target=stored.plain_fasta.local_target
                ),
            }
        )
        request = self._assert_canonical_compilation(
            parameters, operation="prepare_reference"
        )
        self.assertEqual(request["inputs"], [])
        self.assertEqual(request["download"], download.model_dump(mode="json"))
        self.assertEqual(
            request["reference"]["plain_fasta"],  # type: ignore[index]
            {"role": "genomic_fasta", "local_target": "/work/reference/reference.fa"},
        )
        self.assertNotIn("accepted_plain_fasta", request["reference"])  # type: ignore[operator]
        for refused, reason in (
            (
                parameters.model_copy(update={"plain_fasta": stored.plain_fasta}),
                "worker_request_plain_fasta_invalid",
            ),
            (
                stored.model_copy(update={"plain_fasta": parameters.plain_fasta}),
                "worker_request_plain_fasta_invalid",
            ),
            (
                parameters.model_copy(
                    update={
                        "source": download.model_copy(
                            update={"url": "http://publisher.example.test/r.fna.gz"}
                        )
                    }
                ),
                "worker_request_source_url_invalid",
            ),
            (
                parameters.model_copy(
                    update={
                        "source": download.model_copy(
                            update={"object_key": "runs/neutral/receipts/worker.json"}
                        )
                    }
                ),
                "worker_request_identity_collision",
            ),
        ):
            with (
                self.subTest(reason=reason),
                self.assertRaisesRegex(WorkerRequestContractError, reason),
            ):
                compile_worker_request(refused)

    def test_consume_prepared_reference_request_compiles_canonical_bytes_with_fixed_input_output_order(
        self,
    ) -> None:
        parameters = _consume_parameters()
        request = self._assert_canonical_compilation(
            parameters, operation="consume_prepared_reference"
        )
        self.assertEqual(
            [item["role"] for item in request["inputs"]],
            ["read_1", "read_2", *PREPARED_ROLES],
        )
        self.assertEqual(
            [item["role"] for item in request["outputs"]],
            ["aligned_bam", "VCF"],
        )
        expected_parabricks_argv = (
            "pbrun",
            "germline",
            "--ref",
            "/work/reference/reference.fa",
            "--in-fq",
            "/work/inputs/read-1.fastq.gz",
            "/work/inputs/read-2.fastq.gz",
            "--out-bam",
            "/work/outputs/aligned.bam",
            "--out-variants",
            "/work/outputs/variants.vcf",
            "--read-group-sm",
            "neutral-sample",
            "--num-gpus",
            "1",
        )
        self.assertEqual(parameters.parabricks_argv, expected_parabricks_argv)
        self.assertEqual(request["parabricks_argv"], list(expected_parabricks_argv))
        self.assertEqual(
            request["receipt_output"]["role"],  # type: ignore[index]
            "worker_receipt",
        )

    def test_runtime_smoke_request_compiles_canonical_bytes(self) -> None:
        # Canonical compilation is offline contract proof, not live worker proof.
        self._assert_canonical_compilation(
            RuntimeSmokeParameters(request_id="smoke-neutral"),
            operation="runtime_smoke",
        )

    def test_gpu_request_retains_only_explicit_cross_bucket_input_bindings(
        self,
    ) -> None:
        parameters = _consume_parameters()
        inputs = tuple(
            item.model_copy(update={"bucket": "prepared-cache-bucket"})
            if item.role in PREPARED_ROLES
            else item
            for item in parameters.inputs
        )
        compiled = compile_worker_request(
            parameters.model_copy(update={"inputs": inputs})
        )
        wire = json.loads(compiled.request_bytes)
        wire_by_role = {item["role"]: item for item in wire["inputs"]}
        self.assertNotIn("bucket", wire_by_role["read_1"])
        self.assertNotIn("bucket", wire_by_role["read_2"])
        self.assertEqual(
            {wire_by_role[role]["bucket"] for role in PREPARED_ROLES},
            {"prepared-cache-bucket"},
        )

    def test_parameter_schemas_forbid_secret_and_gpu_preparation_fields(
        self,
    ) -> None:
        prepared = _prepare_parameters().model_dump(mode="python")
        prepared["aws_secret_access_key"] = "forbidden"
        with self.assertRaises(ValidationError):
            PrepareReferenceParameters.model_validate(prepared, strict=True)

        consume = _consume_parameters().model_dump(mode="python")
        consume["preparation_commands"] = ("bwa-index",)
        with self.assertRaises(ValidationError):
            ConsumePreparedReferenceParameters.model_validate(consume, strict=True)

    def test_compiler_rejects_untyped_or_oversized_requests(self) -> None:
        with self.assertRaisesRegex(
            WorkerRequestContractError, "worker_request_parameters_invalid"
        ):
            compile_worker_request({"operation": "runtime_smoke"})  # type: ignore[arg-type]

        with self.assertRaisesRegex(
            WorkerRequestContractError, "worker_request_oversized"
        ):
            compile_worker_request(_prepare_parameters(request_id="x" * (64 * 1024)))

    def test_compiled_worker_request_cannot_be_forged(self) -> None:
        compiled = compile_worker_request(
            RuntimeSmokeParameters(request_id="smoke-neutral")
        )
        with self.assertRaises(TypeError):
            CompiledWorkerRequest(  # type: ignore[call-arg]
                request_bytes=compiled.request_bytes,
                identity=compiled.identity,
            )

    def test_prepare_request_rejects_invalid_source_sha256_before_bytes(
        self,
    ) -> None:
        bad_source = _prepare_parameters().model_copy(
            update={
                "source": _prepare_parameters().source.model_copy(
                    update={"sha256": "not-a-hash"}
                )
            }
        )
        with self.assertRaisesRegex(WorkerRequestContractError, "sha256"):
            compile_worker_request(bad_source)

    def test_consume_request_rejects_noncanonical_local_target_before_bytes(
        self,
    ) -> None:
        consume = _consume_parameters()
        changed_inputs = list(consume.inputs)
        changed_inputs[0] = changed_inputs[0].model_copy(
            update={"local_target": "/work/inputs/../changed.fastq.gz"}
        )
        with self.assertRaisesRegex(WorkerRequestContractError, "local_target"):
            compile_worker_request(
                consume.model_copy(update={"inputs": tuple(changed_inputs)})
            )

    def test_consume_request_rejects_parabricks_binding_drift_before_bytes(
        self,
    ) -> None:
        consume = _consume_parameters()
        with self.assertRaisesRegex(WorkerRequestContractError, "parabricks_binding"):
            compile_worker_request(
                consume.model_copy(
                    update={"parabricks_argv": (*consume.parabricks_argv[:-1], "2")}
                )
            )


if __name__ == "__main__":
    unittest.main()
