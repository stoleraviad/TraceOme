import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path

from bio_neb.parabricks.worker_exchange import RECEIPT_ADAPTER
from bio_neb.parabricks.worker_request import (
    RuntimeSmokeParameters,
    compile_worker_request,
)
from tests.approved_execution_facts import (
    _consume_parameters,
    _prepare_parameters,
)

# -I -S prevents the checkout's editable installation and PYTHONPATH from making
# missing Docker COPY inputs accidentally available to this isolated process.
_PACKAGED_PROBE = r"""
import contextlib
import hashlib
import importlib.util
import io
import json
import runpy
import sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
root = Path(sys.argv[1])
worker_path = root / 'reusable_object_storage_parabricks.py'
name = 'bio_neb.parabricks.reusable_object_storage_parabricks'
spec = importlib.util.spec_from_file_location(name, worker_path)
worker = importlib.util.module_from_spec(spec)
sys.modules[name] = worker
spec.loader.exec_module(worker)
if sys.argv[5] == 'drift':
    with contextlib.redirect_stdout(io.StringIO()) as refused_output:
        assert worker._entrypoint([str(Path(sys.argv[3]) / 'runtime_smoke.json')]) == 1
    assert 'worker exchange dependency changed' in refused_output.getvalue()
    assert 'BIO_NEB_WORKER_REQUEST_ACCEPTED' not in refused_output.getvalue()
    try:
        worker.validate_request((Path(sys.argv[3]) / 'runtime_smoke.json').read_bytes())
    except worker.WorkerError as error:
        assert str(error) == 'worker exchange dependency changed', str(error)
        try:
            worker.worker_source_bytes()
        except worker.WorkerError:
            print('dependency-drift-refused')
            raise SystemExit(0)
    raise AssertionError('changed packaged dependency was accepted')
assert hashlib.sha256(worker.worker_source_bytes()).hexdigest() == sys.argv[6]
for operation in ('prepare_reference', 'consume_prepared_reference', 'runtime_smoke'):
    request = worker.load_request(Path(sys.argv[3]) / (operation + '.json'))
    assert request['operation'] == operation
fakes = runpy.run_path(sys.argv[4])
worker._tool_identity = lambda operation: fakes['fixed_tools']()
with contextlib.redirect_stdout(io.StringIO()) as smoke_output:
    assert worker._entrypoint([str(Path(sys.argv[3]) / 'runtime_smoke.json')]) == 0
assert 'BIO_NEB_PARABRICKS_WORKER_SMOKE_COMPLETE' in smoke_output.getvalue()
assert sys.argv[6] in smoke_output.getvalue()
results = []
for operation, factory in [('prepare_reference', 'reference_preparation_request'),
                           ('consume_prepared_reference', 'prepared_reference_consumption_request')]:
    work = root / operation / 'work'
    worker.WORK_ROOT = work
    request, bodies = fakes[factory](work)
    client = fakes['OfflineObjectStorage'](dict(bodies))
    calls = []
    with contextlib.redirect_stdout(io.StringIO()):
        result = worker.execute(request, client, fakes['fixed_tools'](),
            run_command=fakes['fake_tool_runner'](request, calls))
    persisted = json.loads(client.objects[request['receipt_output']['object_key']])
    assert persisted['operation'] == operation
    assert persisted['authority'] == 'non-authoritative-worker-observation'
    assert len(calls) == (3 if operation == 'prepare_reference' else 1)
    assert result == dict(persisted, receipt_object=result['receipt_object'])
    results.append(persisted)
print(json.dumps(results))
"""


def _image_definition() -> tuple[str, str, str, str]:
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "containers/parabricks-worker/Dockerfile").read_text()
    requirements = (root / "containers/parabricks-worker/requirements.txt").read_text()
    excludes = (
        root / "containers/parabricks-worker/Dockerfile.dockerignore"
    ).read_text()
    worker_source = (
        root / "src/bio_neb/parabricks/reusable_object_storage_parabricks.py"
    ).read_text()
    return dockerfile, requirements, excludes, worker_source


class ReusableWorkerImageContractTests(unittest.TestCase):
    def test_docker_packaged_ingress_results_and_dependency_identity(self):
        repo = Path(__file__).resolve().parents[1]
        dockerfile, _, excludes, _ = _image_definition()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            # Use the actual declared source-to-destination layout, not a second
            # hand-maintained inventory of what the worker needs to import.
            for line in dockerfile.splitlines():
                if not line.startswith("COPY src/"):
                    continue
                _, source, destination = line.split()
                self.assertIn("!" + source, excludes)
                target = package / destination.removeprefix("/opt/bio-neb/")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo / source, target)
            requests = root / "requests"
            requests.mkdir()
            for parameters in (
                _prepare_parameters(),
                _consume_parameters(),
                RuntimeSmokeParameters(request_id="smoke-neutral"),
            ):
                compiled = compile_worker_request(parameters)
                (requests / (compiled.identity.operation + ".json")).write_bytes(
                    compiled.request_bytes
                )
            worker_path = package / "reusable_object_storage_parabricks.py"
            digest = hashlib.sha256(worker_path.read_bytes()).hexdigest()
            command = [
                sys.executable,
                "-I",
                "-S",
                "-c",
                _PACKAGED_PROBE,
                str(package),
                sysconfig.get_paths()["purelib"],
                str(requests),
                str(repo / "tests/reusable_parabricks_worker_fakes.py"),
                "normal",
                digest,
            ]
            result = subprocess.run(
                command,
                cwd=root,
                env={**os.environ, "PYTHONPATH": ""},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            receipts = json.loads(result.stdout)
            for receipt in receipts:
                parsed = RECEIPT_ADAPTER.validate_python(receipt, strict=True)
                self.assertEqual(
                    parsed.authority, "non-authoritative-worker-observation"
                )
            self.assertEqual(
                [entry["role"] for entry in receipts[0]["prepared_uploads"]],
                [
                    "genomic_fasta",
                    "fasta_fai",
                    "reference_dict",
                    "bwa_amb",
                    "bwa_ann",
                    "bwa_bwt",
                    "bwa_pac",
                    "bwa_sa",
                ],
            )
            self.assertEqual(
                [entry["role"] for entry in receipts[1]["uploads"]],
                ["aligned_bam", "VCF"],
            )
            # Any extracted source changing alone must refuse the executable's
            # ingress and source attestation without changing its own raw SHA.
            for relative in (
                "bio_neb/parabricks/worker_exchange.py",
                "bio_neb/contracts.py",
                "bio_neb/fasta_selection.py",
            ):
                dependency = package / relative
                original = dependency.read_bytes()
                dependency.write_bytes(original + b"\n# changed packaged behavior\n")
                command[-2] = "drift"
                refused = subprocess.run(
                    command, cwd=root, capture_output=True, text=True
                )
                self.assertEqual(refused.returncode, 0, refused.stderr)
                self.assertEqual(refused.stdout.strip(), "dependency-drift-refused")
                self.assertEqual(
                    hashlib.sha256(worker_path.read_bytes()).hexdigest(), digest
                )
                dependency.write_bytes(original)

    def test_worker_image_pins_base_toolchain_and_dependencies(self):
        dockerfile, requirements, _, _ = _image_definition()
        self.assertIn("4.7.0-1@sha256:28f219", dockerfile)
        self.assertIn("bwa=0.7.17-6", dockerfile)
        for dependency in (
            "boto3==1.43.67",
            "botocore==1.43.67",
            "pydantic==2.13.4",
        ):
            self.assertIn(dependency, requirements)

    def test_worker_image_labels_entrypoint_and_exact_source_contract(self):
        dockerfile, _, excludes, worker_source = _image_definition()
        self.assertIn(
            'io.bio-neb.worker.request-schema="bio-neb.parabricks-worker-request/7"',
            dockerfile,
        )
        self.assertIn(
            'io.bio-neb.worker.receipt-schema="bio-neb.parabricks-worker-receipt/5"',
            dockerfile,
        )
        self.assertIn(
            'io.bio-neb.worker.module="bio_neb.parabricks.'
            'reusable_object_storage_parabricks"',
            dockerfile,
        )
        self.assertIn(
            'ENTRYPOINT ["/opt/bio-neb/venv/bin/python", '
            '"/opt/bio-neb/reusable_object_storage_parabricks.py"]',
            dockerfile,
        )
        self.assertIn(
            "!src/bio_neb/parabricks/reusable_object_storage_parabricks.py",
            excludes,
        )
        for shared in ("contracts.py", "fasta_selection.py"):
            self.assertIn(f"!src/bio_neb/{shared}", excludes)
            self.assertIn(
                f"COPY src/bio_neb/{shared} /opt/bio-neb/bio_neb/{shared}",
                dockerfile,
            )
        boundary = (
            "Nebius Object Storage is the provider; compatible S3/AWS names "
            "imply no Amazon resource."
        )
        self.assertEqual(worker_source.count(boundary), 1)

    def test_worker_image_excludes_secrets_and_donor_specific_values(self):
        dockerfile, requirements, excludes, _ = _image_definition()
        self.assertIn("**/.env", excludes)
        self.assertNotIn("reusable_" + "s3_parabricks", dockerfile + excludes)
        combined = (dockerfile + requirements).casefold()
        for forbidden in (
            "hg001",
            "hg002",
            "deepvariant",
            "sample_id",
            "aws_secret_access_key",
        ):
            self.assertNotIn(forbidden, combined)


if __name__ == "__main__":
    unittest.main()
