"""Offline fakes for reusable Parabricks worker contract tests.

These helpers exercise local worker protocol mechanics only.  They never prove
live Object Storage, provider, container, CPU, or GPU behavior.
"""

import gzip
import hashlib
import io
import subprocess
from pathlib import Path

from bio_neb.parabricks import (
    reusable_object_storage_parabricks as worker,
)


def sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


class CountingRead:
    def __init__(self, stream, path: Path, counts: dict[Path, int]) -> None:
        self._stream = stream
        self._path = path
        self._counts = counts

    def read(self, size=-1):
        body = self._stream.read(size)
        self._counts[self._path] = self._counts.get(self._path, 0) + len(body)
        return body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._stream.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def counting_path_open(counts: dict[Path, int]):
    original = Path.open

    def open_counted(path: Path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        if "r" in mode and "b" in mode:
            return CountingRead(stream, path, counts)
        return stream

    return open_counted


class OfflineObjectStorage:
    """Provider-shaped protocol fake; never live Object Storage evidence."""

    def __init__(
        self, objects: dict[str, bytes], *, fail_key: str | None = None
    ) -> None:
        self.objects = objects
        self.fail_key = fail_key
        self.uploads: list[tuple[str, list[int], dict[str, str]]] = []
        self._multipart: dict[str, dict[str, object]] = {}

    def list_objects_v2(self, **kwargs):
        return {"KeyCount": 0}

    def get_object(self, *, Bucket, Key):
        del Bucket
        return {"Body": io.BytesIO(self.objects[Key])}

    def create_multipart_upload(self, *, Bucket, Key, Metadata):
        del Bucket
        upload_id = f"offline-upload-{len(self._multipart) + 1}"
        self._multipart[upload_id] = {
            "key": Key,
            "metadata": Metadata,
            "parts": [],
        }
        return {"UploadId": upload_id}

    def upload_part(self, *, Bucket, Key, UploadId, PartNumber, Body):
        del Bucket, Key
        self._multipart[UploadId]["parts"].append(bytes(Body))
        return {"ETag": f'"offline-part-{PartNumber}"'}

    def complete_multipart_upload(
        self, *, Bucket, Key, UploadId, MultipartUpload, IfNoneMatch
    ):
        del Bucket, MultipartUpload
        assert IfNoneMatch == "*"
        if Key == self.fail_key:
            raise RuntimeError("injected offline upload failure")
        retained = self._multipart.pop(UploadId)
        parts = retained["parts"]
        body = b"".join(parts)
        metadata = retained["metadata"]
        assert metadata["checksum-sha256"] == sha256(body)
        self.objects[Key] = body
        self.uploads.append((Key, [len(part) for part in parts], metadata))

    def abort_multipart_upload(self, **kwargs):
        self._multipart.pop(kwargs["UploadId"], None)


def target(role: str, key: str, path: Path) -> dict[str, object]:
    return {"role": role, "object_key": key, "local_target": str(path)}


def exact(role: str, key: str, path: Path, body: bytes) -> dict[str, object]:
    return {
        **target(role, key, path),
        "byte_count": len(body),
        "sha256": sha256(body),
    }


def prepared_paths(fasta: Path) -> dict[str, Path]:
    return {
        "genomic_fasta": fasta,
        "fasta_fai": Path(str(fasta) + ".fai"),
        "reference_dict": fasta.with_suffix(".dict"),
        **{
            f"bwa_{extension}": Path(str(fasta) + f".{extension}")
            for extension in ("amb", "ann", "bwt", "pac", "sa")
        },
    }


def reference_preparation_request(
    root: Path,
    *,
    plain: bytes = b">synthetic-reference\nACGT\n",
    excluded: tuple[str, ...] = (),
    selected: bytes | None = None,
):
    """``selected`` is the plain FASTA without the ``excluded`` records, as the caller
    states it (default: ``plain``); the stored source's accepted identity is its."""
    compressed = gzip.compress(plain)
    selected = plain if selected is None else selected
    source = root / "source" / "reference.fa.gz"
    fasta = root / "prepared" / "reference.fa"
    paths = prepared_paths(fasta)
    request = {
        "schema_version": worker.REQUEST_SCHEMA_VERSION,
        "operation": "prepare_reference",
        "request_id": "synthetic-reference-preparation",
        "bucket": "synthetic-offline-bucket",
        "endpoint_url": "https://storage.example.test",
        "region": "synthetic-region",
        "output_prefix": "synthetic/prepared/",
        "workflow": "bwa-samtools-reference-preparation",
        "inputs": [
            exact(
                "reference_source",
                "synthetic/source/reference.fa.gz",
                source,
                compressed,
            )
        ],
        "reference": {
            "source_role": "reference_source",
            "compression": "gzip",
            "excluded_sequences": list(excluded),
            "accepted_plain_fasta": {
                "role": "genomic_fasta",
                "local_target": str(fasta),
                "byte_count": len(selected),
                "sha256": sha256(selected),
            },
            "commands": list(worker.PREPARATION_COMMANDS),
        },
        "prepared_outputs": [
            target(role, f"synthetic/prepared/{role}", paths[role])
            for role in worker.PREPARED_ROLES
        ],
        "receipt_output": target(
            "worker_receipt",
            "synthetic/prepared/worker-receipt.json",
            root / "receipts" / "prepare.json",
        ),
    }
    return request, {"synthetic/source/reference.fa.gz": compressed}


PUBLIC_SOURCE_URL = "https://publisher.example.test/genomes/reference.fna.gz"


def public_reference_preparation_request(
    root: Path,
    *,
    plain: bytes = b">synthetic-reference\nACGT\n",
    excluded: tuple[str, ...] = (),
):
    """The same preparation with the source fetched from its approved publisher URL."""
    request, bodies = reference_preparation_request(
        root, plain=plain, excluded=excluded
    )
    compressed = bodies.pop("synthetic/source/reference.fa.gz")
    stored = request.pop("inputs")[0]
    accepted = request["reference"].pop("accepted_plain_fasta")
    request["inputs"] = []
    request["download"] = {
        "role": "reference_source",
        "url": PUBLIC_SOURCE_URL,
        "byte_count": len(compressed),
        "md5": hashlib.md5(compressed).hexdigest(),
        "object_key": "synthetic/prepared/source/reference.fa.gz",
        "local_target": stored["local_target"],
    }
    request["reference"]["plain_fasta"] = {
        "role": "genomic_fasta",
        "local_target": accepted["local_target"],
    }
    return request, compressed


class OfflinePublisher:
    """A publisher-shaped fetch fake; never live network evidence."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.requests: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout: float):
        self.requests.append((url, timeout))
        return io.BytesIO(self.body)


def prepared_reference_consumption_request(root: Path):
    fasta = root / "prepared" / "reference.fa"
    paths = prepared_paths(fasta)
    prepared_bodies = {
        role: f"synthetic-{role}-exact\n".encode() for role in worker.PREPARED_ROLES
    }
    prepared_bodies["genomic_fasta"] = b">synthetic-reference\nACGT\n"
    read_bodies = {"read_1": b"synthetic-forward\n", "read_2": b"synthetic-reverse\n"}
    inputs = [
        exact(
            role,
            f"synthetic/reads/{role}.fq",
            root / "reads" / f"{role}.fq",
            body,
        )
        for role, body in read_bodies.items()
    ] + [
        exact(
            role,
            f"synthetic/prepared/{role}",
            paths[role],
            prepared_bodies[role],
        )
        for role in worker.PREPARED_ROLES
    ]
    outputs = root / "outputs"
    request = {
        "schema_version": worker.REQUEST_SCHEMA_VERSION,
        "operation": "consume_prepared_reference",
        "request_id": "synthetic-prepared-reference-consumption",
        "bucket": "synthetic-offline-bucket",
        "endpoint_url": "https://storage.example.test",
        "region": "synthetic-region",
        "output_prefix": "synthetic/results/",
        "workflow": "nvidia-parabricks-germline",
        "resources": {"gpu_count": 1, "shm_size_bytes": 14 * 1024**3},
        "inputs": inputs,
        "outputs": [
            target(
                "aligned_bam", "synthetic/results/aligned.bam", outputs / "aligned.bam"
            ),
            target("VCF", "synthetic/results/variants.vcf", outputs / "variants.vcf"),
        ],
        "receipt_output": target(
            "worker_receipt",
            "synthetic/results/worker-receipt.json",
            root / "receipts" / "consume.json",
        ),
        "parabricks_argv": [
            "pbrun",
            "germline",
            "--ref",
            str(fasta),
            "--in-fq",
            str(root / "reads" / "read_1.fq"),
            str(root / "reads" / "read_2.fq"),
            "--out-bam",
            str(outputs / "aligned.bam"),
            "--out-variants",
            str(outputs / "variants.vcf"),
            "--read-group-sm",
            "synthetic-sample",
            "--num-gpus",
            "1",
        ],
    }
    objects = {
        item["object_key"]: body
        for item, body in zip(inputs[:2], read_bodies.values(), strict=True)
    }
    objects.update(
        {f"synthetic/prepared/{role}": body for role, body in prepared_bodies.items()}
    )
    return request, objects


def fixed_tools() -> dict[str, dict[str, str]]:
    return {
        "python": {"path": "/opt/bio-neb/venv/bin/python", "version": "3.10"},
        "bwa": {"path": "/usr/bin/bwa", "version": "0.7.17"},
        "samtools": {"path": "/usr/bin/samtools", "version": "1.13"},
        "pbrun": {
            "path": "/usr/local/parabricks/pbrun",
            "version": "4.7.0-1",
        },
    }


def fake_tool_runner(
    request: dict[str, object], calls: list[tuple], *, omit: str | None = None
):
    """Return an injected executable fake; no subprocess or worker is launched."""

    def run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        if argv[:2] == ("/usr/bin/bwa", "index"):
            for extension in ("amb", "ann", "bwt", "pac", "sa"):
                if f"bwa_{extension}" != omit:
                    Path(f"{argv[2]}.{extension}").write_bytes(extension.encode())
        elif argv[:2] == ("/usr/bin/samtools", "faidx") and omit != "fasta_fai":
            Path(f"{argv[2]}.fai").write_bytes(b"fai")
        elif argv[:2] == ("/usr/bin/samtools", "dict") and omit != "reference_dict":
            Path(argv[argv.index("-o") + 1]).write_bytes(b"dict")
        elif argv[:2] == ("/usr/local/parabricks/pbrun", "germline"):
            Path(request["outputs"][0]["local_target"]).write_bytes(b"B" * 13)
            Path(request["outputs"][1]["local_target"]).write_bytes(b"V" * 9)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    return run
