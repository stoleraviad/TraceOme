"""Shared secret-free runtime selections and liveness facts."""

from __future__ import annotations

import re
from decimal import Decimal

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from bio_neb.contracts import FrozenContract
from bio_neb.providers.runtime import _canonical_text

ATTESTED_RUNTIME_RECEIPT_SHA256 = (
    "12f564eafc27802c23d152ad61d2ce81d79fbd49b78a1a6671673e843d4a4878"
)

ATTESTED_PROVIDER_IMAGE = (
    "cr.eu-north1.nebius.cloud/e00md63vxjxams912c/"
    "bio-neb-parabricks-worker:i464-20261029T150408Z-0447bb17"
)

ATTESTED_IMAGE_INDEX = (
    "sha256:1c1f7eb88b71b56c42e9a31e36a734123b0f1bba9bc97079e03d47fc3afeda5b"
)

ATTESTED_RUNTIME_IMAGE = (
    "cr.eu-north1.nebius.cloud/e00md63vxjxams912c/"
    "bio-neb-parabricks-worker@sha256:"
    "4121c9d26c21cd902938322ce314a63ff26340d6caac1173e0c3ed08ccb7927c"
)

ATTESTED_IMAGE_CHILD = (
    "sha256:4121c9d26c21cd902938322ce314a63ff26340d6caac1173e0c3ed08ccb7927c"
)

ATTESTED_WORKER_SHA256 = (
    "0447bb178c2b6f35322e2fc9169655c2df550ea232c74021c21eedde95db3a56"
)

ATTESTED_BWA_VERSION = "Version: 0.7.17-r1188"

ATTESTED_SAMTOOLS_VERSION = "samtools 1.13"

ATTESTED_PYTHON_VERSION = "3.10.12"

ATTESTED_PBRUN_VERSION = "pbrun: 4.7.0-1"


class RuntimeImage(FrozenContract):
    """Attested image defaults; alternatives remain plain, structurally valid values."""

    provider_image_reference: StrictStr = ATTESTED_PROVIDER_IMAGE
    provider_image_index_digest: StrictStr = Field(
        default=ATTESTED_IMAGE_INDEX, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    worker_runtime_image: StrictStr = ATTESTED_RUNTIME_IMAGE
    worker_sha256: StrictStr = Field(
        default=ATTESTED_WORKER_SHA256, pattern=r"^[0-9a-f]{64}$"
    )

    @field_validator("*", mode="after")
    @classmethod
    def canonical_runtime_values(cls, value):
        return (
            _canonical_text(value, "runtime value") if isinstance(value, str) else value
        )

    @model_validator(mode="after")
    def same_image_repository(self):
        tag_repository, separator, tag = self.provider_image_reference.rpartition(":")
        image_repository, marker, digest = self.worker_runtime_image.rpartition(
            "@sha256:"
        )
        if not separator or "/" not in tag_repository or not tag:
            raise ValueError("provider_image_reference must be tagged")
        if (
            marker != "@sha256:"
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("worker_runtime_image must be digest-pinned")
        if tag_repository != image_repository:
            raise ValueError("provider tag changed the digest-pinned repository")
        return self


class CpuReferenceRuntime(RuntimeImage):
    platform: StrictStr = "cpu-d3"
    preset: StrictStr = "4vcpu-16gb"
    disk_size: StrictStr = "100Gi"
    restart_policy: StrictStr = "never"
    bwa_version: StrictStr = ATTESTED_BWA_VERSION
    samtools_version: StrictStr = ATTESTED_SAMTOOLS_VERSION
    worker_module: StrictStr = "bio_neb.parabricks.reusable_object_storage_parabricks"

    @field_validator("disk_size")
    @classmethod
    def valid_disk_size(cls, value: str) -> str:
        _disk_size_bytes(value)
        return value

    @property
    def disk_size_bytes(self) -> int:
        return _disk_size_bytes(self.disk_size)


class H100Runtime(RuntimeImage):
    platform: StrictStr = "gpu-h100-sxm"
    preset: StrictStr = "1gpu-16vcpu-200gb"
    disk_size: StrictStr = "250Gi"
    restart_policy: StrictStr = "never"
    gpu_count: StrictInt = Field(default=1, ge=1)
    shared_memory_bytes: StrictInt = Field(default=17179869184, ge=1)
    recipe_id: StrictStr
    resource_profile_id: StrictStr
    cpu_count: StrictInt = Field(default=16, ge=1)
    memory_gib: StrictInt = Field(default=200, ge=1)
    accelerator_memory_gib: StrictInt = Field(default=80, ge=1)
    python_path: StrictStr = "/opt/bio-neb/venv/bin/python"
    python_version: StrictStr = ATTESTED_PYTHON_VERSION
    pbrun_path: StrictStr = "/usr/local/parabricks/pbrun"
    pbrun_version: StrictStr = ATTESTED_PBRUN_VERSION

    @field_validator("disk_size")
    @classmethod
    def valid_disk_size(cls, value: str) -> str:
        _disk_size_bytes(value)
        return value

    @property
    def disk_size_bytes(self) -> int:
        return _disk_size_bytes(self.disk_size)


def _disk_size_bytes(value: str) -> int:
    """Translate the selected CLI quantity into the fetched envelope's bytes."""
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]i|[kMGTPE])?", value)
    if match is None:
        raise ValueError("disk_size must be a positive byte quantity")
    number, unit = match.groups()
    exponent = "KMGTPE".index(unit[0].upper()) + 1 if unit else 0
    count = (
        Decimal(number) * (1024 if unit and unit.endswith("i") else 1000) ** exponent
    )
    if count <= 0 or count != count.to_integral_value():
        raise ValueError("disk_size must resolve to a positive whole byte count")
    return int(count)


class Runtimes(FrozenContract):
    cpu: CpuReferenceRuntime = CpuReferenceRuntime()
    h100: H100Runtime
    fallbacks: tuple[H100Runtime, ...] = ()


def runtime_matches_package(runtime: H100Runtime, package) -> bool:
    """One relationship shared by proposal admission and the composition gate."""
    return package is not None and (
        runtime.recipe_id,
        runtime.resource_profile_id,
        runtime.gpu_count,
        runtime.cpu_count,
        runtime.memory_gib,
        runtime.accelerator_memory_gib,
    ) == (
        package.workflow.workflow_id,
        package.resource.profile_id,
        package.resource.accelerator_count,
        package.resource.cpu_count,
        package.resource.memory_gib,
        package.resource.accelerator_memory_gib,
    )


class ExecutionCompositionRuntime(RuntimeImage):
    """Legacy receipt projection; #59 removes it with the old binding."""

    receipt_sha256: StrictStr | None = ATTESTED_RUNTIME_RECEIPT_SHA256
    capability_snapshot_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    nebius_cli_version: StrictStr
    worker_request_schema: StrictStr
    bwa_path: StrictStr
    bwa_version: StrictStr
    samtools_path: StrictStr
    samtools_version: StrictStr
    python_path: StrictStr
    python_version: StrictStr
    pbrun_path: StrictStr
    pbrun_version: StrictStr
