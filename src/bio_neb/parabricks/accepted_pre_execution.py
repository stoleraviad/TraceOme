"""Contact-free binding of the accepted worker capability for planning."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from bio_neb.journey import Journey
from bio_neb.parabricks import (
    reusable_object_storage_parabricks as accepted_worker,
)
from bio_neb.parabricks.capability_policy import (
    CapabilityPolicyImage,
    CapabilityPolicyRecipe,
    CapabilityPolicyResource,
    CapabilityPolicyTool,
    CapabilitySnapshot,
    CapabilitySnapshotAuthoring,
    CapabilitySnapshotError,
    LocalCapabilitySourceInput,
    compile_capability_snapshot_value,
)
from bio_neb.parabricks.execution_runtime import (
    ATTESTED_BWA_VERSION,
    ATTESTED_IMAGE_CHILD,
    ATTESTED_IMAGE_INDEX,
    ATTESTED_PBRUN_VERSION,
    ATTESTED_PROVIDER_IMAGE,
    ATTESTED_SAMTOOLS_VERSION,
    ATTESTED_WORKER_SHA256,
)

if TYPE_CHECKING:
    pass


_CAPABILITY_VERSION = "bio-neb-parabricks-germline-t125j/1"
_WORKER_MODULE = "bio_neb.parabricks.reusable_object_storage_parabricks"
_WORKER_RETRIEVED_AT = "2026-09-16T23:05:28Z"


class AcceptedPreExecutionBindingError(ValueError):
    """Typed internal refusal for drift in accepted application-owned facts."""

    def __init__(self, code: str, fields: tuple[str, ...] = ()) -> None:
        super().__init__(code)
        self.code = code
        self.fields = fields


def _worker_path() -> Path:
    worker_file = accepted_worker.__file__
    if worker_file is None or accepted_worker.__name__ != _WORKER_MODULE:
        raise AcceptedPreExecutionBindingError(
            "accepted_worker_source_unavailable", (_WORKER_MODULE,)
        )
    try:
        accepted_worker.worker_source_bytes()
        return Path(worker_file).resolve(strict=True)
    except (OSError, accepted_worker.WorkerError) as error:
        raise AcceptedPreExecutionBindingError(
            "accepted_worker_source_unavailable", (_WORKER_MODULE,)
        ) from error


def _accepted_capability_snapshot(journey: Journey) -> CapabilitySnapshot:
    worker_path = _worker_path()
    source = LocalCapabilitySourceInput(
        kind="local_bytes",
        locator=worker_path.as_uri(),
        retrieved_at=_WORKER_RETRIEVED_AT,
        path=str(worker_path),
    )
    authoring = CapabilitySnapshotAuthoring(
        schema_version="bio-neb.execution-capability-authoring/1",
        capability_version=_CAPABILITY_VERSION,
        sources=(source,),
        recipes=(
            CapabilityPolicyRecipe(
                recipe_id="germline-approved",
                assay="gDNA",
                read_layout="paired",
                output_kind="VCF",
                requires_prepared_reference=True,
                requires_companion_roles=("genomic_fasta",),
            ),
        ),
        resources=(
            CapabilityPolicyResource(
                profile_id="h100-approved",
                recipe_id="germline-approved",
                image_id="parabricks-image",
                tool_ids=("bwa", "pbrun", "samtools"),
                accelerator_count=1,
                cpu_count=16,
                memory_gib=200,
                accelerator_memory_gib=80,
            ),
        ),
        images=(
            CapabilityPolicyImage(
                image_id="parabricks-image",
                provider_image=ATTESTED_PROVIDER_IMAGE,
                image_index_digest=ATTESTED_IMAGE_INDEX,
                platform="linux/amd64",
                platform_child_digest=ATTESTED_IMAGE_CHILD,
                worker_sha256=ATTESTED_WORKER_SHA256,
                worker_module=_WORKER_MODULE,
                worker_request_schema="bio-neb.parabricks-worker-request/7",
                worker_receipt_schema="bio-neb.parabricks-worker-receipt/5",
                worker_entrypoint=(
                    "/opt/bio-neb/venv/bin/python",
                    "/opt/bio-neb/reusable_object_storage_parabricks.py",
                ),
            ),
        ),
        tools=(
            CapabilityPolicyTool(
                tool_id="bwa",
                execution_scope="worker_image",
                absolute_path="/usr/bin/bwa",
                version=ATTESTED_BWA_VERSION,
                image_id="parabricks-image",
            ),
            CapabilityPolicyTool(
                tool_id="pbrun",
                execution_scope="worker_image",
                absolute_path="/usr/local/parabricks/pbrun",
                version=ATTESTED_PBRUN_VERSION,
                image_id="parabricks-image",
            ),
            CapabilityPolicyTool(
                tool_id="samtools",
                execution_scope="worker_image",
                absolute_path="/usr/bin/samtools",
                version=ATTESTED_SAMTOOLS_VERSION,
                image_id="parabricks-image",
            ),
        ),
    )
    try:
        snapshot = compile_capability_snapshot_value(authoring)
    except CapabilitySnapshotError as error:
        raise AcceptedPreExecutionBindingError(error.code, error.fields) from error
    manifest = snapshot.source_manifest
    if (
        len(manifest) != 1
        or manifest[0].locator != source.locator
        or manifest[0].retrieved_at != source.retrieved_at
        or manifest[0].content_assurance != "application_hashed_local_bytes"
        or manifest[0].content_digest != ATTESTED_WORKER_SHA256
    ):
        raise AcceptedPreExecutionBindingError(
            "accepted_worker_sha256_mismatch", (str(worker_path),)
        )
    retained_snapshot_ids = {
        event.get("payload", {}).get("result", {}).get("source", {}).get("snapshot_id")
        for event in journey.read()
        if event.get("event_type")
        in {"RecipeObservationRecorded", "ResourceObservationRecorded"}
    }
    retained_snapshot_ids.discard(None)
    if retained_snapshot_ids and retained_snapshot_ids != {snapshot.snapshot_id}:
        raise AcceptedPreExecutionBindingError(
            "accepted_capability_snapshot_changed", ("snapshot_id",)
        )
    return snapshot


__all__ = ["AcceptedPreExecutionBindingError"]
