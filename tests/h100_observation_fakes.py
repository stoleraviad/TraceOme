"""Current H100 observation vectors for contact-free operation tests.

Copied from the retained legacy helper without changing seeded values. These
observations are synthetic; current stage tests still exercise the real stage.
"""

from __future__ import annotations

import hashlib

from bio_neb.journey import canonical_json
from bio_neb.parabricks.execution_observations import (
    PreparedReferenceObjectObserved,
    ToolExited,
    ToolStarted,
    UploadCompleted,
    UploadStarted,
    WorkerInputVerifiedObserved,
    WorkerPreparedReferenceVerifiedObserved,
    WorkerReceiptObject,
    WorkerReceiptObserved,
    WorkerReferenceSupportObserved,
    WorkerRequestObserved,
    WorkerStartupObserved,
    WorkerToolIdentity,
    WorkerToolsObserved,
)
from bio_neb.parabricks.h100_recipe import H100SubmissionReceipt
from bio_neb.providers.job_observations import (
    BoundedProviderLogObserved,
    FetchedProviderEnvelope,
    ProviderCreateObserved,
    ProviderStateObserved,
)

_PROVIDER_JOB_ID = "aijob-exact"


def h100_submission(request) -> H100SubmissionReceipt:
    """Bind the one provider job identity every observation of this fake reports."""

    response = b"Job ID: aijob-exact\n"
    return H100SubmissionReceipt(
        provider_job_id=_PROVIDER_JOB_ID,
        deterministic_job_name=request.deterministic_job_name,
        create_response_byte_count=len(response),
        create_response_sha256=hashlib.sha256(response).hexdigest(),
    )


def h100_observations(
    config, request, *, pbrun_exit=0, output_payloads=None, job_id=_PROVIDER_JOB_ID
):
    observations = [
        ProviderCreateObserved(
            provider_job_id=job_id,
            provider_job_name=request.deterministic_job_name,
            disposition="created",
        ),
        FetchedProviderEnvelope(
            provider_job_id=job_id,
            provider_job_name=request.deterministic_job_name,
            project_id=config.nebius.project_id,
            provider_image_reference=config.runtime.provider_image_reference,
            platform=config.runtime.platform,
            preset=config.runtime.preset,
            disk_size_bytes=config.runtime.disk_size_bytes,
            subnet_id=config.nebius.subnet_id,
            # Server-injected: the create request sets no deadline, so the envelope records
            # whatever the provider chose.  A live run observed 86400s.
            timeout="86400s",
            container_command_requested="image-entrypoint",
            container_command_fetched=None,
            normalized_arguments=(config.request_container_path,),
            injected_paths=(config.request_container_path,),
            environment_names=(
                "AWS_ACCESS_KEY_ID",
                "AWS_ENDPOINT_URL",
                "AWS_REGION",
                "AWS_SECRET_ACCESS_KEY",
            ),
            version_bound_secret_environment_names=("AWS_SECRET_ACCESS_KEY",),
            shared_memory_bytes=16 * 1024**3,
            restart_fetched="omitted",
        ),
        ProviderStateObserved(
            provider_job_id=job_id,
            provider_state="COMPLETED",
            created_at="2026-08-10T00:00:00+00:00",
            started_at="2026-08-10T00:01:00+00:00",
            finished_at="2026-08-10T00:02:00+00:00",
        ),
        BoundedProviderLogObserved(
            provider_job_id=job_id,
            available=True,
            byte_count=1,
            line_count=1,
            sha256=hashlib.sha256(b"x").hexdigest(),
            limit_exhausted=False,
        ),
        WorkerStartupObserved(
            interpreter=config.runtime.python_path,
            interpreter_resolved="/usr/bin/python3.10",
            request_schema=request.worker_request_identity.schema_version,
        ),
        WorkerRequestObserved(
            byte_count=request.worker_request_identity.byte_count,
            request_sha256=request.worker_request_identity.sha256,
            worker_sha256=config.runtime.worker_sha256,
        ),
    ]
    tools = (
        WorkerToolIdentity(
            name="python",
            path=config.runtime.python_path,
            resolved_path="/usr/bin/python3.10",
            version=config.runtime.python_version,
        ),
        WorkerToolIdentity(
            name="pbrun",
            path=config.runtime.pbrun_path,
            version=config.runtime.pbrun_version,
        ),
    )
    observations.append(WorkerToolsObserved(tools=tools))
    observations.extend(
        WorkerInputVerifiedObserved(
            role=item.role,
            bucket=config.object_storage.bucket,
            object_key=item.object_key,
            byte_count=item.byte_count,
            sha256=item.sha256,
        )
        for item in request.inputs
    )
    prepared = tuple(
        PreparedReferenceObjectObserved(
            role=item.role,
            object_key=item.object_key,
            byte_count=item.byte_count,
            sha256=item.sha256,
        )
        for item in request.inputs[2:]
    )
    observations.extend(
        (
            WorkerReferenceSupportObserved(mode="verify", objects=prepared),
            WorkerPreparedReferenceVerifiedObserved(objects=prepared),
            ToolStarted(
                command_id="pbrun-germline",
                argv=(config.runtime.pbrun_path, *request.parabricks_argv[1:]),
            ),
            ToolExited(command_id="pbrun-germline", returncode=pbrun_exit),
        )
    )
    uploads = []
    for index, target in enumerate(config.outputs[:2]):
        payload = (output_payloads or {}).get(target.role)
        upload = {
            "role": target.role,
            "bucket": config.object_storage.bucket,
            "object_key": target.object_key,
            "byte_count": len(payload) if payload is not None else index + 301,
            "sha256": hashlib.sha256(payload).hexdigest()
            if payload is not None
            else f"{index + 31:064x}",
        }
        observations.extend(
            (
                UploadStarted(**upload, multipart_chunk_bytes=8 * 1024**2),
                UploadCompleted(**upload),
            )
        )
        uploads.append(upload)
    receipt = {
        "schema_version": "bio-neb.parabricks-worker-receipt/3",
        "authority": "non-authoritative-worker-observation",
        "operation": "consume_prepared_reference",
        "request_id": request.request_id,
        "workflow": "nvidia-parabricks-germline",
        "prepared_inputs": [
            item.model_dump(mode="json", exclude={"kind"}) for item in prepared
        ],
        "uploads": uploads,
        "tools": {
            item.name: item.model_dump(mode="json", exclude={"name"}, exclude_none=True)
            for item in tools
        },
    }
    body = canonical_json(receipt).encode()
    receipt_upload = {
        "role": "worker_receipt",
        "bucket": config.object_storage.bucket,
        "object_key": config.outputs[2].object_key,
        "byte_count": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    observations.extend(
        (
            UploadStarted(**receipt_upload, multipart_chunk_bytes=8 * 1024**2),
            UploadCompleted(**receipt_upload),
            WorkerReceiptObserved(
                receipt=receipt,
                receipt_object=WorkerReceiptObject(**receipt_upload),
            ),
        )
    )
    return observations
