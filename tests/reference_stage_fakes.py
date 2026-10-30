"""Synthetic provider receipts using the retained test worker marker contracts.

Large object heads are scripted identities. No FASTQ/reference bytes are read,
no worker runs, and these receipts establish no provider or scientific proof.
"""

import hashlib
from collections import deque
from types import SimpleNamespace

from bio_neb.content import canonical_json
from bio_neb.observation import AdapterHeartbeat
from bio_neb.parabricks import reference_recipe as contracts
from bio_neb.parabricks.execution_observations import (
    ToolExited,
    ToolStarted,
    UploadCompleted,
    UploadStarted,
    WorkerReceiptObject,
    WorkerReceiptObserved,
    WorkerRequestObserved,
    WorkerSourceAcquiredObserved,
    WorkerStartupObserved,
    WorkerToolIdentity,
    WorkerToolsObserved,
)
from bio_neb.parabricks.reference_recipe import (
    PublicUrlReferenceSource,
    ReferencePreparationRequest,
)
from bio_neb.providers.job_observations import (
    BoundedProviderLogObserved,
    FetchedProviderEnvelope,
    ProviderCreateObserved,
    ProviderStateObserved,
)
from bio_neb.providers.nebius_common import (
    AmbiguousNebiusCreate,
    AwsCredentialMaterial,
    NebiusCliError,
    NebiusCliFailureEvidence,
    NebiusCliResponse,
)
from bio_neb.providers.ports import Receipt
from tests.execution_observation_fakes import durable_receipt, worker_marker
from tests.reference_fakes import source_acquisition


def credentials() -> AwsCredentialMaterial:
    """Return inert credential material for the supplied-effects JobRunner."""
    return AwsCredentialMaterial(
        access_key_id="contact-only-access-key",
        secret_access_key="contact-only-secret-value",
        credentials_profile="object-storage-profile",
        worker_secret_selector="mbsec-secret@version-1",
    )


def cpu_observation_inputs(plan):
    return SimpleNamespace(
        nebius=plan.scope,
        object_storage=plan.storage,
        runtime=plan.runtimes.cpu,
        outputs=tuple(
            SimpleNamespace(role=role, object_key=key)
            for role, key in zip(
                (
                    "genomic_fasta",
                    "fasta_fai",
                    "reference_dict",
                    "bwa_amb",
                    "bwa_ann",
                    "bwa_bwt",
                    "bwa_pac",
                    "bwa_sa",
                ),
                plan.layout.prepared_object_keys,
                strict=True,
            )
        ),
    )


def cpu_observations(config, request, source, submission):
    """Supply protocol facts without constructing a historical execution adapter."""
    yield ProviderCreateObserved(
        provider_job_id=submission.provider_job_id,
        provider_job_name=submission.deterministic_job_name,
        disposition="created",
    )
    yield FetchedProviderEnvelope(
        provider_job_id=submission.provider_job_id,
        provider_job_name=submission.deterministic_job_name,
        project_id=config.nebius.project_id,
        provider_image_reference=config.runtime.provider_image_reference,
        platform=config.runtime.platform,
        preset=config.runtime.preset,
        disk_size_bytes=config.runtime.disk_size_bytes,
        subnet_id=config.nebius.subnet_id,
        timeout="14400s",
        container_command_requested="image-entrypoint",
        container_command_fetched=None,
        normalized_arguments=("/opt/bio-neb/request.json",),
        injected_paths=("/opt/bio-neb/request.json",),
        environment_names=(
            "AWS_ACCESS_KEY_ID",
            "AWS_ENDPOINT_URL",
            "AWS_REGION",
            "AWS_SECRET_ACCESS_KEY",
        ),
        version_bound_secret_environment_names=("AWS_SECRET_ACCESS_KEY",),
        shared_memory_bytes=0,
        restart_fetched="never",
    )
    for observation in (
        ProviderStateObserved(
            provider_job_id=submission.provider_job_id,
            provider_state="STARTING",
            created_at="2026-08-08T10:00:00+00:00",
        ),
        AdapterHeartbeat(),
        ProviderStateObserved(
            provider_job_id=submission.provider_job_id,
            provider_state="RUNNING",
            created_at="2026-08-08T10:00:00+00:00",
            started_at="2026-08-08T10:01:00+00:00",
        ),
        BoundedProviderLogObserved(
            provider_job_id=submission.provider_job_id,
            available=True,
            byte_count=1,
            line_count=1,
            sha256=hashlib.sha256(b"r").hexdigest(),
            limit_exhausted=False,
        ),
        ProviderStateObserved(
            provider_job_id=submission.provider_job_id,
            provider_state="COMPLETED",
            created_at="2026-08-08T10:00:00+00:00",
            started_at="2026-08-08T10:01:00+00:00",
            finished_at="2026-08-08T10:02:00+00:00",
        ),
    ):
        yield observation
    yield BoundedProviderLogObserved(
        provider_job_id=submission.provider_job_id,
        available=True,
        byte_count=1,
        line_count=1,
        sha256=hashlib.sha256(b"x").hexdigest(),
        limit_exhausted=False,
    )
    yield WorkerStartupObserved(
        interpreter="/opt/bio-neb/venv/bin/python",
        interpreter_resolved="/usr/bin/python3.10",
        request_schema="bio-neb.parabricks-worker-request/7",
    )
    yield WorkerRequestObserved(
        byte_count=request.worker_request_identity.byte_count,
        request_sha256=request.worker_request_identity.sha256,
        worker_sha256=config.runtime.worker_sha256,
    )
    tools = (
        WorkerToolIdentity(
            name="bwa", path="/usr/bin/bwa", version=config.runtime.bwa_version
        ),
        WorkerToolIdentity(
            name="samtools",
            path="/usr/bin/samtools",
            version=config.runtime.samtools_version,
        ),
    )
    yield WorkerToolsObserved(tools=tools)
    downloaded = isinstance(request.source, PublicUrlReferenceSource)
    if downloaded:
        # The job fetched the approved URL; its bytes are the fixture source's.
        acquired = source_acquisition()
        yield WorkerSourceAcquiredObserved(
            role="reference_source",
            url=request.source.url,
            byte_count=acquired.source_byte_count,
            md5=acquired.source_md5,
            sha256=acquired.source_sha256,
        )
        stored_source = {
            "role": "reference_source",
            "bucket": config.object_storage.bucket,
            "object_key": request.source.object_key,
            "byte_count": acquired.source_byte_count,
            "sha256": acquired.source_sha256,
        }
        yield UploadStarted(**stored_source, multipart_chunk_bytes=8 * 1024 * 1024)
        yield UploadCompleted(**stored_source)
        plain = (acquired.plain_fasta_byte_count, acquired.plain_fasta_sha256)
    else:
        plain = (
            request.source.plain_fasta_byte_count,
            request.source.plain_fasta_sha256,
        )
    for command_id, argv in zip(
        ("bwa-index", "samtools-faidx", "samtools-dict"),
        contracts.reference_preparation_commands(),
        strict=True,
    ):
        yield ToolStarted(command_id=command_id, argv=argv)
        yield ToolExited(command_id=command_id, returncode=0)
    uploads = []
    for index, item in enumerate(config.outputs):
        byte_count = plain[0] if item.role == "genomic_fasta" else index + 1
        sha256 = plain[1] if item.role == "genomic_fasta" else f"{index + 1:064x}"
        upload = {
            "role": item.role,
            "bucket": config.object_storage.bucket,
            "object_key": item.object_key,
            "byte_count": byte_count,
            "sha256": sha256,
        }
        yield UploadStarted(**upload, multipart_chunk_bytes=8 * 1024 * 1024)
        yield UploadCompleted(**upload)
        uploads.append(upload)
    receipt = {
        "schema_version": "bio-neb.parabricks-worker-receipt/3",
        "authority": "non-authoritative-worker-observation",
        "operation": "prepare_reference",
        "request_id": request.request_id,
        "workflow": "bwa-samtools-reference-preparation",
        "source": {
            "role": "reference_source",
            "object_key": source.object.object_key,
            "byte_count": source.object.byte_count,
            "sha256": source.object.sha256,
            "md5": request.source_md5,
            "origin": request.source.url
            if downloaded
            else f"s3://{config.object_storage.bucket}/{source.object.object_key}",
        },
        "prepared_uploads": uploads,
        "tools": {
            item.name: item.model_dump(mode="json", exclude={"name"}, exclude_none=True)
            for item in tools
        },
    }
    receipt_body = canonical_json(receipt).encode()
    receipt_upload = {
        "role": "worker_receipt",
        "bucket": config.object_storage.bucket,
        "object_key": request.worker_receipt_object_key,
        "byte_count": len(receipt_body),
        "sha256": hashlib.sha256(receipt_body).hexdigest(),
    }
    yield UploadStarted(**receipt_upload, multipart_chunk_bytes=8 * 1024 * 1024)
    yield UploadCompleted(**receipt_upload)
    yield WorkerReceiptObserved(
        receipt=receipt, receipt_object=WorkerReceiptObject(**receipt_upload)
    )


def seed_cpu(provider, plan, approved, *, job_id="aijob-reference-exact"):
    config = cpu_observation_inputs(plan)
    acquired = source_acquisition()
    # Input storage placed the source only for a supplied local FASTA or index.
    downloaded = plan.input_source is None
    request = contracts.compile_cpu_request(plan, None if downloaded else acquired)
    request = request.model_copy(
        update={"deterministic_job_name": plan.layout.cpu_job_name}
    )
    source = SimpleNamespace(
        object=SimpleNamespace(
            object_key=request.source.object_key,
            byte_count=acquired.source_byte_count,
            sha256=acquired.source_sha256,
        )
    )
    observations = list(
        cpu_observations(
            config,
            request,
            source,
            SimpleNamespace(
                provider_job_id=job_id,
                deterministic_job_name=plan.layout.cpu_job_name,
            ),
        )
    )
    marker_names = {
        "worker_startup": "BIO_NEB_PARABRICKS_WORKER_START",
        "worker_request": "BIO_NEB_WORKER_REQUEST_ACCEPTED",
        "worker_tools": "BIO_NEB_WORKER_TOOLS",
        "worker_source_acquired": "BIO_NEB_WORKER_SOURCE_ACQUIRED",
        "tool_started": "BIO_NEB_WORKER_TOOL_START",
        "tool_exited": "BIO_NEB_WORKER_TOOL_EXIT",
        "upload_started": "BIO_NEB_WORKER_UPLOAD_START",
        "upload_completed": "BIO_NEB_WORKER_UPLOAD_COMPLETE",
        "worker_receipt": "BIO_NEB_WORKER_RECEIPT",
    }
    lines = []
    for obs in observations:
        kind = getattr(obs, "kind", "")
        if kind not in marker_names:
            continue
        payload = obs.model_dump(mode="json", exclude={"kind"}, exclude_none=True)
        if kind == "worker_tools":
            payload = {
                "tools": {
                    item.name: item.model_dump(
                        mode="json", exclude={"name"}, exclude_none=True
                    )
                    for item in obs.tools
                }
            }
        lines.append(worker_marker(marker_names[kind], payload, len(lines) + 1))
    created = next(
        obs for obs in observations if getattr(obs, "kind", "") == "provider_create"
    )
    envelope = next(
        obs
        for obs in observations
        if isinstance(obs, contracts.FetchedProviderEnvelope)
    )
    observed = next(
        obs for obs in observations if isinstance(obs, contracts.WorkerReceiptObserved)
    )
    provider.scripts["cpu_preparation", "submit_cpu"] = deque(
        [
            Receipt(
                values={
                    "create": created.model_dump(mode="json"),
                    "envelope": envelope.model_dump(mode="json"),
                },
                provider_ids=(job_id,),
                resource_count=1,
            )
        ]
    )
    provider.scripts["cpu_preparation", "await_cpu"] = deque(
        [
            Receipt(
                values={
                    "state": "COMPLETED",
                    "log": "\n".join(lines),
                    "window": 1000,
                    "details": next(
                        obs.model_dump(mode="json")
                        for obs in observations
                        if isinstance(obs, contracts.ProviderStateObserved)
                        and obs.provider_state == "COMPLETED"
                    ),
                },
                provider_ids=(job_id,),
            )
        ]
    )
    worker = durable_receipt(observations)
    # The older CPU vector predates input/support markers. Supply the current
    # producer contract explicitly, without claiming these values were observed.
    if not downloaded:
        worker["validation"].insert(
            3,
            {
                "kind": "worker_input_verified",
                "role": "reference_source",
                "bucket": plan.storage.bucket,
                "object_key": source.object.object_key,
                "byte_count": source.object.byte_count,
                "sha256": source.object.sha256,
            },
        )
    first_upload = next(
        i
        for i, fact in enumerate(worker["validation"])
        if fact["kind"] == "upload_started" and fact["role"] != "reference_source"
    )
    worker["validation"].insert(
        first_upload,
        {
            "kind": "worker_reference_support",
            "mode": "derive",
            "objects": [
                {k: v for k, v in item.items() if k != "bucket"}
                for item in worker["prepared_uploads"]
            ],
        },
    )
    body = canonical_json(worker).encode()
    provider.objects[plan.storage.bucket, plan.layout.reference_receipt_object_key] = (
        body
    )
    # Only an independent full read heads the produced objects; the source
    # manifest the host then publishes is an ordinary in-memory write.
    audited = (
        [
            *(
                [
                    {
                        "byte_count": acquired.source_byte_count,
                        "sha256": acquired.source_sha256,
                    }
                ]
                if downloaded
                else []
            ),
            *observed.receipt["prepared_uploads"],
        ]
        if plan.verification.effective.reference_preparation == "independent_full_read"
        else []
    )
    provider.scripts["cpu_preparation", "verify_bundle"] = deque(
        [
            Receipt(
                content=body,
                values={"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()},
            ),
            *[
                Receipt(values={"bytes": item["byte_count"], "sha256": item["sha256"]})
                for item in audited
            ],
        ]
    )
    return provider


class CpuCli:
    def __init__(
        self,
        config,
        request: ReferencePreparationRequest,
        *,
        ambiguous: bool = False,
        invalid_json_after_create: bool = False,
        initial_state: str = "STARTING",
        omit_restart: bool = False,
        log_outputs: tuple[str, ...] = (),
    ):
        self.config = config
        self.request = request
        self.ambiguous = ambiguous
        self.invalid_json_after_create = invalid_json_after_create
        self.initial_state = initial_state
        self.omit_restart = omit_restart
        self.log_outputs = iter(log_outputs)
        self.last_log_output = ""
        self.created = False
        self.calls: list[tuple[str, ...]] = []
        self.states = iter(("RUNNING", "RUNNING", "COMPLETED"))
        self.get_count = 0
        self.spec = None

    def __call__(self, *arguments: str):
        if arguments[:3] == ("ai", "job", "create"):
            # json() records the call, so it is not recorded twice here.
            return self.json(*arguments)
        self.calls.append(arguments)
        if arguments[:3] == ("ai", "job", "logs"):
            try:
                self.last_log_output = next(self.log_outputs)
            except StopIteration:
                pass
            return NebiusCliResponse(stdout=self.last_log_output)
        return NebiusCliResponse(stdout="")

    def json(self, *arguments: str):
        self.calls.append(arguments)
        if arguments[:3] == ("ai", "job", "get-by-name"):
            if not self.created:
                raise NebiusCliError(
                    "provider_command_failed",
                    failure_evidence=NebiusCliFailureEvidence(
                        operation="ai.job.get-by-name",
                        child_exit_code=13,
                        stdout_byte_count=0,
                        stdout_sha256=hashlib.sha256(b"").hexdigest(),
                        stderr_byte_count=48,
                        stderr_sha256="f" * 64,
                        stderr_classification="not_found",
                    ),
                )
            return self._job(self.initial_state)
        if arguments[:3] == ("ai", "job", "create"):
            self.created = True
            values = {
                arguments[index]: arguments[index + 1]
                for index in range(3, len(arguments) - 1)
                if arguments[index].startswith("--")
                and arguments[index] not in {"--env", "--env-secret"}
            }
            environment = []
            for index, argument in enumerate(arguments[:-1]):
                if argument == "--env":
                    name, value = arguments[index + 1].split("=", 1)
                    environment.append({"name": name, "value": value})
                elif argument == "--env-secret":
                    name, selector = arguments[index + 1].split("=", 1)
                    secret_id, version_id = selector.rsplit("@", 1)
                    environment.append(
                        {
                            "name": name,
                            "mysterybox_secret": {
                                "secret_id": secret_id,
                                "version_id": version_id,
                            },
                        }
                    )
            self.spec = {
                "image": values["--image"],
                "container_command": values.get("--container-command"),
                "args": values["--args"],
                "environment_variables": environment,
                "volumes": [],
                "platform": values["--platform"],
                "preset": values["--preset"],
                "disk": {"size_bytes": str(100 * 1024**3)},
                "shm_size_bytes": "0",
                "subnet_id": values["--subnet-id"],
                # Server-injected: no --timeout is requested.
                "timeout": "86400s",
                "restart_policy": values["--restart-policy"],
                "injected_files": [
                    {"container_path": values["--inject-file"].split(":", 1)[1]}
                ],
            }
            if self.omit_restart:
                self.spec.pop("restart_policy")
            if self.ambiguous:
                raise AmbiguousNebiusCreate()
            return NebiusCliResponse(stdout="Job ID: aijob-exact" + chr(10))
        if arguments[:3] == ("ai", "job", "get"):
            # The old form replayed the initial state once after an ambiguous create,
            # compensating for a `get` the reconcile path used to perform.  Identity now
            # comes from `get-by-name` instead, so create performs no `get` and
            # there is nothing left to compensate for.
            self.get_count += 1
            return self._job(next(self.states))
        raise AssertionError(arguments)

    def _job(self, state: str):
        return {
            "metadata": {
                "id": "aijob-reference-preparation",
                "name": self.request.deterministic_job_name,
                "parent_id": "project-exact",
                "created_at": "2026-08-08T10:00:00Z",
            },
            "spec": self.spec,
            "status": {
                "state": state,
                "started_at": "2026-08-08T10:00:01Z",
                "finished_at": "2026-08-08T10:02:00Z" if state == "COMPLETED" else None,
            },
        }
