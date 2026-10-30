"""Typed worker recipe, manifest and closure contracts for current stages."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from bio_neb.application.artifact_verification import WorkerTransferProof
from bio_neb.contracts import FrozenContract
from bio_neb.journey import (
    canonical_json,
    canonical_json_digest,
)
from bio_neb.observation import AdapterHeartbeat
from bio_neb.parabricks.execution_observations import (
    ToolExited,
    ToolStarted,
    UploadCompleted,
    UploadStarted,
    WorkerFailureObserved,
    WorkerInputVerifiedObserved,
    WorkerPreparedReferenceVerifiedObserved,
    WorkerReceiptObserved,
    WorkerReferenceSupportObserved,
    WorkerRequestObserved,
    WorkerSourceAcquiredObserved,
    WorkerSourceMismatchObserved,
    WorkerStartupObserved,
    WorkerToolsObserved,
    validate_truthful_observation_order,
)
from bio_neb.parabricks.execution_runtime import CpuReferenceRuntime
from bio_neb.parabricks.worker_exchange import PREPARED_ROLES
from bio_neb.parabricks.worker_request import (
    REQUEST_SCHEMA_VERSION,
    AcceptedPlainFasta,
    ExactWorkerObject,
    PlainFastaTarget,
    PrepareReferenceParameters,
    PublicReferenceDownload,
    WorkerObjectStorage,
    WorkerObjectTarget,
    WorkerRequestIdentity,
    compile_worker_request,
)
from bio_neb.parabricks.worker_request_placement import (
    reconcile_fetched_request_route,
    reconcile_observed_worker_request,
)
from bio_neb.providers.job_observations import (
    BoundedProviderLogObserved,
    FetchedProviderEnvelope,
    ProviderCreateObserved,
    ProviderStateObserved,
)
from bio_neb.providers.runtime import NebiusCliTarget, RunObjectStorageTarget
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_PATTERN,
)
from bio_neb.verified_artifacts import VerifiedObject


def _line(value: str, field: str) -> str:
    if not value or value.strip() != value or any(c in value for c in "\r\n\0"):
        raise ValueError(f"{field} must be one canonical line")
    return value


def _key(value: str, field: str) -> str:
    _line(value, field)
    if value.startswith("/") or value.endswith("/") or "//" in value:
        raise ValueError(f"{field} must be an exact object key")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{field} contains a relative segment")
    return value


def _uuid(value: str, field: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"{field} must be a UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{field} must be canonical")
    return value


class ExistingObjectReferenceSource(FrozenContract):
    """A source input storage already placed; its acquired plain FASTA is reproduced."""

    mode: Literal["existing_object"] = "existing_object"
    object_key: StrictStr
    expected_size_bytes: StrictInt = Field(gt=0)
    expected_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    plain_fasta_byte_count: StrictInt = Field(gt=0)
    plain_fasta_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("object_key")
    @classmethod
    def exact_key(cls, value: str) -> str:
        return _key(value, "object_key")


class PublicUrlReferenceSource(FrozenContract):
    """The approved publisher URL the CPU job fetches into ``object_key``."""

    mode: Literal["public_url"] = "public_url"
    url: StrictStr = Field(pattern=r"^https://")
    object_key: StrictStr
    expected_size_bytes: StrictInt = Field(gt=0)

    @field_validator("object_key")
    @classmethod
    def exact_key(cls, value: str) -> str:
        return _key(value, "object_key")


PreparedReferenceRole = Literal[
    "genomic_fasta",
    "fasta_fai",
    "bwa_amb",
    "bwa_ann",
    "bwa_bwt",
    "bwa_pac",
    "bwa_sa",
    "reference_dict",
]


class PreparedReferenceObject(FrozenContract):
    role: PreparedReferenceRole
    object_key: StrictStr
    interoperability_extra: StrictBool = False

    @field_validator("object_key")
    @classmethod
    def exact_key(cls, value: str) -> str:
        return _key(value, "object_key")

    @model_validator(mode="after")
    def dictionary_is_only_extra(self) -> PreparedReferenceObject:
        if self.interoperability_extra != (self.role == "reference_dict"):
            raise ValueError(
                "only reference_dict is the labeled interoperability extra"
            )
        return self


class ReferencePreparationRequest(FrozenContract):
    request_id: StrictStr = Field(
        pattern=r"^reference-preparation-request:[0-9a-f]{64}$"
    )
    journey_id: StrictStr
    authority_id: StrictStr = Field(pattern=r"^execution-authority:[0-9a-f]{64}$")
    proposal_id: StrictStr = Field(pattern=r"^execution-proposal:[0-9a-f]{64}$")
    configuration_id: StrictStr = Field(
        pattern=r"^reference-preparation-configuration:[0-9a-f]{64}$"
    )
    bundle_id: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    deterministic_job_name: StrictStr = Field(
        pattern=r"^bio-neb-(?:ref|cpu)-[0-9a-f]{16}$"
    )
    reference_asset_id: StrictStr
    assembly_accession: StrictStr
    source_md5: StrictStr = Field(pattern=r"^[0-9a-f]{32}$")
    source: ExistingObjectReferenceSource | PublicUrlReferenceSource = Field(
        discriminator="mode"
    )
    # The records the plain FASTA leaves out: the Run indexes its reference
    # sequences only.
    excluded_sequences: tuple[StrictStr, ...]
    storage: WorkerObjectStorage
    outputs: tuple[PreparedReferenceObject, ...]
    request_object_key: StrictStr
    source_manifest_object_key: StrictStr
    manifest_object_key: StrictStr
    worker_receipt_object_key: StrictStr
    worker_request_identity: WorkerRequestIdentity
    runtime: CpuReferenceRuntime

    @model_validator(mode="after")
    def exact_identity(self) -> ReferencePreparationRequest:
        _uuid(self.journey_id, "journey_id")
        material = self.model_dump(
            mode="json",
            exclude={"request_id", "deterministic_job_name", "worker_request_identity"},
        )
        expected = "reference-preparation-request:" + canonical_json_digest(material)
        if self.request_id != expected or (
            self.deterministic_job_name.startswith("bio-neb-ref-")
            and self.deterministic_job_name
            != f"bio-neb-ref-{expected.rsplit(':', 1)[1][:16]}"
        ):
            raise ValueError(
                "reference preparation request changed deterministic identity"
            )
        if _compile_worker_request(self).identity != self.worker_request_identity:
            raise ValueError("reference preparation changed compiled worker request")
        return self


def _compile_worker_request(request: ReferencePreparationRequest):
    fasta = "/work/reference/reference.fa"
    paths = {
        "genomic_fasta": fasta,
        "fasta_fai": f"{fasta}.fai",
        "reference_dict": "/work/reference/reference.dict",
        **{
            f"bwa_{suffix}": f"{fasta}.{suffix}"
            for suffix in ("amb", "ann", "bwt", "pac", "sa")
        },
    }
    return compile_worker_request(
        PrepareReferenceParameters(
            request_id=request.request_id,
            storage=WorkerObjectStorage(
                bucket=request.storage.bucket,
                endpoint_url=request.storage.endpoint_url,
                region=request.storage.region,
            ),
            output_prefix=f"references/{request.bundle_id}/",
            **_worker_source(request, fasta),
            excluded_sequences=tuple(request.excluded_sequences),
            prepared_outputs=tuple(
                WorkerObjectTarget(
                    role=item.role,
                    object_key=item.object_key,
                    local_target=paths[item.role],
                )
                for item in request.outputs
            ),
            receipt_output=WorkerObjectTarget(
                role="worker_receipt",
                object_key=request.worker_receipt_object_key,
                local_target="/work/receipts/worker.json",
            ),
        )
    )


def _worker_source(request: ReferencePreparationRequest, fasta: str) -> dict:
    source, local = request.source, "/work/source/reference.fa.gz"
    if isinstance(source, PublicUrlReferenceSource):
        return {
            "source": PublicReferenceDownload(
                role="reference_source",
                url=source.url,
                byte_count=source.expected_size_bytes,
                md5=request.source_md5,
                object_key=source.object_key,
                local_target=local,
            ),
            "plain_fasta": PlainFastaTarget(local_target=fasta),
        }
    return {
        "source": ExactWorkerObject(
            role="reference_source",
            object_key=source.object_key,
            local_target=local,
            byte_count=source.expected_size_bytes,
            sha256=source.expected_sha256,
        ),
        "plain_fasta": AcceptedPlainFasta(
            local_target=fasta,
            byte_count=source.plain_fasta_byte_count,
            sha256=source.plain_fasta_sha256,
        ),
    }


def output_targets(plan) -> tuple[PreparedReferenceObject, ...]:
    return tuple(
        PreparedReferenceObject(
            role=role, object_key=key, interoperability_extra=role == "reference_dict"
        )
        for role, key in zip(
            PREPARED_ROLES, plan.layout.prepared_object_keys, strict=True
        )
    )


def configuration_id(plan) -> str:
    return "reference-preparation-configuration:" + canonical_json_digest(
        {
            "scope": plan.scope.model_dump(mode="json"),
            "storage": plan.storage.model_dump(mode="json"),
            "runtime": plan.runtimes.cpu.model_dump(mode="json"),
            "reference": plan.reference.model_dump(mode="json"),
            "bundle_id": plan.layout.reference_bundle_id,
            "outputs": [item.model_dump(mode="json") for item in output_targets(plan)],
            "request_object_key": plan.layout.cpu_request_object_key,
            "source_manifest_object_key": plan.layout.source_manifest_object_key,
            "manifest_object_key": plan.layout.reference_manifest_object_key,
            "worker_receipt_object_key": plan.layout.reference_receipt_object_key,
            "verification_mode": plan.verification.effective.reference_preparation,
        }
    )


def compile_cpu_request(plan, acquisition) -> ReferencePreparationRequest:
    """Compile the approved plan's CPU recipe without another authority/config layer.

    With an acquisition (input storage placed the source) the source and plain
    FASTA identities are its measured values and the worker must reproduce them.
    Without one the job fetches the approved URL, checks its md5 and size, and
    measures them itself.
    """
    source = (
        PublicUrlReferenceSource(
            url=plan.reference.source_url,
            object_key=plan.layout.fetched_source_object_key,
            expected_size_bytes=plan.reference.expected_size_bytes,
        )
        if acquisition is None
        else ExistingObjectReferenceSource(
            object_key=plan.layout.source_object_key,
            expected_size_bytes=acquisition.source_byte_count,
            expected_sha256=acquisition.source_sha256,
            plain_fasta_byte_count=acquisition.plain_fasta_byte_count,
            plain_fasta_sha256=acquisition.plain_fasta_sha256,
        )
    )
    return _reference_request(
        {
            "journey_id": plan.identity.journey_id,
            "authority_id": plan.identity.authority_id,
            "proposal_id": plan.identity.proposal_id,
            "configuration_id": configuration_id(plan),
            "bundle_id": plan.layout.reference_bundle_id,
            "reference_asset_id": plan.reference.reference_asset_id,
            "assembly_accession": plan.reference.assembly_accession,
            "source_md5": plan.reference.expected_md5,
            "source": source,
            "excluded_sequences": list(plan.reference.excluded_sequences),
            "storage": WorkerObjectStorage(
                bucket=plan.storage.bucket,
                endpoint_url=plan.storage.endpoint_url,
                region=plan.storage.region,
            ),
            "outputs": output_targets(plan),
            "request_object_key": plan.layout.cpu_request_object_key,
            "source_manifest_object_key": plan.layout.source_manifest_object_key,
            "manifest_object_key": plan.layout.reference_manifest_object_key,
            "worker_receipt_object_key": plan.layout.reference_receipt_object_key,
            "runtime": plan.runtimes.cpu,
        },
        plan.layout.cpu_job_name,
    )


def _reference_request(
    base: dict, job_name: str | None = None
) -> ReferencePreparationRequest:
    serialized = {
        key: value.model_dump(mode="json")
        if isinstance(value, BaseModel)
        else [item.model_dump(mode="json") for item in value]
        if isinstance(value, tuple)
        else value
        for key, value in base.items()
    }
    request_id = "reference-preparation-request:" + canonical_json_digest(serialized)
    provisional = {
        "request_id": request_id,
        "deterministic_job_name": job_name
        or f"bio-neb-ref-{request_id.rsplit(':', 1)[1][:16]}",
        **base,
    }
    unchecked = ReferencePreparationRequest.model_construct(
        worker_request_identity=WorkerRequestIdentity(
            schema_version=REQUEST_SCHEMA_VERSION,
            operation="prepare_reference",
            byte_count=1,
            sha256="0" * 64,
        ),
        **provisional,
    )
    identity = _compile_worker_request(unchecked).identity
    return ReferencePreparationRequest(
        worker_request_identity=identity,
        **provisional,
    )


def reference_preparation_commands() -> tuple[tuple[str, ...], ...]:
    """The only tool invocations accepted from the compact CPU worker."""

    fasta = "/work/reference/reference.fa"
    return (
        ("/usr/bin/bwa", "index", fasta),
        ("/usr/bin/samtools", "faidx", fasta),
        (
            "/usr/bin/samtools",
            "dict",
            "-o",
            "/work/reference/reference.dict",
            fasta,
        ),
    )


class VerifiedReferenceSource(FrozenContract):
    source_mode: Literal["public_url", "local_file", "existing_object"]
    reference_asset_id: StrictStr
    source_locator: StrictStr
    object: VerifiedObject
    manifest_object: VerifiedObject
    exact_inventory: tuple[StrictStr, ...] = Field(min_length=2, max_length=2)
    verification_mode: Literal["independent_full_read", "reuse_creation_proof"]
    independently_full_read: StrictBool = True
    manifest_published_last: Literal[True] = True

    @model_validator(mode="after")
    def exact_source_inventory(self) -> VerifiedReferenceSource:
        expected = tuple(
            sorted((self.object.object_key, self.manifest_object.object_key))
        )
        if (
            self.object.role != "reference_source"
            or self.manifest_object.role != "source_manifest"
            or self.exact_inventory != expected
            or self.independently_full_read
            != (self.verification_mode == "independent_full_read")
        ):
            raise ValueError("source verification changed exact inventory")
        return self


class ReferenceProviderStateChange(FrozenContract):
    kind: Literal["provider_state"] = "provider_state"
    provider_state: StrictStr


_TERMINAL_PROVIDER_STATES = {
    "COMPLETED",
    "SUCCEEDED",
    "FAILED",
    "ERROR",
    "CANCELLED",
    "CANCELED",
    "TIMED_OUT",
    "TIMEOUT",
}


class ReferencePreparationTerminalSnapshot(FrozenContract):
    """Secret-free exact provider truth yielded before optional log contact."""

    kind: Literal["terminal_snapshot"] = "terminal_snapshot"
    provider_job_id: StrictStr
    deterministic_job_name: StrictStr
    provider_state: StrictStr
    created_at: StrictStr
    started_at: StrictStr
    finished_at: StrictStr
    project_id: StrictStr
    provider_image_reference: StrictStr
    platform: Literal["cpu-d3"]
    preset: Literal["4vcpu-16gb"]
    disk_size_bytes: StrictInt = Field(gt=0)
    subnet_id: StrictStr
    # The provider's own injected deadline, exactly as observed on the fetched
    # envelope.  The APP authors no deadline, so nothing pins this to a value of
    # its own.
    timeout: StrictStr
    restart_policy: Literal["never"]
    container_command: StrictStr | None
    worker_target: Literal["/opt/bio-neb/request.json"]
    request_object_key: StrictStr
    request_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    worker_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_secret_free_terminal(self) -> ReferencePreparationTerminalSnapshot:
        if self.provider_state not in _TERMINAL_PROVIDER_STATES:
            raise ValueError("terminal snapshot retained a nonterminal state")
        timestamps = []
        for name in ("created_at", "started_at", "finished_at"):
            value = _line(getattr(self, name), name)
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("terminal snapshot timestamp is invalid") from error
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("terminal snapshot timestamp lacks timezone")
            timestamps.append(parsed)
        if timestamps != sorted(timestamps):
            raise ValueError("terminal snapshot timestamps changed lifecycle order")
        if self.request_object_key != "/opt/bio-neb/request.json":
            raise ValueError("terminal snapshot changed injected request path")
        return self


class ReferencePreparationExited(FrozenContract):
    kind: Literal["exit"] = "exit"
    exit_code: StrictInt


ReferencePreparationObservation = (
    AdapterHeartbeat
    | ReferenceProviderStateChange
    | ReferencePreparationTerminalSnapshot
    | ReferencePreparationExited
    | ProviderCreateObserved
    | FetchedProviderEnvelope
    | ProviderStateObserved
    | BoundedProviderLogObserved
    | WorkerStartupObserved
    | WorkerRequestObserved
    | WorkerToolsObserved
    | WorkerInputVerifiedObserved
    | WorkerSourceAcquiredObserved
    | WorkerSourceMismatchObserved
    | WorkerReferenceSupportObserved
    | WorkerPreparedReferenceVerifiedObserved
    | ToolStarted
    | ToolExited
    | UploadStarted
    | UploadCompleted
    | WorkerReceiptObserved
    | WorkerFailureObserved
)


def validate_preparation_observation_closure(
    scope: NebiusCliTarget,
    storage: RunObjectStorageTarget,
    runtime: CpuReferenceRuntime,
    request: ReferencePreparationRequest,
    source: VerifiedReferenceSource | None,
    provider_job_id: str,
    observations: Iterable[ReferencePreparationObservation],
    *,
    durable: bool = False,
) -> WorkerReceiptObserved:
    """Validate worker truth before manifest authority mutates Run object storage.

    A stored source (``source`` given) was verified before the job; a downloaded
    source is established here from the worker's acquisition and its upload.
    """
    downloaded = isinstance(request.source, PublicUrlReferenceSource)
    if downloaded != (source is None):
        raise ValueError("prepared worker source mode changed")

    items = tuple(observations)
    validate_truthful_observation_order(items, terminal=True, durable=durable)
    creates = [item for item in items if isinstance(item, ProviderCreateObserved)]
    envelopes = [item for item in items if isinstance(item, FetchedProviderEnvelope)]
    provider_scoped = [
        item
        for item in items
        if isinstance(
            item,
            (
                ProviderCreateObserved,
                FetchedProviderEnvelope,
                ProviderStateObserved,
                BoundedProviderLogObserved,
            ),
        )
    ]
    startups = [item for item in items if isinstance(item, WorkerStartupObserved)]
    requests = [item for item in items if isinstance(item, WorkerRequestObserved)]
    tool_sets = [item for item in items if isinstance(item, WorkerToolsObserved)]
    input_verifications = [
        item for item in items if isinstance(item, WorkerInputVerifiedObserved)
    ]
    reference_support = [
        item for item in items if isinstance(item, WorkerReferenceSupportObserved)
    ]
    prepared_verified = [
        item
        for item in items
        if isinstance(item, WorkerPreparedReferenceVerifiedObserved)
    ]
    acquired = [
        item for item in items if isinstance(item, WorkerSourceAcquiredObserved)
    ]
    if downloaded and len(acquired) != 1:
        raise ValueError("prepared worker lacks its exact source acquisition")
    cpu_marker_extension = durable or bool(
        input_verifications or acquired or reference_support or prepared_verified
    )
    receipts = [item for item in items if isinstance(item, WorkerReceiptObserved)]
    if len(envelopes) == 1:
        reconcile_fetched_request_route(
            expected_container_path="/opt/bio-neb/request.json",
            normalized_arguments=envelopes[0].normalized_arguments,
            injected_paths=envelopes[0].injected_paths,
        )
    if len(startups) == 1 and len(requests) == 1:
        reconcile_observed_worker_request(
            request.worker_request_identity,
            request_schema=startups[0].request_schema,
            byte_count=requests[0].byte_count,
            request_sha256=requests[0].request_sha256,
        )
    if (
        len(creates) != 1
        or any(item.provider_job_id != provider_job_id for item in provider_scoped)
        or creates[0].provider_job_name != request.deterministic_job_name
        or len(envelopes) != 1
        or envelopes[0].provider_job_name != request.deterministic_job_name
        or envelopes[0].project_id != scope.project_id
        or envelopes[0].provider_image_reference != runtime.provider_image_reference
        or envelopes[0].platform != runtime.platform
        or envelopes[0].preset != runtime.preset
        or envelopes[0].disk_size_bytes != runtime.disk_size_bytes
        or envelopes[0].subnet_id != scope.subnet_id
        or envelopes[0].container_command_requested != "image-entrypoint"
        or envelopes[0].container_command_fetched is not None
        or envelopes[0].environment_names
        != (
            "AWS_ACCESS_KEY_ID",
            "AWS_ENDPOINT_URL",
            "AWS_REGION",
            "AWS_SECRET_ACCESS_KEY",
        )
        or envelopes[0].version_bound_secret_environment_names
        != ("AWS_SECRET_ACCESS_KEY",)
        or envelopes[0].shared_memory_bytes != 0
        or envelopes[0].restart_requested != "never"
        or envelopes[0].restart_fetched not in {"never", "omitted"}
        or len(startups) != 1
        or len(requests) != 1
        or requests[0].worker_sha256 != runtime.worker_sha256
        or len(tool_sets) != 1
        or (
            cpu_marker_extension
            and (
                not _source_observed(
                    storage, request, source, input_verifications, acquired
                )
                or len(reference_support) != 1
                or reference_support[0].mode != "derive"
                or prepared_verified
            )
        )
        or len(receipts) != 1
    ):
        raise ValueError("prepared worker startup, tools, and receipt must be unique")
    tools = {item.name: item for item in tool_sets[0].tools}
    bwa = tools.get("bwa")
    samtools = tools.get("samtools")
    if (
        bwa is None
        or bwa.path != "/usr/bin/bwa"
        or bwa.resolved_path not in {None, "/usr/bin/bwa"}
        or bwa.version != runtime.bwa_version
        or samtools is None
        or samtools.path != "/usr/bin/samtools"
        or samtools.resolved_path not in {None, "/usr/bin/samtools"}
        or samtools.version != runtime.samtools_version
    ):
        raise ValueError("prepared worker changed exact BWA or samtools identity")
    command_ids = ("bwa-index", "samtools-faidx", "samtools-dict")
    expected_commands: list[ToolStarted | ToolExited] = []
    for command_id, argv in zip(
        command_ids, reference_preparation_commands(), strict=True
    ):
        expected_commands.extend(
            (
                ToolStarted(command_id=command_id, argv=argv),
                ToolExited(command_id=command_id, returncode=0),
            )
        )
    commands = [item for item in items if isinstance(item, (ToolStarted, ToolExited))]
    if commands != expected_commands:
        raise ValueError("prepared worker changed exact command order or exit")
    starts = [item for item in items if isinstance(item, UploadStarted)]
    completions = [item for item in items if isinstance(item, UploadCompleted)]
    source_roles = ("reference_source",) if downloaded else ()
    expected_roles = (
        (*source_roles, *PREPARED_ROLES)
        if durable
        else (*source_roles, *PREPARED_ROLES, "worker_receipt")
    )
    expected_keys = (
        *((request.source.object_key,) if downloaded else ()),
        *(item.object_key for item in request.outputs),
        request.worker_receipt_object_key,
    )
    if durable:
        expected_keys = expected_keys[:-1]
    if (
        tuple(item.role for item in starts) != expected_roles
        or tuple(item.role for item in completions) != expected_roles
        or tuple((item.bucket, item.object_key) for item in starts)
        != tuple((storage.bucket, key) for key in expected_keys)
        or tuple((item.bucket, item.object_key) for item in completions)
        != tuple((storage.bucket, key) for key in expected_keys)
        or tuple((item.byte_count, item.sha256) for item in starts)
        != tuple((item.byte_count, item.sha256) for item in completions)
        or (
            downloaded
            and (completions[0].byte_count, completions[0].sha256)
            != (acquired[0].byte_count, acquired[0].sha256)
        )
        or (
            not downloaded
            and (completions[0].byte_count, completions[0].sha256)
            != (
                request.source.plain_fasta_byte_count,
                request.source.plain_fasta_sha256,
            )
        )
    ):
        raise ValueError("prepared worker changed ordered upload identities")
    prepared = completions[len(source_roles) : len(source_roles) + 8]
    if cpu_marker_extension and tuple(
        (item.role, item.object_key, item.byte_count, item.sha256)
        for item in reference_support[0].objects
    ) != tuple(
        (item.role, item.object_key, item.byte_count, item.sha256) for item in prepared
    ):
        raise ValueError("prepared worker changed derived reference support")
    observed = receipts[0]
    expected_source = (
        {
            "role": "reference_source",
            "object_key": request.source.object_key,
            "byte_count": acquired[0].byte_count,
            "sha256": acquired[0].sha256,
            "md5": request.source_md5,
            "origin": request.source.url,
        }
        if downloaded
        else {
            "role": "reference_source",
            "object_key": source.object.object_key,
            "byte_count": source.object.byte_count,
            "sha256": source.object.sha256,
            "md5": request.source_md5,
            "origin": f"s3://{storage.bucket}/{source.object.object_key}",
        }
    )
    receipt_bytes = canonical_json(observed.receipt).encode("utf-8")
    if (
        observed.receipt.get("schema_version")
        != (
            "bio-neb.parabricks-worker-receipt/5"
            if durable
            else "bio-neb.parabricks-worker-receipt/3"
        )
        or observed.receipt.get("authority") != "non-authoritative-worker-observation"
        or observed.receipt.get("operation") != "prepare_reference"
        or observed.receipt.get("workflow") != "bwa-samtools-reference-preparation"
        or observed.receipt.get("request_id") != request.request_id
        or observed.receipt.get("source") != expected_source
        or observed.receipt_object.bucket != storage.bucket
        or observed.receipt_object.object_key != request.worker_receipt_object_key
        or (
            not durable
            and (observed.receipt_object.byte_count, observed.receipt_object.sha256)
            != (len(receipt_bytes), hashlib.sha256(receipt_bytes).hexdigest())
        )
    ):
        raise ValueError("prepared worker receipt changed exact source or bytes")
    return observed


def _source_observed(storage, request, source, input_verifications, acquired) -> bool:
    """A stored source is verified from S3; a downloaded one matches its approval."""
    if source is None:
        return (
            not input_verifications
            and len(acquired) == 1
            and (acquired[0].url, acquired[0].byte_count, acquired[0].md5)
            == (
                request.source.url,
                request.source.expected_size_bytes,
                request.source_md5,
            )
        )
    return (
        not acquired
        and len(input_verifications) == 1
        and (
            input_verifications[0].role,
            input_verifications[0].bucket,
            input_verifications[0].object_key,
            input_verifications[0].byte_count,
            input_verifications[0].sha256,
        )
        == (
            "reference_source",
            storage.bucket,
            source.object.object_key,
            source.object.byte_count,
            source.object.sha256,
        )
    )


class PreparedReferenceManifest(FrozenContract):
    schema_version: Literal["bio-neb.prepared-reference-build-manifest/2"]
    authority: Literal["application"] = "application"
    derivation_rule: Literal[
        "decompress_gzip_reference_sequences_then_bwa_samtools"
    ] = "decompress_gzip_reference_sequences_then_bwa_samtools"
    # The approved source identity a later supplied-index match reads.
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    source_md5: StrictStr = Field(pattern=r"^[0-9a-f]{32}$")
    request_id: StrictStr
    configuration_id: StrictStr
    bundle_id: StrictStr
    deterministic_job_name: StrictStr
    source: VerifiedObject
    source_manifest: VerifiedObject
    worker_runtime_image: StrictStr
    worker_sha256: StrictStr
    bwa_version: StrictStr
    samtools_version: StrictStr
    commands: tuple[tuple[StrictStr, ...], ...] = Field(min_length=3, max_length=3)
    objects: tuple[VerifiedObject, ...] = Field(min_length=8, max_length=8)
    worker_receipt: VerifiedObject
    whole_prefix_inventory: tuple[StrictStr, ...] = Field(min_length=9, max_length=9)
    # Older external build manifests are descriptors for fresh candidate admission.
    # Current application receipts below require proof, and replay rebinds it.
    transfer_proof: WorkerTransferProof | None = None

    @model_validator(mode="after")
    def complete_bundle(self) -> PreparedReferenceManifest:
        roles = tuple(item.role for item in self.objects)
        keys = tuple(
            sorted(
                (
                    *[item.object_key for item in self.objects],
                    self.worker_receipt.object_key,
                )
            )
        )
        if roles != PREPARED_ROLES:
            raise ValueError("manifest lacks exact prepared-reference roles")
        if self.whole_prefix_inventory != keys:
            raise ValueError("manifest inventory is not the exact data-object prefix")
        if (
            self.source.role != "reference_source"
            or self.source_manifest.role != "source_manifest"
            or self.source.bucket != self.source_manifest.bucket
            or self.worker_receipt.role != "worker_receipt"
            or self.worker_receipt.bucket != self.source.bucket
        ):
            raise ValueError("manifest changed the verified source inventory")
        if self.commands != reference_preparation_commands():
            raise ValueError("manifest changed the fixed preparation commands")
        return self


class PreparedReferenceObjectClosure(FrozenContract):
    objects: tuple[VerifiedObject, ...] = Field(min_length=8, max_length=8)
    worker_receipt: VerifiedObject
    pre_manifest_inventory: tuple[StrictStr, ...] = Field(min_length=9, max_length=9)

    @model_validator(mode="after")
    def exact_order_and_inventory(self) -> PreparedReferenceObjectClosure:
        if tuple(item.role for item in self.objects) != PREPARED_ROLES:
            raise ValueError("object closure changed prepared-reference roles")
        expected = tuple(
            sorted(
                (
                    *(item.object_key for item in self.objects),
                    self.worker_receipt.object_key,
                )
            )
        )
        if self.pre_manifest_inventory != expected:
            raise ValueError("object closure changed pre-manifest inventory")
        return self


def compile_prepared_reference_manifest(
    storage: RunObjectStorageTarget,
    runtime: CpuReferenceRuntime,
    verification_mode: str,
    request: ReferencePreparationRequest,
    source: VerifiedReferenceSource,
    closure: PreparedReferenceObjectClosure,
) -> PreparedReferenceManifest:
    return PreparedReferenceManifest(
        schema_version="bio-neb.prepared-reference-build-manifest/2",
        assembly_accession=request.assembly_accession,
        source_md5=request.source_md5,
        request_id=request.request_id,
        configuration_id=request.configuration_id,
        bundle_id=request.bundle_id,
        deterministic_job_name=request.deterministic_job_name,
        source=source.object,
        source_manifest=source.manifest_object,
        worker_runtime_image=runtime.worker_runtime_image,
        worker_sha256=runtime.worker_sha256,
        bwa_version=runtime.bwa_version,
        samtools_version=runtime.samtools_version,
        commands=reference_preparation_commands(),
        objects=closure.objects,
        worker_receipt=closure.worker_receipt,
        whole_prefix_inventory=closure.pre_manifest_inventory,
        transfer_proof=WorkerTransferProof.bind(
            storage=storage,
            runtime=runtime,
            configuration_id=request.configuration_id,
            request=request,
            source_proof_ids=(source.manifest_object.sha256,),
            inputs=(source.object,),
            outputs=closure.objects,
            worker_receipt=closure.worker_receipt,
            mode=verification_mode,
        ),
    )


_SHA256 = r"^[0-9a-f]{64}$"


class PreparedReferenceCompatibility(FrozenContract):
    worker_request_schema: Literal["bio-neb.parabricks-worker-request/7"] = (
        REQUEST_SCHEMA_VERSION
    )
    worker_image_child_digest: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    worker_sha256: StrictStr | None = Field(default=None, pattern=_SHA256)
    bwa_path: Literal["/usr/bin/bwa"] = "/usr/bin/bwa"
    bwa_version: StrictStr
    samtools_path: Literal["/usr/bin/samtools"] = "/usr/bin/samtools"
    samtools_version: StrictStr

    @field_validator("bwa_version", "samtools_version")
    @classmethod
    def canonical_versions(cls, value: str, info: Any) -> str:
        return _line(value, info.field_name)


def candidate_compatibility(plan) -> PreparedReferenceCompatibility:
    return PreparedReferenceCompatibility(
        worker_image_child_digest=plan.runtimes.h100.worker_runtime_image.rsplit(
            "@", 1
        )[1],
        worker_sha256=plan.runtimes.h100.worker_sha256,
        bwa_version=plan.runtimes.cpu.bwa_version,
        samtools_version=plan.runtimes.cpu.samtools_version,
    )
