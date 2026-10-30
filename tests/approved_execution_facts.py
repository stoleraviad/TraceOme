"""Established approval, reference and worker facts for offline execution tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from bio_neb.application.commands import (
    RegisterFastqInput,
    RegisterRunCommand,
    register_run,
)
from bio_neb.journey import Journey
from bio_neb.parabricks.capability_policy import (
    CapabilityPolicyImage,
    CapabilityPolicyRecipe,
    CapabilityPolicyResource,
    CapabilityPolicyTool,
    CapabilitySnapshot,
    CapabilitySnapshotAuthoring,
    RetainedCapabilitySourceInput,
    compile_capability_snapshot_value,
)
from bio_neb.parabricks.execution_proposal import (
    project_execution_hitl,
)
from bio_neb.parabricks.execution_runtime import (
    ATTESTED_IMAGE_CHILD,
    ATTESTED_IMAGE_INDEX,
    ATTESTED_PBRUN_VERSION,
    ATTESTED_PROVIDER_IMAGE,
    ATTESTED_WORKER_SHA256,
)
from bio_neb.parabricks.planning_commands import (
    RespondExecutionHitlCommand,
    respond_execution_hitl,
)
from bio_neb.parabricks.worker_request import (
    PREPARED_ROLES,
    AcceptedPlainFasta,
    ConsumePreparedReferenceParameters,
    ExactWorkerObject,
    PrepareReferenceParameters,
    WorkerObjectStorage,
    WorkerObjectTarget,
    WorkerResources,
)
from bio_neb.providers.nebius_context import (
    NebiusExecutionContextSelection,
    NebiusObjectStorageTopologyCandidate,
)
from tests.application_fakes import _ready_journey, _submit
from tests.execution_approval import approve_execution
from tests.proposal_information_fakes import runtime_configuration


def _h100_policy() -> CapabilitySnapshot:
    content_digest = hashlib.sha256(b"execution-preparation-policy").hexdigest()
    return compile_capability_snapshot_value(
        CapabilitySnapshotAuthoring(
            schema_version="bio-neb.execution-capability-authoring/1",
            capability_version="execution-preparation-tests/1",
            sources=(
                RetainedCapabilitySourceInput(
                    kind="retained_nonlocal",
                    locator="https://tests.invalid/execution-preparation-policy",
                    content_digest=content_digest,
                    retrieved_at="2026-08-10T00:00:00Z",
                ),
            ),
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
                    tool_ids=("pbrun",),
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
                    worker_request_schema="bio-neb.parabricks-worker-request/7",
                    worker_receipt_schema="bio-neb.parabricks-worker-receipt/3",
                    worker_module=(
                        "bio_neb.parabricks.reusable_object_storage_parabricks"
                    ),
                    worker_entrypoint=(
                        "/opt/bio-neb/venv/bin/python",
                        "/opt/bio-neb/reusable_object_storage_parabricks.py",
                    ),
                ),
            ),
            tools=(
                CapabilityPolicyTool(
                    tool_id="pbrun",
                    execution_scope="worker_image",
                    absolute_path="/usr/local/parabricks/pbrun",
                    version=ATTESTED_PBRUN_VERSION,
                    image_id="parabricks-image",
                ),
            ),
        )
    )


def _registered_h100(root: Path, *, model: str = "Qwen/Qwen3.5-397B-A17B"):
    root.mkdir(parents=True, exist_ok=True)
    prompt = root / "prompt.txt"
    runtime = root / "runtime.json"
    paths = {"read_1": root / "r1.fastq", "read_2": root / "r2.fastq"}
    prompt.write_text(
        "Analyze rs1234 for sample-001 Homo sapiens and produce VCF.",
        encoding="utf-8",
    )
    runtime.write_text(json.dumps(runtime_configuration(model=model)), encoding="utf-8")
    for index, path in enumerate(paths.values()):
        path.write_bytes(f"reads-{index}".encode())
    receipt = register_run(
        RegisterRunCommand(
            run_root=str(root / "runs"),
            prompt_file=str(prompt),
            runtime_config_file=str(runtime),
            inputs=tuple(
                RegisterFastqInput(role=role, registered_path=str(path))
                for role, path in paths.items()
            ),
        ),
        repository_root=Path.cwd(),
    )
    return receipt, paths


def _authorized_h100_journey(root: Path, *, registration_factory=_registered_h100):
    with (
        patch(
            "tests.application_fakes._registered",
            side_effect=registration_factory,
        ),
        patch(
            "tests.application_fakes.policy",
            return_value=_h100_policy(),
        ),
    ):
        registration, _ = _ready_journey(root)
    pending = _submit(registration.journey_file)
    respond_execution_hitl(
        RespondExecutionHitlCommand(
            journey_file=registration.journey_file,
            request_event_id=pending.request_event_id,
            response=approve_execution(pending.proposal),
        )
    )
    hitl = project_execution_hitl(Journey.open(Path(registration.journey_file)).read())
    return registration, pending.proposal, hitl.authorities[-1]


def _context(input_config: dict[str, object]) -> NebiusExecutionContextSelection:
    nebius = input_config["nebius"]
    assert isinstance(nebius, dict)
    return NebiusExecutionContextSelection(
        controller_runtime="native_ubuntu",
        nebius_executable="/home/researcher/.nebius/bin/nebius",
        resolved_nebius_executable="/home/researcher/.nebius/bin/nebius",
        nebius_cli_version=nebius["cli_version"],
        nebius_executable_byte_count=nebius["executable_byte_count"],
        nebius_executable_sha256=nebius["executable_sha256"],
        topology=NebiusObjectStorageTopologyCandidate.bind(
            profile="browser-profile",
            tenant_id="tenant-exact",
            project_id="project-exact",
            service_account_id="service-account-exact",
            access_key_resource_id="access-key-resource-exact",
            access_key_id="aws-access-key-exact",
            group_id="group-exact",
            group_membership_id="membership-exact",
            access_permit_id="permit-exact",
        ),
    )


def prepared_authorized_journey(root: Path):
    """Real current preparation with explicitly scripted planning contacts."""
    from tests.planning_assignment_fakes import prepare_configured_run

    registration, review, _ = prepare_configured_run(root)
    request = review.execution_hitl.pending_request
    respond_execution_hitl(
        RespondExecutionHitlCommand(
            journey_file=registration.journey_file,
            request_event_id=request.request_event_id,
            response=approve_execution(request.proposal),
        )
    )
    hitl = project_execution_hitl(Journey.open(Path(registration.journey_file)).read())
    return registration, request.proposal, hitl.authorities[-1]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _storage() -> WorkerObjectStorage:
    return WorkerObjectStorage(
        bucket="neutral-bucket",
        endpoint_url="https://storage.example.invalid",
        region="neutral-region",
    )


def _prepared_targets(prefix: str, fasta: str) -> tuple[WorkerObjectTarget, ...]:
    paths = {
        "genomic_fasta": fasta,
        "fasta_fai": fasta + ".fai",
        "reference_dict": "/work/reference/reference.dict",
        **{
            f"bwa_{extension}": fasta + f".{extension}"
            for extension in ("amb", "ann", "bwt", "pac", "sa")
        },
    }
    return tuple(
        WorkerObjectTarget(
            role=role,
            object_key=f"{prefix}prepared/{role}",
            local_target=paths[role],
        )
        for role in reversed(PREPARED_ROLES)
    )


def _prepare_parameters(
    *, request_id: str = "prepare-neutral", excluded_sequences=("NT_187515.1",)
) -> PrepareReferenceParameters:
    prefix = "runs/neutral/"
    fasta = "/work/reference/reference.fa"
    return PrepareReferenceParameters(
        request_id=request_id,
        storage=_storage(),
        output_prefix=prefix,
        source=ExactWorkerObject(
            role="reference_source",
            object_key="inputs/reference.fa.gz",
            local_target="/work/source/reference.fa.gz",
            byte_count=101,
            sha256=_sha("compressed-reference"),
        ),
        plain_fasta=AcceptedPlainFasta(
            local_target=fasta,
            byte_count=202,
            sha256=_sha("plain-reference"),
        ),
        excluded_sequences=excluded_sequences,
        prepared_outputs=_prepared_targets(prefix, fasta),
        receipt_output=WorkerObjectTarget(
            role="worker_receipt",
            object_key=f"{prefix}receipts/worker.json",
            local_target="/work/receipts/worker.json",
        ),
    )


def _consume_parameters() -> ConsumePreparedReferenceParameters:
    prefix = "runs/neutral/"
    fasta = "/work/reference/reference.fa"
    prepared_targets = _prepared_targets("prepared-source/", fasta)
    inputs = [
        ExactWorkerObject(
            role="read_1",
            object_key="inputs/read-1.fastq.gz",
            local_target="/work/inputs/read-1.fastq.gz",
            byte_count=301,
            sha256=_sha("read-1"),
        ),
        ExactWorkerObject(
            role="read_2",
            object_key="inputs/read-2.fastq.gz",
            local_target="/work/inputs/read-2.fastq.gz",
            byte_count=302,
            sha256=_sha("read-2"),
        ),
        *(
            ExactWorkerObject(
                role=item.role,
                object_key=item.object_key,
                local_target=item.local_target,
                byte_count=index + 400,
                sha256=_sha(item.role),
            )
            for index, item in enumerate(prepared_targets)
        ),
    ]
    outputs = (
        WorkerObjectTarget(
            role="VCF",
            object_key=f"{prefix}outputs/variants.vcf",
            local_target="/work/outputs/variants.vcf",
        ),
        WorkerObjectTarget(
            role="aligned_bam",
            object_key=f"{prefix}outputs/aligned.bam",
            local_target="/work/outputs/aligned.bam",
        ),
    )
    return ConsumePreparedReferenceParameters(
        request_id="consume-neutral",
        storage=_storage(),
        output_prefix=prefix,
        resources=WorkerResources(gpu_count=1, shm_size_bytes=1024),
        inputs=tuple(reversed(inputs)),
        outputs=outputs,
        receipt_output=WorkerObjectTarget(
            role="worker_receipt",
            object_key=f"{prefix}receipts/worker.json",
            local_target="/work/receipts/worker.json",
        ),
        parabricks_argv=(
            "pbrun",
            "germline",
            "--ref",
            fasta,
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
        ),
    )
