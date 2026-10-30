"""Current contact-free operation setup and supplied provider effects.

The historical terminal adapter is retained under the #59 test-source archive.
Current tests use the real operation with supplied, inert provider receipts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from bio_neb.parabricks.execution_contact_accounting import ExecutionContactJournal
from bio_neb.storage_contracts import ObjectStorageCredentialReuseReceipt
from tests.approved_execution_facts import (
    _context,
)
from tests.execution_approval import approve_execution
from tests.prerequisite_builds import measured_build
from tests.reference_fakes import source_acquisition


def storage_config(proposal) -> dict[str, object]:
    package = proposal.execution_package
    assert package is not None
    return {
        "schema_version": "nebius-execution-input-storage/1",
        "nebius": {
            "controller_runtime": "native_ubuntu",
            "executable": "/home/researcher/.nebius/bin/nebius",
            "resolved_executable": "/home/researcher/.nebius/bin/nebius",
            "cli_version": "0.12.263",
            "executable_byte_count": 83259554,
            "executable_sha256": "a" * 64,
            "nebius_cli_profile": "browser-profile",
            "tenant_id": "tenant-exact",
            "project_id": "project-exact",
        },
        "object_storage": {
            "endpoint_url": "https://storage.eu-north1.example.test",
            "region": "eu-north1",
        },
        "bucket_name": "fresh-run-bucket",
        "bucket_domain": "storage.eu-north1.example.test",
        "source_object_key": "inputs/reference/source.fa.gz",
        "fastqs": [
            {
                "role": item.role,
                "registered_path": item.registered_path,
                "object_key": f"inputs/reads/{item.role}.fastq.gz",
            }
            for item in package.inputs
        ],
        "heartbeat_interval_seconds": 5,
        "bounded_get_bytes": 1024,
        "verification_mode": "independent_full_read",
        "verification_policy": None,
    }


def input_storage_provider(plan, journal):
    """Attest fixture identities without transferring their large source objects."""
    from bio_neb.providers.fake import FakeProvider
    from bio_neb.providers.ports import Receipt

    # Fresh synthetic files supply acquired identities for metadata-only inputs.
    # These scripted receipts do not prove a provider transfer.
    inputs = [
        (
            item.byte_count,
            item.sha256
            or hashlib.sha256(Path(item.registered_path).read_bytes()).hexdigest(),
            item.object_key,
        )
        for item in plan.fastqs
    ]
    receipts = [
        Receipt(
            values={"bytes": count, "sha256": digest},
            provider_ids=(f"s3://{plan.storage.bucket}/{key}",),
        )
        for count, digest, key in inputs
    ]
    # The discovered fixture reference's bytes: md5 checked, content measured.
    # Without a supplied local FASTA or index the CPU job fetches it instead.
    acquired = source_acquisition()
    if plan.input_source is not None:
        receipts.append(
            Receipt(
                values={
                    "bytes": acquired.source_byte_count,
                    "sha256": acquired.source_sha256,
                    "md5": acquired.source_md5,
                    "decompressed": {
                        "bytes": acquired.plain_fasta_byte_count,
                        "sha256": acquired.plain_fasta_sha256,
                    },
                },
                provider_ids=(
                    f"s3://{plan.storage.bucket}/{plan.layout.source_object_key}",
                ),
            )
        )
    return FakeProvider(
        {
            ("input_storage", "place_inputs"): receipts,
            ("input_storage", "verify_inputs"): receipts,
        }
    )


@measured_build("approved")
def approved_input(
    root: Path,
    *,
    launch_overrides=None,
    configuration=None,
    investigation_request=None,
    failure=None,
    galaxy=None,
    assembly_report=None,
):
    """Current preparation and exact approval of the test's final choices.

    Supplied effects remain synthetic. ``failure`` is the runtime's failure section;
    ``assembly_report`` is NCBI's report planning reads (`reference_fakes`'s default).
    """
    from bio_neb.journey import Journey
    from bio_neb.parabricks.configuration import ParabricksConfiguration
    from bio_neb.parabricks.plan import approved_execution_input
    from bio_neb.parabricks.planning_commands import (
        RespondExecutionHitlCommand,
        respond_execution_hitl,
    )
    from tests.planning_assignment_fakes import (
        execution_configuration,
        prepare_configured_run,
    )

    root.mkdir(parents=True, exist_ok=True)
    configuration = configuration or execution_configuration(root)
    if launch_overrides:
        material = configuration.model_dump(mode="json")
        material["launch"].update(launch_overrides)
        configuration = ParabricksConfiguration.model_validate(material)
    registration, review, _ = prepare_configured_run(
        root,
        configuration=configuration,
        investigation_request=investigation_request,
        failure=failure,
        galaxy=galaxy,
        assembly_report=assembly_report,
    )
    request = review.execution_hitl.pending_request
    respond_execution_hitl(
        RespondExecutionHitlCommand(
            journey_file=registration.journey_file,
            request_event_id=request.request_event_id,
            response=approve_execution(request.proposal),
        )
    )
    return approved_execution_input(Journey.open(Path(registration.journey_file)))


def with_runtime_choices(configuration):
    """Supply both allowed runtime choices before the real compiler and review."""
    from bio_neb.content import canonical_json_digest
    from bio_neb.parabricks.capability_policy import (
        CapabilitySnapshot,
        ExecutionRuntimeSelection,
    )
    from bio_neb.parabricks.configuration import ParabricksConfiguration

    selected = configuration.runtime_selection
    profile = selected.capability.runtime_profiles[0]
    alternate = profile.model_copy(
        update={
            "profile_id": "alternate",
            "h100": profile.h100.model_copy(
                update={
                    "platform": "gpu-alternative",
                    "preset": "alternative-preset",
                    "disk_size": "500Gi",
                }
            ),
        }
    )
    fallback = alternate.model_copy(
        update={
            "profile_id": "fallback",
            "h100": alternate.h100.model_copy(update={"platform": "gpu-fallback"}),
        }
    )
    claims = selected.capability.model_dump(mode="json", exclude={"snapshot_id"})
    claims["runtime_profiles"] = [
        p.model_dump(mode="json") for p in (alternate, fallback)
    ]
    snapshot = CapabilitySnapshot(snapshot_id=canonical_json_digest(claims), **claims)
    material = configuration.model_dump(mode="json")
    material["runtime_selection"] = ExecutionRuntimeSelection(
        capability=snapshot, profile_ids=("alternate", "fallback")
    ).model_dump(mode="json")
    return ParabricksConfiguration.model_validate(material)


def _credential_receipt() -> ObjectStorageCredentialReuseReceipt:
    return ObjectStorageCredentialReuseReceipt(
        service_account_id="service-account-exact",
        access_key_resource_id="access-key-resource-exact",
        access_key_id="aws-access-key-exact",
        exact_topology_match_count=1,
        access_key_state="ACTIVE",
        secret_state="ACTIVE",
        secret_delivery_mode="mystery_box",
        host_secret_resolved=True,
        worker_secret_version_bound=True,
    )


VCF_BYTES = b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
BAM_BYTES = b"bounded synthetic BAM acquisition bytes"


class _OneGet:
    def __init__(self, body: bytes = VCF_BYTES) -> None:
        self.body = body
        self.calls = []

    def download(
        self,
        object_key,
        destination,
        *,
        expected_byte_count,
        expected_sha256,
        observe_part=None,
    ):
        self.calls.append(object_key)
        if (
            expected_byte_count != len(self.body)
            or expected_sha256 != hashlib.sha256(self.body).hexdigest()
        ):
            raise AssertionError("application changed the exact descriptor")
        destination.write(self.body)
        return SimpleNamespace(
            byte_count=len(self.body),
            sha256=hashlib.sha256(self.body).hexdigest(),
            credential_receipt=_credential_receipt(),
            part_bytes=32 * 1024 * 1024,
            part_count=1,
        )


def populate_synthetic_journal(journal: ExecutionContactJournal) -> None:
    contacts: tuple[
        tuple[Literal["local_runtime", "provider", "storage"], str, str], ...
    ] = (
        ("local_runtime", "nebius.version", "nebius-cli:" + "c" * 64),
        ("provider", "iam.whoami", "nebius-cli:" + "c" * 64),
        ("provider", "storage.bucket.create", "nebius-cli:" + "c" * 64),
        ("provider", "ai.job.create", "nebius-cli:" + "c" * 64),
        ("storage", "s3.head-object", "s3-sdk"),
    )
    for category, operation, executable in contacts:
        unit_id = journal.start(
            category=category, operation=operation, executable_identity=executable
        )
        journal.finish(unit_id, disposition="succeeded", exit_status=0)
    credential = journal.start(
        category="credential",
        operation="selected-object-storage-credential",
        executable_identity="credential-resolver",
        credential_names=(
            "control_plane_authentication",
            "host_secret_resolution",
            "worker_secret_binding",
            "session_credentials",
        ),
        selected_identity="nebius-object-storage-topology:" + "1" * 64,
    )
    journal.finish(credential, disposition="succeeded")


def synthetic_result_context():
    """Explicit metadata for pre-existing focused scientific projection vectors."""
    from bio_neb.parabricks.execution_runtime import (
        ATTESTED_BWA_VERSION,
        ATTESTED_IMAGE_INDEX,
        ATTESTED_PBRUN_VERSION,
        ATTESTED_PROVIDER_IMAGE,
        ATTESTED_PYTHON_VERSION,
        ATTESTED_RUNTIME_IMAGE,
        ATTESTED_RUNTIME_RECEIPT_SHA256,
        ATTESTED_SAMTOOLS_VERSION,
        ATTESTED_WORKER_SHA256,
        ExecutionCompositionRuntime,
    )
    from bio_neb.providers.runtime import RunObjectStorageTarget

    context = _context(
        {
            "nebius": {
                "cli_version": "0.12.263",
                "executable_byte_count": 83259554,
                "executable_sha256": "a" * 64,
            }
        }
    )
    runtime = ExecutionCompositionRuntime(
        receipt_sha256=ATTESTED_RUNTIME_RECEIPT_SHA256,
        nebius_cli_version="0.12.263",
        provider_image_reference=ATTESTED_PROVIDER_IMAGE,
        provider_image_index_digest=ATTESTED_IMAGE_INDEX,
        worker_runtime_image=ATTESTED_RUNTIME_IMAGE,
        worker_sha256=ATTESTED_WORKER_SHA256,
        worker_request_schema="bio-neb.parabricks-worker-request/7",
        bwa_path="/usr/bin/bwa",
        bwa_version=ATTESTED_BWA_VERSION,
        samtools_path="/usr/bin/samtools",
        samtools_version=ATTESTED_SAMTOOLS_VERSION,
        python_path="/opt/bio-neb/venv/bin/python",
        python_version=ATTESTED_PYTHON_VERSION,
        pbrun_path="/usr/local/parabricks/pbrun",
        pbrun_version=ATTESTED_PBRUN_VERSION,
    )
    return dict(
        execution_package_id="execution-package:" + "8" * 64,
        composition_binding_id="nebius-execution-composition:" + "8" * 64,
        selected_context=context,
        runtime=runtime,
        storage=RunObjectStorageTarget(
            endpoint_url="https://storage.example.invalid",
            region="test-region",
            bucket="run-bucket",
            mount_path="/mnt/object-storage",
        ),
        reference=dict(
            configuration_id="reference-preparation-configuration:" + "8" * 64,
            reference_asset_id="reference-asset-exact",
            assembly_accession="GCF_000001405.40",
            plain_fasta_byte_count=101,
            plain_fasta_sha256="a" * 64,
        ),
        h100_configuration_id="h100-execution-configuration:" + "8" * 64,
        accounting_journal_file="/offline/journal.jsonl",
        accounting_ledger_file="/offline/ledger.json",
    )


def execute_and_read_path(approved, **kwargs):
    """Exercise the real operation, then inspect checked facts for stage tests.

    This helper is only for tests of execution internals. Public caller tests
    assert the command's result directly in test_operation.
    """
    from bio_neb.journey_path import JourneyPath
    from bio_neb.parabricks.operation import execute_parabricks

    execute_parabricks(approved, **kwargs)
    return JourneyPath.open(Path(approved.journey_file)).state
