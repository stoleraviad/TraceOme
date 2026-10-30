"""Derive the contact-free Execution composition plan from approved facts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from pydantic import (
    Field,
    StrictStr,
    model_validator,
)

from bio_neb.application.input_journey import paired_fastq_roles
from bio_neb.application.verification_policy import (
    ObjectVerificationOverrides,
    ResolvedObjectVerificationPolicy,
    resolve_object_verification_policy,
)
from bio_neb.contracts import FrozenContract
from bio_neb.journey import canonical_json_digest
from bio_neb.journey_path import ExecutionState
from bio_neb.parabricks.configuration import (
    LocalReferenceSource,
    ParabricksLaunch,
)
from bio_neb.parabricks.execution_package import ExecutionInputSource
from bio_neb.parabricks.execution_proposal import (
    ExecutionAuthority,
    ExecutionProposal,
)
from bio_neb.parabricks.execution_runtime import Runtimes, runtime_matches_package
from bio_neb.parabricks.worker_exchange import PREPARED_ROLES
from bio_neb.prebuilt_index.choice import PrebuiltIndexCandidate
from bio_neb.prebuilt_index.contigs import RequestedContig
from bio_neb.providers.ports import TransferSource
from bio_neb.providers.runtime import NebiusCliTarget, RunObjectStorageTarget
from bio_neb.reference_discovery.reference_source import GZIP_REFERENCE_SEQUENCES


class ApprovedParabricksInput(FrozenContract):
    """Facts established by planning and exact approval, with their existing evidence."""

    journey_file: StrictStr
    proposal: ExecutionProposal
    authority: ExecutionAuthority
    launch: ParabricksLaunch

    @model_validator(mode="after")
    def associated_approval(self):
        if (
            self.proposal.configuration is not None
            and self.launch != self.proposal.configuration.launch
        ):
            raise ValueError("execution launch differs from reviewed configuration")
        if (
            self.authority.proposal_id != self.proposal.proposal_id
            or self.proposal.execution_package is None
        ):
            raise ValueError("approved Parabricks input lacks exact proposal/package")
        if (
            not Path(self.journey_file).is_absolute()
            or not Path(self.launch.work_directory).is_absolute()
        ):
            raise ValueError("execution locations must be absolute")
        reference = self.proposal.execution_package.reference
        if (
            reference.source_mode != "public_url"
            or reference.source_representation != "gzip_fasta"
            or reference.derivation_rule != GZIP_REFERENCE_SEQUENCES
            or reference.expected_checksum_algorithm != "md5"
            or not PurePosixPath(urlsplit(reference.source_locator).path).name
        ):
            raise ValueError(
                "Parabricks requires an admitted public gzip FASTA with its publisher md5"
            )
        workflow = self.proposal.execution_package.workflow
        if workflow.read_layout != "paired" or {
            (item.name, item.value) for item in workflow.meaning_parameters
        } != {("capture_intervals", "omitted"), ("known_sites", "omitted")}:
            raise ValueError("Parabricks requires the approved paired germline recipe")
        return self

    @property
    def execution_id(self) -> str:
        return "parabricks-execution:" + canonical_json_digest(
            self.model_dump(mode="json")
        )


@dataclass(frozen=True)
class RetainedExecution:
    """Current planning approval plus the once-folded execution path; no reader."""

    execution: ExecutionState
    authority: ExecutionAuthority | None
    proposal: ExecutionProposal | None


class ExecutionCompositionRefused(ValueError):
    """One of the four composition refusal families, with a precise detail."""

    def __init__(self, code: str, *details: str) -> None:
        super().__init__(code)
        self.code = code
        self.details = details


class ExecutionIdentity(FrozenContract):
    execution_id: str
    journey_id: str
    authority_id: str
    proposal_id: str
    execution_package_id: str


class Fastq(FrozenContract):
    role: str
    file_id: str
    registered_path: str
    byte_count: int
    sha256: str | None
    object_key: str


class Reference(FrozenContract):
    """The approved source identity; acquisition measures the bytes."""

    reference_asset_id: str
    assembly_accession: str
    source_url: str
    object_key: str
    expected_size_bytes: int
    expected_md5: str
    # The Run's reference sequences: a prebuilt index is fitted against them.
    reference_sequences: tuple[RequestedContig, ...]
    # The assembly's other sequences by their FASTA name: the CPU build and input
    # storage's measure drop them.
    excluded_sequences: tuple[str, ...]
    # The Execution Review's bound prebuilt index: ADMIT_CANDIDATE downloads, fits
    # and places it. Without one the Run prepares on CPU.
    prebuilt_index: PrebuiltIndexCandidate | None = None


class _PairedFastqs(FrozenContract):
    items: tuple[ExecutionInputSource, ExecutionInputSource]

    @model_validator(mode="after")
    def paired_roles(self):
        if not paired_fastq_roles(item.role for item in self.items):
            raise ValueError("FASTQs must name one complete forward/reverse pair")
        return self


class Layout(FrozenContract):
    reference_bundle_id: str
    input_manifest_object_key: str
    source_object_key: str
    # Where a CPU job writes the source it fetched; each job intent gets its own
    # key, so a replacement job never collides with an earlier job's write-once upload.
    fetched_source_object_key: str
    prepared_object_keys: tuple[str, ...]
    candidate_object_keys: tuple[str, ...]
    candidate_manifest_object_key: str
    h100_output_keys: tuple[str, str, str]
    h100_manifest_object_key: str
    cpu_request_object_key: str
    source_manifest_object_key: str
    reference_manifest_object_key: str
    reference_receipt_object_key: str
    cpu_job_name: str
    h100_job_name: str


class Policies(FrozenContract):
    heartbeat_interval_seconds: float = Field(gt=0, allow_inf_nan=False)
    observation_interval_seconds: float = Field(default=5, gt=0)


@dataclass(frozen=True)
class ExecutionPlan:
    identity: ExecutionIdentity
    scope: NebiusCliTarget
    storage: RunObjectStorageTarget
    layout: Layout
    reference: Reference
    # The controller streams the source for a supplied local FASTA or a bound
    # prebuilt index; otherwise the CPU preparation job fetches the approved URL itself.
    input_source: TransferSource | None
    fastqs: tuple[Fastq, Fastq]
    runtimes: Runtimes
    verification: ResolvedObjectVerificationPolicy
    launch: ParabricksLaunch
    policies: Policies
    amendable: frozenset[str]


def _compile_layout(approved, selected, cpu_runtime, intent_ids) -> Layout:
    reference = approved.proposal.execution_package.reference
    scope = selected.object_scope
    source_name = PurePosixPath(urlsplit(reference.source_locator).path).name
    if source_name == "manifest.json":
        raise ExecutionCompositionRefused(
            "lineage", "reference_source_layout_collision"
        )
    source_key = f"raw/{scope}/{source_name}"
    suffixes = (
        "reference.fa",
        "reference.fa.fai",
        "reference.dict",
        "reference.fa.amb",
        "reference.fa.ann",
        "reference.fa.bwt",
        "reference.fa.pac",
        "reference.fa.sa",
    )
    cpu_intent = intent_ids.get("cpu_preparation")
    if cpu_intent:
        suffixes = tuple(f"{cpu_intent}/{name}" for name in suffixes)
    bundle_id = canonical_json_digest(
        {
            "reference_asset_id": reference.reference_asset_id,
            "assembly_accession": reference.assembly_accession,
            # The records the plain FASTA leaves out: the bundle indexes the Run's
            # reference sequences only.
            "excluded_sequences": list(reference.excluded_sequences),
            # The approved source identity names the bundle; its bytes and the
            # plain FASTA they decompress to are measured at acquisition.
            "source": {
                "object_key": source_key,
                "expected_size_bytes": reference.expected_size_bytes,
                "expected_md5": reference.expected_checksum,
            },
            "worker_runtime_image": cpu_runtime.worker_runtime_image,
            "worker_sha256": cpu_runtime.worker_sha256,
            "bwa_version": cpu_runtime.bwa_version,
            "samtools_version": cpu_runtime.samtools_version,
            "layout": sorted(
                (role, suffix, role == "reference_dict")
                for role, suffix in zip(PREPARED_ROLES, suffixes, strict=True)
            ),
        }
    )
    candidate_prefix = f"candidate-copy/{scope}/"
    output_prefix = f"execution/{scope}" + (
        "/" + intent_ids["h100"] if "h100" in intent_ids else ""
    )
    return Layout(
        reference_bundle_id=bundle_id,
        input_manifest_object_key=f"inputs/{scope}/manifest.json",
        source_object_key=source_key,
        fetched_source_object_key=f"raw/{scope}/"
        + (f"{cpu_intent}/" if cpu_intent else "")
        + source_name,
        prepared_object_keys=tuple(
            f"references/{bundle_id}/{suffix}" for suffix in suffixes
        ),
        candidate_object_keys=tuple(
            f"{candidate_prefix}{role}" for role in PREPARED_ROLES
        ),
        candidate_manifest_object_key=f"{candidate_prefix}admission-manifest.json",
        h100_output_keys=tuple(
            f"{output_prefix}/{name}"
            for name in ("aligned.bam", "variants.vcf", "worker-receipt.json")
        ),
        h100_manifest_object_key=f"{output_prefix}/manifest.json",
        cpu_request_object_key=f"control/{scope}/"
        + (f"{cpu_intent}/" if cpu_intent else "")
        + "reference-preparation-request.json",
        source_manifest_object_key=f"raw/{scope}/"
        + (f"{cpu_intent}/" if cpu_intent else "")
        + "manifest.json",
        reference_manifest_object_key=f"references/{bundle_id}/manifest.json",
        reference_receipt_object_key=f"references/{bundle_id}/worker-receipt.json",
        cpu_job_name="bio-neb-cpu-"
        + intent_ids.get("cpu_preparation", approved.execution_id.rsplit(":", 1)[1])[
            :16
        ],
        h100_job_name="bio-neb-h100-"
        + intent_ids.get("h100", approved.execution_id.rsplit(":", 1)[1])[:16],
    )


def _controller_streams_source(launch: ParabricksLaunch, prebuilt=None) -> bool:
    """A supplied local FASTA or the Review's bound prebuilt index keeps the
    controller stream (decision 4), so a miss's CPU job finds its source placed."""
    return launch.local_reference_source is not None or prebuilt is not None


def _validate_reviewed_settings(approved):
    configuration = approved.proposal.configuration
    if (
        configuration is None
        or approved.launch != configuration.launch
        or approved.proposal.runtime_selection != configuration.runtime_selection
    ):
        raise ExecutionCompositionRefused("lineage", "reviewed_configuration_changed")


def plan_execution(
    approved: ApprovedParabricksInput, state: RetainedExecution
) -> ExecutionPlan:
    """Compose one approved command without reading a file or contacting a provider."""
    try:
        verification = resolve_object_verification_policy(
            approved.launch.verification_mode,
            ObjectVerificationOverrides.model_validate(
                approved.launch.verification_overrides.model_dump()
            ),
        )
    except ValueError as error:
        raise ExecutionCompositionRefused(
            "verification_policy_invalid", "selected_policy"
        ) from error
    if (
        state.authority is None
        or state.proposal is None
        or (
            state.authority.authority_id != approved.authority.authority_id
            or state.proposal.proposal_id != approved.proposal.proposal_id
        )
    ):
        raise ExecutionCompositionRefused("lineage", "authority_not_current")
    bound = state.execution.bound
    if (
        state.authority != approved.authority
        or state.proposal != approved.proposal
        or (
            bound is not None
            and (
                bound.execution_id != approved.execution_id
                or bound.authority_id != approved.authority.authority_id
                or bound.proposal_id != approved.proposal.proposal_id
                or bound.approved_input != approved.model_dump(mode="json")
            )
        )
    ):
        raise ExecutionCompositionRefused("lineage", "approved_facts_drift")
    launch = approved.launch
    _validate_reviewed_settings(approved)
    amendments = state.execution.amendments
    if not set(amendments) <= {"gpu_class"}:
        raise ExecutionCompositionRefused("lineage", "undeclared_amendment")
    local_source = launch.local_reference_source
    if local_source is not None:
        try:
            local_source = LocalReferenceSource.model_validate(
                local_source.model_dump()
            )
        except ValueError as error:
            raise ExecutionCompositionRefused(
                "lineage", "local_reference_invalid"
            ) from error
    context, scope = launch.context, launch.nebius
    topology = context.topology
    if (
        context.controller_runtime,
        context.nebius_executable,
        context.resolved_nebius_executable,
        context.nebius_cli_version,
        context.nebius_executable_byte_count,
        context.nebius_executable_sha256,
        topology.profile,
        topology.tenant_id,
        topology.project_id,
    ) != (
        scope.controller_runtime,
        scope.executable,
        scope.resolved_executable,
        scope.cli_version,
        scope.executable_byte_count,
        scope.executable_sha256,
        scope.profile,
        scope.tenant_id,
        scope.project_id,
    ):
        raise ExecutionCompositionRefused("nebius_context_drift", "selected_context")
    package = approved.proposal.execution_package
    try:
        _PairedFastqs(items=package.inputs)
    except ValueError as error:
        raise ExecutionCompositionRefused("fastq_roles", "paired_inputs") from error
    selection = approved.proposal.runtime_selection
    try:
        runtimes = (
            type(selection).model_validate(selection.model_dump(mode="json")).resolve()
        )
        if not runtime_matches_package(runtimes.h100, package):
            raise ValueError("approved resource requires an exact runtime capability")
    except ValueError as error:
        raise ExecutionCompositionRefused("lineage", "runtime_capability") from error
    if "gpu_class" in amendments:
        try:
            if amendments["gpu_class"] is None:
                raise ValueError("GPU amendment must name a capability profile")
            runtimes = selection.select(
                amended_profile_id=amendments["gpu_class"]
            ).runtimes
        except (AttributeError, ValueError) as error:
            raise ExecutionCompositionRefused(
                "lineage", "invalid_gpu_class_amendment"
            ) from error
    if state.execution.runtime_profile_id is not None:
        try:
            runtimes = selection.select(
                amended_profile_id=amendments.get("gpu_class"),
                substituted_profile_id=state.execution.runtime_profile_id,
            ).runtimes
        except (AttributeError, ValueError) as error:
            raise ExecutionCompositionRefused(
                "lineage", "unapproved_runtime_substitution"
            ) from error
    cpu = runtimes.cpu
    intent_ids = {
        stage: canonical_json_digest(
            {"execution": approved.execution_id, "stage": stage, "answer": answer}
        )
        for stage, answer in state.execution.job_intent_answer_ids.items()
    }
    layout = _compile_layout(approved, launch, cpu, intent_ids)
    reference = package.reference
    prebuilt = reference.prebuilt_index
    return ExecutionPlan(
        identity=ExecutionIdentity(
            execution_id=approved.execution_id,
            journey_id=package.journey_id,
            authority_id=approved.authority.authority_id,
            proposal_id=approved.proposal.proposal_id,
            execution_package_id=package.package_id,
        ),
        scope=launch.nebius,
        storage=launch.object_storage,
        layout=layout,
        reference=Reference(
            reference_asset_id=package.reference.reference_asset_id,
            assembly_accession=package.reference.assembly_accession,
            source_url=package.reference.source_locator,
            object_key=layout.source_object_key,
            expected_size_bytes=reference.expected_size_bytes,
            expected_md5=reference.expected_checksum,
            reference_sequences=reference.reference_sequences,
            excluded_sequences=reference.excluded_sequences,
            prebuilt_index=prebuilt,
        ),
        input_source=TransferSource(
            "local" if local_source is not None else "url",
            local_source.path
            if local_source is not None
            else package.reference.source_locator,
            reference.expected_size_bytes,
            None,
            md5=reference.expected_checksum,
            measure_gzip_content=True,
            excluded_sequences=reference.excluded_sequences,
        )
        if _controller_streams_source(approved.launch, prebuilt)
        else None,
        fastqs=tuple(
            Fastq(
                role=item.role,
                file_id=item.file_id,
                registered_path=item.registered_path,
                byte_count=item.byte_count,
                sha256=item.sha256,
                object_key=f"inputs/{launch.object_scope}/{item.role}.fastq.gz",
            )
            for item in package.inputs
        ),
        runtimes=runtimes,
        verification=verification,
        launch=launch,
        policies=Policies(
            heartbeat_interval_seconds=launch.heartbeat_interval_seconds,
            observation_interval_seconds=approved.proposal.configuration.observation_interval_seconds,
        ),
        amendable=frozenset({"gpu_class"} if selection else ()),
    )


def rederive_plan(plan: ExecutionPlan, state: ExecutionState) -> ExecutionPlan:
    """Recompose from already checked bound authority after a retained decision."""
    approved = ApprovedParabricksInput.model_validate(state.bound.approved_input)
    if plan.identity.execution_id != approved.execution_id:
        raise ExecutionCompositionRefused("lineage", "approved_facts_drift")
    derived = plan_execution(
        approved, RetainedExecution(state, approved.authority, approved.proposal)
    )
    return replace(derived, policies=plan.policies)


def approved_execution_input(journey):
    """Compose the public operation input from its frozen review and exact answer."""
    from bio_neb.parabricks.execution_proposal import current_execution_approval

    authority, proposal = current_execution_approval(journey.read())
    if authority is None or proposal is None or proposal.configuration is None:
        raise ExecutionCompositionRefused(
            "lineage", "complete_approved_configuration_required"
        )
    return ApprovedParabricksInput(
        journey_file=str(journey.path),
        proposal=proposal,
        authority=authority,
        launch=proposal.configuration.launch,
    )
