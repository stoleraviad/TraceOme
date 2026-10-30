from __future__ import annotations

import hashlib
import json
import socket
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.agentic_journey.planning_assignment import planning_assignment
from bio_neb.application.advancement import (
    AnswerReview,
    PlanningReviewAnswer,
    ResumeRun,
    advance,
    use_run_advancement_composition,
)
from bio_neb.application.commands import (
    RegisterFastqInput,
    RegisterRunCommand,
    register_run,
)
from bio_neb.application.queries import (
    ReviewJourneyQuery,
    review_journey,
)
from bio_neb.input_inspection.tool import (
    FilesystemIdentity,
    InspectedInput,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.parabricks.capability_policy import CapabilitySnapshotAdapter
from bio_neb.parabricks.execution_package import (
    ExecutionPackage,
    describe_execution_input,
)
from bio_neb.parabricks.execution_proposal import (
    ExecutionProposal,
    compile_execution_proposal,
    execution_authority_id,
)
from bio_neb.parabricks.planning_commands import (
    RespondPlanningHitlCommand,
    respond_planning_hitl,
)
from bio_neb.parabricks.planning_hitl import (
    ClarifyPlanningHitlResponse,
    SelectPlanningHitlResponse,
)
from bio_neb.parabricks.requirement_resolution import (
    project_requirement_resolution,
)
from bio_neb.providers.ncbi_datasets import NcbiDatasetsReferenceAdapter
from bio_neb.recording import read_evidence
from bio_neb.reference_discovery.reference_companions_tool import (
    CompanionAdapterObservation,
    CompanionSourceRecord,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    AssemblyAdapterObservation,
    AssemblySourceRecord,
    SourceObservation,
    TaxonomyAdapterObservation,
    TaxonomySourceRecord,
)
from bio_neb.reference_discovery.reference_source import (
    ReferenceSourceAdmission,
    reference_source_admission_id,
)
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.planning_assignment_fakes import (
    PlanningComposition,
    execution_configuration,
    prepare_configured_run,
)
from tests.proposal_information_fakes import runtime_configuration
from tests.reference_fakes import SOURCE_GZIP, AssemblyReportFake

_SAMPLE = "neutral-sample-omega"
_TAXON = "Neutralis exemplaris"
_TAX_ID = 424242
_ASSEMBLY = "GCA_123456789.7"
# A typed position the Run can check (words stating none are asked).
_MARKER = "chr7:42424 A>G"
_NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
_REFERENCE_SIZE = 972_898_531
_REFERENCE_MD5 = hashlib.md5(b"authoritative compressed reference identity").hexdigest()


def _inspected_input(path: Path, **changes) -> InspectedInput:
    observed = path.stat()
    return InspectedInput(
        role="read_1",
        registered_path=changes.get("registered_path", str(path)),
        exists=True,
        is_regular=True,
        size_bytes=changes.get("size_bytes", observed.st_size),
        filesystem_identity=FilesystemIdentity(
            device=changes.get("device", str(observed.st_dev)),
            inode=changes.get("inode", str(observed.st_ino)),
        ),
        modified_at_ns=changes.get("modified_at_ns", observed.st_mtime_ns),
    )


class PreparationOutcomeTests(unittest.TestCase):
    def test_missing_planning_information_returns_gap_without_writes(self):
        from bio_neb.parabricks.planning_readiness import PlanningIncomplete
        from bio_neb.parabricks.preparation import prepare_execution

        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary))
            before = journey.read()
            outcome = prepare_execution(journey)
            self.assertIsInstance(outcome, PlanningIncomplete)
            self.assertEqual(outcome.code, "proposal_planning_not_ready")
            self.assertEqual(journey.read(), before)

    def test_discovered_reference_reaches_review_binding_its_source_identity(self):
        """The prompt and FASTQ pair suffice: nobody types the reference identity.

        The Review's proposal binds the source the NCBI adapter observed (accession,
        asset, file URL, publisher md5, HEAD size); the configuration carries no
        reference byte identity.
        """
        with tempfile.TemporaryDirectory() as temporary:
            _, review, configured = prepare_configured_run(Path(temporary).resolve())
        proposal = review.current_review.request.proposal
        admission = proposal.reference_source_admission
        ftp = "https://ftp.ncbi.nlm.nih.gov/genomes/all/offline"
        self.assertEqual(
            (
                admission.assembly_accession,
                admission.reference_asset_id,
                admission.source_locator,
                admission.expected_checksum_algorithm,
                admission.expected_checksum,
                admission.expected_size_bytes,
                admission.checksum_authority_locator,
            ),
            (
                "GCF_000001405.40",
                "source-owned-opaque-id",
                f"{ftp}/GCF_000001405.40_genomic.fna.gz",
                "md5",
                hashlib.md5(SOURCE_GZIP).hexdigest(),
                len(SOURCE_GZIP),
                f"{ftp}/md5checksums.txt",
            ),
        )
        supplied = configured.model_dump(mode="json")
        self.assertNotIn("reference_source", supplied)
        self.assertNotIn("accepted_plain_fasta", supplied["launch"])


class ExecutionInputDescriptionTests(unittest.TestCase):
    def test_retained_metadata_does_not_claim_content_proof_or_reinspect(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "reads.fastq"
            path.write_bytes(b"abc")
            inspected = _inspected_input(path)
            path.unlink()
            with patch.object(Path, "stat", side_effect=AssertionError("reinspection")):
                source = describe_execution_input(
                    inspected,
                    "11111111-1111-4111-8111-111111111111",
                    "22222222-2222-4222-8222-222222222222",
                )
        self.assertEqual(source.registered_path, str(path))
        self.assertEqual(source.byte_count, 3)
        self.assertIsNone(source.sha256)
        self.assertEqual(source.filesystem_inode, inspected.filesystem_identity.inode)


def _source(authority: str, label: str) -> SourceObservation:
    return SourceObservation(
        authority=authority,
        adapter_version="neutral-fixture/v1",
        source_version="neutral-source/v1",
        request_identity=f"https://neutral.invalid/{label}",
        response_sha256=hashlib.sha256(label.encode()).hexdigest(),
        observed_at=_NOW,
        citation_url=f"https://neutral.invalid/{label}",
        elapsed_time_ms=0,
        response_bytes=1,
    )


class _NeutralReferenceAdapter:
    # The fake agrees with the real port's one-taxon contract (ADR 0020).
    taxon_term_refusal = NcbiDatasetsReferenceAdapter.taxon_term_refusal

    def __init__(self) -> None:
        self.fake_calls = 0

    @staticmethod
    def _record() -> AssemblySourceRecord:
        return AssemblySourceRecord(
            assembly_accession=_ASSEMBLY,
            assembly_name="neutral-assembly-seven",
            organism_name=_TAXON,
            ncbi_tax_id=_TAX_ID,
            assembly_status="current",
            refseq_category=None,
            assembly_level="chromosome",
        )

    def resolve_taxon(self, query: str) -> TaxonomyAdapterObservation:
        self.fake_calls += 1
        if query != _TAXON:
            raise AssertionError(query)
        return TaxonomyAdapterObservation(
            records=(
                TaxonomySourceRecord(
                    ncbi_tax_id=_TAX_ID, scientific_name=_TAXON, rank="species"
                ),
            ),
            source=_source("ncbi_taxonomy", "taxonomy/neutral"),
        )

    def discover_assemblies(self, ncbi_tax_id: int, query):
        self.fake_calls += 1
        if ncbi_tax_id != _TAX_ID:
            raise AssertionError((ncbi_tax_id, query))
        return AssemblyAdapterObservation(
            records=(self._record(),),
            source=_source("ncbi_datasets", "assemblies/neutral"),
            total_count=1,
        )

    def describe_assembly(self, assembly_accession: str) -> AssemblyAdapterObservation:
        self.fake_calls += 1
        if assembly_accession != _ASSEMBLY:
            raise AssertionError(assembly_accession)
        return AssemblyAdapterObservation(
            records=(self._record(),),
            source=_source("ncbi_datasets", "assembly/neutral-seven"),
            total_count=1,
        )


class _NeutralCompanionAdapter(AssemblyReportFake):
    contact_count = 0

    def __init__(self) -> None:
        self.fake_calls = 0

    def describe_companions(self, assembly_accession: str, role: str):
        self.fake_calls += 1
        if (assembly_accession, role) != (_ASSEMBLY, "genomic_fasta"):
            raise AssertionError((assembly_accession, role))
        return CompanionAdapterObservation(
            records=(
                CompanionSourceRecord(
                    asset_id="neutral-authority:assembly-seven:fasta",
                    assembly_accession=_ASSEMBLY,
                    role="genomic_fasta",
                    publisher="Neutral Fixture Authority",
                    release_version="release-seven",
                    representation="neutral genomic nucleotide FASTA",
                    authority_locator="https://neutral.invalid/assembly-seven/fasta",
                    # The publisher file, its md5 and HEAD size, as observed.
                    source_locator="https://neutral.invalid/assembly-seven/fasta.gz",
                    checksum_algorithm="md5",
                    checksum=_REFERENCE_MD5,
                    byte_size=_REFERENCE_SIZE,
                    checksum_authority_locator=(
                        "https://neutral.invalid/assembly-seven/md5checksums.txt"
                    ),
                ),
            ),
            source=_source("ncbi_datasets", "companions/neutral-seven"),
        )


class _NeutralPlanning(PlanningComposition):
    """Scripts the neutral proposal; coordinator, Tools and neutral sources run."""

    def __init__(self, receipt, paths, capability):
        super().__init__(receipt, paths)
        self.reference_adapter = _NeutralReferenceAdapter()
        self.companion_adapter = _NeutralCompanionAdapter()
        self.recipe_adapter = CapabilitySnapshotAdapter(capability)

    async def respond(self, messages, info):
        tools = {tool.name for tool in info.function_tools}
        if (
            not self.calls
            or "inspect_input_metadata" in tools
            or "record_proposal_requirements" not in tools
        ):
            return await super().respond(messages, info)
        inspection = next(
            event
            for event in reversed(self.journey.read())
            if event["event_type"] == "InputMetadataInspected"
        )
        inspected = SimpleNamespace(
            request=self.registered.request,
            registered_input_set=self.registered.registered_input_set,
            inspection=SimpleNamespace(source_event_id=inspection["event_id"]),
        )
        proposal_draft = _draft(inspected, self.paths)
        sample_draft = next(
            item
            for item in proposal_draft["requirements"]
            if item["kind"] == "sample_identity"
        )
        sample_draft["knowledge"] = {
            "state": "to_find",
            "evidence_event_ids": [inspected.request.source_event_id],
        }
        sample_draft["acquisition_route"] = "ask_planning_hitl"
        proposal_draft["requirements"].append(
            {
                "kind": "scientific_investigation",
                "subject_key": "post_execution_variant",
                "domain": "scientific",
                "meaning": "Retain the requested variant for post-execution investigation.",
                # The model types the site its quote writes.
                "knowledge": {
                    "state": "asserted",
                    "quote": _MARKER,
                    "site": {
                        "contig": "chr7",
                        "position": 42424,
                        "ref": "A",
                        "alt": "G",
                    },
                },
                "acquisition_route": "validate_with_tool",
                "validation": {
                    "kind": "tool_evidence",
                    "requirement": "Investigate after exact VCF observations exist.",
                },
                "investigation_purpose": "known_variant",
            }
        )
        self.calls.append("record_proposal_requirements")
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "record_proposal_requirements",
                    proposal_draft,
                    tool_call_id=f"assignment-{len(self.calls)}",
                )
            ]
        )

    def planning_assignment(self, journey):
        self.journey = journey
        self.registered = review_journey(
            ReviewJourneyQuery(journey_file=str(journey.path))
        )
        return planning_assignment(
            reference_adapter=self.reference_adapter,
            companion_adapter=self.companion_adapter,
            galaxy_adapter=self.galaxy_adapter,
            recipe_adapter=self.recipe_adapter,
            model_factory=ScriptedModelEndpoint(self.respond),
        )


def _draft(review, paths: dict[str, Path]) -> dict[str, object]:
    requirements = []
    # Typed as a model would; the organism name states no TaxId, so the TaxId is
    # to find by that name. Execution always produces BAM + VCF: no output requirement.
    for kind, key, knowledge in (
        ("research_goal", "requested_variant", {"quote": _MARKER}),
        ("sample_identity", "sample_identity", {"value": _SAMPLE, "quote": _SAMPLE}),
    ):
        requirements.append(
            {
                "kind": kind,
                "subject_key": key,
                "domain": "scientific",
                "meaning": f"Neutral fixture requirement for {kind}.",
                "knowledge": {"state": "asserted", **knowledge},
                "acquisition_route": "use_supplied_assertion",
                "validation": (
                    {"kind": "none"}
                    if kind == "research_goal"
                    else {
                        "kind": "tool_evidence",
                        "requirement": f"Validate the asserted {kind}.",
                    }
                ),
            }
        )
    for role in paths:
        requirements.append(
            {
                "kind": "registered_input",
                "subject_key": "registered_" + role.casefold().replace("-", "_"),
                "domain": "operational",
                "meaning": "The exact neutral registered FASTQ path.",
                "knowledge": {
                    "state": "observed",
                    "input_role": role,
                    "fact": "registered_path",
                },
                "acquisition_route": "validate_with_tool",
                "validation": {"kind": "none"},
            }
        )
    requirements.append(
        {
            "kind": "taxon_identity",
            "subject_key": "taxon_identity",
            "domain": "scientific",
            "meaning": "Neutral fixture requirement for taxon_identity.",
            "knowledge": {
                "state": "to_find",
                "evidence_event_ids": [review.request.source_event_id],
                "quote": _TAXON,
            },
            "acquisition_route": "validate_with_tool",
            "validation": {
                "kind": "tool_evidence",
                "requirement": "Validate the stated organism.",
            },
        }
    )
    for kind, key in (
        ("reference_assembly_identity", "reference_assembly"),
        ("companion_asset", "reference_companions"),
        ("analysis_recipe", "analysis_recipe"),
        ("resource_profile", "resource_profile"),
    ):
        requirements.append(
            {
                "kind": kind,
                "subject_key": key,
                "domain": "operational" if kind == "resource_profile" else "scientific",
                "meaning": f"Neutral fixture evidence for {kind}.",
                "knowledge": {
                    "state": "to_find",
                    "evidence_event_ids": [review.request.source_event_id],
                },
                "acquisition_route": "validate_with_tool",
                "validation": {
                    "kind": "tool_evidence",
                    "requirement": f"Obtain evidence for {kind}.",
                },
            }
        )
    return {"requirements": requirements}


class ExecutionPackageNeutralAcceptanceTests(unittest.TestCase):
    def test_unresolved_sample_resolves_to_exact_package_and_scientific_handoff(  # noqa: PLR0915 - Preserve the integrated package scenario; test decomposition belongs to264.
        self,
    ):
        read_bodies = (b"neutral forward reads\n", b"neutral reverse reads\n")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = {
                "forward_read": root / "alpha.reads",
                "reverse_read": root / "beta.reads",
            }
            for path, body in zip(paths.values(), read_bodies, strict=True):
                path.write_bytes(body)
            prompt = root / "prompt.txt"
            prompt.write_text(
                f"Analyze {_MARKER} for {_SAMPLE} {_TAXON} and produce germline VCF.",
                encoding="utf-8",
            )
            configuration = execution_configuration(root)
            runtime_settings = runtime_configuration()
            runtime_settings["parabricks"] = configuration.model_dump(mode="json")
            runtime = root / "runtime.json"
            runtime.write_text(json.dumps(runtime_settings), encoding="utf-8")
            receipt = register_run(
                RegisterRunCommand(
                    run_root=str((root / "runs").resolve()),
                    prompt_file=str(prompt.resolve()),
                    runtime_config_file=str(runtime.resolve()),
                    inputs=tuple(
                        RegisterFastqInput(
                            role=role, registered_path=str(path.resolve())
                        )
                        for role, path in paths.items()
                    ),
                ),
                repository_root=Path.cwd(),
            )
            composition = _NeutralPlanning(
                receipt, paths, configuration.runtime_selection.capability
            )
            original_open = Path.open

            def evidence_only_open(path, *args, **kwargs):
                if str(path) in {str(item) for item in paths.values()}:
                    raise AssertionError("preparation read FASTQ bytes before approval")
                return original_open(path, *args, **kwargs)

            with (
                use_run_advancement_composition(composition),
                patch.object(Path, "open", evidence_only_open),
                patch.object(
                    socket,
                    "create_connection",
                    side_effect=AssertionError(
                        "offline tracer attempted network contact"
                    ),
                ),
            ):
                pending_sample = advance(ResumeRun(journey_file=receipt.journey_file))
                self.assertEqual(
                    (pending_sample.status, pending_sample.failure_code),
                    ("planning_hitl_pending", None),
                )
                initial_resolution = pending_sample.requirement_resolution
                initial_sample = next(
                    item
                    for item in initial_resolution.requirements
                    if item.kind == "sample_identity"
                )
                self.assertEqual(initial_sample.state, "needs_review")
                self.assertFalse(initial_resolution.ready_for_execution_compilation)
                self.assertEqual(
                    tuple(
                        item.purpose
                        for item in initial_resolution.scientific_investigation.items
                    ),
                    ("known_variant",),
                )
                sample_request = pending_sample.planning_hitl.pending_request
                self.assertEqual(
                    sample_request.requirement_id, initial_sample.requirement_id
                )
                answered_sample = respond_planning_hitl(
                    RespondPlanningHitlCommand(
                        journey_file=receipt.journey_file,
                        request_event_id=sample_request.request_event_id,
                        response=ClarifyPlanningHitlResponse(
                            action="clarify", instruction=_SAMPLE
                        ),
                    )
                )
                sample_answer_event_id = answered_sample.response_event_id
                answered_review = review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                )
                asserted_resolution = answered_review.requirement_resolution
                asserted_sample = next(
                    item
                    for item in asserted_resolution.requirements
                    if item.kind == "sample_identity"
                )
                self.assertEqual(asserted_sample.state, "needs_review")
                self.assertFalse(asserted_sample.values)
                self.assertEqual(
                    asserted_sample.user_assertions[-1].assertion_kind,
                    "planning_hitl",
                )
                pending_recipe = advance(ResumeRun(journey_file=receipt.journey_file))
                admitted_resolution = project_requirement_resolution(
                    read_evidence(Journey.open(Path(receipt.journey_file)))
                )
                admitted_sample = admitted_resolution.one_value("sample_identity")
                # The attested sample is the researcher's answer, still unverified.
                self.assertEqual(
                    (admitted_sample.source, admitted_sample.status),
                    ("review_answer", "asserted"),
                )
                self.assertIn(sample_answer_event_id, admitted_sample.lineage_event_ids)
                eligibility = pending_recipe.planning_hitl.pending_request
                review = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=eligibility.request_event_id,
                        answer=PlanningReviewAnswer(
                            kind="planning",
                            response=SelectPlanningHitlResponse(
                                action="select",
                                candidate_id=eligibility.candidates[0].candidate_id,
                            ),
                        ),
                    )
                )
            self.assertEqual(review.status, "execution_hitl_pending")
            retained = read_evidence(Journey.open(Path(receipt.journey_file)))
            accepted_timeline = retained[
                : next(
                    index
                    for index, event in enumerate(retained)
                    if event["event_type"] == "ExecutionPackageCompiled"
                )
            ]
            resolution = project_requirement_resolution(retained)
            self.assertTrue(resolution.ready_for_execution_compilation)
            self.assertEqual(
                resolution.scientific_investigation,
                initial_resolution.scientific_investigation,
            )
            sample_fact = resolution.one_value("sample_identity")
            accepted_planned = compile_execution_proposal(accepted_timeline)
            self.assertEqual(
                accepted_planned.requirement_resolution,
                resolution.execution_resolution,
            )
            request = review.execution_hitl.pending_request
            self.assertFalse(review.execution_hitl.authorities)
            self.assertFalse(
                any(
                    item["event_type"] == "ExecutionAuthorityGranted"
                    for item in retained
                )
            )
            proposal = request.proposal
            self.assertEqual(
                proposal.requirement_resolution,
                resolution.execution_resolution,
            )
            from bio_neb.parabricks.preparation import (
                PreparationReady,
                prepare_execution,
            )

            prepared_journey = Journey.open(Path(receipt.journey_file))
            before_replay = prepared_journey.read()
            replay = prepare_execution(prepared_journey)
            self.assertIsInstance(replay, PreparationReady)
            self.assertEqual(replay.proposal, proposal)
            self.assertEqual(prepared_journey.read(), before_replay)
            package = proposal.execution_package
            self.assertIsNotNone(package)
            assert package is not None
            self.assertEqual(package.workflow.sample_id, sample_fact.value)
            self.assertTrue(
                set(sample_fact.lineage_event_ids)
                <= {
                    item.source_event_id
                    for item in package.source_plan.evidence_closure
                }
            )

            package_json = package.model_dump(mode="json")
            self.assertEqual(
                set(package_json),
                {
                    "package_id",
                    "journey_id",
                    "source_plan",
                    "input_set_id",
                    "inputs",
                    "reference",
                    "workflow",
                    "resource",
                    "outputs",
                },
            )
            banned = {
                "prepared_reference",
                "reference_support",
                "bucket",
                "object_key",
                "mount_path",
                "runtime_path",
                "command_argv",
                "provider_envelope",
                "image_lineage",
                "worker_topology",
                "sample_species",
                "readmission",
                "output_prefix",
            }

            def keys(value):
                if isinstance(value, dict):
                    return set(value) | set().union(
                        *(keys(item) for item in value.values()), set()
                    )
                if isinstance(value, list):
                    return set().union(*(keys(item) for item in value), set())
                return set()

            self.assertFalse(keys(package_json) & banned)
            self.assertEqual(
                tuple((item.byte_count, item.sha256) for item in package.inputs),
                tuple((len(body), None) for body in read_bodies),
            )
            self.assertEqual(
                (
                    package.reference.assembly_accession,
                    package.reference.source_mode,
                    package.reference.source_representation,
                    package.reference.expected_size_bytes,
                    package.reference.expected_checksum_algorithm,
                    package.reference.expected_checksum,
                    package.reference.checksum_authority_locator,
                    package.reference.derivation_rule,
                ),
                (
                    _ASSEMBLY,
                    "public_url",
                    "gzip_fasta",
                    _REFERENCE_SIZE,
                    "md5",
                    _REFERENCE_MD5,
                    "https://neutral.invalid/assembly-seven/md5checksums.txt",
                    "decompress_gzip_reference_sequences",
                ),
            )
            self.assertNotIn("local_path", package.reference.model_fields_set)
            self.assertFalse(
                {"byte_count", "sha256", "fetched_sha256", "indexes", "placement"}
                & set(package.reference.model_dump(mode="json"))
            )
            self.assertEqual(package.workflow.sample_id, _SAMPLE)
            self.assertEqual(
                {(item.role, item.format) for item in package.outputs},
                {("aligned_bam", "BAM"), ("variants", "VCF")},
            )

            self.assertEqual(proposal.execution_package, package)
            self.assertEqual(request.proposal.proposal_id, proposal.proposal_id)
            self.assertEqual(composition.reference_adapter.fake_calls, 3)
            self.assertEqual(composition.companion_adapter.fake_calls, 1)
            self.assertEqual(composition.recipe_adapter.contact_count, 3)

    def test_reference_or_output_byte_drift_changes_package_proposal_and_authority_identity(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            from tests.planning_assignment_fakes import prepare_configured_run

            _, review, _ = prepare_configured_run(Path(temporary).resolve())
            request = review.execution_hitl.pending_request
            proposal = request.proposal
            package = proposal.execution_package
            self.assertIsNotNone(package)
            assert package is not None
            admission = proposal.reference_source_admission
            assert admission is not None
            authority_approval_event_id = "00000000-0000-0000-0000-000000000001"
            original_authority_id = execution_authority_id(
                proposal.proposal_id,
                request.request_event_id,
                authority_approval_event_id,
            )
            authority_changes = (
                {"source_locator": "https://reference.invalid/changed-reference.fa.gz"},
                {
                    "source_representation": "plain_fasta",
                    "derivation_rule": "identity",
                },
                {"expected_size_bytes": admission.expected_size_bytes + 1},
                {"expected_checksum": "1" * 32},
                {
                    "expected_checksum_algorithm": "sha256",
                    "expected_checksum": "1" * 64,
                },
                {
                    "checksum_authority_locator": "https://reference.invalid/changed-checksums.md5"
                },
            )
            for reference_changes in authority_changes:
                admission_material = admission.model_dump(
                    mode="json", exclude={"admission_id"}
                )
                admission_material.update(reference_changes)
                changed_admission = ReferenceSourceAdmission(
                    admission_id=reference_source_admission_id(admission_material),
                    **admission_material,
                )
                changed_package_material = package.model_dump(
                    mode="json", exclude={"package_id"}
                )
                changed_package_material["reference"].update(reference_changes)
                changed_package_material["reference"]["source_admission_id"] = (
                    changed_admission.admission_id
                )
                changed_package_id = (
                    "execution-package:"
                    + hashlib.sha256(
                        canonical_json(changed_package_material).encode()
                    ).hexdigest()
                )
                with self.assertRaises(ValidationError):
                    ExecutionPackage(
                        package_id=package.package_id, **changed_package_material
                    )
                changed_package = ExecutionPackage(
                    package_id=changed_package_id, **changed_package_material
                )
                changed_proposal_material = proposal.model_dump(
                    mode="json", exclude={"proposal_id"}
                )
                changed_proposal_material["reference_source_admission"] = (
                    changed_admission.model_dump(mode="json")
                )
                changed_proposal_material["execution_package"] = (
                    changed_package.model_dump(mode="json")
                )
                changed_proposal_id = (
                    "execution-proposal:"
                    + hashlib.sha256(
                        canonical_json(changed_proposal_material).encode()
                    ).hexdigest()
                )
                changed_proposal = ExecutionProposal(
                    proposal_id=changed_proposal_id, **changed_proposal_material
                )
                self.assertNotEqual(changed_proposal.proposal_id, proposal.proposal_id)
                self.assertNotEqual(
                    execution_authority_id(
                        changed_proposal.proposal_id,
                        request.request_event_id,
                        authority_approval_event_id,
                    ),
                    original_authority_id,
                )

            drift_material = package.model_dump(mode="json", exclude={"package_id"})
            drift_material["outputs"] = list(reversed(drift_material["outputs"]))
            with self.assertRaises(ValidationError):
                ExecutionPackage(package_id=package.package_id, **drift_material)
            drift_id = (
                "execution-package:"
                + hashlib.sha256(canonical_json(drift_material).encode()).hexdigest()
            )
            self.assertNotEqual(
                ExecutionPackage(package_id=drift_id, **drift_material).package_id,
                package.package_id,
            )


if __name__ == "__main__":
    unittest.main()
