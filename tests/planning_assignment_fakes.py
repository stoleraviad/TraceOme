"""Current application work uses domain Tools in one adaptive assignment."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.agentic_journey.planning_assignment import planning_assignment
from bio_neb.application.queries import ReviewJourneyQuery, review_journey
from bio_neb.journey import Journey
from bio_neb.paper_intake.tools import PAPER_TOOLS
from bio_neb.parabricks.capability_policy import CapabilitySnapshotAdapter
from tests.application_fakes import _grounded_draft
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.planning_fakes import policy
from tests.prerequisite_builds import measured_build
from tests.reference_fakes import (
    PREBUILT_INDEX_TOOLS,
    CompanionAdapter,
    ReferenceAdapter,
    companion_arguments,
    recipe_selection_arguments,
    reference_call,
    reference_const_arguments,
)


def eligible_tools(journey, function_tools):
    """The shown Tools whose preconditions hold now, as a well-informed model picks.

    Every Tool of the stage is shown; this scripted model calls only the ones
    admission would admit, unless a scenario scripts a refused call itself.
    """
    from bio_neb.agentic_journey.planning_assignment import planning_precondition

    # The live Run owner may be another instance; read the current file.
    deps = SimpleNamespace(journey=Journey.open(Path(journey.path)))
    # A paper Tool is the model's when the researcher's words cite a paper: a scenario
    # that cites one scripts its calls, so this model leaves them alone.
    return {
        tool.name: tool
        for tool in function_tools
        if tool.name == "search_journey"
        or (
            tool.name not in PAPER_TOOLS
            and planning_precondition(deps, tool.name) is None
        )
    }


@contextmanager
def planned_before_required_targets():
    """Plan as before #342, when a plan could name no target: an old Journey.

    Old Journeys are dev-cycle only (owner, #327 Q3), but the request-needed path
    the owner kept for them needs one (#342): this build emulates the earlier
    admission rule, and the scripted plan records no target.
    """
    from bio_neb.parabricks import proposal_information

    earlier = proposal_information.REQUIRED_PROPOSAL_INFORMATION_KINDS - {
        "scientific_investigation"
    }
    with (
        patch.object(
            proposal_information, "REQUIRED_PROPOSAL_INFORMATION_KINDS", earlier
        ),
        patch.object(PlanningComposition, "records_targets", False),
    ):
        yield


class PlanningComposition:
    """Only external model/source adapters are controlled; coordinator and Tools run."""

    environ = {"LOCAL_MODEL_KEY": "secret"}
    # False only under `planned_before_required_targets`: an old Journey's plan.
    records_targets = True
    # The model checks Galaxy's indexes before describing any companion.
    index_first = False
    # The Investigation's NCBI/EVA sources a suggested assembly's targets are checked
    # with; None makes no such contact.
    variant_sources = None

    def __init__(self, receipt, paths, *, hostile=False, registered=None):
        self.capability = policy()
        # Optional persistent recipe adapter, so a scenario can count or fail contacts.
        self.recipe_adapter = None
        self.hostile = hostile
        self.source_mode = "public_url"
        self.source_locator = "https://example.test/reference.fa.gz"
        self.reference_adapter = ReferenceAdapter()
        self.companion_adapter = CompanionAdapter("GCF_000001405.40")
        # Galaxy's datacache for the genomic FASTA's index search: its four
        # repositories list nothing unless a scenario serves its own.
        from bio_neb.providers.galaxy_datacache import GalaxyDatacache
        from tests.prebuilt_index_support import GalaxyOpener

        self.galaxy_adapter = GalaxyDatacache(opener=GalaxyOpener(listed=False))
        # Europe PMC and the PMC bucket for the paper Tools; a scenario that cites
        # a paper gives recorded answers.
        self.paper_adapter = None
        # The cited_paper requirements the scripted plan declares: a function of
        # the Journey, so a plan can cite a paper it retrieved first. None
        # declares none, as a Run whose words cite no paper.
        self.cited_papers = None
        # A scenario's own plan: a function of the Journey and the scripted
        # requirements, returning the requirements the model records; None keeps them.
        self.plan_requirements = None
        self.contacts_at_refused_action = None
        self.paths = paths
        self.journey = Journey.open(Path(receipt.journey_file))
        # A caller binding mid-Attempt supplies its registration projection.
        self.registered = registered or review_journey(
            ReviewJourneyQuery(journey_file=receipt.journey_file)
        )
        self.calls = []
        self.instructions_seen = []
        self.history_results = []
        self.scientific_intent = None
        self.scientific_rsid = "rs429358"
        self.scientific_version = None
        # The site the model types from the quote, if any.
        self.scientific_site = None
        # The targets the model reads in a target answer (the target_answer stage);
        # None reads the scripted default (`_scripted_answer_reading`).
        self.answer_targets = None
        # One scripted endpoint keeps every sent body across this composition's
        # Attempts; it reads `respond` per request, so a scenario may replace it.
        self.model_endpoint = ScriptedModelEndpoint(
            lambda messages, info: self.respond(messages, info)
        )

    async def respond(self, messages, info):
        self.instructions_seen.append(info.instructions)
        tools = eligible_tools(self.journey, info.function_tools)
        if not self.calls:
            name = "search_journey"
            arguments = {"kinds": ["ResearchRequestRegistered"], "limit": 20}
        elif self.hostile and len(self.calls) == 1:
            assert "execute_approved_parabricks" not in tools
            self.contacts_at_refused_action = (
                self.reference_adapter.contact_count
                + self.companion_adapter.contact_count
            )
            name = "execute_approved_parabricks"
            arguments = {"historical_approval": "not-current-authority"}
        elif "record_answered_targets" in tools:
            name = "record_answered_targets"
            arguments = {
                "targets": _scripted_answer_reading(self.journey)
                if self.answer_targets is None
                else self.answer_targets
            }
        elif "inspect_input_metadata" in tools:
            name = "inspect_input_metadata"
            arguments = {
                "input_set_id": self.registered.registered_input_set.input_set_id
            }
        elif "record_proposal_requirements" in tools:
            name = "record_proposal_requirements"
            inspection = next(
                e
                for e in reversed(self.journey.read())
                if e["event_type"] == "InputMetadataInspected"
            )
            context = SimpleNamespace(
                request=self.registered.request,
                registered_input_set=self.registered.registered_input_set,
                inspection=SimpleNamespace(source_event_id=inspection["event_id"]),
            )
            arguments = _grounded_draft(
                context,
                self.paths,
                registered_requests=Journey.open(Path(self.journey.path)).checked_view(
                    "investigation_request"
                ),
            )
            if self.cited_papers is not None:
                arguments["requirements"].extend(
                    self.cited_papers(Journey.open(Path(self.journey.path)))
                )
            if self.plan_requirements is not None:
                arguments["requirements"] = self.plan_requirements(
                    Journey.open(Path(self.journey.path)), arguments["requirements"]
                )
            if not self.records_targets:
                arguments["requirements"] = [
                    item
                    for item in arguments["requirements"]
                    if item["kind"] != "scientific_investigation"
                ]
            if self.scientific_intent is not None:
                # The prompt's own question replaces the default target; an item
                # citing a registered request file stays beside it.
                arguments["requirements"] = [
                    item
                    for item in arguments["requirements"]
                    if item["kind"] != "scientific_investigation"
                    or item["knowledge"]["state"] == "to_find"
                ]
                arguments["requirements"].append(
                    {
                        "kind": "scientific_investigation",
                        "subject_key": "post_execution_variant",
                        "domain": "scientific",
                        "meaning": "Retain the original scientific question after Execution.",
                        "knowledge": {
                            "state": "asserted",
                            "quote": self.scientific_intent,
                            "value": self.scientific_rsid,
                            # The version the quote writes, as the model names
                            # it.
                            **(
                                {"stated_version": self.scientific_version}
                                if self.scientific_version
                                else {}
                            ),
                            **(
                                {"site": self.scientific_site}
                                if self.scientific_site
                                else {}
                            ),
                        },
                        "acquisition_route": "validate_with_tool",
                        "validation": {
                            "kind": "tool_evidence",
                            "requirement": "Verify from RefSNP.",
                        },
                        "investigation_purpose": "known_variant"
                        if self.scientific_rsid
                        else "open_question",
                    }
                )
        elif (
            call := reference_call(
                Journey.open(Path(self.journey.path)),
                tools,
                index_first=self.index_first,
            )
        ) is not None:
            name, arguments = call
        else:
            # A question comes before any other remaining Tool. The index Tools stay
            # callable after the choice; this model chose.
            name = next(
                name
                for name in (
                    "request_planning_hitl",
                    "request_companion_asset_selection",
                    "request_companion_refinement",
                    *tools,
                )
                if name in tools
                and name != "search_journey"
                and name not in PREBUILT_INDEX_TOOLS
            )
            arguments = (
                reference_const_arguments(tools[name])
                if name == "request_planning_hitl"
                else recipe_selection_arguments(self.journey, tools[name])
                if name == "select_recipe_candidate"
                else companion_arguments(tools[name])
            )
        self.calls.append(name)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    name, arguments, tool_call_id=f"assignment-{len(self.calls)}"
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
            recipe_adapter=self.recipe_adapter
            or CapabilitySnapshotAdapter(self.capability),
            model_factory=self.model_endpoint,
            variant_source_factory=None
            if self.variant_sources is None
            else (lambda: self.variant_sources),
            paper_adapter=self.paper_adapter,
        )


def _scripted_answer_reading(journey):
    """A stand-in for the model's reading of the latest target answer: each rsID it
    writes, in its own words, else the whole answer as one target."""
    import re

    answer = next(
        e["payload"]["response"]["instruction"]
        for e in reversed(journey.read())
        if e["event_type"] == "PlanningHitlResponded"
        and e["payload"]["response"]["action"] == "clarify"
    )
    rsids = [
        {"state": "asserted", "value": "rs" + match.group(1), "quote": match.group(0)}
        for match in re.finditer(r"\b[Rr][Ss](\d+)\b", answer)
    ]
    return rsids or [{"state": "asserted", "quote": answer}]


def execution_configuration(root):
    """Pre-review supplied selections, no approved input; it names no prepared
    reference, so the Run prepares on CPU (owner, 2026-10-29, #464)."""
    from bio_neb.content import canonical_json_digest
    from bio_neb.parabricks.capability_policy import (
        CapabilityPolicyTool,
        CapabilityRuntimeProfile,
        CapabilitySnapshot,
        ExecutionRuntimeSelection,
    )
    from bio_neb.parabricks.configuration import ParabricksConfiguration
    from bio_neb.parabricks.execution_runtime import CpuReferenceRuntime, H100Runtime
    from tests.approved_execution_facts import _context, _h100_policy

    base = _h100_policy()
    profile = CapabilityRuntimeProfile(
        profile_id="h100",
        cpu=CpuReferenceRuntime(),
        h100=H100Runtime(
            recipe_id="germline-approved", resource_profile_id="h100-approved"
        ),
    )
    claims = base.model_dump(mode="json", exclude={"snapshot_id"})
    claims["resources"][0]["tool_ids"] = ["bwa", "pbrun", "python", "samtools"]
    claims["tools"] = [
        CapabilityPolicyTool(
            tool_id=name,
            execution_scope="worker_image",
            absolute_path=path,
            version=version,
            image_id="parabricks-image",
        ).model_dump(mode="json")
        for name, path, version in (
            ("bwa", "/usr/bin/bwa", profile.cpu.bwa_version),
            ("pbrun", profile.h100.pbrun_path, profile.h100.pbrun_version),
            ("python", profile.h100.python_path, profile.h100.python_version),
            ("samtools", "/usr/bin/samtools", profile.cpu.samtools_version),
        )
    ]
    claims["runtime_profiles"] = [profile.model_dump(mode="json")]
    capability = CapabilitySnapshot(snapshot_id=canonical_json_digest(claims), **claims)
    context = _context(
        {
            "nebius": {
                "cli_version": "0.12.263",
                "executable_byte_count": 83259554,
                "executable_sha256": "a" * 64,
            }
        }
    )
    configuration = ParabricksConfiguration.model_validate(
        {
            "runtime_selection": ExecutionRuntimeSelection(
                capability=capability, profile_ids=("h100",)
            ).model_dump(mode="json"),
            "launch": {
                "object_scope": "operation/offline",
                "nebius": {
                    "controller_runtime": "native_ubuntu",
                    "executable": context.nebius_executable,
                    "resolved_executable": context.resolved_nebius_executable,
                    "cli_version": context.nebius_cli_version,
                    "executable_byte_count": context.nebius_executable_byte_count,
                    "executable_sha256": context.nebius_executable_sha256,
                    "profile": "browser-profile",
                    "tenant_id": "tenant-exact",
                    "project_id": "project-exact",
                    "subnet_id": "vpcsubnet-exact123",
                    "registry_id": "registry-exact123",
                },
                "object_storage": {
                    "endpoint_url": "https://storage.eu-north1.example.test",
                    "region": "eu-north1",
                    "bucket": "fresh-run-bucket",
                    "mount_path": "/mnt/object-storage",
                },
                "context": context.model_dump(mode="json"),
                "work_directory": str(root / "execution"),
                "cleanup_owner": "offline-test-owner",
                "heartbeat_interval_seconds": 30,
                "bounded_get_bytes": 8388608,
            },
        }
    )
    return configuration


def parabricks_account(root):
    """The per-account part of an offline Parabricks section (#454)."""
    material = execution_configuration(root).model_dump(mode="json")
    launch = material["launch"]
    return {
        "schema_version": "bio-neb.parabricks-account/1",
        "launch": {
            "nebius": launch["nebius"],
            "context": launch["context"],
            "object_storage": {
                "endpoint_url": launch["object_storage"]["endpoint_url"],
                "region": launch["object_storage"]["region"],
            },
            "cleanup_owner": launch["cleanup_owner"],
        },
        "runtime_selection": material["runtime_selection"],
    }


@measured_build("planning-registered")
def configured_planning_run(
    root,
    *,
    supply_configuration=True,
    configuration=None,
    model="Qwen/Qwen3.5-397B-A17B",
    investigation_request=None,
    scientific_intent=None,
    scientific_rsid="rs429358",
    scientific_version=None,
    scientific_site=None,
    agent_invocation=None,
    failure=None,
    failed_turns=None,
    artifact_verification=None,
    galaxy=None,
    organism="Homo sapiens",
    assembly_report=None,
    prompt=None,
    composition_type=None,
    staged_inputs=None,
):
    """A registered Run with a Parabricks section and its scripted planning
    composition. ``prompt`` replaces the default request (a bare citation, #383 A2),
    and ``composition_type`` a scenario's own scripted planning model.
    ``staged_inputs`` (role to ``s3://`` locator) registers staged objects instead of
    the local FASTQs, which are still written for their bytes (#488)."""
    import json

    from bio_neb.application.commands import (
        RegisterFastqInput,
        RegisterRunCommand,
        register_run,
    )
    from tests.proposal_information_fakes import runtime_configuration

    configuration = configuration or execution_configuration(root)
    runtime = runtime_configuration(model=model)
    if supply_configuration:
        runtime["parabricks"] = configuration.model_dump(mode="json")
    if agent_invocation is not None:
        runtime["agent_invocation"] = {
            **runtime["agent_invocation"],
            **agent_invocation,
        }
    if failure is not None:
        runtime["failure"] = failure
    if failed_turns is not None:
        runtime["agent_invocation"] = {
            **runtime["agent_invocation"],
            "failed_turns": failed_turns,
        }
    if artifact_verification is not None:
        runtime["artifact_verification"] = artifact_verification
    root.mkdir(parents=True, exist_ok=True)
    (root / "runtime.json").write_text(json.dumps(runtime))
    (root / "prompt.txt").write_text(
        prompt
        or f"Analyze rs1234 for sample-001 {organism} and produce VCF."
        + ("\n" + scientific_intent if scientific_intent else "")
    )
    paths = {"read_1": root / "r1.fastq", "read_2": root / "r2.fastq"}
    for role, path in paths.items():
        path.write_bytes(("@" + role + "\nACGT\n+\n!!!!\n").encode())
    # One request document, or a list of several, each its own registered file.
    documents = (
        []
        if investigation_request is None
        else investigation_request
        if isinstance(investigation_request, list)
        else [investigation_request]
    )
    request_files = []
    for number, document in enumerate(documents):
        path = root / (
            "snp-request.json" if not number else f"snp-request-{number}.json"
        )
        path.write_text(json.dumps(document))
        request_files.append(str(path))
    receipt = register_run(
        RegisterRunCommand(
            run_root=str(root / "runs"),
            prompt_file=str(root / "prompt.txt"),
            runtime_config_file=str(root / "runtime.json"),
            request_files=tuple(request_files),
            inputs=tuple(
                RegisterFastqInput(role=k, registered_path=str(v))
                for k, v in (staged_inputs or paths).items()
            ),
        )
    )
    composition = (composition_type or PlanningComposition)(
        receipt, staged_inputs or paths
    )
    composition.scientific_intent = scientific_intent
    composition.scientific_rsid = scientific_rsid
    composition.scientific_version = scientific_version
    composition.scientific_site = scientific_site
    composition.capability = configuration.runtime_selection.capability
    if galaxy is not None:
        # Galaxy's datacache answers the genomic FASTA's index search.
        from bio_neb.providers.galaxy_datacache import GalaxyDatacache

        composition.galaxy_adapter = GalaxyDatacache(opener=galaxy)
    if assembly_report is not None:
        # NCBI's assembly report beside the verified FASTA.
        composition.companion_adapter = CompanionAdapter(
            "GCF_000001405.40", report=assembly_report
        )
    return receipt, composition, configuration


def configured_provider(journey_file, constructed):
    """External transfer/worker effects only; local FASTQs have actual bytes.

    The Run names no prebuilt index, so its scripted CPU job prepares the
    reference before the H100 job (owner, 2026-10-29, #464).
    """
    from tests.h100_stage_fakes import h100_provider
    from tests.parabricks_operation_fakes import input_storage_provider
    from tests.reference_stage_fakes import seed_cpu

    executed = SimpleNamespace(journey_file=journey_file)

    def factory(plan, journal):
        constructed.append(plan)
        provider = seed_cpu(input_storage_provider(plan, journal), plan, executed)
        return h100_provider(provider, plan, executed)

    return factory


@measured_build("review-pending")
def prepare_configured_run(
    root,
    *,
    configuration=None,
    model="Qwen/Qwen3.5-397B-A17B",
    investigation_request=None,
    scientific_intent=None,
    scientific_rsid="rs429358",
    failure=None,
    failed_turns=None,
    artifact_verification=None,
    galaxy=None,
    assembly_report=None,
    staged_inputs=None,
):
    """Produce a real current review from registered bytes and scripted planning contacts."""
    from bio_neb.application.advancement import (
        AnswerReview,
        PlanningReviewAnswer,
        ResumeRun,
        advance,
        use_run_advancement_composition,
    )
    from bio_neb.parabricks.planning_hitl import SelectPlanningHitlResponse

    receipt, composition, configured = configured_planning_run(
        root,
        configuration=configuration,
        model=model,
        investigation_request=investigation_request,
        scientific_intent=scientific_intent,
        scientific_rsid=scientific_rsid,
        failure=failure,
        failed_turns=failed_turns,
        artifact_verification=artifact_verification,
        galaxy=galaxy,
        assembly_report=assembly_report,
        staged_inputs=staged_inputs,
    )
    with use_run_advancement_composition(composition):
        review = advance(ResumeRun(journey_file=receipt.journey_file))
        while (
            review.current_review is not None
            and review.current_review.kind == "planning"
        ):
            question = review.current_review.request
            review = advance(
                AnswerReview(
                    journey_file=receipt.journey_file,
                    request_event_id=question.request_event_id,
                    answer=PlanningReviewAnswer(
                        kind="planning",
                        response=SelectPlanningHitlResponse(
                            action="select",
                            candidate_id=question.candidates[0].candidate_id,
                        ),
                    ),
                )
            )
    if review.current_review is None or review.current_review.kind != "execution":
        raise AssertionError((review.status, review.current_review))
    return receipt, review, configured
