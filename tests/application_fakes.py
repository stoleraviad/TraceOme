"""Reusable application Journey builders and their exact offline providers."""

from __future__ import annotations

import copy
import json
import re
import signal
from contextlib import contextmanager
from pathlib import Path

from bio_neb.application.commands import (
    RegisterFastqInput,
    RegisterRunCommand,
    register_run,
)
from bio_neb.input_inspection.commands import (
    InspectRegisteredInputsCommand,
    inspect_registered_inputs,
)
from bio_neb.journey import Journey
from bio_neb.parabricks.capability_policy import CapabilitySnapshotAdapter
from bio_neb.parabricks.planning_commands import (
    RespondPlanningHitlCommand,
    SubmitExecutionProposalCommand,
    respond_planning_hitl,
    submit_execution_proposal,
)
from bio_neb.parabricks.planning_hitl import (
    SelectPlanningHitlResponse,
)
from tests.planning_fakes import policy
from tests.proposal_information_fakes import (
    proposal_draft,
    runtime_configuration,
)


def _registered(
    root: Path,
    *,
    model: str = "Qwen/Qwen3.5-397B-A17B",
    prompt_text: str = "Analyze rs1234 for sample-001 Homo sapiens and produce VCF.",
    agent_invocation=None,
    failure=None,
    paper_files=(),
    request_files=(),
):
    prompt = root / "prompt.txt"
    config = root / "runtime.json"
    paths = {"read_1": root / "r1.fastq", "read_2": root / "r2.fastq"}
    prompt.write_text(prompt_text, encoding="utf-8")
    configuration = runtime_configuration(model=model)
    if agent_invocation is not None:
        configuration["agent_invocation"] = {
            **configuration["agent_invocation"],
            **agent_invocation,
        }
    if failure is not None:
        configuration["failure"] = failure
    config.write_text(json.dumps(configuration), encoding="utf-8")
    for index, path in enumerate(paths.values()):
        path.write_bytes(f"reads-{index}".encode())
    receipt = register_run(
        RegisterRunCommand(
            run_root=str(root / "runs"),
            prompt_file=str(prompt),
            runtime_config_file=str(config),
            inputs=tuple(
                RegisterFastqInput(role=role, registered_path=str(path))
                for role, path in paths.items()
            ),
            paper_files=tuple(str(path) for path in paper_files),
            request_files=tuple(str(path) for path in request_files),
        ),
        repository_root=Path.cwd(),
    )
    return receipt, paths


def _grounded_draft(review, paths, *, registered_requests=()):
    """The scripted plan; each registered request file is cited by its own item, as
    the planner text asks (#367)."""
    draft = proposal_draft(
        review,
        path_by_role={role: str(path) for role, path in paths.items()},
    )
    if registered_requests:
        target = next(
            item
            for item in draft["requirements"]
            if item["kind"] == "scientific_investigation"
        )
        position = draft["requirements"].index(target)
        cited = []
        for number, registered in enumerate(registered_requests):
            item = copy.deepcopy(target)
            if number:
                item["subject_key"] = f"{target['subject_key']}_file_{number + 1}"
            item["knowledge"] = {
                "state": "to_find",
                "evidence_event_ids": [registered.source_event_id],
            }
            item["acquisition_route"] = "validate_with_tool"
            cited.append(item)
        draft["requirements"][position : position + 1] = cited
    assertions = {"sample_identity": "sample-001"}
    # Like a model, type the exact accession the request states, if any.
    accessions = re.findall(r"\bGC[AF]_[0-9]{9}\.[1-9][0-9]*\b", review.request.prompt)
    if accessions:
        assertions["reference_assembly_identity"] = accessions[0]
    for item in draft["requirements"]:
        if item["kind"] == "taxon_identity":
            # An organism name states no TaxId: the TaxId is to find by that name,
            # the one the request writes (human unless it names C. elegans).
            item["knowledge"]["quote"] = next(
                (
                    name
                    for name in ("Caenorhabditis elegans",)
                    if name in review.request.prompt
                ),
                "Homo sapiens",
            )
            continue
        value = assertions.get(item["kind"])
        if value is None:
            continue
        item["knowledge"] = {"state": "asserted", "value": value, "quote": value}
        item["acquisition_route"] = "use_supplied_assertion"
    return draft


def _planning_ready(
    root: Path, *, model: str = "Qwen/Qwen3.5-397B-A17B", agent_invocation=None
):
    """Reach execution-planning readiness through the real planning assignment.

    Only the model and source adapters are scripted; each scoped question takes its
    first candidate. The composition keeps every model-visible instruction.
    """
    # The composition reuses this module's grounded draft.
    from tests.planning_assignment_fakes import PlanningComposition

    receipt, paths = _registered(root, model=model, agent_invocation=agent_invocation)
    journey = Journey.open(Path(receipt.journey_file))
    inspect_registered_inputs(
        InspectRegisteredInputsCommand(journey_file=str(journey.path)), journey=journey
    )
    composition = PlanningComposition(receipt, paths)
    composition.capability = policy()
    for _ in range(8):
        journey = Journey.open(Path(receipt.journey_file))
        ready = composition.planning_assignment(journey).run(
            journey, environ=composition.environ
        )
        if ready.status == "execution_planning_ready":
            break
        pending = ready.planning_hitl.pending_request
        if pending is None:
            raise AssertionError(ready.status)
        respond_planning_hitl(
            RespondPlanningHitlCommand(
                journey_file=receipt.journey_file,
                request_event_id=pending.request_event_id,
                response=SelectPlanningHitlResponse(
                    action="select", candidate_id=pending.candidates[0].candidate_id
                ),
            )
        )
    else:
        raise AssertionError(ready.status)
    return receipt, composition


def _ready_journey(root: Path, *, model: str = "Qwen/Qwen3.5-397B-A17B"):
    """Reach execution-planning readiness; preparation admits the discovered source.

    No execution configuration is supplied.
    """
    receipt, composition = _planning_ready(root, model=model)
    return receipt, CapabilitySnapshotAdapter(composition.capability)


def _submit(journey_file: str):
    review = submit_execution_proposal(
        SubmitExecutionProposalCommand(journey_file=journey_file)
    )
    if review.execution_hitl is None:
        # A planning gap returns the Run at its planning stop; name the gap.
        from bio_neb.parabricks.preparation import prepare_execution

        gap = prepare_execution(Journey.open(Path(journey_file)))
        raise AssertionError(f"submission met a planning gap: {gap.code}")
    return review.execution_hitl.pending_request


@contextmanager
def operator_sigint():
    """SIGINT reaches the Run as an attached Ctrl-C does: Python's default handler."""
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
