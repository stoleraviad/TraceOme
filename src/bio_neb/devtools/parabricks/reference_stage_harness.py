"""DEV offline one-stage probe using explicit historical state and fake receipts."""

import json
from pathlib import Path

from bio_neb.journey_path import (
    Artifact,
    ArtifactPointer,
    JourneyPath,
    project_execution,
)
from bio_neb.parabricks.execution_contact_accounting import ExecutionContactJournal
from bio_neb.parabricks.plan import (
    ApprovedParabricksInput,
    RetainedExecution,
    plan_execution,
)
from bio_neb.parabricks.reference_stages import ADMIT_CANDIDATE, CPU_PREPARATION
from bio_neb.parabricks.runner import run_stage
from bio_neb.providers.fake import FakeProvider
from bio_neb.providers.ports import Receipt


def run_reference_stage(input_file: Path, stage_name: str):
    document = json.loads(input_file.read_text())
    approved = ApprovedParabricksInput.model_validate(document["approved"])
    stage = {"admit-candidate": ADMIT_CANDIDATE, "cpu-preparation": CPU_PREPARATION}[
        stage_name
    ]
    state = project_execution(document["path_events"])
    plan = plan_execution(
        approved, RetainedExecution(state, approved.authority, approved.proposal)
    )
    root = Path(document["run_root"])
    if not root.is_absolute() or root.exists():
        raise ValueError("offline probe requires a fresh absolute run root")
    prepared = JourneyPath.create(root)
    for item in document.get("artifacts", []):
        pointer = ArtifactPointer.model_validate(item["pointer"])
        artifact = Artifact(
            content=bytes.fromhex(item["hex"]),
            role=pointer.role,
            producer=pointer.producer,
            grounding=pointer.grounding,
            proof_ref=pointer.proof_ref,
        )
        if artifact.pointer() != pointer:
            raise ValueError("harness artifact identity mismatch")
        prepared.attach(artifact)
    # Exact synthetic prefix preserves causal ids; it does not authorize live work.
    file = root / "journey.jsonl"
    file.write_text("".join(json.dumps(row) + "\n" for row in document["path_events"]))
    journey = JourneyPath.open(file)
    provider = FakeProvider(
        {
            (stage.name, name): [
                Receipt.model_validate_json(json.dumps(r)) for r in values
            ]
            for name, values in document.get("fixture_receipts", {}).items()
        },
        sources={
            key: bytes.fromhex(value)
            for key, value in document.get("sources_hex", {}).items()
        },
    )
    for item in document.get("objects", []):
        provider.objects[item["bucket"], item["key"]] = bytes.fromhex(item["hex"])
    journal = ExecutionContactJournal(
        root / "journal.jsonl", approved.execution_id, execution_pass=state.pass_
    )
    try:
        return run_stage(
            state, plan, stage, provider, journey=journey, journal=journal, live=None
        )
    finally:
        journal.finalize()
