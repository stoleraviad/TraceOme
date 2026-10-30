"""DEV-only offline INPUT_STORAGE probe using explicit synthetic input bytes.

The document names an approved-input fixture and its in-memory source bytes.
Fixture approval is never live authority. A fresh retained bundle holds the
stage's bound state; this probe does not alter the fixture's original Journey.
"""

import json
from pathlib import Path

from bio_neb.journey_path import (
    ExecutionBound,
    ExecutionState,
    JourneyPath,
    PathFact,
)
from bio_neb.parabricks.execution_contact_accounting import ExecutionContactJournal
from bio_neb.parabricks.input_storage import INPUT_STORAGE
from bio_neb.parabricks.plan import (
    ApprovedParabricksInput,
    RetainedExecution,
    plan_execution,
)
from bio_neb.parabricks.runner import run_stage
from bio_neb.providers.fake import FakeProvider
from bio_neb.providers.ports import Receipt


def run_input_storage(input_file: Path) -> ExecutionState:
    document = json.loads(input_file.read_text())
    approved = ApprovedParabricksInput.model_validate(document["approved"])
    provider = FakeProvider(
        {
            (INPUT_STORAGE.name, step): [Receipt.model_validate(r) for r in receipts]
            for step, receipts in document.get("fixture_receipts", {}).items()
        },
        sources={
            key: bytes.fromhex(value)
            for key, value in document.get("sources_hex", {}).items()
        },
    )
    plan = plan_execution(
        approved,
        RetainedExecution(ExecutionState(), approved.authority, approved.proposal),
    )
    root = Path(document["run_root"])
    if not root.is_absolute() or root.exists():
        raise ValueError("offline probe requires a fresh absolute run root")
    journey = JourneyPath.create(Path(root))
    _, state = journey.path(
        journey.state,
        PathFact(
            event_type="ExecutionBound",
            kind="authority",
            grounding="human",
            **{"pass": 1},
            payload=ExecutionBound(
                execution_id=approved.execution_id,
                authority_id=approved.authority.authority_id,
                proposal_id=approved.proposal.proposal_id,
                run_id=plan.identity.journey_id,
                approved_input=approved.model_dump(mode="json"),
            ),
        ),
    )
    journal = ExecutionContactJournal(
        Path(root) / "journal.jsonl", approved.execution_id
    )
    state = run_stage(
        state,
        plan,
        INPUT_STORAGE,
        provider,
        journey=journey,
        journal=journal,
        live=None,
    )
    journal.finalize()
    return state
