"""Print each Review kind's response schemas per action, as `observe` publishes them, for `gui/tests/schemas.json`.

    uv run --locked --offline python -m tools.gui_schemas > gui/tests/schemas.json

Mirrors `src/bio_neb/application/reviews.py` `_legal_actions` with synthetic request values, plus the finalization
files' schemas (`correction_file`, `reason_file`) that `observe` does not publish. Rerun it on every M merge into the
GUI and whenever a response class or a finalization file shape changes.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from bio_neb.agentic_journey import agent_work_review as aw
from bio_neb.agentic_journey.effort import (
    ExtendAgentEffortResponse,
    StopAgentEffortResponse,
)
from bio_neb.application.reviews import (
    _model_availability_actions,
    finalization_response_commands,
)
from bio_neb.failure import ReviewAnswer
from bio_neb.investigation.investigation import (
    AmendInvestigationFindingResponse,
    ApproveInvestigationFindingResponse,
    ApproveInvestigationResponse,
    ClarifyInvestigationResponse,
    RejectInvestigationResponse,
    ReviseInvestigationResponse,
    StopInvestigationResponse,
)
from bio_neb.journey_path import Answered
from bio_neb.parabricks import execution_proposal as ep
from bio_neb.parabricks import planning_hitl as ph
from bio_neb.parabricks.preparation_review import (
    StopPreparation,
    SupplyPreparationInformation,
)
from bio_neb.reporting.finalization import (
    FinalizationCorrectionAssertion,
    FinalizationStopAssertion,
)


def _schemas(pairs) -> dict:
    return {action: owner.model_json_schema() for action, owner in pairs}


def _finalization() -> dict:
    """The finalization commands' schemas as `observe` publishes them: the Run's ids left out, the action a const."""
    out = {}
    for action, command in finalization_response_commands().items():
        schema = command.model_json_schema()
        for field in ("journey_file", "request_event_id"):
            schema["properties"].pop(field)
            schema["required"].remove(field)
        schema["properties"]["action"] = {"type": "string", "const": action}
        schema["required"].append("action")
        out[action] = schema
    return out


def schemas_by_kind() -> dict:
    candidates = tuple(
        ph.PlanningHitlCandidate.model_construct(candidate_id=c) for c in ("wgs", "wes")
    )
    proposal = SimpleNamespace(
        proposal_id="execution-proposal:" + "1" * 64, ploidy=None
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short=8", "HEAD"], text=True
    ).strip()
    return {
        "_source": f"lane M response schemas at BioNeb {commit}, from tools/gui_schemas "
        "(mirrors reviews.py _legal_actions)",
        "input": _schemas((a, ReviewAnswer) for a in ("retry", "amend", "abort")),
        "junction": _schemas((a, ReviewAnswer) for a in ("retry", "amend", "stop")),
        "investigation_download": _schemas(
            (a, ReviewAnswer) for a in ("retry", "abort")
        ),
        "run_health": _schemas((a, ReviewAnswer) for a in ("retry", "abort")),
        "paper": _schemas(
            (a, ReviewAnswer) for a in ("retry", "amend", "continue", "abort")
        ),
        "execution_path": _schemas(
            (a, Answered) for a in ("approve_cpu_preparation", "abort")
        ),
        "preparation": _schemas(
            (("supply", SupplyPreparationInformation), ("stop", StopPreparation))
        ),
        "planning": {
            a: ph._response_schema(a, candidates)
            for a in ("select", "clarify", "reject", "stop")
        },
        "model_availability": {
            a.action: a.response_schema for a in _model_availability_actions()
        },
        "agent_effort": _schemas(
            (("extend", ExtendAgentEffortResponse), ("stop", StopAgentEffortResponse))
        ),
        "agent_work": _schemas((a, aw._RESPONSES[a]) for a in ("continue", "reject")),
        "execution": {
            a: ep._response_schema(owner, proposal)
            for a, owner in (
                ("approve", ep.ApproveExecutionResponse),
                ("revise", ep.ReviseExecutionResponse),
                ("reject", ep.RejectExecutionResponse),
            )
        },
        "investigation": _schemas(
            (
                ("clarify", ClarifyInvestigationResponse),
                ("reject", RejectInvestigationResponse),
                ("approve", ApproveInvestigationResponse),
                ("revise", ReviseInvestigationResponse),
                ("stop", StopInvestigationResponse),
            )
        ),
        "investigation_finding": _schemas(
            (
                ("approve", ApproveInvestigationFindingResponse),
                ("amend", AmendInvestigationFindingResponse),
                ("reject", RejectInvestigationResponse),
            )
        ),
        "finalization": _finalization(),
        # The files finalization correct/stop name (finalization_commands._load_correction/_load_stop).
        "finalization_files": {
            "correction_file": FinalizationCorrectionAssertion.model_json_schema(),
            "reason_file": FinalizationStopAssertion.model_json_schema(),
        },
    }


def main() -> None:
    print(json.dumps(schemas_by_kind(), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
