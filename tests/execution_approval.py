"""The researcher's approval of an Execution proposal, as the Review's schema asks it."""

from __future__ import annotations

from bio_neb.parabricks.execution_proposal import (
    ApproveExecutionResponse,
    ExecutionProposal,
)


def approval_json(proposal: ExecutionProposal) -> dict[str, object]:
    """Approve ``proposal``, answering its calling ploidy when it states one: the one
    value it offers (owner #470 R2: the researcher chooses diploid explicitly)."""
    return {
        "action": "approve",
        "proposal_id": proposal.proposal_id,
        **(
            {"calling_ploidy": proposal.ploidy.calling_ploidy}
            if proposal.ploidy is not None
            else {}
        ),
    }


def approve_execution(proposal: ExecutionProposal) -> ApproveExecutionResponse:
    return ApproveExecutionResponse.model_validate(approval_json(proposal))
