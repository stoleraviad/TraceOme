"""Typed Journey fact for an accepted proposal-information plan."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from bio_neb.parabricks.proposal_information import (
    AnsweredTarget,
    ProposalInformationPlan,
)


class ProposalRequirementsIdentified(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: Literal["ProposalRequirementsIdentified"] = (
        "ProposalRequirementsIdentified"
    )
    attempt_id: StrictStr
    input_set_id: StrictStr
    plan: ProposalInformationPlan

    @model_validator(mode="after")
    def matching_identity(self) -> "ProposalRequirementsIdentified":
        if self.plan.input_set_id != self.input_set_id:
            raise ValueError("proposal plan input identity changed")
        return self


class TargetAnswerRead(BaseModel):
    """The planning model's reading of a researcher's answer to a target question:
    the targets it states, each in the answer's words (#462, owner 2026-10-26: the
    answer goes back to planning's target reading as a new stage)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: Literal["TargetAnswerRead"] = "TargetAnswerRead"
    attempt_id: StrictStr
    requirement_id: StrictStr
    answer_event_id: StrictStr
    targets: tuple[AnsweredTarget, ...]


def validate_target_answer_read(
    event: Mapping[str, Any],
) -> TargetAnswerRead | None:
    if event.get("event_type") != "TargetAnswerRead":
        return None
    return TargetAnswerRead.model_validate(
        {"event_type": event["event_type"], **event.get("payload", {})}
    )


def validate_proposal_fact(
    event: Mapping[str, Any],
) -> ProposalRequirementsIdentified | None:
    if event.get("event_type") != "ProposalRequirementsIdentified":
        return None
    return ProposalRequirementsIdentified.model_validate(
        {"event_type": event["event_type"], **event.get("payload", {})}
    )
