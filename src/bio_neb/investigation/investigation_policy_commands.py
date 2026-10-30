"""Public contact-free policy amendment and checked Investigation decisions."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from bio_neb.agentic_journey.continuation_policy import InvestigationWorkPolicy
from bio_neb.agentic_journey.refusal_disposition import RefusalDisposition
from bio_neb.application.queries import read_investigation
from bio_neb.investigation.investigation_policy import (
    InvestigationPolicyAmended,
    InvestigationPolicyAmendment,
    InvestigationPolicyDecision,
    project_investigation_policy,
)
from bio_neb.journey import Journey, JourneyFile


class InvestigationPolicyApplicationRefused(ValueError):
    def __init__(self, code: str, *fields: str) -> None:
        super().__init__(code)
        self.code = code
        self.fields = tuple(fields)
        self.disposition = RefusalDisposition(kind="caller_correction_and_reissue")


class AmendInvestigationPolicyCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    journey_file: JourneyFile
    amendment: InvestigationPolicyAmendment


def inspect_investigation_policy(journey_file: str) -> InvestigationPolicyDecision:
    return read_investigation(Journey.open(Path(journey_file))).work_policy


def amend_investigation_policy(
    command: AmendInvestigationPolicyCommand,
    *,
    defaults: InvestigationWorkPolicy | None = None,
) -> InvestigationPolicyDecision:
    """Retain only an exact prospective change, never alter evidence or grants."""
    trusted = AmendInvestigationPolicyCommand.model_validate(command)
    journey = Journey.open(Path(trusted.journey_file))
    read = read_investigation(journey)
    amendment = trusted.amendment
    # Exact reissue is a read even after subsequent work, using original defaults.
    for event in read.timeline:
        if event["event_type"] == "InvestigationPolicyAmended" and event["payload"][
            "amendment"
        ] == amendment.model_dump(mode="json"):
            return read.work_policy
    if not read.timeline or not journey.is_current():
        raise InvestigationPolicyApplicationRefused(
            "investigation_policy_head_changed", "journey_file"
        )
    if defaults is not None and (
        read.work_policy.policy_id is not None or read.evidence is not None
    ):
        raise InvestigationPolicyApplicationRefused(
            "investigation_policy_defaults_already_retained", "defaults"
        )
    held = set(read.work_policy.held_work_item_ids)
    held.update(amendment.hold_work_item_ids)
    held.difference_update(amendment.release_work_item_ids)
    fact = InvestigationPolicyAmended.create(
        amendment=amendment,
        defaults=defaults or read.work_policy.effective,
        held_work_item_ids=tuple(sorted(held)),
    )
    prospective = {
        "event_type": fact.event_type,
        "event_id": "prospective-policy-event",
        "payload": fact.model_dump(mode="json", exclude={"event_type"}),
        "parent_ids": [read.timeline[-1]["event_id"]],
    }
    try:
        project_investigation_policy(
            [*read.timeline, prospective], read.investigation, read.evidence
        )
    except ValueError as error:
        # Existing evidence was checked above. This refusal belongs only to the
        # proposed command; malformed retained history is never reclassified.
        raise InvestigationPolicyApplicationRefused(
            "investigation_policy_amendment_ineligible", "amendment"
        ) from error
    if not journey.is_current():
        raise InvestigationPolicyApplicationRefused(
            "investigation_policy_head_changed", "journey_file"
        )
    journey.record(fact.event_type, prospective["payload"], prospective["parent_ids"])
    return read_investigation(journey).work_policy
