"""Prospective Investigation work policy and its independent Journey interpretation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from pydantic import Field, StrictStr, model_validator

from bio_neb.agentic_journey.continuation_policy import (
    InvestigationCompletion,
    InvestigationWorkPolicy,
    resolve_investigation_work_policy,
)
from bio_neb.contracts import StrictFrozenContract
from bio_neb.investigation.investigation_observation import (
    InvestigationCollectionItemStatus,
)
from bio_neb.journey import canonical_json_digest
from bio_neb.recording import PassClosed, review_head, work_frontier

if TYPE_CHECKING:
    from bio_neb.investigation.investigation import InvestigationProjection
    from bio_neb.investigation.investigation_observation import (
        InvestigationCollectionEvidenceRecorded,
    )


class InvestigationPolicyAmendment(StrictFrozenContract):
    proposal_id: StrictStr
    expected_head_event_id: StrictStr
    previous_policy_id: StrictStr | None
    asserted_by: StrictStr = Field(min_length=1)
    reason: StrictStr = Field(min_length=1)
    hold_work_item_ids: tuple[StrictStr, ...] = ()
    release_work_item_ids: tuple[StrictStr, ...] = ()
    completion: InvestigationCompletion | None = None

    @model_validator(mode="after")
    def exact_changes(self) -> InvestigationPolicyAmendment:
        selected = (*self.hold_work_item_ids, *self.release_work_item_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("policy amendment repeats a member")
        if not selected and self.completion is None:
            raise ValueError("policy amendment requires a prospective change")
        return self


class InvestigationPolicyAmended(StrictFrozenContract):
    event_type: Literal["InvestigationPolicyAmended"] = "InvestigationPolicyAmended"
    amendment: InvestigationPolicyAmendment
    defaults: InvestigationWorkPolicy
    effective: InvestigationWorkPolicy
    held_work_item_ids: tuple[StrictStr, ...]
    policy_id: StrictStr

    @model_validator(mode="after")
    def exact_resolution(self) -> InvestigationPolicyAmended:
        if self.effective != resolve_investigation_work_policy(
            self.defaults, self.amendment.completion
        ) or self.policy_id != "investigation-work-policy:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"event_type", "policy_id"})
        ):
            raise ValueError("Investigation policy resolution changed")
        return self

    @classmethod
    def create(
        cls,
        *,
        amendment: InvestigationPolicyAmendment,
        defaults: InvestigationWorkPolicy,
        held_work_item_ids: tuple[str, ...],
    ) -> InvestigationPolicyAmended:
        material = {
            "amendment": amendment.model_dump(mode="json"),
            "defaults": defaults.model_dump(mode="json"),
            "effective": resolve_investigation_work_policy(
                defaults, amendment.completion
            ).model_dump(mode="json"),
            "held_work_item_ids": held_work_item_ids,
        }
        return cls.model_validate_json(
            json.dumps(
                {
                    **material,
                    "policy_id": "investigation-work-policy:"
                    + canonical_json_digest(material),
                }
            )
        )


class InvestigationItemDecision(StrictFrozenContract):
    work_item_id: StrictStr
    disposition: InvestigationCollectionItemStatus
    held: bool
    continuation: Literal[
        "eligible", "held", "approval_required", "retained", "blocked"
    ]


class InvestigationPolicyDecision(StrictFrozenContract):
    """Completion-policy permission is necessary, not complete command readiness.

    Current head, fresh evidence, exact basis and mandatory approval checks remain
    application-owned. In particular, ``finalization_eligible`` does not authorize
    compilation or report production by itself.
    """

    proposal_id: StrictStr | None = None
    head_event_id: StrictStr | None = None
    amendment_eligible: bool = False
    policy_id: StrictStr | None = None
    effective: InvestigationWorkPolicy = Field(default_factory=InvestigationWorkPolicy)
    held_work_item_ids: tuple[StrictStr, ...] = ()
    eligible_work_item_ids: tuple[StrictStr, ...] = ()
    items: tuple[InvestigationItemDecision, ...] = ()
    finalization_eligible: bool = False
    completion_blockers: tuple[StrictStr, ...] = ()


POLICY_FRONTIERS = {
    "InvestigationReviewRequested",
    "InvestigationAuthorityGranted",
    "InvestigationCollectionEvidenceRecorded",
    "InvestigationPolicyAmended",
    "InvestigationItemEvidenceCollected",
    # An answered download Review lets its grant's preparation run again.
    "InvestigationDownloadReviewAnswered",
}


def _amendment_frontier(
    events: list[dict], proposal_id: str | None, by_id: dict[str, dict]
) -> bool:
    """Whether an amendment may follow ``events[-1]``, the exact head it names.

    A Pass that closed waiting or completed changes no reviewed object, so the
    amendment follows the event that Pass closed on (#287). A held or
    interrupted Pass is no frontier.
    """
    event = events[-1]
    if event["event_type"] == "PassClosed":
        if PassClosed.model_validate(event["payload"]).disposition not in {
            "waiting",
            "completed",
        }:
            return False
        event = review_head(events)
    if event["event_type"] in POLICY_FRONTIERS:
        return True
    # The Investigation projector separately validates the complete Attempt closure.
    # Only its exact successful Review terminal is eligible, never an arbitrary activity.
    if (
        event["event_type"] != "activity_succeeded"
        or event["payload"].get("stage") != "investigation_planning_attempt"
    ):
        return False
    terminal_id = event["payload"].get("result_event_id")
    terminal = by_id.get(terminal_id)
    return bool(
        terminal
        and terminal["event_type"] == "InvestigationReviewRequested"
        and terminal["payload"].get("proposal_id") == proposal_id
        and event.get("parent_ids") == [terminal_id]
    )


def project_investigation_policy(  # noqa: C901 - Existing orchestration complexity; contract-base migration preserves this flow.
    timeline: list[dict],
    investigation: InvestigationProjection,
    evidence: InvestigationCollectionEvidenceRecorded | None,
) -> InvestigationPolicyDecision:
    """Check amendments and later grants against their actual historical membership."""
    proposals = {proposal.proposal_id: proposal for proposal in investigation.proposals}
    current_proposal = None
    current_policy = None
    held: set[str] = set()
    granted: set[str] = set()
    started: set[str] = set()
    failed = False
    retained_policy = None
    by_id = {}
    for index, event in enumerate(timeline):
        by_id[event["event_id"]] = event
        kind, payload = event["event_type"], event["payload"]
        if kind in {
            "InvestigationProposalRecorded",
            "InvestigationProposalCarriedForward",
        }:
            current_proposal = proposals[payload["proposal"]["proposal_id"]]
            held.intersection_update(
                item.work_item_id for item in current_proposal.work_items
            )
        elif kind == "InvestigationWorkItemStarted":
            started.add(payload["work_item_id"])
        elif kind == "InvestigationStopped":
            # A member's fault holds the Run and its member is planned again; an
            # item's failed lookup is accounted (R2). Only a stop ends it.
            failed = True
        elif kind == "InvestigationCollectionEvidenceRecorded":
            recorded = payload.get("work_policy")
            previous_collection_policy = retained_policy
            retained_policy = InvestigationWorkPolicy.model_validate(recorded)
            if recorded != retained_policy.model_dump(mode="json"):
                raise ValueError("retained collection policy must be complete")
            if (
                current_policy is None
                and previous_collection_policy is not None
                and retained_policy != previous_collection_policy
            ):
                raise ValueError("collection policy changed without an amendment")
            if (
                current_policy is not None
                and retained_policy != current_policy.effective
            ) or payload.get("work_policy_id") != (
                current_policy.policy_id if current_policy else None
            ):
                raise ValueError("collection changed its retained work policy")
        elif kind == "InvestigationAuthorityGranted":
            selected = {
                item["work_item_id"]
                for item in payload["authority"]["authorized_work_items"]
            }
            if selected & held:
                raise ValueError("Investigation authority approved a held member")
            granted.update(selected)
        elif kind == "InvestigationPolicyAmended":
            fact = InvestigationPolicyAmended.model_validate_json(json.dumps(payload))
            if fact.model_dump(mode="json", exclude={"event_type"}) != payload:
                raise ValueError("retained Investigation policy must be complete")
            amendment = fact.amendment
            previous = timeline[index - 1] if index else None
            if (
                current_proposal is None
                or amendment.proposal_id != current_proposal.proposal_id
                or previous is None
                or not _amendment_frontier(
                    timeline[:index], current_proposal.proposal_id, by_id
                )
                or amendment.expected_head_event_id != previous["event_id"]
                or event.get("parent_ids") != [previous["event_id"]]
                or amendment.previous_policy_id
                != (current_policy.policy_id if current_policy else None)
                or failed
            ):
                raise ValueError("Investigation amendment changed its exact frontier")
            pending = (
                {item.work_item_id for item in current_proposal.work_items}
                - granted
                - started
            )
            if (
                not set(amendment.hold_work_item_ids) <= pending - held
                or not set(amendment.release_work_item_ids) <= held & pending
            ):
                raise ValueError("Investigation amendment changed pending membership")
            previous_policy = (
                current_policy.effective if current_policy else retained_policy
            )
            if previous_policy is not None and fact.defaults != previous_policy:
                raise ValueError("Investigation amendment changed prior policy")
            held.update(amendment.hold_work_item_ids)
            held.difference_update(amendment.release_work_item_ids)
            if fact.held_work_item_ids != tuple(sorted(held)):
                raise ValueError("Investigation amendment changed held membership")
            current_policy = fact
    policy = (
        current_policy.effective
        if current_policy
        else retained_policy or InvestigationWorkPolicy()
    )
    states = (
        {item.work_item_id: item for item in evidence.item_states} if evidence else {}
    )
    blockers = []
    if evidence is None:
        blockers.append("observation_required")
    if failed or investigation.status == "stopped":
        blockers.append("failed_or_stopped")
    proposal = investigation.proposals[-1] if investigation.proposals else None
    if proposal is not None and proposal.unresolved_request_member_ids:
        blockers.append("unresolved_request_identity")
    items = []
    blocked = (
        failed
        or investigation.status
        not in {"approved", "review_pending", "investigation_work_ready"}
        or not timeline
        or work_frontier(timeline)["event_type"]
        not in POLICY_FRONTIERS | {"InvestigationAssertionRecorded"}
    )
    for item in proposal.work_items if proposal else ():
        identity = item.work_item_id
        state = states.get(identity)
        disposition = (
            state.status
            if state and state.status not in {"approval_pending", "observation_pending"}
            else "observation_pending"
            if identity in granted
            else "approval_pending"
        )
        # A failed lookup is that item's accounted outcome (R2).
        terminal = disposition in {
            "completed",
            "negative",
            "unresolved",
            "refused",
            "failed",
        }
        continuation = (
            "retained"
            if terminal
            else "blocked"
            if blocked
            else "held"
            if identity in held
            else "eligible"
            if identity in granted
            else "approval_required"
        )
        items.append(
            InvestigationItemDecision(
                work_item_id=identity,
                disposition=disposition,
                held=identity in held,
                continuation=continuation,
            )
        )
    if policy.completion == "require_terminal_members" and any(
        item.continuation != "retained" for item in items
    ):
        blockers.append("pending_members")
    return InvestigationPolicyDecision(
        proposal_id=proposal.proposal_id if proposal else None,
        head_event_id=timeline[-1]["event_id"] if timeline else None,
        amendment_eligible=bool(
            proposal
            and timeline
            and not failed
            and _amendment_frontier(timeline, proposal.proposal_id, by_id)
        ),
        policy_id=current_policy.policy_id if current_policy else None,
        effective=policy,
        held_work_item_ids=tuple(sorted(held)),
        eligible_work_item_ids=tuple(
            item.work_item_id for item in items if item.continuation == "eligible"
        ),
        items=tuple(items),
        finalization_eligible=not blockers,
        completion_blockers=tuple(blockers),
    )
