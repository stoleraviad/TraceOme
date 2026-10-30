"""One Investigation item's history: its Attempts and every fact its admission reads.

Owner, #476 alignment G2.3 A (sweep Pattern F, a3 C10): "this item's earlier work" has
one definition. The ancestry is the item's earlier Attempts of its revision, followed
through provider continuations, ``continue`` and work-review restarts
(`item_ancestry`); the repeat guard, the work builder and the reopen checks read it.

The one rule for what a target change keeps (owner #479 Q2 A, the record kept): an
item's target changes at an ``amend`` answer, at a ``clarify`` that types a target (its
own, or by the model's reading of its words), and at a new revision. Reads that do not depend on the target stay for the
item's whole history, earlier revisions included: lookups as evidence, assembly
reports, assembly searches and the sequence names a lookup retained. Facts tied to the
target (lookup decisions, gene-change placements and picks) apply only when recorded
after the latest target change. Target readings stay keyed to the words they read. The
replaced target's last lookup stays shown to the model, marked replaced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# The item's reads that do not depend on its target.
TARGET_FREE_RESULTS = frozenset(
    {
        "RefSnpToolResultRecorded",
        "AssemblyReportToolResultRecorded",
        "AssembliesSearched",
    }
)
# The item's facts tied to its current target.
TARGET_TIED_FACTS = frozenset(
    {
        "InvestigationLookupDecided",
        "InvestigationGeneChangePlaced",
        "InvestigationGeneChangePicked",
    }
)
# A researcher answer that types the item's target again.
TARGET_ANSWERS = frozenset({"clarify", "amend"})


def item_ancestry(events: Sequence[Mapping[str, Any]], attempt_id: str):
    """The item's earlier Attempts of its revision, oldest first: provider
    continuations, ``continue`` (``continued_attempt_id``) and work-review restarts;
    distinct from resuming a model transcript."""
    from bio_neb.agentic_journey.model_availability import prior_model_attempt_ids

    bindings = {
        event["payload"]["attempt_id"]: event["payload"]
        for event in events
        if event["event_type"] == "InvestigationAttemptBound"
    }
    by_id = {event["event_id"]: event for event in events}
    restarts = {}
    for event in events:
        if event["event_type"] == "AttemptStarted" and (
            answer_id := event["payload"].get("continuation_work_review_event_id")
        ):
            answer = by_id[answer_id]
            request = by_id[answer["payload"]["request_event_id"]]
            restarts[event["payload"]["attempt_id"]] = request["payload"]["attempt_id"]
    result: list[str] = []
    visiting: set[str] = set()

    def visit(identity):
        if identity in visiting:
            raise ValueError("Investigation continuation cycles its Attempt identity")
        visiting.add(identity)
        model_prior = prior_model_attempt_ids(events, identity)
        continued = restarts.get(identity) or bindings.get(identity, {}).get(
            "continued_attempt_id"
        )
        for prior in (*model_prior, *((continued,) if continued else ())):
            if prior not in bindings:
                raise ValueError("Investigation continuation lost its prior Attempt")
            visit(prior)
            if prior not in result:
                result.append(prior)
        visiting.remove(identity)

    visit(attempt_id)
    return tuple(result)


def item_predecessor(
    events: Sequence[Mapping[str, Any]], member_id: str, revision_number: int
) -> tuple[str, int] | None:
    """The item a revision's member continues, as ``(member id, revision number)``:
    the revision's own record of it (a reworked Work item's request member, or a
    reworked request member), never an equal request hash; ``None`` for a request the
    revision adds, or a first revision."""
    from bio_neb.investigation.investigation import (
        InvestigationRequestItem,
        investigation_output_member_id,
        request_member_id,
    )

    def requests(bound):
        return [
            InvestigationRequestItem.model_validate(r) for r in bound["work_requests"]
        ]

    bindings = [
        e["payload"] for e in events if e["event_type"] == "InvestigationAttemptBound"
    ]
    record = next(
        (
            e["payload"]["revision"]
            for e in events
            if e["event_type"] == "InvestigationCollectionRevisionRecorded"
            and e["payload"]["revision"]["revision_number"] == revision_number
        ),
        None,
    )
    bound = next((b for b in bindings if b["revision_number"] == revision_number), None)
    if record is None or bound is None:
        return None
    ordinal = next(
        (
            index
            for index, request in enumerate(requests(bound))
            if investigation_output_member_id(request, index) == member_id
        ),
        None,
    )
    if ordinal is None:
        return None
    member = [item for item in record["members"] if item.get("request") is not None][
        ordinal
    ]
    proposal = next(
        e["payload"]["proposal"]
        for e in events
        if e["event_type"] == "InvestigationProposalRecorded"
        and e["payload"]["proposal"]["proposal_id"] == record["supersedes_proposal_id"]
    )
    previous = member.get("previous_request_member_id") or next(
        (
            item["request_member_id"]
            for item in proposal.get("request_members", ())
            if member.get("previous_work_item_id") in item["work_item_ids"]
        ),
        None,
    )
    if previous is None:
        return None
    # The revision that last checked that request member: the superseded proposal's
    # own, or an earlier one when a revision only kept (carried) it (review r2 SP2).
    for earlier in sorted(
        (b for b in bindings if b["revision_number"] <= proposal["revision_number"]),
        key=lambda bound: -bound["revision_number"],
    ):
        for index, request in enumerate(requests(earlier)):
            if request_member_id(earlier["assertion_id"], request, index) == previous:
                return (
                    investigation_output_member_id(request, index),
                    earlier["revision_number"],
                )
    return None


def item_lineage(events: Sequence[Mapping[str, Any]], attempt_id: str):
    """The item's Attempts whose target-free reads it keeps, oldest first, the Attempt
    itself last, and the item's member ids across its revisions: its ancestry, after
    each earlier revision's latest Attempt of the item that revision's record says it
    continues, and that Attempt's ancestry."""
    bindings = [
        event["payload"]
        for event in events
        if event["event_type"] == "InvestigationAttemptBound"
    ]
    current = next(b for b in bindings if b["attempt_id"] == attempt_id)
    member, revision = current.get("collection_member_id"), current["revision_number"]
    chain: list[tuple[str, int]] = []
    while member is not None and (
        previous := item_predecessor(events, member, revision)
    ):
        member, revision = previous
        chain.append(previous)
    result: list[str] = []
    for member, revision in reversed(chain):
        latest = [
            bound["attempt_id"]
            for bound in bindings
            if bound.get("collection_member_id") == member
            and bound["revision_number"] == revision
        ]
        if not latest:
            continue
        for identity in (*item_ancestry(events, latest[-1]), latest[-1]):
            if identity not in result:
                result.append(identity)
    for identity in (*item_ancestry(events, attempt_id), attempt_id):
        if identity not in result:
            result.append(identity)
    members = frozenset(
        {current.get("collection_member_id"), *(member for member, _ in chain)}
    )
    return tuple(result), members


def item_read_attempts(events: Sequence[Mapping[str, Any]], attempt_id: str):
    """The item's Attempts whose target-free reads it keeps (`item_lineage`)."""
    return item_lineage(events, attempt_id)[0]


def target_change(
    events: Sequence[Mapping[str, Any]], member_id: str, revision_number: int
) -> Mapping[str, Any] | None:
    """The event the item's current target starts at: its latest researcher answer
    in this revision that changes it (an amend, or a clarify that types a target, its
    own or by the model's reading of its words), else the revision's first binding of
    the item. A clarify that types no target leaves the target as it was. ``None`` before the item is bound in this
    revision."""
    typed_by_reading = {
        event["payload"]["source_event_id"]
        for event in events
        if event["event_type"] == "InvestigationTargetRead"
        and "target" in event["payload"]
    }
    first = answer = None
    for event in events:
        payload = event["payload"]
        if (
            first is None
            and event["event_type"] == "InvestigationAttemptBound"
            and payload.get("collection_member_id") == member_id
            and payload["revision_number"] == revision_number
        ):
            first = event
        elif (
            event["event_type"] == "InvestigationReviewResponded"
            and payload["response"].get("action") in TARGET_ANSWERS
            and payload["response"].get("collection_member_id") == member_id
            and payload["response"].get("revision_number") == revision_number
            and (
                # An amend corrects the request; a clarify changes the target only
                # when it types one.
                payload["response"]["action"] == "amend"
                or payload["response"].get("target")
                or event["event_id"] in typed_by_reading
            )
        ):
            answer = event
    return answer or first


@dataclass
class ItemHistory:
    """What the item's admission reads, from its one history."""

    attempt_ids: tuple[str, ...]
    # Target-free reads, in record order.
    source_results: list[dict[str, Any]] = field(default_factory=list)
    assembly_searches: list[dict[str, Any]] = field(default_factory=list)
    # Each assembly's sequence names a lookup of the item retained, by accession: the
    # retaining event's id and its bytes (a3 C10).
    sequence_reports: dict[str, tuple[str, bytes]] = field(default_factory=dict)
    # Target-tied facts recorded after the latest target change.
    decisions: list[dict[str, Any]] = field(default_factory=list)
    gene_change_placements: list[dict[str, Any]] = field(default_factory=list)
    gene_change_picks: list[dict[str, Any]] = field(default_factory=list)
    # The event the current target starts at, and the item's last lookup before it
    # when that answer or revision replaced an earlier target.
    target_event_id: str | None = None
    replaced_result: dict[str, Any] | None = None
    replaced_result_ids: frozenset[str] = frozenset()


def item_history(
    events: Sequence[Mapping[str, Any]],
    *,
    attempt_id: str,
    member_id: str,
    revision_number: int,
    read_artifact: Callable[[Mapping[str, Any]], bytes],
) -> ItemHistory:
    """The item's history as of ``events`` for its Attempt ``attempt_id``."""
    attempts, members = item_lineage(events, attempt_id)
    owned = set(attempts)
    change = target_change(events, member_id, revision_number)
    start = None if change is None else events.index(change)
    history = ItemHistory(
        attempt_ids=attempts,
        target_event_id=None if change is None else change["event_id"],
    )
    asked: dict[str, str] = {}
    for index, event in enumerate(events):
        payload = event["payload"]
        if payload.get("attempt_id") not in owned:
            continue
        kind = event["event_type"]
        if kind == "InvestigationSourceContactStarted":
            # The assembly a sequence-names GET asks is its contact's key; its last
            # retained answer is the report (an earlier one is NCBI's links).
            asked[payload["attempt_id"]] = (
                payload["request_key"]
                if payload["operation"] == "ncbi_sequence_names"
                else None
            )
            continue
        if kind == "InvestigationSourceRetrieved":
            accession = asked.get(payload["attempt_id"])
            if (
                accession is not None
                and payload["source"]["operation"] == "ncbi_sequence_names"
                and 200 <= payload["source"]["status_code"] < 300
            ):
                history.sequence_reports[accession] = (
                    event["event_id"],
                    read_artifact(event),
                )
            continue
        if payload.get("collection_member_id") not in members:
            continue
        if kind == "AssembliesSearched":
            history.assembly_searches.append(event)
        elif kind in TARGET_FREE_RESULTS:
            history.source_results.append(event)
            if (
                kind == "RefSnpToolResultRecorded"
                and start is not None
                and index < start
                and _replaces(change)
            ):
                history.replaced_result = event
                history.replaced_result_ids |= {event["event_id"]}
        elif kind in TARGET_TIED_FACTS and (start is None or index > start):
            {
                "InvestigationLookupDecided": history.decisions,
                "InvestigationGeneChangePlaced": history.gene_change_placements,
                "InvestigationGeneChangePicked": history.gene_change_picks,
            }[kind].append(event)
    return history


def _replaces(change) -> bool:
    """Whether the target change replaced an earlier target: an answer, or a later
    revision's first binding."""
    return change["event_type"] == "InvestigationReviewResponded" or (
        change["payload"]["revision_number"] > 1
    )


# An Attempt's own start and end, its member context and its stop's Review request: no
# change to what a call finds when the item's next Attempt repeats an earlier one's.
_ATTEMPT_BOOKKEEPING = frozenset(
    {
        "AttemptStarted",
        "AttemptSucceeded",
        "AttemptStopped",
        "AttemptCancelled",
        "AttemptFailed",
        "InvestigationAttemptBound",
        "InvestigationAttemptCompleted",
        "InvestigationMemberContextRecorded",
        "AgentEffortReviewRequested",
        "AgentWorkReviewRequested",
        "ModelAvailabilityReviewRequested",
    }
)
# A researcher's answer to a Review the item's work stopped at: a change, whatever it
# says (a continue may instruct the work to fetch again).
_REVIEW_ANSWERS = frozenset(
    {
        "AgentEffortReviewResponded",
        "AgentWorkReviewResponded",
        "AgentFailureReviewResponded",
        "ModelAvailabilityReviewResponded",
    }
)


def item_call_history(events, attempt_id: str):
    """The repeat guard's history for an Investigation item (#479 G2.3): its whole
    ancestry, the same as its work builder reads. An earlier Attempt's call repeats
    only while nothing of the item changed: another Attempt's start and stop change
    nothing; a researcher's answer, to the item's question or to a Review its work
    stopped at, does."""
    from bio_neb.agentic_journey.repeated_calls import CallHistory

    bound = next(
        (
            event["payload"]
            for event in events
            if event["event_type"] == "InvestigationAttemptBound"
            and event["payload"]["attempt_id"] == attempt_id
        ),
        {},
    )
    member = bound.get("collection_member_id")
    attempts = frozenset({*item_ancestry(events, attempt_id), attempt_id})
    # Each Review request's own Attempt. A keyed Run-level Review's joined items are
    # answered each by its own request, so a lead's answer is the lead's only.
    asked = {
        event["event_id"]: event["payload"].get("attempt_id")
        for event in events
        if event["event_type"].endswith("ReviewRequested")
    }

    def changes(event) -> bool:
        payload = event["payload"]
        if event["event_type"] in _REVIEW_ANSWERS:
            # An answer to a Review this item's own work stopped at; a sibling's
            # answer changes nothing of this item.
            return asked.get(payload.get("request_event_id")) in attempts
        return (
            event["event_type"] == "InvestigationReviewResponded"
            and payload["response"].get("collection_member_id") == member
        )

    return CallHistory(
        attempt_ids=attempts,
        bookkeeping=lambda event: event["event_type"] in _ATTEMPT_BOOKKEEPING,
        changes=changes,
    )
