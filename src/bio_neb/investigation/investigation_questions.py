"""Request-scoped Investigation clarification through the existing Review junction.

These decisions grant no scientific or execution authority. The original request
identity survives an answer; its attributed text is additional asserted intent.
"""

from bio_neb.application.investigation_read import current_investigation
from bio_neb.junction import evaluate_scoped_answer


def question_answer(requests, receipts, request_event_id, response):
    from bio_neb.investigation.investigation_commands import (
        InvestigationApplicationRefused,
    )

    def refusal(reason, event_id):
        return InvestigationApplicationRefused(
            f"investigation_review_{reason}", request_event_id=event_id
        )

    def validate(request, answer):
        from bio_neb.investigation.investigation import (
            AmendInvestigationFindingResponse,
            ApproveInvestigationFindingResponse,
            ClarifyInvestigationResponse,
            RejectInvestigationResponse,
        )

        question = request.question
        member_scoped = (
            ClarifyInvestigationResponse,
            RejectInvestigationResponse,
            ApproveInvestigationFindingResponse,
            AmendInvestigationFindingResponse,
        )
        if (
            question is None
            or answer.action not in request.legal_actions
            or not isinstance(answer, member_scoped)
        ):
            raise refusal("illegal", request.request_event_id)
        if (
            answer.collection_member_id != question.collection_member_id
            or answer.revision_number != question.revision_number
        ):
            raise refusal("stale", request.request_event_id)

    return evaluate_scoped_answer(
        requests,
        receipts,
        request_event_id,
        response,
        validate=validate,
        refusal=refusal,
    )


def amended_findings(reviews, receipts) -> frozenset[str]:
    """The issue findings the researcher amended: their members are checked again."""
    by_id = {review.request_event_id: review for review in reviews}
    return frozenset(
        by_id[receipt.request_event_id].question.issue.finding_event_id
        for receipt in receipts
        if receipt.response.action == "amend"
        and receipt.request_event_id in by_id
        and by_id[receipt.request_event_id].question is not None
        and by_id[receipt.request_event_id].question.issue is not None
    )


def without_amended_findings(lineages, reviews, receipts) -> tuple:
    """An amended finding is the researcher's new intent for its member, whose
    whole output lineage restarts; every other member keeps its lineage."""
    amended = amended_findings(reviews, receipts)
    return tuple(
        lineage
        for lineage in lineages
        if lineage.accepted_output is None
        or lineage.validation_attempts[-1].validation_event_id not in amended
    )


def eligible_members(
    requests, assertion_id, revision_number, lineages, reviews, receipts
):
    from bio_neb.investigation.investigation import investigation_output_member_id

    completed = {
        lineage.collection_member_id
        for lineage in without_amended_findings(lineages, reviews, receipts)
        if lineage.accepted_output is not None
    }
    answers = {receipt.request_event_id: receipt for receipt in receipts}
    held = {
        review.question.collection_member_id
        for review in reviews
        if review.question is not None
        and review.question.assertion_id == assertion_id
        and review.question.revision_number == revision_number
        and (
            review.pending
            or answers[review.request_event_id].response.action == "reject"
        )
    }
    return tuple(
        investigation_output_member_id(request, ordinal)
        for ordinal, request in enumerate(requests)
        if investigation_output_member_id(request, ordinal) not in completed | held
    )


def latest_collection_frontier(events, origin_id):
    """An operational answer keeps the current exact collection binding."""
    bindings = [
        e["payload"] for e in events if e["event_type"] == "InvestigationAttemptBound"
    ]
    origin = next(
        (bound for bound in bindings if bound["attempt_id"] == origin_id), None
    )
    if origin is None or not bindings:
        raise ValueError("operational recovery lost its Investigation binding")
    # Members are planned in their own Attempts: the frontier is the latest
    # binding of the origin's own member.
    member = origin.get("collection_member_id")
    latest = [
        bound for bound in bindings if bound.get("collection_member_id") == member
    ][-1]

    def work_binding(bound):
        return {
            key: value
            for key, value in bound.items()
            if key not in {"attempt_id", "continued_attempt_id"}
        }

    if work_binding(latest) != work_binding(origin):
        raise ValueError(
            "operational recovery belongs to superseded Investigation work"
        )
    return latest


def request_rejections(reviews, receipts, assertion_id, revision_number):
    from bio_neb.investigation.investigation import InvestigationRequestRejection

    by_id = {review.request_event_id: review for review in reviews}
    return {
        by_id[
            receipt.request_event_id
        ].question.collection_member_id: InvestigationRequestRejection(
            request_event_id=receipt.request_event_id,
            response_event_id=receipt.response_event_id,
            reason=receipt.response.reason,
            asserted_by=receipt.response.asserted_by,
            issue=by_id[receipt.request_event_id].question.issue,
        )
        for receipt in receipts
        if receipt.response.action == "reject"
        and by_id[receipt.request_event_id].question is not None
        # Rejecting an issue removes its item as rejected.
        and by_id[receipt.request_event_id].question.assertion_id == assertion_id
        and by_id[receipt.request_event_id].question.revision_number == revision_number
    }


def member_stop_rejections(timeline, assertion_id, revision_number, checked_once=None):
    """Each member whose stop the researcher ended, settled as not resolved and
    accounted like a rejected request: an Agent work stop rejected (#389 finding
    9) or dismissed (#447), or an effort, operational or model availability stop
    answered stop (member
    scope applies to every stop kind, owner B2; #446 Q3 A). The rejection names
    who ended it only where the answer does: the reject's ``asserted_by``, the
    operational answer's ``by`` (owner, 2026-10-25)."""
    from bio_neb.agentic_journey.agent_work_review import project_agent_work_review
    from bio_neb.agentic_journey.effort import project_effort_review
    from bio_neb.agentic_journey.model_availability import (
        project_model_availability_review,
    )
    from bio_neb.agentic_journey.operational_review import project_operational_review
    from bio_neb.investigation.investigation import InvestigationRequestRejection

    bound = {
        event["payload"]["attempt_id"]: event["payload"]
        for event in timeline
        if event["event_type"] == "InvestigationAttemptBound"
    }
    ended = (
        (
            project_agent_work_review(timeline, checked_once=checked_once),
            ("reject", "dismiss"),
            lambda response: response.asserted_by,
        ),
        (
            project_effort_review(timeline, checked_once=checked_once),
            ("stop",),
            lambda _response: None,
        ),
        (
            project_operational_review(timeline, checked_once=checked_once),
            ("stop",),
            lambda response: response.by,
        ),
        (
            project_model_availability_review(timeline, checked_once=checked_once),
            ("stop",),
            lambda _response: None,
        ),
    )
    rejections = {}
    for reviews, actions, asserter in ended:
        by_id = {request.request_event_id: request for request in reviews.requests}
        for receipt in reviews.receipts:
            request = by_id[receipt.request_event_id]
            binding = bound.get(request.attempt_id)
            if (
                receipt.response.action in actions
                and request.agent_role == "investigation_planner"
                and binding is not None
                and binding.get("collection_member_id") is not None
                and binding["assertion_id"] == assertion_id
                and binding["revision_number"] == revision_number
            ):
                rejections[binding["collection_member_id"]] = (
                    InvestigationRequestRejection(
                        request_event_id=receipt.request_event_id,
                        response_event_id=receipt.response_event_id,
                        reason=receipt.response.reason
                        or f"Its Review was answered {receipt.response.action}.",
                        asserted_by=asserter(receipt.response),
                    )
                )
    return rejections


def answer_target_asserters(
    reviews, receipts, requests, assertion_id, revision_number, reads=()
):
    """Each member whose target a researcher's clarify or amend answer supplies, with
    that answer's asserter (#367 r1 D4): the researcher's typed target, or the one the
    model typed from the answer's words (``reads``, #472 Q3 a) naming another rsID or
    a site. The member's request and its requested rsID stay its own."""
    from bio_neb.investigation.investigation import investigation_output_member_id
    from bio_neb.investigation.investigation_request import (
        RESEARCHER_INSTRUCTION_ACTIONS,
        effective_requested_rsid,
    )

    read_targets = {
        read["source_event_id"]: read["target"]
        for read in reads
        if read["source"] == "answer" and "target" in read
    }

    by_id = {review.request_event_id: review for review in reviews}
    own = {
        investigation_output_member_id(request, ordinal): effective_requested_rsid(
            request
        )
        for ordinal, request in enumerate(requests)
    }
    found = {}
    for receipt in receipts:
        response = receipt.response
        review = by_id.get(receipt.request_event_id)
        if (
            response.action not in RESEARCHER_INSTRUCTION_ACTIONS
            or review is None
            or review.question is None
            or review.question.assertion_id != assertion_id
            or review.question.revision_number != revision_number
            or response.collection_member_id not in own
        ):
            continue
        read = read_targets.get(receipt.response_event_id)
        if response.target is not None or (
            read is not None
            and (
                "position" in read
                or read.get("rsid") != own[response.collection_member_id]
            )
        ):
            found[response.collection_member_id] = response.asserted_by
    return found


def issue_approvals(reviews, receipts, assertion_id, revision_number):
    """Each member whose issue finding the researcher approved: checked, not verified."""
    from bio_neb.investigation.investigation import InvestigationIssueApproval

    by_id = {review.request_event_id: review for review in reviews}
    return {
        by_id[
            receipt.request_event_id
        ].question.collection_member_id: InvestigationIssueApproval(
            request_event_id=receipt.request_event_id,
            response_event_id=receipt.response_event_id,
            asserted_by=receipt.response.asserted_by,
        )
        for receipt in receipts
        if receipt.response.action == "approve"
        and receipt.request_event_id in by_id
        and by_id[receipt.request_event_id].question is not None
        and by_id[receipt.request_event_id].question.issue is not None
        and by_id[receipt.request_event_id].question.assertion_id == assertion_id
        and by_id[receipt.request_event_id].question.revision_number == revision_number
    }


def close_collection(journey, projection, response_event_id=None):
    """Record the collection's proposal once every member has settled (#105).

    Each member is planned in its own Attempt; when none is eligible or held and
    none awaits its Review, every member has its accepted output, its approved
    issue or its rejection, and the proposal closes and is presented. An answer
    that settles the last member closes it from that answer.
    """
    from bio_neb.agentic_journey.continuation_policy import InvestigationReviewPolicy
    from bio_neb.investigation.investigation import (
        ADMISSIBLE_LOOKUP_STATUSES,
        InvestigationProposalRecorded,
        InvestigationRequestItem,
        InvestigationReviewRequested,
        compile_contact_investigation_revision_proposal,
        compile_investigation_collection_proposal,
        deferred_findings,
        investigation_output_member_id,
        investigation_proposal_actions,
        select_investigation_sources,
    )
    from bio_neb.recording import read_evidence

    if (
        # Settlement, not the Run's stage, closes it: a revision's members
        # settle while the earlier grant still reads approved.
        projection.status == "stopped"
        or projection.eligible_member_ids
        # A pending issue Review is checked below with its finding.
        or any(
            r.pending and (r.question is None or r.question.issue is None)
            for r in projection.review_requests
        )
        or not projection.attempts
    ):
        return projection
    timeline = read_evidence(journey)
    bound = next(
        e["payload"]
        for e in reversed(timeline)
        if e["event_type"] == "InvestigationAttemptBound"
    )
    if any(p.revision_number == bound["revision_number"] for p in projection.proposals):
        return projection
    rejections = request_rejections(
        projection.review_requests,
        projection.review_receipts,
        bound["assertion_id"],
        bound["revision_number"],
    ) | member_stop_rejections(
        timeline, bound["assertion_id"], bound["revision_number"]
    )
    approvals = issue_approvals(
        projection.review_requests,
        projection.review_receipts,
        bound["assertion_id"],
        bound["revision_number"],
    )
    assertion = next(
        a for a in projection.assertions if a.assertion_id == bound["assertion_id"]
    )
    answered = answer_target_asserters(
        projection.review_requests,
        projection.review_receipts,
        assertion.requests
        or (
            InvestigationRequestItem(
                request_text=assertion.request_text, rsid=assertion.rsid
            ),
        ),
        bound["assertion_id"],
        bound["revision_number"],
        tuple(
            event["payload"]
            for event in timeline
            if event["event_type"] == "InvestigationTargetRead"
        ),
    )
    lineages = tuple(
        item
        for item in projection.collection_lineages
        if item.accepted_output is not None
        and item.collection_member_id not in rejections
    )
    pairs = {
        item.collection_member_id: select_investigation_sources(
            timeline, item.collection_member_id, item.accepted_output
        )
        for item in lineages
    }
    requests = assertion.requests or (
        InvestigationRequestItem(
            request_text=assertion.request_text, rsid=assertion.rsid
        ),
    )
    member_ids = {
        investigation_output_member_id(request, ordinal)
        for ordinal, request in enumerate(requests)
    }
    # A finding deferred out of the Run asks nothing.
    deferred = deferred_findings(lineages, pairs)
    unanswered_issues = {
        member_id
        for member_id, (refsnp, assembly) in pairs.items()
        if member_id not in deferred
        and not (
            refsnp.status in ADMISSIBLE_LOOKUP_STATUSES
            and assembly is not None
            and assembly.status == "compatible"
        )
        # A located region needs no assembly report.
        and refsnp.status != "region_located"
        and member_id not in approvals
    }
    # The first collection accounts for its members recorded not checked.
    not_checked = (
        {view.collection_member_id: view.outcome for view in projection.not_checked}
        if bound["revision_id"] is None
        else {}
    )
    if (
        set(pairs) | set(rejections) | set(not_checked) != member_ids
        or unanswered_issues
    ):
        return projection
    results = tuple(
        pairs.get(investigation_output_member_id(request, ordinal))
        for ordinal, request in enumerate(requests)
    )
    predecessors = tuple(
        dict.fromkeys(
            [
                result.result_id
                for pair in results
                if pair
                for result in pair
                if result is not None
            ]
            + [r.response_event_id for r in rejections.values()]
            + [a.response_event_id for a in approvals.values()]
            + [n.event_id for n in not_checked.values()]
            + [n.event_id for n in deferred.values()]
        )
    )
    if bound["revision_id"] is None:
        proposal = compile_investigation_collection_proposal(
            investigation_input=projection.investigation_input,
            assertion=assertion,
            source_results=results,
            revision_number=bound["revision_number"],
            supersedes_proposal_id=bound["supersedes_proposal_id"],
            predecessor_ids=predecessors,
            rejections=rejections,
            issue_approvals=approvals,
            not_checked={**not_checked, **deferred},
            target_asserters=answered,
        )
    else:
        proposal = compile_contact_investigation_revision_proposal(
            investigation_input=projection.investigation_input,
            assertion=assertion,
            previous_proposal=next(
                p
                for p in projection.proposals
                if p.proposal_id == bound["supersedes_proposal_id"]
            ),
            revision=next(
                r for r in projection.revisions if r.revision_id == bound["revision_id"]
            ),
            source_results=results,
            rejections=rejections,
            issue_approvals=approvals,
            deferred=deferred,
            target_asserters=answered,
        )
    position = {event["event_id"]: index for index, event in enumerate(timeline)}
    parent = (
        response_event_id
        if response_event_id is not None
        else max(
            (
                item.validation_attempts[-1].validation_link_event_id
                for item in lineages
            ),
            key=position.__getitem__,
        )
    )
    recorded = InvestigationProposalRecorded(
        attempt_id=bound["attempt_id"],
        output_lineages=lineages,
        question_response_event_id=response_event_id,
        proposal=proposal,
    )
    event = journey.record(
        recorded.event_type,
        recorded.model_dump(mode="json", exclude={"event_type"}),
        [parent],
    )
    if not proposal.work_items and not proposal.unresolved_request_member_ids:
        # Every original request has its own retained rejection or approved issue.
        # There is no scientific action left to approve; the coordinator owns
        # Finalization.
        return current_investigation(journey)
    review = InvestigationReviewRequested(
        review_policy=InvestigationReviewPolicy(),
        proposal_event_id=event["event_id"],
        proposal_id=proposal.proposal_id,
        legal_actions=investigation_proposal_actions(proposal),
    )
    journey.record(
        review.event_type,
        review.model_dump(mode="json", exclude={"event_type"}),
        [event["event_id"]],
    )
    return current_investigation(journey)
