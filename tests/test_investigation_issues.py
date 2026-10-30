"""Issue SNPs through the real Agent and application Review seams (#359).

A request member whose check ends with an issue (not found, other assembly only,
conflict) is keyed by the failure module and asks its
own Review; the researcher approves the finding, amends it or rejects it.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.application.queries import read_investigation
from bio_neb.investigation.investigation import (
    AmendInvestigationFindingResponse,
    ApproveInvestigationFindingResponse,
    ClarifyInvestigationResponse,
    InvestigationRequestItem,
    UserAssertion,
)
from bio_neb.investigation.investigation_commands import (
    ContinueInvestigationCommand,
    InvestigationApplicationRefused,
    RespondInvestigationReviewCommand,
    continue_investigation,
    respond_investigation_review,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.recording import read_evidence, review_head
from tests.explainer_fakes import explained_by_default
from tests.observation_input_fakes import RUN_VCF, TABLE_READING_ENVIRON, RunOutputStore
from tests.run_prerequisites import RunPrerequisites
from tests.sequential_investigation_support import (
    ALSO_NOT_FOUND,
    NOT_FOUND,
    AmendedLookupModel,
    IssueSourceResponses,
    SequentialAdaptiveModel,
    _current_member_messages,
    run_attempt,
)
from tests.terminal_journey_fakes import forked_child

# dbSNP's 404 for the unknown rsID, its own answer with no verdict: the source's own
# words, observation only, no instruction.
NOT_FOUND_CONFLICT = (
    f"NCBI RefSNP answered HTTP 404 for {NOT_FOUND}: "
    '{"error":{"code":404,"message":"RefSNP not found"}}.'
)
ISSUE_REQUEST = {
    "schema_version": "bio-neb.investigation-request/2",
    "asserted_by": "researcher:owner",
    "requests": [
        {"request": f"Does HG001 carry {rsid}?", "target": {"rsid": rsid}}
        for rsid in (NOT_FOUND, "rs429358")
    ],
}
ONLY_ISSUE_REQUEST = {
    **ISSUE_REQUEST,
    "requests": [
        {"request": f"Does HG001 carry {NOT_FOUND}?", "target": {"rsid": NOT_FOUND}}
    ],
}
TWO_ISSUE_REQUEST = {
    **ISSUE_REQUEST,
    "requests": [
        {"request": f"Does HG001 carry {rsid}?", "target": {"rsid": rsid}}
        for rsid in (NOT_FOUND, ALSO_NOT_FOUND, "rs429358")
    ],
}


def _assertion(*rsids: str) -> UserAssertion:
    return UserAssertion.create_collection(
        asserted_by="researcher",
        predecessor_ids=("offline-research-request",),
        requests=tuple(
            InvestigationRequestItem(
                request_text=f"Does HG001 carry {rsid}?", rsid=rsid
            )
            for rsid in rsids
        ),
    )


def _answer(journey, review, response):
    return respond_investigation_review(
        RespondInvestigationReviewCommand(
            journey_file=str(journey.path),
            request_event_id=review.request_event_id,
            response=response,
        )
    )


def _approve(review):
    return ApproveInvestigationFindingResponse(
        action="approve",
        collection_member_id=review.question.collection_member_id,
        revision_number=review.question.revision_number,
        asserted_by="researcher",
    )


def _amend(review, instruction):
    return AmendInvestigationFindingResponse(
        action="amend",
        collection_member_id=review.question.collection_member_id,
        revision_number=review.question.revision_number,
        instruction=instruction,
        asserted_by="researcher",
    )


class CitedAskModel(SequentialAdaptiveModel):
    """Asks the target question citing the lookup whose source answered elsewhere
    (#479 A), as a model reading that answer would; otherwise the sequential model.
    ``cite`` replaces the cite it sends (an unknown one is refused first)."""

    def __init__(self, cite: str | None = None) -> None:
        super().__init__()
        self.cite = cite

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        _request, refsnp, _assembly, basis, *_ = _current_member_messages(messages)
        if refsnp is None or refsnp["status"] != "source_answered":
            return await super().respond(messages, info)
        self.calls += 1
        cite, self.cite = self.cite or basis["refsnp_event_id"], None
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "request_investigation_information",
                    canonical_json(
                        {
                            "question": "dbSNP has no record of this rsID. Which "
                            "variant do you mean?",
                            "missing_information": "The source answered that it "
                            "keeps no record for the stated rsID.",
                            "asks_about": "target",
                            "refsnp_event_id": cite,
                        }
                    ),
                    tool_call_id=f"cited-ask-{self.calls}",
                )
            ]
        )


def _continue(journey, model, sources):
    return continue_investigation(
        ContinueInvestigationCommand(journey_file=str(journey.path)),
        environ={"LOCAL_MODEL_KEY": "offline"},
        model_factory=model.model,
        source_adapter_factory=lambda: sources,
    )


class InvestigationIssueTests(unittest.TestCase):
    def test_an_issue_snp_asks_its_own_review_and_its_sibling_is_still_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = IssueSourceResponses()
            model = SequentialAdaptiveModel()
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                assertion=_assertion(NOT_FOUND, "rs429358"),
            )
            review = pending.pending_request
            self.assertEqual(review.legal_actions, ("approve", "amend", "reject"))
            issue = review.question.issue
            self.assertEqual(
                (issue.key, issue.source, issue.verdict, issue.conflicts),
                (
                    "refsnp: source_answered",
                    "refsnp",
                    "source_answered",
                    (NOT_FOUND_CONFLICT,),
                ),
            )
            # The item Review shows the researcher dbSNP's own statement.
            self.assertEqual(review.question.missing_information, NOT_FOUND_CONFLICT)
            self.assertNotIn("source_answers", review.question.missing_information)
            events = read_evidence(Journey.open(journey.path))
            # The issue member's own Attempt; its sibling has its own.
            (asking,) = [
                a.attempt_id
                for a in pending.attempts
                if a.collection_member_id == review.question.collection_member_id
            ]
            (refsnp,) = [
                e
                for e in events
                if e["event_type"] == "RefSnpToolResultRecorded"
                and e["payload"]["attempt_id"] == asking
            ]
            (finding,) = [
                e
                for e in events
                if e["event_type"] == "AgentOutputValidated"
                and e["payload"]["status"] == "valid"
                and e["payload"]["attempt_id"] == asking
            ]
            self.assertEqual(issue.evidence_event_ids, (refsnp["event_id"],))
            self.assertEqual(issue.finding_event_id, finding["event_id"])
            self.assertEqual(review.question.request.rsid, NOT_FOUND)
            # The one item-question shape: it cites the lookup and copies its
            # statement; the finding comes first, then removal (C7).
            self.assertEqual(review.question.cited_event_id, refsnp["event_id"])
            self.assertEqual(
                review.question.statement,
                refsnp["payload"]["result"]["assessment"]["rationale"],
            )
            self.assertEqual(
                review.question.question,
                f"The check of {NOT_FOUND} ended with an issue (refsnp: "
                f"source_answered): {NOT_FOUND_CONFLICT} Approve removing this item "
                "from the Run (reject): the other items go on. Or keep it as checked "
                "but not verified (approve), or amend the request, for example with "
                "a corrected rsID, to check it again (amend).",
            )
            self.assertNotIn("end the Run", review.question.question)
            # The issue ends only its member's Attempt; the sibling is its own
            # Attempt in the same batch and is checked while the issue waits.
            self.assertEqual(pending.status, "review_pending")
            self.assertEqual(pending.eligible_member_ids, ())
            continued = pending
            self.assertEqual(continued.status, "review_pending")
            self.assertEqual(continued.pending_request, review)
            # No proposal, so no lookup, until the issue is answered.
            self.assertEqual(continued.proposals, ())
            self.assertEqual(
                sources.operations,
                [
                    f"fetch_refsnp:{NOT_FOUND}",
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )

    def test_an_approved_issue_is_checked_not_verified_and_the_proposal_opens(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = IssueSourceResponses()
            model = SequentialAdaptiveModel()
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                assertion=_assertion(NOT_FOUND, "rs429358"),
            )
            review = pending.pending_request
            approved = _answer(journey, review, _approve(review))
            (proposal,) = approved.proposals
            issue_member, ready_member = proposal.request_members
            self.assertEqual(issue_member.request.rsid, NOT_FOUND)
            self.assertEqual(issue_member.refsnp.status, "source_answered")
            self.assertEqual(issue_member.work_item_ids, ())
            self.assertEqual(
                (
                    issue_member.issue_approval.request_event_id,
                    issue_member.issue_approval.asserted_by,
                ),
                (review.request_event_id, "researcher"),
            )
            # Checked, not verified: no Work item and no verified identity.
            self.assertIsNone(issue_member.refsnp.verified_rsid)
            self.assertEqual(proposal.unresolved_request_member_ids, ())
            self.assertEqual(
                [item.target.aliases for item in proposal.work_items],
                [("rs429358",)],
            )
            self.assertEqual(
                ready_member.work_item_ids, (proposal.work_items[0].work_item_id,)
            )
            self.assertEqual(approved.pending_request.proposal, proposal)
            self.assertEqual(
                approved.pending_request.legal_actions, ("approve", "revise", "stop")
            )
            policy = read_investigation(Journey.open(journey.path)).work_policy
            self.assertNotIn("unresolved_request_identity", policy.completion_blockers)
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, approved
            )

    def test_an_amended_target_is_asserted_by_the_researcher_who_answered(self):
        # The paper asserted NOT_FOUND; the researcher's amend names rs7412,
        # which the paper never named. The request stays the paper's.
        from bio_neb.investigation.collection_view import collection_view

        paper = UserAssertion.create_collection(
            asserted_by="paper:doi:10.1000/x",
            predecessor_ids=("offline-research-request",),
            requests=tuple(
                InvestigationRequestItem(
                    request_text=f"Does HG001 carry {rsid}?", rsid=rsid
                )
                for rsid in (NOT_FOUND, "rs429358")
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            sources = IssueSourceResponses()
            model = AmendedLookupModel()
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                assertion=paper,
            )
            review = pending.pending_request
            _answer(journey, review, _amend(review, "I meant rs7412."))
            rechecked = _continue(journey, model, sources)

        (proposal,) = rechecked.proposals
        corrected, kept = proposal.request_members
        self.assertEqual(
            (
                corrected.request.rsid,
                corrected.refsnp.verified_rsid,
                corrected.target_asserted_by,
                kept.target_asserted_by,
            ),
            (NOT_FOUND, "rs7412", "researcher", None),
        )
        self.assertEqual(
            [snp.asserted_by for snp in collection_view(proposal).snps],
            ["researcher", None],
        )

    def test_an_answer_during_a_revision_is_asserted_by_its_own_researcher(self):
        # Researcher:a revises the approved issue to ALSO_NOT_FOUND;
        # researcher:b's amend names rs7412. The revised member is b's, the
        # carried one keeps none.
        from bio_neb.investigation.investigation import ReviseInvestigationResponse

        paper = UserAssertion.create_collection(
            asserted_by="paper:doi:10.1000/x",
            predecessor_ids=("offline-research-request",),
            requests=tuple(
                InvestigationRequestItem(
                    request_text=f"Does HG001 carry {rsid}?", rsid=rsid
                )
                for rsid in (NOT_FOUND, "rs429358")
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            sources = IssueSourceResponses()
            model = AmendedLookupModel()
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                assertion=paper,
            )
            issue = pending.pending_request
            review = _answer(
                journey,
                issue,
                _approve(issue).model_copy(update={"asserted_by": "researcher:a"}),
            ).pending_request
            [unresolved] = [
                member
                for member in review.proposal.request_members
                if not member.work_item_ids
            ]
            _answer(
                journey,
                review,
                ReviseInvestigationResponse(
                    action="revise",
                    proposal_id=review.proposal.proposal_id,
                    asserted_by="researcher:a",
                    operations=(
                        {
                            "operation": "amend_request",
                            "request_member_id": unresolved.request_member_id,
                            "request": {
                                "request_text": f"Check {ALSO_NOT_FOUND}",
                                "rsid": ALSO_NOT_FOUND,
                            },
                        },
                    ),
                ),
            )
            again = _continue(journey, model, sources).pending_request
            amend = _amend(again, "I meant rs7412.")
            _answer(
                journey, again, amend.model_copy(update={"asserted_by": "researcher:b"})
            )
            revised = _continue(journey, model, sources).proposals[-1]

        [corrected] = [
            member
            for member in revised.request_members
            if member.previous_request_member_id is not None
        ]
        [carried] = [
            member
            for member in revised.request_members
            if member.previous_request_member_id is None
        ]
        self.assertEqual(
            (
                revised.revision_number,
                corrected.request.rsid,
                corrected.refsnp.verified_rsid,
                corrected.target_asserted_by,
                carried.target_asserted_by,
            ),
            (2, ALSO_NOT_FOUND, "rs7412", "researcher:b", None),
        )

    def test_an_amended_issue_sends_only_its_member_back_before_any_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = IssueSourceResponses()
            model = AmendedLookupModel()
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                assertion=_assertion(NOT_FOUND, "rs429358"),
            )
            review = pending.pending_request
            amended = _answer(journey, review, _amend(review, "I meant rs7412."))
            # Only the amended member is checked again; nothing is proposed yet.
            self.assertEqual(amended.status, "investigation_work_ready")
            self.assertEqual(
                amended.eligible_member_ids, (review.question.collection_member_id,)
            )
            self.assertEqual(amended.proposals, ())
            rechecked = _continue(journey, model, sources)
            self.assertEqual(
                sources.operations,
                [
                    f"fetch_refsnp:{NOT_FOUND}",
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                    "fetch_refsnp:rs7412",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            (proposal,) = rechecked.proposals
            corrected, ready = proposal.request_members
            self.assertEqual(
                (corrected.refsnp.requested_rsid, corrected.refsnp.verified_rsid),
                (NOT_FOUND, "rs7412"),
            )
            self.assertIsNone(corrected.issue_approval)
            self.assertEqual(
                [item.target.aliases for item in proposal.work_items],
                [("rs7412",), ("rs429358",)],
            )
            # The researcher verifies the corrected set before any lookup.
            self.assertEqual(rechecked.pending_request.proposal, proposal)
            self.assertEqual(rechecked.authorities, ())
            # The feed tells the amend by its own consequence (Standards/Spec r2).
            from bio_neb.application.run_reader import read_journey_feed

            self.assertEqual(
                [
                    (item.status, item.text)
                    for item in read_journey_feed(
                        str(journey.path), everything=True
                    ).items
                    if item.event_type == "InvestigationReviewResponded"
                ],
                [("answer", "You amended a variant's request; it is checked again")],
            )
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, rechecked
            )

    def test_mixed_answers_settle_each_issue_on_its_own(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = IssueSourceResponses()
            model = AmendedLookupModel()
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                assertion=_assertion(NOT_FOUND, ALSO_NOT_FOUND, "rs429358"),
            )
            # Both issue members ask in the first batch, each in its own Attempt.
            first = pending.pending_request
            (second,) = [
                r
                for r in pending.review_requests
                if r.pending and r.request_event_id != first.request_event_id
            ]
            # Approve one, amend the other: only the amended member goes back.
            after_approval = _answer(journey, first, _approve(first))
            self.assertEqual(after_approval.proposals, ())
            self.assertTrue(
                next(
                    r
                    for r in after_approval.review_requests
                    if r.request_event_id == second.request_event_id
                ).pending
            )
            amended = _answer(journey, second, _amend(second, "I meant rs7412."))
            self.assertEqual(
                amended.eligible_member_ids, (second.question.collection_member_id,)
            )
            lookups_before = list(sources.operations)
            rechecked = _continue(journey, model, sources)
            self.assertEqual(
                sources.operations[len(lookups_before) :],
                ["fetch_refsnp:rs7412", "fetch_assembly_report:GCF_000001405.40"],
            )
            (proposal,) = rechecked.proposals
            approved_issue, corrected, ready = proposal.request_members
            self.assertEqual(
                approved_issue.issue_approval.request_event_id, first.request_event_id
            )
            self.assertEqual(approved_issue.work_item_ids, ())
            self.assertIsNone(corrected.issue_approval)
            self.assertEqual(corrected.refsnp.verified_rsid, "rs7412")
            self.assertEqual(
                [item.target.aliases for item in proposal.work_items],
                [("rs7412",), ("rs429358",)],
            )
            self.assertEqual(proposal.unresolved_request_member_ids, ())
            self.assertEqual(rechecked.pending_request.proposal, proposal)

    def test_an_amend_that_still_ends_with_an_issue_asks_again(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = IssueSourceResponses()
            model = AmendedLookupModel()
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                assertion=_assertion(NOT_FOUND),
            )
            first = pending.pending_request
            _answer(journey, first, _amend(first, f"Try {ALSO_NOT_FOUND}."))
            again = _continue(journey, model, sources)
            self.assertEqual(again.proposals, ())
            second = again.pending_request
            # The amend replaced the target: the continued member is shown its last
            # lookup for the old one, marked replaced.
            events = read_evidence(Journey.open(journey.path))
            (amend,) = [
                e for e in events if e["event_type"] == "InvestigationReviewResponded"
            ]
            old_lookup = next(
                e
                for e in events
                if e["event_type"] == "RefSnpToolResultRecorded"
                and e["payload"]["query"] == NOT_FOUND
            )
            context = [
                e
                for e in events
                if e["event_type"] == "InvestigationMemberContextRecorded"
            ][-1]
            self.assertEqual(
                (
                    context["payload"]["replaced_result_event_id"],
                    context["payload"]["replaced_by_event_id"],
                ),
                (old_lookup["event_id"], amend["event_id"]),
            )
            self.assertIn("replaced_target", json.dumps(model.visible_messages[-1]))
            self.assertNotEqual(second.request_event_id, first.request_event_id)
            self.assertEqual(
                second.question.collection_member_id,
                first.question.collection_member_id,
            )
            self.assertNotEqual(
                second.question.issue.finding_event_id,
                first.question.issue.finding_event_id,
            )
            self.assertEqual(sources.operations[-1], f"fetch_refsnp:{ALSO_NOT_FOUND}")
            # Approving the second finding settles the only member: accounting
            # alone, so no proposal Review opens; Finalization owns the rest.
            settled = _answer(journey, second, _approve(second))
            (proposal,) = settled.proposals
            (member,) = proposal.request_members
            self.assertEqual(
                member.issue_approval.request_event_id, second.request_event_id
            )
            self.assertEqual(proposal.work_items, ())
            self.assertIsNone(settled.pending_request)

    def test_issue_answers_refuse_the_wrong_member_shape_or_a_changed_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = IssueSourceResponses()
            model = SequentialAdaptiveModel()
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                assertion=_assertion(NOT_FOUND),
            )
            review = pending.pending_request
            before = journey.path.read_bytes()
            question = review.question
            refusals = {
                "investigation_review_stale": _approve(review).model_copy(
                    update={"collection_member_id": "another-member"}
                ),
                "investigation_review_illegal": ClarifyInvestigationResponse(
                    action="clarify",
                    collection_member_id=question.collection_member_id,
                    revision_number=1,
                    instruction="Investigate T to C",
                    asserted_by="researcher",
                ),
            }
            for code, response in refusals.items():
                with (
                    self.subTest(code=code),
                    self.assertRaises(InvestigationApplicationRefused) as refused,
                ):
                    _answer(journey, review, response)
                self.assertEqual(refused.exception.code, code)
                self.assertEqual(journey.path.read_bytes(), before)
            approved = _answer(journey, review, _approve(review))
            after = journey.path.read_bytes()
            # The identical answer is inert; a different one conflicts.
            self.assertEqual(_answer(journey, review, _approve(review)), approved)
            self.assertEqual(journey.path.read_bytes(), after)
            with self.assertRaises(InvestigationApplicationRefused) as refused:
                _answer(journey, review, _amend(review, "I meant rs7412."))
            self.assertEqual(refused.exception.code, "investigation_review_conflict")
            self.assertEqual(journey.path.read_bytes(), after)

    def test_the_model_asks_the_target_question_citing_the_lookup_it_rests_on(self):
        # The model's ask cites the lookup whose source answered elsewhere;
        # code copies that result's statement; the question has no finding,
        # so it is clarified or rejected.
        import copy

        from bio_neb.investigation.investigation import project_investigation

        with tempfile.TemporaryDirectory() as directory:
            sources = IssueSourceResponses()
            model = CitedAskModel(cite="not-a-lookup")
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                assertion=_assertion(NOT_FOUND, "rs429358"),
            )
            events = read_evidence(Journey.open(journey.path))
            review = pending.pending_request
            question = review.question
            (refsnp,) = [
                e
                for e in events
                if e["event_type"] == "RefSnpToolResultRecorded"
                and e["payload"]["query"] == NOT_FOUND
            ]
            # An unknown cite is refused by its own reason first.
            (refused,) = [
                e
                for e in events
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["status"] == "invalid"
            ]
            self.assertEqual(refused["payload"]["errors"], ["question_cite_unknown"])
            self.assertEqual(review.legal_actions, ("clarify", "reject"))
            self.assertEqual(question.asks_about, "target")
            self.assertIsNone(question.issue)
            self.assertEqual(question.cited_event_id, refsnp["event_id"])
            self.assertEqual(
                question.statement,
                refsnp["payload"]["result"]["assessment"]["rationale"],
            )
            # Its sibling is still checked while it waits.
            self.assertIn("fetch_refsnp:rs429358", sources.operations)
            self.assertEqual(project_investigation(events), pending)
            (index,) = [
                i
                for i, e in enumerate(events)
                if e["event_type"] == "InvestigationReviewRequested"
            ]
            hostile = copy.deepcopy(events)
            hostile[index]["payload"]["question"]["statement"] = "Another statement."
            with self.assertRaises(ValueError) as error:
                project_investigation(hostile)
            self.assertIn("cited statement", str(error.exception))

    def test_reopen_refuses_an_issue_whose_verdict_or_finding_changed(self):
        import copy

        from bio_neb.investigation.investigation import project_investigation

        with tempfile.TemporaryDirectory() as directory:
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=SequentialAdaptiveModel().model),
                IssueSourceResponses(),
                assertion=_assertion(NOT_FOUND),
            )
            events = read_evidence(Journey.open(journey.path))
            self.assertEqual(project_investigation(events), pending)
            (index,) = [
                i
                for i, e in enumerate(events)
                if e["event_type"] == "InvestigationReviewRequested"
            ]
            for field, value in (
                ("verdict", "incompatible"),
                ("conflicts", ["NCBI RefSNP had no eligible placement."]),
                ("finding_event_id", events[index - 1]["event_id"]),
            ):
                hostile = copy.deepcopy(events)
                hostile[index]["payload"]["question"]["issue"][field] = value
                if field == "verdict":
                    hostile[index]["payload"]["question"]["issue"]["key"] = (
                        "refsnp: incompatible"
                    )
                with self.subTest(field=field), self.assertRaises(ValueError) as error:
                    project_investigation(hostile)
                self.assertIn("Investigation issue changed", str(error.exception))

    def test_reopen_refuses_a_proposal_recorded_beside_an_unanswered_issue(self):
        # Standards r1 S1: the checked view, not only the producer, holds the
        # proposal until every issue is answered (ADR 0032).
        from unittest.mock import patch

        from bio_neb.investigation import investigation_questions
        from bio_neb.investigation.investigation import project_investigation

        class EveryMember(dict):
            """Claims every member's issue was approved; names no approval."""

            def __contains__(self, member_id):
                return True

        with tempfile.TemporaryDirectory() as directory:
            sources = IssueSourceResponses()
            model = SequentialAdaptiveModel()
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                assertion=_assertion(NOT_FOUND, "rs429358"),
            )
            with (
                patch.object(
                    investigation_questions,
                    "issue_approvals",
                    lambda *_args, **_kwargs: EveryMember(),
                ),
                self.assertRaises(ValueError),
            ):
                # The producer's own guard is bypassed: it records the proposal,
                # and the checked read refuses it.
                investigation_questions.close_collection(
                    journey, project_investigation(read_evidence(journey))
                )
            events = read_evidence(Journey.open(journey.path))
            (recorded,) = [
                e for e in events if e["event_type"] == "InvestigationProposalRecorded"
            ]
            self.assertEqual(
                len(
                    recorded["payload"]["proposal"]["request_members"][0][
                        "work_item_ids"
                    ]
                ),
                0,
            )
            self.assertTrue(
                next(
                    r
                    for r in pending.review_requests
                    if r.request_event_id == pending.pending_request.request_event_id
                ).pending
            )
            with self.assertRaises(ValueError) as refused:
                project_investigation(events)
            self.assertEqual(
                str(refused.exception),
                "Investigation proposal was recorded before its issues were answered",
            )


class InvestigationIssueRunTests(RunPrerequisites, unittest.TestCase):
    """The coordinator's path for an issue SNP, from Investigation to Finalization."""

    def child(self, root: Path, request: dict = ISSUE_REQUEST) -> Journey:
        parent = self.prerequisite(
            "executed",
            investigation_request=request,
            output_payloads={"VCF": RUN_VCF},
        ).journey_file
        return forked_child(root, parent_file=parent)

    def answer(self, journey: Journey, request_event_id: str, response: dict):
        """Answer through the native CLI, as the researcher does."""
        import contextlib
        import io

        from bio_neb.application.advancement import ObserveRun, observe
        from bio_neb.cli import main

        response_file = journey.path.parent / f"answer-{request_event_id}.json"
        response_file.write_text(json.dumps(response), encoding="utf-8")
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as error,
        ):
            code = main(
                [
                    "answer",
                    "--journey-file",
                    str(journey.path),
                    "--request-event-id",
                    request_event_id,
                    "--response-file",
                    str(response_file),
                ]
            )
        self.assertEqual(code, 0, error.getvalue())
        return observe(ObserveRun(journey_file=str(journey.path)))

    def test_approving_the_issue_lets_the_lookups_and_table_reading_reach_finalization(
        self,
    ):
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.queries import read_completion
        from bio_neb.application.reviews import pending_reviews

        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(Path(directory).resolve())
            model = SequentialAdaptiveModel()
            sources = IssueSourceResponses()
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                investigation_model_factory=model.model,
                investigation_source_adapter_factory=lambda: sources,
                investigation_vcf_source_factory=RunOutputStore(),
            ):
                first = advance(ResumeRun(journey_file=str(journey.path)))
                # Both SNPs were checked; only the issue waits, before any lookup.
                (issue,) = pending_reviews(first)
                self.assertEqual(issue.request.question.request.rsid, NOT_FOUND)
                self.assertEqual(
                    [action.action for action in issue.legal_actions],
                    ["approve", "amend", "reject"],
                )
                self.assertEqual(first.investigation.proposals, ())
                proposed = self.answer(
                    journey,
                    issue.request_event_id,
                    {
                        "action": "approve",
                        "collection_member_id": issue.request.question.collection_member_id,
                        "revision_number": 1,
                        "asserted_by": "researcher:owner",
                    },
                )
                (review,) = pending_reviews(proposed)
                proposal = review.request.proposal
                self.assertEqual(proposal.unresolved_request_member_ids, ())
                finalizing = self.answer(
                    journey,
                    review.request_event_id,
                    {"action": "approve", "proposal_id": proposal.proposal_id},
                )
            self.assertEqual(finalizing.status, "finalization_review_pending")
            basis = read_completion(Journey.open(journey.path)).finalization.bases[-1]
            # The lookup, the window and the table reading ran for the ready SNP
            # (ADR 0047).
            (evidence,) = basis.item_evidence
            self.assertEqual(evidence.window.status, "recorded")
            self.assertEqual(evidence.reading.status, "recorded")
            self.assertEqual(
                [item.target.aliases for item in basis.work_items], [("rs429358",)]
            )
            # The issue SNP is listed as such: verdict, evidence, decision.
            (listed,) = basis.issue_items
            self.assertEqual(listed.request.rsid, NOT_FOUND)
            self.assertEqual(
                (listed.issue.key, listed.issue.verdict, listed.issue.conflicts),
                ("refsnp: source_answered", "source_answered", (NOT_FOUND_CONFLICT,)),
            )
            self.assertEqual(listed.issue, issue.request.question.issue)
            self.assertEqual(
                (listed.decision, listed.decided_by, listed.request_event_id),
                ("approved", "researcher:owner", issue.request_event_id),
            )
            from bio_neb.application.run_reader import read_journey_feed

            feed = read_journey_feed(str(journey.path), everything=True).items
            self.assertEqual(
                [
                    (item.status, item.text)
                    for item in feed
                    if item.event_type == "InvestigationReviewResponded"
                ],
                [
                    (
                        "answer",
                        "You approved a variant's finding: checked, not verified",
                    ),
                    ("answer", "You approved the investigation plan"),
                ],
            )
            report = self.publish(journey, basis)
            self.assertEqual(report.issue_items, basis.issue_items)
            rendered = self.reopen(journey, report.report_digest)
            listed_lines = rendered[rendered.index("Issue items") :].splitlines()[:9]
            self.assertEqual(
                listed_lines,
                [
                    "Issue items (checked, not verified)",
                    f"- request: Does HG001 carry {NOT_FOUND}?",
                    f"  rsID: {NOT_FOUND}",
                    "  issue: refsnp: source_answered",
                    "  verdict: source_answered (refsnp)",
                    f"  conflict: {NOT_FOUND_CONFLICT}",
                    f"  evidence: {listed.issue.evidence_event_ids[0]}",
                    f"  finding: {listed.issue.finding_event_id}",
                    "  researcher decision: approved by researcher:owner",
                ],
            )

    def test_rejecting_one_issue_removes_only_that_item_and_other_work_continues(
        self,
    ):
        # Rejecting an issue resolves that item as rejected; its sibling issue is
        # answered on its own and the ready SNP goes on to its lookup.
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.queries import read_completion
        from bio_neb.application.reviews import pending_reviews

        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(Path(directory).resolve(), TWO_ISSUE_REQUEST)
            vcf = RunOutputStore()
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                investigation_model_factory=SequentialAdaptiveModel().model,
                investigation_source_adapter_factory=IssueSourceResponses,
                investigation_vcf_source_factory=vcf,
            ):
                reviews = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                # Members ask concurrently; each Review is its member's.
                order = [NOT_FOUND, ALSO_NOT_FOUND]
                first, second = sorted(
                    reviews,
                    key=lambda review: order.index(
                        review.request.question.request.rsid
                    ),
                )
                self.assertEqual(len(reviews), 2)
                after_reject = self.answer(
                    journey,
                    first.request_event_id,
                    {
                        "action": "reject",
                        "collection_member_id": first.request.question.collection_member_id,
                        "revision_number": 1,
                        "reason": "Not worth pursuing in this Run.",
                        "asserted_by": "researcher:owner",
                    },
                )
                # No Run failure: the other issue still waits on its own answer.
                events = read_evidence(Journey.open(journey.path))
                self.assertEqual(
                    [e for e in events if e["event_type"] == "RunFailed"], []
                )
                (still,) = pending_reviews(after_reject)
                self.assertEqual(still.request_event_id, second.request_event_id)
                proposed = self.answer(
                    journey,
                    second.request_event_id,
                    {
                        "action": "approve",
                        "collection_member_id": second.request.question.collection_member_id,
                        "revision_number": 1,
                        "asserted_by": "researcher:owner",
                    },
                )
                (review,) = pending_reviews(proposed)
                proposal = review.request.proposal
                rejected, approved, ready = proposal.request_members
                self.assertEqual(
                    rejected.rejection.request_event_id, first.request_event_id
                )
                self.assertEqual(rejected.work_item_ids, ())
                self.assertEqual(
                    approved.issue_approval.request_event_id,
                    second.request_event_id,
                )
                self.assertEqual(
                    [item.target.aliases for item in proposal.work_items],
                    [("rs429358",)],
                )
                finalizing = self.answer(
                    journey,
                    review.request_event_id,
                    {"action": "approve", "proposal_id": proposal.proposal_id},
                )
            self.assertEqual(finalizing.status, "finalization_review_pending")
            basis = read_completion(Journey.open(journey.path)).finalization.bases[-1]
            # The ready SNP's lookup ran (other work continued).
            self.assertEqual(
                [item.target.aliases for item in basis.work_items], [("rs429358",)]
            )
            self.assertEqual(
                {
                    (item.request.rsid, item.decision, item.decided_by)
                    for item in basis.issue_items
                },
                {
                    (NOT_FOUND, "rejected", "researcher:owner"),
                    (ALSO_NOT_FOUND, "approved", "researcher:owner"),
                },
            )
            from bio_neb.application.run_reader import read_journey_feed

            feed = read_journey_feed(str(journey.path), everything=True).items
            self.assertEqual(
                [
                    (item.status, item.text)
                    for item in feed
                    if item.event_type == "InvestigationReviewResponded"
                ][0],
                ("answer", "You removed a variant whose check ended with an issue"),
            )

    def test_a_run_whose_every_issue_is_rejected_reaches_finalization(self):
        # Every request independently rejected: ordinary Finalization accounts for
        # it without inventing a finding (PROJECT.md).
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.queries import read_completion
        from bio_neb.application.reviews import pending_reviews

        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(Path(directory).resolve(), ONLY_ISSUE_REQUEST)
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                investigation_model_factory=SequentialAdaptiveModel().model,
                investigation_source_adapter_factory=IssueSourceResponses,
                investigation_vcf_source_factory=RunOutputStore(),
            ):
                (issue,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                settled = self.answer(
                    journey,
                    issue.request_event_id,
                    {
                        "action": "reject",
                        "collection_member_id": issue.request.question.collection_member_id,
                        "revision_number": 1,
                        "reason": "Not worth pursuing in this Run.",
                        "asserted_by": "researcher:owner",
                    },
                )
            self.assertEqual(settled.status, "finalization_review_pending")
            basis = read_completion(Journey.open(journey.path)).finalization.bases[-1]
            self.assertEqual(basis.outcome, "investigation_requests_rejected")
            self.assertEqual(basis.work_items, ())
            self.assertEqual(
                [(item.request.rsid, item.decision) for item in basis.issue_items],
                [(NOT_FOUND, "rejected")],
            )

    def test_a_run_whose_every_snp_is_an_approved_issue_reaches_finalization(self):
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.queries import read_completion
        from bio_neb.application.reviews import pending_reviews

        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(Path(directory).resolve(), ONLY_ISSUE_REQUEST)
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                investigation_model_factory=SequentialAdaptiveModel().model,
                investigation_source_adapter_factory=IssueSourceResponses,
                investigation_vcf_source_factory=RunOutputStore(),
            ):
                (issue,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                settled = self.answer(
                    journey,
                    issue.request_event_id,
                    {
                        "action": "approve",
                        "collection_member_id": issue.request.question.collection_member_id,
                        "revision_number": 1,
                        "asserted_by": "researcher:owner",
                    },
                )
            self.assertEqual(settled.status, "finalization_review_pending")
            basis = read_completion(Journey.open(journey.path)).finalization.bases[-1]
            self.assertEqual(basis.outcome, "investigation_issues_approved")
            self.assertEqual(basis.work_items, ())
            self.assertEqual(
                [(item.request.rsid, item.decision) for item in basis.issue_items],
                [(NOT_FOUND, "approved")],
            )

    def publish(self, journey: Journey, basis):
        """Approve the pending basis, then let the coordinator publish its report."""
        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.application.queries import read_completion
        from bio_neb.reporting.finalization_commands import (
            ApproveFinalizationCommand,
            approve_finalization,
        )
        from bio_neb.reporting.grounded_report import project_grounded_report

        pending = read_completion(Journey.open(journey.path)).finalization
        approve_finalization(
            ApproveFinalizationCommand(
                journey_file=str(journey.path),
                request_event_id=pending.pending_request.request_event_id,
                basis_digest=basis.basis_digest,
                head_event_id=review_head(read_evidence(Journey.open(journey.path)))[
                    "event_id"
                ],
            )
        )
        # The coordinator publishes, then explains the report by default.
        with explained_by_default():
            advance(ResumeRun(journey_file=str(journey.path)))
        return project_grounded_report(read_evidence(Journey.open(journey.path))).report

    def reopen(self, journey: Journey, report_digest: str) -> str:
        import contextlib
        import io

        from bio_neb.cli import main

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                [
                    "grounded-report-reopen",
                    "--journey-file",
                    str(journey.path),
                    "--report-digest",
                    report_digest,
                ]
            )
        self.assertEqual(code, 0)
        return output.getvalue()


if __name__ == "__main__":
    unittest.main()
