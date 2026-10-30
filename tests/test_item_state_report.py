"""Every requested item keeps its real state through Finalization (#339 ``report``).

A Run that ends before, during or after its lookups freezes a basis whose
collection view lists every request of the question, each candidate Work item
with what happened to it, and the gap or limitation that state leaves. The
Grounded report and the native CLI text carry the same states. Listing an item
admits nothing; an observed genotype reads from its own outcome; planning's and
the Investigation's identity slots for one requested rsID read as one entry.

Each scenario runs through the actual callers: the coordinator's ``advance``,
the native ``answer`` and ``grounded-report-reopen`` commands, with offline
model, NCBI and object-store doubles.
"""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.investigation.qualification_origins import (
    RequestIssueRejected,
    RequestNotChecked,
    RequestRejected,
    RequestUnresolved,
)
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.item_state_runs import ItemStateRuns, request_file
from tests.observation_input_fakes import (
    SAMPLE,
    TABLE_READING_ENVIRON,
    RunOutputStore,
)
from tests.refsnp_retained_bodies import RS2228570, RS4149056, RS10735810
from tests.sequential_investigation_support import (
    ALSO_NOT_FOUND,
    NOT_FOUND,
    SourceResponsesOnly,
)

RS429358_TARGET = "NC_000019.10:44908684 T>C"
# rs2228570 carries G only, written as a padded CA>CG record that matches once
# trimmed; an upstream deletion spans rs429358's target.
CASE_VCF = (
    b"##fileformat=VCFv4.2\n"
    b"##contig=<ID=NC_000012.12,length=133275309>\n"
    b"##contig=<ID=NC_000019.10,length=58617616>\n"
    b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + SAMPLE.encode() + b"\n"
    b"NC_000012.12\t47879111\t.\tCA\tCG\t50\tPASS\t.\tGT\t0/1\n"
    b"NC_000019.10\t44908682\t.\tATT\tA\t50\tPASS\t.\tGT\t0/1\n"
)
# The sequential double's rs7412 record reuses rs429358's alleles (synthetic).
RS7412_TARGET = "NC_000019.10:44908822 T>C"


def _texts(qualifications):
    """Each qualification's sentence (#448 D2: the basis keeps its ids too)."""
    return tuple(item.text for item in qualifications)


def _rows(view) -> dict:
    """Each request's state and its items' (target, state, outcome, genotype)."""
    return {
        snp.requested_rsid: (
            snp.state,
            [
                (item.target, item.state, item.outcome, item.genotype)
                for item in snp.items
            ],
        )
        for snp in view.snps
    }


def _section(text: str, heading: str) -> str:
    return text[text.index(f"\n{heading}\n") :].split("\n\n")[0]


class ItemStateReportTests(ItemStateRuns, unittest.TestCase):
    def test_a_run_stopped_before_its_lookups_lists_every_item_with_its_state(self):
        # Shaped like a retained Run: two approved items never looked up, one
        # rsID with two candidates nobody approved, and one dbSNP does not know.
        from bio_neb.application.advancement import ObserveRun, ResumeRun, advance
        from bio_neb.application.advancement import observe as observe_run
        from bio_neb.application.reviews import pending_reviews
        from bio_neb.investigation.investigation import ApproveInvestigationResponse
        from bio_neb.investigation.investigation_commands import (
            RespondInvestigationReviewCommand,
            respond_investigation_review,
        )

        rsids = ("rs429358", "rs7412", "rs4149056", NOT_FOUND)
        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(Path(directory).resolve(), request_file(*rsids))
            store = RunOutputStore()
            with self.dependencies(store, {"rs4149056": RS4149056}):
                (issue,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                (review,) = pending_reviews(self.approve_issue(journey, issue))
                proposal = review.request.proposal
                approved = tuple(
                    item.work_item_id
                    for item in proposal.work_items
                    if item.target.one_based_position in {44908684, 44908822}
                )
                # The approval is recorded; the researcher stops the Run before
                # the coordinator's next Pass looks anything up.
                respond_investigation_review(
                    RespondInvestigationReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=review.request_event_id,
                        response=ApproveInvestigationResponse(
                            action="approve",
                            proposal_id=proposal.proposal_id,
                            work_item_ids=approved,
                        ),
                    )
                )
                (remaining,) = pending_reviews(
                    observe_run(ObserveRun(journey_file=str(journey.path)))
                )
                ended = self.answer(
                    journey,
                    remaining.request_event_id,
                    {
                        "action": "stop",
                        "proposal_id": proposal.proposal_id,
                        "reason": "End the Run before the lookups.",
                    },
                )
            self.assertEqual(ended.status, "finalization_review_pending")
            self.assertEqual(store.gets, [])
            basis = self.basis(journey)
            self.assertEqual(basis.outcome, "investigation_stopped")
            self.assertEqual(
                _rows(basis.collection_view),
                {
                    "rs429358": (
                        "candidates",
                        [(RS429358_TARGET, "approved_not_observed", None, None)],
                    ),
                    "rs7412": (
                        "candidates",
                        [(RS7412_TARGET, "approved_not_observed", None, None)],
                    ),
                    "rs4149056": (
                        "candidates",
                        [
                            ("NC_000012.12:21178615 T>A", "not_approved", None, None),
                            ("NC_000012.12:21178615 T>C", "not_approved", None, None),
                        ],
                    ),
                    NOT_FOUND: ("unresolved", []),
                },
            )
            self.assertEqual(
                [snp.requested_rsid for snp in basis.collection_view.snps], list(rsids)
            )
            # Every state leaves its gap; listing admits nothing.
            items = {
                item.work_item_id: snp.requested_rsid
                for snp in basis.collection_view.snps
                for item in snp.items
            }
            for work_item_id in approved:
                self.assertIn(
                    f"Work item {work_item_id} ({items[work_item_id]} "
                    + (
                        RS429358_TARGET
                        if items[work_item_id] == "rs429358"
                        else RS7412_TARGET
                    )
                    + ") was approved but not observed before Finalization.",
                    _texts(basis.gaps),
                )
            self.assertEqual(
                sum("proposed but not approved" in gap for gap in _texts(basis.gaps)), 2
            )
            self.assertIn(
                f"Request {NOT_FOUND} has no Work item: "
                "refsnp: source_answered (approved, checked not verified).",
                _texts(basis.gaps),
            )
            # Each gap carries the typed origin its state set, never words.
            self.assertEqual(
                sorted(gap.origin.code for gap in basis.gaps),
                [
                    *["item_approved_not_observed"] * 2,
                    *["item_not_approved"] * 2,
                    "request_unresolved",
                ],
            )
            self.assertIn(
                RequestUnresolved(
                    source="refsnp", verdict="source_answered", decision="approved"
                ),
                [gap.origin for gap in basis.gaps],
            )
            self.assertEqual(
                (basis.work_items, basis.sample_observations, basis.admissions),
                ((), (), ()),
            )
            report = self.publish(journey, basis)
            self.assertEqual(report.collection_view, basis.collection_view)
            self.assertEqual(report.gaps, basis.gaps)
            section = _section(
                self.reopen(journey, report.report_digest), "Requested SNPs"
            )
            for line in (
                f"  {RS429358_TARGET}: approved, not observed",
                f"  {RS7412_TARGET}: approved, not observed",
                "  NC_000012.12:21178615 T>A: proposed, not approved",
                "  NC_000012.12:21178615 T>C: proposed, not approved",
                "  no Work item: refsnp: source_answered (approved, checked not verified)",
            ):
                self.assertIn(line, section)
            # With no observed membership, each candidate keeps the target the
            # view recorded, one results row each in its state's short words, and
            # the page names no Work item.
            readable = self.readable(journey, report.report_digest)
            self.assertNotIn("work-item:", readable)
            # The proposal's approval, then the follow-up's stop (the
            # accounting is a set, a follow-up follows the answer it follows).
            self.assertIn(
                "- **Investigation:** the exact proposal approved for 2 positions.\n"
                "- **Investigation:** the exact proposal answered `stop`.\n",
                readable,
            )
            for allele in ("T>A", "T>C"):
                self.assertIn(
                    f"| you | NC_000012.12:21178615 {allele} | not approved | — |",
                    readable,
                )
            # Review r1 M1: every target with no VCF search is listed as not
            # examined, a target with Work items in its items' own state words.
            not_examined = readable.split("\n## 8. Not examined\n")[1].split("\n## ")[0]
            self.assertNotIn("Every target was examined", not_examined)
            self.assertIn(
                "| **T1** | rs429358 | Approved; the Run ended before the VCF was "
                "searched | ",
                not_examined,
            )
            self.assertIn(
                "| **T3** | rs4149056 | Not approved for a VCF search | ", not_examined
            )
            self.assertIn("| **T4** | rs9999999999 | ", not_examined)

    def test_an_observed_genotype_and_its_identity_read_from_its_own_records(self):
        # rs429358 is called 0/1, rs7412 has no record; neither reads
        # "genotype unknown".
        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.application.reviews import pending_reviews

        with tempfile.TemporaryDirectory() as directory:
            # The prompt also names RS429358, so planning holds its own rsID
            # slot for the target the request file names too.
            journey = self.child(
                Path(directory).resolve(),
                request_file("rs429358", "rs7412"),
                scientific_intent="Also check RS429358.",
                scientific_rsid="rs429358",
            )
            with self.dependencies(RunOutputStore()):
                (review,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                finalizing = self.answer(
                    journey,
                    review.request_event_id,
                    {
                        "action": "approve",
                        "proposal_id": review.request.proposal.proposal_id,
                    },
                )
            self.assertEqual(finalizing.status, "finalization_review_pending")
            basis = self.basis(journey)
            self.assertEqual(
                _rows(basis.collection_view),
                {
                    "rs429358": (
                        "candidates",
                        [
                            (
                                RS429358_TARGET,
                                "observed",
                                "exact_allele_observed",
                                "0/1",
                            )
                        ],
                    ),
                    "rs7412": (
                        "candidates",
                        [(RS7412_TARGET, "observed", "no_record", None)],
                    ),
                },
            )
            texts = (
                *_texts(basis.gaps),
                *_texts(basis.limitations),
                *_texts(basis.conflicts),
            )
            self.assertFalse(
                [text for text in texts if "unknown" in text.lower()], texts
            )
            # One entry per requested rsID: the Investigation's verified slot,
            # with planning's slot for the same rsID inside it.
            for rsid, planned in (
                # Planning names no source for an rsID.
                ("rs429358", [("rs429358", None)]),
                ("rs7412", []),
            ):
                (slot,) = [
                    slot for slot in basis.fact_slots if slot.requested_value == rsid
                ]
                self.assertEqual(
                    (slot.kind, slot.status, slot.verified_value),
                    ("requested_rsid", "verified", rsid),
                )
                self.assertEqual(
                    [
                        (earlier.requested_value, earlier.verification_source)
                        for earlier in slot.planning_slots
                    ],
                    planned,
                )
            # Every Investigation model contact is counted, the table reader's
            # included.
            events = read_evidence(Journey.open(journey.path))
            roles = {
                event["payload"]["attempt_id"]: event["payload"]["agent_role"]
                for event in events
                if event["event_type"] == "AttemptStarted"
            }
            contacts = [
                roles.get(event["payload"]["attempt_id"])
                for event in events
                if event["event_type"] == "ModelPhysicalContactStarted"
            ]
            self.assertIn("table_reader", contacts)
            self.assertEqual(
                basis.inherited_model_contact_count,
                sum(
                    role in {"investigation_planner", "table_reader"}
                    for role in contacts
                ),
            )
            report = self.publish(journey, basis)
            self.assertEqual(report.fact_slots, basis.fact_slots)
            text = self.reopen(journey, report.report_digest)
            self.assertIn(
                f"  {RS429358_TARGET}: exact_allele_observed (GT 0/1)",
                _section(text, "Requested SNPs"),
            )
            self.assertNotIn("genotype: unknown", text)
            identities = _section(text, "Requested and verified identities")
            for rsid in ("rs429358", "rs7412"):
                self.assertEqual(identities.count(f"  requested: {rsid}\n"), 1)
            self.assertEqual(identities.count("  planning: "), 1)

    def test_a_merge_and_unselected_candidates_keep_their_rows(self):
        # rs10735810 was merged into rs2228570, which lists three ALTs; only G is
        # approved, so A>C and A>T remain candidates the approval did not select.
        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.application.reviews import pending_reviews

        rsids = ("rs10735810", "rs2228570", "rs429358", NOT_FOUND)
        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(
                Path(directory).resolve(), request_file(*rsids), vcf=CASE_VCF
            )
            store = RunOutputStore(vcf=CASE_VCF)
            bodies = {"rs10735810": RS10735810, "rs2228570": RS2228570}
            with self.dependencies(store, bodies):
                (issue,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                (review,) = pending_reviews(self.approve_issue(journey, issue))
                proposal = review.request.proposal
                selected = [
                    item.work_item_id
                    for item in proposal.work_items
                    if (
                        item.target.contig,
                        item.target.one_based_position,
                        item.target.alt,
                    )
                    in {
                        ("NC_000012.12", 47879112, "G"),
                        ("NC_000019.10", 44908684, "C"),
                    }
                ]
                self.assertEqual(len(selected), 2)
                (remaining,) = pending_reviews(
                    self.answer(
                        journey,
                        review.request_event_id,
                        {
                            "action": "approve",
                            "proposal_id": proposal.proposal_id,
                            "work_item_ids": selected,
                        },
                    )
                )
                self.answer(
                    journey,
                    remaining.request_event_id,
                    {
                        "action": "stop",
                        "proposal_id": proposal.proposal_id,
                        "reason": "The other alleles are not needed.",
                    },
                )
            basis = self.basis(journey)
            alleles = [
                ("NC_000012.12:47879112 A>C", "unselected_candidate", None, None),
                (
                    "NC_000012.12:47879112 A>G",
                    "observed",
                    "exact_allele_observed",
                    "0/1",
                ),
                ("NC_000012.12:47879112 A>T", "unselected_candidate", None, None),
            ]
            rows = _rows(basis.collection_view)
            self.assertEqual(rows["rs10735810"], ("candidates", alleles))
            self.assertEqual(rows["rs2228570"], ("candidates", alleles))
            self.assertEqual(rows[NOT_FOUND], ("unresolved", []))
            (merged,) = [
                snp
                for snp in basis.collection_view.snps
                if snp.requested_rsid == "rs10735810"
            ]
            self.assertEqual(merged.verified_rsid, "rs2228570")
            self.assertIn("merged_identity", merged.cases)
            # Each request lists its candidates, so both rows leave both gaps.
            self.assertEqual(
                sum(
                    "is a candidate the approval did not select" in g
                    for g in _texts(basis.gaps)
                ),
                4,
            )
            # The merge reads as one identity entry: requested and current rsID.
            (slot,) = [
                slot
                for slot in basis.fact_slots
                if slot.requested_value == "rs10735810"
            ]
            self.assertEqual(
                (slot.verified_value, slot.mismatch, slot.summary),
                (
                    "rs2228570",
                    True,
                    "requested rs10735810, but NCBI RefSNP verified rs2228570",
                ),
            )
            report = self.publish(journey, basis)
            # The readable row says how its request was read: the merge in dbSNP's
            # own case words, else the rsID it asked about.
            readable = self.readable(journey, report.report_digest)
            self.assertIn(
                "How each target was read, where it differs from its label:\n\n"
                "- **T1** dbSNP merged rs10735810 into rs2228570, so the request "
                "reads as rs2228570\n",
                readable,
            )
            self.assertNotIn("- **T2** ", readable)
            self.assertIn("| **T2** | rs2228570 | you | NC_000012.12:", readable)
            text = self.reopen(journey, report.report_digest)
            section = _section(text, "Requested SNPs")
            self.assertIn(
                "- rs10735810 (verified rs2228570): Does HG001 carry rs10735810?",
                section,
            )
            self.assertEqual(
                section.count(
                    "  NC_000012.12:47879112 A>C: a candidate the approval did not "
                    "select"
                ),
                2,
            )
            question = _section(text, "Question")
            for ordinal, rsid in enumerate(rsids, start=1):
                self.assertIn(
                    f"  request {ordinal}: Does HG001 carry {rsid}?", question
                )

    def test_every_request_rejected_lists_each_rejection_with_its_attribution(self):
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.reviews import pending_reviews
        from tests.model_endpoint_fakes import ScriptedModelEndpoint

        async def ask(messages, info):  # noqa: ANN001, ANN202
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "request_investigation_information",
                        {
                            "question": "Which allele?",
                            "missing_information": "Allele unspecified.",
                            "asks_about": "target",
                        },
                        tool_call_id="question",
                    )
                ]
            )

        rsids = ("rs429358", "rs7412")
        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(Path(directory).resolve(), request_file(*rsids))
            sources = SourceResponsesOnly()
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                investigation_model_factory=ScriptedModelEndpoint(ask),
                investigation_source_adapter_factory=lambda: sources,
                investigation_vcf_source_factory=RunOutputStore(),
            ):
                questions = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                self.assertEqual(len(questions), 2)
                for question in questions:
                    current = self.answer(
                        journey,
                        question.request_event_id,
                        {
                            "action": "reject",
                            "collection_member_id": question.request.question.collection_member_id,
                            "revision_number": question.request.question.revision_number,
                            "asserted_by": "researcher:owner",
                            "reason": "Not this "
                            + question.request.question.request.rsid
                            + ".",
                        },
                    )
            self.assertEqual(current.status, "finalization_review_pending")
            self.assertEqual(sources.operations, [])
            basis = self.basis(journey)
            self.assertEqual(
                {
                    snp.requested_rsid: (snp.state, snp.state_origin)
                    for snp in basis.collection_view.snps
                },
                {
                    rsid: (
                        "rejected",
                        RequestRejected(
                            by="researcher:owner", reason=f"Not this {rsid}."
                        ),
                    )
                    for rsid in rsids
                },
            )
            for rsid in rsids:
                self.assertIn(
                    f"Request {rsid} was rejected by researcher:owner: Not this {rsid}.",
                    _texts(basis.limitations),
                )
            self.assertEqual((basis.work_items, basis.admissions), ((), ()))
            report = self.publish(journey, basis)
            section = _section(
                self.reopen(journey, report.report_digest), "Requested SNPs"
            )
            for rsid in rsids:
                self.assertIn(
                    f"- {rsid} (not verified): Does HG001 carry {rsid}?", section
                )
            self.assertEqual(section.count("rejected by researcher:owner"), 2)

    def test_targets_not_checked_reach_the_report_each_with_its_reason(self):
        # No rsID, and nothing placed: it is recorded not checked before any model
        # request, asks nothing, and is its own row in the basis and the report,
        # with its asserter. An item with only words is read by the model instead
        # (test_not_checked_targets).
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from tests.model_endpoint_fakes import ScriptedModelEndpoint

        async def never(messages, info):  # noqa: ANN001, ANN202
            raise AssertionError("a not-checked target asks no model")

        request = {
            "schema_version": "bio-neb.investigation-request/3",
            "asserted_by": "researcher:owner",
            "requests": [
                {
                    "request": "a deletion upstream of the promoter (placement)",
                    "target": {"gene_change": {"gene": "APOE", "change": "del"}},
                    "placement": {
                        "step": "none",
                        "reason": "not_found: the annotation names no change del",
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(Path(directory).resolve(), request)
            sources = SourceResponsesOnly()
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                investigation_model_factory=ScriptedModelEndpoint(never),
                investigation_source_adapter_factory=lambda: sources,
                investigation_vcf_source_factory=RunOutputStore(),
            ):
                current = advance(ResumeRun(journey_file=str(journey.path)))
            self.assertEqual(current.status, "finalization_review_pending")
            self.assertEqual(sources.operations, [])
            events = read_evidence(Journey.open(journey.path))
            self.assertEqual(
                [
                    e["event_type"]
                    for e in events
                    if "Attempt" in e["event_type"]
                    and "Investigation" in e["event_type"]
                ],
                [],
            )
            basis = self.basis(journey)
            self.assertEqual(
                [(snp.state, snp.state_origin) for snp in basis.collection_view.snps],
                [
                    (
                        "not_checked",
                        RequestNotChecked(
                            reason_code="not_placed",
                            reason="not_found: the annotation names no change del",
                        ),
                    )
                ],
            )
            self.assertIn(
                "Request a deletion upstream of the promoter (placement) was not "
                "checked: not_found: the annotation names no change del.",
                _texts(basis.limitations),
            )
            self.assertEqual((basis.work_items, basis.admissions), ((), ()))
            report = self.publish(journey, basis)
            section = _section(
                self.reopen(journey, report.report_digest), "Requested SNPs"
            )
            self.assertEqual(section.count("no Work item: not checked"), 0)
            self.assertEqual(section.count("state: not checked ("), 1)
            # The explainer reads its request row as its own state (it read as
            # state_not_recorded).
            from bio_neb.reporting.explanation_members import row_claims
            from bio_neb.reporting.explanation_rows import explanation_rows

            self.assertEqual(
                [c.claim for c in row_claims(explanation_rows(report.collection_view))],
                ["not_checked"],
            )

    def test_a_rejected_issue_keeps_its_row_beside_the_requests_that_went_on(self):
        # Rejecting an issue removes only its item: the other issue is answered on
        # its own, the ready request is proposed, and the researcher then stops.
        # None of them is dropped.
        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.application.reviews import pending_reviews
        from tests.sequential_investigation_support import ALSO_NOT_FOUND

        rsids = (NOT_FOUND, ALSO_NOT_FOUND, "rs429358")
        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(Path(directory).resolve(), request_file(*rsids))
            with self.dependencies(RunOutputStore()):
                reviews = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                rejected, unanswered = sorted(
                    reviews,
                    key=lambda review: rsids.index(
                        review.request.question.request.rsid
                    ),
                )
                after = self.answer(
                    journey,
                    rejected.request_event_id,
                    {
                        "action": "reject",
                        "collection_member_id": rejected.request.question.collection_member_id,
                        "revision_number": 1,
                        "reason": "Not worth pursuing in this Run.",
                        "asserted_by": "researcher:owner",
                    },
                )
                # The other issue still waits on its own answer.
                (still,) = pending_reviews(after)
                self.assertEqual(still.request_event_id, unanswered.request_event_id)
                (review,) = pending_reviews(self.approve_issue(journey, still))
                ended = self.answer(
                    journey,
                    review.request_event_id,
                    {
                        "action": "stop",
                        "proposal_id": review.request.proposal.proposal_id,
                        "reason": "Enough for now.",
                    },
                )
            self.assertEqual(ended.status, "finalization_review_pending")
            basis = self.basis(journey)
            self.assertEqual(
                [
                    (snp.requested_rsid, snp.state, snp.state_origin)
                    for snp in basis.collection_view.snps
                ],
                [
                    (
                        NOT_FOUND,
                        "rejected",
                        RequestIssueRejected(
                            by="researcher:owner",
                            source="refsnp",
                            verdict="source_answered",
                        ),
                    ),
                    (
                        ALSO_NOT_FOUND,
                        "unresolved",
                        RequestUnresolved(
                            source="refsnp",
                            verdict="source_answered",
                            decision="approved",
                        ),
                    ),
                    ("rs429358", "candidates", None),
                ],
            )
            # The rejected request keeps the source answer its check read (dbSNP's
            # 404), as the issue its researcher rejected cites it.
            lookup = basis.collection_view.snps[0].lookup
            self.assertEqual(
                (lookup.authority, lookup.status_code, lookup.verdict),
                ("ncbi", 404, "source_answered"),
            )
            report = self.publish(journey, basis)
            section = _section(
                self.reopen(journey, report.report_digest), "Requested SNPs"
            )
            # The rejected item keeps its issue, its researcher and its evidence.
            self.assertIn(
                "  no Work item: rejected by researcher:owner: refsnp: "
                "source_answered finding",
                section,
            )
            readable = self.readable(journey, report.report_digest)
            # A target not searched shows its source's own answer, copied from the
            # raw artifact the report names; its rejection is tied to it by the
            # proposal member's stored link.
            self.assertIn(
                f"| **T1** | {NOT_FOUND} | Rejected; the VCF was not searched | NCBI "
                '`fetch_refsnp` HTTP 404 (not found): `{"error":{"code":404,'
                '"message":"RefSNP not found"}}` |',
                readable,
            )
            self.assertIn(
                f"- **Target question, T1 {NOT_FOUND}:** reject (reason in L1), by "
                "researcher:owner.\n",
                readable,
            )
            # Review r1 M4 (D7 a): the answer is tied to its target by the proposal
            # member's stored link, never by its reason: given a reason no target
            # holds, the reject still names T1, and the target questions run in
            # target order.
            from bio_neb.reporting.readable_report import (
                ExplanationState,
                RunFacts,
                render_readable_report,
            )

            reasoned = report.model_copy(
                update={
                    "accounting": tuple(
                        member.model_copy(
                            update={
                                "value": member.value.model_copy(
                                    update={
                                        "payload": {
                                            **member.value.payload,
                                            "response": {
                                                **member.value.payload["response"],
                                                "reason": "A reason no target holds.",
                                            },
                                        }
                                    }
                                )
                            }
                        )
                        if member.value.event_type == "InvestigationReviewResponded"
                        and member.value.payload["response"]["action"] == "reject"
                        else member
                        for member in report.accounting
                    )
                }
            )
            for page in (
                readable,
                render_readable_report(
                    reasoned, ExplanationState(None, "none"), RunFacts()
                ),
            ):
                first = page.index(f"- **Target question, T1 {NOT_FOUND}:** reject")
                self.assertLess(
                    first, page.index(f"- **Target question, T2 {ALSO_NOT_FOUND}:** ")
                )

    def _rejected_lookup(self, journey, rsid: str):
        """The rejected row of ``rsid`` in the Finalization basis, and the request its
        own lookup asked (dbSNP's URL for that rsID)."""
        (row,) = [
            snp
            for snp in self.basis(journey).collection_view.snps
            if snp.requested_rsid == rsid
        ]
        self.assertEqual(row.state, "rejected")
        self.assertIsNotNone(row.lookup)
        return row.lookup.asked, (
            "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/"
            + rsid.removeprefix("rs")
        )

    def test_an_amended_request_rejected_after_its_lookup_keeps_that_answer(self):
        """#481 N1 (Standards r1 ST1, Spec r1 SP1; ADR 0039): a replacement's lookup is
        its own, recorded under its revision's request, never its predecessor's."""
        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.application.reviews import pending_reviews

        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(
                Path(directory).resolve(), request_file(NOT_FOUND, "rs429358")
            )
            with self.dependencies(RunOutputStore()):
                (issue,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                (review,) = pending_reviews(self.approve_issue(journey, issue))
                proposal = review.request.proposal
                (unresolved,) = [
                    member
                    for member in proposal.request_members
                    if member.request.rsid == NOT_FOUND
                ]
                (revised_issue,) = pending_reviews(
                    self.answer(
                        journey,
                        review.request_event_id,
                        {
                            "action": "revise",
                            "proposal_id": proposal.proposal_id,
                            "asserted_by": "researcher:owner",
                            "operations": [
                                {
                                    "operation": "amend_request",
                                    "request_member_id": unresolved.request_member_id,
                                    "request": {
                                        "rsid": ALSO_NOT_FOUND,
                                        "request_text": f"Check {ALSO_NOT_FOUND} "
                                        "instead.",
                                    },
                                }
                            ],
                        },
                    )
                )
                self._reject_then_stop(journey, revised_issue)
            asked, own = self._rejected_lookup(journey, ALSO_NOT_FOUND)
            self.assertEqual(asked, own)

    def test_an_added_request_rejected_after_its_lookup_keeps_that_answer(self):
        """#481 N1 (Spec r1 SP1): a request a revision added keeps the answer its own
        lookup read when the researcher rejects it."""
        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.application.reviews import pending_reviews

        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(Path(directory).resolve(), request_file("rs429358"))
            with self.dependencies(RunOutputStore()):
                (review,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                proposal = review.request.proposal
                (issue,) = pending_reviews(
                    self.answer(
                        journey,
                        review.request_event_id,
                        {
                            "action": "revise",
                            "proposal_id": proposal.proposal_id,
                            "asserted_by": "researcher:owner",
                            "operations": [
                                *(
                                    {"operation": "keep", "work_item_id": item_id}
                                    for item_id in (
                                        item.work_item_id
                                        for item in proposal.work_items
                                    )
                                ),
                                {
                                    "operation": "add",
                                    "request": {
                                        "rsid": ALSO_NOT_FOUND,
                                        "request_text": f"Also check {ALSO_NOT_FOUND}.",
                                    },
                                },
                            ],
                        },
                    )
                )
                self._reject_then_stop(journey, issue)
            asked, own = self._rejected_lookup(journey, ALSO_NOT_FOUND)
            self.assertEqual(asked, own)

    def _reject_then_stop(self, journey, issue) -> None:
        """Reject the revision's issue question, then stop at its proposal."""
        from bio_neb.application.reviews import pending_reviews

        (revised,) = pending_reviews(
            self.answer(
                journey,
                issue.request_event_id,
                {
                    "action": "reject",
                    "collection_member_id": issue.request.question.collection_member_id,
                    "revision_number": 2,
                    "reason": "Do not pursue this request.",
                    "asserted_by": "researcher:owner",
                },
            )
        )
        self.answer(
            journey,
            revised.request_event_id,
            {
                "action": "stop",
                "proposal_id": revised.request.proposal.proposal_id,
                "reason": "End after the revision.",
            },
        )

    def test_removed_and_deferred_candidates_and_an_amended_request_keep_their_rows(
        self,
    ):
        # A revision removes rs2228570's A>C, defers its A>T and all of rs429358,
        # and amends the first (unresolved) request; then the researcher stops.
        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.application.reviews import pending_reviews

        rsids = (NOT_FOUND, "rs2228570", "rs429358")
        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(Path(directory).resolve(), request_file(*rsids))
            with self.dependencies(RunOutputStore(), {"rs2228570": RS2228570}):
                (issue,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                (review,) = pending_reviews(self.approve_issue(journey, issue))
                proposal = review.request.proposal
                unresolved = next(
                    member
                    for member in proposal.request_members
                    if not member.work_item_ids
                )
                operation = {
                    (47879112, "G"): "keep",
                    (47879112, "C"): "remove",
                    (47879112, "T"): "defer",
                    (44908684, "C"): "defer",
                }
                (revised,) = pending_reviews(
                    self.answer(
                        journey,
                        review.request_event_id,
                        {
                            "action": "revise",
                            "proposal_id": proposal.proposal_id,
                            "asserted_by": "researcher:owner",
                            "operations": [
                                *(
                                    {
                                        "operation": operation[
                                            (
                                                item.target.one_based_position,
                                                item.target.alt,
                                            )
                                        ],
                                        "work_item_id": item.work_item_id,
                                    }
                                    for item in proposal.work_items
                                ),
                                {
                                    "operation": "amend_request",
                                    "request_member_id": unresolved.request_member_id,
                                    "request": {
                                        "rsid": "rs7412",
                                        "request_text": "Check rs7412 instead.",
                                    },
                                },
                            ],
                        },
                    )
                )
                self.answer(
                    journey,
                    revised.request_event_id,
                    {
                        "action": "stop",
                        "proposal_id": revised.request.proposal.proposal_id,
                        "reason": "Stop after the revision.",
                    },
                )
            basis = self.basis(journey)
            view = basis.collection_view
            # The amended first request keeps its place in the question.
            self.assertEqual(
                [snp.requested_rsid for snp in view.snps],
                ["rs7412", "rs2228570", "rs429358"],
            )
            rows = _rows(view)
            self.assertEqual(
                rows["rs2228570"],
                (
                    "candidates",
                    [
                        ("NC_000012.12:47879112 A>C", "removed", None, None),
                        ("NC_000012.12:47879112 A>G", "not_approved", None, None),
                        ("NC_000012.12:47879112 A>T", "deferred", None, None),
                    ],
                ),
            )
            self.assertEqual(
                rows["rs429358"],
                ("deferred", [(RS429358_TARGET, "deferred", None, None)]),
            )
            notes = {
                (snp.requested_rsid, item.target): item.state_origin.by
                for snp in view.snps
                for item in snp.items
                if item.state in {"removed", "deferred"}
            }
            self.assertEqual(set(notes.values()), {"researcher:owner"})
            self.assertEqual(len(notes), 3)
            for limitation in (
                "Request rs429358 was deferred by researcher:owner.",
                "(rs2228570 NC_000012.12:47879112 A>C) was removed by researcher:owner.",
                "(rs2228570 NC_000012.12:47879112 A>T) was deferred by researcher:owner.",
            ):
                self.assertTrue(
                    any(
                        text.endswith(limitation) for text in _texts(basis.limitations)
                    ),
                    (limitation, _texts(basis.limitations)),
                )
            report = self.publish(journey, basis)
            section = _section(
                self.reopen(journey, report.report_digest), "Requested SNPs"
            )
            self.assertIn(
                "  NC_000012.12:47879112 A>C: removed (by researcher:owner)", section
            )
            self.assertIn(
                "  NC_000012.12:47879112 A>T: deferred (by researcher:owner)", section
            )
            self.assertIn("  state: deferred (by researcher:owner)", section)

    def test_a_removed_item_leaves_only_its_own_id_out_of_a_shared_qualification(
        self,
    ):
        # One sentence every lookup states is one qualification carrying each
        # looked-up item's id. A revision removes rs2228570's A>C before the lookups;
        # the researcher approves the rest and the Run goes on. The shared
        # qualification keeps the other items' ids, and the removed item is accounted
        # for by its own limitation, with its id.
        # Exercised path: the Investigation Review's ``revise`` ``remove``, then
        # ``approve``; the failure module's item Review is not exercised here.
        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.application.reviews import pending_reviews

        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(
                Path(directory).resolve(), request_file("rs2228570", "rs429358")
            )
            with self.dependencies(RunOutputStore(), {"rs2228570": RS2228570}):
                (review,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                proposal = review.request.proposal
                removed = next(
                    item.work_item_id
                    for item in proposal.work_items
                    if (item.target.one_based_position, item.target.alt)
                    == (47879112, "C")
                )
                kept = sorted(
                    item.work_item_id
                    for item in proposal.work_items
                    if item.work_item_id != removed
                )
                self.assertEqual(len(kept), 3)
                (revised,) = pending_reviews(
                    self.answer(
                        journey,
                        review.request_event_id,
                        {
                            "action": "revise",
                            "proposal_id": proposal.proposal_id,
                            "asserted_by": "researcher:owner",
                            "operations": [
                                {
                                    "operation": "remove"
                                    if item.work_item_id == removed
                                    else "keep",
                                    "work_item_id": item.work_item_id,
                                }
                                for item in proposal.work_items
                            ],
                        },
                    )
                )
                self.answer(
                    journey,
                    revised.request_event_id,
                    {
                        "action": "approve",
                        "proposal_id": revised.request.proposal.proposal_id,
                    },
                )
            basis = self.basis(journey)
        # L1 reads per item as its depth was measured or not. Only rs429358
        # (NC_000019.10) lies on a sequence of the offline BAM; the kept rs2228570
        # items' sequence is not in it, so their depth was not measured. The
        # removed item carries neither sentence.
        measured = sorted(
            item.work_item_id
            for item in proposal.work_items
            if item.target.one_based_position == 44908684
        )
        unmeasured = sorted(set(kept) - set(measured))
        self.assertEqual((len(measured), len(unmeasured)), (1, 2))
        by_text = {item.text: item for item in basis.limitations}
        shared = by_text[
            "Coverage, callability, biological absence, function, causality, "
            "and diagnosis are not inferred."
        ]
        depth = by_text[
            "Callability, biological absence, function, causality, and "
            "diagnosis are not inferred; read depth is the Run's measured value."
        ]
        self.assertEqual(shared.work_item_ids, tuple(unmeasured))
        self.assertEqual(depth.work_item_ids, tuple(measured))
        (own,) = [
            item
            for item in basis.limitations
            if item.text.startswith(f"Work item {removed} (")
        ]
        self.assertIn("was removed", own.text)
        self.assertEqual((own.work_item_ids, own.request_member_ids), ((removed,), ()))

    def test_a_request_set_aside_in_mixed_ways_keeps_each_items_disposition(self):
        # rs2228570 has no kept ALT: A>C removed, A>G and A>T deferred, while
        # rs429358 is kept. The request is not summarized as either.
        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.application.reviews import pending_reviews

        operation = {
            (47879112, "C"): "remove",
            (47879112, "G"): "defer",
            (47879112, "T"): "defer",
            (44908684, "C"): "keep",
        }
        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(
                Path(directory).resolve(), request_file("rs2228570", "rs429358")
            )
            with self.dependencies(RunOutputStore(), {"rs2228570": RS2228570}):
                (review,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                proposal = review.request.proposal
                (revised,) = pending_reviews(
                    self.answer(
                        journey,
                        review.request_event_id,
                        {
                            "action": "revise",
                            "proposal_id": proposal.proposal_id,
                            "asserted_by": "researcher:owner",
                            "operations": [
                                {
                                    "operation": operation[
                                        (
                                            item.target.one_based_position,
                                            item.target.alt,
                                        )
                                    ],
                                    "work_item_id": item.work_item_id,
                                }
                                for item in proposal.work_items
                            ],
                        },
                    )
                )
                self.answer(
                    journey,
                    revised.request_event_id,
                    {
                        "action": "stop",
                        "proposal_id": revised.request.proposal.proposal_id,
                        "reason": "Stop after the revision.",
                    },
                )
            basis = self.basis(journey)
            self.assertEqual(
                _rows(basis.collection_view)["rs2228570"],
                (
                    "candidates",
                    [
                        ("NC_000012.12:47879112 A>C", "removed", None, None),
                        ("NC_000012.12:47879112 A>G", "deferred", None, None),
                        ("NC_000012.12:47879112 A>T", "deferred", None, None),
                    ],
                ),
            )
            self.assertFalse(
                [
                    text
                    for text in _texts(basis.limitations)
                    if "Request rs2228570" in text
                ]
            )
            for allele, state in (
                ("C", "removed"),
                ("G", "deferred"),
                ("T", "deferred"),
            ):
                self.assertTrue(
                    any(
                        text.endswith(
                            f"(rs2228570 NC_000012.12:47879112 A>{allele}) was "
                            f"{state} by researcher:owner."
                        )
                        for text in _texts(basis.limitations)
                    ),
                    (allele, _texts(basis.limitations)),
                )

    def test_a_correction_that_changes_nothing_is_refused_without_a_write(self):
        # Donor case F3: an exact correction that changes no membership is
        # refused, and the pending basis stays the one to answer.
        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.application.queries import read_completion
        from bio_neb.application.reviews import pending_reviews
        from bio_neb.reporting.finalization_commands import (
            CorrectFinalizationCommand,
            FinalizationApplicationRefused,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            journey = self.child(root, request_file("rs429358"))
            with self.dependencies(RunOutputStore()):
                (review,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                self.answer(
                    journey,
                    review.request_event_id,
                    {
                        "action": "approve",
                        "proposal_id": review.request.proposal.proposal_id,
                    },
                )
                basis = self.basis(journey)
                correction_file = root / "correction.json"
                correction_file.write_text(
                    json.dumps(
                        {
                            "asserted_by": "researcher:owner",
                            "reason": "Keep the entry as it is.",
                            "requested_changes": [
                                {
                                    "action": "set_report_membership",
                                    "member_id": basis.accounting[0].accounting_id,
                                    "included": True,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                pending = read_completion(
                    Journey.open(journey.path)
                ).finalization.pending_request
                before = journey.path.read_bytes()
                with self.assertRaises(FinalizationApplicationRefused) as refused:
                    advance(
                        CorrectFinalizationCommand(
                            journey_file=str(journey.path),
                            request_event_id=pending.request_event_id,
                            basis_digest=basis.basis_digest,
                            head_event_id=pending.request_event_id,
                            correction_file=str(correction_file),
                        )
                    )
            self.assertEqual(refused.exception.code, "finalization_correction_noop")
            self.assertEqual(journey.path.read_bytes(), before)
            self.assertEqual(self.basis(journey), basis)


class ReconciledFactSlotTests(unittest.TestCase):
    """Planning's rsID slot joins each Investigation slot that requested it."""

    @staticmethod
    def slot(requirement_id, kind, source, requested, verified=None):
        from bio_neb.parabricks.requirement_resolution import FactSlotStatus

        return FactSlotStatus(
            requirement_id=requirement_id,
            kind=kind,
            subject_key=requirement_id,
            status="verified" if verified else "asserted" if requested else "to_find",
            verification_source=source,
            requested_value=requested,
            verified_value=verified,
            mismatch=bool(requested and verified and requested != verified),
            summary="as retained",
        )

    def test_one_requested_rsid_reads_as_one_entry_per_request(self):
        from bio_neb.reporting.finalization import reconciled_fact_slots

        sample = self.slot("sample_identity", "sample_identity", None, "HG001")
        # Planning's rsID slots are scientific investigations naming no source: the
        # Investigation's model chooses it.
        planned = self.slot(
            "plan-rs429358", "scientific_investigation", None, "rs429358"
        )
        other = self.slot("plan-rs7412", "scientific_investigation", None, "rs7412")
        # Two requests for one rsID stay two entries: a shared rsID is not proof
        # of one identity (0039); each carries the planning slot.
        first = self.slot(
            "member-1", "requested_rsid", "NCBI RefSNP", "rs429358", "rs429358"
        )
        second = self.slot(
            "member-2", "requested_rsid", "NCBI RefSNP", "rs429358", "rs429358"
        )
        # Planning's rsID kind is required: a slot of another kind with the same text
        # and no source is not reconciled.
        lookalike = self.slot("plan-x", "sample_identity", None, "rs429358")
        reported = reconciled_fact_slots(
            (sample, planned, other, lookalike), (first, second)
        )
        self.assertEqual(
            [
                (
                    slot.requirement_id,
                    [earlier.requirement_id for earlier in slot.planning_slots],
                )
                for slot in reported
            ],
            [
                ("sample_identity", []),
                ("plan-rs7412", []),
                ("plan-x", []),
                ("member-1", ["plan-rs429358"]),
                ("member-2", ["plan-rs429358"]),
            ],
        )
        # A slot with nothing reconciled serializes as before.
        self.assertNotIn("planning_slots", reported[0].model_dump(mode="json"))


if __name__ == "__main__":
    unittest.main()
