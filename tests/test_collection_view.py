"""The whole collection reviewable together, with cross-SNP cases surfaced (#360 W).

Scoping each member's prompt to its own evidence must not lose the whole picture:
the proposal Review, the Finalization basis and the Grounded report each show
every requested SNP in one view, and flag the cases that are not simple across
SNPs. A flag reports only; it never changes an admission or a claim.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.investigation.investigation import (
    InvestigationRequestItem,
    UserAssertion,
)
from bio_neb.investigation.investigation_request import StatedAssembly, StatedPosition
from bio_neb.investigation.qualification_origins import RequestUnresolved
from bio_neb.journey import Journey
from bio_neb.providers.ncbi_investigation import InvestigationSourceResponse
from bio_neb.recording import read_evidence, review_head
from tests.explainer_fakes import explained_by_default
from tests.observation_input_fakes import SAMPLE, TABLE_READING_ENVIRON, RunOutputStore
from tests.refsnp_retained_bodies import (
    RS429358,
    RS429358_DOCUMENT,
    RS2228570,
    RS10735810,
    RS756427959,
    RS878871521,
    refsnp_body,
)
from tests.run_prerequisites import RunPrerequisites
from tests.sequential_investigation_support import (
    NOT_FOUND,
    IssueSourceResponses,
    SequentialAdaptiveModel,
    SourceResponsesOnly,
    _current_member_messages,
    run_attempt,
)
from tests.terminal_journey_fakes import forked_child

# A second rsID dbSNP places at rs429358's own locus (synthetic: no retained
# record holds two rsIDs at one position).
RS1111111 = refsnp_body(
    {**RS429358_DOCUMENT, "refsnp_id": "1111111", "dbsnp1_merges": []}
)
BODIES = {
    "rs10735810": RS10735810,
    "rs2228570": RS2228570,
    "rs429358": RS429358,
    "rs1111111": RS1111111,
}
# The Run's VCF for the Finalization case: rs2228570 carries G only, written as
# a padded CA>CG record that matches only once trimmed, and an upstream deletion
# spans rs429358's target, so its lookup is overlapped.
CASE_VCF = (
    b"##fileformat=VCFv4.2\n"
    b"##contig=<ID=NC_000012.12,length=133275309>\n"
    b"##contig=<ID=NC_000019.10,length=58617616>\n"
    b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + SAMPLE.encode() + b"\n"
    b"NC_000012.12\t47879111\t.\tCA\tCG\t50\tPASS\t.\tGT\t0/1\n"
    b"NC_000019.10\t44908682\t.\tATT\tA\t50\tPASS\t.\tGT\t0/1\n"
)
CASE_REQUEST = {
    "schema_version": "bio-neb.investigation-request/2",
    "asserted_by": "researcher:owner",
    "requests": [
        {"request": f"Does HG001 carry {rsid}?", "target": {"rsid": rsid}}
        for rsid in ("rs10735810", "rs2228570", "rs429358", NOT_FOUND)
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


def _cases(view):
    """Each case by its code and the requested rsIDs it names, in request order."""
    rsids = {snp.request_member_id: snp.requested_rsid for snp in view.snps}
    return {
        (case.code, tuple(rsids[member] for member in case.request_member_ids))
        for case in view.cases
    }


class ProposalReviewCollectionViewTests(unittest.TestCase):
    def test_the_proposal_review_shows_every_snp_with_its_cross_snp_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            projection, _journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=SequentialAdaptiveModel().model),
                SourceResponsesOnly(refsnp_bodies=BODIES),
                assertion=_assertion(
                    "rs10735810", "rs2228570", "rs429358", "rs1111111"
                ),
            )
            self.assertEqual(projection.status, "review_pending")
            review = projection.pending_request
            view = review.collection_view
            (proposal,) = projection.proposals
            # Every requested SNP, in the researcher's order, with its items.
            self.assertEqual(
                [
                    (snp.requested_rsid, snp.verified_rsid, len(snp.items))
                    for snp in view.snps
                ],
                [
                    ("rs10735810", "rs2228570", 3),
                    ("rs2228570", "rs2228570", 3),
                    ("rs429358", "rs429358", 1),
                    ("rs1111111", "rs1111111", 1),
                ],
            )
            self.assertEqual(
                [snp.request_member_id for snp in view.snps],
                [member.request_member_id for member in proposal.request_members],
            )
            self.assertEqual(
                _cases(view),
                {
                    ("merged_identity", ("rs10735810",)),
                    ("shared_identity", ("rs10735810", "rs2228570")),
                    ("multi_alt", ("rs10735810",)),
                    ("multi_alt", ("rs2228570",)),
                    ("shared_locus", ("rs429358", "rs1111111")),
                },
            )
            # Each case names its evidence: the members' retrieved sources.
            sources = {
                member.request_member_id: {
                    member.refsnp.source.source_id,
                    member.assembly_report.source.source_id,
                }
                for member in proposal.request_members
            }
            for case in view.cases:
                with self.subTest(case=case.code):
                    self.assertEqual(
                        set(case.evidence_ids),
                        set().union(
                            *(sources[member] for member in case.request_member_ids)
                        ),
                    )
                    self.assertTrue(case.origin.audit())
            # Flags report only: the proposal and its items are unchanged.
            self.assertEqual(review.proposal, proposal)
            self.assertEqual(
                {snp.requested_rsid: snp.cases for snp in view.snps}["rs429358"],
                ("shared_locus",),
            )

    def test_requests_that_state_their_rsid_only_in_words_keep_their_identity(self):
        # Spec r1 F1: a request may name its rsID only in its words; the model types
        # it (record_target_reading) and the view reads it as the request's identity,
        # as the fact slots do.
        with tempfile.TemporaryDirectory() as directory:
            projection, _journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=_WordsRsidModel().model),
                SourceResponsesOnly(refsnp_bodies=BODIES),
                assertion=UserAssertion.create_collection(
                    asserted_by="researcher",
                    predecessor_ids=("offline-research-request",),
                    requests=tuple(
                        InvestigationRequestItem(
                            request_text=f"Does HG001 carry {rsid}?"
                        )
                        for rsid in ("rs10735810", "rs2228570")
                    ),
                ),
            )
            self.assertEqual(projection.status, "review_pending")
            view = projection.pending_request.collection_view
            self.assertEqual(
                [snp.requested_rsid for snp in view.snps], ["rs10735810", "rs2228570"]
            )
            self.assertIn(("merged_identity", ("rs10735810",)), _cases(view))
            (shared,) = [case for case in view.cases if case.code == "shared_identity"]
            self.assertEqual(
                shared.origin.audit(),
                "The requests for rs10735810, rs2228570 all resolve to rs2228570.",
            )

    def test_sources_that_disagree_about_one_rsid_are_flagged(self):
        # dbSNP answered the two lookups of rs2228570 differently (the second
        # lists only G): both records stay evidence, and the case is surfaced.
        with tempfile.TemporaryDirectory() as directory:
            projection, _journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=SequentialAdaptiveModel().model),
                _ChangingRecordSources(refsnp_bodies=BODIES),
                assertion=_assertion("rs10735810", "rs2228570"),
            )
            view = projection.pending_request.collection_view
            self.assertIn(
                ("contradictory_sources", ("rs10735810", "rs2228570")), _cases(view)
            )
            self.assertEqual([len(snp.items) for snp in view.snps], [3, 1])


def _unlisted(contig, position, ref, alt, *, rsid=None, text=None):
    """A request stating a position on GRCh38 that no source lists, or an rsID."""
    if rsid is not None:
        return InvestigationRequestItem(
            request_text=f"Does HG001 carry {rsid}?", rsid=rsid
        )
    return InvestigationRequestItem(
        request_text=text or f"Does HG001 carry {contig}:{position} {ref}>{alt}?",
        position=StatedPosition(contig=contig, position=position, ref=ref, alt=alt),
        stated_assembly=StatedAssembly(name="GRCh38", cited_line="Aligned to GRCh38."),
    )


def _position_view(*requests, reference_bases):
    with tempfile.TemporaryDirectory() as directory:
        projection, _journey = run_attempt(
            Path(directory),
            SimpleNamespace(model=SequentialAdaptiveModel().model),
            SourceResponsesOnly(refsnp_bodies=BODIES, reference_bases=reference_bases),
            assertion=UserAssertion.create_collection(
                asserted_by="researcher",
                predecessor_ids=("offline-research-request",),
                requests=requests,
            ),
        )
    assert projection.status == "review_pending", projection.status
    return projection.pending_request.collection_view


class RequestRecordViewTests(unittest.TestCase):
    """#460: each request keeps the position its researcher stated and the source
    answer its check read, so the readable report copies them, never infers them."""

    def test_each_request_keeps_its_stated_position_and_the_answer_its_check_read(
        self,
    ):
        stated = _unlisted("19", 44908684, "T", "G")
        rsid = _unlisted("19", 44908684, "T", "C", rsid="rs429358")
        with tempfile.TemporaryDirectory() as directory:
            projection, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=SequentialAdaptiveModel().model),
                SourceResponsesOnly(
                    refsnp_bodies=BODIES,
                    reference_bases={"NC_000019.10:44908683:1": "T"},
                ),
                assertion=UserAssertion.create_collection(
                    asserted_by="researcher",
                    predecessor_ids=("offline-research-request",),
                    requests=(stated, rsid),
                ),
            )
            # Each lookup result the Run recorded, by the source answer it read.
            recorded = {
                event["payload"]["result"]["source"]["source_id"]: event["payload"][
                    "result"
                ]
                for event in Journey.open(journey.path).read()
                if event["event_type"] == "RefSnpToolResultRecorded"
            }

        view = projection.pending_request.collection_view
        position, record = view.snps
        self.assertEqual(position.stated_position, stated.position)
        self.assertIsNone(record.stated_position)
        for snp in view.snps:
            with self.subTest(request=snp.request_text):
                result = recorded[snp.lookup.source_id]
                self.assertEqual(result.get("requested_rsid"), snp.requested_rsid)
                self.assertEqual(
                    snp.lookup.model_dump(mode="json"),
                    {
                        "authority": result["source"]["authority"],
                        "operation": result["source"]["operation"],
                        "asked": result["source"]["request_identity"],
                        "status_code": result["source"]["status_code"],
                        "outcome": result["status"],
                        "verdict": result["assessment"]["verdict"],
                        "rationale": result["assessment"]["rationale"],
                        "source_id": result["source"]["source_id"],
                    },
                )


class PositionCollectionViewTests(unittest.TestCase):
    """A position no source lists resolves with no rsID; that absence is no identity
    the view may group by (#141 Run 2: four unlisted C. elegans sites read as one)."""

    def test_unlisted_positions_share_no_identity(self):
        view = _position_view(
            _unlisted("19", 44908684, "T", "G"),
            _unlisted("19", 44908822, "C", "A"),
            reference_bases={
                "NC_000019.10:44908683:1": "T",
                "NC_000019.10:44908821:1": "C",
            },
        )

        self.assertEqual([snp.verified_rsid for snp in view.snps], [None, None])
        self.assertEqual(_cases(view), set())

    def test_a_position_at_an_rsids_locus_shares_that_locus(self):
        view = _position_view(
            _unlisted("19", 44908684, "T", "C", rsid="rs429358"),
            _unlisted("19", 44908684, "T", "G"),
            reference_bases={"NC_000019.10:44908683:1": "T"},
        )

        (shared,) = [case for case in view.cases if case.code == "shared_locus"]
        self.assertEqual(
            shared.origin.audit(),
            "NC_000019.10:44908684 T>G, rs429358 share the locus "
            "NC_000019.10:44908684.",
        )
        self.assertNotIn("None", " ".join(case.origin.audit() for case in view.cases))

    def test_one_target_stated_padded_and_trimmed_is_one_identity(self):
        # Standards r7 addendum S3: T>G and its suffix-padded TA>GA are the same
        # normalized Work item; the view finds no second identity and no conflict.
        view = _position_view(
            _unlisted("19", 44908684, "T", "G"),
            _unlisted("19", 44908684, "TA", "GA"),
            reference_bases={
                "NC_000019.10:44908683:1": "T",
                "NC_000019.10:44908683:2": "TA",
            },
        )

        self.assertEqual(
            {item.work_item_id for snp in view.snps for item in snp.items},
            {view.snps[0].items[0].work_item_id},
        )
        self.assertEqual(_cases(view), set())

    def test_unequal_reference_spans_from_one_position_agree(self):
        # Spec r8 S4: normalized, T>TA is an insertion at 44908685 with an empty REF
        # and TAC>T a deletion of AC there; C>G and A>G state that base. Each REF
        # holds on the reference: the locus is shared, and nothing is contradicted.
        for first, second, bases in (
            (
                ("T", "TA"),
                ("C", "G"),
                {"NC_000019.10:44908683:1": "T", "NC_000019.10:44908684:1": "C"},
            ),
            (
                ("TAC", "T"),
                ("A", "G"),
                {"NC_000019.10:44908683:3": "TAC", "NC_000019.10:44908684:1": "A"},
            ),
        ):
            with self.subTest(first=first, second=second):
                view = _position_view(
                    _unlisted("19", 44908684, *first),
                    _unlisted("19", 44908685, *second),
                    reference_bases=bases,
                )

                self.assertEqual(_cases(view), {("shared_locus", (None, None))})

    def test_distinct_targets_worded_alike_still_share_their_locus(self):
        # Standards r6 S2, Spec r6 S3: the typed target, not the request's words,
        # is the identity of a position no source lists.
        view = _position_view(
            _unlisted("19", 44908684, "T", "G", text="Check this position."),
            _unlisted("19", 44908684, "T", "A", text="Check this position."),
            reference_bases={"NC_000019.10:44908683:1": "T"},
        )

        (shared,) = [case for case in view.cases if case.code == "shared_locus"]
        self.assertEqual(len(shared.request_member_ids), 2)
        self.assertEqual(
            shared.origin.audit(),
            "NC_000019.10:44908684 T>A, NC_000019.10:44908684 T>G share the locus "
            "NC_000019.10:44908684.",
        )


class _WordsRsidModel(SequentialAdaptiveModel):
    """Looks up the rsID a request states only in its words, as a model reading
    its assignment's request would."""

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        response = await super().respond(messages, info)
        (part,) = response.parts
        arguments = part.args_as_dict()
        if part.tool_name == "fetch_refsnp" and arguments.get("rsid") is None:
            request, _refsnp, _assembly, _basis, *_ = _current_member_messages(messages)
            arguments["rsid"] = request["request_text"].split()[-1].rstrip("?")
        return ModelResponse(
            parts=[
                ToolCallPart(
                    part.tool_name,
                    json.dumps(arguments),
                    tool_call_id=part.tool_call_id,
                )
            ]
        )


class _ChangingRecordSources(SourceResponsesOnly):
    """The second rs2228570 lookup returns a record that lists G only."""

    def fetch_refsnp(self, rsid):  # noqa: ANN001, ANN201
        if rsid == "rs2228570" and f"fetch_refsnp:{rsid}" in self.operations:
            document = json.loads(RS2228570)
            for placement in document["primary_snapshot_data"][
                "placements_with_allele"
            ]:
                placement["alleles"] = [
                    item
                    for item in placement["alleles"]
                    if item["allele"]["spdi"]["inserted_sequence"] in {"A", "G"}
                ]
            self.refsnp_bodies = {**self.refsnp_bodies, rsid: refsnp_body(document)}
        return super().fetch_refsnp(rsid)


class FinalizationCollectionViewTests(RunPrerequisites, unittest.TestCase):
    def test_finalization_and_the_report_keep_the_per_snp_view_with_vcf_cases(self):
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.queries import read_completion
        from bio_neb.application.reviews import pending_reviews

        with tempfile.TemporaryDirectory() as directory:
            parent = self.prerequisite(
                "executed",
                investigation_request=CASE_REQUEST,
                output_payloads={"VCF": CASE_VCF},
            ).journey_file
            journey = forked_child(Path(directory).resolve(), parent_file=parent)
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                investigation_model_factory=SequentialAdaptiveModel().model,
                investigation_source_adapter_factory=lambda: _CaseSources(
                    refsnp_bodies=BODIES
                ),
                investigation_vcf_source_factory=RunOutputStore(vcf=CASE_VCF),
            ):
                (issue,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
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
                self.assertIn(
                    ("no_work_item", (NOT_FOUND,)),
                    _cases(review.request.collection_view),
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
            basis = read_completion(Journey.open(journey.path)).finalization.bases[-1]
            view = basis.collection_view
            self.assertEqual(
                [snp.requested_rsid for snp in view.snps],
                ["rs10735810", "rs2228570", "rs429358", NOT_FOUND],
            )
            outcomes = {
                snp.requested_rsid: [(item.target, item.outcome) for item in snp.items]
                for snp in view.snps
            }
            self.assertEqual(
                outcomes["rs2228570"],
                [
                    ("NC_000012.12:47879112 A>C", "target_allele_not_carried"),
                    ("NC_000012.12:47879112 A>G", "exact_allele_observed"),
                    ("NC_000012.12:47879112 A>T", "target_allele_not_carried"),
                ],
            )
            self.assertEqual(outcomes["rs10735810"], outcomes["rs2228570"])
            self.assertEqual(
                outcomes["rs429358"],
                [("NC_000019.10:44908684 T>C", "locus_overlapped")],
            )
            self.assertEqual(outcomes[NOT_FOUND], [])
            # Why it has no Work item is its typed origin, never a stored sentence
            # (E): the request's and its case's.
            (no_work,) = [case for case in view.cases if case.code == "no_work_item"]
            unresolved = RequestUnresolved(
                source="refsnp", verdict="source_answered", decision="approved"
            )
            self.assertEqual(
                {snp.requested_rsid: snp.no_work for snp in view.snps}[NOT_FOUND],
                unresolved,
            )
            self.assertEqual(no_work.origin.reason, unresolved)
            self.assertEqual(
                _cases(view),
                {
                    ("merged_identity", ("rs10735810",)),
                    ("shared_identity", ("rs10735810", "rs2228570")),
                    ("multi_alt", ("rs10735810",)),
                    ("multi_alt", ("rs2228570",)),
                    ("mixed_alt_outcomes", ("rs10735810",)),
                    ("mixed_alt_outcomes", ("rs2228570",)),
                    ("trimmed_match", ("rs10735810",)),
                    ("trimmed_match", ("rs2228570",)),
                    ("overlapping_record", ("rs429358",)),
                    ("no_work_item", (NOT_FOUND,)),
                },
            )
            # An rsID NCBI was asked about and did not find reads as such, not as
            # "not yet verified".
            (slot,) = [
                slot
                for slot in basis.fact_slots
                if slot.kind == "requested_rsid" and slot.requested_value == NOT_FOUND
            ]
            self.assertEqual(
                (slot.status, slot.summary),
                (
                    "asserted",
                    "asserted by researcher; NCBI RefSNP checked it: source_answered",
                ),
            )
            # The VCF-side cases link the observations they rest on.
            observations = {
                item.work_item_id: item.observation_id
                for item in basis.sample_observations
            }
            (overlap,) = [c for c in view.cases if c.code == "overlapping_record"]
            self.assertTrue(
                {observations[item] for item in overlap.work_item_ids}
                <= set(overlap.evidence_ids)
            )
            # Admissions are the lookups' own; the view changed none.
            self.assertEqual(len(basis.admissions), len(basis.sample_observations))
            report = self.publish(journey, basis)
            self.assertEqual(report.collection_view, basis.collection_view)
            rendered = self.reopen(journey, report.report_digest)
            question = rendered[rendered.index("\nQuestion\n") :].split("\n\n")[0]
            for rsid in ("rs10735810", "rs2228570", "rs429358", NOT_FOUND):
                self.assertIn(f"Does HG001 carry {rsid}?", question)
            section = rendered[rendered.index("\nRequested SNPs\n") :].split("\n\n")[0]
            self.assertIn(
                "- rs10735810 (verified rs2228570): Does HG001 carry rs10735810?",
                section,
            )
            self.assertIn(
                "  NC_000012.12:47879112 A>G: exact_allele_observed (GT 0/1)", section
            )
            self.assertIn("  NC_000019.10:44908684 T>C: locus_overlapped", section)
            self.assertIn(
                "  no Work item: refsnp: source_answered (approved, checked not verified)",
                section,
            )
            cases = rendered[rendered.index("\nCross-SNP cases\n") :].split("\n\n")[0]
            for code in ("shared_identity", "overlapping_record", "no_work_item"):
                self.assertIn(code, cases)
            # A case reads as its typed origin's audit sentence.
            self.assertIn(
                f"- no_work_item: {NOT_FOUND} has no Work item: refsnp: source_answered "
                "(approved, checked not verified).",
                cases,
            )

    def test_indel_items_read_their_anchored_vcf_records_into_the_report(self):
        # Through the Run: rs756427959 and rs878871521's retained dbSNP records,
        # and their records in a recorded live VCF (14397 CTGT>C 0/1; MT 302
        # A>AC,ACC 1/2). Each Work item holds the trimmed allele; its lookup
        # takes the anchor base from the record's REF.
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.queries import read_completion
        from bio_neb.application.reviews import pending_reviews

        vcf = (
            b"##fileformat=VCFv4.2\n"
            b"##contig=<ID=NC_000001.11,length=248956422>\n"
            b"##contig=<ID=NC_012920.1,length=16569>\n"
            b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + SAMPLE.encode()
            + b"\n"
            b"NC_000001.11\t14397\t.\tCTGT\tC\t166.6\tPASS\t.\tGT\t0/1\n"
            b"NC_012920.1\t302\t.\tA\tAC,ACC\t50\tPASS\t.\tGT\t1/2\n"
        )
        request = {
            "schema_version": "bio-neb.investigation-request/2",
            "asserted_by": "researcher:owner",
            "requests": [
                {"request": f"Does HG001 carry {rsid}?", "target": {"rsid": rsid}}
                for rsid in ("rs756427959", "rs878871521")
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            parent = self.prerequisite(
                "executed",
                investigation_request=request,
                output_payloads={"VCF": vcf},
            ).journey_file
            journey = forked_child(Path(directory).resolve(), parent_file=parent)
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                investigation_model_factory=SequentialAdaptiveModel().model,
                investigation_source_adapter_factory=lambda: SourceResponsesOnly(
                    refsnp_bodies={
                        "rs756427959": RS756427959,
                        "rs878871521": RS878871521,
                    }
                ),
                investigation_vcf_source_factory=RunOutputStore(vcf=vcf),
            ):
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
            basis = read_completion(Journey.open(journey.path)).finalization.bases[-1]
            outcomes = {
                snp.requested_rsid: [(item.target, item.outcome) for item in snp.items]
                for snp in basis.collection_view.snps
            }
            self.assertEqual(
                outcomes,
                {
                    "rs756427959": [
                        ("NC_000001.11:14398 TGT>-", "exact_allele_observed")
                    ],
                    "rs878871521": [
                        ("NC_012920.1:303 ->C", "exact_allele_observed"),
                        ("NC_012920.1:303 ->CC", "exact_allele_observed"),
                        ("NC_012920.1:303 ->CCC", "target_allele_not_carried"),
                        ("NC_012920.1:303 ->" + "C" * 11, "target_allele_not_carried"),
                        ("NC_012920.1:303 ->" + "C" * 14, "target_allele_not_carried"),
                    ],
                },
            )
            (deletion,) = [
                item
                for item in basis.sample_observations
                if item.result.target.ref == "TGT"
            ]
            self.assertEqual(
                (
                    deletion.result.record.one_based_position,
                    deletion.result.record.ref,
                    deletion.result.record.alts,
                    deletion.result.genotype.raw_genotype,
                ),
                (14397, "CTGT", ("C",), "0/1"),
            )
            report = self.publish(journey, basis)
            rendered = self.reopen(journey, report.report_digest)
            section = rendered[rendered.index("\nRequested SNPs\n") :].split("\n\n")[0]
            for line in (
                "  NC_000001.11:14398 TGT>-: exact_allele_observed (GT 0/1)",
                "  NC_012920.1:303 ->CC: exact_allele_observed (GT 1/2)",
                "  NC_012920.1:303 ->CCC: target_allele_not_carried (GT 1/2)",
            ):
                self.assertIn(line, section)

    def test_an_eva_snv_item_reads_its_vcf_record_into_the_report(self):
        # An EVA-backed Work item through the Run's VCF lookup, its Finalization
        # basis and the Grounded report. The executed prerequisite is a GRCh38 Run,
        # and its scripted model chooses EVA (both lookup Tools on every Run, no
        # route); the EVA bodies are synthetic, in the recorded shape
        # (`tests/eva_retained_bodies.py`), placing rs81000001 on GRCh38's chr19
        # (`CM000681.2`, RefSeq `NC_000019.10`).
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.queries import read_completion
        from bio_neb.application.reviews import pending_reviews

        vcf = (
            b"##fileformat=VCFv4.2\n"
            b"##contig=<ID=NC_000019.10,length=58617616>\n"
            b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + SAMPLE.encode()
            + b"\n"
            b"NC_000019.10\t44908684\t.\tT\tC\t99\tPASS\t.\tGT\t0/1\n"
        )
        request = {
            "schema_version": "bio-neb.investigation-request/2",
            "asserted_by": "researcher:owner",
            "requests": [
                {
                    "request": "Does HG001 carry rs81000001?",
                    "target": {"rsid": "rs81000001"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            parent = self.prerequisite(
                "executed",
                investigation_request=request,
                output_payloads={"VCF": vcf},
            ).journey_file
            journey = forked_child(Path(directory).resolve(), parent_file=parent)
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                # The model chooses EVA on this GRCh38 Run (no route forces it).
                investigation_model_factory=SequentialAdaptiveModel(
                    lookups=("fetch_eva_variant",)
                ).model,
                investigation_source_adapter_factory=_EvaOnGrch38,
                investigation_vcf_source_factory=RunOutputStore(vcf=vcf),
            ):
                (review,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                (item,) = review.request.proposal.work_items
                self.assertEqual(item.target.variant_text, "NC_000019.10:44908684 T>C")
                finalizing = self.answer(
                    journey,
                    review.request_event_id,
                    {
                        "action": "approve",
                        "proposal_id": review.request.proposal.proposal_id,
                    },
                )
            self.assertEqual(finalizing.status, "finalization_review_pending")
            basis = read_completion(Journey.open(journey.path)).finalization.bases[-1]
            (snp,) = basis.collection_view.snps
            self.assertEqual(
                [(i.target, i.outcome) for i in snp.items],
                [("NC_000019.10:44908684 T>C", "exact_allele_observed")],
            )
            (observation,) = basis.sample_observations
            self.assertEqual(
                (
                    observation.result.record.one_based_position,
                    observation.result.genotype.raw_genotype,
                ),
                (44908684, "0/1"),
            )
            eva_sources = {
                source.operation
                for source in basis.retrieved_sources
                if source.authority == "eva"
            }
            self.assertEqual(
                eva_sources,
                {"eva_clustered_variant", "eva_contig_alias", "eva_submitted_variants"},
            )
            report = self.publish(journey, basis)
            rendered = self.reopen(journey, report.report_digest)
            section = rendered[rendered.index("\nRequested SNPs\n") :].split("\n\n")[0]
            self.assertIn(
                "  NC_000019.10:44908684 T>C: exact_allele_observed (GT 0/1)", section
            )

    def answer(self, journey: Journey, request_event_id: str, response: dict):
        """Answer through the native CLI, as the researcher does."""
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


class _EvaOnGrch38(SourceResponsesOnly):
    """NCBI's GRCh38 assembly report and synthetic EVA answers for rs81000001, an SNV
    at GRCh38 chr19 44908684 T>C; dbSNP is never asked."""

    _EVA = "https://www.ebi.ac.uk/eva/webservices"

    def __init__(self) -> None:
        # GRCh38's report with its INSDC twin, which the EVA lookup reads, in NCBI's
        # recorded shape.
        super().__init__(
            assembly_body=json.dumps(
                {
                    "reports": [
                        {
                            "accession": "GCF_000001405.40",
                            "paired_accession": "GCA_000001405.29",
                            "assembly_info": {
                                "assembly_name": "GRCh38.p14",
                                "assembly_status": "current",
                            },
                        }
                    ]
                }
            ).encode()
        )

    def fetch_refsnp(self, rsid):  # noqa: ANN001, ANN201
        raise AssertionError("the substituted route never reaches dbSNP")

    def _answer(self, url, document):  # noqa: ANN001, ANN201
        self.operations.append(url)
        return InvestigationSourceResponse(
            request_identity=url,
            status_code=200,
            body=json.dumps(document).encode(),
            observed_at="2026-10-17T12:00:00Z",
            elapsed_time_ms=2,
        )

    def fetch_eva_clustered_variant(self, rsid):  # noqa: ANN001, ANN201
        return self._answer(
            f"{self._EVA}/identifiers/v1/clustered-variants/81000001",
            [
                {
                    "accession": 81000001,
                    "data": {
                        "assemblyAccession": "GCA_000001405.29",
                        "contig": "CM000681.2",
                        "start": 44908684,
                        "type": "SNV",
                    },
                }
            ],
        )

    def fetch_eva_contig_alias(self, insdc_contig):  # noqa: ANN001, ANN201
        return self._answer(
            f"{self._EVA}/contig-alias/v1/chromosomes/genbank/{insdc_contig}",
            {
                "_embedded": {
                    "chromosomeEntities": [
                        {
                            "insdcAccession": "CM000681.2",
                            "refseq": "NC_000019.10",
                            "assembly": {
                                "insdcAccession": "GCA_000001405.29",
                                "refseq": "GCF_000001405.40",
                                "name": "GRCh38.p14",
                            },
                        }
                    ]
                }
            },
        )

    def fetch_eva_submitted_variants(self, rsid):  # noqa: ANN001, ANN201
        return self._answer(
            f"{self._EVA}/identifiers/v1/clustered-variants/81000001/submitted",
            [
                {
                    "accession": 81000002,
                    "data": {
                        "referenceSequenceAccession": "GCA_000001405.29",
                        "contig": "CM000681.2",
                        "start": 44908684,
                        "referenceAllele": "T",
                        "alternateAllele": "C",
                        "clusteredVariantAccession": 81000001,
                        "projectAccession": "PRJEB00001",
                    },
                }
            ],
        )


class _CaseSources(IssueSourceResponses):
    """dbSNP for the Finalization case: the retained records, and one rsID it
    does not know."""

    def fetch_refsnp(self, rsid):  # noqa: ANN001, ANN201
        if rsid == NOT_FOUND:
            return super().fetch_refsnp(rsid)
        return SourceResponsesOnly.fetch_refsnp(self, rsid)


if __name__ == "__main__":
    unittest.main()
