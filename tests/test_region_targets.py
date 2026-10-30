"""A region target (#367, merged from #377; owner option b, 2026-10-23).

A region member is planned in its own planner Attempt: ``locate_variant`` names its
sequence on the Run's assembly and reads the Run's reference bases over it, with no rsID
listed, and ``compile_proposal`` states it once with ``intended_action=observe_region``.
The server checks every stated value against the lookup. The member's one Work item is
the region; each VCF record in it becomes its own row after approval.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.investigation.investigation import RegionTarget, UserAssertion
from bio_neb.investigation.investigation_request import (
    InvestigationRequestItem,
    StatedAssembly,
    StatedRegion,
)
from bio_neb.recording import read_evidence
from tests.item_state_runs import ItemStateRuns
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.observation_input_fakes import (
    SAMPLE,
    TABLE_READING_ENVIRON,
    RunOutputStore,
)
from tests.sequential_investigation_support import (
    SequentialAdaptiveModel,
    SourceResponsesOnly,
    input_and_assertion,
    run_attempt,
    run_taxon,
    source_basis,
)

REGION = StatedRegion(contig="MT", start=300, end=312)
# The model's naming of the region's sequence: GRCh38's mitochondrion, its quote the
# request's own words.
MT_NAMED = {
    "assembly_accession": "GCF_000001405.40",
    "record": "NC_012920.1",
    "quote": "MT:300-312",
}
# Synthetic bases in the retained span's shape; no GET read them.
BASES = "AAACCCCCCCTCC"


def _region_member():
    investigation_input, _ = input_and_assertion()
    request = InvestigationRequestItem(
        request_text="every record in MT:300-312",
        region=REGION,
        normalized_question_action="observe_region",
    )
    assertion = UserAssertion.create_collection(
        asserted_by="researcher",
        requests=(request,),
        predecessor_ids=(investigation_input.input_id,),
    )
    return investigation_input, assertion


def _planner(output, located=None):
    """A planner that locates the region, names its sequence when the first result asks
    (#467; its own fact, #479 G2.2), locates it again, then states ``output`` for it;
    ``located`` adds arguments to its locate_variant calls."""
    calls = []

    async def respond(messages, info):  # noqa: ANN001, ANN202
        calls.append(info.function_tools)
        if len(calls) in (1, 3):
            name = "locate_variant"
            arguments = dict(located or {})
        elif len(calls) == 2:
            name = "name_sequence"
            arguments = {
                "refsnp_event_id": source_basis(info)["refsnp_event_id"],
                **MT_NAMED,
            }
        else:
            name = info.output_tools[0].name
            arguments = {"refsnp_event_id": source_basis(info)["refsnp_event_id"]}
            arguments.update(output)
        return ModelResponse(
            parts=[
                ToolCallPart(name, json.dumps(arguments), tool_call_id=f"r{len(calls)}")
            ]
        )

    return SimpleNamespace(model=ScriptedModelEndpoint(respond)), calls


COMPLETE = {
    "outcome": "complete",
}


class RegionPlanningTests(unittest.TestCase):
    def _run(self, model):
        investigation_input, assertion = _region_member()
        sources = SourceResponsesOnly(reference_bases={"NC_012920.1:299:13": BASES})
        directory = self.enterContext(tempfile.TemporaryDirectory())
        result, journey = run_attempt(
            Path(directory),
            model,
            sources,
            investigation_input=investigation_input,
            assertion=assertion,
        )
        return result, read_evidence(journey), sources

    def test_a_region_member_is_located_and_proposed_as_its_one_region_item(self):
        model, _ = _planner(COMPLETE)

        result, events, sources = self._run(model)

        # The first lookup asked for the sequence; the second located the region.
        *_, lookup = [
            e["payload"]["result"]
            for e in events
            if e["event_type"] == "RefSnpToolResultRecorded"
        ]
        self.assertEqual(
            (lookup["status"], lookup["location"]["reference_bases"]),
            ("region_located", BASES),
        )
        self.assertEqual(
            sources.operations,
            [
                # The first call asks for the region's sequence, the second reads
                # the name the model recorded.
                "fetch_assembly_report:GCF_000001405.40",
                "ncbi_sequence_names:GCF_000001405.40",
                "fetch_assembly_report:GCF_000001405.40",
                "ncbi_sequence_names:GCF_000001405.40",
                "ncbi_reference_span:NC_012920.1:299:13",
            ],
        )
        proposal = result.pending_request.proposal
        [item] = proposal.work_items
        self.assertIsInstance(item.target, RegionTarget)
        self.assertEqual(
            (
                item.intended_action,
                item.target.contig,
                item.target.start,
                item.target.end,
            ),
            ("observe_region", "NC_012920.1", 300, 312),
        )
        self.assertEqual(proposal.request_members[0].assembly_report, None)

    def test_a_region_call_sending_a_null_source_is_read_as_naming_none(self):
        """A region names no variant source (#469); a null source is that same call
        (review r1 ST1/SP1): it is admitted, and the Journey reopens with its result."""
        model, _ = _planner(COMPLETE, located={"source": None})

        result, events, _sources = self._run(model)

        # The first lookup asked for the sequence; the second located the region.
        *_, lookup = [
            e["payload"]["result"]
            for e in events
            if e["event_type"] == "RefSnpToolResultRecorded"
        ]
        self.assertEqual(
            (lookup["status"], lookup["lookup_source"]), ("region_located", "ncbi")
        )
        self.assertEqual(result.status, "review_pending")

    def test_a_region_output_that_changes_the_located_region_is_refused(self):
        """The proposal states only its outcome and cited ids (#478 G2.1): an output
        that writes a region end of its own is refused as a field the draft does not
        hold, and no proposal carries the changed region."""
        model, _ = _planner(COMPLETE | {"region_end": 320})

        result, events, _ = self._run(model)

        refused = [
            e["payload"]
            for e in events
            if e["event_type"] == "AgentOutputValidated"
            and e["payload"]["status"] != "valid"
        ]
        self.assertTrue(refused)
        self.assertEqual(refused[0]["errors"], ["region_end:extra_forbidden"])
        self.assertEqual(refused[0]["output"]["region_end"], 320)
        self.assertIn('\\"type\\": \\"extra_forbidden\\"', json.dumps(refused[0]))
        self.assertNotIn(
            "valid",
            [
                e["payload"]["status"]
                for e in events
                if e["event_type"] == "AgentOutputValidated"
            ],
        )
        # The Attempt stops for the researcher's review; no proposal was compiled.
        self.assertEqual(result.status, "agent_work_review_pending")
        self.assertIsNone(result.pending_request)


# A recorded Run VCF's two records inside NC_012920.1:300-312, as written, and one
# record outside the region (synthetic) that the region read leaves out.
REGION_VCF = (
    b"##fileformat=VCFv4.2\n"
    b"##contig=<ID=NC_012920.1,length=16569>\n"
    b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + SAMPLE.encode() + b"\n"
    b"NC_012920.1\t302\t.\tA\tAC,ACC\t2670.02\t.\t.\tGT\t1/2\n"
    b"NC_012920.1\t310\t.\tT\tTC,C\t2413.06\t.\t.\tGT\t1/2\n"
    b"NC_012920.1\t400\t.\tG\tA\t50\tPASS\t.\tGT\t0/1\n"
)


REGION_REQUEST = {
    "schema_version": "bio-neb.investigation-request/3",
    "asserted_by": "researcher:owner",
    "requests": [
        {
            "request": "every record in MT:300-312",
            "target": {"region": {"contig": "MT", "start": 300, "end": 312}},
            "normalized_question_action": "observe_region",
        }
    ],
}


class RegionRunTests(ItemStateRuns, unittest.TestCase):
    def test_records_past_the_record_limit_are_counted_not_dropped(self):
        from unittest.mock import patch

        from bio_neb.investigation.observation_defaults import RegionDefaults

        with patch(
            "bio_neb.investigation.investigation_observation_commands.RegionDefaults",
            lambda: RegionDefaults(record_limit=1),
        ):
            snp = self._region_basis().collection_view.snps[0]

        self.assertEqual(
            ([row.position for row in snp.records], snp.records_not_reported),
            ([302], 1),
        )

    def _region_basis(self):
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.reviews import pending_reviews

        directory = self.enterContext(tempfile.TemporaryDirectory())
        journey = self.child(Path(directory).resolve(), REGION_REQUEST, vcf=REGION_VCF)
        with use_run_advancement_dependencies(
            environ=TABLE_READING_ENVIRON,
            investigation_model_factory=SequentialAdaptiveModel().model,
            investigation_source_adapter_factory=lambda: SourceResponsesOnly(
                reference_bases={"NC_012920.1:299:13": BASES}
            ),
            investigation_vcf_source_factory=RunOutputStore(vcf=REGION_VCF),
        ):
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
        return self.basis(journey)

    def test_each_record_in_a_region_is_its_own_row_through_the_report(self):
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.reviews import pending_reviews

        request = {
            "schema_version": "bio-neb.investigation-request/3",
            "asserted_by": "researcher:owner",
            "requests": [
                {
                    "request": "every record in MT:300-312",
                    "target": {"region": {"contig": "MT", "start": 300, "end": 312}},
                    "normalized_question_action": "observe_region",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            journey = self.child(Path(directory).resolve(), request, vcf=REGION_VCF)
            sources = SourceResponsesOnly(reference_bases={"NC_012920.1:299:13": BASES})
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                investigation_model_factory=SequentialAdaptiveModel().model,
                investigation_source_adapter_factory=lambda: sources,
                investigation_vcf_source_factory=RunOutputStore(vcf=REGION_VCF),
            ):
                (review,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                [item] = review.request.proposal.work_items
                self.assertEqual(item.intended_action, "observe_region")
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
            # A region-only collection is measured too: the Run's health once,
            # and the region's depth over its own stretch. The offline BAM holds
            # the mitochondrion but no read on it.
            from bio_neb.journey import Journey

            events = Journey.open(journey.path).read()
            kinds = [e["event_type"] for e in events]
            self.assertEqual(kinds.count("RunHealthMeasured"), 1)
            assert basis.run_health is not None
            self.assertIsNotNone(basis.run_health.health.measurement)
            [coverage] = [
                e["payload"]
                for e in events
                if e["event_type"] == "InvestigationItemCoverageRecorded"
            ]
            self.assertEqual(
                (coverage["contig"], coverage["span_start"], coverage["span_end"]),
                ("NC_012920.1", 300, 312),
            )
            self.assertEqual(
                (
                    coverage["measurement"]["depth"],
                    coverage["measurement"]["mean_depth"],
                    coverage["batch_flags"],
                ),
                (0, 0.0, []),
            )
            [snp] = basis.collection_view.snps
            self.assertEqual(
                [
                    (row.position, row.ref, row.alts, row.genotype, row.reference)
                    for row in snp.records
                ],
                [
                    (302, "A", ("AC", "ACC"), "1/2", "matches"),
                    (310, "T", ("TC", "C"), "1/2", "matches"),
                ],
            )
            self.assertEqual(snp.records_not_reported, 0)
            # A region's records admit no finding.
            self.assertEqual(
                [admission.status for admission in basis.admissions], ["not_admitted"]
            )
            report = self.publish(journey, basis)
            text = self.reopen(journey, report.report_digest)
            # The readable page places a region by its interval.
            body = self.readable(journey, report.report_digest).split(
                "\n## Appendix\n"
            )[0]
            self.assertIn("NC_012920.1:300-312", body)
            self.assertNotIn("work-item:", body)
        self.assertIn("  record 302 A>AC,ACC (GT 1/2): REF matches", text)
        self.assertIn("  record 310 T>TC,C (GT 1/2): REF matches", text)
        self.assertNotIn("record 400", text)


def _stated_region_member(version: str):
    investigation_input, _ = input_and_assertion()
    request = InvestigationRequestItem(
        request_text="every record in MT:300-312",
        region=REGION,
        stated_assembly=StatedAssembly(name=version, cited_line=f"on {version}"),
        normalized_question_action="observe_region",
    )
    assertion = UserAssertion.create_collection(
        asserted_by="researcher",
        requests=(request,),
        predecessor_ids=(investigation_input.input_id,),
    )
    return investigation_input, assertion


@run_taxon(9606)
class RegionVersionAndAmendTests(unittest.TestCase):
    """#367 r1 D2: a region's stated version and an answer's replacement target go
    through the one effective-target owner."""

    def _run(self, version):
        investigation_input, assertion = _stated_region_member(version)
        directory = self.enterContext(tempfile.TemporaryDirectory())
        sources = SourceResponsesOnly(reference_bases={"NC_012920.1:299:13": BASES})
        result, journey = run_attempt(
            Path(directory),
            SequentialAdaptiveModel(),
            sources,
            investigation_input=investigation_input,
            assertion=assertion,
        )
        return result, journey, sources

    def test_a_region_stated_on_the_runs_version_is_located_and_proposed(self):
        result, _, _ = self._run("GRCh38")

        [item] = result.pending_request.proposal.work_items
        self.assertEqual(
            (item.intended_action, item.target.start, item.target.end),
            ("observe_region", 300, 312),
        )

    def test_a_region_stated_on_another_version_is_an_issue_not_reread(self):
        result, journey, sources = self._run("GRCh37")

        question = result.pending_request.question
        self.assertEqual(question.issue.verdict, "incompatible")
        # GRCh37 is not the Run's literally: the model names it from NCBI's
        # search, and the named version's lookup is the issue.
        unnamed, lookup = [
            e["payload"]["result"]
            for e in read_evidence(journey)
            if e["event_type"] == "RefSnpToolResultRecorded"
        ]
        self.assertEqual(unnamed["status"], "version_unnamed")
        self.assertEqual(
            lookup["location"]["version_naming"]["assembly_accession"],
            "GCF_000001405.13",
        )
        self.assertIn(
            "states its region on GRCh37; the Run's reference is GRCh38.p14",
            lookup["conflicts"][0],
        )
        # No reference bases are read on the Run's sequence for another version.
        self.assertNotIn("ncbi_reference_span", " ".join(sources.operations))

    def test_a_region_amended_to_an_rsid_is_looked_up_by_that_rsid(self):
        from bio_neb.investigation.investigation import (
            AmendInvestigationFindingResponse,
        )
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )

        result, journey, sources = self._run("GRCh37")
        review = result.pending_request
        respond_investigation_review(
            RespondInvestigationReviewCommand(
                journey_file=str(journey.path),
                request_event_id=review.request_event_id,
                response=AmendInvestigationFindingResponse(
                    action="amend",
                    collection_member_id=review.question.collection_member_id,
                    revision_number=review.question.revision_number,
                    instruction="Check rs429358 instead.",
                    target={"rsid": "rs429358"},
                    asserted_by="researcher",
                ),
            )
        )
        calls = []

        async def respond(messages, info):  # noqa: ANN001, ANN202
            calls.append(info)
            basis = source_basis(info)
            steps = (
                ("fetch_refsnp", {"rsid": "rs429358"}),
                (
                    "fetch_assembly_report",
                    {
                        "refsnp_event_id": basis["refsnp_event_id"],
                        "assembly_accession": "GCF_000001405.40",
                    },
                ),
            )
            if len(calls) <= len(steps):
                name, arguments = steps[len(calls) - 1]
            else:
                name, arguments = (
                    info.output_tools[0].name,
                    {
                        **basis,
                        "outcome": "complete",
                    },
                )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name, json.dumps(arguments), tool_call_id=f"a{len(calls)}"
                    )
                ]
            )

        continued = continue_investigation(
            ContinueInvestigationCommand(journey_file=str(journey.path)),
            environ={"LOCAL_MODEL_KEY": "offline"},
            model_factory=ScriptedModelEndpoint(respond),
            source_adapter_factory=lambda: sources,
        )

        refused = [
            e["payload"]
            for e in read_evidence(journey)
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_name"] == "fetch_refsnp"
            and e["payload"]["status"] != "valid"
        ]
        self.assertEqual(refused, [])
        [item] = continued.pending_request.proposal.work_items
        self.assertEqual(
            (item.intended_action, item.target.one_based_position),
            ("observe_variant", 44908684),
        )

    def test_a_retried_region_is_reread_as_its_region_and_proposed_again(self):
        """#367 r2 S4: a proposal's region Work item retries as its exact region."""
        from bio_neb.investigation.investigation import ReviseInvestigationResponse
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )

        result, journey, sources = self._run("GRCh38.p14")
        review = result.pending_request
        [item] = review.proposal.work_items
        revised = respond_investigation_review(
            RespondInvestigationReviewCommand(
                journey_file=str(journey.path),
                request_event_id=review.request_event_id,
                response=ReviseInvestigationResponse(
                    action="revise",
                    proposal_id=review.proposal.proposal_id,
                    asserted_by="researcher",
                    operations=(
                        {"operation": "retry", "work_item_id": item.work_item_id},
                    ),
                ),
            )
        )
        # The retry names the region on the Run's sequence, as its Work item does.
        [retry] = revised.revisions[-1].work_requests
        self.assertEqual(
            (retry.region, retry.normalized_question_action),
            (
                StatedRegion(contig="NC_012920.1", start=300, end=312),
                "observe_region",
            ),
        )

        continued = continue_investigation(
            ContinueInvestigationCommand(journey_file=str(journey.path)),
            environ={"LOCAL_MODEL_KEY": "offline"},
            model_factory=SequentialAdaptiveModel().model,
            source_adapter_factory=lambda: sources,
        )

        proposal = continued.pending_request.proposal
        [again] = proposal.work_items
        self.assertEqual(
            (
                proposal.revision_number,
                again.previous_work_item_id,
                again.intended_action,
                again.target,
            ),
            (2, item.work_item_id, "observe_region", item.target),
        )
        self.assertEqual(again.target.reference_bases, BASES)
