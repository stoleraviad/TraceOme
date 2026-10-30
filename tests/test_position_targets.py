"""A request's stated position reaches its own Work item through the Investigation
Agent, and the intent check compares the stated position and alleles (#369).

Each test runs one planning Attempt through the shared lifecycle
(`sequential_investigation_support.run_attempt`) with the scripted model and the
scripted NCBI answers; the lookup, the Journey checks and the proposal are real.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bio_neb.investigation.investigation import UserAssertion
from bio_neb.investigation.investigation_request import (
    InvestigationRequestItem,
    StatedAssembly,
    StatedPosition,
    StatedTarget,
)
from bio_neb.recording import read_evidence
from tests.position_move_support import grch37_move_sources
from tests.refsnp_retained_bodies import RS429358
from tests.sequence_naming_support import assembly_report_text
from tests.sequential_investigation_support import (
    SequentialAdaptiveModel,
    SourceResponsesOnly,
    input_and_assertion,
    run_attempt,
    run_taxon,
)


def _assertion(request: InvestigationRequestItem):
    investigation_input, _ = input_and_assertion()
    return investigation_input, UserAssertion.create_collection(
        asserted_by="paper:doi:10.1000/example",
        requests=(request,),
        predecessor_ids=(investigation_input.input_id,),
    )


def _position(contig, position, ref=None, alt=None, *, rsid=None, text=None):
    return InvestigationRequestItem(
        request_text=text or f"The paper reports {contig}:{position}.",
        rsid=rsid,
        position=StatedPosition(contig=contig, position=position, ref=ref, alt=alt),
    )


def _plan(request, sources, model=None):
    investigation_input, assertion = _assertion(request)
    with tempfile.TemporaryDirectory() as directory:
        result, _ = run_attempt(
            Path(directory),
            model or SequentialAdaptiveModel(),
            sources,
            investigation_input=investigation_input,
            assertion=assertion,
        )
    return result


def _targets(result):
    return [
        (
            item.target.contig,
            item.target.one_based_position,
            item.target.ref,
            item.target.alt,
            item.target.aliases,
        )
        for item in result.pending_request.proposal.work_items
    ]


class PositionTargetTests(unittest.TestCase):
    def test_a_stated_position_becomes_the_work_item_of_its_listed_rsid(self):
        sources = SourceResponsesOnly(
            refsnp_bodies={"rs429358": RS429358},
            reference_bases={"NC_000019.10:44908683:1": "T"},
            listed_rsids={"NC_000019.10:44908683:T:C": (429358,)},
        )

        result = _plan(_position("chr19", 44908684, "T", "C"), sources)

        run_names = [
            "fetch_assembly_report:GCF_000001405.40",
            "ncbi_sequence_names:GCF_000001405.40",
        ]
        to_record = [
            *run_names,
            "ncbi_reference_span:NC_000019.10:44908683:1",
            "ncbi_spdi_rsids:NC_000019.10:44908683:T:C",
            "fetch_refsnp:rs429358",
            # No version is stated: the allele is listed on GRCh37 too.
            "ncbi_sequence_names:GCF_000001405.25",
        ]
        self.assertEqual(
            sources.operations,
            [
                # The model names the Run's sequence, then GRCh37's, each after a
                # sequence_unnamed result asks for it.
                *run_names,
                *to_record,
                *to_record,
                # GRCh37's listing answers 404 ("does not match any RSID"): the lookup
                # ends on it, and the model calls again quoting it.
                "ncbi_spdi_rsids:NC_000019.9:44908683:T:C",
                *to_record,
                "ncbi_spdi_rsids:NC_000019.9:44908683:T:C",
                "fetch_assembly_report:GCF_000001405.40",
            ],
        )
        self.assertEqual(
            _targets(result), [("NC_000019.10", 44908684, "T", "C", ("rs429358",))]
        )

    def test_an_unread_other_version_admits_no_work_item(self) -> None:
        # Standards r3 S7 / Spec r3 S6 through the caller: GRCh37's listing answers
        # 200 with no list, so no version is settled and the item asks.
        class _Unread(SourceResponsesOnly):
            def fetch_spdi_rsids(self, spdi):  # noqa: ANN001, ANN201
                if not spdi.startswith("NC_000019.9:"):
                    return super().fetch_spdi_rsids(spdi)
                url = f"https://api.ncbi.nlm.nih.gov/variation/v0/spdi/{spdi}/rsids"
                return self._answer("ncbi_spdi_rsids", spdi, url, 200, b"{}")

        result = _plan(
            _position("chr19", 44908684, "T", "C"),
            _Unread(refsnp_bodies={"rs429358": RS429358}),
        )

        self.assertEqual(result.proposals, ())
        issue = result.pending_request.question.issue
        self.assertEqual((issue.source, issue.verdict), ("refsnp", "malformed"))
        self.assertEqual(
            issue.conflicts,
            # The typed ending's observation.
            ("NCBI's rsIDs at the stated allele on GRCh37.p13 could not be read.",),
        )

    def test_a_wrong_identity_record_authorizes_no_version_get(self) -> None:
        # Standards r4 S8: the listed rsID's GET answers another record, complete
        # with placements; its identity conflict ends the lookup at once.
        import json

        body = json.loads(RS429358)
        body["refsnp_id"] = "99"
        sources = SourceResponsesOnly(
            refsnp_bodies={"rs429358": json.dumps(body).encode()},
            listed_rsids={"NC_000019.10:44908683:T:C": (429358,)},
        )

        result = _plan(_position("chr19", 44908684, "T", "C"), sources)

        self.assertEqual(result.proposals, ())
        self.assertNotIn("ncbi_sequence_names:GCF_000001405.25", sources.operations)
        issue = result.pending_request.question.issue
        self.assertEqual(
            (issue.verdict, issue.conflicts),
            ("conflicted", ("NCBI RefSNP identity did not match the requested rsID.",)),
        )

    def test_a_position_no_source_lists_is_its_own_work_item_without_an_rsid(self):
        sources = SourceResponsesOnly(
            reference_bases={"NC_000019.10:44908683:5": "TGCGG"},
        )

        request = _position("19", 44908684, "TGCGG", "A").model_copy(
            update={
                "stated_assembly": StatedAssembly(
                    name="GRCh38", cited_line="Reads were aligned to GRCh38."
                )
            }
        )
        result = _plan(request, sources)

        self.assertNotIn("fetch_refsnp", " ".join(sources.operations))
        self.assertEqual(
            _targets(result), [("NC_000019.10", 44908684, "TGCGG", "A", ())]
        )

    def test_a_position_without_alleles_or_rsid_is_deferred_with_no_question(self):
        # A recorded request ("Chromosome 19 position 44908822 on GRCh38."): no
        # source lists rsIDs without an allele, so the item is deferred out of the
        # Run, with its reason; it asks nothing, and the other item's Work goes on.
        words = "Chromosome 19 position 44908822 on GRCh38."
        coordinate = InvestigationRequestItem(
            request_text=words,
            position=StatedPosition(contig="19", position=44908822),
            stated_assembly=StatedAssembly(name="GRCh38", cited_line=words),
        )
        investigation_input, _ = input_and_assertion()
        assertion = UserAssertion.create_collection(
            asserted_by="paper:doi:10.1000/example",
            requests=(
                coordinate,
                InvestigationRequestItem(request_text="rs429358", rsid="rs429358"),
            ),
            predecessor_ids=(investigation_input.input_id,),
        )
        sources = SourceResponsesOnly(refsnp_bodies={"rs429358": RS429358})
        with tempfile.TemporaryDirectory() as directory:
            result, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        self.assertIsNone(result.pending_request.question)
        proposal = result.pending_request.proposal
        deferred, resolved = proposal.request_members
        self.assertEqual(
            (deferred.not_checked.reason_code, deferred.work_item_ids),
            ("position_without_alleles", ()),
        )
        self.assertEqual(
            deferred.not_checked.reason,
            "it states 19:44908822 without its alleles or an rsID, and the Run's "
            "sources list a variant only at a stated allele, so no source was asked "
            "about the position",
        )
        self.assertEqual(len(resolved.work_item_ids), 1)
        # The ended view's explanation row for the deferred member carries no lookup
        # (its proposal member keeps none), so its own state words say that no
        # source was asked about it.
        from bio_neb.investigation.collection_view import ended_collection_view
        from bio_neb.reporting.explanation_rows import explanation_rows

        ended = ended_collection_view(result, proposal)
        deferred_row = explanation_rows(ended)[0]
        self.assertEqual(
            (deferred_row.subject, deferred_row.state, deferred_row.words),
            (
                words,
                "not_checked",
                "deferred out of the Run: it was stated without its alleles or an "
                "rsID, so no source was asked about it and the VCF was not searched; "
                "this does not say whether the sample carries a variant there",
            ),
        )
        # The readable report's row reads the same owner's sentence (a minimal
        # report holding the ended view).
        from types import SimpleNamespace

        from bio_neb.reporting.readable_report import readable_rows

        (page_row, _resolved_row) = readable_rows(
            SimpleNamespace(collection_view=ended, work_items=())
        )
        self.assertEqual(
            page_row.not_searched,
            "Deferred out of the Run: it was stated without its alleles or an rsID, "
            "so no source was asked about it and the VCF was not searched; this does "
            "not say whether the sample carries a variant there",
        )
        # Its lookup read only the Run's assembly report, and no question opened.
        self.assertNotIn("ncbi_sequence_names", " ".join(sources.operations[:1]))
        self.assertFalse(
            [e for e in events if e["event_type"] == "InvestigationQuestionRecorded"]
        )

    def test_a_position_without_alleles_added_in_a_revision_is_deferred(self):
        # Spec r1 R1-SP1: a revision's added coordinate with neither alleles nor an
        # rsID is deferred as in a first collection, asking nothing; its carried
        # sibling keeps its Work.
        from bio_neb.investigation.investigation import (
            AddInvestigationMember,
            KeepInvestigationMember,
            ReviseInvestigationResponse,
        )
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )
        from bio_neb.journey import Journey

        words = "Chromosome 19 position 44908822 on GRCh38."
        sources = SourceResponsesOnly(refsnp_bodies={"rs429358": RS429358})
        model = SequentialAdaptiveModel()
        with tempfile.TemporaryDirectory() as directory:
            first, journey = run_attempt(Path(directory), model, sources)
            review = first.pending_request
            (carried,) = review.proposal.work_items
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=review.request_event_id,
                    response=ReviseInvestigationResponse(
                        action="revise",
                        proposal_id=review.proposal.proposal_id,
                        asserted_by="researcher:owner",
                        operations=(
                            KeepInvestigationMember(
                                operation="keep", work_item_id=carried.work_item_id
                            ),
                            AddInvestigationMember(
                                operation="add",
                                request=InvestigationRequestItem(
                                    request_text=words,
                                    position=StatedPosition(
                                        contig="19", position=44908822
                                    ),
                                    stated_assembly=StatedAssembly(
                                        name="GRCh38", cited_line=words
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            )
            second = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model.model,
                source_adapter_factory=lambda: sources,
            )
            events = read_evidence(Journey.open(journey.path))

        self.assertIsNone(second.pending_request.question)
        proposal = second.pending_request.proposal
        self.assertEqual(proposal.revision_number, 2)
        self.assertEqual(
            [item.work_item_id for item in proposal.work_items], [carried.work_item_id]
        )
        (deferred,) = [m for m in proposal.request_members if m.not_checked]
        self.assertEqual(
            (deferred.not_checked.reason_code, deferred.work_item_ids),
            ("position_without_alleles", ()),
        )
        self.assertIn(deferred.not_checked.event_id, proposal.predecessor_ids)
        self.assertFalse(
            [
                e
                for e in events
                if e["event_type"] == "InvestigationReviewRequested"
                and e["payload"].get("question")
            ]
        )

    def test_a_bodyless_listing_404_is_decided_and_becomes_a_work_item(self):
        # A recorded C. elegans reply (EVA's listing answered 404 with no body): the
        # model cites the answer's status with quote_listed_none, the listing reads
        # as empty, and the stated allele is a Work item, with no item question.
        class _Bodyless(SourceResponsesOnly):
            def fetch_spdi_rsids(self, spdi):  # noqa: ANN001, ANN201
                url = f"https://api.ncbi.nlm.nih.gov/variation/v0/spdi/{spdi}/rsids"
                return self._answer("ncbi_spdi_rsids", spdi, url, 404, b"")

        sources = _Bodyless(reference_bases={"NC_000019.10:44908683:5": "TGCGG"})
        request = _position("19", 44908684, "TGCGG", "A").model_copy(
            update={
                "stated_assembly": StatedAssembly(
                    name="GRCh38", cited_line="Reads were aligned to GRCh38."
                )
            }
        )
        investigation_input, assertion = _assertion(request)
        with tempfile.TemporaryDirectory() as directory:
            result, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        (decided,) = [
            e["payload"]["decision"]
            for e in events
            if e["event_type"] == "InvestigationLookupDecided"
            and e["payload"]["decision"]["kind"] == "listed_none"
        ]
        self.assertEqual(decided, {"kind": "listed_none", "status_code": 404})
        self.assertEqual(
            _targets(result), [("NC_000019.10", 44908684, "TGCGG", "A", ())]
        )
        self.assertIsNone(result.pending_request.question)


class PositionIntentTests(unittest.TestCase):
    """The intent check compares what a request states, not only its rsID."""

    def _asked(self, request, sources):
        result = _plan(request, sources)
        self.assertEqual(result.proposals, ())
        question = result.pending_request.question
        self.assertIsNotNone(question.refused_output_event_id)
        return question

    def test_an_rsid_the_source_lists_elsewhere_than_stated_is_asked(self):
        # The paper places rs7412 at rs429358's allele.
        question = self._asked(
            _position(
                "chr19",
                44908684,
                "T",
                "C",
                rsid="rs7412",
                text="The paper reports rs7412 at chr19:44908684 T>C.",
            ),
            SourceResponsesOnly(
                refsnp_bodies={"rs429358": RS429358},
                reference_bases={"NC_000019.10:44908683:1": "T"},
                listed_rsids={"NC_000019.10:44908683:T:C": (429358,)},
            ),
        )

        self.assertIn(
            "requested rs7412, retrieved rs429358 at the stated position",
            question.missing_information,
        )

    def test_a_stated_chromosome_dbsnp_does_not_place_it_on_is_asked(self):
        # Standards r1 S1: the number matches, the chromosome does not.
        question = self._asked(
            _position(
                "chr1",
                44908684,
                rsid="rs429358",
                text="The paper reports rs429358 at chr1:44908684.",
            ),
            SourceResponsesOnly(refsnp_bodies={"rs429358": RS429358}),
        )

        self.assertIn(
            "the request places rs429358 at NC_000001.11:44908684; dbSNP places "
            "it at NC_000019.10:44908684",
            question.missing_information,
        )

    def test_a_stated_position_dbsnp_places_elsewhere_is_asked(self):
        question = self._asked(
            _position(
                "chr19",
                44908000,
                rsid="rs429358",
                text="The paper reports rs429358 at chr19:44908000.",
            ),
            SourceResponsesOnly(refsnp_bodies={"rs429358": RS429358}),
        )

        self.assertIn(
            "the request places rs429358 at NC_000019.10:44908000; dbSNP places "
            "it at NC_000019.10:44908684",
            question.missing_information,
        )
        # The live check's finding 1: a position difference asks for the position.
        self.assertTrue(
            question.question.startswith(
                "The source places the retrieved target differently than this "
                "request states it ("
            ),
            question.question,
        )
        self.assertIn(
            "Clarify with the position, alleles or genome version you mean",
            question.question,
        )
        self.assertNotIn("different rsID", question.question)
        # No version was stated: the record's other placements are cited (Spec r1 S2).
        self.assertIn(
            "with no version stated, dbSNP also places it at NC_000019.9:45411941 "
            "on GRCh37.p13",
            question.missing_information,
        )


class ClarifiedPositionTests(unittest.TestCase):
    def test_a_typed_clarify_corrects_a_refused_position(self) -> None:
        from bio_neb.investigation.investigation import ClarifyInvestigationResponse
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )
        from tests.sequential_investigation_support import SequentialSourceResponses

        sources = SequentialSourceResponses(
            reference_bases={
                "NC_000019.10:44908683:1": "T",
                "NC_000019.10:44908821:1": "T",
            },
            listed_rsids={
                "NC_000019.10:44908683:T:C": (429358,),
                "NC_000019.10:44908821:T:C": (7412,),
            },
        )
        investigation_input, assertion = _assertion(
            _position(
                "chr19",
                44908684,
                "T",
                "C",
                rsid="rs7412",
                text="The paper reports rs7412 at chr19:44908684 T>C.",
            )
        )
        model = SequentialAdaptiveModel()
        with tempfile.TemporaryDirectory() as directory:
            asked, journey = run_attempt(
                Path(directory),
                model,
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            question = asked.pending_request.question
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=asked.pending_request.request_event_id,
                    response=ClarifyInvestigationResponse(
                        action="clarify",
                        collection_member_id=question.collection_member_id,
                        revision_number=question.revision_number,
                        instruction="The paper's table puts rs7412 at 44908822.",
                        asserted_by="researcher",
                        target=StatedTarget(
                            rsid="rs7412",
                            position=StatedPosition(
                                contig="chr19", position=44908822, ref="T", alt="C"
                            ),
                        ),
                    ),
                )
            )
            corrected = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model.model,
                source_adapter_factory=lambda: sources,
            )

        self.assertIn("ncbi_spdi_rsids:NC_000019.10:44908821:T:C", sources.operations)
        self.assertEqual(
            _targets(corrected), [("NC_000019.10", 44908822, "T", "C", ("rs7412",))]
        )


class StatedVersionTests(unittest.TestCase):
    """A position stated on another genome version than the Run's (owner's rule,
    map #4 6236758799): compared with the record's own placement on that version;
    no position is mapped between versions."""

    def _grch37_sources(self):
        # Synthetic, in the retained sequence report's shape: GRCh37's chromosome 19.
        grch37 = (
            b'{"reports":[{"assembly_accession":"GCF_000001405.25","chr_name":"19",'
            b'"ucsc_style_name":"chr19","refseq_accession":"NC_000019.9",'
            b'"genbank_accession":"CM000681.1","role":"assembled-molecule",'
            b'"sequence_name":"19","length":59128983}],"total_count":1}'
        )

        class _Sources(SourceResponsesOnly):
            def fetch_sequence_names(self, assembly_accession):  # noqa: ANN001, ANN201
                if assembly_accession != "GCF_000001405.25":
                    return super().fetch_sequence_names(assembly_accession)
                return self._answer(
                    "ncbi_sequence_names",
                    assembly_accession,
                    "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
                    "GCF_000001405.25/assembly_report.txt",
                    200,
                    assembly_report_text(grch37),
                )

        return _Sources(refsnp_bodies={"rs429358": RS429358})

    def _request(self, position, name):
        return InvestigationRequestItem(
            request_text=f"rs429358 at chr19:{position} T>C on {name}.",
            rsid="rs429358",
            position=StatedPosition(
                contig="chr19", position=position, ref="T", alt="C"
            ),
            # The line states the version as the request names it: a version's
            # quote holds the version.
            stated_assembly=StatedAssembly(name=name, cited_line=f"Methods: {name}"),
        )

    def test_the_papers_own_placement_on_its_version_is_accepted(self):
        sources = self._grch37_sources()

        result = _plan(self._request(45411941, "GRCh37"), sources)

        self.assertEqual(
            sources.operations[:5],
            [
                # GRCh37 is not literally the Run's: the first call reads dbSNP's
                # record and asks the model to name the version from its
                # placements; the second reads it on GRCh37.p13.
                "fetch_assembly_report:GCF_000001405.40",
                "fetch_refsnp:rs429358",
                "fetch_assembly_report:GCF_000001405.40",
                "fetch_refsnp:rs429358",
                "ncbi_sequence_names:GCF_000001405.25",
            ],
        )
        # The Work item is the Run's own placement; the paper's is kept beside it.
        self.assertEqual(
            _targets(result), [("NC_000019.10", 44908684, "T", "C", ("rs429358",))]
        )

    def test_a_version_stated_by_its_accession_is_that_version(self):
        # A request may name a version by its assembly accession: the Run's own is
        # placed directly; another's is read on that version.
        for name, position in (
            ("GCF_000001405.40", 44908684),
            ("GCF_000001405.25", 45411941),
        ):
            with self.subTest(name=name):
                result = _plan(self._request(position, name), self._grch37_sources())

                self.assertEqual(
                    _targets(result),
                    [("NC_000019.10", 44908684, "T", "C", ("rs429358",))],
                )

    def test_a_cross_version_allele_selects_its_own_run_allele(self):
        # Spec r1 S3: a multiallelic record (rs429358 with a synthetic second ALT,
        # G, on both placements); the paper's G on GRCh37 is the Run's G alone.
        import copy
        import json

        body = json.loads(RS429358)
        for placement in body["primary_snapshot_data"]["placements_with_allele"]:
            other = copy.deepcopy(placement["alleles"][-1])
            other["allele"]["spdi"]["inserted_sequence"] = "G"
            placement["alleles"].append(other)
        sources = self._grch37_sources()
        sources.refsnp_bodies = {"rs429358": json.dumps(body).encode()}
        request = self._request(45411941, "GRCh37").model_copy(
            update={
                "position": StatedPosition(
                    contig="chr19", position=45411941, ref="T", alt="G"
                )
            }
        )

        result = _plan(request, sources)

        self.assertEqual(
            _targets(result), [("NC_000019.10", 44908684, "T", "G", ("rs429358",))]
        )

    def _multiallelic(self, *, run_has_g=True, extra=False):
        import copy
        import json

        body = json.loads(RS429358)
        placements = body["primary_snapshot_data"]["placements_with_allele"]
        for placement in placements:
            if placement["is_ptlp"] and not run_has_g:
                continue
            other = copy.deepcopy(placement["alleles"][-1])
            other["allele"]["spdi"]["inserted_sequence"] = "G"
            placement["alleles"].append(other)
        if extra:
            # Synthetic: a second, not top-level GRCh38 placement on another
            # sequence, and a second assembly named by the Run's placement.
            alt = copy.deepcopy(placements[0])
            alt["seq_id"], alt["is_ptlp"] = "NT_187636.1", False
            for allele in alt["alleles"]:
                allele["allele"]["spdi"]["seq_id"] = "NT_187636.1"
            placements.append(alt)
            placements[0]["placement_annot"]["seq_id_traits_by_assembly"].append(
                {
                    "assembly_accession": "GCF_000001405.39",
                    "assembly_name": "GRCh38.p13",
                }
            )
        return json.dumps(body).encode()

    def _paper_g(self):
        return self._request(45411941, "GRCh37").model_copy(
            update={
                "position": StatedPosition(
                    contig="chr19", position=45411941, ref="T", alt="G"
                )
            }
        )

    def test_the_run_placement_the_model_names_corresponds(self):
        """With two placements on the Run's assembly (a synthetic not-top-level
        NT_187636.1 beside NC_000019.10), the stated allele binds the one the model
        names, not a top-level pick by code (ADR 0049; owner #468 Q1 a). Standards r2
        S5, Spec r2 S3 still hold: another assembly a placement names does not widen
        it."""
        for named in ("NC_000019.10", "NT_187636.1"):
            with self.subTest(named=named):
                sources = self._grch37_sources()
                sources.refsnp_bodies = {"rs429358": self._multiallelic(extra=True)}
                model = SequentialAdaptiveModel(placements={"rs429358": named})

                result = _plan(self._paper_g(), sources, model)

                self.assertEqual(
                    _targets(result), [(named, 44908684, "T", "G", ("rs429358",))]
                )
                self.assertIn(
                    '"status":"placement_unnamed"', model.visible_messages[-2]
                )

    def test_a_stated_allele_with_no_run_allele_admits_none(self):
        sources = self._grch37_sources()
        sources.refsnp_bodies = {"rs429358": self._multiallelic(run_has_g=False)}

        result = _plan(self._paper_g(), sources)

        self.assertEqual(result.proposals, ())
        issue = result.pending_request.question.issue
        self.assertEqual((issue.source, issue.verdict), ("refsnp", "conflicted"))
        self.assertEqual(
            issue.conflicts,
            (
                "The allele the request states on GRCh37 has no one corresponding "
                "allele of rs429358 on the Run's GRCh38.p14.",
            ),
        )

    def test_a_merged_rsid_stated_on_another_version_is_followed(self):
        # Standards r2 S3: the merge is followed before the version comparison.
        from tests.refsnp_retained_bodies import RS4149056, RS60037639

        sources = SourceResponsesOnly(
            refsnp_bodies={"rs60037639": RS60037639, "rs4149056": RS4149056}
        )
        request = InvestigationRequestItem(
            request_text="rs60037639 at chr12:21331549 T>C (GRCh37).",
            rsid="rs60037639",
            position=StatedPosition(
                contig="chr12", position=21331549, ref="T", alt="C"
            ),
            stated_assembly=StatedAssembly(name="GRCh37", cited_line="Methods: GRCh37"),
        )

        result = _plan(request, sources)

        self.assertEqual(
            sources.operations[:7],
            [
                # The merge is followed before the version is named.
                "fetch_assembly_report:GCF_000001405.40",
                "fetch_refsnp:rs60037639",
                "fetch_refsnp:rs4149056",
                "fetch_assembly_report:GCF_000001405.40",
                "fetch_refsnp:rs60037639",
                "fetch_refsnp:rs4149056",
                "ncbi_sequence_names:GCF_000001405.25",
            ],
        )
        self.assertEqual(
            _targets(result), [("NC_000012.12", 21178615, "T", "C", ("rs4149056",))]
        )

    def test_a_position_its_version_does_not_hold_is_asked(self):
        result = _plan(self._request(44908684, "GRCh37"), self._grch37_sources())

        self.assertEqual(result.proposals, ())
        self.assertIn(
            "the request places rs429358 at chr19:44908684 on GRCh37; dbSNP places "
            "it at NC_000019.9:45411941 on GRCh37.p13",
            result.pending_request.question.missing_information,
        )


class AmendedPositionTests(unittest.TestCase):
    def test_a_typed_amend_sends_its_member_to_the_stated_position(self):
        from types import SimpleNamespace

        from bio_neb.investigation.investigation import (
            AmendInvestigationFindingResponse,
        )
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )
        from tests.sequential_investigation_support import (
            NOT_FOUND,
            AmendedLookupModel,
            IssueSourceResponses,
        )

        with tempfile.TemporaryDirectory() as directory:
            sources = IssueSourceResponses(
                reference_bases={"NC_000019.10:44908683:1": "T"},
                listed_rsids={"NC_000019.10:44908683:T:C": (429358,)},
            )
            model = AmendedLookupModel()
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                assertion=UserAssertion.create_collection(
                    asserted_by="researcher",
                    predecessor_ids=("offline-research-request",),
                    requests=tuple(
                        InvestigationRequestItem(
                            request_text=f"Does HG001 carry {rsid}?", rsid=rsid
                        )
                        for rsid in (NOT_FOUND, "rs7412")
                    ),
                ),
            )
            review = pending.pending_request
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=review.request_event_id,
                    response=AmendInvestigationFindingResponse(
                        action="amend",
                        collection_member_id=review.question.collection_member_id,
                        revision_number=review.question.revision_number,
                        instruction="The paper's variant is chr19:44908684 T>C.",
                        asserted_by="researcher",
                        target=StatedTarget(
                            position=StatedPosition(
                                contig="chr19", position=44908684, ref="T", alt="C"
                            )
                        ),
                    ),
                )
            )
            looked_up = len(sources.operations)
            rechecked = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model.model,
                source_adapter_factory=lambda: sources,
            )

        names = [
            "fetch_assembly_report:GCF_000001405.40",
            "ncbi_sequence_names:GCF_000001405.40",
        ]
        self.assertEqual(
            sources.operations[looked_up:],
            [
                # The model names the Run's sequence after a sequence_unnamed result
                # asks for it.
                *names,
                *names,
                "ncbi_reference_span:NC_000019.10:44908683:1",
                "ncbi_spdi_rsids:NC_000019.10:44908683:T:C",
                "fetch_refsnp:rs429358",
                "fetch_assembly_report:GCF_000001405.40",
            ],
        )
        (proposal,) = rechecked.proposals
        corrected, _ = proposal.request_members
        # The request keeps its own rsID; the amend's position chose the lookup.
        self.assertEqual(
            (corrected.refsnp.requested_rsid, corrected.refsnp.verified_rsid),
            (NOT_FOUND, "rs429358"),
        )
        self.assertEqual(
            [
                (
                    item.target.contig,
                    item.target.one_based_position,
                    item.target.aliases,
                )
                for item in proposal.work_items
            ],
            [
                ("NC_000019.10", 44908684, ("rs429358",)),
                ("NC_000019.10", 44908822, ("rs7412",)),
            ],
        )


class TypedRsidCorrectionTests(unittest.TestCase):
    def test_a_typed_rsid_amend_replaces_a_position_only_target(self) -> None:
        # Spec r1 S5: the answer's instruction names no rsID; its typed target does.
        from types import SimpleNamespace

        from bio_neb.investigation.investigation import (
            AmendInvestigationFindingResponse,
        )
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )
        from tests.sequential_investigation_support import AmendedLookupModel

        investigation_input, assertion = _assertion(_position("chr99", 10, "A", "C"))
        sources = SourceResponsesOnly(refsnp_bodies={"rs429358": RS429358})
        model = AmendedLookupModel()
        with tempfile.TemporaryDirectory() as directory:
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            review = pending.pending_request
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=review.request_event_id,
                    response=AmendInvestigationFindingResponse(
                        action="amend",
                        collection_member_id=review.question.collection_member_id,
                        revision_number=review.question.revision_number,
                        instruction="Use the variant the paper's Table 2 names.",
                        asserted_by="researcher",
                        target=StatedTarget(rsid="rs429358"),
                    ),
                )
            )
            corrected = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model.model,
                source_adapter_factory=lambda: sources,
            )

        self.assertIn("fetch_refsnp:rs429358", sources.operations)
        self.assertEqual(
            _targets(corrected), [("NC_000019.10", 44908684, "T", "C", ("rs429358",))]
        )


class LookupDecisionFactTests(unittest.TestCase):
    """Each decision a lookup asks for is one fact, written by its own Tool through
    the one constructor, citing the result that asked; the next lookup reads it and
    nothing re-sends it (owner, #476 alignment G2.2 B; #479 Q-b: four Tools)."""

    @run_taxon(9606)
    def test_each_lookup_decision_is_one_fact_its_next_lookup_reads(self):
        from bio_neb.investigation.investigation import DECISION_ASKS

        version = StatedVersionTests()
        placed = InRunPlacementTests()
        words = "Also check chr19:45411941 T>C on hg19."
        scenarios = {
            # Two sequences named, and GRCh37's listing quoted as listing none.
            "sequence and listed none": (
                _position("chr19", 44908684, "T", "C"),
                SourceResponsesOnly(
                    refsnp_bodies={"rs429358": RS429358},
                    reference_bases={"NC_000019.10:44908683:1": "T"},
                    listed_rsids={"NC_000019.10:44908683:T:C": (429358,)},
                ),
                SequentialAdaptiveModel(),
            ),
            # A stated version named from NCBI's genome reports.
            "version": (
                InvestigationRequestItem(
                    request_text=words,
                    position=StatedPosition(
                        contig="chr19", position=45411941, ref="T", alt="C"
                    ),
                    stated_assembly=StatedAssembly(name="hg19", cited_line=words),
                ),
                placed._sources(),
                SequentialAdaptiveModel(),
            ),
            # One of a record's two placements on the Run's assembly named.
            "placement": (
                version._paper_g(),
                version._grch37_sources(),
                SequentialAdaptiveModel(placements={"rs429358": "NT_187636.1"}),
            ),
        }
        scenarios["placement"][1].refsnp_bodies = {
            "rs429358": version._multiallelic(extra=True)
        }
        kinds = set()
        for name, (request, sources, model) in scenarios.items():
            with self.subTest(scenario=name):
                investigation_input, assertion = _assertion(request)
                with tempfile.TemporaryDirectory() as directory:
                    _result, journey = run_attempt(
                        Path(directory),
                        model,
                        sources,
                        investigation_input=investigation_input,
                        assertion=assertion,
                    )
                    events = read_evidence(journey)
                by_id = {event["event_id"]: event for event in events}
                validations = {
                    event["payload"]["tool_call_id"]: event["payload"]
                    for event in events
                    if event["event_type"] == "ToolCallValidated"
                }
                decided = [
                    e for e in events if e["event_type"] == "InvestigationLookupDecided"
                ]
                self.assertTrue(decided)
                lookups = [
                    e for e in events if e["event_type"] == "RefSnpToolResultRecorded"
                ]
                for event in decided:
                    payload = event["payload"]
                    kind = payload["decision"]["kind"]
                    kinds.add(kind)
                    # Its own Tool's admitted call, citing the result that asked.
                    call = validations[payload["tool_call_id"]]
                    self.assertEqual(call["errors"], [])
                    self.assertEqual(
                        call["tool_name"],
                        {
                            "sequence": "name_sequence",
                            "version": "name_version",
                            "placement": "name_placement",
                            "listed_none": "quote_listed_none",
                        }[kind],
                    )
                    asked = by_id[payload["refsnp_event_id"]]
                    self.assertEqual(
                        asked["payload"]["result"]["status"], DECISION_ASKS[kind]
                    )
                    # The next lookup reads it; its call sends none of it.
                    (reader,) = [
                        lookup
                        for lookup in lookups
                        if event["event_id"]
                        in lookup["payload"].get("decision_event_ids", ())
                        and events.index(lookup) > events.index(event)
                    ][:1]
                    sent = validations[reader["payload"]["tool_call_id"]]["arguments"]
                    self.assertFalse(
                        {"sequences", "version", "placement", "listed_none"} & set(sent)
                    )
        self.assertEqual(kinds, {"sequence", "version", "placement", "listed_none"})

    def test_a_dbsnp_rsid_lookup_reads_its_named_placement(self):
        """Review r1 SP1: a placement named for a fetch_refsnp result is read by the
        next fetch_refsnp of that rsID, which resolves on it."""
        sources = SourceResponsesOnly(
            refsnp_bodies={"rs429358": StatedVersionTests()._multiallelic(extra=True)}
        )
        model = SequentialAdaptiveModel(placements={"rs429358": "NT_187636.1"})
        with tempfile.TemporaryDirectory() as directory:
            _result, journey = run_attempt(Path(directory), model, sources)
            events = read_evidence(journey)

        (decided,) = [
            e for e in events if e["event_type"] == "InvestigationLookupDecided"
        ]
        lookups = [e for e in events if e["event_type"] == "RefSnpToolResultRecorded"]
        self.assertEqual(
            lookups[-1]["payload"]["decision_event_ids"], [decided["event_id"]]
        )
        self.assertEqual(
            lookups[-1]["payload"]["result"]["named_placement"], "NT_187636.1"
        )

    def test_reopen_refuses_a_decision_moved_to_another_call(self):
        """Review r1 ST2: a decision names its own admitted call (ADR 0034)."""
        import copy

        from bio_neb.investigation.investigation import project_investigation

        investigation_input, assertion = _assertion(
            _position("chr19", 44908684, "T", "C")
        )
        sources = SourceResponsesOnly(
            refsnp_bodies={"rs429358": RS429358},
            reference_bases={"NC_000019.10:44908683:1": "T"},
            listed_rsids={"NC_000019.10:44908683:T:C": (429358,)},
        )
        with tempfile.TemporaryDirectory() as directory:
            _result, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        index = next(
            i
            for i, e in enumerate(events)
            if e["event_type"] == "InvestigationLookupDecided"
        )
        hostile = copy.deepcopy(events)
        hostile[index]["payload"]["tool_call_id"] = "another-calls-id"
        with self.assertRaisesRegex(ValueError, "lookup decision changed its call"):
            project_investigation(hostile)


if __name__ == "__main__":
    unittest.main()


class InRunPlacementTests(unittest.TestCase):
    """A position the researcher states on another version (here typed in the prompt)
    is moved onto the Run's assembly inside its lookup and becomes the Run's Work item
    (#462 R2, owner 2026-10-26: the moved position goes back through the pipeline as an
    amendment does). #393's development GETs answer NCBI's listing, GRCh37's sequence
    report and the SPDI's equivalents."""

    def _sources(self):
        return grch37_move_sources()

    @run_taxon(9606)
    def test_an_hg19_prompt_position_becomes_the_runs_work_item(self):
        words = "Also check chr19:45411941 T>C on hg19."
        request = InvestigationRequestItem(
            request_text=words,
            position=StatedPosition(
                contig="chr19", position=45411941, ref="T", alt="C"
            ),
            stated_assembly=StatedAssembly(name="hg19", cited_line=words),
        )
        sources = self._sources()

        result = _plan(request, sources)

        self.assertEqual(
            sources.operations[:8],
            [
                # The stated version is the model's to name first: the first call ends
                # there, and the model searches NCBI's genome reports; then the
                # version's sequence is its to name, and the third call moves the
                # position.
                "fetch_assembly_report:GCF_000001405.40",
                "ncbi_taxon_assemblies:9606?filters.assembly_version=all_assemblies"
                "&filters.search_text=hg19&page_size=100",
                "fetch_assembly_report:GCF_000001405.40",
                "ncbi_sequence_names:GCF_000001405.25",
                "fetch_assembly_report:GCF_000001405.40",
                "ncbi_sequence_names:GCF_000001405.25",
                "ncbi_spdi_equivalents:NC_000019.9:45411940:T:C",
                "ncbi_sequence_names:GCF_000001405.40",
            ],
        )
        self.assertEqual(
            _targets(result), [("NC_000019.10", 44908684, "T", "C", ("rs429358",))]
        )
        # Its row keeps where the lookup placed it, beside the position and version
        # as stated, so the readable report can say so.
        from bio_neb.investigation.collection_view import (
            LookupPlacement,
            collection_view,
        )
        from bio_neb.investigation.investigation_request import PlacedPosition

        (snp,) = collection_view(result.pending_request.proposal).snps
        self.assertEqual(
            snp.lookup.placement,
            LookupPlacement(
                stated=request.position,
                stated_version="hg19",
                placed=PlacedPosition(
                    contig="NC_000019.10", position=44908684, ref="T", alt="C"
                ),
            ),
        )

    @run_taxon(9606)
    def test_a_papers_hg19_target_is_placed_by_the_same_flow_and_stays_the_papers(
        self,
    ):
        """#485 Q1 a (owner 2026-10-26, one flow whatever the source): a target a
        paper's reading proposed, beside the researcher's request, reaches the
        Investigation in the paper's words with the version line it cites, and the
        same lookup moves it onto the Run's assembly; it stays the paper's. A
        derived input: the hg19 case above, its words and version line a paper's."""
        line = "Carriers of chr19:45411941 T>C (hg19) were enriched among cases."
        paper = "paper:doi:10.5555/fictional.485"
        request = InvestigationRequestItem(
            request_text=line,
            position=StatedPosition(
                contig="chr19", position=45411941, ref="T", alt="C"
            ),
            stated_assembly=StatedAssembly(name="hg19", cited_line=line),
            asserted_by=paper,
        )
        investigation_input, _ = input_and_assertion()
        assertion = UserAssertion.create_collection(
            asserted_by="researcher:owner",
            requests=(request,),
            predecessor_ids=(investigation_input.input_id,),
        )
        sources = self._sources()
        with tempfile.TemporaryDirectory() as directory:
            result, _ = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )

        self.assertIn(
            "ncbi_spdi_equivalents:NC_000019.9:45411940:T:C", sources.operations
        )
        self.assertEqual(
            _targets(result), [("NC_000019.10", 44908684, "T", "C", ("rs429358",))]
        )
        from bio_neb.investigation.collection_view import collection_view

        (snp,) = collection_view(result.pending_request.proposal).snps
        self.assertEqual(
            (snp.request_text, snp.asserted_by, snp.lookup.placement.stated_version),
            (line, paper, "hg19"),
        )


class ModelNamedSequenceTests(unittest.TestCase):
    """Through the Investigation Agent (#467, owner D1 a, RC5 1a): a written contig
    that is not literally a name of its sequence is named by the model after a
    sequence_unnamed result, from the assembly's searched sequence names, checked by
    membership, and marked model-matched on the result the proposal cites."""

    def _run(self, request, judgments):
        investigation_input, assertion = _assertion(request)
        directory = self.enterContext(tempfile.TemporaryDirectory())
        sources = SourceResponsesOnly(
            refsnp_bodies={"rs429358": RS429358},
            reference_bases={"NC_000019.10:44908683:1": "T"},
            listed_rsids={
                "NC_000019.10:44908683:T:C": (429358,),
                "NC_000019.9:44908683:T:C": (),
            },
        )
        result, journey = run_attempt(
            Path(directory),
            SequentialAdaptiveModel(judgments),
            sources,
            investigation_input=investigation_input,
            assertion=assertion,
        )
        from bio_neb.recording import read_evidence

        return result, read_evidence(journey)

    def test_a_contig_written_as_chromosome_19_becomes_its_runs_work_item(self):
        result, events = self._run(
            _position(
                "chromosome 19",
                44908684,
                "T",
                "C",
                text="The paper reports chromosome 19 at 44908684 T>C.",
            ),
            # The scripted model judges the written contig means molecule 19.
            {"chromosome 19": "19"},
        )

        lookups = [
            e["payload"]["result"]
            for e in events
            if e["event_type"] == "RefSnpToolResultRecorded"
        ]
        self.assertEqual(lookups[0]["status"], "sequence_unnamed")
        searched = [
            e["payload"]["result"]
            for e in events
            if e["event_type"] == "SequenceNamesRead"
        ]
        self.assertEqual(searched[0]["assembly_accession"], "GCF_000001405.40")
        namings = lookups[-1]["location"]["sequence_namings"]
        self.assertEqual(
            [(n["assembly_accession"], n["record"], n["match"]) for n in namings],
            [
                ("GCF_000001405.40", "NC_000019.10", "model_matched"),
                ("GCF_000001405.25", "NC_000019.9", "model_matched"),
            ],
        )
        self.assertEqual(
            _targets(result), [("NC_000019.10", 44908684, "T", "C", ("rs429358",))]
        )

    def test_a_search_before_any_lookup_read_the_names_is_refused(self):
        """Only names a lookup of this member retained are searched (#467): no GET."""
        from types import SimpleNamespace

        from bio_neb.investigation.investigation_agent import (
            investigation_action_errors,
        )

        refused = investigation_action_errors(
            SimpleNamespace(
                investigation_input=SimpleNamespace(
                    assembly_accession="GCF_000001405.40"
                ),
                source_events=[],
                request=_position("chr19", 44908684, "T", "C"),
                researcher_answers=(),
                raw_answers=(),
                target_reads=[],
                started_event_id="started",
                sequence_reports={},
            ),
            SimpleNamespace(
                tool_name="search_sequence_names",
                arguments={"assembly_accession": "GCF_000001405.40"},
            ),
        )

        self.assertEqual(refused, ("sequence_names_unread",))
