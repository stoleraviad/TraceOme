"""The rsID identifier rule and the researcher request document that applies it."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from bio_neb.investigation.investigation import (
    UserAssertion,
    investigation_output_member_id,
)
from bio_neb.investigation.investigation_request import (
    RSID_RULE,
    InvestigationRequestItem,
    InvestigationRequestRefused,
    StatedPosition,
    read_investigation_request,
)
from tests.run_prerequisites import RunPrerequisites
from tests.sequential_investigation_support import begin_in_pass

_MEMBER_ID_AT_4A86EC4D = (
    "request-member:8d5c7d296984e14505b6eff00b14685f50fa91266c217f09208c64ea7eb009d2"
)


class RsidRuleTests(unittest.TestCase):
    def test_a_typed_rsid_matches_any_spelling_its_quote_uses(self) -> None:
        for quote in ("rs1234", "RS1234", "rsID1234", "RsId1234", "rs 1234"):
            with self.subTest(quote=quote):
                self.assertTrue(RSID_RULE.matches("rs1234", quote))
        # No source is named: the Investigation's model chooses dbSNP or EVA.
        self.assertIsNone(RSID_RULE.source)

    def test_a_different_or_untyped_rsid_does_not_match(self) -> None:
        for value, quote in (
            ("rs1234", "RS4321"),
            ("RS1234", "RS1234"),
            ("rs01234", "rs01234"),
            ("rs1234", "snp1234"),
            ("rs1234", "1234"),
        ):
            with self.subTest(value=value, quote=quote):
                self.assertFalse(RSID_RULE.matches(value, quote))

    def test_prose_states_whole_tokens_only(self) -> None:
        text = "Compare RS429358 with rsID7412, not xrs99 or rs42935810x."
        self.assertEqual(RSID_RULE.stated(text), ("rs429358", "rs7412"))
        self.assertNotIn("rs42935", RSID_RULE.stated("observe rs429358 here"))
        self.assertNotIn("rs1234", RSID_RULE.stated("RS4321 only"))


class StatedPositionTests(unittest.TestCase):
    """A request's position target is typed (#369); prose never states one."""

    def _read(self, document: dict) -> tuple:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "request.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return read_investigation_request(str(path)).items

    def test_a_position_target_is_read_with_its_alleles_and_stated_assembly(self):
        (item,) = self._read(
            {
                "schema_version": "bio-neb.investigation-request/2",
                "asserted_by": "paper:doi:10.1000/s3",
                "requests": [
                    {
                        "request": "chr11: 207,469:A>C is the causal change.",
                        "target": {
                            "position": {
                                "contig": "chr11",
                                "position": 207469,
                                "ref": "A",
                                "alt": "C",
                            }
                        },
                        "stated_assembly": {
                            "name": "sacCer1",
                            "cited_line": "Reads were aligned to the UCSC sacCer1 "
                            "reference sequence",
                        },
                    }
                ],
            }
        )
        self.assertIsNone(item.rsid)
        self.assertEqual(
            (item.position.contig, item.position.position),
            ("chr11", 207469),
        )
        self.assertEqual(item.position.identity, ("chr11", 207469, "A", "C"))
        self.assertEqual(item.stated_assembly.name, "sacCer1")

    def test_an_rsid_target_may_also_state_where_the_paper_places_it(self):
        (item,) = self._read(
            {
                "schema_version": "bio-neb.investigation-request/2",
                "asserted_by": "paper:doi:10.1000/h1",
                "requests": [
                    {
                        "request": "chr7:150951552 carries rs199472944.",
                        "target": {
                            "rsid": "rs199472944",
                            "position": {"contig": "chr7", "position": 150951552},
                        },
                    }
                ],
            }
        )
        self.assertEqual(item.rsid, "rs199472944")
        self.assertEqual(item.position.identity, None)
        self.assertIsNone(item.stated_assembly)

    def test_an_allele_is_held_trimmed_of_its_shared_bases(self):
        # The normal form: an indel keeps no context base, an empty side is "".
        for ref, alt, expected in (
            ("CTGT", "C", ("NC_000001.11", 14398, "TGT", "")),
            ("A", "AC", ("NC_000001.11", 14398, "", "C")),
            ("TC", "GA", ("NC_000001.11", 14397, "TC", "GA")),
        ):
            with self.subTest(ref=ref, alt=alt):
                position = StatedPosition(
                    contig="NC_000001.11", position=14397, ref=ref, alt=alt
                )
                self.assertEqual(position.identity, expected)

    def test_a_position_states_both_alleles_or_neither(self):
        for alleles in ({"ref": "A"}, {"alt": "C"}, {"ref": "A", "alt": "A"}):
            with self.subTest(alleles=alleles), self.assertRaises(ValidationError):
                StatedPosition(contig="chr1", position=10, **alleles)

    def test_a_position_written_in_the_request_text_is_not_a_target(self):
        item = InvestigationRequestItem(
            request_text="Retry observe_variant; exact allele "
            "NC_000012.12:21178615 T>C."
        )
        self.assertIsNone(item.position)

    def test_the_first_request_file_version_is_retired(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "request.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "bio-neb.investigation-request/1",
                        "asserted_by": "researcher",
                        "requests": [
                            {"request": "Observe rs429358.", "rsid": "rs429358"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(InvestigationRequestRefused) as raised:
                read_investigation_request(str(path))
        self.assertEqual(raised.exception.code, "investigation_request_invalid")

    def test_an_rsid_member_keeps_its_identity(self):
        # The member digest at 4a86ec4d, before position targets existed: an rsID
        # target's identity does not move when the item gains optional fields.
        item = InvestigationRequestItem(
            request_text="Observe rs429358 in sample HG001.", rsid="rs429358"
        )
        self.assertEqual(
            investigation_output_member_id(item, 0),
            _MEMBER_ID_AT_4A86EC4D,
        )


_V3 = "bio-neb.investigation-request/3"
_GRCH38_LINE = "Reads were aligned to hg38 with bwa mem."


class RequestFileVersionThreeTests(unittest.TestCase):
    """A ``/3`` request (#367): each item's typed placement, region and gene-and-change
    targets, with ``/2`` files still read."""

    def _read(self, document: dict):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "request.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return read_investigation_request(str(path))

    def _refusal(self, document: dict) -> str:
        with self.assertRaises(InvestigationRequestRefused) as raised:
            self._read(document)
        return raised.exception.code

    def _moved(self, **placement):
        return {
            "request": "chr11: 207,469:A>C (placement: moved)",
            "target": {
                "position": {
                    "contig": "chr11",
                    "position": 207469,
                    "ref": "A",
                    "alt": "C",
                }
            },
            "stated_assembly": {"name": "sacCer1", "cited_line": "aligned to sacCer1"},
            "placement": {
                "step": "table",
                "matched_name": "sacCer1",
                "position": {
                    "contig": "NC_001143.9",
                    "position": 207480,
                    "ref": "A",
                    "alt": "C",
                },
                "reason": "moved from sacCer1 chr11:207469 by a flank move",
            }
            | placement,
        }

    def test_a_moved_target_keeps_its_stated_position_and_its_placed_one(self):
        request = self._read(
            {
                "schema_version": _V3,
                "asserted_by": "paper:doi:x",
                "requests": [self._moved()],
            }
        )

        [item] = request.items
        self.assertEqual(
            (item.position.contig, item.position.position, item.stated_assembly.name),
            ("chr11", 207469, "sacCer1"),
        )
        self.assertEqual(
            (
                item.placement.step,
                item.placement.matched_name,
                item.placed.contig,
                item.placed.position,
            ),
            ("table", "sacCer1", "NC_001143.9", 207480),
        )

    def test_a_kept_position_is_the_stated_one(self):
        kept = self._moved(
            step="same_version",
            position={"contig": "chr11", "position": 207469, "ref": "A", "alt": "C"},
        )
        moved_as_kept = self._moved(step="same_version")

        [item] = self._read(
            {"schema_version": _V3, "asserted_by": "p", "requests": [kept]}
        ).items
        self.assertEqual(item.placed.model_dump(), item.position.model_dump())
        self.assertEqual(
            self._refusal(
                {"schema_version": _V3, "asserted_by": "p", "requests": [moved_as_kept]}
            ),
            "investigation_request_invalid",
        )

    def test_an_unplaced_target_has_no_placed_position(self):
        unplaced = self._moved(step="none", reason="no step placed it")
        unplaced["placement"].pop("position")
        unplaced["placement"].pop("matched_name")

        [item] = self._read(
            {"schema_version": _V3, "asserted_by": "p", "requests": [unplaced]}
        ).items

        self.assertEqual((item.placement.step, item.placed), ("none", None))
        self.assertEqual(
            self._refusal(
                {
                    "schema_version": _V3,
                    "asserted_by": "p",
                    "requests": [self._moved(step="none")],
                }
            ),
            "investigation_request_invalid",
        )

    def test_a_position_step_places_a_stated_position_on_its_version(self):
        unstated = self._moved()
        unstated.pop("stated_assembly")

        self.assertEqual(
            self._refusal(
                {"schema_version": _V3, "asserted_by": "p", "requests": [unstated]}
            ),
            "investigation_request_invalid",
        )

    def test_a_gene_and_change_is_placed_by_g15_with_no_stated_position(self):
        item = {
            "request": "point mutations at PBP2 (Y127X) (placement: placed)",
            "target": {"gene_change": {"gene": "PBP2", "change": "Y127X"}},
            "placement": {
                "step": "gene_change",
                "position": {
                    "contig": "NC_001134.8",
                    "position": 1006,
                    "ref": "C",
                    "alt": "A",
                },
                "reason": "candidate 1 of 2",
            },
        }

        [read] = self._read(
            {"schema_version": _V3, "asserted_by": "p", "requests": [item]}
        ).items

        self.assertEqual(
            (read.position, read.gene_change.gene, read.placed.position),
            (None, "PBP2", 1006),
        )

    def test_a_region_is_its_own_target_and_question(self):
        region = {
            "request": "every record in NC_012920.1:300-312",
            "target": {"region": {"contig": "NC_012920.1", "start": 300, "end": 312}},
            "normalized_question_action": "observe_region",
        }

        [item] = self._read(
            {"schema_version": _V3, "asserted_by": "researcher", "requests": [region]}
        ).items

        self.assertEqual(
            (item.region.contig, item.region.start, item.region.end),
            ("NC_012920.1", 300, 312),
        )
        self.assertEqual(item.normalized_question_action, "observe_region")
        for broken in (
            region | {"normalized_question_action": "observe_variant"},
            region | {"target": {"region": {"contig": "MT", "start": 9, "end": 3}}},
            region
            | {
                "request": "rs1 in NC_012920.1:1-3",
                "target": {
                    "rsid": "rs1",
                    "region": {"contig": "MT", "start": 1, "end": 3},
                },
            },
        ):
            with self.subTest(broken=broken):
                self.assertEqual(
                    self._refusal(
                        {
                            "schema_version": _V3,
                            "asserted_by": "r",
                            "requests": [broken],
                        }
                    ),
                    "investigation_request_invalid",
                )

    def test_an_answer_states_an_rsid_a_position_or_a_typed_gene_change(self):
        """A researcher's typed answer states no region, and its gene change is typed:
        code reads no change text (#473 owner Q1 a, Q2 a; ADR 0049)."""
        from bio_neb.investigation.investigation import ClarifyInvestigationResponse

        answer = {
            "action": "clarify",
            "collection_member_id": "member",
            "revision_number": 1,
            "instruction": "check this",
            "asserted_by": "researcher",
        }
        ClarifyInvestigationResponse.model_validate(
            answer | {"target": {"rsid": "rs1"}}
        )
        typed = {
            "gene": "PBP2",
            "change": "Y2X",
            "protein": {"ref": "Y", "codon": 2, "alt": "*"},
        }
        ClarifyInvestigationResponse.model_validate(
            answer | {"target": {"gene_change": typed}}
        )
        for target in (
            {"region": {"contig": "MT", "start": 1, "end": 3}},
            {"gene_change": {"gene": "PBP2", "change": "Y2X"}},
        ):
            with self.subTest(target=target), self.assertRaises(ValidationError):
                ClarifyInvestigationResponse.model_validate(answer | {"target": target})

    def test_a_version_two_file_still_reads_and_states_no_placement(self):
        position = {"contig": "chr7", "position": 150951552}
        earlier = {
            "schema_version": "bio-neb.investigation-request/2",
            "asserted_by": "researcher",
            "requests": [
                {
                    "request": "chr7:150951552 rs199472944",
                    "target": {"rsid": "rs199472944", "position": position},
                }
            ],
        }

        [item] = self._read(earlier).items

        self.assertEqual(
            (item.rsid, item.position.position, item.placement),
            ("rs199472944", 150951552, None),
        )
        self.assertEqual(
            self._refusal(earlier | {"requests": [self._moved()]}),
            "investigation_request_invalid",
        )


class RequestDocumentRsidTests(unittest.TestCase):
    def test_request_text_may_spell_the_rsid_differently(self) -> None:
        item = InvestigationRequestItem(
            request_text="Observe RS429358 in sample HG001.", rsid="rs429358"
        )
        self.assertEqual(item.rsid, "rs429358")

    def test_document_rsid_is_typed_in_its_one_spelling(self) -> None:
        with self.assertRaises(ValidationError):
            InvestigationRequestItem(
                request_text="Observe rs429358 in sample HG001.", rsid="rsID429358"
            )

    def test_a_prefix_or_a_different_rsid_is_not_adjacent(self) -> None:
        for request_text, rsid in (
            ("Observe rs429358 in sample HG001.", "rs42935"),
            ("Observe RS4321 in sample HG001.", "rs1234"),
        ):
            with self.subTest(rsid=rsid), self.assertRaises(ValidationError) as raised:
                InvestigationRequestItem(request_text=request_text, rsid=rsid)
            self.assertEqual(
                raised.exception.errors()[0]["type"],
                "investigation_request_item_missing_adjacent_rsid",
            )

    def test_reader_refuses_a_prefix_rsid_with_its_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "request.json"
            path.write_text(
                json.dumps(
                    {
                        "asserted_by": "researcher",
                        "request": "Observe rs429358 in sample HG001.",
                        "target": {"rsid": "rs42935"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(InvestigationRequestRefused) as raised:
                read_investigation_request(str(path))
        self.assertEqual(
            raised.exception.code, "investigation_request_missing_adjacent_rsid"
        )
        self.assertEqual(raised.exception.fields, ("request_file.requests[0].request",))

    def test_user_assertion_applies_the_same_rule(self) -> None:
        accepted = UserAssertion.create(
            asserted_by="researcher",
            request_text="Observe RS429358 in sample HG001.",
            rsid="rs429358",
            predecessor_ids=("event",),
        )
        self.assertEqual(accepted.rsid, "rs429358")
        with self.assertRaises(ValidationError):
            UserAssertion.create(
                asserted_by="researcher",
                request_text="Observe rs429358 in sample HG001.",
                rsid="rs42935",
                predecessor_ids=("event",),
            )


class InvestigationBeginPassTests(RunPrerequisites, unittest.TestCase):
    def test_begin_outside_a_processing_pass_refuses_before_any_write(self):
        from bio_neb.investigation.investigation_commands import (
            BeginInvestigationCommand,
            InvestigationApplicationRefused,
            begin_investigation,
        )

        def no_contact(*_args):
            raise AssertionError("a refused begin constructs no model or source")

        parent = self.prerequisite("executed").journey_file
        before = parent.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            request.write_text(
                json.dumps(
                    {
                        "asserted_by": "researcher:owner",
                        "request": "Investigate rs429358 in sample-001.",
                        "target": {"rsid": "rs429358"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(InvestigationApplicationRefused) as refused:
                begin_investigation(
                    BeginInvestigationCommand(
                        journey_file=str(parent), request_files=(str(request),)
                    ),
                    environ={},
                    model_factory=no_contact,
                    source_adapter_factory=no_contact,
                )
        self.assertEqual(refused.exception.code, "investigation_begin_outside_pass")
        self.assertEqual(parent.read_bytes(), before)


class InvestigationRequestEntryTests(unittest.TestCase):
    def test_later_begin_replay_preserves_initial_request_after_scoped_revision(self):
        from bio_neb.investigation.investigation import (
            AmendInvestigationMember,
            ReviseInvestigationResponse,
        )
        from bio_neb.investigation.investigation_commands import (
            BeginInvestigationCommand,
            InvestigationApplicationRefused,
            RespondInvestigationReviewCommand,
            respond_investigation_review,
        )
        from tests.sequential_investigation_support import (
            SequentialAdaptiveModel,
            SequentialSourceResponses,
            forked_child,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            original = {
                "asserted_by": "researcher:owner",
                "request": "Investigate rs429358 in sample HG001.",
                "target": {"rsid": "rs429358"},
            }
            journey = forked_child(root, investigation_request=original)
            request = root / "request.json"
            request.write_text(json.dumps(original), encoding="utf-8")
            command = BeginInvestigationCommand(
                journey_file=str(journey.path),
                request_files=(str(request),),
            )
            model = SequentialAdaptiveModel()
            sources = SequentialSourceResponses()
            initial = begin_in_pass(
                command,
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model.model,
                source_adapter_factory=lambda: sources,
            )
            pending = initial.pending_request
            revised = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.request_event_id,
                    response=ReviseInvestigationResponse(
                        action="revise",
                        proposal_id=pending.proposal.proposal_id,
                        asserted_by="researcher:owner",
                        operations=(
                            AmendInvestigationMember(
                                operation="amend",
                                work_item_id=pending.proposal.work_items[
                                    0
                                ].work_item_id,
                                request=InvestigationRequestItem(
                                    request_text="Recheck rs429358 in sample HG001.",
                                    rsid="rs429358",
                                ),
                            ),
                        ),
                    ),
                )
            )
            before = journey.path.read_bytes()

            def forbidden():
                self.fail("replay must not construct an Agent or source adapter")

            # Document formatting is not scientific request identity.
            request.write_text(json.dumps(original, indent=2), encoding="utf-8")
            self.assertEqual(
                begin_in_pass(
                    command,
                    environ={},
                    model_factory=lambda *_: forbidden(),
                    source_adapter_factory=forbidden,
                ),
                revised,
            )
            self.assertEqual(
                begin_in_pass(
                    command.model_copy(update={"request_file": None}),
                    environ={},
                    model_factory=lambda *_: forbidden(),
                    source_adapter_factory=forbidden,
                ),
                revised,
            )
            self.assertEqual(journey.path.read_bytes(), before)

            request.write_text(
                json.dumps({**original, "request": "Recheck rs429358."}),
                encoding="utf-8",
            )
            with self.assertRaises(InvestigationApplicationRefused) as conflict:
                begin_in_pass(command, environ={})
            self.assertEqual(conflict.exception.code, "investigation_request_conflict")
            request.write_text(json.dumps(original), encoding="utf-8")
            self.assertEqual(journey.path.read_bytes(), before)

    def test_fresh_remaining_scope_can_stop_without_erasing_prior_approval(self):
        from bio_neb.application.queries import read_investigation
        from bio_neb.investigation.investigation import StopInvestigationResponse
        from bio_neb.investigation.investigation_commands import (
            RespondInvestigationReviewCommand,
            respond_investigation_review,
        )
        from bio_neb.journey import Journey
        from tests.sequential_investigation_support import (
            approved_collection_observation,
        )

        with tempfile.TemporaryDirectory() as directory:
            _, journey, authority, _, _ = approved_collection_observation(
                Path(directory).resolve(),
                cardinality=2,
                approve_count=1,
            )
            current = read_investigation(Journey.open(journey.path)).investigation
            request = current.pending_request
            stopped = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=request.request_event_id,
                    response=StopInvestigationResponse(
                        action="stop",
                        proposal_id=request.proposal.proposal_id,
                        reason="Stop the unapproved remaining item.",
                    ),
                )
            )
            self.assertEqual(stopped.status, "stopped")
            self.assertIn(authority, stopped.authorities)
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, stopped
            )


class PlannedRequestOwnershipTests(unittest.TestCase):
    """Planning owns the target list; a registered request file is planning's to
    place, never a list that overrides the plan unseen (#342, ADR 0010)."""

    REQUEST = {
        "schema_version": "bio-neb.investigation-request/2",
        "asserted_by": "researcher:owner",
        "requests": [
            {"request": "Observe rs7412 in HG001.", "target": {"rsid": "rs7412"}},
            {"request": "Observe rs429358 in HG001.", "target": {"rsid": "rs429358"}},
        ],
    }

    def _planned(self, root, request, **choices):
        """Register the request file, record the scripted plan, stop at a Review."""
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_composition,
        )
        from tests.planning_assignment_fakes import configured_planning_run

        receipt, composition, _ = configured_planning_run(
            root, investigation_request=request, **choices
        )
        with use_run_advancement_composition(composition):
            stopped = advance(ResumeRun(journey_file=receipt.journey_file))
        return receipt, composition, stopped

    PAPER = {
        "schema_version": "bio-neb.investigation-request/3",
        "asserted_by": "paper:doi:10.1000/x",
        "requests": [
            {"request": "rs429358 in the paper", "target": {"rsid": "rs429358"}},
            {"request": "rs4149056 in the paper", "target": {"rsid": "rs4149056"}},
        ],
    }

    def test_two_registered_files_are_each_cited_and_keep_their_asserters(self):
        # A researcher's list beside a paper's /3. Planning cites each file by its
        # own item; the targets keep the files' order, then the prompt's, each once,
        # and each its own asserter.
        from bio_neb.investigation.investigation_request import (
            current_investigation_request,
        )
        from bio_neb.journey import Journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, _, stopped = self._planned(
                Path(directory),
                [self.REQUEST, self.PAPER],
                scientific_intent="Also check rs1801133.",
                scientific_rsid="rs1801133",
            )
            journey = Journey.open(Path(receipt.journey_file))
            files = journey.checked_view("investigation_request")
            planned = current_investigation_request(journey)

        handoff = stopped.requirement_resolution.scientific_investigation
        self.assertEqual(
            [
                i.registered_request_event_id
                for i in handoff.items
                if i.registered_request_event_id
            ],
            [file.source_event_id for file in files],
        )
        self.assertEqual(planned.request.asserted_by, "researcher:owner")
        self.assertEqual(
            [(item.rsid, item.asserted_by) for item in planned.request.items],
            [
                ("rs7412", None),
                ("rs429358", None),
                ("rs4149056", "paper:doi:10.1000/x"),
                ("rs1801133", "researcher"),
            ],
        )

    def test_planning_sees_the_registered_request_and_must_cite_it(self):
        from bio_neb.agentic_journey.assignment_context import (
            current_assignment_context,
        )
        from bio_neb.journey import Journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, _, stopped = self._planned(Path(directory), self.REQUEST)
            journey = Journey.open(Path(receipt.journey_file))
            (registered,) = journey.checked_view("investigation_request")
            context = current_assignment_context(journey, "plan")

        # The planner's context names each file and its size, not every item.
        self.assertEqual(
            context["registered_requests"],
            [
                {
                    "event_id": registered.source_event_id,
                    "request_file": registered.request_file,
                    "asserted_by": "researcher:owner",
                    "target_count": 2,
                }
            ],
        )
        (item,) = stopped.requirement_resolution.scientific_investigation.items
        self.assertEqual(
            (item.registered_request_event_id, item.targets),
            (registered.source_event_id, ()),
        )
        self.assertEqual(stopped.requirement_resolution.unsettled_targets, ())

    def test_a_cited_request_file_is_the_target_list_in_its_own_order(self):
        from bio_neb.investigation.investigation_request import (
            current_investigation_request,
        )
        from bio_neb.journey import Journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, _, _ = self._planned(Path(directory), self.REQUEST)
            planned = current_investigation_request(
                Journey.open(Path(receipt.journey_file))
            )

        self.assertEqual(planned.request.asserted_by, "researcher:owner")
        self.assertEqual(
            [(item.request_text, item.rsid) for item in planned.request.items],
            [
                ("Observe rs7412 in HG001.", "rs7412"),
                ("Observe rs429358 in HG001.", "rs429358"),
            ],
        )
        self.assertEqual(planned.research_goals, ("rs1234",))

    def test_a_prompt_target_the_file_does_not_list_is_checked_without_a_question(self):
        # The request also names rs4149056, which the file does not list. The
        # researcher asked for it, so no planning Review asks again; it follows the
        # file's targets in the planned request.
        from bio_neb.investigation.investigation_request import (
            current_investigation_request,
        )
        from bio_neb.journey import Journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, _, stopped = self._planned(
                Path(directory),
                self.REQUEST,
                scientific_intent="Also check rs4149056.",
                scientific_rsid="rs4149056",
            )
            planned = current_investigation_request(
                Journey.open(Path(receipt.journey_file))
            )

        self.assertEqual(stopped.requirement_resolution.unsettled_targets, ())
        self.assertEqual(
            [
                request.owner
                for request in stopped.planning_hitl.pending_requests
                if request.owner == "requirement_resolution"
            ],
            [],
        )
        self.assertEqual(
            [(item.request_text, item.rsid) for item in planned.request.items],
            [
                ("Observe rs7412 in HG001.", "rs7412"),
                ("Observe rs429358 in HG001.", "rs429358"),
                ("Also check rs4149056.", "rs4149056"),
            ],
        )

    def test_a_prompt_target_the_plan_leaves_untyped_is_asked_before_execution(self):
        """ADR 0049 ("code reading free text for a value" is a rule a surface may not
        set), #472 Q5 a: the plan quotes "Also check rs1801133." without typing its
        rsID; no code picks the rsID out of the quote (#367 live F3's answer_value
        fallback, retired), so the untyped target is asked before Execution (#462
        R3)."""
        with tempfile.TemporaryDirectory() as directory:
            _receipt, _, stopped = self._planned(
                Path(directory),
                [self.REQUEST, self.PAPER],
                scientific_intent="Also check rs1801133.",
                scientific_rsid=None,
            )

        self.assertEqual(
            [u.reason for u in stopped.requirement_resolution.unsettled_targets],
            ["untyped_target"],
        )

    def test_a_position_the_prompt_states_is_a_typed_planned_target(self):
        # "chr7:117559590 C>T on hg38" beside the file is one typed position (the
        # model types the site) on the version its words name (the plan names it,
        # SP4), asked about by no Review.
        from bio_neb.investigation.investigation_request import (
            StatedAssembly,
            StatedPosition,
            current_investigation_request,
            not_checked_reason,
        )
        from bio_neb.journey import Journey

        words = "Also check chr7:117559590 C>T on hg38."
        with tempfile.TemporaryDirectory() as directory:
            receipt, _, stopped = self._planned(
                Path(directory),
                self.REQUEST,
                scientific_intent=words,
                scientific_rsid=None,
                scientific_version="hg38",
                scientific_site={
                    "contig": "chr7",
                    "position": 117559590,
                    "ref": "C",
                    "alt": "T",
                },
            )
            planned = current_investigation_request(
                Journey.open(Path(receipt.journey_file))
            )

        self.assertEqual(stopped.requirement_resolution.unsettled_targets, ())
        item = planned.request.items[-1]
        self.assertEqual(
            (item.request_text, item.rsid, item.position, item.stated_assembly),
            (
                words,
                None,
                StatedPosition(contig="chr7", position=117559590, ref="C", alt="T"),
                StatedAssembly(name="hg38", cited_line=words),
            ),
        )
        self.assertIsNone(not_checked_reason(item))

    def _ask_target(self, directory):
        receipt, composition, stopped = self._planned(
            Path(directory),
            self.REQUEST,
            scientific_intent="Also check the variant near unc-22.",
            scientific_rsid=None,
        )
        question = next(
            request
            for request in stopped.planning_hitl.pending_requests
            if request.owner == "requirement_resolution"
        )
        return receipt, composition, stopped, question

    def _answer(self, receipt, composition, question, instruction):
        from bio_neb.application.advancement import (
            AnswerReview,
            PlanningReviewAnswer,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.parabricks.planning_hitl import ClarifyPlanningHitlResponse

        with use_run_advancement_composition(composition):
            return advance(
                AnswerReview(
                    journey_file=receipt.journey_file,
                    request_event_id=question.request_event_id,
                    answer=PlanningReviewAnswer(
                        kind="planning",
                        response=ClarifyPlanningHitlResponse(
                            action="clarify", instruction=instruction
                        ),
                    ),
                )
            )

    def _settle_planning_questions(self, current, composition):
        """Answer each open question the planning model asked (here the recipe
        eligibility one) with its first candidate."""
        import json

        from bio_neb.application.advancement import (
            advance,
            answer_review_from_json,
            use_run_advancement_composition,
        )

        with use_run_advancement_composition(composition):
            for _ in range(8):
                sibling = next(
                    (
                        request
                        for request in current.planning_hitl.pending_requests
                        if request.owner != "requirement_resolution"
                    ),
                    None,
                )
                if sibling is None:
                    return current
                current = advance(
                    answer_review_from_json(
                        current,
                        sibling.request_event_id,
                        json.dumps(
                            {
                                "action": "select",
                                "candidate_id": sibling.candidates[0].candidate_id,
                            }
                        ),
                    )
                )
        return current

    def test_an_answer_waits_while_the_planning_model_asks_its_own_question(self):
        # Spec r3 SP5: no answer is read while the planning model's own question is
        # open, so a failed read never opens a second Review beside it; the answer
        # is read once that question is answered.
        from bio_neb.journey import Journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, composition, _, question = self._ask_target(directory)
            waiting = self._answer(receipt, composition, question, "rs1234")
            unread = [
                e
                for e in Journey.open(Path(receipt.journey_file)).read()
                if e["event_type"] == "TargetAnswerRead"
            ]
            read = self._settle_planning_questions(waiting, composition)
            events = Journey.open(Path(receipt.journey_file)).read()

        self.assertEqual(unread, [])
        self.assertEqual(
            [u.reason for u in waiting.requirement_resolution.unsettled_targets],
            ["answer_unread"],
        )
        self.assertEqual(waiting.current_review.request.owner, "agent_attempt")
        (fact,) = [e for e in events if e["event_type"] == "TargetAnswerRead"]
        self.assertEqual(fact["payload"]["targets"][0]["rsid"], "rs1234")
        self.assertEqual(read.requirement_resolution.unsettled_targets, ())

    def test_prompt_words_that_state_no_target_are_asked_and_answered(self):
        # Words the Run cannot check open planning's target Review before
        # Execution. The answer goes back to planning (a new target_answer stage):
        # the model reads it as it reads a prompt and names its targets and the
        # version, which code does not check.
        from bio_neb.investigation.investigation_request import (
            StatedAssembly,
            StatedPosition,
            current_investigation_request,
        )
        from bio_neb.journey import Journey

        answer = "Check chr7:117559590 C>T on hg38."
        with tempfile.TemporaryDirectory() as directory:
            receipt, composition, stopped, question = self._ask_target(directory)
            self.assertIsNone(stopped.execution_hitl)
            composition.answer_targets = [
                {
                    "state": "asserted",
                    "quote": "chr7:117559590 C>T on hg38",
                    "site": {
                        "contig": "chr7",
                        "position": 117559590,
                        "ref": "C",
                        "alt": "T",
                    },
                    "stated_version": "hg38",
                }
            ]
            answered = self._settle_planning_questions(
                self._answer(receipt, composition, question, answer), composition
            )
            journey = Journey.open(Path(receipt.journey_file))
            planned = current_investigation_request(journey)
            events = journey.read()

        self.assertEqual(
            (question.issue, question.question, question.candidates),
            (
                "The request's words for this target, \"Also check the variant near "
                'unc-22.", state no rsID or position the Investigation can check.',
                "Which target do these words name? Give its rsID, or its position as "
                "sequence:position REF>ALT with the genome version it is on.",
                (),
            ),
        )
        (read,) = [e for e in events if e["event_type"] == "TargetAnswerRead"]
        started = next(
            e
            for e in events
            if e["event_type"] == "AttemptStarted"
            and events.index(e)
            > events.index(
                next(e for e in events if e["event_type"] == "PlanningHitlResponded")
            )
        )
        self.assertEqual(
            read["payload"]["attempt_id"], started["payload"]["attempt_id"]
        )
        self.assertIn("record_answered_targets", composition.calls)
        self.assertEqual(answered.requirement_resolution.unsettled_targets, ())
        item = planned.request.items[-1]
        self.assertEqual(
            (item.request_text, item.position, item.stated_assembly),
            (
                "chr7:117559590 C>T on hg38",
                StatedPosition(contig="chr7", position=117559590, ref="C", alt="T"),
                StatedAssembly(name="hg38", cited_line="chr7:117559590 C>T on hg38"),
            ),
        )

    def test_an_answer_the_model_finds_no_target_in_is_asked_again(self):
        # No code reads an answer before the model ("remove the check"), and an
        # answer with no target the Run can check opens the Review again, showing
        # that answer ("ask again each time").
        with tempfile.TemporaryDirectory() as directory:
            receipt, composition, _, question = self._ask_target(directory)
            composition.answer_targets = []
            answered = self._settle_planning_questions(
                self._answer(
                    receipt, composition, question, "the one in the APOE paper"
                ),
                composition,
            )

        (again,) = [
            request
            for request in answered.planning_hitl.pending_requests
            if request.owner == "requirement_resolution"
        ]
        self.assertNotEqual(again.request_event_id, question.request_event_id)
        self.assertEqual(
            again.issue,
            'The answer "the one in the APOE paper" names no target the '
            "Investigation can check.",
        )
        self.assertIsNone(answered.execution_hitl)

    def test_a_provider_retry_of_an_answer_reading_keeps_reading(self):
        # Spec r3 SP6: a 503 while planning reads the answer opens the availability
        # Review; its retry continues the reading Attempt and resends its exact
        # request, so the answer is read once and the target settles.
        import json

        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.application.advancement import (
            advance,
            answer_review_from_json,
            use_run_advancement_composition,
        )
        from bio_neb.journey import Journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, composition, current, question = self._ask_target(directory)
            # Settle the planning model's own questions first.
            current = self._settle_planning_questions(current, composition)
            respond = composition.respond
            unavailable = True

            async def fail_answer_reading(messages, info):
                if unavailable and any(
                    tool.name == "record_answered_targets"
                    for tool in info.function_tools
                ):
                    raise ModelHTTPError(
                        503,
                        "offline-answer-reading-unavailable",
                        body={},
                        headers={"retry-after": "999"},
                    )
                return await respond(messages, info)

            composition.respond = fail_answer_reading
            waiting = self._answer(receipt, composition, question, "rs1234")
            pending = waiting.model_availability_review.pending_request
            self.assertIsNotNone(pending)
            unavailable = False
            with use_run_advancement_composition(composition):
                continued = advance(
                    answer_review_from_json(
                        waiting,
                        pending.request_event_id,
                        json.dumps({"action": "retry"}),
                    )
                )
            events = Journey.open(Path(receipt.journey_file)).read()

        (read,) = [e for e in events if e["event_type"] == "TargetAnswerRead"]
        self.assertEqual(read["payload"]["targets"][0]["rsid"], "rs1234")
        self.assertIsNone(continued.model_availability_review.pending_request)
        self.assertEqual(continued.requirement_resolution.unsettled_targets, ())
        resent = next(
            e
            for e in events
            if e["event_type"] == "ModelRequestRecorded"
            and e["payload"]["continuation_request_event_id"] is not None
        )
        original = next(
            e
            for e in events
            if e["event_id"] == resent["payload"]["continuation_request_event_id"]
        )
        self.assertEqual(
            resent["payload"]["compatible_adapter_evidence"]["request"],
            original["payload"]["compatible_adapter_evidence"]["request"],
        )

    def test_a_target_the_request_and_file_both_name_is_one_target(self):
        from bio_neb.investigation.investigation_request import (
            current_investigation_request,
        )
        from bio_neb.journey import Journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, _, stopped = self._planned(
                Path(directory),
                self.REQUEST,
                scientific_intent="Also check RS429358.",
                scientific_rsid="rs429358",
            )
            planned = current_investigation_request(
                Journey.open(Path(receipt.journey_file))
            )

        self.assertEqual(stopped.requirement_resolution.unsettled_targets, ())
        # The file's two targets, in its order and words; the prompt's
        # RS429358 is the file's rs429358, not a third member.
        self.assertEqual(
            [(item.request_text, item.rsid) for item in planned.request.items],
            [
                ("Observe rs7412 in HG001.", "rs7412"),
                ("Observe rs429358 in HG001.", "rs429358"),
            ],
        )

    def test_a_long_request_file_hands_over_every_target_in_order(self):
        # No count cap: a list too long for the model arrives as the typed
        # request file, and planning cites it instead of restating it.
        from bio_neb.investigation.investigation_request import (
            current_investigation_request,
        )
        from bio_neb.journey import Journey

        rsids = [f"rs{1_000_000 + index}" for index in range(10_000)]
        request = {
            "schema_version": "bio-neb.investigation-request/2",
            "asserted_by": "researcher:owner",
            "requests": [{"request": rsid, "target": {"rsid": rsid}} for rsid in rsids],
        }
        with tempfile.TemporaryDirectory() as directory:
            receipt, _, stopped = self._planned(Path(directory), request)
            planned = current_investigation_request(
                Journey.open(Path(receipt.journey_file))
            )
            assertion = UserAssertion.create_collection(
                asserted_by=planned.request.asserted_by,
                requests=planned.request.items,
                predecessor_ids=(planned.source_event_id,),
                research_goals=planned.research_goals,
            )

        self.assertEqual([item.rsid for item in planned.request.items], rsids)
        self.assertEqual(stopped.requirement_resolution.unsettled_targets, ())
        self.assertEqual(len(assertion.requests), 10_000)


if __name__ == "__main__":
    unittest.main()
