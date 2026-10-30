from __future__ import annotations

import copy
import unittest
import uuid

from bio_neb.parabricks.planning_hitl import (
    ClarifyPlanningHitlResponse,
    PlanningHitlRequested,
    RejectPlanningHitlResponse,
    StopPlanningHitlResponse,
    make_planning_hitl_responded,
)
from bio_neb.parabricks.proposal_information import ToolEvidenceValidation
from bio_neb.parabricks.requirement_resolution import (
    planned_fact_slots,
    project_requirement_resolution,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    RequirementGrounding,
    SampleAttestationInput,
    VerifySampleAttestationTool,
)
from bio_neb.reference_discovery.reference_journey import SampleAttestationRecorded
from tests.planning_fakes import add_goal_requirement, add_requirement
from tests.reference_fakes import (
    INPUT_SET_ID,
    JOURNEY_ID,
    PLAN_ID,
    SAMPLE_REQUIREMENT_ID,
    append,
    build_timeline,
    identity,
    payload,
)
from tests.reference_fakes import sample_requirement as _sample_requirement
from tests.reference_fakes import unresolved_sample_review as _unresolved_sample_review


def _sample_view(resolution):
    return next(
        item for item in resolution.requirements if item.kind == "sample_identity"
    )


def _resolvable(timeline):
    """The reference timeline with the companion requirement its projection needs."""

    add_requirement(timeline, "companion_asset", "reference_companions", "7")
    return timeline


_TARGET_ID = "requirement_" + "e" * 32


def _open_target():
    """A plan with one target item the request names in no words of its own."""

    timeline, stages = build_timeline()
    timeline = copy.deepcopy(timeline[: stages["plan"]])
    plan, _sample = _sample_requirement(timeline)
    plan["payload"]["plan"]["requirements"].append(
        {
            "requirement_id": _TARGET_ID,
            "kind": "scientific_investigation",
            "subject_key": "requested_targets",
            "domain": "scientific",
            "meaning": "Check whether the sample carries the APOE e4 allele.",
            "knowledge_state": "to_find",
            "value": None,
            "evidence": [
                {
                    "source_kind": "planning_context",
                    "source_event_id": timeline[0]["event_id"],
                }
            ],
            "acquisition_route": "ask_planning_hitl",
            "validation": {"kind": "researcher_decision", "requirement": "Name them."},
            "investigation_purpose": "known_variant",
        }
    )
    return timeline


def _goal_view(resolution):
    return next(
        item for item in resolution.requirements if item.kind == "research_goal"
    )


def _view(resolution, kind):
    return next(item for item in resolution.requirements if item.kind == kind)


def _answer_target_review(timeline, response):
    """Append the coordinator's target question for the open item and one answer."""

    plan = timeline[-1]
    head = plan["event_id"]
    request = PlanningHitlRequested(
        attempt_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL, f"{JOURNEY_ID}:{_TARGET_ID}:{head}:resolution"
            )
        ),
        owner="requirement_resolution",
        plan_event_id=head,
        plan_id=PLAN_ID,
        requirement_id=_TARGET_ID,
        subject_key="requested_targets",
        issue="The request names no target for this investigation item.",
        question="Which targets should the Investigation check?",
        evidence_event_ids=(head,),
    )
    request_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"{JOURNEY_ID}:{_TARGET_ID}:{head}:review")
    )
    timeline.append(
        {
            "journey_id": JOURNEY_ID,
            "event_id": request_id,
            "event_type": request.event_type,
            "parent_ids": [head],
            "payload": payload(request),
        }
    )
    answer = payload(make_planning_hitl_responded(request_id, response))
    return append(timeline, "target-answer", "PlanningHitlResponded", answer)


def _site(contig, position, ref=None, alt=None):
    """A site as the model types it (#472 Q1 a): the contig as written, the position,
    and REF and ALT when written."""
    return {
        "contig": contig,
        "position": position,
        **({"ref": ref, "alt": alt} if ref is not None else {}),
    }


_SITE_V = _site("V", 2129317, "C", "T")


def _read_answer(timeline, answer, *targets):
    """Append planning's reading of ``answer`` (#462 target_answer stage): each
    target is ``(quote, rsid, stated_version)`` in the answer's words, with the site
    the model typed as a fourth item when it typed one (#472 Q1 a)."""
    instruction = answer["payload"]["response"]["instruction"]
    return append(
        timeline,
        "target-answer-read",
        "TargetAnswerRead",
        {
            "attempt_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "read-attempt")),
            "requirement_id": _TARGET_ID,
            "answer_event_id": answer["event_id"],
            "targets": [
                {
                    "quote": quote,
                    "start_offset": instruction.index(quote),
                    **({"rsid": rsid} if rsid else {}),
                    **({"stated_version": version} if version else {}),
                    **({"site": rest[0]} if rest else {}),
                }
                for quote, rsid, version, *rest in targets
            ],
        },
    )


def _with_investigations(timeline):
    changed = copy.deepcopy(timeline)
    plan, _sample = _sample_requirement(changed)
    context_id = changed[0]["event_id"]
    for suffix, purpose, value, typed in (
        ("variant", "known_variant", "RS1234", "rs1234"),
        ("question", "open_question", "What explains the observed phenotype?", None),
    ):
        plan["payload"]["plan"]["requirements"].append(
            {
                "requirement_id": "requirement_" + suffix[0] * 32,
                "kind": "scientific_investigation",
                "subject_key": f"investigation_{suffix}",
                "domain": "scientific",
                "meaning": "Post-execution scientific work requested by the researcher.",
                "knowledge_state": "asserted",
                "value": typed or value,
                "evidence": [
                    {
                        "source_kind": "research_request",
                        "source_event_id": context_id,
                        "start_offset": 0,
                        "end_offset": len(value),
                        "exact_text": value,
                    }
                ],
                "acquisition_route": "validate_with_tool",
                "validation": {
                    "kind": "tool_evidence",
                    "requirement": "Investigate after VCF observations are available.",
                },
                "investigation_purpose": purpose,
                **(
                    {
                        "identifier_rule": "rsid",
                        "verification": {"source": "NCBI dbSNP"},
                    }
                    if typed
                    else {}
                ),
            }
        )
    return changed


def _with_target(
    timeline, quote, *, reference=None, stated_version=None, site=None, gene_change=None
):
    """A plan whose one target item quotes ``quote`` from the request; with
    ``reference``, the request also asserts the Run's reference accession; with
    ``stated_version``, the plan names the version the quote writes (#462 SP4); with
    ``site``, the site the model typed from the quote (#472 Q1 a)."""
    changed = copy.deepcopy(timeline)
    plan, _sample = _sample_requirement(changed)
    context_id = changed[0]["event_id"]

    def quoted(text):
        return {
            "source_kind": "research_request",
            "source_event_id": context_id,
            "start_offset": 0,
            "end_offset": len(text),
            "exact_text": text,
        }

    requirements = plan["payload"]["plan"]["requirements"]
    requirements.append(
        {
            "requirement_id": _TARGET_ID,
            "kind": "scientific_investigation",
            "subject_key": "investigation_site",
            "domain": "scientific",
            "meaning": "Post-execution scientific work requested by the researcher.",
            "knowledge_state": "asserted",
            "value": quote,
            "evidence": [quoted(quote)],
            "acquisition_route": "validate_with_tool",
            "validation": {"kind": "tool_evidence", "requirement": "Look it up."},
            "investigation_purpose": "known_variant",
        }
    )
    if stated_version is not None:
        requirements[-1]["stated_version"] = stated_version
    if site is not None:
        requirements[-1]["site"] = site
    if gene_change is not None:
        requirements[-1]["gene_change"] = gene_change
    if reference is not None:
        accession, line = reference
        item = next(
            r for r in requirements if r["kind"] == "reference_assembly_identity"
        )
        item.update(
            knowledge_state="asserted",
            value=accession,
            evidence=[quoted(line)],
            acquisition_route="use_supplied_assertion",
            identifier_rule="assembly_accession",
        )
    return changed


def _set_taxon(timeline, **fields):
    plan = next(
        event
        for event in timeline
        if event["event_type"] == "ProposalRequirementsIdentified"
    )
    taxon = next(
        item
        for item in plan["payload"]["plan"]["requirements"]
        if item["kind"] == "taxon_identity"
    )
    taxon.update(fields)


class RequirementResolutionTests(unittest.TestCase):
    def test_unresolved_sample_is_projected_and_can_open_review(self) -> None:
        timeline, request_event_id = _unresolved_sample_review()
        before = project_requirement_resolution(
            [e for e in timeline if e["event_id"] != request_event_id][:-1]
        )
        sample = _sample_view(before)
        self.assertEqual(sample.state, "needs_review")
        self.assertIn(
            sample.requirement_id, before.unresolved_execution_requirement_ids
        )
        self.assertEqual(
            (sample.fact_slot.status, sample.fact_slot.summary),
            ("to_find", "to find"),
        )
        pending = project_requirement_resolution(timeline)
        self.assertEqual(_sample_view(pending).state, "needs_review")

    def test_the_attested_sample_stays_asserted_by_the_researcher(self) -> None:
        timeline = _resolvable(build_timeline()[0])

        resolution = project_requirement_resolution(timeline)
        sample = resolution.one_value("sample_identity")

        self.assertEqual(
            (sample.value, sample.source, sample.status),
            ("sample-001", "research_request", "asserted"),
        )
        self.assertIn(
            next(
                e["event_id"]
                for e in timeline
                if e["event_type"] == "SampleAttestationRecorded"
            ),
            sample.lineage_event_ids,
        )
        view = _sample_view(resolution)
        self.assertFalse(view.user_assertions[0].scientifically_verified)
        self.assertEqual(
            view.fact_slot.model_dump(mode="json"),
            {
                "requirement_id": SAMPLE_REQUIREMENT_ID,
                "kind": "sample_identity",
                "subject_key": "sample_identity",
                "status": "asserted",
                "verification_source": None,
                "requested_value": "sample-001",
                "verified_value": None,
                "mismatch": False,
                "summary": "asserted by researcher, unverified",
            },
        )
        self.assertEqual(view.state, "resolved")

    def test_a_source_verified_slot_is_verified_and_a_different_request_is_flagged(
        self,
    ) -> None:
        timeline = _resolvable(build_timeline()[0])
        found = project_requirement_resolution(timeline)
        taxon = _view(found, "taxon_identity").fact_slot
        self.assertEqual(
            (taxon.status, taxon.requested_value, taxon.verified_value, taxon.mismatch),
            ("verified", None, "9606", False),
        )
        self.assertEqual(taxon.summary, "verified by NCBI Taxonomy")
        self.assertEqual(
            [(v.source, v.status) for v in _view(found, "taxon_identity").values],
            [("tool_result", "verified")],
        )
        reference = _view(found, "reference_assembly_identity").fact_slot
        self.assertEqual(
            (reference.status, reference.verification_source),
            ("verified", "NCBI Datasets"),
        )
        self.assertEqual(
            tuple(item.kind for item in found.fact_slots),
            ("sample_identity", "taxon_identity", "reference_assembly_identity"),
        )

        for requested, mismatch in (("9606", False), ("10090", True)):
            changed = copy.deepcopy(timeline)
            _set_taxon(changed, knowledge_state="asserted", value=requested)
            _set_taxon(changed, identifier_rule="ncbi_taxon_id")
            slot = _view(project_requirement_resolution(changed), "taxon_identity")
            with self.subTest(requested=requested):
                self.assertEqual(slot.state, "resolved")
                self.assertEqual(
                    (slot.fact_slot.requested_value, slot.fact_slot.mismatch),
                    (requested, mismatch),
                )
                if mismatch:
                    self.assertEqual(
                        slot.fact_slot.summary,
                        "requested 10090, but NCBI Taxonomy verified 9606",
                    )

    def test_planned_fact_slots_are_the_plan_s_and_none_without_a_plan(
        self,
    ) -> None:
        timeline = _resolvable(build_timeline()[0])
        self.assertEqual(
            planned_fact_slots(timeline),
            project_requirement_resolution(timeline).fact_slots,
        )
        self.assertEqual(planned_fact_slots(timeline[:1]), ())

    def test_a_selected_candidate_is_the_slots_latest_requested_value(
        self,
    ) -> None:
        # A Review selection's typed value is copied from the candidate's minted id, so
        # the researcher's selection, later than the request's own accession, is the
        # slot's requested value; the mismatch flag itself is
        # test_a_source_verified_slot_is_verified_and_a_different_request_is_flagged's.
        timeline = _resolvable(build_timeline()[0])
        plan = next(
            e for e in timeline if e["event_type"] == "ProposalRequirementsIdentified"
        )
        reference = next(
            item
            for item in plan["payload"]["plan"]["requirements"]
            if item["kind"] == "reference_assembly_identity"
        )
        reference.update(
            knowledge_state="asserted",
            value="GCF_000001405.40",
            identifier_rule="assembly_accession",
        )
        reference["evidence"] = [
            *reference["evidence"],
            {
                "source_kind": "research_request",
                "source_event_id": timeline[0]["event_id"],
                "start_offset": 0,
                "end_offset": 16,
                "exact_text": "GCF_000001405.40",
            },
        ]

        slot = _view(
            project_requirement_resolution(timeline), "reference_assembly_identity"
        ).fact_slot

        self.assertEqual(
            (slot.status, slot.requested_value, slot.verified_value, slot.mismatch),
            ("verified", "GCF_000009999.1", "GCF_000009999.1", False),
        )
        self.assertEqual(slot.summary, "verified by NCBI Datasets")

    def test_an_asserted_slot_awaits_its_source(self) -> None:
        timeline, stages = build_timeline()
        planned = copy.deepcopy(timeline[: stages["plan"]])
        _set_taxon(
            planned,
            knowledge_state="asserted",
            value="9606",
            identifier_rule="ncbi_taxon_id",
        )
        taxon = _view(project_requirement_resolution(planned), "taxon_identity")
        self.assertEqual(
            (taxon.fact_slot.status, taxon.fact_slot.summary),
            ("asserted", "asserted by researcher; not yet verified by NCBI Taxonomy"),
        )

    def test_scientific_questions_survive_without_blocking_execution(self) -> None:
        timeline = _resolvable(build_timeline()[0])

        execution_only = project_requirement_resolution(timeline)
        resolution = project_requirement_resolution(_with_investigations(timeline))

        # Post-execution items never hold execution back.
        self.assertEqual(
            resolution.unresolved_execution_requirement_ids,
            execution_only.unresolved_execution_requirement_ids,
        )
        self.assertEqual(
            tuple(item.purpose for item in resolution.scientific_investigation.items),
            ("known_variant", "open_question"),
        )
        self.assertEqual(
            tuple(item.state for item in resolution.scientific_investigation.items),
            ("open", "open"),
        )
        self.assertEqual(
            resolution.scientific_investigation.items[0].question_or_assertion,
            "RS1234",
        )
        rsid = _view(resolution, "scientific_investigation").fact_slot
        self.assertEqual(
            (rsid.kind, rsid.status, rsid.requested_value, rsid.summary),
            (
                "scientific_investigation",
                "asserted",
                "rs1234",
                "asserted by researcher; not yet verified by NCBI dbSNP",
            ),
        )

    def test_asserted_research_goal_is_a_researcher_value(self) -> None:
        timeline, stages = build_timeline()
        planned = copy.deepcopy(timeline[: stages["plan"]])
        add_goal_requirement(planned)
        resolution = project_requirement_resolution(planned)
        goal = _goal_view(resolution)

        self.assertEqual(goal.state, "resolved")
        self.assertFalse(goal.execution_consequence)
        self.assertIsNone(goal.fact_slot)
        self.assertEqual(
            [(v.value, v.source, v.status) for v in goal.values],
            [("Analyze sample-001", "research_request", "asserted")],
        )
        self.assertEqual(goal.values[0].lineage_event_ids, (planned[0]["event_id"],))
        self.assertIn(goal.values[0], resolution.values)
        # The goal travels to the Investigation in the researcher's words.
        self.assertEqual(
            [
                (g.requirement_id, g.words, g.source_event_id)
                for g in resolution.scientific_investigation.research_goals
            ],
            [(goal.requirement_id, "Analyze sample-001", planned[0]["event_id"])],
        )
        self.assertNotIn(
            goal.requirement_id, resolution.unresolved_execution_requirement_ids
        )

    def test_research_goal_with_remaining_validation_has_no_value(self) -> None:
        timeline, stages = build_timeline()
        planned = copy.deepcopy(timeline[: stages["plan"]])
        add_goal_requirement(
            planned,
            validation=ToolEvidenceValidation(
                requirement="Confirm the goal with exact Tool evidence."
            ),
        )

        goal = _goal_view(project_requirement_resolution(planned))

        self.assertEqual((goal.state, goal.values), ("needs_tool", ()))

    def test_same_value_with_new_lineage_changes_exact_execution_resolution(
        self,
    ) -> None:
        timeline = _resolvable(build_timeline()[0])
        before = project_requirement_resolution(timeline)
        original = next(
            item
            for item in timeline
            if item["event_type"] == "SampleAttestationRecorded"
        )
        append(
            timeline,
            "same-sample-new-lineage",
            "SampleAttestationRecorded",
            copy.deepcopy(original["payload"]),
        )

        after = project_requirement_resolution(timeline)

        self.assertEqual(
            before.one_value("sample_identity").value,
            after.one_value("sample_identity").value,
        )
        self.assertNotEqual(
            before.one_value("sample_identity").value_id,
            after.one_value("sample_identity").value_id,
        )

    def test_contradictory_tool_admissions_remain_visible_and_block_compilation(
        self,
    ) -> None:
        timeline = _resolvable(build_timeline()[0])
        plan = next(
            e for e in timeline if e["event_type"] == "ProposalRequirementsIdentified"
        )
        assertion = append(
            timeline,
            "second-sample-assertion",
            "ResearchRequestRegistered",
            {"prompt": "Use sample-002."},
        )
        attempt_id = identity("second-sample-attempt")
        append(
            timeline,
            "second-sample-attempt-start",
            "AttemptStarted",
            {
                "attempt_id": attempt_id,
                "agent_role": "planning_agent",
                "input_set_id": INPUT_SET_ID,
            },
        )
        _plan, sample = _sample_requirement(timeline)
        grounding = RequirementGrounding(
            journey_id=JOURNEY_ID,
            plan_id=PLAN_ID,
            requirement_id=SAMPLE_REQUIREMENT_ID,
            kind="sample_identity",
            subject_key="sample_identity",
            source_event_ids=(
                plan["event_id"],
                *(item["source_event_id"] for item in sample["evidence"]),
                assertion["event_id"],
            ),
        )
        tool_input = SampleAttestationInput(
            grounding=grounding,
            input_set_id=INPUT_SET_ID,
            sample_identifier="sample-002",
            assertion_event_id=assertion["event_id"],
            quote="sample-002",
        )
        result = VerifySampleAttestationTool().verify(
            tool_input,
            expected_grounding=grounding,
            expected_input_set_id=INPUT_SET_ID,
            expected_sample_identifier="sample-002",
        )
        append(
            timeline,
            "second-sample-result",
            "SampleAttestationRecorded",
            payload(
                SampleAttestationRecorded(
                    attempt_id=attempt_id,
                    tool_input=tool_input,
                    result=result,
                )
            ),
        )

        resolution = project_requirement_resolution(timeline)
        sample = _sample_view(resolution)

        self.assertEqual(sample.state, "conflicted")
        self.assertEqual(set(sample.contradiction.values), {"sample-001", "sample-002"})
        self.assertFalse(resolution.ready_for_execution_compilation)

    def test_assertions_read_the_recorded_typed_value_and_quote(self) -> None:
        timeline, stages = build_timeline()
        timeline = copy.deepcopy(timeline[: stages["plan"]])
        _plan, sample = _sample_requirement(timeline)
        # The recorded typed value is read as recorded, never re-derived from the quote.
        sample["value"] = "recorded-sample"

        assertion = _sample_view(
            project_requirement_resolution(timeline)
        ).user_assertions[0]

        self.assertEqual(
            (assertion.asserted_value, assertion.identifier_rule, assertion.value),
            ("sample-001", "sample_identifier", "recorded-sample"),
        )

    def test_a_to_find_quote_is_said_but_states_no_value(self) -> None:
        timeline, stages = build_timeline()
        timeline = copy.deepcopy(timeline[: stages["plan"]])

        taxon = _view(project_requirement_resolution(timeline), "taxon_identity")

        self.assertEqual(
            [(a.asserted_value, a.value) for a in taxon.user_assertions],
            [("Homo sapiens", None)],
        )
        self.assertEqual(taxon.fact_slot.status, "to_find")

    def test_each_target_travels_in_the_researchers_own_words(self) -> None:
        # The quote is the target; its typed rsID rides with it.
        resolution = project_requirement_resolution(
            _with_investigations(_resolvable(build_timeline()[0]))
        )

        self.assertEqual(
            [
                [(t.words, t.rsid) for t in item.targets]
                for item in resolution.scientific_investigation.items
            ],
            [
                [("RS1234", "rs1234")],
                [("What explains the observed phenotype?", None)],
            ],
        )
        # The question's words state no rsID or position, so it is asked.
        self.assertEqual(
            [(u.requirement_id, u.reason) for u in resolution.unsettled_targets],
            [("requirement_" + "q" * 32, "untyped_target")],
        )

    def test_model_text_is_never_a_target_so_an_open_item_is_asked(self) -> None:
        # The model's ``meaning`` never stands in for the researcher.
        resolution = project_requirement_resolution(_open_target())

        (item,) = resolution.scientific_investigation.items
        self.assertEqual(item.targets, ())
        self.assertEqual(
            [(u.requirement_id, u.reason) for u in resolution.unsettled_targets],
            [(_TARGET_ID, "no_target")],
        )

    def test_an_answer_planning_has_not_read_holds_its_item(self) -> None:
        # An answer goes back to planning's target reading; until the model reads it,
        # no code reads a target out of it.
        timeline = _open_target()
        answer = _answer_target_review(
            timeline,
            ClarifyPlanningHitlResponse(
                action="clarify", instruction="Check RS429358 and rs7412."
            ),
        )

        resolution = project_requirement_resolution(timeline)

        self.assertEqual(resolution.scientific_investigation.items[0].targets, ())
        self.assertEqual(
            [
                (u.reason, u.answer, u.answer_event_id)
                for u in resolution.unsettled_targets
            ],
            [("answer_unread", "Check RS429358 and rs7412.", answer["event_id"])],
        )

    def test_a_target_answer_names_the_targets_planning_read_in_it(self) -> None:
        timeline = _open_target()
        answer = _answer_target_review(
            timeline,
            ClarifyPlanningHitlResponse(
                action="clarify", instruction="Check RS429358 and rs7412."
            ),
        )
        _read_answer(
            timeline, answer, ("RS429358", "rs429358", None), ("rs7412", "rs7412", None)
        )

        resolution = project_requirement_resolution(timeline)

        (item,) = resolution.scientific_investigation.items
        self.assertEqual(
            [(t.words, t.rsid, t.source_event_id) for t in item.targets],
            [
                ("RS429358", "rs429358", answer["event_id"]),
                ("rs7412", "rs7412", answer["event_id"]),
            ],
        )
        self.assertEqual(resolution.unsettled_targets, ())

    def test_an_answer_quote_the_model_typed_nothing_for_states_no_target(self) -> None:
        # ADR 0049 ("code reading free text for a value" is a rule a surface may not
        # set): an rsID is the model's typed value, never one code picks out of the
        # quote; the untyped target is asked again.
        timeline = _open_target()
        answer = _answer_target_review(
            timeline,
            ClarifyPlanningHitlResponse(action="clarify", instruction="Check rs7412."),
        )
        _read_answer(timeline, answer, ("rs7412", None, None))

        resolution = project_requirement_resolution(timeline)

        (target,) = resolution.scientific_investigation.items[0].targets
        self.assertEqual((target.rsid, target.position), (None, None))
        self.assertEqual(
            [u.reason for u in resolution.unsettled_targets], ["untyped_target"]
        )

    def test_a_position_the_request_states_is_a_typed_target_on_its_version(self):
        # The site the model typed from "V:2129317 C>T" is the target's position,
        # on the version the request states for the Run's reference (the
        # researcher's accession and words).
        from bio_neb.investigation.investigation_request import (
            StatedAssembly,
            StatedPosition,
        )

        line = "Align to GCF_000002985.6 (WBcel235)."
        resolution = project_requirement_resolution(
            _with_target(
                _resolvable(build_timeline()[0]),
                "V:2129317 C>T",
                reference=("GCF_000002985.6", line),
                site=_SITE_V,
            )
        )

        (item,) = resolution.scientific_investigation.items
        (target,) = item.targets
        self.assertEqual(
            (target.words, target.rsid, target.position, target.stated_assembly),
            (
                "V:2129317 C>T",
                None,
                StatedPosition(contig="V", position=2129317, ref="C", alt="T"),
                StatedAssembly(name="GCF_000002985.6", cited_line=line),
            ),
        )
        self.assertEqual(resolution.unsettled_targets, ())

    def test_a_typed_gene_change_is_a_target_the_run_places(self):
        """#473 Q1 a: the gene change the model typed from "BRAF V600E" is the
        target's, settled with no rsID or position, and reaches the Run's request
        item, where the Investigation places it."""
        from bio_neb.investigation.investigation_request import (
            StatedChange,
            planned_investigation_request,
        )

        change = {
            "gene": "BRAF",
            "change": "V600E",
            "protein": {"ref": "V", "codon": 600, "alt": "E"},
        }
        resolution = project_requirement_resolution(
            _with_target(
                _resolvable(build_timeline()[0]), "BRAF V600E", gene_change=change
            )
        )

        (item,) = resolution.scientific_investigation.items
        (target,) = item.targets
        self.assertEqual(
            (target.rsid, target.position, target.gene_change),
            (None, None, StatedChange.model_validate(change)),
        )
        self.assertEqual(resolution.unsettled_targets, ())
        captured = planned_investigation_request(resolution, ())
        (request,) = captured.request.items
        self.assertEqual(
            (request.request_text, request.gene_change),
            ("BRAF V600E", StatedChange.model_validate(change)),
        )

    def test_an_untyped_gene_change_is_still_to_settle(self):
        """Code reads no change text (#473 Q2 a): a gene and change with no typed part
        settles nothing, and planning asks before the Execution proposal (#462 R3)."""
        resolution = project_requirement_resolution(
            _with_target(
                _resolvable(build_timeline()[0]),
                "BRAF V600E",
                gene_change={"gene": "BRAF", "change": "V600E"},
            )
        )

        self.assertEqual(
            [u.reason for u in resolution.unsettled_targets], ["untyped_target"]
        )

    def test_the_version_the_plan_names_in_the_quote_is_the_targets_own(self):
        # The model names the version the quote writes for the site, as written;
        # grounding found it in the quote.
        from bio_neb.investigation.investigation_request import StatedAssembly

        for quote, version, site in (
            ("chr7:117559590 C>T on hg19", "hg19", _site("chr7", 117559590, "C", "T")),
            ("chr19:45411941 T>C (hg19)", "hg19", _site("chr19", 45411941, "T", "C")),
            (
                "chr19:45411941 T>C on CanFam 3.1",
                "CanFam 3.1",
                _site("chr19", 45411941, "T", "C"),
            ),
        ):
            with self.subTest(quote=quote):
                resolution = project_requirement_resolution(
                    _with_target(
                        _resolvable(build_timeline()[0]),
                        quote,
                        reference=("GCF_000001405.40", "Use GCF_000001405.40."),
                        stated_version=version,
                        site=site,
                    )
                )

                (target,) = resolution.scientific_investigation.items[0].targets
                self.assertEqual(
                    target.stated_assembly,
                    StatedAssembly(name=version, cited_line=quote),
                )

    def test_code_reads_no_version_out_of_a_quote(self):
        # With no version the plan names, the quote's words are not read for one; the
        # request's stated reference applies.
        from bio_neb.investigation.investigation_request import StatedAssembly

        resolution = project_requirement_resolution(
            _with_target(
                _resolvable(build_timeline()[0]),
                "chr7:117559590 C>T on hg19",
                reference=("GCF_000001405.40", "Use GCF_000001405.40."),
                site=_site("chr7", 117559590, "C", "T"),
            )
        )

        (target,) = resolution.scientific_investigation.items[0].targets
        self.assertEqual(
            target.stated_assembly,
            StatedAssembly(name="GCF_000001405.40", cited_line="Use GCF_000001405.40."),
        )

    def test_a_position_with_no_version_stated_has_none(self):
        resolution = project_requirement_resolution(
            _with_target(
                _resolvable(build_timeline()[0]), "V:2129317 C>T", site=_SITE_V
            )
        )

        (target,) = resolution.scientific_investigation.items[0].targets
        self.assertIsNotNone(target.position)
        self.assertIsNone(target.stated_assembly)

    def test_code_reads_no_position_out_of_a_quote(self):
        # ADR 0049 ("code reading free text for a value" is a rule a surface may not
        # set): a quote that writes a site is no position until the model types it;
        # the untyped target is asked before Execution.
        resolution = project_requirement_resolution(
            _with_target(_resolvable(build_timeline()[0]), "V:2129317 C>T")
        )

        (target,) = resolution.scientific_investigation.items[0].targets
        self.assertIsNone(target.position)
        self.assertEqual(
            [(u.requirement_id, u.reason) for u in resolution.unsettled_targets],
            [(_TARGET_ID, "untyped_target")],
        )

    def test_an_rsid_target_keeps_its_shape_and_digest(self):
        # The new fields are left out when empty, so an rsID target's handoff
        # material is what it was.
        resolution = project_requirement_resolution(
            _with_investigations(_resolvable(build_timeline()[0]))
        )

        target = resolution.scientific_investigation.items[0].targets[0]
        self.assertEqual(
            set(target.model_dump(mode="json")), {"words", "rsid", "source_event_id"}
        )

    def test_words_that_state_no_rsid_or_position_are_asked(self):
        # A target whose words the Run cannot check is a dead end asked before
        # Execution, never recorded "not checked" in silence.
        resolution = project_requirement_resolution(
            _with_target(_resolvable(build_timeline()[0]), "the variant near unc-22")
        )

        self.assertEqual(
            [(u.requirement_id, u.reason) for u in resolution.unsettled_targets],
            [(_TARGET_ID, "untyped_target")],
        )

    def test_a_target_answer_keeps_its_position_beside_its_one_rsid(self):
        # Spec r1 SP2: "rs429358 at chr19:45411941 T>C on hg19" is one target with both.
        from bio_neb.investigation.investigation_request import (
            StatedAssembly,
            StatedPosition,
        )

        instruction = "Check rs429358 at chr19:45411941 T>C on hg19."
        quote = "rs429358 at chr19:45411941 T>C on hg19"
        timeline = _open_target()
        answer = _answer_target_review(
            timeline,
            ClarifyPlanningHitlResponse(action="clarify", instruction=instruction),
        )
        _read_answer(
            timeline,
            answer,
            (quote, "rs429358", "hg19", _site("chr19", 45411941, "T", "C")),
        )

        (target,) = (
            project_requirement_resolution(timeline)
            .scientific_investigation.items[0]
            .targets
        )
        self.assertEqual(
            (target.words, target.rsid, target.position, target.stated_assembly),
            (
                quote,
                "rs429358",
                StatedPosition(contig="chr19", position=45411941, ref="T", alt="C"),
                StatedAssembly(name="hg19", cited_line=quote),
            ),
        )

    def test_a_sample_id_beside_a_position_leaves_the_asserted_reference(self):
        # Spec r1 SP1: "HG001" is a sample, not the position's genome version.
        from bio_neb.investigation.investigation_request import StatedAssembly

        resolution = project_requirement_resolution(
            _with_target(
                _resolvable(build_timeline()[0]),
                "Does sample HG001 carry chr19:44908684 T>C?",
                reference=("GCF_000001405.40", "Use GCF_000001405.40."),
                site=_site("chr19", 44908684, "T", "C"),
            )
        )

        (target,) = resolution.scientific_investigation.items[0].targets
        self.assertEqual(
            target.stated_assembly,
            StatedAssembly(name="GCF_000001405.40", cited_line="Use GCF_000001405.40."),
        )

    def test_a_target_answer_may_state_one_position(self):
        from bio_neb.investigation.investigation_request import StatedPosition

        timeline = _open_target()
        answer = _answer_target_review(
            timeline,
            ClarifyPlanningHitlResponse(
                action="clarify", instruction="Check V:2129317 C>T."
            ),
        )
        _read_answer(timeline, answer, ("V:2129317 C>T", None, None, _SITE_V))

        resolution = project_requirement_resolution(timeline)

        (target,) = resolution.scientific_investigation.items[0].targets
        self.assertEqual(
            (target.words, target.position, target.source_event_id),
            (
                "V:2129317 C>T",
                StatedPosition(contig="V", position=2129317, ref="C", alt="T"),
                answer["event_id"],
            ),
        )
        self.assertEqual(resolution.unsettled_targets, ())

    def test_every_target_rejected_leaves_nothing_to_execute(self) -> None:
        timeline = _open_target()
        _answer_target_review(
            timeline,
            RejectPlanningHitlResponse(action="reject", reason="No variant to check."),
        )

        resolution = project_requirement_resolution(timeline)

        self.assertEqual(
            _view(resolution, "scientific_investigation").state, "rejected"
        )
        self.assertEqual(resolution.unsettled_targets, ())
        self.assertIn(_TARGET_ID, resolution.unresolved_execution_requirement_ids)
        self.assertFalse(resolution.ready_for_execution_compilation)

    def test_a_sample_answer_states_no_value_until_the_model_types_it(self) -> None:
        """ADR 0049 ("code reading free text for a value" is a rule a surface may not
        set), #472 Q2 a: a planning answer's words are the researcher's assertion and
        no code picks a value out of them; the sample stage types the name with its
        quote (`attest_sample`). Replaces C11's matcher test (retired with
        `planning_answer_reading`)."""
        for instruction in ("HG002", "sample HG002", "The sample is HG002."):
            timeline, request_event_id = _unresolved_sample_review()
            response = ClarifyPlanningHitlResponse(
                action="clarify", instruction=instruction
            )
            fact = make_planning_hitl_responded(request_event_id, response)
            append(timeline, "sample-answer", fact.event_type, payload(fact))

            view = _sample_view(project_requirement_resolution(timeline))
            assertion = view.user_assertions[-1]

            with self.subTest(instruction=instruction):
                self.assertEqual(
                    (
                        assertion.assertion_kind,
                        assertion.asserted_value,
                        assertion.identifier_rule,
                        assertion.value,
                    ),
                    ("planning_hitl", instruction, None, None),
                )
                self.assertNotIn("value", payload(fact))

    def test_stop_response_is_typed_researcher_confirmation(self) -> None:
        response = StopPlanningHitlResponse(
            action="stop", reason="No trustworthy sample label is available."
        )
        fact = make_planning_hitl_responded(identity("review"), response)

        self.assertEqual(fact.response.action, "stop")
        self.assertFalse(fact.scientifically_verified)
        self.assertFalse(fact.authorizes_execution)


if __name__ == "__main__":
    unittest.main()
