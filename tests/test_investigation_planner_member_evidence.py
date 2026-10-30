"""The Investigation planner's member prompts and output repairs (#360).

Each member's prompt carries only its own source evidence, and an incomplete
complete draft is refused with a code that names each missing field and the value
the cited source holds for it.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse, ToolCallPart

from bio_neb.investigation.investigation import (
    AmendInvestigationFindingResponse,
    InvestigationPlanDraft,
    InvestigationRequestItem,
    UserAssertion,
)
from bio_neb.investigation.investigation_commands import (
    ContinueInvestigationCommand,
    RespondInvestigationReviewCommand,
    continue_investigation,
    respond_investigation_review,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.recording import read_evidence
from tests.model_endpoint_fakes import ScriptedModelEndpoint, corrections
from tests.refsnp_retained_bodies import RS4988235
from tests.sequential_investigation_support import (
    NOT_FOUND,
    AmendedLookupModel,
    IssueSourceResponses,
    SequentialAdaptiveModel,
    SequentialSourceResponses,
    SourceResponsesOnly,
    _current_member_messages,
    run_attempt,
)

SOURCE_RESULT_KINDS = {"RefSnpToolResultRecorded", "AssemblyReportToolResultRecorded"}


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


def _member_evidence_marks(events, member_id):
    """Every identity a member's source evidence carries: its result events, their
    retrieved sources and their typed result ids."""
    marks = set()
    for event in events:
        if (
            event["event_type"] in SOURCE_RESULT_KINDS
            and event["payload"]["collection_member_id"] == member_id
        ):
            marks.update(
                (
                    event["event_id"],
                    event["payload"]["source_event_id"],
                    event["payload"]["result"]["result_id"],
                )
            )
    return marks


def _continue(journey, model, sources):
    return continue_investigation(
        ContinueInvestigationCommand(journey_file=str(journey.path)),
        environ={"LOCAL_MODEL_KEY": "offline"},
        model_factory=model.model,
        source_adapter_factory=lambda: sources,
    )


def _member_requests(events, member_id):
    return [
        event
        for event in events
        if event["event_type"] == "ModelRequestRecorded"
        and event["payload"]["member_id"] == member_id
    ]


class MemberEvidenceTests(unittest.TestCase):
    def test_a_member_prompt_holds_no_other_member_evidence(self):
        # A recorded live Run: every member after the first was sent the
        # previous member's RefSNP and assembly results as its current
        # evidence.
        with tempfile.TemporaryDirectory() as directory:
            _projection, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=SequentialAdaptiveModel().model),
                SequentialSourceResponses(),
                assertion=_assertion("rs429358", "rs7412"),
            )
            events = read_evidence(Journey.open(journey.path))
            members = list(
                dict.fromkeys(
                    event["payload"]["member_id"]
                    for event in events
                    if event["event_type"] == "ModelRequestRecorded"
                )
            )
            self.assertEqual(len(members), 2)
            first, second = members
            first_marks = _member_evidence_marks(
                events, first
            ) - _member_evidence_marks(events, second)
            self.assertTrue(first_marks)
            second_requests = _member_requests(events, second)
            self.assertTrue(second_requests)
            for request in second_requests:
                sent = canonical_json(request["payload"])
                self.assertEqual(
                    {mark for mark in first_marks if mark in sent},
                    set(),
                    "the second member's request carries the first member's evidence",
                )

    def test_a_rechecked_member_prompt_shows_its_own_source_results_only(self):
        # An amended issue SNP is checked again after its sibling: its prompt shows
        # its own retained results, never the sibling's later ones.
        with tempfile.TemporaryDirectory() as directory:
            sources = IssueSourceResponses()
            model = AmendedLookupModel()
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                assertion=_assertion(NOT_FOUND, "rs429358"),
            )
            # The sibling is checked in the first batch, its own Attempt.
            review = pending.pending_request
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=review.request_event_id,
                    response=AmendInvestigationFindingResponse(
                        action="amend",
                        collection_member_id=review.question.collection_member_id,
                        revision_number=review.question.revision_number,
                        instruction="I meant rs7412.",
                        asserted_by="researcher",
                    ),
                )
            )
            before = read_evidence(Journey.open(journey.path))
            amended = review.question.collection_member_id
            (sibling,) = {
                event["payload"]["collection_member_id"]
                for event in before
                if event["event_type"] in SOURCE_RESULT_KINDS
            } - {amended}
            (own,) = [
                event["event_id"]
                for event in before
                if event["event_type"] == "RefSnpToolResultRecorded"
                and event["payload"]["collection_member_id"] == amended
            ]
            _continue(journey, model, sources)
            after = read_evidence(Journey.open(journey.path))
            rechecked = _member_requests(after, amended)[
                len(_member_requests(before, amended))
            ]
            evidence = _member_context(rechecked, member_id=amended)["assignment"][
                "current_evidence"
            ]
            self.assertEqual(
                {kind: item["event_id"] for kind, item in evidence.items()},
                {"RefSnpToolResultRecorded": own},
            )
            self.assertEqual(
                {
                    mark
                    for mark in _member_evidence_marks(before, sibling)
                    - _member_evidence_marks(before, amended)
                    if mark in canonical_json(rechecked["payload"])
                },
                set(),
            )


def _member_context(request, *, member_id):
    contexts = []
    for message in request["payload"]["compatible_adapter_evidence"]["request"][
        "messages"
    ]:
        for part in message.get("parts", ()):
            if part.get("part_kind") != "user-prompt" or not isinstance(
                part.get("content"), str
            ):
                continue
            try:
                context = json.loads(part["content"])
            except json.JSONDecodeError:
                continue
            if isinstance(context, dict) and "user_assertion" in context:
                contexts.append(context)
    (context,) = contexts
    assert context["collection_member_id"] == member_id
    return context


# A recorded live Run: the rs4988235 member's complete draft carried only
# the schema-required fields and two constants; its RefSNP event id is this
# test's own.
RECORDED_INCOMPLETE_DRAFT = {
    "aliases": ["rs4988235"],
    "intended_action": "observe_variant",
    "outcome": "complete",
    "unresolved_conflicts": [],
}


class Rs4988235ReplayModel:
    """Replays #107's rs4988235 member: both lookups, the recorded incomplete
    draft, then the complete draft the continued Attempt sent (response
    ``1f4b2f6a``)."""

    def __init__(
        self, *, incomplete_drafts: int = 1, draft: dict = RECORDED_INCOMPLETE_DRAFT
    ) -> None:
        self.calls = 0
        self.incomplete_drafts = incomplete_drafts
        self.draft = draft
        self.seen: list[str] = []
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        self.calls += 1
        self.seen.append(ModelMessagesTypeAdapter.dump_json(messages).decode())
        _request, refsnp, assembly, basis, *_ = _current_member_messages(messages)
        if refsnp is None:
            name, arguments = "fetch_refsnp", {"rsid": "rs4988235"}
        elif assembly is None:
            name = "fetch_assembly_report"
            arguments = {
                "assembly_accession": "GCF_000001405.40",
                "refsnp_event_id": basis["refsnp_event_id"],
            }
        elif self.calls <= 2 + self.incomplete_drafts:
            name = "compile_proposal"
            arguments = {**self.draft, "refsnp_event_id": basis["refsnp_event_id"]}
        else:
            # The model's choice and its citations only.
            name = "compile_proposal"
            arguments = {**basis, "outcome": "complete"}
        return ModelResponse(
            parts=[
                ToolCallPart(
                    name, canonical_json(arguments), tool_call_id=f"replay-{self.calls}"
                )
            ]
        )


class CodeHeldValueTests(unittest.TestCase):
    """The model re-types nothing code holds (#478 G2.1, ADR 0049): a draft that
    sends a copied field is refused by that field, never overwritten, and code
    copies every value from the results the model cites."""

    def test_a_draft_that_retypes_a_code_held_value_is_refused_by_that_field(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Rs4988235ReplayModel()
            projection, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                SourceResponsesOnly(refsnp_body=RS4988235),
                assertion=_assertion("rs4988235"),
            )
            events = read_evidence(Journey.open(journey.path))
            refused, accepted = [
                event["payload"]
                for event in events
                if event["event_type"] == "AgentOutputValidated"
            ]
            # The recorded draft re-typed three fields code holds.
            self.assertEqual(refused["status"], "invalid")
            self.assertEqual(
                set(refused["errors"]),
                # One derived validator names each field and its error.
                {
                    "aliases:extra_forbidden",
                    "intended_action:extra_forbidden",
                    "unresolved_conflicts:extra_forbidden",
                },
            )
            self.assertEqual((accepted["status"], accepted["errors"]), ("valid", []))
            self.assertEqual(
                set(accepted["output"]),
                {"outcome", "refsnp_event_id", "assembly_event_id"},
            )
            self.assertEqual(projection.status, "review_pending")
            (proposal,) = projection.proposals
            # Code copied the locus from the cited record.
            self.assertEqual(
                [
                    (
                        item.target.contig,
                        item.target.one_based_position,
                        item.target.ref,
                    )
                    for item in proposal.work_items
                ],
                [("NC_000002.12", 135851076, "G")] * 3,
            )
            self.assertEqual(
                [item.target.alt for item in proposal.work_items], ["A", "C", "T"]
            )

    def test_the_default_policy_allows_a_second_repair_of_the_same_draft(self):
        # The recorded next request resent the previous draft byte for byte; one
        # repair was all the Run allowed. The default allowed two, and now four.
        with tempfile.TemporaryDirectory() as directory:
            model = Rs4988235ReplayModel(incomplete_drafts=2)
            projection, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                SourceResponsesOnly(refsnp_body=RS4988235),
                assertion=_assertion("rs4988235"),
            )
            events = read_evidence(Journey.open(journey.path))
            self.assertEqual(
                [
                    event["payload"]["status"]
                    for event in events
                    if event["event_type"] == "AgentOutputValidated"
                ],
                ["invalid", "invalid", "valid"],
            )
            (started,) = [
                event["payload"]
                for event in events
                if event["event_type"] == "AttemptStarted"
            ]
            self.assertEqual(started["continuation_policy"]["failed_turns"], 4)
            self.assertEqual(projection.status, "review_pending")

    def test_the_draft_takes_only_the_outcome_and_its_citations(self):
        self.assertEqual(
            set(InvestigationPlanDraft.model_fields),
            {"outcome", "refsnp_event_id", "assembly_event_id"},
        )
        for field in ("contig", "unresolved_conflicts"):
            with self.subTest(field=field), self.assertRaises(ValidationError) as sent:
                InvestigationPlanDraft.model_validate_json(
                    json.dumps(
                        {"refsnp_event_id": "e", "outcome": "unresolved", field: []}
                    )
                )
            ((error,),) = (sent.exception.errors(),)
            self.assertEqual(
                (error["type"], error["loc"]), ("extra_forbidden", (field,))
            )


class _ReshapedFirstOutputModel(SequentialAdaptiveModel):
    """Sends its first output reshaped by ``reshape``, then the lawful one; it
    keeps every correction the Attempt sent it."""

    def __init__(self, reshape) -> None:
        super().__init__()
        self.reshape = reshape
        self.outputs = 0
        self.corrections: list[str] = []

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        for message in messages[-1:]:
            self.corrections.extend(corrections(message))
        response = await super().respond(messages, info)
        (part,) = response.parts
        if part.tool_name != "compile_proposal":
            return response
        self.outputs += 1
        if self.outputs > 1:
            return response
        return ModelResponse(
            parts=[
                ToolCallPart(
                    part.tool_name,
                    canonical_json(self.reshape(part.args_as_dict())),
                    tool_call_id=part.tool_call_id,
                )
            ]
        )


class UnresolvedRefusalVisibilityTests(unittest.TestCase):
    """An unresolved-path refusal states why it refused and the outcome to send
    (#372 D3, owner 2026-10-20: visibility, not a changed outcome). The planted
    rs9999999999 still ends unresolved, with the source's own conflicts, which
    code copies (#478 G2.1)."""

    def _refused(self, reshape):
        with tempfile.TemporaryDirectory() as directory:
            model = _ReshapedFirstOutputModel(reshape)
            _projection, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                IssueSourceResponses(),
                assertion=_assertion(NOT_FOUND),
            )
            events = read_evidence(Journey.open(journey.path))
        (lookup,) = [
            event["payload"]["result"]
            for event in events
            if event["event_type"] == "RefSnpToolResultRecorded"
        ]
        refused, accepted = [
            event["payload"]
            for event in events
            if event["event_type"] == "AgentOutputValidated"
        ]
        self.assertEqual(
            (accepted["status"], accepted["output"]["outcome"]), ("valid", "unresolved")
        )
        self.assertEqual(refused["status"], "invalid")
        return refused, json.dumps(lookup["conflicts"]), model

    def test_a_complete_draft_of_a_lookup_that_found_nothing_is_told_unresolved(self):
        def complete(arguments):
            return {**arguments, "outcome": "complete"}

        refused, conflicts, _model = self._refused(complete)
        # The outcome to send and what the cited source observed.
        self.assertEqual(
            refused["errors"],
            [
                "unresolved_sources_marked_complete",
                'outcome: expected="unresolved"',
                *json.loads(conflicts),
            ],
        )


class _ResultIdCitingModel(SequentialAdaptiveModel):
    """Cites its lookup's result_id, not its event id, in the first assembly
    report call (as #389-#401 members did), then the event id."""

    def __init__(self) -> None:
        super().__init__()
        self.reports = 0
        self.corrections: list[str] = []

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        for message in messages[-1:]:
            self.corrections.extend(corrections(message))
        response = await super().respond(messages, info)
        (part,) = response.parts
        if part.tool_name != "fetch_assembly_report":
            return response
        self.reports += 1
        if self.reports > 1:
            return response
        arguments = part.args_as_dict()
        arguments["refsnp_event_id"] = "refsnp-result:not-an-event-id"
        return ModelResponse(
            parts=[
                ToolCallPart(
                    part.tool_name,
                    canonical_json(arguments),
                    tool_call_id=part.tool_call_id,
                )
            ]
        )


class AssemblyReportRefusalTests(unittest.TestCase):
    """An assembly report call that cites no lookup of the member is told the
    member's lookup event ids; it is never told to look up again (#380, #372)."""

    def test_a_result_id_cited_for_an_existing_lookup_names_its_event_id(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _ResultIdCitingModel()
            projection, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                SourceResponsesOnly(refsnp_body=RS4988235),
                assertion=_assertion("rs4988235"),
            )
            events = read_evidence(Journey.open(journey.path))
        (lookup,) = [
            event["event_id"]
            for event in events
            if event["event_type"] == "RefSnpToolResultRecorded"
        ]
        (refused,) = [
            event["payload"]
            for event in events
            if event["event_type"] == "ToolCallValidated"
            and event["payload"]["status"] == "invalid"
        ]
        self.assertEqual(
            refused["errors"], ["fetch_assembly_report_refsnp_event_unknown"]
        )
        (correction,) = model.corrections
        self.assertIn(
            '`refsnp_event_id` is "refsnp-result:not-an-event-id": No lookup result '
            "of this member has that event id. Send the event id of one of this "
            f"member's lookup results, not its result_id: `{lookup}` (resolved).",
            correction,
        )
        self.assertNotIn("Call fetch_refsnp first", correction)
        self.assertEqual(projection.status, "review_pending")


class _ChecksUnresolvedLookupModel(SequentialAdaptiveModel):
    """Asks for an assembly report of a lookup that found nothing, once."""

    def __init__(self) -> None:
        super().__init__()
        self.asked = False
        self.corrections: list[str] = []

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        for message in messages[-1:]:
            self.corrections.extend(corrections(message))
        _request, refsnp, _assembly, basis, *_ = _current_member_messages(messages)
        if refsnp is None or self.asked:
            return await super().respond(messages, info)
        self.asked = True
        self.calls += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "fetch_assembly_report",
                    canonical_json(
                        {
                            "refsnp_event_id": basis["refsnp_event_id"],
                            "assembly_accession": "GCF_000001405.40",
                        }
                    ),
                    tool_call_id="unresolved-assembly-check",
                )
            ]
        )


class UnresolvedLookupAssemblyRefusalTests(unittest.TestCase):
    def test_an_unresolved_lookup_is_told_its_status_and_the_output_to_return(self):
        # The refusal names the lookup's status and the output that ends the
        # member; code copies the conflicts.
        with tempfile.TemporaryDirectory() as directory:
            model = _ChecksUnresolvedLookupModel()
            _projection, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                IssueSourceResponses(),
                assertion=_assertion(NOT_FOUND),
            )
            events = read_evidence(Journey.open(journey.path))
        (lookup,) = [
            event["payload"]["result"]
            for event in events
            if event["event_type"] == "RefSnpToolResultRecorded"
        ]
        (correction,) = model.corrections
        self.assertIn(
            f"That lookup result is {lookup['status']}, so no assembly report checks "
            'it. Return compile_proposal with outcome "unresolved", citing this '
            "refsnp_event_id.",
            correction,
        )


class _FirstCallsModel(SequentialAdaptiveModel):
    """Sends these calls first, one per reply, then works the member as usual;
    keeps every correction it reads."""

    def __init__(self, *calls) -> None:
        super().__init__()
        self.first = list(calls)
        self.corrections: list[str] = []

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        for message in messages[-1:]:
            self.corrections.extend(corrections(message))
        if not self.first:
            return await super().respond(messages, info)
        name, arguments = self.first.pop(0)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    name,
                    canonical_json(arguments),
                    tool_call_id=f"first-{len(self.first)}",
                )
            ]
        )


def _position_member(**position):
    from bio_neb.investigation.investigation_request import StatedPosition

    return UserAssertion.create_collection(
        asserted_by="paper:doi:10.1000/example",
        predecessor_ids=("offline-research-request",),
        requests=(
            InvestigationRequestItem(
                request_text="The paper reports chr19:44908684.",
                position=StatedPosition(**position),
            ),
        ),
    )


class MemberLookupRefusalTests(unittest.TestCase):
    """A refused member lookup names the member's own stated target and the call
    to make, never the argument dict it was sent (#409, #372 map G2)."""

    def _corrections(self, model, assertion, sources=None):
        with tempfile.TemporaryDirectory() as directory:
            run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources or SourceResponsesOnly(),
                assertion=assertion,
            )
        return model.corrections

    def test_an_rsid_lookup_of_a_position_member_is_shown_its_position(self):
        model = _FirstCallsModel(("fetch_refsnp", {"rsid": "rs429358"}))
        corrections = self._corrections(
            model,
            _position_member(contig="chr19", position=44908684, ref="T", alt="C"),
        )
        self.assertIn(
            '`stated position` is "chr19:44908684 T>C": A member whose request '
            "states a position or a region is looked up with `locate_variant()`, "
            "which reads a position's rsID and follows a dbSNP merge too.",
            corrections[0],
        )

    def test_a_location_of_a_member_with_no_position_names_its_lookup(self):
        model = _FirstCallsModel(("locate_variant", {}))
        corrections = self._corrections(model, _assertion("rs4988235"))
        self.assertIn(
            "`stated position` has no value: This member's request states no "
            "position or region, so locate_variant does not apply; look up "
            "`rs4988235` with `fetch_refsnp` or `fetch_eva_variant`, the source that "
            "suits the Run's organism.",
            corrections[0],
        )
        self.assertNotIn("is {}", corrections[0])

    def test_a_member_with_only_words_is_told_to_read_them_before_a_lookup(self):
        # The target-missing refusal is retired: a member stating its target only
        # in words is read by the model first, so a lookup before the reading is
        # refused naming the reading Tool and the words' event.
        model = _FirstCallsModel(("fetch_refsnp", {"rsid": "rs1"}))
        assertion = UserAssertion.create_collection(
            asserted_by="researcher",
            predecessor_ids=("offline-research-request",),
            requests=(InvestigationRequestItem(request_text="Is HG001 at risk?"),),
        )
        (correction,) = self._corrections(model, assertion)
        self.assertIn("Refused before execution (target_words_unread).", correction)
        self.assertIn("call record_target_reading for each of", correction)
        self.assertNotIn("investigation_target_missing", correction)

    def test_a_position_located_with_no_source_is_told_to_name_one(self):
        """A position's rsIDs are listed by the variant source the model names (#469,
        owner Q1 a, ADR 0049; it supersedes #366's route): the refusal names both
        sources and what each keeps, never the empty argument dict it was sent."""
        model = _FirstCallsModel(("locate_variant", {}))
        corrections = self._corrections(
            model,
            _position_member(contig="chr19", position=44908684, ref="T", alt="C"),
        )
        self.assertIn(
            "`source` has no value: A stated position's rsIDs are listed by the "
            'variant source you name: source="ncbi_dbsnp" (dbSNP: human variants) '
            'or source="eva" (EVA: non-human variants).',
            corrections[0],
        )
        self.assertNotIn("is {}", corrections[0])

    def test_evas_variant_type_sent_with_dbsnp_names_the_value(self):
        """variant_type is EVA's own option (#469, owner R1 a, SP4 a): sent with dbSNP,
        the refusal names the option and its value and the source it goes with."""
        model = _FirstCallsModel(
            ("locate_variant", {"source": "ncbi_dbsnp", "variant_type": "SNV"})
        )
        corrections = self._corrections(
            model,
            _position_member(contig="chr19", position=44908684, ref="T", alt="C"),
        )
        self.assertIn(
            '`EVA option` is "variant_type=SNV": variant_type and contig_naming are '
            'EVA\'s own options for its GETs; send them only with source="eva".',
            corrections[0],
        )


class MemberQuestionRefusalTests(unittest.TestCase):
    def test_a_question_not_needed_names_the_state_that_holds(self):
        # Dormant: the value is the member's state, not the question sent.
        from bio_neb.agentic_journey.action_policy import ProposedAgentAction
        from bio_neb.agentic_journey.tool_execution import function_tool_contracts
        from bio_neb.failure import sentence_with_facts
        from bio_neb.investigation.investigation_agent import (
            investigation_action_errors,
            investigation_refusal_facts,
        )

        (row,) = [
            item
            for contract in function_tool_contracts("investigation_planner")
            for item in contract.preconditions
            if item.code == "investigation_question_not_needed"
        ]
        action = ProposedAgentAction(
            "request_investigation_information",
            "call",
            {
                "question": "Which allele?",
                "missing_information": "x",
                "asks_about": "target",
            },
        )
        for held, value, reason in (
            (
                {"question_event": {"event_id": "q"}, "validation_event_id": None},
                "question pending",
                "this member's question is pending; its answer comes back on a "
                "later request.",
            ),
            (
                {"question_event": None, "validation_event_id": "v"},
                "output accepted",
                "this member's output is already accepted.",
            ),
        ):
            with self.subTest(value=value):
                deps = SimpleNamespace(
                    investigation_input=SimpleNamespace(
                        assembly_accession="GCF_000001405.40"
                    ),
                    request=InvestigationRequestItem(
                        request_text="Does HG001 carry rs429358?", rsid="rs429358"
                    ),
                    researcher_answers=(),
                    raw_answers=(),
                    target_reads=[],
                    started_event_id="started",
                    source_events=[],
                    **held,
                )
                (code,) = investigation_action_errors(deps, action)
                facts = investigation_refusal_facts(deps, action, code)
                self.assertEqual((row.slot, facts["value"]), ("member state", value))
                self.assertEqual(
                    sentence_with_facts(row.explanation, facts),
                    f"No researcher question is needed: {reason}",
                )


class EvaLookupRefusalTests(unittest.TestCase):
    def test_an_eva_lookup_of_another_rsid_names_the_stated_ones(self):
        from bio_neb.agentic_journey.action_policy import ProposedAgentAction
        from bio_neb.agentic_journey.tool_execution import function_tool_contracts
        from bio_neb.failure import sentence_with_facts
        from bio_neb.investigation.investigation_agent import (
            investigation_action_errors,
            investigation_refusal_facts,
        )

        request = InvestigationRequestItem(
            request_text="Does the sample carry rs5334475172?", rsid="rs5334475172"
        )
        deps = SimpleNamespace(
            investigation_input=SimpleNamespace(assembly_accession="GCF_000003025.6"),
            request=request,
            researcher_answers=(),
            raw_answers=(),
            target_reads=[],
            started_event_id="started",
            source_events=[],
        )
        action = ProposedAgentAction("fetch_eva_variant", "call", {"rsid": "rs1"})
        (code,) = investigation_action_errors(deps, action)
        (sentence,) = {
            item.explanation
            for contract in function_tool_contracts("investigation_planner")
            if contract.tool_name == "fetch_eva_variant"
            for item in contract.preconditions
            if item.code == code
        }
        self.assertEqual(code, "eva_query_not_requested")
        self.assertEqual(
            sentence_with_facts(
                sentence, investigation_refusal_facts(deps, action, code)
            ),
            "Look up only an rsID this member's researcher stated (in the request "
            "or a retained answer), a current rsID one of this member's merged "
            "results names in merged_into: `rs5334475172`, or an rsID one of this "
            "member's lookup results' source_answers names as an id.",
        )


class FollowUpAdmissionTests(unittest.TestCase):
    """A lookup may follow an rsID a source's own answer names (owner #468 Q3 a, ADR
    0049): admission checks only that one of this member's retained answers names it as
    an id, as its source writes it. Bodies are synthetic, in EVA's and NCBI's shapes."""

    def _errors(self, tool, rsid, answers):
        from bio_neb.agentic_journey.action_policy import ProposedAgentAction
        from bio_neb.investigation.investigation_agent import (
            investigation_action_errors,
        )

        deps = SimpleNamespace(
            investigation_input=SimpleNamespace(
                assembly_accession="GCF_000003025.6"
                if tool == "fetch_eva_variant"
                else "GCF_000001405.40"
            ),
            request=InvestigationRequestItem(
                request_text="Does the sample carry rs160719159?", rsid="rs160719159"
            ),
            researcher_answers=(),
            raw_answers=(),
            target_reads=[],
            started_event_id="started",
            source_events=[
                {
                    "event_type": "RefSnpToolResultRecorded",
                    "event_id": "lookup-1",
                    "payload": {
                        "result": {"status": "source_answered"},
                        "source_answers": [
                            {"operation": operation, "media": media, "body": body}
                            for operation, media, body in answers
                        ],
                    },
                }
            ],
        )
        return investigation_action_errors(
            deps, ProposedAgentAction(tool, "call", {"rsid": rsid})
        )

    def test_an_rsid_an_answer_names_as_an_id_is_admitted(self) -> None:
        cases = {
            "EVA's clustered accession": (
                "eva_clustered_variant",
                "json",
                [{"accession": 7, "version": 1}],
            ),
            "EVA's merge destination": (
                "eva_clustered_variant",
                "json",
                [{"accession": 160719159, "mergedInto": 7}],
            ),
            "a submitted variant's cluster": (
                "eva_submitted_variants",
                "json",
                [{"accession": 900, "data": {"clusteredVariantAccession": 7}}],
            ),
            "dbSNP's merged_into as strings": (
                "fetch_refsnp",
                "json",
                {"merged_snapshot_data": {"merged_into": ["7"]}},
            ),
            "an rsID in a JSON string": (
                "fetch_refsnp",
                "json",
                {"error": {"message": "merged into rs7."}},
            ),
            "an rsID in a text answer": (
                "eva_clustered_variant",
                "text",
                "rs160719159 was merged into (rs7).",
            ),
        }
        for name, answer in cases.items():
            for tool in ("fetch_eva_variant", "fetch_refsnp"):
                with self.subTest(name, tool=tool):
                    self.assertEqual(self._errors(tool, "rs7", [answer]), ())

    def test_an_rsid_no_answer_names_as_an_id_is_refused(self) -> None:
        """Membership is in the fields the source writes an rsID in, by its own record
        (Standards r1 ST1, Spec r1 SP1, r2 SP1): a coordinate, an HTTP code, a taxon's
        or a submitted variant's own accession, or a longer token names none."""
        cases = {
            "a coordinate": (
                "eva_clustered_variant",
                "json",
                [{"accession": 160719159, "data": {"start": 7}}],
            ),
            "EVA's taxon accession": (
                "eva_clustered_variant",
                "json",
                [{"accession": 160719159, "data": {"taxonomyAccession": 7}}],
            ),
            "a submitted variant's own accession": (
                "eva_submitted_variants",
                "json",
                [{"accession": 7, "data": {"clusteredVariantAccession": 160719159}}],
            ),
            "an HTTP status in the body": (
                "fetch_refsnp",
                "json",
                {"error": {"code": 7, "message": "RefSNP not found"}},
            ),
            "a suffixed token in text": (
                "eva_clustered_variant",
                "text",
                "See rs7other and ars7x.",
            ),
            "a suffixed token in a JSON string": (
                "fetch_refsnp",
                "json",
                {"message": "rs7x moved"},
            ),
        }
        for name, answer in cases.items():
            for tool, code in (
                ("fetch_eva_variant", "eva_query_not_requested"),
                ("fetch_refsnp", "fetch_refsnp_query_not_requested"),
            ):
                with self.subTest(name, tool=tool):
                    self.assertEqual(self._errors(tool, "rs7", [answer]), (code,))

    def test_a_lookup_without_its_retained_answers_is_an_integrity_fault(self) -> None:
        """A recorded lookup always keeps its source answers (#458): a payload without
        them is no empty evidence set (Standards r1 ST2; ADR 0049 Q7 a, no compatibility
        reading)."""
        from bio_neb.investigation.investigation_agent import named_in_answers

        with self.assertRaises(KeyError):
            named_in_answers("rs7", [{"payload": {}}])


if __name__ == "__main__":
    unittest.main()
