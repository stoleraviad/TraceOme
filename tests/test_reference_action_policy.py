"""Contact-free reference readers and the reference Tools' preconditions."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.agentic_journey.action_policy import PreconditionUnmet
from bio_neb.journey import Journey
from bio_neb.parabricks.planning_commands import (
    RespondPlanningHitlCommand,
    respond_planning_hitl,
)
from bio_neb.parabricks.planning_hitl import ClarifyPlanningHitlResponse
from bio_neb.recording import read_evidence
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_RULE,
    SAMPLE_IDENTIFIER_RULE,
)
from bio_neb.reference_discovery.reference_journey import statement_states
from bio_neb.reference_discovery.reference_tools import reference_tool_values
from tests.planning_assignment_fakes import eligible_tools
from tests.reference_fakes import ReferenceAdapter
from tests.reference_query_fakes import (
    ChoosingPlanning,
    UnverifiedReferenceAdapter,
    planning_run,
    run_planning_attempt,
)


def event(event_id, event_type, payload):
    return {"event_id": event_id, "event_type": event_type, "payload": payload}


class StatementMembershipTests(unittest.TestCase):
    """A value the model typed from a researcher statement is admitted only when the
    statement it cites holds its quote and the quote states it (#472 Q2 a, ADR 0049
    membership); no code reads the value out of the statement."""

    EVENTS = (
        event(
            "prompt",
            "ResearchRequestRegistered",
            {"prompt": "Align to GCF_000001405.40 for sample HG001."},
        ),
        event(
            "answer",
            "PlanningHitlResponded",
            {"response": {"action": "clarify", "instruction": "Use the latest one."}},
        ),
        event(
            "revision",
            "ExecutionHitlResponded",
            {
                "response": {
                    "action": "revise",
                    "revalidation_scope": "reference",
                    "instruction": "Use GCF_000001405.39 instead.",
                }
            },
        ),
        event("plan", "ProposalRequirementsIdentified", {"plan": {}}),
    )

    def test_a_cited_statement_that_writes_the_value_admits_it(self):
        for event_id, quote, accession in (
            ("prompt", "GCF_000001405.40", "GCF_000001405.40"),
            ("revision", "Use GCF_000001405.39", "GCF_000001405.39"),
        ):
            with self.subTest(event_id=event_id):
                self.assertTrue(
                    statement_states(
                        self.EVENTS, event_id, quote, ASSEMBLY_ACCESSION_RULE, accession
                    )
                )
        self.assertTrue(
            statement_states(
                self.EVENTS, "prompt", "sample HG001", SAMPLE_IDENTIFIER_RULE, "HG001"
            )
        )

    def test_a_value_the_cited_statement_does_not_write_is_refused(self):
        for event_id, quote, accession in (
            # The answer writes no accession; the request's is not its own.
            ("answer", "GCF_000001405.40", "GCF_000001405.40"),
            # The quote stands, but states another accession.
            ("prompt", "GCF_000001405.40", "GCF_000001405.39"),
            # A planning fact is no researcher statement.
            ("plan", "GCF_000001405.40", "GCF_000001405.40"),
        ):
            with self.subTest(event_id=event_id, accession=accession):
                self.assertFalse(
                    statement_states(
                        self.EVENTS, event_id, quote, ASSEMBLY_ACCESSION_RULE, accession
                    )
                )


def _planning_events(candidate="first", adapter=None, reason=None):
    """One real planning Attempt whose scripted model chooses from discovery."""

    with tempfile.TemporaryDirectory() as directory:
        journey_file, composition = planning_run(
            Path(directory).resolve(), ChoosingPlanning
        )
        composition.candidate = candidate
        composition.reason = reason or composition.reason
        composition.reference_adapter = adapter or ReferenceAdapter(ambiguous=True)
        run_planning_attempt(journey_file, composition)
        return read_evidence(Journey.open(Path(journey_file)))


def _through(events, event_type):
    """The prefix ending at the first event of this type."""

    index = next(i for i, e in enumerate(events) if e["event_type"] == event_type)
    return events[: index + 1]


def _event(events, event_type):
    return next(e for e in events if e["event_type"] == event_type)


def _offers(prefix):
    """Each reference Tool's bound values at this prefix; None when it is refused.

    Every reference-stage Tool is shown; an unmet precondition refuses the call.
    """

    deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: prefix))

    def bound(name):
        try:
            return reference_tool_values(deps, name)
        except PreconditionUnmet:
            return None

    return {
        name: bound(name)
        for name in (
            "request_planning_hitl",
            "resolve_taxon",
            "discover_assemblies",
            "inspect_reference",
        )
    }


class SuppliedStatementScopeTests(unittest.TestCase):
    """#472 Q2 a: the model types a supplied accession from a researcher statement it
    cites. A reference question's answer is one, even when the question opened on a
    discovery page before any choice (review r2 SP3); another Review's answer is not,
    so it never changes the offered call (review r1 W1, the pending-retry guarantee of
    `test_run_observation_application…test_target_answer_before_provider_retry_preserves_current_tool_ancestry`)."""

    def test_a_reference_answer_before_any_choice_can_be_cited_and_others_cannot(
        self,
    ):
        import copy

        from bio_neb.parabricks.planning_hitl import make_planning_hitl_responded
        from bio_neb.reference_discovery.reference_identity_agent import (
            _planning_request,
        )
        from tests.reference_fakes import (
            SAMPLE_REQUIREMENT_ID,
            append,
            attempt_start,
            build_timeline,
            payload,
        )

        events, stages = build_timeline()
        events = copy.deepcopy(events[: stages["discovery"]])
        events[0]["payload"]["prompt"] = "Find the reference I mean."
        deps = SimpleNamespace(
            journey=SimpleNamespace(read=lambda: events),
            attempt_id=attempt_start(events, "reference-question-before-choice"),
        )
        self.assertNotIn("ReferenceChoiceRecorded", [e["event_type"] for e in events])
        reference_tool_values(deps, "request_planning_hitl")
        request = _planning_request(deps)
        asked = append(
            events, "reference-question", request.event_type, payload(request)
        )
        answer = append(
            events,
            "reference-answer",
            "PlanningHitlResponded",
            payload(
                make_planning_hitl_responded(
                    asked["event_id"],
                    ClarifyPlanningHitlResponse(
                        action="clarify", instruction="Use GCF_000001405.40."
                    ),
                )
            ),
        )
        offered = reference_tool_values(deps, "record_supplied_reference").enums[
            "assertion_event_id"
        ]
        self.assertIn(answer["event_id"], offered)
        self.assertIn(events[0]["event_id"], offered)

        # The same words, answering a question about another requirement, state no
        # reference: that answer is never offered for citation.
        hostile = copy.deepcopy(events[:-2])
        scoped = SimpleNamespace(
            journey=SimpleNamespace(read=lambda: hostile), attempt_id=deps.attempt_id
        )
        sample = request.model_copy(
            update={
                "requirement_id": SAMPLE_REQUIREMENT_ID,
                "subject_key": "sample_identity",
            }
        )
        other = append(hostile, "sample-question", sample.event_type, payload(sample))
        unrelated = append(
            hostile,
            "sample-answer",
            "PlanningHitlResponded",
            payload(
                make_planning_hitl_responded(
                    other["event_id"],
                    ClarifyPlanningHitlResponse(
                        action="clarify", instruction="Use GCF_000001405.40."
                    ),
                )
            ),
        )
        self.assertNotIn(
            unrelated["event_id"],
            reference_tool_values(scoped, "record_supplied_reference").enums[
                "assertion_event_id"
            ],
        )


class SampleQuestionTests(unittest.TestCase):
    """#472 Q2 a (ADR 0049, ADR 0031: a dead end opens the stage's Review): code reads
    no sample out of the researcher's words, so the sample question stays open until
    a sample is attested whenever the researcher stated none, answered a question
    (the model reads whether the answer names one), or attestation refused what they
    stated (review r1 SP2)."""

    def test_a_sample_answer_naming_none_can_be_asked_again(self):
        from bio_neb.parabricks.planning_hitl import make_planning_hitl_responded
        from tests.reference_fakes import append, payload, unresolved_sample_review

        events, request_id = unresolved_sample_review()
        fact = make_planning_hitl_responded(
            request_id,
            ClarifyPlanningHitlResponse(
                action="clarify", instruction="I have not decided on a name yet."
            ),
        )
        append(events, "sample-answer-without-name", fact.event_type, payload(fact))
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: events))

        values = reference_tool_values(deps, "request_planning_hitl")

        self.assertEqual(
            values.constants["requirement_id"],
            events[-2]["payload"]["requirement_id"],
        )

    def test_the_sample_question_opens_until_a_sample_is_attested(self):
        from unittest.mock import patch

        from bio_neb.reference_discovery import reference_tools

        refused = SimpleNamespace(event_type="SampleAttestationRecorded")
        for case, sample, facts, attested, kind in (
            ("stated none", None, (), None, "sample_identity"),
            ("an answer", "planning_hitl", (), None, "sample_identity"),
            ("refused", "research_request", (refused,), None, "sample_identity"),
            # The prompt's statement is attested first: the taxon is asked next.
            ("prompt", "research_request", (), None, "taxon_identity"),
            ("attested", "planning_hitl", (refused,), "sample", "taxon_identity"),
        ):
            assertions = {
                "sample_identity": None
                if sample is None
                else SimpleNamespace(assertion_kind=sample),
                "taxon_identity": None,
            }
            view = SimpleNamespace(
                status="sample_identity",
                sample_event_id=attested,
                facts=facts,
                plan_event_id="plan",
            )
            with (
                self.subTest(case=case),
                patch.object(
                    reference_tools,
                    "reference_assertion",
                    lambda _deps, _view, which, a=assertions: a[which],
                ),
                # No cited paper's reading keeps an organism either.
                patch.object(reference_tools, "_paper_organism", lambda *_: None),
            ):
                self.assertEqual(
                    reference_tools.reference_question(None, view), (kind, "plan")
                )


class ReferenceQuestionOfferTests(unittest.TestCase):
    """The reference question opens only when the Agent's choice needs the researcher."""

    @classmethod
    def setUpClass(cls):
        cls.concrete = _planning_events()
        cls.ambiguous = _planning_events(candidate="none")
        cls.unverified = _planning_events(adapter=UnverifiedReferenceAdapter())
        cls.sole = _planning_events(adapter=ReferenceAdapter())

    def test_concrete_choice_offers_only_inspection_until_a_result_exists(self):
        offers = _offers(_through(self.concrete, "ReferenceChoiceRecorded"))

        self.assertIsNone(offers["request_planning_hitl"])
        self.assertIsNone(offers["resolve_taxon"])
        self.assertIsNone(offers["discover_assemblies"])
        self.assertIsNotNone(offers["inspect_reference"])

    def test_no_choice_over_a_multi_candidate_shortlist_offers_the_question(self):
        prefix = _through(self.concrete, "ReferenceAssembliesRecorded")
        self.assertEqual(len(prefix[-1]["payload"]["result"]["rows"]), 2)

        offer = _offers(prefix)["request_planning_hitl"]

        self.assertEqual(offer.constants["evidence_event_id"], prefix[-1]["event_id"])

    def test_a_single_row_page_offers_the_question_like_any_page(self):
        """Code judges no page: one listed row is no verdict that the reference is
        settled, and the model asks when what it read conflicts with the
        researcher's words (#471; ADR 0049)."""
        prefix = _through(self.sole, "ReferenceAssembliesRecorded")
        self.assertEqual(len(prefix[-1]["payload"]["result"]["rows"]), 1)

        offer = _offers(prefix)["request_planning_hitl"]

        self.assertEqual(offer.constants["evidence_event_id"], prefix[-1]["event_id"])

    def test_explicit_ambiguity_opens_the_question_citing_the_choice(self):
        prefix = _through(self.ambiguous, "ReferenceChoiceRecorded")
        choice = prefix[-1]
        self.assertIsNone(choice["payload"]["assembly_accession"])
        self.assertIsNotNone(_offers(prefix)["request_planning_hitl"])

        request = _event(self.ambiguous, "PlanningHitlRequested")["payload"]
        discovery = _event(self.ambiguous, "ReferenceAssembliesRecorded")

        self.assertEqual(
            request["evidence_event_ids"],
            [discovery["event_id"], choice["event_id"]],
        )
        self.assertIn(
            "Neither isolate is documented for the stated sample.", request["question"]
        )
        self.assertEqual(len(request["candidates"]), 2)
        self.assertIsNone(request.get("recommendation"))

    def test_a_new_discovery_after_a_clarify_answer_reopens_the_question(self):
        """A researcher's answer holds for the page its question cited; once the
        model reads another page, it may ask again on that page (#471 live a1 F3:
        a clarify answer must not close the reference for the rest of the Run)."""

        class Reasking(ChoosingPlanning):
            candidate = "none"

            async def respond(self, messages, info):
                tools = eligible_tools(self.journey, info.function_tools)
                answered = any(
                    e["event_type"] == "PlanningHitlResponded"
                    for e in self.journey.read()
                )
                pages = [
                    e
                    for e in self.journey.read()
                    if e["event_type"] == "ReferenceAssembliesRecorded"
                ]
                if answered and len(pages) == 1 and "discover_assemblies" in tools:
                    self.calls.append("discover_assemblies")
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "discover_assemblies",
                                {
                                    "taxid": pages[0]["payload"]["tool_input"]["taxon"][
                                        "ncbi_tax_id"
                                    ],
                                    "search_text": ["isolate-B"],
                                },
                                tool_call_id=f"assignment-{len(self.calls)}",
                            )
                        ]
                    )
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            journey_file, composition = planning_run(
                Path(directory).resolve(), Reasking
            )
            composition.reference_adapter = ReferenceAdapter(ambiguous=True)
            pending = run_planning_attempt(journey_file, composition)
            respond_planning_hitl(
                RespondPlanningHitlCommand(
                    journey_file=journey_file,
                    request_event_id=pending.planning_hitl.pending_request.request_event_id,
                    response=ClarifyPlanningHitlResponse(
                        action="clarify", instruction="The isolate-B assembly."
                    ),
                )
            )
            run_planning_attempt(journey_file, composition)
            events = read_evidence(Journey.open(Path(journey_file)))

        pages = [
            e["event_id"]
            for e in events
            if e["event_type"] == "ReferenceAssembliesRecorded"
        ]
        questions = [
            e["payload"]["evidence_event_ids"]
            for e in events
            if e["event_type"] == "PlanningHitlRequested"
        ]
        self.assertEqual(len(pages), 2)
        self.assertEqual(len(questions), 2)
        self.assertIn(pages[1], questions[1])

    def test_a_maximal_reason_is_cut_to_the_question_limit_and_kept_by_the_choice(
        self,
    ):
        reason = "Provenance " + "x" * 1989
        events = _planning_events(candidate="none", reason=reason)
        self.assertEqual(
            _event(events, "ReferenceChoiceRecorded")["payload"]["reason"], reason
        )

        question = _event(events, "PlanningHitlRequested")["payload"]["question"]

        self.assertEqual(len(question), 2000)
        self.assertTrue(question.endswith("x…"))
        self.assertIn("The planning Agent's reason: Provenance xxx", question)

    def test_unresolved_inspection_of_the_choice_opens_the_question_citing_it(self):
        choice = _event(self.unverified, "ReferenceChoiceRecorded")
        self.assertIsNone(
            _offers(_through(self.unverified, "ReferenceChoiceRecorded"))[
                "request_planning_hitl"
            ]
        )
        prefix = _through(self.unverified, "ReferenceDescriptionRecorded")
        description = prefix[-1]
        self.assertEqual(description["payload"]["result"]["resolution"], "absent")
        self.assertIsNotNone(_offers(prefix)["request_planning_hitl"])

        request = _event(self.unverified, "PlanningHitlRequested")["payload"]
        discovery = _event(self.unverified, "ReferenceAssembliesRecorded")

        self.assertEqual(
            request["evidence_event_ids"],
            [discovery["event_id"], choice["event_id"], description["event_id"]],
        )
        self.assertIn(choice["payload"]["reason"], request["question"])
        self.assertIn("GCF_000001405.40", request["issue"])


def _row(code, tool=None):
    from bio_neb.agentic_journey.tool_execution import function_tool_contracts

    (row,) = {
        (item.slot, item.explanation)
        for contract in function_tool_contracts("planning_agent")
        if tool in (None, contract.tool_name)
        for item in contract.preconditions
        if item.code == code
    }
    return row


class ReferenceRefusalTextTests(unittest.TestCase):
    """Each reference refusal names a readable value and one next action, from
    the same reads admission makes (#409, #372 map R- and C- rows)."""

    @classmethod
    def setUpClass(cls):
        cls.events = _planning_events()

    def _refused(self, event_type, name):
        prefix = _through(self.events, event_type)
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: prefix))
        with self.assertRaises(PreconditionUnmet) as refused:
            reference_tool_values(deps, name)
        return refused.exception

    def _sentence(self, unmet):
        from bio_neb.failure import sentence_with_facts

        _slot, explanation = _row(unmet.code)
        return sentence_with_facts(explanation, unmet.facts)

    def test_a_choice_before_discovery_has_no_discovery_to_name(self):
        unmet = self._refused("SampleAttestationRecorded", "record_reference_choice")
        self.assertEqual(
            (unmet.code, unmet.value), ("reference_discovery_not_open", None)
        )

    def test_a_question_not_needed_names_the_reference_tool_to_call(self):
        unmet = self._refused("SampleAttestationRecorded", "request_planning_hitl")
        self.assertEqual(unmet.code, "reference_question_not_needed")
        self.assertEqual(
            self._sentence(unmet),
            # record_supplied_reference stays callable: whether the researcher
            # supplied an accession is the model's to read.
            "No reference requirement needs the researcher now. Continue the "
            "reference work with `resolve_taxon`, `record_supplied_reference`.",
        )

    def test_a_supplied_accession_after_a_choice_names_the_inspection(self):
        unmet = self._refused("ReferenceChoiceRecorded", "record_supplied_reference")
        self.assertEqual(unmet.code, "reference_choice_already_recorded")
        self.assertEqual(
            self._sentence(unmet),
            "A reference choice is already recorded; a supplied accession is "
            "recorded before any choice, or after the researcher answers a "
            "question about the current choice. Continue the reference work with "
            "`inspect_reference`.",
        )

    def test_a_supplied_accession_after_verification_says_the_reference_is_done(
        self,
    ):
        unmet = self._refused(
            "CompanionObservationRecorded", "record_supplied_reference"
        )
        self.assertIn(
            "The reference is verified; the reference work is done.",
            self._sentence(unmet),
        )

    def test_a_supplied_accession_the_researchers_words_do_not_write_is_refused(self):
        """The model types a supplied accession from the researcher statement it cites
        (#472 Q2 a, ADR 0049 membership): a quote the statement does not hold, or one
        that writes no such accession, is refused with the next step to take."""
        from bio_neb.agentic_journey.action_policy import ProposedAgentAction
        from bio_neb.failure import sentence_with_facts
        from bio_neb.reference_discovery.reference_tools import reference_tool_errors

        prefix = _through(self.events, "SampleAttestationRecorded")
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: prefix))
        request = next(
            e for e in prefix if e["event_type"] == "ResearchRequestRegistered"
        )
        written = request["payload"]["prompt"].split()[0]

        def errors(quote):
            return reference_tool_errors(
                deps,
                ProposedAgentAction(
                    "record_supplied_reference",
                    "call",
                    {
                        "assembly_accession": "GCF_000009999.1",
                        "assertion_event_id": request["event_id"],
                        "quote": quote,
                        "reason": "The request names it.",
                    },
                ),
            )

        self.assertEqual(
            errors("words the researcher never wrote"), ("supplied_quote_ungrounded",)
        )
        (code,) = errors(written)
        self.assertEqual(code, "supplied_accession_not_in_quote")
        _slot, explanation = _row(code, "record_supplied_reference")
        self.assertEqual(
            sentence_with_facts(explanation, {}),
            "The accession must be one the quote writes, versioned, exactly as it "
            "writes it; when the researcher's words name no accession, continue from "
            "them, for example with `discover_assemblies`, or ask.",
        )

    def test_the_index_tools_open_with_the_verified_reference_then_wait_on_a_search(
        self,
    ):
        """A Galaxy index is the verified assembly's, searched before any companion
        is described (owner, #459 D1, 2026-10-30), and a read and a choice follow a
        search (owner #464; ADR 0040 preconditions)."""
        prefix = _through(self.events, "ReferenceDescriptionRecorded")
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: prefix))
        reference_tool_values(deps, "search_prebuilt_indexes")
        for name in ("read_prebuilt_index", "record_prebuilt_index_choice"):
            with self.subTest(tool=name):
                unmet = self._refused("ReferenceDescriptionRecorded", name)
                self.assertEqual(unmet.code, "prebuilt_index_unsearched")

    def test_finished_companion_work_says_execution_planning_opens_next(self):
        unmet = self._refused("PrebuiltIndexChosen", "describe_companion")
        self.assertEqual(unmet.code, "companion_work_complete")
        self.assertEqual(
            self._sentence(unmet),
            "Companion work is complete; the next request opens execution planning.",
        )

    def test_a_choice_quoting_words_the_researcher_never_wrote_is_refused(self):
        """A choice rests on the researcher's own words (#471 Q2 a): a quote that
        stands in no request or answer is refused, and so is a row named with no
        quote; a quote the request holds, of a row of the cited page, is admitted.
        No invented value steers the reference (the guarantee the retired discovery
        refinement test held, owner-approved retirement)."""
        from bio_neb.agentic_journey.action_policy import ProposedAgentAction
        from bio_neb.reference_discovery.reference_journey import researcher_words
        from bio_neb.reference_discovery.reference_tools import reference_tool_errors

        prefix = _through(self.events, "ReferenceAssembliesRecorded")
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: prefix))
        written = researcher_words(prefix)[0].split()[0]

        def errors(**arguments):
            return reference_tool_errors(
                deps,
                ProposedAgentAction(
                    "record_reference_choice",
                    "call",
                    {
                        "discovery_event_id": prefix[-1]["event_id"],
                        "assembly_accession": "GCF_000001405.40",
                        "reason": "Closest.",
                        **arguments,
                    },
                ),
            )

        self.assertEqual(
            errors(quote="the strain the researcher never named"),
            ("choice_quote_ungrounded",),
        )
        self.assertEqual(errors(), ("choice_quote_missing",))
        self.assertEqual(errors(quote=written), ())

    def test_an_accession_not_on_the_cited_page_is_told_to_read_another(self):
        from bio_neb.agentic_journey.action_policy import ProposedAgentAction
        from bio_neb.failure import sentence_with_facts
        from bio_neb.reference_discovery.reference_tools import (
            reference_refusal_facts,
            reference_tool_errors,
        )

        prefix = _through(self.events, "ReferenceAssembliesRecorded")
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: prefix))
        action = ProposedAgentAction(
            "record_reference_choice",
            "call",
            {
                "discovery_event_id": prefix[-1]["event_id"],
                "assembly_accession": "GCF_000001405.39",
                "quote": "Homo sapiens",
                "reason": "Closest.",
            },
        )
        (code,) = reference_tool_errors(deps, action)
        _slot, explanation = _row(code, "record_reference_choice")
        self.assertEqual(code, "assembly_not_on_discovery_page")
        self.assertEqual(
            sentence_with_facts(
                explanation, reference_refusal_facts(deps, action, code)
            ),
            "Name an accession that is a row of the discovery page you cite, or "
            '"none"; read another page with `discover_assemblies` and its '
            "page_token, or another query.",
        )


if __name__ == "__main__":
    unittest.main()
