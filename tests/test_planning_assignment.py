"""Current application work uses domain Tools in one adaptive assignment."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from bio_neb.agentic_journey.agent_work_review import (
    AgentWorkReviewResponseRefused,
    ContinueAgentWorkResponse,
    RejectAgentWorkResponse,
)
from bio_neb.agentic_journey.effort import ExtendAgentEffortResponse
from bio_neb.application.advancement import (
    AgentEffortReviewAnswer,
    AgentWorkReviewAnswer,
    AnswerReview,
    PlanningReviewAnswer,
    ResumeRun,
    ReviewAnswerRefused,
    advance,
    use_run_advancement_composition,
)
from bio_neb.application.planning_read import read_planning
from bio_neb.application.queries import (
    ReviewJourneyQuery,
    project_journey_events,
    review_journey,
)
from bio_neb.application.run_interaction import run_interaction
from bio_neb.journey import Journey
from bio_neb.parabricks.planning_hitl import SelectPlanningHitlResponse
from bio_neb.recording import read_evidence
from bio_neb.reference_discovery.reference_identity_tool import (
    TaxonomyAdapterObservation,
)
from tests.application_fakes import _planning_ready, _registered
from tests.model_endpoint_fakes import corrections
from tests.planning_assignment_fakes import PlanningComposition, eligible_tools
from tests.reference_fakes import ReferenceAdapter, reference_source


def _semantic(events):
    """Journey facts without the processing-Pass journal markers."""
    return tuple(
        e for e in events if e["event_type"] not in {"PassStarted", "PassClosed"}
    )


def _shown(tool_name, fact, returned_texts):
    """Whether the composition read ``fact`` back as its Tool row's view (#478
    G1.3); it keeps each return as its text."""
    import ast

    from bio_neb.agentic_journey.tool_execution import model_view_of, tool_model_view

    view = tool_model_view(model_view_of(tool_name), fact)
    return any(
        returned.startswith("{") and ast.literal_eval(returned) == view
        for returned in returned_texts
    )


class _DifferentRefusals(PlanningComposition):
    """Four different refused actions in a row."""

    async def respond(self, messages, info):
        response = await super().respond(messages, info)
        arguments = response.parts[0].args
        if len(self.calls) == 1:
            arguments["before_event_id"] = "None"
        elif len(self.calls) <= 4:
            requirements = {item["kind"]: item for item in arguments["requirements"]}
            if len(self.calls) == 2:
                requirements["sample_identity"]["subject_key"] = "SampleID"
                requirements["companion_asset"]["acquisition_route"] = (
                    "use_supplied_assertion"
                )
            elif len(self.calls) == 3:
                requirements["sample_identity"]["validation"] = {"kind": "none"}
            else:
                requirements["scientific_investigation"]["knowledge"] = {
                    "state": "asserted",
                    "exact_text": "variant_calls",
                }
        return response


_REFUSED_HISTORY_KEY = "search_journey: search_outside_run"


class _RefusedPlanning(PlanningComposition):
    """The same refused history read four times: three corrections, and the fourth
    failed turn in a row stops; later responses are the scripted valid work,
    refused only while ``refusals_left`` remains."""

    repeats_left = 4
    refusals_left = 0

    async def respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        if self.repeats_left:
            self.repeats_left -= 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "search_journey",
                        {
                            "kinds": ["ResearchRequestRegistered"],
                            "before_event_id": "None",
                        },
                        tool_call_id=f"refused-read-{self.repeats_left}",
                    )
                ]
            )
        response = await super().respond(messages, info)
        if self.refusals_left:
            self.refusals_left -= 1
            response.parts[0].args["unexpected"] = "refused"
        return response


_TAXID_ASSERTION = "Homo sapiens (TaxId 9606)"


class _TaxIdTaxonomy(ReferenceAdapter):
    """Taxonomy finds a candidate only for the listed queries and records each query.

    ``outcomes`` maps a query to the records taxonomy returns for it instead.
    """

    def __init__(self, resolvable=(), outcomes=None):
        super().__init__()
        self.resolvable = resolvable
        self.outcomes = outcomes or {}
        self.queries = []

    def resolve_taxon(self, query):
        self.queries.append(query)
        if query in self.resolvable:
            return super().resolve_taxon(query)
        self.contact_count += 1
        return TaxonomyAdapterObservation(
            records=self.outcomes.get(query, ()),
            source=reference_source("ncbi_taxonomy", "taxonomy/not-found"),
        )


class _TaxIdPlanning(PlanningComposition):
    """Asserts the organism with its TaxId and tries its own taxon ``terms`` in
    order, reading each result; asks the researcher once they are exhausted.

    ``premature`` holds ``(after, name, arguments)``: once the model's last call
    was ``after``, it sends that call, once, whatever its preconditions; arguments
    may be a function of the Journey, for a call that cites a recorded event.
    """

    terms = (_TAXID_ASSERTION, "9606")
    premature = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent_terms = []
        self.tools_at_question = None
        # What came back to the model: Tool results and refusal corrections.
        self.returned = []

    async def respond(self, messages, info):
        from pydantic_ai.messages import (
            ModelResponse,
            RetryPromptPart,
            ToolCallPart,
            ToolReturnPart,
        )

        from tests.reference_fakes import reference_const_arguments

        self.returned.extend(
            str(part.content)
            for part in messages[-1].parts
            if isinstance(part, ToolReturnPart | RetryPromptPart)
        )
        journey = Journey.open(Path(self.journey.path))
        view = read_planning(journey).reference
        tools = eligible_tools(self.journey, info.function_tools)

        def call(name, arguments):
            self.calls.append(name)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name, arguments, tool_call_id=f"assignment-{len(self.calls)}"
                    )
                ]
            )

        if self.premature and self.calls and self.calls[-1] == self.premature[0][0]:
            (_, name, arguments), self.premature = self.premature[0], self.premature[1:]
            return call(name, arguments(journey) if callable(arguments) else arguments)
        unsent = [t for t in self.terms if t not in self.sent_terms]
        if view is not None and view.resolved_taxon is None:
            if "resolve_taxon" in tools and unsent:
                self.sent_terms.append(unsent[0])
                return call("resolve_taxon", {"term": unsent[0]})
            if "request_planning_hitl" in tools and not unsent:
                self.tools_at_question = set(tools)
                return call(
                    "request_planning_hitl",
                    reference_const_arguments(tools["request_planning_hitl"]),
                )
        response = await super().respond(messages, info)
        if response.parts[0].tool_name == "record_proposal_requirements":
            taxon_requirement = next(
                item
                for item in response.parts[0].args["requirements"]
                if item["kind"] == "taxon_identity"
            )
            taxon_requirement["knowledge"] = {
                "state": "asserted",
                "value": 9606,
                "quote": _TAXID_ASSERTION,
            }
            taxon_requirement["acquisition_route"] = "use_supplied_assertion"
        return response


def _taxid_run(
    directory,
    *,
    supplied=False,
    resolvable=(),
    outcomes=None,
    terms=None,
    premature=(),
):
    """Advance a Run whose taxon assertion carries its TaxId, optionally with a
    supplied exact accession, against taxonomy that resolves only resolvable."""
    reference = " using reference GCF_000001405.40" if supplied else ""
    receipt, paths = _registered(
        Path(directory),
        prompt_text=(
            f"Analyze rs1234 for sample-001 {_TAXID_ASSERTION}{reference} "
            "and produce VCF."
        ),
    )
    composition = _TaxIdPlanning(receipt, paths)
    composition.terms = terms or composition.terms
    composition.premature = premature
    composition.reference_adapter = _TaxIdTaxonomy(
        resolvable=resolvable, outcomes=outcomes
    )
    with use_run_advancement_composition(composition):
        result = advance(ResumeRun(journey_file=receipt.journey_file))
    events = read_evidence(Journey.open(Path(receipt.journey_file)))
    return receipt, composition, result, events


def _taxon_requirement_id(events):
    return next(
        item["requirement_id"]
        for e in events
        if e["event_type"] == "ProposalRequirementsIdentified"
        for item in e["payload"]["plan"]["requirements"]
        if item["kind"] == "taxon_identity"
    )


class PlanningAssignmentTests(unittest.TestCase):
    def test_planning_requests_carry_the_runs_one_tool_choice(self):
        # Planning has no output Tool, yet nothing forces a call: every request
        # records the Run's one choice, auto.
        with tempfile.TemporaryDirectory() as directory:
            receipt, _composition = _planning_ready(Path(directory))
            requests = [
                e["payload"]["compatible_adapter_evidence"]["request"]
                for e in read_evidence(Journey.open(Path(receipt.journey_file)))
                if e["event_type"] == "ModelRequestRecorded"
            ]
        self.assertGreater(len(requests), 1)
        for number, request in enumerate(requests, start=1):
            with self.subTest(request=number):
                offered = {tool["name"] for tool in request["function_tools"]}
                self.assertTrue(offered - {"search_journey"})
                self.assertEqual(request["output_tools"], [])
                self.assertEqual(request["model_settings"].get("tool_choice"), "auto")

    def test_different_refused_planning_calls_are_failed_turns_in_a_row(self):
        # One failure rule counts replies, not keys. Four different refused
        # calls in a row are four failed turns: three corrections reach the
        # model and the fourth stops the Attempt for the Agent work Review.
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _DifferentRefusals(receipt, paths)
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(
                (result.status, result.failure_code),
                ("agent_work_review_pending", None),
            )
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            self.assertFalse(
                any(e["event_type"] in {"AttemptFailed", "RunFailed"} for e in events)
            )
            self.assertEqual(
                sum(e["event_type"] == "ModelRequestRecorded" for e in events), 4
            )
            validations = {
                e["event_id"]: e["payload"]
                for e in events
                if e["event_type"] == "ToolCallValidated"
            }
            refused = [
                e
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
                and e["payload"]["disposition"] == "singleton_refused"
            ]
            # Each key names its own call; the count is the reply's place in the
            # failed turns in a row.
            self.assertEqual(
                [
                    (
                        turn["payload"]["correction_key"].split(" @ ")[0],
                        turn["payload"]["correction_count"],
                    )
                    for turn in refused
                ],
                [
                    (
                        validations[turn["payload"]["validation_event_ids"][0]][
                            "tool_name"
                        ]
                        + ": "
                        + ", ".join(
                            validations[turn["payload"]["validation_event_ids"][0]][
                                "errors"
                            ]
                        ),
                        count,
                    )
                    for count, turn in enumerate(refused, start=1)
                ],
            )
            self.assertEqual(
                len({turn["payload"]["correction_key"] for turn in refused}), 4
            )
            self.assertTrue(all(turn["payload"]["feedback"] for turn in refused))
            self.assertEqual(
                [payload["status"] for payload in validations.values()],
                ["invalid"] * 4,
            )
            request = result.current_review.request
            self.assertEqual(
                (request.cause, request.source_event_id),
                ("failed_turns", refused[-1]["event_id"]),
            )

    def test_same_refused_planning_call_is_corrected_three_times_then_stops_for_review(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _RefusedPlanning(receipt, paths)
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(
                (result.status, result.failure_code),
                ("agent_work_review_pending", None),
            )
            self.assertEqual(composition.reference_adapter.contact_count, 0)
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            self.assertFalse(
                any(e["event_type"] in {"AttemptFailed", "RunFailed"} for e in events)
            )
            self.assertEqual(
                sum(e["event_type"] == "ModelRequestRecorded" for e in events), 4
            )
            validations = [e for e in events if e["event_type"] == "ToolCallValidated"]
            self.assertEqual(
                [(e["payload"]["status"], e["payload"]["errors"]) for e in validations],
                [("invalid", ["search_outside_run"])] * 4,
            )
            turns = [
                e
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
            ]
            refused = [e["payload"] for e in turns]
            # Three corrections of the same call reach the model; the fourth failed
            # turn in a row stops.
            self.assertEqual(
                [
                    (
                        turn["disposition"],
                        turn["correction_key"].split(" @ ")[0],
                        turn["correction_count"],
                    )
                    for turn in refused
                ],
                [
                    ("singleton_refused", _REFUSED_HISTORY_KEY, count)
                    for count in range(1, 5)
                ],
            )
            self.assertTrue(all(turn["feedback"] for turn in refused))
            # The stop is the review request's own terminal, not a failure.
            request = result.current_review.request
            (stopped,) = [e for e in events if e["event_type"] == "AttemptStopped"]
            self.assertEqual(
                stopped["payload"],
                {
                    "attempt_id": result.attempts[-1].attempt_id,
                    "code": "failed_turns",
                    "source_event_id": request.request_event_id,
                },
            )
            self.assertEqual(result.current_review.kind, "agent_work")
            self.assertEqual(
                (request.cause, request.agent_role, request.source_event_id),
                ("failed_turns", "planning_agent", turns[-1]["event_id"]),
            )
            self.assertEqual(
                [
                    (
                        item.event_id,
                        item.tool_name,
                        item.errors,
                        item.correction_key.split(" @ ")[0],
                        item.correction_count,
                    )
                    for item in request.faults
                ],
                [
                    (
                        e["event_id"],
                        "search_journey",
                        ("search_outside_run",),
                        _REFUSED_HISTORY_KEY,
                        count,
                    )
                    for count, e in enumerate(validations, start=1)
                ],
            )
            self.assertEqual(
                [item.action for item in request.legal_actions], ["continue", "reject"]
            )
            continue_effect, reject_effect = (
                item.effect for item in request.legal_actions
            )
            self.assertIn("new Attempt", continue_effect)
            self.assertIn("fresh count of failed turns", continue_effect)
            self.assertIn("no further Attempt of this role", reject_effect)
            reopened = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            self.assertEqual(reopened.status, "agent_work_review_pending")
            self.assertEqual(reopened.current_review.request, request)

            instruction = "Record the proposal requirements from the inspected inputs."
            answer = AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=request.request_event_id,
                answer=AgentWorkReviewAnswer(
                    kind="agent_work",
                    response=ContinueAgentWorkResponse(
                        action="continue", instruction=instruction
                    ),
                ),
            )
            # One more refusal in the new Attempt: its count of failed turns is fresh.
            composition.refusals_left = 1
            with use_run_advancement_composition(composition):
                resumed = advance(answer)
                self.assertEqual(resumed.status, "planning_hitl_pending")
                calls = len(composition.calls)
                after = read_evidence(Journey.open(Path(receipt.journey_file)))
                self.assertEqual(advance(answer).status, "planning_hitl_pending")
                with self.assertRaises(AgentWorkReviewResponseRefused) as conflict:
                    advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=request.request_event_id,
                            answer=AgentWorkReviewAnswer(
                                kind="agent_work",
                                response=RejectAgentWorkResponse(
                                    action="reject",
                                    reason="Stop planning.",
                                    asserted_by="researcher",
                                ),
                            ),
                        )
                    )
                self.assertEqual(
                    conflict.exception.code, "agent_work_review_response_conflict"
                )
                with self.assertRaises(ReviewAnswerRefused) as mismatch:
                    advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=request.request_event_id,
                            answer=AgentEffortReviewAnswer(
                                kind="agent_effort",
                                response=ExtendAgentEffortResponse(
                                    action="extend", additional_requests=1
                                ),
                            ),
                        )
                    )
                self.assertEqual(mismatch.exception.code, "review_answer_kind_mismatch")
            self.assertEqual(len(composition.calls), calls)
            self.assertEqual(
                _semantic(read_evidence(Journey.open(Path(receipt.journey_file)))),
                _semantic(after),
            )
            self.assertFalse(any(e["event_type"] == "RunFailed" for e in after))
            first, second = resumed.attempts[-2:]
            self.assertEqual(first.attempt_id, stopped["payload"]["attempt_id"])
            continued = next(
                e
                for e in after
                if e["event_type"] == "AttemptStarted"
                and e["payload"]["attempt_id"] == second.attempt_id
            )
            lifecycle = next(
                e for e in after if e["event_id"] == continued["parent_ids"][0]
            )
            response_event = next(
                e for e in after if e["event_type"] == "AgentWorkReviewResponded"
            )
            self.assertEqual(lifecycle["parent_ids"], [response_event["event_id"]])
            self.assertEqual(
                second.effective_invocation_policy.continuation.failed_turns, 4
            )
            second_validations = [
                e["payload"]["status"]
                for e in after
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["attempt_id"] == second.attempt_id
            ]
            self.assertEqual(second_validations[:2], ["invalid", "valid"])
            first_request = next(
                e
                for e in after
                if e["event_type"] == "ModelRequestRecorded"
                and e["payload"]["attempt_id"] == second.attempt_id
            )
            self.assertIn(instruction, json.dumps(first_request["payload"]))

    def test_availability_retry_after_agent_work_continuation_reenters_planning(self):
        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.agentic_journey.model_availability import (
            RetryModelAvailabilityResponse,
        )
        from bio_neb.application.advancement import ModelAvailabilityReviewAnswer

        class _RefusedThenOutage(_RefusedPlanning):
            outage = False

            async def respond(self, messages, info):
                if self.outage:
                    self.outage = False
                    raise ModelHTTPError(
                        503, "adaptive-planning", headers={"retry-after": "999"}
                    )
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _RefusedThenOutage(receipt, paths)
            with use_run_advancement_composition(composition):
                work_stop = advance(ResumeRun(journey_file=receipt.journey_file))
                self.assertEqual(work_stop.status, "agent_work_review_pending")
                composition.outage = True
                outage = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=work_stop.current_review.request_event_id,
                        answer=AgentWorkReviewAnswer(
                            kind="agent_work",
                            response=ContinueAgentWorkResponse(
                                action="continue",
                                instruction="Record the proposal requirements.",
                            ),
                        ),
                    )
                )
                self.assertEqual(outage.status, "model_availability_review_pending")
                pending = outage.model_availability_review.pending_request
                resumed = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=pending.request_event_id,
                        answer=ModelAvailabilityReviewAnswer(
                            kind="model_availability",
                            response=RetryModelAvailabilityResponse(action="retry"),
                        ),
                    )
                )
            # The retry re-enters the stopped planning work from the continued
            # Attempt's frontier, the recorded Agent work answer.
            self.assertEqual(resumed.status, "planning_hitl_pending")
            self.assertEqual(len(resumed.attempts), 3)
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            retry_start = next(
                e
                for e in events
                if e["event_type"] == "AttemptStarted"
                and e["payload"]["attempt_id"] == resumed.attempts[-1].attempt_id
            )
            lifecycle = next(
                e for e in events if e["event_id"] == retry_start["parent_ids"][0]
            )
            (retry_answer,) = [
                e
                for e in events
                if e["event_type"] == "ModelAvailabilityReviewResponded"
                and e["payload"]["request_event_id"] == pending.request_event_id
            ]
            self.assertEqual(lifecycle["parent_ids"], [retry_answer["event_id"]])

    def test_fault_history_spans_the_provider_retry_of_its_attempt(self):
        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.agentic_journey.model_availability import (
            RetryModelAvailabilityResponse,
        )
        from bio_neb.application.advancement import ModelAvailabilityReviewAnswer

        class _OutageBeforeLastRefusal(_RefusedPlanning):
            outage = True

            async def respond(self, messages, info):
                if self.repeats_left == 1 and self.outage:
                    self.outage = False
                    raise ModelHTTPError(
                        503, "adaptive-planning", headers={"retry-after": "999"}
                    )
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _OutageBeforeLastRefusal(receipt, paths)
            with use_run_advancement_composition(composition):
                outage = advance(ResumeRun(journey_file=receipt.journey_file))
                self.assertEqual(outage.status, "model_availability_review_pending")
                pending = outage.model_availability_review.pending_request
                stopped = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=pending.request_event_id,
                        answer=ModelAvailabilityReviewAnswer(
                            kind="model_availability",
                            response=RetryModelAvailabilityResponse(action="retry"),
                        ),
                    )
                )
            self.assertEqual(stopped.status, "agent_work_review_pending")
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            refused = [e for e in events if e["event_type"] == "ToolCallValidated"]
            # Three corrections before the outage and the stopping fourth after the
            # retry are one Attempt's history: the Review shows all four.
            self.assertEqual(
                [
                    (
                        item.event_id,
                        item.correction_key.split(" @ ")[0],
                        item.correction_count,
                    )
                    for item in stopped.current_review.request.faults
                ],
                [
                    (e["event_id"], _REFUSED_HISTORY_KEY, count)
                    for count, e in enumerate(refused, start=1)
                ],
            )
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).current_review.request,
                stopped.current_review.request,
            )

    def test_reentry_without_work_carrying_the_recorded_assignment_holds(self):
        import dataclasses

        from bio_neb.application.advancement import RunAdvancementRefused
        from bio_neb.application.agent_reentry import AgentReentryRefused

        class _WithoutPlanningWork(_RefusedPlanning):
            def planning_assignment(self, journey):
                # The coordinator offers no Agent work recorded as planning.
                return dataclasses.replace(
                    super().planning_assignment(journey), agent_role="other_work"
                )

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            with use_run_advancement_composition(_RefusedPlanning(receipt, paths)):
                work_stop = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(work_stop.status, "agent_work_review_pending")
            before = read_evidence(Journey.open(Path(receipt.journey_file)))
            composition = _WithoutPlanningWork(receipt, paths)
            with (
                use_run_advancement_composition(composition),
                self.assertRaises(RunAdvancementRefused) as refused,
            ):
                advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=work_stop.current_review.request_event_id,
                        answer=AgentWorkReviewAnswer(
                            kind="agent_work",
                            response=ContinueAgentWorkResponse(
                                action="continue", instruction="Continue."
                            ),
                        ),
                    )
                )
            self.assertEqual(
                (refused.exception.code, refused.exception.status),
                ("agent_reentry_unavailable", "run_held"),
            )
            self.assertIsInstance(refused.exception.__cause__, AgentReentryRefused)
            self.assertIn("'planning_agent'", refused.exception.failure.cause)
            self.assertEqual(composition.calls, [])
            # The answer is retained and the Run held; no Attempt starts from it.
            after = read_evidence(Journey.open(Path(receipt.journey_file)))
            self.assertEqual(after[: len(before)], before)
            self.assertEqual(
                [e["event_type"] for e in _semantic(after[len(before) :])],
                ["AgentWorkReviewResponded"],
            )
            self.assertEqual(after[-1]["event_type"], "PassClosed")
            self.assertEqual(after[-1]["payload"]["disposition"], "held")

    def test_rejected_invalid_action_review_retains_work_and_holds_planning(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _RefusedPlanning(receipt, paths)
            with use_run_advancement_composition(composition):
                stopped = advance(ResumeRun(journey_file=receipt.journey_file))
            retained = read_evidence(Journey.open(Path(receipt.journey_file)))
            answer = AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=stopped.current_review.request_event_id,
                answer=AgentWorkReviewAnswer(
                    kind="agent_work",
                    response=RejectAgentWorkResponse(
                        action="reject",
                        reason="The planning direction is wrong.",
                        asserted_by="researcher",
                    ),
                ),
            )
            with use_run_advancement_composition(composition):
                rejected = advance(answer)
                self.assertEqual(rejected.status, "finalization_review_pending")
                self.assertEqual(rejected.current_review.kind, "finalization")
                self.assertEqual(
                    rejected.finalization.pending_request.basis.outcome,
                    "agent_work_rejected",
                )
                self.assertEqual(run_interaction(rejected).frontier, "review_required")
                self.assertEqual(advance(answer).status, "finalization_review_pending")
                self.assertEqual(
                    advance(ResumeRun(journey_file=receipt.journey_file)).status,
                    "finalization_review_pending",
                )
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            self.assertEqual(events[: len(retained)], retained)
            self.assertEqual(
                [e["event_type"] for e in _semantic(events[len(retained) :])],
                ["AgentWorkReviewResponded", "FinalizationBasisCompiled"],
            )
            journey = Journey.open(Path(receipt.journey_file))
            with self.assertRaisesRegex(ValueError, "exact continue answer"):
                composition.planning_assignment(journey).run(
                    journey, environ=composition.environ
                )
            self.assertEqual(
                _semantic(read_evidence(Journey.open(Path(receipt.journey_file)))),
                _semantic(events),
            )
            # Only the four refused reads reached the model; nothing after the stop.
            self.assertEqual((composition.repeats_left, composition.calls), (0, []))

    def test_requirement_schema_types_each_kind_and_its_quote(self):
        import re

        class Correcting(PlanningComposition):
            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                if self.calls[-1] == "record_proposal_requirements":
                    tool = next(
                        t
                        for t in info.function_tools
                        if t.name == "record_proposal_requirements"
                    )
                    self.requirement_schema = tool.parameters_json_schema
                    output = next(
                        item
                        for item in response.parts[0].args["requirements"]
                        if item["kind"] == "scientific_investigation"
                    )
                    # Same distinction seen live, using synthetic prompt and local inputs.
                    # The key is valid; claiming the request states it is not.
                    output["subject_key"] = "variant_calls"
                    if len(self.calls) == 2:
                        output["knowledge"] = {
                            "state": "asserted",
                            "value": "rs1234",
                            "quote": "variant_calls",
                        }
                    else:
                        self.contacts_before_repair = (
                            self.reference_adapter.contact_count
                            + self.companion_adapter.contact_count
                        )
                return response

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = Correcting(receipt, paths)
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            definitions = composition.requirement_schema["$defs"]
            key = definitions["SampleIdentityRequirementDraft"]["properties"][
                "subject_key"
            ]
            self.assertIn("pattern", key)
            self.assertIsNotNone(re.fullmatch(key["pattern"], "variant_calls"))
            self.assertIsNone(re.fullmatch(key["pattern"], "Variant Calls"))
            self.assertIn("not a quotation", key["description"])
            # Each kind types its own value; the quote states it.
            self.assertEqual(
                definitions["AssertedTaxonDraft"]["properties"]["value"]["type"],
                "integer",
            )
            # Execution always produces BAM + VCF: no output kind is recorded.
            self.assertNotIn("AssertedOutputDraft", definitions)
            self.assertIn(
                "pattern", definitions["AssertedAccessionDraft"]["properties"]["value"]
            )
            self.assertEqual(
                sorted(definitions["AssertedSampleDraft"]["required"]),
                ["quote", "value"],
            )
            self.assertEqual(definitions["AssertedTextDraft"]["required"], ["quote"])
            quote = definitions["AssertedSampleDraft"]["properties"]["quote"][
                "description"
            ]
            for phrase in ("copied exactly", "whole words", "more than once"):
                self.assertIn(phrase, quote)
            self.assertNotIn(
                "quote", definitions["ToFindKnowledgeDraft"].get("required", [])
            )
            meaning = definitions["SampleIdentityRequirementDraft"]["properties"][
                "meaning"
            ]
            self.assertIn("normalized", meaning["description"])
            self.assertEqual(composition.contacts_before_repair, 0)
            self.assertNotEqual(result.status, "failed")
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            refused = [
                e
                for e in events
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["status"] == "invalid"
            ]
            self.assertEqual(len(refused), 1)
            self.assertEqual(
                refused[0]["payload"]["errors"],
                ["variant_calls:request_assertion_ungrounded"],
            )
            plans = [
                e for e in events if e["event_type"] == "ProposalRequirementsIdentified"
            ]
            self.assertEqual(len(plans), 1)
            output = next(
                item
                for item in plans[0]["payload"]["plan"]["requirements"]
                if item["kind"] == "scientific_investigation"
            )
            self.assertEqual(output["subject_key"], "variant_calls")
            self.assertEqual(output["value"], "rs1234")
            self.assertEqual(output["evidence"][0]["exact_text"], "rs1234")

    def test_each_slot_producer_tool_is_a_planning_tool(self):
        # The slot table names the Tools that supply each kind in its refusal
        # guidance; a renamed or retired Tool must not leave a stale name there.
        from bio_neb.agentic_journey.planning_assignment import _planning_tools
        from bio_neb.parabricks.slot_satisfaction import SLOT_ROWS

        named = {tool for row in SLOT_ROWS for tool in row.tools}
        self.assertTrue(named)
        self.assertEqual(named - set(_planning_tools()), set())

    def test_a_plan_missing_kinds_is_told_each_missing_kind(self):
        # Through the real admission: a schema-valid draft without two kinds is
        # refused by grounding, and the correction names exactly those.

        class Incomplete(PlanningComposition):
            def __init__(self, *args):
                super().__init__(*args)
                self.drafts = 0
                self.feedback = []

            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                if self.calls[-1] != "record_proposal_requirements":
                    return response
                self.drafts += 1
                retries = [
                    text for message in messages for text in corrections(message)
                ]
                if retries:
                    self.feedback.append(retries[-1])
                if self.drafts == 1:
                    args = response.parts[0].args
                    args["requirements"] = [
                        item
                        for item in args["requirements"]
                        if item["kind"] not in {"taxon_identity", "resource_profile"}
                    ]
                return response

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = Incomplete(receipt, paths)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(composition.drafts, 2)
            (feedback,) = composition.feedback
            self.assertIn(
                "(proposal_requirement_kinds_incomplete).\n"
                "Tool `record_proposal_requirements` was not run.",
                feedback,
            )
            self.assertIn(
                "still missing: `taxon_identity`, `resource_profile`.", feedback
            )

    def _first_draft_feedback(self, change):
        """The correction the model reads after its first draft, changed by
        ``change``, is refused; the second draft is the composition's own."""

        class Changed(PlanningComposition):
            def __init__(self, *args):
                super().__init__(*args)
                self.drafts = 0
                self.feedback = []

            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                if self.calls[-1] != "record_proposal_requirements":
                    return response
                self.drafts += 1
                retries = [
                    text for message in messages for text in corrections(message)
                ]
                if retries:
                    self.feedback.append(retries[-1])
                if self.drafts == 1:
                    change(response.parts[0].args["requirements"])
                return response

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = Changed(receipt, paths)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
        (feedback,) = composition.feedback
        return feedback

    def test_a_knowledge_without_its_state_is_told_the_kinds_states(self):
        # The field error lists the states this kind's knowledge takes, and
        # the requirement list is not also called empty because its one bad
        # item failed.
        def drop_state(requirements):
            # A recorded draft: one requirement, its knowledge without a
            # state.
            sample = next(
                item for item in requirements if item["kind"] == "sample_identity"
            )
            del sample["knowledge"]["state"]
            requirements[:] = [sample]

        feedback = self._first_draft_feedback(drop_state)
        self.assertIn(
            "knowledge names its state: one of 'asserted', 'to_find'.", feedback
        )
        self.assertNotIn("Unable to extract tag", feedback)
        self.assertNotIn("too_short", feedback)

    def test_a_to_find_citation_outside_the_scope_is_told_the_allowed_ids(self):
        # A keyed grounding refusal reads its own item's facts through the real
        # admission.
        def cite_unknown(requirements):
            item = next(
                item for item in requirements if item["knowledge"]["state"] == "to_find"
            )
            item["knowledge"]["evidence_event_ids"] = ["not-an-event"]

        feedback = self._first_draft_feedback(cite_unknown)
        self.assertIn(
            "to_find knowledge may cite only the research request, the registered "
            "request file, the input registration, the inspection or a recorded "
            "paper's events: `",
            feedback,
        )

    def _first_planning_task(self, instruction_form=None):
        from dataclasses import replace

        from pydantic_ai.messages import ModelMessagesTypeAdapter

        from bio_neb.agentic_journey import model_settings

        qwen = model_settings.QWEN_ENDPOINT_POLICY
        entries = (
            ()
            if instruction_form is None
            else (replace(qwen, instruction_form=instruction_form),)
        )
        registry = model_settings.model_registry().replacing(*entries)
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            with (
                model_settings.use_model_registry(registry),
                use_run_advancement_composition(PlanningComposition(receipt, paths)),
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            request = next(
                e["payload"]["compatible_adapter_evidence"]["request"]
                for e in read_evidence(Journey.open(Path(receipt.journey_file)))
                if e["event_type"] == "ModelRequestRecorded"
            )
        # The task message comes first; Nemotron's injected notices follow it.
        prompt = next(
            part.content
            for message in ModelMessagesTypeAdapter.validate_python(request["messages"])
            for part in message.parts
            if part.part_kind == "user-prompt"
        )
        return json.loads(prompt)["task"]

    def test_nemotrons_planning_message_states_planning_job(self):
        # Planning's own last message states its job: the request's sentence is
        # the overall question, recorded once as research_goal, never a target.
        # The job is each form's own planning block.
        from bio_neb.agentic_journey.role_instructions import role_instructions

        for form in ("nemotron_3_5_super", "nemotron_3_ultra"):
            with self.subTest(form=form):
                task = self._first_planning_task(form)
                selected = role_instructions("planning_agent", form=form)
                self.assertEqual(task, selected.message("job"))
                self.assertIn("never as a scientific_investigation target", task)
        # Qwen's bytes stay: its task is the objective every form's context
        # states.
        qwen = role_instructions("planning_agent", form="qwen_3_5")
        self.assertEqual(self._first_planning_task(), qwen.message("objective"))
        self.assertEqual(qwen.message("job"), qwen.message("objective"))

    def test_requirement_routes_expose_replay_safe_repair_guidance(self):
        class Correcting(PlanningComposition):
            def __init__(self, *args):
                super().__init__(*args)
                self.drafts = 0
                self.feedback = []
                self.contacts_before_repair = []

            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                if self.calls[-1] != "record_proposal_requirements":
                    return response
                self.drafts += 1
                requirement_tool = next(
                    tool
                    for tool in info.function_tools
                    if tool.name == "record_proposal_requirements"
                )
                self.requirement_schema = requirement_tool.parameters_json_schema
                self.requirement_description = requirement_tool.description
                retries = [
                    text for message in messages for text in corrections(message)
                ]
                if retries:
                    self.feedback.append(retries[-1])
                self.contacts_before_repair.append(
                    self.reference_adapter.contact_count
                    + self.companion_adapter.contact_count
                )
                requirements = response.parts[0].args["requirements"]
                for item in requirements:
                    if item["kind"] == "registered_input" and self.drafts == 1:
                        item["acquisition_route"] = "use_supplied_assertion"
                    if item["kind"] == "resource_profile":
                        item["knowledge"] = {"state": "asserted", "quote": "H100"}
                        item["acquisition_route"] = "use_supplied_assertion"
                        if self.drafts <= 2:
                            item["validation"] = {"kind": "none"}
                return response

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory),
                prompt_text="Analyze rs1234 for sample-001 Homo sapiens and produce VCF. Requested resources: an H100.",
            )
            composition = Correcting(receipt, paths)
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            # The cross-field rules are stated once, in the Tool's description.
            for rule in (
                "knowledge.state='asserted'",
                "registered_input",
                "validate_with_tool",
                "sample_identity",
                "taxon_identity",
                "reference_assembly_identity",
                "companion_asset",
                "analysis_recipe",
            ):
                self.assertIn(rule, composition.requirement_description)
            self.assertEqual(composition.drafts, 3)
            self.assertEqual(composition.contacts_before_repair, [0, 0, 0])
            self.assertEqual(result.status, "planning_hitl_pending")
            self.assertEqual(len(composition.feedback), 2)
            # The Tool's own description states the cross-field rules.
            for rule in (
                "knowledge.state='asserted'",
                "validation.kind='tool_evidence'",
                "resource_profile",
            ):
                self.assertIn(rule, composition.requirement_description)
            for feedback in composition.feedback:
                self.assertTrue(
                    feedback.startswith(
                        "Refused before execution (tool_arguments_invalid).\n"
                    ),
                    feedback,
                )
                self.assertIn("`record_proposal_requirements` was not run.", feedback)
                self.assertIn("- Field `requirements.8`: you sent {", feedback)
                # The paper Tools are the plan stage's on every Run: code cannot
                # read whether the words cite a paper.
                self.assertIn(
                    "Allowed now: one of `find_paper`, `record_proposal_requirements`, "
                    "`search_journey`.\n"
                    "Waiting on their step, not callable now: `request_planning_hitl`, "
                    "`retrieve_paper`.",
                    feedback,
                )
                self.assertIn("For record_proposal_requirements:", feedback)
                self.assertIn("resource_profile", feedback)
                self.assertIn("validation.kind='tool_evidence'", feedback)
                self.assertIn("knowledge.state='asserted'", feedback)
            # Each refusal names the shape no source supplies and its kind's
            # lawful routes.
            input_refusal = (
                "no Tool, fact or Review can ever supply registered_input with "
                "knowledge.state='observed', acquisition_route='use_supplied_assertion'"
            )
            self.assertIn(input_refusal, composition.feedback[0])
            self.assertNotIn(input_refusal, composition.feedback[1])
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            validations = [
                event
                for event in events
                if event["event_type"] == "ToolCallValidated"
                and event["payload"]["tool_name"] == "record_proposal_requirements"
            ]
            resource_error = (
                "tool_arguments_invalid:requirements.8:value_error:Value error, "
                "no Tool, fact or Review can ever supply resource_profile with "
                "knowledge.state='asserted', acquisition_route='use_supplied_assertion' "
                "and validation.kind='none'; valid for it: "
                "acquisition_route='ask_planning_hitl'/'use_supplied_assertion'/"
                "'validate_with_tool' with validation.kind='tool_evidence' "
                "(the select_resource_candidate Tool)"
            )
            self.assertEqual(
                validations[0]["payload"]["errors"],
                [
                    f"tool_arguments_invalid:requirements.{index}:value_error:"
                    "Value error, no Tool, fact or Review can ever supply "
                    "registered_input with knowledge.state='observed', "
                    "acquisition_route='use_supplied_assertion' and "
                    "validation.kind='none'; valid for it: "
                    "acquisition_route='ask_planning_hitl'/'derive_deterministically'/"
                    "'validate_with_tool' with validation.kind="
                    "'deterministic_derivation'/'none'/'researcher_decision'/"
                    "'tool_evidence' (the inspect_input_metadata Tool)"
                    for index in (1, 2)
                ]
                + [resource_error],
            )
            self.assertEqual(validations[1]["payload"]["errors"], [resource_error])
            self.assertEqual(validations[2]["payload"]["errors"], [])
            plan = next(
                event
                for event in events
                if event["event_type"] == "ProposalRequirementsIdentified"
            )["payload"]["plan"]
            resource = next(
                item
                for item in plan["requirements"]
                if item["kind"] == "resource_profile"
            )
            self.assertEqual(resource["value"], "H100")
            self.assertEqual(resource["validation"]["kind"], "tool_evidence")
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).status,
                result.status,
            )

    def test_a_clarified_taxon_resumes_and_explores_without_replacing_supplied_reference(
        self,
    ):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        from bio_neb.parabricks.planning_hitl import ClarifyPlanningHitlResponse
        from bio_neb.reference_discovery.reference_identity_tool import (
            TaxonomyAdapterObservation,
        )
        from tests.reference_fakes import ReferenceAdapter, reference_source

        class ClarifiedSource(ReferenceAdapter):
            def __init__(self):
                super().__init__()
                self.queries = []

            def resolve_taxon(self, query):
                self.queries.append(query)
                if len(self.queries) == 1:
                    self.contact_count += 1
                    return TaxonomyAdapterObservation(
                        records=(),
                        source=reference_source("ncbi_taxonomy", "taxonomy/not-found"),
                    )
                return super().resolve_taxon(query)

        class Correcting(PlanningComposition):
            resumed = False
            repairs = 0
            sent: list = []

            async def respond(self, messages, info):
                tools = eligible_tools(self.journey, info.function_tools)
                if not self.resumed and "request_planning_hitl" in tools:
                    name = "request_planning_hitl"
                    arguments = {
                        key: field["const"]
                        for key, field in tools[name]
                        .parameters_json_schema["properties"]
                        .items()
                        if "const" in field
                    }
                    self.calls.append(name)
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                name,
                                arguments,
                                tool_call_id=f"request-taxon-{len(self.calls)}",
                            )
                        ]
                    )
                response = await super().respond(messages, info)
                if (
                    self.resumed
                    and self.calls[-1] == "inspect_reference"
                    and self.repairs < 2
                ):
                    self.repairs += 1
                    arguments = response.parts[0].args
                    self.contacts_before_success = self.reference_adapter.contact_count
                    # An unrelated lookup stays evidence, not a replacement selection.
                    if self.repairs == 1:
                        arguments["accession"] = "GCF_000009999.1"
                    self.sent.append(dict(arguments))
                return response

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory),
                prompt_text="Analyze rs1234 for sample-001 Homo sapiens using reference GCF_000001405.40 and produce VCF.",
            )
            composition = Correcting(receipt, paths)
            source = ClarifiedSource()
            composition.reference_adapter = source
            with use_run_advancement_composition(composition):
                pending = advance(ResumeRun(journey_file=receipt.journey_file))
                self.assertEqual(pending.status, "planning_hitl_pending")
                before = read_evidence(Journey.open(Path(receipt.journey_file)))
                question = pending.planning_hitl.pending_request
                composition.resumed = True
                result = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=question.request_event_id,
                        answer=PlanningReviewAnswer(
                            kind="planning",
                            response=ClarifyPlanningHitlResponse(
                                action="clarify", instruction="9606"
                            ),
                        ),
                    )
                )
            self.assertEqual(composition.repairs, 2)
            self.assertEqual(
                composition.sent,
                [
                    {"accession": "GCF_000009999.1"},
                    {"accession": "GCF_000001405.40"},
                ],
            )
            # Two taxonomy lookups and the mismatched inspection reached the source.
            self.assertEqual(composition.contacts_before_success, 3)
            self.assertEqual(source.queries, ["Homo sapiens", "9606"])
            self.assertEqual(result.status, "planning_hitl_pending")
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            self.assertEqual(events[: len(before)], before)
            validations = [
                event
                for event in events[len(before) :]
                if event["event_type"] == "ToolCallValidated"
                and event["payload"]["tool_name"] == "inspect_reference"
            ]
            self.assertEqual(
                [event["payload"]["errors"] for event in validations[:2]],
                [[], []],
            )
            # The record keeps what the model sent; nothing is coerced.
            self.assertEqual(
                validations[0]["payload"]["arguments"],
                {"accession": "GCF_000009999.1"},
            )
            self.assertEqual(
                sum(
                    event["event_type"] == "SampleAttestationRecorded"
                    for event in events
                ),
                1,
            )
            self.assertEqual(
                sum(
                    event["event_type"] == "InputMetadataInspected" for event in events
                ),
                1,
            )
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).status,
                result.status,
            )

    def test_source_tool_transport_uses_its_active_child_and_preserves_source_failure(
        self,
    ):
        import copy
        import io
        import json
        import urllib.error
        from unittest.mock import patch

        from bio_neb.agentic_journey.source_contacts import source_contact_units
        from bio_neb.providers.ncbi_datasets import (
            NcbiDatasetsReferenceAdapter,
            UrllibNcbiTransport,
        )

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory),
                prompt_text="Analyze rs1234 for sample-001 Homo sapiens using reference GCF_000001405.40 and produce VCF.",
            )
            requests = []

            def open_source(request, *, timeout):
                requests.append(request.full_url)
                if "/taxonomy/" not in request.full_url:
                    raise urllib.error.HTTPError(
                        request.full_url, 503, "controlled upstream failure", {}, None
                    )
                response = io.BytesIO(
                    json.dumps(
                        {
                            "taxonomy_nodes": [
                                {
                                    "taxonomy": {
                                        "tax_id": 9606,
                                        "current_scientific_name": {
                                            "name": "Homo sapiens"
                                        },
                                        "rank": "species",
                                    }
                                }
                            ]
                        }
                    ).encode()
                )
                response.status = 200
                response.headers = {}
                return response

            with patch("time.sleep"), patch("urllib.request.build_opener") as opener:
                opener.return_value.open.side_effect = open_source
                source = NcbiDatasetsReferenceAdapter(UrllibNcbiTransport())
                composition = PlanningComposition(receipt, paths)
                composition.reference_adapter = source
                composition.companion_adapter = source
                with use_run_advancement_composition(composition):
                    result = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(len(requests), 4)
            self.assertEqual(result.status, "operational_review_pending")
            reopened = Journey.open(Path(receipt.journey_file))
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file),
                    journey=reopened,
                ),
                result,
            )
            events = read_evidence(reopened)
            scopes = [
                e for e in events if e["event_type"] == "SourceContactScopeStarted"
            ]
            starts = [
                e for e in events if e["event_type"] == "SourcePhysicalContactStarted"
            ]
            finishes = [
                e for e in events if e["event_type"] == "SourcePhysicalContactFinished"
            ]
            # One scope per source call: the taxon lookup, then the inspection.
            self.assertEqual(
                [e["payload"]["tool_name"] for e in scopes],
                ["resolve_taxon", "inspect_reference"],
            )
            self.assertEqual(
                [e["payload"]["scope_event_id"] for e in starts],
                [scopes[0]["event_id"], *([scopes[1]["event_id"]] * 3)],
            )
            self.assertEqual(
                [e["payload"]["disposition"] for e in finishes],
                ["succeeded", "failed", "failed", "failed"],
            )
            self.assertEqual(finishes[1]["payload"]["error_type"], "HTTPError")
            taxonomy = next(
                e for e in events if e["event_type"] == "TaxonResolutionRecorded"
            )
            self.assertEqual(taxonomy["payload"]["result"]["status"], "observed")
            description = next(
                e for e in events if e["event_type"] == "ReferenceDescriptionRecorded"
            )
            self.assertEqual(
                description["payload"]["result"]["failure"]["code"], "http_error"
            )
            self.assertTrue(description["payload"]["result"]["failure"]["retryable"])
            units, complete = source_contact_units(events)
            self.assertEqual(len(units), 4)
            self.assertTrue(complete)
            changed = copy.deepcopy(events)
            scope_finish = next(
                e for e in changed if e["event_type"] == "SourceContactScopeFinished"
            )
            next(
                e for e in changed if e["event_type"] == "SourcePhysicalContactStarted"
            )["payload"]["scope_event_id"] = scope_finish["event_id"]
            with self.assertRaisesRegex(ValueError, "outside its open scope"):
                source_contact_units(changed)
            self.assertEqual(len(requests), 4)

    def test_exact_asset_selection_resumes_description_without_repeating_reference(
        self,
    ):
        from tests.reference_fakes import CompanionAdapter

        class ExploringSelection(PlanningComposition):
            async def respond(self, messages, info):
                if self.calls.count("describe_companion") == 1:
                    from pydantic_ai.messages import ModelResponse, ToolCallPart

                    self.calls.append("describe_companion")
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "describe_companion",
                                {
                                    "asset": "genomic_fasta",
                                    "accession": "GCF_000000001.1",
                                },
                                tool_call_id=f"explore-{len(self.calls)}",
                            )
                        ]
                    )
                response = await super().respond(messages, info)
                if (
                    self.calls[-1] == "describe_companion"
                    and self.calls.count("describe_companion") == 3
                ):
                    response.parts[0].args["accession"] = "GCF_000000001.1"
                return response

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = ExploringSelection(receipt, paths)
            composition.companion_adapter = CompanionAdapter(
                "GCF_000001405.40", asset_ids=("asset-A", "asset-B")
            )
            with use_run_advancement_composition(composition):
                pending = advance(ResumeRun(journey_file=receipt.journey_file))
                question = next(
                    q
                    for q in pending.planning_hitl.pending_requests
                    if "asset" in q.question.lower()
                )
                self.assertEqual(
                    [c.asserted_value for c in question.candidates],
                    ["asset-A", "asset-B"],
                )
                result = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=question.request_event_id,
                        answer=PlanningReviewAnswer(
                            kind="planning",
                            response=SelectPlanningHitlResponse(
                                action="select",
                                candidate_id=question.candidates[0].candidate_id,
                            ),
                        ),
                    )
                )
            self.assertEqual(result.reference_facts.status, "reference_facts")
            self.assertEqual(composition.reference_adapter.contact_count, 3)
            self.assertEqual(composition.companion_adapter.contact_count, 4)
            # The selected asset is described again; the reference is not.
            self.assertEqual(composition.calls.count("inspect_reference"), 1)
            self.assertEqual(composition.calls.count("describe_companion"), 4)
            self.assertEqual(
                [f.result["outcome"] for f in result.reference_facts.facts],
                ["multiple", "incompatible", "incompatible", "verified"],
            )
            self.assertEqual(
                [
                    f.tool_input["selected_asset_id"]
                    for f in result.reference_facts.facts
                ],
                [None, None, "asset-A", "asset-A"],
            )
            self.assertEqual(
                result.reference_facts.facts[-1].result["candidates"][0]["asset_id"],
                "asset-A",
            )
            import copy

            from bio_neb.application.planning_read import read_planning_events

            forged = copy.deepcopy(
                read_evidence(Journey.open(Path(receipt.journey_file)))
            )
            last = next(
                e
                for e in reversed(forged)
                if e["event_type"] == "CompanionObservationRecorded"
            )
            last["payload"]["result"]["candidates"][0]["asset_id"] = "asset-B"
            with self.assertRaisesRegex(ValueError, "changed the selected asset"):
                read_planning_events(forged)

    def test_partial_companion_asks_exact_refinement_without_new_contact(self):
        import hashlib

        from bio_neb.journey import canonical_json

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            composition.companion_adapter = _partial_companion_adapter()
            with use_run_advancement_composition(composition):
                pending = advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            partial = next(
                e for e in events if e["event_type"] == "CompanionObservationRecorded"
            )
            started = next(
                e
                for e in events
                if e["event_type"] == "ObservedToolExecutionStarted"
                and e["payload"]["tool_name"] == "request_companion_refinement"
            )
            succeeded = next(
                e
                for e in events
                if e["event_type"] == "ObservedToolExecutionSucceeded"
                and e["payload"]["started_event_id"] == started["event_id"]
            )
            question = pending.planning_hitl.pending_request

            self.assertEqual(pending.status, "planning_hitl_pending")
            self.assertEqual(partial["payload"]["result"]["outcome"], "partial")
            self.assertEqual(question.evidence_event_ids, (partial["event_id"],))
            self.assertEqual(question.candidates, ())
            self.assertEqual(composition.companion_adapter.contact_count, 1)
            self.assertEqual(started["payload"]["adapter_identity"], "deterministic")
            self.assertEqual(
                succeeded["payload"]["stage_result_event_id"],
                question.request_event_id,
            )
            self.assertEqual(
                succeeded["payload"]["model_visible_result_digest"],
                hashlib.sha256(
                    # The question Tool's view, written out: the model is told the
                    # Attempt ends at its question.
                    canonical_json(
                        {
                            "status": "asked",
                            "next": "The Attempt ends at this question; the "
                            "researcher's answer comes back as your next "
                            "instruction.",
                        }
                    ).encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).status,
                pending.status,
            )

    def test_an_unexpected_tool_failure_is_shown_once_then_holds_the_run(self):
        """Owner, #477 C8 (alignment gate 2026-10-28): an undeclared Tool failure
        is shown to the model once as "the Tool failed: <type>" and the Attempt
        continues; the same Tool and exception type again holds the Run."""
        for tool_name, helper, companion in _FAILING_COMPANION_REQUESTS:
            with (
                self.subTest(tool_name=tool_name),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, _composition, refused = (
                    _advance_with_failing_companion_request(
                        Path(directory), helper, companion()
                    )
                )
                review = review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                )
                events = read_evidence(Journey.open(Path(receipt.journey_file)))
                started = [
                    e
                    for e in events
                    if e["event_type"] == "ObservedToolExecutionStarted"
                    and e["payload"]["tool_name"] == tool_name
                ]
                failed = [
                    e
                    for e in events
                    if e["event_type"] == "ObservedToolExecutionFailed"
                    and e["payload"]["started_event_id"]
                    in {item["event_id"] for item in started}
                ]
                shown = [e for e in events if e["event_type"] == "ToolExecutionRefused"]
                attempt_failed = [
                    e for e in events if e["event_type"] == "AttemptFailed"
                ]

                # The model retried the call it was told failed; it ran again.
                self.assertEqual(len(started), 2)
                self.assertEqual(
                    [item["payload"]["mechanical_error_type"] for item in failed],
                    ["RuntimeError", "RuntimeError"],
                )
                self.assertEqual(
                    [item["payload"] for item in shown],
                    [
                        {
                            "attempt_id": started[0]["payload"]["attempt_id"],
                            "failed_event_id": failed[0]["event_id"],
                            "reason": "tool_failed",
                            "statement": "the Tool failed: RuntimeError",
                            "fault_key": f"{tool_name}: RuntimeError",
                            # The failure module's next step and the reply's
                            # place, recorded as shown (F8).
                            "next": "Call another Tool callable now; the same "
                            "failure again holds the Run for the researcher.",
                            "count": "Count: failed turn 1 of 4 in a row. At 4 the "
                            "Attempt stops and the researcher reviews it.",
                        }
                    ],
                )
                (attempt_failed,) = attempt_failed
                self.assertEqual(
                    (
                        attempt_failed["payload"]["code"],
                        attempt_failed["payload"]["invariant_id"],
                        attempt_failed["payload"]["source_event_id"],
                    ),
                    ("invariant", "tool_failed_again", failed[1]["event_id"]),
                )
                self.assertEqual(
                    (refused.failure.invariant_id, refused.status, review.status),
                    ("planning_attempt_failed", "run_held", "run_held"),
                )
                self.assertFalse(
                    any(e["event_type"] == "PlanningHitlRequested" for e in events)
                )
                self.assertNotIn("RunFailed", [e["event_type"] for e in events])

    def test_a_planning_configuration_failure_holds_the_run_on_every_resume(self):
        # D5: each planning dispatch without its credential records one
        # RuntimeConfigurationFailed and holds the Run; nothing ends it.
        from bio_neb.application.advancement import RunAdvancementRefused
        from bio_neb.application.queries import ObserveRun, observe

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            journey_file = receipt.journey_file
            composition = PlanningComposition(receipt, paths)
            credential, composition.environ = composition.environ, {}
            refusals = []
            with use_run_advancement_composition(composition):
                for _ in range(2):
                    with self.assertRaises(RunAdvancementRefused) as refused:
                        advance(ResumeRun(journey_file=journey_file))
                    refusals.append(refused.exception)
            observed = observe(ObserveRun(journey_file=journey_file))
            events = read_evidence(Journey.open(Path(journey_file)))
            held_calls = list(composition.calls)
            # With the credential back, resume starts the Attempt from the head
            # the failures left as side leaves (I-65 reads the same path).
            composition.environ = credential
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=journey_file))
            resumed = read_evidence(Journey.open(Path(journey_file)))
        self.assertEqual([e["event_type"] for e in resumed].count("AttemptStarted"), 1)
        failures = [
            e for e in events if e["event_type"] == "RuntimeConfigurationFailed"
        ]
        self.assertEqual(
            [f["payload"]["code"] for f in failures],
            ["model_credential_unavailable"] * 2,
        )
        self.assertEqual(
            [
                (
                    r.status,
                    r.failure.failure_class,
                    r.failure.invariant_id,
                    r.failure.evidence,
                )
                for r in refusals
            ],
            [
                (
                    "run_held",
                    "invariant",
                    "model_credential_unavailable",
                    {
                        "configuration_failure_event_id": f["event_id"],
                        "violations": f["payload"]["violations"],
                    },
                )
                for f in failures
            ],
        )
        self.assertRegex(refusals[0].failure.cause, r"^RuntimeConfigError@")
        self.assertEqual(observed.status, "run_held")
        self.assertEqual(held_calls, [])
        types = [e["event_type"] for e in events]
        self.assertNotIn("RunFailed", types)
        self.assertNotIn("AttemptStarted", types)
        self.assertEqual(
            [
                e["payload"]["disposition"]
                for e in events
                if e["event_type"] == "PassClosed"
            ][-2:],
            ["held", "held"],
        )

    def test_instructions_that_cannot_render_hold_the_run_until_the_fix(self):
        # I-65: a selected block source missing from the package is refused at the
        # Attempt's configuration, before any model is acquired. It is recorded as
        # RuntimeConfigurationFailed and holds the Run under its own typed id,
        # never coordinator_unhandled_error; resume after the fix starts the
        # Attempt.
        import shutil
        from unittest.mock import patch

        from bio_neb.agentic_journey import role_instructions as module
        from bio_neb.application.advancement import RunAdvancementRefused

        published = module._published("planning_agent.v22.blocks.toml")
        module.role_instructions.cache_clear()
        self.addCleanup(module.role_instructions.cache_clear)
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "instructions"
            shutil.copytree(Path(str(published)).parent, package)
            (package / "planning_agent.v22.blocks.toml").unlink()
            (Path(directory) / "run").mkdir()
            receipt, paths = _registered(Path(directory) / "run")
            journey_file = receipt.journey_file
            composition = PlanningComposition(receipt, paths)
            with (
                use_run_advancement_composition(composition),
                patch.object(module, "_published", lambda name: package / name),
                self.assertRaises(RunAdvancementRefused) as refused,
            ):
                advance(ResumeRun(journey_file=journey_file))
            held = read_evidence(Journey.open(Path(journey_file)))
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=journey_file))
            resumed = read_evidence(Journey.open(Path(journey_file)))
        [failure] = [e for e in held if e["event_type"] == "RuntimeConfigurationFailed"]
        self.assertEqual(
            failure["payload"]["violations"],
            ["unknown instruction version: planning_agent:v22"],
        )
        self.assertEqual(
            (
                refused.exception.status,
                refused.exception.failure.failure_class,
                refused.exception.failure.invariant_id,
                refused.exception.failure.evidence,
                failure["payload"]["code"],
            ),
            (
                "run_held",
                "invariant",
                "instructions_unrenderable",
                {
                    "configuration_failure_event_id": failure["event_id"],
                    "violations": failure["payload"]["violations"],
                },
                "instructions_unrenderable",
            ),
        )
        self.assertRegex(refused.exception.failure.cause, r"^RuntimeConfigError@")
        self.assertNotIn("AttemptStarted", [e["event_type"] for e in held])
        self.assertEqual(held[-1]["payload"]["disposition"], "held")
        # The fix is in: resume dispatches the Attempt with its instructions.
        [started] = [e for e in resumed if e["event_type"] == "AttemptStarted"]
        self.assertEqual(
            started["payload"]["instruction_identity"], "planning_agent:v22:qwen_3_5"
        )

    def test_a_broken_instruction_file_holds_the_run_before_any_model(self):
        # I-65 (SP2): a selected file that does not parse, or that states the
        # role's messages or purposes incompletely or unfillably, is the same
        # typed configuration hold, before the model is acquired.
        import re
        import shutil
        from unittest.mock import Mock, patch

        from bio_neb.agentic_journey import role_instructions as module
        from bio_neb.application.advancement import RunAdvancementRefused

        def without(name):
            return lambda text: re.sub(
                rf'\n\[\[block\]\]\nname = "{name}"\n.*?(?=\n\[\[block\]\]|\Z)',
                "",
                text,
                flags=re.S,
            )

        blocks, qwen = (
            "planning_agent.v22.blocks.toml",
            "planning_agent.v22.qwen_3_5.toml",
        )
        cases = {
            "malformed block source": (
                blocks,
                lambda _: "[[block]\n",
                "^block source planning_agent:v22 does not parse: TOMLDecodeError$",
            ),
            "malformed form text": (
                qwen,
                lambda _: "[[block]\n",
                "^form text planning_agent:v22:qwen_3_5 does not parse: TOMLDecodeError$",
            ),
            "no block table": (
                blocks,
                lambda _: 'unrelated = "x"\n',
                "^block source planning_agent:v22 does not parse: KeyError$",
            ),
            "missing message": (
                blocks,
                without("objective"),
                r"^instructions planning_agent:v22:qwen_3_5 state messages "
                r"\['correction_task', 'job', 'target_answer_task'\], not",
            ),
            "missing purpose": (
                blocks,
                without("search_journey"),
                r"^instructions planning_agent:v22:qwen_3_5 state no purpose for "
                r"\['search_journey'\]",
            ),
            "unformattable message": (
                qwen,
                lambda text: text.replace(
                    "text = '''Resolve the information", "text = '''unfinished { the"
                ),
                "^instructions planning_agent:v22:qwen_3_5 message job does not format",
            ),
        }
        published = Path(str(module._published(blocks))).parent
        for case, (name, change, reason) in cases.items():
            module.role_instructions.cache_clear()
            self.addCleanup(module.role_instructions.cache_clear)
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                package = Path(directory) / "instructions"
                shutil.copytree(published, package)
                target = package / name
                changed = change(target.read_text(encoding="utf-8"))
                self.assertNotEqual(changed, target.read_text(encoding="utf-8"))
                target.write_text(changed, encoding="utf-8")
                (Path(directory) / "run").mkdir()
                receipt, paths = _registered(Path(directory) / "run")
                composition = PlanningComposition(receipt, paths)
                composition.model_endpoint = Mock(wraps=composition.model_endpoint)
                with (
                    use_run_advancement_composition(composition),
                    patch.object(module, "_published", lambda n, p=package: p / n),
                    self.assertRaises(RunAdvancementRefused) as refused,
                ):
                    advance(ResumeRun(journey_file=receipt.journey_file))
                events = read_evidence(Journey.open(Path(receipt.journey_file)))
                [failure] = [
                    e["payload"]
                    for e in events
                    if e["event_type"] == "RuntimeConfigurationFailed"
                ]
                self.assertEqual(
                    (
                        refused.exception.failure.invariant_id,
                        failure["code"],
                        composition.model_endpoint.call_count,
                    ),
                    ("instructions_unrenderable", "instructions_unrenderable", 0),
                )
                [violation] = failure["violations"]
                self.assertRegex(violation, reason)
                self.assertNotIn("AttemptStarted", [e["event_type"] for e in events])

    def test_a_planning_fault_before_any_model_response_fails_its_attempt(self):
        # A fault that is neither a Tool's nor the model's is our bug: the Attempt
        # fails as an invariant with its id, and resume after the fix starts afresh.
        from unittest.mock import patch

        from bio_neb.agentic_journey import planning_assignment as assignment
        from bio_neb.application.advancement import RunAdvancementRefused

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            with (
                use_run_advancement_composition(composition),
                patch.object(
                    assignment,
                    "planning_stage_findings",
                    side_effect=RuntimeError("private planning bug"),
                ),
                self.assertRaises(RunAdvancementRefused) as refused,
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            started = next(e for e in events if e["event_type"] == "AttemptStarted")
            failed = next(e for e in events if e["event_type"] == "AttemptFailed")
            self.assertEqual(
                failed["payload"],
                {
                    "attempt_id": started["payload"]["attempt_id"],
                    "code": "invariant",
                    "invariant_id": "attempt_unhandled_error",
                    "source_event_id": started["event_id"],
                },
            )
            self.assertEqual(
                (
                    refused.exception.status,
                    refused.exception.failure.invariant_id,
                    refused.exception.failure.evidence["code"],
                ),
                ("run_held", "planning_attempt_failed", "invariant"),
            )
            with use_run_advancement_composition(composition):
                resumed = advance(ResumeRun(journey_file=receipt.journey_file))
            after = [
                e["event_type"]
                for e in read_evidence(Journey.open(Path(receipt.journey_file)))
            ]
        self.assertEqual(resumed.status, "planning_hitl_pending")
        self.assertEqual(
            (after.count("AttemptStarted"), after.count("AttemptFailed")), (2, 1)
        )
        self.assertNotIn("RunFailed", after)

    def test_each_resume_after_a_failed_planning_attempt_starts_exactly_one_attempt(
        self,
    ):
        # Q9: a failed planning Attempt holds the Run; only the researcher's resume
        # starts the next Attempt, once, and nothing reads the Run as failed.
        from unittest.mock import patch

        from bio_neb.application.advancement import RunAdvancementRefused
        from bio_neb.application.queries import ObserveRun, observe

        _, helper, companion = _FAILING_COMPANION_REQUESTS[0]
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            journey_file = receipt.journey_file
            composition = PlanningComposition(receipt, paths)
            composition.companion_adapter = companion()

            def counts():
                types = [
                    e["event_type"]
                    for e in read_evidence(Journey.open(Path(journey_file)))
                ]
                return tuple(
                    types.count(t) for t in ("AttemptStarted", "AttemptFailed")
                ) + ("RunFailed" in types,)

            seen = []
            with (
                use_run_advancement_composition(composition),
                patch(
                    f"bio_neb.reference_discovery.reference_facts_agent.{helper}",
                    side_effect=RuntimeError("private planning failure"),
                ),
            ):
                for _ in range(3):
                    with self.assertRaises(RunAdvancementRefused) as refused:
                        advance(ResumeRun(journey_file=journey_file))
                    seen.append(
                        (
                            refused.exception.status,
                            observe(ObserveRun(journey_file=journey_file)).status,
                            *counts(),
                        )
                    )
            self.assertEqual(
                seen,
                [
                    ("run_held", "run_held", 1, 1, False),
                    ("run_held", "run_held", 2, 2, False),
                    ("run_held", "run_held", 3, 3, False),
                ],
            )
            # With the Tool fixed, one resume starts one Attempt that reaches its stop.
            with use_run_advancement_composition(composition):
                fixed = advance(ResumeRun(journey_file=journey_file))
            self.assertEqual(fixed.status, "planning_hitl_pending")
            self.assertEqual(counts(), (4, 3, False))

    def test_exhausted_taxon_lookup_asks_exact_taxon_question_before_discovery(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        from bio_neb.reference_discovery.reference_identity_tool import (
            TaxonomyAdapterObservation,
        )
        from tests.reference_fakes import (
            ReferenceAdapter,
            reference_const_arguments,
            reference_source,
        )

        class UnknownTaxon(ReferenceAdapter):
            def resolve_taxon(self, query):
                self.contact_count += 1
                return TaxonomyAdapterObservation(
                    records=(),
                    source=reference_source("ncbi_taxonomy", "taxonomy/not-found"),
                )

        class Asking(PlanningComposition):
            async def respond(self, messages, info):
                tools = eligible_tools(self.journey, info.function_tools)
                if "request_planning_hitl" not in tools:
                    return await super().respond(messages, info)
                self.calls.append("request_planning_hitl")
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "request_planning_hitl",
                            reference_const_arguments(tools["request_planning_hitl"]),
                            tool_call_id="request-taxon",
                        )
                    ]
                )

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory),
                prompt_text="Analyze rs1234 for sample-001 Homo sapiens using reference GCF_000001405.40 and produce VCF.",
            )
            composition = Asking(receipt, paths)
            composition.reference_adapter = UnknownTaxon()
            with use_run_advancement_composition(composition):
                pending = advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            taxon_facts = [
                e for e in events if e["event_type"] == "TaxonResolutionRecorded"
            ]
            started = next(
                e
                for e in events
                if e["event_type"] == "ObservedToolExecutionStarted"
                and e["payload"]["tool_name"] == "request_planning_hitl"
            )
            succeeded = next(
                e
                for e in events
                if e["event_type"] == "ObservedToolExecutionSucceeded"
                and e["payload"]["started_event_id"] == started["event_id"]
            )
            question = pending.planning_hitl.pending_request
            taxon_requirement = next(
                r
                for r in pending.proposal_plan.plan.requirements
                if r.kind == "taxon_identity"
            )

            self.assertEqual(pending.status, "planning_hitl_pending")
            self.assertEqual(question.requirement_id, taxon_requirement.requirement_id)
            self.assertEqual(
                question.evidence_event_ids, (taxon_facts[-1]["event_id"],)
            )
            self.assertEqual(question.candidates, ())
            self.assertEqual(composition.reference_adapter.contact_count, 1)
            self.assertFalse(
                any(
                    e["event_type"]
                    in {"ReferenceAssembliesRecorded", "ReferenceDescriptionRecorded"}
                    for e in events
                )
            )
            self.assertEqual(started["payload"]["adapter_identity"], "planning_hitl")
            self.assertEqual(
                succeeded["payload"]["stage_result_event_id"],
                question.request_event_id,
            )
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).status,
                pending.status,
            )

    def test_a_comma_term_is_refused_by_the_port_before_contact_and_the_model_recovers(
        self,
    ):
        comma = "Homo sapiens, NCBI taxon identifier 9606"
        with tempfile.TemporaryDirectory() as directory:
            _, composition, result, events = _taxid_run(
                directory, terms=(comma, "9606"), resolvable=("9606",)
            )

        # No source contact for the comma term; the next term resolves.
        self.assertEqual(composition.reference_adapter.queries, ["9606"])
        refused = next(
            e
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["arguments"] == {"term": comma}
        )
        self.assertEqual(refused["payload"]["status"], "invalid")
        self.assertEqual(refused["payload"]["errors"], ["taxon_term_not_one_taxon"])
        self.assertFalse(
            any(
                e["event_type"] == "ObservedToolExecutionStarted"
                and e["payload"]["tool_call_id"] == refused["payload"]["tool_call_id"]
                for e in events
            )
        )
        self.assertTrue(
            any(
                f'`term` is "{comma}": NCBI Taxonomy reads a comma as a list of '
                "several taxa" in returned
                for returned in composition.returned
            )
        )
        self.assertTrue(
            any(e["event_type"] == "ReferenceAssembliesRecorded" for e in events)
        )
        self.assertFalse(any(e["event_type"] == "RunFailed" for e in events))
        self.assertNotEqual(result.status, "failed")

    def test_a_not_found_term_returns_to_the_model_which_chooses_the_next(self):
        with tempfile.TemporaryDirectory() as directory:
            _, composition, _, events = _taxid_run(directory, resolvable=("9606",))

            self.assertEqual(
                composition.reference_adapter.queries, [_TAXID_ASSERTION, "9606"]
            )
            first = next(
                e for e in events if e["event_type"] == "TaxonResolutionRecorded"
            )
            self.assertEqual(first["payload"]["result"]["resolution"], "not_found")
            self.assertTrue(_shown("resolve_taxon", first, composition.returned))
            self.assertTrue(
                any(e["event_type"] == "ReferenceAssembliesRecorded" for e in events)
            )
            self.assertIsNone(composition.tools_at_question)
            self.assertFalse(
                any(
                    e["event_type"] == "PlanningHitlRequested"
                    and e["payload"]["requirement_id"] == _taxon_requirement_id(events)
                    for e in events
                )
            )

    def test_the_resolved_taxon_verifies_its_slot_and_a_different_one_is_flagged(self):
        from bio_neb.reference_discovery.reference_identity_tool import (
            TaxonomySourceRecord,
        )

        mouse = (
            TaxonomySourceRecord(ncbi_tax_id=10090, scientific_name="Mus musculus"),
        )
        # (terms in order, verified TaxId, mismatch); the latest resolution is current.
        cases = (
            (("9606",), "9606", False),
            (("10090",), "10090", True),
            (("10090", "9606"), "9606", False),
            (("9606", "10090"), "10090", True),
        )
        for terms, verified, mismatch in cases:
            with (
                self.subTest(terms=terms),
                tempfile.TemporaryDirectory() as directory,
            ):
                # A second resolved lookup is the model's own correction.
                premature = tuple(
                    ("resolve_taxon", "resolve_taxon", {"term": term})
                    for term in terms[1:]
                )
                receipt, composition, _, events = _taxid_run(
                    directory,
                    terms=terms[:1],
                    resolvable=("9606",),
                    outcomes={"10090": mouse},
                    premature=premature,
                )
                self.assertEqual(composition.reference_adapter.queries, list(terms))
                slots = {
                    slot.kind: slot
                    for slot in read_planning(
                        Journey.open(Path(receipt.journey_file))
                    ).requirements.fact_slots
                }
                taxon = slots["taxon_identity"]
                self.assertEqual(
                    (taxon.status, taxon.requested_value, taxon.verified_value),
                    ("verified", "9606", verified),
                )
                self.assertEqual(taxon.mismatch, mismatch)
                self.assertEqual(slots["sample_identity"].status, "asserted")
                # Reference work went on from the current taxon.
                discovery = next(
                    e
                    for e in events
                    if e["event_type"] == "ReferenceAssembliesRecorded"
                )
                self.assertEqual(
                    discovery["payload"]["tool_input"]["taxon"]["ncbi_tax_id"],
                    int(verified),
                )

    def test_a_provisional_or_deleted_taxon_is_resolved_with_ncbis_flag_copied(self):
        """NCBI's provisional or deleted flag is copied onto the one candidate, which
        resolves the taxon; the model weighs the flag and discovery goes on. Code
        stamps no verdict of no taxon on it (#471 Q5 a, #375 C4)."""
        from bio_neb.reference_discovery.reference_identity_tool import (
            TaxonomySourceRecord,
        )

        for flag in ("provisional", "deleted"):
            with (
                self.subTest(flag=flag),
                tempfile.TemporaryDirectory() as directory,
            ):
                record = TaxonomySourceRecord(
                    ncbi_tax_id=9606, scientific_name="Homo sapiens", **{flag: True}
                )
                _, composition, _, events = _taxid_run(
                    directory, terms=("9606",), outcomes={"9606": (record,)}
                )

                fact = next(
                    e for e in events if e["event_type"] == "TaxonResolutionRecorded"
                )
                result = fact["payload"]["result"]
                self.assertEqual(result["resolution"], "resolved")
                self.assertEqual(result["candidates"][0]["identity_status"], flag)
                self.assertIn("discover_assemblies", composition.calls)

    def test_each_negative_taxonomy_result_returns_and_the_question_names_it(self):
        from bio_neb.reference_discovery.reference_identity_tool import (
            TaxonomySourceRecord,
        )

        human = TaxonomySourceRecord(ncbi_tax_id=9606, scientific_name="Homo sapiens")
        outcomes = {
            "not_found": (),
            "ambiguous": (
                human,
                TaxonomySourceRecord(
                    ncbi_tax_id=63221, scientific_name="Homo sapiens neanderthalensis"
                ),
            ),
        }
        for outcome, records in outcomes.items():
            with (
                self.subTest(outcome=outcome),
                tempfile.TemporaryDirectory() as directory,
            ):
                _, composition, pending, events = _taxid_run(
                    directory, terms=("9606",), outcomes={"9606": records}
                )

                fact = next(
                    e for e in events if e["event_type"] == "TaxonResolutionRecorded"
                )
                self.assertEqual(fact["payload"]["result"]["resolution"], outcome)
                # The result went back to the model, which asked; no terminal.
                self.assertTrue(_shown("resolve_taxon", fact, composition.returned))
                self.assertEqual(composition.calls[-1], "request_planning_hitl")
                self._assert_taxon_question(pending, events)
                issue = pending.planning_hitl.pending_request.issue
                self.assertTrue(
                    issue.startswith(
                        f'NCBI Taxonomy returned {outcome} for the query "9606"'
                    ),
                    issue,
                )
                self.assertNotIn("returned no candidates", issue)
                if outcome == "ambiguous":
                    self.assertIn("Homo sapiens neanderthalensis (TaxId 63221)", issue)
                self.assertFalse(
                    any(
                        e["event_type"] in {"ReferenceAssembliesRecorded", "RunFailed"}
                        for e in events
                    )
                )

    def test_a_citation_the_tool_does_not_offer_is_refused_before_its_domain_check(
        self,
    ):
        """#476 (owner, 2026-10-28, G3; sweep P3): an offered choice is enforced at
        admission, as a bound constant is, for every Tool; no domain re-check is
        needed for it to hold."""

        def uncited(journey):
            inputs = next(
                event
                for event in read_evidence(journey)
                if event["event_type"] == "RunInputsRegistered"
            )
            return {
                "assembly_accession": "GCF_000001405.40",
                "assertion_event_id": inputs["event_id"],
                "quote": "GCF_000001405.40",
                "reason": "The researcher supplied this accession.",
            }

        premature = (
            ("record_proposal_requirements", "record_supplied_reference", uncited),
        )
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, events = _taxid_run(
                directory, supplied=True, resolvable=("9606",), premature=premature
            )
        refused_id, refused = next(
            (e["event_id"], e["payload"])
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_name"] == "record_supplied_reference"
        )
        self.assertEqual(refused["status"], "invalid")
        self.assertEqual(refused["errors"], ["assertion_event_id:not_offered"])
        (turn,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
            and refused_id in e["payload"]["validation_event_ids"]
        ]
        self.assertIn("which this Tool does not offer", turn["feedback"])

    def test_supplied_accession_is_inspected_once_the_model_resolves_its_taxon(self):
        premature = (
            ("attest_sample", "inspect_reference", {"accession": "GCF_000001405.40"}),
        )
        with tempfile.TemporaryDirectory() as directory:
            _, composition, _, events = _taxid_run(
                directory, supplied=True, resolvable=("9606",), premature=premature
            )

            # Inspecting before the taxon resolves is refused with its reason.
            refused = next(
                e["payload"]
                for e in events
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["tool_name"] == "inspect_reference"
            )
            self.assertEqual(refused["status"], "invalid")
            self.assertEqual(refused["errors"], ["taxon_unresolved"])
            self.assertEqual(
                composition.reference_adapter.queries, [_TAXID_ASSERTION, "9606"]
            )
            self.assertNotIn("discover_assemblies", composition.calls)
            description = next(
                e for e in events if e["event_type"] == "ReferenceDescriptionRecorded"
            )
            self.assertEqual(description["payload"]["result"]["resolution"], "resolved")
            self.assertIsNone(composition.tools_at_question)
            (reference,) = (
                slot
                for slot in read_planning(
                    Journey.open(Path(composition.journey.path))
                ).requirements.fact_slots
                if slot.kind == "reference_assembly_identity"
            )
            self.assertEqual(
                (reference.status, reference.verified_value, reference.mismatch),
                ("verified", "GCF_000001405.40", False),
            )

    def test_supplied_reference_reopens_by_its_assertion_not_a_rule_rescan(self):
        from unittest.mock import patch

        from bio_neb.identifier_rule import IdentifierRule

        with tempfile.TemporaryDirectory() as directory:
            receipt, _, _, events = _taxid_run(
                directory, supplied=True, resolvable=("9606",)
            )
            (choice,) = (
                e for e in events if e["event_type"] == "ReferenceChoiceRecorded"
            )
            self.assertEqual(
                choice["payload"]["assembly_accession"], "GCF_000001405.40"
            )
            # The plan recorded the typed accession, and reopen checks only that its
            # quote holds it: a matcher that lists another value too cannot change
            # the choice.
            wider = IdentifierRule(
                name="assembly_accession",
                stated=lambda text: ("GCF_000001405.39", "GCF_000001405.40"),
            )
            with patch(
                "bio_neb.reference_discovery.reference_journey.ASSEMBLY_ACCESSION_RULE",
                wider,
            ):
                reopened = read_planning(Journey.open(Path(receipt.journey_file)))
            self.assertEqual(
                reopened.reference.verified_reference.assembly_accession,
                "GCF_000001405.40",
            )

    def test_an_unverified_supplied_accession_asks_and_a_clarified_one_is_inspected(
        self,
    ):
        from bio_neb.parabricks.planning_hitl import ClarifyPlanningHitlResponse
        from bio_neb.reference_discovery.reference_identity_tool import (
            AssemblyAdapterObservation,
        )

        class Replaced(_TaxIdTaxonomy):
            """NCBI Datasets has no record of the supplied .40; .39 is current."""

            def __init__(self):
                super().__init__(resolvable=("9606",))
                self.described = []

            def describe_assembly(self, assembly_accession):
                self.described.append(assembly_accession)
                if assembly_accession == "GCF_000001405.39":
                    return super().describe_assembly(assembly_accession)
                self.contact_count += 1
                return AssemblyAdapterObservation(
                    records=(),
                    source=reference_source(
                        "ncbi_datasets", f"genome/accession/{assembly_accession}"
                    ),
                    total_count=0,
                )

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory),
                prompt_text=(
                    f"Analyze rs1234 for sample-001 {_TAXID_ASSERTION} using "
                    "reference GCF_000001405.40 and produce VCF."
                ),
            )
            composition = _TaxIdPlanning(receipt, paths)
            composition.terms = ("9606",)
            adapter = composition.reference_adapter = Replaced()
            with use_run_advancement_composition(composition):
                pending = advance(ResumeRun(journey_file=receipt.journey_file))
                events = read_evidence(Journey.open(Path(receipt.journey_file)))
                choice = next(
                    e for e in events if e["event_type"] == "ReferenceChoiceRecorded"
                )
                description = next(
                    e
                    for e in events
                    if e["event_type"] == "ReferenceDescriptionRecorded"
                )
                self.assertEqual(
                    description["payload"]["result"]["resolution"], "absent"
                )
                # The absent result reached the model, which asked; nothing failed.
                self.assertTrue(
                    _shown("inspect_reference", description, composition.returned)
                )
                self.assertEqual(composition.calls[-1], "request_planning_hitl")
                question = pending.planning_hitl.pending_request
                self.assertEqual(
                    question.evidence_event_ids,
                    (choice["event_id"], description["event_id"]),
                )
                self.assertIn("GCF_000001405.40", question.issue)
                self.assertIn("absent", question.issue)
                result = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=question.request_event_id,
                        answer=PlanningReviewAnswer(
                            kind="planning",
                            response=ClarifyPlanningHitlResponse(
                                action="clarify",
                                instruction="Use GCF_000001405.39 instead.",
                            ),
                        ),
                    )
                )
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            self.assertEqual(
                result.reference_identity.verified_reference.assembly_accession,
                "GCF_000001405.39",
            )
            self.assertEqual(
                adapter.described, ["GCF_000001405.40", "GCF_000001405.39"]
            )
            # The clarification did not repeat unchanged reference work.
            self.assertEqual(adapter.queries, ["9606"])
            self.assertEqual(
                sum(e["event_type"] == "SampleAttestationRecorded" for e in events), 1
            )
            self.assertEqual(composition.calls.count("record_supplied_reference"), 2)

    def test_an_answer_without_an_accession_continues_by_discovery_not_the_stale_choice(
        self,
    ):
        from bio_neb.parabricks.planning_hitl import ClarifyPlanningHitlResponse
        from bio_neb.reference_discovery.reference_identity_tool import (
            AssemblyAdapterObservation,
        )

        class Replaced(_TaxIdTaxonomy):
            """The supplied .40 is absent; discovery lists the current .39."""

            def __init__(self):
                super().__init__(resolvable=("9606",))
                self.described = []

            def discover_assemblies(self, ncbi_tax_id, query):
                self.contact_count += 1
                return AssemblyAdapterObservation(
                    records=(self._assembly("GCF_000001405.39", ncbi_tax_id, None),),
                    source=reference_source("ncbi_datasets", "genome/taxon/9606"),
                    total_count=1,
                )

            def describe_assembly(self, assembly_accession):
                self.described.append(assembly_accession)
                if assembly_accession == "GCF_000001405.39":
                    return super().describe_assembly(assembly_accession)
                self.contact_count += 1
                return AssemblyAdapterObservation(
                    records=(),
                    source=reference_source(
                        "ncbi_datasets", f"genome/accession/{assembly_accession}"
                    ),
                    total_count=0,
                )

        def stale_arguments(journey):
            # The model types the request's old accession while citing the answer.
            answer = next(
                event
                for event in reversed(read_evidence(journey))
                if event["event_type"] == "PlanningHitlResponded"
            )
            return {
                "assembly_accession": "GCF_000001405.40",
                "assertion_event_id": answer["event_id"],
                "quote": "GCF_000001405.40",
                "reason": "Inspect the supplied accession again.",
            }

        stale = ("request_planning_hitl", "record_supplied_reference", stale_arguments)
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory),
                prompt_text=(
                    f"Analyze rs1234 for sample-001 {_TAXID_ASSERTION} using "
                    "reference GCF_000001405.40 and produce VCF."
                ),
            )
            composition = _TaxIdPlanning(receipt, paths)
            composition.terms = ("9606",)
            # Right after its question is answered, the model tries the old choice.
            composition.premature = (stale,)
            adapter = composition.reference_adapter = Replaced()
            with use_run_advancement_composition(composition):
                pending = advance(ResumeRun(journey_file=receipt.journey_file))
                question = pending.planning_hitl.pending_request
                result = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=question.request_event_id,
                        answer=PlanningReviewAnswer(
                            kind="planning",
                            response=ClarifyPlanningHitlResponse(
                                action="clarify",
                                instruction="Use the latest human reference assembly.",
                            ),
                        ),
                    )
                )
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            refused = [
                e["payload"]
                for e in events
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["tool_name"] == "record_supplied_reference"
                and e["payload"]["status"] == "invalid"
            ]
            # Only the answer's own words replace the choice, and they write no
            # accession.
            self.assertEqual(
                [r["errors"] for r in refused], [["supplied_quote_ungrounded"]]
            )
            # No stale supplied choice was written; discovery continued, and the
            # model chose the row its page lists.
            self.assertEqual(
                [
                    (
                        e["payload"]["assembly_accession"],
                        e["payload"].get("discovery_event_id") is not None,
                    )
                    for e in events
                    if e["event_type"] == "ReferenceChoiceRecorded"
                ],
                [("GCF_000001405.40", False), ("GCF_000001405.39", True)],
            )
            self.assertEqual(
                adapter.described, ["GCF_000001405.40", "GCF_000001405.39"]
            )
            self.assertEqual(
                result.reference_identity.verified_reference.assembly_accession,
                "GCF_000001405.39",
            )
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).reference_identity,
                result.reference_identity,
            )

    def test_supplied_accession_with_an_unresolved_taxon_asks_the_taxon_question(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            _, composition, pending, events = _taxid_run(directory, supplied=True)

            self.assertEqual(
                composition.reference_adapter.queries, [_TAXID_ASSERTION, "9606"]
            )
            self._assert_taxon_question(pending, events)
            self.assertFalse(
                any(e["event_type"] == "ReferenceDescriptionRecorded" for e in events)
            )

    def test_replay_refuses_a_taxon_fact_its_admitted_term_did_not_produce(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _, _, events = _taxid_run(directory, resolvable=("9606",))
            resolved = next(
                e
                for e in events
                if e["event_type"] == "TaxonResolutionRecorded"
                and e["payload"]["tool_input"]["query"] == "9606"
            )
            validation = next(
                e
                for e in events
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["arguments"] == {"term": "9606"}
            )
            project_journey_events(events, journey_file=receipt.journey_file)
            # The model call and its admission name another term than the fact.
            hostile = copy.deepcopy(events)
            for event in hostile:
                if event["event_id"] == validation["event_id"]:
                    event["payload"]["arguments"]["term"] = "Homo sapiens"
                if event["event_type"] == "ModelResponseRecorded":
                    for part in event["payload"]["compatible_adapter_evidence"][
                        "response"
                    ]["message"]["parts"]:
                        if (
                            part.get("tool_call_id")
                            == validation["payload"]["tool_call_id"]
                        ):
                            # The wire returns a call's arguments as JSON text.
                            part["args"] = json.dumps(
                                {**json.loads(part["args"]), "term": "Homo sapiens"}
                            )
            # The retained fact itself is untouched; only its admission changed.
            self.assertIn(resolved, hostile)
            with self.assertRaisesRegex(ValueError, "changed its admitted arguments"):
                project_journey_events(hostile, journey_file=receipt.journey_file)

    def _assert_taxon_question(self, pending, events):
        taxon_facts = [
            e for e in events if e["event_type"] == "TaxonResolutionRecorded"
        ]
        question = pending.planning_hitl.pending_request
        self.assertEqual(pending.status, "planning_hitl_pending")
        self.assertEqual(question.requirement_id, _taxon_requirement_id(events))
        self.assertEqual(question.evidence_event_ids, (taxon_facts[-1]["event_id"],))

    def test_ambiguous_discovery_can_be_refined_with_retained_isolate_evidence(self):
        from tests.reference_fakes import CompanionAdapter, ReferenceAdapter
        from tests.reference_query_fakes import RefiningPlanning as Refine

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = Refine(receipt, paths)
            composition.reference_adapter = ReferenceAdapter(ambiguous=True)
            composition.companion_adapter = CompanionAdapter("GCF_000009999.1")
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(
                result.reference_identity.verified_reference.assembly_accession,
                "GCF_000009999.1",
            )
            self.assertEqual(composition.calls.count("discover_assemblies"), 2)
            self.assertEqual(composition.reference_adapter.contact_count, 4)
            self.assertEqual(
                sum(
                    f.event_type == "ReferenceAssembliesRecorded"
                    for f in result.reference_identity.facts
                ),
                2,
            )

    def test_source_failure_is_retained_and_never_becomes_empty_discovery(self):
        from unittest.mock import patch

        from bio_neb.reference_discovery.reference_identity_tool import (
            ReferenceAdapterError,
        )
        from tests.reference_fakes import ReferenceAdapter

        class Unavailable(ReferenceAdapter):
            def resolve_taxon(self, query):
                self.contact_count += 1
                raise ReferenceAdapterError(
                    "transport_error", "Source unavailable.", retryable=True
                )

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            composition.reference_adapter = Unavailable()
            with patch("time.sleep"), use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(result.status, "operational_review_pending")
            self.assertEqual(composition.reference_adapter.contact_count, 3)
            self.assertIsNone(result.reference_identity.discovery_event_id)
            self.assertIsNone(result.reference_identity.verified_reference)
            self.assertEqual(
                result.reference_identity.facts[-1].result["status"], "failed"
            )
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).status,
                result.status,
            )

    def test_explicit_ambiguity_retains_reason_and_stops_for_useful_clarification(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        from tests.reference_fakes import ReferenceAdapter

        class Ambiguous(PlanningComposition):
            async def respond(self, messages, info):
                result = await super().respond(messages, info)
                call = result.parts[0]
                if (
                    call.tool_name == "record_reference_choice"
                    and self.calls.count(call.tool_name) == 1
                ):
                    args = dict(call.args)
                    args.update(
                        # The named choice that no listed row fits.
                        assembly_accession="none",
                        reason="Both supported assemblies remain plausible; sample provenance is insufficient to choose.",
                    )
                    args.pop("quote", None)
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                call.tool_name, args, tool_call_id=call.tool_call_id
                            )
                        ]
                    )
                return result

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = Ambiguous(receipt, paths)
            composition.reference_adapter = ReferenceAdapter(ambiguous=True)
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(result.status, "planning_hitl_pending")
            choice = result.reference_identity.choices[-1]
            self.assertIsNone(choice.assembly_accession)
            self.assertIn("provenance", choice.reason)
            self.assertIsNone(result.reference_identity.verified_reference)
            self.assertNotIn("inspect_reference", composition.calls)
            self.assertEqual(len(result.planning_hitl.pending_request.candidates), 2)
            pending = result.planning_hitl.pending_request
            with use_run_advancement_composition(composition):
                resumed = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=pending.request_event_id,
                        answer=PlanningReviewAnswer(
                            kind="planning",
                            response=SelectPlanningHitlResponse(
                                action="select",
                                candidate_id=pending.candidates[0].candidate_id,
                            ),
                        ),
                    )
                )
            self.assertIsNotNone(resumed.reference_identity.verified_reference)
            self.assertEqual(len(resumed.reference_identity.choices), 2)
            self.assertIsNone(resumed.reference_identity.choices[0].assembly_accession)
            self.assertEqual(composition.calls.count("discover_assemblies"), 1)

    def test_unsupported_selected_source_is_a_gap_without_reference_substitution(self):
        from tests.reference_fakes import CompanionAdapter

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory),
                prompt_text="Analyze rs1234 for sample-001 Homo sapiens using exact reference GCF_000009999.1 and produce VCF.",
            )
            composition = PlanningComposition(receipt, paths)
            composition.companion_adapter = CompanionAdapter("GCF_000009999.1")
            with use_run_advancement_composition(composition):
                pending = advance(ResumeRun(journey_file=receipt.journey_file))
                question = pending.planning_hitl.pending_request
                result = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=question.request_event_id,
                        answer=PlanningReviewAnswer(
                            kind="planning",
                            response=SelectPlanningHitlResponse(
                                action="select",
                                candidate_id=question.candidates[0].candidate_id,
                            ),
                        ),
                    )
                )
            # The coordinator asks for the launch settings through the preparation
            # Review; the reference source is the discovered one, never asked for,
            # and no fixed reference descriptor is substituted.
            self.assertEqual(result.current_review.kind, "preparation")
            self.assertNotIn("request_preparation_information", composition.calls)
            self.assertEqual(
                tuple(result.current_review.request.missing_fields),
                ("launch", "runtime_selection"),
            )
            self.assertEqual(
                result.reference_identity.verified_reference.assembly_accession,
                "GCF_000009999.1",
            )
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            self.assertFalse(
                any(e["event_type"] == "ReferenceSourceAdmitted" for e in events)
            )
            context = json.dumps(result.current_review.request.context)
            self.assertIn("GCF_000009999.1", context)
            self.assertNotIn("GCF_000001405", context)

    def test_local_reference_replica_reaches_review_unopened(self):
        from bio_neb.parabricks.configuration import LocalReferenceSource
        from tests.planning_assignment_fakes import (
            execution_configuration,
            prepare_configured_run,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            baseline = execution_configuration(root)
            local = LocalReferenceSource(
                path=str(root / "not-acquired-reference.fa.gz")
            )
            configuration = baseline.model_copy(
                update={
                    "launch": baseline.launch.model_copy(
                        update={"local_reference_source": local}
                    )
                }
            )
            receipt, review, _ = prepare_configured_run(
                root, configuration=configuration
            )
            proposal = review.execution_hitl.pending_request.proposal
            self.assertEqual(
                proposal.configuration.launch.local_reference_source, local
            )
            self.assertFalse(Path(local.path).exists())
            self.assertEqual(review.execution_hitl.authorities, ())
            self.assertFalse(
                any(
                    e["event_type"] == "ReferencePrepared"
                    for e in read_evidence(Journey.open(Path(receipt.journey_file)))
                )
            )

    def test_reference_correction_preserves_old_review_and_requires_new_approval(self):
        from bio_neb.application.advancement import ExecutionReviewAnswer
        from bio_neb.parabricks.execution_proposal import ReviseExecutionResponse
        from tests.planning_assignment_fakes import configured_planning_run

        with tempfile.TemporaryDirectory() as directory:
            receipt, composition, _ = configured_planning_run(Path(directory).resolve())
            with use_run_advancement_composition(composition):
                reviewed = advance(ResumeRun(journey_file=receipt.journey_file))
                while (
                    reviewed.current_review
                    and reviewed.current_review.kind == "planning"
                ):
                    question = reviewed.current_review.request
                    reviewed = advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=question.request_event_id,
                            answer=PlanningReviewAnswer(
                                kind="planning",
                                response=SelectPlanningHitlResponse(
                                    action="select",
                                    candidate_id=question.candidates[0].candidate_id,
                                ),
                            ),
                        )
                    )
                old = reviewed.execution_hitl.pending_request
                revised = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=old.request_event_id,
                        answer=ExecutionReviewAnswer(
                            kind="execution",
                            response=ReviseExecutionResponse(
                                action="revise",
                                proposal_id=old.proposal.proposal_id,
                                instruction="Reinspect the chosen exact reference and retain a fresh validation before approval.",
                                revalidation_scope="reference",
                            ),
                        ),
                    )
                )
            self.assertEqual(revised.status, "execution_hitl_pending")
            self.assertEqual(revised.execution_hitl.requests[0].proposal, old.proposal)
            new = revised.execution_hitl.pending_request.proposal
            self.assertEqual(new.supersedes_proposal_id, old.proposal.proposal_id)
            self.assertNotEqual(new.reference_event_id, old.proposal.reference_event_id)
            self.assertNotEqual(
                new.reference_basis.choice.source_event_id,
                old.proposal.reference_basis.choice.source_event_id,
            )
            self.assertEqual(len(revised.reference_identity.choices), 2)
            self.assertEqual(revised.execution_hitl.authorities, ())
            self.assertEqual(composition.calls.count("discover_assemblies"), 1)
            self.assertEqual(composition.calls.count("inspect_reference"), 2)

    def test_supplied_exact_reference_bypasses_discovery_and_reopens(self):
        from tests.application_fakes import _registered

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, paths = _registered(
                root,
                prompt_text="Analyze rs1234 for sample-001 Homo sapiens using exact reference GCF_000001405.40 and produce VCF.",
            )
            composition = PlanningComposition(receipt, paths)
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertIn(
                "Consumer-known genomic FASTA requirements are not a questionnaire",
                composition.instructions_seen[0],
            )
            self.assertNotIn("discover_assemblies", composition.calls)
            self.assertIn("inspect_reference", composition.calls)
            self.assertIsNone(result.reference_identity.discovery_event_id)
            self.assertEqual(
                result.reference_identity.verified_reference.assembly_accession,
                "GCF_000001405.40",
            )
            self.assertEqual(
                result.reference_identity.choices[0].assembly_accession,
                "GCF_000001405.40",
            )
            self.assertEqual(composition.reference_adapter.contact_count, 2)
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).reference_identity,
                result.reference_identity,
            )

    def test_reference_capabilities_choose_and_inspect_without_technical_question(self):
        from tests.reference_fakes import ReferenceAdapter

        class ChoiceComposition(PlanningComposition):
            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                if self.calls[-1] == "record_reference_choice":
                    response.parts[0].args.update(
                        assembly_accession="GCF_000001405.40",
                        reason="The current RefSeq assembly matches the stated human sample and downstream capability.",
                    )
                return response

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = ChoiceComposition(receipt, paths)
            composition.reference_adapter = ReferenceAdapter(ambiguous=True)
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertIn("discover_assemblies", composition.calls)
            self.assertIn("inspect_reference", composition.calls)
            self.assertNotIn("request_companion_roles", composition.calls)
            self.assertEqual(
                result.reference_identity.verified_reference.assembly_accession,
                "GCF_000001405.40",
            )
            self.assertEqual(result.reference_facts.required_roles, ("genomic_fasta",))
            self.assertEqual(len(result.reference_identity.choices), 1)
            self.assertEqual(
                result.reference_identity.choices[0].assembly_accession,
                "GCF_000001405.40",
            )
            contacts = (
                composition.reference_adapter.contact_count
                + composition.companion_adapter.contact_count
            )
            reopened = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            self.assertEqual(reopened.reference_identity, result.reference_identity)
            self.assertEqual(
                contacts,
                composition.reference_adapter.contact_count
                + composition.companion_adapter.contact_count,
            )
            import copy

            from bio_neb.application.planning_read import read_planning_events

            forged = copy.deepcopy(
                read_evidence(Journey.open(Path(receipt.journey_file)))
            )
            description = next(
                e for e in forged if e["event_type"] == "ReferenceDescriptionRecorded"
            )
            forged_reference = description["payload"]["result"]["references"][0]
            forged_reference["assembly_accession"] = "GCF_000009999.1"
            forged_reference["record"]["assembly_accession"] = "GCF_000009999.1"
            with self.assertRaisesRegex(ValueError, "exact queried identity"):
                read_planning_events(forged[: forged.index(description) + 1])

    def test_mismatched_inspection_can_ask_without_replacing_concrete_choice(self):
        from tests.reference_query_fakes import ChoosingPlanning

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = ChoosingPlanning(receipt, paths)
            composition.reference_adapter = ReferenceAdapter(ambiguous=True)
            composition.broken_inspections = 1
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            descriptions = [
                e for e in events if e["event_type"] == "ReferenceDescriptionRecorded"
            ]
            self.assertEqual(len(descriptions), 1)
            self.assertEqual(
                descriptions[0]["payload"]["result"]["resolution"],
                "source_disagreement",
            )
            self.assertEqual(
                descriptions[0]["payload"]["result"]["assembly_accession"],
                "GCF_000000001.1",
            )
            choices = [
                e for e in events if e["event_type"] == "ReferenceChoiceRecorded"
            ]
            self.assertEqual(len(choices), 1)
            self.assertEqual(
                choices[0]["payload"]["assembly_accession"], "GCF_000001405.40"
            )
            question = next(
                e for e in events if e["event_type"] == "PlanningHitlRequested"
            )
            self.assertIn(
                descriptions[0]["event_id"], question["payload"]["evidence_event_ids"]
            )
            self.assertEqual(composition.feedback, [])

    def test_application_combines_input_plan_reference_and_history(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        receipt, paths = _registered(tmp_path)
        composition = PlanningComposition(receipt, paths)
        with use_run_advancement_composition(composition):
            result = advance(ResumeRun(journey_file=receipt.journey_file))
        assert result.status == "planning_hitl_pending"
        assert [attempt.agent_role for attempt in result.attempts] == ["planning_agent"]
        assert (
            result.reference_identity.verified_reference.assembly_accession
            == "GCF_000001405.40"
        )
        events = read_evidence(Journey.open(Path(receipt.journey_file)))
        assert (
            len([e for e in events if e["event_type"] == "InputMetadataInspected"]) == 1
        )
        assert (
            len(
                [
                    e
                    for e in events
                    if e["event_type"] == "ProposalRequirementsIdentified"
                ]
            )
            == 1
        )
        history = next(e for e in events if e["event_type"] == "JourneySearchRead")
        assert (
            history["payload"]["result"]["events"][0]["payload"]["prompt"]
            == result.request.prompt
        )
        assert history["payload"]["result"]["journey_id"] == result.journey_id
        reopened = review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
        assert reopened.reference_identity == result.reference_identity

    def test_clarification_resumes_new_assignment_without_repeating_completed_tools(
        self,
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        from tests.planning_assignment_fakes import configured_planning_run

        receipt, composition, _ = configured_planning_run(tmp_path)
        with use_run_advancement_composition(composition):
            pending = advance(ResumeRun(journey_file=receipt.journey_file))
            request = pending.planning_hitl.pending_request
            answer = AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=request.request_event_id,
                answer=PlanningReviewAnswer(
                    kind="planning",
                    response=SelectPlanningHitlResponse(
                        action="select", candidate_id=request.candidates[0].candidate_id
                    ),
                ),
            )
            resumed = advance(answer)
            before_replay = list(composition.calls)
            replayed = advance(answer)
        assert resumed.status == "execution_hitl_pending"
        assert len(resumed.attempts) == 2
        assert resumed.attempts[0].attempt_id != resumed.attempts[1].attempt_id
        assert resumed.reference_facts.status == "reference_facts"
        assert composition.calls.count("inspect_input_metadata") == 0
        assert composition.calls.count("inspect_reference") == 1
        assert composition.calls.count("record_proposal_requirements") == 1
        assert composition.calls == before_replay
        assert replayed == resumed
        proposal = resumed.execution_hitl.pending_request.proposal
        self.assertEqual(
            proposal.reference_basis.choice.assembly_accession, "GCF_000001405.40"
        )
        self.assertEqual(
            proposal.reference_basis.validation_event_id, proposal.reference_event_id
        )
        self.assertEqual(
            proposal.reference_basis.alternatives[0].assembly_accession,
            "GCF_000001405.40",
        )
        self.assertIn("sample organism", proposal.reference_basis.choice.reason)

    def test_history_read_does_not_admit_unapproved_execution(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        receipt, paths = _registered(tmp_path)
        composition = PlanningComposition(receipt, paths, hostile=True)
        with use_run_advancement_composition(composition):
            result = advance(ResumeRun(journey_file=receipt.journey_file))
        assert result.status == "planning_hitl_pending"
        events = read_evidence(Journey.open(Path(receipt.journey_file)))
        refused = next(
            e
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_name"] == "execute_approved_parabricks"
        )
        assert refused["payload"]["status"] == "invalid"
        assert refused["payload"]["errors"] == ["tool_name_ineligible"]
        assert composition.contacts_at_refused_action == 0
        assert composition.reference_adapter.contact_count == 3
        assert composition.companion_adapter.contact_count == 1
        assert len(
            [e for e in events if e["event_type"] == "ModelRequestRecorded"]
        ) == len(composition.calls)
        assert not any(
            e["event_type"] == "ObservedToolExecutionStarted"
            and e["payload"]["tool_name"] == "execute_approved_parabricks"
            for e in events
        )
        assert not any(e["event_type"].startswith("Execution") for e in events)

    def test_adaptive_provider_retry_preserves_completed_tools_and_request_policy(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.agentic_journey.model_availability import (
            RetryModelAvailabilityResponse,
        )
        from bio_neb.application.advancement import ModelAvailabilityReviewAnswer

        class OutageComposition(PlanningComposition):
            failed = False

            async def respond(self, messages, info):
                if len(self.calls) == 3 and not self.failed:
                    self.failed = True
                    raise ModelHTTPError(
                        503, "adaptive-planning", headers={"retry-after": "999"}
                    )
                return await super().respond(messages, info)

        receipt, paths = _registered(tmp_path)
        composition = OutageComposition(receipt, paths)
        with use_run_advancement_composition(composition):
            stopped = advance(ResumeRun(journey_file=receipt.journey_file))
            assert stopped.status == "model_availability_review_pending"
            pending = stopped.model_availability_review.pending_request
            answer = AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=pending.request_event_id,
                answer=ModelAvailabilityReviewAnswer(
                    kind="model_availability",
                    response=RetryModelAvailabilityResponse(action="retry"),
                ),
            )
            resumed = advance(answer)
            assert resumed.status == "planning_hitl_pending"
            contacts = composition.reference_adapter.contact_count
            calls = list(composition.calls)
            assert advance(answer) == resumed
        assert composition.calls == calls
        assert composition.reference_adapter.contact_count == contacts == 3
        assert calls.count("inspect_input_metadata") == 0
        assert calls.count("record_proposal_requirements") == 1
        assert (
            resumed.attempts[-1].effective_invocation_policy
            == stopped.attempts[-1].effective_invocation_policy
        )
        assert len(resumed.attempts) == 2


def _advance_with_failing_companion_request(directory, helper, companion_adapter):
    """Advance until a companion donor Tool raises inside its observed execution.

    The model is shown the failure once and calls again; the same failure then
    holds the Run (#477 C8). The hold refusal is returned.
    """
    from unittest.mock import patch

    from bio_neb.application.advancement import RunAdvancementRefused

    receipt, paths = _registered(directory)
    composition = PlanningComposition(receipt, paths)
    composition.companion_adapter = companion_adapter
    with (
        use_run_advancement_composition(composition),
        patch(
            f"bio_neb.reference_discovery.reference_facts_agent.{helper}",
            side_effect=RuntimeError("private planning failure"),
        ),
    ):
        try:
            advance(ResumeRun(journey_file=receipt.journey_file))
        except RunAdvancementRefused as refused:
            return receipt, composition, refused
    raise AssertionError("a failed planning Attempt must hold the Run")


def _partial_companion_adapter():
    from tests.reference_fakes import RecordedCompanionAdapter, companion_record

    return RecordedCompanionAdapter(
        records=(
            companion_record(assembly_accession="GCF_000001405.40", authority=False),
        )
    )


def _multiple_companion_adapter():
    from tests.reference_fakes import CompanionAdapter

    return CompanionAdapter("GCF_000001405.40", asset_ids=("asset-A", "asset-B"))


_FAILING_COMPANION_REQUESTS = (
    (
        "request_companion_asset_selection",
        "_asset_candidates",
        _multiple_companion_adapter,
    ),
    ("request_companion_refinement", "_projection", _partial_companion_adapter),
)
