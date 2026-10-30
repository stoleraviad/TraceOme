"""Execution-planning Tools are reached only through the planning assignment."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

from bio_neb.agentic_journey.planning_assignment import DOMAIN_TOOLS
from bio_neb.agentic_journey.tool_execution import (
    function_tool_contracts,
    function_tool_names,
    negative_domain_result,
)
from bio_neb.application.advancement import (
    AnswerReview,
    ExecutionReviewAnswer,
    JunctionReviewAnswer,
    PlanningReviewAnswer,
    ResumeRun,
    advance,
    use_run_advancement_composition,
)
from bio_neb.application.queries import (
    ReviewJourneyQuery,
    project_journey_events,
    review_journey,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.parabricks.capability_policy import CapabilitySnapshotAdapter
from bio_neb.parabricks.execution_planning_action_policy import (
    EXECUTION_PLANNING_TOOLS,
)
from bio_neb.parabricks.execution_proposal import ReviseExecutionResponse
from bio_neb.parabricks.planning_hitl import SelectPlanningHitlResponse
from bio_neb.recording import read_evidence
from tests.application_fakes import _grounded_draft, _registered
from tests.planning_assignment_fakes import (
    PlanningComposition,
    configured_planning_run,
    eligible_tools,
)
from tests.planning_fakes import policy
from tests.reference_fakes import (
    PREBUILT_INDEX_TOOLS,
    companion_arguments,
    reference_call,
)


class _UngroundedFirstRecipeLookup(PlanningComposition):
    """The first recipe lookup names another input set; later calls are exact."""

    ungrounded_sent = False

    async def respond(self, messages, info):
        tools = {tool.name: tool for tool in info.function_tools}
        name = "list_compatible_analysis_recipes"
        if self.calls and name in tools and not self.ungrounded_sent:
            self.ungrounded_sent = True
            arguments = companion_arguments(tools[name])
            arguments["input_set_id"] = str(uuid.uuid4())
            self.calls.append(name)
            return ModelResponse(
                parts=[ToolCallPart(name, arguments, tool_call_id="ungrounded-list")]
            )
        return await super().respond(messages, info)


class _TwoRecipeRequirements(PlanningComposition):
    """Record a second recipe subject so one resource needs an owning recipe."""

    async def respond(self, messages, info):
        tools = {tool.name for tool in info.function_tools}
        if (
            self.calls
            and "record_proposal_requirements" in tools
            and "inspect_input_metadata" not in tools
        ):
            draft = _grounded_draft(self.registered, self.paths)
            recipe = next(
                item
                for item in draft["requirements"]
                if item["kind"] == "analysis_recipe"
            )
            sibling = copy.deepcopy(recipe)
            sibling["subject_key"] = "secondary_analysis_recipe"
            draft["requirements"].append(sibling)
            self.calls.append("record_proposal_requirements")
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "record_proposal_requirements",
                        draft,
                        tool_call_id=f"assignment-{len(self.calls)}",
                    )
                ]
            )
        return await super().respond(messages, info)


class _ReadsNegativeResult(PlanningComposition):
    """Records each Tool return it reads; with no admissible Tool left it says so."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_returns = []

    async def respond(self, messages, info):
        self.tool_returns.extend(
            (part.tool_name, part.model_response_object())
            for part in messages[-1].parts
            if isinstance(part, ToolReturnPart)
        )
        # The genomic FASTA's index Tools stay callable after its choice.
        tools = eligible_tools(self.journey, info.function_tools)
        if (
            self.calls
            and set(tools) - set(PREBUILT_INDEX_TOOLS) <= {"search_journey"}
            and reference_call(Journey.open(Path(self.journey.path)), tools) is None
        ):
            self.calls.append("text")
            return ModelResponse(
                parts=[TextPart("No compatible resource profile remains.")]
            )
        return await super().respond(messages, info)


def _answer_first_candidate(receipt, review):
    question = review.planning_hitl.pending_request
    assert question is not None and question.candidates, review.status
    return advance(
        AnswerReview(
            journey_file=receipt.journey_file,
            request_event_id=question.request_event_id,
            answer=PlanningReviewAnswer(
                kind="planning",
                response=SelectPlanningHitlResponse(
                    action="select", candidate_id=question.candidates[0].candidate_id
                ),
            ),
        )
    )


def _planning_run(directory, *, capability=None, composition=PlanningComposition):
    """Register inputs; only model, source and recipe adapters are scripted."""

    receipt, paths = _registered(Path(directory))
    planning = composition(receipt, paths)
    if capability is not None:
        planning.capability = capability
    planning.recipe_adapter = CapabilitySnapshotAdapter(planning.capability)
    return receipt, planning


def _advance(receipt, planning, *, answers=0):
    with use_run_advancement_composition(planning):
        review = advance(ResumeRun(journey_file=receipt.journey_file))
        for _ in range(answers):
            review = _answer_first_candidate(receipt, review)
    return review


def _advance_into_capability_failure(test, directory, method, *, after_question=False):
    """Advance until one recipe-adapter method raises inside its observed Tool.

    The failed planning Attempt holds the Run with its cause (#216).
    """
    from bio_neb.application.advancement import RunAdvancementRefused

    receipt, planning = _planning_run(directory)
    question = _advance(receipt, planning) if after_question else None
    with (
        test.assertRaises(RunAdvancementRefused) as refused,
        use_run_advancement_composition(planning),
        patch.object(
            planning.recipe_adapter,
            method,
            side_effect=RuntimeError("private capability detail"),
        ),
    ):
        if question is None:
            advance(ResumeRun(journey_file=receipt.journey_file))
        else:
            _answer_first_candidate(receipt, question)
    test.assertEqual(
        (refused.exception.status, refused.exception.failure.invariant_id),
        ("run_held", "planning_attempt_failed"),
    )
    test.assertRegex(refused.exception.failure.cause, r"^RuntimeError@")
    return receipt, planning


def _events(receipt):
    return read_evidence(Journey.open(Path(receipt.journey_file)))


def _executions(events, tool_name):
    """Pair each observed start of one Tool with its terminal record."""

    starts = [
        event
        for event in events
        if event["event_type"] == "ObservedToolExecutionStarted"
        and event["payload"]["tool_name"] == tool_name
    ]
    return [
        (
            start,
            next(
                event
                for event in events
                if event["event_type"]
                in {"ObservedToolExecutionSucceeded", "ObservedToolExecutionFailed"}
                and event["payload"]["started_event_id"] == start["event_id"]
            ),
        )
        for start in starts
    ]


def _by_id(events):
    return {event["event_id"]: event for event in events}


def _recorded_event_digest(tool_name, event):
    """The digest of the model view the Tool's row shows of ``event`` (#478)."""
    from bio_neb.agentic_journey.tool_execution import model_view_of, tool_model_view

    return hashlib.sha256(
        canonical_json(tool_model_view(model_view_of(tool_name), event)).encode("utf-8")
    ).hexdigest()


class ExecutionPlanningToolContractTests(unittest.TestCase):
    def test_every_execution_planning_action_is_an_unchanged_planning_contract(self):
        contracts = {
            item.tool_name: item
            for item in function_tool_contracts()
            if item.tool_name in EXECUTION_PLANNING_TOOLS
        }
        recorded = ("compiled_capability_snapshot", "ToolExecutionFailed")
        deterministic = ("deterministic", None)
        expected = {
            "list_compatible_analysis_recipes": (
                *recorded,
                "RecipeObservationRecorded",
                "_outcome",
            ),
            "lookup_compatible_resource_profiles": (
                *recorded,
                "ResourceObservationRecorded",
                "_outcome",
            ),
            **{
                name: (*deterministic, "PlanningHitlRequested", "_asked")
                for name in (
                    "request_recipe_eligibility_assertion",
                    "request_recipe_selection",
                    "request_resource_selection",
                    "request_resource_recipe_dependency",
                )
            },
            **{
                name: (*deterministic, "PlanningCandidateSelected", "_recorded")
                for name in ("select_recipe_candidate", "select_resource_candidate")
            },
        }
        self.assertEqual(
            {
                name: (
                    item.adapter_identity,
                    item.stage_failure_event_type,
                    item.stage_result_event_type,
                    # Each row's declared view reads its status.
                    item.model_view.status.__name__,
                )
                for name, item in contracts.items()
            },
            expected,
        )
        self.assertEqual(
            {item.agent_role for item in contracts.values()}, {"planning_agent"}
        )
        self.assertLessEqual(set(contracts), function_tool_names("planning_agent"))
        self.assertEqual(
            {DOMAIN_TOOLS[name] for name in EXECUTION_PLANNING_TOOLS},
            {"execution"},
        )


class ExecutionPlanningThroughAssignmentTests(unittest.TestCase):
    def test_transient_capability_success_retains_each_failure_and_policy_in_journal(
        self,
    ):
        from tests.planning_fakes import ExceptionalAdapter

        for method, tool, answers in (
            ("list_recipes", "list_compatible_analysis_recipes", 0),
            ("lookup_resources", "lookup_compatible_resource_profiles", 1),
        ):
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, planning = _planning_run(directory)
                original = getattr(planning.recipe_adapter, method)
                failed = getattr(ExceptionalAdapter(planning.capability), method)
                calls = []

                def transient_then_success(
                    tool_input, original=original, failed=failed, calls=calls
                ):
                    calls.append(tool_input)
                    if len(calls) < 3:
                        return failed(tool_input).model_copy(
                            update={"failure_class": "transient"}
                        )
                    return original(tool_input)

                with (
                    patch.object(
                        planning.recipe_adapter,
                        method,
                        side_effect=transient_then_success,
                    ),
                    patch("random.uniform", side_effect=lambda low, high: high),
                    patch("time.sleep") as sleep,
                ):
                    result = _advance(receipt, planning, answers=answers)
                self.assertNotEqual(result.status, "operational_review_pending")
                self.assertEqual(len(calls), 3)
                self.assertEqual(calls, [calls[0]] * 3)
                self.assertEqual(
                    [call.args[0] for call in sleep.call_args_list], [1, 2]
                )
                events = _events(receipt)
                executions = _executions(events, tool)
                self.assertEqual(len(executions), 1)
                failure = [
                    event
                    for event in events
                    if event["event_type"] == "ToolOperationFailureRecorded"
                ]
                self.assertEqual(len(failure), 2)
                self.assertEqual(
                    [event["payload"]["contact_attempt"] for event in failure], [1, 2]
                )
                self.assertEqual(
                    [event["payload"]["retry_delay_seconds"] for event in failure],
                    [1, 2],
                )
                payload = failure[0]["payload"]
                self.assertEqual(
                    payload["started_event_id"], executions[0][0]["event_id"]
                )
                self.assertEqual(payload["contact_attempt"], 1)
                self.assertEqual(payload["failure_class"], "transient")
                self.assertEqual(payload["original_failure_class"], "transient")
                self.assertEqual(payload["retry_delay_seconds"], 1)
                self.assertEqual(payload["policy"]["effective"]["retry_attempts"], 3)
                self.assertFalse(
                    any(
                        event["event_type"] == "OperationalFailureRecorded"
                        for event in events
                    )
                )
                path = Path(receipt.journey_file)
                retained = path.read_bytes()
                self.assertEqual(
                    review_journey(ReviewJourneyQuery(journey_file=str(path))).status,
                    result.status,
                )
                self.assertEqual(path.read_bytes(), retained)
                for field, value in (
                    ("contact_attempt", 2),
                    ("retry_delay_seconds", 2),
                    ("started_event_id", str(uuid.uuid4())),
                ):
                    with self.subTest(changed=field):
                        hostile = copy.deepcopy(events)
                        changed = next(
                            event
                            for event in hostile
                            if event["event_id"] == failure[0]["event_id"]
                        )
                        changed["payload"][field] = value
                        with self.assertRaisesRegex(
                            ValueError,
                            "observed Tool execution failure changed its scoped policy or attempt",
                        ):
                            project_journey_events(hostile, journey_file=str(path))

    def test_classified_capability_results_review_and_retry_only_the_failed_call(self):
        from tests.planning_fakes import ExceptionalAdapter

        for method, tool, answers in (
            ("list_recipes", "list_compatible_analysis_recipes", 0),
            ("lookup_resources", "lookup_compatible_resource_profiles", 1),
        ):
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, planning = _planning_run(directory)
                original = getattr(planning.recipe_adapter, method)
                failing = getattr(ExceptionalAdapter(planning.capability), method)
                with patch.object(
                    planning.recipe_adapter, method, side_effect=failing
                ) as fault:
                    stopped = _advance(receipt, planning, answers=answers)
                self.assertEqual(stopped.status, "operational_review_pending")
                review = stopped.current_review
                before = _events(receipt)
                classified = next(
                    e for e in before if e["event_type"] == "OperationalFailureRecorded"
                )
                self.assertEqual(classified["payload"]["failure_class"], "rejected")
                self.assertEqual(classified["payload"]["family"], "provider")
                self.assertEqual(classified["payload"]["contact_attempt"], 1)
                # One default policy asks; with no entry nothing tunes it, and
                # each record names its Tool surface.
                for recorded in [
                    e["payload"]["policy"]
                    for e in before
                    if e["event_type"]
                    in {"OperationalFailureRecorded", "ToolOperationFailureRecorded"}
                ]:
                    self.assertEqual(
                        recorded["surface"], {"scope": "tool", "name": tool}
                    )
                    self.assertEqual(recorded["surface_overrides"], {})
                    self.assertEqual(recorded["effective"]["semantic"], "ask")
                self.assertEqual(len(_executions(before, tool)), 1)
                hostile = copy.deepcopy(before)
                retry_fault = next(
                    event
                    for event in hostile
                    if event["event_type"] == "ToolOperationFailureRecorded"
                )
                for resolved in ("defaults", "effective"):
                    retry_fault["payload"]["policy"][resolved]["backoff_seconds"] = 2
                with self.assertRaisesRegex(
                    ValueError,
                    "observed Tool execution failure changed its terminal operation fault",
                ):
                    project_journey_events(hostile, journey_file=receipt.journey_file)
                with (
                    patch.object(
                        planning.recipe_adapter, method, side_effect=original
                    ) as retry,
                    use_run_advancement_composition(planning),
                ):
                    resumed = advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=review.request_event_id,
                            answer=JunctionReviewAnswer(
                                kind="junction",
                                response={
                                    "request_id": review.request.request.request_id,
                                    "action": "retry",
                                    "by": "researcher",
                                },
                            ),
                        )
                    )
                self.assertEqual(retry.call_args_list, fault.call_args_list)
                self.assertNotEqual(resumed.status, "operational_review_pending")
                after = _events(receipt)
                self.assertEqual(after[: len(before)], before)
                self.assertEqual(len(_executions(after, tool)), 2)
                self.assertNotIn("RunFailed", [e["event_type"] for e in after])
                path = Path(receipt.journey_file)
                retained = path.read_bytes()
                self.assertEqual(
                    review_journey(ReviewJourneyQuery(journey_file=str(path))).status,
                    resumed.status,
                )
                self.assertEqual(path.read_bytes(), retained)

    def test_same_response_resource_lookup_refuses_recipe_work_before_effects(self):
        class RecipeThenResource(PlanningComposition):
            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                part = response.parts[0]
                if part.tool_name == "select_recipe_candidate":
                    from bio_neb.agentic_journey.planning_assignment import (
                        planning_bound_values,
                    )

                    # Server-bound values never reach the model; read them
                    # where they are held, as the request that offered the Tool
                    # did.
                    bound = planning_bound_values(
                        SimpleNamespace(journey=Journey.open(Path(self.journey.path))),
                        part.tool_name,
                    )
                    selected = {**bound.constants, **part.args}
                    response.parts.append(
                        ToolCallPart(
                            "lookup_compatible_resource_profiles",
                            {
                                "work_item_id": selected["work_item_id"],
                                "recipe_candidate_id": selected["candidate_id"],
                            },
                            tool_call_id="resource-with-recipe-work",
                        )
                    )
                return response

        with tempfile.TemporaryDirectory() as directory:
            receipt, planning = _planning_run(directory, composition=RecipeThenResource)
            result = _advance(receipt, planning, answers=1)
            events = _events(receipt)
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
            self.assertIn(
                "select_recipe_candidate",
                planning.calls,
                (result.status, planning.calls),
            )
        refused = next(
            e
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_call_id"] == "resource-with-recipe-work"
        )
        self.assertEqual(
            refused["payload"]["errors"], ["work_item_id:bound_value_mismatch"]
        )
        self.assertFalse(
            any(
                e["event_type"] == "ObservedToolExecutionStarted"
                and e["payload"]["validation_event_id"] == refused["event_id"]
                for e in events
            )
        )
        turn = next(
            e["payload"]
            for e in events
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
            and refused["event_id"] in e["payload"]["validation_event_ids"]
        )
        self.assertEqual(turn["disposition"], "multi_call_refused")
        # The server-bound form names the field and the allowed form.
        self.assertIn("- Field `work_item_id`: you sent", turn["feedback"])
        self.assertIn(
            "Allowed form: omit it; the server supplies it.", turn["feedback"]
        )
        ((start, done),) = _executions(events, "lookup_compatible_resource_profiles")
        args = _by_id(events)[start["payload"]["validation_event_id"]]["payload"][
            "arguments"
        ]
        self.assertTrue(args["work_item_id"].startswith("resource-work:"))
        fact = _by_id(events)[done["payload"]["stage_result_event_id"]]
        self.assertEqual(
            args["work_item_id"],
            fact["payload"]["tool_input"]["work_item"]["work_item_id"],
        )

    def test_ungrounded_recipe_lookup_is_refused_before_the_eligibility_question(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, planning = _planning_run(
                directory, composition=_UngroundedFirstRecipeLookup
            )
            review = _advance(receipt, planning)
            events = _events(receipt)

        validations = [
            event
            for event in events
            if event["event_type"] == "ToolCallValidated"
            and event["payload"]["tool_name"] == "list_compatible_analysis_recipes"
        ]
        self.assertEqual(
            [
                (item["payload"]["status"], item["payload"]["errors"])
                for item in validations
            ],
            [("invalid", ["input_set_id:bound_value_mismatch"]), ("valid", [])],
        )
        ((list_start, list_done),) = _executions(
            events, "list_compatible_analysis_recipes"
        )
        ((ask_start, ask_done),) = _executions(
            events, "request_recipe_eligibility_assertion"
        )
        by_id = _by_id(events)
        observation = by_id[list_done["payload"]["stage_result_event_id"]]
        question = review.planning_hitl.pending_request
        self.assertEqual(planning.recipe_adapter.contact_count, 1)
        self.assertEqual(
            list_start["payload"]["validation_event_id"], validations[1]["event_id"]
        )
        self.assertEqual(
            list_start["payload"]["adapter_identity"], "compiled_capability_snapshot"
        )
        self.assertEqual(observation["event_type"], "RecipeObservationRecorded")
        self.assertEqual(ask_start["payload"]["adapter_identity"], "deterministic")
        self.assertEqual(
            ask_done["payload"]["stage_result_event_id"], question.request_event_id
        )
        self.assertEqual(
            ask_done["payload"]["model_visible_result_digest"],
            _recorded_event_digest(
                ask_start["payload"]["tool_name"], by_id[question.request_event_id]
            ),
        )
        self.assertEqual(question.evidence_event_ids, (observation["event_id"],))
        self.assertEqual(review.status, "planning_hitl_pending")

    def test_agent_selections_are_bound_to_their_admitted_calls_until_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, planning = _planning_run(directory)
            review = _advance(receipt, planning, answers=1)
            events = _events(receipt)

        by_id = _by_id(events)
        executions = [
            pair
            for name in (
                "select_recipe_candidate",
                "lookup_compatible_resource_profiles",
                "select_resource_candidate",
            )
            for pair in _executions(events, name)
        ]
        self.assertEqual(
            [
                (
                    start["payload"]["tool_name"],
                    start["payload"]["adapter_identity"],
                    by_id[done["payload"]["stage_result_event_id"]]["event_type"],
                )
                for start, done in executions
            ],
            [
                (
                    "select_recipe_candidate",
                    "deterministic",
                    "PlanningCandidateSelected",
                ),
                (
                    "lookup_compatible_resource_profiles",
                    "compiled_capability_snapshot",
                    "ResourceObservationRecorded",
                ),
                (
                    "select_resource_candidate",
                    "deterministic",
                    "PlanningCandidateSelected",
                ),
            ],
        )
        for start, done in (executions[0], executions[2]):
            selected = by_id[done["payload"]["stage_result_event_id"]]["payload"]
            self.assertEqual(selected["authority"], "agent_ranking")
            self.assertEqual(
                selected["authority_event_id"],
                start["payload"]["validation_event_id"],
            )
        self.assertEqual(planning.recipe_adapter.contact_count, 3)
        self.assertEqual(review.execution_planning.status, "execution_planning_ready")

    def test_researcher_recipe_choice_then_incompatible_resource_returns_to_the_model(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            receipt, planning = _planning_run(
                directory,
                capability=policy(recipes=2),
                composition=_ReadsNegativeResult,
            )
            review = _advance(receipt, planning, answers=2)
            events = _events(receipt)

        by_id = _by_id(events)
        answer = next(
            event
            for event in events
            if event["event_type"] == "PlanningHitlResponded"
            and by_id[event["payload"]["request_event_id"]]["payload"]["question"]
            .lower()
            .startswith("which exact recipe candidate")
        )
        ((select_start, select_done),) = _executions(events, "select_recipe_candidate")
        ((_, lookup_done),) = _executions(events, "lookup_compatible_resource_profiles")
        selected = by_id[select_done["payload"]["stage_result_event_id"]]["payload"]
        resource = by_id[lookup_done["payload"]["stage_result_event_id"]]
        self.assertEqual(
            (selected["authority"], selected["authority_event_id"]),
            ("researcher_selection", answer["event_id"]),
        )
        self.assertNotEqual(
            select_start["payload"]["validation_event_id"], answer["event_id"]
        )
        self.assertTrue(
            all(
                item["compatibility"] == "incompatible"
                for item in resource["payload"]["result"]["candidates"]
            )
        )
        # The model's next request carries the exact Tool return; no typed-failure
        # terminal ended the Attempt first.
        from bio_neb.agentic_journey.tool_execution import (
            model_view_of,
            tool_model_view,
        )

        self.assertIn(
            (
                "lookup_compatible_resource_profiles",
                tool_model_view(
                    model_view_of("lookup_compatible_resource_profiles"), resource
                ),
            ),
            planning.tool_returns,
            "the resource result did not reach the model",
        )
        after = events[events.index(resource) :]
        self.assertIn("ModelRequestRecorded", [e["event_type"] for e in after])
        self.assertNotIn("SliceCheckpointReached", [e["event_type"] for e in after])
        self.assertTrue(negative_domain_result("planning_agent", resource))
        self.assertNotEqual(review.status, "execution_planning_ready")

    def test_no_compatible_recipe_returns_to_the_model_before_any_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, planning = _planning_run(
                directory,
                capability=policy(recipes=0),
                composition=_ReadsNegativeResult,
            )
            _advance(receipt, planning)
            events = _events(receipt)

        by_id = _by_id(events)
        ((_, listed),) = _executions(events, "list_compatible_analysis_recipes")
        recipe = by_id[listed["payload"]["stage_result_event_id"]]
        self.assertEqual(recipe["payload"]["result"]["outcome"], "missing")
        from bio_neb.agentic_journey.tool_execution import (
            model_view_of,
            tool_model_view,
        )

        self.assertIn(
            (
                "list_compatible_analysis_recipes",
                tool_model_view(
                    model_view_of("list_compatible_analysis_recipes"), recipe
                ),
            ),
            planning.tool_returns,
            "the recipe result did not reach the model",
        )
        self.assertNotIn(
            "SliceCheckpointReached",
            [e["event_type"] for e in events[events.index(recipe) :]],
        )
        self.assertTrue(negative_domain_result("planning_agent", recipe))

    def test_selection_and_dependency_questions_are_observed_requests(self):
        cases = (
            (
                "request_recipe_selection",
                dict(capability=policy(recipes=2)),
                1,
                "recipe-candidate-disposition:",
            ),
            (
                "request_resource_selection",
                dict(capability=policy(resources=2)),
                1,
                "resource-candidate-disposition:",
            ),
            (
                "request_resource_recipe_dependency",
                dict(composition=_TwoRecipeRequirements),
                2,
                "resource-recipe-dependency:",
            ),
        )
        for tool_name, setup, answers, candidate_prefix in cases:
            with (
                self.subTest(tool_name=tool_name),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, planning = _planning_run(directory, **setup)
                review = _advance(receipt, planning, answers=answers)
                events = _events(receipt)
                question = review.planning_hitl.pending_request
                ((start, done),) = _executions(events, tool_name)
                request = _by_id(events)[done["payload"]["stage_result_event_id"]]
                self.assertEqual(start["payload"]["adapter_identity"], "deterministic")
                self.assertEqual(request["event_id"], question.request_event_id)
                self.assertEqual(request["event_type"], "PlanningHitlRequested")
                self.assertEqual(
                    done["payload"]["model_visible_result_digest"],
                    _recorded_event_digest(tool_name, request),
                )
                self.assertEqual(len(question.candidates), 2)
                self.assertTrue(
                    all(
                        item.candidate_id.startswith(candidate_prefix)
                        for item in question.candidates
                    )
                )

    def test_capability_failures_retain_their_exact_stage_failure(self):
        cases = (
            ("list_compatible_analysis_recipes", "list_recipes", False, 0),
            ("lookup_compatible_resource_profiles", "lookup_resources", True, 2),
        )
        for tool_name, method, after_question, contacts in cases:
            with (
                self.subTest(tool_name=tool_name),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, planning = _advance_into_capability_failure(
                    self, directory, method, after_question=after_question
                )
                events = _events(receipt)
                ((start, failed),) = _executions(events, tool_name)
                stage_failure = next(
                    event
                    for event in events
                    if event["event_type"] == "ToolExecutionFailed"
                    and event["payload"]["tool_name"] == tool_name
                )
                attempt_failed = next(
                    event for event in events if event["event_type"] == "AttemptFailed"
                )
                self.assertEqual(
                    start["payload"]["adapter_identity"],
                    "compiled_capability_snapshot",
                )
                self.assertEqual(stage_failure["payload"]["error_type"], "RuntimeError")
                self.assertEqual(
                    failed["payload"],
                    {
                        "attempt_id": start["payload"]["attempt_id"],
                        "started_event_id": start["event_id"],
                        "mechanical_error_type": "RuntimeError",
                        "stage_failure_event_id": stage_failure["event_id"],
                    },
                )
                self.assertEqual(
                    attempt_failed["payload"]["source_event_id"],
                    stage_failure["event_id"],
                )
                self.assertEqual(planning.recipe_adapter.contact_count, contacts)

    def test_planning_attempt_failed_by_a_capability_error_reads_back(self):
        # A recipe stage failure closes only its own Tool's evidence.
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _advance_into_capability_failure(
                self, directory, "list_recipes"
            )
            review = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
        self.assertEqual(review.attempts[-1].outcome, "failed")
        self.assertEqual(review.status, "run_held")

    def test_unexpected_selection_request_failure_is_generically_observed(self):
        from bio_neb.application.advancement import RunAdvancementRefused

        failing = patch(
            "bio_neb.parabricks.execution_planning_agent._projection",
            side_effect=RuntimeError("private deterministic detail"),
        )

        started = []

        class FailAtRecipeSelection(PlanningComposition):
            async def respond(self, messages, info):
                if not started and "request_recipe_selection" in eligible_tools(
                    self.journey, info.function_tools
                ):
                    started.append(failing.start())
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            receipt, planning = _planning_run(
                directory,
                capability=policy(recipes=2),
                composition=FailAtRecipeSelection,
            )
            try:
                with self.assertRaises(RunAdvancementRefused) as refused:
                    _advance(receipt, planning, answers=1)
            finally:
                if started:
                    failing.stop()
            events = _events(receipt)

        # The model is shown the failure once; the same failure again holds the
        # Run with its cause.
        self.assertEqual(
            refused.exception.failure.invariant_id, "planning_attempt_failed"
        )
        (shown_start, shown), (start, failed) = _executions(
            events, "request_recipe_selection"
        )
        (refusal,) = (e for e in events if e["event_type"] == "ToolExecutionRefused")
        self.assertEqual(
            (refusal["payload"]["failed_event_id"], refusal["payload"]["fault_key"]),
            (shown["event_id"], "request_recipe_selection: RuntimeError"),
        )
        attempt_failed = next(
            event for event in events if event["event_type"] == "AttemptFailed"
        )
        self.assertEqual(start["payload"]["adapter_identity"], "deterministic")
        self.assertEqual(
            failed["payload"],
            {
                "attempt_id": start["payload"]["attempt_id"],
                "started_event_id": start["event_id"],
                "mechanical_error_type": "RuntimeError",
                "stage_failure_event_id": None,
            },
        )
        self.assertFalse(
            any(
                event["event_type"] == "PlanningHitlRequested"
                and event["payload"]["attempt_id"] == start["payload"]["attempt_id"]
                for event in events
            )
        )
        self.assertEqual(
            attempt_failed["payload"]["source_event_id"], failed["event_id"]
        )
        self.assertEqual(planning.recipe_adapter.contact_count, 2)

    def test_execution_revision_reenters_planning_with_its_instruction_visible(self):
        instruction = "Revalidate execution policy exactly from retained facts."
        with tempfile.TemporaryDirectory() as directory:
            receipt, planning, _ = configured_planning_run(Path(directory))
            planning.recipe_adapter = CapabilitySnapshotAdapter(planning.capability)
            with use_run_advancement_composition(planning):
                review = advance(ResumeRun(journey_file=receipt.journey_file))
                while (
                    review.current_review is not None
                    and review.current_review.kind == "planning"
                ):
                    review = _answer_first_candidate(receipt, review)
                proposal = review.execution_hitl.pending_request.proposal
                contacts = planning.recipe_adapter.contact_count
                before = len(_events(receipt))
                revised = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=review.execution_hitl.pending_request.request_event_id,
                        answer=ExecutionReviewAnswer(
                            kind="execution",
                            response=ReviseExecutionResponse(
                                action="revise",
                                proposal_id=proposal.proposal_id,
                                instruction=instruction,
                                revalidation_scope="execution_planning",
                            ),
                        ),
                    )
                )
            reentry = _events(receipt)[before:]

        requests = [
            (index, event)
            for index, event in enumerate(reentry)
            if event["event_type"] == "ModelRequestRecorded"
        ]
        first_request = canonical_json(
            requests[0][1]["payload"]["compatible_adapter_evidence"]["request"]
        )
        # A revise first reopens the genomic FASTA's index choice, its words shown
        # with it; execution planning then reopens on the superseded proposal.
        self.assertIn(instruction, first_request)
        (rechosen,) = [
            index
            for index, event in enumerate(reentry)
            if event["event_type"] == "PrebuiltIndexChosen"
        ]
        execution_request = canonical_json(
            next(event for index, event in requests if index > rechosen)["payload"][
                "compatible_adapter_evidence"
            ]["request"]
        )
        self.assertIn("superseded_proposal", execution_request)
        self.assertIn(proposal.proposal_id, execution_request)
        self.assertIn(proposal.companions[0].candidate.asset_id, execution_request)
        self.assertEqual(
            {
                event["payload"]["agent_role"]
                for event in reentry
                if event["event_type"] == "AttemptStarted"
            },
            {"planning_agent"},
        )
        self.assertEqual(planning.recipe_adapter.contact_count, contacts + 1)
        self.assertEqual(revised.status, "planning_hitl_pending")


if __name__ == "__main__":
    unittest.main()
