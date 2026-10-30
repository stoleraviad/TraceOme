import asyncio
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from bio_neb.agentic_journey.adapters import create_nebius_model_endpoint
from bio_neb.agentic_journey.planning_assignment import (
    planning_assignment,
)
from bio_neb.application.planning_read import read_planning_events
from bio_neb.application.queries import (
    ObserveRun,
    ReviewJourneyQuery,
    observe,
    project_journey_events,
    read_investigation,
    read_planning,
    read_pre_execution,
    review_journey,
)
from bio_neb.investigation.investigation import project_investigation
from bio_neb.investigation.investigation_commands import (
    BeginInvestigationCommand,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.recording import read_evidence
from tests.execution_approval import approve_execution
from tests.input_correction_fakes import prepare_input_correction
from tests.proposal_information_fakes import registered_run
from tests.sequential_investigation_support import (
    SourceResponsesOnly,
    begin_in_pass,
    forked_child,
)


def transport_factory(respond, clients):
    def factory(loaded, selected):
        adapter = create_nebius_model_endpoint(loaded, selected)
        clients.append(adapter.client._client)
        adapter.client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=False
        )
        clients.append(adapter.client._client)
        return adapter

    return factory


async def close_clients(clients):
    for client in clients:
        await client.aclose()


def tool_call_completion(
    body, *, response_id, call_id, tool_name, arguments, content=None
):
    """One compatible chat completion that calls one Tool."""
    return httpx.Response(
        200,
        json={
            "id": response_id,
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                }
            ],
        },
    )


def truncated_completion(body, *, response_id, output_tokens, content="partial"):
    """One compatible chat completion cut off on its output ceiling."""
    return httpx.Response(
        200,
        json={
            "id": response_id,
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": output_tokens,
                "total_tokens": 100 + output_tokens,
            },
        },
    )


def grounded_requirements(journey_file, registered, paths):
    """A grounded proposal draft whose sample identity remains to be validated."""
    from types import SimpleNamespace

    from tests.application_fakes import _grounded_draft

    inspected = next(
        e
        for e in Journey.open(Path(journey_file)).read()
        if e["event_type"] == "InputMetadataInspected"
    )
    arguments = _grounded_draft(
        SimpleNamespace(
            request=registered.request,
            registered_input_set=registered.registered_input_set,
            inspection=SimpleNamespace(source_event_id=inspected["event_id"]),
        ),
        paths,
    )
    sample = next(
        item for item in arguments["requirements"] if item["kind"] == "sample_identity"
    )
    sample["knowledge"] = {
        "state": "to_find",
        "evidence_event_ids": [registered.request.source_event_id],
    }
    sample["acquisition_route"] = "validate_with_tool"
    return arguments


def offered_tool(body, tool_name):
    return next(
        t["function"] for t in body["tools"] if t["function"]["name"] == tool_name
    )


def bound_arguments(tool):
    """The constant arguments an offered Tool schema binds."""
    return {
        key: value["const"]
        for key, value in tool["parameters"]["properties"].items()
        if "const" in value
    }


def chooser_transport_composition(chooser, requests, clients):
    """Drive a scripted planning chooser through the real transport Adapter."""
    from types import SimpleNamespace

    from bio_neb.parabricks.capability_policy import CapabilitySnapshotAdapter

    async def respond(request):
        requests.append(request)
        body = json.loads(request.content)
        info = SimpleNamespace(
            instructions="retained actual transport instruction",
            function_tools=[
                SimpleNamespace(
                    name=t["function"]["name"],
                    parameters_json_schema=t["function"]["parameters"],
                )
                for t in body["tools"]
            ],
        )
        action = (await chooser.respond(body["messages"], info)).parts[0]
        return tool_call_completion(
            body,
            response_id=f"planning-{len(requests)}",
            call_id=action.tool_call_id,
            tool_name=action.tool_name,
            arguments=action.args,
        )

    class TransportComposition:
        environ = chooser.environ

        def planning_assignment(self, journey):
            chooser.journey = journey
            chooser.registered = review_journey(
                ReviewJourneyQuery(journey_file=str(journey.path))
            )
            return planning_assignment(
                model_factory=transport_factory(respond, clients),
                reference_adapter=chooser.reference_adapter,
                companion_adapter=chooser.companion_adapter,
                galaxy_adapter=chooser.galaxy_adapter,
                recipe_adapter=CapabilitySnapshotAdapter(chooser.capability),
            )

    return TransportComposition()


def select_planning_reviews(result, journey_file, *, limit=3):
    """Select each planning question's first candidate, at most ``limit`` times."""
    from bio_neb.application.advancement import (
        AnswerReview,
        PlanningReviewAnswer,
        advance,
    )
    from bio_neb.parabricks.planning_hitl import SelectPlanningHitlResponse

    for _ in range(limit):
        if result.current_review is None or result.current_review.kind != "planning":
            break
        question = result.current_review.request
        result = advance(
            AnswerReview(
                journey_file=journey_file,
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
    return result


def plan_without_preparation_configuration(root):
    """Drive transport-aware planning on a Run lacking Parabricks settings."""
    from bio_neb.application.advancement import (
        ResumeRun,
        advance,
        use_run_advancement_composition,
    )
    from bio_neb.parabricks.accepted_pre_execution import (
        _accepted_capability_snapshot,
    )
    from tests.planning_assignment_fakes import configured_planning_run

    receipt, chooser, _ = configured_planning_run(root, supply_configuration=False)
    chooser.capability = _accepted_capability_snapshot(
        Journey.open(Path(receipt.journey_file))
    )
    requests, clients = [], []
    composition = chooser_transport_composition(chooser, requests, clients)
    try:
        with use_run_advancement_composition(composition):
            result = select_planning_reviews(
                advance(ResumeRun(journey_file=receipt.journey_file)),
                receipt.journey_file,
            )
    finally:
        asyncio.run(close_clients(clients))
    return receipt, chooser, requests, result


class TransportAllowanceBoundTests(unittest.TestCase):
    def fact(self, final_max_completion_tokens):
        from bio_neb.agentic_journey.model_settings import ModelContextBudget
        from bio_neb.agentic_journey.model_transport import (
            ModelTransportRequestRecorded,
        )

        return ModelTransportRequestRecorded(
            attempt_id="00000000-0000-4000-8000-000000000001",
            request_number=58,
            logical_request_event_id="00000000-0000-4000-8000-000000000002",
            contact_started_event_id="00000000-0000-4000-8000-000000000003",
            contact_ordinal=1,
            unbudgeted_body_byte_count=254315,
            unbudgeted_body_sha256="a" * 64,
            request_body_byte_count=254315,
            request_body_sha256="b" * 64,
            requested_max_tokens=245760,
            final_max_completion_tokens=final_max_completion_tokens,
            context_budget=ModelContextBudget(
                context_window=262144,
                input_reserve=16384,
                serialized_body_safety_margin=8192,
            ),
        )

    def test_recorded_allowance_may_not_exceed_the_estimated_room(self):
        # A recorded request body: 254,315 bytes estimate 127,158 tokens,
        # which leave 262,144 - 127,158 - 8,192 = 126,794 completion tokens.
        self.assertEqual(self.fact(126794).final_max_completion_tokens, 126794)
        with self.assertRaisesRegex(ValueError, "final allowance exceeds its budget"):
            self.fact(126795)


class ModelTransportEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch(
                "bio_neb.agentic_journey.model_observation.random.random",
                return_value=0.0,
            )
        )

    def test_valid_planning_action_after_repairs_keeps_transport_evidence_for_next_offer(
        self,
    ):
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.recording import read_evidence
        from tests.application_fakes import _registered
        from tests.planning_assignment_fakes import PlanningComposition

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            journey = Journey.open(Path(receipt.journey_file))
            registered = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            requests, clients = [], []

            def respond(request):
                requests.append(request)
                body = json.loads(request.content)
                if len(requests) <= 4:
                    arguments = grounded_requirements(
                        receipt.journey_file, registered, paths
                    )
                    if len(requests) <= 3:
                        next(
                            item
                            for item in arguments["requirements"]
                            if item["kind"] == "sample_identity"
                        )["subject_key"] = "Invalid Identifier"
                    tool_name = "record_proposal_requirements"
                else:
                    tool_name = "request_planning_hitl"
                    tool = offered_tool(body, tool_name)
                    # The wire offers no bound value to restate.
                    self.assertEqual(tool["parameters"]["properties"], {})
                    arguments = bound_arguments(tool)
                    self.assertEqual(arguments, {})
                return tool_call_completion(
                    body,
                    response_id=f"planning-{len(requests)}",
                    call_id=f"action-{len(requests)}",
                    tool_name=tool_name,
                    arguments=arguments,
                )

            class TransportPlanning(PlanningComposition):
                def planning_assignment(self, _journey):
                    nonlocal journey
                    journey = _journey
                    return planning_assignment(
                        model_factory=transport_factory(respond, clients),
                        reference_adapter=self.reference_adapter,
                        companion_adapter=self.companion_adapter,
                        galaxy_adapter=self.galaxy_adapter,
                    )

            composition = TransportPlanning(receipt, paths)
            try:
                with use_run_advancement_composition(composition):
                    result = advance(ResumeRun(journey_file=receipt.journey_file))
            finally:
                asyncio.run(close_clients(clients))
            self.assertEqual(result.status, "planning_hitl_pending")
            self.assertEqual(len(requests), 5)
            self.assertEqual(composition.reference_adapter.contact_count, 0)
            self.assertEqual(composition.companion_adapter.contact_count, 0)
            reopened = Journey.open(journey.path)
            self.assertEqual(
                observe(
                    ObserveRun(journey_file=receipt.journey_file), journey=reopened
                ),
                result,
            )
            evidence = read_evidence(reopened)
            validations = [
                e for e in evidence if e["event_type"] == "ToolCallValidated"
            ]
            self.assertEqual(
                [e["payload"]["status"] for e in validations],
                ["invalid"] * 3 + ["valid"] * 2,
            )
            # The binding lives in the recorded request, not the offered schema, and
            # the admitted call records its effective arguments and bound names.
            question = validations[-1]["payload"]
            self.assertEqual(question["tool_name"], "request_planning_hitl")
            recorded_tool = next(
                tool
                for tool in [
                    e for e in evidence if e["event_type"] == "ModelRequestRecorded"
                ][-1]["payload"]["compatible_adapter_evidence"]["request"][
                    "function_tools"
                ]
                if tool["name"] == "request_planning_hitl"
            )
            self.assertEqual(
                recorded_tool["metadata"]["bound_arguments"], question["arguments"]
            )
            self.assertEqual(
                question["bound_argument_names"],
                ["evidence_event_id", "requirement_id"],
            )

            self.assertEqual(
                sum(
                    e["event_type"] == "ProposalRequirementsIdentified"
                    for e in evidence
                ),
                1,
            )
            self.assertFalse(any(e["event_type"] == "AttemptFailed" for e in evidence))

            changed = reopened.read()
            transports = [
                e for e in changed if e["event_type"] == "ModelTransportRequestRecorded"
            ]
            transports[-1]["payload"]["contact_started_event_id"] = transports[0][
                "payload"
            ]["contact_started_event_id"]
            journey.path.write_text(
                "".join(canonical_json(e) + "\n" for e in changed), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "model transport changed its open physical contact"
            ):
                read_planning(Journey.open(journey.path))
            self.assertEqual(len(requests), 5)

    def test_transport_planning_admits_source_and_execution_review_then_approves_once(
        self,
    ):
        from bio_neb.application.advancement import (
            AnswerReview,
            ExecutionReviewAnswer,
            ResumeRun,
            advance,
            use_run_advancement_composition,
            use_run_advancement_dependencies,
        )
        from bio_neb.parabricks.execution_proposal import (
            ApproveExecutionResponse,
            ExecutionHitlResponseRefused,
        )
        from bio_neb.recording import read_evidence
        from tests.planning_assignment_fakes import (
            configured_planning_run,
            configured_provider,
        )

        with tempfile.TemporaryDirectory() as directory:
            receipt, chooser, configuration = configured_planning_run(
                Path(directory).resolve()
            )
            requests, clients, effects = [], [], []
            composition = chooser_transport_composition(chooser, requests, clients)
            try:
                with (
                    use_run_advancement_composition(composition),
                    use_run_advancement_dependencies(
                        parabricks_infrastructure_factory=configured_provider(
                            receipt.journey_file, effects
                        )
                    ),
                ):
                    result = select_planning_reviews(
                        advance(ResumeRun(journey_file=receipt.journey_file)),
                        receipt.journey_file,
                    )
                    self.assertEqual(result.status, "execution_hitl_pending")
                    self.assertEqual(effects, [])
                    request = result.execution_hitl.pending_request
                    self.assertEqual(request.proposal.configuration, configuration)
                    self.assertEqual(
                        request.proposal.reference_source_admission.assembly_accession,
                        "GCF_000001405.40",
                    )
                    journey = Journey.open(Path(receipt.journey_file))
                    evidence = read_evidence(journey)
                    for kind in (
                        "ModelPhysicalContactStarted",
                        "ModelTransportRequestRecorded",
                        "ReferenceSourceAdmitted",
                        "ExecutionPackageCompiled",
                        "ExecutionProposalSubmitted",
                    ):
                        self.assertTrue(
                            any(e["event_type"] == kind for e in evidence), kind
                        )
                    self.assertEqual(
                        observe(ObserveRun(journey_file=receipt.journey_file)), result
                    )
                    answer = AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=request.request_event_id,
                        answer=ExecutionReviewAnswer(
                            kind="execution",
                            response=approve_execution(request.proposal),
                        ),
                    )
                    before = journey.path.read_bytes()
                    with self.assertRaises(ExecutionHitlResponseRefused) as illegal:
                        advance(
                            answer.model_copy(
                                update={
                                    "answer": ExecutionReviewAnswer(
                                        kind="execution",
                                        response=ApproveExecutionResponse(
                                            action="approve",
                                            proposal_id="execution-proposal:"
                                            + "0" * 64,
                                        ),
                                    ),
                                }
                            )
                        )
                    self.assertEqual(
                        illegal.exception.code, "execution_hitl_response_illegal"
                    )
                    self.assertEqual(journey.path.read_bytes(), before)
                    self.assertEqual(effects, [])
                    completed = advance(answer)
                    # The approval's call verifies Execution, then begins Investigation
                    # from the plan's target; this transport serves planning only.
                    from bio_neb.parabricks.result import (
                        read_completed_parabricks_result,
                    )

                    self.assertEqual(
                        read_completed_parabricks_result(
                            receipt.journey_file
                        ).disposition,
                        "succeeded",
                    )
                    self.assertEqual(len(effects), 1)
                    self.assertEqual(
                        effects[0].identity.proposal_id, request.proposal.proposal_id
                    )
                    before = journey.path.read_bytes()
                    count = len(requests)
                    self.assertEqual(advance(answer), completed)
                    self.assertEqual(journey.path.read_bytes(), before)
                    self.assertEqual(len(effects), 1)
                    self.assertEqual(len(requests), count)
            finally:
                asyncio.run(close_clients(clients))

    def test_checked_planning_readers_keep_physical_contacts_and_replay_prefixes(
        self,
    ):
        from types import SimpleNamespace

        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.application.planning_read import (
            current_planning,
            planning_evidence,
        )
        from tests.planning_assignment_fakes import configured_planning_run

        with tempfile.TemporaryDirectory() as directory:
            receipt, chooser, _ = configured_planning_run(Path(directory).resolve())
            requests, clients = [], []
            composition = chooser_transport_composition(chooser, requests, clients)
            try:
                with use_run_advancement_composition(composition):
                    advance(ResumeRun(journey_file=receipt.journey_file))
            finally:
                asyncio.run(close_clients(clients))
            journey = Journey.open(Path(receipt.journey_file))
            semantic = journey.read()
            transports = [
                e
                for e in semantic
                if e["event_type"] == "ModelTransportRequestRecorded"
            ]
            self.assertGreater(len(requests), 0)
            self.assertEqual(len(transports), len(requests))
            named = {e["payload"]["contact_started_event_id"] for e in transports}
            # The semantic-only prefix drops the contacts its transport facts name.
            self.assertFalse(named & {e["event_id"] for e in semantic})
            with self.assertRaisesRegex(
                ValueError, "model transport changed its open physical contact"
            ):
                read_planning_events(semantic)

            evidence = planning_evidence(journey)
            self.assertLessEqual(
                named,
                {
                    e["event_id"]
                    for e in evidence
                    if e["event_type"] == "ModelPhysicalContactStarted"
                },
            )
            semantic_ids = {e["event_id"] for e in semantic}
            self.assertEqual(
                [e for e in evidence if e["event_id"] in semantic_ids], semantic
            )
            self.assertEqual(current_planning(journey), read_planning(journey))
            # Agent callers share the owner's checked events read-only; the
            # list is theirs, the facts are not.
            with self.assertRaises(TypeError):
                evidence[0]["payload"]["changed_by_caller"] = True
            with self.assertRaises(TypeError):
                evidence[-1]["parent_ids"].append(evidence[0]["event_id"])
            evidence.append({"event_id": "caller-only"})
            self.assertEqual(planning_evidence(journey), evidence[:-1])
            self.assertNotIn(
                "changed_by_caller", planning_evidence(journey)[0]["payload"]
            )

            # Historical replay keeps its exact merged prefix; nothing reopens.
            end = max(evidence.index(e) for e in evidence if e["event_id"] in named)
            prefix = planning_evidence(journey)[: end + 1]
            replay = SimpleNamespace(read=lambda: prefix)
            self.assertEqual(planning_evidence(replay), prefix)
            self.assertEqual(current_planning(replay), read_planning_events(prefix))
            self.assertEqual(len(requests), len(transports))

    def test_preparation_context_reads_no_transport_or_contact_evidence(self):
        """The writer's semantic record yields the merged-evidence preparation context."""
        from bio_neb.application.planning_read import planning_evidence
        from bio_neb.parabricks.preparation_review import preparation_context

        with tempfile.TemporaryDirectory() as directory:
            receipt, _, _, _ = plan_without_preparation_configuration(
                Path(directory).resolve()
            )
            journey = Journey.open(Path(receipt.journey_file))
            semantic = journey.read()
            evidence = planning_evidence(journey)
            self.assertTrue(
                any(
                    e["event_type"] == "ModelTransportRequestRecorded" for e in semantic
                )
            )
            self.assertFalse(
                any(e["event_type"] == "ModelPhysicalContactStarted" for e in semantic)
            )
            expected = preparation_context(evidence)
            self.assertTrue(expected["resources"])
            self.assertEqual(preparation_context(semantic), expected)
            # Emptied transport facts change nothing: preparation never reads them.
            emptied = [
                {**e, "payload": {}}
                if e["event_type"] == "ModelTransportRequestRecorded"
                else e
                for e in semantic
            ]
            self.assertEqual(preparation_context(emptied), expected)

    def test_transport_planning_reaches_the_preparation_review_and_reopens(self):
        """Transport-aware planning reaches the coordinator's preparation Review.

        The coordinator reads checked evidence; the Journey writer's preparation
        review admission recompiles preparation meaning over its semantic-only prefix.
        """
        from bio_neb.recording import read_evidence

        with tempfile.TemporaryDirectory() as directory:
            receipt, chooser, requests, result = plan_without_preparation_configuration(
                Path(directory).resolve()
            )
            self.assertIsNotNone(result.current_review, result.status)
            self.assertEqual(result.current_review.kind, "preparation")
            self.assertNotIn("request_preparation_information", chooser.calls)
            reopened = Journey.open(Path(receipt.journey_file))
            evidence = read_evidence(reopened)
            self.assertEqual(
                sum(e["event_type"] == "ModelPhysicalContactStarted" for e in evidence),
                len(requests),
            )
            self.assertEqual(
                sum(
                    e["event_type"] == "ModelTransportRequestRecorded" for e in evidence
                ),
                len(requests),
            )
            self.assertFalse(any(e["event_type"] == "AttemptFailed" for e in evidence))
            self.assertEqual(
                observe(
                    ObserveRun(journey_file=receipt.journey_file), journey=reopened
                ),
                result,
            )

    def test_history_reads_stay_bounded_and_reopen(self):
        self._exercise_history_transport()

    def test_planning_agent_retrieves_referenced_history_payload_before_continuing(
        self,
    ):
        self._exercise_history_transport(retrieve_payload=True)

    def _exercise_history_transport(self, retrieve_payload=False):
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.recording import read_evidence
        from tests.application_fakes import _registered
        from tests.planning_assignment_fakes import PlanningComposition

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            journey = Journey.open(Path(receipt.journey_file))
            registered = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            requests, clients = [], []
            payload_parts = []
            payload_binding = {}
            offered_history_schemas = []

            def respond(request):
                requests.append(request)
                body = json.loads(request.content)
                offered_history_schemas.append(
                    next(
                        tool["function"]["parameters"]
                        for tool in body["tools"]
                        if tool["function"]["name"] == "search_journey"
                    )
                )
                number = len(requests)
                if retrieve_payload and number in (5, 6, 7):
                    # The newest Tool result; a refused call's answer in its slot is
                    # a failed result, not a read. A search reads as {"status":
                    # "read", "result":..., "next":...}.
                    returned = next(
                        shown["result"]
                        for message in reversed(body["messages"])
                        if message["role"] == "tool"
                        and (shown := json.loads(message["content"])).get("status")
                        == "read"
                    )
                    if number == 5:
                        reference = next(
                            event["payload_reference"]
                            for event in returned["events"]
                            if event["event_type"] == "ModelResponseRecorded"
                        )
                        payload_binding.update(event_id=reference["event_id"])
                    else:
                        chunk = returned["payload_slice"]
                        self.assertEqual(chunk["event_id"], payload_binding["event_id"])
                        payload_parts.append(chunk["text"])
                        if number == 6:
                            self.assertEqual(chunk["next_offset"], 128)
                        else:
                            self.assertIsNone(chunk["next_offset"])
                            recovered = json.loads("".join(payload_parts))
                            self.assertEqual(recovered["request_number"], 1)
                            self.assertIn("outside-this-run", "".join(payload_parts))
                if number <= 4:
                    tool_name = "search_journey"
                    arguments = {"limit": 100}
                    if number == 1:
                        arguments["before_event_id"] = "outside-this-run"
                elif retrieve_payload and number in (5, 6):
                    tool_name = "search_journey"
                    arguments = {
                        "payload_slice": {
                            **payload_binding,
                            "offset": 0 if number == 5 else 128,
                            "length": 128 if number == 5 else 4096,
                        }
                    }
                elif number == (7 if retrieve_payload else 5):
                    tool_name = "record_proposal_requirements"
                    arguments = grounded_requirements(
                        receipt.journey_file, registered, paths
                    )
                else:
                    tool_name = "request_planning_hitl"
                    arguments = bound_arguments(offered_tool(body, tool_name))
                return tool_call_completion(
                    body,
                    response_id=f"history-{number}",
                    call_id=f"history-action-{number}",
                    tool_name=tool_name,
                    arguments=arguments,
                )

            class HistoryPlanning(PlanningComposition):
                def planning_assignment(self, owner):
                    nonlocal journey
                    journey = owner
                    return planning_assignment(
                        model_factory=transport_factory(respond, clients),
                        reference_adapter=self.reference_adapter,
                        companion_adapter=self.companion_adapter,
                        galaxy_adapter=self.galaxy_adapter,
                    )

            composition = HistoryPlanning(receipt, paths)
            try:
                with use_run_advancement_composition(composition):
                    result = advance(ResumeRun(journey_file=receipt.journey_file))
            finally:
                asyncio.run(close_clients(clients))
            if retrieve_payload:
                self.assertIn("payload_slice", offered_history_schemas[0]["properties"])
            self.assertEqual(result.status, "planning_hitl_pending")
            self.assertEqual(len(requests), 8 if retrieve_payload else 6)
            self.assertEqual(composition.reference_adapter.contact_count, 0)
            self.assertEqual(
                observe(ObserveRun(journey_file=receipt.journey_file)), result
            )
            events = read_evidence(Journey.open(journey.path))
            histories = [e for e in events if e["event_type"] == "JourneySearchRead"]
            # Requests 3 and 4 repeat request 2's search with only the loop's own
            # records added: the loop guard refuses them before they run.
            self.assertEqual(len(histories), 3 if retrieve_payload else 1)
            self.assertEqual(
                [
                    e["payload"]["tool_call_id"]
                    for e in events
                    if e["event_type"] == "ToolCallValidated"
                    and e["payload"]["errors"] == ["repeated_call"]
                ],
                ["history-action-3", "history-action-4"],
            )
            self.assertLess(max(len(r.content) for r in requests), 200000)
            for event in histories:
                self.assertLess(
                    len(canonical_json(event["payload"]["result"]).encode()),
                    65536,
                )
                self.assertIn(
                    event["payload"]["result"]["representation"],
                    {"retained_evidence_v2", "retained_payload_slice_v1"},
                )
            if retrieve_payload:
                original = next(
                    e for e in events if e["event_id"] == payload_binding["event_id"]
                )
                self.assertEqual(
                    "".join(payload_parts), canonical_json(original["payload"])
                )
                self._assert_history_transport_artifacts(histories, events, journey)
            changed = Journey.open(journey.path).read()
            changed_history = next(
                e for e in reversed(changed) if e["event_type"] == "JourneySearchRead"
            )
            if retrieve_payload:
                changed_history["payload"]["result"]["payload_slice"]["text"] = (
                    "forged payload"
                )
            else:
                changed_history["payload"]["result"]["events"][0]["event_id"] = (
                    "different-history-event"
                )
            journey.path.write_text(
                "".join(canonical_json(e) + "\n" for e in changed),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "search receipt changed its exact Run prefix"
            ):
                observe(ObserveRun(journey_file=receipt.journey_file))

    def _assert_history_transport_artifacts(self, histories, events, journey):
        linked_transports = [
            entry
            for receipt in histories
            for entry in receipt["payload"]["result"].get("events", [])
            if entry["event_type"] == "ModelTransportRequestRecorded"
        ]
        self.assertTrue(linked_transports)
        reopened = Journey.open(journey.path)
        for entry in linked_transports:
            retained = next(
                event for event in events if event["event_id"] == entry["event_id"]
            )
            self.assertEqual(
                entry["artifact_references"],
                retained["artifact_references"],
            )
            self.assertTrue(entry["artifact_references"])
            for reference in entry["artifact_references"]:
                self.assertEqual(
                    len(reopened.read_artifact(reference)),
                    reference["byte_count"],
                )

    def _plan_through_second_request(
        self,
        directory,
        *,
        overflow=True,
        truncated=False,
        rejected=None,
        agent_invocation=None,
    ):
        """Run real planning until its second request ends the first Attempt.

        With ``overflow`` the first response carries a large text part beside one
        history read, so the next request's serialized body, at two bytes per
        estimated token, exceeds the configured 262,144-token window; otherwise the
        endpoint rejects the second request.
        With ``truncated`` the first response is a refused action and the next
        ``truncated`` responses are cut off on their output ceiling. Only the
        transport is controlled; the Adapter budget is real. The endpoint rejects
        request number ``rejected``.
        """
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_composition,
        )
        from tests.application_fakes import _registered
        from tests.planning_assignment_fakes import PlanningComposition

        receipt, paths = _registered(Path(directory), agent_invocation=agent_invocation)
        registered = review_journey(
            ReviewJourneyQuery(journey_file=receipt.journey_file)
        )
        requests, clients = [], []
        self.addCleanup(lambda: asyncio.run(close_clients(clients)))

        def respond(request):
            requests.append(request)
            body = json.loads(request.content)
            number = len(requests)
            if number in (rejected if isinstance(rejected, tuple) else (rejected,)):
                return httpx.Response(400, json={"error": {"message": "rejected"}})
            if truncated and number == 1:
                tool_name, arguments = "record_proposal_requirements", {}
            elif number == 1:
                tool_name, arguments = "search_journey", {"limit": 100}
            elif truncated and number <= 1 + truncated:
                return truncated_completion(
                    body,
                    response_id=f"cut-off-{number}",
                    output_tokens=32768,
                    content=f"looping partial reply {number}",
                )
            elif not overflow and not truncated:
                return httpx.Response(400, json={"error": {"message": "rejected"}})
            elif number == 2 + truncated:
                tool_name = "record_proposal_requirements"
                arguments = grounded_requirements(
                    receipt.journey_file, registered, paths
                )
            else:
                tool_name = "request_planning_hitl"
                arguments = bound_arguments(offered_tool(body, tool_name))
            return tool_call_completion(
                body,
                response_id=f"context-{number}",
                call_id=f"context-action-{number}",
                tool_name=tool_name,
                arguments=arguments,
                content="large local context " * 32000
                if overflow and not truncated and number == 1
                else None,
            )

        class ContextPlanning(PlanningComposition):
            def planning_assignment(self, owner):
                del owner
                return planning_assignment(
                    model_factory=transport_factory(respond, clients),
                    reference_adapter=self.reference_adapter,
                    companion_adapter=self.companion_adapter,
                    galaxy_adapter=self.galaxy_adapter,
                )

        composition = ContextPlanning(receipt, paths)
        with use_run_advancement_composition(composition):
            stopped = advance(ResumeRun(journey_file=receipt.journey_file))
        return receipt.journey_file, composition, requests, stopped

    def test_local_context_exhaustion_stops_for_review_and_continues_fresh(self):
        from bio_neb.agentic_journey.agent_work_review import (
            ContinueAgentWorkResponse,
            project_agent_work_review,
        )
        from bio_neb.application.advancement import (
            AgentWorkReviewAnswer,
            AnswerReview,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.recording import read_evidence

        def semantic(journey_file):
            return [
                e
                for e in read_evidence(Journey.open(Path(journey_file)))
                if e["event_type"] not in {"PassStarted", "PassClosed"}
            ]

        with tempfile.TemporaryDirectory() as directory:
            journey_file, composition, requests, stopped = (
                self._plan_through_second_request(directory)
            )
            # The overflowing request never reached the transport.
            self.assertEqual(
                (stopped.status, len(requests)), ("agent_work_review_pending", 1)
            )
            self.assertEqual(observe(ObserveRun(journey_file=journey_file)), stopped)
            events = read_evidence(Journey.open(Path(journey_file)))
            attempt_id = stopped.attempts[-1].attempt_id
            refused = [
                e
                for e in events
                if e["payload"].get("attempt_id") == attempt_id
                and e["payload"].get("request_number") == 2
            ]
            self.assertEqual(
                [e["event_type"] for e in refused],
                [
                    "ModelRequestRecorded",
                    "ModelPhysicalContactStarted",
                    "ModelPhysicalContactFailed",
                    "ModelRequestFailed",
                ],
            )
            contact, failure = refused[2]["payload"], refused[3]
            self.assertEqual(
                (
                    contact["failure_class"],
                    contact["completion_certainty"],
                    contact["failure_reason"],
                    contact["policy_disposition"],
                ),
                (
                    "exhausted",
                    "not_sent",
                    "context_window_exhausted",
                    "terminal_failure",
                ),
            )
            self.assertEqual(
                (
                    failure["payload"]["error_type"],
                    failure["payload"]["failure_reason"],
                ),
                ("ModelContextExhausted", "context_window_exhausted"),
            )
            # Accounting keeps the refused request a logical request, not a send.
            postmortem = stopped.model_contact_postmortem
            self.assertEqual(
                (postmortem.logical_request_count, postmortem.physical_contact_count),
                (2, len(requests)),
            )
            self.assertFalse(
                any(e["event_type"] in {"AttemptFailed", "RunFailed"} for e in events)
            )
            self.assertEqual(stopped.current_review.kind, "agent_work")
            request = stopped.current_review.request
            self.assertEqual(
                (request.cause, request.source_event_id, request.faults),
                ("context_exhausted", failure["event_id"], ()),
            )
            self.assertIn("262144-token context window", request.reason)
            self.assertEqual(
                [item.action for item in request.legal_actions], ["continue", "reject"]
            )
            (attempt_stopped,) = [
                e for e in events if e["event_type"] == "AttemptStopped"
            ]
            self.assertEqual(
                attempt_stopped["payload"],
                {
                    "attempt_id": attempt_id,
                    "code": "context_exhausted",
                    "source_event_id": request.request_event_id,
                },
            )
            # The request must name its own locally refused request.
            response_id = next(
                e for e in events if e["event_type"] == "ModelResponseRecorded"
            )["event_id"]
            forgeries = {
                "another source": ("AgentWorkReviewRequested", "source_event_id"),
                "another failure": ("ModelRequestFailed", "failure_reason"),
            }
            for label, (event_type, field) in forgeries.items():
                changed = copy.deepcopy(events)
                (forged,) = [e for e in changed if e["event_type"] == event_type]
                forged["payload"][field] = (
                    response_id if field == "source_event_id" else "non_http_failure"
                )
                with (
                    self.subTest(forged=label),
                    self.assertRaisesRegex(ValueError, "exact refused request"),
                ):
                    project_agent_work_review(changed)

            def answer(response, request_event_id=request.request_event_id):
                return AnswerReview(
                    journey_file=journey_file,
                    request_event_id=request_event_id,
                    answer=AgentWorkReviewAnswer(kind="agent_work", response=response),
                )

            continued = answer(ContinueAgentWorkResponse(action="continue"))
            with use_run_advancement_composition(composition):
                resumed = advance(continued)
                self.assertEqual(resumed.status, "planning_hitl_pending")
                after, sent = semantic(journey_file), len(requests)
                self.assertEqual(advance(continued).status, "planning_hitl_pending")
                # Conflicting, stale and wrong-kind answers are pinned once, in
                # test_planning_assignment's five-correction stop and the narrow
                # test_agent_work_review answer test.
            self.assertEqual((semantic(journey_file), len(requests)), (after, sent))
            # A new Attempt begins from the assignment context, not the transcript.
            first, second = resumed.attempts[-2:]
            self.assertEqual((first.attempt_id, first.outcome), (attempt_id, "stopped"))
            started = next(
                e
                for e in after
                if e["event_type"] == "AttemptStarted"
                and e["payload"]["attempt_id"] == second.attempt_id
            )
            response_event = next(
                e for e in after if e["event_type"] == "AgentWorkReviewResponded"
            )
            self.assertEqual(
                started["payload"]["continuation_work_review_event_id"],
                response_event["event_id"],
            )
            first_request = next(
                e
                for e in after
                if e["event_type"] == "ModelRequestRecorded"
                and e["payload"]["attempt_id"] == second.attempt_id
            )
            self.assertEqual(first_request["payload"]["request_number"], 1)
            self.assertNotIn("large local context", requests[1].content.decode())
            self.assertFalse(
                any(e["event_type"] in {"AttemptFailed", "RunFailed"} for e in after)
            )

    def test_rejected_request_failure_reopens_only_on_its_causal_chain(self):
        from bio_neb.recording import read_evidence

        with tempfile.TemporaryDirectory() as directory:
            journey_file, _, requests, failed = self._plan_through_second_request(
                directory, overflow=False
            )
            # The first refusal is shown to the model in the provider's words;
            # the same refusal of the request that carries them waits for the
            # Review with that body.
            self.assertEqual(
                (failed.status, len(requests)), ("model_availability_review_pending", 3)
            )
            self.assertEqual(observe(ObserveRun(journey_file=journey_file)), failed)
            events = read_evidence(Journey.open(Path(journey_file)))
            statement = '{"error":{"message":"rejected"}}'
            (rejected,) = [
                e for e in events if e["event_type"] == "ModelRequestRejected"
            ]
            self.assertEqual(
                (
                    rejected["payload"]["http_status_code"],
                    rejected["payload"]["provider_statement"],
                ),
                (400, statement),
            )
            self.assertTrue(
                any(
                    statement in str(message.get("content"))
                    for message in json.loads(requests[2].content)["messages"]
                )
            )
            (failure,) = [e for e in events if e["event_type"] == "ModelRequestFailed"]
            (terminal,) = [e for e in events if e["event_type"] == "AttemptFailed"]
            (review,) = [
                e
                for e in events
                if e["event_type"] == "ModelAvailabilityReviewRequested"
            ]
            self.assertEqual(
                (
                    failure["payload"]["failure_reason"],
                    failure["payload"]["provider_statement"],
                    failure["payload"]["fault_key"],
                    terminal["payload"]["source_event_id"],
                    review["payload"]["provider_statement"],
                ),
                (
                    "request_rejected",
                    statement,
                    rejected["payload"]["fault_key"],
                    failure["event_id"],
                    statement,
                ),
            )
            # Every model endpoint error points to its retained body (G4.2).
            response = next(
                e
                for e in events
                if e["event_id"] == failure["payload"]["response_event_id"]
            )
            self.assertEqual(response["event_type"], "ModelTransportResponseRecorded")
            journey_path = Path(journey_file)
            changed = Journey.open(journey_path).read()
            (forged,) = [e for e in changed if e["event_type"] == "AttemptFailed"]
            forged["payload"]["source_event_id"] = next(
                e for e in changed if e["event_type"] == "ModelResponseRecorded"
            )["event_id"]
            journey_path.write_text(
                "".join(canonical_json(e) + "\n" for e in changed), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "availability request changed its logical request"
            ):
                observe(ObserveRun(journey_file=journey_file))

    def test_planning_truncation_is_retried_once_without_the_cut_off_reply(self):
        from bio_neb.recording import read_evidence

        with tempfile.TemporaryDirectory() as directory:
            journey_file, _composition, requests, result = (
                self._plan_through_second_request(directory, truncated=1)
            )
            self.assertEqual(
                (result.status, len(requests)), ("planning_hitl_pending", 4)
            )
            cut_off = json.loads(requests[1].content)
            offered = sorted(tool["function"]["name"] for tool in cut_off["tools"])
            # C7's truncation correction, built from the cut-off request's offer.
            correction = (
                "Your reply was cut at 32768 tokens before any tool call.\n"
                "Allowed form: a short reply that calls one of "
                + ", ".join(f"`{name}`" for name in offered)
                # The refused call was failed turn 1; the cut-off is the second.
                + ".\nCount: failed turn 2 of 4 in a row. At 4 the Attempt stops and "
                "the researcher reviews it."
            )
            retry = json.loads(requests[2].content)
            sent = json.dumps(retry["messages"])
            self.assertNotIn("looping partial reply", sent)
            # The cut-off reply is kept with no call and no text, so it is sent as
            # empty content before its correction, never its text.
            self.assertEqual(retry["messages"][:-2], cut_off["messages"])
            self.assertEqual(
                retry["messages"][-2], {"role": "assistant", "content": ""}
            )
            self.assertEqual(retry["messages"][-1]["role"], "user")
            self.assertIn(correction, retry["messages"][-1]["content"])
            events = read_evidence(Journey.open(Path(journey_file)))
            (truncation,) = [
                e for e in events if e["event_type"] == "ModelCompletionTruncated"
            ]
            self.assertEqual(
                (
                    truncation["payload"]["request_number"],
                    truncation["payload"]["correction_count"],
                    truncation["payload"]["feedback"],
                ),
                (2, 2, correction),
            )
            self.assertFalse(
                any(
                    e["event_type"]
                    in {"AgentWorkReviewRequested", "AttemptFailed", "RunFailed"}
                    for e in events
                )
            )
            self.assertEqual(observe(ObserveRun(journey_file=journey_file)), result)

    def test_reopen_refuses_failure_body_substitution(self):
        """ADR 0032/0033 (#477 G4.2; Standards r1 ST1): an HTTP failure's pointer,
        key and words are its own retained answer's."""
        import copy
        import uuid

        from bio_neb.application.queries import project_journey_events
        from bio_neb.recording import read_evidence

        with tempfile.TemporaryDirectory() as directory:
            path, _, _, _ = self._plan_through_second_request(directory, overflow=False)
            events = read_evidence(Journey.open(Path(path)))
            project_journey_events(events, journey_file=path)
            for mutation in ("missing response", "invented statement", "invented key"):
                hostile = copy.deepcopy(events)
                bogus_id = str(uuid.uuid4())
                for event in hostile:
                    payload = event["payload"]
                    if mutation == "missing response" and event["event_type"] in {
                        "ModelPhysicalContactFailed",
                        "ModelRequestFailed",
                        "ModelRequestRejected",
                    }:
                        payload["response_event_id"] = bogus_id
                    if mutation == "invented statement" and event["event_type"] in {
                        "ModelRequestFailed",
                        "ModelAvailabilityReviewRequested",
                    }:
                        payload["provider_statement"] = (
                            "These words were never sent by the provider."
                        )
                    if mutation == "invented key":
                        for field in ("fault_key", "after_rejection_fault_key"):
                            if field in payload:
                                payload[field] = (
                                    "a cause unrelated to the response bytes"
                                )
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    project_journey_events(hostile, journey_file=path)

    def test_retried_truncation_is_not_failure_evidence_of_a_later_failure(self):
        from bio_neb.recording import read_evidence

        with tempfile.TemporaryDirectory() as directory:
            # The refusal is shown once; the same refusal again asks.
            journey_file, _, requests, failed = self._plan_through_second_request(
                directory, truncated=1, rejected=(3, 4)
            )
            self.assertEqual(
                (failed.status, len(requests)), ("model_availability_review_pending", 4)
            )
            events = read_evidence(Journey.open(Path(journey_file)))
            (truncation,) = [
                e for e in events if e["event_type"] == "ModelCompletionTruncated"
            ]
            (failure,) = [e for e in events if e["event_type"] == "ModelRequestFailed"]
            (terminal,) = [e for e in events if e["event_type"] == "AttemptFailed"]
            self.assertIsNotNone(truncation["payload"]["feedback"])
            self.assertEqual(
                terminal["payload"]["source_event_id"], failure["event_id"]
            )
            # The corrected truncation stays in the Attempt; the failure still reads.
            self.assertEqual(observe(ObserveRun(journey_file=journey_file)), failed)

    def test_planning_second_truncation_stops_for_review_and_continues(self):
        from bio_neb.agentic_journey.agent_work_review import (
            ContinueAgentWorkResponse,
            project_agent_work_review,
        )
        from bio_neb.application.advancement import (
            AgentWorkReviewAnswer,
            AnswerReview,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.recording import read_evidence

        with tempfile.TemporaryDirectory() as directory:
            # The refused call and two cut-off replies are three failed turns in
            # a row; at a limit of three the third stops.
            journey_file, composition, requests, stopped = (
                self._plan_through_second_request(
                    directory, truncated=2, agent_invocation={"failed_turns": 3}
                )
            )
            self.assertEqual(
                (stopped.status, len(requests)), ("agent_work_review_pending", 3)
            )
            self.assertEqual(observe(ObserveRun(journey_file=journey_file)), stopped)
            events = read_evidence(Journey.open(Path(journey_file)))
            attempt_id = stopped.attempts[-1].attempt_id
            (refused,) = [
                e
                for e in events
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["status"] == "invalid"
            ]
            retried, truncation = [
                e for e in events if e["event_type"] == "ModelCompletionTruncated"
            ]
            owned = [e for e in events if e["payload"].get("attempt_id") == attempt_id]
            self.assertEqual(
                [e["event_type"] for e in owned[-4:]],
                [
                    "ModelCompletionTruncated",
                    "AgentTurnRouted",
                    "AgentWorkReviewRequested",
                    "AttemptStopped",
                ],
            )
            self.assertFalse(
                any(e["event_type"] in {"AttemptFailed", "RunFailed"} for e in events)
            )
            self.assertEqual(stopped.current_review.kind, "agent_work")
            request = stopped.current_review.request
            self.assertEqual(
                (request.cause, request.source_event_id),
                ("failed_turns", truncation["event_id"]),
            )
            # The ordered fault history: the corrected refusal, the retried
            # truncation with its correction, then the stopping truncation.
            self.assertEqual(
                [
                    (
                        item.event_id,
                        item.correction_key.split(" @ ")[0],
                        item.correction_count,
                    )
                    for item in request.faults
                ],
                [
                    (
                        refused["event_id"],
                        "record_proposal_requirements: "
                        + ", ".join(refused["payload"]["errors"]),
                        1,
                    ),
                    (retried["event_id"], "output_truncated", 2),
                    (truncation["event_id"], "output_truncated", 3),
                ],
            )
            # Each fault keeps the correction written for it; the stopping one is
            # recorded, never sent.
            self.assertEqual(
                [item.feedback is not None for item in request.faults],
                [True, True, True],
            )
            # The Review shows the cut-off request, its ceiling and what it spent.
            self.assertEqual(
                (
                    request.truncation.event_id,
                    request.truncation.request_number,
                    request.truncation.requested_max_tokens,
                    request.truncation.completion_tokens,
                ),
                (
                    truncation["event_id"],
                    3,
                    truncation["payload"]["requested_max_tokens"],
                    32768,
                ),
            )
            self.assertIsNotNone(request.truncation.requested_max_tokens)
            self.assertEqual(
                [item.action for item in request.legal_actions], ["continue", "reject"]
            )
            (attempt_stopped,) = [
                e for e in events if e["event_type"] == "AttemptStopped"
            ]
            self.assertEqual(
                attempt_stopped["payload"],
                {
                    "attempt_id": attempt_id,
                    "code": "failed_turns",
                    "source_event_id": request.request_event_id,
                },
            )
            # The request must name its Attempt's own exact truncation.
            response_id = next(
                e
                for e in events
                if e["event_type"] == "ModelResponseRecorded"
                and e["payload"]["request_number"] == 3
            )["event_id"]
            forgeries = {
                "another source": (
                    "AgentWorkReviewRequested",
                    "source_event_id",
                    response_id,
                    "exact stopping reply",
                ),
                "another budget": (
                    "ModelCompletionTruncated",
                    "completion_tokens",
                    1,
                    "model truncation",
                ),
            }
            for label, (event_type, field, value, message) in forgeries.items():
                changed = copy.deepcopy(events)
                forged = [e for e in changed if e["event_type"] == event_type][-1]
                forged["payload"][field] = value
                with (
                    self.subTest(forged=label),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    project_agent_work_review(changed)

            continued = AnswerReview(
                journey_file=journey_file,
                request_event_id=request.request_event_id,
                answer=AgentWorkReviewAnswer(
                    kind="agent_work",
                    response=ContinueAgentWorkResponse(
                        action="continue", instruction="Keep reasoning short."
                    ),
                ),
            )
            with use_run_advancement_composition(composition):
                resumed = advance(continued)
            self.assertEqual(resumed.status, "planning_hitl_pending")
            first, second = resumed.attempts[-2:]
            self.assertEqual((first.attempt_id, first.outcome), (attempt_id, "stopped"))
            after = read_evidence(Journey.open(Path(journey_file)))
            started = next(
                e
                for e in after
                if e["event_type"] == "AttemptStarted"
                and e["payload"]["attempt_id"] == second.attempt_id
            )
            response_event = next(
                e for e in after if e["event_type"] == "AgentWorkReviewResponded"
            )
            self.assertEqual(
                started["payload"]["continuation_work_review_event_id"],
                response_event["event_id"],
            )
            self.assertFalse(
                any(e["event_type"] in {"AttemptFailed", "RunFailed"} for e in after)
            )

    def test_separate_causal_investigation_retains_transport_and_reopens_failure(self):
        for truncated in (False, True):
            with self.subTest(truncated=truncated):
                self._separate_investigation_case(truncated=truncated)

    def _separate_investigation_case(self, *, truncated):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {
                "schema_version": "bio-neb.investigation-request/2",
                "asserted_by": "researcher:owner",
                "request": "Investigate rs429358 in sample HG001.",
                "target": {"rsid": "rs429358"},
            }
            child = forked_child(root, investigation_request=request)
            request_file = root / "investigation-request.json"
            request_file.write_text(json.dumps(request), encoding="utf-8")
            requests, clients = [], []
            sources = SourceResponsesOnly()

            def respond(request):
                requests.append(request)
                if truncated:
                    return httpx.Response(
                        200,
                        json={
                            "id": "truncated-investigation",
                            "object": "chat.completion",
                            "created": 0,
                            "model": json.loads(request.content)["model"],
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "partial",
                                    },
                                    "finish_reason": "length",
                                }
                            ],
                        },
                    )
                return httpx.Response(
                    503, json={"error": {"message": "offline failure"}}
                )

            try:
                result = begin_in_pass(
                    BeginInvestigationCommand(
                        journey_file=str(child.path),
                        request_files=(str(request_file),),
                    ),
                    environ={"LOCAL_MODEL_KEY": "sentinel-secret"},
                    model_factory=transport_factory(respond, clients),
                    source_adapter_factory=lambda: sources,
                )
            finally:
                asyncio.run(close_clients(clients))
            reopened = Journey.open(child.path)
            events = reopened.read()
            bound = next(
                event
                for event in events
                if event["event_type"] == "InvestigationAttemptBound"
            )
            # The forked child also holds its parent's planning transport records.
            bodies = [
                event
                for event in events
                if event["event_type"] == "ModelTransportRequestRecorded"
                and event["payload"]["attempt_id"] == bound["payload"]["attempt_id"]
            ]
            # A failed send is sent once more; a cut-off reply is a failed turn,
            # and the fourth in a row stops.
            sent = 4 if truncated else 2
            self.assertEqual(len(bodies), sent)
            self.assertEqual(len(requests), sent)
            self.assertEqual(sources.contact_count, 0)
            self.assertEqual(
                reopened.read_artifact(bodies[0]["artifact_references"][0]),
                requests[0].content,
            )
            self.assertEqual(read_investigation(reopened).investigation, result)
            if truncated:
                truncation = [
                    event
                    for event in events
                    if event["event_type"] == "ModelCompletionTruncated"
                ][-1]
                requested = next(
                    event
                    for event in events
                    if event["event_type"] == "AgentWorkReviewRequested"
                )
                self.assertEqual(
                    (
                        result.status,
                        requested["payload"]["cause"],
                        requested["payload"]["source_event_id"],
                    ),
                    (
                        "agent_work_review_pending",
                        "failed_turns",
                        truncation["event_id"],
                    ),
                )
                self.assertFalse(
                    any(
                        event["event_type"] == "InvestigationAttemptFailed"
                        for event in events
                    )
                )
                # Merged evidence: the inherited planning transport records
                # are read with their contacts.
                changed = copy.deepcopy(read_evidence(reopened))
                [
                    event
                    for event in changed
                    if event["event_type"] == "ModelCompletionTruncated"
                ][-1]["payload"]["completion_tokens"] = 1
                with self.assertRaisesRegex(ValueError, "model truncation"):
                    project_investigation(changed)
            before = child.path.read_bytes()
            for view in (
                observe(ObserveRun(journey_file=str(child.path)), journey=reopened),
                review_journey(
                    ReviewJourneyQuery(journey_file=str(child.path)), journey=reopened
                ),
            ):
                self.assertIsNotNone(view.investigation)
            self.assertEqual(child.path.read_bytes(), before)

    def test_public_attempt_retains_exact_transport_body_and_reopens(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = registered_run(Path(directory))
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            journey = Journey.open(Path(receipt.journey_file))
            requests = []
            clients = []

            def respond(request):
                requests.append(request)
                return httpx.Response(
                    200,
                    json={
                        "id": "transport-response",
                        "object": "chat.completion",
                        "created": 0,
                        "model": json.loads(request.content)["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": "metadata-call",
                                            "type": "function",
                                            "function": {
                                                "name": "inspect_input_metadata",
                                                "arguments": json.dumps(
                                                    {"input_set_id": input_set_id}
                                                ),
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    },
                )

            try:
                result = planning_assignment(
                    model_factory=transport_factory(respond, clients)
                ).run(
                    prepare_input_correction(journey),
                    environ={"LOCAL_MODEL_KEY": "sentinel-secret"},
                )
            finally:
                asyncio.run(close_clients(clients))

            self.assertEqual(result.status, "inspection_succeeded")
            self.assertEqual(len(requests), 1)
            retained = journey.read()
            transport = [
                event
                for event in retained
                if event["event_type"] == "ModelTransportRequestRecorded"
            ]
            self.assertEqual(len(transport), 1)
            event = transport[0]
            body = requests[0].content
            self.assertEqual(
                journey.read_artifact(event["artifact_references"][0]), body
            )
            self.assertEqual(event["payload"]["request_body_byte_count"], len(body))
            self.assertEqual(
                event["payload"]["request_body_sha256"],
                hashlib.sha256(body).hexdigest(),
            )
            self.assertEqual(event["payload"]["requested_max_tokens"], 245760)
            payload = event["payload"]
            # Two body bytes per estimated token, rounded up.
            expected = min(
                payload["requested_max_tokens"],
                payload["context_budget"]["context_window"]
                - -(-payload["unbudgeted_body_byte_count"] // 2)
                - payload["context_budget"]["serialized_body_safety_margin"],
            )
            self.assertEqual(payload["final_max_completion_tokens"], expected)
            self.assertEqual(json.loads(body)["max_completion_tokens"], expected)
            self.assertNotIn(b"sentinel-secret", body)
            self.assertNotIn(b"authorization", body.lower())
            self.assertIn("sentinel-secret", requests[0].headers["authorization"])

            before = Path(receipt.journey_file).read_bytes()
            reopened = Journey.open(Path(receipt.journey_file))
            for view in (
                observe(
                    ObserveRun(journey_file=receipt.journey_file), journey=reopened
                ),
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file),
                    journey=reopened,
                ),
            ):
                self.assertIn(
                    event["event_id"],
                    [item.source_event_id for item in view.attempt.model_events],
                )
            self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
            self.assertEqual(len(requests), 1)

            for field, value in (
                ("request_number", 2),
                ("final_max_completion_tokens", 1),
                ("request_body_sha256", "0" * 64),
            ):
                changed = copy.deepcopy(retained)
                next(
                    item
                    for item in changed
                    if item["event_type"] == "ModelTransportRequestRecorded"
                )["payload"][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    project_journey_events(changed, journey_file=receipt.journey_file)
            missing = [
                item
                for item in retained
                if item["event_type"] != "ModelTransportRequestRecorded"
            ]
            with self.assertRaises(ValueError):
                project_journey_events(missing, journey_file=receipt.journey_file)

            # A coherent detached envelope cannot establish what the admitted bytes say.
            forged = copy.deepcopy(retained)
            forged_transport = next(
                e for e in forged if e["event_type"] == "ModelTransportRequestRecorded"
            )
            model = json.loads(body)["model"]
            forged_body = body.replace(model.encode(), b"x" * len(model.encode()))
            digest = hashlib.sha256(forged_body).hexdigest()
            forged_transport["payload"]["request_body_sha256"] = digest
            forged_transport["payload"]["unbudgeted_body_sha256"] = digest
            artifact = forged_transport["artifact_references"][0]
            artifact["artifact_id"] = "sha256:" + digest
            journey.artifact_path(artifact).write_bytes(forged_body)
            journey.path.write_text(
                "".join(canonical_json(e) + "\n" for e in forged), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "open physical contact"):
                read_planning_events(forged)
            self.assertIsNone(read_planning(journey).plan)
            self.assertEqual(len(read_pre_execution(journey).history.attempts), 1)
            self.assertFalse(journey.is_current())
            forged_owner = Journey.open(journey.path)
            with self.assertRaisesRegex(ValueError, "body changed its model"):
                read_planning(forged_owner)
            with self.assertRaisesRegex(ValueError, "body changed its model"):
                read_pre_execution(forged_owner)
            self.assertEqual(len(requests), 1)

    def test_every_contact_keeps_its_exact_response_body_beside_its_request(self):
        # Each physical contact that receives an HTTP response keeps the
        # provider's raw body, content-addressed and bound to that contact; an
        # error body too, so a lost call can be traced to its layer.
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = registered_run(Path(directory))
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            journey = Journey.open(Path(receipt.journey_file))
            sent, clients = [], []

            def respond(request):
                if not sent:
                    sent.append(b"upstream busy")
                    return httpx.Response(
                        503,
                        content=sent[-1],
                        headers={"content-type": "text/plain; charset=utf-8"},
                    )
                completion = tool_call_completion(
                    json.loads(request.content),
                    response_id="after-busy",
                    call_id="metadata-call",
                    tool_name="inspect_input_metadata",
                    arguments={"input_set_id": input_set_id},
                ).json()
                # Bytes no SDK re-serialization would reproduce.
                sent.append(json.dumps(completion, indent=3).encode() + b"\n")
                return httpx.Response(
                    200,
                    content=sent[-1],
                    headers={"content-type": "application/json"},
                )

            try:
                result = planning_assignment(
                    model_factory=transport_factory(respond, clients)
                ).run(
                    prepare_input_correction(journey),
                    environ={"LOCAL_MODEL_KEY": "offline"},
                )
            finally:
                asyncio.run(close_clients(clients))

            self.assertEqual(result.status, "inspection_succeeded")
            # Contacts live in the Contact journal; the bodies are Journey facts.
            evidence = read_evidence(journey)

            def of(kind, events=evidence):
                return [e for e in events if e["event_type"] == kind]

            contacts = of("ModelPhysicalContactStarted")
            bodies = of("ModelTransportResponseRecorded")
            self.assertEqual(len(contacts), 2)
            self.assertTrue(
                all(c["payload"]["response_body_expected"] for c in contacts)
            )
            self.assertEqual(
                [
                    (
                        b["payload"]["contact_started_event_id"],
                        b["payload"]["status_code"],
                        b["artifact_references"][0]["media_type"],
                        journey.read_artifact(b["artifact_references"][0]),
                    )
                    for b in bodies
                ],
                [
                    (contacts[0]["event_id"], 503, "text/plain", sent[0]),
                    (contacts[1]["event_id"], 200, "application/json", sent[1]),
                ],
            )
            for body, raw in zip(bodies, sent, strict=True):
                self.assertEqual(
                    (
                        body["payload"]["response_body_byte_count"],
                        body["payload"]["response_body_sha256"],
                    ),
                    (len(raw), hashlib.sha256(raw).hexdigest()),
                )
            # Each body follows its request body and comes before its outcome.
            (failed,) = of("ModelPhysicalContactFailed")
            (response,) = of("ModelResponseRecorded")
            requests = of("ModelTransportRequestRecorded")
            for request, body, outcome in zip(
                requests, bodies, (failed, response), strict=True
            ):
                self.assertLess(evidence.index(request), evidence.index(body))
                self.assertLess(evidence.index(body), evidence.index(outcome))
            self.assertEqual(
                review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file)),
                result,
            )

            project_journey_events(evidence, journey_file=receipt.journey_file)
            for field, value in (
                ("contact_ordinal", 2),
                ("status_code", 500),
                ("response_body_sha256", "0" * 64),
            ):
                changed = copy.deepcopy(evidence)
                of("ModelTransportResponseRecorded", changed)[0]["payload"][field] = (
                    value
                )
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(ValueError, "transport.* response"),
                ):
                    project_journey_events(changed, journey_file=receipt.journey_file)
            # A received response without its body is refused: the HTTP failure's
            # and the admitted response's alike. The rest of the graph is kept
            # whole (each child takes the removed body's parents).
            for body in bodies:
                missing = [
                    {
                        **e,
                        "parent_ids": [
                            parent
                            for item in e["parent_ids"]
                            for parent in (
                                body["parent_ids"]
                                if item == body["event_id"]
                                else [item]
                            )
                        ],
                    }
                    for e in evidence
                    if e["event_id"] != body["event_id"]
                ]
                with (
                    self.subTest(status=body["payload"]["status_code"]),
                    self.assertRaisesRegex(
                        ValueError, "lost its expected raw response"
                    ),
                ):
                    project_journey_events(missing, journey_file=receipt.journey_file)
            # A changed body under the same envelope is refused when it is read.
            journey.artifact_path(bodies[1]["artifact_references"][0]).write_bytes(
                sent[1].replace(b"after-busy", b"after-bus!")
            )
            with self.assertRaisesRegex(ValueError, "artifact"):
                read_planning(Journey.open(journey.path))

    def test_a_failure_to_retain_a_response_body_is_ours_and_is_not_resent(self):
        # The SDK wraps anything raised inside its send as a connection fault; a
        # local failure to record the body is our invariant.
        from bio_neb.agentic_journey import model_observation
        from bio_neb.failure import Invariant

        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = registered_run(Path(directory))
            journey = Journey.open(Path(receipt.journey_file))
            sent, clients = [], []

            def respond(request):
                sent.append(request)
                return tool_call_completion(
                    json.loads(request.content),
                    response_id="kept-locally-failed",
                    call_id="metadata-call",
                    tool_name="inspect_input_metadata",
                    arguments={},
                )

            try:
                with (
                    patch.object(
                        model_observation.ModelTransportResponseRecorded,
                        "from_received",
                        side_effect=ValueError("response artifact identity changed"),
                    ),
                    self.assertRaises(Invariant),
                ):
                    planning_assignment(
                        model_factory=transport_factory(respond, clients)
                    ).run(
                        prepare_input_correction(journey),
                        environ={"LOCAL_MODEL_KEY": "offline"},
                    )
            finally:
                asyncio.run(close_clients(clients))

            self.assertEqual(len(sent), 1)
            (failed,) = [
                e["payload"]
                for e in read_evidence(journey)
                if e["event_type"] == "ModelPhysicalContactFailed"
            ]
            self.assertEqual(
                (
                    failed["error_type"],
                    failed["failure_class"],
                    failed["policy_disposition"],
                ),
                ("ValueError", "invariant", "terminal_failure"),
            )

    def test_malformed_provider_body_is_resent_once_then_reviewed(self):
        from bio_neb.recording import read_evidence

        for body in ({"choices": []}, {"choices": "invalid"}, "not a chat completion"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                receipt, _ = registered_run(Path(directory))
                journey = Journey.open(Path(receipt.journey_file))
                requests, clients, replied = [], [], []

                def respond(request, requests=requests, body=body, replied=replied):
                    requests.append(request.content)
                    replied.append(httpx.Response(200, json=body))
                    return replied[-1]

                try:
                    result = planning_assignment(
                        model_factory=transport_factory(respond, clients)
                    ).run(
                        prepare_input_correction(journey),
                        environ={"LOCAL_MODEL_KEY": "offline"},
                    )
                finally:
                    asyncio.run(close_clients(clients))
                self.assertEqual(result.status, "model_availability_review_pending")
                self.assertEqual(
                    result.model_availability_review.pending_request.exhaustion_reason,
                    "malformed_response",
                )
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0], requests[1])
                events = read_evidence(journey)
                self.assertEqual(
                    sum(e["event_type"] == "ModelRequestRecorded" for e in events), 1
                )
                self.assertFalse(any(e["event_type"] == "RunFailed" for e in events))
                # Each malformed reply was received: its exact bytes are kept.
                self.assertEqual(
                    [
                        journey.read_artifact(e["artifact_references"][0])
                        for e in events
                        if e["event_type"] == "ModelTransportResponseRecorded"
                    ],
                    [response.content for response in replied],
                )
                self.assertEqual(
                    observe(ObserveRun(journey_file=receipt.journey_file)), result
                )

    def test_failed_send_keeps_prepared_body_and_physical_failure_source(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = registered_run(Path(directory))
            journey = Journey.open(Path(receipt.journey_file))
            requests, clients = [], []

            def respond(request):
                requests.append(request)
                raise httpx.ConnectError("transport unavailable", request=request)

            try:
                result = planning_assignment(
                    model_factory=transport_factory(respond, clients)
                ).run(
                    prepare_input_correction(journey),
                    environ={"LOCAL_MODEL_KEY": "sentinel-secret"},
                )
            finally:
                asyncio.run(close_clients(clients))

            self.assertEqual(result.status, "model_availability_review_pending")
            from bio_neb.recording import read_evidence

            events = read_evidence(journey)
            bodies = [
                event
                for event in events
                if event["event_type"] == "ModelTransportRequestRecorded"
            ]
            failed = [
                event
                for event in events
                if event["event_type"] == "ModelPhysicalContactFailed"
            ]
            self.assertEqual(len(requests), 2)
            self.assertEqual(len(bodies), 2)
            self.assertEqual(len(failed), 2)
            self.assertFalse(
                any(event["event_type"] == "ModelResponseRecorded" for event in events)
            )
            for prepared, failure, request in zip(
                bodies, failed, requests, strict=True
            ):
                self.assertEqual(
                    journey.read_artifact(prepared["artifact_references"][0]),
                    request.content,
                )
                self.assertEqual(
                    prepared["payload"]["contact_started_event_id"],
                    failure["payload"]["contact_started_event_id"],
                )
                self.assertEqual(
                    failure["payload"]["completion_certainty"], "completion_unknown"
                )
                self.assertLess(events.index(prepared), events.index(failure))
            logical_failure = next(
                event for event in events if event["event_type"] == "ModelRequestFailed"
            )
            self.assertEqual(
                logical_failure["payload"]["last_physical_contact_event_id"],
                failed[-1]["event_id"],
            )
            self.assertEqual(
                review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file)),
                result,
            )

    def test_truncated_response_stops_on_its_retained_truncation(self):
        from bio_neb.recording import read_evidence

        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = registered_run(Path(directory))
            journey = Journey.open(Path(receipt.journey_file))
            clients = []

            def respond(request):
                return httpx.Response(
                    200,
                    json={
                        "id": "truncated-response",
                        "object": "chat.completion",
                        "created": 0,
                        "model": json.loads(request.content)["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "partial"},
                                "finish_reason": "length",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 245760,
                            "total_tokens": 245860,
                        },
                    },
                )

            try:
                result = planning_assignment(
                    model_factory=transport_factory(respond, clients)
                ).run(
                    prepare_input_correction(journey),
                    environ={"LOCAL_MODEL_KEY": "sentinel-secret"},
                )
            finally:
                asyncio.run(close_clients(clients))

            # The endpoint cuts off every retry: the fourth cut-off in a row is the
            # limit of failed turns and stops. The contact journal is part of the
            # evidence the readers check.
            events = read_evidence(journey)
            prepared = [
                event
                for event in events
                if event["event_type"] == "ModelTransportRequestRecorded"
            ][-1]
            response = [
                event
                for event in events
                if event["event_type"] == "ModelResponseRecorded"
            ][-1]
            truncation = [
                event
                for event in events
                if event["event_type"] == "ModelCompletionTruncated"
            ][-1]
            requested = next(
                event
                for event in events
                if event["event_type"] == "AgentWorkReviewRequested"
            )
            self.assertEqual(
                response["payload"]["physical_contact_started_event_id"],
                prepared["payload"]["contact_started_event_id"],
            )
            self.assertEqual(
                truncation["payload"]["response_event_id"], response["event_id"]
            )
            self.assertEqual(
                (
                    result.status,
                    requested["payload"]["cause"],
                    requested["payload"]["source_event_id"],
                ),
                (
                    "agent_work_review_pending",
                    "failed_turns",
                    truncation["event_id"],
                ),
            )
            self.assertFalse(
                any(e["event_type"] in {"AttemptFailed", "RunFailed"} for e in events)
            )
            self.assertEqual(
                (
                    truncation["payload"]["completion_tokens"],
                    truncation["payload"]["correction_count"],
                    sum(e["event_type"] == "ModelCompletionTruncated" for e in events),
                ),
                (245760, 4, 4),
            )
            # A truncation moved off its response loses its exact response.
            for field, value, message in (
                (
                    "response_event_id",
                    prepared["event_id"],
                    "model truncation lost its exact request or response",
                ),
                ("completion_tokens", 1, "model truncation"),
                ("requested_max_tokens", 1, "model truncation"),
            ):
                changed = copy.deepcopy(events)
                [
                    item
                    for item in changed
                    if item["event_type"] == "ModelCompletionTruncated"
                ][-1]["payload"][field] = value
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    project_journey_events(changed, journey_file=receipt.journey_file)
            self.assertEqual(
                review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file)),
                result,
            )
