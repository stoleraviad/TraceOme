"""Effort is an explicit scoped stop; continuation preserves work and allowances."""

import copy
import json
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic_ai.exceptions import ModelHTTPError

from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
from bio_neb.agentic_journey.effort import (
    AgentEffortReviewResponseRefused,
    ExtendAgentEffortResponse,
    StopAgentEffortResponse,
    pending_effort_work,
    project_effort_review,
    respond_effort_review,
)
from bio_neb.agentic_journey.model_availability import RetryModelAvailabilityResponse
from bio_neb.application.advancement import (
    AgentEffortReviewAnswer,
    AnswerReview,
    ModelAvailabilityReviewAnswer,
    ResumeRun,
    advance,
    use_run_advancement_composition,
    use_run_advancement_dependencies,
)
from bio_neb.application.queries import (
    ObserveRun,
    observe,
)
from bio_neb.input_inspection.commands import (
    InspectRegisteredInputsCommand,
    inspect_registered_inputs,
)
from bio_neb.investigation.investigation import (
    InvestigationRequestItem,
    UserAssertion,
    project_investigation,
)
from bio_neb.investigation.investigation_commands import (
    BeginInvestigationCommand,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.recording import read_evidence
from tests.application_fakes import _registered
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.planning_assignment_fakes import PlanningComposition
from tests.sequential_investigation_support import (
    ASSEMBLY_BODY,
    SequentialAdaptiveModel,
    SequentialSourceResponses,
    begin_in_pass,
    forked_child,
    input_and_assertion,
    members_one_at_a_time,
    run_attempt,
)


def _semantic(events):
    """Journey facts without the processing-Pass journal markers."""
    return tuple(
        e for e in events if e["event_type"] not in {"PassStarted", "PassClosed"}
    )


def _inspected_planning_case(root):
    """Registered and inspected inputs with the scripted planning composition."""
    receipt, paths = _registered(root)
    journey = Journey.open(Path(receipt.journey_file))
    inspect_registered_inputs(
        InspectRegisteredInputsCommand(journey_file=str(journey.path)), journey=journey
    )
    return receipt, journey, PlanningComposition(receipt, paths)


class AgentEffortTests(unittest.TestCase):
    def test_findings_change_effort_stop_reopens_with_exact_pending_context(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey, composition = _inspected_planning_case(Path(directory))
            role = composition.planning_assignment(journey)
            stopped = role.run(
                journey,
                environ=composition.environ,
                invocation_overrides=AgentInvocationOverrides(model_requests=2),
            )
            self.assertEqual(
                composition.calls, ["search_journey", "record_proposal_requirements"]
            )
            self.assertEqual(stopped.status, "effort_review_pending")
            self.assertEqual(stopped.attempts[-1].outcome, "stopped")
            events = read_evidence(Journey.open(journey.path))
            self.assertEqual(
                sum(
                    e["event_type"] == "ProposalRequirementsIdentified" for e in events
                ),
                1,
            )
            self.assertFalse(any(e["event_type"] == "RunFailed" for e in events))
            self.assertEqual(
                observe(ObserveRun(journey_file=str(journey.path))).status,
                "effort_review_pending",
            )
            request = stopped.agent_effort_review.pending_request
            self.assertEqual(
                request.messages[-1]["instructions"], composition.instructions_seen[-1]
            )
            from bio_neb.agentic_journey.planning_assignment import planning_stage

            self.assertEqual(
                planning_stage(SimpleNamespace(journey=journey)), "reference"
            )
            before_count = len(composition.instructions_seen)
            with use_run_advancement_composition(composition):
                resumed = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=request.request_event_id,
                        answer=AgentEffortReviewAnswer(
                            kind="agent_effort",
                            response=ExtendAgentEffortResponse(
                                action="extend", additional_requests=8
                            ),
                        ),
                    )
                )
            self.assertEqual(
                composition.instructions_seen[before_count],
                request.messages[-1]["instructions"],
            )
            self.assertTrue(
                any(
                    text != request.messages[-1]["instructions"]
                    for text in composition.instructions_seen[before_count + 1 :]
                )
            )
            self.assertEqual(composition.calls.count("record_proposal_requirements"), 1)
            self.assertNotEqual(resumed.status, "failed")
            self.assertEqual(
                observe(ObserveRun(journey_file=str(journey.path))), resumed
            )

    def test_assignment_effort_retains_completed_tool_and_extends_only_request_allowance(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey, composition = _inspected_planning_case(Path(directory))
            role = composition.planning_assignment(journey)
            stopped = role.run(
                journey,
                environ={"LOCAL_MODEL_KEY": "offline"},
                invocation_overrides=AgentInvocationOverrides(model_requests=1),
            )
            self.assertEqual(composition.calls, ["search_journey"])
            self.assertEqual(stopped.status, "effort_review_pending")
            self.assertEqual(stopped.attempts[-1].outcome, "stopped")
            self.assertFalse(
                any(e["event_type"] == "RunFailed" for e in journey.read())
            )
            request = stopped.agent_effort_review.pending_request
            self.assertEqual(request.requests_used, 1)
            self.assertTrue(request.messages)
            retained = tuple(journey.read())
            retained_messages = "effort transcript changed retained model messages"
            for change, violation in (
                ("prompt", retained_messages),
                ("response", retained_messages),
                (
                    "pending_prompt",
                    "effort transcript changed its pending Tool continuation",
                ),
                (
                    # A tag alone is no injection: this Attempt switches none on.
                    "pending_injection",
                    "effort transcript changed its pending Tool continuation",
                ),
                ("missing_history", retained_messages),
            ):
                hostile = copy.deepcopy(read_evidence(journey))
                snapshot = next(
                    e["payload"]["messages"]
                    for e in hostile
                    if e["event_type"] == "AgentEffortReviewRequested"
                )
                if change == "prompt":
                    snapshot[0]["parts"][0]["content"] = (
                        "Invented original instructions"
                    )
                elif change == "response":
                    snapshot[-2]["parts"][0]["args"] = {"invented": True}
                elif change == "pending_prompt":
                    snapshot[-1]["parts"].append(
                        {"part_kind": "user-prompt", "content": "Invented new work"}
                    )
                elif change == "pending_injection":
                    snapshot[-1]["parts"].append(
                        {
                            "part_kind": "user-prompt",
                            "content": "[injection:unregistered]\nInvented new work",
                        }
                    )
                else:
                    snapshot.pop(0)
                with (
                    self.subTest(change=change),
                    self.assertRaisesRegex(ValueError, violation),
                ):
                    project_effort_review(hostile)
            with self.assertRaisesRegex(ValueError, "exact extension"):
                role.run(journey, environ={"LOCAL_MODEL_KEY": "offline"})
            self.assertEqual(_semantic(journey.read()), _semantic(retained))
            # Eleven further requests are exactly the scripted work up to the first
            # scoped question (the genomic FASTA's index search and choice among
            # them); a smaller allowance would stop for effort again.
            response = ExtendAgentEffortResponse(
                action="extend", additional_requests=11
            )
            accepted = respond_effort_review(
                journey, request.request_event_id, response
            )
            # A recorded extension reopens the Attempt only through its exact answer.
            unconsumed = journey.read()
            with self.assertRaisesRegex(ValueError, "exact extension"):
                role.run(journey, environ={"LOCAL_MODEL_KEY": "offline"})
            self.assertEqual(_semantic(journey.read()), _semantic(unconsumed))
            # Detached shared-owner proof: another role does not consume this answer.
            independent = copy.deepcopy(
                next(
                    e
                    for e in journey.read()
                    if e["event_type"] == "AttemptStarted"
                    and e["payload"]["agent_role"] == "planning_agent"
                )
            )
            independent["payload"]["agent_role"] = "investigation_planner"
            independent["event_id"] = str(uuid.uuid4())
            independent["payload"]["attempt_id"] = str(uuid.uuid4())
            project_effort_review([*read_evidence(journey), independent])
            # The shared review reader accepts another scoped semantic fact after
            # an answer. A detached copy of the Run, with that fact appended, supplies
            # the interleaving without inventing a second scientific caller in this
            # planning scenario or writing to the Run that resumes below.
            note = dict(independent, event_type="IndependentAssignmentNote", payload={})
            detached_root = Path(directory) / "detached" / journey.run_directory.name
            shutil.copytree(journey.run_directory, detached_root)
            detached_path = detached_root / journey.path.relative_to(
                journey.run_directory
            )
            with detached_path.open("a", encoding="utf-8") as retained_lines:
                retained_lines.write(canonical_json(note) + "\n")
            detached = Journey.open(detached_path)
            pending, _, _ = pending_effort_work(
                detached,
                accepted.response_event_id,
                agent_role="planning_agent",
            )
            self.assertEqual(pending.request_event_id, request.request_event_id)
            resumed = role.run(
                journey,
                environ={"LOCAL_MODEL_KEY": "offline"},
                effort_review_response_event_id=accepted.response_event_id,
            )
            self.assertEqual(resumed.status, "planning_hitl_pending")
            self.assertEqual(composition.calls.count("search_journey"), 1)
            self.assertEqual(len(composition.calls), 12)
            self.assertEqual(
                resumed.attempts[-1].effective_invocation_policy.model_requests, 12
            )
            self.assertEqual(
                resumed.attempts[-1].effective_invocation_policy.continuation,
                stopped.attempts[-1].effective_invocation_policy.continuation,
            )
            with self.assertRaisesRegex(ValueError, "unused"):
                pending_effort_work(
                    journey,
                    accepted.response_event_id,
                    agent_role="planning_agent",
                )
            after = tuple(journey.read())
            hostile = copy.deepcopy(read_evidence(journey))
            resumed_request = next(
                e
                for e in hostile
                if e["event_type"] == "ModelRequestRecorded"
                and e["payload"]["attempt_id"] == resumed.attempts[-1].attempt_id
            )
            resumed_request["payload"]["compatible_adapter_evidence"]["request"][
                "messages"
            ][0]["parts"][0]["content"] = "Substituted continuation prompt"
            with self.assertRaisesRegex(ValueError, "pending messages"):
                project_effort_review(hostile)
            self.assertEqual(
                respond_effort_review(journey, request.request_event_id, response),
                accepted,
            )
            self.assertEqual(tuple(journey.read()), after)
            with self.assertRaises(AgentEffortReviewResponseRefused) as conflict:
                respond_effort_review(
                    journey,
                    request.request_event_id,
                    ExtendAgentEffortResponse(action="extend", additional_requests=2),
                )
            self.assertEqual(
                conflict.exception.code, "agent_effort_review_response_conflict"
            )
            self.assertEqual(tuple(journey.read())[: len(retained)], retained)

    # The scripted outage follows the first extension.
    @members_one_at_a_time()
    def test_application_extension_resumes_only_its_member_without_repeating_sources(
        self,
    ):
        # Each member is its own Attempt with its own allowance: both stop on
        # their own two requests; each extension resumes only its member, an
        # outage on that resumed member holds it alone, and no source repeats.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {
                "schema_version": "bio-neb.investigation-request/2",
                "asserted_by": "researcher:owner",
                "requests": [
                    {
                        "request": f"Investigate {rsid} in sample-001.",
                        "target": {"rsid": rsid},
                        "normalized_question_action": "observe_variant",
                    }
                    for rsid in SequentialAdaptiveModel.rsids[:2]
                ],
            }
            journey = forked_child(root, investigation_request=request)
            request_file = root / "effort-investigation.json"
            request_file.write_text(json.dumps(request))
            model = SequentialAdaptiveModel()
            outage = {"armed": False, "fired": False}

            async def respond(messages, info):
                if outage["armed"] and not outage["fired"]:
                    outage["fired"] = True
                    raise ModelHTTPError(
                        503,
                        "offline-sequential-investigation-planner",
                        headers={"retry-after": "999"},
                    )
                if outage["fired"] and "stage_findings" in info.instructions:
                    findings = json.loads(info.instructions.rpartition("\n")[2])[
                        "stage_findings"
                    ]
                    # The resumed member sees its own retained, usable sources.
                    statuses = [item["status"] for item in findings["findings"]]
                    self.assertTrue(statuses)
                    self.assertLessEqual(set(statuses), {"resolved", "compatible"})
                return await model.respond(messages, info)

            model.model = ScriptedModelEndpoint(respond)
            sources = SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)
            begin_in_pass(
                BeginInvestigationCommand(
                    journey_file=str(journey.path),
                    request_files=(str(request_file),),
                    invocation_overrides=AgentInvocationOverrides(model_requests=2),
                ),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model.model,
                source_adapter_factory=lambda: sources,
            )
            stopped = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertEqual(stopped.status, "effort_review_pending")
            requests = project_effort_review(
                read_evidence(Journey.open(journey.path))
            ).requests
            self.assertEqual(
                [(r.requests_used, r.request_limit) for r in requests],
                [(2, 2), (2, 2)],
            )
            self.assertEqual(len({r.attempt_id for r in requests}), 2)
            self.assertEqual(sources.contact_count, 4)
            for change in (
                "return_content",
                "return_call",
                "missing_transcript",
                "pending_instructions",
            ):
                hostile = copy.deepcopy(read_evidence(Journey.open(journey.path)))
                review = next(
                    e
                    for e in hostile
                    if e["event_type"] == "AgentEffortReviewRequested"
                )
                if change == "missing_transcript":
                    review["payload"]["messages"] = []
                elif change == "pending_instructions":
                    review["payload"]["messages"][-1]["instructions"] = (
                        "Invented source context"
                    )
                else:
                    returned = next(
                        part
                        for message in review["payload"]["messages"]
                        for part in message["parts"]
                        if part["part_kind"] == "tool-return"
                    )
                    if change == "return_content":
                        returned["content"] = {"fabricated": "source result"}
                    else:
                        returned["tool_call_id"] = "wrong-call"
                with self.subTest(change=change), self.assertRaises(ValueError):
                    project_investigation(hostile)

            def extend(request):
                return AnswerReview(
                    journey_file=str(journey.path),
                    request_event_id=request.request_event_id,
                    answer=AgentEffortReviewAnswer(
                        kind="agent_effort",
                        response=ExtendAgentEffortResponse(
                            action="extend", additional_requests=2
                        ),
                    ),
                )

            with (
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "offline"}),
                use_run_advancement_dependencies(
                    investigation_model_factory=model.model,
                    investigation_source_adapter_factory=lambda: sources,
                ),
            ):
                waiting = advance(ResumeRun(journey_file=str(journey.path)))
                self.assertEqual(waiting.status, "effort_review_pending")
                outage["armed"] = True
                first = extend(requests[0])
                unavailable = advance(first)
                # The resumed member's outage holds it alone; its sibling still
                # waits on its own effort Review.
                retry_request = unavailable.model_availability_review.pending_request
                self.assertIsNotNone(retry_request)
                self.assertEqual(
                    [
                        r.attempt_id
                        for r in unavailable.agent_effort_review.requests
                        if r.pending
                    ],
                    [requests[1].attempt_id],
                )
                resumed = advance(
                    AnswerReview(
                        journey_file=str(journey.path),
                        request_event_id=retry_request.request_event_id,
                        answer=ModelAvailabilityReviewAnswer(
                            kind="model_availability",
                            response=RetryModelAvailabilityResponse(action="retry"),
                        ),
                    )
                )
                self.assertEqual(resumed.status, "effort_review_pending")
                completed = advance(extend(requests[1]))
                self.assertEqual(completed.status, "investigation_review_pending")
                self.assertEqual(
                    {
                        attempt.effective_invocation_policy.model_requests
                        for attempt in completed.investigation.attempts[2:]
                    },
                    {4},
                )
                self.assertEqual(sources.contact_count, 4)
                calls = model.calls
                advance(first)
                self.assertEqual(model.calls, calls)
                self.assertEqual(sources.contact_count, 4)
            events = Journey.open(journey.path).read()
            self.assertEqual(len(project_effort_review(events).receipts), 2)
            self.assertFalse(any(e["event_type"] == "RunFailed" for e in events))

    def test_a_member_spending_its_allowance_exactly_leaves_its_sibling_its_own(
        self,
    ):
        # Each scripted member spends exactly three requests (RefSNP, assembly
        # report, typed output). Each member is its own Attempt with its own
        # allowance, so a limit of three completes both members; no member
        # starts on another's spent requests (Errors 18935528).
        investigation_input, single = input_and_assertion()
        assertion = UserAssertion.create_collection(
            asserted_by=single.asserted_by,
            requests=tuple(
                InvestigationRequestItem(
                    request_text=f"Investigate {rsid} in sample HG001.", rsid=rsid
                )
                for rsid in SequentialAdaptiveModel.rsids[:2]
            ),
            predecessor_ids=(investigation_input.input_id,),
        )
        with tempfile.TemporaryDirectory() as directory:
            model = SequentialAdaptiveModel()
            sources = SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)
            completed, journey = run_attempt(
                Path(directory).resolve(),
                model,
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
                invocation_overrides={"model_requests": 3},
            )
            self.assertEqual(completed.status, "review_pending")
            self.assertEqual(project_effort_review(read_evidence(journey)).requests, ())
            self.assertEqual(
                sources.operations,
                [
                    operation
                    for rsid in SequentialAdaptiveModel.rsids[:2]
                    for operation in (
                        f"fetch_refsnp:{rsid}",
                        "fetch_assembly_report:GCF_000001405.40",
                    )
                ],
            )
            self.assertEqual(model.calls, 6)
            self.assertEqual(
                [
                    len(attempt.model_request_event_ids)
                    for attempt in completed.attempts
                ],
                [3, 3],
            )
            self.assertEqual(
                [
                    len(lineage.validation_attempts)
                    for lineage in completed.collection_lineages
                ],
                [1, 1],
            )

    def test_application_extend_answer_reopens_planning_without_repeating_completed_tool(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey, composition = _inspected_planning_case(Path(directory))
            stopped = composition.planning_assignment(journey).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "offline"},
                invocation_overrides=AgentInvocationOverrides(model_requests=1),
            )
            request = stopped.agent_effort_review.pending_request
            answer = AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=request.request_event_id,
                answer=AgentEffortReviewAnswer(
                    kind="agent_effort",
                    response=ExtendAgentEffortResponse(
                        action="extend", additional_requests=11
                    ),
                ),
            )
            with use_run_advancement_composition(composition):
                waiting = advance(ResumeRun(journey_file=receipt.journey_file))
                self.assertEqual(waiting.status, "effort_review_pending")
                self.assertEqual(composition.calls, ["search_journey"])
                resumed = advance(answer)
                self.assertEqual(resumed.status, "planning_hitl_pending")
                # Eleven further requests are exactly the scripted work up to the
                # first scoped question (the index search and the choice among
                # them), so the reopened limit is 1 + 11.
                self.assertEqual(len(composition.calls), 12)
                self.assertEqual(
                    resumed.attempts[-1].effective_invocation_policy.model_requests, 12
                )
                self.assertEqual(advance(answer).status, "planning_hitl_pending")
                self.assertEqual(len(composition.calls), 12)
            events = Journey.open(journey.path).read()
            history_starts = [
                e["payload"]["attempt_id"]
                for e in read_evidence(Journey.open(journey.path))
                if e["event_type"] == "ObservedToolExecutionStarted"
                and e["payload"]["tool_name"] == "search_journey"
            ]
            self.assertEqual(history_starts, [stopped.attempts[-1].attempt_id])
            self.assertEqual(len(project_effort_review(events).receipts), 1)
            self.assertFalse(any(e["event_type"] == "RunFailed" for e in events))

    def test_application_stop_answer_preserves_work_and_never_starts_another_model(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey, composition = _inspected_planning_case(Path(directory))
            stopped = composition.planning_assignment(journey).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "offline"},
                invocation_overrides=AgentInvocationOverrides(model_requests=1),
            )
            request = stopped.agent_effort_review.pending_request
            retained = journey.read()
            answer = AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=request.request_event_id,
                answer=AgentEffortReviewAnswer(
                    kind="agent_effort",
                    response=StopAgentEffortResponse(
                        action="stop", reason="Enough evidence retained"
                    ),
                ),
            )
            finished = advance(answer)
            self.assertEqual(finished.status, "finalization_review_pending")
            from bio_neb.application.run_interaction import run_interaction

            self.assertEqual(run_interaction(finished).frontier, "review_required")
            self.assertEqual(finished.current_review.kind, "finalization")
            self.assertEqual(
                finished.finalization.pending_request.basis.outcome, "effort_stopped"
            )
            self.assertEqual(len(composition.calls), 1)
            self.assertEqual(advance(answer).status, "finalization_review_pending")
            self.assertEqual(
                advance(ResumeRun(journey_file=receipt.journey_file)).status,
                "finalization_review_pending",
            )
            events = Journey.open(journey.path).read()
            self.assertEqual(events[: len(retained)], retained)
            self.assertFalse(any(e["event_type"] == "RunFailed" for e in events))
            self.assertFalse(
                any(
                    e["event_type"] in {"AttemptStarted", "ModelRequestRecorded"}
                    for e in events[len(retained) :]
                )
            )
            self.assertEqual(len(composition.calls), 1)
            with self.assertRaisesRegex(ValueError, "exact extension"):
                composition.planning_assignment(journey).run(
                    journey, environ={"LOCAL_MODEL_KEY": "offline"}
                )
            self.assertEqual(
                _semantic(Journey.open(journey.path).read()), _semantic(events)
            )
            self.assertEqual(len(composition.calls), 1)
            independent = copy.deepcopy(
                next(
                    e
                    for e in events
                    if e["event_type"] == "AttemptStarted"
                    and e["payload"]["agent_role"] == "planning_agent"
                )
            )
            independent["payload"]["agent_role"] = "investigation_planner"
            independent["event_id"] = str(uuid.uuid4())
            independent["payload"]["attempt_id"] = str(uuid.uuid4())
            project_effort_review(
                [*read_evidence(Journey.open(journey.path)), independent]
            )
            independent["payload"]["agent_role"] = "planning_agent"
            with self.assertRaisesRegex(ValueError, "exact extension"):
                project_effort_review(
                    [*read_evidence(Journey.open(journey.path)), independent]
                )
            with self.assertRaises(AgentEffortReviewResponseRefused) as conflict:
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
            self.assertEqual(
                conflict.exception.code, "agent_effort_review_response_conflict"
            )
