"""A non-planning assignment runs through the shared Attempt lifecycle's hooks.

The test-local assignment binds its own runtime and causal head, drives two
member invocations inside one Attempt and records terminals planning never
writes. Model contact is the real Adapter over a scripted transport; no provider is reached.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_ai.messages import (
    ModelResponse,
    ToolCallPart,
)

from bio_neb.agentic_journey.agent_roles import (
    AttemptAdmission,
    AttemptAssignment,
    RuntimeBinding,
)
from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
from bio_neb.agentic_journey.invocation import (
    ActionAdmissionSpec,
    AssignmentOutput,
    assemble_agent,
)
from bio_neb.agentic_journey.runtime_config import retained_runtime_selections
from bio_neb.failure import Invariant
from bio_neb.journey import Journey
from bio_neb.providers.nebius_model_endpoint import NebiusModelEndpointAdapter
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.model_lifetime_support import OwnedModelEndpoint
from tests.proposal_information_fakes import registered_run

_ENVIRON = {"LOCAL_MODEL_KEY": "offline"}
_MEMBERS = ("member-a", "member-b")


class ProbeTerminal(BaseModel):
    """A terminal only the test-local assignment records."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    event_type: Literal["ProbeTerminalRecorded"] = "ProbeTerminalRecorded"
    attempt_id: str
    outcome: Literal["succeeded", "failed"]
    detail: str


class ProbeMemberRejected(Exception):
    pass


class _ProbeAssignment:
    """Two members per Attempt, each done once it has read retained history:
    member-a after one read, member-b after two. A text ending is a correction
    in every role, so the probe completes through its Tool, never in text."""

    def __init__(self, *, reject_first_member: bool = False, tools=None, output=None):
        self.reject_first_member = reject_first_member
        self.tools = tools or {}
        self.output = output
        self.model_requests = 0
        self.continuation_checks = []
        self.model_constructions = 0
        self.model = ScriptedModelEndpoint(self._respond)

    async def _respond(self, messages, info):
        self.model_requests += 1
        # Each read differs: an identical read with nothing changed would be a
        # repeat the loop guard stops before it runs (ADR 0044).
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "search_journey",
                    {"limit": self.model_requests},
                    tool_call_id=f"history-{self.model_requests}",
                )
            ]
        )

    def _model_factory(self, _loaded, _selected):
        self.model_constructions += 1
        return self.model(_loaded, _selected)

    def assignment(self) -> AttemptAssignment:
        return AttemptAssignment(
            agent_role="investigation_planner",
            activity_stage="investigation_planning_attempt",
            admit=self._admit,
            validate_continuation=self._validate_continuation,
            open=self._open,
            drive=self._drive,
            complete=self._complete,
            fail=self._fail,
            read_result=lambda journey: journey.read(),
            model_factory=self._model_factory,
        )

    def _admit(self, journey, reentry):
        events = journey.read()
        registered = next(
            e["payload"] for e in events if e["event_type"] == "RunInputsRegistered"
        )
        head = events[-1]["event_id"]
        return AttemptAdmission(
            _MEMBERS,
            RuntimeBinding(
                retained_runtime_selections(registered["runtime_selections"]),
                registered["input_set_id"],
            ),
            causal_head=head,
            # A fresh Attempt's availability stop retries from its causal head;
            # the availability review refuses any other frontier.
            availability_retry_frontier=head,
        )

    def _validate_continuation(
        self, events, *, attempt_id, policy, next_request, member_id=None
    ):
        # Each member's first request is its own progress; planning's rule
        # would refuse the second member after a text-only first answer.
        self.continuation_checks.append((attempt_id, next_request))

    @contextmanager
    def _open(self, context, observe):
        observed = observe()
        members = SimpleNamespace(current=None)

        def member_agent():
            return assemble_agent(
                observed,
                assignment="investigation_planner",
                instructions="Answer the member request in text.",
                tools=self.tools,
                stage=lambda _deps: "source_reading",
                bound_values=lambda _deps, _name: None,
                admission=ActionAdmissionSpec(validator=lambda _deps, _action: ()),
                policy=context.effective_policy,
                model_policy=context.model,
                role_text=context.instructions,
                journey=context.journey,
                attempt_id=context.attempt_id,
                parent_activity=context.activity,
                failure_source=context.failure_source,
                dependencies=lambda _admission, _observed: SimpleNamespace(
                    journey=context.journey
                ),
                output=self.output,
            )

        members.agent = member_agent
        yield members

    async def _drive(self, context, members, invoke_member, message_history):
        used = []
        for reads, member_id in enumerate(context.admitted, start=1):
            members.current = member_id
            agent, deps = members.agent()
            first = self.model_requests

            def done(first=first, reads=reads, member_id=member_id):
                if self.model_requests - first < reads:
                    return None
                return SimpleNamespace(checkpoint=f"{member_id}-done")

            await invoke_member(
                agent,
                f"Resolve {member_id}.",
                deps=deps,
                member_id=member_id,
                completed_work=done,
            )
            used.append(self.model_requests)
            if self.reject_first_member:
                raise ProbeMemberRejected("member answer rejected")
        return tuple(used)

    def _complete(self, context, used):
        terminal = context.activity.record(
            ProbeTerminal(
                attempt_id=context.attempt_id,
                outcome="succeeded",
                detail=",".join(str(count) for count in used),
            )
        )
        context.activity.succeeded(terminal)

    def _fail(self, context, members, error):
        if not isinstance(error, ProbeMemberRejected):
            return False
        # The failing member is per-Attempt state that only ``open`` created.
        terminal = context.activity.record(
            ProbeTerminal(
                attempt_id=context.attempt_id,
                outcome="failed",
                detail=f"{members.current}: {error}",
            )
        )
        context.activity.failed(terminal)
        return True


def _of(events, event_type):
    return [e for e in events if e["event_type"] == event_type]


class AttemptLifecycleHookTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        receipt, _ = registered_run(Path(directory.name))
        self.journey_file = Path(receipt.journey_file)

    def owned_assignment(self, probe, *, close_error=None):
        models = []

        def factory(loaded, selected):
            model = OwnedModelEndpoint(probe._respond, close_error=close_error)
            models.append(model)
            return model(loaded, selected)

        return replace(probe.assignment(), model_factory=factory), models

    def test_repeated_attempts_release_each_transport_on_its_own_loop(self):
        models = []
        for _ in range(2):
            probe = _ProbeAssignment()

            def factory(loaded, selected, probe=probe):
                model = OwnedModelEndpoint(probe._respond)
                models.append(model)
                return model(loaded, selected)

            events = replace(probe.assignment(), model_factory=factory).run(
                Journey.open(self.journey_file), environ=_ENVIRON
            )
            self.assertEqual(
                _of(events, "ProbeTerminalRecorded")[-1]["payload"]["outcome"],
                "succeeded",
            )
            transport = models[-1].transport
            self.assertEqual((transport.requests, transport.close_started), (3, 1))
            self.assertTrue(transport.closed)
            self.assertTrue(transport.loop.is_closed())
        self.assertEqual(len(models), 2)
        self.assertIsNot(models[0].transport.loop, models[1].transport.loop)

    def test_cleanup_fault_preserves_success_and_fails_without_replaying_members(self):
        cleanup = RuntimeError("controlled cleanup failure")
        assignment, models = self.owned_assignment(
            _ProbeAssignment(), close_error=cleanup
        )
        with self.assertRaises(RuntimeError) as raised:
            assignment.run(Journey.open(self.journey_file), environ=_ENVIRON)
        self.assertIs(raised.exception, cleanup)
        events = Journey.open(self.journey_file).read()
        (terminal,) = _of(events, "ProbeTerminalRecorded")
        self.assertEqual(terminal["payload"]["outcome"], "succeeded")
        self.assertEqual(len(_of(events, "AttemptStarted")), 1)
        self.assertEqual(_of(events, "RunFailed"), [])
        self.assertEqual(models[0].transport.requests, 3)
        self.assertTrue(models[0].transport.closed)

    def test_pending_review_survives_cleanup_fault_without_a_failure_terminal(self):
        cleanup = RuntimeError("controlled cleanup failure")
        assignment, models = self.owned_assignment(
            _ProbeAssignment(), close_error=cleanup
        )
        with self.assertRaises(RuntimeError) as raised:
            assignment.run(
                Journey.open(self.journey_file),
                environ=_ENVIRON,
                invocation_overrides=AgentInvocationOverrides(model_requests=2),
            )
        self.assertIs(raised.exception, cleanup)
        events = Journey.open(self.journey_file).read()
        self.assertEqual(len(_of(events, "AgentEffortReviewRequested")), 1)
        self.assertEqual(_of(events, "ProbeTerminalRecorded"), [])
        self.assertEqual(models[0].transport.requests, 2)
        self.assertTrue(models[0].transport.closed)

    def test_handled_primary_failure_is_retained_and_wins_when_cleanup_also_fails(self):
        assignment, models = self.owned_assignment(
            _ProbeAssignment(reject_first_member=True),
            close_error=RuntimeError("private cleanup detail"),
        )
        with self.assertRaisesRegex(
            ProbeMemberRejected, "member answer rejected"
        ) as raised:
            assignment.run(Journey.open(self.journey_file), environ=_ENVIRON)
        self.assertEqual(
            raised.exception.__notes__,
            ["Model client cleanup also failed: RuntimeError"],
        )
        events = Journey.open(self.journey_file).read()
        (terminal,) = _of(events, "ProbeTerminalRecorded")
        self.assertEqual(terminal["payload"]["outcome"], "failed")
        self.assertEqual(models[0].transport.requests, 1)
        self.assertTrue(models[0].transport.closed)

    def test_unknown_failure_reraises_after_retaining_result_and_releasing_client(self):
        assignment, models = self.owned_assignment(
            _ProbeAssignment(reject_first_member=True)
        )
        with self.assertRaisesRegex(ProbeMemberRejected, "member answer rejected"):
            assignment.run(Journey.open(self.journey_file), environ=_ENVIRON)
        events = Journey.open(self.journey_file).read()
        (terminal,) = _of(events, "ProbeTerminalRecorded")
        self.assertEqual(terminal["payload"]["outcome"], "failed")
        self.assertEqual(models[0].transport.requests, 1)
        self.assertEqual(models[0].transport.close_started, 1)
        self.assertTrue(models[0].transport.closed)

    def test_context_entry_failure_releases_its_acquired_resource(self):
        primary = ValueError("controlled entry failure")
        probe = _ProbeAssignment()
        models = []

        class RefusedEntry(NebiusModelEndpointAdapter):
            async def __aenter__(self):
                raise primary

        def factory(loaded, selected):
            model = OwnedModelEndpoint(probe._respond, adapter=RefusedEntry)
            models.append(model)
            return model(loaded, selected)

        before = self.journey_file.read_bytes()
        with self.assertRaises(ValueError) as raised:
            replace(probe.assignment(), model_factory=factory).run(
                Journey.open(self.journey_file), environ=_ENVIRON
            )
        self.assertIs(raised.exception, primary)
        self.assertEqual(self.journey_file.read_bytes(), before)
        self.assertEqual(models[0].transport.requests, 0)
        self.assertEqual(models[0].transport.close_started, 1)
        self.assertTrue(models[0].transport.closed)

    def test_unhandled_failure_and_cancellation_release_before_reraising_primary(self):
        for primary in (
            ValueError("primary failure"),
            asyncio.CancelledError("cancelled"),
        ):
            for cleanup in (None, RuntimeError("private cleanup detail")):
                with self.subTest(
                    primary=type(primary).__name__, cleanup=cleanup is not None
                ):

                    async def fail(messages, info, primary=primary):
                        raise primary

                    probe = _ProbeAssignment()
                    probe._respond = fail
                    assignment, models = self.owned_assignment(
                        probe, close_error=cleanup
                    )
                    with self.assertRaises(type(primary)) as raised:
                        assignment.run(
                            Journey.open(self.journey_file), environ=_ENVIRON
                        )
                    self.assertIs(raised.exception, primary)
                    self.assertEqual(models[0].transport.close_started, 1)
                    self.assertTrue(models[0].transport.closed)
                    self.assertTrue(models[0].transport.loop.is_closed())
                    if cleanup is not None:
                        self.assertIn(
                            "Model client cleanup also failed: RuntimeError",
                            primary.__notes__,
                        )

    def test_assembly_failure_releases_the_acquired_client_before_any_request(self):
        def undocumented(ctx):
            return "unused"

        assignment, models = self.owned_assignment(
            _ProbeAssignment(tools={"undocumented": undocumented})
        )
        with self.assertRaisesRegex(RuntimeError, "offered Tool states no purpose"):
            assignment.run(Journey.open(self.journey_file), environ=_ENVIRON)
        self.assertEqual(models[0].transport.requests, 0)
        self.assertEqual(models[0].transport.close_started, 1)
        self.assertTrue(models[0].transport.closed)

    def test_advertised_endpoint_refusal_releases_before_attempt_or_model_contact(self):
        from bio_neb.agentic_journey.runtime_config import NebiusModelEndpointIdentity

        models = []
        probe = _ProbeAssignment()

        def factory(loaded, selected):
            model = OwnedModelEndpoint(probe._respond)
            models.append(model)
            adapter = model(loaded, selected)
            adapter.nebius_model_endpoint = NebiusModelEndpointIdentity(
                provider="nebius",
                model="unselected",
                contacted_url="https://offline.invalid/v1",
            )
            return adapter

        before = Journey.open(self.journey_file).read()
        with self.assertRaises(Invariant) as held:
            replace(probe.assignment(), model_factory=factory).run(
                Journey.open(self.journey_file), environ=_ENVIRON
            )
        self.assertEqual(held.exception.invariant_id, "runtime_config_changed")
        added = Journey.open(self.journey_file).read()[len(before) :]
        self.assertEqual(
            [e["event_type"] for e in added],
            ["PassStarted", "RuntimeConfigurationFailed", "PassClosed"],
        )
        self.assertEqual(models[0].transport.requests, 0)
        self.assertTrue(models[0].transport.closed)

    def test_public_planning_releases_every_attempt_before_returning_its_review(self):
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.application.queries import ReviewJourneyQuery, review_journey
        from tests.application_fakes import _registered
        from tests.planning_assignment_fakes import PlanningComposition

        models = []

        class OwnedPlanning(PlanningComposition):
            def planning_assignment(self, journey):
                assignment = super().planning_assignment(journey)

                def factory(loaded, selected):
                    model = OwnedModelEndpoint(self.respond)
                    models.append(model)
                    return model(loaded, selected)

                return replace(assignment, model_factory=factory)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = OwnedPlanning(receipt, paths)
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(result.status, "planning_hitl_pending")
            journey = Journey.open(Path(receipt.journey_file))
            before = journey.path.read_bytes()
            reopened = review_journey(
                ReviewJourneyQuery(journey_file=str(journey.path))
            )
            self.assertEqual(reopened.status, "planning_hitl_pending")
            self.assertEqual(journey.path.read_bytes(), before)
            self.assertEqual(len(models), len(_of(journey.read(), "AttemptStarted")))
            self.assertGreater(len(models), 0)
            for model in models:
                self.assertTrue(model.transport.closed)
                self.assertEqual(model.transport.close_started, 1)
                self.assertTrue(model.transport.loop.is_closed())
            self.assertEqual(_of(journey.read(), "RunFailed"), [])

    def test_configuration_failure_records_the_fault_at_the_causal_head_before_any_attempt(
        self,
    ):
        probe = _ProbeAssignment()
        before = Journey.open(self.journey_file).read()

        with self.assertRaises(Invariant) as held:
            probe.assignment().run(Journey.open(self.journey_file), environ={})

        self.assertEqual(held.exception.invariant_id, "model_credential_unavailable")
        _, failure, _ = Journey.open(self.journey_file).read()[len(before) :]
        self.assertEqual(failure["event_type"], "RuntimeConfigurationFailed")
        self.assertEqual(failure["parent_ids"], [before[-1]["event_id"]])
        self.assertEqual(
            held.exception.evidence["configuration_failure_event_id"],
            failure["event_id"],
        )
        self.assertEqual((probe.model_constructions, probe.model_requests), (0, 0))

    def test_two_members_share_one_attempt_and_record_its_own_success(self):
        probe = _ProbeAssignment()

        events = probe.assignment().run(
            Journey.open(self.journey_file), environ=_ENVIRON
        )

        (started,) = _of(events, "AttemptStarted")
        attempt_id = started["payload"]["attempt_id"]
        self.assertEqual(started["payload"]["agent_role"], "investigation_planner")
        self.assertEqual(
            started["payload"]["effective_invocation_policy"]["model_requests"], 200
        )
        responses = _of(events, "ModelResponseRecorded")
        self.assertEqual(
            [r["payload"]["attempt_id"] for r in responses], [attempt_id] * 3
        )
        self.assertEqual(probe.continuation_checks, [(attempt_id, True)] * 3)
        (terminal,) = _of(events, "ProbeTerminalRecorded")
        # member-b's two requests are counted on top of member-a's one.
        self.assertEqual(
            terminal["payload"],
            {"attempt_id": attempt_id, "outcome": "succeeded", "detail": "1,3"},
        )
        self.assertEqual(
            _of(events, "activity_succeeded")[-1]["payload"]["result_event_id"],
            terminal["event_id"],
        )
        for planning_terminal in ("SliceCheckpointReached", "AttemptSucceeded"):
            self.assertEqual(_of(events, planning_terminal), [])

    def test_a_role_offered_only_the_history_tool_is_not_forced_to_call_it(self):
        # The probe has no output Tool and its stage shows no domain Tool: with
        # nothing to do but read history, the request stays unforced, under the
        # Run's one recorded choice.
        events = (
            _ProbeAssignment()
            .assignment()
            .run(Journey.open(self.journey_file), environ=_ENVIRON)
        )

        requests = [
            e["payload"]["compatible_adapter_evidence"]["request"]
            for e in _of(events, "ModelRequestRecorded")
        ]
        self.assertEqual(len(requests), 3)
        for request in requests:
            self.assertEqual(
                [tool["name"] for tool in request["function_tools"]],
                ["search_journey"],
            )
            self.assertEqual(request["model_settings"]["tool_choice"], "auto")

    def test_a_declared_tool_refusal_is_the_calls_result_and_a_failed_turn(self):
        """Owner, #477 C8 (alignment gate 2026-10-28): a Tool's declared refusal
        is shown to the model as the call's result, the Attempt continues, and
        the reply that held it is one more failed turn."""
        from unittest.mock import patch

        from bio_neb.agentic_journey import invocation
        from bio_neb.agentic_journey.continuation_policy import model_output_faults
        from bio_neb.failure import ToolRefused
        from bio_neb.recording import read_evidence

        reads = []
        search = invocation.search_journey

        def refuse_first(journey, **query):
            reads.append(query)
            # member-b's first read: its second request carries the refusal.
            if len(reads) == 2:
                raise ToolRefused("probe_unready", "the probe has nothing to read yet")
            return search(journey, **query)

        with patch.object(invocation, "search_journey", refuse_first):
            _ProbeAssignment().assignment().run(
                Journey.open(self.journey_file), environ=_ENVIRON
            )
        events = read_evidence(Journey.open(self.journey_file))
        (refused,) = _of(events, "ToolExecutionRefused")
        (failed,) = _of(events, "ObservedToolExecutionFailed")
        self.assertEqual(
            refused["payload"],
            {
                "attempt_id": failed["payload"]["attempt_id"],
                "failed_event_id": failed["event_id"],
                "reason": "probe_unready",
                "statement": "the probe has nothing to read yet",
                "fault_key": "search_journey: probe_unready",
                # The failure module's next step and the reply's place in the
                # failed turns, recorded as shown (F8).
                "next": "Call it again only with arguments that can produce its "
                "result, or call another Tool callable now.",
                "count": refused["payload"]["count"],
            },
        )
        self.assertRegex(
            refused["payload"]["count"],
            r"^Count: failed turn 1 of ([1-9][0-9]*) in a row\. At \1 the Attempt "
            r"stops and the researcher reviews it\.$",
        )
        self.assertEqual(failed["payload"]["mechanical_error_type"], "ToolRefused")
        # The next request carries the refusal as the call's result.
        returned = [
            part
            for request in _of(events, "ModelRequestRecorded")
            for message in request["payload"]["compatible_adapter_evidence"]["request"][
                "messages"
            ]
            for part in message.get("parts", [])
            if part.get("part_kind") == "tool-return"
            and part.get("tool_call_id") == "history-2"
        ]
        self.assertTrue(returned)
        self.assertEqual(
            returned[0]["content"],
            {
                "status": "refused",
                "reason": "probe_unready",
                "statement": "the probe has nothing to read yet",
                # Then where to go and the reply's place (F8).
                "next": "Call it again only with arguments that can produce its "
                "result, or call another Tool callable now.",
                "count": refused["payload"]["count"],
            },
        )
        # The refusal's own id rides beside the view, never sent.
        self.assertEqual(returned[0]["metadata"], {"event_id": refused["event_id"]})
        self.assertEqual(_of(events, "AttemptFailed"), [])
        (fault,) = model_output_faults(
            events, attempt_id=refused["payload"]["attempt_id"]
        )
        self.assertEqual(
            (fault.key, fault.count, fault.failed),
            ("tool_refused: search_journey: probe_unready", 1, True),
        )

    def test_assembly_refuses_a_tool_that_states_no_purpose(self):
        def undocumented_probe_tool(ctx) -> str:
            return "never offered"

        probe = _ProbeAssignment(
            tools={"undocumented_probe_tool": undocumented_probe_tool}
        )

        with self.assertRaisesRegex(
            RuntimeError, "offered Tool states no purpose: undocumented_probe_tool$"
        ):
            probe.assignment().run(Journey.open(self.journey_file), environ=_ENVIRON)

        self.assertEqual(probe.model_requests, 0)

    def test_assembly_refuses_an_output_tool_that_states_no_purpose(self):
        probe = _ProbeAssignment(
            output=AssignmentOutput(
                output_type=ProbeTerminal,
                name="record_probe_terminal",
                description=" ",
                validator=lambda _ctx, value: value,
            )
        )

        with self.assertRaisesRegex(
            RuntimeError, "offered Tool states no purpose: record_probe_terminal$"
        ):
            probe.assignment().run(Journey.open(self.journey_file), environ=_ENVIRON)

        self.assertEqual(probe.model_requests, 0)

    def test_second_member_stops_for_effort_after_the_first_used_the_allowance(self):
        probe = _ProbeAssignment()
        head = Journey.open(self.journey_file).read()[-1]["event_id"]

        events = probe.assignment().run(
            Journey.open(self.journey_file),
            environ=_ENVIRON,
            invocation_overrides=AgentInvocationOverrides(model_requests=2),
        )

        # member-b alone fits two requests; after member-a it gets only one.
        self.assertEqual(probe.model_requests, 2)
        (attempt_id,) = {
            e["payload"]["attempt_id"] for e in _of(events, "AttemptStarted")
        }
        (review,) = _of(events, "AgentEffortReviewRequested")
        self.assertEqual(review["payload"]["attempt_id"], attempt_id)
        self.assertEqual(
            (review["payload"]["requests_used"], review["payload"]["request_limit"]),
            (2, 2),
        )
        # The effort review retries from the causal head the assignment admitted.
        self.assertEqual(review["payload"]["retry_frontier_event_id"], head)
        self.assertEqual(len(_of(events, "AttemptStopped")), 1)
        self.assertEqual(_of(events, "ProbeTerminalRecorded"), [])

    def test_assignment_failure_terminal_is_its_own_and_leaves_the_run_open(self):
        probe = _ProbeAssignment(reject_first_member=True)

        with self.assertRaisesRegex(ProbeMemberRejected, "member answer rejected"):
            probe.assignment().run(Journey.open(self.journey_file), environ=_ENVIRON)
        events = Journey.open(self.journey_file).read()

        self.assertEqual(probe.model_requests, 1)
        (terminal,) = _of(events, "ProbeTerminalRecorded")
        self.assertEqual(
            (terminal["payload"]["outcome"], terminal["payload"]["detail"]),
            ("failed", "member-a: member answer rejected"),
        )
        self.assertEqual(
            _of(events, "activity_failed")[-1]["payload"]["failure_event_id"],
            terminal["event_id"],
        )
        for planning_failure in ("AttemptFailed", "RunFailed"):
            self.assertEqual(_of(events, planning_failure), [])


if __name__ == "__main__":
    unittest.main()
