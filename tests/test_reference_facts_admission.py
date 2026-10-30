from __future__ import annotations

import unittest
import uuid

from pydantic_ai import ModelRetry

from bio_neb.agentic_journey.action_admission import (
    ActionAdmissionSession,
    ProposedAgentAction,
)
from bio_neb.agentic_journey.action_policy import call_identity
from bio_neb.agentic_journey.agent_work_review import AgentFailedTurns
from bio_neb.agentic_journey.attempt_journey import AttemptFailureSource
from bio_neb.agentic_journey.continuation_policy import (
    ContinuationPolicy,
    validate_continuation,
)
from bio_neb.failure import CorrectionWording
from bio_neb.reference_discovery.reference_facts_journey import project_reference_facts
from bio_neb.reference_discovery.reference_tools import (
    COMPANION_TOOLS,
    reference_tool_errors,
    reference_tool_values,
)
from tests.agent_attempt_fakes import received_reply
from tests.reference_fakes import (
    append_observation,
    companion_record,
    companion_timeline,
    compatible_record,
    payload,
)


class _Journey:
    def __init__(self, timeline):
        self.timeline = timeline

    def read(self):
        return self.timeline


class _Activity:
    def __init__(self):
        self.events = []

    def record(self, fact):
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": fact.event_type,
            "payload": payload(fact),
            "parent_ids": [],
        }
        self.events.append(event)
        return event


class _Dependencies:
    def __init__(self, timeline):
        self.journey = _Journey(timeline)


class ReferenceFactsAdmissionTests(unittest.TestCase):
    # The first failed turn is the limit: its correction is never sent.
    POLICY = ContinuationPolicy(failed_turns=1)

    def _session(self, timeline):
        activity = _Activity()
        dependencies = _Dependencies(timeline)
        self.attempt_id = str(uuid.uuid4())
        # The Attempt's start and the request this reply answers.
        self.attempt_events = [
            {
                "event_id": str(uuid.uuid4()),
                "event_type": "AttemptStarted",
                "payload": {
                    "attempt_id": self.attempt_id,
                    "continuation_policy": self.POLICY.model_dump(mode="json"),
                    "continuation_policy_id": self.POLICY.policy_id,
                },
            },
            {
                "event_id": str(uuid.uuid4()),
                "event_type": "ModelRequestRecorded",
                "payload": {"attempt_id": self.attempt_id},
            },
        ]
        session = ActionAdmissionSession(
            wording=CorrectionWording(),
            attempt_id=self.attempt_id,
            parent_activity=activity,
            failure_source=AttemptFailureSource(),
            eligible_tool_names=frozenset(COMPANION_TOOLS),
            action_validator=lambda action: reference_tool_errors(dependencies, action),
            action_fingerprint=call_identity,
            admitted_repeat_error="reference_facts_action_repeated",
            continuation=self.POLICY,
            events=lambda: [*self.attempt_events, *activity.events],
        )
        return session, activity, dependencies

    def _next_request_stops(self, activity):
        with self.assertRaises(AgentFailedTurns) as stopped:
            validate_continuation(
                [*self.attempt_events, *activity.events],
                attempt_id=self.attempt_id,
                policy=self.POLICY,
                next_request=True,
            )
        return stopped.exception

    def _offered_action(self, dependencies):
        # The model names the due companion role; nothing else is bound.
        self.assertEqual(
            reference_tool_values(dependencies, "describe_companion").constants, {}
        )
        return ProposedAgentAction(
            tool_name="describe_companion",
            tool_call_id="describe-1",
            arguments={"asset": "genomic_fasta", "accession": "GCF_000001405.40"},
        )

    def test_ambiguous_and_multiple_results_allow_exploration_without_asset_selection(
        self,
    ):
        # Synthetic source records exercise public eligibility and the real typed
        # Tool/projector; integrated planning and call binding have separate flow proof.
        for outcome, authority in (("multiple", True), ("ambiguous", False)):
            with self.subTest(outcome=outcome):
                timeline, work, _ = companion_timeline()
                append_observation(
                    timeline,
                    work,
                    label="unsettled-assets",
                    records=(
                        compatible_record(work, "asset-A"),
                        compatible_record(work, "asset-B", authority=authority),
                    ),
                )
                dependencies = _Dependencies(timeline)
                self.assertEqual(
                    reference_tool_values(dependencies, "describe_companion").constants,
                    {},
                )
                changed = ProposedAgentAction(
                    tool_name="describe_companion",
                    tool_call_id="explore-other",
                    arguments={
                        "asset": "genomic_fasta",
                        "accession": "GCF_000000001.1",
                    },
                )
                self.assertEqual(reference_tool_errors(dependencies, changed), ())
                append_observation(
                    timeline,
                    work,
                    label="other-assembly",
                    assembly_accession="GCF_000000001.1",
                    records=(
                        companion_record("other", assembly_accession="GCF_000000001.1"),
                    ),
                )
                projected = project_reference_facts(timeline)
                self.assertEqual(
                    [f.result["outcome"] for f in projected.facts],
                    [outcome, "incompatible"],
                )
                self.assertEqual(projected.asset_dispositions, ())
                self.assertEqual(projected.completed_work_item_ids, ())
                self.assertEqual(projected.pending_work_item_ids, (work.work_item_id,))
                selection = reference_tool_values(
                    dependencies, "request_companion_asset_selection"
                )
                self.assertEqual(
                    selection.constants["basis_event_id"],
                    projected.facts[0].source_event_id,
                )

    def test_a_role_the_reference_does_not_need_is_refused_with_its_reason(self):
        timeline, _, _ = companion_timeline()
        dependencies = _Dependencies(timeline)
        offered = self._offered_action(dependencies)

        self.assertEqual(reference_tool_errors(dependencies, offered), ())
        self.assertEqual(
            reference_tool_errors(
                dependencies,
                ProposedAgentAction(
                    tool_name=offered.tool_name,
                    tool_call_id="other-role",
                    arguments={"asset": "annotation", "accession": "GCF_000001405.40"},
                ),
            ),
            ("companion_asset_not_due",),
        )

    def test_invalid_singleton_at_the_limit_stops_before_its_correction_is_sent(
        self,
    ):
        timeline, _, _ = companion_timeline()
        session, activity, dependencies = self._session(timeline)
        action = self._offered_action(dependencies)
        invalid = ProposedAgentAction(
            tool_name=action.tool_name,
            tool_call_id=action.tool_call_id,
            arguments={"asset": "annotation", "accession": "GCF_000001405.40"},
        )
        # Admission records the refusal and its correction; the next request's
        # check stops the Attempt, so that correction is never sent.
        with self.assertRaises(ModelRetry):
            session.admit_turn(
                response_event_id=str(uuid.uuid4()),
                actions=(invalid,),
                exposed_tool_names=frozenset({action.tool_name}),
            )
        (disposition,) = activity.events[-1:]
        self.assertEqual(
            (
                disposition["payload"]["disposition"],
                disposition["payload"]["correction_count"],
            ),
            ("singleton_refused", 1),
        )
        self.assertEqual(
            activity.events[0]["payload"]["errors"], ["companion_asset_not_due"]
        )
        stopped = self._next_request_stops(activity)
        self.assertEqual(stopped.cause, "failed_turns")
        self.assertEqual(stopped.source_event_id, disposition["event_id"])

    def test_a_mixed_reply_names_its_unoffered_call_and_copy_before_any_runs(self):
        timeline, _, _ = companion_timeline()
        offered = self._offered_action(_Dependencies(timeline))
        actions = (
            offered,
            ProposedAgentAction(
                tool_name="not_a_reference_facts_action",
                tool_call_id="off-menu",
                arguments={},
            ),
            ProposedAgentAction(
                tool_name=offered.tool_name,
                tool_call_id="malformed",
                arguments={"invalid_json": "{"},
                arguments_invalid_json=True,
            ),
            ProposedAgentAction(
                tool_name=offered.tool_name,
                tool_call_id="copy",
                arguments=offered.arguments,
            ),
        )
        reply, turn = received_reply(actions)
        session, activity, _dependencies = self._session(timeline)
        # The unoffered call is refused on its own and the copy collapses onto
        # its first call; routing never stops the Attempt: the kept calls go on
        # to admission, one by one.
        routing = session.route_reply(reply, turn, frozenset({offered.tool_name}))
        self.assertEqual(
            [call.tool_call_id for call in routing.run_order],
            [offered.tool_call_id, "malformed"],
        )
        self.assertIsNotNone(routing.note)
        validations = [
            event["payload"]
            for event in activity.events
            if event["event_type"] == "ToolCallValidated"
        ]
        self.assertEqual(
            [(v["tool_call_id"], v["errors"]) for v in validations],
            [("off-menu", ["tool_name_ineligible"])],
        )
        (routed,) = [
            event
            for event in activity.events
            if event["event_type"] == "AgentTurnRouted"
        ]
        self.assertEqual(
            (routed["payload"]["run_order"], routed["payload"]["copies"]),
            ([offered.tool_call_id, "malformed"], {"copy": offered.tool_call_id}),
        )
        self.assertEqual(
            [(f["key"], f["count"]) for f in routed["payload"]["faults"]],
            [("copied_calls", 1), ("unoffered_calls", 1)],
        )
        self.assertFalse(
            any(
                event["event_type"]
                in {
                    "AgentActionTurnDispositionRecorded",
                    "AgentActionReviewRequested",
                    "ObservedToolExecutionStarted",
                }
                for event in activity.events
            )
        )
        # Were none of its calls to run, the reply would be a failed turn: at
        # this limit of one, the next request's check stops on the routing.
        self.assertEqual(
            self._next_request_stops(activity).source_event_id, routed["event_id"]
        )


if __name__ == "__main__":
    unittest.main()
