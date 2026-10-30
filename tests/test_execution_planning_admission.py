from __future__ import annotations

import inspect
import unittest
import uuid
from dataclasses import replace

from pydantic import ValidationError
from pydantic_ai import ModelRetry, ToolFailed

from bio_neb.agentic_journey.action_admission import (
    ActionAdmissionSession,
    ProposedAgentAction,
)
from bio_neb.agentic_journey.action_policy import PreconditionUnmet, call_identity
from bio_neb.agentic_journey.attempt_journey import (
    AttemptFailureSource,
    SliceCheckpointReached,
    ToolCallValidated,
)
from bio_neb.agentic_journey.continuation_policy import (
    ContinuationPolicy,
    failed_turns_spent,
)
from bio_neb.failure import CorrectionWording
from bio_neb.parabricks import (
    execution_planning_agent as execution_planning_module,
)
from bio_neb.parabricks.capability_policy import CapabilitySnapshotAdapter
from bio_neb.parabricks.execution_planning_action_policy import (
    EXECUTION_PLANNING_TOOLS,
    execution_tool_errors,
    execution_tool_values,
)
from bio_neb.parabricks.execution_planning_journey import (
    PlanningCandidateSelected,
    project_execution_planning,
)
from bio_neb.parabricks.planning_hitl import (
    PlanningHitlCandidate,
    PlanningHitlRequested,
    SelectPlanningHitlResponse,
    make_planning_hitl_responded,
)
from bio_neb.parabricks.recipe_resources import candidate_disposition_id
from tests.agent_attempt_fakes import FakeActivity, FakeJourney, received_reply
from tests.planning_fakes import (
    add_inspection,
    add_requirement,
    add_target_requirement,
    append_asserted_recipe_observation,
    policy,
)
from tests.reference_fakes import (
    append,
    append_observation,
    attempt_start,
    companion_timeline,
    compatible_record,
    payload,
)


class _Dependencies:
    def __init__(self, timeline):
        self.journey = FakeJourney(timeline)


def _offered(dependencies):
    """The execution actions whose preconditions hold at this prefix."""
    offered = set()
    for name in EXECUTION_PLANNING_TOOLS:
        try:
            execution_tool_values(dependencies, name)
        except PreconditionUnmet as unmet:
            assert unmet.code == "execution_step_not_current", unmet.code
        else:
            offered.add(name)
    return offered


def _planning_timeline():
    timeline, companion_work, _ = companion_timeline()
    add_requirement(timeline, "analysis_recipe", "analysis_recipe", "1")
    add_requirement(timeline, "resource_profile", "resource_profile", "2")
    add_target_requirement(timeline)
    add_inspection(timeline)
    append_observation(
        timeline,
        companion_work,
        label="execution-admission-companion",
        records=(compatible_record(companion_work),),
    )
    return timeline


def _taxon_record(timeline):
    return next(e for e in timeline if e["event_type"] == "TaxonResolutionRecorded")


def _judged(timeline, **changes):
    """The model's ploidy judgment, citing the Run's taxonomy record by its words."""
    return {
        "organism_ploidy": "diploid",
        "organism_record_event_id": _taxon_record(timeline)["event_id"],
        "organism_quote": "Homo sapiens",
        **changes,
    }


class OrganismPloidyCitationTests(unittest.TestCase):
    """Code checks the model's ploidy judgment only by membership: the cited record
    is one the reference stage kept for this Run and the quote is its words (ADR 0049;
    #470 Q1 a; #469 Q2 names the organism record)."""

    def setUp(self):
        self.timeline = _planning_timeline()
        append_asserted_recipe_observation(
            self.timeline, CapabilitySnapshotAdapter(policy()), "ploidy-recipe"
        )
        self.dependencies = _Dependencies(self.timeline)

    def _errors(self, **changes):
        return execution_tool_errors(
            self.dependencies,
            ProposedAgentAction(
                "select_recipe_candidate", "call", _judged(self.timeline, **changes)
            ),
        )

    def test_the_offered_records_are_the_runs_organism_records(self):
        offered = execution_tool_values(
            self.dependencies, "select_recipe_candidate"
        ).enums["organism_record_event_id"]

        self.assertEqual(
            sorted(offered),
            sorted(
                e["event_id"]
                for e in self.timeline
                if e["event_type"]
                in {
                    "TaxonResolutionRecorded",
                    "ReferenceAssembliesRecorded",
                    "ReferenceDescriptionRecorded",
                }
            ),
        )
        self.assertEqual(self._errors(), ())

    def test_a_record_the_run_did_not_keep_is_refused(self):
        plan = next(
            e
            for e in self.timeline
            if e["event_type"] == "ProposalRequirementsIdentified"
        )

        self.assertEqual(
            self._errors(organism_record_event_id=plan["event_id"]),
            ("organism_record_event_id_ungrounded",),
        )

    def test_words_the_record_does_not_hold_are_refused(self):
        for quote in ("Saccharomyces cerevisiae", "Homo sap"):
            with self.subTest(quote=quote):
                self.assertEqual(
                    self._errors(organism_quote=quote),
                    ("organism_quote_not_in_record",),
                )

    def test_a_non_diploid_judgment_is_the_models_own_choice(self):
        for judgment in ("likely_non_diploid", "unknown"):
            with self.subTest(judgment=judgment):
                self.assertEqual(self._errors(organism_ploidy=judgment), ())


class ExecutionPlanningOfferTests(unittest.TestCase):
    def test_initial_frontier_offers_only_the_recipe_lookup(self):
        dependencies = _Dependencies(_planning_timeline())
        self.assertEqual(_offered(dependencies), {"list_compatible_analysis_recipes"})

    def test_selected_recipe_frontier_offers_resource_lookup_only(self):
        timeline = _planning_timeline()
        initial = project_execution_planning(timeline)
        adapter = CapabilitySnapshotAdapter(policy())
        work, recipe_result, recipe_event = append_asserted_recipe_observation(
            timeline, adapter, "selected-frontier-recipe"
        )
        observed = project_execution_planning(timeline)
        candidate = recipe_result.candidates[0]
        selection_attempt = attempt_start(timeline, "selected-frontier-attempt")
        validated = append(
            timeline,
            "selected-frontier-validated",
            "ToolCallValidated",
            payload(
                ToolCallValidated(
                    attempt_id=selection_attempt,
                    tool_name="select_recipe_candidate",
                    tool_call_id="select-recipe",
                    arguments={
                        "work_item_id": work.work_item_id,
                        "candidate_id": candidate.candidate_id,
                        "basis_event_id": recipe_event["event_id"],
                    },
                    status="valid",
                    errors=(),
                )
            ),
        )
        append(
            timeline,
            "selected-frontier-selected",
            "PlanningCandidateSelected",
            payload(
                PlanningCandidateSelected(
                    attempt_id=selection_attempt,
                    kind="recipe",
                    work_item_id=work.work_item_id,
                    candidate_id=candidate.candidate_id,
                    basis_event_id=recipe_event["event_id"],
                    authority="agent_ranking",
                    authority_event_id=validated["event_id"],
                )
            ),
        )
        dependencies = _Dependencies(timeline)
        self.assertEqual(initial.status, "recipe_validation_pending")
        self.assertEqual(observed.status, "recipe_selection_pending")
        self.assertEqual(
            next(
                item for item in observed.requirements if item.kind == "analysis_recipe"
            ).state,
            "needs_selection",
        )
        self.assertEqual(
            project_execution_planning(timeline).status,
            "resource_validation_pending",
        )
        self.assertEqual(
            _offered(dependencies), {"lookup_compatible_resource_profiles"}
        )

    def test_planning_hitl_selection_offers_the_exact_researcher_recipe(self):
        timeline = _planning_timeline()
        adapter = CapabilitySnapshotAdapter(policy(recipes=2))
        work, result, basis = append_asserted_recipe_observation(
            timeline, adapter, "hitl-recipe"
        )
        projected = project_execution_planning(timeline)
        request_attempt = attempt_start(timeline, "hitl-recipe-request")
        requirement = next(
            item for item in projected.requirements if item.kind == "analysis_recipe"
        )
        candidates = tuple(
            PlanningHitlCandidate(
                candidate_id=candidate_disposition_id(
                    "recipe", work.work_item_id, item.candidate_id
                ),
                asserted_value=item.candidate_id,
                display_label=item.recipe_id,
                evidence_event_ids=(basis["event_id"],),
            )
            for item in result.candidates
        )
        request = PlanningHitlRequested(
            attempt_id=request_attempt,
            plan_event_id=projected.plan_event_id,
            plan_id=projected.plan_id,
            requirement_id=requirement.requirement_id,
            subject_key=requirement.subject_key,
            issue="Multiple compatible recipe policies.",
            question="Select exact recipe.",
            evidence_event_ids=(basis["event_id"],),
            candidates=candidates,
        )
        requested = append(
            timeline, "hitl-recipe-requested", request.event_type, payload(request)
        )
        responded = make_planning_hitl_responded(
            requested["event_id"],
            SelectPlanningHitlResponse(
                action="select", candidate_id=candidates[1].candidate_id
            ),
        )
        append(
            timeline, "hitl-recipe-responded", responded.event_type, payload(responded)
        )
        offer = execution_tool_values(
            _Dependencies(timeline), "select_recipe_candidate"
        )
        self.assertEqual(
            offer.constants["candidate_id"], result.candidates[1].candidate_id
        )
        self.assertEqual(offer.constants["basis_event_id"], basis["event_id"])


class ExecutionStepRefusalTests(unittest.TestCase):
    """An out-of-step call is told the current step and its Tool (#372 W5).

    The step's Tools are those whose own precondition holds: the same guards
    admission runs, so the text has no second source of truth.
    """

    def _refused(self, timeline, tool_name):
        with self.assertRaises(PreconditionUnmet) as refused:
            execution_tool_values(_Dependencies(timeline), tool_name)
        return refused.exception

    def _assert_step(self, timeline, step, tools):
        for name in sorted(set(EXECUTION_PLANNING_TOOLS) - set(tools)):
            with self.subTest(refused=name):
                unmet = self._refused(timeline, name)
                self.assertEqual(unmet.code, "execution_step_not_current")
                self.assertEqual(unmet.value, step)
                self.assertEqual(unmet.facts, {"tools": tools})

    def test_recipe_validation_names_the_recipe_lookup(self):
        self._assert_step(
            _planning_timeline(),
            "recipe validation",
            ("list_compatible_analysis_recipes",),
        )

    def test_recipe_selection_with_one_compatible_recipe_names_its_selection(self):
        timeline = _planning_timeline()
        append_asserted_recipe_observation(
            timeline, CapabilitySnapshotAdapter(policy()), "one-compatible-recipe"
        )
        self._assert_step(timeline, "recipe selection", ("select_recipe_candidate",))

    def test_recipe_selection_among_several_names_both_ways_to_choose(self):
        # Several compatible recipes, ranked by the planning Agent itself: it may
        # select one or ask the researcher, and the refusal names both.
        timeline = _planning_timeline()
        attempt_start(timeline, "several-compatible-attempt")
        append_asserted_recipe_observation(
            timeline,
            CapabilitySnapshotAdapter(policy(recipes=2)),
            "several-compatible-recipes",
        )
        self._assert_step(
            timeline,
            "recipe selection",
            ("request_recipe_selection", "select_recipe_candidate"),
        )

    def test_an_unoffered_recipe_candidate_is_told_the_offered_ones(self):
        # The refusal lists the candidates the schema enum already offers.
        from bio_neb.agentic_journey.action_policy import ProposedAgentAction
        from bio_neb.agentic_journey.tool_execution import function_tool_contracts
        from bio_neb.failure import sentence_with_facts
        from bio_neb.parabricks.execution_planning_action_policy import (
            execution_refusal_facts,
        )

        timeline = _planning_timeline()
        attempt_start(timeline, "several-compatible-attempt")
        append_asserted_recipe_observation(
            timeline,
            CapabilitySnapshotAdapter(policy(recipes=2)),
            "several-compatible-recipes",
        )
        dependencies = _Dependencies(timeline)
        offered = execution_tool_values(dependencies, "select_recipe_candidate").enums[
            "candidate_id"
        ]
        action = ProposedAgentAction(
            "select_recipe_candidate",
            "call",
            {"candidate_id": "not-offered", **_judged(timeline)},
        )
        (code,) = execution_tool_errors(dependencies, action)
        (explanation,) = {
            item.explanation
            for contract in function_tool_contracts("planning_agent")
            if contract.tool_name == "select_recipe_candidate"
            for item in contract.preconditions
            if item.code == code
        }
        self.assertEqual(len(offered), 2)
        self.assertEqual(
            sentence_with_facts(
                explanation, execution_refusal_facts(dependencies, action, code)
            ),
            "Choose one of the offered compatible recipe candidates: "
            + ", ".join(f"`{candidate}`" for candidate in offered)
            + ".",
        )

    def test_the_registry_sentence_names_the_current_steps_tool(self):
        from bio_neb.agentic_journey.tool_execution import function_tool_contracts
        from bio_neb.failure import sentence_with_facts

        # Every step row shares one slot and sentence; only its dependency order
        # (``met_by``) differs.
        ((slot, explanation),) = {
            (item.slot, item.explanation)
            for contract in function_tool_contracts("planning_agent")
            for item in contract.preconditions
            if item.code == "execution_step_not_current"
        }
        unmet = self._refused(_planning_timeline(), "select_recipe_candidate")
        self.assertEqual(slot, "current execution-planning step")
        self.assertEqual(
            sentence_with_facts(explanation, unmet.facts),
            "This Tool does not act on that step; the Tool for it is "
            "`list_compatible_analysis_recipes`.",
        )


class ExecutionPlanningAdmissionTests(unittest.TestCase):
    def test_execution_checkpoint_has_dedicated_exclusive_evidence_field(self):
        value = str(uuid.uuid4())
        checkpoint = SliceCheckpointReached(
            attempt_id=str(uuid.uuid4()),
            checkpoint="execution_planning_ready",
            input_set_id=value,
            proposal_plan_event_id=value,
            execution_planning_evidence_event_id=value,
        )
        self.assertEqual(checkpoint.execution_planning_evidence_event_id, value)
        with self.assertRaises(ValidationError):
            SliceCheckpointReached(
                attempt_id=str(uuid.uuid4()),
                checkpoint="execution_planning_ready",
                input_set_id=value,
                proposal_plan_event_id=value,
                reference_evidence_event_id=value,
                execution_planning_evidence_event_id=value,
            )

    POLICY = ContinuationPolicy(failed_turns=2)

    def _request(self, timeline):
        """The request the next reply answers: it opens that reply's count."""
        append(
            timeline,
            f"request-{uuid.uuid4()}",
            "ModelRequestRecorded",
            {"attempt_id": self.attempt_id},
        )

    def _spent(self, timeline):
        return failed_turns_spent(
            timeline, attempt_id=self.attempt_id, policy=self.POLICY
        )

    def _session(self, timeline):
        # The shared admission session with the planning assignment's repeat identity
        # and error; the execution policy grounds the action. The limit of two
        # failed turns in a row is this test's input, not a product configuration.
        dependencies = _Dependencies(timeline)
        activity = FakeActivity(dependencies.journey)
        self.attempt_id = str(uuid.uuid4())
        session = ActionAdmissionSession(
            wording=CorrectionWording(),
            attempt_id=self.attempt_id,
            parent_activity=activity,
            failure_source=AttemptFailureSource(),
            eligible_tool_names=frozenset(EXECUTION_PLANNING_TOOLS),
            action_validator=lambda action: execution_tool_errors(dependencies, action),
            action_fingerprint=call_identity,
            admitted_repeat_error="planning_action_repeated",
            continuation=self.POLICY,
            events=lambda: dependencies.journey.timeline,
        )
        offer = execution_tool_values(dependencies, "list_compatible_analysis_recipes")
        assert offer is not None
        action = ProposedAgentAction(
            tool_name="list_compatible_analysis_recipes",
            tool_call_id="planning-call",
            arguments=offer.constants,
        )
        return session, dependencies.journey.timeline, action

    def test_repeat_identity_includes_every_work_and_basis_sensitive_argument(self):
        _, _, action = self._session(_planning_timeline())
        fingerprint = call_identity(action)
        self.assertIsNotNone(fingerprint)
        self.assertEqual(
            fingerprint,
            call_identity(
                ProposedAgentAction(
                    tool_name=action.tool_name,
                    tool_call_id="another-call",
                    arguments=dict(action.arguments),
                )
            ),
        )
        for name, value in action.arguments.items():
            changed = dict(action.arguments)
            changed[name] = "changed" if value is not None else "asserted"
            self.assertNotEqual(
                fingerprint,
                call_identity(
                    ProposedAgentAction(
                        tool_name=action.tool_name,
                        tool_call_id=action.tool_call_id,
                        arguments=changed,
                    )
                ),
                name,
            )

    def test_tool_module_has_no_local_offer_or_disposition_policy(self):
        source = inspect.getsource(execution_planning_module)
        self.assertNotIn("def _offer(", source)
        self.assertNotIn("def _selected_disposition(", source)
        # No offer or disposition function is defined in the Tool module itself.
        local = {
            name
            for name, value in vars(execution_planning_module).items()
            if inspect.isfunction(value)
            and value.__module__ == execution_planning_module.__name__
        }
        self.assertEqual(
            {name for name in local if "offer" in name or "disposition" in name},
            set(),
        )

    def test_admitted_repeat_is_corrected_then_its_second_failed_turn_stops(self):
        session, timeline, action = self._session(_planning_timeline())
        self._request(timeline)
        session.admit_turn(
            response_event_id=str(uuid.uuid4()),
            actions=(action,),
            exposed_tool_names=frozenset({action.tool_name}),
        )
        repeated = ProposedAgentAction(
            tool_name=action.tool_name,
            tool_call_id="planning-repeat",
            arguments=action.arguments,
        )
        for _ in range(2):
            self.assertFalse(self._spent(timeline))
            self._request(timeline)
            # Admission records each refusal and its correction; it never stops.
            with self.assertRaises(ModelRetry):
                session.admit_turn(
                    response_event_id=str(uuid.uuid4()),
                    actions=(repeated,),
                    exposed_tool_names=frozenset({action.tool_name}),
                )
        # The second failed turn in a row is the limit: the next request's check
        # stops the Attempt, so its correction is never sent.
        self.assertTrue(self._spent(timeline))
        validations = [
            event for event in timeline if event["event_type"] == "ToolCallValidated"
        ]
        self.assertEqual(
            [event["payload"]["status"] for event in validations],
            ["valid", "invalid", "invalid"],
        )
        self.assertEqual(
            validations[-1]["payload"]["errors"],
            ["planning_action_repeated"],
        )
        # Both repeats are the same refused call under one key; each names its
        # reply's place in the failed turns in a row.
        refusals = [
            (
                event["payload"]["correction_key"].split(" @ ")[0],
                event["payload"]["correction_count"],
            )
            for event in timeline
            if event["event_type"] == "AgentActionTurnDispositionRecorded"
            and event["payload"]["disposition"] == "singleton_refused"
        ]
        self.assertEqual(
            refusals,
            [
                (f"{action.tool_name}: planning_action_repeated", 1),
                (f"{action.tool_name}: planning_action_repeated", 2),
            ],
        )

    def test_a_mixed_reply_runs_its_offered_call_and_failed_turns_in_a_row_stop(
        self,
    ):
        session, timeline, action = self._session(_planning_timeline())
        offered = frozenset({action.tool_name})

        def route(*actions):
            self._request(timeline)
            reply, turn = received_reply(actions)
            return reply.response_event_id, session.route_reply(reply, turn, offered)

        off_menu = ProposedAgentAction(
            tool_name="not_execution_planning", tool_call_id="off-menu", arguments={}
        )
        response_id, routing = route(action, off_menu)
        # The unoffered call is refused on its own; the offered call still runs.
        self.assertEqual(
            [item.tool_call_id for item in routing.actions], [action.tool_call_id]
        )
        self.assertIn("(tool_name_ineligible)", routing.note)
        self.assertIn("failed turn 1 of 2 in a row", routing.note)
        session.admit_turn(response_id, routing.actions, offered, routed=True)
        session.admit_call(action.tool_call_id)
        invalid = ProposedAgentAction(
            tool_name=action.tool_name,
            tool_call_id="grounding-drift",
            arguments={**action.arguments, "input_set_id": str(uuid.uuid4())},
        )
        self._request(timeline)
        with self.assertRaises(ModelRetry):
            # The request bound the offer; a contradicting sent value is refused.
            # The reply before it ran a call, so this is the first failed turn.
            session.admit_turn(
                response_event_id=str(uuid.uuid4()),
                actions=(invalid,),
                exposed_tool_names=offered,
                bound_arguments={action.tool_name: dict(action.arguments)},
            )
        # A second reply naming an unoffered Tool keeps its offered call, which
        # repeats the call that already ran: nothing of the reply works, so it is
        # the second failed turn in a row and the Attempt stops at its limit.
        response_id, routing = route(
            replace(action, tool_call_id="planning-again"),
            replace(off_menu, tool_call_id="off-menu-again"),
        )
        self.assertIn("failed turn 2 of 2 in a row", routing.note)
        session.admit_turn(response_id, routing.actions, offered, routed=True)
        with self.assertRaises(ToolFailed):
            session.admit_call("planning-again")
        self.assertTrue(self._spent(timeline))
        turns = [
            e["payload"]
            for e in timeline
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
        ]
        self.assertEqual(
            [(turn["disposition"], turn.get("correction_count")) for turn in turns],
            [
                ("multi_call_admitted", None),
                ("singleton_refused", 1),
                ("multi_call_refused", 2),
            ],
        )
        self.assertIn("input_set_id:bound_value_mismatch", turns[1]["feedback"])
        self.assertIn("planning_action_repeated", turns[2]["correction_key"])
        routed = [
            e["payload"]["faults"]
            for e in timeline
            if e["event_type"] == "AgentTurnRouted"
        ]
        self.assertEqual(
            routed,
            [
                [{"key": "unoffered_calls", "count": 1}],
                [{"key": "unoffered_calls", "count": 2}],
            ],
        )
        self.assertNotIn(
            "AgentActionReviewRequested", {e["event_type"] for e in timeline}
        )


if __name__ == "__main__":
    unittest.main()
