"""Scoped bridge proof; public planning/retry tests own the complete Journey path.

These minimal semantic graphs exercise the shared binding boundary, not domain
fact validation, contact evidence or transcript/policy validation.
"""

import copy
import unittest

from bio_neb.agentic_journey.continuation import validate_attempt_continuation

KINDS = (
    ("availability", "ModelAvailability", "retry"),
    ("effort", "AgentEffort", "extend"),
    ("work", "AgentWork", "continue"),
    ("operational", "AgentFailure", "retry"),
)


def bridge(kind, *, answer_first):
    field, owner, action = kind
    events = []

    def append(identity, event_type, parents, **payload):
        event = dict(
            event_id=identity,
            event_type=event_type,
            parent_ids=parents,
            payload=payload,
        )
        events.append(event)
        return event

    append(
        "original",
        "AttemptStarted",
        [],
        attempt_id="stopped",
        agent_role="planning_agent",
        input_set_id="inputs",
    )
    append(
        "review",
        owner + "ReviewRequested",
        ["original"],
        attempt_id="stopped",
        agent_role="planning_agent",
    )
    append("closed", "activity_succeeded", ["review"])
    append(
        "output", "PlanningHitlRequested", ["closed"], owner="requirement_resolution"
    )
    if answer_first:
        append(
            "answer",
            owner + "ReviewResponded",
            ["review", "output"],
            request_event_id="review",
            response={"action": action},
        )
        append(
            "output-answer",
            "PlanningHitlResponded",
            ["output", "answer"],
            request_event_id="output",
        )
    else:
        append(
            "output-answer",
            "PlanningHitlResponded",
            ["output"],
            request_event_id="output",
        )
        append(
            "answer",
            owner + "ReviewResponded",
            ["review", "output-answer"],
            request_event_id="review",
            response={"action": action},
        )
    append("pass-closed", "PassClosed", [events[-1]["event_id"]])
    append("pass-started", "PassStarted", ["pass-closed"])
    append("lifecycle", "activity_started", ["pass-closed"])
    started = append(
        "resumed",
        "AttemptStarted",
        ["lifecycle"],
        attempt_id="resumed",
        agent_role="planning_agent",
        input_set_id="inputs",
        **{f"continuation_{field}_review_event_id": "answer"},
    )
    return events, started


class AttemptContinuationTests(unittest.TestCase):
    def test_independent_answer_orders_and_passes_preserve_one_consumed_direction(self):
        for kind in KINDS:
            for answer_first in (False, True):
                with self.subTest(kind=kind[0], answer_first=answer_first):
                    events, started = bridge(kind, answer_first=answer_first)
                    binding = validate_attempt_continuation(events, started)
                    self.assertEqual(binding.answer["event_id"], "answer")
                    self.assertEqual(binding.request["event_id"], "review")
                    self.assertEqual(
                        binding.previous_attempt["payload"]["attempt_id"], "stopped"
                    )
                    self.assertEqual(binding.causal_head, "pass-closed")

    def test_a_check_its_owner_holds_equals_the_fresh_check_at_every_later_prefix(
        self,
    ):
        # A reading owner keeps each start's check, keyed by the start; it is fixed
        # by the prefix up to that start, so later events never change it, and a
        # refusal is never held.
        for kind in KINDS:
            for answer_first in (False, True):
                with self.subTest(kind=kind[0], answer_first=answer_first):
                    events, started = bridge(kind, answer_first=answer_first)
                    held = {}

                    def once(key, check, held=held):
                        if key not in held:
                            held[key] = check()
                        return held[key]

                    first = validate_attempt_continuation(
                        events, started, checked_once=once
                    )
                    self.assertEqual(first.answer["event_id"], "answer")
                    self.assertEqual(list(held), [("attempt_continuation", "resumed")])
                    later = [
                        *events,
                        dict(
                            event_id="closed-again",
                            event_type="PassClosed",
                            parent_ids=["resumed"],
                            payload={},
                        ),
                    ]
                    for prefix in (events, later):
                        self.assertEqual(
                            validate_attempt_continuation(
                                prefix, started, checked_once=once
                            ),
                            validate_attempt_continuation(prefix, started),
                        )
                    hostile, refused = bridge(kind, answer_first=answer_first)
                    next(e for e in hostile if e["event_id"] == "lifecycle")[
                        "parent_ids"
                    ] = ["answer"]
                    refusals = {}

                    def refusing_once(key, check, held=refusals):
                        if key not in held:
                            held[key] = check()
                        return held[key]

                    for checked_once in (refusing_once, None):
                        with self.assertRaisesRegex(ValueError, "current causal head"):
                            validate_attempt_continuation(
                                hostile, refused, checked_once=checked_once
                            )
                    self.assertEqual(refusals, {})

    def test_wrong_parent_role_request_field_and_missing_binding_refuse(self):
        for kind in KINDS:
            for change, message in (
                ("answer_parent", "exact request or causal head"),
                ("lifecycle_parent", "current causal head"),
                ("role", "Agent role"),
                ("request", "exact request kind"),
                ("field", "typed direction"),
                ("two_fields", "two review directions"),
                ("missing", "lost its exact direction binding"),
            ):
                with self.subTest(kind=kind[0], change=change):
                    events, started = bridge(kind, answer_first=False)
                    by_id = {event["event_id"]: event for event in events}
                    field = f"continuation_{kind[0]}_review_event_id"
                    other = (
                        "continuation_work_review_event_id"
                        if kind[0] != "work"
                        else "continuation_effort_review_event_id"
                    )
                    if change == "answer_parent":
                        by_id["answer"]["parent_ids"] = ["review"]
                    elif change == "lifecycle_parent":
                        by_id["lifecycle"]["parent_ids"] = ["answer"]
                    elif change == "role":
                        started["payload"]["agent_role"] = "investigation_planner"
                    elif change == "request":
                        by_id["answer"]["payload"]["request_event_id"] = "output"
                    elif change in {"field", "two_fields"}:
                        started["payload"][other] = "answer"
                        if change == "field":
                            del started["payload"][field]
                    else:
                        del started["payload"][field]
                        started["payload"].pop("continuation_request_event_id", None)
                    with self.assertRaisesRegex(ValueError, message):
                        validate_attempt_continuation(events, started)

    def test_a_consumed_answer_cannot_authorize_another_attempt(self):
        for kind in KINDS:
            with self.subTest(kind=kind[0]):
                events, started = bridge(kind, answer_first=True)
                second_lifecycle = dict(
                    event_id="second-lifecycle",
                    event_type="activity_started",
                    parent_ids=["resumed"],
                    payload={},
                )
                second = copy.deepcopy(started)
                second["event_id"] = "second"
                second["parent_ids"] = ["second-lifecycle"]
                second["payload"]["attempt_id"] = "second"
                events.extend((second_lifecycle, second))
                with self.assertRaisesRegex(ValueError, "unused authority"):
                    validate_attempt_continuation(events, second)
