"""One declaration per Agent role over one global role default (#446, ADR 0040)."""

import dataclasses
import unittest
from typing import get_args

from bio_neb.agentic_journey.role_declaration import (
    ROLE_DEFAULT,
    ROLES,
    AgentRole,
    AttemptClosure,
    RoleDeclaration,
    role,
)

# Every field a role must state: none has a default.
_FIELDS = (
    "name",
    "jobs",
    "stages",
    "output_tools",
    "closure",
    "member_scope",
    "member_reject_effect",
    "dismiss_effect",
    "settles_member_stops",
    "instruction_version",
)


def _table_reader(**changed):
    values = {
        "name": "table_reader",
        "jobs": {"vcf_table_reading": "job"},
        "stages": ("vcf_table_reading",),
        "output_tools": frozenset({"record_item_readings"}),
        "closure": AttemptClosure(
            "investigation_table_reading_attempt",
            "InvestigationTableChunkRead",
            "InvestigationTableChunkFailed",
        ),
        "member_scope": "role",
        "member_reject_effect": None,
        "dismiss_effect": "Dismiss this chunk.",
        "settles_member_stops": False,
        "instruction_version": "v4",
    }
    return RoleDeclaration(**{**values, **changed})


class RoleDeclarationTests(unittest.TestCase):
    def test_every_role_has_one_declaration_and_nothing_else_is_declared(self):
        self.assertEqual(set(ROLES), set(get_args(AgentRole)))
        for name, declaration in ROLES.items():
            self.assertEqual(declaration.name, name)
            self.assertIs(role(name), declaration)
        with self.assertRaisesRegex(ValueError, "^not an Agent role: planner$"):
            role("planner")

    def test_a_declaration_omitting_any_field_is_refused(self):
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(RoleDeclaration)), _FIELDS
        )
        declared = _table_reader()
        complete = {name: getattr(declared, name) for name in _FIELDS}
        for name in _FIELDS:
            with self.subTest(field=name), self.assertRaises(TypeError):
                RoleDeclaration(**{k: v for k, v in complete.items() if k != name})

    def test_a_declaration_states_a_job_for_each_stage_and_its_scope_effect(self):
        for changed, message in (
            ({"stages": ()}, "declares no stage"),
            ({"jobs": {}}, "names no job block for stage vcf_table_reading"),
            ({"jobs": {"vcf_table_reading": ""}}, "names no job block"),
            (
                {"jobs": {"vcf_table_reading": "job", "other": "job"}},
                "names a job for a stage it has not",
            ),
            ({"member_scope": "member"}, "member scope states its reject effect"),
            (
                {"member_reject_effect": "Stop this chunk."},
                "role scope states no member reject effect",
            ),
            ({"member_scope": "chunk"}, "scope is member or role"),
            ({"name": "planner"}, "not an Agent role: planner"),
        ):
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(ValueError, message),
            ):
                _table_reader(**changed)

    def test_the_declarations_state_each_roles_differences(self):
        self.assertEqual(
            {
                name: (
                    item.stages,
                    tuple(sorted(item.output_tools)),
                    item.closure.stage,
                    item.member_scope,
                )
                for name, item in ROLES.items()
            },
            {
                "planning_agent": (
                    (
                        "input_correction",
                        "target_answer",
                        "proposal_requirements",
                        # A plan that cites a paper settles it here.
                        "paper",
                        "reference",
                        "companions",
                        "execution_planning",
                    ),
                    (),
                    "planning_attempt",
                    "role",
                ),
                "investigation_planner": (
                    ("source_reading",),
                    ("compile_proposal",),
                    "investigation_planning_attempt",
                    "member",
                ),
                "table_reader": (
                    ("vcf_table_reading",),
                    ("record_item_readings",),
                    "investigation_table_reading_attempt",
                    "role",
                ),
                "report_explainer": (
                    ("report_explanation", "report_combination"),
                    ("record_combined_answer", "record_explanation"),
                    "report_explanation_attempt",
                    "member",
                ),
                # ADR 0039: a reading item is a member, so its stop holds only
                # itself.
                "paper_reader": (
                    ("paper_reading",),
                    ("answer_reading_item", "propose_reading_areas"),
                    "paper_reading_attempt",
                    "member",
                ),
            },
        )

    def test_one_global_default_holds_the_per_attempt_budgets_and_the_stops(self):
        # One failure rule for every role: the fourth failed turn in a row stops
        # the Attempt.
        self.assertEqual(
            (ROLE_DEFAULT.model_requests, ROLE_DEFAULT.failed_turns), (200, 4)
        )
        # A refused output is one more failed turn: its stop is the Agent work stop.
        self.assertEqual(
            ROLE_DEFAULT.stops,
            ("agent_work", "effort", "operational", "model_availability"),
        )
        # No role is excluded from the rule, so no declaration names an entitlement.
        for item in ROLES.values():
            self.assertFalse(hasattr(item, "output_repair_allowed"))
        # The per-kind allowances are retired: no default or declaration holds one.
        for allowance in (
            "same_call_correction",
            "repeated_call",
            "truncation_retry",
            "output_repair",
        ):
            self.assertFalse(hasattr(ROLE_DEFAULT, allowance))
            self.assertNotIn(allowance, _FIELDS)


# The job sentences the three roles had no block for: the table reader's moved
# from its Python task text unchanged, the paper reader's is new (approved as
# written), and the explainer's rewritten ones.
_JOBS = {
    ("table_reader", "job"): (
        "Read every row of this chunk of the Run's VCF table and record one reading "
        "per row."
    ),
    # The explainer's two job sentences for the readable report: a member says
    # why its request ended so, the combination writes the conclusion; code
    # still fills each target's claim. They name the request and the position
    # searched, never rows or items.
    ("report_explainer", "job"): (
        "Explain one request of this approved Grounded report to the researcher, "
        "from this request's own targets: say why it ended as its targets record, "
        "and keep each target's claim and, for a position that carries one, its "
        "genotype, as the targets show them. Cover each qualification once and cite "
        "the evidence index. position_evidence keeps each searched position's "
        "outcome apart from its nested observation; its events are the ones to read "
        "with search_journey."
    ),
    ("report_explainer", "combination_job"): (
        "Write the conclusion of this approved Grounded report over every request. "
        "Each entry of requests is one request's member: its claim per target and "
        "why it ended so, or why it was not explained. The conclusion covers every "
        "request from those claims. Cover each qualification once and cite the "
        "evidence index."
    ),
    ("paper_reader", "job"): (
        "Read this paper for the researcher's question, one reading item at a time: "
        "the paper item proposes the areas to read with propose_reading_areas, and "
        "an area or id item answers with answer_reading_item, citing each value by "
        "the span id that writes it. You do not decide whether a finding is true: "
        "code checks every cited value against the paper's own words, so write "
        "only what the paper states."
    ),
}


class RoleJobTests(unittest.TestCase):
    def test_every_declared_job_is_a_message_of_its_roles_instructions(self):
        from bio_neb.agentic_journey.role_instructions import _FORMS, role_instructions

        for name, item in ROLES.items():
            for form in _FORMS:
                messages = dict(role_instructions(name, form=form).messages)
                for stage, block in item.jobs.items():
                    with self.subTest(role=name, form=form, stage=stage):
                        self.assertTrue(messages.get(block, "").strip())
                        expected = _JOBS.get((name, block))
                        if expected is not None:
                            self.assertEqual(messages[block], expected)


# Each role's effective invocation policy on the default entry, recorded at the
# predecessor head 09c60355 before the role default was built.
_POLICY_IDS_AT_09C60355 = {
    "planning_agent": "invocation-policy:b4832ffe027089b8677ed8277a10b39e8f8f9c2b3c355e43d9d3fd4f0f97ced8",
    "investigation_planner": "invocation-policy:6811a56f2eb1b7b98cc1b7799d7d9cf6db4d178512344bb0fd177ab2c3fd5c22",
    "table_reader": "invocation-policy:7848a4472889004143217121b3a5365a24069d320863b42883069160f4edd6a2",
    "report_explainer": "invocation-policy:d387a4738c44daeb5c5f03b2cd2675393e5f43b73714d6f09e5ecaf000420d8c",
    "paper_reader": "invocation-policy:c9bd83b1cca6a4d02d990fa82a4e5e552dbb3e3cb37c3cc2d06035218fa9e52e",
}


def _policies(document=None):
    from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
    from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration

    selections = RuntimeConfiguration.model_validate(
        {"contract_version": "1.0", **(document or {})}
    ).runtime_selections()
    return selections, {
        name: selections.agent_invocation_policy(name, AgentInvocationOverrides())
        for name in ROLES
    }


# The same policies under the one failure rule that replaced the allowances.
_POLICY_IDS_AT_447 = {
    "planning_agent": "invocation-policy:cb7fa44b8ad810acb3379f4ca8b94643baa6b82f75123057f48773a6697971a9",
    "investigation_planner": "invocation-policy:395d1b9bb746f7a8cb435283892bb08a2ae0ca7edbb9f7d9820d2a902a23bc99",
    "table_reader": "invocation-policy:b81927cfac11b9228f47eef962509716ea2440be3db1686bda32e3e64667c3eb",
    "report_explainer": "invocation-policy:2a497016c792ce55cabedfcceddf8d850a6adc8a65811ace889004fbc41c8a02",
    "paper_reader": "invocation-policy:02df833a7c441c98ca709094317d92a6dce35335cff57fbca93753b99473db54",
}


# The target_answer planning stage (a researcher's target answer goes back to
# planning's target reading) has its stage settings in planning's policy.
_POLICY_IDS_AT_462 = {
    **_POLICY_IDS_AT_447,
    "planning_agent": "invocation-policy:7c186f530b322777e1df57a7b0205b559ad31d85da7aa555217b0bd15fa0c876",
}


# The one Tool choice is recorded once, globally: each stage's choice reads from
# the global scope, not the role's copy.
_POLICY_IDS_AT_477 = {
    "planning_agent": "invocation-policy:cf9d28d14cc725bbc7f7d6f321350967c4513ce364617b802ed257c3c4aff218",
    "investigation_planner": "invocation-policy:7437a09344ef03663bddf2372552d09a2de8d0e29e36192ea612df3a576b9d79",
    "table_reader": "invocation-policy:bc06dd929e63e691ce3aa9ac9cef8cd4a64549d77dd7c4331bc122ab288ee1fe",
    "report_explainer": "invocation-policy:7011868cf68be2890ec8b5ea0b5b91fb998ce7a0a03f73bc9c8b3dc1c2f32c02",
    "paper_reader": "invocation-policy:b43bcb91c1061d2f95356b6ef81fe8ec2f345ec24ade37409abde3ccc62e0af6",
}


# The paper planning stage (a plan that cites a paper settles it before the
# reference work) has its stage settings in planning's policy.
_POLICY_IDS_AT_420 = {
    **_POLICY_IDS_AT_477,
    "planning_agent": "invocation-policy:a63119beec8f63cd3a862daaf1ec31ad13e5e0c69ebc77ecd1c99841fdc43600",
}


def _before_420(policy):
    """The policy's values with #420's paper stage put back."""
    values = policy.model_dump(mode="json", exclude_none=True)
    values.get("stage_settings", {}).pop("paper", None)
    return values


def _as_at_477(policy):
    from bio_neb.journey import canonical_json_digest

    return "invocation-policy:" + canonical_json_digest(_before_420(policy))


def _before_477(policy):
    """The policy's values with the per-role Tool-choice copies put back."""
    values = _before_420(policy)
    for entry in values.get("stage_settings", {}).values():
        sources = entry.get("settings_sources", {})
        if sources.get("tool_choice") == "global":
            sources["tool_choice"] = "role"
    return values


def _as_at_462(policy):
    from bio_neb.journey import canonical_json_digest

    return "invocation-policy:" + canonical_json_digest(_before_477(policy))


def _before_462(policy):
    """The policy's values with #462's target_answer stage put back."""
    values = _before_477(policy)
    values.get("stage_settings", {}).pop("target_answer", None)
    return values


def _as_at_447(policy):
    from bio_neb.journey import canonical_json_digest

    return "invocation-policy:" + canonical_json_digest(_before_462(policy))


def _as_at_09c60355(name, policy):
    """The policy's identity with #447 B's three changes put back: the
    allowances, the effort-notice threshold (8) and the exhaustion keys."""
    from bio_neb.journey import canonical_json_digest

    values = _before_462(policy)
    values["continuation"] = {
        "output_repair": 0 if name == "planning_agent" else 4,
        "same_call_correction": 10,
        "repeated_call": 6,
        "truncation_retry": 2,
    }
    values["injection"]["patterns"]["progress_notice"]["after_requests"] = 8
    values["exhaustion"] = {
        "model_availability": "review",
        "output_repair": "review",
        "singleton_action": "review",
    }
    return "invocation-policy:" + canonical_json_digest(values)


def _budgets(policies):
    return {
        name: (policy.model_requests, policy.continuation.model_dump(mode="json"))
        for name, policy in policies.items()
    }


class RoleDefaultResolutionTests(unittest.TestCase):
    def test_every_role_on_the_default_entry_keeps_its_effective_policy(self):
        selections, policies = _policies()
        self.assertEqual(
            {name: policy.policy_id for name, policy in policies.items()},
            _POLICY_IDS_AT_420,
        )
        # With the paper stage taken out again, every role's identity is the one
        # recorded before it; nothing else changed.
        self.assertEqual(
            {name: _as_at_477(policy) for name, policy in policies.items()},
            _POLICY_IDS_AT_477,
        )
        # With the per-role Tool-choice copies put back, every role's identity
        # is the one recorded before the global Tool choice; nothing else
        # changed.
        self.assertEqual(
            {name: _as_at_462(policy) for name, policy in policies.items()},
            _POLICY_IDS_AT_462,
        )
        # With the target_answer stage taken out again, every role's identity is
        # the one recorded before it; nothing else changed.
        self.assertEqual(
            {name: _as_at_447(policy) for name, policy in policies.items()},
            _POLICY_IDS_AT_447,
        )
        # With the per-role allowances put back in place of the one failure
        # rule, every role's identity is the one recorded before it; nothing
        # else changed.
        self.assertEqual(
            {name: _as_at_09c60355(name, policy) for name, policy in policies.items()},
            _POLICY_IDS_AT_09C60355,
        )
        # One failure rule for every role, Tool-driven ones too.
        self.assertEqual(
            _budgets(policies),
            {name: (200, {"failed_turns": 4}) for name in ROLES},
        )
        # Registration records the one override path and no per-role wiring; an
        # unset failure rule records no key.
        registered = selections.model_dump(mode="json")
        self.assertEqual(registered["agent_invocation"]["model_requests"], 200)
        self.assertNotIn("failed_turns", registered["agent_invocation"])
        for retired in (
            "investigation_continuation",
            "role_output_repair",
            "agent_continuation",
        ):
            self.assertNotIn(retired, registered)

    def test_a_runtime_file_overrides_the_global_default_and_one_role(self):
        _, policies = _policies(
            {
                "agent_invocation": {
                    "model_requests": 150,
                    "failed_turns": 2,
                    "scopes": {
                        "role": {
                            "table_reader": {"model_requests": 50, "failed_turns": 6}
                        }
                    },
                }
            }
        )
        self.assertEqual(
            {
                name: (requests, continuation["failed_turns"])
                for name, (requests, continuation) in _budgets(policies).items()
            },
            {
                "planning_agent": (150, 2),
                "investigation_planner": (150, 2),
                "table_reader": (50, 6),
                "report_explainer": (150, 2),
                "paper_reader": (150, 2),
            },
        )

    def test_a_failed_turns_number_is_refused_where_no_scope_can_hold_one(self):
        from pydantic import ValidationError

        tool_unbuilt = "Value error, this scope cannot set it until its seam is built"
        for document, refused in (
            # A request does not know which Tool the model will call.
            (
                {
                    "agent_invocation": {
                        "scopes": {"tool": {"search_journey": {"failed_turns": 2}}}
                    }
                },
                (
                    (
                        "agent_invocation",
                        "scopes",
                        "tool",
                        "search_journey",
                        "failed_turns",
                    ),
                    "value_error",
                    tool_unbuilt,
                ),
            ),
            (
                {"agent_invocation": {"failed_turns": 0}},
                (
                    ("agent_invocation", "failed_turns"),
                    "greater_than_equal",
                    "Input should be greater than or equal to 1",
                ),
            ),
            # The retired per-kind allowances are not read, at any scope.
            *(
                (
                    {"agent_invocation": {name: 1}},
                    (
                        ("agent_invocation", name),
                        "extra_forbidden",
                        "Extra inputs are not permitted",
                    ),
                )
                for name in (
                    "output_repair",
                    "same_call_correction",
                    "repeated_call",
                    "truncation_retry",
                )
            ),
            (
                {
                    "agent_invocation": {
                        "scopes": {"role": {"table_reader": {"output_repair": 1}}}
                    }
                },
                (
                    (
                        "agent_invocation",
                        "scopes",
                        "role",
                        "table_reader",
                        "output_repair",
                    ),
                    "extra_forbidden",
                    "Extra inputs are not permitted",
                ),
            ),
            (
                {
                    "agent_invocation": {
                        "scopes": {"stage": {"reference": {"output_repair": 1}}}
                    }
                },
                (
                    (
                        "agent_invocation",
                        "scopes",
                        "stage",
                        "reference",
                        "output_repair",
                    ),
                    "extra_forbidden",
                    "Extra inputs are not permitted",
                ),
            ),
            # The retired per-role wiring is not read.
            *(
                (
                    {name: value},
                    ((name,), "extra_forbidden", "Extra inputs are not permitted"),
                )
                for name, value in (
                    ("investigation_continuation", {"output_repair": 4}),
                    ("role_output_repair", {"table_reader": {"output_repair": 6}}),
                    ("agent_continuation", {"planning_agent": {"output_repair": 0}}),
                )
            ),
        ):
            with self.subTest(document=document):
                with self.assertRaises(ValidationError) as raised:
                    _policies(document)
                self.assertEqual(
                    [
                        (error["loc"], error["type"], error["msg"])
                        for error in raised.exception.errors()
                    ],
                    [refused],
                )
        # A stage scope and the Tool-driven role's own scope now hold one.
        _, policies = _policies(
            {
                "agent_invocation": {
                    "scopes": {
                        "role": {"planning_agent": {"failed_turns": 3}},
                        "stage": {"reference": {"failed_turns": 6}},
                    }
                }
            }
        )
        planning = policies["planning_agent"].continuation
        self.assertEqual(
            (planning.limit("reference"), planning.limit("input_correction")), (6, 3)
        )
        # Only the role owning the stage records it.
        self.assertEqual(
            {
                name: policy.continuation.stage_failed_turns
                for name, policy in policies.items()
                if policy.continuation.stage_failed_turns
            },
            {"planning_agent": {"reference": 6}},
        )

    def test_a_role_that_selects_another_model_takes_the_same_failure_rule(self):
        # The per-model allowances are retired: no entry states its own number,
        # so the role that selects Qwen takes the role default's like every
        # other role.
        selections, policies = _policies(
            {
                "model_roster": {"qwen": {"model": "Qwen/Qwen3.5-397B-A17B"}},
                "agent_invocation": {
                    "scopes": {"role": {"table_reader": {"model_reference": "qwen"}}}
                },
            }
        )
        self.assertEqual(
            {name: budget[1] for name, budget in _budgets(policies).items()},
            {name: {"failed_turns": 4} for name in ROLES},
        )
        self.assertEqual(policies["table_reader"].model_reference, "qwen")
        # A file naming a role's number keeps its own.
        _, named = _policies(
            {
                "model_roster": {"qwen": {"model": "Qwen/Qwen3.5-397B-A17B"}},
                "agent_invocation": {
                    "scopes": {
                        "role": {
                            "table_reader": {
                                "model_reference": "qwen",
                                "failed_turns": 7,
                            }
                        }
                    }
                },
            }
        )
        self.assertEqual(_budgets(named)["table_reader"][1], {"failed_turns": 7})


if __name__ == "__main__":
    unittest.main()
