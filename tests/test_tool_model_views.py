"""What the model is shown before it calls and after (#478, owner #476 alignment).

Each request names the Tools callable now and those waiting, with the reason a call
of each would be refused now, from the one source the refusal reads (G1.1 B); every
Tool row declares the model view its return is built from (G1.3 A).
"""

import unittest
from types import SimpleNamespace

from bio_neb.agentic_journey import tool_execution
from bio_neb.agentic_journey.action_policy import PreconditionUnmet, ProposedAgentAction
from bio_neb.agentic_journey.assignment_context import SEARCH_TOOL_NAME
from bio_neb.agentic_journey.invocation import _StagedTools
from bio_neb.agentic_journey.role_declaration import role
from bio_neb.agentic_journey.role_instructions import role_instructions
from bio_neb.agentic_journey.tool_execution import (
    FunctionToolContract,
    ModelView,
    function_tool_contracts,
    stage_tool_names,
    tool_model_view,
)

# The assembled roles.
ROLES = ("planning_agent", "investigation_planner", "table_reader", "report_explainer")
_GENERIC = {"researcher_question_opened", "tool_name_unavailable"}


class ToolsNowTests(unittest.TestCase):
    def test_every_role_s_waiting_tools_read_the_refusal_s_own_sentence(self):
        for agent_role in ROLES:
            contracts = {c.tool_name: c for c in function_tool_contracts(agent_role)}
            for stage in role(agent_role).stages:
                tools = {
                    name: None
                    for name in stage_tool_names(agent_role, stage)
                    if name != SEARCH_TOOL_NAME
                }
                # Each stage Tool with a declared reason of its own waits on it.
                reasons = {
                    name: next(
                        (
                            item.code
                            for item in contracts[name].preconditions
                            # A sentence with typed facts needs its role's
                            # computed facts; these name none.
                            if item.code not in _GENERIC and "{" not in item.explanation
                        ),
                        None,
                    )
                    for name in tools
                }

                def bound(_deps, name, reasons=reasons):
                    if reasons[name] is not None:
                        raise PreconditionUnmet(reasons[name])
                    return SimpleNamespace(constants={}, enums={})

                staged = _StagedTools(
                    agent_role,
                    tools,
                    lambda _deps, stage=stage: stage,
                    bound,
                    None,
                    role_instructions(agent_role),
                )
                output = frozenset(role(agent_role).output_tools)
                always = frozenset({SEARCH_TOOL_NAME}) | output
                with self.subTest(role=agent_role, stage=stage):
                    view = staged.tools_now(None, stage, always)
                    waiting = {item["tool"]: item["reason"] for item in view["waiting"]}
                    self.assertEqual(
                        set(waiting),
                        {name for name, code in reasons.items() if code is not None},
                    )
                    self.assertEqual(
                        sorted([*view["callable_now"], *waiting]),
                        sorted(frozenset(tools) | always),
                    )
                    for name, reason in waiting.items():
                        # The sentence a premature call of it is refused with.
                        action = ProposedAgentAction(name, "premature", {})
                        staged.precondition(None, action)
                        self.assertEqual(
                            reason, staged.explain(None, action, reasons[name])
                        )
                        self.assertNotEqual(reason, reasons[name])


class ModelViewTests(unittest.TestCase):
    def test_every_row_declares_its_model_view_and_no_recorded_shape_remains(self):
        rows = function_tool_contracts()
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(tool=row.tool_name, role=row.agent_role):
                self.assertIsInstance(row.model_view, ModelView)
        self.assertNotIn(
            "model_visible_result_shape", FunctionToolContract.__dataclass_fields__
        )
        for retired in (
            "recorded_result",
            "recorded_source_result",
            "model_visible_result",
            "ToolModelVisibleResultShape",
        ):
            self.assertFalse(hasattr(tool_execution, retired), retired)

    def test_a_view_shows_no_hash_timing_or_internal_id_and_one_next_step(self):
        view = ModelView(
            next_step={"resolved": "Check it.", "*": "Read the answer."},
            statement=lambda payload: payload["result"]["assessment"]["rationale"],
            cite="refsnp_event_id",
        )
        event = {
            "event_id": "fact-1",
            "event_type": "LookupRecorded",
            "payload": {
                "attempt_id": "attempt-1",
                "tool_call_id": "call-1",
                "result": {
                    "result_id": "refsnp-result:" + "0" * 64,
                    "status": "source_answered",
                    "summary": "kept for the Journey",
                    "rows": [{"name": "A", "observed_at": "2026-01-01T00:00:00Z"}],
                    "source": {"raw_response_sha256": "0" * 64, "elapsed_time_ms": 3},
                    "assessment": {"rationale": "NCBI answered HTTP 410 for rs1."},
                },
                "source_answers": [
                    {
                        "operation": "fetch_refsnp",
                        "request_identity": "https://example.test/rs1",
                        "status_code": 410,
                        "byte_count": 9,
                        "sha256": "0" * 64,
                        "media": "text",
                        "shown": "whole",
                        "body": "withdrawn",
                    }
                ],
            },
        }
        self.assertEqual(
            tool_model_view(view, event),
            {
                "status": "source_answered",
                "statement": "NCBI answered HTTP 410 for rs1.",
                "result": {"rows": [{"name": "A"}]},
                "source_answers": [
                    {
                        "operation": "fetch_refsnp",
                        "request": "https://example.test/rs1",
                        "status_code": 410,
                        "shown": "whole",
                        "body": "withdrawn",
                    }
                ],
                "cite": {"refsnp_event_id": "fact-1"},
                "next": "Read the answer.",
            },
        )


if __name__ == "__main__":
    unittest.main()


class MessageShapeTests(unittest.TestCase):
    """A refusal, a repeat refusal and a provider's refusal read in one shape: what
    happened, then where to go, then the reply's place in its count, written by one
    function of the failure module (#478 G1.4; #457 F8)."""

    def test_every_correction_reads_its_place_from_the_one_count_line(self):
        from unittest.mock import patch

        from bio_neb.failure import _correction as correction
        from bio_neb.failure._correction import (
            ActionRefused,
            CorrectionWording,
            FaultCount,
            OutputRefused,
            ProviderRejected,
            RepeatedCall,
            render_correction,
        )

        wording = CorrectionWording()
        cases = {
            "refusal": (
                ActionRefused(
                    tool_name="locate_variant",
                    errors=("stated_position_needs_location",),
                    arguments={},
                    offered=frozenset({"locate_variant", "search_journey"}),
                    fault=FaultCount(1, 4),
                ),
                "Count: failed turn 1 of 4 in a row.",
            ),
            "repeat": (
                RepeatedCall(
                    tool_name="locate_variant",
                    count=2,
                    cap=4,
                    offered=frozenset({"locate_variant", "search_journey"}),
                    earlier_status="sequence_unnamed",
                    earlier_next="Name the sequence with `sequences`.",
                ),
                "Count: failed turn 2 of 4 in a row.",
            ),
            "output": (
                OutputRefused(reason="no source cited", fault=FaultCount(3, 4)),
                "Count: failed turn 3 of 4 in a row.",
            ),
            "provider": (
                ProviderRejected(status=400, statement="tool grammar refused"),
                "Count: refused-request fault 1 of 2.",
            ),
        }
        for name, (refused, count) in cases.items():
            with (
                self.subTest(case=name),
                patch.object(
                    correction, "count_line", wraps=correction.count_line
                ) as written,
            ):
                text = render_correction(refused, wording)
                written.assert_called_once()
                self.assertIn(count, text)
                # The count follows what happened and where to go.
                self.assertGreater(text.index(count), 0)
        repeat = render_correction(cases["repeat"][0], wording)
        self.assertIn(
            "your earlier `locate_variant` call with the same arguments ended "
            "`sequence_unnamed`",
            repeat,
        )
        self.assertIn("Its next step: Name the sequence with `sequences`.", repeat)
        self.assertNotIn("same Tool, same arguments", repeat)
        provider = render_correction(cases["provider"][0], wording)
        self.assertLess(
            provider.index("tool grammar refused"), provider.index("Change what")
        )


class RefusedExecutionTests(unittest.TestCase):
    """A Tool that ran and could not produce its fact answers in the one shape,
    written by the failure module, with the reply's place in the failed turns
    (#477 C8; #478 G1.4, F8; review r1 ST1, ST2, SP2)."""

    @staticmethod
    def _returns(events, call_id):
        return [
            part
            for event in events
            if event["event_type"] == "ModelRequestRecorded"
            for message in event["payload"]["compatible_adapter_evidence"]["request"][
                "messages"
            ]
            for part in message.get("parts", [])
            if part.get("part_kind") == "tool-return"
            and part.get("tool_call_id") == call_id
        ]

    def test_an_undeclared_failure_shown_once_keeps_its_exact_return_on_reopen(self):
        import copy
        import tempfile
        import uuid
        from pathlib import Path

        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.application.queries import project_journey_events
        from bio_neb.journey import Journey
        from bio_neb.recording import read_evidence
        from tests.application_fakes import _registered
        from tests.planning_assignment_fakes import PlanningComposition
        from tests.reference_fakes import ReferenceAdapter

        class FailsOnce(ReferenceAdapter):
            failures = 0

            def discover_assemblies(self, ncbi_tax_id, query):
                self.failures += 1
                if self.failures == 1:
                    raise RuntimeError("offline fault")
                return super().discover_assemblies(ncbi_tax_id, query)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            composition.reference_adapter = FailsOnce()
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            project_journey_events(events, journey_file=receipt.journey_file)
            by_id = {e["event_id"]: e for e in events}
            (refusal,) = [
                e for e in events if e["event_type"] == "ToolExecutionRefused"
            ]
            failed = by_id[refusal["payload"]["failed_event_id"]]
            call_id = by_id[failed["payload"]["started_event_id"]]["payload"][
                "tool_call_id"
            ]
            (shown, *_) = self._returns(events, call_id)
            self.assertEqual(
                (
                    shown["content"]["statement"],
                    shown["content"]["next"],
                    shown["metadata"],
                ),
                (
                    "the Tool failed: RuntimeError",
                    "Call another Tool callable now; the same failure again holds "
                    "the Run for the researcher.",
                    {"event_id": refusal["event_id"]},
                ),
            )
            self.assertTrue(shown["content"]["count"].startswith("Count: failed turn "))
            # The return cannot lose or borrow its recorded refusal (ADR 0034).
            for label, metadata in (
                ("missing", None),
                ("substituted", {"event_id": str(uuid.uuid4())}),
            ):
                hostile = copy.deepcopy(events)
                for part in self._returns(hostile, call_id):
                    part["metadata"] = metadata
                with (
                    self.subTest(label),
                    self.assertRaisesRegex(
                        ValueError, "refused execution lost its exact result"
                    ),
                ):
                    project_journey_events(hostile, journey_file=receipt.journey_file)
