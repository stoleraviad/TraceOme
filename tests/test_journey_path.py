"""Contract proof from the end-of-dev-cleanup Run's path, with synthetic variations.

The rows come from `tests/retained_execution_path.py`; review, failure and CPU-preparation
branches the Run never took are explicit synthetic variations.
"""

from __future__ import annotations

import ast
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from bio_neb.journey_path import (
    PATH_REGISTRY,
    Answered,
    Artifact,
    Grounding,
    Hit,
    JourneyPath,
    PathEvent,
    PathFact,
    PathKind,
    Pending,
    Prepared,
    Producer,
    project_execution,
)
from bio_neb.parabricks.execution_contact_accounting import (
    ExecutionContactJournal,
    ExecutionContactJournalRow,
    _ScannedJournal,
    _valid_prefix,
)
from bio_neb.providers.ports import ContactReceipt
from tests.retained_execution_path import path_events, prepared_path_events

_EXECUTION_ID = (
    "parabricks-execution:"
    "a0e0817ce35dc4d438c96502778e9680cc4b70b207cf1ccca4aed344a7677345"
)


def _journal_step(sequence, stage):
    """A synthetic verified step row of the current journal schema."""
    return {
        "schema_version": "bio-neb.execution-contact-journal/1",
        "sequence": sequence,
        "execution_id": _EXECUTION_ID,
        "row_kind": "step",
        "stage": stage,
        "step": "verify_artifacts",
        "phase": "verified",
        "attempt": 1,
        "ceiling_count": 0,
        "receipt": {"kind": "transfer_proof", "ref": f"retained-journal:1:{sequence}"},
        "provider_ids": [],
        "duration": None,
    }


def _bind_prebuilt_index(bound_row):
    """A synthetic bound prebuilt index: the fold reads only whether one is bound."""
    bound_row["payload"]["approved_input"]["proposal"]["execution_package"][
        "reference"
    ]["prebuilt_index"] = {"selected": True}


def fixture():
    # The derived approval binds no prebuilt index (the Run's index is that index or
    # one CPU job), so the six rows take the synthetic CPU-preparation branch.
    events = prepared_path_events()
    return {
        "events": events,
        "journal_rows": [
            _journal_step(1, "input_storage"),
            _journal_step(2, "cpu_preparation"),
            _journal_step(3, "h100"),
        ],
    }


def fact(
    event_type,
    kind,
    payload,
    *,
    rests_on=(),
    grounding="verified",
    pass_=1,
    event_id=None,
):
    return {
        "event_type": event_type,
        "kind": kind,
        "grounding": grounding,
        "pass": pass_,
        "payload": payload,
        "rests_on": list(rests_on),
        "event_id": event_id or event_type,
    }


def review_events():
    events = fixture()["events"][:3]
    events.append(
        fact(
            "CpuPreparationReviewRequested",
            "review",
            {
                "request_id": "cpu-review",
                "reason": "candidate missing",
                "legal_actions": ["abort", "approve_cpu_preparation"],
            },
            rests_on=[events[-1]["event_id"]],
        )
    )
    return events


def path_imports(source, *, stage=False):
    """Reject writer modules; StageDefs may import their declared data contracts."""
    allowed = {
        "journey": set(),
        "journey_path": {
            "Artifact",
            "ArtifactPointer",
            "Hit",
            "PathFact",
            "Prepared",
            "Producer",
            "ReferenceAcquisition",
            "Verified",
            "Unresolved",
            "AdmissionFailed",
            "Answered",
            "Drifted",
            "Miss",
        }
        if stage
        else set(),
        "parabricks.runner": {"StageDef", "StepContext"} if stage else set(),
    }
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for item in node.names:
                if any(
                    item.name == f"bio_neb.{module}"
                    or item.name.startswith(f"bio_neb.{module}.")
                    for module in allowed
                ):
                    found.append(item.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for owner, names in allowed.items():
                if module == f"bio_neb.{owner}" or (node.level and module == owner):
                    found.extend(
                        f"{module}.{item.name}"
                        for item in node.names
                        if item.name not in names
                    )
                elif module.startswith(f"bio_neb.{owner}.") or (
                    node.level and module.startswith(f"{owner}.")
                ):
                    found.append(module)
                elif module in {"", "bio_neb", "bio_neb.parabricks"}:
                    found.extend(
                        item.name
                        for item in node.names
                        if item.name == owner
                        or (
                            module == "bio_neb.parabricks"
                            and f"parabricks.{item.name}" == owner
                        )
                    )
    return found


class JourneyPathTests(unittest.TestCase):
    def test_no_selected_candidate_can_prepare_directly_from_original_authority(self):
        """Owner decision 2026-10-29 (#464): only the Execution Review's bound
        prebuilt index is a candidate; without one the Run prepares on CPU, and
        with one CPU preparation needs admission or the CPU Review's approval."""
        events = fixture()["events"][:2]
        prepared = fixture()["events"][3]
        prepared["rests_on"] = [event["event_id"] for event in events]
        state = project_execution(events + [prepared])
        self.assertIsInstance(state.reference, Prepared)
        _bind_prebuilt_index(events[0])
        with self.assertRaisesRegex(
            ValueError, "requires admission or review approval"
        ):
            project_execution(events + [prepared])

    def test_selected_candidate_miss_drift_and_admission_failure_require_cpu_approval(
        self,
    ):
        for resolution in ("miss", "drifted", "admission_failed"):
            with self.subTest(resolution=resolution):
                events = fixture()["events"][:4]
                _bind_prebuilt_index(events[0])
                events[2]["payload"]["reference"] = {
                    "state": resolution,
                    "reason": "synthetic candidate refusal",
                    "evidence": [events[1]["payload"]["pointers"][0]],
                }
                with self.assertRaisesRegex(
                    ValueError, "CPU preparation lacks review approval"
                ):
                    project_execution(events)

    def test_generic_failure_review_rejects_invalid_authority_and_abort_continuation(
        self,
    ):
        events = [
            fixture()["events"][0],
            fact(
                "StageFailed",
                "failure",
                {
                    "module": "input_stage",
                    "stage": "inputs",
                    "step": "upload",
                    "failure_class": "exhausted",
                    "detail": "retry budget exhausted",
                    "evidence": [fixture()["events"][1]["payload"]["pointers"][0]],
                },
            ),
        ]
        request = fact(
            "FailureReviewRequested",
            "review",
            {
                "stage": "inputs",
                "failure_event_id": "StageFailed",
                "request_id": "review",
                "reason": "exhausted",
                "legal_actions": ["abort"],
            },
            rests_on=["StageFailed"],
        )
        for field, value in [("stage", "other"), ("failure_event_id", "unknown")]:
            invalid = copy.deepcopy(request)
            invalid["payload"][field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "matching unresolved"),
            ):
                project_execution(events + [invalid])
        invariant = copy.deepcopy(events)
        invariant[-1]["payload"].update(failure_class="invariant", invariant_id="bug")
        with self.assertRaisesRegex(ValueError, "invariant failure holds for a fix"):
            project_execution(invariant + [request])
        events += [
            request,
            fact(
                "ExecutionOutcome",
                "outcome",
                {
                    "disposition": "incomplete",
                    "contact_journal": "journal.jsonl",
                },
                rests_on=["FailureReviewRequested"],
            ),
        ]
        answer = fact(
            "FailureReviewAnswered",
            "authority",
            {
                "request_id": "review",
                "action": "abort",
                "by": "owner",
            },
            grounding="human",
            rests_on=["FailureReviewRequested"],
            pass_=2,
        )
        invalid = copy.deepcopy(answer)
        invalid["payload"]["action"] = "retry"
        with self.assertRaisesRegex(ValueError, "declared authority"):
            project_execution(events + [invalid])
        invalid = copy.deepcopy(answer)
        invalid["grounding"] = "agent_composed"
        with self.assertRaisesRegex(ValueError, "human grounding"):
            project_execution(events + [invalid])
        events += [answer]
        with self.assertRaisesRegex(ValueError, "human abort"):
            project_execution(events + [dict(request, event_id="again", **{"pass": 2})])
        terminal = fact(
            "ExecutionOutcome",
            "outcome",
            {
                "disposition": "failed",
                "contact_journal": "pass2.jsonl",
            },
            rests_on=["FailureReviewAnswered"],
            pass_=2,
            event_id="terminal",
        )
        self.assertEqual(project_execution(events + [terminal]).disposition, "failed")

    def test_generic_failure_review_can_retry_before_reference_resolution(self):
        from bio_neb.failure import Ask, ReviewAnswer
        from bio_neb.journey_path import ExecutionState, apply_execution_event

        data = fixture()["events"]
        events = [
            data[0],
            fact(
                "StageFailed",
                "failure",
                {
                    "module": "input_stage",
                    "stage": "inputs",
                    "step": "upload",
                    "failure_class": "exhausted",
                    "detail": "retry budget exhausted",
                    "evidence": [data[1]["payload"]["pointers"][0]],
                },
            ),
        ]
        request = Ask(
            stage="inputs",
            failure_event_id="StageFailed",
            request_id="failure-review",
            reason="retry budget exhausted",
        )
        events.append(
            fact(
                "FailureReviewRequested",
                "review",
                request.model_dump(),
                rests_on=["StageFailed"],
            )
        )
        events.append(
            fact(
                "ExecutionOutcome",
                "outcome",
                {
                    "disposition": "incomplete",
                    "contact_journal": "journal.jsonl",
                },
                rests_on=["FailureReviewRequested"],
            )
        )
        answer = ReviewAnswer(request_id="failure-review", action="retry", by="owner")
        events.append(
            fact(
                "FailureReviewAnswered",
                "authority",
                answer.model_dump(),
                grounding="human",
                rests_on=["FailureReviewRequested"],
                pass_=2,
            )
        )
        inputs = copy.deepcopy(data[1])
        inputs["pass"] = 2
        events.append(inputs)
        state = project_execution(events)
        self.assertEqual(state.inputs.state, "verified")
        self.assertEqual(state.resolved_failure_ids, ("StageFailed",))
        self.assertEqual(len(state.failures), 1)
        incremental = ExecutionState()
        for event in events:
            incremental = apply_execution_event(incremental, event)
        self.assertEqual(incremental, state)

    def test_retained_run_path_folds_to_its_verified_outcome(self):
        # Expected values are the Run's recorded pointers and objects, not a
        # second fold.
        events = path_events()
        state = project_execution(events)
        self.assertEqual(state.disposition, "succeeded")
        self.assertEqual(state.frontier, "artifacts_verified")
        self.assertEqual(state.status, "execution_artifacts_verified")
        self.assertEqual(state.pass_, 1)
        self.assertEqual(
            [p.role for p in state.inputs.pointers],
            ["read_1", "read_2", "reference_source", "input_manifest"],
        )
        self.assertIsInstance(state.reference, Hit)
        self.assertEqual(len(state.reference.pointers), 10)
        self.assertEqual(
            [p.role for p in state.h100.pointers],
            ["aligned_bam", "VCF", "worker_receipt", "execution_manifest"],
        )
        self.assertEqual(
            state.h100.receipt_id,
            "sha256:966d04cf902ee4e002e007d1e54d657db43eda9dbe81b8ec51a80760d5cf6716",
        )
        self.assertEqual(state.failures, ())
        self.assertEqual(state.h100.pointers[1].bytes, 48_808_433)
        self.assertEqual(
            state.h100.pointers[1].content_id,
            "sha256:3dbe1dcdf18f83a5d4cd5fd3a193ca0133fb3ad23db666e8f46002947a2d022f",
        )
        self.assertEqual(
            state.rests_on["8e88175d-4eb4-4c64-a408-cd7847793b2f"],
            ("597ee58d-e0fa-4efc-9c8e-87ca31216cfe",),
        )

    def test_registered_path_and_failure_reviews_keep_two_closed_axes(self):
        self.assertEqual(sum(map(len, PATH_REGISTRY.values())), 12)
        self.assertEqual(
            set(PathKind),
            {
                "decision",
                "authority",
                "admission",
                "verified_artifact",
                "review",
                "failure",
                "outcome",
                "observation",
            },
        )
        self.assertEqual(
            set(Grounding),
            {"verified", "tool_succeeded", "agent_composed", "ambiguous", "human"},
        )
        for field, value in [
            ("kind", "heartbeat"),
            ("grounding", "probably"),
            ("event_type", "Observed"),
        ]:
            row = copy.deepcopy(fixture()["events"][0])
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                PathEvent.model_validate(row)

    def test_typed_payload_and_grounding_refuse_false_verified_claim(self):
        for mutate in [
            lambda r: r["payload"].update(receipt_id=None),
            lambda r: r.update(grounding="human"),
            lambda r: r["payload"].update(extra="unregistered"),
        ]:
            row = copy.deepcopy(fixture()["events"][1])
            mutate(row)
            with self.subTest(row=row), self.assertRaises(ValidationError):
                PathEvent.model_validate(row)

    def test_rests_on_required_for_each_dependent_kind(self):
        samples = fixture()["events"][1:] + review_events()[-1:]
        for row in samples:
            row = dict(row, rests_on=[])
            with (
                self.subTest(kind=row["kind"]),
                self.assertRaisesRegex(ValueError, "requires rests_on"),
            ):
                PathEvent.model_validate(row)
        row = dict(fixture()["events"][0], rests_on=["unrelated"])
        with self.assertRaisesRegex(
            ValueError, "only legal.*applicable authority answers"
        ):
            PathEvent.model_validate(row)

    def test_unknown_duplicate_and_future_dependencies_refused(self):
        events = fixture()["events"]
        for changed in [
            events[:1] * 2,
            [events[0], dict(events[1], rests_on=["future"])],
            [events[0], dict(events[1], rests_on=[events[0]["event_id"]] * 2)],
        ]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                project_execution(changed)

    def test_rests_on_must_name_current_stage_prerequisite(self):
        events = fixture()["events"]
        events[4]["rests_on"] = [events[0]["event_id"]]
        with self.assertRaisesRegex(ValueError, "prerequisite"):
            project_execution(events)

    def test_reference_cannot_change_after_verification(self):
        events = fixture()["events"][:-1]
        changed = dict(events[2], event_id="changed-reference")
        with self.assertRaisesRegex(ValueError, "requires an unresolved decision"):
            project_execution(events + [changed])

    def test_verified_pointer_cannot_be_grounded_only_by_human(self):
        event = fixture()["events"][1]
        event["payload"]["pointers"][0]["grounding"] = "human"
        with self.assertRaisesRegex(ValueError, "unverified artifact"):
            PathEvent.model_validate(event)

    def test_success_cannot_skip_verification(self):
        events = fixture()["events"]
        events[-1]["rests_on"] = [events[0]["event_id"]]
        with self.assertRaisesRegex(ValueError, "requires verified H100"):
            project_execution([events[0], events[-1]])

    def test_review_continuation_across_passes(self):
        events = review_events()
        self.assertIsInstance(project_execution(events).review, Pending)
        self.assertEqual(
            project_execution(events).status, "cpu_preparation_review_pending"
        )
        events.append(
            fact(
                "ExecutionOutcome",
                "outcome",
                {"disposition": "incomplete", "contact_journal": "journal.jsonl"},
                rests_on=[events[-1]["event_id"]],
            )
        )
        events.append(
            fact(
                "CpuPreparationReviewAnswered",
                "authority",
                {
                    "request_id": "cpu-review",
                    "action": "approve_cpu_preparation",
                    "by": "researcher",
                },
                grounding="human",
                rests_on=["CpuPreparationReviewRequested"],
                pass_=2,
            )
        )
        state = project_execution(events)
        self.assertIsInstance(state.review, Answered)
        self.assertEqual(state.pass_, 2)
        self.assertEqual(state.status, "cpu_preparation_authorized")
        self.assertIsNone(state.outcome)
        prepared = fixture()["events"][3]
        prepared["pass"] = 2
        prepared["rests_on"].append("CpuPreparationReviewAnswered")
        self.assertIsInstance(
            project_execution(events + [prepared]).reference, Prepared
        )

    def test_review_rejects_wrong_request_action_and_amendment_field(self):
        """Owner decision 2026-10-29 (#464): the CPU Review amends nothing, so a
        location amendment is refused like any undeclared field."""
        for payload in [
            {
                "request_id": "cpu-review",
                "action": "amend",
                "field": "location",
                "value": "s3://corrected/reference",
                "by": "owner",
            },
            {"request_id": "wrong", "action": "approve_cpu_preparation", "by": "owner"},
            {
                "request_id": "cpu-review",
                "action": "amend",
                "field": "credentials",
                "value": "bad",
                "by": "owner",
            },
        ]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                project_execution(
                    review_events()
                    + [
                        fact(
                            "ExecutionOutcome",
                            "outcome",
                            {
                                "disposition": "incomplete",
                                "contact_journal": "journal.jsonl",
                            },
                            rests_on=["CpuPreparationReviewRequested"],
                        ),
                        fact(
                            "CpuPreparationReviewAnswered",
                            "authority",
                            payload,
                            grounding="human",
                            rests_on=["CpuPreparationReviewRequested"],
                            pass_=2,
                        ),
                    ]
                )

    def test_only_designed_stop_can_open_next_pass(self):
        events = fixture()["events"]
        for prefix, last in [
            (events[:1], dict(events[1], **{"pass": 2})),
            (events, dict(events[-1], event_id="after")),
            ([], dict(events[0], **{"pass": 2})),
        ]:
            with self.subTest(last=last), self.assertRaises(ValueError):
                project_execution(prefix + [last])

    def test_unresolved_stage_failure_blocks_later_success(self):
        events = fixture()["events"]
        failure = fact(
            "StageFailed",
            "failure",
            {
                "module": "h100",
                "stage": "h100",
                "step": "verify",
                "failure_class": "semantic",
                "detail": "closure failed",
                "evidence": [events[1]["payload"]["pointers"][0]],
            },
        )
        with self.assertRaisesRegex(ValueError, "unresolved"):
            project_execution(events[:-1] + [failure, events[-1]])
        with self.assertRaisesRegex(ValueError, "unresolved"):
            project_execution(events[:4] + [failure, events[4]])

    def test_review_answer_cannot_bypass_stop_within_one_pass(self):
        events = review_events()
        answer = fact(
            "CpuPreparationReviewAnswered",
            "authority",
            {
                "request_id": "cpu-review",
                "action": "approve_cpu_preparation",
                "by": "owner",
            },
            grounding="human",
            rests_on=["CpuPreparationReviewRequested"],
        )
        with self.assertRaisesRegex(ValueError, "next pass"):
            project_execution(events + [answer])

    def test_cpu_review_cannot_name_or_resolve_a_stage_failure(self):
        base = fixture()["events"][:3]
        base.append(
            fact(
                "StageFailed",
                "failure",
                {
                    "module": "cpu",
                    "stage": "cpu_preparation",
                    "step": "observe",
                    "failure_class": "worker_failed",
                    "detail": "worker stopped",
                    "evidence": [base[1]["payload"]["pointers"][0]],
                },
            )
        )
        request = review_events()[-1]
        request["payload"].update(failure_event_id="StageFailed")
        request["rests_on"].append("StageFailed")
        with self.assertRaisesRegex(
            ValueError, "failure_event_id|generic failure review"
        ):
            project_execution(base + [request])
        request["payload"].pop("failure_event_id")
        with self.assertRaisesRegex(ValueError, "generic failure review"):
            project_execution(base + [request])

    def test_failure_evidence_keeps_module_identity_and_sets_failed_outcome(self):
        events = fixture()["events"][:3]
        evidence = events[1]["payload"]["pointers"][0]
        failure = {
            "module": "provider_nebius",
            "stage": "cpu_preparation",
            "step": "observe",
            "failure_class": "invariant",
            "invariant_id": "manifest-last",
            "detail": "bad manifest",
            "context": {"attempt": 1},
            "evidence": [evidence],
        }
        events.append(
            fact("StageFailed", "failure", failure, grounding="tool_succeeded")
        )
        events.append(
            fact(
                "ExecutionOutcome",
                "outcome",
                {"disposition": "failed", "contact_journal": "journal.jsonl"},
                rests_on=["StageFailed"],
            )
        )
        state = project_execution(events)
        self.assertEqual(state.disposition, "failed")
        self.assertEqual(state.status, "execution_failed")
        self.assertEqual(state.failures[0].module, "provider_nebius")
        self.assertEqual(state.failures[0].invariant_id, "manifest-last")
        failure["evidence"] = []
        with self.assertRaises(ValueError):
            project_execution(events)

    def test_an_invariant_without_outcome_holds_and_reenters_only_resting_on_it(self):
        # Held, not failed; the next pass's first fact rests on the held failure
        # and resolves it. No new event type.
        events = fixture()["events"][:3]
        evidence = events[1]["payload"]["pointers"][0]
        invariant = {
            "module": "bio_neb.parabricks.reference_stages",
            "stage": "cpu_preparation",
            "step": "verify_bundle",
            "failure_class": "invariant",
            "invariant_id": "worker_receipt_read_invalid",
            "detail": "worker_receipt_read_invalid",
            "evidence": [evidence],
        }
        held = fact("StageFailed", "failure", invariant, grounding="ambiguous")
        state = project_execution([*events, held])
        self.assertEqual(state.status, "run_held")
        self.assertEqual(state.disposition, "incomplete")
        self.assertEqual(state.held_failure_id, "StageFailed")
        prepared = dict(fixture()["events"][3], **{"pass": 2})
        with self.assertRaisesRegex(ValueError, "re-entry must rest on its held"):
            project_execution([*events, held, prepared])
        with self.assertRaisesRegex(ValueError, "prevents further progress"):
            project_execution([*events, held, dict(prepared, **{"pass": 1})])
        resumed = project_execution(
            [
                *events,
                held,
                dict(prepared, rests_on=[*prepared["rests_on"], "StageFailed"]),
            ]
        )
        self.assertEqual(resumed.pass_, 2)
        self.assertEqual(resumed.resolved_failure_ids, ("StageFailed",))
        self.assertEqual(resumed.status, "prepared_reference_verified")
        again = fact(
            "StageFailed",
            "failure",
            invariant,
            grounding="ambiguous",
            pass_=2,
            rests_on=["StageFailed"],
            event_id="StageFailedAgain",
        )
        self.assertEqual(
            project_execution([*events, held, again]).held_failure_id,
            "StageFailedAgain",
        )
        for wrong in (
            dict(again, rests_on=["StageFailed", events[0]["event_id"]]),
            dict(again, **{"pass": 1}),
        ):
            with self.assertRaisesRegex(ValueError, "held failure"):
                project_execution([*events, held, wrong])

    def test_writer_and_reader_preserve_checked_state(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = JourneyPath.create(Path(directory))
            ids = {}
            for row in fixture()["events"]:
                row = dict(row)
                old_id = row.pop("event_id")
                row["rests_on"] = [ids[x] for x in row["rests_on"]]
                ids[old_id], state = journey.path(
                    journey.state, PathFact.model_validate(row)
                )
            self.assertEqual(journey.state.status, "execution_artifacts_verified")
            reopened = JourneyPath.open(
                journey.file, expected_prefix_sha256=journey.prefix_sha256
            )
            self.assertEqual(reopened.read(), journey.read())
            with self.assertRaisesRegex(ValueError, "prefix changed"):
                JourneyPath.open(journey.file, expected_prefix_sha256="0" * 64)

    def test_writer_refuses_success_without_artifacts_without_changing_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = JourneyPath.create(Path(directory))
            bound = dict(fixture()["events"][0])
            bound.pop("event_id")
            bound_id, state = journey.path(
                journey.state, PathFact.model_validate(bound)
            )
            before = journey.file.read_bytes(), journey.prefix_sha256, journey.read()
            invalid_row = fact(
                "ExecutionOutcome",
                "outcome",
                {"disposition": "succeeded", "contact_journal": "unused.jsonl"},
                rests_on=[bound_id],
            )
            invalid_row.pop("event_id")
            invalid = PathFact.model_validate(invalid_row)
            with self.assertRaisesRegex(ValueError, "success requires verified H100"):
                journey.path(journey.state, invalid)
            self.assertEqual(
                (journey.file.read_bytes(), journey.prefix_sha256, journey.read()),
                before,
            )

    def test_registered_but_uninterpreted_event_cannot_acquire_outcome_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = JourneyPath.create(Path(directory))
            for row in fixture()["events"][:2]:
                value = {k: v for k, v in row.items() if k != "event_id"}
                value["rests_on"] = list(journey.state.event_ids)
                _, state = journey.path(journey.state, PathFact.model_validate(value))
            with patch.dict(
                PATH_REGISTRY[PathKind.OUTCOME],
                {"FutureOutcome": PATH_REGISTRY[PathKind.OUTCOME]["ExecutionOutcome"]},
            ):
                event = PathFact(
                    event_type="FutureOutcome",
                    kind="outcome",
                    grounding="verified",
                    **{"pass": 1},
                    rests_on=(state.latest_event_ids["InputsVerified"],),
                    payload={
                        "disposition": "incomplete",
                        "milestone": "input_storage",
                        "contact_journal": "unused.jsonl",
                    },
                )
                before = journey.file.read_bytes()
                with self.assertRaisesRegex(ValueError, "no execution interpretation"):
                    journey.path(state, event)
                self.assertEqual(journey.file.read_bytes(), before)

    def test_writer_binds_state_to_prefix_and_returns_the_recorded_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = JourneyPath.create(Path(directory))
            initial = journey.state
            row = dict(fixture()["events"][0])
            row.pop("event_id")
            with patch.object(Path, "read_bytes", side_effect=AssertionError("polled")):
                identity, state = journey.path(initial, PathFact.model_validate(row))
            self.assertEqual(state.event_ids, (identity,))
            self.assertEqual(state.latest_event_ids, {"ExecutionBound": identity})
            self.assertEqual(state.bound.run_id, "3b17ac33-133c-47f2-8271-b95fbec21a71")
            self.assertEqual(state, project_execution(journey.read()))
            input_row = dict(fixture()["events"][1])
            input_row.pop("event_id")
            input_row["rests_on"] = [identity]
            incoming = PathFact.model_validate(input_row)
            before = journey.file.read_bytes(), journey.prefix_sha256, journey.read()
            altered = state.model_copy(update={"pass_": 2})
            for wrong in (initial, altered):
                with self.assertRaisesRegex(ValueError, "writer semantic prefix"):
                    journey.path(wrong, incoming)
            with patch("bio_neb.journey_path.uuid.uuid4", return_value=identity):
                with self.assertRaisesRegex(ValueError, "duplicate event"):
                    journey.path(state, incoming)
            self.assertEqual(
                (journey.file.read_bytes(), journey.prefix_sha256, journey.read()),
                before,
            )
            _, accepted = journey.path(state, incoming)
            self.assertEqual(
                accepted.inputs.receipt_id, input_row["payload"]["receipt_id"]
            )
            self.assertEqual(accepted, JourneyPath.open(journey.file).state)
            accepted.bound.approved_input["mutated"] = True
            self.assertNotIn("mutated", journey.state.bound.approved_input)
            self.assertNotIn("mutated", journey.read()[0].payload.approved_input)
            self.assertEqual(
                project_execution(journey.read()), JourneyPath.open(journey.file).state
            )

    def test_io_failure_returns_no_state_and_poisoned_writer_cannot_continue(self):
        for boundary in ("write", "sync"):
            with (
                self.subTest(boundary=boundary),
                tempfile.TemporaryDirectory() as directory,
            ):
                journey = JourneyPath.create(Path(directory))
                state = journey.state
                row = dict(fixture()["events"][0])
                row.pop("event_id")
                incoming = PathFact.model_validate(row)
                target = (
                    "pathlib.Path.open"
                    if boundary == "write"
                    else "bio_neb.journey_path.os.fsync"
                )
                with patch(target, side_effect=OSError("disk refused")):
                    with self.assertRaisesRegex(OSError, "disk refused"):
                        journey.path(state, incoming)
                self.assertEqual(journey.read(), ())
                self.assertEqual(journey.state, state)
                with self.assertRaisesRegex(
                    ValueError, "unusable after an I/O failure"
                ):
                    journey.path(state, incoming)
                # A sync failure may have written bytes. No rollback/recovery claim.
                if boundary == "write":
                    self.assertEqual(journey.file.read_bytes(), b"")

    def test_dev_milestone_outcomes_do_not_authorize_continuation_passes(self):
        from bio_neb.journey_path import apply_execution_event

        for milestone, count in (("input_storage", 2), ("reference_preparation", 4)):
            with self.subTest(milestone=milestone):
                prefix = fixture()["events"][:count]
                stop = fact(
                    "ExecutionOutcome",
                    "outcome",
                    {
                        "disposition": "incomplete",
                        "milestone": milestone,
                        "contact_journal": "dev-only.jsonl",
                    },
                    rests_on=[prefix[-1]["event_id"]],
                )
                state = project_execution(prefix + [stop])
                candidate = dict(fixture()["events"][count], **{"pass": 2})
                with self.assertRaisesRegex(
                    ValueError, "new pass requires a designed stop"
                ):
                    apply_execution_event(state, candidate)

    def test_attach_and_read_verify_immutable_small_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = JourneyPath.create(Path(directory))
            artifact = Artifact(
                content=b'{"manifest":"proof"}',
                role="manifest",
                producer=Producer(stage="h100"),
                grounding=Grounding.VERIFIED,
                proof_ref="cj:1:3",
            )
            pointer = journey.attach(artifact)
            self.assertEqual(journey.attach(artifact), pointer)
            self.assertEqual(journey.read_artifact(pointer), artifact.content)
            (journey.file.parent / pointer.location).write_bytes(b"broken")
            with self.assertRaises(ValueError):
                journey.read_artifact(pointer)
            with self.assertRaises(ValueError):
                journey.attach(artifact)

    def test_owner_backed_bundle_artifact_is_verified_once_per_owner(self):
        from bio_neb.journey import Journey

        with tempfile.TemporaryDirectory() as directory:
            owner = Journey.create(Path(directory).resolve())
            owner.record("JourneyStarted", {})
            journey = JourneyPath.from_journey(owner)
            artifact = Artifact(
                content=b'{"manifest":"proof"}',
                role="manifest",
                producer=Producer(stage="h100"),
                grounding=Grounding.VERIFIED,
                proof_ref="cj:1:3",
            )
            pointer = journey.attach(artifact)
            location = journey.file.parent / pointer.location
            reads = []
            original_read = Path.read_bytes

            def read_bytes(selected):
                if selected == location:
                    reads.append(selected)
                return original_read(selected)

            with patch.object(Path, "read_bytes", read_bytes):
                self.assertEqual(journey.read_artifact(pointer), artifact.content)
                again = JourneyPath.from_journey(owner)
                self.assertEqual(again.read_artifact(pointer), artifact.content)
            self.assertEqual(len(reads), 1)
            location.write_bytes(b"broken")
            # In-owner reuse trusts the first verification (0029).
            self.assertEqual(journey.read_artifact(pointer), artifact.content)
            # An independent open is the boundary for current disk bytes.
            fresh = JourneyPath.from_journey(Journey.open(owner.path))
            with self.assertRaisesRegex(ValueError, "artifact integrity changed"):
                fresh.read_artifact(pointer)

    def test_path_read_is_an_isolated_snapshot_without_disk_polling(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = JourneyPath.create(Path(directory))
            row = dict(fixture()["events"][0])
            row.pop("event_id")
            journey.path(journey.state, PathFact.model_validate(row))
            read = journey.read()
            read[0].payload.approved_input["mutated"] = True
            with patch.object(Path, "read_bytes", side_effect=AssertionError("polled")):
                self.assertNotIn("mutated", journey.read()[0].payload.approved_input)

    def test_effect_modules_cannot_import_path(self):
        root = Path(__file__).parents[1] / "src/bio_neb"
        effects = list((root / "providers").rglob("*.py")) + list(
            (root / "agentic_journey/adapters").rglob("*.py")
        )
        effects += [
            root / "parabricks" / name
            for name in [
                "object_storage_tracer.py",
                "reusable_object_storage_parabricks.py",
            ]
        ]
        effects += [root / "parabricks/storage_readiness.py"]
        for file in effects:
            with self.subTest(file=file.relative_to(root)):
                self.assertEqual(path_imports(file.read_text()), [])
        for name in ("input_storage.py", "reference_stages.py", "h100_stage.py"):
            file = root / "parabricks" / name
            with self.subTest(stage=name):
                self.assertEqual(path_imports(file.read_text(), stage=True), [])

    def test_import_guard_detects_injected_violations(self):
        for source in [
            "import bio_neb.journey_path",
            "from bio_neb.journey_path import JourneyPath",
            "from bio_neb import journey_path",
            "from ..journey_path import JourneyPath",
            "from .. import journey_path",
            "import bio_neb.journey as retained",
            "from bio_neb.journey import Journey as Writer",
            "from bio_neb import journey",
            "from ..journey import Journey",
            "from .. import journey",
            "import bio_neb.parabricks.runner as execution",
            "from bio_neb.parabricks.runner import record_outcome",
            "from bio_neb.parabricks import runner",
        ]:
            with self.subTest(source=source):
                self.assertTrue(path_imports(source))
                self.assertTrue(path_imports(source, stage=True))

    def test_stage_import_guard_allows_facts_but_not_writer_capabilities(self):
        for source in (
            "from bio_neb.journey_path import PathFact, Artifact, Producer",
            "from ..journey_path import Verified as InputState",
            "from bio_neb.parabricks.runner import StageDef, StepContext",
        ):
            with self.subTest(allowed=source):
                self.assertEqual(path_imports(source, stage=True), [])
                self.assertTrue(path_imports(source))
        for source in (
            "from bio_neb.journey_path import PathFact, JourneyPath",
            "from ..journey_path import JourneyPath as Writer",
            "from bio_neb.parabricks.runner import StageDef, run_stage",
            "from ..parabricks.runner import record_outcome as finish",
            "from bio_neb.journey_path import *",
            "from bio_neb.parabricks.runner import *",
        ):
            with self.subTest(forbidden=source):
                self.assertTrue(path_imports(source, stage=True))

    def test_advance_routes_only_retained_kind_files_without_an_owner(self):
        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.content import canonical_json
        from bio_neb.journey import Journey, JourneyValidationError
        from bio_neb.journey_path import (
            has_execution_path,
            is_historical_execution_file,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            legacy = root / "legacy" / "journey.jsonl"
            legacy.parent.mkdir()
            legacy.write_bytes(
                b"".join(
                    (canonical_json(event) + "\n").encode()
                    for event in fixture()["events"][:3]
                )
            )
            semantic = Journey.create(root / "runs")
            semantic.record("JourneyStarted", {})
            corrupt = root / "corrupt.jsonl"
            corrupt.write_bytes(b"{not json\n")

            self.assertTrue(is_historical_execution_file(legacy))
            self.assertTrue(has_execution_path(legacy))
            self.assertIsNone(JourneyPath.open(legacy)._owner)
            for other in (semantic.path, corrupt, root / "missing.jsonl"):
                with self.subTest(file=other.name):
                    self.assertFalse(is_historical_execution_file(other))

            routed = []
            with patch(
                "bio_neb.application.advancement._advance",
                side_effect=lambda intent: routed.append(
                    Journey.processing_owner(Path(intent.journey_file))
                ),
            ):
                advance(ResumeRun(journey_file=str(legacy)))
                self.assertEqual(routed, [None])
                for other in (corrupt, root / "missing.jsonl"):
                    with (
                        self.subTest(file=other.name),
                        self.assertRaises(JourneyValidationError),
                    ):
                        advance(ResumeRun(journey_file=str(other)))
                self.assertEqual(routed, [None])


class ContactJournalExtensionTests(unittest.TestCase):
    def test_adapted_journal_rows_roundtrip(self):
        for row in fixture()["journal_rows"]:
            typed = ExecutionContactJournalRow.model_validate(row)
            self.assertEqual(
                typed,
                ExecutionContactJournalRow.model_validate_json(typed.model_dump_json()),
            )

    def test_engine_phases_and_receipts_share_existing_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = ExecutionContactJournal(
                Path(directory) / "journal.jsonl", "parabricks-execution:" + "a" * 64
            )
            for phase in ["claimed", "invoked", "verified", "failed"]:
                ref = journal.step(
                    stage="h100",
                    step="submit",
                    phase=phase,
                    attempt=2,
                    ceiling_count=1,
                    receipt=ContactReceipt(kind="job_submission", ref="receipt.json"),
                    provider_ids=("job-one",),
                    duration=0.25,
                )
            self.assertEqual(ref, "cj:1:4")
            journal.finalize()
            content = journal.path.read_bytes()
            rows, _, truncated, discrepancies = _valid_prefix(content)
            self.assertFalse(truncated)
            self.assertEqual(discrepancies, ())
            self.assertEqual(rows[2][0].phase, "verified")
            self.assertEqual(rows[2][0].provider_ids, ("job-one",))
            self.assertEqual(rows[2][0].duration, 0.25)
            scanned = _ScannedJournal(content, journal.execution_id)
            self.assertTrue(scanned.complete)
            self.assertEqual(scanned.discrepancies, [])
            malformed = content + b'{"partial":'
            self.assertEqual(_valid_prefix(malformed)[1], len(content))
            self.assertTrue(_valid_prefix(malformed)[2])

    def test_closed_or_failed_sink_never_returns_a_proof_locator(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = ExecutionContactJournal(
                Path(directory) / "journal.jsonl", "parabricks-execution:" + "a" * 64
            )
            journal.finalize()
            with self.assertRaisesRegex(ValueError, "not writable"):
                journal.step(stage="h100", step="submit", phase="invoked")

    def test_invalid_columns_do_not_enter_valid_prefix(self):
        row = fixture()["journal_rows"][0]
        for field, value in [
            ("phase", "running"),
            ("attempt", 0),
            ("duration", -1.0),
            ("ceiling_count", -1),
            ("stage", None),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                ExecutionContactJournalRow.model_validate(dict(row, **{field: value}))
