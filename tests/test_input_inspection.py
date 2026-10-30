"""Supplied input metadata through real application and deterministic callers."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from bio_neb.application.commands import RegisterRunCommand, register_run
from bio_neb.application.input_journey import RunInputsRegistered
from bio_neb.application.queries import ReviewJourneyQuery, review_journey
from bio_neb.input_inspection.commands import (
    InspectRegisteredInputsCommand,
    inspect_registered_inputs,
)
from bio_neb.input_inspection.local_files import LocalFilesAdapter
from bio_neb.journey import Journey
from tests.proposal_information_fakes import runtime_configuration


class InputInspectionTests(unittest.TestCase):
    def register(self, root):
        prompt, runtime = root / "prompt.txt", root / "runtime.json"
        prompt.write_text("Analyze the supplied reads.")
        runtime.write_text(json.dumps(runtime_configuration()))
        entries = []
        for role, name in [("read_1", "r1.fastq"), ("read_2", "r2.fastq")]:
            path = root / name
            path.write_bytes(b"reads")
            entries.append(dict(role=role, registered_path=str(path)))
        receipt = register_run(
            RegisterRunCommand(
                run_root=str(root / "run"),
                prompt_file=str(prompt),
                runtime_config_file=str(runtime),
                inputs=entries,
            )
        )
        return receipt, entries

    def test_registration_admits_no_supplied_prepared_reference(self):
        # The prepared index is a launch choice; registration carries FASTQs only.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, entries = self.register(root)
            reference = dict(role="reference_fasta", registered_path=str(root / "ref"))
            with self.assertRaises(ValidationError):
                RegisterRunCommand(
                    run_root=str(root / "other"),
                    prompt_file=str(root / "prompt.txt"),
                    runtime_config_file=str(root / "runtime.json"),
                    inputs=entries,
                    prepared_reference_inputs=[reference],
                )
            registration = next(
                event
                for event in Journey.open(Path(receipt.journey_file)).read()
                if event["event_type"] == "RunInputsRegistered"
            )
            self.assertEqual(registration["payload"]["prepared_reference_inputs"], [])
            with self.assertRaises(ValidationError):
                RunInputsRegistered.model_validate(
                    {
                        **registration["payload"],
                        "prepared_reference_inputs": [reference],
                    }
                )

    def test_supplied_inputs_share_one_stat_only_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, entries = self.register(Path(directory))
            calls = []

            def stat(path, calls=calls, **kwargs):
                calls.append(path)
                return os.stat(path, **kwargs)

            result = inspect_registered_inputs(
                InspectRegisteredInputsCommand(journey_file=receipt.journey_file),
                adapter=LocalFilesAdapter(stat_provider=stat),
            )
            self.assertEqual(calls, [e["registered_path"] for e in entries])
            self.assertEqual(result.status, "inspection_succeeded")
            self.assertEqual(
                [i.size_bytes for i in result.inspection.result.inputs],
                [5, 5],
            )
            with patch.object(
                LocalFilesAdapter,
                "observe",
                side_effect=AssertionError("reopen contacted filesystem"),
            ):
                reopened = review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                )
            self.assertEqual(reopened, result)
            self.assertEqual(len(reopened.attempts), 0)

    def test_actual_input_error_opens_shared_review_before_any_model_contact(self):
        from bio_neb.application.advancement import ResumeRun, advance

        with tempfile.TemporaryDirectory() as directory:
            receipt, entries = self.register(Path(directory))
            Path(entries[0]["registered_path"]).unlink()
            result = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(result.status, "input_review_pending")
            self.assertEqual(result.current_review.kind, "input")
            self.assertEqual(
                result.current_review.request.request.failure_event_id,
                result.inspection.source_event_id,
            )
            self.assertEqual(len(result.attempts), 0)

    def test_exact_amendment_routes_through_planning_tool_and_preserves_both_observations(
        self,
    ):
        from bio_neb.application.advancement import (
            AnswerReview,
            InputReviewAnswer,
            ResumeRun,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.failure import ReviewAnswer
        from tests.application_fakes import _registered
        from tests.planning_assignment_fakes import PlanningComposition

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, paths = _registered(root)
            original = paths["read_1"]
            original.unlink()
            pending = advance(ResumeRun(journey_file=receipt.journey_file))
            replacement = root / "corrected.fastq"
            replacement.write_bytes(b"corrected-reads")
            paths["read_1"] = replacement
            composition = PlanningComposition(receipt, paths)
            request = pending.current_review.request
            answer = AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=request.request_event_id,
                answer=InputReviewAnswer(
                    response=ReviewAnswer(
                        request_id=request.request.request_id,
                        action="amend",
                        by="researcher",
                        field="read_1",
                        value=str(replacement),
                    )
                ),
            )
            with use_run_advancement_composition(composition):
                result = advance(answer)
                calls = list(composition.calls)
                self.assertEqual(advance(answer), result)
                self.assertEqual(composition.calls, calls)
            self.assertEqual(result.status, "planning_hitl_pending")
            self.assertEqual(composition.calls.count("inspect_input_metadata"), 1)
            self.assertEqual(
                result.inspection.result.inputs[0].registered_path, str(replacement)
            )
            self.assertEqual(result.inspection.result.inputs[0].size_bytes, 15)
            self.assertEqual(
                result.registered_input_set.inputs[0].registered_path, str(original)
            )
            observations = result.input_selection.inspections
            self.assertEqual(len(observations), 2)
            self.assertEqual(
                observations[0]["payload"]["result"]["failures"][0]["code"],
                "input_missing",
            )
            self.assertEqual(
                observations[1]["payload"]["selection_event_id"],
                result.input_selection.receipts[0].response_event_id,
            )
            self.assertEqual(
                review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file)),
                result,
            )

    def test_illegal_amendment_leaves_exact_review_open_and_writes_nothing(self):
        from bio_neb.application.advancement import (
            AnswerReview,
            InputReviewAnswer,
            ResumeRun,
            advance,
        )
        from bio_neb.failure import ReviewAnswer
        from bio_neb.journey import Journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, entries = self.register(Path(directory))
            Path(entries[0]["registered_path"]).unlink()
            pending = advance(ResumeRun(journey_file=receipt.journey_file))
            request = pending.current_review.request
            for field, value in [
                ("unknown_role", "/tmp/other"),
                ("read_1", "relative"),
            ]:
                before = Journey.open(Path(receipt.journey_file)).read()
                with self.assertRaises(ValueError):
                    advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=request.request_event_id,
                            answer=InputReviewAnswer(
                                response=ReviewAnswer(
                                    request_id=request.request.request_id,
                                    action="amend",
                                    by="researcher",
                                    field=field,
                                    value=value,
                                )
                            ),
                        )
                    )
                # A Pass may honestly record the interrupted command, but no answer
                # or new input observation can be admitted.
                after = Journey.open(Path(receipt.journey_file)).read()
                self.assertEqual(
                    [e for e in after if e["event_type"].startswith("Input")],
                    [e for e in before if e["event_type"].startswith("Input")],
                )
                self.assertTrue(
                    review_journey(
                        ReviewJourneyQuery(journey_file=receipt.journey_file)
                    ).current_review.request.pending
                )

    def test_unrequested_second_inspection_refuses_without_filesystem_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = self.register(Path(directory))
            command = InspectRegisteredInputsCommand(journey_file=receipt.journey_file)
            inspect_registered_inputs(command)
            with patch.object(
                LocalFilesAdapter,
                "observe",
                side_effect=AssertionError("unexpected recheck"),
            ):
                with self.assertRaisesRegex(ValueError, "input_inspection_ineligible"):
                    inspect_registered_inputs(command)

    def test_unexpected_stat_error_is_retained_and_reraised_without_retry(self):
        from bio_neb.journey import Journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = self.register(Path(directory))
            error = PermissionError("private OS diagnostic")
            with (
                patch.object(LocalFilesAdapter, "observe", side_effect=error),
                self.assertRaises(PermissionError) as caught,
            ):
                inspect_registered_inputs(
                    InspectRegisteredInputsCommand(journey_file=receipt.journey_file)
                )
            self.assertIs(caught.exception, error)
            events = Journey.open(Path(receipt.journey_file)).read()
            self.assertEqual(events[-1]["event_type"], "InputInspectionErrored")
            self.assertEqual(events[-1]["payload"]["error_type"], "PermissionError")
            self.assertNotIn("private OS diagnostic", json.dumps(events))
            self.assertTrue(
                Journey.open(Path(receipt.journey_file))
                .checked_view("input_selection")
                .stopped
            )

    def test_reader_refuses_observation_substitution_and_wrong_selection(self):
        import copy

        from bio_neb.application.queries import project_journey_events
        from bio_neb.journey import Journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = self.register(Path(directory))
            inspect_registered_inputs(
                InspectRegisteredInputsCommand(journey_file=receipt.journey_file)
            )
            events = Journey.open(Path(receipt.journey_file)).read()
            for mutation in ("path", "selection", "size", "duplicate"):
                hostile = copy.deepcopy(events)
                inspected = next(
                    e for e in hostile if e["event_type"] == "InputMetadataInspected"
                )
                if mutation == "path":
                    inspected["payload"]["result"]["observation"]["inputs"][0][
                        "registered_path"
                    ] = "/different.fastq"
                elif mutation == "selection":
                    inspected["payload"]["selection_event_id"] = events[0]["event_id"]
                elif mutation == "size":
                    inspected["payload"]["result"]["inputs"][0]["size_bytes"] = 999
                else:
                    duplicate = copy.deepcopy(inspected)
                    duplicate["event_id"] = str(__import__("uuid").uuid4())
                    duplicate["parent_ids"] = [inspected["event_id"]]
                    hostile.append(duplicate)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    project_journey_events(hostile, journey_file=receipt.journey_file)

    def test_failed_correction_returns_to_review_without_failing_the_run(self):
        from bio_neb.agentic_journey.planning_assignment import planning_assignment
        from bio_neb.application.advancement import (
            AnswerReview,
            InputReviewAnswer,
            ResumeRun,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.failure import ReviewAnswer
        from bio_neb.journey import Journey
        from tests.proposal_information_fakes import InspectionModel

        with tempfile.TemporaryDirectory() as directory:
            receipt, entries = self.register(Path(directory))
            Path(entries[0]["registered_path"]).unlink()
            pending = advance(ResumeRun(journey_file=receipt.journey_file))
            request = pending.current_review.request
            model = InspectionModel(pending.registered_input_set.input_set_id)

            class Composition:
                environ = {"LOCAL_MODEL_KEY": "offline"}

                def planning_assignment(self, journey):
                    return planning_assignment(model_factory=model.model)

            with use_run_advancement_composition(Composition()):
                result = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=request.request_event_id,
                        answer=InputReviewAnswer(
                            response=ReviewAnswer(
                                request_id=request.request.request_id,
                                action="retry",
                                by="researcher",
                            )
                        ),
                    )
                )
            self.assertEqual(result.status, "input_review_pending")
            self.assertNotEqual(
                result.current_review.request_event_id, request.request_event_id
            )
            self.assertEqual(len(result.input_selection.inspections), 2)
            self.assertFalse(
                any(
                    e["event_type"] == "RunFailed"
                    for e in Journey.open(Path(receipt.journey_file)).read()
                )
            )
