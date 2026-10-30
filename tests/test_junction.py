"""Execution path Reviews answered on real stops of an approved Run.

The Execution Review binds the offline Galaxy prebuilt index, the Run's only
candidate besides one CPU job (owner, 2026-10-29, #464). Each stop is a real
Execution Pass over offline fake providers, so its outcome names the Pass's
coordinator contact journal; no external contact.
"""

import tempfile
import unittest
from collections import deque
from contextlib import contextmanager
from pathlib import Path

from bio_neb.failure import ProviderError
from bio_neb.journey_path import JourneyPath, PathFact, project_execution
from bio_neb.parabricks.plan import RetainedExecution, plan_execution
from tests.h100_stage_fakes import h100_provider
from tests.parabricks_operation_fakes import (
    approved_input,
    execute_and_read_path,
    input_storage_provider,
)
from tests.prebuilt_index_support import (
    GALAXY_FASTA,
    GalaxyOpener,
    offline_galaxy_files,
)
from tests.reference_fakes import SOURCE_GZIP
from tests.reference_stage_fakes import seed_cpu


class JunctionTests(unittest.TestCase):
    def test_generic_failure_answer_uses_the_public_route_without_granting_cpu_approval(
        self,
    ):
        from bio_neb.application.advancement import (
            AnswerReview,
            ExecutionPathReviewAnswer,
            advance,
        )

        failure_id, request_id = self.failure_stop("admit_candidate")
        before = self.path.file.read_bytes()
        with self.offline_continuation(), self.assertRaises(ValueError):
            advance(
                AnswerReview(
                    journey_file=self.approved.journey_file,
                    request_event_id=request_id,
                    answer=ExecutionPathReviewAnswer(
                        kind="execution_path",
                        response={"action": "approve_cpu_preparation", "by": "owner"},
                    ),
                )
            )
        self.assertEqual(self.path.file.read_bytes(), before)
        with self.offline_continuation():
            result = advance(
                AnswerReview(
                    journey_file=self.approved.journey_file,
                    request_event_id=request_id,
                    answer=ExecutionPathReviewAnswer(
                        kind="execution_path",
                        response={"action": "retry", "by": "owner"},
                    ),
                )
            )
        # The retry continues the Run to its next stop: the next Pass admits the
        # candidate, Execution completes and Investigation begins from the plan's
        # target, stopping at its first Review.
        self.assertEqual(result.status, "investigation_review_pending")
        public_result = result
        result = JourneyPath.open(self.path.file).state
        self.assertEqual(result.failure_review.action, "retry")
        self.assertEqual(result.failure_event_ids, (failure_id,))
        self.assertEqual(result.resolved_failure_ids, (failure_id,))
        answer_id = result.latest_event_ids["FailureReviewAnswered"]
        self.assertEqual(result.rests_on[answer_id], (request_id,))
        self.assertEqual(result.rests_on[request_id], (failure_id,))
        self.assertIsNone(result.review)
        self.assertEqual(result.unresolved_failure_ids, ())
        self.assertNotEqual(public_result.status, "cpu_preparation_authorized")
        self.assertNotIn("CpuPreparationReviewAnswered", result.latest_event_ids)
        self.assertEqual(
            project_execution(JourneyPath.open(self.path.file).read()), result
        )

    def test_public_answer_command_routes_semantic_cpu_review_without_contact(
        self,
    ):
        import io
        import json
        from contextlib import redirect_stdout

        from bio_neb.cli import main

        request_id = self.stop()
        output = io.StringIO()
        with self.offline_continuation(), redirect_stdout(output):
            code = main(
                [
                    "answer",
                    "--journey-file",
                    self.approved.journey_file,
                    "--request-event-id",
                    request_id,
                    "--response-json",
                    json.dumps({"action": "approve"}),
                ]
            )
        self.assertEqual(code, 0)
        # The answer authorizes CPU preparation and the Run continues: the next
        # Pass prepares the reference on that authority, Execution completes and
        # Investigation begins from the plan's target.
        self.assertEqual(
            json.loads(output.getvalue())["status"], "investigation_review_pending"
        )
        state = project_execution(JourneyPath.open(self.path.file).read())
        self.assertIsNone(state.failure_review)
        answer_id = state.latest_event_ids["CpuPreparationReviewAnswered"]
        self.assertEqual(state.rests_on[answer_id], (request_id,))
        self.assertIn(
            answer_id, state.rests_on[state.latest_event_ids["ReferencePrepared"]]
        )
        self.assertEqual(state.disposition, "succeeded")

    def failure_stop(self, stage):
        """Run the next Pass to a rejected provider effect at stage's Review."""
        self.failing_stage = stage
        state = self.execute()
        self.assertEqual(state.status, "execution_review_pending")
        self.assertEqual(state.failures[-1].stage, stage)
        return (
            state.unresolved_failure_ids[-1],
            state.latest_event_ids["FailureReviewRequested"],
        )

    def assert_new_pass_requires_answer(self, request_id):
        from bio_neb.journey_path import Outcome

        state = self.path.state
        before = self.path.file.read_bytes(), self.path.prefix_sha256, self.path.read()
        invalid = PathFact(
            event_type="ExecutionOutcome",
            kind="outcome",
            grounding="agent_composed",
            **{"pass": state.pass_ + 1},
            rests_on=(request_id,),
            payload=Outcome(
                disposition="incomplete", contact_journal=state.outcome.contact_journal
            ),
        )
        with self.assertRaisesRegex(
            ValueError, "new pass must answer its current review"
        ):
            self.path.path(state, invalid)
        self.assertEqual(
            (self.path.file.read_bytes(), self.path.prefix_sha256, self.path.read()),
            before,
        )

    def test_cpu_stop_cannot_start_a_new_pass_without_its_answer(self):
        self.assert_new_pass_requires_answer(self.stop())

    def test_failure_stop_cannot_start_a_new_pass_without_its_answer(self):
        _, request_id = self.failure_stop("admit_candidate")
        self.assert_new_pass_requires_answer(request_id)

    def test_cli_failure_answer_preserves_exact_failure_identity_and_actions(self):
        import io
        import json
        from contextlib import redirect_stdout

        from bio_neb.cli import main

        failure_id, request_id = self.failure_stop("admit_candidate")
        output = io.StringIO()
        with self.offline_continuation(), redirect_stdout(output):
            code = main(
                [
                    "answer",
                    "--journey-file",
                    self.approved.journey_file,
                    "--request-event-id",
                    request_id,
                    "--response-json",
                    json.dumps({"action": "retry"}),
                ]
            )
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        # The retry continues the Run: the next Pass admits the candidate the
        # failed Pass could not place, Execution completes, and Investigation
        # begins from the plan's target.
        self.assertEqual(result["status"], "investigation_review_pending")
        retained = JourneyPath.open(self.path.file).state
        self.assertEqual((retained.disposition, retained.pass_), ("succeeded", 2))
        self.assertEqual(retained.failure_event_ids, (failure_id,))
        self.assertEqual(retained.resolved_failure_ids, (failure_id,))
        answer_id = retained.latest_event_ids["FailureReviewAnswered"]
        self.assertEqual(retained.rests_on[answer_id], (request_id,))
        self.assertEqual(retained.rests_on[request_id], (failure_id,))

    def test_old_identical_answer_preserves_new_review_and_retry_keeps_cpu_permission(
        self,
    ):
        from bio_neb.parabricks.review import answer_review

        cpu_request = self.stop()
        cpu_answer = {"action": "approve_cpu_preparation", "by": "owner"}
        approved = answer_review(self.approved.journey_file, cpu_request, cpu_answer)
        failure_id, request_id = self.failure_stop("cpu_preparation")
        pending = self.path.state
        before = self.path.file.read_bytes()
        replayed = answer_review(self.approved.journey_file, cpu_request, cpu_answer)
        self.assertEqual(replayed, pending)
        self.assertEqual(replayed.failure_review.failure_event_id, failure_id)
        self.assertEqual(self.path.file.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "already answered differently"):
            answer_review(self.approved.journey_file, cpu_request, {"action": "abort"})
        self.assertEqual(self.path.file.read_bytes(), before)
        retried = answer_review(
            self.approved.journey_file, request_id, {"action": "retry"}
        )
        self.assertEqual(retried.review, approved.review)
        self.assertEqual(
            retried.latest_event_ids["CpuPreparationReviewAnswered"],
            approved.latest_event_ids["CpuPreparationReviewAnswered"],
        )
        self.assertEqual(retried.failure_event_ids, (failure_id,))
        self.assertEqual(retried.resolved_failure_ids, (failure_id,))
        later_failure, later_request = self.failure_stop("cpu_preparation")
        self.assertNotEqual(later_failure, failure_id)
        self.assertNotEqual(later_request, request_id)
        self.assertEqual(self.path.state.failure_event_ids, (failure_id, later_failure))
        self.assertEqual(self.path.state.unresolved_failure_ids, (later_failure,))
        before = self.path.file.read_bytes()
        answer_review(self.approved.journey_file, request_id, {"action": "retry"})
        self.assertEqual(self.path.file.read_bytes(), before)

    def test_an_answer_breaking_its_contract_is_a_command_refusal(self):
        # The caller's input, not the coordinator's own validation.
        from bio_neb.failure import CommandInputInvalid
        from bio_neb.parabricks.review import answer_review

        request_id = self.stop()
        before = self.path.file.read_bytes()
        with self.assertRaises(CommandInputInvalid) as refused:
            answer_review(
                self.approved.journey_file, request_id, {"action": "approve", "by": ""}
            )
        self.assertEqual(
            (refused.exception.code, refused.exception.fields),
            ("command_invalid", ("answer.response.by",)),
        )
        self.assertEqual(self.path.file.read_bytes(), before)

    def test_an_answer_beyond_its_declared_authority_is_refused_before_the_pass(
        self,
    ):
        # An undeclared action is the caller's to correct, never a held Run.
        from bio_neb.application.advancement import (
            AnswerReview,
            ExecutionPathReviewAnswer,
            advance,
        )

        _, request_id = self.failure_stop("admit_candidate")
        # An admission fault after approval offers retry or abort only; it amends
        # no location.
        request = JourneyPath.open(self.path.file).state.failure_review
        self.assertEqual(
            (request.legal_actions, request.amendable), (("retry", "abort"), ())
        )
        before = self.path.file.read_bytes()
        for response, field in (
            (
                {"action": "approve_cpu_preparation", "by": "owner"},
                "answer.response.action",
            ),
            (
                {
                    "action": "amend",
                    "field": "location",
                    "value": "s3://prepared/",
                    "by": "owner",
                },
                "answer.response.action",
            ),
        ):
            with self.subTest(field):
                with (
                    self.offline_continuation(),
                    self.assertRaises(ValueError) as refused,
                ):
                    advance(
                        AnswerReview(
                            journey_file=self.approved.journey_file,
                            request_event_id=request_id,
                            answer=ExecutionPathReviewAnswer(
                                kind="execution_path", response=response
                            ),
                        )
                    )
                self.assertEqual(
                    (refused.exception.code, refused.exception.fields),
                    ("review_answer_exceeds_authority", (field,)),
                )
                self.assertEqual(self.path.file.read_bytes(), before)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.files = offline_galaxy_files(root / "galaxy")
        self.approved = approved_input(
            root / "run", galaxy=GalaxyOpener(files=self.files)
        )
        # Galaxy serves no FASTA until an admission failure publishes it, so the
        # first Pass misses (an unreachable file).
        self.unserved = {k: v for k, v in self.files.items() if k != GALAXY_FASTA}
        self.path = JourneyPath.open(Path(self.approved.journey_file))
        self.failing_stage = None
        self.candidate_published = False
        self.provider = None

    def factory(self, plan, journal):
        """One offline provider across Passes; a scripted rejection fails one Pass.

        The bound prebuilt index misses, its FASTA unserved, until an admission
        failure publishes it, so a retry after that failure admits it.
        """
        scripted = seed_cpu(input_storage_provider(plan, journal), plan, self.approved)
        if self.provider is None:
            self.provider = scripted
        else:
            self.provider.scripts.update(scripted.scripts)
            for key, content in scripted.objects.items():
                self.provider.objects.setdefault(key, content)
        provider = self.provider
        failing, self.failing_stage = self.failing_stage, None
        if failing == "admit_candidate":
            self.candidate_published = True
        provider.sources.update(
            self.files if self.candidate_published else self.unserved
        )
        provider.sources[plan.reference.source_url] = SOURCE_GZIP
        if failing == "admit_candidate":
            provider.scripts["admit_candidate", "place_candidate"] = deque(
                [ProviderError("rejected", "copy_refused", {"operation": "copy"})]
            )
        elif failing == "cpu_preparation":
            provider.scripts["cpu_preparation", "submit_cpu"] = deque(
                [ProviderError("rejected", "copy_refused")]
            )
        return h100_provider(provider, plan, self.approved)

    @contextmanager
    def offline_continuation(self):
        """Keep an answer's dispatched Execution, and the Investigation it begins,
        on this case's fake providers."""
        from bio_neb.application.advancement import use_run_advancement_dependencies
        from tests.sequential_investigation_support import (
            offline_investigation_dependencies,
        )

        with use_run_advancement_dependencies(
            parabricks_infrastructure_factory=self.factory,
            **offline_investigation_dependencies(),
        ):
            yield

    def execute(self):
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        self.path = JourneyPath.open(Path(self.approved.journey_file))
        self.state = state
        self.plan = plan_execution(
            self.approved,
            RetainedExecution(state, self.approved.authority, self.approved.proposal),
        )
        return state

    def stop(self):
        """Run the first Pass to its unserved-index CPU preparation Review."""
        state = self.execute()
        self.assertEqual(state.status, "cpu_preparation_review_pending")
        return state.latest_event_ids["CpuPreparationReviewRequested"]

    def _cli_answer(self, request_id, response):
        import io
        import json
        from contextlib import redirect_stderr, redirect_stdout
        from unittest.mock import patch

        from bio_neb.cli import main

        output, errors = io.StringIO(), io.StringIO()
        with (
            patch(
                "bio_neb.parabricks.operation.execute_parabricks",
                side_effect=AssertionError("the answer dispatched Execution"),
            ),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            code = main(
                [
                    "answer",
                    "--journey-file",
                    self.approved.journey_file,
                    "--request-event-id",
                    request_id,
                    "--response-json",
                    json.dumps(response),
                ]
            )
        return code, output.getvalue(), errors.getvalue()

    def test_cli_replays_an_answered_cpu_review_without_dispatch(self):
        from bio_neb.parabricks.review import answer_review

        request_id = self.stop()
        answer_review(self.approved.journey_file, request_id, {"action": "approve"})
        before = self.path.file.read_bytes()
        code, _, errors = self._cli_answer(request_id, {"action": "approve"})
        self.assertEqual(code, 0, errors)
        self.assertEqual(self.path.file.read_bytes(), before)

    def test_cli_refuses_an_unknown_request_id_with_a_typed_refusal(self):
        import json
        import uuid

        self.stop()
        before = self.path.file.read_bytes()
        unknown = str(uuid.uuid4())
        code, output, errors = self._cli_answer(unknown, {"action": "approve"})
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        refusal = json.loads(errors)
        self.assertEqual(
            (refusal["frontier"], refusal["code"], refusal["request_event_id"]),
            ("refused", "review_answer_request_unknown", unknown),
        )
        self.assertEqual(self.path.file.read_bytes(), before)

    def test_cli_refuses_an_amendment_of_the_cpu_review_and_leaves_the_review(self):
        """Owner decision 2026-10-29 (#464): the CPU preparation Review offers
        abort or one CPU job and amends nothing, so an amendment is the caller's
        command input to correct: refused without a hold or any write (#214)."""
        import json

        request_id = self.stop()
        request = JourneyPath.open(self.path.file).state.review
        self.assertEqual(request.legal_actions, ("abort", "approve_cpu_preparation"))
        before = self.path.file.read_bytes()
        for field in ("location", "input_correction"):
            with self.subTest(field=field):
                code, output, errors = self._cli_answer(
                    request_id, {"action": "amend", "field": field, "value": {}}
                )
                self.assertEqual((code, output), (2, ""))
                refusal = json.loads(errors)
                self.assertEqual(refusal["code"], "command_invalid")
                self.assertIn("answer.response.action", refusal["fields"])
        self.assertEqual(self.path.file.read_bytes(), before)
        self.assertEqual(
            self.path.state.latest_event_ids["CpuPreparationReviewRequested"],
            request_id,
        )

    def test_typed_answers_name_an_unknown_id_apart_from_another_familys_review(self):
        import uuid

        from bio_neb.application.advancement import (
            AnswerReview,
            ExecutionPathReviewAnswer,
            JunctionReviewAnswer,
            ReviewAnswerRefused,
            advance,
        )

        request_id = self.stop()
        before = self.path.file.read_bytes()
        unknown = str(uuid.uuid4())
        for answer, kind, request, code in (
            (
                ExecutionPathReviewAnswer,
                "execution_path",
                unknown,
                "review_answer_request_unknown",
            ),
            (
                JunctionReviewAnswer,
                "junction",
                unknown,
                "review_answer_request_unknown",
            ),
            (
                JunctionReviewAnswer,
                "junction",
                request_id,
                "review_answer_kind_mismatch",
            ),
        ):
            with self.subTest(kind=kind, request=request):
                with self.assertRaises(ReviewAnswerRefused) as refused:
                    advance(
                        AnswerReview(
                            journey_file=self.approved.journey_file,
                            request_event_id=request,
                            answer=answer(kind=kind, response={"action": "approve"}),
                        )
                    )
                self.assertEqual(refused.exception.code, code)
                self.assertEqual(self.path.file.read_bytes(), before)

    def test_answer_records_exact_human_authority_and_identical_replay_is_a_noop(self):
        from bio_neb.parabricks.review import answer_review

        request_id = self.stop()
        answer = {"action": "approve_cpu_preparation", "by": "owner"}
        state = answer_review(self.approved.journey_file, request_id, answer)
        self.assertEqual(state.status, "cpu_preparation_authorized")
        self.assertEqual(state.pass_, 2)
        event = JourneyPath.open(self.path.file).read()[-1]
        self.assertEqual(
            (event.kind, event.grounding, event.rests_on),
            ("authority", "human", (request_id,)),
        )
        before = self.path.file.read_bytes()
        self.assertEqual(
            answer_review(self.approved.journey_file, request_id, answer), state
        )
        self.assertEqual(self.path.file.read_bytes(), before)
        with self.assertRaises(ValueError):
            answer_review(
                self.approved.journey_file,
                request_id,
                {"action": "abort", "by": "owner"},
            )
        self.assertEqual(self.path.file.read_bytes(), before)
