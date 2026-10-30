"""Input milestone proof; historical approval facts and scripted large transfers.

The scripts attest the old fixture's declared sizes and digests without reading
the large FASTQs/reference. Real stream verification is tested at NebiusProvider.

The approved Run names no prebuilt index, so after input storage it prepares its
reference on CPU (owner, 2026-10-29, #464). `input_provider` scripts that CPU job's
submission as refused: each Run stops at the CPU stage's failure Review, a designed
stop with nothing after input storage verified. Without a supplied local FASTA the
CPU job fetches the source (#300), so input storage places only the FASTQs and its
manifest.
"""

import hashlib
import io
import json
import tempfile
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from bio_neb.cli import main
from bio_neb.failure import ProviderError
from bio_neb.journey_path import JourneyPath, ReferenceAcquisition, project_execution
from bio_neb.parabricks.operation import execute_parabricks
from bio_neb.providers.fake import FakeProvider
from bio_neb.providers.ports import Receipt
from tests.execution_contact_journal import (
    execution_contact_row,
    execution_contact_rows,
    rewrite_journey,
)
from tests.parabricks_operation_fakes import (
    execute_and_read_path,
    input_storage_provider,
)
from tests.reference_fakes import SOURCE_GZIP, SOURCE_PLAIN
from tests.run_prerequisites import RunPrerequisites


def input_provider(plan, journal):
    """Input storage's scripted effects; the CPU job's submission is refused."""
    from collections import deque

    provider = input_storage_provider(plan, journal)
    provider.scripts["cpu_preparation", "submit_cpu"] = deque(
        [ProviderError("rejected", "cpu_job_refused")]
    )
    return provider


def _with_local_reference_replica(configuration):
    """Select a local replica of the fake source's reviewed reference bytes."""
    from bio_neb.parabricks.configuration import ParabricksConfiguration

    material = configuration.model_dump(mode="json")
    launch = material["launch"]
    # A path only: upload checks the approved md5 and size while streaming.
    launch["local_reference_source"] = {
        "path": str(Path(launch["work_directory"]) / "reference.gz")
    }
    return ParabricksConfiguration.model_validate(material)


class InputStorageTests(RunPrerequisites, unittest.TestCase):
    def test_reviewed_sources_acquire_identity_in_upload_without_rewriting_approval(
        self,
    ):
        self._exercise_acquisition_retry(None)

    def test_later_transfer_failure_reuses_acquired_identity_across_attempts(self):
        self._exercise_acquisition_retry("transient")

    def test_human_retry_reuses_acquired_identity_from_closed_pass(self):
        self._exercise_acquisition_retry("rejected")

    def _exercise_acquisition_retry(self, failure_class):
        import hashlib

        self.approved = self.prerequisite("approved").value
        self.file = Path(self.approved.journey_file)
        self.prefix = self.file.read_bytes()
        before = self.approved.proposal.model_dump(mode="json")
        self.assertTrue(
            all(
                item.sha256 is None
                for item in self.approved.proposal.execution_package.inputs
            )
        )
        transferred = []
        expected = {}
        failed = False
        real = FakeProvider()

        def factory(plan, journal):
            provider = input_provider(plan, journal)
            # Small real FASTQ bytes exercise the stateful effect adapter; only the
            # separate large reference remains a scripted provider receipt.
            source_bytes = {
                item.registered_path: Path(item.registered_path).read_bytes()
                for item in plan.fastqs
            }
            real.sources.update(source_bytes)
            original = provider.for_step

            def for_step(stage, step):
                ports = original(stage, step)
                ports.store.head = lambda *a, **kw: self.fail("verification-only HEAD")
                if (stage, step) == ("input_storage", "place_inputs"):
                    put = ports.store.put

                    def upload(bucket, key, source):
                        nonlocal failed
                        if source.locator in source_bytes:
                            if failure_class and len(transferred) == 1 and not failed:
                                failed = True
                                raise ProviderError(
                                    failure_class, "later_transfer_failed"
                                )
                            receipt = real.store.put(bucket, key, source)
                            transferred.append(source.locator)
                            expected[key] = hashlib.sha256(
                                source_bytes[source.locator]
                            ).hexdigest()
                            self.assertIsNone(
                                receipt.values["artifact_proof"]["expected_identity"]
                            )
                            return receipt
                        # Only the reference transfer uses a scripted identity.
                        while len(ports.store.script) > 1:
                            ports.store.script.popleft()
                        return put(bucket, key, source)

                    ports.store.put = upload
                return ports

            provider.for_step = for_step
            return provider

        result = execute_parabricks(self.approved, infrastructure_factory=factory)
        if failure_class == "rejected":
            from bio_neb.parabricks.review import answer_review

            self.assertEqual(result.status, "execution_review_pending")
            stopped = JourneyPath.open(self.file)
            self.assertNotIn("InputsVerified", stopped.state.latest_event_ids)
            answer_review(str(self.file), result.request_event_id, {"action": "retry"})
            result = execute_parabricks(self.approved, infrastructure_factory=factory)
        self.assertEqual(result.status, "execution_review_pending")
        self.assertEqual(len(transferred), 2)
        self.assertEqual(failed, failure_class is not None)
        state = JourneyPath.open(self.file).state
        fastqs = [p for p in state.inputs.pointers if p.role in {"read_1", "read_2"}]
        self.assertEqual(len(fastqs), 2)
        self.assertTrue(
            next(p for p in fastqs if p.role == "read_1").proof_ref.startswith("cj:1:")
        )
        for pointer in fastqs:
            self.assertEqual(
                pointer.content_id,
                "sha256:" + expected[pointer.location.split("/", 3)[3]],
            )
            self.assertEqual(
                pointer.producer.authority_id, self.approved.authority.authority_id
            )
        self.assertEqual(state.bound.approved_input["proposal"], before)
        self.assertTrue(self.file.read_bytes().startswith(self.prefix))

    def test_created_inputs_retain_upload_proof_without_remote_rechecks(self):
        self.approve()

        def factory(plan, journal):
            provider = input_provider(plan, journal)
            original = provider.for_step

            def for_step(stage, step):
                ports = original(stage, step)
                ports.store.head = lambda *a, **kw: self.fail(
                    "ordinary input reuse must not issue a verification HEAD"
                )
                return ports

            provider.for_step = for_step
            return provider

        result = execute_parabricks(self.approved, infrastructure_factory=factory)
        self.assertEqual(result.status, "execution_review_pending")
        state = JourneyPath.open(self.file).state
        self.assertEqual(len(state.inputs.pointers), 3)
        for pointer in state.inputs.pointers:
            self.assertTrue(pointer.proof_ref.startswith("cj:1:"))

    def test_public_queries_reopen_semantic_path_without_contact_or_mutation(self):
        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.application.queries import (
            ObserveRun,
            ReviewJourneyQuery,
            observe,
            review_journey,
        )

        self.approve()
        expected = execute_parabricks(
            self.approved, infrastructure_factory=input_provider
        )
        before = {
            path: path.read_bytes()
            for path in self.file.parent.rglob("*")
            if path.is_file()
        }
        with patch(
            "bio_neb.parabricks.operation._make_provider",
            side_effect=AssertionError("query constructed execution infrastructure"),
        ):
            observed = observe(ObserveRun(journey_file=str(self.file)))
            reviewed = review_journey(ReviewJourneyQuery(journey_file=str(self.file)))
            resumed = advance(ResumeRun(journey_file=str(self.file)))
        self.assertEqual(observed, expected)
        self.assertEqual(reviewed, expected)
        self.assertEqual(resumed, expected)
        self.assertEqual(observed.disposition, "incomplete")
        self.assertEqual(observed.status, "execution_review_pending")
        self.assertIsNotNone(observed.pending_review)
        self.assertEqual(observed.request_event_id, expected.request_event_id)
        self.assertEqual(
            {
                path: path.read_bytes()
                for path in self.file.parent.rglob("*")
                if path.is_file()
            },
            before,
        )

    def test_public_queries_refuse_damaged_path_instead_of_falling_back_to_planning(
        self,
    ):
        from bio_neb.application.queries import (
            ObserveRun,
            ReviewJourneyQuery,
            observe,
            review_journey,
        )

        self.approve()
        execute_and_read_path(self.approved, infrastructure_factory=input_provider)

        def rest_outside_the_path(events):
            # The last path fact rests on the approval: an earlier Journey event,
            # but outside the Execution path, with its envelope kept consistent.
            approval = next(
                event["event_id"]
                for event in events
                if event["event_type"] == "ExecutionHitlResponded"
            )
            outcome = [
                event
                for event in events
                if event["event_type"] == "ExecutionPathRecorded"
            ][-1]
            outcome["parent_ids"] = outcome["payload"]["rests_on"] = [approval]

        damaged = rewrite_journey(self.file, rest_outside_the_path)
        for reader, query in (
            (observe, ObserveRun),
            (review_journey, ReviewJourneyQuery),
        ):
            with self.subTest(reader=reader.__name__):
                with self.assertRaisesRegex(ValueError, "rests_on outside"):
                    reader(query(journey_file=str(self.file)))
                self.assertEqual(self.file.read_bytes(), damaged)

    def test_manifest_publication_loss_retries_without_overwriting_prior_proof(self):
        from bio_neb.parabricks.review import answer_review

        self.approve()
        provider = None
        lose_publication = True

        def factory(plan, journal):
            nonlocal provider
            scripted = input_provider(plan, journal)
            if provider is None:
                provider = scripted
            else:
                provider.scripts.update(scripted.scripts)
            original = provider.for_step

            def for_step(stage, step):
                ports = original(stage, step)
                if (stage, step) == ("input_storage", "publish_manifest"):
                    put = ports.store.put

                    def checked(*args, **kwargs):
                        nonlocal lose_publication
                        result = put(*args, **kwargs)
                        if lose_publication:
                            lose_publication = False
                            raise ProviderError("rejected", "manifest_publication_lost")
                        return result

                    ports.store.put = checked
                return ports

            provider.for_step = for_step
            return provider

        pending = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(pending.failures[-1].detail, "manifest_publication_lost")
        before = dict(provider.objects)
        self.assertEqual(len(before), 1)
        answered = answer_review(
            self.approved.journey_file,
            pending.latest_event_ids["FailureReviewRequested"],
            {"action": "retry"},
        )
        done = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(done.inputs.state, "verified")
        self.assertIn(
            answered.latest_event_ids["FailureReviewAnswered"],
            done.rests_on[done.latest_event_ids["InputsVerified"]],
        )
        self.assertEqual(done.status, "execution_review_pending")
        self.assertEqual(provider.objects, before)
        document = json.loads(next(iter(before.values())))
        self.assertEqual(
            [p["proof_ref"] for p in document["inputs"]],
            [p.proof_ref for p in done.inputs.pointers if p.role != "input_manifest"],
        )
        events = JourneyPath.open(Path(self.approved.journey_file)).read()
        enabling = answered.latest_event_ids["FailureReviewAnswered"]
        damaged = [
            event.model_copy(
                update={"rests_on": tuple(i for i in event.rests_on if i != enabling)}
            )
            if event.event_type == "InputsVerified"
            else event
            for event in events
        ]
        with self.assertRaisesRegex(ValueError, "enabling stage answer"):
            project_execution(damaged)

    def test_answered_input_failure_retries_in_the_next_pass_then_reaches_cpu(self):
        from bio_neb.parabricks.review import answer_review

        self.approve()
        state = execute_and_read_path(
            self.approved,
            infrastructure_factory=lambda plan, journal: FakeProvider(
                {
                    ("input_storage", "preflight"): [
                        ProviderError("rejected", "fix_credentials")
                    ]
                }
            ),
        )
        self.assertIsNotNone(state.failure_review)
        answered = answer_review(
            self.approved.journey_file,
            state.latest_event_ids["FailureReviewRequested"],
            {"action": "retry"},
        )
        self.assertIsNone(answered.review)
        state = execute_and_read_path(
            self.approved, infrastructure_factory=input_provider
        )
        self.assertEqual(state.pass_, 2)
        self.assertIsNone(state.outcome.milestone)
        self.assertEqual(state.inputs.state, "verified")
        # The input failure is resolved; the next is the CPU job's own refusal.
        self.assertEqual(len(state.failures), 2)
        self.assertEqual(len(state.unresolved_failure_ids), 1)
        self.assertEqual(state.failure_review.stage, "cpu_preparation")
        self.assertIsNone(state.review)

    def test_public_input_milestone_retains_selected_local_reference_replica(self):
        # The replica is part of the reviewed configuration, so it is approved
        # with the Run rather than edited into the launch afterwards.
        approved = self.approve(configure=_with_local_reference_replica)
        replica = Path(approved.launch.work_directory) / "reference.gz"

        def selected_provider(plan, journal):
            self.assertEqual(plan.input_source.kind, "local")
            self.assertEqual(plan.input_source.locator, str(replica))
            return input_provider(plan, journal)

        state = execute_and_read_path(
            approved, infrastructure_factory=selected_provider
        )
        self.assertIsNone(state.outcome.milestone)
        self.assertEqual(
            state.bound.approved_input["launch"]["local_reference_source"],
            {"path": str(replica)},
        )

    def test_offline_harness_sources_hex_moves_small_bytes_and_refuses_changed_source(
        self,
    ):
        import hashlib

        from bio_neb.content import canonical_json_digest
        from bio_neb.devtools.stage_harness import main as harness_main
        from bio_neb.parabricks.execution_proposal import execution_authority_id

        self.approve()
        # This derived approval exists only inside the offline harness document.
        # Its small byte identities are synthetic; the retained planning prefix stays intact.
        approved = self.approved.model_dump(mode="json")
        proposal = approved["proposal"]
        package = proposal["execution_package"]
        sources = {}
        for item in package["inputs"]:
            content = f"synthetic-{item['role']}".encode()
            sources[item["registered_path"]] = content
            item.update(
                byte_count=len(content), sha256=hashlib.sha256(content).hexdigest()
            )
            item["file_id"] = "execution-file:" + canonical_json_digest(
                {k: v for k, v in item.items() if k != "file_id"}
            )
        # The discovered reference's own small gzip bytes, as its md5 approves them,
        # from a supplied local replica the controller streams (without one the CPU
        # job fetches the source).
        reference_bytes = SOURCE_GZIP
        replica = str(Path(self.temp.name) / "replica" / "reference.gz")
        for launch in (approved["launch"], proposal["configuration"]["launch"]):
            launch["local_reference_source"] = {"path": replica}
        sources[replica] = reference_bytes
        package["package_id"] = "execution-package:" + canonical_json_digest(
            {k: v for k, v in package.items() if k != "package_id"}
        )
        proposal["proposal_id"] = "execution-proposal:" + canonical_json_digest(
            {k: v for k, v in proposal.items() if k != "proposal_id"}
        )
        authority = approved["authority"]
        authority["proposal_id"] = proposal["proposal_id"]
        authority["authority_id"] = execution_authority_id(
            authority["proposal_id"],
            authority["request_event_id"],
            authority["approval_event_id"],
        )
        for name, corrupt in (("small-bytes", False), ("changed-source", True)):
            with self.subTest(case=name):
                source_hex = {key: value.hex() for key, value in sources.items()}
                if corrupt:
                    source_hex[replica] = b"changed".hex()
                root = Path(self.temp.name) / name
                document = {
                    "approved": approved,
                    "run_root": str(root),
                    "sources_hex": source_hex,
                }
                file = Path(self.temp.name) / f"{name}.json"
                file.write_text(json.dumps(document))
                output = io.StringIO()
                with (
                    redirect_stdout(output),
                    patch(
                        "bio_neb.providers.nebius.NebiusProvider",
                        side_effect=AssertionError(
                            "offline harness reached live provider"
                        ),
                    ),
                ):
                    code = harness_main(
                        [
                            "--stage",
                            "input-storage",
                            "--mode",
                            "offline",
                            "--input",
                            str(file),
                        ]
                    )
                result = json.loads(output.getvalue())
                state = JourneyPath.open(root / "journey.jsonl").state
                if corrupt:
                    self.assertEqual(code, 2)
                    self.assertNotIn("InputsVerified", state.latest_event_ids)
                    self.assertEqual(
                        result["failures"][0]["detail"], "source_identity_mismatch"
                    )
                else:
                    self.assertEqual(code, 0)
                    self.assertEqual(len(state.inputs.pointers), 4)
                    for pointer in state.inputs.pointers:
                        if pointer.role == "input_manifest":
                            continue
                        expected = (
                            reference_bytes
                            if pointer.role == "reference_source"
                            else sources[
                                next(
                                    item["registered_path"]
                                    for item in package["inputs"]
                                    if item["role"] == pointer.role
                                )
                            ]
                        )
                        self.assertEqual(pointer.bytes, len(expected))
                        self.assertEqual(
                            pointer.content_id,
                            "sha256:" + hashlib.sha256(expected).hexdigest(),
                        )
        self.assertEqual(self.file.read_bytes(), self.prefix)

    def test_contact_rows_retain_content_identity_and_verification_evidence(self):
        self.approve()
        state = execute_and_read_path(
            self.approved, infrastructure_factory=input_provider
        )
        for pointer in state.inputs.pointers:
            row = execution_contact_row(
                state.outcome.contact_journal, pointer.proof_ref
            )
            proof = json.loads(Path(row["receipt"]["ref"]).read_text())
            self.assertEqual(proof["operation"], "put")
            self.assertEqual(
                proof["values"]["sha256"], pointer.content_id.removeprefix("sha256:")
            )
            self.assertEqual(proof["values"]["bytes"], pointer.bytes)

    def test_input_manifest_is_readable_from_the_journey_without_provider_contact(self):
        self.approve()
        state = execute_and_read_path(
            self.approved, infrastructure_factory=input_provider
        )
        manifest = next(
            pointer
            for pointer in state.inputs.pointers
            if pointer.role == "input_manifest"
        )
        body = JourneyPath.open(self.file).read_artifact(manifest)
        self.assertEqual(json.loads(body)["execution_id"], self.approved.execution_id)
        self.assertEqual(len(json.loads(body)["inputs"]), 2)

    def test_quiet_transfer_emits_disposable_heartbeat(self):
        import time

        approved = self.approve(heartbeat_interval_seconds=0.01)
        seen = []

        def slow(plan, journal):
            time.sleep(0.04)
            return input_provider(plan, journal)

        state = execute_and_read_path(
            approved, infrastructure_factory=slow, progress=seen.append
        )
        self.assertIn("execution_flow_heartbeat", [row.get("kind") for row in seen])
        self.assertEqual(state.status, "execution_review_pending")

    def test_a_second_approval_cannot_follow_new_execution_facts(self):
        # The single-stream Journey has no planning-after-Execution rule; the
        # closest current guard is the approval: a second Execution approval
        # appended after the path names no pending request, and the public
        # queries refuse it without writing.
        from bio_neb.application.queries import (
            ObserveRun,
            ReviewJourneyQuery,
            observe,
            review_journey,
        )

        self.approve()
        execute_and_read_path(self.approved, infrastructure_factory=input_provider)

        def approve_again(events):
            approval = next(
                event
                for event in events
                if event["event_type"] == "ExecutionHitlResponded"
            )
            events.append(
                {
                    **approval,
                    "event_id": str(uuid.uuid4()),
                    "parent_ids": [events[-1]["event_id"]],
                }
            )

        damaged = rewrite_journey(self.file, approve_again)
        for reader, query in (
            (observe, ObserveRun),
            (review_journey, ReviewJourneyQuery),
        ):
            with self.subTest(reader=reader.__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "Execution HITL response did not target the pending request",
                ):
                    reader(query(journey_file=str(self.file)))
                self.assertEqual(self.file.read_bytes(), damaged)

    def test_interrupted_first_pass_reopens_as_incomplete_and_requires_new_run(self):
        from bio_neb.parabricks.result import read_parabricks_result

        self.approve()

        def interrupted(plan, journal):
            raise RuntimeError("offline factory interruption")

        with self.assertRaisesRegex(RuntimeError, "offline factory interruption"):
            execute_and_read_path(self.approved, infrastructure_factory=interrupted)
        before = self.file.read_bytes()
        retained = read_parabricks_result(str(self.file))
        self.assertEqual(retained.disposition, "incomplete")
        self.assertIsNone(retained.outcome_event_id)
        with self.assertRaisesRegex(
            ValueError, "unfinished_execution_requires_new_run"
        ):
            execute_and_read_path(
                self.approved,
                infrastructure_factory=lambda *_: self.fail("unexpected contact"),
            )
        self.assertEqual(self.file.read_bytes(), before)

    def test_artifact_sync_failure_never_records_its_verified_fact(self):
        import os

        self.approve()
        real_fsync = os.fsync
        failed = []

        def fail_artifact_sync(fd):
            # Only the first artifact the stage retains fails; the Pass's own
            # closing artifact is written afterwards and must still sync.
            target = Path(os.readlink(f"/proc/self/fd/{fd}"))
            if target.parent.name == "sha256" and not failed:
                failed.append(target)
                self.assertNotIn(
                    b'"event_type":"InputsVerified"', self.file.read_bytes()
                )
                raise OSError("offline artifact sync failure")
            return real_fsync(fd)

        with patch("os.fsync", side_effect=fail_artifact_sync):
            with self.assertRaisesRegex(OSError, "offline artifact sync failure"):
                execute_and_read_path(
                    self.approved, infrastructure_factory=input_provider
                )
        self.assertEqual(len(failed), 1)
        state = JourneyPath.open(self.file).state
        self.assertNotIn("InputsVerified", state.latest_event_ids)
        self.assertIsNone(state.outcome)
        before = self.file.read_bytes()
        with self.assertRaisesRegex(
            ValueError, "unfinished_execution_requires_new_run"
        ):
            execute_parabricks(
                self.approved,
                infrastructure_factory=lambda *_: self.fail("unexpected contact"),
            )
        self.assertEqual(self.file.read_bytes(), before)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def approve(self, *, configure=None, **launch_overrides):
        """Restore the class's approved Run for this case's final launch choices."""
        self.approved = self.prerequisite(
            "approved", configure=configure, **launch_overrides
        ).value
        self.file = Path(self.approved.journey_file)
        self.prefix = self.file.read_bytes()
        return self.approved

    def test_public_operation_preserves_inputs_and_reaches_the_cpu_jobs_review(
        self,
    ):
        self.approve()
        state = execute_and_read_path(
            self.approved, infrastructure_factory=input_provider
        )
        self.assertEqual(state.status, "execution_review_pending")
        self.assertIsNone(state.outcome.milestone)
        self.assertEqual(state.disposition, "incomplete")
        self.assertTrue(self.file.read_bytes().startswith(self.prefix))
        path = JourneyPath.open(self.file)
        self.assertEqual(
            [e.event_type for e in path.read()],
            [
                "ExecutionBound",
                "InputsVerified",
                "StageFailed",
                "FailureReviewRequested",
                "ExecutionOutcome",
            ],
        )
        self.assertEqual(state.failure_review.stage, "cpu_preparation")
        self.assertEqual(len(state.inputs.pointers), 3)
        rows = execution_contact_rows(state.outcome.contact_journal)
        self.assertTrue(rows[-1]["complete"])
        self.assertEqual(rows[-1]["row_kind"], "final")
        self.assertTrue(
            all(p.proof_ref.startswith("cj:1:") for p in state.inputs.pointers)
        )
        self.assertEqual(state, project_execution(path.read()))

    def test_pending_review_reopens_without_constructing_provider_or_continuing(self):
        self.approve()
        first = execute_and_read_path(
            self.approved, infrastructure_factory=input_provider
        )
        with patch(
            "bio_neb.parabricks.operation._make_provider",
            side_effect=AssertionError("contact"),
        ):
            self.assertEqual(execute_and_read_path(self.approved), first)
        before = self.file.read_bytes()
        reopened = execute_parabricks(
            self.approved,
            infrastructure_factory=lambda *_: self.fail("unexpected contact"),
        )
        self.assertEqual(
            (reopened.status, reopened.request_event_id),
            (
                "execution_review_pending",
                first.latest_event_ids["FailureReviewRequested"],
            ),
        )
        self.assertEqual(self.file.read_bytes(), before)

    def test_cli_interface_runs_the_new_operation(self):
        self.approve()
        approved_file = Path(self.temp.name) / "approved.json"
        approved_file.write_text(self.approved.model_dump_json())
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                ["resume", "--journey-file", self.approved.journey_file],
                parabricks_infrastructure_factory=input_provider,
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(stdout.getvalue())["status"], "execution_review_pending"
        )

    def test_historical_execution_binding_cannot_be_treated_as_fresh_approval(self):
        from bio_neb.journey import Journey

        self.approve()
        retained = Journey.open(self.file)
        retained.record(
            "ParabricksExecutionBound",
            {"approved_input": self.approved.model_dump(mode="json")},
            parent_ids=[retained.read()[-1]["event_id"]],
        )
        before = self.file.read_bytes()
        with patch(
            "bio_neb.parabricks.operation._make_provider",
            side_effect=AssertionError("historical binding reached provider"),
        ):
            for launch in (
                self.approved.launch,
                self.approved.launch.model_copy(
                    update={"object_scope": "changed-scope"}
                ),
            ):
                with self.subTest(scope=launch.object_scope):
                    with self.assertRaisesRegex(
                        ValueError, "legacy_execution_requires_migration"
                    ):
                        execute_and_read_path(
                            self.approved.model_copy(update={"launch": launch})
                        )
        self.assertEqual(self.file.read_bytes(), before)
        self.assertFalse((self.file.parent / "execution-contact").exists())

    def test_missing_current_approval_is_refused_before_provider_construction(self):
        self.approve()
        self.file.write_bytes(b"")
        with patch(
            "bio_neb.parabricks.operation._make_provider",
            side_effect=AssertionError("contact"),
        ):
            with self.assertRaisesRegex(ValueError, "lineage"):
                execute_and_read_path(self.approved)
        self.assertEqual(self.file.read_bytes(), b"")

    def test_discovered_source_is_acquired_against_its_approved_md5(self):
        """Upload checks the approved publisher md5 and records the acquired bytes;
        a changed source stops the Run with a typed cause and keeps the approval."""
        changed = bytearray(SOURCE_GZIP)
        changed[12] ^= 1
        for name, served in (("approved", SOURCE_GZIP), ("changed", bytes(changed))):
            with self.subTest(source=name):
                # The controller streams a supplied local replica; without one the
                # CPU job fetches the source.
                self.approve(configure=_with_local_reference_replica)

                def serving(plan, journal, served=served):
                    provider = input_provider(plan, journal)
                    # The reference transfers its bytes; FASTQs stay scripted.
                    provider.scripts["input_storage", "place_inputs"].pop()
                    provider.sources[plan.input_source.locator] = served
                    return provider

                state = execute_and_read_path(
                    self.approved, infrastructure_factory=serving
                )
                if name == "approved":
                    self.assertEqual(
                        state.inputs.reference_acquisition,
                        ReferenceAcquisition(
                            source_md5=hashlib.md5(SOURCE_GZIP).hexdigest(),
                            source_byte_count=len(SOURCE_GZIP),
                            source_sha256=hashlib.sha256(SOURCE_GZIP).hexdigest(),
                            plain_fasta_byte_count=len(SOURCE_PLAIN),
                            plain_fasta_sha256=hashlib.sha256(SOURCE_PLAIN).hexdigest(),
                        ),
                    )
                    continue
                self.assertEqual(state.status, "execution_review_pending")
                self.assertEqual(state.failures[-1].detail, "source_identity_mismatch")
                self.assertNotIn("InputsVerified", state.latest_event_ids)
                self.assertTrue(self.file.read_bytes().startswith(self.prefix))
                self.assertEqual(
                    state.bound.approved_input,
                    self.approved.model_dump(mode="json"),
                )

    def test_upload_identity_mismatch_fails_without_inputs_fact_or_manifest(self):
        self.approve()
        providers = []

        def drifting(plan, journal):
            provider = input_provider(plan, journal)
            from collections import deque

            provider.scripts["input_storage", "place_inputs"] = deque(
                [Receipt(values={"bytes": 1, "sha256": "0" * 64})]
            )
            providers.append(provider)
            return provider

        state = execute_and_read_path(self.approved, infrastructure_factory=drifting)
        # The semantic mismatch asks its Review on default policy.
        self.assertEqual(state.status, "execution_review_pending")
        self.assertEqual(state.failures[-1].detail, "upload_identity_mismatch")
        self.assertNotIn("InputsVerified", state.latest_event_ids)
        self.assertFalse(providers[0].objects)
        self.assertFalse(providers[0].jobs_by_name)

    def test_provider_failure_retains_generic_review_and_journal(self):
        self.approve()

        def failed(plan, journal):
            return FakeProvider(
                {
                    ("input_storage", "preflight"): [
                        ProviderError("rejected", "identity_refused")
                    ]
                }
            )

        state = execute_and_read_path(self.approved, infrastructure_factory=failed)
        self.assertEqual(state.status, "execution_review_pending")
        self.assertEqual(state.failure_review.reason, "identity_refused")
        self.assertTrue(Path(state.outcome.contact_journal).exists())

    def test_throwing_observer_does_not_change_inputs_or_write_reporting_path_fact(
        self,
    ):
        self.approve()

        def observer(record):
            raise RuntimeError("display unavailable")

        state = execute_and_read_path(
            self.approved, infrastructure_factory=input_provider, progress=observer
        )
        self.assertEqual(state.status, "execution_review_pending")
        self.assertEqual(len(state.event_ids), 5)

    def test_dev_harness_crosses_run_stage_with_explicit_inert_fixture_receipts(self):
        from bio_neb.devtools.stage_harness import main as harness_main
        from bio_neb.journey_path import ExecutionState
        from bio_neb.parabricks.plan import RetainedExecution, plan_execution

        self.approve()
        plan = plan_execution(
            self.approved,
            RetainedExecution(
                ExecutionState(), self.approved.authority, self.approved.proposal
            ),
        )
        provider = input_storage_provider(plan, None)
        document = {
            "approved": self.approved.model_dump(mode="json"),
            "run_root": str(Path(self.temp.name) / "offline-probe"),
            "fixture_receipts": {
                step: [r.model_dump(mode="json") for r in receipts]
                for (_, step), receipts in provider.scripts.items()
            },
        }
        path = Path(self.temp.name) / "probe.json"
        path.write_text(json.dumps(document))
        output = io.StringIO()
        with (
            redirect_stdout(output),
            patch(
                "bio_neb.providers.nebius.NebiusProvider",
                side_effect=AssertionError("contact"),
            ),
        ):
            code = harness_main(
                ["--stage", "input-storage", "--mode", "offline", "--input", str(path)]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["inputs"]["state"], "verified")
        self.assertEqual(self.file.read_bytes(), self.prefix)
