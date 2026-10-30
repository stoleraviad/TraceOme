"""Current complete operation and public readers with supplied provider effects.

Worker observations and large-object identities are synthetic; these tests establish
flow, authority and artifact integrity, not actual worker or scientific correctness.
"""

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from bio_neb.cli import main
from bio_neb.journey import Journey
from bio_neb.journey_path import JourneyPath
from bio_neb.parabricks.operation import execute_parabricks
from bio_neb.parabricks.result import (
    read_completed_parabricks_result,
    read_parabricks_result,
)
from tests.h100_stage_fakes import h100_provider
from tests.parabricks_operation_fakes import (
    input_storage_provider,
)
from tests.reference_stage_fakes import seed_cpu
from tests.run_prerequisites import RunPrerequisites
from tests.tiny_bam import tiny_bam

# These Runs capture the researcher's SNP request at registration, so a
# verified Execution continues into Investigation instead of waiting for a
# request.
_CAPTURED_REQUEST = {
    "asserted_by": "researcher:owner",
    "request": "Investigate rs429358 in this sample.",
    "target": {"rsid": "rs429358"},
}


class OperationTests(RunPrerequisites, unittest.TestCase):
    def test_execution_shares_coordinator_pass_and_retains_exact_early_binding(self):
        from bio_neb.recording import (
            ContactScope,
            physical_contact,
            processing_pass,
            read_passes,
        )

        self.approve()
        owner = Journey.open(Path(self.approved.journey_file))
        with owner.processing(), processing_pass(owner) as interval:
            request = owner.record("Request", {"attempt_id": "before-execution"})
            with physical_contact(
                ContactScope(
                    attempt_id="before-execution", request_id=request["event_id"]
                ),
                "inspect",
            ):
                pass
            result = execute_parabricks(
                self.approved, infrastructure_factory=self.provider
            )
            self.assertEqual(result.disposition, "succeeded")
            execution = JourneyPath.open(owner.path)
            outcome = execution.state.outcome
            early_bytes = Path(outcome.contact_journal).read_bytes()
            self.assertEqual(len(early_bytes), outcome.journal_bytes)
            self.assertEqual(
                hashlib.sha256(early_bytes).hexdigest(), outcome.journal_sha256
            )
            self.assertFalse(interval.closed)
            owner.record(
                "CoordinatorResultObserved", {"execution_id": result.execution_id}
            )
            request = owner.record("Request", {"attempt_id": "after-execution"})
            with physical_contact(
                ContactScope(
                    attempt_id="after-execution", request_id=request["event_id"]
                ),
                "inspect",
            ):
                pass
            interval.close("completed")
        reopened = Journey.open(owner.path)
        passes = read_passes(reopened)
        self.assertTrue(passes[-1].journal_complete)
        self.assertEqual(sum(row.kind == "start" for row in passes[-1].contacts), 2)
        self.assertEqual(read_parabricks_result(str(owner.path)), result)
        self.assertEqual(Path(outcome.contact_journal).read_bytes(), early_bytes)
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )

        self.assertEqual(review_execution_contacts(str(owner.path)).status, "complete")

    def test_shared_pass_closure_damage_preserves_verified_execution(self):
        from bio_neb.recording import processing_pass, read_passes

        self.approve()
        owner = Journey.open(Path(self.approved.journey_file))
        with owner.processing(), processing_pass(owner) as interval:
            result = execute_parabricks(
                self.approved, infrastructure_factory=self.provider
            )
            original = interval.journal.snapshot()
            with patch(
                "bio_neb.recording.append_durable",
                side_effect=OSError("closure failed"),
            ):
                interval.close("completed")
        reopened = Journey.open(owner.path)
        passes = read_passes(reopened)
        self.assertEqual(len(passes), self.planning_pass_count + 1)
        retained = passes[-1]
        self.assertEqual(retained.accounting_error, "OSError")
        self.assertFalse(retained.journal_complete)
        self.assertEqual(Path(result.accounting_journal_file).read_bytes(), original)
        self.assertEqual(read_parabricks_result(str(owner.path)), result)

    def test_shared_execution_contacts_refuse_another_semantic_owner(self):
        from bio_neb.recording import read_passes, validate_contact_ownership

        self.approve()
        execute_parabricks(self.approved, infrastructure_factory=self.provider)
        owner = Journey.open(Path(self.approved.journey_file))
        passes = read_passes(owner)
        self.assertEqual(len(passes), self.planning_pass_count + 1)
        retained = passes[-1]
        rows = list(retained.contacts)
        index = next(i for i, row in enumerate(rows) if row.domain is not None)
        domain = rows[index].domain
        rows[index] = rows[index].model_copy(
            update={
                "domain": domain.model_copy(
                    update={
                        "owner_event_id": owner.read()[0]["event_id"],
                    }
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "exact bound operation"):
            validate_contact_ownership(
                rows, owner.read(), retained.started_event_id, retained.closed_event_id
            )

    def test_interrupted_answered_execution_pass_cannot_repeat_effects(self):
        from bio_neb.parabricks.review import answer_review
        from bio_neb.recording import read_passes

        self.approve_indexed()
        first = execute_parabricks(
            self.approved,
            infrastructure_factory=lambda plan, journal: h100_provider(
                seed_cpu(input_storage_provider(plan, journal), plan, self.approved),
                plan,
                self.approved,
            ),
        )
        self.assertEqual(first.status, "cpu_preparation_review_pending")
        answer_review(
            self.approved.journey_file, first.request_event_id, {"action": "approve"}
        )
        with self.assertRaisesRegex(RuntimeError, "construction interrupted"):
            execute_parabricks(
                self.approved,
                infrastructure_factory=lambda *_: (_ for _ in ()).throw(
                    RuntimeError("construction interrupted")
                ),
            )
        before = {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }
        with self.assertRaisesRegex(
            ValueError, "unfinished_execution_requires_new_run"
        ):
            execute_parabricks(
                self.approved,
                infrastructure_factory=lambda *_: self.fail(
                    "replayed provider construction"
                ),
            )
        self.assertEqual(
            {
                path: path.read_bytes()
                for path in self.root.rglob("*")
                if path.is_file()
            },
            before,
        )
        passes = read_passes(Journey.open(Path(self.approved.journey_file)))
        self.assertEqual(len(passes), self.planning_pass_count + 2)
        self.assertEqual(passes[-1].disposition, "interrupted")
        self.assertFalse(passes[-1].journal_complete)

    def test_execution_prefix_cannot_disagree_with_its_retained_pass(self):
        from bio_neb.content import canonical_json
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )

        self.approve()
        result = execute_parabricks(self.approved, infrastructure_factory=self.provider)
        snapshot = Path(result.accounting_journal_file)
        rows = [json.loads(line) for line in snapshot.read_bytes().splitlines()]
        changed = next(
            row["domain"]["payload"]
            for row in rows
            if row.get("domain") and row["domain"]["payload"].get("phase") == "verified"
        )
        changed["duration"] = 123.0
        prefix = b""
        for row in rows:
            if row.get("domain") and row["domain"]["phase"] == "closed":
                row["domain"]["payload"]["prefix_sha256"] = hashlib.sha256(
                    prefix
                ).hexdigest()
            prefix += (canonical_json(row) + "\n").encode()
        digest = hashlib.sha256(prefix).hexdigest()
        substitute = snapshot.with_name(f"execution-prefix-{digest}.jsonl")
        substitute.write_bytes(prefix)
        path = Path(self.approved.journey_file)
        events = [json.loads(line) for line in path.read_bytes().splitlines()]
        outcome = next(
            event["payload"]["payload"]
            for event in events
            if event["event_type"] == "ExecutionPathRecorded"
            and event["payload"]["event_type"] == "ExecutionOutcome"
        )
        outcome.update(
            contact_journal=str(substitute),
            journal_sha256=digest,
            journal_bytes=len(prefix),
        )
        path.write_text("".join(canonical_json(event) + "\n" for event in events))
        review = review_execution_contacts(str(path))
        self.assertIn("execution_contact_prefix_drift", review.review.discrepancies)
        self.assertFalse(review.review.complete)

    def test_first_execution_contact_failure_retains_incomplete_pass(self):
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )
        from bio_neb.recording import read_passes

        self.approve()
        with patch(
            "bio_neb.recording.append_durable",
            side_effect=OSError("first contact failed"),
        ):
            with self.assertRaisesRegex(OSError, "first contact failed"):
                execute_parabricks(
                    self.approved,
                    infrastructure_factory=lambda *_: self.fail("unrecorded contact"),
                )
        owner = Journey.open(Path(self.approved.journey_file))
        passes = read_passes(owner)
        self.assertEqual(len(passes), self.planning_pass_count + 1)
        retained = passes[-1]
        self.assertEqual(retained.accounting_error, "OSError")
        self.assertFalse(retained.journal_complete)
        self.assertEqual(
            review_execution_contacts(self.approved.journey_file).status, "incomplete"
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.providers = []

    def approve(self, *, captured=True, old_journey=False, **launch_overrides):
        """Restore the class's approved Run for this case's final launch choices.

        Each call, including each subTest, gets the approved baseline restored
        in place at its own Run root; execution runs per case. `old_journey`
        restores a Run planned before #342, naming no target.
        """
        from bio_neb.recording import read_passes

        baseline = self.prerequisite(
            "approved",
            investigation_request=_CAPTURED_REQUEST if captured else None,
            **({"old_journey": True} if old_journey else {}),
            **launch_overrides,
        )
        self.root = baseline.root
        self.approved = baseline.value
        self.planning_pass_count = len(
            read_passes(Journey.open(Path(self.approved.journey_file)))
        )
        return self.approved

    def approve_indexed(self):
        """A fresh approval binding the offline Galaxy prebuilt index (#365) for a
        captured request. Its files are not served, so admission misses and the
        CPU preparation Review opens: a miss is the only route to that Review
        (owner, 2026-10-29, #464)."""
        from bio_neb.recording import read_passes
        from tests.parabricks_operation_fakes import approved_input
        from tests.prebuilt_index_support import GalaxyOpener, offline_galaxy_files

        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.approved = approved_input(
            self.root / "run",
            investigation_request=_CAPTURED_REQUEST,
            galaxy=GalaxyOpener(files=offline_galaxy_files(self.root / "galaxy")),
        )
        self.assertIsNotNone(
            self.approved.proposal.execution_package.reference.prebuilt_index
        )
        self.planning_pass_count = len(
            read_passes(Journey.open(Path(self.approved.journey_file)))
        )
        return self.approved

    def provider(self, plan, journal):
        """Input storage, one CPU preparation job and the H100 job: the approval
        binds no prebuilt index (owner, 2026-10-29, #464)."""
        provider = seed_cpu(input_storage_provider(plan, journal), plan, self.approved)
        self.providers.append(provider)
        return h100_provider(provider, plan, self.approved)

    def scripted_investigation(self):
        """Script the Investigation model and sources a captured request continues into.

        After verified Execution the coordinator closes the Pass at the child point
        and begins Investigation in a fresh Pass (ADR 0031); only the external model
        and source effects are scripted.
        """
        from tests.sequential_investigation_support import (
            ASSEMBLY_BODY,
            SequentialAdaptiveModel,
            SequentialSourceResponses,
        )

        self.investigation_model = SequentialAdaptiveModel()
        self.investigation_sources = SequentialSourceResponses(
            assembly_body=ASSEMBLY_BODY
        )
        return {
            "investigation_model_factory": self.investigation_model.model,
            "investigation_source_adapter_factory": lambda: self.investigation_sources,
        }

    def investigation_dependencies(self):
        from bio_neb.application.advancement import use_run_advancement_dependencies

        return use_run_advancement_dependencies(
            environ={"LOCAL_MODEL_KEY": "not-retained"},
            **self.scripted_investigation(),
        )

    def assert_continued_to_collection_review(self, result):
        """Verified Execution continued into Investigation and stopped at its review."""
        self.assertEqual(result.status, "investigation_review_pending")
        self.assertEqual(result.current_review.kind, "investigation")
        verified = read_parabricks_result(self.approved.journey_file)
        self.assertEqual(
            (verified.status, verified.disposition),
            ("execution_artifacts_verified", "succeeded"),
        )
        self.assertEqual(self.investigation_model.calls, 3)
        self.assertEqual(self.investigation_sources.contact_count, 2)

    def test_approved_execution_does_not_repeat_upstream_compilation_or_hitl_review(
        self,
    ):
        # The Review's prebuilt-index hit is proven in
        # tests/test_prebuilt_index_execution.py; this Run prepares on CPU.
        self.approve()
        with (
            patch(
                "bio_neb.parabricks.execution_proposal._compile_execution_proposal",
                side_effect=AssertionError("upstream proposal recompiled"),
            ),
            patch(
                "bio_neb.parabricks.execution_proposal.validate_execution_hitl_response",
                side_effect=AssertionError("upstream approval revalidated"),
            ),
        ):
            state = execute_parabricks(
                self.approved, infrastructure_factory=self.provider
            )
        self.assertEqual(state.disposition, "succeeded")
        self.assertEqual(state.authority_id, self.approved.authority.authority_id)
        self.assertEqual(state.proposal_id, self.approved.proposal.proposal_id)
        self.assertEqual(state.execution_id, self.approved.execution_id)
        self.assertEqual(state.cache_outcome, "miss")
        # One CPU preparation job, then the H100 job.
        self.assertEqual(len(self.providers[-1].jobs_by_name), 2)

    def test_interactive_resume_presents_completion_or_answers_exact_pending_review(
        self,
    ):
        for completed in (False, True):
            with self.subTest(completed=completed):
                # A captured request continues past verified Execution into
                # Investigation; the completed stop exists only without one, in
                # an old Journey planned before plans named targets (a current
                # plan names its targets, so its Execution continues into
                # Investigation too).
                if completed:
                    self.approve(captured=False, old_journey=True)
                else:
                    self.approve_indexed()
                first = execute_parabricks(
                    self.approved,
                    infrastructure_factory=self.provider
                    if completed
                    else input_storage_provider,
                )
                self.assertEqual(
                    first.disposition, "succeeded" if completed else "incomplete"
                )
                before = Path(self.approved.journey_file).read_bytes()
                output, errors = io.StringIO(), io.StringIO()
                response = io.StringIO('{"action":"approve"}\n')
                continuations = []

                def continue_execution(
                    plan,
                    journal,
                    *,
                    was_completed=completed,
                    recorded_continuations=continuations,
                ):
                    self.assertFalse(
                        was_completed, "completed resume contacted provider"
                    )
                    recorded_continuations.append(plan.identity.execution_id)
                    return h100_provider(
                        seed_cpu(
                            input_storage_provider(plan, journal), plan, self.approved
                        ),
                        plan,
                        self.approved,
                    )

                with (
                    redirect_stdout(output),
                    redirect_stderr(errors),
                    patch("sys.stdin", response),
                    patch(
                        "bio_neb.parabricks.operation._make_provider",
                        side_effect=AssertionError(
                            "resume contacted execution infrastructure"
                        ),
                    ),
                    patch.dict("os.environ", {"LOCAL_MODEL_KEY": "not-retained"}),
                ):
                    code = main(
                        [
                            "resume",
                            "--journey-file",
                            self.approved.journey_file,
                            "--interactive",
                        ],
                        parabricks_infrastructure_factory=continue_execution,
                        **self.scripted_investigation(),
                    )
                self.assertEqual(code, 0, errors.getvalue())
                documents = [
                    json.loads(line) for line in output.getvalue().splitlines()
                ]
                if completed:
                    document = dict(documents[0])
                    self.assertEqual(
                        document.pop("status"), "investigation_request_needed"
                    )
                    self.assertEqual(
                        document.pop("schema_version"),
                        "bio-neb.investigation-request-needed/1",
                    )
                    self.assertEqual(
                        [action["action"] for action in document.pop("legal_actions")],
                        ["supply_investigation_request"],
                    )
                    self.assertEqual(
                        document,
                        first.model_dump(
                            mode="json", exclude={"schema_version", "status"}
                        ),
                    )
                    self.assertEqual(self.investigation_model.calls, 0)
                    self.assertEqual(continuations, [])
                    self.assertEqual(len(documents), 1)
                    self.assertEqual(response.tell(), 0)
                    self.assertEqual(
                        Path(self.approved.journey_file).read_bytes(), before
                    )
                    self.assertEqual(errors.getvalue(), "")
                else:
                    self.assertEqual(documents[0], first.model_dump(mode="json"))
                    self.assertEqual(continuations, [first.execution_id])
                    self.assertEqual(len(documents), 2)
                    # The answered CPU review continues Execution, which verifies
                    # and continues into Investigation's review in the same call.
                    self.assertEqual(
                        documents[1]["status"], "investigation_review_pending"
                    )
                    verified = read_parabricks_result(self.approved.journey_file)
                    self.assertEqual(verified.status, "execution_artifacts_verified")
                    self.assertEqual(verified.disposition, "succeeded")
                    self.assertEqual(verified.execution_id, first.execution_id)
                    self.assertEqual(verified.execution_pass, 2)
                    self.assertEqual(self.investigation_model.calls, 3)
                    self.assertEqual(self.investigation_sources.contact_count, 2)
                    rows = JourneyPath.open(Path(self.approved.journey_file)).read()
                    answer = next(
                        row
                        for row in rows
                        if row.event_type == "CpuPreparationReviewAnswered"
                    )
                    self.assertIn(first.request_event_id, answer.rests_on)

    def test_new_pending_proposal_makes_prior_approved_execution_ineligible(self):
        from bio_neb.parabricks.plan import ExecutionCompositionRefused

        self.approve()
        newer = self.prerequisite(
            "approved", investigation_request=_CAPTURED_REQUEST, slot="newer"
        ).value
        latest = next(
            event
            for event in Journey.open(Path(newer.journey_file)).read()
            if event["event_type"] == "ExecutionProposalSubmitted"
        )
        journey = Journey.open(Path(self.approved.journey_file))
        journey.record(
            "ExecutionProposalSubmitted",
            latest["payload"],
            parent_ids=[journey.read()[-1]["event_id"]],
        )
        before = {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }
        with self.assertRaisesRegex(ExecutionCompositionRefused, "lineage") as refused:
            execute_parabricks(
                self.approved,
                infrastructure_factory=lambda *_: self.fail(
                    "stale approval constructed provider"
                ),
            )
        self.assertEqual(refused.exception.details, ("authority_not_current",))
        self.assertEqual(
            {
                path: path.read_bytes()
                for path in self.root.rglob("*")
                if path.is_file()
            },
            before,
        )

    def test_cli_resumes_retained_approval_and_runs_real_operation(self):
        self.approve()
        input_file = self.root / "approved.json"
        input_file.write_text(self.approved.model_dump_json())
        with (
            redirect_stdout(io.StringIO()) as output,
            redirect_stderr(io.StringIO()) as progress,
            patch.dict("os.environ", {"LOCAL_MODEL_KEY": "not-retained"}),
        ):
            code = main(
                ["resume", "--journey-file", self.approved.journey_file],
                parabricks_infrastructure_factory=self.provider,
                **self.scripted_investigation(),
            )
        self.assertEqual(code, 0, progress.getvalue())
        # The verified Execution continues into Investigation in the same call;
        # the CLI presents that stop, and the reader returns the Execution result.
        self.assertEqual(
            json.loads(output.getvalue())["status"], "investigation_review_pending"
        )
        self.assertEqual(self.investigation_model.calls, 3)
        self.assertEqual(self.investigation_sources.contact_count, 2)
        presented = read_parabricks_result(self.approved.journey_file).model_dump(
            mode="json"
        )
        self.assertIn("verified_objects", presented)
        self.assertEqual(
            presented["authority_id"], self.approved.authority.authority_id
        )
        self.assertEqual(presented["execution_id"], self.approved.execution_id)
        self.assertNotIn("h100", presented)
        self.assertNotIn("bound", presented)
        bam = next(
            item
            for item in presented["verified_objects"]
            if item["role"] == "aligned_bam"
        )
        self.assertEqual(bam["byte_count"], len(tiny_bam()))
        self.assertEqual(bam["sha256"], hashlib.sha256(tiny_bam()).hexdigest())
        self.assertEqual(bam["bucket"], "fresh-run-bucket")
        self.assertEqual(presented["provider_job_id"], "aijob-exact")
        for proof in presented["output_proofs"]:
            self.assertEqual(
                proof["producer"]["authority_id"], self.approved.authority.authority_id
            )
            self.assertRegex(proof["proof_ref"], r"^cj:1:[1-9][0-9]*$")
        self.assertEqual(
            {item["role"] for item in presented.get("output_proofs", [])},
            {"aligned_bam", "VCF", "worker_receipt", "execution_manifest"},
        )
        self.assertEqual(presented["disposition"], "succeeded")
        records = [json.loads(line) for line in progress.getvalue().splitlines()]
        kinds = [record.get("kind") for record in records]
        self.assertEqual(kinds[0], "execution_flow_started")
        self.assertEqual(kinds.count("execution_flow_finished"), 1)
        self.assertEqual(len(self.providers), 1)
        completed = read_completed_parabricks_result(self.approved.journey_file)
        self.assertEqual(
            tuple(item.role for item in completed.verified_objects),
            ("aligned_bam", "VCF", "worker_receipt", "execution_manifest"),
        )

    def test_result_contract_import_and_read_work_without_execution_modules(self):
        self.approve()
        state = execute_parabricks(self.approved, infrastructure_factory=self.provider)
        source = """
import importlib.abc, sys
class ForbidExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.startswith('bio_neb.parabricks._') or fullname in {'bio_neb.parabricks.runner', 'bio_neb.parabricks.operation', 'bio_neb.parabricks.nebius_execution_composition', 'bio_neb.parabricks.plan', 'bio_neb.parabricks.input_storage', 'bio_neb.parabricks.reference_stages', 'bio_neb.parabricks.h100_stage'}:
            raise AssertionError('execution import: ' + fullname)
sys.meta_path.insert(0, ForbidExecution())
from bio_neb.parabricks.result import read_completed_parabricks_result
from bio_neb.investigation.vcf_materialization import materialize_current_vcf
from bio_neb.investigation.bam_materialization import materialize_current_bam
print(read_completed_parabricks_result(sys.argv[1]).execution_id)
"""
        child = subprocess.run(
            [sys.executable, "-c", source, self.approved.journey_file],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(child.stdout.strip(), state.execution_id)

    def test_retained_result_refuses_changed_execution_evidence(self):
        self.approve()
        execute_parabricks(self.approved, infrastructure_factory=self.provider)
        journey = JourneyPath.open(Path(self.approved.journey_file))
        pointer = next(
            p for p in journey.state.h100.pointers if p.role == "execution_manifest"
        )
        manifest_file = journey.file.parent / pointer.location
        content = manifest_file.read_bytes()
        self.assertIn(b'"provider_job_id":"aijob-exact"', content)
        # Same size, valid JSON: the hash check must detect this changed producer.
        manifest_file.write_bytes(
            content.replace(
                b'"provider_job_id":"aijob-exact"', b'"provider_job_id":"aijob-other"'
            )
        )
        with self.assertRaisesRegex(ValueError, "artifact integrity changed"):
            read_completed_parabricks_result(self.approved.journey_file)
        self.assertEqual(len(self.providers), 1)

    def test_retained_result_refuses_ambiguous_output_proof_without_contact(self):
        self.approve()
        execute_parabricks(self.approved, infrastructure_factory=self.provider)
        path = Path(self.approved.journey_file)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        verified = next(
            row["payload"]
            for row in rows
            if row["event_type"] == "ExecutionPathRecorded"
            and row["payload"]["event_type"] == "H100ArtifactsVerified"
        )
        pointers = verified["payload"]["pointers"]
        vcf = next(pointer for pointer in pointers if pointer["role"] == "VCF")
        # Deliberately malformed retained evidence: two claims for the same role,
        # one wrong identity followed by the original. Last-wins is not proof.
        pointers.insert(0, {**vcf, "content_id": "sha256:" + "b" * 64})
        from bio_neb.content import canonical_json

        path.write_text("".join(canonical_json(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "duplicate.*proof"):
            read_completed_parabricks_result(str(path))
        self.assertEqual(len(self.providers), 1)

    def test_retained_result_refuses_forged_input_or_reference_proof(self):
        self.approve()
        execute_parabricks(self.approved, infrastructure_factory=self.provider)
        path = Path(self.approved.journey_file)
        original = path.read_bytes()
        result = read_completed_parabricks_result(str(path))
        from bio_neb.content import canonical_json

        def forge(event_type, role, change):
            rows = [json.loads(line) for line in original.decode().splitlines()]
            fact = next(
                row["payload"]
                for row in rows
                if row["event_type"] == "ExecutionPathRecorded"
                and row["payload"]["event_type"] == event_type
            )
            payload = fact["payload"]
            pointers = payload.get("pointers") or payload["reference"]["pointers"]
            change(next(pointer for pointer in pointers if pointer["role"] == role))
            path.write_text("".join(canonical_json(row) + "\n" for row in rows))

        reference_event = (
            "ReferenceResolved"
            if result.cache_outcome == "hit"
            else "ReferencePrepared"
        )
        reference_role = result.reference_proofs[0].role
        foreign = "execution-authority:" + "f" * 64
        # Deliberately forged retained proofs: each one still claims verified
        # grounding, so only the producer-boundary inventory check can refuse it.
        forgeries = {
            "input from another authority": (
                "InputsVerified",
                "read_1",
                lambda p: p["producer"].update(authority_id=foreign),
            ),
            "input from another producer": (
                "InputsVerified",
                "read_2",
                lambda p: p["producer"].update(stage="h100"),
            ),
            "input outside the Run bucket": (
                "InputsVerified",
                "read_2",
                lambda p: p.update(
                    location=p["location"].replace("s3://", "s3://foreign-", 1)
                ),
            ),
            "reference without authority": (
                reference_event,
                reference_role,
                lambda p: p["producer"].update(authority_id=None),
            ),
            "reference from the input producer": (
                reference_event,
                reference_role,
                lambda p: p["producer"].update(stage="input_storage"),
            ),
        }
        for name, (event_type, role, change) in forgeries.items():
            with self.subTest(name):
                forge(event_type, role, change)
                with self.assertRaisesRegex(ValueError, "input or reference proof"):
                    read_completed_parabricks_result(str(path))
        path.write_bytes(original)
        self.assertEqual(read_completed_parabricks_result(str(path)), result)
        self.assertEqual(len(self.providers), 1)

    def test_result_contract_refuses_incomplete_or_unbound_proof_inventory(self):
        from bio_neb.parabricks.result import ParabricksExecutionOutcome

        self.approve()
        execute_parabricks(self.approved, infrastructure_factory=self.provider)
        valid = read_completed_parabricks_result(self.approved.journey_file).model_dump(
            mode="json"
        )
        manifest_role = (
            "prepared_reference_admission"
            if valid["cache_outcome"] == "hit"
            else "prepared_reference_manifest"
        )

        def without(field, role):
            return [p for p in valid[field] if p["role"] != role]

        def changed(field, role, **update):
            return [{**p, **update} if p["role"] == role else p for p in valid[field]]

        # Deliberately malformed outcome documents at the producer contract.
        cases = {
            "duplicate input role": {
                "input_proofs": [*valid["input_proofs"], valid["input_proofs"][0]]
            },
            # The CPU job fetched the source, so its proof is a reference proof.
            "missing reference_source": {
                "reference_proofs": without("reference_proofs", "reference_source")
            },
            "missing input_manifest": {
                "input_proofs": without("input_proofs", "input_manifest")
            },
            "missing reference manifest": {
                "reference_proofs": without("reference_proofs", manifest_role)
            },
            "unverified reference proof": {
                "reference_proofs": changed(
                    "reference_proofs", manifest_role, grounding="ambiguous"
                )
            },
            "bundle location not its content": {
                "input_proofs": changed(
                    "input_proofs",
                    "input_manifest",
                    location="artifacts/sha256/" + "0" * 64,
                )
            },
            "receipt names another manifest": {
                "prepared_reference_receipt_id": "sha256:" + "0" * 64
            },
        }
        self.assertEqual(
            ParabricksExecutionOutcome.model_validate(valid).model_dump(mode="json"),
            valid,
        )
        for name, update in cases.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(ValueError, "input or reference proof"),
            ):
                ParabricksExecutionOutcome.model_validate({**valid, **update})

    def test_throwing_observer_is_disabled_without_changing_execution(self):
        self.approve()
        records = []

        def broken(record):
            records.append(record)
            raise RuntimeError("credential-value-not-for-display")

        state = execute_parabricks(
            self.approved, infrastructure_factory=self.provider, progress=broken
        )
        self.assertEqual(state.disposition, "succeeded")
        self.assertEqual(len(records), 1)
        # The CPU preparation job and the H100 job (no account candidate).
        self.assertEqual(len(self.providers[0].jobs_by_name), 2)
        before = {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }
        self.assertFalse(
            any(
                b"credential-value-not-for-display" in content
                for content in before.values()
            )
        )
        reopened = execute_parabricks(
            self.approved,
            infrastructure_factory=lambda *_: self.fail("terminal replay contact"),
            progress=broken,
        )
        self.assertEqual(reopened, state)
        self.assertEqual(len(records), 1)
        self.assertEqual(
            {
                path: path.read_bytes()
                for path in self.root.rglob("*")
                if path.is_file()
            },
            before,
        )

    def test_quiet_operation_keeps_heartbeat_and_excludes_private_receipt_fields(self):
        self.approve(heartbeat_interval_seconds=0.01)
        heartbeat = threading.Event()
        records = []

        def report(record):
            records.append(record)
            if record.get("kind") == "execution_flow_heartbeat":
                heartbeat.set()

        def factory(plan, journal):
            if not heartbeat.wait(2):
                raise AssertionError("quiet operation lost heartbeat")
            provider = self.provider(plan, journal)
            original = provider.for_step

            def for_step(stage, step):
                ports = original(stage, step)
                if (stage, step) == ("h100", "await_h100"):
                    receipt = provider.scripts[stage, step][0]
                    provider.scripts[stage, step][0] = receipt.model_copy(
                        update={
                            "values": {
                                **receipt.values,
                                "private": "credential-value-not-for-display",
                            }
                        }
                    )
                return ports

            provider.for_step = for_step
            return provider

        state = execute_parabricks(
            self.approved, infrastructure_factory=factory, progress=report
        )
        self.assertEqual(state.disposition, "succeeded")
        self.assertTrue(heartbeat.is_set())
        self.assertEqual(records[0]["kind"], "execution_flow_started")
        self.assertEqual(records[-1]["kind"], "execution_flow_finished")
        samples = [
            record
            for record in records
            if record.get("kind") == "execution_observation"
        ]
        self.assertTrue(samples)
        self.assertEqual(samples[-1]["state"], "COMPLETED")
        self.assertEqual(samples[-1]["provider_ids"], ["aijob-exact"])
        self.assertNotIn("credential-value-not-for-display", repr(records))
        # The CPU preparation job and the H100 job (no account candidate).
        self.assertEqual(len(self.providers[0].jobs_by_name), 2)

    def test_explicit_miss_cpu_failure_waits_before_h100_with_retained_cause(self):
        from bio_neb.failure import ProviderError

        self.approve()

        def factory(plan, journal):
            provider = self.provider(plan, journal)
            provider.scripts["cpu_preparation", "await_cpu"] = deque(
                [
                    ProviderError(
                        "worker_failed",
                        "execution_nonzero",
                        {"provider_job_id": "aijob-reference-exact"},
                    )
                ]
            )
            return provider

        state = execute_parabricks(self.approved, infrastructure_factory=factory)
        self.assertEqual(state.status, "execution_review_pending")
        self.assertEqual(state.failures[-1].failure_class, "worker_failed")
        self.assertEqual(state.failures[-1].detail, "execution_nonzero")
        self.assertIsNotNone(state.pending_review)
        self.assertEqual(
            state.failure_event_ids, (state.pending_review.failure_event_id,)
        )
        self.assertEqual(state.disposition, "incomplete")
        self.assertNotIn("verified_objects", state.model_dump())
        self.assertEqual(len(self.providers[0].jobs_by_name), 1)

    def test_approved_review_continues_one_binding_and_reader_returns_current_pass(
        self,
    ):
        self.approve_indexed()

        def factory(plan, journal):
            provider = h100_provider(
                seed_cpu(input_storage_provider(plan, journal), plan, self.approved),
                plan,
                self.approved,
            )
            self.providers.append(provider)
            return provider

        first = execute_parabricks(self.approved, infrastructure_factory=factory)
        self.assertEqual(first.status, "cpu_preparation_review_pending")
        presented = first.model_dump(mode="json")
        self.assertIn("pending_review", presented)
        self.assertEqual(
            presented["pending_review"]["legal_actions"],
            ["abort", "approve_cpu_preparation"],
        )
        self.assertNotIn("h100", presented)
        self.assertEqual(first.execution_pass, 1)
        self.assertFalse(self.providers[0].jobs_by_name)
        with (
            redirect_stdout(io.StringIO()) as output,
            patch.dict("os.environ", {"LOCAL_MODEL_KEY": "not-retained"}),
        ):
            code = main(
                [
                    "answer",
                    "--journey-file",
                    self.approved.journey_file,
                    "--request-event-id",
                    first.request_event_id,
                    "--response-json",
                    json.dumps({"action": "approve"}),
                ],
                parabricks_infrastructure_factory=factory,
                **self.scripted_investigation(),
            )
        self.assertEqual(code, 0)
        second = read_parabricks_result(self.approved.journey_file)
        # The answer's call verifies Execution and continues into Investigation.
        answered = json.loads(output.getvalue())
        self.assertEqual(answered["status"], "investigation_review_pending")
        self.assertEqual(self.investigation_model.calls, 3)
        before = Path(self.approved.journey_file).read_bytes()
        with redirect_stdout(io.StringIO()):
            replay = main(
                [
                    "answer",
                    "--journey-file",
                    self.approved.journey_file,
                    "--request-event-id",
                    first.request_event_id,
                    "--response-json",
                    json.dumps({"action": "approve"}),
                ],
                parabricks_infrastructure_factory=lambda *_: self.fail(
                    "replayed CPU effect"
                ),
            )
        self.assertEqual(replay, 0)
        self.assertEqual(Path(self.approved.journey_file).read_bytes(), before)
        self.assertEqual(second.disposition, "succeeded")
        self.assertEqual(second.execution_pass, 2)
        self.assertEqual(second.execution_id, first.execution_id)
        self.assertEqual(second.authority_id, first.authority_id)
        self.assertEqual(len(self.providers), 2)
        self.assertEqual(len(self.providers[1].jobs_by_name), 2)
        self.assertEqual(read_parabricks_result(self.approved.journey_file), second)
        journey = JourneyPath.open(Path(self.approved.journey_file))
        self.assertEqual(
            sum(event.event_type == "ExecutionBound" for event in journey.read()), 1
        )
        self.assertIn(
            journey.state.latest_event_ids["CpuPreparationReviewAnswered"],
            journey.state.rests_on[journey.state.latest_event_ids["ReferencePrepared"]],
        )

    def test_noncontinuable_stops_reopen_through_operation_and_cli_without_changes(
        self,
    ):
        from bio_neb.failure import Ask, ProviderError
        from bio_neb.journey_path import Pending

        for mode in ("pending", "declined", "failure_pending", "succeeded"):
            with self.subTest(mode=mode):
                # Successful Execution waits only when no scientific request was
                # registered, in an old Journey planned before plans named
                # targets; a captured request must continue into Investigation.
                if mode in {"pending", "declined"}:
                    self.approve_indexed()
                else:
                    self.approve(
                        captured=mode != "succeeded", old_journey=mode == "succeeded"
                    )

                def factory(plan, journal, mode=mode):
                    if mode in {"pending", "declined"}:
                        return input_storage_provider(plan, journal)
                    provider = self.provider(plan, journal)
                    if mode == "failure_pending":
                        provider.scripts["input_storage", "preflight"] = deque(
                            [ProviderError("rejected", "storage_access_refused")]
                        )
                    return provider

                retained = execute_parabricks(
                    self.approved, infrastructure_factory=factory
                )
                if mode == "declined":
                    with redirect_stdout(io.StringIO()) as answer_output:
                        code = main(
                            [
                                "answer",
                                "--journey-file",
                                self.approved.journey_file,
                                "--request-event-id",
                                retained.request_event_id,
                                "--response-json",
                                json.dumps({"action": "decline"}),
                            ]
                        )
                    self.assertEqual(code, 0)
                    retained = read_parabricks_result(self.approved.journey_file)
                    self.assertEqual(
                        json.loads(answer_output.getvalue())["status"],
                        "finalization_review_pending",
                    )
                expected = {
                    "pending": ("incomplete", Pending),
                    "declined": ("failed", type(None)),
                    "failure_pending": ("incomplete", Ask),
                    "succeeded": ("succeeded", type(None)),
                }[mode]
                if mode == "failure_pending":
                    self.assertEqual(
                        retained.failures[-1].detail, "storage_access_refused"
                    )
                approved_file = self.root / "approved.json"
                approved_file.write_text(self.approved.model_dump_json())
                before = {
                    path: path.read_bytes()
                    for path in self.root.rglob("*")
                    if path.is_file()
                }

                def forbid(*_):
                    self.fail("noncontinuable stop constructed provider")

                reopened = execute_parabricks(
                    self.approved, infrastructure_factory=forbid
                )
                self.assertEqual(reopened, retained)
                self.assertEqual(
                    (
                        reopened.disposition,
                        type(getattr(reopened, "pending_review", None)),
                    ),
                    expected,
                )
                with (
                    redirect_stdout(io.StringIO()) as output,
                    redirect_stderr(io.StringIO()) as errors,
                ):
                    status = main(
                        ["resume", "--journey-file", self.approved.journey_file],
                        parabricks_infrastructure_factory=forbid,
                    )
                self.assertEqual(status, 0, errors.getvalue())
                if mode == "declined":
                    self.assertEqual(
                        json.loads(output.getvalue())["status"],
                        "finalization_review_pending",
                    )
                elif mode == "succeeded":
                    document = json.loads(output.getvalue())
                    self.assertEqual(
                        document.pop("status"), "investigation_request_needed"
                    )
                    self.assertEqual(
                        document.pop("schema_version"),
                        "bio-neb.investigation-request-needed/1",
                    )
                    actions = document.pop("legal_actions")
                    self.assertEqual(
                        [action["action"] for action in actions],
                        ["supply_investigation_request"],
                    )
                    expected_result = retained.model_dump(
                        mode="json", exclude={"schema_version", "status"}
                    )
                    self.assertEqual(document, expected_result)
                else:
                    self.assertEqual(
                        json.loads(output.getvalue()), retained.model_dump(mode="json")
                    )
                diagnostics = [
                    json.loads(line) for line in errors.getvalue().splitlines()
                ]
                self.assertEqual(diagnostics, [])
                self.assertEqual(
                    {
                        path: path.read_bytes()
                        for path in self.root.rglob("*")
                        if path.is_file()
                    },
                    before,
                )

    def test_changed_approved_input_is_refused_before_continuation_effects(self):
        from bio_neb.parabricks.plan import ExecutionCompositionRefused
        from bio_neb.parabricks.review import answer_review

        self.approve_indexed()
        first = execute_parabricks(
            self.approved, infrastructure_factory=input_storage_provider
        )
        self.assertEqual(first.status, "cpu_preparation_review_pending")
        answer_review(
            self.approved.journey_file,
            first.request_event_id,
            {"action": "approve"},
        )
        retained = read_parabricks_result(self.approved.journey_file)
        changed = self.approved.model_copy(
            update={
                "launch": self.approved.launch.model_copy(
                    update={"heartbeat_interval_seconds": 31}
                )
            }
        )
        before = {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }
        with self.assertRaises(ExecutionCompositionRefused) as refusal:
            execute_parabricks(
                changed,
                infrastructure_factory=lambda *_: self.fail(
                    "changed approval contacted provider"
                ),
            )
        self.assertEqual(
            (refusal.exception.code, refusal.exception.details),
            ("lineage", ("approved_facts_drift",)),
        )
        self.assertEqual(read_parabricks_result(self.approved.journey_file), retained)
        self.assertEqual(
            {
                path: path.read_bytes()
                for path in self.root.rglob("*")
                if path.is_file()
            },
            before,
        )

    def test_approved_identity_and_verification_choices_reach_all_stage_receipts(self):
        from bio_neb.application.verification_policy import ObjectVerificationOverrides

        for full_read in (False, True):
            with self.subTest(full_read=full_read):
                overrides = (
                    ObjectVerificationOverrides(
                        input_storage="independent_full_read",
                        h100_outputs="independent_full_read",
                        reason="audit selected uploads",
                    )
                    if full_read
                    else ObjectVerificationOverrides()
                )
                self.approve(verification_overrides=overrides)
                plans = []

                def factory(plan, journal, plans=plans):
                    plans.append(plan)
                    return self.provider(plan, journal)

                state = execute_parabricks(
                    self.approved, infrastructure_factory=factory
                )
                self.assertEqual(state.disposition, "succeeded")
                self.assertEqual(len(plans), 1)
                plan = plans[0]
                reference = self.approved.proposal.execution_package.reference
                self.assertEqual(
                    (
                        reference.expected_checksum_algorithm,
                        plan.reference.expected_md5,
                        plan.reference.expected_size_bytes,
                    ),
                    ("md5", reference.expected_checksum, reference.expected_size_bytes),
                )
                self.assertEqual(
                    tuple(item.role for item in plan.fastqs), ("read_1", "read_2")
                )
                self.assertEqual(plan.launch.context, self.approved.launch.context)
                self.assertEqual(plan.storage, self.approved.launch.object_storage)
                selected = (
                    "independent_full_read" if full_read else "reuse_creation_proof"
                )
                self.assertEqual(
                    plan.verification.effective.model_dump(),
                    {
                        "input_storage": selected,
                        "h100_outputs": selected,
                        "prepared_candidate": "reuse_creation_proof",
                        "reference_preparation": "reuse_creation_proof",
                    },
                )
                self.assertEqual(plan.verification.overrides, overrides)
                rows = [
                    json.loads(line)
                    for line in Path(state.accounting_journal_file)
                    .read_text()
                    .splitlines()
                ]
                rows = [
                    row["domain"]["payload"]
                    for row in rows
                    if row.get("domain") and row["domain"]["payload"]
                ]
                self.assertEqual(
                    [
                        row["resource_kind"]
                        for row in rows
                        if row.get("logical_operation") == "jobs.submit"
                        and row.get("phase") == "invoked"
                    ],
                    ["cpu_job", "h100_job"],
                )
                publications = [
                    row
                    for row in rows
                    if row.get("logical_operation") == "store.put"
                    and row.get("phase") == "verified"
                ]
                self.assertEqual(
                    (publications[-1]["stage"], publications[-1]["step"]),
                    ("h100", "publish_manifest"),
                )
                modes = {}
                for row in rows:
                    if row.get("receipt"):
                        receipt = json.loads(Path(row["receipt"]["ref"]).read_text())
                        mode = receipt.get("values", {}).get("verification_mode")
                        if mode:
                            modes.setdefault((row["stage"], row["step"]), []).append(
                                mode
                            )
                # Only the FASTQs: with no supplied index the CPU job fetches
                # the reference.
                self.assertEqual(
                    modes.get(("input_storage", "verify_inputs"), []),
                    [selected] * 2 if full_read else [],
                )
                self.assertEqual(
                    modes.get(("cpu_preparation", "verify_bundle"), []),
                    [],
                )
                self.assertEqual(
                    modes.get(("h100", "verify_outputs"), []),
                    [selected] * 2 if full_read else [],
                )
                journey = JourneyPath.open(Path(self.approved.journey_file))
                manifest = next(
                    pointer
                    for pointer in state.output_proofs
                    if pointer.role == "execution_manifest"
                )
                proof = json.loads(journey.read_artifact(manifest))["transfer_proof"]
                self.assertEqual(
                    proof["host_audit_bytes"], proof["upload_bytes"] if full_read else 0
                )
                self.assertGreater(proof["upload_bytes"], 0)

    def test_production_construction_shares_authentication_through_actual_callers(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from bio_neb.journey_path import ExecutionState
        from bio_neb.parabricks.operation import _make_provider
        from bio_neb.parabricks.plan import RetainedExecution, plan_execution
        from bio_neb.providers.nebius_common import AwsCredentialMaterial
        from tests.parabricks_operation_fakes import _credential_receipt

        approved = self.approve()
        plan = plan_execution(
            approved,
            RetainedExecution(ExecutionState(), approved.authority, approved.proposal),
        )
        target = plan.scope
        runner = Mock()
        runner.preflight.return_value = SimpleNamespace(
            executable=target.executable,
            resolved_executable=target.resolved_executable,
            version=target.cli_version,
            byte_count=target.executable_byte_count,
            sha256=target.executable_sha256,
        )
        calls = []

        def contact(*args):
            operation = args[:2] if args[:2] == ("iam", "whoami") else args[:3]
            calls.append(operation)
            if operation == ("iam", "whoami"):
                return {
                    "service_account_profile": {
                        "info": {
                            "status": {"active": True},
                            "metadata": {"parent_id": target.project_id},
                        }
                    }
                }
            if operation == ("iam", "project", "get"):
                return {
                    "metadata": {
                        "id": target.project_id,
                        "parent_id": target.tenant_id,
                    }
                }
            self.assertEqual(operation, ("vpc", "subnet", "get"))
            return {
                "metadata": {
                    "id": target.subnet_id,
                    "parent_id": target.project_id,
                },
                "status": {"state": "READY"},
            }

        runner.json.side_effect = contact
        with (
            patch(
                "bio_neb.providers.nebius.make_accepted_nebius_cli_runner",
                return_value=runner,
            ),
            patch(
                "bio_neb.providers.nebius.SelectedNebiusCredentialResolver",
                return_value=Mock(
                    return_value=AwsCredentialMaterial(
                        access_key_id="offline-key",
                        secret_access_key="offline-secret",
                        credentials_profile="offline-storage",
                        worker_secret_selector="offline-secret@offline-version",
                    ),
                    receipt=_credential_receipt(),
                ),
            ),
        ):
            provider = _make_provider(plan, None)
        self.assertEqual(calls, [])
        first = provider.for_step("input_storage", "preflight").identity.whoami()
        second = provider.for_step("input_storage", "preflight").identity.whoami()
        self.assertEqual(first, second)
        self.assertEqual(first.values["project_id"], target.project_id)
        provider.jobs.contact.preflight(target)
        self.assertEqual(
            calls,
            [
                ("iam", "whoami"),
                ("iam", "project", "get"),
                ("iam", "project", "get"),
                ("vpc", "subnet", "get"),
            ],
        )

    def test_unexpected_provider_fault_preserves_cause_and_interrupted_prefix(self):
        for via_cli in (False, True):
            with self.subTest(via_cli=via_cli):
                self.approve()
                original = RuntimeError("original execution failure")
                cause = OSError("original cause")
                original.__cause__ = cause
                journals = []

                def fail(original=original):
                    raise original

                def factory(plan, journal, journals=journals):
                    journals.append(journal.path)
                    provider = input_storage_provider(plan, journal)
                    for_step = provider.for_step

                    def ports(stage, step):
                        result = for_step(stage, step)
                        result.identity.whoami = fail
                        return result

                    provider.for_step = ports
                    return provider

                approved_file = self.root / "approved.json"
                approved_file.write_text(self.approved.model_dump_json())
                errors = io.StringIO()
                if via_cli:
                    # Under `resume` the untyped fault reaches the coordinator's
                    # failure module: the Run holds and the CLI names it.
                    with redirect_stdout(io.StringIO()), redirect_stderr(errors):
                        exit_code = main(
                            ["resume", "--journey-file", self.approved.journey_file],
                            parabricks_infrastructure_factory=factory,
                        )
                    refusal = json.loads(errors.getvalue().splitlines()[-1])
                    self.assertEqual(
                        (exit_code, refusal["code"], refusal["status"]),
                        (2, "coordinator_unhandled_error", "run_held"),
                    )
                    self.assertTrue(
                        refusal["failure"]["cause"].startswith("RuntimeError@")
                    )
                else:
                    with (
                        redirect_stdout(io.StringIO()),
                        redirect_stderr(errors),
                        self.assertRaises(RuntimeError) as raised,
                    ):
                        execute_parabricks(
                            self.approved, infrastructure_factory=factory
                        )
                    self.assertIs(raised.exception, original)
                    self.assertIs(raised.exception.__cause__, cause)
                self.assertEqual(len(journals), 1)
                rows = [
                    json.loads(line) for line in journals[0].read_text().splitlines()
                ]
                self.assertEqual(rows[-1]["kind"], "final")
                self.assertFalse(rows[-1]["complete"])
                rows = [
                    row["domain"]["payload"]
                    for row in rows
                    if row.get("domain") and row["domain"]["payload"]
                ]
                self.assertEqual(rows[-1]["phase"], "invoked")
                self.assertEqual(rows[-1]["logical_operation"], "identity.whoami")
                retained = read_parabricks_result(self.approved.journey_file)
                self.assertEqual(retained.disposition, "incomplete")
                self.assertIsNone(retained.outcome_event_id)
                self.assertEqual(retained.status, "execution_inputs_pending")
                self.assertNotIn("verified_objects", retained.model_dump())
                before = {
                    path: path.read_bytes()
                    for path in self.root.rglob("*")
                    if path.is_file()
                }
                with self.assertRaisesRegex(
                    ValueError, "unfinished_execution_requires_new_run"
                ):
                    execute_parabricks(
                        self.approved,
                        infrastructure_factory=lambda *_: self.fail(
                            "interrupted execution contacted provider"
                        ),
                    )
                self.assertEqual(
                    {
                        path: path.read_bytes()
                        for path in self.root.rglob("*")
                        if path.is_file()
                    },
                    before,
                )

    def test_accounting_damage_keeps_verified_outcome_and_reopen_is_read_only(self):
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )

        for mode in ("ledger_collision", "unreadable_journal"):
            with self.subTest(mode=mode):
                self.approve()
                journal_paths = []

                def factory(plan, journal, mode=mode, journal_paths=journal_paths):
                    journal_paths.append(journal.path)
                    if mode == "ledger_collision":
                        journal.path.with_name(
                            "predecessor-contact-ledger.json"
                        ).write_bytes(b"retained collision")
                    return self.provider(plan, journal)

                read_bytes = Path.read_bytes

                def read(
                    path, mode=mode, journal_paths=journal_paths, read_bytes=read_bytes
                ):
                    if mode == "unreadable_journal" and path in journal_paths:
                        raise PermissionError("journal unavailable")
                    return read_bytes(path)

                with patch.object(Path, "read_bytes", read):
                    if mode == "ledger_collision":
                        with self.assertRaises(FileExistsError):
                            execute_parabricks(
                                self.approved, infrastructure_factory=factory
                            )
                    else:
                        result = execute_parabricks(
                            self.approved, infrastructure_factory=factory
                        )
                        self.assertEqual(result.disposition, "succeeded")
                        self.assertEqual(
                            review_execution_contacts(
                                self.approved.journey_file
                            ).status,
                            "incomplete",
                        )
                retained = read_parabricks_result(self.approved.journey_file)
                self.assertEqual(retained.disposition, "succeeded")
                self.assertEqual(len(retained.verified_objects), 4)
                review = review_execution_contacts(self.approved.journey_file)
                self.assertEqual(review.status, "drift")
                self.assertIn(
                    "materialized_ledger_malformed"
                    if mode == "ledger_collision"
                    else "materialized_ledger_projection_drift",
                    review.review.discrepancies,
                )
                self.assertFalse(review.review.fixture_eligible)
                if mode == "ledger_collision":
                    self.assertEqual(
                        journal_paths[0]
                        .with_name("predecessor-contact-ledger.json")
                        .read_bytes(),
                        b"retained collision",
                    )
                else:
                    self.assertIsNone(
                        JourneyPath.open(
                            Path(self.approved.journey_file)
                        ).state.outcome.journal_sha256
                    )
                before = {
                    path: path.read_bytes()
                    for path in self.root.rglob("*")
                    if path.is_file()
                }
                self.assertEqual(
                    execute_parabricks(
                        self.approved,
                        infrastructure_factory=lambda *_: self.fail(
                            "accounting damage restarted execution"
                        ),
                    ),
                    retained,
                )
                self.assertEqual(
                    {
                        path: path.read_bytes()
                        for path in self.root.rglob("*")
                        if path.is_file()
                    },
                    before,
                )
