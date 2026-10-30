import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

from bio_neb.failure import ProviderError
from bio_neb.journey_path import JourneyPath, Verified, project_execution
from bio_neb.parabricks.operation import execute_parabricks
from bio_neb.parabricks.plan import ExecutionCompositionRefused
from bio_neb.parabricks.review import answer_review
from bio_neb.providers.ports import Receipt
from tests.execution_contact_journal import (
    execution_contact_row,
    execution_contact_rows,
    execution_path_fact,
    rewrite_journey,
)
from tests.h100_stage_fakes import OUTPUTS, h100_observation_inputs, h100_provider
from tests.parabricks_operation_fakes import (
    execute_and_read_path,
    input_storage_provider,
)
from tests.reference_stage_fakes import seed_cpu
from tests.run_prerequisites import RunPrerequisites


class H100StageTests(RunPrerequisites, unittest.TestCase):
    def test_full_operation_uses_creation_proof_without_verification_heads(self):
        self.approve()

        def factory(plan, journal):
            provider = self.factory(plan, journal)
            original = provider.for_step

            def for_step(stage, step):
                ports = original(stage, step)
                ports.store.head = lambda *a, **kw: self.fail(
                    f"verification-only HEAD at {stage}/{step}"
                )
                return ports

            provider.for_step = for_step
            return provider

        result = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(result.disposition, "succeeded")
        # A worker-produced object rests on the read of its producer's worker
        # receipt; an object this application publishes rests on its own put.
        read_receipt, put = "store.read_small", "store.put"
        produced = {
            ("cpu_preparation", role): ("verify_bundle", read_receipt)
            for role in (
                "genomic_fasta",
                "fasta_fai",
                "reference_dict",
                "bwa_amb",
                "bwa_ann",
                "bwa_bwt",
                "bwa_pac",
                "bwa_sa",
                "worker_receipt",
                # With no supplied index the job fetched the source.
                "reference_source",
            )
        }
        produced.update(
            {
                ("h100", role): ("verify_outputs", read_receipt)
                for role in ("aligned_bam", "VCF", "worker_receipt")
            }
        )
        expected = {
            **produced,
            ("cpu_preparation", "source_manifest"): ("verify_bundle", put),
            ("cpu_preparation", "prepared_reference_manifest"): (
                "publish_prepared",
                put,
            ),
            ("h100", "execution_manifest_object"): ("publish_manifest", put),
            ("h100", "execution_manifest"): ("publish_manifest", put),
            # The result context is local content with no provider contact.
            ("h100", "execution_result_context"): ("publish_manifest", None),
        }
        pointers = (*result.reference.pointers, *result.h100.pointers)
        self.maxDiff = None
        proofs = {}
        for pointer in pointers:
            row = execution_contact_row(
                result.outcome.contact_journal, pointer.proof_ref
            )
            proofs[(pointer.producer.stage, pointer.role)] = (
                row["step"],
                row["logical_operation"],
            )
        self.assertEqual(proofs, expected)
        receipts = {
            pointer.producer.stage: pointer.proof_ref
            for pointer in pointers
            if pointer.role == "worker_receipt"
        }
        for pointer in pointers:
            if (pointer.producer.stage, pointer.role) in produced:
                self.assertEqual(
                    pointer.proof_ref, receipts[pointer.producer.stage], pointer.role
                )

    def test_successful_semantic_run_exports_inert_artifacts_with_unknown_counts(self):
        import io
        from contextlib import redirect_stdout

        from bio_neb.devtools.parabricks_fixture_export import main as export_main

        self.approve()
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        target = Path(self.temp.name) / "execution-fixture.json"
        with redirect_stdout(io.StringIO()):
            code = export_main(
                [
                    "export-execution",
                    "--journey-file",
                    self.approved.journey_file,
                    "--source-commit",
                    "a" * 40,
                    "--source-ticket",
                    "59",
                    "--evidence-kind",
                    "synthetic",
                    "--output",
                    str(target),
                ]
            )
        self.assertEqual(code, 0)
        document = json.loads(target.read_bytes())
        self.assertFalse(document["fixture_grants_live_authority"])
        scenario = document["scenarios"][0]
        self.assertEqual(scenario["disposition"], "succeeded")
        self.assertEqual(scenario["evidence_kind"], "synthetic")
        self.assertEqual(
            [item["role"] for item in scenario["verified_objects"]],
            ["aligned_bam", "VCF", "worker_receipt", "execution_manifest"],
        )
        self.assertEqual(
            scenario["contacts"]["provider_contact"]["status"], "unmeasurable"
        )
        self.assertNotIn("approved_input", target.read_text())
        from bio_neb.devtools.parabricks_fixture_export import load_execution_fixture

        content = target.read_bytes()
        with patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError("loader read original evidence"),
        ):
            loaded = load_execution_fixture(content)
        self.assertEqual(loaded.scenarios[0].journey_id, state.bound.run_id)
        self.assertIsNone(loaded.scenarios[0].contacts.provider_contact.count)
        document["scenarios"][0]["verified_objects"].pop()
        with self.assertRaisesRegex(
            ValueError, "completed execution fixture roles/path required"
        ):
            load_execution_fixture(json.dumps(document).encode())

    def test_cli_keeps_heartbeat_after_running_cpu_and_gpu_samples(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout

        from bio_neb.application.advancement import use_run_advancement_dependencies
        from bio_neb.cli import main
        from tests.sequential_investigation_support import (
            offline_investigation_dependencies,
        )

        self.approve(heartbeat_interval_seconds=0.01)
        approved_file = Path(self.temp.name) / "approved.json"
        approved_file.write_text(self.approved.model_dump_json())
        stdout, stderr = io.StringIO(), io.StringIO()

        def factory(plan, journal):
            provider = self.factory(plan, journal)
            original = provider.for_step

            def for_step(stage, step):
                ports = original(stage, step)
                if step in {"await_cpu", "await_h100"}:
                    terminal = provider.scripts[stage, step][0]
                    running = terminal.model_copy(
                        update={"values": {**terminal.values, "state": "RUNNING"}}
                    )
                    provider.scripts[stage, step].appendleft(running)
                return ports

            provider.for_step = for_step
            return provider

        # The Run's plan names its target, so the resumed Run begins
        # Investigation after the verified Execution, on offline adapters.
        with (
            use_run_advancement_dependencies(**offline_investigation_dependencies()),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = main(
                ["resume", "--journey-file", self.approved.journey_file],
                parabricks_infrastructure_factory=factory,
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(stdout.getvalue())["status"], "investigation_review_pending"
        )
        self.assertEqual(
            JourneyPath.open(Path(self.approved.journey_file)).state.disposition,
            "succeeded",
        )
        records = [json.loads(line) for line in stderr.getvalue().splitlines()]
        for stage in ("cpu_preparation", "h100"):
            index = next(
                i
                for i, row in enumerate(records)
                if row.get("kind") == "execution_observation"
                and row.get("stage") == stage
                and row.get("state") == "RUNNING"
            )
            self.assertTrue(
                any(
                    row.get("kind") == "execution_flow_heartbeat"
                    for row in records[index + 1 :]
                ),
                [
                    (row.get("kind"), row.get("stage"), row.get("state"))
                    for row in records
                ],
            )
        self.assertEqual(records[-1]["kind"], "execution_flow_finished")

    def test_complete_worker_checkpoints_survive_missing_logs_and_broken_display(self):
        self.approve()

        def no_logs(provider, plan):
            for stage, step in (
                ("cpu_preparation", "await_cpu"),
                ("h100", "await_h100"),
            ):
                if provider.scripts.get((stage, step)):
                    queue = provider.scripts[stage, step]
                    receipt = queue[0]
                    queue[0] = receipt.model_copy(
                        update={"values": {**receipt.values, "log": None}}
                    )

        def factory(plan, journal):
            provider = seed_cpu(
                input_storage_provider(plan, journal), plan, self.approved
            )
            no_logs(provider, plan)
            provider = h100_provider(provider, plan, self.approved, amend=no_logs)
            self.providers.append(provider)
            return provider

        def broken(record):
            raise BrokenPipeError("display closed")

        state = execute_and_read_path(
            self.approved, infrastructure_factory=factory, progress=broken
        )
        self.assertEqual(state.disposition, "succeeded", state.failures)
        self.assertEqual(state.reference.state, "prepared")
        self.assertIsInstance(state.h100, Verified)
        self.assertEqual(len(self.providers[-1].jobs_by_name), 2)

    def test_missing_internal_receipt_bytes_are_invariant_for_both_consumers(self):
        from bio_neb.failure import Invariant

        for stage, step in (
            ("cpu_preparation", "verify_bundle"),
            ("h100", "verify_outputs"),
        ):
            with self.subTest(stage=stage):
                self.approve()

                def corrupt(provider, plan, stage=stage, step=step):
                    provider.scripts[stage, step] = deque(
                        [Receipt(values={"bytes": 2, "sha256": "0" * 64})]
                    )

                def factory(plan, journal, stage=stage, corrupt=corrupt):
                    provider = seed_cpu(
                        input_storage_provider(plan, journal), plan, self.approved
                    )
                    if stage == "cpu_preparation":
                        corrupt(provider, plan)
                    return h100_provider(
                        provider,
                        plan,
                        self.approved,
                        amend=corrupt if stage == "h100" else None,
                    )

                with self.assertRaises(Invariant) as refused:
                    execute_and_read_path(self.approved, infrastructure_factory=factory)
                self.assertEqual(
                    refused.exception.invariant_id, "worker_receipt_read_invalid"
                )
                state = JourneyPath.open(Path(self.approved.journey_file)).state
                self.assertEqual(state.failures[-1].failure_class, "invariant")
                self.assertEqual(state.failures[-1].context["route"], "hold")
                self.assertNotIn("H100ArtifactsVerified", state.latest_event_ids)
                # Our bug holds the Execution for a fix: no outcome ends it, and
                # the escaping invariant names the retained failure (Q2).
                self.assertIsNone(state.outcome)
                self.assertEqual(state.status, "run_held")
                self.assertEqual(
                    refused.exception.evidence["stage_failed_event_id"],
                    state.latest_event_ids["StageFailed"],
                )
                self.assertEqual(refused.exception.evidence["stage"], stage)

    def test_invalid_durable_producer_facts_refuse_both_stage_checkpoints(self):
        import hashlib

        from bio_neb.content import canonical_json

        for stage, step in (
            ("cpu_preparation", "verify_bundle"),
            ("h100", "verify_outputs"),
        ):
            for defect in ("tool_exited", "worker_input_verified", "worker_request"):
                with self.subTest(stage=stage, defect=defect):
                    self.approve()

                    def corrupt(provider, plan, stage=stage, step=step, defect=defect):
                        key = (
                            plan.layout.reference_receipt_object_key
                            if stage == "cpu_preparation"
                            else plan.layout.h100_output_keys[2]
                        )
                        worker = json.loads(provider.objects[plan.storage.bucket, key])
                        # A job that fetched its source records the acquisition
                        # where a stored input records its verification.
                        kind = (
                            "worker_source_acquired"
                            if (stage, defect)
                            == ("cpu_preparation", "worker_input_verified")
                            else defect
                        )
                        fact = next(
                            fact
                            for fact in worker["validation"]
                            if fact["kind"] == kind
                        )
                        if defect == "tool_exited":
                            fact["returncode"] = 9
                        elif defect == "worker_input_verified":
                            fact["sha256"] = "0" * 64
                        else:
                            fact["request_sha256"] = "0" * 64
                        body = canonical_json(worker).encode()
                        provider.objects[plan.storage.bucket, key] = body
                        provider.scripts[stage, step] = deque(
                            [
                                Receipt(
                                    content=body,
                                    values={
                                        "bytes": len(body),
                                        "sha256": hashlib.sha256(body).hexdigest(),
                                    },
                                )
                            ]
                        )

                    def factory(plan, journal, stage=stage, corrupt=corrupt):
                        provider = seed_cpu(
                            input_storage_provider(plan, journal), plan, self.approved
                        )
                        if stage == "cpu_preparation":
                            corrupt(provider, plan)
                        return h100_provider(
                            provider,
                            plan,
                            self.approved,
                            amend=corrupt if stage == "h100" else None,
                        )

                    state = execute_and_read_path(
                        self.approved, infrastructure_factory=factory
                    )
                    self.assertEqual(state.status, "execution_review_pending")
                    self.assertEqual(state.failure_review.stage, stage)
                    self.assertEqual(state.disposition, "incomplete")
                    self.assertEqual(state.failures[-1].failure_class, "semantic")
                    self.assertEqual(
                        state.failures[-1].detail,
                        "cpu_receipt_invalid"
                        if stage == "cpu_preparation"
                        else "h100_receipt_invalid",
                    )
                    self.assertNotIn(
                        "ReferencePrepared"
                        if stage == "cpu_preparation"
                        else "H100ArtifactsVerified",
                        state.latest_event_ids,
                    )

    def test_malformed_logs_preserve_terminal_replacement_authority(self):
        for stage, step, name_field in (
            ("cpu_preparation", "await_cpu", "cpu_job_name"),
            ("h100", "await_h100", "h100_job_name"),
        ):
            for status in ("FAILED", "CANCELLED", "ERROR", "COMPLETED", "RUNNING"):
                with self.subTest(stage=stage, status=status):
                    self._exercise_malformed_log(stage, step, name_field, status)

    def _exercise_malformed_log(self, stage, step, name_field, status):
        approved = self.approve()
        provider = None
        plans = []
        bad_log = True

        def factory(plan, journal):
            nonlocal provider
            plans.append(plan)
            scripted = seed_cpu(
                input_storage_provider(plan, journal),
                plan,
                approved,
                job_id="job:" + plan.layout.cpu_job_name,
            )
            if provider is None:
                provider = scripted
            else:
                provider.scripts.update(scripted.scripts)
                for key, content in scripted.objects.items():
                    provider.objects.setdefault(key, content)

            def malformed(fake, selected):
                if bad_log:
                    fake.scripts[stage, step] = deque(
                        [
                            fake.scripts[stage, step][0].model_copy(
                                update={
                                    "values": {
                                        **fake.scripts[stage, step][0].values,
                                        "state": status,
                                        "log": "BIO_NEB_BROKEN",
                                    }
                                }
                            )
                        ]
                    )

            if stage == "cpu_preparation":
                malformed(provider, plan)
            return h100_provider(
                provider,
                plan,
                approved,
                amend=malformed if stage == "h100" else None,
                job_id="job:" + plan.layout.h100_job_name,
            )

        from bio_neb.parabricks.execution_observations import JobObservation

        with (
            patch(
                "bio_neb.parabricks.reference_stages.CPU_OBSERVATION",
                JobObservation(max_wait=0),
            ),
            patch(
                "bio_neb.parabricks.h100_stage.H100_OBSERVATION",
                JobObservation(max_wait=0),
            ),
        ):
            pending = execute_and_read_path(approved, infrastructure_factory=factory)
        if status == "COMPLETED":
            self.assertEqual(pending.disposition, "succeeded")
            self.assertEqual(len(provider.jobs_by_name), 2)
            return
        replacement = status in {"FAILED", "CANCELLED", "ERROR"}
        self.assertEqual(
            pending.failure_review.retry_consequence,
            "replace_job" if replacement else "reuse_intent",
        )
        journey = JourneyPath.open(Path(approved.journey_file))
        evidence = json.loads(journey.read_artifact(pending.failures[-1].evidence[0]))[
            "evidence"
        ]
        if replacement:
            self.assertEqual(evidence["log_window"]["window_text"], "BIO_NEB_BROKEN")
        else:
            self.assertEqual(pending.failures[-1].failure_class, "exhausted")
        original_name = getattr(plans[-1].layout, name_field)
        answer_review(
            approved.journey_file,
            pending.latest_event_ids["FailureReviewRequested"],
            {"action": "retry"},
        )
        bad_log = False
        done = execute_and_read_path(approved, infrastructure_factory=factory)
        self.assertEqual(done.disposition, "succeeded")
        self.assertEqual(
            getattr(plans[-1].layout, name_field) == original_name,
            not replacement,
        )
        self.assertEqual(len(provider.jobs_by_name), 3 if replacement else 2)

    def test_precreate_capacity_fallback_keeps_one_intent_and_request_versions(self):
        from bio_neb.parabricks.plan import rederive_plan
        from bio_neb.providers.fake import FakeProvider
        from tests.parabricks_operation_fakes import with_runtime_choices

        self.approve(configure=with_runtime_choices)
        provider = None
        requests = []
        selected_platform = None

        def factory(plan, journal):
            nonlocal provider
            provider = seed_cpu(
                input_storage_provider(plan, journal), plan, self.approved
            )
            original = provider.for_step

            def for_step(stage, step):
                nonlocal selected_platform
                if stage == "h100":
                    state = JourneyPath.open(Path(self.approved.journey_file)).state
                    current = rederive_plan(plan, state)
                    if current.runtimes.h100.platform != selected_platform:
                        selected_platform = current.runtimes.h100.platform
                        scripted = h100_provider(FakeProvider(), current, self.approved)
                        scripted.for_step(stage, step)
                        provider.scripts.update(scripted.scripts)
                        # Synthetic future worker bytes; no original job exists.
                        provider.objects.update(scripted.objects)
                        if selected_platform == "gpu-alternative":
                            provider.scripts["h100", "submit_h100"] = deque(
                                [
                                    ProviderError(
                                        "rejected",
                                        "no_capacity",
                                        {"resource_absent": True},
                                    )
                                ]
                            )
                ports = original(stage, step)
                if (stage, step) == ("h100", "submit_h100"):
                    submit = ports.jobs.submit

                    def captured(name, body):
                        requests.append((name, body["platform"]))
                        return submit(name, body)

                    ports.jobs.submit = captured
                return ports

            provider.for_step = for_step
            return provider

        done = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(done.disposition, "succeeded")
        self.assertEqual(
            [platform for _, platform in requests], ["gpu-alternative", "gpu-fallback"]
        )
        self.assertEqual(requests[0][0], requests[1][0])
        self.assertEqual(len(provider.jobs_by_name), 2)
        files = list(
            Path(self.approved.launch.work_directory).rglob("h100-request.json")
        )
        self.assertEqual(len(files), 2)
        self.assertNotEqual(files[0].read_bytes(), files[1].read_bytes())

    def test_concrete_job_contact_replacement_reuse_and_envelope_refusal(self):
        from types import SimpleNamespace

        from bio_neb.parabricks.h100_stage import compile_request
        from bio_neb.providers.nebius import NebiusProvider
        from bio_neb.providers.ports import Ports
        from tests.h100_stage_fakes import H100Cli
        from tests.reference_stage_fakes import credentials

        self.approve()
        backend = {}
        submissions = []
        fake = None

        def factory(plan, journal):
            nonlocal fake
            if fake is None:
                fake = seed_cpu(
                    input_storage_provider(plan, journal), plan, self.approved
                )
            job_id = "aijob-original" if not backend else "aijob-replacement"
            h100_provider(fake, plan, self.approved, job_id=job_id)
            original = fake.for_step
            concrete = None

            def for_step(stage, step):
                nonlocal concrete
                ports = original(stage, step)
                if stage != "h100":
                    return ports
                if concrete is None:
                    state = JourneyPath.open(Path(self.approved.journey_file)).state
                    config = h100_observation_inputs(plan)
                    request = compile_request(plan, state)
                    terminal = fake.scripts["h100", "await_h100"][0]
                    cli = H100Cli(
                        config,
                        request,
                        job_id=job_id,
                        initial_state="FAILED" if not backend else "COMPLETED",
                        log_outputs=(terminal.values["log"],),
                        ambiguous=True,
                    )
                    cli.states = iter((cli.initial_state,) * 10)
                    backend[request.deterministic_job_name] = cli

                    def make_provider():
                        return NebiusProvider(
                            selection=plan.launch.context,
                            endpoint_url=plan.storage.endpoint_url,
                            region=plan.storage.region,
                            journal=None,
                            cli_runner=cli,
                            credential_resolver=credentials,
                            registry_resolver=lambda _: SimpleNamespace(
                                index_digest=config.runtime.provider_image_index_digest,
                                linux_amd64_child_digest=config.runtime.worker_runtime_image.rsplit(
                                    "@", 1
                                )[1],
                            ),
                        )

                    concrete = make_provider()
                    submit = concrete.jobs.submit

                    def submitted(name, body):
                        receipt = submit(name, body)
                        self.assertEqual(
                            make_provider().jobs.submit(name, body).provider_ids,
                            receipt.provider_ids,
                        )
                        submissions.append((name, body, make_provider))
                        return receipt

                    concrete.jobs.submit = submitted
                return Ports(ports.store, concrete.jobs, ports.identity)

            fake.for_step = for_step
            return fake

        pending = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(pending.failures[-1].detail, "h100_job_failed")
        answer_review(
            self.approved.journey_file,
            pending.latest_event_ids["FailureReviewRequested"],
            {"action": "retry"},
        )
        done = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(done.disposition, "succeeded")
        self.assertEqual(
            {cli.job_id for cli in backend.values()},
            {"aijob-original", "aijob-replacement"},
        )
        for name, body, make_provider in submissions:
            prior = json.dumps(backend[name].spec, sort_keys=True)
            with self.assertRaises(ProviderError) as refused:
                make_provider().jobs.submit(
                    name, {**body, "preset": "unapproved-preset"}
                )
            self.assertEqual(refused.exception.failure_class, "rejected")
            self.assertEqual(refused.exception.detail, "provider_envelope_mismatch")
            self.assertEqual(json.dumps(backend[name].spec, sort_keys=True), prior)
            self.assertEqual(
                sum(
                    call[:3] == ("ai", "job", "create") for call in backend[name].calls
                ),
                1,
            )

    def test_gpu_amendment_replaces_existing_job_through_public_operation(self):
        from tests.parabricks_operation_fakes import with_runtime_choices

        self.approve(configure=with_runtime_choices)
        provider = None
        plans = []
        fail_worker = True

        def factory(plan, journal):
            nonlocal provider
            plans.append(plan)
            if provider is None:
                provider = seed_cpu(
                    input_storage_provider(plan, journal), plan, self.approved
                )

            def amend(fake, selected):
                if fail_worker:
                    fake.scripts["h100", "await_h100"] = deque(
                        [Receipt(values={"state": "FAILED"})]
                    )

            return h100_provider(
                provider,
                plan,
                self.approved,
                amend=amend,
                job_id="job:" + plan.layout.h100_job_name,
            )

        pending = execute_and_read_path(self.approved, infrastructure_factory=factory)
        old = dict(provider.objects)
        request_id = pending.latest_event_ids["FailureReviewRequested"]
        before = Path(self.approved.journey_file).read_bytes()
        with self.assertRaises(ExecutionCompositionRefused) as refused:
            answer_review(
                self.approved.journey_file,
                request_id,
                {"action": "amend", "field": "gpu_class", "value": "unapproved"},
            )
        self.assertEqual(refused.exception.code, "lineage")
        self.assertEqual(refused.exception.details, ("invalid_gpu_class_amendment",))
        self.assertEqual(Path(self.approved.journey_file).read_bytes(), before)
        answer = answer_review(
            self.approved.journey_file,
            request_id,
            {"action": "amend", "field": "gpu_class", "value": "fallback"},
        )
        fail_worker = False
        done = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(done.disposition, "succeeded")
        self.assertEqual(len(provider.jobs_by_name), 3)
        self.assertEqual(plans[1].runtimes.h100.platform, "gpu-fallback")
        self.assertEqual(plans[1].runtimes.h100.disk_size_bytes, 536870912000)
        self.assertNotEqual(
            plans[0].layout.h100_job_name, plans[1].layout.h100_job_name
        )
        for key, content in old.items():
            self.assertEqual(provider.objects[key], content)
        self.assertIn(
            answer.latest_event_ids["FailureReviewAnswered"],
            done.rests_on[done.latest_event_ids["H100ArtifactsVerified"]],
        )

    def test_replacement_then_publication_retry_keeps_one_job_per_answer(self):
        self.approve()
        provider = None
        phase = "failed_worker"
        plans = []

        def factory(plan, journal):
            nonlocal provider
            plans.append(plan)
            if provider is None:
                provider = seed_cpu(
                    input_storage_provider(plan, journal), plan, self.approved
                )

            def amend(fake, selected):
                if phase == "failed_worker":
                    fake.scripts["h100", "await_h100"] = deque(
                        [Receipt(values={"state": "FAILED"})]
                    )

            h100_provider(
                provider,
                plan,
                self.approved,
                amend=amend,
                job_id="job:" + plan.layout.h100_job_name,
            )
            original = provider.for_step

            def for_step(stage, step):
                ports = original(stage, step)
                if (stage, step) == (
                    "h100",
                    "publish_manifest",
                ) and phase == "lost_publication":
                    put = ports.store.put

                    def lost(*args, **kwargs):
                        put(*args, **kwargs)
                        raise ProviderError("exhausted", "manifest_publication_lost")

                    ports.store.put = lost
                return ports

            provider.for_step = for_step
            return provider

        pending = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(pending.failure_review.retry_consequence, "replace_job")
        old = dict(provider.objects)
        replaced = answer_review(
            self.approved.journey_file,
            pending.latest_event_ids["FailureReviewRequested"],
            {"action": "retry"},
        )
        phase = "lost_publication"
        publication = execute_and_read_path(
            self.approved, infrastructure_factory=factory
        )
        self.assertEqual(publication.failures[-1].detail, "manifest_publication_lost")
        self.assertEqual(publication.failure_review.retry_consequence, "reuse_intent")
        self.assertEqual(len(provider.jobs_by_name), 3)
        self.assertNotEqual(
            plans[0].layout.h100_job_name, plans[1].layout.h100_job_name
        )
        self.assertTrue(
            set(plans[0].layout.h100_output_keys).isdisjoint(
                plans[1].layout.h100_output_keys
            )
        )
        for key, content in old.items():
            self.assertEqual(provider.objects[key], content)
        prior_manifest = provider.objects[
            plans[-1].storage.bucket, plans[-1].layout.h100_manifest_object_key
        ]
        retried = answer_review(
            self.approved.journey_file,
            publication.latest_event_ids["FailureReviewRequested"],
            {"action": "retry"},
        )
        phase = "complete"
        done = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(done.disposition, "succeeded")
        self.assertEqual(len(provider.jobs_by_name), 3)
        self.assertEqual(plans[1].layout, plans[2].layout)
        self.assertEqual(
            provider.objects[
                plans[-1].storage.bucket, plans[-1].layout.h100_manifest_object_key
            ],
            prior_manifest,
        )
        dependencies = done.rests_on[done.latest_event_ids["H100ArtifactsVerified"]]
        for answer in (replaced, retried):
            self.assertIn(
                answer.latest_event_ids["FailureReviewAnswered"], dependencies
            )
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )

        review = review_execution_contacts(self.approved.journey_file)
        self.assertEqual(review.review.ledger.h100_create.count, 2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.providers = []
        self.manifest_targets = []
        self.amend = None

    def approve(self, *, configure=None, **launch_overrides):
        """Restore the class's approved Run for this case's final choices.

        No case binds a prebuilt index, so the CPU preparation stage runs before
        H100 (owner, 2026-10-29, #464). Each call, including each subTest,
        gets the approved baseline restored in place; execution runs per case.
        """
        self.approved = self.prerequisite(
            "approved",
            configure=configure,
            **launch_overrides,
        ).value
        return self.approved

    def factory(self, plan, journal):
        self.manifest_targets.append(
            (plan.storage.bucket, plan.layout.h100_manifest_object_key)
        )
        provider = seed_cpu(input_storage_provider(plan, journal), plan, self.approved)
        provider = h100_provider(provider, plan, self.approved, amend=self.amend)
        self.providers.append(provider)
        return provider

    def refusing_first_cpu_preflight(self):
        """`factory`, with the first Pass's CPU stage refused at its identity
        preflight, before any job contact: the Run stops at that stage's failure
        Review, the designed two-Pass stop now that a Run without a bound
        prebuilt index prepares on CPU with no Review (owner, 2026-10-29, #464)."""
        refused = []

        def factory(plan, journal):
            provider = self.factory(plan, journal)
            if not refused:
                refused.append(plan)
                provider.scripts["cpu_preparation", "preflight"] = deque(
                    [ProviderError("rejected", "identity_refused")]
                )
            return provider

        return factory

    def test_full_operation_verifies_outputs_and_reopens_without_contact(self):
        self.approve()
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        self.assertEqual(state.disposition, "succeeded")
        self.assertEqual(state.status, "execution_artifacts_verified")
        self.assertIsInstance(state.h100, Verified)
        journey = JourneyPath.open(Path(self.approved.journey_file))
        self.assertEqual(
            [e.event_type for e in journey.read()],
            [
                "ExecutionBound",
                "InputsVerified",
                "ReferencePrepared",
                "H100ArtifactsVerified",
                "ExecutionOutcome",
            ],
        )
        self.assertEqual(state, project_execution(journey.read()))
        self.assertEqual(len(self.providers[-1].jobs_by_name), 2)
        manifest = next(
            p for p in state.h100.pointers if p.role == "execution_manifest"
        )
        self.assertTrue(manifest.location.startswith("artifacts/sha256/"))
        document = json.loads(journey.read_artifact(manifest))
        self.assertEqual(
            document["prepared_reference_receipt_id"], state.reference.receipt_id
        )
        self.assertEqual(
            [i["role"] for i in document["outputs"]], ["aligned_bam", "VCF"]
        )
        self.assertEqual(document["outputs"][1]["byte_count"], len(OUTPUTS["VCF"]))
        rows = execution_contact_rows(state.outcome.contact_journal)
        self.assertFalse(any("log" in row for row in rows))
        self.assertEqual(rows[-1]["row_kind"], "final")
        self.assertTrue(rows[-1]["complete"])
        with patch(
            "bio_neb.parabricks.operation._make_provider",
            side_effect=AssertionError("contact"),
        ):
            self.assertEqual(execute_and_read_path(self.approved), state)
        before = Path(self.approved.journey_file).read_bytes()
        reopened = execute_parabricks(
            self.approved,
            infrastructure_factory=lambda *_: self.fail("succeeded run contacted"),
        )
        self.assertEqual(
            (reopened.disposition, reopened.outcome_event_id),
            ("succeeded", state.latest_event_ids["ExecutionOutcome"]),
        )
        self.assertEqual(Path(self.approved.journey_file).read_bytes(), before)

    def test_output_drift_cannot_publish_manifest_or_success(self):
        from bio_neb.application.verification_policy import ObjectVerificationOverrides

        self.approve(
            verification_overrides=ObjectVerificationOverrides(
                h100_outputs="independent_full_read",
                reason="explicit drift audit",
            )
        )

        def amend(provider, plan):
            provider.objects[plan.storage.bucket, plan.layout.h100_output_keys[1]] = (
                b"changed"
            )

        self.amend = amend
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        self.assertNotIn("H100ArtifactsVerified", state.latest_event_ids)
        self.assertEqual(state.failures[-1].detail, "artifact_identity_mismatch")
        self.assertNotIn(self.manifest_targets[-1], self.providers[-1].objects)

    def test_missing_durable_validation_never_publishes_h100_success(self):
        self.approve()

        def amend(provider, plan):
            key = plan.storage.bucket, plan.layout.h100_output_keys[2]
            worker = json.loads(provider.objects[key])
            worker.pop("validation")
            from bio_neb.content import canonical_json

            provider.objects[key] = canonical_json(worker).encode()

        self.amend = amend
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        self.assertEqual(state.status, "execution_review_pending")
        self.assertEqual(state.failure_review.stage, "h100")
        self.assertEqual(state.disposition, "incomplete")
        self.assertEqual(state.failures[-1].failure_class, "semantic")
        self.assertEqual(state.failures[-1].detail, "h100_receipt_invalid")
        self.assertNotIn("H100ArtifactsVerified", state.latest_event_ids)
        self.assertNotIn(self.manifest_targets[-1], self.providers[-1].objects)

    def test_local_request_placement_failure_stops_before_each_job_submission(self):
        for stage, submitted in (("cpu_preparation", 0), ("h100", 1)):
            with self.subTest(stage=stage):
                approved = self.approve()
                providers = []

                def factory(
                    plan, journal, approved=approved, stage=stage, providers=providers
                ):
                    provider = seed_cpu(
                        input_storage_provider(plan, journal), plan, approved
                    )
                    provider = h100_provider(provider, plan, approved)
                    job_name = (
                        plan.layout.cpu_job_name
                        if stage == "cpu_preparation"
                        else plan.layout.h100_job_name
                    )
                    target = Path(plan.launch.work_directory) / job_name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("obstruct request directory")
                    providers.append(provider)
                    return provider

                state = execute_and_read_path(approved, infrastructure_factory=factory)
                self.assertEqual(state.disposition, "incomplete")
                failure = state.failures[-1]
                self.assertEqual(
                    (failure.stage, failure.failure_class, failure.detail),
                    (stage, "rejected", "worker_request_placement_failed"),
                )
                self.assertEqual(len(providers[-1].jobs_by_name), submitted)
                before = Path(approved.journey_file).read_bytes()
                reopened = execute_parabricks(
                    approved,
                    infrastructure_factory=lambda *_: self.fail("review contacted"),
                )
                self.assertEqual(
                    (reopened.disposition, reopened.pending_review),
                    ("incomplete", state.failure_review),
                )
                self.assertEqual(Path(approved.journey_file).read_bytes(), before)

    def test_completed_result_accepts_equivalent_local_path_spellings(self):
        import os

        from bio_neb.parabricks.result import read_completed_parabricks_result

        self.approve()
        execute_and_read_path(self.approved, infrastructure_factory=self.factory)
        expected = read_completed_parabricks_result(self.approved.journey_file)
        file = Path(self.approved.journey_file)
        alias = file.parent / "alias"
        alias.symlink_to(file.parent, target_is_directory=True)
        for spelling in (
            os.path.relpath(file),
            str(alias / file.name),
            str(file.parent) + "/./" + file.name,
        ):
            with self.subTest(spelling=spelling):
                self.assertEqual(read_completed_parabricks_result(spelling), expected)

    def test_result_context_distinguishes_composed_plan_from_verified_transfer(self):
        from bio_neb.parabricks.result import read_completed_parabricks_result

        self.approve()
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        pointers = {p.role: p for p in state.h100.pointers}
        context = pointers["execution_result_context"]
        remote = pointers["execution_manifest_object"]
        self.assertEqual(context.grounding, "agent_composed")
        self.assertEqual(remote.grounding, "verified")
        self.assertNotEqual(context.proof_ref, remote.proof_ref)
        row = execution_contact_row(state.outcome.contact_journal, context.proof_ref)
        self.assertEqual(
            (row["stage"], row["step"], row["phase"]),
            ("h100", "publish_manifest", "verified"),
        )
        event = state.latest_event_ids["H100ArtifactsVerified"]
        self.assertIn(state.latest_event_ids["ExecutionBound"], state.rests_on[event])
        result = read_completed_parabricks_result(self.approved.journey_file)
        self.assertEqual(result.selected_context, self.approved.launch.context)
        self.assertEqual(result.proposal_id, self.approved.proposal.proposal_id)
        self.assertEqual(result.authority_id, self.approved.authority.authority_id)
        self.assertEqual(result.storage, self.approved.launch.object_storage)

    def test_result_consumers_and_contact_review_use_new_path(self):
        from bio_neb.investigation.bam_materialization import (
            _execution_lineage as bam_lineage,
        )
        from bio_neb.investigation.vcf_materialization import _lineage as vcf_lineage
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )
        from bio_neb.parabricks.result import read_completed_parabricks_result

        self.approve()
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        result = read_completed_parabricks_result(self.approved.journey_file)
        self.assertEqual(
            result.h100_checkpoint_event_id,
            state.latest_event_ids["H100ArtifactsVerified"],
        )
        self.assertEqual(vcf_lineage(result)[1].role, "VCF")
        self.assertEqual(bam_lineage(result)[1].role, "aligned_bam")
        reviewed = review_execution_contacts(self.approved.journey_file)
        self.assertEqual(reviewed.status, "complete", reviewed.review.discrepancies)
        self.assertEqual(reviewed.review.ledger.fresh_bucket.count, 1)
        self.assertEqual(reviewed.review.ledger.cpu_preparation_create.count, 1)
        self.assertEqual(reviewed.review.ledger.h100_create.count, 1)
        self.assertEqual(reviewed.review.ledger.network_contact.status, "unmeasurable")
        self.assertTrue(reviewed.review.fixture_eligible)

    def test_scientific_continuation_publication_consumes_new_path(self):
        from bio_neb.investigation.scientific_continuation import (
            PublishScientificContinuationCommand,
            publish_scientific_continuation,
        )

        self.approve()
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        output = Path(self.temp.name) / "continuation.json"
        receipt = publish_scientific_continuation(
            PublishScientificContinuationCommand(
                execution_journey_file=self.approved.journey_file, output=str(output)
            )
        )
        self.assertEqual(
            receipt.frontier_event_id, state.latest_event_ids["H100ArtifactsVerified"]
        )
        packet = json.loads(output.read_text())
        self.assertEqual(packet["vcf"]["byte_count"], len(OUTPUTS["VCF"]))
        self.assertFalse(packet["grants_live_access"])

    def test_review_binds_journal_and_contact_receipt_bytes(self):
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )

        self.approve()
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        journal = Path(state.outcome.contact_journal)
        original = journal.read_bytes()
        ref = next(
            row["receipt"]["ref"]
            for row in execution_contact_rows(journal)
            if row["receipt"] and Path(row["receipt"]["ref"]).is_absolute()
        )
        receipt = Path(ref)
        receipt.write_text("{}")
        review = review_execution_contacts(self.approved.journey_file)
        self.assertEqual(review.status, "drift")
        self.assertIn("contact_receipt_drift", review.review.discrepancies)
        journal.write_bytes(original[:-1])
        review = review_execution_contacts(self.approved.journey_file)
        self.assertFalse(review.review.complete)
        self.assertFalse(review.review.binding_matches)

    def test_missing_and_malformed_ledger_keep_available_journal_resources(self):
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )

        self.approve()
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        ledger = Path(state.outcome.contact_journal).with_name(
            "predecessor-contact-ledger.json"
        )
        ledger.unlink()
        missing = review_execution_contacts(self.approved.journey_file)
        self.assertEqual(missing.status, "incomplete")
        self.assertIn("materialized_ledger_missing", missing.review.discrepancies)
        ledger.write_text("{broken")
        malformed = review_execution_contacts(self.approved.journey_file)
        self.assertEqual(malformed.status, "drift")
        self.assertIn("materialized_ledger_malformed", malformed.review.discrepancies)
        self.assertEqual(missing.review.ledger.h100_create.unit_ids, ("aijob-exact",))
        self.assertEqual(malformed.review.ledger.h100_create.unit_ids, ("aijob-exact",))

    def test_approved_second_pass_counts_each_named_resource_once(self):
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )

        self.approved = self.prerequisite("approved").value
        factory = self.refusing_first_cpu_preflight()
        pending = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(pending.status, "execution_review_pending")
        self.assertEqual(pending.failure_review.stage, "cpu_preparation")
        first_journal = Path(pending.outcome.contact_journal).read_bytes()
        answer_review(
            self.approved.journey_file,
            pending.latest_event_ids["FailureReviewRequested"],
            {"action": "retry"},
        )
        completed = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(completed.disposition, "succeeded")
        self.assertEqual(completed.pass_, 2)
        review = review_execution_contacts(self.approved.journey_file)
        self.assertEqual(review.status, "complete", review.review.discrepancies)
        self.assertEqual(review.review.ledger.fresh_bucket.count, 1)
        self.assertEqual(review.review.ledger.cpu_preparation_create.count, 1)
        self.assertEqual(review.review.ledger.h100_create.count, 1)
        self.assertEqual(len(review.review.ledger.prior_pass_journal_bindings), 1)
        self.assertEqual(
            Path(pending.outcome.contact_journal).read_bytes(), first_journal
        )

    def test_review_includes_interrupted_second_pass_and_its_submitted_job(self):
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )
        from bio_neb.parabricks.result import read_parabricks_result

        self.approved = self.prerequisite("approved").value
        pending = execute_and_read_path(
            self.approved, infrastructure_factory=self.refusing_first_cpu_preflight()
        )
        original = Path(pending.outcome.contact_journal).read_bytes()
        answer_review(
            self.approved.journey_file,
            pending.latest_event_ids["FailureReviewRequested"],
            {"action": "retry"},
        )

        def interrupted_factory(plan, journal):
            provider = self.factory(plan, journal)
            original_for_step = provider.for_step

            def for_step(stage, step):
                ports = original_for_step(stage, step)
                if (stage, step) == ("cpu_preparation", "await_cpu"):

                    def interrupted(*args, **kwargs):
                        raise RuntimeError("offline observation interrupted")

                    ports.jobs.observe = interrupted
                return ports

            provider.for_step = for_step
            return provider

        with self.assertRaisesRegex(RuntimeError, "offline observation interrupted"):
            execute_and_read_path(
                self.approved, infrastructure_factory=interrupted_factory
            )
        before = Path(self.approved.journey_file).read_bytes()
        with self.assertRaisesRegex(
            ValueError, "unfinished_execution_requires_new_run"
        ):
            execute_and_read_path(
                self.approved,
                infrastructure_factory=lambda *_: self.fail(
                    "interrupted pass contacted"
                ),
            )
        self.assertEqual(Path(self.approved.journey_file).read_bytes(), before)
        self.assertEqual(
            read_parabricks_result(self.approved.journey_file).execution_pass, 2
        )
        report = review_execution_contacts(self.approved.journey_file)
        self.assertEqual(report.status, "incomplete")
        self.assertIsNone(report.journey_pointer_event_id)
        self.assertEqual(
            [p.execution_pass for p in report.review.retained_passes], [1, 2]
        )
        # Journals are keyed by the coordinator Pass: the review names the
        # interrupted second pass's own journal, not the first one.
        self.assertEqual(
            report.journal_locator, report.review.retained_passes[1].journal_locator
        )
        self.assertNotEqual(report.journal_locator, pending.outcome.contact_journal)
        self.assertEqual(report.review.retained_passes[0].disposition, "incomplete")
        self.assertIsNone(report.review.retained_passes[1].disposition)
        self.assertIn("unfinished_step", report.review.discrepancies)
        metric = report.review.ledger.cpu_preparation_create
        self.assertEqual(metric.status, "unmeasurable")
        self.assertIn("aijob-reference-exact", metric.observed_unit_ids)
        self.assertEqual(len(report.review.ledger.fresh_bucket.observed_unit_ids), 1)
        self.assertEqual(Path(pending.outcome.contact_journal).read_bytes(), original)

    def test_failed_submission_can_retry_at_designed_h100_review(self):
        self.approve()
        persistent = None
        names = []

        def factory(plan, journal):
            nonlocal persistent
            names.append(plan.layout.h100_job_name)
            if persistent is None:
                persistent = seed_cpu(
                    input_storage_provider(plan, journal), plan, self.approved
                )
            return h100_provider(persistent, plan, self.approved, amend=self.amend)

        def amend(provider, plan):
            provider.scripts["h100", "submit_h100"] = deque(
                [ProviderError("rejected", "fix_configuration")]
            )

        self.amend = amend
        pending = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(pending.status, "execution_review_pending")
        # The approved input carries a runtime selection, so the H100 Review also
        # offers a GPU class amendment.
        self.assertEqual(pending.failure_review.amendable, ("gpu_class",))
        self.assertEqual(
            pending.failure_review.legal_actions, ("retry", "abort", "amend")
        )
        self.assertEqual(pending.failure_review.retry_consequence, "reuse_intent")
        self.assertEqual(len(persistent.jobs_by_name), 1)
        answer_review(
            self.approved.journey_file,
            pending.latest_event_ids["FailureReviewRequested"],
            {"action": "retry"},
        )
        self.amend = None
        done = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(len(names), 2)
        self.assertEqual(done.disposition, "succeeded")
        self.assertEqual(done.pass_, 2)
        self.assertEqual(done.unresolved_failure_ids, ())
        self.assertEqual(len(persistent.jobs_by_name), 2)
        self.assertEqual(names[0], names[1])

    def test_a_read_fault_on_a_created_job_offers_retry_or_abort_only(self):
        # A poll fault on a job that went on to complete offered
        # `amend gpu_class`, a second paid H100. A fault read from a job that exists
        # and has not failed offers retry or abort only; retry re-attaches by name.
        self.approve()
        persistent = None

        def factory(plan, journal):
            nonlocal persistent
            if persistent is None:
                persistent = seed_cpu(
                    input_storage_provider(plan, journal), plan, self.approved
                )
            return h100_provider(persistent, plan, self.approved, amend=self.amend)

        def amend(provider, plan):
            provider.scripts["h100", "await_h100"] = deque(
                [
                    ProviderError(
                        "rejected",
                        "terminal provider state changed: COMPLETED -> RUNNING",
                        {"provider_job_id": "aijob-exact"},
                    )
                ]
            )

        self.amend = amend
        pending = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(pending.status, "execution_review_pending")
        self.assertEqual(pending.failure_review.legal_actions, ("retry", "abort"))
        self.assertEqual(pending.failure_review.amendable, ())
        self.assertEqual(pending.failure_review.amendment_consequences, {})
        # The CPU preparation's job and this one H100 job; no second H100.
        self.assertEqual(len(persistent.jobs_by_name), 2)

    def test_failed_submission_review_keeps_known_id_and_cause_without_exact_total(
        self,
    ):
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )

        self.approve()

        def amend(provider, plan):
            provider.scripts["h100", "submit_h100"] = deque(
                [
                    ProviderError(
                        "rejected",
                        "submission_response_lost",
                        {"provider_job_id": "aijob-known-failed"},
                    )
                ]
            )

        self.amend = amend
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        self.assertEqual(state.status, "execution_review_pending")
        report = review_execution_contacts(self.approved.journey_file)
        metric = report.review.ledger.h100_create
        self.assertEqual(metric.status, "unmeasurable")
        self.assertEqual(metric.observed_unit_ids, ("aijob-known-failed",))
        failed = next(a for a in report.review.logical_attempts if a.failure_class)
        self.assertEqual(failed.operation, "jobs.submit")
        self.assertEqual(failed.resource_kind, "h100_job")
        self.assertEqual(failed.failure_class, "rejected")
        self.assertEqual(failed.failure_detail, "submission_response_lost")

    def test_real_job_runner_consumes_h100_recipe_and_reuses_named_job(self):
        from types import SimpleNamespace

        from bio_neb.parabricks.h100_stage import compile_request
        from bio_neb.providers.nebius import NebiusProvider
        from bio_neb.providers.ports import Ports
        from tests.reference_stage_fakes import CpuCli, credentials

        self.approve()

        class H100Cli(CpuCli):
            def _job(inner, state):
                result = super()._job(state)
                result["metadata"]["id"] = "aijob-exact"
                if result["spec"]:
                    result["spec"]["disk"]["size_bytes"] = str(
                        inner.config.runtime.disk_size_bytes
                    )
                    result["spec"]["shm_size_bytes"] = str(
                        inner.config.runtime.shared_memory_bytes
                    )
                return result

        def factory(plan, journal):
            fake = self.factory(plan, journal)
            original = fake.for_step
            jobs = None

            def for_step(stage, step):
                nonlocal jobs
                ports = original(stage, step)
                if stage != "h100":
                    return ports
                if jobs is None:
                    state = project_execution(
                        JourneyPath.open(Path(self.approved.journey_file)).read()
                    )
                    config = h100_observation_inputs(plan)
                    request = compile_request(plan, state)
                    terminal = fake.scripts["h100", "await_h100"][0]
                    cli = H100Cli(
                        config, request, log_outputs=(terminal.values["log"],)
                    )
                    cli.states = iter(("COMPLETED",))
                    self.cli = cli

                    def provider():
                        return NebiusProvider(
                            selection=plan.launch.context,
                            endpoint_url=plan.storage.endpoint_url,
                            region=plan.storage.region,
                            journal=None,
                            cli_runner=cli,
                            credential_resolver=credentials,
                            registry_resolver=lambda _: SimpleNamespace(
                                index_digest=config.runtime.provider_image_index_digest,
                                linux_amd64_child_digest=config.runtime.worker_runtime_image.rsplit(
                                    "@", 1
                                )[1],
                            ),
                        )

                    class LostResponse:
                        def submit(inner, name, values):
                            provider().jobs.submit(name, values)
                            cli.initial_state = "COMPLETED"
                            inner.current = provider().jobs
                            return inner.current.submit(name, values)

                        def observe(inner, job_id, *, window):
                            return inner.current.observe(job_id, window=window)

                    jobs = LostResponse()
                return Ports(ports.store, jobs, ports.identity)

            return SimpleNamespace(for_step=for_step)

        done = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(self.cli.config.runtime.disk_size, "250Gi")
        self.assertEqual(done.disposition, "succeeded")
        self.assertEqual(
            sum(call[:3] == ("ai", "job", "create") for call in self.cli.calls), 1
        )

    def test_public_materialization_acquires_exact_bam_and_vcf(self):
        from bio_neb.investigation.bam_materialization import (
            MaterializeCurrentBamCommand,
            materialize_current_bam,
        )
        from bio_neb.investigation.vcf_materialization import (
            MaterializeCurrentVcfCommand,
            materialize_current_vcf,
        )
        from bio_neb.parabricks.result import read_completed_parabricks_result
        from tests.h100_stage_fakes import OneGet as _OneGet

        self.approve()
        execute_and_read_path(self.approved, infrastructure_factory=self.factory)
        result = read_completed_parabricks_result(self.approved.journey_file)
        vcf = _OneGet(OUTPUTS["VCF"])
        materialized = materialize_current_vcf(
            MaterializeCurrentVcfCommand(
                execution_journey_file=self.approved.journey_file,
                output_directory=str(Path(self.temp.name) / "vcf"),
            ),
            object_read_factory=lambda _: vcf,
        )
        self.assertEqual(len(vcf.calls), 1)
        self.assertEqual(
            Path(materialized.receipt.materialized_vcf.path).read_bytes(),
            OUTPUTS["VCF"],
        )
        bam = _OneGet(OUTPUTS["aligned_bam"])
        descriptor = result.verified_objects[0]
        acquired = materialize_current_bam(
            MaterializeCurrentBamCommand(
                execution_journey_file=self.approved.journey_file,
                output_directory=str(Path(self.temp.name) / "bam"),
            ),
            object_read_factory=lambda _: bam,
        )
        # The one GET names the Run's own verified BAM; no caller supplied it.
        self.assertEqual(bam.calls, [descriptor.object_key])
        self.assertEqual(
            Path(acquired.receipt.materialized_bam.path).read_bytes(),
            OUTPUTS["aligned_bam"],
        )

        self.assertEqual(materialized.receipt.lineage.execution_id, result.execution_id)
        self.assertEqual(acquired.receipt.lineage.execution_id, result.execution_id)
        self.assertEqual(
            materialized.receipt.lineage.h100_request_id, result.h100_request_id
        )
        self.assertEqual(
            acquired.receipt.lineage.provider_job_id, result.provider_job_id
        )
        self.assertEqual(
            materialized.receipt.materialized_vcf.sha256,
            result.verified_objects[1].sha256,
        )
        self.assertEqual(acquired.materialized_bam.sha256, descriptor.sha256)

    def test_completed_result_rejects_a_noncanonical_remote_location(self):
        from bio_neb.parabricks.result import read_completed_parabricks_result

        self.approve()
        execute_and_read_path(self.approved, infrastructure_factory=self.factory)
        path = Path(self.approved.journey_file)
        original = path.read_bytes()
        for suffix in ("?other-object", "#other-object"):

            def relocate(events, suffix=suffix):
                verified = execution_path_fact(events, "H100ArtifactsVerified")
                pointer = next(
                    p
                    for p in verified["payload"]["pointers"]
                    if p["role"] == "execution_manifest_object"
                )
                pointer["location"] += suffix

            path.write_bytes(original)
            rewrite_journey(path, relocate)
            with (
                self.subTest(suffix=suffix),
                self.assertRaisesRegex(ValueError, "object_location_invalid"),
            ):
                read_completed_parabricks_result(str(path))

    def test_runtime_disk_quantity_rejects_invalid_or_fractional_bytes(self):
        from bio_neb.parabricks.execution_runtime import (
            CpuReferenceRuntime,
            H100Runtime,
        )

        self.assertEqual(CpuReferenceRuntime().disk_size_bytes, 107374182400)
        for value, expected in (
            ("500Gi", 500 * 1024**3),
            ("1.5Gi", 1610612736),
            ("500G", 500_000_000_000),
            ("42", 42),
        ):
            runtime = H100Runtime(
                recipe_id="recipe", resource_profile_id="profile", disk_size=value
            )
            self.assertEqual(runtime.disk_size_bytes, expected)
            self.assertEqual(
                CpuReferenceRuntime(disk_size=value).disk_size_bytes, expected
            )
        for value in ("0Gi", "-1Gi", "0.1", "500garbage"):
            with self.subTest(cpu_disk=value), self.assertRaises(ValueError):
                CpuReferenceRuntime(disk_size=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                H100Runtime(
                    recipe_id="recipe", resource_profile_id="profile", disk_size=value
                )

    def test_abort_answer_preserves_the_closed_pass_accounting(self):
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )

        self.approved = self.prerequisite("approved").value
        pending = execute_and_read_path(
            self.approved, infrastructure_factory=self.refusing_first_cpu_preflight()
        )
        before = review_execution_contacts(self.approved.journey_file)
        answer_review(
            self.approved.journey_file,
            pending.latest_event_ids["FailureReviewRequested"],
            {"action": "abort"},
        )
        after = review_execution_contacts(self.approved.journey_file)
        self.assertEqual(before, after)
        self.assertEqual(after.status, "complete")
        with patch(
            "bio_neb.parabricks.operation._make_provider",
            side_effect=AssertionError("contact"),
        ):
            self.assertEqual(execute_and_read_path(self.approved).disposition, "failed")
