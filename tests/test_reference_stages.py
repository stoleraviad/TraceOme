import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

from bio_neb.failure import ProviderError
from bio_neb.journey_path import JourneyPath, Prepared, project_execution
from bio_neb.parabricks.review import answer_review
from bio_neb.providers.ports import Receipt
from tests.execution_contact_journal import execution_contact_rows
from tests.h100_stage_fakes import h100_provider
from tests.issue98_execution_fixture import load_issue98_execution_fixture
from tests.parabricks_operation_fakes import (
    approved_input,
    execute_and_read_path,
    input_storage_provider,
)
from tests.planning_assignment_fakes import execution_configuration
from tests.prebuilt_index_support import (
    GalaxyOpener,
    offline_galaxy_files,
)
from tests.reference_fakes import SOURCE_GZIP, source_acquisition
from tests.reference_stage_fakes import cpu_observation_inputs, seed_cpu
from tests.run_prerequisites import RunPrerequisites


class ReferenceStagesTests(RunPrerequisites, unittest.TestCase):
    """A Run's reference is the Execution Review's bound prebuilt index or one CPU
    preparation job (owner, 2026-10-29, #464). ``approve(indexed=True)`` binds the
    offline Galaxy index; ``self.served`` is what Galaxy serves when its files stream
    (none by default, so admission misses and the CPU preparation Review opens)."""

    def test_a_recorded_stage_policy_asks_the_failure_review(self):
        # A recorded Execution tuning is one failure entry per Execution stage,
        # without the retired word "terminal"; a semantic failure asks.
        a1 = {"retry_attempts": 1, "transient": "ask"}
        self.approve(
            indexed=True,
            failure={
                "scopes": {
                    "stage": {
                        name: {"operation": dict(a1)}
                        for name in (
                            "input_storage",
                            "admit_candidate",
                            "cpu_preparation",
                            "h100",
                        )
                    }
                }
            },
        )

        def factory(plan, journal):
            provider = self.factory(plan, journal)
            # The bound index's stream fails transiently: a stage failure the
            # recorded policy asks about, not a candidate miss.
            provider.scripts["admit_candidate", "inspect_candidate"] = deque(
                [ProviderError("transient", "gateway_timeout", {"operation": "get"})]
            )
            return provider

        state = execute_and_read_path(self.approved, infrastructure_factory=factory)
        policy = state.failures[-1].context["failure_policy"]
        self.assertEqual(
            policy["surface"], {"scope": "stage", "name": "admit_candidate"}
        )
        self.assertEqual(policy["surface_overrides"], a1)
        # The effective values the recorded whole-policy runtime
        # recorded.
        self.assertEqual(
            policy["effective"],
            {
                "retry_attempts": 1,
                "backoff_seconds": 1,
                "max_backoff_seconds": 30,
                "transient": "ask",
                "exhausted": "ask",
                "rejected": "ask",
                "worker_failed": "ask",
                "semantic": "ask",
                "invariant": "hold",
            },
        )
        self.assertEqual(state.failures[-1].context["route"], "ask")
        self.assertEqual(state.status, "execution_review_pending")
        self.assertEqual(state.disposition, "incomplete")

    def test_one_stage_entry_tunes_only_that_stage_end_to_end(self):
        # One value in the failure section changes its surface's behaviour and
        # recorded policy; every other surface keeps the defaults.
        from bio_neb.journey import Journey
        from bio_neb.recording import read_evidence
        from bio_neb.reference_discovery.reference_identity_tool import (
            ReferenceAdapterError,
        )
        from tests.reference_fakes import ReferenceAdapter

        defaults = {
            "retry_attempts": 3,
            "backoff_seconds": 1,
            "max_backoff_seconds": 30,
            "transient": "retry",
            "exhausted": "ask",
            "rejected": "ask",
            "worker_failed": "ask",
            "semantic": "ask",
            "invariant": "hold",
        }
        resolve_taxon = ReferenceAdapter.resolve_taxon
        unavailable = []

        def briefly_unavailable(adapter, query):
            if not unavailable:
                unavailable.append(query)
                raise ReferenceAdapterError(
                    "http_error", "Temporarily unavailable.", retryable=True
                )
            return resolve_taxon(adapter, query)

        with (
            patch.object(ReferenceAdapter, "resolve_taxon", briefly_unavailable),
            patch("time.sleep"),
        ):
            self.approve(
                indexed=True,
                failure={
                    "scopes": {
                        "stage": {
                            "admit_candidate": {"operation": {"retry_attempts": 2}}
                        }
                    }
                },
            )

        def factory(plan, journal):
            provider = self.factory(plan, journal)
            provider.scripts["admit_candidate", "inspect_candidate"] = deque(
                ProviderError("transient", "gateway_timeout", {"operation": "get"})
                for _ in range(3)
            )
            self.inspections = provider.scripts["admit_candidate", "inspect_candidate"]
            return provider

        with patch("time.sleep"):
            state = execute_and_read_path(self.approved, infrastructure_factory=factory)
        # The tuned stage stops after its two attempts, not the default three.
        self.assertEqual(len(self.inspections), 1)
        failure = state.failures[-1]
        self.assertEqual(failure.stage, "admit_candidate")
        self.assertEqual(failure.failure_class, "exhausted")
        self.assertEqual(failure.context["route"], "ask")
        self.assertEqual(state.status, "execution_review_pending")
        policy = failure.context["failure_policy"]
        self.assertEqual(
            policy["surface"], {"scope": "stage", "name": "admit_candidate"}
        )
        self.assertEqual(policy["surface_overrides"], {"retry_attempts": 2})
        self.assertEqual(policy["effective"], {**defaults, "retry_attempts": 2})

        journey = Journey.open(Path(self.approved.journey_file))
        # Another Execution stage records the defaults on its contact rows.
        steps = [
            row
            for journal in journey.run_directory.glob("contacts/*/journal.jsonl")
            for row in execution_contact_rows(journal)
        ]
        storage = [
            step["failure_policy"]
            for step in steps
            if step["stage"] == "input_storage" and step["failure_policy"]
        ]
        self.assertTrue(storage)
        for recorded in storage:
            self.assertEqual(
                recorded["surface"], {"scope": "stage", "name": "input_storage"}
            )
            self.assertEqual(recorded["effective"], defaults)
        self.assertEqual(len(unavailable), 1)
        events = read_evidence(journey)
        # A Tool surface's recorded retry keeps the defaults.
        (source_fault,) = [
            e["payload"]["policy"]
            for e in events
            if e["event_type"] == "SourceOperationFailureRecorded"
        ]
        self.assertEqual(
            source_fault["surface"], {"scope": "tool", "name": "resolve_taxon"}
        )
        self.assertEqual(source_fault["effective"], defaults)
        # A role's model contact keeps the defaults.
        started = [e for e in events if e["event_type"] == "AttemptStarted"]
        self.assertTrue(started)
        for event in started:
            policy = event["payload"]["effective_invocation_policy"]
            self.assertEqual(
                policy["model_contact"],
                # The per-call window.
                {
                    "contacts": 2,
                    "base_delay_seconds": 30,
                    "max_delay_seconds": 60,
                    "max_wait_seconds": 60,
                    "stall_seconds": 180,
                },
            )
            self.assertEqual(set(policy["model_contact_sources"].values()), {"default"})

    def test_rejected_placement_preserves_failure_and_does_not_offer_cpu(self):
        """A run problem placing the approved index's files goes to the stage
        failure Review, retry or abort, never the CPU Review (owner, 2026-10-29,
        #464)."""
        self.approve(indexed=True)
        self.serve_bound_index()

        def factory(plan, journal):
            provider = self.factory(plan, journal)
            provider.scripts["admit_candidate", "place_candidate"] = deque(
                [ProviderError("rejected", "destination_denied", {"operation": "put"})]
            )
            return provider

        state = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(state.status, "execution_review_pending")
        self.assertNotIn("CpuPreparationReviewRequested", state.latest_event_ids)
        self.assertNotIn("ReferenceResolved", state.latest_event_ids)
        self.assertFalse(self.providers[0].jobs_by_name)
        failure = state.failures[-1]
        self.assertEqual(failure.failure_class, "rejected")
        self.assertEqual(failure.detail, "destination_denied")
        self.assertEqual(
            (state.failure_review.legal_actions, state.failure_review.amendable),
            (("retry", "abort"), ()),
        )
        journey = JourneyPath.open(Path(self.approved.journey_file))
        self.assertEqual(
            json.loads(journey.read_artifact(failure.evidence[0]))["evidence"],
            {"operation": "put"},
        )

    def test_rejected_metadata_preserves_failure_and_does_not_offer_cpu(self):
        from bio_neb.application.verification_policy import ObjectVerificationOverrides

        self.approve(
            indexed=True,
            verification_overrides=ObjectVerificationOverrides(
                prepared_candidate="independent_full_read",
                reason="explicit drift audit",
            ),
        )
        self.serve_bound_index()

        def factory(plan, journal):
            provider = self.factory(plan, journal)
            provider.scripts["admit_candidate", "candidate_metadata"] = deque(
                [
                    ProviderError(
                        "rejected", "storage_access_refused", {"operation": "head"}
                    )
                ]
            )
            return provider

        state = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(state.status, "execution_review_pending")
        self.assertNotIn("CpuPreparationReviewRequested", state.latest_event_ids)
        self.assertNotIn("ReferenceResolved", state.latest_event_ids)
        self.assertFalse(self.providers[0].jobs_by_name)
        failure = state.failures[-1]
        self.assertEqual(failure.failure_class, "rejected")
        self.assertEqual(failure.detail, "storage_access_refused")
        journey = JourneyPath.open(Path(self.approved.journey_file))
        self.assertEqual(
            json.loads(journey.read_artifact(failure.evidence[0]))["evidence"],
            {"operation": "head"},
        )

    def _no_index_flow(self, *, failed_log=None):
        """Run with no supplied index; optionally fail the CPU job with ``failed_log``."""
        self.approve()
        captured = {}

        def factory(plan, journal):
            provider = seed_cpu(
                input_storage_provider(plan, journal), plan, self.approved
            )
            if failed_log is not None:
                provider.scripts["cpu_preparation", "await_cpu"] = deque(
                    [
                        Receipt(
                            values={"state": "FAILED", "log": failed_log(plan)},
                            provider_ids=("aijob-reference-exact",),
                        )
                    ]
                )
            original = provider.for_step

            def for_step(stage, step):
                ports = original(stage, step)
                jobs = ports.jobs

                class CapturedJobs:
                    def submit(inner, name, request):
                        captured.setdefault(stage, request)
                        return jobs.submit(name, request)

                    def observe(inner, job_id, *, window):
                        return jobs.observe(job_id, window=window)

                return Ports(ports.store, CapturedJobs(), ports.identity)

            provider.for_step = for_step
            captured["plan"] = plan
            return h100_provider(provider, plan, self.approved)

        from bio_neb.providers.ports import Ports

        state = execute_and_read_path(self.approved, infrastructure_factory=factory)
        return state, captured

    def test_without_a_supplied_index_the_cpu_job_fetches_the_approved_url(self):
        state, captured = self._no_index_flow()
        plan = captured["plan"]
        acquired = source_acquisition()
        # Input storage placed only the FASTQs; nothing streamed the reference.
        self.assertIsNone(plan.input_source)
        self.assertNotIn("reference_source", [p.role for p in state.inputs.pointers])
        self.assertIsNone(state.inputs.reference_acquisition)
        # The CPU request the job received names the approved URL and md5.
        source_path = captured["cpu_preparation"]["injection_provider_values"][0]
        request = json.loads(Path(source_path.split(":", 1)[0]).read_bytes())
        self.assertEqual(request["inputs"], [])
        self.assertEqual(
            request["download"],
            {
                "role": "reference_source",
                "url": plan.reference.source_url,
                "byte_count": plan.reference.expected_size_bytes,
                "md5": plan.reference.expected_md5,
                "object_key": plan.layout.fetched_source_object_key,
                "local_target": "/work/source/reference.fa.gz",
            },
        )
        self.assertNotIn("accepted_plain_fasta", request["reference"])
        # The job's acquisition is the prepared reference's receipt.
        self.assertIsInstance(state.reference, Prepared)
        self.assertEqual(state.reference.reference_acquisition, acquired)
        source = next(
            p for p in state.reference.pointers if p.role == "reference_source"
        )
        self.assertEqual(
            (source.location, source.bytes, source.content_id),
            (
                f"s3://{plan.storage.bucket}/{plan.layout.fetched_source_object_key}",
                acquired.source_byte_count,
                f"sha256:{acquired.source_sha256}",
            ),
        )
        self.assertEqual(state.h100.state, "verified", state.failures)

    def test_an_md5_mismatch_in_the_cpu_job_is_a_typed_failure_with_evidence(self):
        from tests.execution_observation_fakes import worker_marker

        def failed_log(plan):
            markers = (
                (
                    "BIO_NEB_PARABRICKS_WORKER_START",
                    {
                        "interpreter": "/opt/bio-neb/venv/bin/python",
                        "interpreter_resolved": "/usr/bin/python3.10",
                        "request_schema": "bio-neb.parabricks-worker-request/7",
                    },
                ),
                (
                    "BIO_NEB_WORKER_REQUEST_ACCEPTED",
                    {
                        "byte_count": 1,
                        "request_sha256": "1" * 64,
                        "worker_sha256": "2" * 64,
                    },
                ),
                (
                    "BIO_NEB_WORKER_TOOLS",
                    {"tools": {"bwa": {"path": "/usr/bin/bwa", "version": "0.7.17"}}},
                ),
                (
                    "BIO_NEB_WORKER_SOURCE_MISMATCH",
                    {
                        "role": "reference_source",
                        "url": plan.reference.source_url,
                        "expected_byte_count": plan.reference.expected_size_bytes,
                        "observed_byte_count": plan.reference.expected_size_bytes,
                        "expected_md5": plan.reference.expected_md5,
                        "observed_md5": "f" * 32,
                    },
                ),
                (
                    "BIO_NEB_PARABRICKS_WORKER_ERROR",
                    {"reason": "reference source identity mismatch"},
                ),
            )
            return "\n".join(
                worker_marker(name, payload, second)
                for second, (name, payload) in enumerate(markers, start=1)
            )

        state, captured = self._no_index_flow(failed_log=failed_log)
        plan = captured["plan"]
        failure = state.failures[-1]
        self.assertEqual(
            (failure.stage, failure.failure_class, failure.detail),
            ("cpu_preparation", "semantic", "source_identity_mismatch"),
        )
        evidence = json.loads(
            JourneyPath.open(Path(self.approved.journey_file)).read_artifact(
                failure.evidence[0]
            )
        )["evidence"]
        self.assertEqual(
            {
                key: evidence[key]
                for key in ("url", "expected_md5", "observed_md5", "provider_job_id")
            },
            {
                "url": plan.reference.source_url,
                "expected_md5": plan.reference.expected_md5,
                "observed_md5": "f" * 32,
                "provider_job_id": "aijob-reference-exact",
            },
        )
        self.assertNotIn("ReferencePrepared", state.latest_event_ids)
        self.assertNotIn("h100", captured)

    def test_a_failed_cpu_sample_must_carry_its_log_and_job_identity(self):
        from bio_neb.failure import Invariant
        from bio_neb.parabricks.reference_stages import cpu_terminal

        for missing in ("log", "provider_job_id"):
            values = {"state": "FAILED", "log": None, "provider_job_id": "job"}
            del values[missing]
            with (
                self.subTest(missing=missing),
                self.assertRaises(Invariant) as refused,
            ):
                cpu_terminal([Receipt(values=values)])
            self.assertEqual(
                (refused.exception.invariant_id, str(refused.exception)),
                ("worker_observation_invalid", "terminal sample lacks log or job id"),
            )
        # An unavailable log is an explicit null: an ordinary job failure.
        with self.assertRaisesRegex(ProviderError, "cpu_job_failed"):
            cpu_terminal(
                [
                    Receipt(
                        values={"state": "FAILED", "log": None, "provider_job_id": "j"}
                    )
                ]
            )

    def test_a_replacement_fetch_job_writes_its_source_under_a_fresh_key(self):
        self.approve()
        plans, provider, fail_worker = [], None, True

        def factory(plan, journal):
            nonlocal provider
            plans.append(plan)
            scripted = seed_cpu(
                input_storage_provider(plan, journal),
                plan,
                self.approved,
                job_id="job:" + plan.layout.cpu_job_name,
            )
            if provider is None:
                provider = scripted
            else:
                provider.scripts.update(scripted.scripts)
                for key, content in scripted.objects.items():
                    provider.objects.setdefault(key, content)
            if fail_worker:
                provider.scripts["cpu_preparation", "await_cpu"] = deque(
                    [
                        Receipt(
                            values={"state": "FAILED", "log": None},
                            provider_ids=("job:" + plan.layout.cpu_job_name,),
                        )
                    ]
                )
            return h100_provider(provider, plan, self.approved)

        first = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(first.failures[-1].detail, "cpu_job_failed")
        answer_review(
            self.approved.journey_file,
            first.latest_event_ids["FailureReviewRequested"],
            {"action": "retry"},
        )
        fail_worker = False
        second = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(second.h100.state, "verified", second.failures)
        earlier, replacement = plans[0].layout, plans[-1].layout
        self.assertNotEqual(
            earlier.fetched_source_object_key, replacement.fetched_source_object_key
        )
        self.assertNotEqual(
            earlier.source_manifest_object_key, replacement.source_manifest_object_key
        )
        source = next(
            p for p in second.reference.pointers if p.role == "reference_source"
        )
        self.assertTrue(
            source.location.endswith("/" + replacement.fetched_source_object_key)
        )

    def test_terminal_cpu_replacement_retains_permission_and_distinct_intents(self):
        # The bound index's files are not served: it misses, and the CPU
        # preparation Review grants the permission the replacements keep.
        self.approve(indexed=True)
        provider = None
        plans = []
        fail_worker = True

        def factory(plan, journal):
            nonlocal provider
            plans.append(plan)
            scripted = seed_cpu(
                input_storage_provider(plan, journal),
                plan,
                self.approved,
                job_id="job:" + plan.layout.cpu_job_name,
            )
            if provider is None:
                provider = scripted
            else:
                provider.scripts.update(scripted.scripts)
                for key, content in scripted.objects.items():
                    provider.objects.setdefault(key, content)
            if fail_worker:
                provider.scripts["cpu_preparation", "await_cpu"] = deque(
                    [
                        Receipt(
                            values={"state": "FAILED", "log": None},
                            provider_ids=("job:" + plan.layout.cpu_job_name,),
                        )
                    ]
                )
            return h100_provider(provider, plan, self.approved)

        pending = execute_and_read_path(self.approved, infrastructure_factory=factory)
        approved = answer_review(
            self.approved.journey_file,
            pending.latest_event_ids["CpuPreparationReviewRequested"],
            {"action": "approve"},
        )
        first = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(first.failures[-1].detail, "cpu_job_failed")
        self.assertEqual(first.failure_review.retry_consequence, "replace_job")
        original_objects = dict(provider.objects)
        original_name = plans[-1].layout.cpu_job_name
        request_id = first.latest_event_ids["FailureReviewRequested"]
        replacement = answer_review(
            self.approved.journey_file, request_id, {"action": "retry"}
        )
        self.assertEqual(replacement.review, approved.review)
        second = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(second.failures[-1].detail, "cpu_job_failed")
        self.assertEqual(len(provider.jobs_by_name), 2)
        self.assertNotEqual(plans[-1].layout.cpu_job_name, original_name)
        self.assertNotEqual(
            plans[-1].layout.reference_receipt_object_key,
            plans[-2].layout.reference_receipt_object_key,
        )
        self.assertEqual(
            answer_review(self.approved.journey_file, request_id, {"action": "retry"}),
            second,
        )
        answer_review(
            self.approved.journey_file,
            second.latest_event_ids["FailureReviewRequested"],
            {"action": "retry"},
        )
        third = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(third.failures[-1].detail, "cpu_job_failed")
        self.assertEqual(len(provider.jobs_by_name), 3)
        for key, content in original_objects.items():
            self.assertEqual(provider.objects[key], content)
        accepted = answer_review(
            self.approved.journey_file,
            third.latest_event_ids["FailureReviewRequested"],
            {"action": "retry"},
        )
        fail_worker = False
        done = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(done.disposition, "succeeded")
        dependencies = done.rests_on[done.latest_event_ids["ReferencePrepared"]]
        self.assertIn(
            approved.latest_event_ids["CpuPreparationReviewAnswered"], dependencies
        )
        self.assertIn(
            replacement.latest_event_ids["FailureReviewAnswered"], dependencies
        )
        self.assertIn(accepted.latest_event_ids["FailureReviewAnswered"], dependencies)
        self.assertEqual(len(provider.jobs_by_name), 5)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.providers = []
        self.served = {}

    def approve(
        self, *, configuration=None, failure=None, indexed=False, **launch_overrides
    ):
        """Approve this case's Run with its final choices.

        The default choices, binding no prebuilt index, come from the class's
        approved Run restored in place for each call; other choices (the bound
        Galaxy index, configuration, verification or failure policy) build their
        own Run once.
        """
        if (
            configuration is None
            and failure is None
            and not indexed
            and not launch_overrides
        ):
            self.approved = self.prerequisite("approved").value
            return self.approved
        galaxy = None
        if indexed:
            self.galaxy_files = offline_galaxy_files(
                Path(self.enterContext(tempfile.TemporaryDirectory()))
            )
            galaxy = GalaxyOpener(files=self.galaxy_files)
        self.approved = approved_input(
            Path(self.temp.name),
            configuration=configuration,
            launch_overrides=launch_overrides or None,
            failure=failure,
            galaxy=galaxy,
        )
        if indexed:
            self.assertIsNotNone(
                self.approved.proposal.execution_package.reference.prebuilt_index
            )
        return self.approved

    def serve_bound_index(self):
        """Serve the bound index's own files, so they stream and are admitted."""
        self.served = dict(self.galaxy_files)

    def factory(self, plan, journal):
        provider = seed_cpu(input_storage_provider(plan, journal), plan, self.approved)
        provider.sources.update(self.served)
        provider.sources[plan.reference.source_url] = SOURCE_GZIP
        self.providers.append(provider)
        return h100_provider(provider, plan, self.approved)

    def test_explicit_no_candidate_prepares_then_runs_h100_without_dev_milestone_stop(
        self,
    ):
        self.approve()
        from bio_neb.parabricks.operation import execute_parabricks

        with patch(
            "bio_neb.journey_path.project_execution", wraps=project_execution
        ) as fold:
            result = execute_parabricks(
                self.approved, infrastructure_factory=self.factory
            )
        self.assertEqual(fold.call_count, 1)
        self.assertEqual(result.disposition, "succeeded")
        state = JourneyPath.open(Path(self.approved.journey_file)).state
        self.assertIsInstance(state.reference, Prepared)
        self.assertEqual(state.disposition, "succeeded")
        events = JourneyPath.open(Path(self.approved.journey_file)).read()
        self.assertEqual(
            [e.event_type for e in events],
            [
                "ExecutionBound",
                "InputsVerified",
                "ReferencePrepared",
                "H100ArtifactsVerified",
                "ExecutionOutcome",
            ],
        )
        self.assertEqual(state, project_execution(events))
        self.assertEqual(len(self.providers[0].jobs_by_name), 2)
        with patch(
            "bio_neb.parabricks.operation._make_provider",
            side_effect=AssertionError("contact"),
        ):
            self.assertEqual(execute_and_read_path(self.approved), state)

    def test_approved_nondefault_disks_reach_submission_and_worker_closure(self):
        self.check_approved_runtime_flow()

    def test_cpu_disk_drift_stops_before_h100_or_prepared_publication(self):
        self.check_approved_runtime_flow(observed_cpu_bytes=107374182400)

    def check_approved_runtime_flow(self, *, observed_cpu_bytes=None):
        # Current preparation/review supplies authority; the live export is only
        # an independent positive artifact/flow baseline for synthetic effects.
        from bio_neb.content import canonical_json_digest
        from bio_neb.parabricks.capability_policy import (
            CapabilitySnapshot,
            ExecutionRuntimeSelection,
        )
        from bio_neb.parabricks.configuration import ParabricksConfiguration
        from bio_neb.providers.ports import Ports

        fixture = load_issue98_execution_fixture()
        self.assertEqual(len(fixture.scenarios), 1)
        baseline = fixture.scenarios[0]
        milestones = (
            "ExecutionBound",
            "InputsVerified",
            "ReferencePrepared",
            "H100ArtifactsVerified",
            "ExecutionOutcome",
        )
        observed_milestones = tuple(
            name for name in baseline.path_events if name in milestones
        )
        self.assertEqual(observed_milestones, milestones)
        configuration = execution_configuration(Path(self.temp.name))
        selected = configuration.runtime_selection
        profile = selected.capability.runtime_profiles[0]
        alternate = profile.model_copy(
            update={
                "profile_id": "alternate",
                "cpu": profile.cpu.model_copy(update={"disk_size": "200Gi"}),
                "h100": profile.h100.model_copy(
                    update={
                        "platform": "gpu-alternative",
                        "preset": "alternative-preset",
                        "disk_size": "500Gi",
                    }
                ),
            }
        )
        fallback = alternate.model_copy(
            update={
                "profile_id": "fallback",
                "h100": alternate.h100.model_copy(update={"platform": "gpu-fallback"}),
            }
        )
        claims = selected.capability.model_dump(mode="json", exclude={"snapshot_id"})
        claims["runtime_profiles"] = [
            item.model_dump(mode="json") for item in (alternate, fallback)
        ]
        snapshot = CapabilitySnapshot(
            snapshot_id=canonical_json_digest(claims), **claims
        )
        material = configuration.model_dump(mode="json")
        material["runtime_selection"] = ExecutionRuntimeSelection(
            capability=snapshot, profile_ids=("alternate", "fallback")
        ).model_dump(mode="json")
        approved = self.approve(
            configuration=ParabricksConfiguration.model_validate(material),
        )
        captured = {}
        submissions = {}

        def factory(plan, journal):
            provider = h100_provider(
                seed_cpu(input_storage_provider(plan, journal), plan, approved),
                plan,
                approved,
            )
            captured.update(plan=plan, provider=provider)
            if observed_cpu_bytes is not None:
                receipt = provider.scripts["cpu_preparation", "submit_cpu"][0]
                envelope = {
                    **receipt.values["envelope"],
                    "disk_size_bytes": observed_cpu_bytes,
                }
                provider.scripts["cpu_preparation", "submit_cpu"][0] = (
                    receipt.model_copy(
                        update={"values": {**receipt.values, "envelope": envelope}}
                    )
                )
            original = provider.for_step

            def for_step(stage, step):
                ports = original(stage, step)
                jobs = ports.jobs

                class CapturedJobs:
                    def submit(inner, name, request):
                        submissions[stage] = (name, request)
                        return jobs.submit(name, request)

                    def observe(inner, job_id, *, window):
                        return jobs.observe(job_id, window=window)

                return Ports(ports.store, CapturedJobs(), ports.identity)

            provider.for_step = for_step
            return provider

        state = execute_and_read_path(approved, infrastructure_factory=factory)
        journey = JourneyPath.open(Path(approved.journey_file))
        plan, provider = captured["plan"], captured["provider"]
        self.assertEqual(state, project_execution(journey.read()))
        # The retained Run streamed the source through input storage; with no
        # supplied index the CPU job now fetches it, so it leaves the inputs.
        self.assertEqual(
            tuple(p.role for p in state.inputs.pointers),
            tuple(p.role for p in baseline.inputs if p.role != "reference_source"),
        )
        self.assertIsNone(state.inputs.reference_acquisition)
        self.assertEqual(submissions["cpu_preparation"][1]["disk_size"], "200Gi")
        self.assertEqual(
            submissions["cpu_preparation"][1]["disk_size_bytes"], 214748364800
        )
        if observed_cpu_bytes is not None:
            self.assertEqual(state.failures[-1].detail, "cpu_receipt_invalid")
            self.assertNotIn("h100", submissions)
            self.assertNotIn("ReferencePrepared", state.latest_event_ids)
            for event_type in observed_milestones[2:-1]:
                self.assertNotIn(event_type, state.latest_event_ids)
            self.assertNotEqual(state.disposition, "succeeded")
            self.assertNotIn(
                (plan.storage.bucket, plan.layout.reference_manifest_object_key),
                provider.objects,
            )
            return
        self.assertEqual(submissions["h100"][1]["disk_size"], "500Gi")
        self.assertEqual(submissions["h100"][1]["disk_size_bytes"], 536870912000)
        self.assertEqual(submissions["h100"][1]["platform"], "gpu-alternative")
        self.assertEqual(submissions["h100"][1]["preset"], "alternative-preset")
        self.assertEqual(state.reference.state, "prepared")
        self.assertEqual(state.h100.state, "verified", state.failures)
        # The job-fetched source and its Run-bucket manifest join the prepared
        # reference's proof; the manifest is no longer a pre-job Journey document.
        prepared_roles = tuple(p.role for p in baseline.reference)
        self.assertEqual(
            tuple(p.role for p in state.reference.pointers),
            (
                *prepared_roles[:8],
                "reference_source",
                "source_manifest",
                *(role for role in prepared_roles[8:] if role != "source_manifest"),
            ),
        )
        self.assertEqual(state.reference.reference_acquisition, source_acquisition())
        output_roles = tuple(item.role for item in baseline.verified_objects)
        self.assertEqual(
            output_roles, ("aligned_bam", "VCF", "worker_receipt", "execution_manifest")
        )
        self.assertEqual(
            tuple(p.role for p in state.h100.pointers if p.role in output_roles),
            output_roles,
        )
        self.assertEqual(
            tuple(
                event.event_type
                for event in journey.read()
                if event.event_type in milestones
            ),
            observed_milestones,
        )

        from bio_neb.parabricks.h100_recipe import H100ExecutionRequest
        from bio_neb.parabricks.h100_stage import compile_request

        request = compile_request(plan, state)
        self.assertEqual(request.deterministic_job_name, plan.layout.h100_job_name)
        self.assertEqual(submissions["h100"][0], request.deterministic_job_name)
        H100ExecutionRequest.model_validate_json(request.model_dump_json())
        values = submissions["h100"][1]
        self.assertEqual(values["normalized_arguments"], ["/opt/bio-neb/request.json"])
        self.assertEqual(values["injected_paths"], ["/opt/bio-neb/request.json"])
        source, target = values["injection_provider_values"][0].split(":", 1)
        self.assertEqual(target, "/opt/bio-neb/request.json")
        document = json.loads(Path(source).read_bytes())
        self.assertEqual(
            [item["role"] for item in document["inputs"]],
            [
                "read_1",
                "read_2",
                "genomic_fasta",
                "fasta_fai",
                "reference_dict",
                "bwa_amb",
                "bwa_ann",
                "bwa_bwt",
                "bwa_pac",
                "bwa_sa",
            ],
        )
        self.assertEqual(
            document["parabricks_argv"],
            [
                "pbrun",
                "germline",
                "--ref",
                "/work/reference/reference.fa",
                "--in-fq",
                "/work/inputs/read-1.fastq.gz",
                "/work/inputs/read-2.fastq.gz",
                "--out-bam",
                "/work/outputs/aligned.bam",
                "--out-variants",
                "/work/outputs/variants.vcf",
                "--read-group-sm",
                "sample-001",
                "--num-gpus",
                "1",
            ],
        )

    def test_binding_sync_failure_prevents_provider_and_dependent_facts(self):
        import os

        self.approve()
        journey_file = Path(self.approved.journey_file).resolve()
        real_fsync = os.fsync
        refused = []

        def refuse_binding_sync(fd):
            # Only the Journey sync that would make the ExecutionBound binding
            # durable fails; every other write, including the Pass's, syncs.
            target = Path(os.readlink(f"/proc/self/fd/{fd}"))
            if (
                target == journey_file
                and not refused
                and b'"event_type":"ExecutionBound"' in journey_file.read_bytes()
            ):
                refused.append(target)
                raise OSError("sync refused")
            return real_fsync(fd)

        with patch("os.fsync", side_effect=refuse_binding_sync):
            with self.assertRaisesRegex(OSError, "sync refused"):
                execute_and_read_path(
                    self.approved, infrastructure_factory=self.factory
                )
        self.assertEqual(refused, [journey_file])
        self.assertEqual(self.providers, [])
        events = JourneyPath.open(Path(self.approved.journey_file)).read()
        self.assertEqual([event.event_type for event in events], ["ExecutionBound"])
        # Written bytes may survive failed sync; no accepted return or continuation.

    def test_candidate_miss_waits_then_exact_approval_prepares_in_second_pass(self):
        """A bound prebuilt index that misses opens the CPU preparation Review,
        abort or one CPU job only (owner, 2026-10-29, #464)."""
        self.approve(indexed=True)
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        self.assertEqual(state.status, "cpu_preparation_review_pending")
        self.assertEqual(state.reference.reason, "candidate_absent")
        self.assertEqual(
            state.review.legal_actions, ("abort", "approve_cpu_preparation")
        )
        self.assertFalse(self.providers[0].jobs_by_name)
        answer_review(
            self.approved.journey_file,
            state.latest_event_ids["CpuPreparationReviewRequested"],
            {"action": "approve"},
        )
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        self.assertIsInstance(state.reference, Prepared)
        self.assertEqual(state.pass_, 2)
        self.assertEqual(len(self.providers[1].jobs_by_name), 2)
        events = JourneyPath.open(Path(self.approved.journey_file)).read()
        prepared = next(e for e in events if e.event_type == "ReferencePrepared")
        self.assertIn(
            state.latest_event_ids["CpuPreparationReviewAnswered"], prepared.rests_on
        )

    def test_decline_never_submits(self):
        self.approve(indexed=True)
        state = execute_and_read_path(
            self.approved, infrastructure_factory=self.factory
        )
        declined = answer_review(
            self.approved.journey_file,
            state.latest_event_ids["CpuPreparationReviewRequested"],
            {"action": "decline"},
        )
        with patch(
            "bio_neb.parabricks.operation._make_provider",
            side_effect=AssertionError("contact"),
        ):
            self.assertEqual(execute_and_read_path(self.approved), declined)
        self.assertEqual(declined.disposition, "failed")

    def test_exhausted_submission_requests_generic_review_without_reference_fact(self):
        self.approve()

        def factory(plan, journal):
            provider = self.factory(plan, journal)
            provider.scripts["cpu_preparation", "submit_cpu"] = deque(
                [ProviderError("transient", "busy")] * 3
            )
            return h100_provider(provider, plan, self.approved)

        with patch("bio_neb.parabricks.runner.time.sleep"):
            state = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(state.status, "execution_review_pending")
        self.assertNotIn("ReferencePrepared", state.latest_event_ids)
        self.assertIsNone(state.review)
        self.assertEqual(state.failures[-1].step, "submit_cpu")

    def test_cpu_metadata_drift_does_not_publish_manifest(self):
        from bio_neb.application.verification_policy import ObjectVerificationOverrides

        self.approve(
            verification_overrides=ObjectVerificationOverrides(
                reference_preparation="independent_full_read",
                reason="explicit drift audit",
            ),
        )

        def factory(plan, journal):
            provider = self.factory(plan, journal)
            provider.scripts["cpu_preparation", "verify_bundle"][1] = Receipt(
                values={"bytes": 1, "sha256": "0" * 64}
            )
            self.manifest_key = (
                plan.storage.bucket,
                plan.layout.reference_manifest_object_key,
            )
            return h100_provider(provider, plan, self.approved)

        state = execute_and_read_path(self.approved, infrastructure_factory=factory)
        # The semantic drift asks its Review on default policy.
        self.assertEqual(state.status, "execution_review_pending")
        self.assertEqual(state.failure_review.stage, "cpu_preparation")
        self.assertNotIn("ReferencePrepared", state.latest_event_ids)
        self.assertNotIn(self.manifest_key, self.providers[0].objects)

    def test_harness_success_is_not_failed_by_resolved_historical_failure(self):
        import io
        from contextlib import redirect_stdout

        from bio_neb.devtools.stage_harness import main
        from bio_neb.parabricks.plan import RetainedExecution, plan_execution

        self.approve()

        def rejected(plan, journal):
            provider = seed_cpu(
                input_storage_provider(plan, journal), plan, self.approved
            )
            provider.scripts["cpu_preparation", "submit_cpu"] = deque(
                [ProviderError("rejected", "submission_refused")]
            )
            return provider

        failed = execute_and_read_path(self.approved, infrastructure_factory=rejected)
        answer_review(
            self.approved.journey_file,
            failed.latest_event_ids["FailureReviewRequested"],
            {"action": "retry"},
        )
        journey = JourneyPath.open(Path(self.approved.journey_file))
        state = journey.state
        plan = plan_execution(
            self.approved,
            RetainedExecution(state, self.approved.authority, self.approved.proposal),
        )
        provider = seed_cpu(input_storage_provider(plan, None), plan, self.approved)
        document = {
            "approved": self.approved.model_dump(mode="json"),
            "path_events": [
                event.model_dump(mode="json", by_alias=True) for event in journey.read()
            ],
            "artifacts": [
                {
                    "pointer": pointer.model_dump(mode="json"),
                    "hex": journey.read_artifact(pointer).hex(),
                }
                for pointer in (
                    *state.inputs.pointers,
                    *(p for failure in state.failures for p in failure.evidence),
                )
                if pointer.location.startswith("artifacts/")
            ],
            "run_root": str(Path(self.temp.name) / "recovered-harness"),
            "fixture_receipts": {
                step: [r.model_dump(mode="json") for r in receipts]
                for (stage, step), receipts in provider.scripts.items()
                if stage == "cpu_preparation"
            },
            "objects": [
                {"bucket": bucket, "key": key, "hex": content.hex()}
                for (bucket, key), content in provider.objects.items()
            ],
        }
        input_file = Path(self.temp.name) / "recovered-probe.json"
        input_file.write_text(json.dumps(document))
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(
                [
                    "--stage",
                    "cpu-preparation",
                    "--mode",
                    "offline",
                    "--input",
                    str(input_file),
                ]
            )
        result = json.loads(output.getvalue())
        self.assertEqual(result["reference"]["state"], "prepared")
        self.assertEqual(len(result["failures"]), 1)
        self.assertEqual(result["resolved_failure_ids"], result["failure_event_ids"])
        self.assertEqual(code, 0)

    def test_offline_harness_crosses_both_reference_stage_seams(self):
        import io
        import json
        from contextlib import redirect_stdout

        from bio_neb.devtools.stage_harness import main
        from bio_neb.parabricks.plan import RetainedExecution, plan_execution

        self.approve()
        execute_and_read_path(self.approved, infrastructure_factory=self.factory)
        prefix = JourneyPath.open(Path(self.approved.journey_file)).read()[:2]
        state = project_execution(prefix)
        plan = plan_execution(
            self.approved,
            RetainedExecution(state, self.approved.authority, self.approved.proposal),
        )
        for name, expected in (
            ("admit-candidate", "miss"),
            ("cpu-preparation", "prepared"),
        ):
            with self.subTest(name=name):
                provider = seed_cpu(
                    input_storage_provider(plan, None), plan, self.approved
                )
                doc = {
                    "approved": self.approved.model_dump(mode="json"),
                    "path_events": [
                        row.model_dump(mode="json", by_alias=True) for row in prefix
                    ],
                    "artifacts": [
                        {
                            "pointer": pointer.model_dump(mode="json"),
                            "hex": JourneyPath.open(Path(self.approved.journey_file))
                            .read_artifact(pointer)
                            .hex(),
                        }
                        for pointer in state.inputs.pointers
                        if pointer.location.startswith("artifacts/")
                    ],
                    "run_root": str(Path(self.temp.name) / name),
                    "fixture_receipts": {
                        step: [r.model_dump(mode="json") for r in receipts]
                        for (stage, step), receipts in provider.scripts.items()
                        if stage == name.replace("-", "_")
                    },
                    "objects": [
                        {"bucket": bucket, "key": key, "hex": content.hex()}
                        for (bucket, key), content in provider.objects.items()
                    ],
                }
                path = Path(self.temp.name) / (name + ".json")
                path.write_text(json.dumps(doc))
                output = io.StringIO()
                with (
                    redirect_stdout(output),
                    patch(
                        "bio_neb.providers.nebius.NebiusProvider",
                        side_effect=AssertionError("live contact"),
                    ),
                ):
                    code = main(
                        ["--stage", name, "--mode", "offline", "--input", str(path)]
                    )
                self.assertEqual(code, 0, output.getvalue())
                self.assertEqual(
                    json.loads(output.getvalue())["reference"]["state"], expected
                )

    def test_evicted_worker_windows_preserve_durable_validation(self):
        self.approve()

        def factory(plan, journal):
            provider = self.factory(plan, journal)
            terminal = provider.scripts["cpu_preparation", "await_cpu"][0]
            lines = terminal.values["log"].splitlines()
            provider.scripts["cpu_preparation", "await_cpu"] = deque(
                [
                    terminal.model_copy(
                        update={
                            "values": {
                                **terminal.values,
                                "state": "RUNNING",
                                "log": "\n".join(lines[:8]),
                            }
                        }
                    ),
                    terminal.model_copy(
                        update={
                            "values": {
                                **terminal.values,
                                "state": "RUNNING",
                                "log": "ordinary short window",
                            }
                        }
                    ),
                    terminal.model_copy(
                        update={
                            "values": {**terminal.values, "log": "\n".join(lines[7:])}
                        }
                    ),
                ]
            )
            return h100_provider(provider, plan, self.approved)

        with patch("bio_neb.observation.time.sleep"):
            state = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(state.reference.state, "prepared")

    def test_absent_durable_validation_never_publishes_reference_success(self):
        self.approve()

        def factory(plan, journal):
            import hashlib

            from bio_neb.content import canonical_json

            provider = self.factory(plan, journal)
            receipt = provider.scripts["cpu_preparation", "verify_bundle"][0]
            worker = json.loads(receipt.content)
            worker.pop("validation")
            body = canonical_json(worker).encode()
            provider.scripts["cpu_preparation", "verify_bundle"][0] = Receipt(
                content=body,
                values={"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()},
            )
            return h100_provider(provider, plan, self.approved)

        state = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(state.status, "execution_review_pending")
        self.assertEqual(state.failure_review.stage, "cpu_preparation")
        self.assertEqual(state.failures[-1].failure_class, "semantic")
        self.assertEqual(state.failures[-1].detail, "cpu_receipt_invalid")
        self.assertNotIn("ReferencePrepared", state.latest_event_ids)
        self.assertEqual(state.disposition, "incomplete")

    def test_completed_job_reuse_reaches_cpu_closure_with_the_real_job_runner(self):
        from types import SimpleNamespace

        from bio_neb.parabricks import reference_recipe as contracts
        from bio_neb.providers.nebius import NebiusProvider
        from bio_neb.providers.ports import Ports
        from tests.reference_stage_fakes import CpuCli, credentials

        self.approve()

        def factory(plan, journal):
            fake = self.factory(plan, journal)
            config = cpu_observation_inputs(plan)
            request = contracts.compile_cpu_request(
                plan, source_acquisition()
            ).model_copy(update={"deterministic_job_name": plan.layout.cpu_job_name})
            output = fake.scripts["cpu_preparation", "await_cpu"][0].values["log"]
            cli = CpuCli(config, request, log_outputs=(output,))
            cli.states = iter(("COMPLETED",))
            self.reused_cli = cli

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

            class LostSubmissionResponse:
                def submit(inner, name, request):
                    provider().jobs.submit(name, request)
                    cli.initial_state = "COMPLETED"
                    inner.jobs = provider().jobs
                    return inner.jobs.submit(name, request)

                def observe(inner, job_id, *, window):
                    return inner.jobs.observe(job_id, window=window)

            jobs = LostSubmissionResponse()
            return SimpleNamespace(
                for_step=lambda stage, step: (
                    fake.for_step(stage, step)
                    if stage == "h100"
                    else Ports(
                        fake.for_step(stage, step).store,
                        jobs,
                        fake.for_step(stage, step).identity,
                    )
                )
            )

        state = execute_and_read_path(self.approved, infrastructure_factory=factory)
        self.assertEqual(state.reference.state, "prepared")
        self.assertEqual(
            sum(call[:3] == ("ai", "job", "create") for call in self.reused_cli.calls),
            1,
        )

    def test_submission_failure_retains_the_name_before_any_provider_result(self):
        import json

        self.approve()

        def factory(plan, journal):
            provider = self.factory(plan, journal)
            self.job_name = plan.layout.cpu_job_name
            provider.scripts["cpu_preparation", "submit_cpu"] = deque(
                [ProviderError("rejected", "lookup_failed_after_create")]
            )
            return h100_provider(provider, plan, self.approved)

        state = execute_and_read_path(self.approved, infrastructure_factory=factory)
        intent = next(
            row
            for row in execution_contact_rows(state.outcome.contact_journal)
            if row["step"] == "submit_cpu" and row["phase"] == "invoked"
        )
        body = json.loads(Path(intent["receipt"]["ref"]).read_text())
        self.assertEqual(body["name"], self.job_name)
        self.assertEqual(body["phase"], "requested")
