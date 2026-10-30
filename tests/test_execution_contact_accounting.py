from __future__ import annotations

import asyncio
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bio_neb.cli import main
from bio_neb.failure import ModelContactPolicy
from bio_neb.journey import Journey, canonical_json
from bio_neb.parabricks.execution_contact_accounting import (
    COMPILER_VERSION,
    ExecutionContactJournal,
    ExecutionContactJournalBinding,
    ExecutionContactJournalRow,
    compile_execution_contact_evidence,
)
from bio_neb.providers.nebius_common import (
    AwsCredentialMaterial,
    NebiusCliError,
    NebiusCliInstallIdentity,
    NebiusCliRunner,
)
from bio_neb.providers.nebius_object_storage import (
    NebiusObjectStorageMechanics,
)
from bio_neb.recording import processing_pass, read_evidence, read_passes
from tests.run_prerequisites import RunPrerequisites
from tests.source_limits_support import declaring

_EXECUTION_ID = "parabricks-execution:" + "a" * 64
_JOURNEY_BINDING = "journey:" + "b" * 64


def _journey_events(
    approved_journey_file: Path,
    binding: ExecutionContactJournalBinding | None = None,
) -> tuple[dict[str, object], ...]:
    # Compiler-only synthetic execution path around a real prepared planning prelude.
    # This is not operation, provider, or scientific execution proof.
    planning = read_evidence(Journey.open(approved_journey_file))
    events = (
        *planning,
        {
            "event_type": "ExecutionBound",
            "event_id": "bound",
            "kind": "authority",
            "payload": {},
        },
    )
    if binding is None:
        return events
    return (
        *events,
        {
            "event_type": "ExecutionOutcome",
            "event_id": "outcome",
            "pass": 1,
            "payload": {
                "contact_journal": binding.locator,
                "journal_sha256": binding.sha256,
                "journal_bytes": binding.byte_count,
            },
        },
    )


def _planning_model_contacts(journey_file) -> int:
    """The physical model contacts a baseline's real planning prelude retained.

    Baselines plan through the scripted HTTP model endpoint, so each request is
    one recorded physical contact the ledger counts with the Run's others.
    """
    contacts = sum(
        event["event_type"] == "ModelPhysicalContactStarted"
        for event in read_evidence(Journey.open(Path(journey_file)))
    )
    if contacts == 0:
        raise AssertionError("the baseline planned without a model contact")
    return contacts


class _ExplodingObserver:
    def start(self, **_: object) -> str:
        raise OSError("diagnostic sink unavailable")

    def finish(self, unit_id: str | None, **_: object) -> None:
        raise OSError(f"diagnostic sink unavailable: {unit_id}")


class _CollectingObserver:
    def __init__(self) -> None:
        self.started: list[dict[str, object]] = []
        self.finished: list[tuple[str | None, dict[str, object]]] = []

    def start(self, **values: object) -> str:
        unit_id = f"unit-{len(self.started) + 1}"
        self.started.append(dict(values))
        return unit_id

    def finish(self, unit_id: str | None, **values: object) -> None:
        self.finished.append((unit_id, dict(values)))


class ExecutionContactAccountingTests(RunPrerequisites, unittest.TestCase):
    def test_invalid_invocation_refs_are_inconsistent_in_closure_and_review(self):
        from bio_neb.parabricks.execution_contact_accounting import (
            read_complete_execution_contacts,
        )

        for ref in ("cj:2:1", "broken", "cj:1:99"):
            with self.subTest(ref=ref), tempfile.TemporaryDirectory() as directory:
                journal = ExecutionContactJournal(
                    Path(directory) / "journal.jsonl", _EXECUTION_ID
                )
                identity = dict(
                    stage="cpu",
                    step="launch",
                    logical_operation="jobs.submit",
                    resource_kind="cpu_job",
                    resource_intent="intent",
                )
                journal.step(phase="invoked", **identity)
                journal.step(
                    phase="verified",
                    invocation_ref=ref,
                    provider_ids=("job-known",),
                    ceiling_count=1,
                    **identity,
                )
                binding = journal.finalize()
                self.assertFalse(
                    json.loads(journal.path.read_bytes().splitlines()[-1])["complete"]
                )
                with self.assertRaisesRegex(ValueError, "incomplete_or_inconsistent"):
                    read_complete_execution_contacts(
                        journal.path.read_bytes(), _EXECUTION_ID
                    )
                review = compile_execution_contact_evidence(
                    journal_bytes=journal.path.read_bytes(),
                    binding=binding,
                    expected_execution_id=_EXECUTION_ID,
                    journey_events=[
                        {
                            "event_type": "ExecutionBound",
                            "event_id": "bound",
                            "payload": {},
                            "kind": "authority",
                        }
                    ],
                    journey_binding="synthetic-path",
                )
                self.assertIn("logical_result_without_invocation", review.discrepancies)
                self.assertEqual(
                    review.ledger.cpu_preparation_create.observed_unit_ids,
                    ("job-known",),
                )

    def test_duplicate_results_preserve_both_ids_and_failure_causes(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = ExecutionContactJournal(
                Path(directory) / "journal.jsonl", _EXECUTION_ID
            )
            identity = dict(
                stage="cpu",
                step="launch",
                logical_operation="jobs.submit",
                resource_kind="cpu_job",
                resource_intent="job-intent",
            )
            invocation = journal.step(phase="invoked", **identity)
            journal.step(
                phase="failed",
                invocation_ref=invocation,
                provider_ids=("job-old",),
                failure_class="rejected",
                failure_detail="response_lost",
                **identity,
            )
            journal.step(
                phase="verified",
                invocation_ref=invocation,
                provider_ids=("job-new",),
                ceiling_count=1,
                **identity,
            )
            binding = journal.finalize()
            review = compile_execution_contact_evidence(
                journal_bytes=journal.path.read_bytes(),
                binding=binding,
                expected_execution_id=_EXECUTION_ID,
                journey_events=[
                    {
                        "event_type": "ExecutionBound",
                        "event_id": "bound",
                        "payload": {},
                        "kind": "authority",
                    }
                ],
                journey_binding="synthetic-path",
            )
            self.assertIn("duplicate_logical_result", review.discrepancies)
            self.assertEqual(
                review.ledger.cpu_preparation_create.observed_unit_ids,
                ("job-new", "job-old"),
            )
            self.assertEqual(review.logical_attempts[0].disposition, "inconsistent")
            self.assertIn(
                "response_lost", [row.failure_detail for row in review.logical_results]
            )

    def test_conflicting_resource_attestations_preserve_all_known_ids(self):
        # Synthetic journal vectors exercise the supported compiler, not a provider.
        with tempfile.TemporaryDirectory() as directory:
            journal = ExecutionContactJournal(
                Path(directory) / "journal.jsonl", _EXECUTION_ID
            )
            for job_id, count in (("job-old", 1), ("job-new", 2)):
                identity = dict(
                    stage="cpu",
                    step="renamed",
                    logical_operation="jobs.submit",
                    resource_kind="cpu_job",
                    resource_intent="same-intent",
                )
                invocation = journal.step(phase="invoked", **identity)
                journal.step(
                    phase="verified",
                    invocation_ref=invocation,
                    provider_ids=(job_id,),
                    ceiling_count=count,
                    **identity,
                )
            binding = journal.finalize()
            from bio_neb.parabricks.execution_contact_accounting import (
                read_complete_execution_contacts,
            )

            with self.assertRaisesRegex(ValueError, "incomplete_or_inconsistent"):
                read_complete_execution_contacts(
                    journal.path.read_bytes(), _EXECUTION_ID
                )
            review = compile_execution_contact_evidence(
                journal_bytes=journal.path.read_bytes(),
                binding=binding,
                expected_execution_id=_EXECUTION_ID,
                journey_binding="synthetic-path",
                journey_events=[
                    {
                        "event_type": "ExecutionBound",
                        "event_id": "bound",
                        "payload": {},
                        "kind": "authority",
                    },
                    {
                        "event_type": "ExecutionOutcome",
                        "event_id": "outcome",
                        "pass": 1,
                        "payload": {
                            "contact_journal": str(journal.path),
                            "journal_sha256": binding.sha256,
                            "journal_bytes": binding.byte_count,
                        },
                    },
                ],
            )
            self.assertFalse(review.complete)
            self.assertIn("resource_attestation_conflict", review.discrepancies)
            self.assertIn("resource_intent_identity_conflict", review.discrepancies)
            self.assertEqual(
                review.ledger.cpu_preparation_create.status, "unmeasurable"
            )
            self.assertEqual(
                review.ledger.cpu_preparation_create.observed_unit_ids,
                ("job-new", "job-old"),
            )

    def test_closed_writer_rejects_all_supported_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = ExecutionContactJournal(
                Path(directory) / "journal.jsonl", _EXECUTION_ID
            )
            unit = journal.start(
                category="source",
                operation="http.get",
                executable_identity="http-client",
            )
            journal.finish(unit, disposition="succeeded")
            journal.finalize()
            before = journal.path.read_bytes()
            for action in (
                lambda: journal.start(
                    category="source",
                    operation="http.get",
                    executable_identity="http-client",
                ),
                lambda: journal.finish(unit, disposition="succeeded"),
                lambda: journal.step(stage="cpu", step="submit", phase="claimed"),
                lambda: journal.retain_receipt(
                    kind="job_submission", body={"name": "job"}
                ),
            ):
                with (
                    self.subTest(action=action),
                    self.assertRaisesRegex(ValueError, "not writable"),
                ):
                    action()
            self.assertEqual(journal.path.read_bytes(), before)

    def test_public_versions_and_consumers_keep_diagnostic_contract(self):
        from bio_neb.application.queries import ReviewJourneyQuery, review_journey
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            command_file = str(
                self.prerequisite("executed", writable=True).journey_file
            )
            public = review_execution_contacts(command_file)
            self.assertEqual(public.status, "complete")
            # E12 admits bounded artifact/flow fixtures even when physical
            # network totals remain unknown in this offline operation.
            self.assertTrue(public.review.fixture_eligible)
            self.assertEqual(
                public.review.ledger.model_contact.count,
                _planning_model_contacts(command_file),
            )
            # The Galaxy index search's contacts are retained, the NCBI fakes' are
            # not: the evidence is incomplete.
            self.assertEqual(
                (
                    public.review.ledger.source_contact.count,
                    public.review.ledger.source_contact.reason,
                ),
                (None, "source_contact_evidence_incomplete"),
            )
            self.assertIsNone(public.review.ledger.network_contact.count)
            self.assertEqual(
                public.review.compiler_version, "bio-neb.execution-contact-compiler/2"
            )
            self.assertEqual(
                public.review.ledger.schema_version,
                "bio-neb.fresh-full-predecessor-contact-ledger/3",
            )
            self.assertEqual(
                public.schema_version, "bio-neb.public-execution-contact-review/2"
            )
            before = Path(command_file).read_bytes()
            self.assertEqual(review_execution_contacts(command_file), public)
            observed = review_journey(ReviewJourneyQuery(journey_file=command_file))
            self.assertEqual(observed.disposition, "succeeded")
            self.assertEqual(observed.disposition, "succeeded")
            self.assertEqual(len(observed.verified_objects), 4)
            self.assertEqual(Path(command_file).read_bytes(), before)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "execution-contact-review",
                            "--execution-journey-file",
                            command_file,
                        ]
                    ),
                    0,
                )
            ledger = Path(public.ledger_locator)
            document = json.loads(ledger.read_bytes())
            document["schema_version"] = (
                "bio-neb.fresh-full-predecessor-contact-ledger/unsupported"
            )
            ledger.write_text(canonical_json(document))
            retained = {
                path: path.read_bytes() for path in root.rglob("*") if path.is_file()
            }
            refused = review_execution_contacts(command_file)
            self.assertEqual(refused.status, "drift")
            self.assertIn("materialized_ledger_malformed", refused.review.discrepancies)
            self.assertFalse(refused.review.fixture_eligible)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "execution-contact-review",
                            "--execution-journey-file",
                            command_file,
                        ]
                    ),
                    2,
                )
            self.assertEqual(
                {path: path.read_bytes() for path in root.rglob("*") if path.is_file()},
                retained,
            )

    def test_orphan_agent_activity_cannot_be_projected_as_an_attempt(self):
        from bio_neb.agentic_journey.lifecycle import ActivityLifecycle
        from bio_neb.application.queries import project_journey_events
        from tests.application_fakes import _registered

        # A registered Run has no Attempt yet, so an Agent activity started on it
        # is orphaned without breaking any parentage the reader checks first.
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _registered(Path(directory).resolve())
            journey = Journey.open(Path(receipt.journey_file))
            ActivityLifecycle(
                journey, causal_head=journey.read()[-1]["event_id"]
            ).start(
                adapter="pydantic_ai",
                stage="planning_attempt",
            )
            before = journey.path.read_bytes()
            with self.assertRaisesRegex(
                ValueError, "activity lifecycle evidence has no owning Attempt"
            ):
                project_journey_events(
                    read_evidence(journey), journey_file=str(journey.path)
                )
            self.assertEqual(journey.path.read_bytes(), before)

    def test_current_typed_decisions_drive_metrics_and_reject_conflicts(self):
        self._assert_model_retry_contacts()
        self._assert_terminal_contact_conflicts()
        self._assert_failed_cpu_create_contacts()

    def _assert_model_retry_contacts(self):
        from pydantic_ai.exceptions import ModelHTTPError
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models import ModelRequestParameters

        from bio_neb.agentic_journey.attempt_journey import AttemptFailureSource
        from bio_neb.agentic_journey.lifecycle import ActivityLifecycle
        from bio_neb.agentic_journey.model_observation import JourneyObservedModel
        from tests.model_endpoint_fakes import scripted_adapter

        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory) / "retry")
            anchor = journey.record("AccountingTestStarted", {})
            with processing_pass(journey):
                parent = ActivityLifecycle(
                    journey, causal_head=anchor["event_id"]
                ).start(
                    adapter="pydantic_ai",
                    stage="planning_attempt",
                )
                attempts = 0
                delays = []

                async def respond(messages, info):
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        raise ModelHTTPError(
                            503, "Qwen/Qwen3.5-397B-A17B", body=None, headers={}
                        )
                    return ModelResponse(parts=[TextPart("done")])

                async def sleep(seconds):
                    delays.append(seconds)

                model = JourneyObservedModel(
                    scripted_adapter(respond, model="Qwen/Qwen3.5-397B-A17B"),
                    contact_policy=ModelContactPolicy(),
                    parent_activity=parent,
                    attempt_id="10000000-0000-0000-0000-000000000009",
                    failure_source=AttemptFailureSource(),
                    nebius_model_endpoint={
                        "provider": "nebius",
                        "model": "Qwen/Qwen3.5-397B-A17B",
                        "contacted_url": "https://model.example.test/v1",
                    },
                    availability_sleep=sleep,
                    # The one retry waits base_delay_seconds (30) times the jitter.
                    availability_jitter=lambda: 0.5,
                )
                asyncio.run(
                    model.request(
                        [],
                        {},
                        ModelRequestParameters(
                            function_tools=[],
                            output_tools=[],
                            allow_text_output=True,
                            output_mode="text",
                        ),
                    )
                )
                self.assertEqual((attempts, delays), (2, [15.0]))
                events = read_evidence(journey)
                logical = [
                    e for e in events if e["event_type"] == "ModelRequestRecorded"
                ]
                physical = [
                    e
                    for e in events
                    if e["event_type"] == "ModelPhysicalContactStarted"
                ]
                self.assertEqual(len(logical), 1)
                self.assertEqual(len(physical), 2)
                self.assertEqual(
                    {e["payload"]["logical_request_event_id"] for e in physical},
                    {logical[0]["event_id"]},
                )
                review = compile_execution_contact_evidence(
                    journal_bytes=b"",
                    binding=None,
                    expected_execution_id=_EXECUTION_ID,
                    journey_events=events,
                    journey_binding=_JOURNEY_BINDING,
                )
                self.assertEqual(review.ledger.model_contact.count, attempts)

    def _assert_terminal_contact_conflicts(self):
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )
        from bio_neb.parabricks.result import read_completed_parabricks_result

        command_file = str(self.prerequisite("executed", writable=True).journey_file)
        packet = read_completed_parabricks_result(command_file)
        public = review_execution_contacts(command_file)
        journal = SimpleNamespace(path=Path(public.journal_locator))
        self.assertEqual(public.status, "complete")
        self.assertEqual(public.review.ledger.fresh_bucket.count, 1)
        self.assertEqual(public.review.ledger.cpu_preparation_create.count, 1)
        self.assertEqual(public.review.ledger.h100_create.count, 1)
        self.assertTrue(public.review.fixture_eligible)
        self.assertEqual(review_execution_contacts(command_file), public)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(
                main(
                    [
                        "execution-contact-review",
                        "--execution-journey-file",
                        command_file,
                    ]
                ),
                0,
            )
        ledger_path = Path(public.ledger_locator)
        ledger_bytes = ledger_path.read_bytes()
        drifted = json.loads(ledger_bytes)
        drifted["source_contact"]["evidence_bindings"] = ["journey:foreign"]
        ledger_path.write_text(canonical_json(drifted) + "\n", encoding="utf-8")
        drifted_bytes = ledger_path.read_bytes()
        self.assertEqual(review_execution_contacts(command_file).status, "drift")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(
                main(
                    [
                        "execution-contact-review",
                        "--execution-journey-file",
                        command_file,
                    ]
                ),
                2,
            )
        self.assertEqual(ledger_path.read_bytes(), drifted_bytes)
        ledger_path.write_bytes(ledger_bytes)

        import copy

        events = [
            json.loads(line) for line in Path(command_file).read_bytes().splitlines()
        ]
        # Model contacts are mechanical evidence in the Pass contact journals;
        # the review reads them merged with the Journey's semantic facts.
        evidence = read_evidence(Journey.open(Path(command_file)))
        model_contact = next(
            e for e in evidence if e["event_type"] == "ModelPhysicalContactStarted"
        )
        for field, value in (
            ("logical_request_event_id", "10000000-0000-0000-0000-000000000099"),
            ("contact_ordinal", 7),
        ):
            with self.subTest(model_conflict=field):
                hostile = copy.deepcopy(evidence)
                next(e for e in hostile if e["event_id"] == model_contact["event_id"])[
                    "payload"
                ][field] = value
                with self.assertRaises(ValueError):
                    compile_execution_contact_evidence(
                        journal_bytes=journal.path.read_bytes(),
                        binding=public.review.binding,
                        expected_execution_id=packet.execution_id,
                        journey_events=hostile,
                        journey_binding=_JOURNEY_BINDING,
                    )
        # Semantic result projection independently rejects changed reference identity.
        original_bytes = Path(command_file).read_bytes()
        hostile = copy.deepcopy(events)
        prepared = next(
            e["payload"]
            for e in hostile
            if e["event_type"] == "ExecutionPathRecorded"
            and e["payload"]["event_type"] == "ReferencePrepared"
        )
        prepared["payload"]["receipt_id"] = "sha256:" + "0" * 64
        Path(command_file).write_text(
            "".join(canonical_json(e) + "\n" for e in hostile)
        )
        with self.assertRaisesRegex(ValueError, "parabricks_result_identity_drift"):
            read_completed_parabricks_result(command_file)
        Path(command_file).write_bytes(original_bytes)

    def _assert_failed_cpu_create_contacts(self):
        import copy
        from collections import deque

        from bio_neb.failure import ProviderError
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )
        from bio_neb.parabricks.operation import execute_parabricks
        from tests.parabricks_operation_fakes import input_storage_provider
        from tests.reference_stage_fakes import seed_cpu

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            approved = self.prerequisite("approved").value

            def factory(plan, journal):
                provider = seed_cpu(
                    input_storage_provider(plan, journal), plan, approved
                )
                provider.scripts["cpu_preparation", "submit_cpu"] = deque(
                    [
                        ProviderError(
                            "rejected",
                            "submission_failed",
                            {"provider_job_id": "job-known"},
                        )
                    ]
                )
                return provider

            state = execute_parabricks(approved, infrastructure_factory=factory)
            self.assertEqual(state.status, "execution_review_pending")
            self.assertEqual(state.disposition, "incomplete")
            self.assertNotIn("verified_objects", state.model_dump())
            public = review_execution_contacts(approved.journey_file)
            attempts = [
                a
                for a in public.review.logical_attempts
                if a.resource_kind == "cpu_job"
            ]
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0].disposition, "failed")
            self.assertEqual(attempts[0].failure_detail, "submission_failed")
            self.assertEqual(
                public.review.ledger.cpu_preparation_create.observed_unit_ids,
                ("job-known",),
            )
            self.assertIsNone(public.review.ledger.cpu_preparation_create.count)
            self.assertFalse(public.review.fixture_eligible)

        from bio_neb.investigation.investigation_observation_commands import (
            investigation_observe,
        )
        from tests.sequential_investigation_support import (
            ObservationAdapter,
            ObservationEvidence,
            approved_collection_observation,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            selected, journey, _, descriptor, command = approved_collection_observation(
                root, cardinality=1
            )
            adapter = ObservationAdapter(command, descriptor, selected.sample_id)
            investigation_observe(
                command,
                adapter=adapter,
                evidence_reader=ObservationEvidence(descriptor),
            )
            calls = [argv for argv in adapter.calls if argv[1] == "query"]
            self.assertEqual(len(calls), 1)
            review = compile_execution_contact_evidence(
                journal_bytes=b"",
                binding=None,
                expected_execution_id=_EXECUTION_ID,
                journey_events=read_evidence(Journey.open(journey.path)),
                journey_binding=_JOURNEY_BINDING,
            )
            self.assertEqual(review.ledger.post_execution_vcf_query.count, 1)
            events = read_evidence(Journey.open(journey.path))
            for conflict in (
                "missing_authority",
                "foreign_parent",
                "reordered_evidence",
            ):
                with self.subTest(vcf_conflict=conflict):
                    hostile = copy.deepcopy(events)
                    observed = next(
                        e
                        for e in hostile
                        if e["event_type"] == "InvestigationEvidenceRecorded"
                    )
                    if conflict == "missing_authority":
                        hostile = [
                            e
                            for e in hostile
                            if e["event_type"] != "InvestigationAuthorityGranted"
                        ]
                    elif conflict == "foreign_parent":
                        observed["parent_ids"] = [hostile[0]["event_id"]]
                    else:
                        hostile.remove(observed)
                        hostile.insert(0, observed)
                    with self.assertRaises(ValueError):
                        compile_execution_contact_evidence(
                            journal_bytes=b"",
                            binding=None,
                            expected_execution_id=_EXECUTION_ID,
                            journey_events=hostile,
                            journey_binding=_JOURNEY_BINDING,
                        )

        class InterruptedObservation(ObservationAdapter):
            query_attempts = 0

            def run(self, argv, *, timeout_seconds, stdout_path=None):
                if argv[1:4] == ("query", "--allow-undef-tags", "--regions-file"):
                    self.query_attempts += 1
                    if self.query_attempts == 2:
                        raise KeyboardInterrupt("controlled collection interruption")
                return super().run(
                    argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            selected, journey, _, descriptor, command = approved_collection_observation(
                root, cardinality=2
            )
            adapter = InterruptedObservation(command, descriptor, selected.sample_id)
            with self.assertRaisesRegex(
                KeyboardInterrupt, "controlled collection interruption"
            ):
                investigation_observe(
                    command,
                    adapter=adapter,
                    evidence_reader=ObservationEvidence(descriptor),
                )
            events = read_evidence(Journey.open(journey.path))
            self.assertEqual(adapter.query_attempts, 2)
            self.assertEqual(
                sum(e["event_type"] == "InvestigationWorkItemStarted" for e in events),
                2,
            )
            retained = [
                e for e in events if e["event_type"] == "InvestigationEvidenceRecorded"
            ]
            self.assertEqual(len(retained), 1)
            review = compile_execution_contact_evidence(
                journal_bytes=b"",
                binding=None,
                expected_execution_id=_EXECUTION_ID,
                journey_events=events,
                journey_binding=_JOURNEY_BINDING,
            )
            metric = review.ledger.post_execution_vcf_query
            self.assertEqual(metric.status, "unmeasurable")
            self.assertIsNone(metric.count)
            self.assertEqual(metric.observed_count, 1)
            self.assertEqual(
                metric.observed_unit_ids, ("journey-vcf:" + retained[0]["event_id"],)
            )
            self.assertFalse(review.fixture_eligible)
            self.assertEqual(
                type(review).model_validate_json(review.model_dump_json()), review
            )
            self.assertEqual(
                compile_execution_contact_evidence(
                    journal_bytes=b"",
                    binding=None,
                    expected_execution_id=_EXECUTION_ID,
                    journey_events=read_evidence(Journey.open(journey.path)),
                    journey_binding=_JOURNEY_BINDING,
                ),
                review,
            )

    def test_investigation_source_transport_contacts_reach_compiler(self):
        self._assert_investigation_source_contacts()

    def test_source_transport_multiplicity_failures_and_refusal_are_retained(self):
        # Reference source science is reached only through the shared planning
        # assignment: each source Tool call binds its own contacts to one scope.
        self._assert_reference_source_multiplicity()
        self._assert_reference_source_interruption_and_refusal()

    def test_a_source_contact_past_its_limit_is_refused_before_it_is_sent(self):
        # A breach is a failure that asks the surface's Review. The first GET
        # times out; the retry its policy schedules is past the limit. A call's
        # budget is that call's; the Run's total counts every Pass.
        from bio_neb.application.advancement import (
            AnswerReview,
            JunctionReviewAnswer,
            ResumeRun,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.failure import ContactLimitOverrides
        from bio_neb.providers.ncbi_datasets import (
            NcbiDatasetsReferenceAdapter,
            UrllibNcbiTransport,
        )
        from bio_neb.recording import read_evidence
        from tests.application_fakes import _registered
        from tests.planning_assignment_fakes import PlanningComposition

        # After the retry answer a new call has its own budget, but the Run's
        # spent total still refuses its first contact: no Pass resets it.
        for limit, failure, declared, sent_after_retry, refused_attempts in (
            (
                "contacts",
                None,
                {"resolve_taxon": ContactLimitOverrides(contacts=1)},
                2,
                [2, 2],
            ),
            ("run_contacts", {"limits": {"run_contacts": 1}}, {}, 1, [2, 1]),
        ):
            with (
                self.subTest(limit=limit),
                tempfile.TemporaryDirectory() as directory,
                patch("urllib.request.build_opener") as opener,
                patch("time.sleep"),
                declaring(declared),
            ):
                opener.return_value.open.side_effect = TimeoutError("controlled")
                receipt, paths = _registered(Path(directory), failure=failure)
                composition = PlanningComposition(receipt, paths)
                source = NcbiDatasetsReferenceAdapter(UrllibNcbiTransport())
                composition.reference_adapter = source
                composition.companion_adapter = source
                with use_run_advancement_composition(composition):
                    stopped = advance(ResumeRun(journey_file=receipt.journey_file))
                    self.assertEqual(opener.return_value.open.call_count, 1)
                    request = stopped.current_review
                    advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=request.request_event_id,
                            answer=JunctionReviewAnswer(
                                kind="junction",
                                response={
                                    "request_id": request.request.request.request_id,
                                    "action": "retry",
                                    "by": "researcher",
                                },
                            ),
                        )
                    )
                self.assertEqual(opener.return_value.open.call_count, sent_after_retry)
                events = read_evidence(Journey.open(Path(receipt.journey_file)))
                breaches = [
                    e["payload"]
                    for e in events
                    if e["event_type"] == "SourceContactLimitReached"
                ]
                self.assertEqual(
                    [(b["limit"], b["bound"], b["observed"]) for b in breaches],
                    [(limit, 1, 1)] * 2,
                )
                self.assertEqual(
                    breaches[0]["limits"]["surface"],
                    {"scope": "tool", "name": "resolve_taxon"},
                )
                self.assertEqual(
                    breaches[0]["limits"]["run_contacts"],
                    1 if limit == "run_contacts" else None,
                )
                # Each refusal is the exhausted failure its policy routes to a Review.
                refused = [
                    e["payload"]["contact_attempt"]
                    for e in events
                    if e["event_type"] == "SourceOperationFailureRecorded"
                    and e["payload"]["failure_class"] == "exhausted"
                ]
                self.assertEqual(refused, refused_attempts)
                self.assertEqual(
                    sum(
                        e["event_type"] == "AgentFailureReviewRequested" for e in events
                    ),
                    2,
                )
                self.assertEqual(
                    sum(e["event_type"] == "PassStarted" for e in events), 2
                )

    @classmethod
    def _failed_source_tool(cls):
        """#143 live (143-live-61bd-a1): a guard refuses the GET inside a source
        Tool, so the Tool fails mechanically after the sample attestation's own
        call. One refused action comes first, as the live Run corrected its own.
        The model is shown the failure once; the same failure again holds the
        Run (#477 C8; pinned by test_planning_assignment's shown-once test)."""
        from functools import partial

        from bio_neb.providers.ncbi_datasets import UrllibNcbiTransport
        from tests.planning_assignment_fakes import PlanningComposition

        class GuardRefused(RuntimeError):
            pass

        class RefusingGuard(UrllibNcbiTransport):
            def get(self, url, *, max_bytes):
                raise GuardRefused("source GET ceiling exhausted")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            events = cls._advance_planning_with_reference_source(
                root,
                RefusingGuard(),
                partial(PlanningComposition, hostile=True),
                held=True,
            )
            (journey_file,) = root.rglob("journey.jsonl")
        by_type = {}
        for event in events:
            by_type.setdefault(event["event_type"], []).append(event)
        return events, by_type, str(journey_file)

    def test_a_failed_source_tool_reads_back_with_its_cause(self):
        # The model is shown the first failure; the same failure again holds
        # the Run.
        _events, by_type, _journey_file = self._failed_source_tool()

        shown_failed, failed = by_type["ObservedToolExecutionFailed"]
        (attempt_failed,) = by_type["AttemptFailed"]
        (refusal,) = by_type["ToolExecutionRefused"]
        self.assertEqual(
            (refusal["payload"]["failed_event_id"], refusal["payload"]["fault_key"]),
            (shown_failed["event_id"], "resolve_taxon: GuardRefused"),
        )
        _shown_stage_failure, stage_failure = by_type["ToolExecutionFailed"]
        self.assertEqual(len(by_type["SampleAttestationRecorded"]), 1)
        self.assertNotIn("TaxonResolutionRecorded", by_type)
        self.assertEqual(stage_failure["payload"]["tool_name"], "resolve_taxon")
        self.assertIn(
            "invalid", [e["payload"]["status"] for e in by_type["ToolCallValidated"]]
        )
        self.assertEqual(
            (
                failed["payload"]["mechanical_error_type"],
                stage_failure["payload"]["error_type"],
            ),
            ("GuardRefused", "GuardRefused"),
        )
        self.assertEqual(
            (
                attempt_failed["payload"]["code"],
                attempt_failed["payload"]["invariant_id"],
                attempt_failed["payload"]["source_event_id"],
            ),
            ("invariant", "tool_failed_again", failed["event_id"]),
        )

    def test_failed_source_tool_permits_its_inner_failure_only_with_its_cause(self):
        import copy

        from bio_neb.application.queries import project_journey_events

        events, by_type, journey_file = self._failed_source_tool()
        _shown_stage_failure, stage_failure = by_type["ToolExecutionFailed"]
        hostile = copy.deepcopy(events)
        hostile[events.index(stage_failure)]["payload"]["error_type"] = "OSError"
        with self.assertRaisesRegex(ValueError, "outside its active causal chain"):
            project_journey_events(hostile, journey_file=journey_file)

    @staticmethod
    def _source_http_response(body):
        value = io.BytesIO(body)
        value.status = 200
        value.headers = {}
        return value

    def _assert_investigation_source_contacts(self):
        from bio_neb.providers.ncbi_datasets import UrllibNcbiTransport
        from bio_neb.providers.ncbi_investigation import (
            NcbiInvestigationSourceAdapter,
        )
        from tests.sequential_investigation_support import (
            ASSEMBLY_BODY,
            REFSNP_BODY,
            SequentialAdaptiveModel,
            input_and_assertion,
            run_attempt,
        )

        response = self._source_http_response

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch("urllib.request.build_opener") as opener:
                opener.return_value.open.side_effect = [
                    response(REFSNP_BODY),
                    response(ASSEMBLY_BODY),
                ]
                adapter = NcbiInvestigationSourceAdapter(UrllibNcbiTransport())
                _, journey = run_attempt(
                    root / "investigation",
                    SequentialAdaptiveModel(),
                    adapter,
                    investigation_input=input_and_assertion()[0],
                    assertion=input_and_assertion()[1],
                )
                self.assertEqual(opener.return_value.open.call_count, 2)
            raw, binding = self._healthy_journal(root)
            review = compile_execution_contact_evidence(
                journal_bytes=raw,
                binding=binding,
                expected_execution_id=_EXECUTION_ID,
                journey_events=read_evidence(journey),
                journey_binding=_JOURNEY_BINDING,
            )
            self.assertEqual(review.ledger.source_contact.count, 2)

    @staticmethod
    def _advance_planning_with_reference_source(
        root, transport, composition_type=None, *, held=False, galaxy=None
    ):
        """Advance the real planning assignment with one NCBI adapter for both roles.

        Only the scripted model and the transport's opener are controlled; the
        coordinator, grouped reference Tools, adapter, transport and recorder run.
        ``held`` expects the failure module to hold the Run instead of a stop.
        """
        from bio_neb.application.advancement import (
            ResumeRun,
            RunAdvancementRefused,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.providers.ncbi_datasets import NcbiDatasetsReferenceAdapter
        from bio_neb.recording import read_evidence
        from tests.application_fakes import _registered
        from tests.planning_assignment_fakes import PlanningComposition

        receipt, paths = _registered(root)
        composition = (composition_type or PlanningComposition)(receipt, paths)
        source = NcbiDatasetsReferenceAdapter(transport)
        composition.reference_adapter = source
        composition.companion_adapter = source
        if galaxy is not None:
            composition.galaxy_adapter = galaxy
        with use_run_advancement_composition(composition):
            try:
                advance(ResumeRun(journey_file=receipt.journey_file))
            except RunAdvancementRefused:
                if not held:
                    raise
            else:
                if held:
                    raise AssertionError("the Run was not held")
        return read_evidence(Journey.open(Path(receipt.journey_file)))

    @staticmethod
    def _source_scopes(events):
        """Each retained Tool scope with its physical attempts' terminals, in order."""
        terminals = {
            e["payload"]["started_event_id"]: e["payload"]
            for e in events
            if e["event_type"] == "SourcePhysicalContactFinished"
        }
        scopes = []
        for scope in events:
            if scope["event_type"] != "SourceContactScopeStarted":
                continue
            outcomes = []
            for start in events:
                if (
                    start["event_type"] != "SourcePhysicalContactStarted"
                    or start["payload"]["scope_event_id"] != scope["event_id"]
                ):
                    continue
                terminal = terminals.get(start["event_id"])
                if terminal is None:
                    outcomes.append("unfinished")
                elif terminal["error_type"] is None:
                    outcomes.append(terminal["disposition"])
                else:
                    outcomes.append(
                        f"{terminal['disposition']}:{terminal['error_type']}"
                    )
            scopes.append((scope["payload"]["tool_name"], outcomes))
        return scopes

    def _assert_reference_source_multiplicity(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        from bio_neb.providers.galaxy_datacache import GalaxyDatacache
        from bio_neb.providers.ncbi_datasets import UrllibNcbiTransport
        from tests.planning_assignment_fakes import PlanningComposition

        response = self._source_http_response

        class RefineOnce(PlanningComposition):
            """Refine an ambiguous discovery once by a retained isolate."""

            refined = False

            async def respond(self, messages, info):
                from bio_neb.application.planning_read import read_planning
                from bio_neb.journey import Journey
                from tests.planning_assignment_fakes import eligible_tools

                view = read_planning(Journey.open(Path(self.journey.path))).reference
                facts = () if view is None else view.facts
                discovery = next(
                    (
                        f.result
                        for f in facts
                        if f.event_type == "ReferenceAssembliesRecorded"
                    ),
                    None,
                )
                if (
                    self.refined
                    or discovery is None
                    or len(discovery.get("rows", ())) < 2
                    or "discover_assemblies"
                    not in eligible_tools(self.journey, info.function_tools)
                ):
                    return await super().respond(messages, info)
                self.refined = True
                self.calls.append("discover_assemblies")
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "discover_assemblies",
                            {
                                "taxid": view.resolved_taxon.ncbi_tax_id,
                                "search_text": ["isolate-B"],
                            },
                            tool_call_id=f"refine-{len(self.calls)}",
                        )
                    ]
                )

        def assembly(accession, tax_id, strain, status="current"):
            return {
                "accession": accession,
                "organism": {
                    "organism_name": "Homo sapiens",
                    "tax_id": tax_id,
                    "infraspecific_names": {"strain": strain},
                },
                "assembly_info": {
                    "assembly_name": "assembly-" + accession,
                    "assembly_status": status,
                    "refseq_category": "reference genome",
                    "assembly_level": "chromosome",
                },
            }

        ok = "succeeded"
        scenarios = (
            # One scope per source call: taxon, discovery, exact description and
            # companion summary. The companion summary body is invalid, but the
            # request is still a paid, mechanically completed one.
            (
                "single_candidate",
                None,
                4,
                [
                    ("resolve_taxon", [ok]),
                    ("discover_assemblies", [ok]),
                    ("inspect_reference", [ok]),
                    ("describe_companion", [ok]),
                ],
            ),
            # The model's refined query is one GET: discovery checks no row's
            # lineage. The genomic FASTA companion reads its summary, the
            # assembly name, the publisher's md5 listing, the file's HEAD and the
            # assembly report beside it under one scope; the index search reads
            # Galaxy BRC's two catalogues under its own.
            (
                "refined_query",
                False,
                11,
                [
                    ("resolve_taxon", [ok]),
                    ("discover_assemblies", [ok]),
                    ("discover_assemblies", [ok]),
                    ("inspect_reference", [ok]),
                    ("describe_companion", [ok, ok, ok, ok, ok]),
                    ("search_prebuilt_indexes", [ok, ok]),
                ],
            ),
            # The refined query's GET times out: the shared operation window (three
            # attempts by default) repeats it, every failed contact is retained and
            # counted, and the discovery stops after the last attempt.
            (
                "failed_refined_query",
                True,
                2 + 3,
                [
                    ("resolve_taxon", [ok]),
                    ("discover_assemblies", [ok]),
                    ("discover_assemblies", ["failed:TimeoutError"] * 3),
                ],
            ),
        )
        for scenario, fail_lineage, contacts, expected_scopes in scenarios:
            with (
                self.subTest(scenario=scenario),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                requested = []
                methods = []

                def open_reference(
                    request,
                    *,
                    timeout,
                    fail_lineage=fail_lineage,
                    requested=requested,
                    methods=methods,
                ):
                    url = request.full_url
                    requested.append(url)
                    methods.append(request.get_method())
                    if "/genome/taxon/" in url:
                        if fail_lineage is None:
                            reports = [assembly("GCF_000009999.1", 9606, "isolate-B")]
                        elif sum("/genome/taxon/" in item for item in requested) == 1:
                            reports = [
                                assembly("GCF_000001405.40", 9606, "isolate-A"),
                                assembly("GCF_000009999.1", 9606, "isolate-B"),
                            ]
                        elif fail_lineage:
                            raise TimeoutError("controlled discovery timeout")
                        else:
                            reports = [
                                assembly("GCF_000009999.1", 10091, "isolate-B"),
                                assembly(
                                    "GCF_000000777.1", 10092, "isolate-B", "suppressed"
                                ),
                            ]
                        return response(json.dumps({"reports": reports}).encode())
                    if "/taxonomy/" in url:
                        tax_id = (
                            10091
                            if "/10091/" in url
                            else 10092
                            if "/10092/" in url
                            else 9606
                        )
                        return response(
                            json.dumps(
                                {
                                    "taxonomy_nodes": [
                                        {
                                            "taxonomy": {
                                                "tax_id": tax_id,
                                                "current_scientific_name": {
                                                    "name": "Homo sapiens"
                                                },
                                                "rank": "species",
                                                "lineage": [
                                                    {
                                                        "tax_id": 9606,
                                                        "scientific_name": "Homo sapiens",
                                                    }
                                                ],
                                            }
                                        }
                                    ]
                                }
                            ).encode()
                        )
                    if "datacache.galaxyproject.org" in url:
                        # Galaxy's catalogues list no index.
                        return response(b"")
                    if url.endswith("_assembly_report.txt"):
                        return response(
                            b"one\tassembled-molecule\t1\tChromosome\tCM1.1\t=\t"
                            b"NC_1.1\tPrimary Assembly\t4096\tchr1\n"
                        )
                    if url.endswith("/md5checksums.txt"):
                        listed = url.split("/")[-2] + "_genomic.fna.gz"
                        return response(f"{'c' * 32}  ./{listed}\n".encode())
                    if request.get_method() == "HEAD":
                        head = response(b"")
                        head.headers = {"Content-Length": "4096"}
                        return head
                    if url.endswith("/download_summary"):
                        return response(
                            b"{}"
                            if fail_lineage is None
                            else json.dumps(
                                {
                                    "available_files": {
                                        "all_genomic_fasta": {"file_count": 1}
                                    }
                                }
                            ).encode()
                        )
                    return response(
                        json.dumps(
                            {
                                "reports": [
                                    assembly(
                                        "GCF_000009999.1",
                                        9606 if fail_lineage is None else 10091,
                                        "isolate-B",
                                    )
                                ]
                            }
                        ).encode()
                    )

                with patch("urllib.request.build_opener") as opener:
                    opener.return_value.open.side_effect = open_reference
                    events = self._advance_planning_with_reference_source(
                        root,
                        UrllibNcbiTransport(),
                        PlanningComposition if fail_lineage is None else RefineOnce,
                        galaxy=GalaxyDatacache(),
                    )
                    self.assertEqual(opener.return_value.open.call_count, contacts)
                starts = [
                    e
                    for e in events
                    if e["event_type"] == "SourcePhysicalContactStarted"
                ]
                # Each retained attempt is exactly one invoked request, in order.
                self.assertEqual(
                    [e["payload"]["request_identity"] for e in starts], requested
                )
                self.assertEqual(self._source_scopes(events), expected_scopes)
                # The retained Journey method and Pass-journal operation are the
                # ones sent on the wire: only the companion's size read is a HEAD.
                self.assertEqual(
                    methods.count("HEAD"), 1 if fail_lineage is False else 0
                )
                self.assertEqual([e["payload"]["method"] for e in starts], methods)
                (journey_file,) = (root / "runs").rglob("journey.jsonl")
                journal_operations = [
                    row.operation
                    for view in read_passes(Journey.open(journey_file))
                    for row in view.contacts
                    if row.kind == "start" and row.operation.startswith("source.")
                ]
                self.assertEqual(
                    journal_operations, ["source." + m.lower() for m in methods]
                )
                raw, binding = self._healthy_journal(root)
                review = compile_execution_contact_evidence(
                    journal_bytes=raw,
                    binding=binding,
                    expected_execution_id=_EXECUTION_ID,
                    journey_events=events,
                    journey_binding=_JOURNEY_BINDING,
                )
                self.assertEqual(review.ledger.source_contact.status, "measured")
                self.assertEqual(review.ledger.source_contact.count, contacts)
                self.assertEqual(
                    review.ledger.source_contact.unit_ids,
                    tuple("journey-source:" + e["event_id"] for e in starts),
                )
                if fail_lineage is None:
                    companion = next(
                        e["payload"]["result"]
                        for e in events
                        if e["event_type"] == "CompanionObservationRecorded"
                    )
                    self.assertEqual(
                        (companion["status"], companion["failure"]["code"]),
                        ("failed", "invalid_response"),
                    )
                    self._assert_source_lineage_integrity(events)
                if fail_lineage is False:
                    companion = next(
                        e["payload"]["result"]
                        for e in events
                        if e["event_type"] == "CompanionObservationRecorded"
                    )
                    (fasta,) = companion["candidates"]
                    self.assertEqual(
                        (
                            fasta["source_locator"],
                            fasta["checksum"],
                            fasta["byte_size"],
                        ),
                        (requested[methods.index("HEAD")], "c" * 32, 4096),
                    )
                    discovery = [
                        e
                        for e in events
                        if e["event_type"] == "ReferenceAssembliesRecorded"
                    ][-1]
                    # Every row of the refined page is copied, the suppressed one
                    # with its status.
                    self.assertEqual(
                        [
                            row["assembly_status"]
                            for row in discovery["payload"]["result"]["rows"]
                        ],
                        ["current", "suppressed"],
                    )

    def _assert_reference_source_interruption_and_refusal(self):
        from bio_neb.agentic_journey.lifecycle import Activity
        from bio_neb.providers.ncbi_datasets import UrllibNcbiTransport

        class RefusingTransport(UrllibNcbiTransport):
            def get(self, url, *, max_bytes):
                return super().get(
                    "http://unapproved.example.test", max_bytes=max_bytes
                )

        original_record_contact = Activity.record_contact

        def interrupted_terminal(activity, fact):
            if fact.event_type == "SourcePhysicalContactFinished":
                raise OSError("controlled terminal persistence failure")
            return original_record_contact(activity, fact)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with (
                patch("urllib.request.build_opener") as opener,
                patch.object(Activity, "record_contact", interrupted_terminal),
            ):
                opener.return_value.open.return_value = self._source_http_response(
                    json.dumps(
                        {
                            "taxonomy_nodes": [
                                {
                                    "taxonomy": {
                                        "tax_id": 9606,
                                        "current_scientific_name": {
                                            "name": "Homo sapiens"
                                        },
                                    }
                                }
                            ]
                        }
                    ).encode()
                )
                events = self._advance_planning_with_reference_source(
                    root, UrllibNcbiTransport(), held=True
                )
                self.assertEqual(opener.return_value.open.call_count, 1)
            self.assertEqual(
                self._source_scopes(events), [("resolve_taxon", ["unfinished"])]
            )
            # A local evidence-write failure is not a transport fault: it is not
            # retried, and the failure module holds the Run for a fix.
            (hold,) = [
                event["payload"]["failure"]
                for event in events
                if event["event_type"] == "PassClosed"
                and event["payload"]["disposition"] == "held"
            ]
            self.assertEqual(hold["invariant_id"], "planning_attempt_failed")
            self.assertRegex(hold["cause"], r"^OSError@")
            raw, binding = self._healthy_journal(root)
            review = compile_execution_contact_evidence(
                journal_bytes=raw,
                binding=binding,
                expected_execution_id=_EXECUTION_ID,
                journey_events=events,
                journey_binding=_JOURNEY_BINDING,
            )
            metric = review.ledger.source_contact
            self.assertEqual(metric.status, "unmeasurable")
            self.assertEqual(metric.reason, "source_contact_evidence_incomplete")
            self.assertIsNone(metric.count)
            self.assertEqual(metric.observed_count, 1)
            self.assertEqual(
                metric.observed_unit_ids,
                tuple(
                    "journey-source:" + e["event_id"]
                    for e in events
                    if e["event_type"] == "SourcePhysicalContactStarted"
                ),
            )
            self.assertEqual(
                type(review).model_validate_json(review.model_dump_json()), review
            )
            self.assertFalse(review.fixture_eligible)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch("urllib.request.build_opener") as opener:
                events = self._advance_planning_with_reference_source(
                    root, RefusingTransport()
                )
                self.assertEqual(opener.return_value.open.call_count, 0)
            # The transport was observed and refused before any physical attempt.
            self.assertEqual(self._source_scopes(events), [("resolve_taxon", [])])
            raw, binding = self._healthy_journal(root)
            review = compile_execution_contact_evidence(
                journal_bytes=raw,
                binding=binding,
                expected_execution_id=_EXECUTION_ID,
                journey_events=events,
                journey_binding=_JOURNEY_BINDING,
            )
            self.assertEqual(review.ledger.source_contact.status, "measured")
        self.assertEqual(review.ledger.source_contact.count, 0)

    def _assert_source_lineage_integrity(self, events):
        import copy

        from bio_neb.agentic_journey.source_contacts import (
            source_contact_units,
        )

        refusals = (
            ("detached_scope", "source attempt is outside its open scope"),
            ("foreign_adapter", "source Tool changed its declared Adapter identity"),
            ("unbound_result", "unsupported unbound stage evidence"),
            ("call_identity", "changed its exact model call"),
            ("call_name", "changed its exact model call"),
            ("call_arguments", "changed its exact model call"),
            ("response_order", "changed its admitted response or validation"),
            ("disposition_order", "changed its exact admission disposition"),
        )
        # Per source Tool: its one result fact, another Tool's name, and its
        # admitted argument with a foreign value.
        tools = {
            "resolve_taxon": (
                "TaxonResolutionRecorded",
                "inspect_reference",
                ("term", "Mus musculus"),
            ),
            "inspect_reference": (
                "ReferenceDescriptionRecorded",
                "resolve_taxon",
                ("accession", "GCF_000000001.1"),
            ),
        }
        for tool_name, (result_type, other_tool, foreign_argument) in tools.items():
            for conflict, reason in refusals:
                with self.subTest(source_tool=tool_name, source_conflict=conflict):
                    hostile = copy.deepcopy(events)
                    started = next(
                        e
                        for e in hostile
                        if e["event_type"] == "ObservedToolExecutionStarted"
                        and e["payload"]["tool_name"] == tool_name
                    )
                    if conflict == "detached_scope":
                        # Mechanical contact evidence names its scope by identity.
                        scope = next(
                            e
                            for e in hostile
                            if e["event_type"] == "SourceContactScopeStarted"
                            and e["payload"]["tool_call_id"]
                            == started["payload"]["tool_call_id"]
                        )
                        next(
                            e
                            for e in hostile
                            if e["event_type"] == "SourcePhysicalContactStarted"
                            and e["payload"]["scope_event_id"] == scope["event_id"]
                        )["payload"]["scope_event_id"] = hostile[0]["event_id"]
                    elif conflict == "foreign_adapter":
                        started["payload"]["adapter_identity"] = "foreign-adapter"
                    elif conflict == "unbound_result":
                        # A copy outside any Tool execution binds to no source call.
                        unbound = copy.deepcopy(
                            next(
                                e
                                for e in hostile[hostile.index(started) :]
                                if e["event_type"] == result_type
                            )
                        )
                        unbound["event_id"] = "10000000-0000-0000-0000-000000000099"
                        hostile.append(unbound)
                    else:
                        self._change_admitted_source_call(
                            hostile, started, conflict, other_tool, foreign_argument
                        )
                    with self.assertRaisesRegex(ValueError, reason):
                        source_contact_units(hostile)
        started_index = next(
            i
            for i, e in enumerate(events)
            if e["event_type"] == "SourcePhysicalContactStarted"
        )
        prefix_units, prefix_complete = source_contact_units(
            events[: started_index + 1]
        )
        self.assertEqual(len(prefix_units), 1)
        self.assertFalse(prefix_complete)

    @staticmethod
    def _change_admitted_source_call(
        hostile, started, conflict, other_tool, foreign_argument
    ):
        model_response = next(
            e
            for e in hostile
            if e["event_id"] == started["payload"]["response_event_id"]
        )
        validation = next(
            e
            for e in hostile
            if e["event_id"] == started["payload"]["validation_event_id"]
        )
        call = next(
            part
            for part in model_response["payload"]["compatible_adapter_evidence"][
                "response"
            ]["message"]["parts"]
            if part.get("part_kind") == "tool-call"
            and part.get("tool_call_id") == started["payload"]["tool_call_id"]
        )
        if conflict == "call_identity":
            call["tool_call_id"] = "foreign-call"
        elif conflict == "call_name":
            call["tool_name"] = other_tool
        elif conflict == "call_arguments":
            name, value = foreign_argument
            # The wire returns a call's arguments as JSON text.
            arguments = json.loads(call["args"])
            call["args"] = json.dumps({**arguments, name: value})
        elif conflict == "response_order":
            hostile.remove(model_response)
            hostile.insert(hostile.index(validation) + 1, model_response)
        else:
            disposition = next(
                e
                for e in hostile
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
                and e["payload"]["response_event_id"] == model_response["event_id"]
            )
            hostile.remove(disposition)
            hostile.insert(hostile.index(validation), disposition)

    def _healthy_journal(
        self, root: Path
    ) -> tuple[bytes, ExecutionContactJournalBinding]:
        journal = ExecutionContactJournal(root / "journal.jsonl", _EXECUTION_ID)
        local = journal.start(
            category="local_runtime",
            operation="nebius.version",
            executable_identity="nebius-cli:" + "c" * 64,
        )
        journal.finish(local, disposition="succeeded", exit_status=0)
        provider = journal.start(
            category="provider",
            operation="iam.whoami",
            executable_identity="nebius-cli:" + "c" * 64,
        )
        journal.finish(provider, disposition="succeeded", exit_status=0)
        bucket_create = journal.start(
            category="provider",
            operation="storage.bucket.create",
            executable_identity="nebius-cli:" + "c" * 64,
        )
        journal.finish(bucket_create, disposition="succeeded", exit_status=0)
        h100_create = journal.start(
            category="provider",
            operation="ai.job.create",
            executable_identity="nebius-cli:" + "c" * 64,
        )
        journal.finish(h100_create, disposition="succeeded", exit_status=0)
        storage = journal.start(
            category="storage",
            operation="s3.head-object",
            executable_identity="s3-sdk",
        )
        journal.finish(storage, disposition="failed", parse_classification="transient")
        credential = journal.start(
            category="credential",
            operation="selected-object-storage-credential",
            executable_identity="credential-resolver",
            credential_names=(
                "control_plane_authentication",
                "host_secret_resolution",
                "worker_secret_binding",
                "session_credentials",
            ),
            selected_identity="nebius-object-storage-topology:" + "1" * 64,
        )
        journal.finish(credential, disposition="succeeded")
        for kind, operation, identifier in (
            ("bucket", "store.fresh_bucket", "bucket-one"),
            ("h100_job", "jobs.submit", "h100-one"),
        ):
            identity = dict(
                stage="synthetic",
                step="create",
                logical_operation=operation,
                resource_kind=kind,
                resource_intent=identifier,
            )
            invocation = journal.step(phase="invoked", **identity)
            journal.step(
                phase="verified",
                invocation_ref=invocation,
                provider_ids=(identifier,),
                ceiling_count=1,
                **identity,
            )
        binding = journal.finalize()
        if binding is None:
            raise AssertionError("healthy journal did not produce a binding")
        return journal.path.read_bytes(), binding

    def test_healthy_evidence_separates_physical_units_resources_and_unknown_network(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            content, binding = self._healthy_journal(Path(temporary))
            approved_file = self.prerequisite("approved").journey_file

            review = compile_execution_contact_evidence(
                journal_bytes=content,
                binding=binding,
                expected_execution_id=_EXECUTION_ID,
                journey_events=_journey_events(approved_file, binding),
                journey_binding=_JOURNEY_BINDING,
            )
            planning_contacts = _planning_model_contacts(approved_file)

        self.assertEqual(review.compiler_version, COMPILER_VERSION)
        self.assertTrue(review.complete)
        # E12 permits unknown physical contact totals for artifact/flow fixtures
        # when the resource ceilings are measured and the journal is complete.
        self.assertTrue(review.fixture_eligible)
        self.assertEqual(review.ledger.provider_contact.count, 3)
        self.assertEqual(review.ledger.storage_contact.count, 1)
        self.assertEqual(review.ledger.credential_access.count, 1)
        self.assertEqual(review.ledger.model_contact.count, planning_contacts)
        # The prelude's NCBI Tools ran on offline fakes that open no contact scope,
        # while its Galaxy index search's contacts are retained: the physical source
        # contacts are unknown, not zero, and their evidence incomplete.
        self.assertEqual(
            (
                review.ledger.source_contact.count,
                review.ledger.source_contact.reason,
            ),
            (None, "source_contact_evidence_incomplete"),
        )
        self.assertIsNone(review.ledger.network_contact.count)
        self.assertEqual(
            review.ledger.network_contact.reason,
            "cli_registry_contact_coverage_incomplete",
        )
        self.assertEqual(review.ledger.fresh_bucket.count, 1)
        self.assertEqual(review.ledger.cpu_preparation_create.count, 0)
        self.assertEqual(review.ledger.h100_create.count, 1)
        self.assertEqual(review.ledger.post_execution_vcf_query.count, 0)

    def test_historical_coarse_rows_remain_unclassified_on_review(self):
        # Synthetic historical shape, not a reconstructed provider history.
        with tempfile.TemporaryDirectory() as directory:
            journal = ExecutionContactJournal(
                Path(directory) / "journal.jsonl", _EXECUTION_ID
            )
            unit = journal.start(
                category="provider",
                operation="nebius.allowlisted-other",
                executable_identity="nebius-cli:" + "c" * 64,
            )
            journal.finish(unit, disposition="succeeded", exit_status=0)
            credential = journal.start(
                category="credential",
                operation="selected-object-storage-credential",
                executable_identity="credential-resolver",
                credential_names=(
                    "control_plane_authentication",
                    "host_secret_resolution",
                    "worker_secret_binding",
                    "session_credentials",
                ),
                selected_identity="nebius-object-storage-topology:" + "1" * 64,
            )
            journal.finish(
                credential,
                disposition="succeeded",
                parse_classification="selected_identity",
            )
            binding = journal.finalize()
            content = journal.path.read_bytes()
            review = compile_execution_contact_evidence(
                journal_bytes=content,
                binding=binding,
                expected_execution_id=_EXECUTION_ID,
                journey_events=_journey_events(
                    self.prerequisite("approved").journey_file, binding
                ),
                journey_binding=_JOURNEY_BINDING,
            )
            self.assertEqual(journal.path.read_bytes(), content)
        self.assertEqual(
            [unit.operation for unit in review.units],
            ["nebius.allowlisted-other", "selected-object-storage-credential"],
        )
        self.assertEqual(
            [unit.parse_classification for unit in review.units],
            [None, "selected_identity"],
        )

    def test_current_path_without_journal_does_not_invent_contacts_or_resources(
        self,
    ) -> None:
        review = compile_execution_contact_evidence(
            journal_bytes=b"",
            binding=None,
            expected_execution_id=_EXECUTION_ID,
            journey_events=_journey_events(self.prerequisite("approved").journey_file),
            journey_binding=_JOURNEY_BINDING,
        )

        self.assertFalse(review.complete)
        self.assertEqual(review.ledger.provider_contact.status, "unmeasurable")
        self.assertIsNone(review.ledger.provider_contact.count)
        self.assertIsNone(review.ledger.fresh_bucket.count)
        self.assertIsNone(review.ledger.h100_create.count)
        self.assertFalse(review.fixture_eligible)

    def test_crash_prefix_preserves_unmatched_start_and_truncated_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "journal.jsonl"
            journal = ExecutionContactJournal(path, _EXECUTION_ID)
            journal.start(
                category="provider",
                operation="ai.job.get",
                executable_identity="nebius-cli:" + "d" * 64,
            )
            prefix = path.read_bytes()
        content = prefix + b'{"schema_version":"truncated"'
        binding = ExecutionContactJournalBinding(
            locator=str(path.resolve()),
            byte_count=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            execution_id=_EXECUTION_ID,
            compiler_version=COMPILER_VERSION,
        )

        review = compile_execution_contact_evidence(
            journal_bytes=content,
            binding=binding,
            expected_execution_id=_EXECUTION_ID,
            journey_events=_journey_events(
                self.prerequisite("approved").journey_file, binding
            ),
            journey_binding=_JOURNEY_BINDING,
        )

        self.assertFalse(review.complete)
        self.assertTrue(review.truncated_tail)
        self.assertEqual(len(review.unmatched_unit_ids), 1)
        self.assertEqual(review.longest_valid_prefix_byte_count, len(prefix))
        self.assertEqual(review.ledger.provider_contact.status, "unmeasurable")
        self.assertEqual(review.ledger.post_execution_vcf_query.status, "unmeasurable")

    def test_supported_prerequisites_retain_exact_secret_free_operation_names(self):
        # Actual CLI/writer connection; only the process and executable are synthetic.
        operations = (
            (("iam", "project", "get"), "iam.project.get"),
            (("iam", "project", "list"), "iam.project.list"),
            (("iam", "tenant", "list"), "iam.tenant.list"),
            (("iam", "v2", "access-key", "list"), "iam.v2.access-key.list"),
            (
                ("iam", "group-membership", "list-member-of"),
                "iam.group-membership.list-member-of",
            ),
            (
                ("iam", "group-membership", "list-members"),
                "iam.group-membership.list-members",
            ),
            (("iam", "access-permit", "list"), "iam.access-permit.list"),
            (("mysterybox", "payload", "get"), "mysterybox.payload.get"),
            (("iam", "v2", "access-key", "get"), "iam.v2.access-key.get"),
            (("iam", "group-membership", "get"), "iam.group-membership.get"),
            (("iam", "service-account", "get"), "iam.service-account.get"),
            (("iam", "access-permit", "get"), "iam.access-permit.get"),
            (("iam", "group", "get"), "iam.group.get"),
            (("iam", "whoami"), "iam.whoami"),
            (("mysterybox", "secret", "get"), "mysterybox.secret.get"),
            (("mysterybox", "secret-version", "get"), "mysterybox.secret-version.get"),
            (
                ("mysterybox", "secret-version", "payload", "get"),
                "mysterybox.secret-version.payload.get",
            ),
            (("iam", "unsupported-private-command"), "nebius.allowlisted-other"),
        )
        identity = NebiusCliInstallIdentity(
            executable="/opt/nebius",
            resolved_executable="/opt/nebius",
            version="0.12.263",
            byte_count=10,
            sha256="e" * 64,
        )

        def process(argv, **kwargs):
            stdout = (
                identity.version
                if argv[-1] == "version"
                else ('{"data":[{"string_value":"private-payload"}]}')
            )
            return subprocess.CompletedProcess(argv, 0, stdout, "")

        with tempfile.TemporaryDirectory() as directory:
            journal = ExecutionContactJournal(
                Path(directory) / "journal.jsonl", _EXECUTION_ID
            )
            runner = NebiusCliRunner(
                expected_identity=identity,
                process_runner=process,
                executable_inspector=lambda _: identity,
                execution_contact_observer=journal,
            )
            for arguments, _ in operations:
                runner.json(
                    *arguments,
                    "--id",
                    "private-selector",
                    "--profile",
                    "private-profile",
                )
            binding = journal.finalize()
            self.assertIsNotNone(binding)
            content = journal.path.read_bytes()
            rows = [
                ExecutionContactJournalRow.model_validate_json(line)
                for line in content.splitlines()
            ]
            starts = [row for row in rows if row.row_kind == "start"]
            self.assertEqual(
                [row.operation for row in starts],
                ["nebius.version", *(label for _, label in operations)],
            )
            for secret in (
                b"private-payload",
                b"private-selector",
                b"private-profile",
                b"unsupported-private-command",
            ):
                self.assertNotIn(secret, content)

    def test_nebius_logger_degradation_does_not_change_adapter_result(self) -> None:
        identity = NebiusCliInstallIdentity(
            executable="/opt/nebius",
            resolved_executable="/opt/nebius",
            version="0.12.263",
            byte_count=10,
            sha256="e" * 64,
        )

        def process(
            argv: tuple[str, ...], **_: object
        ) -> subprocess.CompletedProcess[str]:
            stdout = "0.12.263\n" if argv[-1] == "version" else '{"ok":true}'
            return subprocess.CompletedProcess(argv, 0, stdout, "")

        runner = NebiusCliRunner(
            expected_identity=identity,
            process_runner=process,
            executable_inspector=lambda _: SimpleNamespace(
                resolved_executable="/opt/nebius", byte_count=10, sha256="e" * 64
            ),
            execution_contact_observer=_ExplodingObserver(),
        )

        self.assertEqual(runner.json("iam", "whoami"), {"ok": True})

    def test_zero_exit_json_unit_succeeds_while_application_parse_is_pending(
        self,
    ) -> None:
        observer = _CollectingObserver()
        identity = NebiusCliInstallIdentity(
            executable="/opt/nebius",
            resolved_executable="/opt/nebius",
            version="0.12.263",
            byte_count=10,
            sha256="e" * 64,
        )

        def process(
            argv: tuple[str, ...], **_: object
        ) -> subprocess.CompletedProcess[str]:
            stdout = "0.12.263\n" if argv[-1] == "version" else "not-json"
            return subprocess.CompletedProcess(argv, 0, stdout, "")

        runner = NebiusCliRunner(
            expected_identity=identity,
            process_runner=process,
            executable_inspector=lambda _: SimpleNamespace(
                resolved_executable="/opt/nebius", byte_count=10, sha256="e" * 64
            ),
            execution_contact_observer=observer,
        )

        with self.assertRaises(NebiusCliError) as raised:
            runner.json("iam", "whoami")

        self.assertEqual(raised.exception.code, "provider_json_invalid")
        self.assertEqual(observer.finished[-1][1]["disposition"], "succeeded")
        self.assertEqual(
            observer.finished[-1][1]["parse_classification"],
            "json_parse_pending",
        )

    def test_each_explicit_s3_attempt_is_one_observed_unit(self) -> None:
        observer = _CollectingObserver()

        class Client:
            attempts = 0

            def head_object(self, **_: object) -> dict[str, object]:
                self.attempts += 1
                if self.attempts == 1:
                    raise OSError("offline transient")
                return {
                    "ContentLength": 3,
                    "Metadata": {"checksum-sha256": "f" * 64},
                }

        client = Client()
        mechanics = NebiusObjectStorageMechanics(
            endpoint_url="https://storage.example",
            region="test",
            bucket="bucket",
            credential_resolver=lambda: AwsCredentialMaterial(
                access_key_id="offline-access",
                secret_access_key="offline-secret",
                credentials_profile="offline-profile",
            ),
            client_factory=lambda *_: client,
            execution_contact_observer=observer,
        )

        result = mechanics.head("object")

        self.assertEqual(result.value.sha256, "f" * 64)
        self.assertEqual(len(observer.started), 2)
        self.assertEqual(len(observer.finished), 2)
        self.assertEqual({row["category"] for row in observer.started}, {"storage"})
        self.assertEqual(
            [row[1]["disposition"] for row in observer.finished],
            ["failed", "succeeded"],
        )
        self.assertEqual(
            [row["operation"] for row in observer.started],
            ["s3.head-object", "s3.head-object"],
        )

    def test_server_side_copy_requests_are_normalized_storage_units(self) -> None:
        """The issue 31 copy operations journal under their own names, not "other"."""

        with tempfile.TemporaryDirectory() as directory:
            journal = ExecutionContactJournal(
                Path(directory) / "journal.jsonl", _EXECUTION_ID
            )
            for operation in ("s3.copy-object", "s3.upload-part-copy"):
                unit_id = journal.start(
                    category="storage",
                    operation=operation,
                    executable_identity="s3-sdk",
                )
                journal.finish(unit_id, disposition="succeeded", exit_status=0)
            rows = [
                json.loads(line)
                for line in (Path(directory) / "journal.jsonl").read_text().splitlines()
            ]
        self.assertEqual(
            [row["operation"] for row in rows if row["row_kind"] == "start"],
            ["s3.copy-object", "s3.upload-part-copy"],
        )

    def test_journal_rows_are_canonical_and_exclude_unsafe_material(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            content, _ = self._healthy_journal(Path(temporary))
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines()]

        self.assertTrue(
            all(canonical_json(row) in content.decode("utf-8") for row in rows)
        )
        forbidden = {
            "stdout",
            "stderr",
            "headers",
            "payload",
            "exception",
            "credential_value",
        }
        self.assertFalse(forbidden & set().union(*(row.keys() for row in rows)))

    def test_journal_semantic_allowlists_refuse_secret_shaped_safe_fields(
        self,
    ) -> None:
        provider = {
            "sequence": 1,
            "execution_id": _EXECUTION_ID,
            "row_kind": "start",
            "unit_id": f"execution-contact:{'a' * 64}:00000001",
            "category": "provider",
            "operation": "iam.whoami",
            "executable_identity": "nebius-cli:" + "c" * 64,
            "disposition": "attempted",
        }
        cases = (
            {**provider, "operation": "iam.whoami secret=value"},
            {**provider, "executable_identity": "nebius-cli:secret-value"},
            {
                **provider,
                "category": "storage",
                "operation": "head object-key=secret",
                "executable_identity": "s3-sdk",
            },
            {
                **provider,
                "row_kind": "terminal",
                "disposition": "failed",
                "parse_classification": "exception: secret=value",
            },
            {
                **provider,
                "category": "credential",
                "operation": "selected-object-storage-credential",
                "executable_identity": "credential-resolver",
                "credential_names": ("secret_value",),
                "selected_identity": ("nebius-object-storage-topology:" + "1" * 64),
            },
            {
                **provider,
                "category": "credential",
                "operation": "selected-object-storage-credential",
                "executable_identity": "credential-resolver",
                "credential_names": (
                    "control_plane_authentication",
                    "host_secret_resolution",
                    "worker_secret_binding",
                    "session_credentials",
                ),
                "selected_identity": "secret-identity-value",
            },
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ExecutionContactJournalRow(**values)

    def test_current_completion_binds_one_outcome_and_review_is_read_only(self):
        from bio_neb.journey_path import JourneyPath
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            command_file = str(
                self.prerequisite("executed", writable=True).journey_file
            )
            journey = JourneyPath.open(Path(command_file))
            outcomes = [
                event
                for event in journey.read()
                if event.event_type == "ExecutionOutcome"
            ]
            self.assertEqual(len(outcomes), 1)
            payload = outcomes[0].payload
            journal_bytes = Path(payload.contact_journal).read_bytes()
            self.assertEqual(
                payload.journal_sha256, hashlib.sha256(journal_bytes).hexdigest()
            )
            self.assertEqual(payload.journal_bytes, len(journal_bytes))
            retained = {
                path: path.read_bytes() for path in root.rglob("*") if path.is_file()
            }
            with (
                patch("socket.socket", side_effect=AssertionError("network contact")),
                patch.object(
                    subprocess, "Popen", side_effect=AssertionError("process contact")
                ),
            ):
                first = review_execution_contacts(command_file)
                second = review_execution_contacts(command_file)
                output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(io.StringIO()):
                    result = main(
                        [
                            "execution-contact-review",
                            "--execution-journey-file",
                            command_file,
                        ]
                    )
            self.assertEqual(first, second)
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "complete")
            self.assertTrue(first.journal_artifact_equal)
            self.assertTrue(first.projection_equal)
            self.assertEqual(
                {path: path.read_bytes() for path in root.rglob("*") if path.is_file()},
                retained,
            )

    def test_second_pass_journal_continues_unit_ordinals_and_composes_one_ledger(self):
        from collections import deque

        from bio_neb.failure import ProviderError
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )
        from bio_neb.parabricks.operation import execute_parabricks
        from bio_neb.parabricks.review import answer_review
        from tests.h100_stage_fakes import h100_provider
        from tests.parabricks_operation_fakes import input_storage_provider
        from tests.reference_stage_fakes import seed_cpu

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            approved = self.prerequisite("approved").value
            journals = []

            def provider(plan, journal):
                # Two supplied physical observations prove ordinal composition,
                # not actual provider contact or complete network coverage.
                unit = journal.start(
                    category="provider",
                    operation="iam.whoami",
                    executable_identity="nebius-cli:" + "c" * 64,
                )
                journal.finish(unit, disposition="succeeded", exit_status=0)
                journals.append(journal)
                scripted = seed_cpu(
                    input_storage_provider(plan, journal), plan, approved
                )
                if len(journals) == 1:
                    # The first Pass stops at the CPU stage's failure Review before
                    # any job contact: a Run without a bound prebuilt index prepares
                    # on CPU with no Review.
                    scripted.scripts["cpu_preparation", "preflight"] = deque(
                        [ProviderError("rejected", "identity_refused")]
                    )
                return h100_provider(scripted, plan, approved)

            state = execute_parabricks(approved, infrastructure_factory=provider)
            self.assertEqual(state.status, "execution_review_pending")
            self.assertEqual(state.pending_review.stage, "cpu_preparation")
            first = review_execution_contacts(approved.journey_file)
            self.assertEqual(first.status, "complete")
            self.assertEqual(first.review.ledger.prior_pass_journal_bindings, ())
            first_bytes = journals[0].path.read_bytes()
            answer_review(
                approved.journey_file,
                state.request_event_id,
                {"action": "retry"},
            )
            state = execute_parabricks(approved, infrastructure_factory=provider)
            self.assertEqual(state.disposition, "succeeded")
            review = review_execution_contacts(approved.journey_file)
            self.assertEqual(review.status, "complete")
            self.assertTrue(review.review.fixture_eligible)
            self.assertEqual(
                journals[1].path.parent.parent, journals[0].path.parent.parent
            )
            self.assertNotEqual(journals[1].path.parent, journals[0].path.parent)
            self.assertEqual(review.journal_locator, journals[1].binding().locator)
            self.assertEqual(
                [
                    item.locator
                    for item in review.review.ledger.prior_pass_journal_bindings
                ],
                [journals[0].binding().locator],
            )
            self.assertEqual(
                [unit.unit_id.rsplit(":", 1)[-1] for unit in review.review.units],
                ["00000001", "00000002"],
            )
            self.assertEqual(
                [item.execution_pass for item in review.review.retained_passes], [1, 2]
            )
            self.assertEqual(review.review.ledger.fresh_bucket.count, 1)
            self.assertEqual(review.review.ledger.cpu_preparation_create.count, 1)
            self.assertEqual(review.review.ledger.h100_create.count, 1)
            self.assertEqual(journals[0].path.read_bytes(), first_bytes)
            for journal in journals:
                with self.assertRaisesRegex(ValueError, "not writable"):
                    journal.start(
                        category="provider",
                        operation="iam.whoami",
                        executable_identity="synthetic",
                    )
            before = {
                path: path.read_bytes() for path in root.rglob("*") if path.is_file()
            }
            reopened = execute_parabricks(
                approved,
                infrastructure_factory=lambda *_: self.fail(
                    "completed execution contacted provider"
                ),
            )
            self.assertEqual(reopened, state)
            self.assertEqual(review_execution_contacts(approved.journey_file), review)
            self.assertEqual(
                {path: path.read_bytes() for path in root.rglob("*") if path.is_file()},
                before,
            )

    def test_accounting_degradation_at_closure_does_not_override_success(self):
        from bio_neb.parabricks.execution_contact_accounting import (
            review_execution_contacts,
        )
        from bio_neb.parabricks.operation import execute_parabricks
        from tests.h100_stage_fakes import h100_provider
        from tests.parabricks_operation_fakes import input_storage_provider
        from tests.reference_stage_fakes import seed_cpu

        approved = self.prerequisite("approved").value

        def provider(plan, journal):
            return h100_provider(
                seed_cpu(input_storage_provider(plan, journal), plan, approved),
                plan,
                approved,
            )

        finalize = ExecutionContactJournal.finalize

        def degraded_closure(journal):
            journal.finish(None, disposition="unknown")
            return finalize(journal)

        with patch.object(ExecutionContactJournal, "finalize", degraded_closure):
            state = execute_parabricks(approved, infrastructure_factory=provider)
        self.assertEqual(state.disposition, "succeeded")
        review = review_execution_contacts(approved.journey_file)
        self.assertEqual(review.status, "incomplete")
        self.assertTrue(review.journal_artifact_equal)
        self.assertTrue(review.projection_equal)
        self.assertFalse(review.review.fixture_eligible)
        self.assertEqual(review.review.ledger.provider_contact.status, "unmeasurable")
        self.assertEqual(
            review.review.ledger.post_execution_vcf_query.status, "unmeasurable"
        )
        self.assertFalse(review.review.retained_passes[0].journal_complete)


if __name__ == "__main__":
    unittest.main()
