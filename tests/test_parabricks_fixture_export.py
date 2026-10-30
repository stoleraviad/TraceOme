from __future__ import annotations

import hashlib
import io
import json
import socket
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from bio_neb.devtools import parabricks_fixture_export as fixture_export
from bio_neb.journey import Journey, canonical_json
from tests.parabricks_run_fixture import (
    ACTIVE_PARABRICKS_FIXTURE_BUNDLE_BYTE_COUNT,
    ACTIVE_PARABRICKS_FIXTURE_BUNDLE_PATH,
    ACTIVE_PARABRICKS_FIXTURE_BUNDLE_SHA256,
    ACTIVE_PARABRICKS_FIXTURE_BUNDLE_SOURCE_TICKETS,
    get_active_parabricks_fixture_scenario,
    load_active_parabricks_fixture_bundle,
)
from tests.run_prerequisites import RunPrerequisites, State


def assert_refused(
    test: unittest.TestCase,
    refused: fixture_export.FixtureExportRefused,
    reason: str,
    cause: str | type[BaseException] | None = None,
) -> None:
    """The refusal's reason and, where several causes share it, which one fired."""
    test.assertEqual(refused.reason, reason)
    if isinstance(cause, type):
        test.assertIsInstance(refused.__cause__, cause)
    elif cause is not None:
        test.assertIn(cause, str(refused.__cause__))


SHA = "a" * 64
SYNTHETIC_SOURCE_SHA256 = "9" * 64


def _registered(root):
    """A Run registered but not yet planned."""
    from tests.planning_assignment_fakes import configured_planning_run

    registered, _, _ = configured_planning_run(root)
    return Path(registered.journey_file), None


_REGISTERED = State("registered", _registered)


def _with_local_reference_replica(configuration):
    """Select a local replica of the reviewed reference, so input storage places it.

    The historical milestone shape holds a placed reference source; without a
    supplied FASTA or a bound prebuilt index the CPU job fetches it instead
    (owner, 2026-10-29, #464). A path only: the scripted upload reads no bytes.
    """
    from bio_neb.parabricks.configuration import ParabricksConfiguration

    material = configuration.model_dump(mode="json")
    launch = material["launch"]
    launch["local_reference_source"] = {
        "path": str(Path(launch["work_directory"]) / "reference.gz")
    }
    return ParabricksConfiguration.model_validate(material)


def input_only_milestone(approved, provider_factory):
    """Exercise the input stage and retain the historical milestone fixture shape."""
    from bio_neb.journey_path import (
        ExecutionBound,
        JourneyPath,
        Outcome,
        PathFact,
        project_execution,
    )
    from bio_neb.parabricks.execution_contact_accounting import ExecutionContactJournal
    from bio_neb.parabricks.input_storage import INPUT_STORAGE
    from bio_neb.parabricks.plan import RetainedExecution, plan_execution
    from bio_neb.parabricks.runner import run_stage

    journey = JourneyPath.open(Path(approved.journey_file))
    journey.path(
        journey.state,
        PathFact(
            event_type="ExecutionBound",
            kind="authority",
            grounding="human",
            **{"pass": 1},
            payload=ExecutionBound(
                execution_id=approved.execution_id,
                authority_id=approved.authority.authority_id,
                proposal_id=approved.proposal.proposal_id,
                run_id=approved.proposal.journey_id,
                approved_input=approved.model_dump(mode="json"),
            ),
        ),
    )
    state = project_execution(journey.read())
    plan = plan_execution(
        approved, RetainedExecution(state, approved.authority, approved.proposal)
    )
    file = journey.file.parent / "input-contacts.jsonl"
    journal = ExecutionContactJournal(file, approved.execution_id)
    state = run_stage(
        state,
        plan,
        INPUT_STORAGE,
        provider_factory(plan, journal),
        journey=journey,
        journal=journal,
        live=None,
    )
    journal.finalize()
    journey.path(
        journey.state,
        PathFact(
            event_type="ExecutionOutcome",
            kind="outcome",
            grounding="verified",
            **{"pass": 1},
            rests_on=(state.latest_event_ids["InputsVerified"],),
            payload=Outcome(
                disposition="incomplete",
                milestone="input_storage",
                contact_journal=str(file),
            ),
        ),
    )
    return project_execution(journey.read())


class InputStorageFixtureExportTests(RunPrerequisites, unittest.TestCase):
    """Synthetic transfers exercise export integrity, not live storage proof."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prepare_input_milestone(self):
        """Approve the default Run and stop it after verified input storage."""
        from tests.parabricks_operation_fakes import (
            input_storage_provider,
        )

        self.approved = self.prerequisite(
            "approved", configure=_with_local_reference_replica
        ).value
        self.state = input_only_milestone(self.approved, input_storage_provider)
        self.journey_file = Path(self.approved.journey_file)

    def compile(self):
        return fixture_export.compile_input_storage_fixture(
            self.journey_file, source_commit="a" * 40
        )

    def test_export_preserves_verified_inputs_and_declared_milestone(self):
        self.prepare_input_milestone()
        fixture = json.loads(self.compile())
        self.assertEqual(fixture["milestone"], "input_storage")
        self.assertEqual(fixture["disposition"], "incomplete")
        self.assertEqual(
            fixture["path_events"],
            ["ExecutionBound", "InputsVerified", "ExecutionOutcome"],
        )
        self.assertEqual(
            [item["role"] for item in fixture["inputs"]],
            ["read_1", "read_2", "reference_source", "input_manifest"],
        )
        self.assertEqual(len(fixture["verification_receipts"]), 4)
        self.assertEqual(fixture["execution_id"], self.state.bound.execution_id)
        self.assertEqual(len(fixture["manifest"]["inputs"]), 3)
        self.assertNotIn("approved_input", fixture)

    def test_incomplete_journal_is_not_exportable(self):
        self.prepare_input_milestone()
        journal = Path(self.state.outcome.contact_journal)
        journal.write_bytes(
            b"".join(journal.read_bytes().splitlines(keepends=True)[:-1])
        )
        with self.assertRaises(fixture_export.FixtureExportRefused) as refused:
            self.compile()
        assert_refused(
            self,
            refused.exception,
            "input_storage_evidence_invalid",
            "execution_contact_journal_incomplete_or_inconsistent",
        )

    def test_changed_verification_receipt_is_not_exportable(self):
        self.prepare_input_milestone()
        journal = Path(self.state.outcome.contact_journal)
        row = json.loads(
            journal.read_text().splitlines()[
                int(self.state.inputs.pointers[0].proof_ref.rsplit(":", 1)[1]) - 1
            ]
        )
        Path(row["receipt"]["ref"]).write_text("{}")
        with self.assertRaises(fixture_export.FixtureExportRefused) as refused:
            self.compile()
        assert_refused(
            self,
            refused.exception,
            "input_storage_evidence_invalid",
            "input receipt identity changed",
        )

    def test_matching_bytes_at_another_object_do_not_verify_the_input_location(self):
        from collections import deque

        from tests.parabricks_operation_fakes import (
            input_storage_provider,
        )

        def wrong_object(plan, journal):
            provider = input_storage_provider(plan, journal)
            key = ("input_storage", "place_inputs")
            provider.scripts[key] = deque(
                receipt.model_copy(
                    update={"provider_ids": ("s3://another-bucket/key",)}
                )
                for receipt in provider.scripts[key]
            )
            return provider

        approved = self.prerequisite(
            "approved",
            slot="different-object",
            configure=_with_local_reference_replica,
        ).value
        input_only_milestone(approved, wrong_object)
        with self.assertRaises(fixture_export.FixtureExportRefused) as refused:
            fixture_export.compile_input_storage_fixture(
                Path(approved.journey_file), source_commit="a" * 40
            )
        assert_refused(
            self,
            refused.exception,
            "input_storage_evidence_invalid",
            "input verification receipt differs",
        )

    def test_journey_change_after_validation_is_not_exportable(self):
        from bio_neb.journey_path import JourneyPath

        self.prepare_input_milestone()
        original = JourneyPath.open

        def append_after_read(path):
            journey = original(path)
            with path.open("ab") as stream:
                stream.write(b"{}\n")
            return journey

        with (
            patch.object(JourneyPath, "open", side_effect=append_after_read),
            self.assertRaises(fixture_export.FixtureExportRefused) as refused,
        ):
            self.compile()
        assert_refused(
            self,
            refused.exception,
            "input_storage_evidence_invalid",
            "Journey changed after validation",
        )

    def test_dev_command_publishes_exclusively_without_provider_contact(self):
        self.prepare_input_milestone()
        output = self.root / "input-storage.json"
        argv = [
            "export-input-storage",
            "--journey-file",
            str(self.journey_file),
            "--source-commit",
            "a" * 40,
            "--output",
            str(output),
        ]
        with (
            redirect_stdout(io.StringIO()),
            patch("socket.socket", side_effect=AssertionError("network contact")),
        ):
            self.assertEqual(fixture_export.main(argv), 0)
            self.assertEqual(
                json.loads(output.read_bytes())["milestone"], "input_storage"
            )
            before = b"retained different fixture"
            output.write_bytes(before)
            with self.assertRaises(fixture_export.FixtureExportRefused) as refused:
                fixture_export.main(argv)
            assert_refused(self, refused.exception, "output_conflict")
            self.assertEqual(output.read_bytes(), before)


class PreExecutionFixtureExportTests(RunPrerequisites, unittest.TestCase):
    """A scripted Run at the Execution review tests export integrity, never live proof."""

    def setUp(self):
        baseline = self.prerequisite("review-pending")
        self.journey_file, self.review = baseline.journey_file, baseline.value
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def _registered_journey_file(self):
        """A restored Run registered but not yet planned."""
        return self.prerequisite(_REGISTERED).journey_file

    def _compile(self, journey_file):
        return fixture_export.compile_pre_execution_fixture(
            journey_file,
            source_commit="c" * 40,
            source_ticket=150,
            evidence_kind="synthetic",
        )

    def test_export_keeps_the_path_to_the_exact_unanswered_execution_review(self):
        content = self._compile(self.journey_file)

        fixture = fixture_export.load_pre_execution_fixture(content)
        (scenario,) = fixture.scenarios
        journey_bytes = self.journey_file.read_bytes()
        self.assertFalse(fixture.fixture_grants_live_authority)
        self.assertEqual(
            (scenario.journey.byte_count, scenario.journey.sha256),
            (len(journey_bytes), hashlib.sha256(journey_bytes).hexdigest()),
        )
        self.assertEqual(scenario.status, "execution_hitl_pending")
        self.assertEqual(
            scenario.execution_review_event_id,
            self.review.current_review.request.request_event_id,
        )
        self.assertEqual(
            scenario.execution_proposal["proposal_id"],
            self.review.current_review.request.proposal.proposal_id,
        )
        self.assertEqual(scenario.path_events[0], "JourneyStarted")
        self.assertEqual(
            scenario.path_events[-2:], ("ExecutionProposalSubmitted", "PassClosed")
        )
        self.assertNotIn("ExecutionHitlResponded", scenario.path_events)
        # The scripted researcher selected the first candidate of every question.
        self.assertTrue(scenario.planning_reviews)
        for review in scenario.planning_reviews:
            self.assertEqual(review.response["action"], "select")
            self.assertEqual(
                review.response["candidate_id"], review.candidates[0].candidate_id
            )
            self.assertFalse(review.authorizes_execution)
        self.assertTrue(scenario.model_turns)
        self.assertTrue(scenario.tool_calls)
        self.assertTrue(scenario.contacts.complete)
        # No eligibility verdict: the Contact review covers execution paths only.
        self.assertEqual(scenario.fixture_eligibility, "not_computed")
        self.assertNotIn("fixture_eligible", json.loads(content)["scenarios"][0])

    def test_export_keeps_the_scripted_turns_and_their_measured_accounting(self):
        (scenario,) = fixture_export.load_pre_execution_fixture(
            self._compile(self.journey_file)
        ).scenarios

        # The scripted planning model's calls, one per response, in its own order.
        scripted = [
            "search_journey",
            "record_proposal_requirements",
            "attest_sample",
            "resolve_taxon",
            "discover_assemblies",
            "record_reference_choice",
            "inspect_reference",
            "describe_companion",
            # The genomic FASTA's index: Galaxy lists none, so none is chosen.
            "search_prebuilt_indexes",
            "record_prebuilt_index_choice",
            "list_compatible_analysis_recipes",
            "request_recipe_eligibility_assertion",
            "list_compatible_analysis_recipes",
            "select_recipe_candidate",
            "lookup_compatible_resource_profiles",
            "select_resource_candidate",
        ]
        self.assertEqual(
            [call.tool_name for turn in scenario.model_turns for call in turn.calls],
            scripted,
        )
        self.assertEqual([call.tool_name for call in scenario.tool_calls], scripted)
        self.assertEqual(
            {call.disposition for call in scenario.tool_calls}, {"singleton_admitted"}
        )
        # Scripted NCBI adapters make no source GET; the Galaxy index search's two
        # catalogue GETs are its own; every model request and Tool is one row.
        self.assertEqual(
            scenario.contacts.operations,
            {"model.request": 16, "source.get": 2, "tool.execute": 16},
        )
        self.assertEqual(
            (
                scenario.contacts.model_logical_requests,
                scenario.contacts.model_physical_contacts,
            ),
            (16, 16),
        )
        self.assertEqual(
            [
                (review.subject_key, review.authority)
                for review in scenario.planning_reviews
            ],
            [("analysis_recipe", "user_assertion")],
        )

    def _refused_with_observed_passes(self, edit):
        """Controlled observation input: the real view with its passes edited."""
        from bio_neb.application import run_interaction as interaction

        observed = interaction.run_interaction

        def controlled(review):
            view = observed(review)
            return view.model_copy(update={"passes": edit(view.passes)})

        output = self.root / f"refused-{len(list(self.root.iterdir()))}.json"
        argv = [
            "export-pre-execution",
            "--journey-file",
            str(self.journey_file),
            "--source-commit",
            "c" * 40,
            "--source-ticket",
            "150",
            "--evidence-kind",
            "synthetic",
            "--output",
            str(output),
        ]
        with (
            patch.object(interaction, "run_interaction", controlled),
            patch("socket.socket", side_effect=AssertionError("network contact")),
            self.assertRaises(fixture_export.FixtureExportRefused) as refused,
        ):
            fixture_export.main(argv)
        self.assertEqual(refused.exception.reason, "pre_execution_evidence_invalid")
        self.assertFalse(output.exists())

    def test_incomplete_or_failed_pass_accounting_is_not_exportable(self):
        for update in (
            {"journal_complete": False},
            {"accounting_error": "truncated_prefix"},
            {"disposition": "active"},
        ):
            with self.subTest(update=update):
                self._refused_with_observed_passes(
                    lambda passes, update=update: (
                        passes[0].model_copy(update=update),
                        *passes[1:],
                    )
                )

    def test_contact_outside_planning_operations_is_not_exportable(self):
        def storage_contact(passes):
            first, *rest = passes
            rows = list(first.contacts)
            index = next(
                index
                for index, row in enumerate(rows)
                if row.kind == "start" and row.operation == "model.request"
            )
            rows[index] = rows[index].model_copy(
                update={"operation": "storage.put_object"}
            )
            return (first.model_copy(update={"contacts": tuple(rows)}), *rest)

        self._refused_with_observed_passes(storage_contact)

    def test_export_is_byte_stable(self):
        self.assertEqual(
            self._compile(self.journey_file), self._compile(self.journey_file)
        )

    def test_a_run_short_of_the_execution_review_is_not_exportable(self):
        with self.assertRaises(fixture_export.FixtureExportRefused) as refused:
            self._compile(self._registered_journey_file())

        self.assertEqual(refused.exception.reason, "pre_execution_evidence_invalid")

    def test_a_journey_whose_bytes_no_longer_validate_is_not_exportable(self):
        import shutil

        copy = self.root / "tampered"
        shutil.copytree(self.journey_file.parent, copy)
        journal = copy / "journey.jsonl"
        lines = journal.read_bytes().splitlines(keepends=True)
        journal.write_bytes(b"".join(lines[:-1]) + lines[-1].replace(b"a", b"b", 1))

        with self.assertRaises(fixture_export.FixtureExportRefused) as refused:
            self._compile(journal)

        self.assertEqual(refused.exception.reason, "pre_execution_evidence_invalid")

    def test_dev_command_publishes_exclusively_without_provider_contact(self):
        output = self.root / "pre-execution.json"
        argv = [
            "export-pre-execution",
            "--journey-file",
            str(self.journey_file),
            "--source-commit",
            "c" * 40,
            "--source-ticket",
            "150",
            "--evidence-kind",
            "synthetic",
            "--output",
            str(output),
        ]
        with (
            redirect_stdout(io.StringIO()),
            patch("socket.socket", side_effect=AssertionError("network contact")),
        ):
            self.assertEqual(fixture_export.main(argv), 0)
            self.assertEqual(output.read_bytes(), self._compile(self.journey_file))
            before = b"retained different fixture"
            output.write_bytes(before)
            with self.assertRaises(fixture_export.FixtureExportRefused) as refused:
                fixture_export.main(argv)
            assert_refused(self, refused.exception, "output_conflict")
            self.assertEqual(output.read_bytes(), before)


def _object(role: str, key: str, *, sha256: str = SHA) -> dict[str, object]:
    return {
        "role": role,
        "object_key": key,
        "byte_count": 17,
        "sha256": sha256,
        "metadata_sha256": "b" * 64,
    }


def _inventory(pre_manifest: tuple[str, ...], manifest: str) -> dict[str, object]:
    return {
        "pre_manifest_keys": sorted(pre_manifest),
        "manifest_key": manifest,
        "final_keys": sorted((*pre_manifest, manifest)),
        "manifest_published_last": True,
        "multipart_upload_count": 0,
    }


def _facts() -> dict[str, object]:
    input_objects = (
        _object(
            "reference_source",
            "inputs/reference.fa.gz",
            sha256=SYNTHETIC_SOURCE_SHA256,
        ),
        _object("read_1", "inputs/read-1.fastq.gz"),
        _object("read_2", "inputs/read-2.fastq.gz"),
    )
    prepared_roles = (
        "genomic_fasta",
        "fasta_fai",
        "reference_dict",
        "bwa_amb",
        "bwa_ann",
        "bwa_bwt",
        "bwa_pac",
        "bwa_sa",
    )
    prepared = tuple(_object(role, f"prepared/{role}.data") for role in prepared_roles)
    prepared_manifest = _object("prepared_reference_manifest", "prepared/manifest.json")
    prepared_worker_receipt = _object("worker_receipt", "prepared/worker-receipt.json")
    outputs = (
        _object("aligned_bam", "execution/aligned.bam"),
        _object("VCF", "execution/variants.vcf"),
        _object("worker_receipt", "execution/worker-receipt.json"),
        _object("execution_manifest", "execution/manifest.json"),
    )
    return {
        "request_id": "research-request:exact",
        "proposal_id": "execution-proposal:exact",
        "authority_id": "execution-authority:exact",
        "packet_binding_id": "prepared-execution-packet:exact",
        "reference_asset_id": "reference-asset:synthetic",
        "assembly_accession": "GCF_000000000.0",
        "source_derivation_rule": "decompress_gzip",
        "input_objects": input_objects,
        "prepared_objects": prepared,
        "prepared_worker_receipt": prepared_worker_receipt,
        "prepared_manifest": prepared_manifest,
        "output_objects": outputs,
        "mutation_counts": {
            "bucket": {"claimed": 1, "contacted": 1, "created": 1},
            "cpu": {"claimed": 0, "contacted": 0, "created": 0},
            "h100": {"claimed": 1, "contacted": 1, "created": 1},
        },
        "provider": {
            "bucket_create_disposition": "created",
            "cpu_create_disposition": "not_claimed",
            "h100_create_disposition": "created",
            "h100_job_id": "aijob-exact",
            "requested_envelope_sha256": "c" * 64,
            "fetched_envelope_sha256": "c" * 64,
            "envelopes_match": True,
            "provider_states": ["RUNNING", "COMPLETED"],
            "timestamp_facts_sha256": "d" * 64,
            "detail_facts_sha256": "e" * 64,
            "bounded_log_sha256": "f" * 64,
        },
        "worker": {
            "request_schema": "bio-neb.parabricks-worker-request/7",
            "receipt_schema": "bio-neb.parabricks-worker-receipt/3",
            "request_sha256": "1" * 64,
            "worker_sha256": "2" * 64,
            "image_child_digest": "sha256:" + "3" * 64,
            "runtime_identities": [
                {
                    "name": "python",
                    "path": "/opt/bio-neb/venv/bin/python",
                    "version": "3.10",
                },
                {
                    "name": "pbrun",
                    "path": "/usr/local/parabricks/pbrun",
                    "version": "4.7.0-1",
                },
                {"name": "bwa", "path": "/usr/bin/bwa", "version": "0.7.17"},
                {
                    "name": "samtools",
                    "path": "/usr/bin/samtools",
                    "version": "1.19",
                },
            ],
            "germline_argv_sha256": "4" * 64,
            "tool_started_count": 1,
            "tool_exited_count": 1,
            "germline_exit_code": 0,
            "worker_input_verified_count": 10,
        },
        "input_inventory": {
            "exact_keys": sorted(item["object_key"] for item in input_objects),
            "multipart_upload_count": 0,
        },
        "prepared_inventory": _inventory(
            (
                *(item["object_key"] for item in prepared),
                prepared_worker_receipt["object_key"],
            ),
            prepared_manifest["object_key"],
        ),
        "output_inventory": _inventory(
            tuple(item["object_key"] for item in outputs[:-1]),
            outputs[-1]["object_key"],
        ),
        "prepared_reference_receipt_id": "prepared-reference-admission:exact",
        "h100_receipt_id": "h100-execution:exact",
        "status": "execution_artifacts_verified",
        "frontier_state": "artifacts_verified",
        "cache_outcome": "hit",
        "prepared_reference_source": "cache_hit",
    }


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value).encode("utf-8"))


def _identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "byte_count": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _evidence(
    root: Path,
) -> tuple[fixture_export.CacheHitEvidenceInput, dict[str, Path]]:
    root.mkdir(parents=True, exist_ok=True)
    journey = Journey.create(root / "journey")
    parent: list[str] = []

    def record(event_type: str, payload: dict[str, object] | None = None) -> None:
        nonlocal parent
        event = journey.record(event_type, payload or {}, parent)
        parent = [event["event_id"]]

    facts = _facts()
    record("ExecutionProposalSubmitted", {"proposal_id": facts["proposal_id"]})
    record("ExecutionHitlResponded", {"authority_id": facts["authority_id"]})
    record("RunInputStorageBucketCreateClaimed")
    record("RunInputStorageBucketCreateInvoked")
    record("execution_inputs_verified")
    record(
        "prepared_reference_cache_hit",
        {"admission_id": facts["prepared_reference_receipt_id"]},
    )
    record("H100ExecutionCreateClaimed")
    record("H100ExecutionCreateInvoked")
    record("H100ExecutionSubmitted", {"provider_job_id": "aijob-exact"})
    record(
        "H100ExecutionArtifactsVerified",
        {"receipt_id": facts["h100_receipt_id"]},
    )
    paths = {
        "journey": journey.path,
        "live_receipt": root / "live-receipt.json",
        "independent_verification": root / "independent-verification.json",
        "hostile_reopen": root / "hostile-reopen.json",
        "evidence_index": root / "evidence-index.json",
    }
    journey_identity = _identity(journey.path)
    live = {
        "schema_version": "bio-neb.ticket94-live-receipt/1",
        "journey_id": journey.journey_id,
        "journey_sha256": journey_identity["sha256"],
        "facts": facts,
    }
    independent = {
        **live,
        "schema_version": "bio-neb.ticket94-independent-verification/1",
    }
    hostile = {
        "schema_version": "bio-neb.ticket94-hostile-reopen/1",
        "journey_id": journey.journey_id,
        "journey_byte_count": journey_identity["byte_count"],
        "journey_sha256": journey_identity["sha256"],
        "returncodes": [0, 0],
        "stdout_byte_count": 311,
        "stdout_sha256": "5" * 64,
        "stdout_byte_stable": True,
        "journey_byte_stable": True,
        "model_tool_provider_storage_process_contact_count": 0,
        "filesystem_write_count": 0,
        "fixture_contact_count": 0,
    }
    _write(paths["live_receipt"], live)
    _write(paths["independent_verification"], independent)
    _write(paths["hostile_reopen"], hostile)
    identities = {
        name: _identity(path)
        for name, path in paths.items()
        if name != "evidence_index"
    }
    _write(
        paths["evidence_index"],
        {
            "schema_version": "bio-neb.ticket94-evidence-index/1",
            **identities,
            "facts_sha256": hashlib.sha256(
                json.dumps(
                    facts,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
    )
    return fixture_export.CacheHitEvidenceInput.model_validate(
        {
            "journey": identities["journey"],
            "live_receipt": identities["live_receipt"],
            "independent_verification": identities["independent_verification"],
            "hostile_reopen": identities["hostile_reopen"],
            "evidence_index": _identity(paths["evidence_index"]),
        }
    ), paths


def _cache_miss_scenario() -> dict[str, object]:
    return {
        "cache_outcome": "miss",
        "prepared_reference_source": "cpu_built",
        "fixture_grants_live_authority": False,
        "source_ticket": 93,
        "run_label": "accepted-cache-miss-run",
        "journey_id": "cache-miss-journey",
        "journey_byte_count": 101,
        "journey_sha256": "6" * 64,
        "live_receipt_sha256": "7" * 64,
        "independent_verification_sha256": "8" * 64,
        "evidence_index_sha256": "9" * 64,
        "hostile_reopen_sha256": "0" * 64,
        "proposal_id": "execution-proposal:cache-miss",
        "authority_id": "execution-authority:cache-miss",
        "packet_binding_id": "prepared-execution-packet:cache-miss",
        "mutation_counts": {
            name: {"claimed": 1, "contacted": 1, "created": 1}
            for name in ("bucket", "cpu", "h100")
        },
        "prepared_reference_receipt_id": "prepared-reference-receipt:cache-miss",
        "prepared_manifest": {
            "object_key": "prepared/manifest.json",
            "byte_count": 41,
            "sha256": "a" * 64,
        },
        "h100_receipt_id": "h100-execution-receipt:cache-miss",
        "execution_manifest": {
            "object_key": "execution/manifest.json",
            "byte_count": 43,
            "sha256": "b" * 64,
        },
        "status": "execution_artifacts_verified",
        "frontier_state": "artifacts_verified",
        "hostile_reopen_projection": {
            "returncodes": [0, 0],
            "stdout_byte_count": 47,
            "stdout_sha256": "c" * 64,
            "stdout_byte_stable": True,
            "journey_byte_stable": True,
            "external_contact_count": 0,
            "filesystem_mutation_count": 0,
        },
    }


def _bundle_evidence(
    root: Path,
) -> tuple[fixture_export.ParabricksFixtureExportInput, dict[str, Path]]:
    hit, paths = _evidence(root / "hit")
    miss = root / "cache-miss-scenario.json"
    _write(miss, _cache_miss_scenario())
    paths["cache_miss_scenario"] = miss
    return fixture_export.ParabricksFixtureExportInput(
        cache_miss_scenario=fixture_export.EvidenceFileIdentity.model_validate(
            _identity(miss)
        ),
        cache_hit_run_label="accepted-cache-hit-run",
        cache_hit_evidence=hit,
    ), paths


def _replace_hit_identity(
    manifest: fixture_export.ParabricksFixtureExportInput,
    name: str,
    path: Path,
) -> fixture_export.ParabricksFixtureExportInput:
    hit = manifest.cache_hit_evidence.model_copy(
        update={
            name: fixture_export.EvidenceFileIdentity.model_validate(_identity(path))
        }
    )
    return manifest.model_copy(update={"cache_hit_evidence": hit})


class PreExecutionContactSummaryTests(unittest.TestCase):
    """The planning contact vocabulary the exporter admits.

    A constructed Pass view stands in for a Run: no scripted Run reaches the
    Execution review with a companion HEAD (the planning flow in
    ``tests.test_execution_contact_accounting`` stops at Planning HITL).
    """

    @staticmethod
    def _view(*operations):
        from types import SimpleNamespace

        rows = tuple(SimpleNamespace(kind="start", operation=o) for o in operations)
        return SimpleNamespace(
            passes=(
                SimpleNamespace(
                    disposition="completed",
                    journal_complete=True,
                    accounting_error=None,
                    contacts=rows,
                ),
            ),
            model_contact_postmortem=SimpleNamespace(
                logical_request_count=1, physical_contact_count=1
            ),
        )

    def test_a_companion_head_is_a_planning_contact(self):
        summary = fixture_export._pre_execution_contacts(
            self._view(
                "model.request",
                "source.get",
                "source.get",
                "source.head",
                "tool.execute",
            )
        )
        self.assertEqual(
            summary.operations,
            {"model.request": 1, "source.get": 2, "source.head": 1, "tool.execute": 1},
        )
        with self.assertRaisesRegex(ValueError, "outside planning operations"):
            fixture_export._pre_execution_contacts(self._view("source.post"))


class ParabricksFixtureExportTests(unittest.TestCase):
    def test_checked_in_bundle_is_the_sole_active_traced_run_source(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        active_fixtures = tuple(
            repository_root.glob("tests/fixtures/parabricks-run-fixture*.json")
        )
        self.assertEqual(active_fixtures, (ACTIVE_PARABRICKS_FIXTURE_BUNDLE_PATH,))

        content = ACTIVE_PARABRICKS_FIXTURE_BUNDLE_PATH.read_bytes()
        self.assertEqual(len(content), ACTIVE_PARABRICKS_FIXTURE_BUNDLE_BYTE_COUNT)
        self.assertEqual(
            hashlib.sha256(content).hexdigest(),
            ACTIVE_PARABRICKS_FIXTURE_BUNDLE_SHA256,
        )
        self.assertEqual(ACTIVE_PARABRICKS_FIXTURE_BUNDLE_SOURCE_TICKETS, (93, 94))

        bundle = load_active_parabricks_fixture_bundle()
        self.assertEqual(
            bundle.schema_version,
            "bio-neb.parabricks-run-fixture-bundle/1",
        )
        self.assertEqual(
            bundle.exporter_version,
            "bio-neb.parabricks-fixture-exporter/2",
        )
        self.assertEqual(
            tuple(item.cache_outcome for item in bundle.scenarios),
            ("miss", "hit"),
        )
        miss = get_active_parabricks_fixture_scenario("miss")
        hit = get_active_parabricks_fixture_scenario("hit")
        self.assertEqual(miss.source_ticket, 93)
        self.assertEqual(
            miss.run_label,
            "ticket93-cache-miss-20260815T145000Z-a14",
        )
        self.assertEqual(
            miss.journey_id,
            "5d56aa39-51f5-4a8f-8729-eec75f6d4f2e",
        )
        self.assertEqual(
            miss.hostile_reopen_sha256,
            "6a67f092ad7e780b35c83ed8935685a33bc962f58868ce59bf5f443e6751fe62",
        )
        self.assertEqual(
            (
                miss.mutation_counts.bucket.created,
                miss.mutation_counts.cpu.created,
                miss.mutation_counts.h100.created,
            ),
            (1, 1, 1),
        )
        self.assertEqual(hit.source_ticket, 94)
        self.assertEqual(
            hit.run_label,
            "ticket94-cache-hit-20260815T190000Z-a7",
        )
        self.assertEqual(hit.source_evidence.ticket, 94)
        self.assertEqual(
            hit.source_evidence.journey_id,
            "7a1470d1-a3cb-484e-8f48-22df21b493a7",
        )
        self.assertEqual(
            hit.source_evidence.journey_sha256,
            "fde311786c0782973c03735722233ce3b3f4b8d5a50835f215e371e8d829b013",
        )
        self.assertEqual(
            hit.source_evidence.live_receipt_sha256,
            "d5166179516e91be92c3c7f5b2b01688f335aff70f0cb27a671dd3fa30c983f4",
        )
        self.assertEqual(
            hit.source_evidence.independent_verification_sha256,
            "033bf0f44c14ff2162dbb7c7a3291a8d4b25e2002c8e8374916da2990c0f5745",
        )
        self.assertEqual(
            hit.source_evidence.evidence_index_sha256,
            "a21d97965d70a5d24e91ac63debdb8f200798df3a65f6fb62b922c07412eba0b",
        )
        self.assertEqual(
            hit.source_evidence.hostile_reopen_sha256,
            "6b73cd9fedd4d612bfd59d012bd17ee47cd2bafbd6cfabdfe288b74aa3623828",
        )
        self.assertEqual(
            hit.facts.proposal_id,
            "execution-proposal:3665b5e4f119c229ba6cc09d48603899a9568257b4bd598470f7c187fd255952",
        )
        self.assertEqual(
            hit.facts.authority_id,
            "execution-authority:ea0114a07fd830afcb1a4399183a533fc2ee1c4cc6fc20b9253de8e8405359b5",
        )
        self.assertEqual(
            hit.facts.packet_binding_id,
            "prepared-execution-packet:e8a6d74537d1cc9c08b173172bcd02daf638b836d4d6023f42507bc72926e0d6",
        )
        self.assertEqual(hit.status, "execution_artifacts_verified")
        self.assertEqual(hit.frontier_state, "artifacts_verified")
        self.assertEqual(hit.prepared_reference_source, "cache_hit")
        self.assertEqual(hit.mutation_counts.bucket.created, 1)
        self.assertEqual(hit.mutation_counts.cpu.created, 0)
        self.assertEqual(hit.mutation_counts.h100.created, 1)
        self.assertTrue(hit.hostile_reopen.journey_byte_stable)
        self.assertTrue(hit.hostile_reopen.stdout_byte_stable)
        self.assertEqual(hit.hostile_reopen.returncodes, (0, 0))
        self.assertEqual(
            hit.hostile_reopen.model_tool_provider_storage_process_contact_count,
            0,
        )
        self.assertFalse(miss.fixture_grants_live_authority)
        self.assertFalse(hit.fixture_grants_live_authority)
        self.assertFalse(bundle.fixture_grants_live_authority)

    def test_active_fixture_missing_tamper_and_hash_mismatch_fail_closed(self) -> None:
        content = ACTIVE_PARABRICKS_FIXTURE_BUNDLE_PATH.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(
                fixture_export.FixtureExportRefused, "fixture_invalid"
            ):
                fixture_export.review_fixture(
                    root / "missing.json",
                    expected_sha256=ACTIVE_PARABRICKS_FIXTURE_BUNDLE_SHA256,
                )

            candidate = root / "candidate.json"
            candidate.write_bytes(content)
            with self.assertRaisesRegex(
                fixture_export.FixtureExportRefused, "fixture_hash_mismatch"
            ):
                fixture_export.review_fixture(candidate, expected_sha256="0" * 64)

            candidate.write_bytes(content[:-1] + b" ")
            with self.assertRaises(fixture_export.FixtureExportRefused) as refused:
                fixture_export.review_fixture(
                    candidate,
                    expected_sha256=ACTIVE_PARABRICKS_FIXTURE_BUNDLE_SHA256,
                )
            assert_refused(
                self, refused.exception, "fixture_invalid", json.JSONDecodeError
            )

    def test_reference_expected_value_is_absent_outside_owned_catalog(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        accepted_catalog = (
            repository_root / "src/bio_neb/parabricks/accepted_pre_execution.py"
        ).resolve()
        historical_tokens = (
            "ticket" + "69",
            "ticket-" + "69",
            "ticket_" + "69",
        )
        historical_source_sha256 = "".join(
            (
                "11912a45a545bf01a10b2a7f10eb7a429",
                "24436b4d19b476b1899834fb7ba74a3",
            )
        )
        # Registered and historical fixtures retain observed identities; none is a
        # second product catalog or live authority (derived, not restated).
        ownership = json.loads(
            (repository_root / "tests/testing_ownership.json").read_text(
                encoding="utf-8"
            )
        )
        fixture_files = {
            (repository_root / name).resolve()
            for name in (
                ownership["execution_fixture"]["path"],
                *(
                    registration["path"]
                    for registration in ownership[
                        "additional_execution_fixtures"
                    ].values()
                ),
                *(
                    name
                    for name, entry in ownership["unregistered_fixtures"].items()
                    if entry["kind"] == "historical"
                ),
            )
        }
        for active_root in (repository_root / "src", repository_root / "tests"):
            for path in active_root.rglob("*"):
                if path.suffix not in {".py", ".json"}:
                    continue
                retained = path.read_text(encoding="utf-8").lower()
                for token in historical_tokens:
                    self.assertNotIn(
                        token,
                        retained,
                        f"historical full-Run pointer remains active in {path}",
                    )
                if path.resolve() not in {*fixture_files, accepted_catalog}:
                    self.assertNotIn(
                        historical_source_sha256,
                        retained,
                        f"historical expected value remains active in {path}",
                    )

    def test_current_active_shape_exports_deterministically_and_idempotently(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest, _ = _bundle_evidence(root)
            first = fixture_export.compile_fixture(manifest)
            second = fixture_export.compile_fixture(manifest)
            self.assertEqual(first, second)
            output = root / "candidate.json"
            receipt = fixture_export.export_fixture(manifest, output)
            self.assertEqual(output.read_bytes(), first)
            self.assertEqual(fixture_export.export_fixture(manifest, output), receipt)
            self.assertEqual(
                fixture_export.review_fixture(output, expected_sha256=receipt.sha256),
                first,
            )
            value = json.loads(first)
            self.assertFalse(value["fixture_grants_live_authority"])
            self.assertEqual(
                value["scenarios"][1]["facts"]["input_objects"][0]["sha256"],
                SYNTHETIC_SOURCE_SHA256,
            )
            self.assertEqual(
                [item["cache_outcome"] for item in value["scenarios"]],
                ["miss", "hit"],
            )
            retained = first.decode("utf-8").lower()
            for forbidden in (
                "https://",
                "s3://",
                "/home/",
                "credential",
                "selector",
                "provider_body",
                "raw_log",
            ):
                self.assertNotIn(forbidden, retained)

    def test_missing_hash_schema_and_conflicting_evidence_fail_closed(self) -> None:
        cases = ("missing", "hash", "schema", "conflict", "miss", "miss_key")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                manifest, paths = _bundle_evidence(root)
                if case == "missing":
                    paths["independent_verification"].unlink()
                elif case == "hash":
                    paths["hostile_reopen"].write_bytes(b"{}")
                elif case == "schema":
                    value = json.loads(paths["live_receipt"].read_bytes())
                    value["schema_version"] = "bio-neb.ticket94-live-receipt/changed"
                    _write(paths["live_receipt"], value)
                    manifest = _replace_hit_identity(
                        manifest, "live_receipt", paths["live_receipt"]
                    )
                elif case == "conflict":
                    value = json.loads(paths["independent_verification"].read_bytes())
                    value["facts"]["request_id"] = "research-request:conflict"
                    _write(paths["independent_verification"], value)
                    manifest = _replace_hit_identity(
                        manifest,
                        "independent_verification",
                        paths["independent_verification"],
                    )
                    index = json.loads(paths["evidence_index"].read_bytes())
                    index["independent_verification"] = _identity(
                        paths["independent_verification"]
                    )
                    _write(paths["evidence_index"], index)
                    manifest = _replace_hit_identity(
                        manifest, "evidence_index", paths["evidence_index"]
                    )
                else:
                    value = json.loads(paths["cache_miss_scenario"].read_bytes())
                    if case == "miss":
                        value["mutation_counts"]["cpu"]["created"] = 0
                    else:
                        value["prepared_manifest"]["object_key"] = "../manifest.json"
                    _write(paths["cache_miss_scenario"], value)
                    manifest = manifest.model_copy(
                        update={
                            "cache_miss_scenario": (
                                fixture_export.EvidenceFileIdentity.model_validate(
                                    _identity(paths["cache_miss_scenario"])
                                )
                            )
                        }
                    )
                reason, cause = {
                    "missing": ("evidence_unavailable", FileNotFoundError),
                    "hash": ("evidence_size_mismatch", None),
                    "schema": ("live_receipt_invalid", "schema_version"),
                    "conflict": ("independent_facts_conflict", None),
                    "miss": (
                        "cache_miss_scenario_invalid",
                        "cache-miss scenario lost an exact 1/1/1 triplet",
                    ),
                    "miss_key": (
                        "cache_miss_scenario_invalid",
                        "fixture manifest key must be",
                    ),
                }[case]
                with self.assertRaises(fixture_export.FixtureExportRefused) as refused:
                    fixture_export.compile_fixture(manifest)
                assert_refused(self, refused.exception, reason, cause)

    def test_direct_module_export_cli_uses_the_strict_input_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest, _ = _bundle_evidence(root)
            manifest_path = root / "fixture-export-input.json"
            _write(manifest_path, manifest.model_dump(mode="json"))
            self.assertEqual(fixture_export.load_export_input(manifest_path), manifest)
            output = root / "candidate.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = fixture_export.main(
                    [
                        "export",
                        "--input-manifest",
                        str(manifest_path),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(result, 0)
            receipt = fixture_export.ExportedFixtureReceipt.model_validate_json(
                stdout.getvalue()
            )
            self.assertEqual(receipt.byte_count, output.stat().st_size)
            self.assertEqual(
                receipt.sha256, hashlib.sha256(output.read_bytes()).hexdigest()
            )

    def test_forbidden_locator_and_conflicting_output_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest, paths = _bundle_evidence(root)
            live = json.loads(paths["live_receipt"].read_bytes())
            independent = json.loads(paths["independent_verification"].read_bytes())
            for value in (live, independent):
                value["facts"]["request_id"] = "https://provider.invalid/?token=raw"
            _write(paths["live_receipt"], live)
            _write(paths["independent_verification"], independent)
            for name in ("live_receipt", "independent_verification"):
                manifest = _replace_hit_identity(manifest, name, paths[name])
            index = json.loads(paths["evidence_index"].read_bytes())
            index["live_receipt"] = _identity(paths["live_receipt"])
            index["independent_verification"] = _identity(
                paths["independent_verification"]
            )
            index["facts_sha256"] = hashlib.sha256(
                canonical_json(live["facts"]).encode("utf-8")
            ).hexdigest()
            _write(paths["evidence_index"], index)
            manifest = _replace_hit_identity(
                manifest, "evidence_index", paths["evidence_index"]
            )
            with self.assertRaisesRegex(
                fixture_export.FixtureExportRefused,
                "fixture_contains_forbidden_content",
            ):
                fixture_export.compile_fixture(manifest)

            clean, _ = _bundle_evidence(root / "clean")
            output = root / "candidate.json"
            output.write_bytes(b"different")
            with self.assertRaisesRegex(
                fixture_export.FixtureExportRefused, "output_conflict"
            ):
                fixture_export.export_fixture(clean, output)

    def test_hostile_review_rejects_tampering_without_contact_or_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest, _ = _bundle_evidence(root)
            output = root / "candidate.json"
            receipt = fixture_export.export_fixture(manifest, output)
            before = output.read_bytes()
            with (
                patch.object(socket, "socket", side_effect=AssertionError("network")),
                patch.object(
                    subprocess, "Popen", side_effect=AssertionError("process")
                ),
                patch.object(Path, "write_bytes", side_effect=AssertionError("write")),
            ):
                self.assertEqual(
                    fixture_export.review_fixture(
                        output, expected_sha256=receipt.sha256
                    ),
                    before,
                )
            output.write_bytes(before[:-1] + b" ")
            with self.assertRaises(fixture_export.FixtureExportRefused) as refused:
                fixture_export.review_fixture(output, expected_sha256=receipt.sha256)
            assert_refused(
                self, refused.exception, "fixture_invalid", json.JSONDecodeError
            )


if __name__ == "__main__":
    unittest.main()
