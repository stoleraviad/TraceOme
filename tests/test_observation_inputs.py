"""Approved Investigation work supplies the Run's own trusted VCF and tools (#107).

These tests drive the coordinator (``advance``) with a controlled object store and
the locked environment's real bcftools on a real VCF; nothing names a path, so
every input comes from the Run's own verified Execution and registration.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bio_neb.application.advancement import (
    ResumeRun,
    RunAdvancementRefused,
    advance,
    review_response_from_json,
    use_run_advancement_dependencies,
)
from bio_neb.application.queries import ObserveRun, observe, read_investigation
from bio_neb.journey import Journey
from bio_neb.recording import read_contacts
from tests.observation_input_fakes import (
    RUN_BAM,
    RUN_VCF,
    TABLE_READING_ENVIRON,
    RunOutputStore,
)
from tests.run_prerequisites import RunPrerequisites
from tests.sequential_investigation_support import (
    approved_collection_observation,
    collection_request,
)
from tests.vcf_lookup_fakes import TableReaderModel

CAUSE_SCHEMA = "bio-neb.investigation-object-contact/4"


def _pass_journal_rows(journey: Journey):
    """The last Pass journal's rows, through the retained contact reader."""
    started = [
        e for e in Journey.open(journey.path).read() if e["event_type"] == "PassStarted"
    ][-1]
    pass_id = started["payload"]["pass_id"]
    path = journey.run_directory / "contacts" / pass_id / "journal.jsonl"
    return read_contacts(path.read_bytes(), run_id=journey.journey_id, pass_id=pass_id)


def _grant_contacts(journey: Journey) -> list[dict]:
    """Every grant download row of the Run's Pass journals, in Pass order."""
    rows = []
    for started in Journey.open(journey.path).read():
        if started["event_type"] != "PassStarted":
            continue
        pass_id = started["payload"]["pass_id"]
        path = journey.run_directory / "contacts" / pass_id / "journal.jsonl"
        if not path.exists():
            continue
        for row in read_contacts(
            path.read_bytes(), run_id=journey.journey_id, pass_id=pass_id
        ):
            if row.domain is not None and row.domain.phase == "record":
                if row.domain.payload.get("schema_version") == CAUSE_SCHEMA:
                    rows.append(row.domain.payload)
    return rows


class AutomaticObservationInputTests(RunPrerequisites, unittest.TestCase):
    def _approved(self, root: Path):
        parent = self.prerequisite(
            "executed",
            investigation_request=collection_request(3),
            output_payloads={"VCF": RUN_VCF},
        ).journey_file
        return approved_collection_observation(
            root, cardinality=3, approve_count=1, forked=True, parent_file=parent
        )

    def _advance(self, journey, store, intent=None):
        with use_run_advancement_dependencies(
            environ=TABLE_READING_ENVIRON,
            investigation_model_factory=TableReaderModel().model,
            investigation_vcf_source_factory=store,
        ):
            return advance(intent or ResumeRun(journey_file=str(journey.path)))

    def test_approval_supplies_the_runs_verified_vcf_bam_and_probed_tools_once(self):
        from bio_neb.parabricks.result import read_completed_parabricks_result

        with tempfile.TemporaryDirectory() as temporary:
            _, journey, first, descriptor, _ = self._approved(Path(temporary).resolve())
            execution = read_completed_parabricks_result(str(journey.path))
            bam = next(
                item
                for item in execution.verified_objects
                if item.role == "aligned_bam"
            )
            store = RunOutputStore()
            observed = self._advance(journey, store)

            # One download each of the Run's own verified VCF and BAM, under the
            # grant, by their recorded identities: no command, path or flag
            # named them. Each fits one part.
            self.assertEqual(store.gets, [descriptor.object_key, bam.object_key])

            def rows(role, item, byte_count):
                identity = {
                    "schema_version": CAUSE_SCHEMA,
                    "authority_id": first.authority_id,
                    "role": role,
                    "bucket": item.bucket,
                    "object_key": item.object_key,
                    "part": 1,
                    "offset": 0,
                    "part_byte_count": byte_count,
                    "attempt": 1,
                }
                return [
                    {
                        **identity,
                        "stage": "started",
                        "classification": None,
                        "byte_count": None,
                        "cause": None,
                    },
                    {
                        **identity,
                        "stage": "ended",
                        "classification": "success",
                        "byte_count": byte_count,
                        "cause": {
                            "kind": "success",
                            "status": None,
                            "code": None,
                            "error_type": None,
                        },
                    },
                ]

            self.assertEqual(
                _grant_contacts(journey),
                rows("VCF", descriptor, len(RUN_VCF)) + rows("BAM", bam, len(RUN_BAM)),
            )
            read = read_investigation(Journey.open(journey.path))
            command = read.collections[-1].evidence.command
            root = journey.run_directory / "investigation"
            self.assertEqual(command.vcf_file, str(root / "vcf" / "variants.vcf"))
            self.assertEqual(Path(command.vcf_file).read_bytes(), RUN_VCF)
            # The BAM and its receipt lie beside the VCF, named by `aligned_bam`.
            self.assertEqual((root / "bam" / "aligned.bam").read_bytes(), RUN_BAM)
            receipt = json.loads(
                (root / "bam" / "current-bam-acquisition-receipt.json").read_bytes()
            )
            self.assertEqual(receipt["descriptor"], bam.model_dump(mode="json"))
            # Its lineage is the Execution's, which this Fork inherits.
            self.assertEqual(receipt["lineage"]["execution_id"], execution.execution_id)
            probe = json.loads(Path(command.bcftools_probe_receipt_file).read_bytes())
            self.assertEqual(
                [tool["name"] for tool in probe["tools"]], ["bcftools", "samtools"]
            )
            self.assertEqual(
                command.bcftools_executable, probe["tools"][0]["executable"]
            )
            # The real bcftools looked up the approved item in the real VCF.
            member = next(
                m
                for m in read.recorded_members
                if m.evidence.work_item.work_item_id
                == first.authorized_work_items[0].work_item_id
            )
            self.assertEqual(
                member.evidence.sample_observation.preparation.runtime.executable_path,
                command.bcftools_executable,
            )
            self.assertEqual(
                member.evidence.evidence_state.outcome, "exact_allele_observed"
            )
            # The two unapproved items stay in the remaining-scope Review.
            self.assertEqual(len(observed.investigation.pending_work_item_ids), 2)

    def _observe_grant(self, journey, store):
        """The coordinator's observation seam for one eligible grant."""
        from bio_neb.investigation.investigation_observation_commands import (
            advance_approved_observation,
        )
        from bio_neb.recording import processing_pass

        owner = Journey.open(journey.path)
        with owner.processing(), processing_pass(owner) as interval:
            observed = advance_approved_observation(owner, object_read_factory=store)
            interval.close("completed")
        return observed

    def test_a_later_grant_reuses_the_local_inputs_without_transfer_or_reads(self):
        from bio_neb.investigation.investigation import ApproveInvestigationResponse
        from bio_neb.investigation.investigation_commands import (
            RespondInvestigationReviewCommand,
            respond_investigation_review,
        )

        with tempfile.TemporaryDirectory() as temporary:
            _, journey, _, _, _ = self._approved(Path(temporary).resolve())
            store = RunOutputStore()
            self.assertTrue(self._observe_grant(journey, store))
            first_gets = list(store.gets)
            first_rows = _grant_contacts(journey)
            bam_file = journey.run_directory / "investigation" / "bam" / "aligned.bam"
            command = (
                read_investigation(Journey.open(journey.path))
                .collections[-1]
                .evidence.command
            )
            receipts = {
                path: Path(path).read_bytes()
                for path in (
                    command.materialization_receipt_file,
                    command.bcftools_probe_receipt_file,
                    str(bam_file.parent / "current-bam-acquisition-receipt.json"),
                )
            }
            pending = read_investigation(Journey.open(journey.path)).investigation
            request = pending.pending_request
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=request.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve",
                        proposal_id=request.proposal.proposal_id,
                        work_item_ids=(request.proposal.work_items[1].work_item_id,),
                    ),
                )
            )
            reads: dict[str, int] = {}
            original_fdopen = os.fdopen

            def metered(descriptor, *args, **kwargs):
                path = os.readlink(f"/proc/self/fd/{descriptor}")
                if path in (command.vcf_file, str(bam_file)):
                    reads[path] = reads.get(path, 0) + 1
                return original_fdopen(descriptor, *args, **kwargs)

            with patch.object(os, "fdopen", metered):
                self.assertTrue(self._observe_grant(journey, store))
            read = read_investigation(Journey.open(journey.path))
            self.assertEqual(len(read.collections), 2)
            self.assertEqual(
                read.collections[-1].evidence.command.vcf_file, command.vcf_file
            )
            # No second download, probe or byte read: the retained VCF and BAM
            # are trusted.
            self.assertEqual(len(first_gets), 2)
            self.assertEqual(store.gets, first_gets)
            self.assertEqual(_grant_contacts(journey), first_rows)
            self.assertEqual(reads, {})
            for path, content in receipts.items():
                self.assertEqual(Path(path).read_bytes(), content)

    def test_a_transient_part_fault_retries_by_the_runs_recorded_policy(self):
        from tests.observation_input_fakes import transient

        with tempfile.TemporaryDirectory() as temporary:
            _, journey, first, _, _ = self._approved(Path(temporary).resolve())
            # The VCF's part, then the BAM's, each fault once transiently.
            store = RunOutputStore(faults=[transient(), None, transient()])
            import bio_neb.failure._faults as failure
            import bio_neb.investigation.bam_materialization as bam_materialization
            import bio_neb.investigation.vcf_materialization as materialization

            bound = []

            def recorded(resolve):
                def policy(surface, tuning=None):
                    # Each download resolves its policy from the Run's section.
                    bound.append((surface.name, failure._bound_tuning.get()))
                    return resolve(surface, tuning)

                return policy

            with (
                patch("time.sleep"),
                patch.object(
                    materialization, "policy_for", recorded(materialization.policy_for)
                ),
                patch.object(
                    bam_materialization,
                    "policy_for",
                    recorded(bam_materialization.policy_for),
                ),
            ):
                self._advance(journey, store)
            registered = next(
                e
                for e in Journey.open(journey.path).read()
                if e["event_type"] == "RunInputsRegistered"
            )["payload"]["runtime_selections"]["failure"]
            self.assertEqual(
                [(name, tuning.model_dump(mode="json")) for name, tuning in bound],
                [
                    ("vcf_materialization", registered),
                    ("bam_materialization", registered),
                ],
            )
            self.assertEqual(
                [
                    (
                        row["role"],
                        row["part"],
                        row["attempt"],
                        row["stage"],
                        row["classification"],
                    )
                    for row in _grant_contacts(journey)
                ],
                [
                    ("VCF", 1, 1, "started", None),
                    ("VCF", 1, 1, "ended", "transient"),
                    ("VCF", 1, 2, "started", None),
                    ("VCF", 1, 2, "ended", "success"),
                    ("BAM", 1, 1, "started", None),
                    ("BAM", 1, 1, "ended", "transient"),
                    ("BAM", 1, 2, "started", None),
                    ("BAM", 1, 2, "ended", "success"),
                ],
            )
            self.assertEqual(len(store.gets), 4)

    def test_the_bam_downloads_in_ranged_parts_under_its_grant(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, journey, first, _, _ = self._approved(Path(temporary).resolve())
            store = RunOutputStore(part_bytes=8)
            self._advance(journey, store)
            bam_rows = [
                (row["part"], row["offset"], row["part_byte_count"], row["stage"])
                for row in _grant_contacts(journey)
                if row["role"] == "BAM"
            ]
            # The BAM in 8-byte parts, the last one short, each started then
            # ended (the tiny real BAM).
            parts = [
                (offset // 8 + 1, offset, min(8, len(RUN_BAM) - offset))
                for offset in range(0, len(RUN_BAM), 8)
            ]
            self.assertNotEqual(len(RUN_BAM) % 8, 0)
            self.assertEqual(
                bam_rows,
                [
                    (part, offset, size, stage)
                    for part, offset, size in parts
                    for stage in ("started", "ended")
                ],
            )
            bam = journey.run_directory / "investigation" / "bam" / "aligned.bam"
            self.assertEqual(bam.read_bytes(), RUN_BAM)
            # The Pass journal validates and closes each file's scope complete.
            closed = [
                row.domain.iteration
                for row in _pass_journal_rows(journey)
                if row.domain is not None and row.domain.phase == "closed"
            ]
            self.assertEqual(closed, [1, 2])

    def test_each_file_scope_opens_with_its_limits_and_counts_its_lookups(self):
        # On the product path the grant's journal is the credential resolver's
        # observer, so each file's lookup is a counted row under the grant,
        # after its scope opens with its stage's resolved part limits.
        import bio_neb.investigation.bam_materialization as bam_materialization
        import bio_neb.investigation.vcf_materialization as vcf_materialization

        with tempfile.TemporaryDirectory() as temporary:
            _, journey, first, _, _ = self._approved(Path(temporary).resolve())
            store = RunOutputStore()
            with (
                patch.object(
                    vcf_materialization,
                    "execution_object_storage",
                    store.execution_storage,
                ),
                patch.object(
                    bam_materialization,
                    "execution_object_storage",
                    store.execution_storage,
                ),
                use_run_advancement_dependencies(
                    environ=TABLE_READING_ENVIRON,
                    investigation_model_factory=TableReaderModel().model,
                ),
            ):
                advance(ResumeRun(journey_file=str(journey.path)))
            scopes = [
                (
                    row.domain.iteration,
                    row.domain.phase,
                    row.domain.payload.get("schema_version"),
                    row.domain.payload.get("limits", {}).get("surface"),
                    row.domain.payload.get("stage"),
                    row.domain.payload.get("outcome"),
                )
                for row in _pass_journal_rows(journey)
                if row.domain is not None
                and row.domain.operation_id == first.authority_id
            ]
            lookup = "bio-neb.investigation-credential-contact/1"
            limits = "bio-neb.investigation-part-limits/1"
            self.assertEqual(
                scopes,
                [
                    (
                        iteration,
                        phase,
                        schema,
                        surface and {"scope": "stage", "name": surface},
                        stage,
                        outcome,
                    )
                    for stage_name, iteration in (
                        ("vcf_materialization", 1),
                        ("bam_materialization", 2),
                    )
                    for phase, schema, surface, stage, outcome in (
                        ("started", None, None, None, None),
                        ("record", limits, stage_name, None, None),
                        ("record", lookup, None, "started", None),
                        ("record", lookup, None, "ended", "credential_resolved"),
                        ("record", CAUSE_SCHEMA, None, "started", None),
                        ("record", CAUSE_SCHEMA, None, "ended", None),
                        ("closed", CAUSE_SCHEMA, None, None, None),
                    )
                ],
            )
            opening = next(
                row.domain.payload
                for row in _pass_journal_rows(journey)
                if row.domain is not None and "limits" in row.domain.payload
            )
            self.assertEqual(
                opening["limits"]["effective"],
                {"part_seconds": 300, "process_seconds": 600},
            )
            self.assertEqual(opening["limits"]["stage_overrides"], {})

    def _own_collections(self, journey):
        return [
            collection
            for collection in read_investigation(Journey.open(journey.path)).collections
            if collection.evidence.command.journey_file == str(journey.path)
        ]

    def _answer_download(self, journey, store, action, **fields):
        current = observe(ObserveRun(journey_file=str(journey.path)))
        review = current.current_review
        return self._advance(
            journey,
            store,
            review_response_from_json(
                current,
                review.request_event_id,
                json.dumps(
                    {
                        "request_id": review.request.request.request_id,
                        "action": action,
                        "by": "researcher:owner",
                        **fields,
                    }
                ),
            ),
        )

    def test_a_refused_bam_asks_its_review_and_a_retry_downloads_it_afresh(self):
        from tests.observation_input_fakes import access_refused

        with tempfile.TemporaryDirectory() as temporary:
            _, journey, first, descriptor, _ = self._approved(Path(temporary).resolve())
            # The VCF downloads; the BAM's GET is refused.
            store = RunOutputStore(faults=[None, access_refused()])
            asked = self._advance(journey, store)

            # The failure module asks the researcher: the Run waits at the
            # BAM's Review; nothing was observed yet.
            self.assertEqual(asked.status, "investigation_download_review_pending")
            review = asked.current_review
            self.assertEqual(review.kind, "investigation_download")
            self.assertEqual(
                (review.request.request.stage, review.request.request.legal_actions),
                ("bam_materialization", ("retry", "abort")),
            )
            self.assertEqual(self._own_collections(journey), [])
            root = journey.run_directory / "investigation"
            self.assertEqual((root / "vcf" / "variants.vcf").read_bytes(), RUN_VCF)
            self.assertFalse((root / "bam").exists())

            observed = self._answer_download(journey, store, "retry")
            # Only the BAM downloads again, from scratch; then the lookups run.
            bam_key = store.gets[1]
            self.assertEqual(store.gets, [descriptor.object_key, bam_key, bam_key])
            self.assertEqual((root / "bam" / "aligned.bam").read_bytes(), RUN_BAM)
            (collection,) = self._own_collections(journey)
            # The approved item is observed; the two unapproved ones still wait.
            self.assertEqual(
                [item.status for item in collection.evidence.item_states],
                ["completed", "approval_pending", "approval_pending"],
            )
            self.assertNotEqual(
                observed.status, "investigation_download_review_pending"
            )
            # The Run's story tells the refusal, the question and the answer.
            from bio_neb.application.run_reader import read_journey_feed

            told = [
                (item.status, item.text)
                for item in read_journey_feed(str(journey.path)).items
                if item.event_type.startswith("InvestigationDownload")
            ]
            self.assertEqual(
                told,
                [
                    (
                        "failed",
                        "Your run's aligned reads (BAM) could not be downloaded",
                    ),
                    (
                        "question",
                        "Question for you: retry downloading your run's aligned "
                        "reads (BAM), or go on without it?",
                    ),
                    ("answer", "You chose to download the file again"),
                ],
            )

    def test_an_aborted_bam_is_unavailable_and_the_vcf_work_goes_on(self):
        from tests.observation_input_fakes import access_refused

        with tempfile.TemporaryDirectory() as temporary:
            _, journey, first, _, _ = self._approved(Path(temporary).resolve())
            store = RunOutputStore(faults=[None, access_refused()])
            self._advance(journey, store)
            self._answer_download(
                journey, store, "abort", reason="the BAM is not needed now"
            )
            # No second BAM GET; the approved item is observed in the VCF.
            self.assertEqual(len(store.gets), 2)
            self.assertFalse((journey.run_directory / "investigation" / "bam").exists())
            (collection,) = self._own_collections(journey)
            # The approved item is observed; the two unapproved ones still wait.
            self.assertEqual(
                [item.status for item in collection.evidence.item_states],
                ["completed", "approval_pending", "approval_pending"],
            )
            member = next(
                m
                for m in read_investigation(Journey.open(journey.path)).recorded_members
                if m.evidence.work_item.work_item_id
                == first.authorized_work_items[0].work_item_id
            )
            self.assertEqual(
                member.evidence.evidence_state.outcome, "exact_allele_observed"
            )

    def test_a_bam_interrupted_after_the_vcf_resumes_without_a_second_vcf(self):
        # Spec r1 S1: resume reuses the published VCF and downloads the BAM.
        with tempfile.TemporaryDirectory() as temporary:
            _, journey, first, descriptor, _ = self._approved(Path(temporary).resolve())
            store = RunOutputStore(faults=[None, KeyboardInterrupt])
            with self.assertRaises(KeyboardInterrupt):
                self._advance(journey, store)
            root = journey.run_directory / "investigation"
            self.assertFalse((root / "bam").exists())
            self._advance(journey, store)
            bam_key = store.gets[1]
            self.assertEqual(store.gets, [descriptor.object_key, bam_key, bam_key])
            self.assertEqual((root / "bam" / "aligned.bam").read_bytes(), RUN_BAM)
            (collection,) = self._own_collections(journey)
            self.assertEqual(
                [item.status for item in collection.evidence.item_states],
                ["completed", "approval_pending", "approval_pending"],
            )

    def test_an_aborted_vcf_fails_the_shared_preparation(self):
        from tests.observation_input_fakes import access_refused

        with tempfile.TemporaryDirectory() as temporary:
            _, journey, first, _, _ = self._approved(Path(temporary).resolve())
            store = RunOutputStore(faults=[access_refused()])
            asked = self._advance(journey, store)
            self.assertEqual(
                asked.current_review.request.request.stage, "vcf_materialization"
            )
            # The question states the VCF abort's real consequence (r1 S3).
            from bio_neb.application.run_reader import read_journey_feed

            self.assertIn(
                (
                    "question",
                    "Question for you: retry downloading your run's variant calls "
                    "(VCF)? Without it, the approved variant checks fail.",
                ),
                [
                    (item.status, item.text)
                    for item in read_journey_feed(str(journey.path)).items
                ],
            )
            # The BAM is not attempted after a failed VCF.
            self.assertEqual(len(store.gets), 1)
            # An aborted VCF fails the shared preparation: the item is
            # accounted failed with the download's own refusal, and the Run is
            # held.
            with self.assertRaises(RunAdvancementRefused) as held:
                self._answer_download(journey, store, "abort", reason="give up")
            self.assertEqual(
                (held.exception.status, held.exception.code),
                ("run_held", "investigation_shared_observation_failed"),
            )
            self.assertEqual(len(store.gets), 1)
            (collection,) = self._own_collections(journey)
            failed = next(
                item
                for item in collection.evidence.item_states
                if item.work_item_id == first.authorized_work_items[0].work_item_id
            )
            self.assertEqual(
                (failed.status, failed.reason),
                (
                    "failed",
                    "observation_exception:VcfMaterializationRefused:"
                    "vcf_materialization_object_read_failed",
                ),
            )

    def test_a_download_review_names_its_grant_frontier_and_execution_file(self):
        # Standards r1 S1: the failure fact is checked at append and on reopen.
        from bio_neb.investigation.download_review import ask_download_review
        from bio_neb.recording import processing_pass, work_frontier
        from tests.observation_input_fakes import access_refused

        with tempfile.TemporaryDirectory() as temporary:
            _, journey, first, descriptor, _ = self._approved(Path(temporary).resolve())
            store = RunOutputStore(faults=[access_refused()])
            self._advance(journey, store)
            self._answer_download(journey, store, "retry")
            owner = Journey.open(journey.path)
            timeline = owner.read()
            lawful = {
                "parent_event_id": work_frontier(timeline)["event_id"],
                "authority_id": first.authority_id,
                "role": "VCF",
                "object_key": descriptor.object_key,
                "code": "vcf_materialization_object_read_failed",
                "details": (),
            }
            for name, change in (
                ("not a grant", {"authority_id": "not-a-grant"}),
                ("not the work frontier", {"parent_event_id": timeline[0]["event_id"]}),
                ("another file", {"object_key": "not-an-execution-object"}),
                (
                    "the other file's stage",
                    {"code": "bam_materialization_object_read_failed"},
                ),
            ):
                with self.subTest(name):
                    before = journey.path.read_bytes()
                    with (
                        owner.processing(),
                        processing_pass(owner) as interval,
                        self.assertRaisesRegex(ValueError, "download failure"),
                    ):
                        try:
                            ask_download_review(owner, **{**lawful, **change})
                        finally:
                            interval.close("completed")
                    self.assertEqual(
                        [
                            line
                            for line in journey.path.read_bytes().splitlines()
                            if b"InvestigationDownload" in line
                        ],
                        [
                            line
                            for line in before.splitlines()
                            if b"InvestigationDownload" in line
                        ],
                    )

    def test_an_interrupted_part_stays_unfinished_in_the_pass_journal(self):
        from tests.observation_input_fakes import transient

        for faults, rows in (
            ([KeyboardInterrupt], [(1, "started", None)]),
            (
                [transient(), KeyboardInterrupt],
                [
                    (1, "started", None),
                    (1, "ended", "transient"),
                    (2, "started", None),
                ],
            ),
        ):
            with (
                self.subTest(
                    faults=[getattr(f, "__name__", type(f).__name__) for f in faults]
                ),
                tempfile.TemporaryDirectory() as temporary,
            ):
                _, journey, _, _, _ = self._approved(Path(temporary).resolve())
                store = RunOutputStore(faults=faults)
                with patch("time.sleep"), self.assertRaises(KeyboardInterrupt):
                    self._advance(journey, store)
                # Each started GET is retained; the interrupted one has no end and
                # the grant's scope never closes, so its Pass reads incomplete.
                self.assertEqual(
                    [
                        (row["attempt"], row["stage"], row["classification"])
                        for row in _grant_contacts(journey)
                    ],
                    rows,
                )
                self.assertEqual(len(store.gets), len(faults))
                interrupted = _pass_journal_rows(journey)[-1]
                self.assertEqual(
                    (interrupted.kind, interrupted.complete), ("final", False)
                )
                self.assertFalse(
                    any(
                        row.domain is not None and row.domain.phase == "closed"
                        for row in _pass_journal_rows(journey)
                    )
                )

    def test_a_stopped_investigation_with_a_collection_reaches_finalization(self):
        # A recorded live Run: the approved items were observed, a revise/stop
        # Review held completion, and the researcher stopped. Every ended Run
        # reaches the one Finalization path (PROJECT.md). The rsID with no lawful
        # Work item is an issue approved first and holds nothing; here one of
        # rs429358's two candidates stays unapproved under a strict completion
        # policy, so its remaining-scope Review is the one the stop answers.
        import copy

        from bio_neb.investigation.investigation_commands import (
            BeginInvestigationCommand,
        )
        from bio_neb.journey import canonical_json
        from bio_neb.providers.ncbi_investigation import InvestigationSourceResponse
        from tests.sequential_investigation_support import (
            ASSEMBLY_BODY,
            REFSNP_BODY,
            SequentialAdaptiveModel,
            SequentialSourceResponses,
            begin_in_pass,
            forked_child,
        )

        class UnknownRsidSource(SequentialSourceResponses):
            def fetch_refsnp(self, rsid):  # noqa: ANN001, ANN201
                if rsid != "rs9999999999":
                    body = json.loads(REFSNP_BODY)
                    alleles = body["primary_snapshot_data"]["placements_with_allele"][
                        0
                    ]["alleles"]
                    second = copy.deepcopy(alleles[1])
                    second["allele"]["spdi"]["inserted_sequence"] = "A"
                    alleles.append(second)
                    self.operations.append(f"fetch_refsnp:{rsid}")
                    return InvestigationSourceResponse(
                        request_identity=(
                            "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/"
                            + rsid.removeprefix("rs")
                        ),
                        status_code=200,
                        body=canonical_json(body).encode(),
                        observed_at="2026-08-26T00:00:00Z",
                        elapsed_time_ms=7,
                    )
                self.operations.append(f"fetch_refsnp:{rsid}")
                return InvestigationSourceResponse(
                    request_identity=(
                        "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/9999999999"
                    ),
                    status_code=404,
                    body=b'{"error": {"code": 404, "message": "not found"}}',
                    observed_at="2026-08-26T00:00:00Z",
                    elapsed_time_ms=7,
                )

        request = {
            "schema_version": "bio-neb.investigation-request/2",
            "asserted_by": "researcher:owner",
            "requests": [
                {"request": f"Does sample-001 carry {rsid}?", "target": {"rsid": rsid}}
                for rsid in ("rs429358", "rs9999999999")
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey = forked_child(
                root,
                parent_file=self.prerequisite(
                    "executed",
                    investigation_request=request,
                    output_payloads={"VCF": RUN_VCF},
                ).journey_file,
            )
            request_file = root / "request.json"
            request_file.write_text(json.dumps(request), encoding="utf-8")
            asked = begin_in_pass(
                BeginInvestigationCommand(
                    journey_file=str(journey.path), request_files=(str(request_file),)
                ),
                environ=TABLE_READING_ENVIRON,
                model_factory=SequentialAdaptiveModel().model,
                source_adapter_factory=lambda: UnknownRsidSource(
                    assembly_body=ASSEMBLY_BODY
                ),
            )
            issue = asked.pending_request
            self.assertEqual(issue.question.issue.verdict, "source_answered")
            store = RunOutputStore()

            def answer(document):
                current = observe(ObserveRun(journey_file=str(journey.path)))
                review = current.investigation.pending_request
                return self._advance(
                    journey,
                    store,
                    review_response_from_json(
                        current, review.request_event_id, json.dumps(document)
                    ),
                )

            proposed = answer(
                {
                    "action": "approve",
                    "collection_member_id": issue.question.collection_member_id,
                    "revision_number": 1,
                    "asserted_by": "researcher:owner",
                }
            )
            proposal = proposed.investigation.pending_request.proposal
            self.assertEqual(proposal.unresolved_request_member_ids, ())
            first, second = proposal.work_items
            # Completion waits for every member, so the unapproved one holds it.
            from bio_neb.investigation.investigation_policy import (
                InvestigationPolicyAmendment,
            )
            from bio_neb.investigation.investigation_policy_commands import (
                AmendInvestigationPolicyCommand,
                amend_investigation_policy,
            )

            amend_investigation_policy(
                AmendInvestigationPolicyCommand(
                    journey_file=str(journey.path),
                    amendment=InvestigationPolicyAmendment(
                        proposal_id=proposal.proposal_id,
                        expected_head_event_id=Journey.open(journey.path).read()[-1][
                            "event_id"
                        ],
                        previous_policy_id=None,
                        asserted_by="researcher:owner",
                        reason="Hold completion until every candidate is settled.",
                        completion="require_terminal_members",
                    ),
                )
            )
            approved = answer(
                {
                    "action": "approve",
                    "proposal_id": proposal.proposal_id,
                    "work_item_ids": [first.work_item_id],
                }
            )
            # The unapproved candidate keeps its remaining-scope Review open.
            remaining = approved.investigation.pending_request
            self.assertEqual(remaining.remaining_work_item_ids, (second.work_item_id,))
            self.assertEqual(remaining.legal_actions, ("approve", "revise", "stop"))
            # One download each of the VCF and the BAM.
            self.assertEqual(len(store.gets), 2)
            stopped = answer(
                {
                    "action": "stop",
                    "proposal_id": proposal.proposal_id,
                    "reason": "the other allele is not needed",
                }
            )
            self.assertEqual(stopped.status, "finalization_review_pending")
            basis = stopped.finalization.pending_request.basis
            self.assertEqual(basis.outcome, "investigation_stopped")
            self.assertEqual(
                [observation.work_item_id for observation in basis.sample_observations],
                [first.work_item_id],
            )
            self.assertEqual(
                [(item.request.rsid, item.decision) for item in basis.issue_items],
                [("rs9999999999", "approved")],
            )

    def test_a_missing_tool_holds_the_run_and_resume_observes_after_the_fix(self):
        import importlib.metadata

        with tempfile.TemporaryDirectory() as temporary:
            _, journey, first, _, _ = self._approved(Path(temporary).resolve())
            store = RunOutputStore()
            installed = importlib.metadata.version

            def absent(name):
                if name == "pysam":
                    raise importlib.metadata.PackageNotFoundError(name)
                return installed(name)

            with patch("importlib.metadata.version", absent):
                with self.assertRaises(RunAdvancementRefused) as held:
                    self._advance(journey, store)
            self.assertEqual(
                (
                    held.exception.status,
                    held.exception.failure.failure_class,
                    held.exception.failure.invariant_id,
                ),
                ("run_held", "invariant", "runtime_config_invalid"),
            )
            events = Journey.open(journey.path).read()
            failure = next(
                e for e in events if e["event_type"] == "RuntimeConfigurationFailed"
            )
            self.assertEqual(
                failure["payload"]["violations"],
                [
                    "investigation_tools.bcftools.package_version",
                    "investigation_tools.samtools.package_version",
                ],
            )
            self.assertEqual(store.gets, [])
            held_status = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertEqual(held_status.status, "run_held")

            # The environment is fixed: resume probes, transfers and observes in
            # a fresh Pass, past the retained configuration fault.
            self._advance(journey, store)
            events = Journey.open(journey.path).read()
            started = next(
                e for e in events if e["event_type"] == "InvestigationWorkItemStarted"
            )
            failure = next(
                e for e in events if e["event_type"] == "RuntimeConfigurationFailed"
            )
            # Both hang at the same causal head: the side leaf changed nothing.
            self.assertEqual(started["parent_ids"], failure["parent_ids"])
            # One download each of the VCF and the BAM, after the fix only.
            self.assertEqual(len(store.gets), 2)
            self.assertEqual(len(set(store.gets)), 2)
            self.assertEqual(
                [
                    e["payload"]["work_item"]["work_item_id"]
                    for e in events
                    if e["event_type"] == "InvestigationEvidenceRecorded"
                ],
                [first.authorized_work_items[0].work_item_id],
            )


if __name__ == "__main__":
    unittest.main()


class RecordedContactVersionTests(unittest.TestCase):
    """Each grant's GET rows read under the contract their version names (D3 A)."""

    GRANT = "investigation-authority:" + "a" * 64

    def _row(self, phase, payload, complete=None, iteration=1):
        from types import SimpleNamespace

        from bio_neb.recording import DomainContactRecord

        return SimpleNamespace(
            domain=DomainContactRecord(
                owner_event_id="grant-event",
                after_event_id="head-event",
                operation_id=self.GRANT,
                iteration=iteration,
                phase=phase,
                complete=complete,
                payload=payload,
            )
        )

    def _payload(self, version, attempt, **fields):
        return {
            "schema_version": f"bio-neb.investigation-object-contact/{version}",
            "authority_id": self.GRANT,
            "role": "VCF",
            "bucket": "run-bucket",
            "object_key": "execution/variants.vcf",
            "attempt": attempt,
            **fields,
        }

    def _closed(self, version):
        return self._row(
            "closed",
            {"schema_version": f"bio-neb.investigation-object-contact/{version}"},
            complete=True,
        )

    def test_recorded_result_rows_read_under_their_own_contract(self):
        from bio_neb.investigation.observation_inputs import (
            validate_investigation_contacts,
        )

        first_run = [
            self._row("started", {}),
            self._row(
                "record",
                self._payload(1, 1, classification="transient", byte_count=None),
            ),
            self._row(
                "record", self._payload(1, 2, classification="success", byte_count=9)
            ),
            self._closed(1),
        ]
        validate_investigation_contacts(first_run)
        for name, reason, rows in (
            (
                "out of order",
                "attempts are out of order",
                [
                    self._row(
                        "record",
                        self._payload(1, 2, classification="success", byte_count=9),
                    )
                ],
            ),
            (
                "after success",
                "attempts are out of order",
                [
                    self._row(
                        "record",
                        self._payload(1, 1, classification="success", byte_count=9),
                    ),
                    self._row(
                        "record",
                        self._payload(1, 2, classification="success", byte_count=9),
                    ),
                ],
            ),
            (
                "mixed versions",
                "changed its schema version",
                [
                    self._row(
                        "record",
                        self._payload(
                            1, 1, classification="transient", byte_count=None
                        ),
                    ),
                    self._row("record", self._payload(2, 2, stage="started")),
                ],
            ),
            (
                "closed with an open start",
                "changed its scoped closure",
                [
                    self._row("record", self._payload(2, 1, stage="started")),
                    self._closed(2),
                ],
            ),
        ):
            with self.subTest(name), self.assertRaisesRegex(ValueError, reason):
                validate_investigation_contacts(rows)

    def test_recorded_whole_object_attempt_rows_still_read(self):
        from bio_neb.investigation.observation_inputs import (
            validate_investigation_contacts,
        )

        # Recorded live Runs: one whole-object VCF GET, /2.
        validate_investigation_contacts(
            [
                self._row("started", {}),
                self._row("record", self._payload(2, 1, stage="started")),
                self._row(
                    "record",
                    self._payload(
                        2, 1, stage="ended", classification="success", byte_count=9
                    ),
                ),
                self._closed(2),
            ]
        )

    def _part(self, role, part, offset, size, attempt, stage, classification=None):
        return self._row(
            "record",
            {
                **self._payload(3, attempt, stage=stage),
                "role": role,
                "object_key": f"execution/{role}",
                "part": part,
                "offset": offset,
                "part_byte_count": size,
                "classification": classification,
                "byte_count": size if classification == "success" else None,
            },
            iteration={"VCF": 1, "BAM": 2}[role],
        )

    def test_part_rows_follow_each_file_and_part_in_order(self):
        from bio_neb.investigation.observation_inputs import (
            validate_investigation_contacts,
        )

        bam = [
            self._row("started", {}, iteration=2),
            self._part("BAM", 1, 0, 8, 1, "started"),
            self._part("BAM", 1, 0, 8, 1, "ended", "success"),
            self._part("BAM", 2, 8, 4, 1, "started"),
            self._part("BAM", 2, 8, 4, 1, "ended", "transient"),
            self._part("BAM", 2, 8, 4, 2, "started"),
            self._part("BAM", 2, 8, 4, 2, "ended", "success"),
            self._row(
                "closed",
                {"schema_version": "bio-neb.investigation-object-contact/3"},
                complete=True,
                iteration=2,
            ),
        ]
        vcf = [
            self._row("started", {}),
            self._part("VCF", 1, 0, 5, 1, "started"),
            self._part("VCF", 1, 0, 5, 1, "ended", "success"),
        ]
        # Each file is its own scope under the grant; their rows may follow
        # each other in one Pass journal.
        validate_investigation_contacts(vcf + bam)
        for name, rows in (
            (
                "a part skipped",
                [
                    self._part("BAM", 1, 0, 8, 1, "started"),
                    self._part("BAM", 1, 0, 8, 1, "ended", "success"),
                    self._part("BAM", 3, 16, 4, 1, "started"),
                ],
            ),
            (
                "a gap before the next part",
                [
                    self._part("BAM", 1, 0, 8, 1, "started"),
                    self._part("BAM", 1, 0, 8, 1, "ended", "success"),
                    self._part("BAM", 2, 9, 4, 1, "started"),
                ],
            ),
            (
                "a retry of a finished part",
                [
                    self._part("BAM", 1, 0, 8, 1, "started"),
                    self._part("BAM", 1, 0, 8, 1, "ended", "success"),
                    self._part("BAM", 1, 0, 8, 2, "started"),
                ],
            ),
            (
                "the next part after a failure",
                [
                    self._part("BAM", 1, 0, 8, 1, "started"),
                    self._part("BAM", 1, 0, 8, 1, "ended", "transient"),
                    self._part("BAM", 2, 8, 4, 1, "started"),
                ],
            ),
        ):
            with (
                self.subTest(name),
                self.assertRaisesRegex(ValueError, "attempts are out of order"),
            ):
                validate_investigation_contacts(rows)

    def _opening(self, stage="vcf_materialization", iteration=1, part_seconds=None):
        from bio_neb.failure import FailureSurface, limits_for

        limits = limits_for(FailureSurface(scope="stage", name=stage)).model_dump(
            mode="json"
        )
        if part_seconds is not None:
            limits["effective"]["part_seconds"] = part_seconds
        return self._row(
            "record",
            {"schema_version": "bio-neb.investigation-part-limits/1", "limits": limits},
            iteration=iteration,
        )

    def _caused(self, attempt, stage, classification=None, cause=None, part=1):
        return self._row(
            "record",
            {
                **self._payload(4, attempt, stage=stage),
                "part": part,
                "offset": 0,
                "part_byte_count": 5,
                "classification": classification,
                "byte_count": 5 if classification == "success" else None,
                "cause": None if cause is None else {"kind": cause},
            },
        )

    def _lookup(self, lookup, stage, outcome=None, role="VCF"):
        return self._row(
            "record",
            {
                "schema_version": "bio-neb.investigation-credential-contact/1",
                "authority_id": self.GRANT,
                "role": role,
                "lookup": lookup,
                "stage": stage,
                "outcome": outcome,
            },
        )

    def test_cause_rows_open_with_limits_and_count_lookups(self):
        from bio_neb.investigation.observation_inputs import (
            validate_investigation_contacts,
        )

        # The scope opens with its stage's part limits, counts the file's
        # credential lookups, and every part end names its cause.
        scope = [
            self._opening(),
            self._lookup(1, "started"),
            self._lookup(1, "ended", "credential_resolved"),
            self._caused(1, "started"),
            self._caused(1, "ended", "transient", "deadline"),
            self._caused(2, "started"),
            self._caused(2, "ended", "success", "success"),
            self._closed(4),
        ]
        validate_investigation_contacts(scope)
        # A lookup refused before any part closes its scope with no part.
        validate_investigation_contacts(
            [
                self._opening(),
                self._lookup(1, "started"),
                self._lookup(1, "ended", "credential_unavailable"),
                self._closed(4),
            ]
        )
        for name, reason, rows in (
            (
                "an end without its cause",
                "cause belongs to its end",
                [*scope[:4], self._caused(1, "ended", "transient")],
            ),
            (
                "a success ended by its deadline",
                "cause belongs to its end",
                [*scope[:4], self._caused(1, "ended", "success", "deadline")],
            ),
            (
                "parts without the scope's limits",
                "changed its scoped opening",
                [self._caused(1, "started")],
            ),
            (
                "another stage's limits",
                "changed its scoped opening",
                [self._opening(stage="bam_materialization")],
            ),
            (
                "limits that disagree with their layers",
                "disagree with the stage",
                [self._opening(part_seconds=30)],
            ),
            (
                "a lookup inside a part attempt",
                "lookups are out of order",
                [
                    self._opening(),
                    self._caused(1, "started"),
                    self._lookup(1, "started"),
                ],
            ),
            (
                "a part inside a lookup",
                "attempts are out of order",
                [
                    self._opening(),
                    self._lookup(1, "started"),
                    self._caused(1, "started"),
                ],
            ),
            (
                "closed with a lookup open",
                "changed its scoped closure",
                [self._opening(), self._lookup(1, "started"), self._closed(4)],
            ),
            (
                "another file's lookup",
                "changed its grant",
                [self._opening(), self._lookup(1, "started", role="BAM")],
            ),
            (
                "another file's parts under this file's limits and lookup",
                "changed its grant",
                [
                    *scope[:3],
                    self._row(
                        "record",
                        {
                            **self._caused(1, "started").domain.payload,
                            "role": "BAM",
                            "object_key": "execution/aligned.bam",
                        },
                    ),
                ],
            ),
            (
                "a lookup in a scope without causes",
                "changed its schema version",
                [self._row("started", {}), self._lookup(1, "started")],
            ),
        ):
            with self.subTest(name), self.assertRaisesRegex(ValueError, reason):
                validate_investigation_contacts(rows)
