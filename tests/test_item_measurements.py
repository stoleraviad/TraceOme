"""A Run measures its own health and each item's coverage, never a decision (#414).

The default composition's first table-stage step runs the Run's probed samtools
over its local BAM (a real tiny BAM, ``tests.tiny_bam``): one health fact and one
coverage fact per item, folded once into the item evidence, which Finalization
and the Grounded report cite. A failed measurement asks its Review; ``abort``
records it not measured and the Run reaches the same decisions.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bio_neb.application.advancement import (
    advance,
    review_response_from_json,
    use_run_advancement_dependencies,
)
from bio_neb.application.queries import ObserveRun, observe
from bio_neb.failure import StageLimitOverrides
from bio_neb.investigation.local_bcftools import LocalBcftoolsAdapter
from bio_neb.investigation.qualification_origins import (
    DepthBelowMinimum,
    ObservationNotInferred,
    ObservationNotInferredDepthMeasured,
)
from bio_neb.journey import Journey
from bio_neb.reporting.finalization_commands import (
    ApproveFinalizationCommand,
    approve_finalization,
)
from bio_neb.reporting.report_commands import (
    PublishGroundedReportCommand,
    publish_grounded_report,
)
from tests.observation_input_fakes import RUN_VCF, TABLE_READING_ENVIRON, RunOutputStore
from tests.run_prerequisites import RunPrerequisites
from tests.sequential_investigation_support import (
    approved_collection_observation,
    collection_request,
)
from tests.tiny_bam import COVERED_POSITION, READ_COUNT
from tests.vcf_lookup_fakes import TableReaderModel

# The Run's VCF here: the offline APOE call (T>C, a transition) and one A>C
# transversion far from every item, so Ti/Tv is 1 / 1 = 1.00.
MEASURED_VCF = RUN_VCF + b"NC_000019.10\t45000000\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\n"


class RecordingAdapter(LocalBcftoolsAdapter):
    """The Run's local process adapter, recording each measuring process's limit."""

    def __init__(self) -> None:
        super().__init__()
        self.limits: list[tuple[str, str, int]] = []

    def run(self, argv, *, timeout_seconds, stdout_path=None):
        tool = Path(argv[0]).name
        if tool == "bio-neb-samtools" or argv[1] == "stats":
            self.limits.append((tool, argv[1], timeout_seconds))
        return super().run(
            argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
        )


class SamtoolsTimesOut(LocalBcftoolsAdapter):
    """The Run's local process adapter whose samtools runs past its limit while
    ``failing``; bcftools runs as it does."""

    def __init__(self) -> None:
        super().__init__()
        self.failing = True
        self.samtools_calls = 0

    def run(self, argv, *, timeout_seconds, stdout_path=None):
        if Path(argv[0]).name == "bio-neb-samtools":
            self.samtools_calls += 1
            if self.failing:
                raise subprocess.TimeoutExpired(argv, timeout_seconds)
        return super().run(
            argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
        )


def _texts(qualifications):
    """Each qualification's sentence (#448 D2: the basis keeps its ids too)."""
    return tuple(item.text for item in qualifications)


class MeasuredRunTests(RunPrerequisites, unittest.TestCase):
    def _approved(self, root: Path):
        _, journey, _, _, _ = approved_collection_observation(
            root,
            cardinality=3,
            approve_count=1,
            forked=True,
            parent_file=self.prerequisite(
                "executed",
                investigation_request=collection_request(3),
                output_payloads={"VCF": MEASURED_VCF},
            ).journey_file,
        )
        before = observe(ObserveRun(journey_file=str(journey.path)))
        request = before.investigation.pending_request
        command = review_response_from_json(
            before,
            request.request_event_id,
            json.dumps(
                {
                    "action": "approve",
                    "proposal_id": request.proposal.proposal_id,
                    "work_item_ids": list(before.investigation.pending_work_item_ids),
                }
            ),
        )
        return journey, command

    def _advance(self, command, adapter=None):
        with use_run_advancement_dependencies(
            environ=TABLE_READING_ENVIRON,
            investigation_model_factory=TableReaderModel().model,
            investigation_vcf_source_factory=RunOutputStore(vcf=MEASURED_VCF),
            investigation_observation_adapter=adapter,
        ):
            return advance(command)

    def _answer_health(self, journey: Journey, action: str, adapter):
        observation = observe(ObserveRun(journey_file=str(journey.path)))
        self.assertEqual(observation.status, "run_health_review_pending")
        (request,) = observation.run_health.requests
        return self._advance(
            review_response_from_json(
                observation,
                request.request_event_id,
                json.dumps(
                    {
                        "request_id": request.request.request_id,
                        "action": action,
                        "by": "researcher:owner",
                    }
                ),
            ),
            adapter,
        )

    def _finalize(self, journey: Journey, completed):
        self.assertEqual(completed.status, "finalization_review_pending")
        pending = completed.finalization.pending_request
        approved = approve_finalization(
            ApproveFinalizationCommand(
                journey_file=str(journey.path),
                request_event_id=pending.request_event_id,
                basis_digest=pending.basis.basis_digest,
                head_event_id=pending.request_event_id,
            )
        )
        report = publish_grounded_report(
            PublishGroundedReportCommand(
                journey_file=str(journey.path),
                report_basis_digest=approved.report_basis.report_basis_digest,
                head_event_id=Journey.open(journey.path).read()[-1]["event_id"],
            )
        ).report
        return pending.basis, report, Journey.open(journey.path).read()

    def test_the_default_composition_measures_the_run_and_reports_it(self) -> None:
        adapter = RecordingAdapter()
        with tempfile.TemporaryDirectory() as temporary:
            journey, command = self._approved(Path(temporary).resolve())
            # Run health's own StageLimits declaration bounds each process (SP3).
            with mock.patch.dict(
                "bio_neb.failure._faults.STAGE_LIMITS",
                {"run_health": StageLimitOverrides(process_seconds=1234)},
            ):
                completed = self._advance(command, adapter)
            basis, report, events = self._finalize(journey, completed)
            health_dir = journey.run_directory / "investigation" / "health"
            manifests = list(health_dir.glob("run-*/run-health-manifest.json"))
            self.assertEqual(len(manifests), 1)
            manifest = manifests[0].read_bytes()
            # The BAM measured is the one the grant's download published (SP2).
            receipt = json.loads(
                (
                    journey.run_directory
                    / "investigation"
                    / "bam"
                    / "current-bam-acquisition-receipt.json"
                ).read_bytes()
            )
            self.assertEqual(
                json.loads(manifest)["bam"]["path"],
                receipt["materialized_bam"]["path"],
            )

        self.assertEqual(
            {(tool, sub) for tool, sub, _ in adapter.limits},
            {
                ("bio-neb-samtools", "flagstat"),
                ("bio-neb-samtools", "coverage"),
                ("bio-neb-samtools", "index"),
                ("bio-neb-samtools", "idxstats"),
                ("bio-neb-samtools", "depth"),
                ("bio-neb-bcftools", "stats"),
            },
        )
        self.assertEqual({seconds for *_, seconds in adapter.limits}, {1234})
        kinds = [event["event_type"] for event in events]
        (health_event,) = [e for e in events if e["event_type"] == "RunHealthMeasured"]
        coverage_events = [
            e for e in events if e["event_type"] == "InvestigationItemCoverageRecorded"
        ]
        # The first table-stage step: health, then each item, before any window.
        first_window = kinds.index("InvestigationItemWindowRecorded")
        self.assertLess(kinds.index("RunHealthMeasured"), first_window)
        self.assertLess(
            max(
                index
                for index, kind in enumerate(kinds)
                if kind == "InvestigationItemCoverageRecorded"
            ),
            first_window,
        )
        items = [item.work_item_id for item in basis.collection_item_states]
        self.assertEqual(
            {e["payload"]["work_item_id"] for e in coverage_events}, set(items)
        )
        # The tiny BAM's own numbers, as samtools states them, and its manifest.
        health = basis.run_health
        assert health is not None
        numbers = health.health.measurement
        self.assertEqual(health.health_event_id, health_event["event_id"])
        self.assertEqual(
            (numbers.reads, numbers.mapped_percent, numbers.covered_contigs),
            (READ_COUNT, 100.0, 1),
        )
        # Ti/Tv through the one VCF reader: one transition, one transversion.
        self.assertEqual((numbers.ti_tv, numbers.ti_tv_reason), (1.0, None))
        self.assertEqual(numbers.manifest_byte_count, len(manifest))
        self.assertEqual(numbers.manifest_sha256, hashlib.sha256(manifest).hexdigest())
        # Each item is either measured (a depth line) or a typed not-measured gap,
        # read from its folded coverage field only.
        fold = next(
            e for e in events if e["event_type"] == "InvestigationItemEvidenceCollected"
        )
        coverage = {
            item["work_item_id"]: item["coverage"] for item in fold["payload"]["items"]
        }
        lines = {line.work_item_id: line for line in health.items}
        # Every started item sits on a sequence of the tiny BAM: each is measured,
        # read from its folded coverage field, and its own L1 says so (ST5).
        self.assertEqual(set(lines), set(items))
        for work_item_id in items:
            field = coverage[work_item_id]
            line = lines[work_item_id]
            self.assertEqual(field["status"], "measured")
            self.assertEqual(line.depth, field["measurement"]["depth"])
            self.assertEqual(line.coverage_event_id, field["coverage_event_id"])
            own = [
                q.origin for q in basis.limitations if work_item_id in q.work_item_ids
            ]
            self.assertIn(ObservationNotInferredDepthMeasured(), own)
            self.assertNotIn(ObservationNotInferred(), own)
        # Each measured span's depth as the tiny BAM holds it: the APOE base
        # under all twelve reads meets the minimum; a span off the reads is zero
        # deep, below it, with no unsure flag (no read nearby).
        spans = {
            e["payload"]["work_item_id"]: (
                e["payload"]["span_start"],
                e["payload"]["span_end"],
            )
            for e in coverage_events
        }
        self.assertTrue(lines)
        for line in lines.values():
            start, end = spans[line.work_item_id]
            limitations = [
                q.origin
                for q in basis.limitations
                if q.work_item_ids == (line.work_item_id,)
            ]
            if start <= COVERED_POSITION <= end:
                self.assertEqual((line.depth, line.batch_flags), (READ_COUNT, ()))
                self.assertFalse(
                    any(isinstance(o, DepthBelowMinimum) for o in limitations)
                )
            else:
                self.assertEqual((line.depth, line.batch_flags), (0, ()))
                self.assertIn(DepthBelowMinimum(depth=0, minimum=10), limitations)
        self.assertIn(
            READ_COUNT, [line.depth for line in lines.values()], "the APOE item"
        )
        self.assertEqual(report.run_health, health)

    def test_a_failed_measurement_asks_its_review_and_abort_changes_no_decision(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journey, command = self._approved(Path(temporary).resolve())
            measured, _, _ = self._finalize(journey, self._advance(command))
        adapter = SamtoolsTimesOut()
        with tempfile.TemporaryDirectory() as temporary:
            journey, command = self._approved(Path(temporary).resolve())
            held = self._advance(command, adapter)
            # The Review holds the table stage: no window was read.
            self.assertEqual(held.status, "run_health_review_pending")
            kinds = [e["event_type"] for e in Journey.open(journey.path).read()]
            self.assertNotIn("InvestigationItemWindowRecorded", kinds)
            self.assertIn("RunHealthMeasurementFailed", kinds)
            unmeasured, _, events = self._finalize(
                journey, self._answer_health(journey, "abort", adapter)
            )

        (health,) = [e for e in events if e["event_type"] == "RunHealthMeasured"]
        self.assertIsNone(health["payload"]["measurement"])
        self.assertEqual(
            health["payload"]["not_measured"]["code"], "measurement_failed"
        )
        # The failure's typed code and its process, as the Review recorded them.
        self.assertEqual(
            health["payload"]["not_measured"]["detail"],
            "timeout: bio-neb-samtools flagstat ran past 3600 s",
        )
        (failure,) = [
            e for e in events if e["event_type"] == "RunHealthMeasurementFailed"
        ]
        self.assertEqual(failure["payload"]["code"], "timeout")
        for item in unmeasured.collection_item_states:
            self.assertIn(
                "measurement_failed",
                " ".join(
                    q.text
                    for q in unmeasured.gaps
                    if q.work_item_ids == (item.work_item_id,)
                ),
            )
        self.assertNotIn(ObservationNotInferredDepthMeasured(), _origins(unmeasured))
        self.assertIn(ObservationNotInferred(), _origins(unmeasured))
        # No decision differs from the measured Run.
        for field in ("outcome", "collection_status"):
            self.assertEqual(getattr(measured, field), getattr(unmeasured, field))
        self.assertEqual(
            [(s.status, s.outcome) for s in measured.collection_item_states],
            [(s.status, s.outcome) for s in unmeasured.collection_item_states],
        )
        self.assertEqual(
            [a.status for a in measured.admissions],
            [a.status for a in unmeasured.admissions],
        )

        def decisions(value):
            return sorted(
                (m.member_role, m.admitted, m.included)
                for m in value.report_membership
                if m.member_role != "accounting"
            )

        self.assertEqual(decisions(measured), decisions(unmeasured))

    def test_retry_measures_the_run_again(self) -> None:
        adapter = SamtoolsTimesOut()
        with tempfile.TemporaryDirectory() as temporary:
            journey, command = self._approved(Path(temporary).resolve())
            self._advance(command, adapter)
            adapter.failing = False
            basis, _, events = self._finalize(
                journey, self._answer_health(journey, "retry", adapter)
            )
        kinds = [e["event_type"] for e in events]
        self.assertEqual(
            [k for k in kinds if k.startswith("RunHealth")],
            [
                "RunHealthMeasurementFailed",
                "RunHealthReviewRequested",
                "RunHealthReviewAnswered",
                "RunHealthMeasured",
            ],
        )
        assert basis.run_health is not None
        self.assertEqual(basis.run_health.health.measurement.reads, READ_COUNT)


def _origins(basis):
    return [q.origin for q in basis.limitations]


class DepthLimitationTests(unittest.TestCase):
    def test_depth_below_the_minimum_states_its_typed_origin(self) -> None:
        # The audit sentence, read from the origin the basis keeps.
        self.assertEqual(
            DepthBelowMinimum(depth=4, minimum=10).audit("work-item:x"),
            "Work item work-item:x: read depth 4 at the least-covered base of its"
            " span, below the minimum of 10.",
        )


if __name__ == "__main__":
    unittest.main()
