"""One exporter freezes any closed-Pass prefix of a retained Journey as an inert fixture.

A scripted coordinator-built Run supplies every closed boundary: the executed
parent's after-Execution point and, in its Fork child driven by ``advance`` and
the native ``answer`` command, the Investigation Review, the point before
Finalization, the Finalization Review and the published report. These tests
prove export integrity and the cut rule, never live behaviour.
"""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from bio_neb.devtools import reached_fixture_export as export
from bio_neb.devtools.parabricks_fixture_export import FixtureExportRefused
from tests.observation_input_fakes import RUN_VCF, RunOutputStore
from tests.run_prerequisites import RunPrerequisites, State

_REQUEST = {
    "schema_version": "bio-neb.investigation-request/2",
    "asserted_by": "researcher:owner",
    "requests": [
        {"request": f"Does HG001 carry {rsid}?", "target": {"rsid": rsid}}
        for rsid in ("rs429358", "rs7412")
    ],
}
_PARENT = {"investigation_request": _REQUEST, "output_payloads": {"VCF": RUN_VCF}}


def _answer(journey_file: Path, request_event_id: str, response: dict) -> None:
    """Answer through the native CLI, as the researcher does."""
    from bio_neb.cli import main

    response_file = journey_file.parent / f"answer-{request_event_id}.json"
    response_file.write_text(json.dumps(response), encoding="utf-8")
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()) as error,
    ):
        code = main(
            [
                "answer",
                "--journey-file",
                str(journey_file),
                "--request-event-id",
                request_event_id,
                "--response-file",
                str(response_file),
            ]
        )
    if code != 0:
        raise AssertionError(error.getvalue())


def _published(root: Path, parent_file: Path) -> tuple[Path, None]:
    """A Fork child taken through Investigation to a published Grounded report."""
    from bio_neb.application.advancement import (
        ResumeRun,
        advance,
        use_run_advancement_dependencies,
    )
    from bio_neb.application.queries import read_completion
    from bio_neb.application.reviews import pending_reviews
    from bio_neb.journey import Journey
    from bio_neb.recording import read_evidence, review_head
    from tests.explainer_fakes import explained_by_default
    from tests.observation_input_fakes import TABLE_READING_ENVIRON
    from tests.sequential_investigation_support import (
        IssueSourceResponses,
        SequentialAdaptiveModel,
    )
    from tests.terminal_journey_fakes import forked_child

    journey_file = forked_child(root, parent_file=parent_file).path
    with use_run_advancement_dependencies(
        environ=TABLE_READING_ENVIRON,
        investigation_model_factory=SequentialAdaptiveModel().model,
        investigation_source_adapter_factory=IssueSourceResponses,
        investigation_vcf_source_factory=RunOutputStore(),
    ):
        (review,) = pending_reviews(advance(ResumeRun(journey_file=str(journey_file))))
        _answer(
            journey_file,
            review.request_event_id,
            {"action": "approve", "proposal_id": review.request.proposal.proposal_id},
        )
    finalization = read_completion(Journey.open(journey_file)).finalization
    # The published report is explained by default.
    with explained_by_default():
        _answer(
            journey_file,
            finalization.pending_request.request_event_id,
            {
                "action": "approve",
                "basis_digest": finalization.bases[-1].basis_digest,
                "head_event_id": review_head(read_evidence(Journey.open(journey_file)))[
                    "event_id"
                ],
            },
        )
    return journey_file, None


_PUBLISHED = State(
    "published-report",
    _published,
    forks_from="executed",
    forks_from_choices=_PARENT,
)


def _closed_points(journey_file: Path) -> list[tuple[str, str]]:
    """Each PassClosed this file holds, with its disposition, in order."""
    return [
        (event["event_id"], event["payload"]["disposition"])
        for event in map(json.loads, journey_file.read_bytes().splitlines())
        if event["event_type"] == "PassClosed"
    ]


class ReachedFixtureExportTests(RunPrerequisites, unittest.TestCase):
    def setUp(self):
        self.parent = self.prerequisite("executed", **_PARENT).journey_file
        self.child = self.prerequisite(_PUBLISHED).journey_file

    def _compile(self, journey_file, point_event_id=None, **parameters):
        return export.compile_reached_fixture(
            journey_file,
            point_event_id=point_event_id,
            source_commit="c" * 40,
            source_ticket=215,
            evidence_kind="synthetic",
            **parameters,
        )

    def test_the_after_execution_point_exports_its_checked_view_and_accounting(self):
        content = self._compile(self.parent)

        fixture = export.load_reached_fixture(content)
        (scenario,) = fixture.scenarios
        journey_bytes = self.parent.read_bytes()
        (point, disposition) = _closed_points(self.parent)[-1]
        self.assertFalse(fixture.fixture_grants_live_authority)
        self.assertFalse(scenario.fixture_grants_live_authority)
        self.assertEqual(
            (scenario.journey.byte_count, scenario.journey.sha256),
            (len(journey_bytes), hashlib.sha256(journey_bytes).hexdigest()),
        )
        self.assertEqual(
            (scenario.prefix.head_event_id, scenario.point_disposition),
            (point, disposition),
        )
        self.assertEqual(disposition, "completed")
        # The observation at this point is the verified Execution result itself.
        self.assertEqual(
            (scenario.view, scenario.frontier, set(scenario.projections)),
            ("execution_result", None, {"execution_result"}),
        )
        result = scenario.projections["execution_result"]
        self.assertEqual(scenario.status, result["status"])
        self.assertEqual(
            [item["role"] for item in result["verified_objects"]],
            ["aligned_bam", "VCF", "worker_receipt", "execution_manifest"],
        )
        self.assertIn("ExecutionHitlResponded", scenario.path_events)
        self.assertEqual(scenario.path_events[-1], "PassClosed")
        # Every Pass the prefix holds is accounted, the last one closing at the point.
        self.assertEqual(scenario.passes[-1].closed_event_id, point)
        self.assertTrue(all(item.journal_complete for item in scenario.passes))
        self.assertEqual(scenario.eligibility.seam, "not_computed")

    def test_each_closed_point_of_the_child_exports_the_view_it_reached(self):
        # The child's own closed Passes, in order: the Investigation Review, the
        # completed collection before Finalization, the Finalization Review and
        # the published report.
        expected = [
            (
                "waiting",
                "review_required",
                "investigation_review_pending",
                ["investigation"],
            ),
            ("completed", "checkpoint", "investigation_evidence_ready", []),
            (
                "waiting",
                "review_required",
                "finalization_review_pending",
                ["finalization"],
            ),
            ("completed", "report_published", "grounded_report_published", []),
        ]
        points = _closed_points(self.child)
        self.assertEqual(
            [disposition for _, disposition in points], [row[0] for row in expected]
        )
        for (point, _), (disposition, frontier, status, reviews) in zip(
            points, expected, strict=True
        ):
            with self.subTest(status=status):
                (scenario,) = export.load_reached_fixture(
                    self._compile(self.child, point)
                ).scenarios
                self.assertEqual(
                    (
                        scenario.prefix.head_event_id,
                        scenario.point_disposition,
                        scenario.view,
                        scenario.frontier,
                        scenario.status,
                    ),
                    (point, disposition, "run", frontier, status),
                )
                self.assertEqual(
                    [
                        review["kind"]
                        for review in scenario.projections["pending_reviews"]
                    ],
                    reviews,
                )
                # The inherited prefix is part of the reached timeline.
                self.assertEqual(scenario.path_events.count("RunForked"), 1)
                self.assertEqual(scenario.passes[-1].closed_event_id, point)
                # The parent's Passes are accounted as inherited, never as the child's.
                own = {
                    json.loads(line)["event_id"]
                    for line in self.child.read_bytes().splitlines()
                }
                self.assertEqual(
                    [item.inherited for item in scenario.passes],
                    [item.started_event_id not in own for item in scenario.passes],
                )
                self.assertTrue(scenario.passes[0].inherited)
                self.assertFalse(scenario.passes[-1].inherited)

    def test_the_run_at_its_execution_review_exports_that_review(self):
        journey_file = self.prerequisite("review-pending").journey_file

        (scenario,) = export.load_reached_fixture(self._compile(journey_file)).scenarios

        self.assertEqual(
            (scenario.view, scenario.status, scenario.point_disposition),
            ("run", "execution_hitl_pending", "waiting"),
        )
        self.assertEqual(
            [review["kind"] for review in scenario.projections["pending_reviews"]],
            ["execution"],
        )
        self.assertNotIn("ExecutionHitlResponded", scenario.path_events)

    def test_a_point_inside_an_open_pass_is_refused_with_the_fork_code(self):
        events = [json.loads(line) for line in self.child.read_bytes().splitlines()]
        inside = next(
            event["event_id"]
            for event in events
            if event["event_type"] == "InvestigationReviewRequested"
        )
        inherited = _closed_points(self.parent)[-1][0]
        for point, code in (
            (inside, "fork_point_not_closed"),
            ("00000000-0000-4000-8000-000000000000", "fork_point_unknown"),
            # The child's file holds only its own events; its parent's point
            # is exported from the parent's own file.
            (inherited, "fork_point_unknown"),
        ):
            with self.subTest(code=code):
                with self.assertRaises(FixtureExportRefused) as refused:
                    self._compile(self.child, point)
                self.assertEqual(refused.exception.reason, code)

    def test_only_the_selected_projections_are_retained(self):
        point = _closed_points(self.child)[0][0]

        (scenario,) = export.load_reached_fixture(
            self._compile(
                self.child,
                point,
                projections=("investigation_policy", "investigation_request"),
            )
        ).scenarios

        self.assertEqual(
            set(scenario.projections),
            {"investigation_policy", "investigation_request"},
        )
        for unknown in ((), ("execution_authority",), ("review",), ("passes",)):
            with self.subTest(projections=unknown):
                with self.assertRaises(FixtureExportRefused) as refused:
                    self._compile(self.child, point, projections=unknown)
                self.assertEqual(refused.exception.reason, "projection_unknown")

    def test_no_projection_retains_the_execution_authority(self):
        from bio_neb.application.queries import ObserveRun, observe

        authority = observe(
            ObserveRun(journey_file=str(self.child))
        ).execution_hitl.authorities[0]
        point = _closed_points(self.child)[0][0]

        content = self._compile(self.child, point, projections=export.PROJECTIONS)

        self.assertNotIn(b'"execution_authority"', content)
        self.assertNotIn(authority.authority_id.encode(), content)
        self.assertNotIn(authority.approval_event_id.encode(), content)

    def test_no_point_retains_an_authority_document_in_any_projection(self):
        # After the approvals, the Investigation grant sits inside several Run
        # view fields (the Investigation, its evidence, the basis, the report).
        from bio_neb.application.queries import ObserveRun, observe
        from bio_neb.investigation.investigation import (
            InvestigationAuthority,
            InvestigationReviewResponded,
        )
        from bio_neb.journey import canonical_json
        from bio_neb.parabricks.execution_proposal import (
            ExecutionAuthority,
            ExecutionHitlResponded,
        )

        kinds = (
            InvestigationAuthority,
            ExecutionAuthority,
            ExecutionHitlResponded,
            InvestigationReviewResponded,
        )

        def documents(value):
            """Every object that reads as a grant or a raw approval answer once
            dumped, as JSON reads it."""
            if isinstance(value, dict):
                for kind in kinds:
                    try:
                        yield kind.model_validate_json(canonical_json(value))
                    except ValueError:
                        pass
                for item in value.values():
                    yield from documents(item)
            elif isinstance(value, list):
                for item in value:
                    yield from documents(item)

        # The check finds an intact dumped grant of either kind and each raw
        # approval answer as the Journey recorded it.
        observed = observe(ObserveRun(journey_file=str(self.child)))
        events = [
            json.loads(line)
            for path in (self.parent, self.child)
            for line in path.read_bytes().splitlines()
        ]
        answers = [
            event["payload"]
            for event in events
            if event["event_type"]
            in {"ExecutionHitlResponded", "InvestigationReviewResponded"}
        ]
        controls = [
            observed.investigation.authority.model_dump(mode="json"),
            observed.execution_hitl.authorities[0].model_dump(mode="json"),
            *answers,
        ]
        self.assertGreaterEqual(len(answers), 2)
        self.assertEqual(len(list(documents({"held": controls}))), len(controls))
        for point, _ in _closed_points(self.child):
            with self.subTest(point=point):
                content = self._compile(
                    self.child, point, projections=export.PROJECTIONS
                )
                (scenario,) = json.loads(content)["scenarios"]
                self.assertEqual(list(documents(scenario["projections"])), [])

    def test_loading_refuses_an_inserted_grant_or_approval_answer(self):
        from bio_neb.application.queries import ObserveRun, observe

        observed = observe(ObserveRun(journey_file=str(self.child)))
        grant = observed.investigation.authority.model_dump(mode="json")
        answer = observed.execution_hitl.receipts[-1].model_dump(mode="json")
        self.assertEqual(answer["response"]["action"], "approve")
        (fact,) = [
            event["payload"]
            for event in map(json.loads, self.parent.read_bytes().splitlines())
            if event["event_type"] == "ExecutionHitlResponded"
        ]
        point = _closed_points(self.child)[1][0]
        content = self._compile(self.child, point, projections=("investigation",))
        export.load_reached_fixture(content)
        for name, document in (
            ("grant", grant),
            ("answer", answer),
            ("recorded answer fact", fact),
        ):
            with self.subTest(document=name):
                data = json.loads(content)
                data["scenarios"][0]["projections"]["investigation"]["inserted"] = [
                    document
                ]
                with self.assertRaisesRegex(ValueError, "approval-bearing"):
                    export.load_reached_fixture(json.dumps(data).encode())

    def test_every_run_view_field_is_a_projection_or_deliberately_excluded(self):
        from bio_neb.application.run_interaction import RunInteraction

        header = {
            "schema_version",
            "view",
            "frontier",
            "journey_id",
            "journey_file",
            "status",
        }
        self.assertEqual(
            set(RunInteraction.model_fields),
            header | set(export.PROJECTIONS[1:]) | set(export.EXCLUDED_FIELDS),
        )

    def test_eligibility_is_computed_only_by_the_seam_asked_for(self):
        (default,) = export.load_reached_fixture(self._compile(self.parent)).scenarios
        (reviewed,) = export.load_reached_fixture(
            self._compile(self.parent, eligibility="execution_contact_review")
        ).scenarios

        self.assertEqual(
            (
                default.eligibility.contact_review_status,
                default.eligibility.fixture_eligible,
            ),
            (None, None),
        )
        self.assertEqual(
            (reviewed.eligibility.seam, reviewed.eligibility.contact_review_status),
            ("execution_contact_review", "complete"),
        )
        self.assertIsInstance(reviewed.eligibility.fixture_eligible, bool)
        # The Contact review reads only an execution path.
        with self.assertRaises(FixtureExportRefused) as refused:
            self._compile(
                self.prerequisite("review-pending").journey_file,
                eligibility="execution_contact_review",
            )
        self.assertEqual(refused.exception.reason, "eligibility_seam_inapplicable")

    def test_export_is_byte_stable_and_leaves_the_journey_unchanged(self):
        before = self.child.read_bytes()
        point = _closed_points(self.child)[1][0]

        self.assertEqual(
            self._compile(self.child, point), self._compile(self.child, point)
        )
        self.assertEqual(self.child.read_bytes(), before)

    def test_loading_refuses_granted_authority_and_a_moved_point(self):
        content = self._compile(self.parent)
        for mutation in (
            "bundle_authority",
            "scenario_authority",
            "point",
            "projection",
        ):
            with self.subTest(mutation=mutation):
                data = json.loads(content)
                scenario = data["scenarios"][0]
                if mutation == "bundle_authority":
                    data["fixture_grants_live_authority"] = True
                elif mutation == "scenario_authority":
                    scenario["fixture_grants_live_authority"] = True
                elif mutation == "point":
                    scenario["passes"][-1]["closed_event_id"] = scenario["passes"][0][
                        "started_event_id"
                    ]
                else:
                    scenario["projections"]["execution_authority"] = []
                with self.assertRaises(ValueError):
                    export.load_reached_fixture(json.dumps(data).encode())

    def test_dev_command_publishes_exclusively_without_provider_contact(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "reached.json"
            point = _closed_points(self.child)[0][0]
            argv = [
                "export-reached",
                "--journey-file",
                str(self.child),
                "--point-event-id",
                point,
                "--projection",
                "pending_reviews",
                "--source-commit",
                "c" * 40,
                "--source-ticket",
                "215",
                "--evidence-kind",
                "synthetic",
                "--output",
                str(output),
            ]
            with (
                contextlib.redirect_stdout(io.StringIO()) as printed,
                patch("socket.socket", side_effect=AssertionError("network contact")),
            ):
                self.assertEqual(export.main(argv), 0)
                expected = self._compile(
                    self.child, point, projections=("pending_reviews",)
                )
                self.assertEqual(output.read_bytes(), expected)
                self.assertEqual(
                    json.loads(printed.getvalue())["sha256"],
                    hashlib.sha256(expected).hexdigest(),
                )
                before = b"retained different fixture"
                output.write_bytes(before)
                with self.assertRaises(FixtureExportRefused) as refused:
                    export.main(argv)
                self.assertEqual(refused.exception.reason, "output_conflict")
                self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
