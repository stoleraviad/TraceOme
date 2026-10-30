"""Runs taken through Investigation to a published Grounded report (#108, #109).

A test forks its own child of the executed parent for one researcher request,
drives it with the coordinator's ``advance`` and the native ``answer`` command,
and publishes the report through the native Finalization approval, with offline
model, NCBI and object-store doubles. The report's own claims stay with each
test module.
"""

import contextlib
import io
import json
from pathlib import Path

from bio_neb.journey import Journey
from bio_neb.recording import read_evidence, review_head
from tests.observation_input_fakes import (
    RUN_VCF,
    TABLE_READING_ENVIRON,
    RunOutputStore,
)
from tests.run_prerequisites import RunPrerequisites, State
from tests.sequential_investigation_support import (
    IssueSourceResponses,
    SequentialAdaptiveModel,
    SourceResponsesOnly,
)
from tests.terminal_journey_fakes import forked_child


def request_file(*rsids: str) -> dict:
    """The researcher's request document, one request per rsID, in order."""
    return {
        "schema_version": "bio-neb.investigation-request/2",
        "asserted_by": "researcher:owner",
        "requests": [
            {"request": f"Does HG001 carry {rsid}?", "target": {"rsid": rsid}}
            for rsid in rsids
        ],
    }


class ItemStateSources(IssueSourceResponses):
    """dbSNP: retained records where given, one rsID it does not know, and the
    sequential records for rs429358 and rs7412."""

    def fetch_refsnp(self, rsid):  # noqa: ANN001, ANN201
        if rsid in self.refsnp_bodies:
            return SourceResponsesOnly.fetch_refsnp(self, rsid)
        return super().fetch_refsnp(rsid)


def item_state_dependencies(
    store: RunOutputStore, bodies: dict | None = None, *, report_model_factory=None
):
    """The offline model, NCBI and object-store doubles these Runs use; a
    published report is explained by default (#460), by the scripted explainer
    unless the test binds its own."""
    from bio_neb.application.advancement import use_run_advancement_dependencies
    from tests.explainer_fakes import ExplainerModel

    return use_run_advancement_dependencies(
        environ=TABLE_READING_ENVIRON,
        investigation_model_factory=SequentialAdaptiveModel().model,
        investigation_source_adapter_factory=lambda: ItemStateSources(
            refsnp_bodies=bodies or {}
        ),
        investigation_vcf_source_factory=store,
        report_model_factory=report_model_factory or ExplainerModel().model,
    )


def native_answer(
    journey_file: Path, request_event_id: str, response: dict
) -> tuple[int, str]:
    """One native ``answer`` command alone: its exit code and stderr."""
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
    return code, error.getvalue()


class ItemStateRuns(RunPrerequisites):
    """The native steps from an executed parent to a published report."""

    def child(
        self, root: Path, request: dict, vcf: bytes = RUN_VCF, **planning
    ) -> Journey:
        parent = self.prerequisite(
            "executed",
            investigation_request=request,
            output_payloads={"VCF": vcf},
            **planning,
        ).journey_file
        return forked_child(root, parent_file=parent)

    def dependencies(
        self,
        store: RunOutputStore,
        bodies: dict | None = None,
        *,
        report_model_factory=None,
    ):
        return item_state_dependencies(
            store, bodies, report_model_factory=report_model_factory
        )

    def answer(self, journey: Journey, request_event_id: str, response: dict):
        """Answer through the native CLI, as the researcher does."""
        from bio_neb.application.advancement import ObserveRun, observe

        code, error = native_answer(journey.path, request_event_id, response)
        self.assertEqual(code, 0, error)
        return observe(ObserveRun(journey_file=str(journey.path)))

    def approve_issue(self, journey: Journey, review):
        return self.answer(
            journey,
            review.request_event_id,
            {
                "action": "approve",
                "collection_member_id": review.request.question.collection_member_id,
                "revision_number": 1,
                "asserted_by": "researcher:owner",
            },
        )

    def basis(self, journey: Journey):
        from bio_neb.application.queries import read_completion

        return read_completion(Journey.open(journey.path)).finalization.bases[-1]

    def publish(self, journey: Journey, basis, *, explainer=None):
        """Approve the basis through the native answer; the coordinator publishes
        and explains the report by default (#460), with ``explainer`` or the
        scripted one."""
        from bio_neb.application.advancement import use_run_advancement_dependencies
        from bio_neb.application.queries import read_completion
        from bio_neb.reporting.grounded_report import project_grounded_report
        from tests.explainer_fakes import ExplainerModel

        pending = read_completion(Journey.open(journey.path)).finalization
        with use_run_advancement_dependencies(
            environ=TABLE_READING_ENVIRON,
            report_model_factory=(explainer or ExplainerModel()).model,
        ):
            published = self.answer(
                journey,
                pending.pending_request.request_event_id,
                {
                    "action": "approve",
                    "basis_digest": basis.basis_digest,
                    "head_event_id": review_head(
                        read_evidence(Journey.open(journey.path))
                    )["event_id"],
                },
            )
        self.assertEqual(published.status, "grounded_report_published")
        report = project_grounded_report(read_evidence(Journey.open(journey.path)))
        return report.report

    def readable(self, journey: Journey, report_digest: str) -> str:
        """The native, contact-free readable report (#460)."""
        from bio_neb.cli import main

        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = main(
                [
                    "report-readable",
                    "--journey-file",
                    str(journey.path),
                    "--report-digest",
                    report_digest,
                ]
            )
        self.assertEqual((code, error.getvalue()), (0, ""), output.getvalue())
        return output.getvalue()

    def reopen(self, journey: Journey, report_digest: str) -> str:
        from bio_neb.cli import main

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                [
                    "grounded-report-reopen",
                    "--journey-file",
                    str(journey.path),
                    "--report-digest",
                    report_digest,
                ]
            )
        self.assertEqual(code, 0)
        return output.getvalue()


# The two-request Run the read guards share, planned by its executed parent; each
# state is built through the coordinator and restored in place before use.
_EXECUTED_CHOICES = {
    "investigation_request": request_file("rs429358", "rs7412"),
    "output_payloads": {"VCF": RUN_VCF},
}


def _investigation_review_pending(root: Path, parent_file: Path):
    """A Fork child of the executed parent at its two-member Investigation Review;
    value: that Review's request and its approval."""
    from bio_neb.application.advancement import ResumeRun, advance
    from bio_neb.application.reviews import pending_reviews

    journey = forked_child(root, parent_file=parent_file)
    with item_state_dependencies(RunOutputStore()):
        (review,) = pending_reviews(advance(ResumeRun(journey_file=str(journey.path))))
    return journey.path, (
        review.request_event_id,
        {"action": "approve", "proposal_id": review.request.proposal.proposal_id},
    )


def _finalization_pending(root: Path, parent_file: Path):
    """The same child once approved and observed, at its Finalization Review;
    value: that Review's request and its approval."""
    from bio_neb.application.queries import read_completion
    from bio_neb.recording import review_head

    journey_file, (request_id, response) = _investigation_review_pending(
        root, parent_file
    )
    with item_state_dependencies(RunOutputStore()):
        code, error = native_answer(journey_file, request_id, response)
    assert code == 0, error
    owner = Journey.open(journey_file)
    pending = read_completion(owner).finalization
    return journey_file, (
        pending.pending_request.request_event_id,
        {
            "action": "approve",
            "basis_digest": pending.bases[-1].basis_digest,
            "head_event_id": review_head(read_evidence(owner))["event_id"],
        },
    )


INVESTIGATION_REVIEW_PENDING = State(
    "investigation-review-pending",
    _investigation_review_pending,
    forks_from="executed",
    forks_from_choices=_EXECUTED_CHOICES,
)
FINALIZATION_PENDING = State(
    "finalization-pending",
    _finalization_pending,
    forks_from="executed",
    forks_from_choices=_EXECUTED_CHOICES,
)
