"""An optional Agent explanation of an approved Grounded report (#109, #408, ADR 0038).

The researcher asks for an explanation of one published report through the
native ``report-explain`` command; the coordinator runs the explanation Agent
through the shared Attempt lifecycle. Each request of the report is explained by
its own member Attempt over its own rows, stating one claim per row that
admission compares with the row's recorded state; one combination Attempt then
writes the combined answer over the admitted members (owner, #408 parts 1-3).
The Agent reads only retained evidence the exact approved Report basis names.
The explanation is a separate immutable artifact with its own identity; the
deterministic report stays byte-identical whatever the explanation does.

Each scenario runs the actual callers (``advance``, the native ``answer``,
``report-explain`` and ``report-explanation-reopen`` commands) with offline
model, NCBI and object-store doubles. The explanation model is a script at the
HTTP layer: each member answers from the rows and the claim words it is sent,
and the combination from the members' sections it is sent.
"""

import contextlib
import copy
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.journey import Journey, canonical_json
from bio_neb.recording import read_evidence, review_head
from tests.explainer_fakes import (
    ExplainerModel,
    _sent_task,
    combination_draft,
    member_draft,
)
from tests.item_state_runs import (
    ItemStateRuns,
    item_state_dependencies,
    native_answer,
    request_file,
)
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.observation_input_fakes import RUN_VCF, RunOutputStore
from tests.refsnp_retained_bodies import RS4149056
from tests.run_prerequisites import State
from tests.sequential_investigation_support import NOT_FOUND
from tests.terminal_journey_fakes import forked_child

RS429358_TARGET = "NC_000019.10:44908684 T>C"
RS7412_TARGET = "NC_000019.10:44908822 T>C"


def _observed_report(root: Path, parent_file: Path) -> tuple[Path, str]:
    """rs429358 called 0/1 and rs7412 with no record, reported (#339 flow A),
    through the native answers; value: the report's digest."""
    from bio_neb.application.advancement import ResumeRun, advance
    from bio_neb.application.queries import read_completion
    from bio_neb.application.reviews import pending_reviews

    journey = forked_child(root, parent_file=parent_file)
    with item_state_dependencies(RunOutputStore()):
        (review,) = pending_reviews(advance(ResumeRun(journey_file=str(journey.path))))
        code, error = native_answer(
            journey.path,
            review.request_event_id,
            {"action": "approve", "proposal_id": review.request.proposal.proposal_id},
        )
        assert code == 0, error
    basis = read_completion(Journey.open(journey.path)).finalization.bases[-1]
    report = publish_unexplained(journey, basis)
    return journey.path, report.report_digest


def publish_unexplained(journey: Journey, basis):
    """Approve and publish through their own commands, not the coordinator, so
    a test's request is the report's first: the coordinator explains every
    report it publishes by default (#460 Q5 b)."""
    from bio_neb.application.queries import read_completion
    from bio_neb.reporting.finalization_commands import (
        ApproveFinalizationCommand,
        approve_finalization,
    )
    from bio_neb.reporting.report_commands import (
        PublishGroundedReportCommand,
        publish_grounded_report,
    )

    request = read_completion(Journey.open(journey.path)).finalization
    approved = approve_finalization(
        ApproveFinalizationCommand(
            journey_file=str(journey.path),
            request_event_id=request.pending_request.request_event_id,
            basis_digest=basis.basis_digest,
            head_event_id=review_head(read_evidence(Journey.open(journey.path)))[
                "event_id"
            ],
        )
    )
    return publish_grounded_report(
        PublishGroundedReportCommand(
            journey_file=str(journey.path),
            report_basis_digest=approved.report_basis.report_basis_digest,
            head_event_id=Journey.open(journey.path).read()[-1]["event_id"],
        )
    ).report


# The published two-request report the explanation guards share.
_OBSERVED_REPORT = State(
    "observed-report",
    _observed_report,
    forks_from="executed",
    forks_from_choices={
        "investigation_request": request_file("rs429358", "rs7412"),
        "output_payloads": {"VCF": RUN_VCF},
        "failed_turns": 2,
    },
)


# Whole-history copies in one fixture ``report-explain`` (three Attempts): 25
# before the loop's admission session and Attempt request count read the owner's
# shared evidence; 16 then, 17 once the published explanation writes the Run's
# readable page. The owner set it as a ceiling at twice that: it catches a
# runaway, not one more read.
EXPLAIN_HISTORY_COPY_CEILING = 34


class _AskingExplainer(ExplainerModel):
    """Asks the researcher once on one member's first request, then explains."""

    def __init__(self, step: str, **options) -> None:
        super().__init__(**options)
        self.asks = step

    def respond(self, messages, info):  # noqa: ANN001, ANN201
        task = _sent_task(messages)
        if self.asks is not None and self.step(task) == self.asks:
            self.asks = None
            self.tasks.setdefault(self.step(task), []).append(task)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "ask_researcher",
                        {
                            "found_so_far": "Both rows read 0/1 at the site.",
                            "conflict": "The request names no allele to report.",
                            "question": "Which allele should the answer name?",
                        },
                        tool_call_id="ask",
                    )
                ]
            )
        return super().respond(messages, info)


def steps_of(journey: Journey) -> dict[str, str]:
    """Each explanation Attempt and the step it is: a request's subject, or
    ``"combination"``."""
    return {
        event["payload"]["attempt_id"]: event["payload"]["targets"][0]["subject"]
        for event in _events(journey, "ReportExplanationMemberBound")
    } | {
        event["payload"]["attempt_id"]: "combination"
        for event in _events(journey, "ReportExplanationCombinationBound")
    }


def _keys(value) -> set[str]:
    """Every key a JSON value names, at any depth."""
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


def _repeated_copies(page: str, limit: int = 80) -> list[str]:
    """Each run of ``limit`` or more characters that two lines of a page's facts
    (sections 1-11) share: the design's "printed once" rule (#491, README §2),
    read from the page itself."""
    from difflib import SequenceMatcher

    lines = [
        line for line in page.split("\n## 12. ")[0].splitlines() if len(line) >= limit
    ]
    found = []
    for index, line in enumerate(lines):
        for other in lines[index + 1 :]:
            match = SequenceMatcher(
                None, line, other, autojunk=False
            ).find_longest_match()
            if match.size >= limit:
                found.append(line[match.a : match.a + match.size])
    return found


def _events(journey: Journey, event_type: str) -> list[dict]:
    return [
        event
        for event in read_evidence(Journey.open(journey.path))
        if event["event_type"] == event_type
    ]


class ReportExplanationTests(ItemStateRuns, unittest.TestCase):
    def child(self, root: Path, request: dict, *args, **planning) -> Journey:
        # One correction: the Run's `agent_invocation.failed_turns` the explainer
        # reads; its second failed turn in a row stops.
        return super().child(
            root,
            request,
            *args,
            failed_turns=2,
            **planning,
        )

    def explain(self, journey: Journey, report_digest: str, *extra: str):
        from bio_neb.cli import main

        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = main(
                [
                    "report-explain",
                    "--journey-file",
                    str(journey.path),
                    "--report-digest",
                    report_digest,
                    "--requested-by",
                    "researcher:owner",
                    *extra,
                ]
            )
        return code, output.getvalue(), error.getvalue()

    def reopen_explanation(self, journey: Journey, explanation_id: str) -> str:
        """The native contact-free reopen of one published explanation."""
        from bio_neb.cli import main

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                [
                    "report-explanation-reopen",
                    "--journey-file",
                    str(journey.path),
                    "--explanation-id",
                    explanation_id,
                ]
            )
        self.assertEqual(code, 0)
        return output.getvalue()

    def observed_report(self, root: Path, store: RunOutputStore):
        """rs429358 called 0/1 and rs7412 with no record, reported (#339 flow A)."""
        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.application.reviews import pending_reviews

        journey = self.child(root, request_file("rs429358", "rs7412"))
        with self.dependencies(store):
            (review,) = pending_reviews(
                advance(ResumeRun(journey_file=str(journey.path)))
            )
            self.answer(
                journey,
                review.request_event_id,
                {
                    "action": "approve",
                    "proposal_id": review.request.proposal.proposal_id,
                },
            )
            report = publish_unexplained(journey, self.basis(journey))
        return journey, report

    def published(self, journey: Journey) -> dict:
        (published,) = _events(journey, "ReportExplanationPublished")
        return published["payload"]["explanation"]

    def test_each_request_is_explained_by_its_own_member_then_one_combined_answer(
        self,
    ):
        from bio_neb.application.advancement import ObserveRun, observe
        from bio_neb.reporting.grounded_report import REPORT_MEDIA_TYPE

        explainer = ExplainerModel()
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            gets = list(store.gets)
            report_text = self.reopen(journey, report.report_digest)
            with self.dependencies(store, report_model_factory=explainer.model):
                code, output, error = self.explain(journey, report.report_digest)
            self.assertEqual((code, error), (0, ""), output)

            # Each member is sent only its own request's targets (the request and
            # each position searched for it), numbered as the report numbers
            # them, with the claim-word table.
            self.assertEqual(
                {
                    step: [
                        (row["number"], row["kind"], row["subject"], row["state"])
                        for row in tasks[0]["targets"]
                    ]
                    for step, tasks in explainer.tasks.items()
                    if step != "combination"
                },
                {
                    "rs429358": [
                        (1, "request", "rs429358", "candidates"),
                        (2, "position", RS429358_TARGET, "observed"),
                    ],
                    "rs7412": [
                        (3, "request", "rs7412", "candidates"),
                        (4, "position", RS7412_TARGET, "observed"),
                    ],
                },
            )
            # Every task reads its qualifications as their origins' plain words,
            # never the report's audit sentence, and names no row, item or Work
            # item id in its fields or the words code writes (a model's own
            # earlier prose and notes are its own).
            for step, tasks in explainer.tasks.items():
                with self.subTest(step=step):
                    self.assertFalse(
                        {"row", "rows", "request_row", "work_item_id", "state_note"}
                        & _keys(tasks[0]),
                        step,
                    )
                    self.assertNotRegex(
                        json.dumps([tasks[0]["task"], tasks[0]["qualifications"]]),
                        r"(?i)\brows?\b|\bitems?\b|work.item",
                    )
                    self.assertTrue(tasks[0]["qualifications"])
                    for item in tasks[0]["qualifications"]:
                        self.assertEqual(
                            set(item),
                            {"qualification", "kind", "origin", "words", "targets"},
                        )
                        self.assertNotIn("Work item", item["words"])
            (rs7412_task,) = explainer.tasks["rs7412"]
            # The Run measured rs7412's position off the default BAM's reads.
            self.assertIn(
                {
                    "kind": "limitation",
                    "origin": "depth_below_minimum",
                    "words": "Read depth here was 0 at its lowest, below the minimum "
                    "of 10.",
                    "targets": [4],
                },
                [
                    {key: item[key] for key in ("kind", "origin", "words", "targets")}
                    for item in rs7412_task["qualifications"]
                ],
            )
            self.assertEqual(
                [
                    (row.get("outcome"), row.get("genotype"))
                    for step in ("rs429358", "rs7412")
                    for row in explainer.tasks[step][0]["targets"]
                ],
                [
                    (None, None),
                    ("exact_allele_observed", "0/1"),
                    (None, None),
                    ("no_record", None),
                ],
            )
            # Both members ran before the combination, which was sent each
            # member's admitted claims.
            self.assertEqual(explainer.order[-1], "combination")
            self.assertEqual(sorted(explainer.order[:-1]), ["rs429358", "rs7412"])
            (combined,) = explainer.tasks["combination"]
            self.assertEqual(
                [
                    (entry["request"], entry["status"], entry["claim_words"])
                    for entry in combined["requests"]
                ],
                [
                    (1, "explained", ["candidates", "carries_target_allele"]),
                    (3, "explained", ["candidates", "searched_no_record"]),
                ],
            )
            explanation = self.published(journey)
            self.assertEqual(
                explanation["schema_version"], "bio-neb.report-explanation/5"
            )
            # Each genotype a member's text writes is kept with its target and
            # quote, its target's own call.
            self.assertEqual(
                [
                    (item["request"], item["genotypes"])
                    for item in explanation["requests"]
                ],
                [
                    (
                        1,
                        [{"target": 2, "genotype": "0/1", "quote": "genotype 0/1"}],
                    ),
                    (3, []),
                ],
            )
            self.assertEqual(explanation["genotypes"], [])
            self.assertEqual(
                [
                    (
                        item["request"],
                        item["status"],
                        [
                            (claim["target"], claim["claim"], claim.get("genotype"))
                            for claim in item["claims"]
                        ],
                    )
                    for item in explanation["requests"]
                ],
                [
                    (
                        1,
                        "explained",
                        [(1, "candidates", None), (2, "carries_target_allele", "0/1")],
                    ),
                    (
                        3,
                        "explained",
                        [(3, "candidates", None), (4, "searched_no_record", None)],
                    ),
                ],
            )
            self.assertEqual(
                (
                    explanation["report_digest"],
                    explanation["report_basis_digest"],
                    explanation["agent_written"],
                ),
                (report.report_digest, report.report_basis_digest, True),
            )
            (published,) = _events(journey, "ReportExplanationPublished")
            (artifact,) = published["artifact_references"]
            self.assertNotEqual(artifact["media_type"], REPORT_MEDIA_TYPE)
            # The conclusion and each request's own why read on their own.
            self.assertIn("Explanation (Agent-written, not a finding)\n", output)
            self.assertIn(
                "rs429358: candidates, carries target allele. rs7412: candidates, "
                "searched no record.",
                output,
            )
            self.assertIn(
                "  request rs429358 (target 1): explained\n"
                "    target 1 request rs429358: candidates\n"
                f"    target 2 position {RS429358_TARGET}: carries_target_allele, "
                "genotype 0/1\n",
                output,
            )
            self.assertIn(
                "    why: rs7412: row 3 reads candidates; row 4 reads searched no record.",
                output,
            )
            reopened = self.reopen_explanation(journey, explanation["explanation_id"])
            self.assertIn("rs429358: candidates, carries target allele.", reopened)
            self.assertIn("  request rs7412 (target 3): explained\n", reopened)
            # The report is untouched and no NCBI or storage contact was made.
            self.assertEqual(self.reopen(journey, report.report_digest), report_text)
            self.assertEqual(
                observe(ObserveRun(journey_file=str(journey.path))).status,
                "grounded_report_published",
            )
            self.assertEqual(store.gets, gets)
            from bio_neb.cli import main

            feed = io.StringIO()
            with contextlib.redirect_stdout(feed):
                self.assertEqual(
                    main(["journey-feed", "--journey-file", str(journey.path)]), 0
                )
            self.assertIn(
                "An explanation of your report was published (Agent-written)",
                feed.getvalue(),
            )

    def test_a_finished_run_writes_and_prints_its_own_readable_report(self):
        """#476 (owner, 2026-10-28, G5/C13: "i want to see the the run created as a
        repot"): once the report and its default explanation publish, the Run
        writes the readable page as ``report.md`` in its Run root, keeps its
        sha256 in the Journey and prints it at the end of the ``answer``."""
        import hashlib

        from bio_neb.application.advancement import use_run_advancement_dependencies
        from bio_neb.application.queries import read_completion
        from bio_neb.application.reviews import pending_reviews
        from tests.observation_input_fakes import TABLE_READING_ENVIRON

        explainer = ExplainerModel()
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey = self.child(
                Path(directory).resolve(), request_file("rs429358", "rs7412")
            )
            with self.dependencies(store, report_model_factory=explainer.model):
                from bio_neb.application.advancement import ResumeRun, advance

                (review,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                self.answer(
                    journey,
                    review.request_event_id,
                    {
                        "action": "approve",
                        "proposal_id": review.request.proposal.proposal_id,
                    },
                )
                basis = self.basis(journey)
                pending = read_completion(Journey.open(journey.path)).finalization
                with use_run_advancement_dependencies(
                    environ=TABLE_READING_ENVIRON,
                    report_model_factory=explainer.model,
                ):
                    code, printed = native_answer(
                        journey.path,
                        pending.pending_request.request_event_id,
                        {
                            "action": "approve",
                            "basis_digest": basis.basis_digest,
                            "head_event_id": review_head(
                                read_evidence(Journey.open(journey.path))
                            )["event_id"],
                        },
                    )
            self.assertEqual(code, 0, printed)
            report_digest = self.published(journey)["report_digest"]
            readable = self.readable(journey, report_digest)
            written = journey.path.parent.parent / "report.md"
            self.assertEqual(written.read_text(encoding="utf-8"), readable)
            (fact,) = _events(journey, "ReadableReportWritten")
            self.assertEqual(
                {
                    key: fact["payload"][key]
                    for key in ("report_digest", "file_name", "sha256", "byte_count")
                },
                {
                    "report_digest": report_digest,
                    "file_name": "report.md",
                    "sha256": hashlib.sha256(written.read_bytes()).hexdigest(),
                    "byte_count": len(written.read_bytes()),
                },
            )
            self.assertIn(readable, printed)

    def test_a_report_the_coordinator_publishes_is_explained_by_default(self):
        # Every published report gets its explanation (ADR 0038); the Run
        # itself requests it, and a second request of the same report must
        # name the one it rewrites.
        from bio_neb.application.advancement import ObserveRun, ResumeRun, advance
        from bio_neb.application.advancement import observe as observe_run
        from bio_neb.application.reviews import pending_reviews

        explainer = ExplainerModel()
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey = self.child(
                Path(directory).resolve(), request_file("rs429358", "rs7412")
            )
            with self.dependencies(store, report_model_factory=explainer.model):
                (review,) = pending_reviews(
                    advance(ResumeRun(journey_file=str(journey.path)))
                )
                self.answer(
                    journey,
                    review.request_event_id,
                    {
                        "action": "approve",
                        "proposal_id": review.request.proposal.proposal_id,
                    },
                )
                report = self.publish(journey, self.basis(journey), explainer=explainer)
            (requested,) = _events(journey, "ReportExplanationRequested")
            self.assertEqual(
                (
                    requested["payload"]["requested_by"],
                    requested["payload"]["report_digest"],
                ),
                ("run_default", report.report_digest),
            )
            self.assertEqual(
                self.published(journey)["report_digest"], report.report_digest
            )
            self.assertEqual(
                sorted(explainer.tasks), ["combination", "rs429358", "rs7412"]
            )
            self.assertEqual(
                observe_run(ObserveRun(journey_file=str(journey.path))).status,
                "grounded_report_published",
            )
            code, _output, error = self.explain(journey, report.report_digest)
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(error)["code"], "report_explanation_exists")
            readable = self.readable(journey, report.report_digest)
        # The Run report: twelve sections, 1-11 retrieved facts and 12 the
        # explainer's own text, each marked; the explainer titles the page.
        facts, agent = readable.split("\n## 12. The agent's summary and caveats\n")
        self.assertTrue(readable.startswith("# Run report: sample-001\n\n"))
        self.assertIn(
            "*Each request of the report with its member's claims* ✓ (the report "
            "explainer's title)",
            facts,
        )
        self.assertEqual(
            re.findall(r"^## (\d+)\. ", readable, re.MULTILINE),
            [str(number) for number in range(1, 13)],
        )
        self.assertNotIn("✓", facts.split("\n## 1. Run\n")[1])
        self.assertIn(
            "> **2 targets · 2 examined in the VCF · 1 with the target allele called "
            "· 0 not examined**",
            facts,
        )
        # One results row per Work item: the VCF's cells, the Run's depth at the
        # site (the default BAM's twelve reads at rs429358) and its L-lines.
        self.assertIn(
            "| **T1** | rs429358 | you | NC_000019.10:44908684 T>C | called: carries "
            "the target allele | 0/1 | — | — | PASS / 50 | 12 (min 10) | L1, L2, L3 |\n"
            "| **T2** | rs7412 | you | NC_000019.10:44908822 T>C | no call | — | — | — "
            "| — / — | 0 (min 10) | L1, L2, L3, L4 |\n",
            facts,
        )
        self.assertIn(
            "## 4. Run health\n\n| Measure | Value |\n|---|---|\n| Reads | 12 |\n"
            "| Mapped | 100.0% |\n",
            facts,
        )
        self.assertIn("| Ti/Tv | none: the VCF has no transversion |", facts)
        # Key decisions from the report's accounting.
        self.assertIn(
            "- **Execution:** the exact Parabricks proposal approved, calling ploidy 2 "
            "(`user_decision`).\n- **Investigation:** the exact proposal approved.\n"
            "- **Finalization:** the exact Report basis approved.\n",
            facts,
        )
        self.assertIn("`report_explainer:v20:qwen_3_5`", facts)
        self.assertIn("## 7. Papers\n\nNo paper was read in this Run.", facts)
        self.assertIn(
            "## 8. Not examined\n\nEvery target was examined in the VCF.", facts
        )
        # The method copies the calling ploidy the researcher approved.
        self.assertEqual(report.method.calling_ploidy, 2)
        self.assertIn(
            "fq2bam (BWA-MEM), then HaplotypeCaller, calling ploidy 2\n", facts
        )
        # Each qualification once in its typed origin's plain words, as an L-line
        # led by every target its ids cover; the audit sentences and Work item ids
        # are not on the page.
        self.assertIn(
            "- **L1** T1, T2: callability, biological absence, function, causality and "
            "diagnosis are not inferred from it; its read depth was measured.\n"
            "- **L2** T1, T2: the result holds for this sample's VCF at this target "
            "only.\n"
            "- **L3** T1, T2: it comes from a Run forked from earlier work, not a "
            "fresh analysis.\n"
            "- **L4** T2: read depth here was 0 at its lowest, below the minimum of "
            "10.\n",
            facts,
        )
        for internal in ("work-item:", "request-member:", "Work item", report.outcome):
            with self.subTest(internal=internal):
                self.assertNotIn(internal, readable)
        for identifier in (
            report.report_digest,
            report.report_basis_digest,
            report.method.vcf_content_id,
        ):
            self.assertIn(f"`{identifier}`", facts.split("\n## 11. ")[1])
        # The explainer's words, each marked, its caveats tagged by the L-lines
        # their qualification numbers name.
        self.assertIn(
            "**Summary ✓** rs429358: candidates, carries target allele. rs7412: "
            "candidates, searched no record.\n\n**What would change it ✓** A call at a "
            "request's position would change its answer.\n",
            agent,
        )
        self.assertIn(
            "- **T1 rs429358 ✓** rs429358: row 1 reads candidates; row 2 reads "
            "carries target allele with genotype 0/1.\n",
            agent,
        )
        self.assertIn(
            "- *the whole Run, on L1, L3:* These qualifications apply to every "
            "request.\n- *T1, on L2:* This request's qualifications apply as the "
            "report states.\n- *T2, on L2, L4:* This request's qualifications apply "
            "as the report states.\n",
            agent,
        )

    def test_a_report_published_by_its_own_command_is_explained_on_resume(self):
        # A report published outside the coordinator is explained by default on
        # the Run's next resume. The default request is recorded inside that
        # Pass, after its PassStarted, on the report's own publication.
        from bio_neb.application.advancement import ObserveRun, ResumeRun, advance
        from bio_neb.application.advancement import observe as observe_run

        explainer = ExplainerModel()
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                advance(ResumeRun(journey_file=str(journey.path)))
            (requested,) = _events(journey, "ReportExplanationRequested")
            self.assertEqual(requested["payload"]["requested_by"], "run_default")
            self.assertEqual(
                self.published(journey)["report_digest"], report.report_digest
            )
            self.assertEqual(
                observe_run(ObserveRun(journey_file=str(journey.path))).status,
                "grounded_report_published",
            )

    def test_a_report_with_no_explanation_reads_not_written(self):
        # Until an explanation is published, the copied facts stand and each
        # written cell says why it is not written.
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            readable = self.readable(journey, report.report_digest)
        self.assertIn("*Title: Not written: no explanation was requested.*", readable)
        self.assertIn(
            "| **T1** | rs429358 | you | NC_000019.10:44908684 T>C | called: carries "
            "the target allele | 0/1 |",
            readable,
        )
        # The report's own limitations stand without an explanation.
        self.assertIn("- **L1** T1, T2: ", readable)
        self.assertTrue(
            readable.endswith(
                "## 12. The agent's summary and caveats\n\n*Written by the report "
                "explainer. Every row and event it cites is in this report; its "
                "meaning is not checked. Where it differs from a fact above, the fact "
                "is the report's.*\n\nNot written: no explanation was requested.\n"
            ),
            readable,
        )
        self.assertNotIn("✓ ", readable)

    def test_a_sequence_the_model_matched_is_shown_with_its_source_answer(self):
        """A contig the written words do not literally name, matched by the model, is
        shown to the researcher (owner #467 RC5 1a, Q2 a); a literal one adds no line."""
        from bio_neb.investigation.lookup_origins import ModelMatchedName
        from bio_neb.reporting.readable_report import (
            ExplanationState,
            RunFacts,
            render_readable_report,
        )

        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
        view = report.collection_view
        first = view.snps[0]
        matched = report.model_copy(
            update={
                "collection_view": view.model_copy(
                    update={
                        "snps": (
                            first.model_copy(
                                update={
                                    "lookup": first.lookup.model_copy(
                                        update={
                                            "model_matched": (
                                                ModelMatchedName(
                                                    written="chr19",
                                                    sequence_name="19",
                                                    record="NC_000019.10",
                                                    assembly_accession=(
                                                        "GCF_000001405.40"
                                                    ),
                                                ),
                                            )
                                        }
                                    )
                                }
                            ),
                            *view.snps[1:],
                        )
                    }
                )
            }
        )

        self.assertIn(
            "  The model matched the written sequence name: chr19 -> 19 "
            "(NC_000019.10) on GCF_000001405.40.",
            render_readable_report(matched, ExplanationState(None, "none"), RunFacts()),
        )
        self.assertNotIn(
            "The model matched",
            render_readable_report(report, ExplanationState(None, "none"), RunFacts()),
        )

    def test_the_papers_section_and_rows_name_what_a_paper_found_and_asserted(self):
        """#485 Q3 a (live #383 L2): the page says what each paper read in the Run
        found, citing the chunks its reading kept, and a row whose target the paper
        asserted names the paper, not the researcher; the explainer's row words name
        it too. A derived input: the Run's report is given one fictional paper and
        rs429358 is given as that paper's target; every other fact is the Run's."""
        from bio_neb.paper_reading.contracts import KeptFinding, KeptValue, PaperAnswer
        from bio_neb.paper_reading.report import ReportedPaper
        from bio_neb.reporting.explanation_rows import explanation_rows
        from bio_neb.reporting.readable_report import (
            ExplanationState,
            RunFacts,
            render_readable_report,
        )
        from bio_neb.target_request.paper_targets import PaperIdentity, SourceLine

        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
        sentence = "Carriers of rs429358 were enriched among cases."
        quote = SourceLine(
            document_id="main", section="Results", start=120, end=129, text="rs429358"
        )
        paragraph = SourceLine(
            document_id="main", section="Results", start=108, end=155, text=sentence
        )
        paper = ReportedPaper(
            answer_event_id="answer-1",
            paper_event_id="paper-1",
            question="Which APOE SNPs does the paper describe?",
            paper=PaperIdentity(
                doi="10.5555/fictional.485",
                title="A fictional APOE study",
                year="2024",
                resolved_by="europe_pmc",
            ),
            asserted_by="paper:doi:10.5555/fictional.485",
            answer=PaperAnswer(
                question="Which APOE SNPs does the paper describe?",
                findings=(
                    KeptFinding(
                        passages=(quote, paragraph),
                        reason="The paper names it an APOE SNP.",
                        gene="APOE",
                        values=(
                            KeptValue(
                                slot="rsid",
                                written="rs429358",
                                span="s1",
                                source=quote,
                                value="rs429358",
                            ),
                        ),
                    ),
                    # A second finding resting on the same quote: printed once.
                    KeptFinding(
                        passages=(quote, paragraph),
                        reason="The paper names it again among the carriers.",
                        values=(
                            KeptValue(
                                slot="rsid",
                                written="rs429358",
                                span="s1",
                                source=quote,
                                value="rs429358",
                            ),
                        ),
                    ),
                ),
            ),
        )
        view = report.collection_view
        first = view.snps[0].model_copy(update={"asserted_by": paper.asserted_by})
        derived = report.model_copy(
            update={
                "papers": (paper,),
                "collection_view": view.model_copy(
                    update={"snps": (first, *view.snps[1:])}
                ),
            }
        )

        page = render_readable_report(
            derived, ExplanationState(None, "none"), RunFacts()
        )
        # What the paper reports, copied with where, beside what this Run found
        # for the targets whose rsID the finding names; the quote's paragraph is
        # named as the span holding it, never printed again.
        self.assertIn(
            "## 7. Papers\n\n"
            "### P1 · A fictional APOE study (2024)\n\n"
            "doi:10.5555/fictional.485. Read for: “Which APOE SNPs does the paper "
            "describe?”\n"
            "Its targets are asserted by the paper doi:10.5555/fictional.485, not by "
            "you.\n\n"
            "| # | The paper reports (copied, with where) | This Run found |\n"
            "|---|---|---|\n"
            "| 1 | rs429358 (APOE): “rs429358” (Results; main 120–129, within main "
            "108–155) | T1 rs429358: called: carries the target allele |\n"
            "| 2 | rs429358: the quote of finding 1 | T1 rs429358: called: carries the "
            "target allele |\n",
            page,
        )
        self.assertNotIn(sentence, page)
        self.assertEqual(page.count("“rs429358”"), 1)
        self.assertEqual(_repeated_copies(page), [])
        self.assertIn("| **T1** | rs429358 | P1 | NC_000019.10:44908684 T>C |", page)
        self.assertIn("| **T2** | rs7412 | you | NC_000019.10:44908822 T>C |", page)
        self.assertNotIn("No paper was read in this Run.", page)
        self.assertEqual(
            [
                (row.subject, row.asserted_by)
                for row in explanation_rows(derived.collection_view)
                if row.kind == "request"
            ],
            [
                ("rs429358", "the paper doi:10.5555/fictional.485"),
                ("rs7412", None),
            ],
        )

    def test_a_position_placed_from_another_version_reads_as_stated_and_placed(self):
        """#481 H2 (owner Q1 a): a row whose lookup moved its position onto the
        Run's assembly says where it was stated, on which version, and where it was
        placed. A derived input: rs429358's lookup is given #457 human a1's GRCh37
        APOE placement, the Work item it placed at unchanged."""
        from bio_neb.investigation.collection_view import LookupPlacement
        from bio_neb.investigation.investigation_request import (
            PlacedPosition,
            StatedPosition,
        )
        from bio_neb.reporting.readable_report import (
            ExplanationState,
            RunFacts,
            render_readable_report,
        )

        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
        view = report.collection_view
        first = view.snps[0]
        placed = report.model_copy(
            update={
                "collection_view": view.model_copy(
                    update={
                        "snps": (
                            first.model_copy(
                                update={
                                    "lookup": first.lookup.model_copy(
                                        update={
                                            "placement": LookupPlacement(
                                                stated=StatedPosition(
                                                    contig="19",
                                                    position=45411941,
                                                    ref="T",
                                                    alt="C",
                                                ),
                                                stated_version="GRCh37",
                                                placed=PlacedPosition(
                                                    contig="NC_000019.10",
                                                    position=44908684,
                                                    ref="T",
                                                    alt="C",
                                                ),
                                            )
                                        }
                                    )
                                }
                            ),
                            *view.snps[1:],
                        )
                    }
                )
            }
        )

        page = render_readable_report(
            placed, ExplanationState(None, "none"), RunFacts()
        )
        self.assertIn(
            "- **T1** 19:45411941 T>C as stated on GRCh37; placed at "
            "NC_000019.10:44908684 T>C\n",
            page,
        )
        self.assertIn("| **T1** | rs429358 | you | NC_000019.10:44908684 T>C |", page)

    def test_a_pipe_in_the_researchers_words_stays_in_its_cell(self):
        """#481 H6: a label taken from the researcher's free text escapes a pipe,
        so the row keeps its cells. A derived input: rs429358's request is given
        no rsID and a request text with a pipe, which becomes its label (#491)."""
        from bio_neb.reporting.readable_report import (
            ExplanationState,
            RunFacts,
            render_readable_report,
        )

        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
        view = report.collection_view
        first = view.snps[0]
        piped = report.model_copy(
            update={
                "collection_view": view.model_copy(
                    update={
                        "snps": (
                            first.model_copy(
                                update={
                                    "request_text": "APOE | rs429358, please",
                                    "requested_rsid": None,
                                }
                            ),
                            *view.snps[1:],
                        )
                    }
                )
            }
        )

        self.assertIn(
            "| **T1** | APOE \\| rs429358, please | you | NC_000019.10:44908684 T>C |",
            render_readable_report(piped, ExplanationState(None, "none"), RunFacts()),
        )

    def test_the_run_report_keeps_its_template_rules(self):
        """#491 (README §2 rules, owner 2026-10-30): a target's label is its rsID,
        its stated position, else its words up to 60 characters with a named cut,
        never a paper's paragraph (Mullins F1); a long request's words are not
        printed; no fact line passes 700 characters and no copied text over 80
        characters is printed twice; an explanation written before titles says
        so. A derived input: rs429358's request is given no rsID and a 1,400
        character paragraph as its words, as a paper's target was."""
        from bio_neb.investigation.investigation_request import StatedPosition
        from bio_neb.reporting.explanation import ReportExplanationByRequest
        from bio_neb.reporting.readable_report import (
            ExplanationState,
            RunFacts,
            readable_rows,
            render_readable_report,
        )

        explainer = ExplainerModel()
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                self.assertEqual(self.explain(journey, report.report_digest)[0], 0)
            explanation = ReportExplanationByRequest.model_validate_json(
                json.dumps(self.published(journey))
            )
            # The Run's facts beside the report (D1 a): the publication time, and
            # each source's own answer the report's accounting names, verified; a
            # named answer that does not read or verify refuses the page, and a
            # source the report names no answer for reads None.
            import hashlib

            from bio_neb.journey import JourneyValidationError
            from bio_neb.reporting.readable_report import ReadableReportInconsistent
            from bio_neb.reporting.report_commands import _run_facts

            live = Journey.open(journey.path)
            facts = _run_facts(live, report)
            (published,) = _events(journey, "GroundedReportPublished")
            self.assertEqual(facts.published_at, published["occurred_at"])
            source = report.retrieved_sources[0].value
            self.assertEqual(
                hashlib.sha256(facts.source_answer(source)).hexdigest(),
                source.raw_response_sha256,
            )
            self.assertIsNone(
                facts.source_answer(
                    source.model_copy(update={"raw_artifact_id": "sha256:" + "0" * 64})
                )
            )
            for case, change in (
                ("other bytes", {"return_value": b"tampered"}),
                ("unreadable", {"side_effect": JourneyValidationError("gone")}),
            ):
                with (
                    self.subTest(case=case),
                    patch.object(live, "read_artifact", **change),
                    self.assertRaises(ReadableReportInconsistent) as refused,
                ):
                    facts.source_answer(source)
                self.assertEqual(
                    refused.exception.faults,
                    (
                        f"source {source.source_id}: its retained answer does not verify",
                    ),
                )
        paragraph = " ".join(
            f"The APOE gene's allele {n} binds lipids in chylomicrons."
            for n in range(30)
        )
        view = report.collection_view
        first, second = view.snps

        def derived(**changes):
            return report.model_copy(
                update={
                    "collection_view": view.model_copy(
                        update={"snps": (first.model_copy(update=changes), second)}
                    )
                }
            )

        for changes, label in (
            ({}, "rs429358"),
            (
                {
                    "requested_rsid": None,
                    "stated_position": StatedPosition(
                        contig="19", position=44908684, ref="T", alt="C"
                    ),
                },
                "19:44908684 T>C",
            ),
            (
                {"requested_rsid": None, "request_text": "APOE e4, please"},
                "APOE e4, please",
            ),
            (
                {"requested_rsid": None, "request_text": paragraph},
                "The APOE gene's allele 0 binds lipids in chylomicrons. The …",
            ),
        ):
            with self.subTest(label=label):
                self.assertEqual(readable_rows(derived(**changes))[0].label, label)
                self.assertLessEqual(len(label), 60)
        long = derived(requested_rsid=None, request_text=paragraph)
        page = render_readable_report(long, ExplanationState(explanation), RunFacts())
        self.assertNotIn(paragraph[:80], page)
        facts = page.split("\n## 12. ")[0]
        self.assertEqual([line for line in facts.splitlines() if len(line) > 700], [])
        self.assertEqual(_repeated_copies(page), [])
        # The explainer's title is its own: an explanation written before it has
        # none, and the page says so.
        self.assertIn(
            "*Title: Not written: this explanation was written before titles.*",
            render_readable_report(
                report,
                ExplanationState(explanation.model_copy(update={"title": None})),
                RunFacts(),
            ),
        )
        # A target's caveats are each printed whole; another target's caveat on
        # the same L-lines is named with where its words are kept (M5).
        first, second = explanation.requests
        (caveat,) = first.limitations
        twice = explanation.model_copy(
            update={
                "requests": (
                    first.model_copy(
                        update={
                            "limitations": (
                                caveat,
                                caveat.model_copy(
                                    update={"text": "A second caveat of T1."}
                                ),
                            )
                        }
                    ),
                    second.model_copy(
                        update={
                            "limitations": (
                                caveat.model_copy(update={"text": "T2's own caveat."}),
                            )
                        }
                    ),
                )
            }
        )
        page = render_readable_report(report, ExplanationState(twice), RunFacts())
        self.assertIn(
            f"- *T1, on L2:* {caveat.text}\n"
            "- *T1, on L2:* A second caveat of T1. (alike for T2; each one's words: "
            "`report-explanation-reopen`)\n",
            page,
        )
        self.assertNotIn("T2's own caveat.", page)

    def test_every_qualification_origin_reads_in_its_plain_words(self):
        """#474 acceptance: the page states every typed origin in its plain words,
        led by the target it concerns, and never the report's audit sentence
        (#460 LSP6, option c; #491: no appendix). A derived input: the explained
        report's qualifications are replaced by one of every origin, each the
        first request's, and one written limitation of the Run covers them all."""
        from bio_neb.investigation.qualification_origins import plain_words
        from bio_neb.reporting.explanation import (
            ExplainedLimitation,
            ExplainedRequest,
            ReportExplanationByRequest,
        )
        from bio_neb.reporting.finalization import Qualification
        from bio_neb.reporting.readable_report import (
            ExplanationState,
            RunFacts,
            render_readable_report,
        )
        from tests.qualification_origin_pins import AUDIT_WORDING, PINS

        explainer = ExplainerModel()
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                self.assertEqual(self.explain(journey, report.report_digest)[0], 0)
            explanation = ReportExplanationByRequest.model_validate_json(
                json.dumps(self.published(journey))
            )
        first = report.collection_view.snps[0]
        stated = {
            kind: tuple(
                sorted(
                    {
                        origin.audit(subject): Qualification(
                            text=origin.audit(subject),
                            origin=origin,
                            request_member_ids=(first.request_member_id,),
                        )
                        for origin, subject, origin_kind, _audit, _words in PINS
                        if origin_kind == kind
                    }.values(),
                    key=lambda item: item.text,
                )
            )
            for kind in ("limitation", "conflict", "gap")
        }
        every = report.model_copy(
            update={
                "limitations": stated["limitation"],
                "conflicts": stated["conflict"],
                "gaps": stated["gap"],
            }
        )
        count = sum(len(items) for items in stated.values())
        covering = explanation.model_copy(
            update={
                "limitations": (
                    ExplainedLimitation(
                        qualifications=tuple(range(1, count + 1)),
                        text="Each applies as the report states.",
                        references=explanation.conclusion.references,
                    ),
                ),
                "requests": tuple(
                    section.model_copy(update={"limitations": ()})
                    if isinstance(section, ExplainedRequest)
                    else section
                    for section in explanation.requests
                ),
            }
        )
        page = render_readable_report(every, ExplanationState(covering), RunFacts())
        limitations = page.split("\n## 9. Limitations\n")[1].split("\n## ")[0]
        self.assertIsNone(AUDIT_WORDING.search(limitations), limitations)
        for items in stated.values():
            for item in items:
                with self.subTest(origin=item.origin):
                    self.assertIn(
                        plain_words(item.origin, ("T1",)).replace("..", "."),
                        limitations,
                    )
                    # The report's audit sentence is not on the page.
                    self.assertNotIn(item.text, page)

    def test_a_row_that_disagrees_with_its_report_refuses_the_page(self):
        # A row stating a VCF search, an outcome, a carried allele, a genotype, a
        # placement or a source answer the report does not hold fails, never ships.
        import dataclasses

        from bio_neb.reporting.readable_report import (
            ExplanationState,
            ReadableReportInconsistent,
            RunFacts,
            consistency_faults,
            readable_rows,
            render_readable_report,
        )

        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
        render_readable_report(report, ExplanationState(None, "none"), RunFacts())
        view = report.collection_view
        called = view.snps[0]
        hostile = {
            "no observations": report.model_copy(update={"sample_observations": ()}),
            "a genotype not called": report.model_copy(
                update={
                    "collection_view": view.model_copy(
                        update={
                            "snps": (
                                called.model_copy(
                                    update={
                                        "items": (
                                            called.items[0].model_copy(
                                                update={"genotype": "1/1"}
                                            ),
                                        )
                                    }
                                ),
                                *view.snps[1:],
                            )
                        }
                    )
                }
            ),
            "a lookup's answer not retained": report.model_copy(
                update={"retrieved_sources": ()}
            ),
        }
        for case, value in hostile.items():
            with (
                self.subTest(case=case),
                self.assertRaises(ReadableReportInconsistent) as raised,
            ):
                render_readable_report(
                    value, ExplanationState(None, "none"), RunFacts()
                )
            self.assertTrue(raised.exception.faults)
            self.assertTrue(
                all(fault.startswith("row ") for fault in raised.exception.faults)
            )
        # A rendered fact corrupted while its observation stays valid: rs429358's
        # observed 0/1 exact allele read as no record, as another genotype, placed
        # elsewhere, or its source answer named by another authority.
        carried, unseen = readable_rows(report)
        (item,) = carried.items
        rows = {"carried": carried, "unseen": unseen}
        for case, row, expected in (
            (
                "no record read for an observed allele",
                dataclasses.replace(
                    carried,
                    items=(
                        dataclasses.replace(
                            item, key="no_record", carries=False, genotype=None
                        ),
                    ),
                ),
                "reads no_record, observed exact_allele_observed",
            ),
            (
                "a genotype the VCF did not call",
                dataclasses.replace(
                    carried, items=(dataclasses.replace(item, genotype="1/1"),)
                ),
                "genotype 1/1 is not called",
            ),
            (
                "a carried allele on a no-record row",
                dataclasses.replace(
                    unseen,
                    items=(dataclasses.replace(unseen.items[0], carries=True),),
                ),
                "its carried allele differs from the observation",
            ),
            (
                "another placement",
                dataclasses.replace(
                    carried,
                    items=(dataclasses.replace(item, placed="NC_000019.10:1 T>C"),),
                ),
                "placed apart from its Work item",
            ),
            (
                "its own item left out",
                dataclasses.replace(carried, items=()),
                "its items are not its request's",
            ),
            (
                "a sibling request's item",
                dataclasses.replace(carried, items=unseen.items),
                "its items are not its request's",
            ),
            (
                "a sibling request's target words",
                dataclasses.replace(carried, target=unseen.target),
                "its target is not its request's",
            ),
            (
                "a sibling request's target label",
                dataclasses.replace(carried, label=unseen.label),
                "its target is not its request's",
            ),
            (
                "a sibling request's source answer",
                dataclasses.replace(carried, lookup=unseen.lookup),
                "its source answer is not its request's",
            ),
            (
                "another source authority",
                dataclasses.replace(
                    carried,
                    lookup=carried.lookup.model_copy(update={"authority": "eva"}),
                ),
                "its source answer is not retained",
            ),
        ):
            with self.subTest(case=case):
                # The corrupted row in its own place among the report's rows.
                faults = consistency_faults(
                    report,
                    tuple(
                        row
                        if other.request_member_id == row.request_member_id
                        else other
                        for other in rows.values()
                    ),
                )
                self.assertTrue(
                    any(expected in fault for fault in faults), (expected, faults)
                )
        # A whole request left out of the table.
        self.assertIn(
            "the rows are not the report's requests, in its order",
            consistency_faults(report, (carried,)),
        )

    def test_each_recorded_outcome_reads_in_its_own_plain_words(self):
        # The wording table is the contract, word for word; a sentence that drifts
        # from its key fails here. The words live in one place (``state_words``).
        from bio_neb.investigation.state_words import ITEM_WORDS

        self.assertEqual(
            {key: words.cell for key, words in ITEM_WORDS.items()},
            {
                "exact_allele_observed": "Found: the sample carries the target allele",
                "target_allele_not_carried": (
                    "Called here: the target allele is not carried"
                ),
                "target_genotype_unresolved": "Called here: the genotype is unresolved",
                "locus_present_identity_mismatch": (
                    "A different variant was called here"
                ),
                "locus_overlapped": "A nearby variant call overlaps this position",
                "no_record": (
                    "Searched the VCF: no variant call here; genotype not established"
                ),
                "ambiguous_records": "Several variant calls here; not decided",
                "refused": "The VCF search was refused",
                "region_observed": "Searched the VCF across this region",
                "failed": "The VCF search failed",
                "approved_not_observed": (
                    "Approved; the Run ended before the VCF was searched"
                ),
                "not_approved": "Not approved for a VCF search",
                "unselected_candidate": "Another allele was chosen instead",
                "removed": "Removed by the researcher",
                "deferred": "Deferred by the researcher",
            },
        )
        # The results table's short words, one per outcome or state.
        self.assertEqual(
            {key: words.table for key, words in ITEM_WORDS.items()},
            {
                "exact_allele_observed": "called: carries the target allele",
                "target_allele_not_carried": "called: target allele not carried",
                "target_genotype_unresolved": "called: genotype unresolved",
                "locus_present_identity_mismatch": "a different variant is called here",
                "locus_overlapped": "an overlapping call",
                "no_record": "no call",
                "ambiguous_records": "several calls; not decided",
                "refused": "search refused",
                "region_observed": "region searched",
                "failed": "search failed",
                "approved_not_observed": "approved; not searched",
                "not_approved": "not approved",
                "unselected_candidate": "another allele chosen",
                "removed": "removed",
                "deferred": "deferred",
            },
        )
        self.assertEqual(
            {key for key, words in ITEM_WORDS.items() if words.carries},
            {"exact_allele_observed"},
        )

    def test_the_members_run_together_and_each_reads_only_its_own_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            events = read_evidence(Journey.open(journey.path))
            # A planner model request: retained, but no part of the basis.
            outside = next(
                event["event_id"]
                for event in events
                if event["event_type"] == "ModelRequestRecorded"
            )
            explainer = ExplainerModel(
                searches={
                    "rs429358": [
                        lambda task: {"event_ids": [outside]},
                        lambda task: {
                            "kinds": ["InvestigationEvidenceRecorded"],
                            "limit": 10,
                        },
                    ]
                }
            )
            with self.dependencies(store, report_model_factory=explainer.model):
                code, output, error = self.explain(journey, report.report_digest)
            self.assertEqual((code, error), (0, ""), output)

            bound = {
                event["payload"]["request_member_id"]: event
                for event in _events(journey, "ReportExplanationMemberBound")
            }
            self.assertEqual(len(bound), 2)
            # Both members started before either was admitted: one batch.
            ordered = [
                event["event_type"]
                for event in read_evidence(Journey.open(journey.path))
                if event["event_type"]
                in {"ReportExplanationMemberBound", "ReportExplanationMemberAdmitted"}
            ]
            self.assertEqual(ordered[:2], ["ReportExplanationMemberBound"] * 2)
            # Each member's evidence names its own request: the report's
            # publication is in both, and the other request's own lookup is not.
            scope = {
                item["payload"]["targets"][0]["subject"]: set(
                    item["payload"]["evidence_event_ids"]
                )
                for item in bound.values()
            }
            (whole,) = _events(journey, "ReportExplanationCombinationBound")
            (publication,) = _events(journey, "GroundedReportPublished")
            for own, other in (("rs429358", "rs7412"), ("rs7412", "rs429358")):
                self.assertIn(publication["event_id"], scope[own])
                self.assertLess(scope[own], set(whole["payload"]["evidence_event_ids"]))
                # Some of each member's evidence names its request alone.
                self.assertTrue(scope[own] - scope[other])
            self.assertNotIn(
                outside,
                {
                    item["event_id"]
                    for item in explainer.tasks["rs429358"][0]["evidence"]
                },
            )
            # Each target the model is sent carries its typed owners' plain words,
            # not only its claim code.
            searched = {
                "exact_allele_observed": "Found: the sample carries the target allele",
                "locus_overlapped": "A nearby variant call overlaps this position",
                "no_record": (
                    "Searched the VCF: no variant call here; genotype not established"
                ),
            }
            targets = explainer.tasks["rs429358"][0]["targets"]
            self.assertEqual(
                [(target["kind"], target["words"]) for target in targets],
                [("request", "each position proposed for it is its own target")]
                + [("position", searched[target["outcome"]]) for target in targets[1:]],
            )
            self.assertGreater(len(targets), 1)
            rs429358 = next(
                item
                for item in bound.values()
                if item["payload"]["targets"][0]["subject"] == "rs429358"
            )
            searches = [
                (
                    event["payload"]["arguments"],
                    event["payload"]["status"],
                    event["payload"]["errors"],
                )
                for event in _events(journey, "ToolCallValidated")
                if event["payload"]["tool_name"] == "search_journey"
                and event["payload"]["attempt_id"] in steps_of(journey)
            ]
            self.assertEqual(
                searches,
                [
                    ({"event_ids": [outside]}, "invalid", ["search_outside_scope"]),
                    (
                        {"kinds": ["InvestigationEvidenceRecorded"], "limit": 10},
                        "valid",
                        [],
                    ),
                ],
            )
            (read,) = [
                event["payload"]
                for event in _events(journey, "JourneySearchRead")
                if event["payload"]["attempt_id"] in steps_of(journey)
            ]
            self.assertEqual(read["query"]["scope_event_id"], rs429358["event_id"])
            found = {item["event_id"] for item in read["result"]["events"]}
            self.assertTrue(found)
            self.assertLessEqual(found, set(rs429358["payload"]["evidence_event_ids"]))
            # Only the scoped search and the step's own output Tool are offered.
            self.assertEqual(
                {
                    step: set().union(*offered)
                    for step, offered in explainer.offered.items()
                },
                {
                    "rs429358": {
                        "search_journey",
                        "ask_researcher",
                        "record_explanation",
                    },
                    "rs7412": {
                        "search_journey",
                        "ask_researcher",
                        "record_explanation",
                    },
                    "combination": {
                        "search_journey",
                        "ask_researcher",
                        "record_combined_answer",
                    },
                },
            )
            # Each request names the same Tools as callable now, none
            # waiting.
            for step, views in explainer.views.items():
                for view in views:
                    with self.subTest(step=step):
                        self.assertEqual(
                            (set(view["callable_now"]), view["waiting"]),
                            (set().union(*explainer.offered[step]), []),
                        )

    def test_a_draft_in_387s_shape_gets_the_checks_one_correction_and_repairs(
        self,
    ):
        # A recorded fault (17 of 20 drafts): a claim list beside the answer and
        # its references under another name. The framework refuses its shape
        # first; the correction the member reads is the one check's, which the
        # record keeps, not the schema's dump.
        def as_387_sent(task):
            draft = member_draft(task)
            return {
                "why": draft["why"],
                "references": draft["why_references"],
                "claims": [
                    {"row": row["number"], "claim": row["claim"]}
                    for row in task["targets"]
                ],
                "limitations": draft["limitations"],
            }

        def scalar_qualifications(task):
            # Standards ST1 / Spec SP1 r1: a wrong JSON type inside a limitation
            # is a refused output the member repairs, not an error.
            draft = member_draft(task)
            draft["limitations"][0]["qualifications"] = 7
            return draft

        explainer = ExplainerModel(
            drafts={"rs7412": [as_387_sent], "rs429358": [scalar_qualifications]}
        )
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                code, output, error = self.explain(journey, report.report_digest)
            self.assertEqual((code, error), (0, ""), output)
            steps = steps_of(journey)
            validations = {}
            for event in _events(journey, "AgentOutputValidated"):
                payload = event["payload"]
                if payload["agent_role"] != "report_explainer":
                    continue
                validations.setdefault(steps[payload["attempt_id"]], []).append(
                    (payload["status"], payload["errors"], payload.get("feedback"))
                )
            published = self.published(journey)
        scalar = validations.pop("rs429358")
        self.assertEqual(
            [status for status, _errors, _feedback in scalar], ["invalid", "valid"]
        )
        self.assertEqual(scalar[0][1][0], "limitations.0.qualifications:tuple_type")
        self.assertTrue(
            scalar[0][1][-1].startswith("explanation_qualifications_missing:")
        )
        self.assertEqual(
            {step: [item[:2] for item in items] for step, items in validations.items()},
            {
                "rs7412": [
                    (
                        "invalid",
                        [
                            "claims:extra_forbidden",
                            "references:extra_forbidden",
                            "why_references:missing",
                            # Recorded drafts listed no genotype; the list is
                            # part of the draft's shape.
                            "genotypes:missing",
                        ],
                    ),
                    ("valid", []),
                ],
                "combination": [("valid", [])],
            },
        )
        # Only the refused member reads the correction: the check's, as recorded
        # (the record keeps the framework's closing line too).
        (correction,) = explainer.feedback["rs7412"][1]
        self.assertTrue(validations["rs7412"][0][2].startswith(correction))
        self.assertIn("- claims is no field of this output", correction)
        self.assertIn("- why_references is missing", correction)
        self.assertNotIn('"loc"', correction)
        # Code filled the claims the model no longer writes.
        self.assertEqual(
            [claim["claim"] for claim in published["requests"][1]["claims"]],
            ["candidates", "searched_no_record"],
        )

    def test_a_member_left_unadmitted_is_settled_not_explained_by_its_reject(self):
        self._settled_not_explained("reject")

    def test_a_dismissed_member_is_settled_not_explained_and_the_rest_go_on(self):
        # Dismiss settles only this request's member, as its reject does.
        self._settled_not_explained("dismiss")

    def test_a_members_question_is_observed_and_its_answer_reenters_the_member(self):
        # The shared researcher question through the public reader: the member's
        # Review shows it, the continue answer re-enters that member with the
        # instruction, and observation and reopen hold its exact return.
        from bio_neb.application.advancement import ObserveRun, observe
        from bio_neb.application.reviews import pending_reviews
        from bio_neb.reporting.report_commands import read_report_explanations

        explainer = _AskingExplainer("rs7412")
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                code, output, error = self.explain(journey, report.report_digest)
            self.assertEqual((code, error), (0, ""), output)
            held = observe(ObserveRun(journey_file=str(journey.path)))
            (review,) = pending_reviews(held)
            request = review.request
            self.assertEqual(
                (review.kind, request.agent_role, request.cause),
                ("agent_work", "report_explainer", "researcher_question"),
            )
            self.assertEqual(request.question, "Which allele should the answer name?")
            self.assertIn("dismiss", [item.action for item in request.legal_actions])
            with self.dependencies(store, report_model_factory=explainer.model):
                self.answer(
                    journey,
                    review.request_event_id,
                    {"action": "continue", "instruction": "Name the T allele."},
                )
            (explained,) = read_report_explanations(str(journey.path)).requests
            self.assertEqual(explained.status, "published")
            self.assertEqual(
                explainer.tasks["rs7412"][-1]["researcher_instruction"],
                "Name the T allele.",
            )
            observe(ObserveRun(journey_file=str(journey.path)))

    def _settled_not_explained(self, action):
        from bio_neb.application.advancement import ObserveRun, observe
        from bio_neb.application.reviews import pending_reviews
        from bio_neb.reporting.report_commands import read_report_explanations

        def wrong(task):
            draft = member_draft(task)
            draft["why_references"] = ["outside-the-evidence-index"]
            return draft

        # The Run's two failed turns: two refusals stop only rs7412's member.
        explainer = ExplainerModel(drafts={"rs7412": [wrong] * 2})
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            report_text = self.reopen(journey, report.report_digest)
            with self.dependencies(store, report_model_factory=explainer.model):
                code, output, error = self.explain(journey, report.report_digest)
            self.assertEqual((code, error), (0, ""), output)
            self.assertNotIn("combination", explainer.tasks)
            (request,) = read_report_explanations(str(journey.path)).requests
            self.assertEqual(request.status, "review_pending")
            self.assertEqual(
                [(member.status) for member in request.members],
                ["admitted", "review_pending"],
            )
            held = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertEqual(held.status, "grounded_report_published")
            (review,) = pending_reviews(held)
            self.assertEqual(
                (review.kind, review.request.agent_role),
                ("agent_work", "report_explainer"),
            )
            # The member's reject and dismiss say what they do to this request alone.
            effects = {
                item.action: item.effect for item in review.request.legal_actions
            }
            self.assertIn("settles this request as not explained", effects["reject"])
            self.assertIn("settled as not explained", effects["dismiss"])
            with self.dependencies(store, report_model_factory=explainer.model):
                self.answer(
                    journey,
                    review.request_event_id,
                    {
                        "action": action,
                        "reason": "Its citations never matched its evidence index.",
                        "asserted_by": "researcher:owner",
                    },
                )
            (request,) = read_report_explanations(str(journey.path)).requests
            self.assertEqual(request.status, "published")
            (combined,) = explainer.tasks["combination"]
            self.assertEqual(
                combined["requests"][1],
                {
                    "request": 3,
                    "subject": "rs7412",
                    "status": "not_explained",
                    "reason": "Its citations never matched its evidence index.",
                },
            )
            explanation = self.published(journey)
            self.assertEqual(
                [
                    (item["status"], item.get("reason"))
                    for item in explanation["requests"]
                ],
                [
                    ("explained", None),
                    (
                        "not_explained",
                        "Its citations never matched its evidence index.",
                    ),
                ],
            )
            self.assertEqual(
                explanation["request_claims"][1],
                {"request": 3, "claims": ["not_explained"]},
            )
            reopened = self.reopen_explanation(journey, explanation["explanation_id"])
            self.assertIn(
                "  request rs7412 (target 3): not explained: Its citations never matched "
                "its evidence index.\n",
                reopened,
            )
            self.assertEqual(self.reopen(journey, report.report_digest), report_text)
            # After its Review the readable report says why the request's words
            # are not written, beside the facts code copied.
            readable = self.readable(journey, report.report_digest)
            self.assertIn(
                "| **T2** | rs7412 | you | NC_000019.10:44908822 T>C | no call |",
                readable,
            )
            self.assertIn(
                "- **T2 rs7412:** Not written: Its citations never matched its "
                "evidence index.\n",
                readable,
            )

    def test_a_member_stopped_at_its_effort_review_is_not_explained_and_the_rest_go_on(
        self,
    ):
        # Spec r1 S4: an effort stop is the member's own decision too; it settles
        # that request as not explained and holds no combination.
        from bio_neb.agentic_journey import attempt_runner
        from bio_neb.agentic_journey.continuation_policy import (
            AgentInvocationOverrides,
        )
        from bio_neb.application.advancement import ObserveRun, observe
        from bio_neb.application.reviews import pending_reviews
        from bio_neb.reporting.report_commands import read_report_explanations

        unit = attempt_runner.AttemptUnit

        def two_requests(assignment, reentry=None, invocation_overrides=None):
            # Each member's request budget is two; its sibling needs one.
            return unit(
                assignment,
                reentry or attempt_runner.Reentry(None, None),
                AgentInvocationOverrides(model_requests=2),
            )

        searching = [
            lambda task, n=n: {"text": f"rs7412 {n}", "limit": 1} for n in range(3)
        ]
        explainer = ExplainerModel(searches={"rs7412": searching})
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with (
                self.dependencies(store, report_model_factory=explainer.model),
                patch.object(attempt_runner, "AttemptUnit", two_requests),
            ):
                self.assertEqual(self.explain(journey, report.report_digest)[0], 0)
                (review,) = pending_reviews(
                    observe(ObserveRun(journey_file=str(journey.path)))
                )
                self.assertEqual(
                    (review.kind, review.request.agent_role),
                    ("agent_effort", "report_explainer"),
                )
                self.answer(
                    journey,
                    review.request_event_id,
                    {"action": "stop", "reason": "It only searched."},
                )
            (request,) = read_report_explanations(str(journey.path)).requests
            self.assertEqual(request.status, "published")
            self.assertEqual(
                [member.status for member in request.members],
                ["admitted", "not_explained"],
            )
            self.assertEqual(
                [
                    (item["status"], item.get("reason"))
                    for item in self.published(journey)["requests"]
                ],
                [("explained", None), ("not_explained", "It only searched.")],
            )

    def test_a_continue_reenters_only_that_member_with_its_granted_repairs(self):
        from bio_neb.application.advancement import ObserveRun, observe
        from bio_neb.application.reviews import pending_reviews

        def wrong(task):
            draft = member_draft(task)
            draft["why_references"] = ["outside-the-evidence-index"]
            return draft

        # Two refusals are the Run's two failed turns; the continued member,
        # granted four, is refused three times more, then records a valid draft.
        explainer = ExplainerModel(drafts={"rs429358": [wrong] * 5})
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                self.assertEqual(self.explain(journey, report.report_digest)[0], 0)
                (review,) = pending_reviews(
                    observe(ObserveRun(journey_file=str(journey.path)))
                )
                held = self.answer(
                    journey,
                    review.request_event_id,
                    {"action": "continue", "failed_turns": 4},
                )
            self.assertFalse(pending_reviews(held))
            started = [
                event["payload"]
                for event in _events(journey, "AttemptStarted")
                if event["payload"]["agent_role"] == "report_explainer"
            ]
            bound = {
                event["payload"]["attempt_id"]: event["payload"]
                for event in _events(journey, "ReportExplanationMemberBound")
            }
            members = [
                (
                    bound[item["attempt_id"]]["targets"][0]["subject"],
                    item["continuation_policy"]["failed_turns"],
                )
                for item in started
                if item["attempt_id"] in bound
            ]
            self.assertEqual(sorted(members[:2]), [("rs429358", 2), ("rs7412", 2)])
            # Only the stopped member re-entered, under the granted failed turns, and
            # it continues its own earlier Attempt.
            self.assertEqual(members[2:], [("rs429358", 4)])
            third = started[2]
            self.assertEqual(third["invocation_overrides"], {"failed_turns": 4})
            first = next(
                attempt
                for attempt, payload in bound.items()
                if payload["targets"][0]["subject"] == "rs429358"
                and payload["continued_attempt_id"] is None
            )
            self.assertEqual(bound[third["attempt_id"]]["continued_attempt_id"], first)
            self.assertEqual(len(explainer.tasks["rs7412"]), 1)
            # The continued member is sent its last refused draft to correct.
            self.assertEqual(
                explainer.tasks["rs429358"][2]["previous_attempt"]["draft"],
                wrong(explainer.tasks["rs429358"][0]),
            )
            self.assertEqual(
                self.published(journey)["requests"][0]["status"], "explained"
            )

    def test_the_combined_answers_request_claims_are_its_members(self):
        # Code fills each request's claim words from its member's
        # section; a request list the model writes is refused.
        def dropped(task):
            draft = combination_draft(task)
            draft["requests"] = [{"row": 1, "claims": ["carries_target_allele"]}]
            return draft

        explainer = ExplainerModel(drafts={"combination": [dropped]})
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                code, output, error = self.explain(journey, report.report_digest)
            self.assertEqual((code, error), (0, ""), output)
            refused = [
                event["payload"]["errors"]
                for event in _events(journey, "AgentOutputValidated")
                if event["payload"]["status"] == "invalid"
            ]
            self.assertEqual(refused, [["requests:extra_forbidden"]])
            (correction,) = explainer.feedback["combination"][1]
            self.assertIn("- requests is no field of this output", correction)
            self.assertEqual(
                self.published(journey)["request_claims"],
                [
                    {"request": 1, "claims": ["candidates", "carries_target_allele"]},
                    {"request": 3, "claims": ["candidates", "searched_no_record"]},
                ],
            )

    def stopped_report(self, root: Path, store: RunOutputStore):
        """Shaped like retained Run #300: stopped before any lookup (#108)."""
        from bio_neb.application.advancement import ObserveRun, ResumeRun, advance
        from bio_neb.application.advancement import observe as observe_run
        from bio_neb.application.reviews import pending_reviews
        from bio_neb.investigation.investigation import ApproveInvestigationResponse
        from bio_neb.investigation.investigation_commands import (
            RespondInvestigationReviewCommand,
            respond_investigation_review,
        )

        journey = self.child(
            root, request_file("rs429358", "rs7412", "rs4149056", NOT_FOUND)
        )
        with self.dependencies(store, {"rs4149056": RS4149056}):
            (issue,) = pending_reviews(
                advance(ResumeRun(journey_file=str(journey.path)))
            )
            (review,) = pending_reviews(self.approve_issue(journey, issue))
            proposal = review.request.proposal
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=review.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve",
                        proposal_id=proposal.proposal_id,
                        work_item_ids=tuple(
                            item.work_item_id
                            for item in proposal.work_items
                            if item.target.one_based_position in {44908684, 44908822}
                        ),
                    ),
                )
            )
            (remaining,) = pending_reviews(
                observe_run(ObserveRun(journey_file=str(journey.path)))
            )
            self.answer(
                journey,
                remaining.request_event_id,
                {
                    "action": "stop",
                    "proposal_id": proposal.proposal_id,
                    "reason": "End the Run before the lookups.",
                },
            )
            report = publish_unexplained(journey, self.basis(journey))
        return journey, report

    def test_a_stopped_runs_members_claim_each_row_in_its_recorded_state(self):
        explainer = ExplainerModel()
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.stopped_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                code, output, error = self.explain(journey, report.report_digest)
            self.assertEqual((code, error), (0, ""), output)
            self.assertEqual(store.gets, [])
            self.assertEqual(report.outcome, "investigation_stopped")
            explanation = self.published(journey)
            self.assertEqual(
                [
                    (item["subject"], [claim["claim"] for claim in item["claims"]])
                    for item in explanation["requests"]
                ],
                [
                    ("rs429358", ["candidates", "approved_not_looked_up"]),
                    ("rs7412", ["candidates", "approved_not_looked_up"]),
                    ("rs4149056", ["candidates", "not_approved", "not_approved"]),
                    (NOT_FOUND, ["unresolved"]),
                ],
            )
            self.assertFalse(
                [
                    claim
                    for item in explanation["requests"]
                    for claim in item["claims"]
                    if "genotype" in claim
                ]
            )

    def test_qualifications_go_to_the_requests_whose_targets_they_concern(self):
        """#448 D2, Q5 (b): ids, never words, place each qualification, and each is
        given as its origin's code and plain words, never the report's audit
        sentence (#474 Q3 a). A limitation quoting those words, coverage words
        included, is admitted: ADR 0049 retires the word ban and the check is
        membership only (#474 owner Q2 a)."""
        from bio_neb.investigation.qualification_origins import plain_words

        def quoted(task):
            draft = member_draft(task)
            draft["limitations"][0]["text"] = "As the report records: " + " ".join(
                item["words"] for item in task["qualifications"]
            )
            return draft

        explainer = ExplainerModel(drafts={"rs429358": [quoted]})
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                code, output, error = self.explain(journey, report.report_digest)
            self.assertEqual((code, error), (0, ""), output)
            expected = [
                (kind, item.origin.code, plain_words(item.origin))
                for kind, items in (
                    ("limitation", report.limitations),
                    ("conflict", report.conflicts),
                    ("gap", report.gaps),
                )
                for item in items
            ]
            explanation = self.published(journey)
            self.assertEqual(
                [
                    (item["kind"], item["origin"], item["words"])
                    for item in explanation["qualifications"]
                ],
                expected,
            )
            # Each position's depth limitation and its own lookup's claim scope
            # concern one position, so they are its request's; the limitations both
            # searched positions state are the combination's.
            given = {
                step: [
                    (item["origin"], item["words"])
                    for item in tasks[0]["qualifications"]
                ]
                for step, tasks in explainer.tasks.items()
            }
            scope = (
                "observation_scope",
                "The result holds for this sample's VCF at this target only.",
            )
            # The Run measured both positions: rs429358 under the default BAM's
            # reads, rs7412 off them, below the minimum.
            self.assertEqual(given["rs429358"], [scope])
            self.assertEqual(
                given["rs7412"],
                [
                    scope,
                    (
                        "depth_below_minimum",
                        "Read depth here was 0 at its lowest, below the minimum of 10.",
                    ),
                ],
            )
            numbers = {
                step: [item["qualification"] for item in tasks[0]["qualifications"]]
                for step, tasks in explainer.tasks.items()
            }
            self.assertEqual(
                sorted(numbers["combination"]),
                sorted(
                    set(range(1, len(expected) + 1))
                    - set(numbers["rs429358"] + numbers["rs7412"])
                ),
            )
            self.assertTrue(numbers["combination"])
            # Every qualification is named once, by its owner.
            named = sorted(
                number
                for limitation in (
                    *explanation["limitations"],
                    *(
                        limitation
                        for item in explanation["requests"]
                        for limitation in item["limitations"]
                    ),
                )
                for number in limitation["qualifications"]
            )
            self.assertEqual(named, list(range(1, len(expected) + 1)))
            # The quote of the report's words was admitted on its first draft.
            self.assertEqual(
                [
                    event["payload"]["errors"]
                    for event in _events(journey, "AgentOutputValidated")
                    if event["payload"]["status"] == "invalid"
                ],
                [],
            )
            for number, (kind, _code, words) in enumerate(expected, start=1):
                self.assertIn(f"  Q{number} {kind}: {words}\n", output)

    def test_an_unavailable_model_stops_one_member_and_a_retry_resumes_it(self):
        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.application.advancement import ObserveRun, observe
        from bio_neb.application.reviews import pending_reviews
        from bio_neb.reporting.report_commands import read_report_explanations

        explainer = ExplainerModel()
        failures = [ModelHTTPError(503, "offline", headers={"retry-after": "999"})]

        def rs7412_once_unavailable(messages, info):  # noqa: ANN001, ANN202
            if failures and ExplainerModel.step(_sent_task(messages)) == "rs7412":
                raise failures.pop()
            return explainer.respond(messages, info)

        model = ScriptedModelEndpoint(rs7412_once_unavailable)
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            report_text = self.reopen(journey, report.report_digest)
            with self.dependencies(store, report_model_factory=model):
                code, output, error = self.explain(journey, report.report_digest)
                self.assertEqual((code, error), (0, ""), output)
                self.assertIn("Explanation request: review_pending\n", output)
                held = observe(ObserveRun(journey_file=str(journey.path)))
                self.assertEqual(held.status, "grounded_report_published")
                (review,) = pending_reviews(held)
                self.assertEqual(
                    (review.kind, review.request.agent_role),
                    ("model_availability", "report_explainer"),
                )
                (request,) = read_report_explanations(str(journey.path)).requests
                self.assertEqual(
                    [member.status for member in request.members],
                    ["admitted", "review_pending"],
                )
                self.answer(journey, review.request_event_id, {"action": "retry"})
            (request,) = read_report_explanations(str(journey.path)).requests
            self.assertEqual(request.status, "published")
            self.assertEqual(
                [len(member.attempt_ids) for member in request.members], [1, 2]
            )
            self.assertEqual(self.reopen(journey, report.report_digest), report_text)

    def test_an_unavailable_model_stopped_settles_only_its_member_not_explained(self):
        # Member scope covers every stop kind: a stop on the availability Review
        # settles that request as not explained and the combination still answers
        # over the others.
        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.application.advancement import ObserveRun, observe
        from bio_neb.application.reviews import pending_reviews
        from bio_neb.reporting.report_commands import read_report_explanations

        explainer = ExplainerModel()

        def rs7412_unavailable(messages, info):  # noqa: ANN001, ANN202
            if ExplainerModel.step(_sent_task(messages)) == "rs7412":
                raise ModelHTTPError(503, "offline", headers={"retry-after": "999"})
            return explainer.respond(messages, info)

        model = ScriptedModelEndpoint(rs7412_unavailable)
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=model):
                self.assertEqual(self.explain(journey, report.report_digest)[0], 0)
                (review,) = pending_reviews(
                    observe(ObserveRun(journey_file=str(journey.path)))
                )
                self.assertEqual(
                    (review.kind, review.request.agent_role),
                    ("model_availability", "report_explainer"),
                )
                self.answer(
                    journey,
                    review.request_event_id,
                    {"action": "stop", "reason": "The endpoint stays offline."},
                )
            (request,) = read_report_explanations(str(journey.path)).requests
            self.assertEqual(request.status, "published")
            self.assertEqual(
                [member.status for member in request.members],
                ["admitted", "not_explained"],
            )
            self.assertEqual(
                [
                    (item["status"], item.get("reason"))
                    for item in self.published(journey)["requests"]
                ],
                [("explained", None), ("not_explained", "The endpoint stays offline.")],
            )

    def test_a_rewrite_keeps_the_prior_version_and_requests_are_refused_by_cause(self):
        from bio_neb.reporting.report_commands import read_report_explanations

        def worded(text):
            def draft(task):
                value = combination_draft(task)
                value["conclusion"] = text
                return value

            return draft

        explainer = ExplainerModel(drafts={"combination": [worded("First wording.")]})
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            before = journey.path.read_bytes()
            # No report of this digest was published: refused before any write.
            code, _output, error = self.explain(journey, "grounded-report:" + "0" * 64)
            self.assertEqual(code, 2)
            self.assertEqual(
                json.loads(error)["code"], "report_explanation_report_absent"
            )
            self.assertEqual(journey.path.read_bytes(), before)
            with self.dependencies(store, report_model_factory=explainer.model):
                self.assertEqual(self.explain(journey, report.report_digest)[0], 0)
            (first,) = read_report_explanations(str(journey.path)).explanations
            first_id = first.explanation.explanation_id

            # A second explanation must name the version it rewrites.
            before = journey.path.read_bytes()
            code, _output, error = self.explain(journey, report.report_digest)
            self.assertEqual(
                (code, json.loads(error)["code"]), (2, "report_explanation_exists")
            )
            code, _output, error = self.explain(
                journey,
                report.report_digest,
                "--supersedes",
                "report-explanation:" + "1" * 64,
            )
            self.assertEqual(
                (code, json.loads(error)["code"]),
                (2, "report_explanation_prior_unknown"),
            )
            self.assertEqual(journey.path.read_bytes(), before)

            explainer.drafts["combination"] = [worded("Second wording.")]
            with self.dependencies(store, report_model_factory=explainer.model):
                code, output, error = self.explain(
                    journey, report.report_digest, "--supersedes", first_id
                )
            self.assertEqual((code, error), (0, ""), output)
            self.assertIn(f"supersedes: {first_id}\n", output)
            first_again, second = read_report_explanations(
                str(journey.path)
            ).explanations
            self.assertEqual(first_again, first)
            self.assertEqual(
                (
                    second.explanation.supersedes_explanation_id,
                    second.explanation.report_basis_digest,
                ),
                (first_id, first.explanation.report_basis_digest),
            )
            # Each version reopens as published, without a write.
            before = journey.path.read_bytes()
            reopened = self.reopen_explanation(journey, first_id)
            self.assertIn(f"explanation id: {first_id}\n", reopened)
            self.assertIn("First wording.", reopened)
            self.assertEqual(journey.path.read_bytes(), before)

    def test_a_journey_carrying_a_one_attempt_explanation_is_refused_by_name(self):
        # The one-Attempt explanation (/1) and its binding are retired facts. A
        # Journey carrying either is refused by name, before any write, rather
        # than misread as a fresh request or failing untyped.
        from bio_neb.application.queries import project_journey_events
        from bio_neb.reporting.explanation import project_report_explanations
        from bio_neb.reporting.report_commands import (
            ReportExplanationRefused,
            read_report_explanations,
        )

        explainer = ExplainerModel()
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                self.assertEqual(self.explain(journey, report.report_digest)[0], 0)
            events = read_evidence(Journey.open(journey.path))
            (published,) = [
                e for e in events if e["event_type"] == "ReportExplanationPublished"
            ]
            explanation_id = published["payload"]["explanation"]["explanation_id"]

            def one_attempt_binding(hostile):
                bound = next(
                    e
                    for e in hostile
                    if e["event_type"] == "ReportExplanationMemberBound"
                )
                bound["event_type"] = "ReportExplanationBound"

            def one_attempt_artifact(hostile):
                explanation = next(
                    e
                    for e in hostile
                    if e["event_type"] == "ReportExplanationPublished"
                )["payload"]["explanation"]
                explanation["schema_version"] = "bio-neb.report-explanation/1"

            for drift, refusal in (
                (one_attempt_binding, "retired_journey_fact:ReportExplanationBound"),
                (
                    one_attempt_artifact,
                    "retired_journey_fact:bio-neb.report-explanation/1",
                ),
            ):
                hostile = copy.deepcopy(events)
                drift(hostile)
                # The explanation projection, and the whole Journey's read.
                for name, reader in (
                    ("explanations", project_report_explanations),
                    (
                        "journey",
                        lambda hostile: project_journey_events(
                            hostile, journey_file=str(journey.path)
                        ),
                    ),
                ):
                    with (
                        self.subTest(drift=drift.__name__, reader=name),
                        self.assertRaisesRegex(ValueError, f"^{refusal}$"),
                    ):
                        reader(hostile)
            # A retained Journey that recorded the binding explains nothing more:
            # the command refuses it by name, sends nothing and writes nothing.
            journey = Journey.open(journey.path)
            journey.record(
                "ReportExplanationBound", {}, [journey.read()[-1]["event_id"]]
            )
            before, sent = journey.path.read_bytes(), len(explainer.order)
            with self.dependencies(store, report_model_factory=explainer.model):
                code, _output, error = self.explain(
                    journey, report.report_digest, "--supersedes", explanation_id
                )
            self.assertEqual(
                (code, json.loads(error)["code"]),
                (2, "report_explanation_unverifiable"),
            )
            self.assertEqual(len(explainer.order), sent)
            self.assertEqual(journey.path.read_bytes(), before)
            # Its cause names the retired fact.
            with self.assertRaises(ReportExplanationRefused) as refused:
                read_report_explanations(str(journey.path))
            self.assertEqual(
                str(refused.exception.__cause__),
                "retired_journey_fact:ReportExplanationBound",
            )

    def test_a_malformed_explanation_publication_is_refused_by_the_public_reader(self):
        # Spec r1 SP1: the retired-schema check reads a publication's
        # explanation only when it is an object; any other shape still meets the
        # public typed refusal, with nothing written.
        from bio_neb.reporting.report_commands import (
            ReportExplanationRefused,
            read_report_explanations,
        )

        explainer = ExplainerModel()
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                self.assertEqual(self.explain(journey, report.report_digest)[0], 0)
            (published,) = [
                e
                for e in read_evidence(Journey.open(journey.path))
                if e["event_type"] == "ReportExplanationPublished"
            ]
            for shape in (None, [], "bio-neb.report-explanation/1"):
                with self.subTest(shape=shape):
                    copy_root = Path(directory) / f"copy-{type(shape).__name__}"
                    copied = copy_root / journey.path.parent.name
                    shutil.copytree(journey.path.parent, copied)
                    hostile = Journey.open(copied / journey.path.name)
                    hostile.record(
                        "ReportExplanationPublished",
                        {**published["payload"], "explanation": shape},
                        [hostile.read()[-1]["event_id"]],
                    )
                    before = hostile.path.read_bytes()
                    with self.assertRaises(ReportExplanationRefused) as refused:
                        read_report_explanations(str(hostile.path))
                    self.assertEqual(
                        refused.exception.code, "report_explanation_unverifiable"
                    )
                    self.assertIsInstance(refused.exception.__cause__, ValueError)
                    self.assertEqual(hostile.path.read_bytes(), before)

    def test_a_binding_or_publication_that_drifts_from_its_lineage_is_refused(self):
        from bio_neb.reporting.explanation import project_report_explanations

        explainer = ExplainerModel()
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                self.assertEqual(self.explain(journey, report.report_digest)[0], 0)
            events = read_evidence(Journey.open(journey.path))
        outside = next(
            e["event_id"] for e in events if e["event_type"] == "ModelRequestRecorded"
        )
        # The unchanged Journey reads.
        project_report_explanations(events)

        def first(hostile, event_type):
            return next(e for e in hostile if e["event_type"] == event_type)

        def widened(hostile):
            payload = first(hostile, "ReportExplanationMemberBound")["payload"]
            payload["evidence_event_ids"] = sorted(
                [*payload["evidence_event_ids"], outside]
            )

        def another_requests_row(hostile):
            members = [
                e for e in hostile if e["event_type"] == "ReportExplanationMemberBound"
            ]
            payload = members[0]["payload"]
            payload["targets"] = [
                *payload["targets"],
                members[1]["payload"]["targets"][-1],
            ]

        def dropped_qualification(hostile):
            payload = first(hostile, "ReportExplanationMemberBound")["payload"]
            payload["qualifications"] = payload["qualifications"][:-1]

        def other_section(hostile):
            payload = first(hostile, "ReportExplanationCombinationBound")["payload"]
            payload["sections"][0]["claims"] = ["candidates"]

        def other_claim(hostile):
            payload = first(hostile, "ReportExplanationMemberAdmitted")["payload"]
            payload["claims"][-1]["claim"] = "target_allele_not_carried"

        def substituted(hostile):
            payload = first(hostile, "ReportExplanationPublished")["payload"]
            payload["explanation"]["conclusion"]["text"] = "Something else."

        def start(hostile, bound_type):
            attempt = first(hostile, bound_type)["payload"]["attempt_id"]
            return next(
                e["payload"]
                for e in hostile
                if e["event_type"] == "AttemptStarted"
                and e["payload"]["attempt_id"] == attempt
            )

        # Standards r1 S1: the scope the shared Reviews read from a start is its
        # binding's work.
        def held_from_the_report(hostile):
            start(hostile, "ReportExplanationMemberBound")["holds_from_event_id"] = (
                first(hostile, "GroundedReportPublished")["event_id"]
            )

        def another_member(hostile):
            start(hostile, "ReportExplanationMemberBound")["member_id"] = "member-x"

        def combination_as_a_member(hostile):
            start(hostile, "ReportExplanationCombinationBound")["member_id"] = first(
                hostile, "ReportExplanationMemberBound"
            )["payload"]["request_member_id"]

        for drift, refusal in (
            (widened, "changed its approved report binding"),
            (another_requests_row, "binds one request's own targets"),
            (dropped_qualification, "changed its approved report binding"),
            (other_section, "changed its approved report binding"),
            (other_claim, "changed its accepted output"),
            (substituted, "identity changed|changed its accepted output"),
            (held_from_the_report, "changed its hold scope"),
            (another_member, "changed its hold scope"),
            (combination_as_a_member, "changed its hold scope"),
        ):
            hostile = copy.deepcopy(events)
            drift(hostile)
            with (
                self.subTest(drift=drift.__name__),
                self.assertRaisesRegex(ValueError, refusal),
            ):
                project_report_explanations(hostile)

    def test_public_reopen_refuses_an_explanation_whose_attempt_lifecycle_is_invalid(
        self,
    ):
        from bio_neb.cli import main
        from bio_neb.reporting.explanation import project_report_explanations
        from bio_neb.reporting.report_commands import read_report_explanations

        explainer = ExplainerModel()
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                self.assertEqual(self.explain(journey, report.report_digest)[0], 0)
            (published,) = read_report_explanations(str(journey.path)).explanations
            explanation_id = published.explanation.explanation_id
            # Only one member Attempt's start changes: its lifecycle parent
            # becomes the request, which is no lifecycle start.
            lines = journey.path.read_text(encoding="utf-8").splitlines()
            events = [json.loads(line) for line in lines]
            self.assertEqual([canonical_json(event) for event in events], lines)
            request = next(
                e for e in events if e["event_type"] == "ReportExplanationRequested"
            )
            started = next(
                e
                for e in events
                if e["event_type"] == "AttemptStarted"
                and e["payload"]["agent_role"] == "report_explainer"
            )
            started["parent_ids"] = [request["event_id"]]
            journey.path.write_text(
                "".join(canonical_json(event) + "\n" for event in events),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "lacks lifecycle evidence"):
                project_report_explanations(read_evidence(Journey.open(journey.path)))
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                code = main(
                    [
                        "report-explanation-reopen",
                        "--journey-file",
                        str(journey.path),
                        "--explanation-id",
                        explanation_id,
                    ]
                )
            self.assertEqual(
                (code, json.loads(error.getvalue())["code"]),
                (2, "report_explanation_unverifiable"),
            )

    def test_each_step_reads_its_own_stage_text_in_its_models_form(self):
        """Each Attempt sends the text it recorded, in its model's form (#239,
        #379), and only its own step's section of it."""
        import shutil

        from bio_neb.agentic_journey import role_instructions as module
        from bio_neb.agentic_journey.instruction_evidence import (
            read_attempt_instructions,
        )
        from bio_neb.agentic_journey.role_instructions import (
            instruction_blocks,
            stage_text,
        )

        # A test-local Qwen form text gives each step its own words, so a request
        # that read the default form's text would show the base wording.
        base = {b.name: b.text for b in instruction_blocks("report_explainer")}
        own = {
            "member_call": base["member_call"].replace(
                "You explain one request", "In this form's words, explain one request"
            ),
            "combination_call": base["combination_call"].replace(
                "You write the report's title and its conclusion",
                "In this form's words, title and conclude",
            ),
        }
        self.assertTrue(all(own[name] != base[name] for name in own))
        module.role_instructions.cache_clear()
        self.addCleanup(module.role_instructions.cache_clear)
        explainer = ExplainerModel()
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "instructions"
            shutil.copytree(
                Path(str(module._published("report_explainer.v20.blocks.toml"))).parent,
                package,
            )
            (package / "report_explainer.v20.qwen_3_5.toml").write_text(
                "".join(
                    f"[[block]]\nname = {json.dumps(name)}\ntext = {json.dumps(text)}\n"
                    for name, text in own.items()
                ),
                encoding="utf-8",
            )
            store = RunOutputStore()
            journey, report = self.observed_report(
                (Path(directory) / "run").resolve(), store
            )
            with (
                self.dependencies(store, report_model_factory=explainer.model),
                patch.object(module, "_published", lambda name: package / name),
                patch.dict(
                    module._FORM_TEXTS, {"report_explainer": frozenset({"qwen_3_5"})}
                ),
            ):
                code, output, error = self.explain(journey, report.report_digest)
            self.assertEqual((code, error), (0, ""), output)
            events = read_evidence(Journey.open(journey.path))
            stages = {
                event["payload"]["attempt_id"]: "report_explanation"
                for event in events
                if event["event_type"] == "ReportExplanationMemberBound"
            } | {
                event["payload"]["attempt_id"]: "report_combination"
                for event in events
                if event["event_type"] == "ReportExplanationCombinationBound"
            }
            starts = [
                event
                for event in events
                if event["event_type"] == "AttemptStarted"
                and event["payload"]["agent_role"] == "report_explainer"
            ]
            self.assertEqual(len(starts), 3)
            for start in starts:
                attempt_id = start["payload"]["attempt_id"]
                self.assertEqual(
                    start["payload"]["instruction_identity"],
                    "report_explainer:v20:qwen_3_5",
                )
                retained = read_attempt_instructions(
                    Journey.open(journey.path), attempt_id
                )
                requests = [
                    event
                    for event in events
                    if event["event_type"] == "ModelRequestRecorded"
                    and event["payload"]["attempt_id"] == attempt_id
                ]
                self.assertTrue(requests)
                for request in requests:
                    messages = request["payload"]["compatible_adapter_evidence"][
                        "request"
                    ]["messages"]
                    sent = [
                        m["instructions"] for m in messages if m.get("instructions")
                    ]
                    # The stage's text, then the request's view.
                    text, _, view = sent[-1].strip().rpartition("\n")
                    self.assertEqual(
                        text, stage_text(retained.text, stages[attempt_id]).strip()
                    )
                    self.assertIn("callable_now", json.loads(view))
                    step = (
                        "member_call"
                        if stages[attempt_id] == "report_explanation"
                        else "combination_call"
                    )
                    other = ({"member_call", "combination_call"} - {step}).pop()
                    self.assertIn(own[step].strip(), sent[-1])
                    self.assertNotIn(base[step].strip(), sent[-1])
                    self.assertNotIn(own[other].strip(), sent[-1])
                    self.assertNotIn(base[other].strip(), sent[-1])

    def test_explanation_reopen_reads_the_report_from_the_rebuild_snapshot(self):
        # Explainer commands once spent minutes before the Attempt's binding:
        # each explanation Attempt's reopen derived the whole report again on a
        # snapshot of its own.
        import bio_neb.reporting.grounded_report as grounded_report
        from bio_neb.application.advancement import ObserveRun, observe

        explainer = ExplainerModel()
        with tempfile.TemporaryDirectory() as directory:
            store = RunOutputStore()
            journey, report = self.observed_report(Path(directory).resolve(), store)
            with self.dependencies(store, report_model_factory=explainer.model):
                code, output, error = self.explain(journey, report.report_digest)
            self.assertEqual((code, error), (0, ""), output)
            detached = []
            original = grounded_report.PrefixSnapshot

            def snapshot(events, **kwargs):
                detached.append(len(events))
                return original(events, **kwargs)

            with patch.object(grounded_report, "PrefixSnapshot", snapshot):
                observation = observe(ObserveRun(journey_file=str(journey.path)))

        self.assertEqual(observation.status, "grounded_report_published")
        self.assertEqual(detached, [], "report prefixes derived on a detached snapshot")

    def test_one_explain_derives_each_requests_report_once(self):
        # A recorded rehearsal: one report-explain derived each request's report
        # on 185 prefixes, 250 s of 703 s; the prefix before a request never
        # changes, so the command derives its report once for every reader.
        from collections import Counter

        import bio_neb.reporting.grounded_report as grounded_report

        explainer = ExplainerModel()
        derived = Counter()
        original = grounded_report.project_grounded_report

        def counted(events, **kwargs):
            if kwargs.get("report_digest") is not None:
                derived[(len(events), kwargs["report_digest"])] += 1
            return original(events, **kwargs)

        baseline = self.prerequisite(_OBSERVED_REPORT)
        journey = Journey.open(baseline.journey_file)
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                self.dependencies(
                    RunOutputStore(), report_model_factory=explainer.model
                )
            )
            for module in list(sys.modules.values()):
                if getattr(module, "project_grounded_report", None) is original:
                    stack.enter_context(
                        patch.object(module, "project_grounded_report", counted)
                    )
            code, output, error = self.explain(journey, baseline.value)

        self.assertEqual((code, error), (0, ""), output)
        self.assertEqual(len(explainer.order), 3)
        self.assertEqual(list(derived.values()), [1], derived)

    def test_one_explain_copies_the_whole_history_a_bounded_number_of_times(self):
        # The member Attempts read the owner's shared read-only evidence; each
        # whole-history copy (``Journey.read``) cost about 0.36 s per member on a
        # recorded Journey. A ceiling may only shrink, and equals the measured count.
        explainer = ExplainerModel()
        copies = []
        original = Journey.read

        def counted(journey):
            copies.append(journey.journey_id)
            return original(journey)

        baseline = self.prerequisite(_OBSERVED_REPORT)
        journey = Journey.open(baseline.journey_file)
        with (
            self.dependencies(RunOutputStore(), report_model_factory=explainer.model),
            patch.object(Journey, "read", counted),
        ):
            code, output, error = self.explain(journey, baseline.value)

        self.assertEqual((code, error), (0, ""), output)
        self.assertEqual(len(explainer.order), 3)
        self.assertLessEqual(len(copies), EXPLAIN_HISTORY_COPY_CEILING)

    def test_the_explanation_view_is_reread_when_the_active_journal_grows(self):
        # The view read the merged evidence but was held under the head
        # event, which a journal row does not move; it is the current
        # evidence's view, re-derived when the active Pass journal grows.
        from bio_neb.agentic_journey.source_contacts import SourceContactScopeStarted
        from bio_neb.recording import processing_pass, record_contact_evidence
        from bio_neb.reporting.explanation import project_report_explanations
        from bio_neb.reporting.explanation_agent import _explanations

        owner = Journey.open(self.prerequisite(_OBSERVED_REPORT).journey_file)
        with owner.processing(), processing_pass(owner):

            def scope(call):
                return record_contact_evidence(
                    owner,
                    SourceContactScopeStarted(
                        attempt_id="attempt-a",
                        tool_call_id=call,
                        tool_name="fetch_refsnp",
                    ),
                )

            scope("call-a")  # The first row starts the Pass.
            before = _explanations(owner)
            head = owner.head_event_id
            scope("call-b")
            after = _explanations(owner)
            evidence = read_evidence(owner)
            self.assertEqual(owner.head_event_id, head)
            self.assertEqual(evidence[-1]["event_type"], "SourceContactScopeStarted")
            self.assertIsNot(after, before)
            self.assertEqual(after, project_report_explanations(evidence))


if __name__ == "__main__":
    unittest.main()
