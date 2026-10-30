"""The paper reader in the product (ADR 0046, #383): reading items as Attempts of the shared
lifecycle over a Run's stored paper, span ids, values kept only where their span writes
them, each finding decided on its own, the reading's product facts and the evidence view.
"""

import contextlib
import hashlib
import inspect
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pydantic import ValidationError
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.agentic_journey.agent_roles import RuntimeBinding
from bio_neb.agentic_journey.assignment_context import SEARCH_PRECONDITIONS
from bio_neb.agentic_journey.instruction_evidence import read_attempt_instructions
from bio_neb.agentic_journey.role_instructions import role_instructions
from bio_neb.agentic_journey.runtime_config import read_runtime_config
from bio_neb.agentic_journey.tool_execution import (
    function_tool_contracts,
    negative_domain_result,
    validate_observed_function_tool_role,
)
from bio_neb.content import canonical_json, shown_to_model
from bio_neb.journey import Journey
from bio_neb.journey_contracts import semantic_families
from bio_neb.paper_reading.answer import (
    answer_markdown,
    findings_repair,
    judge,
    judge_all,
    kept_finding,
    merged_findings,
    paper_answer,
    reference_problems,
    repair_text,
    stated_reference,
    verify,
)
from bio_neb.paper_reading.contracts import (
    MAX_AREAS,
    FindingPointer,
    ItemAnswer,
    PaperAnswer,
    PaperDocument,
    ReadingPlan,
    ReferencePointer,
    TableCell,
    TableRow,
    ValuePointer,
)
from bio_neb.paper_reading.journey import PAYLOADS as READING_FACTS
from bio_neb.paper_reading.reader import (
    Paper,
    grep,
    lookup,
    outline,
    read,
    rows,
)
from bio_neb.paper_reading.reading import (
    RequestCounts,
    answer_question,
    found_in_code,
    item_problems,
    plan_problems,
)
from bio_neb.paper_reading.targets import paper_targets
from bio_neb.paper_reading.tools import READING_TOOLS, reading_admission_errors
from bio_neb.recording import read_evidence
from bio_neb.target_request import request_document
from bio_neb.target_request.paper_targets import PaperIdentity, candidate_id
from tests.model_endpoint_fakes import ScriptedModelEndpoint, corrections
from tests.paper_reading_support import (
    document,
    finding,
    kept_answer,
    reading_of,
    stored_journey,
    value,
)
from tests.proposal_information_fakes import runtime_configuration
from tools.paper_retrieval.__main__ import main

DOI = "10.1371/journal.pgen.1006651"

# The one request writer's wording beside the paper's line in a Run's /3 request file.
REQUEST_WRITERS = (
    request_document.request_items,
    request_document._unplaced,
    request_document._candidate,
    request_document._text,
)


def ask(
    root: Path,
    paper: Paper,
    question: str,
    model,
    *,
    config: Path,
    environ=None,
):
    """Read ``paper`` for ``question`` through the reading's product entry, the paper
    stored in a fresh Journey as a Run's intake stores it (#420)."""

    journey, paper_event_id = stored_journey(
        root / f"run-{len(list(root.glob('run-*')))}", paper
    )
    selections = read_runtime_config(config).runtime_selections()
    return answer_question(
        journey,
        paper_event_id=paper_event_id,
        question=question,
        runtime=RuntimeBinding(selections, paper_event_id),
        environ=environ or {"LOCAL_MODEL_KEY": "offline"},
        model_factory=model,
    )


# ---------------------------------------------------------------------------
# Answering a question: the model points, code fills
# ---------------------------------------------------------------------------

# s16's recorded identity: a recorded live run on s16 dog, a byte copy of its
# `candidates.json`; the retired candidates in it are not read.
S16_CANDIDATES = Path(__file__).parent / "fixtures/paper_retrieval/s16-candidates.json"
WHY_ASPRV1 = (
    "The paper reports this de novo missense variant in ASPRV1 as the likely cause of the "
    "dog's ichthyosis."
)
WHY_CANFAM = "The reads were mapped to CanFam 3.1, so the S1 positions are on it."


def s16_identity() -> PaperIdentity:
    recorded = json.loads(S16_CANDIDATES.read_text(encoding="utf-8"))
    return PaperIdentity.model_validate(recorded["identity"])


# s16 written small with its documents, so the reading Tools can read it (the recorded
# documents are not in the repository): the main text names CanFam 3.1 and ASPRV1's coding
# change; S1 is a workbook sheet whose first row locates ASPRV1, beside 27 other rows.
# Candidate ids derive from the values, so these are s16's recorded ids above.
S16_MAIN = (
    "# A de novo ASPRV1 variant in a dog with ichthyosis\n## Results\n"
    "The reads were mapped to the CanFam 3.1 reference genome assembly.\n"
    "We found a de novo missense variant in ASPRV1, c.1052T>C, p.(Leu351Pro), in the "
    "affected dog.\n"
)
S16_S1_ID = "supp:pgen.1006651.s002.xlsx"
S16_S1 = "## Sheet1 (29 rows)\nCHROM\tPOS\tREF\tALT\tGene\n" + "\n".join(
    [
        "10\t68587027\tA\tG\tASPRV1",
        *(f"{1 + n}\t{1000100 + 7 * n}\tC\tT\tGENE{n}" for n in range(27)),
    ]
)


def s16_like() -> Paper:
    identity = s16_identity()
    return Paper(
        identity,
        [
            document("main", S16_MAIN),
            PaperDocument(
                document_id=S16_S1_ID,
                kind="supplementary",
                url="https://journals.plos.org/plosgenetics/article/file?id=10.1371/journal.pgen.1006651.s002",
                title="pgen.1006651.s002.xlsx",
                text=S16_S1,
            ),
        ],
    )


# s16's span ids: main's "The reads were..." is its third line, "We found..." its fourth;
# S1's sheet is table t1, its header row r1 and ASPRV1's row r2.
CANFAM_REFERENCE = {"name": "CanFam 3.1", "span": "main:p3", "reason": WHY_CANFAM}
ASPRV1_WORDS = "ASPRV1, c.1052T>C, p.(Leu351Pro)"
ASPRV1_ROW = "10\t68587027\tA\tG\tASPRV1"


def s1_locus(row: str, contig: str, position: str, ref: str, alt: str) -> list[dict]:
    """The locus values an S1 row writes, each by its cell."""

    return [
        value("contig", contig, row, column="CHROM"),
        value("position", position, row, column="POS", number=int(position)),
        value("ref", ref, row, column="REF"),
        value("alt", alt, row, column="ALT"),
    ]


# The finding as the model reads it: the main text's gene and change names, and the S1
# row that locates it, each value by its span.
ASPRV1_FINDING = finding(
    value("gene", "ASPRV1", "main:p4"),
    value("change", "c.1052T>C", "main:p4"),
    value("change", "p.(Leu351Pro)", "main:p4"),
    *s1_locus("t1:r2", "10", "68587027", "A", "G"),
    reason=WHY_ASPRV1,
)


def reading(*replies: list[tuple[str, dict]], seen: list | None = None):
    """A recorded model that sends each reply in turn across the reading items: a reply is
    the Tool calls of one response, as (Tool name, arguments)."""

    queued = iter(replies)

    async def respond(messages, info):
        if seen is not None:
            seen.append(messages)
        calls = next(queued)
        return ModelResponse(
            parts=[
                ToolCallPart(name, arguments, tool_call_id=f"call-{len(messages)}-{n}")
                for n, (name, arguments) in enumerate(calls)
            ]
        )

    return ScriptedModelEndpoint(respond)


ASPRV1_AREA = {
    "area": "the ASPRV1 variant of the results and its S1 row",
    "views": ["s2", "t1"],
    "reason": "The question asks for the paper's most meaningful finding.",
}


def one_area(*answers: dict, plan: dict | None = None, seen: list | None = None):
    """A recorded model whose paper item proposes one area, then answers that area with each
    answer in turn (a refused one goes back for repair)."""

    plan = plan or {
        "reference": CANFAM_REFERENCE,
        "areas": [ASPRV1_AREA],
    }
    return reading(
        [("propose_reading_areas", plan)],
        *([("answer_reading_item", answer)] for answer in answers),
        seen=seen,
    )


def repairs(request) -> list[str]:
    return corrections(request[-1])


class PaperAnswerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = self.root / "runtime.json"
        self.config.write_text(json.dumps(runtime_configuration()), encoding="utf-8")
        self.paper = s16_like()

    def ask(self, model, question="What is the most meaningful finding in this paper?"):
        return ask(
            self.root,
            self.paper,
            question,
            model,
            config=self.config,
        )

    def test_the_most_meaningful_finding_is_kept_from_the_values_the_model_read(self):
        model = one_area({"findings": [ASPRV1_FINDING]})

        outcome = self.ask(model)

        answer = outcome.answer
        self.assertEqual(answer.reference.candidate.name, "CanFam 3.1")
        self.assertEqual(answer.reference.reason, WHY_CANFAM)
        [canfam_line] = answer.reference.candidate.sources
        self.assertEqual(
            canfam_line.text,
            "The reads were mapped to the CanFam 3.1 reference genome assembly.",
        )
        [kept] = answer.findings
        self.assertEqual((kept.reason, kept.gene), (WHY_ASPRV1, "ASPRV1"))
        # Code chose the quote: the span naming the change, first to last kept value.
        self.assertEqual(kept.passages[0].text, ASPRV1_WORDS)
        [variant] = kept.variants
        self.assertEqual(
            (variant.contig, variant.position, variant.ref, variant.alt, variant.hgvs),
            ("10", 68587027, "A", "G", ("c.1052T>C", "p.(Leu351Pro)")),
        )
        # The S1 row states the locus; the main text's line writes the change names.
        s1_line, main_line = variant.sources
        self.assertEqual((s1_line.document_id, s1_line.text), (S16_S1_ID, ASPRV1_ROW))
        self.assertIn(ASPRV1_WORDS, main_line.text)
        self.assertEqual([d.document_id for d in answer.documents], ["main", S16_S1_ID])
        # The paper item and its area, through main's shared Attempt lifecycle, in a
        # Journey that is no Run.
        self.assertEqual(outcome.requests.model_requests, 2)
        events = Journey.open(Path(outcome.journey_file)).read()
        self.assertEqual(
            [
                e["payload"]["agent_role"]
                for e in events
                if e["event_type"] == "AttemptStarted"
            ],
            ["paper_reader", "paper_reader"],
        )
        self.assertNotIn(
            "RunInputsRegistered", {event["event_type"] for event in events}
        )

    def test_a_reference_sent_as_json_text_is_read_as_the_object_it_writes(self):
        # Recorded live runs: under tool choice "auto" the endpoint returned the nullable
        # reference object as JSON text; it is the same object, so the plan is checked as
        # written and kept.
        model = one_area(
            {"findings": [ASPRV1_FINDING]},
            plan={"reference": json.dumps(CANFAM_REFERENCE), "areas": [ASPRV1_AREA]},
        )

        outcome = self.ask(model)

        self.assertEqual(outcome.answer.reference.candidate.name, "CanFam 3.1")
        self.assertEqual(len(outcome.answer.findings), 1)
        self.assertEqual(outcome.requests.refused_answers, 0)
        # Text that is not one JSON object stays the reference's own schema refusal.
        for text in ("CanFam 3.1", json.dumps([CANFAM_REFERENCE])):
            with self.subTest(text=text):
                with self.assertRaises(ValidationError) as refused:
                    ReadingPlan.model_validate({"reference": text, "areas": []})
                self.assertEqual(
                    [(e["loc"], e["type"]) for e in refused.exception.errors()],
                    [(("reference",), "model_type")],
                )

    def test_a_value_the_paper_does_not_write_goes_back_for_repair(self):
        seen = []
        wrong = finding(
            *ASPRV1_FINDING["values"][:4],
            value("position", "68587027", "t1:r2", column="POS", number=68587028),
            *ASPRV1_FINDING["values"][5:],
            reason=WHY_ASPRV1,
        )
        model = one_area(
            {"findings": [wrong]},
            {"findings": [{**ASPRV1_FINDING, "corrects": 1}]},
            seen=seen,
        )

        outcome = self.ask(model)

        [repair] = repairs(seen[2])
        self.assertIn("number_differs:position=68587027,number=68587028", repair)
        # The item's repair names the refused finding by number and its values.
        self.assertIn('Refused 1 (gene "ASPRV1" in main:p4, ', repair)
        validations = [
            e["payload"]
            for e in Journey.open(Path(outcome.journey_file)).read()
            if e["event_type"] == "AgentOutputValidated"
        ]
        # The paper item's plan, then the area's refused and repaired answers.
        self.assertEqual(
            [(v["status"], v["errors"]) for v in validations],
            [
                ("valid", []),
                (
                    "invalid",
                    ["number_differs:position=68587027,number=68587028@refused 1"],
                ),
                ("valid", []),
            ],
        )
        [kept] = outcome.answer.findings
        self.assertEqual([v.position for v in kept.variants], [68587027])
        self.assertEqual(outcome.requests.model_requests, 3)

    def test_a_value_at_an_unknown_span_is_repaired_and_the_found_one_kept(self):
        text, paper = s3_like()
        span = f"main:p{paragraph(text, S3_LINE)}"
        seen = []

        def reading_of(at, **more):
            return {
                "findings": [
                    finding(
                        value("gene", "PBP2", at),
                        value("change", "Y127X", at),
                        reason="The paper reports PBP2 (Y127X).",
                        **more,
                    )
                ]
            }

        model = one_area(
            reading_of("main:p99"),
            reading_of(span, corrects=1),
            plan={
                "reference": None,
                "not_answered": "The paper names no reference assembly.",
                "areas": [
                    {
                        "area": "the point mutations of the results",
                        "views": ["s2"],
                        "reason": "The question asks for every point mutation.",
                    }
                ],
            },
            seen=seen,
        )

        outcome = ask(
            self.root,
            paper,
            "Which point mutations does the paper report?",
            model,
            config=self.config,
        )

        [repair] = repairs(seen[2])
        self.assertIn("unknown_span:main:p99", repair)
        [kept] = outcome.answer.findings
        self.assertEqual(kept.gene, "PBP2")
        self.assertEqual([v.hgvs for v in kept.variants], [("Y127X",)])
        self.assertEqual([p.text for p in kept.passages], ["PBP2 (Y127X)", S3_LINE])
        self.assertEqual(
            (outcome.requests.model_requests, outcome.requests.refused_answers), (3, 1)
        )

    def test_an_answer_call_with_empty_arguments_is_refused_by_its_required_fields(
        self,
    ):
        seen = []
        plan = {"reference": CANFAM_REFERENCE, "areas": [ASPRV1_AREA]}
        model = reading(
            [("propose_reading_areas", {})],
            [("propose_reading_areas", plan)],
            [("answer_reading_item", {})],
            [
                (
                    "answer_reading_item",
                    {"findings": [ASPRV1_FINDING]},
                )
            ],
            seen=seen,
        )

        outcome = self.ask(model)

        schema = [
            e["payload"]["errors"]
            for e in Journey.open(Path(outcome.journey_file)).read()
            if e["event_type"] == "AgentOutputValidated"
            and e["payload"]["status"] == "invalid"
        ]
        # One error form for every role's output, derived from its output type in the
        # output session: each error's location, then its type.
        self.assertEqual(
            schema,
            [
                ["reference:missing", "areas:missing"],
                ["findings:missing"],
            ],
        )
        [plan_repair] = repairs(seen[1])
        self.assertIn("reference", plan_repair)
        self.assertIn("areas", plan_repair)
        [answer_repair] = repairs(seen[3])
        self.assertIn("findings", answer_repair)
        self.assertEqual(len(outcome.answer.findings), 1)

    def test_a_finding_still_refused_when_repairs_run_out_is_recorded_as_refused(self):
        # The item completes on its last allowed call; the refused finding is kept
        # with its reason for a person, never as an accepted finding.
        nowhere = {
            "findings": [finding(value("gene", "ASPRV1", "main:p99"), reason="x")]
        }

        outcome = self.ask(one_area(*[nowhere] * 4))

        self.assertEqual((outcome.answer.findings, outcome.stopped), ((), ()))
        [refused] = outcome.answer.refused
        self.assertEqual(
            (refused.outcome, refused.reasons), ("refused", ("unknown_span:main:p99",))
        )
        events = Journey.open(Path(outcome.journey_file)).read()
        completed = [
            e["payload"] for e in events if e["event_type"] == "ReadingItemCompleted"
        ]
        self.assertEqual([c["item"]["kind"] for c in completed], ["paper", "area"])
        # Each call's refusal is its own fact, each answering the one before; the
        # last stands, and the item's completion names all four.
        judged = [e for e in events if e["event_type"] == "ReadingFindingJudged"]
        self.assertEqual(
            [(j["payload"]["outcome"], tuple(j["payload"]["reasons"])) for j in judged],
            [("refused", ("unknown_span:main:p99",))] * 4,
        )
        self.assertEqual(
            [j["payload"]["answers"] for j in judged],
            [None, *(j["event_id"] for j in judged[:-1])],
        )
        self.assertEqual(
            completed[1]["finding_event_ids"], [j["event_id"] for j in judged]
        )
        # The shared default of four failed turns (the paper reader keeps it):
        # three refusals asked again, then the fourth call, whose failed turn would
        # stop, records it.
        self.assertEqual(
            [
                (e["payload"]["status"], tuple(e["payload"]["errors"]))
                for e in events
                if e["event_type"] == "AgentOutputValidated"
            ][1:],
            [("invalid", ("unknown_span:main:p99@refused 1",))] * 3 + [("valid", ())],
        )

    def refused_accounted(self, outcome):
        self.assertEqual([f.reason for f in outcome.answer.findings], [WHY_ASPRV1])
        self.assertEqual(
            [(f.stated.reason, f.reasons) for f in outcome.answer.refused],
            [("Unfixed finding", ("unknown_span:main:p99",))],
        )
        view = answer_markdown(outcome.answer, outcome.stopped, self.paper)
        self.assertIn("## Refused (1)", view)
        self.assertIn("unknown_span:main:p99", view)

    def test_a_refused_finding_the_repair_leaves_out_stays_recorded_as_refused(self):
        # Spec r1 P1 (d2): omission infers no withdrawal.
        bad = finding(value("gene", "ASPRV1", "main:p99"), reason="Unfixed finding")

        outcome = self.ask(
            one_area({"findings": [ASPRV1_FINDING, bad]}, {"findings": []})
        )

        self.assertEqual(outcome.stopped, ())
        self.refused_accounted(outcome)

    def test_a_schema_stop_ends_the_item_at_its_review_and_the_answer_waits(self):
        # ADR 0039: a stopped reading item holds only itself; pending work stays
        # pending, so the reading records no answer until its Review settles it. What
        # code decided before the stop stays in the item's recorded validations, the
        # refused finding with its reason.
        bad = finding(value("gene", "ASPRV1", "main:p99"), reason="Unfixed finding")

        # Four failed turns (the paper reader keeps the shared default): the
        # fourth, a schema refusal, stops the item.
        outcome = self.ask(
            one_area(
                {"findings": [ASPRV1_FINDING, bad]}, *[{"findings": [bad]}] * 2, {}
            )
        )

        self.assertIsNone(outcome.answer)
        self.assertEqual(
            outcome.stopped, (f"area item {ASPRV1_AREA['area']!r}: failed_turns",)
        )
        events = Journey.open(Path(outcome.journey_file)).read()
        self.assertEqual(
            [
                e["payload"]["item"]["item_id"]
                for e in events
                if e["event_type"] == "ReadingItemCompleted"
            ],
            ["paper"],
        )
        self.assertNotIn("PaperAnswerRecorded", {e["event_type"] for e in events})
        [review] = [e for e in events if e["event_type"] == "AgentWorkReviewRequested"]
        self.assertEqual(review["payload"]["attempt_id"], outcome.items[1].attempt_id)
        refused = [
            e["payload"]["errors"]
            for e in events
            if e["event_type"] == "AgentOutputValidated"
            and e["payload"]["attempt_id"] == outcome.items[1].attempt_id
        ]
        self.assertEqual(refused[0], ["unknown_span:main:p99@refused 1"])

    def test_a_stopped_item_keeps_its_findings_and_its_continue_asks_only_the_rest(
        self,
    ):
        """Owner, #383 A1b ("we want the lane P functionality but updated and agentic
        in the M patterns"): a stopped item keeps every finding it recorded; its
        continue answer re-enters it with the kept ones shown as done, and the model is
        asked only for the one still refused. Nothing recorded is redone or dropped."""
        from bio_neb.agentic_journey.agent_work_review import (
            ContinueAgentWorkResponse,
            respond_agent_work_review,
        )
        from bio_neb.paper_reading.answer import stated_text
        from bio_neb.paper_reading.reading import advance_reading, reading_outcome

        bad = finding(value("gene", "ASPRV1", "main:p99"), reason="Unfixed finding")
        fixed = finding(
            value("gene", "ASPRV1", "main:p4"), reason="second try", corrects=1
        )
        seen = []
        # Four failed turns stop the area; the fifth reply is the re-entry's.
        model = one_area(
            {"findings": [ASPRV1_FINDING, bad]},
            *[{"findings": [bad]}] * 2,
            {},
            {"findings": [fixed]},
            seen=seen,
        )
        outcome = self.ask(model)
        journey = Journey.open(Path(outcome.journey_file))
        stopped = outcome.items[1].attempt_id

        def judged(attempt_id):
            return [
                e
                for e in journey.read()
                if e["event_type"] == "ReadingFindingJudged"
                and e["payload"]["attempt_id"] == attempt_id
            ]

        # The stop keeps what code judged before it: the accepted finding, and the
        # refused one asked again until the stop.
        before = judged(stopped)
        self.assertEqual(
            [e["payload"]["outcome"] for e in before],
            ["accepted", "refused", "refused", "refused"],
        )
        [review] = [
            e for e in journey.read() if e["event_type"] == "AgentWorkReviewRequested"
        ]
        respond_agent_work_review(
            journey, review["event_id"], ContinueAgentWorkResponse(action="continue")
        )
        (paper_event,) = [
            e for e in journey.read() if e["event_type"] == "PaperRecorded"
        ]
        runtime = RuntimeBinding(
            read_runtime_config(self.config).runtime_selections(),
            paper_event["event_id"],
        )

        def advance():
            return advance_reading(
                journey,
                outcome.reading_id,
                runtime=runtime,
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model,
            )

        self.assertTrue(advance())
        # The re-entered item's message shows the kept finding as done, and only the
        # refusal still standing, numbered for its correction.
        shown = json.loads(seen[-1][0].parts[0].content)["findings_recorded"]
        self.assertEqual(
            shown["kept"],
            [f"accepted: {stated_text(FindingPointer.model_validate(ASPRV1_FINDING))}"],
        )
        self.assertIn('Refused 1 (gene "ASPRV1" in main:p99):', shown["refused"])
        self.assertIn("Code keeps 1 finding of this item", shown["refused"])
        self.assertTrue(advance())
        after = reading_outcome(journey, outcome.reading_id)

        reentered = after.items[1].attempt_id
        # The re-entry recorded only the correction, answering the standing refusal;
        # the item's completion names every finding of both Attempts.
        [correction] = judged(reentered)
        self.assertEqual(
            (correction["payload"]["outcome"], correction["payload"]["answers"]),
            ("accepted", before[-1]["event_id"]),
        )
        [completed] = [
            e
            for e in journey.read()
            if e["event_type"] == "ReadingItemCompleted"
            and e["payload"]["attempt_id"] == reentered
        ]
        self.assertEqual(
            completed["payload"]["finding_event_ids"],
            [e["event_id"] for e in (*before, correction)],
        )
        self.assertEqual(
            [f.reason for f in after.answer.findings], [WHY_ASPRV1, "second try"]
        )
        self.assertEqual(after.answer.refused, ())
        self.assertEqual(
            after.requests.model_requests - outcome.requests.model_requests, 1
        )

    def test_a_correction_naming_its_refused_finding_clears_it(self):
        seen = []
        bad = finding(value("gene", "ASPRV1", "main:p99"), reason="first try")
        fixed = finding(
            value("gene", "ASPRV1", "main:p4"), reason="second try", corrects=1
        )
        model = one_area(
            {"findings": [ASPRV1_FINDING, bad]}, {"findings": [fixed]}, seen=seen
        )

        outcome = self.ask(model)

        [repair] = repairs(seen[2])
        self.assertIn('Refused 1 (gene "ASPRV1" in main:p99):', repair)
        self.assertIn("corrects", repair)
        self.assertEqual(
            [f.reason for f in outcome.answer.findings], [WHY_ASPRV1, "second try"]
        )
        self.assertEqual(outcome.answer.refused, ())

    def test_a_resend_restating_a_refused_findings_values_settles_it_without_corrects(
        self,
    ):
        # A recorded c2 run: the repair resent SV1 and SV2 without their refused change
        # and without corrects; the refusals stayed recorded beside the accepted findings.
        text, self.paper = c2_like()
        span = f"main:p{paragraph(text, C2_PARAGRAPH)}"
        sv1 = (
            value("contig", "chr 2", span),
            value("position", "86,770, 373", span, number=86770373),
        )
        sv2 = (
            value("contig", "chr 2", span),
            value("position", "86,870,271", span, number=86870271),
        )
        area = {"area": "SNPs at the Rp locus", "views": ["s2"], "reason": "Asked."}
        model = one_area(
            {
                "findings": [
                    finding(*sv1, value("change", "SV1", span), reason="SV1."),
                    finding(*sv2, value("change", "SV2", span), reason="SV2."),
                ]
            },
            {"findings": [finding(*sv1, reason="SV1."), finding(*sv2, reason="SV2.")]},
            plan={"reference": None, "areas": [area]},
        )

        outcome = self.ask(model)

        self.assertEqual(
            [v.position for f in outcome.answer.findings for v in f.variants],
            [86770373, 86870271],
        )
        self.assertEqual(outcome.answer.refused, ())

    def test_a_resend_correcting_a_refused_value_settles_its_refusal_without_corrects(
        self,
    ):
        # A recorded live c3 run: the contig cited at the row-label cell was refused;
        # the resends corrected it and restated the kept position, REF and ALT, without
        # corrects, and each round's refusals stayed recorded beside the accepted SNPs.
        self.paper = c3_like()

        def label_cell(snp: dict) -> dict:
            contig, *rest = snp["values"]
            return {**snp, "values": [{**contig, "column": "CHROM"}, *rest]}

        snps = [c3_snp(*C3_SNPS[0]), c3_snp(*C3_SNPS[1])]
        area = {"area": "Table 4's SNPs.", "views": ["t1"], "reason": "Asked."}
        model = one_area(
            {"findings": [label_cell(s) for s in snps]},
            {"findings": snps},
            plan={"reference": None, "areas": [area]},
        )

        outcome = self.ask(model)

        self.assertEqual(
            [v.position for f in outcome.answer.findings for v in f.variants],
            [22375729, 20640731],
        )
        self.assertEqual(outcome.answer.refused, ())

    def test_a_resend_correcting_every_value_s_column_settles_its_refusal(self):
        # A recorded live c3 run: each value cited at its row with the row-label column
        # (code kept none); the resend restated the same text at the same spans with the
        # variant's column, without corrects, and the refusals stayed recorded.
        self.paper = c3_like()

        def label_columns(snp: dict) -> dict:
            labels = ("CHROM", "Position", "Reference", "ALT")
            return {
                **snp,
                "values": [
                    {**v, "column": label}
                    for v, label in zip(snp["values"], labels, strict=True)
                ],
            }

        snps = [c3_snp(*C3_SNPS[0]), c3_snp(*C3_SNPS[3]), c3_snp(*C3_SNPS[4])]
        area = {"area": "Table 4's SNPs.", "views": ["t1"], "reason": "Asked."}
        model = one_area(
            {"findings": [label_columns(s) for s in snps]},
            {"findings": snps},
            plan={"reference": None, "areas": [area]},
        )

        outcome = self.ask(model)

        self.assertEqual(
            [v.position for f in outcome.answer.findings for v in f.variants],
            [22375729, 19577767, 28814241],
        )
        self.assertEqual(outcome.answer.refused, ())

    def test_a_repair_asks_only_for_the_refused_finding_and_keeps_its_sibling(self):
        # c2's SV2 was once lost for SV1's refusal; now SV2 stays kept.
        seen = []
        bad = finding(value("gene", "ASPRV1", "main:p99"), reason="first try")
        fixed = finding(
            value("gene", "ASPRV1", "main:p4"), reason="second try", corrects=1
        )
        model = one_area(
            {"findings": [ASPRV1_FINDING, bad]}, {"findings": [fixed]}, seen=seen
        )

        outcome = self.ask(model)

        [repair] = repairs(seen[2])
        self.assertIn(
            "Code keeps 1 finding of this item: do not send it again.", repair
        )
        self.assertIn('Refused 1 (gene "ASPRV1" in main:p99):', repair)
        self.assertEqual(
            [f.reason for f in outcome.answer.findings], [WHY_ASPRV1, "second try"]
        )
        self.assertEqual(outcome.answer.refused, ())

    def test_two_items_text_only_findings_of_one_gene_in_one_paragraph_stay_two(self):
        text = (
            "# Yeast\n## Results\n"
            "The PBP2 changes Y127X and D200N were found in different evolved strains.\n"
        )
        paper = Paper(None, [document("main", text)])
        area = {
            "area": "the PBP2 changes of the results",
            "views": ["s2"],
            "reason": "The question asks for every point mutation.",
        }

        def text_only(change):
            return {
                "findings": [
                    finding(
                        value("change", change, "main:p3"),
                        value("gene", "PBP2", "main:p3"),
                        reason=f"The paper reports PBP2 {change}.",
                    )
                ]
            }

        model = reading(
            [
                (
                    "propose_reading_areas",
                    {
                        "reference": None,
                        "not_answered": "The paper names no reference assembly.",
                        "areas": [area, {**area, "area": "the results again"}],
                    },
                )
            ],
            [("answer_reading_item", text_only("Y127X"))],
            [("answer_reading_item", text_only("D200N"))],
        )

        outcome = ask(
            self.root,
            paper,
            "Which point mutations does the paper report?",
            model,
            config=self.config,
        )

        # A shared gene and paragraph do not make one mutation: nothing joins them.
        self.assertEqual(
            [f.reason for f in outcome.answer.findings],
            ["The paper reports PBP2 Y127X.", "The paper reports PBP2 D200N."],
        )

    def test_findings_sharing_a_change_name_at_two_loci_stay_two(self):
        at_row, at_other = (
            kept_finding(j)
            for j in judge_all(
                [
                    FindingPointer.model_validate(ASPRV1_FINDING),
                    FindingPointer.model_validate(
                        finding(
                            *ASPRV1_FINDING["values"][1:3],
                            *s1_locus("t1:r3", "1", "1000100", "C", "T"),
                            reason="b",
                        )
                    ),
                ],
                self.paper,
            )
        )

        # One variant is one finding: the values make the identity, so two loci stay two.
        self.assertEqual(
            [
                [(v.contig, v.position) for v in f.variants]
                for f in merged_findings([at_row, at_other])
            ],
            [[("10", 68587027)], [("1", 1000100)]],
        )
        self.assertEqual(merged_findings([at_row, at_row]), (at_row,))

    def test_a_joined_variant_keeps_every_identifier_each_finding_checked(self):
        # Standards r1 S2: the later finding's rsID and change name join too.
        text = "## Results\nGENE1 chr1:1234567 A>G (rs123, c.10A>G) was found.\n"
        paper = Paper(None, [document("main", text)])
        locus = [
            value("contig", "chr1", "main:p2"),
            value("position", "1234567", "main:p2", number=1234567),
            value("alleles", "A>G", "main:p2"),
        ]
        first, second = (
            kept_finding(j)
            for j in judge_all(
                [
                    FindingPointer.model_validate(finding(*locus, reason="a")),
                    FindingPointer.model_validate(
                        finding(
                            *locus,
                            value("rsid", "rs123", "main:p2"),
                            value("change", "c.10A>G", "main:p2"),
                            reason="b",
                        )
                    ),
                ],
                paper,
            )
        )

        [joined] = merged_findings([first, second])

        [variant] = joined.variants
        self.assertEqual((variant.rsids, variant.hgvs), (("rs123",), ("c.10A>G",)))
        self.assertTrue(
            verify(paper_answer("Q?", paper, None, [joined], None), paper).verified
        )

    def test_every_variant_asked_for_comes_back_kept_with_no_cap(self):
        rows_ = S16_S1.splitlines()[2:]
        findings = []
        for n, row in enumerate(rows_):
            contig, position, ref, alt, _gene = row.split("\t")
            findings.append(
                finding(
                    *s1_locus(f"t1:r{n + 2}", contig, position, ref, alt),
                    reason=f"reported variant {n}",
                )
            )
        model = one_area({"findings": findings})

        outcome = self.ask(model, question="List every variant this paper reports.")

        self.assertEqual(len(rows_), 28)
        self.assertEqual(
            [f.variants[0].position for f in outcome.answer.findings],
            [int(row.split("\t")[1]) for row in rows_],
        )
        self.assertEqual(
            [f.reason for f in outcome.answer.findings],
            [f"reported variant {n}" for n in range(28)],
        )

    def test_counts_name_logical_requests_and_the_physical_contacts_beside_them(self):
        configuration = runtime_configuration()
        configuration["failure"] = {"model_contact": {"base_delay_seconds": 0}}
        self.config.write_text(json.dumps(configuration), encoding="utf-8")
        contacts = []
        replies = iter(
            [
                ("grep_paper", {"pattern": "ASPRV1"}),
                (
                    "propose_reading_areas",
                    {
                        "reference": CANFAM_REFERENCE,
                        "areas": [ASPRV1_AREA],
                    },
                ),
                ("answer_reading_item", {"findings": [ASPRV1_FINDING]}),
            ]
        )

        async def respond(messages, info):
            contacts.append(len(messages))
            if len(contacts) == 1:
                raise ModelHTTPError(503, "scripted/model", "unavailable")
            name, arguments = next(replies)
            return ModelResponse(
                parts=[ToolCallPart(name, arguments, tool_call_id=f"c{len(contacts)}")]
            )

        outcome = self.ask(ScriptedModelEndpoint(respond))

        self.assertEqual(len(contacts), 4)
        self.assertEqual(
            outcome.requests,
            RequestCounts(
                model_requests=3,
                physical_contacts=4,
                failed_contacts=1,
                tool_calls=("grep_paper",),
                refused_answers=0,
            ),
        )

    def run_command(self, model, out: Path, *options: str) -> int:
        """The developer entry over a retrieve folder: its downloaded files, which the
        command stores as a Run paper through the product's intake (s16 written small:
        the main text as text, S1's sheet as a tab-separated file)."""
        folder = self.root / "paper"
        (folder / "files").mkdir(parents=True)
        main_text = self.paper.documents["main"].text
        sheet = self.paper.documents[S16_S1_ID].text.split("\n", 1)[1]
        files = [
            ("article.txt", "supplied", main_text),
            ("pgen.1006651.s002.tsv", "supplement", sheet),
        ]
        for name, _role, text in files:
            (folder / "files" / name).write_text(text, encoding="utf-8")
        identity = self.paper.identity
        (folder / "download.json").write_text(
            json.dumps(
                {
                    "reference": f"doi:{DOI}",
                    "resolved": {
                        "doi": identity.doi,
                        "pmid": identity.pmid,
                        "pmcid": identity.pmcid,
                        "title": identity.title,
                    },
                    "documents": [
                        {
                            "name": name,
                            "url": f"https://example.org/{name}",
                            "role": role,
                            "media_type": "text/plain",
                        }
                        for name, role, _text in files
                    ],
                }
            ),
            encoding="utf-8",
        )
        argv = [
            "answer",
            "--paper",
            str(folder),
            "--question",
            "What is the most meaningful finding in this paper?",
            "--runtime-config",
            str(self.config),
            "--run-root",
            str(self.root / "runs"),
            "--out",
            str(out),
            *options,
        ]
        with (
            mock.patch(
                "bio_neb.paper_reading.reading.create_nebius_model_endpoint", model
            ),
            mock.patch.dict("os.environ", {"LOCAL_MODEL_KEY": "offline"}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return main(argv)

    def test_the_answer_command_writes_the_answer_its_model_and_its_counts(self):
        out = self.root / "out"
        model = one_area({"findings": [ASPRV1_FINDING]})

        self.assertEqual(self.run_command(model, out), 0)

        written = json.loads((out / "answer.json").read_text(encoding="utf-8"))
        self.assertEqual(written["paper"]["doi"], DOI)
        self.assertEqual(written["reference"]["candidate"]["name"], "CanFam 3.1")
        [finding] = written["findings"]
        self.assertEqual(finding["reason"], WHY_ASPRV1)
        self.assertEqual(
            [(v["contig"], v["position"]) for v in finding["variants"]],
            [("10", 68587027)],
        )
        requests = json.loads((out / "requests.json").read_text(encoding="utf-8"))
        # The configuration's default model; the recorded endpoint stands in for its wire.
        self.assertEqual(requests["model"], "Qwen/Qwen3.5-397B-A17B")
        self.assertEqual(
            (requests["model_requests"], requests["physical_contacts"]), (2, 2)
        )
        self.assertEqual(
            [(i["kind"], i["stopped"]) for i in requests["items"]],
            [("paper", None), ("area", None)],
        )
        self.assertTrue(Path(requests["journey_file"]).is_file())
        readable = (out / "answer.md").read_text(encoding="utf-8")
        for expected in (
            "CanFam 3.1",
            WHY_ASPRV1,
            "position **68587027** · column POS in `t1:r2`",
            "> We found a de novo missense variant in ==ASPRV1==, ==c.1052T>C==, "
            "==p.(Leu351Pro)==, in the affected dog.",
        ):
            self.assertIn(expected, readable)

    def test_the_answer_command_counts_refused_calls_and_refused_findings_apart(self):
        # Live run 1 L6: refused_answers counts invalid output calls; the answer's
        # findings are counted by their outcome, from the recorded answer.
        out = self.root / "out"
        bad = finding(value("gene", "ASPRV1", "main:p99"), reason="Unfixed finding")
        model = one_area({"findings": [ASPRV1_FINDING, bad]}, {"findings": []})

        self.assertEqual(self.run_command(model, out), 0)

        requests = json.loads((out / "requests.json").read_text(encoding="utf-8"))
        self.assertEqual(requests["refused_answers"], 1)
        self.assertEqual(
            requests["findings"], {"accepted": 1, "unresolved": 0, "refused": 1}
        )

    def test_a_captured_answer_tags_every_span_with_its_trace_attributes(self):
        out = self.root / "out"
        spans = self.root / "spans.jsonl"
        model = reading(
            [
                (
                    "propose_reading_areas",
                    {
                        "reference": CANFAM_REFERENCE,
                        "areas": [ASPRV1_AREA],
                    },
                )
            ],
            [("grep_paper", {"pattern": "ASPRV1"})],
            [("answer_reading_item", {"findings": [ASPRV1_FINDING]})],
        )
        lane_p = {"lane": "P", "ticket": "349", "scenario": "s16", "run": "1"}

        status = self.run_command(
            model,
            out,
            "--capture",
            str(spans),
            *(f"--trace-attribute={k}={v}" for k, v in lane_p.items()),
        )

        self.assertEqual(status, 0)
        rows = [json.loads(line) for line in spans.read_text().splitlines()]
        self.assertTrue(rows)
        for row in rows:
            self.assertLessEqual(lane_p.items(), row["attributes"].items(), row["name"])
        # The metadata joins main's own correlation, never replaces it.
        chats = [r for r in rows if r["name"].startswith("chat ")]
        tools = [r for r in rows if r["name"].startswith("execute_tool ")]
        self.assertEqual(len(chats), 3)
        self.assertEqual([r["name"] for r in tools], ["execute_tool grep_paper"])
        for row in chats + tools:
            self.assertEqual(row["attributes"]["agent_role"], "paper_reader")
        requests = json.loads((out / "requests.json").read_text(encoding="utf-8"))
        self.assertEqual(
            requests["capture"],
            {"path": str(spans), "cloud": False, "trace": lane_p, "errors": []},
        )

    def test_a_capture_that_cannot_start_leaves_the_answer_and_says_why(self):
        out = self.root / "out"
        spans = self.root / "spans.jsonl"
        model = one_area({"findings": [ASPRV1_FINDING]})

        # Cloud export without a write token: capture is off, the reading is not.
        with mock.patch.dict("os.environ"):
            os.environ.pop("LOGFIRE_TOKEN", None)
            status = self.run_command(
                model,
                out,
                "--capture",
                str(spans),
                "--cloud",
                "--trace-attribute=lane=P",
            )

        self.assertEqual(status, 0)
        self.assertTrue((out / "answer.json").is_file())
        requests = json.loads((out / "requests.json").read_text(encoding="utf-8"))
        self.assertEqual(
            requests["capture"],
            {
                "path": str(spans),
                "cloud": True,
                "trace": {"lane": "P"},
                "errors": ["ValueError"],
            },
        )

    def test_a_gene_named_only_as_part_of_a_longer_name_is_refused_and_repaired(self):
        model = one_area(
            {
                "findings": [
                    finding(
                        value("gene", "ASPR", "main:p4"),
                        *ASPRV1_FINDING["values"][1:],
                        reason=WHY_ASPRV1,
                    )
                ]
            },
            {"findings": [{**ASPRV1_FINDING, "corrects": 1}]},
        )

        outcome = self.ask(model)

        refused = [
            e["payload"]["errors"]
            for e in Journey.open(Path(outcome.journey_file)).read()
            if e["event_type"] == "AgentOutputValidated"
            and e["payload"]["status"] == "invalid"
        ]
        self.assertEqual(
            refused, [["written_not_in_span:gene=ASPR in main:p4@refused 1"]]
        )
        self.assertEqual(outcome.answer.findings[0].gene, "ASPRV1")


# A paper shaped like s17 (PMC7544121), written small: the main text with Table 1 inside its
# Results, and S6 as the reader writes a workbook (sheets chr4 and chr6, every line a row).
TITLE = "# Hypocalcemia in Thoroughbred foals\n"
RESULTS_LINE = (
    "The only associated 'HIGH' effect nonsense variant was on chr4 (EquCab3.0 "
    "chr4:54108297) in exon 26 of RAPGEF5 (c.2624C>A p.Ser875*) and was exclusive to TBs."
)
RESULTS = (
    f"## Results\n{RESULTS_LINE}\nTable 1\nVariants associated with hypocalcemia.\n"
)
TABLE_1 = "CHROM\tPOS (EquCab3.0)\tREF\tALT\n4\t54108297\tG\tT\n"
METHODS = "## Methods\nReads were mapped to EquCab3.0."
MAIN = TITLE + RESULTS + TABLE_1 + METHODS
CHR4 = (
    "#CHROM\tCHROM\tPOS\tREF\tALT\tINFO\n"
    "NC_009147.3\t4\t54108297\tG\tT\tANN=T|stop_gained|HIGH|RAPGEF5|c.2624C>A|p.Ser875*\n"
    "\n"
    "NC_009147.3\t4\t54108300\tA\tC\tANN=C|intron_variant|MODIFIER|RAPGEF5"
)
CHR6 = (
    "S2 Table. Variants on chr6.\n"
    "#CHROM\tCHROM\tPOS\tREF\tALT\tINFO\n"
    "NC_009149.3\t6\t15346206\tT\tG\t."
)
S6 = f"## chr4 (4 rows)\n{CHR4}\n## chr6 (3 rows)\n{CHR6}"
S6_ID = "supp:pgen.1009028.s006.xlsx"


def s17_like() -> Paper:
    return Paper(
        None,
        [
            PaperDocument(
                document_id="main",
                kind="main_text",
                url="https://pmc-oa-opendata.s3.amazonaws.com/PMC7544121.1/PMC7544121.1.xml",
                text=MAIN,
            ),
            PaperDocument(
                document_id=S6_ID,
                kind="supplementary",
                url="https://pmc-oa-opendata.s3.amazonaws.com/PMC7544121.1/pgen.1009028.s006.xlsx",
                title="pgen.1009028.s006.xlsx",
                text=S6,
            ),
        ],
    )


class PaperViewTests(unittest.TestCase):
    def test_each_document_is_a_root_with_its_text_sections_and_tables_apart(self):
        main, s6 = outline(s17_like()).documents

        self.assertEqual(
            (main.document_id, main.kind, main.characters),
            ("main", "main_text", len(MAIN)),
        )
        # Table 1's rows leave the Results text: the section counts its text only.
        results_end = len(TITLE) + len(RESULTS) + len(TABLE_1)
        self.assertEqual(
            [
                (s.section_id, s.title, s.start, s.end, s.characters)
                for s in main.sections
            ],
            [
                ("s1", "Hypocalcemia in Thoroughbred foals", 0, len(TITLE), len(TITLE)),
                ("s2", "Results", len(TITLE), results_end, len(RESULTS)),
                ("s3", "Methods", results_end, len(MAIN), len(METHODS)),
            ],
        )
        [table_1] = main.tables
        table_start = len(TITLE) + len(RESULTS)
        self.assertEqual(
            table_1.model_dump(),
            {
                "table_id": "t1",
                "location": "Table 1",
                "section": "Results",
                "start": table_start,
                "end": table_start + len(TABLE_1) - 1,
                "header_row": 1,
                "columns": ("CHROM", "POS (EquCab3.0)", "REF", "ALT"),
                "rows": 1,
            },
        )

        # A workbook is tables only: each sheet keeps its name, and line r is row r.
        self.assertEqual(s6.sections, ())
        chr4_start = len("## chr4 (4 rows)\n")
        chr6_start = len(f"## chr4 (4 rows)\n{CHR4}\n## chr6 (3 rows)\n")
        self.assertEqual(
            [t.model_dump() for t in s6.tables],
            [
                {
                    "table_id": "t2",
                    "location": "chr4",
                    "section": "chr4",
                    "start": chr4_start,
                    "end": chr4_start + len(CHR4),
                    "header_row": 1,
                    "columns": ("#CHROM", "CHROM", "POS", "REF", "ALT", "INFO"),
                    "rows": 2,  # row 3 is empty
                },
                {
                    "table_id": "t3",
                    "location": "chr6",
                    "section": "chr6",
                    "start": chr6_start,
                    "end": chr6_start + len(CHR6),
                    "header_row": 2,  # row 1 is the sheet's title
                    "columns": ("#CHROM", "CHROM", "POS", "REF", "ALT", "INFO"),
                    "rows": 1,
                },
            ],
        )


# s3 (PMC2829521), its Results line as retrieved: no pattern reads the three protein changes.
S3_LINE = (
    "Applying these heuristics, we detected four single-point differences between the "
    "laboratory evolved strain and that of its parent: the previously-described point "
    "mutations at PBP2 (Y127X), SGF73 (E294X), and UPF3 (G6W) [4], and an additional point "
    "mutation intergenic to RRN3 and YPK1 (chr11: 207,469:A>C) which we confirmed by Sanger "
    "sequencing (Figure 1)."
)


class GrepTests(unittest.TestCase):
    def test_every_match_is_counted_per_document_and_each_hit_says_where_it_is(self):
        paper = s17_like()

        result = grep(paper, "54108297")

        self.assertEqual(result.matches, {"main": 2, S6_ID: 1})
        self.assertEqual(result.table_matches, {"t1": 1, "t2": 1})
        prose, table_row, sheet_row = result.hits
        self.assertEqual(
            (prose.document_id, prose.offset, prose.line, prose.section, prose.table),
            ("main", MAIN.index("54108297"), RESULTS_LINE, "Results", None),
        )
        self.assertEqual(
            (table_row.offset, table_row.line, table_row.section, table_row.table),
            (
                MAIN.index("\t54108297") + 1,
                "4\t54108297\tG\tT",
                None,
                TableCell(
                    table_id="t1", location="Table 1", row=2, column="POS (EquCab3.0)"
                ),
            ),
        )
        self.assertEqual(
            (sheet_row.document_id, sheet_row.offset, sheet_row.table),
            (
                S6_ID,
                S6.index("54108297"),
                TableCell(table_id="t2", location="chr4", row=2, column="POS"),
            ),
        )

    def test_a_hit_cap_hides_no_table_and_every_document_is_counted(self):
        paper = s17_like()

        capped = grep(paper, "RAPGEF5", max_hits=1)
        absent = grep(paper, "rs1800562|rs4244285")

        [hit] = capped.hits
        self.assertEqual((hit.document_id, hit.section), ("main", "Results"))
        self.assertEqual(capped.matches, {"main": 1, S6_ID: 2})
        self.assertEqual(capped.table_matches, {"t2": 2})
        # s15: neither rsID is in any document, and the result says so for each one.
        self.assertEqual(absent.matches, {"main": 0, S6_ID: 0})
        self.assertEqual((absent.table_matches, absent.hits), ({}, ()))

    def test_one_sentence_gives_a_hit_at_each_change_it_names(self):
        text = f"# Yeast\n## Results\n{S3_LINE}"
        paper = Paper(None, [document("main", text)])

        result = grep(paper, "Y127X|E294X|G6W")

        self.assertEqual(result.matches, {"main": 3})
        # Each hit keeps 200 characters of the line on each side of its match.
        expected = []
        for change in ("Y127X", "E294X", "G6W"):
            at = S3_LINE.index(change)
            line = S3_LINE[max(0, at - 200) : at + len(change) + 200]
            expected.append((text.index(change), line, "Results"))
        self.assertEqual([(h.offset, h.line, h.section) for h in result.hits], expected)

    def test_a_long_line_is_cut_to_the_text_around_the_match(self):
        line = "a" * 300 + "54108297" + "b" * 300
        paper = Paper(None, [document("main", line)])

        [hit] = grep(paper, "54108297").hits

        self.assertEqual(hit.line, "a" * 200 + "54108297" + "b" * 200)

    def test_a_pattern_that_is_no_regular_expression_is_searched_as_written(self):
        result = grep(s17_like(), "p.Ser875*)")

        self.assertTrue(result.literal)
        self.assertEqual(result.matches, {"main": 1, S6_ID: 0})


def paragraph(text: str, line: str) -> int:
    """The 1-based number of the line holding ``line`` in ``text``."""

    return text[: text.index(line)].count("\n") + 1


class SpanTests(unittest.TestCase):
    """Code issues an id for each paragraph and table row a reading shows; a finding cites
    those ids, never copied text (#404 d7: a paragraph, or a table row with cells by column)."""

    def test_each_hit_and_row_names_its_paragraph_or_row_by_a_span_id(self):
        paper = s17_like()

        prose, table_row, sheet_row = grep(paper, "54108297").hits
        [row] = rows(paper, "t2", where={"POS": "54108297"}).rows

        self.assertEqual(prose.span, f"main:p{paragraph(MAIN, RESULTS_LINE)}")
        self.assertEqual(
            (table_row.span, sheet_row.span, row.span), ("t1:r2", "t2:r2", "t2:r2")
        )

    def test_a_span_id_gives_back_its_whole_paragraph_or_row_and_a_row_its_cells(self):
        paper = s17_like()
        number = paragraph(MAIN, RESULTS_LINE)

        text = paper.unit(f"main:p{number}")
        table_row = paper.unit("t1:r2")
        cell = table_row.cell("POS (EquCab3.0)")

        start = MAIN.index(RESULTS_LINE)
        self.assertEqual(
            (text.line.document_id, text.line.start, text.line.text, text.columns),
            ("main", start, RESULTS_LINE, None),
        )
        self.assertEqual(table_row.line.text, "4\t54108297\tG\tT")
        self.assertEqual(table_row.columns, ("CHROM", "POS (EquCab3.0)", "REF", "ALT"))
        at = MAIN.index("\t54108297\t") + 1
        self.assertEqual((cell.start, cell.end, cell.text), (at, at + 8, "54108297"))
        self.assertIsNone(table_row.cell("GENE"))

    def test_an_id_code_never_issued_names_no_span(self):
        paper = s17_like()
        blank = paragraph(S6, "\n\nNC_009147.3\t4\t54108300") + 1

        for span in (
            "main:p999",
            "t9:r1",
            "t1:r9",
            "main",
            "p3",
            "",
            f"{S6_ID}:p{blank}",
        ):
            with self.subTest(span=span):
                self.assertIsNone(paper.unit(span))

    def test_the_same_paper_gives_the_same_ids_on_every_read(self):
        first, again = s17_like(), s17_like()

        self.assertEqual(
            [h.span for h in grep(first, "54108297|RAPGEF5").hits],
            [h.span for h in grep(again, "54108297|RAPGEF5").hits],
        )


class ShownTextTests(unittest.TestCase):
    # A quote copied with a row's raw tabs is no JSON string, and the endpoint
    # dropped the call whole; the model is shown no control character but the line break.
    # Lane P's own copy retired: every reading's result is
    # ``content.shown_to_model``'s.
    def test_each_control_character_but_the_line_break_is_one_space(self):
        self.assertEqual(shown_to_model("a\tb\r\nc\x07"), "a b \nc ")

    def test_every_string_of_a_result_is_shown_names_included(self):
        result = {"cells\t": ["x\ty", {"Gene\x07": "1\t2"}], "rows": 3, "none": None}

        self.assertEqual(
            shown_to_model(result),
            {"cells ": ["x y", {"Gene ": "1 2"}], "rows": 3, "none": None},
        )


class ShownIdentityTests(unittest.TestCase):
    # Review r8 (Standards S9): a document id is shown as it is used, and stays distinct.
    def test_a_document_id_is_read_by_the_id_the_outline_shows(self):
        paper = Paper(
            None,
            [
                document("supp:table\x07a.tsv", "## Results\nGENE1 c.100A>G\n"),
                document("supp:table a.tsv", "## Results\nGENE2 c.200C>T\n"),
                document("supp:table\ta.tsv", "## Results\nGENE3 c.300G>A\n"),
            ],
        )

        shown_ids = [
            d["document_id"]
            for d in shown_to_model(outline(paper).model_dump(mode="json"))["documents"]
        ]

        self.assertEqual(
            shown_ids,
            ["supp:table a.tsv#2", "supp:table a.tsv", "supp:table a.tsv#3"],
        )
        read_back = [
            [(p.span, p.text) for p in read(paper, document_id).passages]
            for document_id in shown_ids
        ]
        self.assertEqual(
            read_back,
            [
                [(f"{d}:p1", "## Results"), (f"{d}:p2", f"GENE{n} c.{n}00{change}")]
                for n, (d, change) in enumerate(
                    zip(shown_ids, ("A>G", "C>T", "G>A"), strict=True), start=1
                )
            ],
        )
        # Each shown id's span names its own document's line.
        self.assertEqual(
            [paper.unit(f"{d}:p2").line.text for d in shown_ids],
            ["GENE1 c.100A>G", "GENE2 c.200C>T", "GENE3 c.300G>A"],
        )


class ReadTests(unittest.TestCase):
    def test_a_section_is_read_from_its_heading_and_names_the_tables_inside_it(self):
        paper = s17_like()
        start = len(TITLE)

        whole = read(paper, "s2")
        rest = read(paper, "s2", offset=start + 11, length=20)

        self.assertEqual(
            (whole.document_id, whole.section_id, whole.start, whole.end),
            ("main", "s2", start, start + len(RESULTS) + len(TABLE_1)),
        )
        # Each line comes back under the id a finding cites: a paragraph, or a table row.
        self.assertEqual(
            [(p.span, p.text) for p in whole.passages],
            [
                ("main:p2", "## Results"),
                ("main:p3", RESULTS_LINE),
                ("main:p4", "Table 1"),
                ("main:p5", "Variants associated with hypocalcemia."),
                ("t1:r1", "CHROM\tPOS (EquCab3.0)\tREF\tALT"),
                ("t1:r2", "4\t54108297\tG\tT"),
            ],
        )
        self.assertEqual(whole.tables, ("t1",))
        # A read that starts or ends inside a line shows that part, under the line's id.
        self.assertEqual(
            (rest.start, [(p.span, p.text) for p in rest.passages]),
            (start + 11, [("main:p3", MAIN[start + 11 : start + 31])]),
        )
        self.assertEqual(rest.tables, ())

    def test_a_document_is_read_at_an_offset_and_never_more_than_50000_characters(self):
        # The read cap is 50,000 characters; the model chooses its read length below it
        # (3,000 by default).
        text = "".join(f"line {n:05d}\n" for n in range(10000))
        paper = Paper(None, [document("main", text)])

        first = read(paper, "main")
        later = read(paper, "main", offset=5000, length=9000)
        capped = read(paper, "main", offset=1000, length=60000)

        shown_text = lambda result: "\n".join(p.text for p in result.passages)  # noqa: E731
        self.assertEqual((first.start, first.end), (0, 3000))
        self.assertEqual(shown_text(first), text[:3000].rstrip("\n"))
        self.assertEqual(first.passages[1].span, "main:p2")
        self.assertEqual((later.start, later.end), (5000, 14000))
        self.assertEqual(
            (shown_text(later), later.characters),
            (text[5000:14000].rstrip("\n"), len(text)),
        )
        self.assertEqual((capped.start, capped.end), (1000, 51000))

    def test_the_read_tool_takes_a_length_up_to_the_cap_and_refuses_one_past_it(
        self,
    ):
        # The read_paper row's length argument stops at MAX_READ.
        import typing

        from pydantic import TypeAdapter

        from bio_neb.paper_reading.reader import MAX_READ
        from bio_neb.paper_reading.tools import read_paper

        length = TypeAdapter(
            typing.get_type_hints(read_paper, include_extras=True)["length"]
        )
        self.assertEqual(MAX_READ, 50_000)
        self.assertEqual(length.validate_python(50_000), 50_000)
        with self.assertRaises(ValidationError):
            length.validate_python(50_001)
        default = inspect.signature(read_paper).parameters["length"].default
        self.assertEqual(default, 3000)

    def test_a_span_id_is_read_inside_its_paragraph_or_row_within_the_read_limit(self):
        # Live run 1 L4: the model read paragraphs by their span ids; S7 keeps the
        # 4,000-character limit, so a long paragraph is read in pages.
        paper = s17_like()
        long_text = "x" * 3000 + " chr4:54108297 " + "y" * 3000
        long = Paper(None, [document("main", f"# T\n{long_text}\nnext line")])

        short = read(paper, f"main:p{paragraph(MAIN, RESULTS_LINE)}")
        row = read(paper, "t1:r2")
        first = read(long, "main:p2")
        later = read(long, "main:p2", offset=first.end, length=4000)

        self.assertEqual(
            [(p.span, p.text) for p in short.passages],
            [(f"main:p{paragraph(MAIN, RESULTS_LINE)}", RESULTS_LINE)],
        )
        self.assertEqual(
            [(p.span, p.text) for p in row.passages], [("t1:r2", "4\t54108297\tG\tT")]
        )
        self.assertEqual((first.start, first.end - first.start), (4, 3000))
        self.assertEqual(
            (later.start, later.end, later.passages[0].text),
            (first.end, 4 + len(long_text), long_text[3000:]),
        )
        # A read never runs past its span into the next line.
        self.assertNotIn("next line", "".join(p.text for p in later.passages))
        for unknown in ("main:p99", "t9:r1"):
            with self.subTest(unknown=unknown):
                self.assertEqual(read(paper, unknown).code, "unknown_view")

    def test_a_span_id_is_admitted_as_a_read_target(self):
        deps = SimpleNamespace(paper=s17_like(), reads=set())

        def errors(at: str) -> tuple[str, ...]:
            action = SimpleNamespace(tool_name="read_paper", arguments={"at": at})
            return reading_admission_errors(deps, action)

        self.assertEqual(errors("t1:r2"), ())
        self.assertEqual(errors(f"main:p{paragraph(MAIN, RESULTS_LINE)}"), ())
        self.assertEqual(errors("main:p99"), ("unknown_view",))

    def test_a_table_id_is_read_as_its_rows(self):
        # A recorded c2 run: the area item called read_paper on table t7 twice and was
        # refused each time; a table id now reads its rows by span.
        paper = s17_like()
        deps = SimpleNamespace(paper=paper, reads=set())
        action = SimpleNamespace(tool_name="read_paper", arguments={"at": "t2"})
        document_id, table = paper.tables["t2"]
        lines = paper.documents[document_id].text[table.start : table.end].splitlines()

        result = read(paper, "t2")

        self.assertEqual(reading_admission_errors(deps, action), ())
        self.assertEqual(result.tables, ("t2",))
        self.assertEqual(
            [p.text for p in result.passages if p.span.startswith("t2:r")], lines
        )
        self.assertIn(
            ("t2:r2", CHR4.splitlines()[1]), [(p.span, p.text) for p in result.passages]
        )

    def test_a_long_unknown_name_is_still_refused_as_a_result(self):
        # Review r1 (Standards S1): a 600-character name broke the 500-character message.
        refused = read(s17_like(), "x" * 600)

        self.assertEqual(refused.code, "unknown_view")
        self.assertLessEqual(len(refused.message), 500)

    def test_a_name_that_is_no_section_or_document_is_refused(self):
        refused = read(s17_like(), "s99")

        self.assertEqual(refused.code, "unknown_view")
        self.assertIn("'s99'", refused.message)


class RowsTests(unittest.TestCase):
    def test_a_table_is_filtered_by_named_columns_and_each_row_points_at_its_line(self):
        result = rows(s17_like(), "t2", where={"CHROM": "4", "POS": "54108297"})

        row_2 = CHR4.splitlines()[1]
        self.assertEqual((result.tables, result.matched), (("t2",), 1))
        self.assertEqual(
            result.rows,
            (
                TableRow(
                    table_id="t2",
                    location="chr4",
                    row=2,
                    span="t2:r2",
                    document_id=S6_ID,
                    offset=S6.index(row_2),
                    cells=dict(
                        zip(
                            ("#CHROM", "CHROM", "POS", "REF", "ALT", "INFO"),
                            row_2.split("\t"),
                            strict=True,
                        )
                    ),
                ),
            ),
        )

    def test_without_a_table_every_table_naming_the_columns_is_searched(self):
        # S6's sheets share one header, so a filter need not know the sheet (Table 1 names
        # no POS column): chr6's row sits under its title row and header.
        result = rows(
            s17_like(),
            where={"CHROM": "6", "POS": "15,346,206"},
            columns=["REF", "ALT"],
        )

        self.assertEqual((result.tables, result.matched), (("t2", "t3"), 1))
        [row] = result.rows
        self.assertEqual(
            (row.table_id, row.location, row.row, row.cells),
            ("t3", "chr6", 3, {"REF": "T", "ALT": "G"}),
        )

    def test_rows_past_the_limit_are_counted_not_returned(self):
        result = rows(s17_like(), "t2", max_rows=1)

        self.assertEqual(result.matched, 2)  # row 3 is empty
        self.assertEqual([r.row for r in result.rows], [2])

    def test_each_column_keeps_its_own_name_and_value(self):
        # Review r1 (both): a blank header cell named by its letter met a real header "B",
        # and a row wider than its header met a header "C"; each lost a value.
        def table(text):
            return Paper(
                None,
                [
                    PaperDocument(
                        document_id="supp:S1.xlsx",
                        kind="supplementary",
                        url="https://example.org/S1.xlsx",
                        text=text,
                    )
                ],
            )

        blank = table("## Sheet1 (2 rows)\nB\t\tB (B)\ngene\tallele\t123456")
        wide = table("## Sheet1 (2 rows)\nC\tPOS\ngene\t123456\tallele")

        [blank_table] = outline(blank).documents[0].tables
        self.assertEqual(blank_table.columns, ("B", "B (B)", "B (B) (C)"))
        self.assertEqual(
            rows(blank, "t1", where={"B": "gene"}).rows[0].cells,
            {"B": "gene", "B (B)": "allele", "B (B) (C)": "123456"},
        )
        [wide_table] = outline(wide).documents[0].tables
        self.assertEqual(wide_table.columns, ("C", "POS", "C (C)"))
        self.assertEqual(
            rows(wide, "t1").rows[0].cells,
            {"C": "gene", "POS": "123456", "C (C)": "allele"},
        )
        [hit] = grep(wide, "allele").hits
        self.assertEqual(hit.table.column, "C (C)")

    def test_an_empty_sheet_is_a_table_with_no_rows(self):
        text = "## Notes (0 rows)\n## chr4 (2 rows)\nCHROM\tPOS\n4\t54108297"
        paper = Paper(
            None,
            [
                PaperDocument(
                    document_id="supp:S6.xlsx",
                    kind="supplementary",
                    url="https://example.org/S6.xlsx",
                    text=text,
                )
            ],
        )

        [notes, chr4] = outline(paper).documents[0].tables
        result = rows(paper, where={"CHROM": "4"})
        empty = rows(paper, "t1")

        self.assertEqual(
            (notes.location, notes.start, notes.end, notes.rows),
            ("Notes", len("## Notes (0 rows)"), len("## Notes (0 rows)"), 0),
        )
        self.assertEqual((result.tables, result.matched), (("t2",), 1))
        self.assertEqual((empty.matched, empty.rows), (0, ()))

    def test_a_long_unknown_table_or_column_is_still_refused_as_a_result(self):
        # Review r1 (Standards S1): the names went into the message whole.
        paper = s17_like()

        refusals = [
            rows(paper, "t" * 600),
            rows(paper, where={"Y" * 600: "1"}),
            rows(paper, "t2", where={"Y" * 600: "1"}),
        ]

        self.assertEqual(
            [r.code for r in refusals],
            ["unknown_view", "unknown_column", "unknown_column"],
        )
        self.assertTrue(all(len(r.message) <= 500 for r in refusals))

    def test_a_table_or_column_that_is_not_there_is_refused_with_what_is(self):
        paper = s17_like()

        no_table = rows(paper, "t9", where={"POS": "1"})
        no_column = rows(paper, "t2", where={"Position": "1"})
        nowhere = rows(paper, where={"Position": "1"})

        self.assertEqual(no_table.code, "unknown_view")
        self.assertIn("'t9'", no_table.message)
        self.assertEqual(no_column.code, "unknown_column")
        self.assertIn("Position", no_column.message)
        self.assertIn("'#CHROM', 'CHROM', 'POS'", no_column.message)
        self.assertEqual(nowhere.code, "unknown_column")
        self.assertIn("Position", nowhere.message)


def s3_like() -> tuple[str, Paper]:
    text = f"# Yeast\n## Results\n{S3_LINE}"
    return text, Paper(None, [document("main", text)])


S3_FINDINGS = (("PBP2", "Y127X"), ("SGF73", "E294X"), ("UPF3", "G6W"))


# c2, Araucana chicken (PMC6903725), its Results paragraph as a recorded run retrieved
# it: the stray space in "86,770, 373" is in the publisher's XML.
C2_PARAGRAPH = (
    "Confirming our earlier finding, the two candidate causative SNPs for rumplessness (Rp) "
    "were identified proximal to IRX1 and IRX2 [14]. Both of these SNPs (chr 2: 86,770, "
    "373, SV1 and 86,870,271, SV2) were present in the filtered SNP dataset and held true "
    "for the Rp mutation. The single tailed NAA and all KA, KD, and WL were wildtype A/A "
    "at the Rp loci, as expected. The five rumpless NAA were A/C or C/C at both Rp loci."
)
# c1, CASQ2 case report (PMC12666932): Table 1 is transposed, one column per variant
# (main@5013-5734, two of its rows).
C1_TABLE = (
    "Table 1\nGene\tCASQ2\tANK2\n"
    "Chromosome: position (Hg38)\t1:115701412 C>G\t4:113355007 G>A\n"
    "Reference SNP number (NCBI)\trs2101052355\trs769038996\n"
)


# h1's KCNH2 row (a retained answer).
H1_KCNH2_ROW = (
    "KCNH2\tchr7:150951552/NM_000238.4\tc.1841C>T p. (Ala614Val)\tMissense variant"
    "\tD/PD\trs199472944\tPathogenic"
)


# c3, Mediterranean rice (PMC12624038), Table 4 as a recorded run retrieved it: each SNP
# a column, its fields as rows (the duplicate header "11" is the reader's "11 (F)").
C3_TABLE = (
    "Table 4\n"
    "CHROM\t4\t6\t10\t11\t11\n"
    "Position\t22,375,729\t20,640,731\t6,918,411\t19,577,767\t28,814,241\n"
    "Reference\tG\tC\tC\tC\tG\n"
    "ALT\tT\tT\tT\tT\tA\n"
    "Albatros\t0/0\t0/0\t0/0\t1/1\t0/1\n"
)
C3_SNPS = (
    ("4", "4", "22,375,729", "G", "T"),
    ("6", "6", "20,640,731", "C", "T"),
    ("10", "10", "6,918,411", "C", "T"),
    ("11", "11", "19,577,767", "C", "T"),
    ("11 (F)", "11", "28,814,241", "G", "A"),
)


def c3_like() -> Paper:
    return Paper(
        PaperIdentity(doi="10.1038/s41598-025-23999-5", resolved_by="user_supplied"),
        [document("main", f"# Rice\n## Results\n{C3_TABLE}")],
    )


def c3_snp(column: str, contig: str, position: str, ref: str, alt: str) -> dict:
    """One c3 SNP as the model's third call stated it: each value at its row, by column."""

    return finding(
        value("contig", contig, "t1:r1", column=column),
        value(
            "position",
            position,
            "t1:r2",
            column=column,
            number=int(position.replace(",", "")),
        ),
        value("ref", ref, "t1:r3", column=column),
        value("alt", alt, "t1:r4", column=column),
        reason=f"Table 4 selects the SNP on chromosome {contig}.",
    )


def c2_like() -> tuple[str, Paper]:
    text = f"# Araucana\n## Results\n{C2_PARAGRAPH}"
    return text, Paper(None, [document("main", text)])


def c1_like() -> tuple[str, Paper]:
    text = f"# CASQ2\n## Results\n{C1_TABLE}"
    return text, Paper(None, [document("main", text)])


class FindingEvidenceTests(unittest.TestCase):
    """The model states each value as written and the span id that writes it; code finds
    the written text there, reads it into its slot and decides each finding on its own
    (#404 d1-d3, d6, d7). Nothing the model sends is a copy of a paragraph."""

    def test_no_output_field_carries_a_paragraph(self):
        # A finding states span ids and short written values only.
        schema = json.dumps(ItemAnswer.model_json_schema())

        self.assertNotIn('"quote"', schema)
        self.assertEqual(
            set(FindingPointer.model_fields), {"values", "spans", "reason", "corrects"}
        )
        self.assertEqual(
            ValuePointer.model_fields["written"].metadata[-1].max_length, 200
        )

    def test_values_cited_by_span_are_kept_where_the_paper_writes_them(self):
        text, paper = s3_like()
        span = f"main:p{paragraph(text, S3_LINE)}"
        stated = finding(
            value("gene", "PBP2", span),
            value("change", "Y127X", span),
            reason="The paper reports Y127X in PBP2.",
        )

        [kept] = kept_answer(
            "Which point mutations?", reading_of(stated), paper
        ).findings

        line = paper.unit(span).line
        self.assertEqual(
            [(v.slot, v.written, v.span, v.source.start, v.form) for v in kept.values],
            [
                ("gene", "PBP2", span, text.index("PBP2"), "as_written"),
                ("change", "Y127X", span, text.index("Y127X"), "as_written"),
            ],
        )
        # The quote is the paper's own text from the first to the last kept value; code
        # chose it, and the span's whole line follows it.
        quote, whole = kept.passages
        # A bracket the quote opens is closed where the paper closes it.
        self.assertEqual(quote.text, "PBP2 (Y127X)")
        self.assertEqual(text[quote.start : quote.end], quote.text)
        self.assertEqual(whole, line)
        [variant] = kept.variants
        self.assertEqual(
            (variant.hgvs, variant.position, kept.gene), (("Y127X",), None, "PBP2")
        )

    def test_a_text_the_span_repeats_is_kept_beside_its_finding_s_other_values(self):
        line = "The evolved strain carries PBP2 (Y127X) and PBP2 (Q30X) in one clone."
        paper = Paper(None, [document("main", f"# Y\n{line}")])

        [kept] = kept_answer(
            "Q?",
            reading_of(
                finding(
                    value("gene", "PBP2", "main:p2"), value("change", "Q30X", "main:p2")
                )
            ),
            paper,
        ).findings

        gene = kept.values[0]
        self.assertEqual(gene.source.start, len("# Y\n") + line.index("PBP2 (Q30X)"))
        self.assertEqual(kept.passages[0].text, "PBP2 (Q30X)")

    def test_a_quote_stops_before_another_variant_s_words_in_a_shared_sentence(self):
        # A recorded c2 run: SV2 keeps the contig both SNPs share, and its quote ran
        # "chr 2: 86,770, 373, SV1 and 86,870,271", across SV1's position.
        text, paper = c2_like()
        span = f"main:p{paragraph(text, C2_PARAGRAPH)}"
        sv1 = finding(
            value("contig", "chr 2", span),
            value("position", "86,770, 373", span, number=86770373),
        )
        sv2 = finding(
            value("contig", "chr 2", span),
            value("position", "86,870,271", span, number=86870271),
        )

        answer = kept_answer("Which SNPs?", reading_of(sv1, sv2), paper)

        self.assertEqual(
            [(f.passages[0].text, f.variants[0].contig) for f in answer.findings],
            [("chr 2: 86,770, 373", "chr 2"), ("86,870,271", "chr 2")],
        )

    def test_c2_s_damaged_group_is_kept_with_its_number_and_form(self):
        # stated-number-checked: the written text, the proper number and the span;
        # code finds the text and checks that its digits spell the number.
        text, paper = c2_like()
        span = f"main:p{paragraph(text, C2_PARAGRAPH)}"
        sv1 = finding(
            value("contig", "chr 2", span),
            value("position", "86,770, 373", span, number=86770373),
        )
        sv2 = finding(
            value("contig", "chr 2", span),
            value("position", "86,870,271", span, number=86870271),
        )

        answer = kept_answer(
            "Which SNPs cause rumplessness?", reading_of(sv1, sv2), paper
        )

        first, second = answer.findings
        position = first.values[1]
        self.assertEqual(
            (position.value, position.form, position.written),
            (86770373, "damaged_thousands_group", "86,770, 373"),
        )
        self.assertEqual(
            text[position.source.start : position.source.end], "86,770, 373"
        )
        self.assertEqual(
            (first.variants[0].contig, first.variants[0].position), ("chr 2", 86770373)
        )
        # One span gives SV2 its contig too: a span is evidence for every finding it holds.
        self.assertEqual(
            (second.values[1].form, second.variants[0].position),
            ("as_written", 86870271),
        )

    def test_a_number_its_written_digits_do_not_spell_is_refused_by_name(self):
        text, paper = c2_like()
        span = f"main:p{paragraph(text, C2_PARAGRAPH)}"

        def reasons(written: str, number: int | None) -> tuple[str, ...]:
            stated = finding(value("position", written, span, number=number))
            return judge(FindingPointer.model_validate(stated), paper).reasons

        self.assertEqual(
            reasons("86,770, 373", 86770),
            ("number_differs:position=86,770, 373,number=86770",),
        )
        self.assertEqual(
            reasons("86,770, 373", None), ("number_not_stated:position=86,770, 373",)
        )
        # Written as the model would print it, not as the paper does: not in the span.
        self.assertEqual(
            reasons("86,770,373", 86770373),
            (f"written_not_in_span:position=86,770,373 in {span}",),
        )
        # A part of a number is no number the paper writes.
        self.assertEqual(
            reasons("770", 770), (f"written_not_in_span:position=770 in {span}",)
        )

    def test_3_01_and_30_1_never_match_and_a_decimal_comma_stays_unresolved(self):
        line = "A depth of 30.1 at 3,01 sites gave P = 3.01 for chr 5: 1,234,567."
        text = f"# Q\n{line}"
        paper = Paper(None, [document("main", text)])
        span = "main:p2"

        def judged(written: str, number: int | None = None):
            stated = finding(value("position", written, span, number=number))
            return judge(FindingPointer.model_validate(stated), paper)

        self.assertEqual(judged("3.01", 301).reasons, ("not_read:position=3.01",))
        self.assertEqual(judged("30.1", 301).reasons, ("not_read:position=30.1",))
        # "3.0" is inside "30.1"'s and "3.01"'s text, but no whole number the line writes.
        self.assertEqual(
            judged("3.0", 30).reasons, (f"written_not_in_span:position=3.0 in {span}",)
        )
        unclear = judged("3,01", 301)
        self.assertEqual(
            (unclear.outcome, unclear.reasons),
            ("unresolved", ("ambiguous_number:position=3,01",)),
        )

    def test_a_transposed_table_binds_each_value_to_its_own_column(self):
        # C1's row holds both variants' loci; a value is read in its cell only.
        text, paper = c1_like()
        casq2 = finding(
            value("contig", "1", "t1:r2", column="CASQ2"),
            value("position", "115701412", "t1:r2", column="CASQ2", number=115701412),
            value("alleles", "C>G", "t1:r2", column="CASQ2"),
            value("rsid", "rs2101052355", "t1:r3", column="CASQ2"),
        )
        stolen = finding(
            value("position", "115701412", "t1:r2", column="ANK2", number=115701412)
        )

        answer = kept_answer("Which variants?", reading_of(casq2, stolen), paper)

        [kept] = answer.findings
        [variant] = kept.variants
        self.assertEqual(
            (variant.contig, variant.position, variant.ref, variant.alt, variant.rsids),
            ("1", 115701412, "C", "G", ("rs2101052355",)),
        )
        self.assertEqual({v.column for v in kept.values}, {"CASQ2"})
        self.assertEqual(kept.passages[0].text, "1:115701412 C>G")
        [refused] = answer.refused
        self.assertEqual(
            refused.reasons, ("written_not_in_span:position=115701412 in t1:r2",)
        )

    def test_a_variant_listed_down_one_column_is_kept_with_its_own_target(self):
        # Live run 1 L1: c3's Table 4 lists each SNP as a column; the locus is read down
        # that one column, REF and ALT by their row labels.
        paper = c3_like()

        answer = kept_answer(
            "Which SNPs?", reading_of(*(c3_snp(*snp) for snp in C3_SNPS)), paper
        )

        self.assertEqual((answer.refused, answer.unresolved), ((), ()))
        self.assertEqual(
            [
                (v.contig, v.position, v.ref, v.alt)
                for f in answer.findings
                for v in f.variants
            ],
            [
                ("4", 22375729, "G", "T"),
                ("6", 20640731, "C", "T"),
                ("10", 6918411, "C", "T"),
                ("11", 19577767, "C", "T"),
                ("11", 28814241, "G", "A"),
            ],
        )
        self.assertTrue(verify(answer, paper).verified)
        targets = paper_targets(answer).targets
        # Five targets: each SNP's quote is its own position cell, so the two
        # chromosome-11 SNPs stay two.
        self.assertEqual(
            [t.request.request_text for t in targets],
            [snp[2] for snp in C3_SNPS],
        )
        self.assertEqual(
            [(t.request.position.contig, t.request.position.position) for t in targets],
            [
                ("4", 22375729),
                ("6", 20640731),
                ("10", 6918411),
                ("11", 19577767),
                ("11", 28814241),
            ],
        )

    def test_two_column_variants_at_one_written_position_stay_two_targets(self):
        # Standards live2 S8: c3's positions are unique; these are not. Main joins targets
        # by their words, so each quote must tell its variant apart, in the paper's text.
        paper = Paper(
            PaperIdentity(doi="10.1000/x", resolved_by="user_supplied"),
            [
                document(
                    "main",
                    "# Q\nTable 1\nCHROM\t4\t6\nPosition\t1,234,567\t1,234,567\n"
                    "Reference\tA\tC\nALT\tG\tT\n",
                )
            ],
        )

        def snp(column: str, ref: str, alt: str) -> dict:
            return finding(
                value("contig", column, "t1:r1", column=column),
                value("position", "1,234,567", "t1:r2", column=column, number=1234567),
                value("ref", ref, "t1:r3", column=column),
                value("alt", alt, "t1:r4", column=column),
            )

        answer = kept_answer(
            "Q?", reading_of(snp("4", "A", "G"), snp("6", "C", "T")), paper
        )

        self.assertTrue(verify(answer, paper).verified)
        targets = paper_targets(answer).targets
        # Two located targets, each with its own words from the paper.
        self.assertEqual(
            [(t.request.position.contig, t.not_checked) for t in targets],
            [("4", None), ("6", None)],
        )
        self.assertEqual(len({t.request.request_text for t in targets}), 2)

    def column_targets(self, *columns: tuple[str, str, str, str, str]):
        """The reviewers' S8/P4 layout: one column per variant, its fields as rows."""

        rows = ["Name\t" + "\t".join(c[0] for c in columns)]
        for label, index in (
            ("CHROM", 1),
            ("Position", 2),
            ("Reference", 3),
            ("ALT", 4),
        ):
            rows.append(label + "\t" + "\t".join(c[index] for c in columns))
        paper = Paper(
            PaperIdentity(doi="10.1000/x", resolved_by="user_supplied"),
            [document("main", "# Q\n" + "\n".join(rows))],
        )
        stated = [
            finding(
                value("contig", contig, "t1:r2", column=column),
                value(
                    "position",
                    position,
                    "t1:r3",
                    column=column,
                    number=int(position.replace(",", "")),
                ),
                value("ref", ref, "t1:r4", column=column),
                value("alt", alt, "t1:r5", column=column),
            )
            for column, contig, position, ref, alt in columns
        ]
        answer = kept_answer("Which variants?", reading_of(*stated), paper)
        self.assertEqual(len(answer.findings), len(columns))
        self.assertTrue(verify(answer, paper).verified)
        return paper_targets(answer).targets

    def test_swapped_allele_pairs_at_one_locus_stay_two_targets(self):
        # Compare by typed slot.
        targets = self.column_targets(
            ("v1", "4", "1,234,567", "A", "G"), ("v2", "4", "1,234,567", "G", "A")
        )

        self.assertEqual(
            [(t.request.position.ref, t.request.position.alt) for t in targets],
            [("A", "G"), ("G", "A")],
        )

    def test_a_replacement_quote_never_takes_another_finding_s_words(self):
        # Standards live3 S8 / Spec P4: check every final target's words.
        targets = self.column_targets(
            ("v1", "4", "1,234,567", "A", "G"),
            ("v2", "6", "1,234,567", "C", "T"),
            ("v3", "8", "4", "G", "T"),
        )

        self.assertEqual(
            sorted(
                (t.request.position.contig, t.request.position.position)
                for t in targets
            ),
            [("4", 1234567), ("6", 1234567), ("8", 4)],
        )

    def test_a_column_locus_never_mixes_columns_or_reads_a_genotype(self):
        paper = c3_like()

        def reasons(*values: dict) -> tuple[str, ...]:
            return judge(FindingPointer.model_validate(finding(*values)), paper).reasons

        # Position from column 4, REF and ALT from column 6: two variants' values.
        self.assertEqual(
            reasons(
                value("contig", "4", "t1:r1", column="4"),
                value("position", "22,375,729", "t1:r2", column="4", number=22375729),
                value("ref", "C", "t1:r3", column="6"),
                value("alt", "T", "t1:r4", column="6"),
            ),
            ("located_values_apart",),
        )
        # Standards live2 S10: the repair names both lawful ways to cite a locus.
        self.assertIn(
            "or down one column of one table",
            findings_repair(
                [
                    judge(
                        FindingPointer.model_validate(
                            finding(
                                value("contig", "4", "t1:r1", column="4"),
                                value("ref", "C", "t1:r3", column="6"),
                            )
                        ),
                        paper,
                    )
                ]
            ),
        )
        # A genotype row is no REF or ALT, whatever its cells hold.
        self.assertEqual(
            reasons(
                value("contig", "4", "t1:r1", column="4"),
                value("position", "22,375,729", "t1:r2", column="4", number=22375729),
                value("ref", "0/0", "t1:r5", column="4"),
                value("alt", "T", "t1:r4", column="4"),
            )[0],
            "not_read:ref=0/0",
        )
        # Code names no REF or ALT row the finding leaves out.
        self.assertEqual(
            reasons(
                value("contig", "4", "t1:r1", column="4"),
                value("position", "22,375,729", "t1:r2", column="4", number=22375729),
            ),
            (),
        )

    def test_the_agent_names_ref_and_alt_cells_and_code_copies_whole_base_cells(self):
        # Keywords guide the agent's search and the agent names REF and ALT from the
        # chunk; code copies a whole cell of bases in the variant's own row or
        # column, whatever its label says.
        paper = Paper(
            None,
            [
                document(
                    "main",
                    "# Q\nTable 2\nSNP\tsnp1\tsnp2\n"
                    "Chromosome\t3\t5\nPosition (bp)\t1,234,567\t7,654,321\n"
                    "WT allele\tA\tC\nMutant allele\tG\tT\n"
                    "Table 3\nGene\tReference amino acid\tAlt aa\nGENE1\tAla\tGly\n"
                    "Table 4\nGene\tReference amino acid\tAlt aa\nGENE2\tA\tG\n"
                    "Table 5\nSNP\tsnp3\nProtein residue ref\tC\nProtein residue alt\tT\n",
                )
            ],
        )

        def judged(*values: dict):
            return judge(FindingPointer.model_validate(finding(*values)), paper)

        column = [
            value("contig", "3", "t1:r2", column="snp1"),
            value("position", "1,234,567", "t1:r3", column="snp1", number=1234567),
        ]
        kept = judged(
            *column,
            value("ref", "A", "t1:r4", column="snp1"),
            value("alt", "G", "t1:r5", column="snp1"),
        )
        self.assertEqual((kept.outcome, kept.reasons), ("accepted", ()))
        # Code names no REF or ALT cell a finding leaves out.
        self.assertEqual(judged(*column).reasons, ())
        protein = judged(
            value("ref", "Ala", "t2:r2", column="Reference amino acid"),
            value("alt", "Gly", "t2:r2", column="Alt aa"),
        )
        self.assertEqual(protein.reasons, ("not_read:ref=Ala", "not_read:alt=Gly"))
        # A whole cell of bases the model names is copied under any column or row label,
        # protein words included: no label vetoes it (ADR 0046).
        for labelled in (
            judged(
                value("ref", "A", "t3:r2", column="Reference amino acid"),
                value("alt", "G", "t3:r2", column="Alt aa"),
            ),
            judged(
                value("ref", "C", "t4:r2", column="snp3"),
                value("alt", "T", "t4:r3", column="snp3"),
            ),
        ):
            self.assertEqual((labelled.outcome, labelled.reasons), ("accepted", ()))

    def test_a_written_text_never_spans_two_cells(self):
        paper = s17_like()

        stated = finding(value("position", "4 54108297", "t1:r2", number=454108297))

        self.assertEqual(
            judge(FindingPointer.model_validate(stated), paper).reasons,
            ("written_not_in_span:position=4 54108297 in t1:r2",),
        )

    def test_an_unknown_span_or_column_is_refused_by_name(self):
        paper = s17_like()

        for written, reason in (
            (value("change", "c.2624C>A", "main:p99"), "unknown_span:main:p99"),
            (value("change", "c.2624C>A", "t7:r1"), "unknown_span:t7:r1"),
            (
                value("position", "54108297", "t1:r2", column="GENE", number=54108297),
                "unknown_column:t1:r2:GENE",
            ),
        ):
            with self.subTest(reason=reason):
                judged = judge(FindingPointer.model_validate(finding(written)), paper)
                self.assertEqual(
                    (judged.outcome, judged.reasons), ("refused", (reason,))
                )
        extra = finding(value("change", "p.Ser875*", "main:p3"), spans=["main:p0"])
        self.assertEqual(
            judge(FindingPointer.model_validate(extra), paper).reasons,
            ("unknown_span:main:p0",),
        )

    def test_a_genotype_or_a_change_name_s_bases_are_no_ref_and_alt(self):
        # A/C is a genotype. c.1841C>T's bases are a transcript's.
        text, paper = c2_like()
        span = f"main:p{paragraph(text, C2_PARAGRAPH)}"
        h1 = Paper(None, [document("main", f"# H1\n{H1_KCNH2_ROW}")])

        genotype = judge(
            FindingPointer.model_validate(finding(value("alleles", "A/C", span))), paper
        )
        [hit] = grep(h1, "c.1841C>T").hits
        from_change = judge(
            FindingPointer.model_validate(finding(value("alleles", "C>T", hit.span))),
            h1,
        )

        self.assertEqual(genotype.reasons, ("alleles_without_direction:A/C",))
        self.assertEqual(from_change.reasons, ("alleles_from_change_name:c.1841C>T",))

    def test_ref_and_alt_are_kept_only_from_their_own_columns_and_together(self):
        paper = s17_like()

        def reasons(*values: dict) -> tuple[str, ...]:
            return judge(FindingPointer.model_validate(finding(*values)), paper).reasons

        self.assertEqual(
            reasons(
                value("ref", "G", "t1:r2", column="REF"),
                value("alt", "T", "t1:r2", column="ALT"),
            ),
            (),
        )
        # A base in another column is no allele; one allele alone is incomplete.
        self.assertEqual(
            reasons(
                value("ref", "4", "t1:r2", column="CHROM"),
                value("alt", "T", "t1:r2", column="ALT"),
            ),
            ("not_read:ref=4",),
        )
        self.assertEqual(
            reasons(value("ref", "G", "t1:r2", column="REF")), ("alleles_incomplete",)
        )

    def test_a_locus_in_two_spans_or_a_contig_alone_is_refused(self):
        text, paper = s3_like()
        span = f"main:p{paragraph(text, S3_LINE)}"
        other = Paper(
            None,
            [document("main", f"# Y\n{S3_LINE}\nThe locus chr11 was resequenced.")],
        )

        apart = finding(
            value("contig", "chr11", "main:p3"),
            value("position", "207,469", "main:p2", number=207469),
        )
        alone = finding(value("contig", "chr11", span))

        self.assertEqual(
            judge(FindingPointer.model_validate(apart), other).reasons,
            ("located_values_apart",),
        )
        self.assertEqual(
            judge(FindingPointer.model_validate(alone), paper).reasons,
            ("contig_without_position:chr11",),
        )

    def test_a_locus_value_its_span_writes_beside_the_position_must_be_stated(self):
        # Standards r4 S6 and live s3 run 1, on spans: a position stated without the
        # contig or alleles written with it locates no target.
        text, paper = s3_like()
        span = f"main:p{paragraph(text, S3_LINE)}"
        position = value("position", "207,469", span, number=207469)

        def reasons(*values: dict) -> tuple[str, ...]:
            return judge(FindingPointer.model_validate(finding(*values)), paper).reasons

        self.assertEqual(
            reasons(position, value("alleles", "A>C", span)),
            ("value_not_stated:contig=chr11",),
        )
        self.assertEqual(
            reasons(position, value("contig", "chr11", span)),
            ("value_not_stated:alleles=A>C",),
        )
        self.assertEqual(
            reasons(
                position, value("contig", "chr11", span), value("alleles", "A>C", span)
            ),
            (),
        )
        # A table row's base cells are no alleles code names: the model names REF and
        # ALT.
        self.assertEqual(
            judge(
                FindingPointer.model_validate(
                    finding(
                        value("contig", "4", "t1:r2", column="CHROM"),
                        value(
                            "position",
                            "54108297",
                            "t1:r2",
                            column="POS (EquCab3.0)",
                            number=54108297,
                        ),
                    )
                ),
                s17_like(),
            ).reasons,
            (),
        )

    def test_a_bare_chromosome_name_is_a_contig_only_before_its_position_or_alone(self):
        text, paper = c1_like()

        kept = judge(
            FindingPointer.model_validate(
                finding(
                    value("contig", "4", "t1:r2", column="ANK2"),
                    value(
                        "position",
                        "113355007",
                        "t1:r2",
                        column="ANK2",
                        number=113355007,
                    ),
                    value("alleles", "G>A", "t1:r2", column="ANK2"),
                )
            ),
            paper,
        )
        prose = Paper(
            None, [document("main", "# Q\nIn 4 dogs the variant at 5,000 was seen.")]
        )
        counted = judge(
            FindingPointer.model_validate(
                finding(
                    value("contig", "4", "main:p2"),
                    value("position", "5,000", "main:p2", number=5000),
                )
            ),
            prose,
        )

        self.assertEqual(kept.outcome, "accepted")
        self.assertEqual(counted.reasons, ("not_read:contig=4",))

    def test_a_gene_is_kept_only_as_a_whole_symbol(self):
        paper = Paper(None, [document("main", "# G\nThe ASPRV1-related variant.")])

        self.assertEqual(
            judge(
                FindingPointer.model_validate(
                    finding(value("gene", "ASPR", "main:p2"))
                ),
                paper,
            ).reasons,
            ("written_not_in_span:gene=ASPR in main:p2",),
        )
        self.assertEqual(
            judge(
                FindingPointer.model_validate(
                    finding(value("gene", "ASPRV1", "main:p2"))
                ),
                paper,
            ).reasons,
            ("written_not_in_span:gene=ASPRV1 in main:p2",),
        )

    def test_a_genomic_change_name_gives_the_locus_it_writes(self):
        paper = Paper(
            None,
            [
                document(
                    "main", "# G\nthe variant NC_000004.12:g.54108297G>T was absent"
                )
            ],
        )

        [kept] = kept_answer(
            "Q?",
            reading_of(
                finding(value("change", "NC_000004.12:g.54108297G>T", "main:p2"))
            ),
            paper,
        ).findings

        [variant] = kept.variants
        self.assertEqual(
            (variant.contig, variant.position, variant.ref, variant.alt),
            ("NC_000004.12", 54108297, "G", "T"),
        )

    def test_a_finding_with_no_variant_value_gives_no_variant(self):
        # There is no quote, and a gene alone locates nothing.
        text, paper = s3_like()
        span = f"main:p{paragraph(text, S3_LINE)}"

        [kept] = kept_answer(
            "Q?", reading_of(finding(value("gene", "PBP2", span))), paper
        ).findings

        self.assertEqual((kept.variants, kept.gene), ((), "PBP2"))
        self.assertEqual(
            judge(FindingPointer.model_validate(finding()), paper).reasons,
            ("no_values",),
        )

    def test_one_refused_finding_leaves_its_siblings_outcomes_intact(self):
        # Each finding is decided on its own; c2 once lost SV2 to SV1's refusal.
        text, paper = c2_like()
        span = f"main:p{paragraph(text, C2_PARAGRAPH)}"
        good = finding(
            value("contig", "chr 2", span),
            value("position", "86,870,271", span, number=86870271),
        )
        bad = finding(value("alleles", "A/C", span), reason="Genotypes.")
        unclear = Paper(None, [document("main", f"# Q\n{C2_PARAGRAPH} Depth 3,01.")])

        answer = kept_answer("Q?", reading_of(good, bad), paper)

        self.assertEqual(
            (len(answer.findings), len(answer.unresolved), len(answer.refused)),
            (1, 0, 1),
        )
        [refused] = answer.refused
        self.assertEqual(
            (refused.outcome, refused.reasons, refused.stated.reason),
            ("refused", ("alleles_without_direction:A/C",), "Genotypes."),
        )
        self.assertEqual(refused.passages, (paper.unit(span).line,))
        [one] = kept_answer(
            "Q?",
            reading_of(finding(value("position", "3,01", "main:p2", number=301))),
            unclear,
        ).unresolved
        self.assertEqual(one.reasons, ("ambiguous_number:position=3,01",))

    def test_the_repair_asks_again_only_for_the_refused_findings(self):
        text, paper = c2_like()
        span = f"main:p{paragraph(text, C2_PARAGRAPH)}"
        judged = judge_all(
            [
                FindingPointer.model_validate(f)
                for f in (
                    finding(value("position", "86,870,271", span, number=86870271)),
                    finding(value("position", "86,770,373", span, number=86770373)),
                )
            ],
            paper,
        )

        repair = findings_repair([j for j in judged if j.outcome == "refused"], kept=1)

        self.assertIn(
            "Code keeps 1 finding of this item: do not send it again.", repair
        )
        self.assertIn(
            f'Refused 1 (position "86,770,373" in {span}):\n'
            f"- written_not_in_span:position=86,770,373 in {span} (this span does not "
            "write this text exactly;",
            repair,
        )
        self.assertNotIn("86,870,271", repair)
        self.assertIn("corrects, the refused finding's number", repair)

    def test_the_evidence_view_marks_each_value_at_its_span_and_lists_the_rest(self):
        # The chunk with each kept value marked and labelled by slot, built from the
        # recorded answer; unresolved and refused findings are listed.
        text, paper = c2_like()
        span = f"main:p{paragraph(text, C2_PARAGRAPH)}"
        sv1 = finding(
            value("contig", "chr 2", span),
            value("position", "86,770, 373", span, number=86770373),
            reason="SV1.",
        )
        genotype = finding(value("alleles", "A/C", span), reason="Genotypes.")
        answer = kept_answer("Which SNPs?", reading_of(sv1, genotype), paper)

        view = answer_markdown(answer, paper=paper)

        self.assertIn(f"#### `{span}` · `main` · Results · finding 1, refused 1", view)
        self.assertIn("SNPs (==chr 2==: ==86,770, 373==, SV1 and 86,870,271", view)
        self.assertIn(
            'position **86770373** (written "86,770, 373", damaged_thousands_group)',
            view,
        )
        self.assertIn("- accepted finding 1: contig **chr 2** · position", view)
        self.assertIn("## Refused (1)", view)
        self.assertIn("- Reason: alleles_without_direction:A/C", view)
        # Each mark is the paper's own text at its recorded place.
        for kept in answer.findings[0].values:
            self.assertIn(f"=={text[kept.source.start : kept.source.end]}==", view)

    def test_a_refused_value_holding_a_line_break_still_comes_back_as_a_repair(self):
        text, paper = s3_like()
        span = f"main:p{paragraph(text, S3_LINE)}"
        judged = judge_all(
            [
                FindingPointer.model_validate(
                    finding(
                        value("gene", "WRONG\nNAME", span),
                        value("change", "Y127X", span),
                    )
                )
            ],
            paper,
        )

        repair = findings_repair(judged)

        self.assertEqual(
            judged[0].reasons, (f"written_not_in_span:gene=WRONG\nNAME in {span}",)
        )
        self.assertIn(f'Refused 1 (gene "WRONG\nNAME" in {span}, ', repair)
        self.assertIn(
            f"- written_not_in_span:gene=WRONG\nNAME in {span} (this span does not",
            repair,
        )

    def test_an_item_with_no_finding_and_no_reason_is_refused_whole_once(self):
        empty = ItemAnswer.model_validate({"findings": []})

        self.assertEqual(item_problems(empty), ["item_unanswered"])
        # A repair call that sends nothing more keeps the findings an earlier call gave.
        self.assertEqual(item_problems(empty, kept=1), [])
        self.assertIn(
            "- item_unanswered (give the findings", repair_text(item_problems(empty))
        )

    def test_a_plan_refusal_keeps_its_whole_detail_and_names_no_finding(self):
        # Standards r2 S2: a plan's view id is the model's text; a trailing
        # "@finding N" in it is data, never a finding of the paper item.
        _text, paper = s3_like()
        for view in ("main@finding 2", "main@finding 0"):
            with self.subTest(view=view):
                plan = ReadingPlan.model_validate(
                    {
                        "reference": None,
                        "areas": [{"area": "A.", "views": [view], "reason": "R."}],
                    }
                )

                problems = tuple(plan_problems(plan, paper, ids_listed=False))
                repair = repair_text(problems)

                self.assertEqual(problems, (f"unknown_view:{view}",))
                self.assertIn(f"- unknown_view:{view} (no section", repair)
                self.assertNotIn("Finding ", repair)

    def test_a_version_the_paper_writes_two_ways_is_kept_by_its_name_and_aliases(self):
        # C2 and c3 each named both names in one, and each cost a repair; owner L3 (a): one
        # name, the others as aliases, and code records the name.
        c2 = (
            "Trimmed and concordant reads were mapped to the Gallus_gallus-5.0 reference "
            "genome (galGal5), generated from a female Red Junglefowl."
        )
        c3 = (
            "High-quality reads were then mapped to the reference genome Nipponbare 1.0 "
            "(IRGSP-1.0, release 52) using bwa-mem (v0.7.17) with default parameters24."
        )

        def plan(line: str, name: str, aliases: list[str]) -> tuple[Paper, ReadingPlan]:
            paper = Paper(None, [document("main", f"# Methods\n{line}")])
            reference = {"name": name, "aliases": aliases, "span": "main:p2"}
            return paper, ReadingPlan.model_validate(
                {
                    "reference": {**reference, "reason": "R."},
                    "areas": [],
                    "not_answered": "N.",
                }
            )

        for line, name, aliases in (
            (c2, "Gallus_gallus-5.0", ["galGal5"]),
            (c3, "IRGSP-1.0", ["Nipponbare 1.0"]),
        ):
            with self.subTest(name=name):
                paper, kept = plan(line, name, aliases)

                self.assertEqual(plan_problems(kept, paper, ids_listed=False), [])
                stated = stated_reference(kept.reference, paper).candidate
                self.assertEqual((stated.name, stated.spellings), (name, (name,)))
        paper, unwritten = plan(c2, "Gallus_gallus-5.0", ["galGal6"])
        problems = plan_problems(unwritten, paper, ids_listed=False)
        self.assertEqual(problems, ["reference_not_written:galGal6"])
        # The run 2 shape, both names in one, is still refused; the repair names aliases.
        paper, joined = plan(c2, "Gallus_gallus-5.0 (galGal5)", [])
        problems = plan_problems(joined, paper, ids_listed=False)
        self.assertEqual(
            problems,
            [
                "reference_not_written:Gallus_gallus-5.0 (galGal5):Gallus_gallus-5.0:galGal5"
            ],
        )
        self.assertIn(
            "the others the span writes for the same version in aliases",
            repair_text(problems),
        )

    def test_an_alias_the_span_writes_as_a_phrase_is_kept_and_never_recorded(self):
        # A recorded live c3 run: the alias "IRGSP-1.0, release 52" is written in the
        # span but is no one version name, and it was refused twice. An alias is never
        # recorded, so it needs only to be written there; the name stays one version.
        line = (
            "High-quality reads were then mapped to the reference genome Nipponbare 1.0 "
            "(IRGSP-1.0, release 52) using bwa-mem (v0.7.17) with default parameters24."
        )
        paper = Paper(None, [document("main", f"# Methods\n{line}")])

        def problems(name: str, aliases: list[str]) -> list[str]:
            reference = {"name": name, "aliases": aliases, "span": "main:p2"}
            plan = ReadingPlan.model_validate(
                {
                    "reference": {**reference, "reason": "R."},
                    "areas": [],
                    "not_answered": "N.",
                }
            )
            return plan_problems(plan, paper, ids_listed=False)

        self.assertEqual(problems("IRGSP-1.0", ["IRGSP-1.0, release 52"]), [])
        self.assertEqual(
            problems("IRGSP-1.0", ["release 53"]), ["reference_not_written:release 53"]
        )
        self.assertEqual(
            problems("IRGSP-1.0, release 52", []),
            ["reference_not_written:IRGSP-1.0, release 52:IRGSP-1.0"],
        )

    def test_a_span_id_is_kept_as_an_area_view(self):
        # A recorded c2 run: the paper item named the paragraph's span id as an area
        # view and the plan went back for repair; the area item reads it with
        # read_paper.
        text, paper = c2_like()
        span = f"main:p{paragraph(text, C2_PARAGRAPH)}"
        area = {"area": "SNPs at the Rp locus.", "views": [span, "s2"], "reason": "R."}
        plan = ReadingPlan.model_validate({"reference": None, "areas": [area]})

        self.assertEqual(plan_problems(plan, paper, ids_listed=False), [])
        unknown = ReadingPlan.model_validate(
            {"reference": None, "areas": [{**area, "views": ["main:p99"]}]}
        )
        self.assertEqual(
            plan_problems(unknown, paper, ids_listed=False), ["unknown_view:main:p99"]
        )

    def test_a_plan_above_the_area_ceiling_is_refused_with_its_count(self):
        # Each area is its own item with its own calls.
        _text, paper = c2_like()

        def plan(count: int) -> ReadingPlan:
            area = {"area": "SNPs.", "views": ["s2"], "reason": "R."}
            return ReadingPlan.model_validate(
                {"reference": None, "areas": [area] * count}
            )

        problems = plan_problems(plan(MAX_AREAS + 1), paper, ids_listed=False)

        self.assertEqual(problems, [f"too_many_areas:{MAX_AREAS + 1}"])
        self.assertIn(
            f"- too_many_areas:{MAX_AREAS + 1} (propose at most {MAX_AREAS} areas",
            repair_text(problems),
        )
        self.assertEqual(plan_problems(plan(MAX_AREAS), paper, ids_listed=False), [])

    def test_a_position_is_never_a_fragment_of_a_longer_written_number(self):
        # Pre-review helper D3, D7: a part of a grouped number passed the token check,
        # and a position 0 was read. Each finding states its contig, so the refusal is
        # the position's own.
        text, paper = c2_like()
        span = f"main:p{paragraph(text, C2_PARAGRAPH)}"
        spaced = Paper(None, [document("main", "# Q\nAt chr5: 1 234 567 it was.")])
        zero = Paper(None, [document("main", "# Q\nchr4: 0 here")])

        for contig, written, number, in_paper, at in (
            ("chr 2", "86,770", 86770, paper, span),
            ("chr 2", "373", 373, paper, span),
            ("chr5", "1 234", 1234, spaced, "main:p2"),
            ("chr5", "234 567", 234567, spaced, "main:p2"),
            ("chr4", "0", 0, zero, "main:p2"),
        ):
            with self.subTest(written=written):
                judged = judge(
                    FindingPointer.model_validate(
                        finding(
                            value("contig", contig, at),
                            value("position", written, at, number=number),
                        )
                    ),
                    in_paper,
                )
                self.assertEqual(judged.outcome, "refused")
                self.assertRegex(
                    judged.reasons[0], r"^(written_not_in_span|not_read):position="
                )
        self.assertEqual(
            judge(
                FindingPointer.model_validate(
                    finding(
                        value("contig", "chr5", "main:p2"),
                        value("position", "1 234 567", "main:p2", number=1234567),
                    )
                ),
                spaced,
            ).outcome,
            "accepted",
        )

    def test_an_rsid_is_kept_in_main_s_spelling_and_joins_its_other_spellings(self):
        # Pre-review helper D2: "RS1800562" was kept as written and broke the targets.
        paper = Paper(
            PaperIdentity(doi="10.1000/x", resolved_by="user_supplied"),
            [document("main", "# Q\nHFE RS1800562 was common.\nThe rs1800562 allele.")],
        )

        answer = kept_answer(
            "Q?",
            reading_of(
                finding(value("rsid", "RS1800562", "main:p2")),
                finding(value("rsid", "rs1800562", "main:p3")),
            ),
            paper,
        )

        [joined] = merged_findings(answer.findings)
        self.assertEqual(joined.variants[0].rsids, ("rs1800562",))
        self.assertEqual(answer.findings[0].values[0].written, "RS1800562")
        [target] = paper_targets(
            answer.model_copy(update={"findings": (joined,)})
        ).targets
        self.assertEqual(target.request.rsid, "rs1800562")

    def test_a_written_text_across_a_control_character_is_not_in_its_span(self):
        # Pre-review helper D8: the shown text matched, the paper's did not; no crash.
        paper = Paper(None, [document("main", "# Q\nKIT p.\x0c(Arg67Cys) was found.")])

        judged = judge(
            FindingPointer.model_validate(
                finding(value("change", "p. (Arg67Cys)", "main:p2"))
            ),
            paper,
        )

        self.assertEqual(
            judged.reasons, ("written_not_in_span:change=p. (Arg67Cys) in main:p2",)
        )

    def test_a_number_after_a_word_and_colon_is_no_contig_to_state(self):
        # Pre-review helper: "exon 2: 207,469" names no chromosome 2.
        paper = Paper(
            None, [document("main", "# Q\nIn exon 2: 207,469 is the KIT change.")]
        )

        judged = judge(
            FindingPointer.model_validate(
                finding(value("position", "207,469", "main:p2", number=207469))
            ),
            paper,
        )

        self.assertEqual(judged.outcome, "accepted")

    def test_every_cited_span_gets_its_whole_card(self):
        # Pre-review helper D5, D6: a quote that is the whole row, and a merged finding's
        # quote, never take the place of the span's own card.
        paper = s17_like()
        row = finding(
            value("contig", "4", "t1:r2", column="CHROM"),
            value(
                "position",
                "54108297",
                "t1:r2",
                column="POS (EquCab3.0)",
                number=54108297,
            ),
            value("ref", "G", "t1:r2", column="REF"),
            value("alt", "T", "t1:r2", column="ALT"),
            spans=["main:p9"],
        )
        answer = kept_answer("Q?", reading_of(row), paper)

        view = answer_markdown(answer, paper=paper)

        self.assertIn("#### `t1:r2` · `main` · Results · finding 1", view)
        self.assertIn("> ==4== ==54108297== ==G== ==T==", view)
        self.assertIn("#### `main:p9`", view)

    def test_outcomes_are_numbered_alike_in_their_lists_and_cards(self):
        text, paper = c2_like()
        span = f"main:p{paragraph(text, C2_PARAGRAPH)}"
        unclear = Paper(None, [document("main", f"# Q\n{C2_PARAGRAPH} Depth 3,01.")])
        answer = kept_answer(
            "Q?",
            reading_of(
                finding(value("position", "3,01", "main:p2", number=301)),
                finding(value("alleles", "A/C", "main:p2"), reason="Genotypes."),
            ),
            unclear,
        )

        view = answer_markdown(answer, paper=unclear)

        self.assertIn("### refused 1: Genotypes.", view)
        self.assertIn("- refused 1: no value in this span", view)
        self.assertIn("- unresolved 1: no value in this span", view)
        self.assertNotIn("refused refused", view)
        self.assertTrue(span)

    def test_verify_checks_the_chunks_of_unresolved_and_refused_findings(self):
        text, paper = c2_like()
        span = f"main:p{paragraph(text, C2_PARAGRAPH)}"
        answer = kept_answer(
            "Q?", reading_of(finding(value("alleles", "A/C", span), reason="G.")), paper
        )
        data = answer.model_dump(mode="json")
        data["refused"][0]["passages"][0]["text"] = data["refused"][0]["passages"][0][
            "text"
        ].replace("wildtype", "Wildtype")

        problems = verify(PaperAnswer.model_validate(data), paper).problems

        self.assertEqual(
            problems, (f"passage_differs:refused 1:main@{paper.unit(span).line.start}",)
        )

    def test_verify_binds_each_variant_field_to_a_kept_value(self):
        # Standards r1 S1: a stored projection no kept value supports fails.
        paper = Paper(
            None, [document("main", "# Q\nchr5: 301 A>G in GENE1 was found.")]
        )
        answer = kept_answer(
            "Q?",
            reading_of(
                finding(
                    value("contig", "chr5", "main:p2"),
                    value("position", "301", "main:p2", number=301),
                    value("alleles", "A>G", "main:p2"),
                    value("gene", "GENE1", "main:p2"),
                )
            ),
            paper,
        )
        self.assertTrue(verify(answer, paper).verified)

        def altered(**changes) -> PaperAnswer:
            data = answer.model_dump(mode="json")
            variant = data["findings"][0]["variants"][0]
            variant.update(changes)
            key = f"{variant['contig']}:{variant['position']}:{variant['ref']}>{variant['alt']}"
            variant["candidate_id"] = candidate_id("v", key)
            return PaperAnswer.model_validate(data)

        self.assertEqual(
            verify(altered(position=999), paper).problems,
            ("variant_unsupported:position=999:1",),
        )
        self.assertEqual(
            verify(altered(alt="C"), paper).problems,
            ("variant_unsupported:alleles=A>C:1",),
        )
        stripped = answer.model_dump(mode="json")
        stripped["findings"][0]["values"] = []
        self.assertIn(
            "values_missing:1",
            verify(PaperAnswer.model_validate(stripped), paper).problems,
        )

    def test_verify_binds_a_gene_only_finding_and_a_lone_allele_too(self):
        # Standards r2 S1: no early return for a finding without values or a variant,
        # and an ALT without a REF is checked.
        paper = Paper(
            None, [document("main", "# Q\nchr5: 301 A>G in GENE1 was found.")]
        )
        gene_only = kept_answer(
            "Q?", reading_of(finding(value("gene", "GENE1", "main:p2"))), paper
        )
        located = kept_answer(
            "Q?",
            reading_of(
                finding(
                    value("contig", "chr5", "main:p2"),
                    value("position", "301", "main:p2", number=301),
                    value("alleles", "A>G", "main:p2"),
                )
            ),
            paper,
        )
        stripped = gene_only.model_dump(mode="json")
        stripped["findings"][0].update(values=[], gene="FABRICATED")
        lone = located.model_dump(mode="json")
        variant = lone["findings"][0]["variants"][0]
        variant.update(ref=None, alt="C", candidate_id=candidate_id("v", "chr5:301"))

        self.assertIn(
            "values_missing:1",
            verify(PaperAnswer.model_validate(stripped), paper).problems,
        )
        self.assertIn(
            "variant_unsupported:alleles=None>C:1",
            verify(PaperAnswer.model_validate(lone), paper).problems,
        )

    def test_verify_reports_an_unreadable_recorded_rsid_without_raising(self):
        # Standards r2 S6: a projection is drawn only from values code can read.
        paper = Paper(
            None, [document("main", "# Q\nchr5: 301 A>G in GENE1 was found.")]
        )
        data = kept_answer(
            "Q?", reading_of(finding(value("gene", "GENE1", "main:p2"))), paper
        ).model_dump(mode="json")
        data["findings"][0]["values"][0]["slot"] = "rsid"

        result = verify(PaperAnswer.model_validate(data), paper)

        self.assertFalse(result.verified)
        self.assertIn("not_read:rsid=GENE1:1", result.problems)

    def test_verify_refuses_a_kept_value_at_a_fragment_judge_refuses(self):
        # Standards r1 S2: verify makes judge's whole-token check too.
        paper = Paper(
            None, [document("main", "# Q\nA depth of 301 in ASPRV1 was measured.")]
        )
        answer = kept_answer(
            "Q?",
            reading_of(
                finding(
                    value("position", "301", "main:p2", number=301),
                    value("gene", "ASPRV1", "main:p2"),
                )
            ),
            paper,
        )
        data = answer.model_dump(mode="json")
        position, gene = data["findings"][0]["values"]
        position.update(written="1", value=1)
        position["source"].update(start=position["source"]["end"] - 1, text="1")
        data["findings"][0]["variants"] = []
        gene.update(written="ASPR")
        gene["source"].update(end=gene["source"]["start"] + 4, text="ASPR")
        data["findings"][0]["gene"] = "ASPR"

        problems = verify(PaperAnswer.model_validate(data), paper).problems

        self.assertIn("written_not_in_span:position=1 in main:p2:1", problems)
        self.assertIn("written_not_in_span:gene=ASPR in main:p2:1", problems)

    def test_alleles_stated_two_ways_must_agree(self):
        # Standards r1 S3: no allele pair wins by dictionary priority.
        paper = Paper(
            None,
            [
                document(
                    "main",
                    "# Q\nCHROM\tPOS\tREF\tALT\tPAIR\n5\t301\tA\tG\tC>T\n5\t302\tA\tG\tA>G\n",
                )
            ],
        )

        def judged(row: str, pair: str):
            return judge(
                FindingPointer.model_validate(
                    finding(
                        value("contig", "5", row, column="CHROM"),
                        value(
                            "position",
                            row[-1] == "2" and "301" or "302",
                            row,
                            column="POS",
                            number=301 if row.endswith("2") else 302,
                        ),
                        value("ref", "A", row, column="REF"),
                        value("alt", "G", row, column="ALT"),
                        value("alleles", pair, row, column="PAIR"),
                    )
                ),
                paper,
            )

        self.assertEqual(
            judged("t1:r2", "C>T").reasons, ("alleles_disagree:C>T,ref=A,alt=G",)
        )
        self.assertEqual(judged("t1:r3", "A>G").outcome, "accepted")
        prose = Paper(
            None,
            [document("main", "# Q\nchr4:54108297 G>T is NC_000004.12:g.54108298G>T.")],
        )
        self.assertEqual(
            judge(
                FindingPointer.model_validate(
                    finding(
                        value("contig", "chr4", "main:p2"),
                        value("position", "54108297", "main:p2", number=54108297),
                        value("alleles", "G>T", "main:p2"),
                        value("change", "NC_000004.12:g.54108298G>T", "main:p2"),
                    )
                ),
                prose,
            ).reasons,
            ("locus_disagrees:NC_000004.12:g.54108298G>T",),
        )

    def test_a_genomic_change_at_position_0_is_no_change_code_reads(self):
        # Standards r1 S4: refused before any outcome is recorded, never a crash.
        for change in ("g.0A>G", "m.0A>G"):
            with self.subTest(change=change):
                paper = Paper(
                    None, [document("main", f"# Q\nThe reported change is {change}.")]
                )
                stated = finding(value("change", change, "main:p2"))

                answer = kept_answer("Q?", reading_of(stated), paper)

                [refused] = answer.refused
                self.assertEqual(refused.reasons, (f"not_read:change={change}",))

    def test_the_reference_is_kept_only_where_its_span_writes_the_name(self):
        # A version cited at a table's header row is refused by name.
        paper = s17_like()

        self.assertEqual(
            reference_problems(
                ReferencePointer(name="EquCab3.0", span="t1:r1", reason="R."), paper
            ),
            [],
        )
        self.assertEqual(
            reference_problems(
                ReferencePointer(name="EquCab3.0", span="t1:r2", reason="R."), paper
            ),
            ["reference_not_written:EquCab3.0"],
        )
        self.assertEqual(
            reference_problems(
                ReferencePointer(name="EquCab3.0", span="t1:r9", reason="R."), paper
            ),
            ["unknown_span:t1:r9"],
        )


class LookupTests(unittest.TestCase):
    def test_each_id_of_a_list_is_looked_up_in_code_and_each_miss_is_marked(self):
        found = lookup(
            s17_like(),
            ["4:54108297", "chr6:15,346,206", "rs1800562", "RAPGEF5", "6:54108297"],
        )

        self.assertEqual(
            [(f.id, f.kind, f.found, f.lines) for f in found],
            [
                ("4:54108297", "position", True, 3),
                ("chr6:15,346,206", "position", True, 1),
                ("rs1800562", "rsid", False, 0),
                ("RAPGEF5", "name", True, 3),
                ("6:54108297", "position", False, 0),
            ],
        )
        position, chr6, absent, gene, elsewhere = found
        # The Results line, Table 1's row and S6's row, each at the position's number.
        self.assertEqual(
            [(h.document_id, h.offset) for h in position.hits],
            [
                ("main", MAIN.index("54108297")),
                ("main", MAIN.index("\t54108297") + 1),
                (S6_ID, S6.index("54108297")),
            ],
        )
        self.assertEqual(
            position.hits[2].table,
            TableCell(table_id="t2", location="chr4", row=2, column="POS"),
        )
        self.assertEqual(
            chr6.hits[0].table,
            TableCell(table_id="t3", location="chr6", row=3, column="POS"),
        )
        self.assertEqual((absent.hits, absent.near), ((), ()))
        self.assertEqual([h.offset for h in gene.hits][0], MAIN.index("RAPGEF5"))
        # The number is in the paper, but never beside contig 6.
        self.assertEqual(elsewhere.hits, ())
        self.assertEqual(len(elsewhere.near), 3)

    def test_a_name_is_found_whatever_its_case(self):
        paper = Paper(
            None, [document("main", "## Results\nThe c.2624C>A change, 1052T>C in S1.")]
        )

        found = lookup(paper, ["C.2624c", "1052t", "2624C"])

        self.assertEqual(
            [(f.kind, f.found) for f in found],
            [("name", True), ("name", True), ("name", False)],
        )

    def test_a_hundred_positions_from_a_table_resolve_in_code(self):
        positions = [(4, 54_000_000 + 7 * n) for n in range(100)]
        sheet = "\n".join(
            ["CHROM\tPOS\tREF\tALT"] + [f"{c}\t{p}\tG\tT" for c, p in positions]
        )
        paper = Paper(
            None,
            [
                PaperDocument(
                    document_id="supp:S6.xlsx",
                    kind="supplementary",
                    url="https://example.org/S6.xlsx",
                    text=f"## chr4 (101 rows)\n{sheet}",
                )
            ],
        )
        ids = [f"{c}:{p}" for c, p in positions] + ["4:1", "5:54000000", "rs1"]

        found = lookup(paper, ids)

        self.assertEqual([f.found for f in found], [True] * 100 + [False] * 3)
        # Two digits are no position (as for the candidates): "4:1" is a word, not found.
        self.assertEqual([f.kind for f in found[100:]], ["name", "position", "rsid"])
        self.assertEqual(
            [(f.lines, f.hits[0].table.row) for f in found[:100]],
            [(1, row) for row in range(2, 102)],
        )


def s17_answer(paper: Paper) -> PaperAnswer:
    """The s17 finding as the model reads it: the Results words that name it, the S6 row
    stating its locus, and the version as the Methods write it."""

    [row] = rows(paper, "t2", where={"POS": "54108297"}).rows
    results = f"main:p{paragraph(MAIN, RESULTS_LINE)}"
    reading = reading_of(
        finding(
            value("gene", "RAPGEF5", results),
            value("change", "c.2624C>A", results),
            value("change", "p.Ser875*", results),
            value("contig", "4", row.span, column="CHROM"),
            value("position", "54108297", row.span, column="POS", number=54108297),
            value("ref", "G", row.span, column="REF"),
            value("alt", "T", row.span, column="ALT"),
            reason="The only HIGH-effect variant, exclusive to TBs.",
        ),
        reference={
            "name": "EquCab3.0",
            "span": f"main:p{paragraph(MAIN, 'Reads were mapped')}",
            "reason": "The positions are on EquCab3.0.",
        },
        not_answered=None,
    )
    return kept_answer("What is the most meaningful finding?", reading, paper)


def changed(answer: PaperAnswer, path: tuple, old: str, new: str) -> PaperAnswer:
    """The answer with one copied value edited, as a later reader might find it."""

    data = answer.model_dump(mode="json")
    holder = data
    for key in path[:-1]:
        holder = holder[key]
    assert old in holder[path[-1]]
    holder[path[-1]] = holder[path[-1]].replace(old, new, 1)
    return PaperAnswer.model_validate(data)


class VerifyTests(unittest.TestCase):
    def test_every_kept_value_is_read_again_where_the_answer_cites_it(self):
        paper = s17_like()
        answer = s17_answer(paper)

        result = verify(answer, paper)

        self.assertEqual(
            (result.verified, result.findings, result.problems), (True, 1, ())
        )

    def test_one_changed_copied_byte_fails_and_names_its_pointer(self):
        paper = s17_like()
        answer = s17_answer(paper)
        _quote, results, row = answer.findings[0].passages
        reference = answer.reference.candidate.sources[0]

        in_passage = changed(
            answer, ("findings", 0, "passages", 2, "text"), "G\tT", "G\tA"
        )
        in_line = changed(
            answer,
            ("findings", 0, "variants", 0, "sources", 1, "text"),
            "exclusive",
            "Exclusive",
        )
        in_reference = changed(
            answer, ("reference", "candidate", "sources", 0, "text"), "EquCab", "EquCaB"
        )

        self.assertEqual(
            verify(in_passage, paper).problems,
            (f"passage_differs:1:{row.document_id}@{row.start}",),
        )
        self.assertEqual(
            verify(in_line, paper).problems,
            (f"passage_differs:1:{results.document_id}@{results.start}",),
        )
        # The edited line no longer writes the version's name either.
        self.assertEqual(
            verify(in_reference, paper).problems,
            (
                f"passage_differs:reference:main@{reference.start}",
                "reference_not_written:reference:EquCab3.0",
            ),
        )
        self.assertFalse(verify(in_passage, paper).verified)

    def test_every_recorded_alias_must_be_written_whole_in_the_reference_line(self):
        # Aliases reach targets.json and placement, so verify reads them as plan admission
        # does; a recorded c3 run's line and aliases.
        line = (
            "High-quality reads were then mapped to the reference genome Nipponbare 1.0 "
            "(IRGSP-1.0, release 52) using bwa-mem (v0.7.17) with default parameters24."
        )
        paper = Paper(None, [document("main", f"# Methods\n{line}")])
        reading = reading_of(
            reference={
                "name": "Nipponbare 1.0",
                "aliases": ["IRGSP-1.0, release 52"],
                "span": "main:p2",
                "reason": "R.",
            },
            not_answered="N.",
        )
        answer = kept_answer("Q?", reading, paper)
        unwritten = changed(
            answer, ("reference", "aliases", 0), "IRGSP-1.0, release 52", "IRGSP-1.1"
        )

        self.assertEqual(verify(answer, paper).problems, ())
        self.assertEqual(
            verify(unwritten, paper).problems,
            ("reference_not_written:reference:alias=IRGSP-1.1",),
        )

    def test_a_kept_value_read_otherwise_than_recorded_fails(self):
        paper = s17_like()
        data = s17_answer(paper).model_dump(mode="json")
        position = data["findings"][0]["values"][4]
        self.assertEqual(position["slot"], "position")
        position["form"] = "damaged_thousands_group"
        moved = s17_answer(paper).model_dump(mode="json")
        moved["findings"][0]["values"][4]["span"] = "t2:r4"

        self.assertEqual(
            verify(PaperAnswer.model_validate(data), paper).problems,
            ("value_differs:position=54108297:1",),
        )
        self.assertEqual(
            verify(PaperAnswer.model_validate(moved), paper).problems,
            ("value_differs:position=54108297:1",),
        )

    def test_an_answer_verifies_against_the_same_texts_however_they_were_reached(self):
        # The same stored texts, reached another way (another url, a Run's
        # PaperDocumentRead): a document is its text's digest.
        other = Paper(
            None,
            [
                d.model_copy(
                    update={
                        "url": "https://elsewhere.example/copy",
                        "read_event_id": "e1",
                    }
                )
                for d in s17_like().documents.values()
            ],
        )

        result = verify(s17_answer(other), s17_like())

        self.assertEqual((result.verified, result.problems), (True, ()))

    def test_a_pointer_the_paper_no_longer_holds_fails(self):
        paper = s17_like()
        answer = s17_answer(paper)
        shorter = Paper(
            None,
            [
                document("main", TITLE + METHODS),
                PaperDocument(
                    document_id=S6_ID,
                    kind="supplementary",
                    url="https://pmc-oa-opendata.s3.amazonaws.com/PMC7544121.1/pgen.1009028.s006.xlsx",
                    text=S6,
                ),
            ],
        )

        problems = verify(answer, shorter).problems

        quote = answer.findings[0].passages[0]
        self.assertIn(f"passage_differs:1:main@{quote.start}", problems)
        self.assertIn("document_differs:main", problems)

    def test_a_document_the_paper_does_not_have_fails_as_a_problem(self):
        # Standards r2 S5: a missing document is a mismatch, never an exception.
        answer = s17_answer(s17_like())
        without_s6 = Paper(None, [document("main", MAIN)])

        result = verify(answer, without_s6)

        _quote, _results, row = answer.findings[0].passages
        self.assertFalse(result.verified)
        self.assertIn(f"passage_differs:1:{S6_ID}@{row.start}", result.problems)
        self.assertIn(f"document_differs:{S6_ID}", result.problems)


# ---------------------------------------------------------------------------
# Reading items: the paper_reader role reads a paper through main's one loop
# ---------------------------------------------------------------------------

S6_RAPGEF5_ROW = (
    "NC_009147.3\t4\t54108297\tG\tT\tANN=T|stop_gained|HIGH|RAPGEF5|c.2624C>A|p.Ser875*"
)
# The row's change names, each by its INFO cell (t2:r2: the chr4 sheet's second row).
S6_ROW_NAMES = [
    value("change", "c.2624C>A", "t2:r2", column="INFO"),
    value("change", "p.Ser875*", "t2:r2", column="INFO"),
]
S6_ROW_GENE = value("gene", "RAPGEF5", "t2:r2", column="INFO")
WHY_RAPGEF5 = (
    "The paper's only HIGH-effect nonsense variant, exclusive to Thoroughbreds."
)


S17_AREA = {
    "area": "the stop-gained RAPGEF5 row of the chr4 sheet",
    "views": ["t2"],
    "reason": "S6 lists the associated variants by position",
}
NO_REF = "The paper names EquCab3.0 only beside its positions."


READING_TOOL_NAMES = ("outline_paper", "grep_paper", "read_paper", "filter_paper_rows")


GOLD = Path(__file__).parent / "fixtures/paper_retrieval"
# A gold paper's stated version names are marks; general genome vocabulary is
# not.
GENERAL_VERSIONS = {"hg19", "hg38", "GRCh37", "GRCh38"}


def gold_marks() -> set[str]:
    """Each gold paper's values in their written forms: positions, rsIDs, HGVS names,
    phrases and the paper's own version names (c1-c3, h1, s3, s16, s17)."""
    marks: set[str] = set()
    for name in ("gold.json", "gold-v2.json"):
        for scenario, entry in json.loads(
            (GOLD / name).read_text(encoding="utf-8")
        ).items():
            if scenario.startswith("_"):
                continue
            marks |= set(entry["reference"]["names"]) - GENERAL_VERSIONS
            for target in entry["key"]:
                position = target.get("pos")
                if position:
                    marks |= {str(position), f"{position:,}"}
                    digits = f"{position:,}".split(",")
                    marks.add(", ".join([",".join(digits[:-1]), digits[-1]]))
                marks |= {target[k] for k in ("rsid",) if target.get(k)}
                marks |= set(target.get("hgvs", ()))
                marks |= set(target.get("phrases", ()))
    return marks


class ReadingItemTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = self.root / "runtime.json"
        self.config.write_text(json.dumps(runtime_configuration()), encoding="utf-8")

    def ask(self, model, paper, question="What is the most meaningful finding?"):
        return ask(
            self.root,
            paper,
            question,
            model,
            config=self.config,
        )

    def test_a_paper_item_proposes_an_area_and_the_area_is_read_as_its_own_attempt(
        self,
    ):
        model = reading(
            [("outline_paper", {})],
            [
                (
                    "propose_reading_areas",
                    {
                        "reference": None,
                        "areas": [
                            {
                                "area": "the stop-gained RAPGEF5 row of the chr4 sheet",
                                "views": ["t2"],
                                "reason": "S6 lists the associated variants by position",
                            }
                        ],
                        "not_answered": "The paper names EquCab3.0 only in a header.",
                    },
                )
            ],
            [("filter_paper_rows", {"table_id": "t2", "where": {"POS": "54108297"}})],
            [
                (
                    "answer_reading_item",
                    {
                        "findings": [
                            finding(S6_ROW_GENE, *S6_ROW_NAMES, reason=WHY_RAPGEF5)
                        ]
                    },
                )
            ],
        )

        outcome = self.ask(model, s17_like())

        [kept] = outcome.answer.findings
        self.assertEqual((kept.gene, kept.reason), ("RAPGEF5", WHY_RAPGEF5))
        quote, row = kept.passages
        self.assertEqual(quote.text, "RAPGEF5|c.2624C>A|p.Ser875*")
        self.assertEqual((row.document_id, row.text), (S6_ID, S6_RAPGEF5_ROW))
        events = Journey.open(Path(outcome.journey_file)).read()
        starts = [e for e in events if e["event_type"] == "AttemptStarted"]
        # The paper item, then the area it proposed: two Attempts of the one role.
        self.assertEqual(
            [s["payload"]["agent_role"] for s in starts], ["paper_reader"] * 2
        )
        self.assertEqual(len({s["payload"]["attempt_id"] for s in starts}), 2)
        self.assertEqual(outcome.requests.model_requests, 4)
        self.assertEqual(
            outcome.requests.tool_calls, ("outline_paper", "filter_paper_rows")
        )

    def test_a_call_naming_no_view_is_refused_with_its_tool_argument_value_and_reason(
        self,
    ):
        seen = []
        model = reading(
            [
                (
                    "propose_reading_areas",
                    {"reference": None, "areas": [S17_AREA], "not_answered": NO_REF},
                )
            ],
            [("read_paper", {"at": "s99"})],
            [("filter_paper_rows", {"table_id": "t2", "where": {"Gene": "RAPGEF5"}})],
            [("read_paper", {"at": "s2"})],
            [("answer_reading_item", {"findings": [], "not_answered": "x"})],
            seen=seen,
        )

        outcome = self.ask(model, s17_like())

        # Each refused call comes back as its own correction, naming the Tool, the
        # argument, the value sent and the registry row's reason; it never ran.
        view, column = (repairs(seen[n])[-1] for n in (2, 3))
        for expected in ("`read_paper` was not run", '`at` is "s99"', "No section"):
            self.assertIn(expected, view)
        for expected in (
            "`filter_paper_rows` was not run",
            '`where` is {"Gene": "RAPGEF5"}',
            "not a column of the table",
        ):
            self.assertIn(expected, column)
        events = Journey.open(Path(outcome.journey_file)).read()
        refused = [
            (e["payload"]["tool_name"], e["payload"]["errors"])
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["status"] == "invalid"
        ]
        self.assertEqual(
            refused,
            [
                ("read_paper", ["unknown_view"]),
                ("filter_paper_rows", ["unknown_filter_column"]),
            ],
        )
        self.assertEqual(
            [
                (e["payload"]["tool_name"], e["payload"]["arguments"].get("at"))
                for e in events
                if e["event_type"] == "PaperReadingRecorded"
            ],
            [("read_paper", "s2")],
        )

    def test_a_stopped_item_holds_only_itself_and_its_continue_answer_re_enters_it(
        self,
    ):
        """ADR 0039 and owner #383 Q23: a reading item is a member of its reading's
        collection, so its stop holds only itself (the shared hold rule, #105); the
        reading's answer waits for it, and its Review's continue answer re-enters it."""
        from bio_neb.agentic_journey.agent_work_review import (
            ContinueAgentWorkResponse,
            respond_agent_work_review,
        )
        from bio_neb.paper_reading.reading import advance_reading, reading_outcome

        second = {
            **S17_AREA,
            "area": "the RAPGEF5 sentence of the results",
            "views": ["s2"],
        }
        model = reading(
            [
                (
                    "propose_reading_areas",
                    {
                        "reference": None,
                        "areas": [S17_AREA, second],
                        "not_answered": NO_REF,
                    },
                )
            ],
            *[[("outline_paper", {})]] * 5,
            [("answer_reading_item", {"findings": [], "not_answered": "Not here."})],
            [("answer_reading_item", {"findings": [], "not_answered": "Read again."})],
        )

        outcome = self.ask(model, s17_like())

        # The call runs once; each repeat is refused by the shared admission's repeat
        # identity, and the fourth refused repeat in a row is the fourth failed turn,
        # which stops the item at the Agent work Review. The stop holds only that item:
        # the second area runs.
        self.assertEqual(
            [(o.item.item_id, o.stopped) for o in outcome.items],
            [("paper", None), ("area-1", "failed_turns"), ("area-2", None)],
        )
        self.assertIsNone(outcome.answer)
        journey = Journey.open(Path(outcome.journey_file))
        events = journey.read()
        first_area = outcome.items[1].attempt_id
        self.assertEqual(
            [
                e["payload"]["errors"]
                for e in events
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["attempt_id"] == first_area
            ],
            [[], *[["repeated_read"]] * 4],
        )
        self.assertEqual(outcome.requests.tool_calls, ("outline_paper",))
        self.assertEqual(
            [
                e["payload"]["member_id"].rsplit(":", 1)[1]
                for e in events
                if e["event_type"] == "AttemptStarted"
            ],
            ["paper", "area-1", "area-2"],
        )
        [review] = [e for e in events if e["event_type"] == "AgentWorkReviewRequested"]
        respond_agent_work_review(
            journey,
            review["event_id"],
            ContinueAgentWorkResponse(action="continue", instruction="Read the table."),
        )
        (paper_event,) = [e for e in events if e["event_type"] == "PaperRecorded"]
        runtime = RuntimeBinding(
            read_runtime_config(self.config).runtime_selections(),
            paper_event["event_id"],
        )

        def advance():
            return advance_reading(
                journey,
                outcome.reading_id,
                runtime=runtime,
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model,
            )

        self.assertTrue(advance())
        reentered = reading_outcome(journey, outcome.reading_id)
        # The continue answer re-entered the stopped item with the researcher's
        # instruction; it completed, settling every item.
        self.assertEqual([o.state for o in reentered.items], ["completed"] * 3)
        self.assertIsNone(reentered.answer)
        # The coordinator's next step records the answer, once.
        self.assertTrue(advance())
        after = reading_outcome(journey, outcome.reading_id)
        self.assertEqual([o.stopped for o in after.items], [None, None, None])
        self.assertNotEqual(after.items[1].attempt_id, first_area)
        self.assertEqual(after.answer.findings, ())
        self.assertIn("Read again.", after.answer.not_answered)
        self.assertIn("Not here.", after.answer.not_answered)
        self.assertFalse(advance())

    def test_a_rejected_or_dismissed_last_item_settles_and_the_answer_is_recorded(
        self,
    ):
        """ADR 0039: a reject or dismiss answer settles the stopped item alone, as not
        read with the researcher's reason, and is accounted, not dropped; once every
        item is settled the coordinator records the answer (owner, #383 A1b)."""
        from bio_neb.agentic_journey.agent_work_review import (
            DismissAgentWorkResponse,
            RejectAgentWorkResponse,
            respond_agent_work_review,
        )
        from bio_neb.paper_reading.reading import advance_reading, reading_outcome

        second = {**S17_AREA, "area": "the chr4 sheet's RAPGEF5 row again"}
        for response, action, state in (
            (RejectAgentWorkResponse, "reject", "rejected"),
            (DismissAgentWorkResponse, "dismiss", "dismissed"),
        ):
            with self.subTest(state=state):
                model = reading(
                    [
                        (
                            "propose_reading_areas",
                            {
                                "reference": None,
                                "areas": [S17_AREA, second],
                                "not_answered": NO_REF,
                            },
                        )
                    ],
                    # Area 1 sends one read again until its fourth failed turn stops it.
                    *[[("outline_paper", {})]] * 5,
                    [
                        (
                            "answer_reading_item",
                            {
                                "findings": [
                                    finding(
                                        S6_ROW_GENE, *S6_ROW_NAMES, reason=WHY_RAPGEF5
                                    )
                                ]
                            },
                        )
                    ],
                )
                outcome = self.ask(model, s17_like())
                self.assertEqual(
                    [o.state for o in outcome.items], ["completed", "open", "completed"]
                )
                self.assertIsNone(outcome.answer)
                journey = Journey.open(Path(outcome.journey_file))
                [review] = [
                    e
                    for e in journey.read()
                    if e["event_type"] == "AgentWorkReviewRequested"
                ]
                receipt = respond_agent_work_review(
                    journey,
                    review["event_id"],
                    response(
                        action=action,
                        reason="Not needed for this question.",
                        asserted_by="researcher",
                    ),
                )
                (paper_event,) = [
                    e for e in journey.read() if e["event_type"] == "PaperRecorded"
                ]
                runtime = RuntimeBinding(
                    read_runtime_config(self.config).runtime_selections(),
                    paper_event["event_id"],
                )

                def advance(
                    journey=journey,
                    runtime=runtime,
                    model=model,
                    reading_id=outcome.reading_id,
                ):
                    return advance_reading(
                        journey,
                        reading_id,
                        runtime=runtime,
                        environ={"LOCAL_MODEL_KEY": "offline"},
                        model_factory=model,
                    )

                # One step settles the item as not read, the next records the answer.
                self.assertEqual([advance(), advance(), advance()], [True, True, False])
                after = reading_outcome(journey, outcome.reading_id)
                self.assertEqual(
                    [o.state for o in after.items], ["completed", state, "completed"]
                )
                events = journey.read()
                [not_read] = [
                    e for e in events if e["event_type"] == "ReadingItemNotRead"
                ]
                # The settlement's stored shape, unchanged since a stop of any kind
                # settles an item too.
                self.assertEqual(
                    not_read["payload"],
                    {
                        "reading_id": outcome.reading_id,
                        "item_id": "area-1",
                        "outcome": state,
                        "review_response_event_id": receipt.response_event_id,
                        "reason": "Not needed for this question.",
                    },
                )
                self.assertEqual(
                    [f.reason for f in after.answer.findings], [WHY_RAPGEF5]
                )
                self.assertIn(
                    f"{S17_AREA['area']}: not read, {state} by the researcher: "
                    "Not needed for this question.",
                    after.answer.not_answered,
                )
                self.assertEqual(after.recorded.item_event_ids[1], not_read["event_id"])
                self.assertEqual(
                    sum(e["event_type"] == "PaperAnswerRecorded" for e in events), 1
                )

    def advance(self, outcome, model) -> bool:
        """One step of ``outcome``'s reading under this test's runtime and model."""
        from bio_neb.paper_reading.reading import advance_reading

        journey = Journey.open(Path(outcome.journey_file))
        (paper_event,) = [
            e for e in journey.read() if e["event_type"] == "PaperRecorded"
        ]
        return advance_reading(
            journey,
            outcome.reading_id,
            runtime=RuntimeBinding(
                read_runtime_config(self.config).runtime_selections(),
                paper_event["event_id"],
            ),
            environ={"LOCAL_MODEL_KEY": "offline"},
            model_factory=model,
        )

    def assert_stop_settles_only_its_item(self, outcome, model, response_event_id):
        """The stop answer settles the stopped area alone as not read, with the
        researcher's reason, and grants no new model work; the answer follows."""
        from bio_neb.paper_reading.reading import reading_outcome

        attempts = sum(
            e["event_type"] == "AttemptStarted"
            for e in Journey.open(Path(outcome.journey_file)).read()
        )
        # One step settles the item as not read, the next records the answer.
        self.assertEqual(
            [self.advance(outcome, model) for _ in range(3)], [True, True, False]
        )
        journey = Journey.open(Path(outcome.journey_file))
        after = reading_outcome(journey, outcome.reading_id)
        self.assertEqual(
            [o.state for o in after.items], ["completed", "stopped", "completed"]
        )
        events = journey.read()
        self.assertEqual(
            sum(e["event_type"] == "AttemptStarted" for e in events), attempts
        )
        [not_read] = [e for e in events if e["event_type"] == "ReadingItemNotRead"]
        self.assertEqual(
            not_read["payload"],
            {
                "reading_id": outcome.reading_id,
                "item_id": "area-1",
                "outcome": "stopped",
                "review_response_event_id": response_event_id,
                "reason": "Enough reading.",
            },
        )
        self.assertIn(
            f"{S17_AREA['area']}: not read, stopped by the researcher: Enough reading.",
            after.answer.not_answered,
        )
        self.assertIn("Not here.", after.answer.not_answered)
        self.assertEqual(after.recorded.item_event_ids[1], not_read["event_id"])

    def two_areas(self, *replies):
        """A paper item proposing two areas, then ``replies`` across them."""

        second = {**S17_AREA, "area": "the RAPGEF5 sentence of the results"}
        return reading(
            [
                (
                    "propose_reading_areas",
                    {
                        "reference": None,
                        "areas": [S17_AREA, second],
                        "not_answered": NO_REF,
                    },
                )
            ],
            *replies,
        )

    def test_an_effort_stop_settles_its_item_as_not_read_and_its_siblings_go_on(
        self,
    ):
        """ADR 0031/0040: a member-scoped role's stop of any kind settles that member
        alone (``RoleDeclaration``; the reader's ``member_reject_effect``), so an
        effort Review answered stop records its item as not read (ADR 0039) without
        an extension, and the reading records its answer."""
        from bio_neb.agentic_journey.effort import (
            StopAgentEffortResponse,
            respond_effort_review,
        )

        configured = runtime_configuration()
        configured["agent_invocation"]["scopes"] = {
            "role": {"paper_reader": {"model_requests": 1}}
        }
        self.config.write_text(json.dumps(configured), encoding="utf-8")
        # Area 1 spends its one request on a read; its next asks for more effort.
        model = self.two_areas(
            [("outline_paper", {})],
            [("answer_reading_item", {"findings": [], "not_answered": "Not here."})],
        )
        outcome = self.ask(model, s17_like())
        self.assertEqual(
            [o.state for o in outcome.items], ["completed", "open", "completed"]
        )
        journey = Journey.open(Path(outcome.journey_file))
        [request] = [
            e for e in journey.read() if e["event_type"] == "AgentEffortReviewRequested"
        ]
        receipt = respond_effort_review(
            journey,
            request["event_id"],
            StopAgentEffortResponse(action="stop", reason="Enough reading."),
        )

        self.assert_stop_settles_only_its_item(
            outcome, model, receipt.response_event_id
        )

    def test_a_model_availability_stop_holds_then_settles_only_its_item(self):
        """ADR 0031/0040: an item's own model availability Review holds that item
        alone while it is pending (the shared hold rule holds only planning's role by
        it), and its stop answer settles the item as not read (ADR 0039).

        The answer is recorded as the application's answer command records it (the
        answer and its ``ModelAvailabilityStopped``); that command reads a whole Run,
        which this reading-only Journey is not."""
        from bio_neb.agentic_journey.continuation import review_answer_parents
        from bio_neb.agentic_journey.model_availability import (
            ModelAvailabilityReviewResponded,
            ModelAvailabilityStopped,
            StopModelAvailabilityResponse,
        )

        configured = runtime_configuration()
        configured["failure"] = {
            "model_contact": {"base_delay_seconds": 0, "contacts": 1}
        }
        self.config.write_text(json.dumps(configured), encoding="utf-8")
        second = {**S17_AREA, "area": "the RAPGEF5 sentence of the results"}
        replies = iter(
            [
                (
                    "propose_reading_areas",
                    {
                        "reference": None,
                        "areas": [S17_AREA, second],
                        "not_answered": NO_REF,
                    },
                ),
                None,  # area 1's one contact fails: its availability is exhausted
                ("answer_reading_item", {"findings": [], "not_answered": "Not here."}),
            ]
        )

        async def respond(messages, info):
            # A fresh Attempt of the held item would read it again.
            reply = next(
                replies,
                ("answer_reading_item", {"findings": [], "not_answered": "Again."}),
            )
            if reply is None:
                raise ModelHTTPError(503, "scripted/model", "unavailable")
            return ModelResponse(
                parts=[ToolCallPart(*reply, tool_call_id=f"c{len(messages)}")]
            )

        model = ScriptedModelEndpoint(respond)
        outcome = self.ask(model, s17_like())

        # Area 1's pending Review holds it alone: no fresh Attempt, area 2 is read.
        self.assertEqual(
            [o.state for o in outcome.items], ["completed", "open", "completed"]
        )
        journey = Journey.open(Path(outcome.journey_file))
        events = journey.read()
        self.assertEqual(
            [
                e["payload"]["member_id"].rsplit(":", 1)[1]
                for e in events
                if e["event_type"] == "AttemptStarted"
            ],
            ["paper", "area-1", "area-2"],
        )
        self.assertFalse(self.advance(outcome, model))
        [request] = [
            e for e in events if e["event_type"] == "ModelAvailabilityReviewRequested"
        ]
        answer = ModelAvailabilityReviewResponded(
            request_event_id=request["event_id"],
            response=StopModelAvailabilityResponse(
                action="stop", reason="Enough reading."
            ),
        )
        recorded = journey.record(
            answer.event_type,
            answer.model_dump(mode="json", exclude={"event_type"}),
            review_answer_parents(events, request["event_id"]),
        )
        stopped = ModelAvailabilityStopped(
            request_event_id=request["event_id"],
            response_event_id=recorded["event_id"],
            reason="Enough reading.",
        )
        journey.record(
            stopped.event_type,
            stopped.model_dump(mode="json", exclude={"event_type"}),
            [recorded["event_id"]],
        )

        self.assert_stop_settles_only_its_item(outcome, model, recorded["event_id"])

    def test_the_fold_settles_an_item_by_its_own_operational_stop_only(self):
        """ADR 0031/0040 (a stop of any kind settles a member-scoped role's member)
        and ADR 0032 (a false settlement is refused): the reading's fold accepts an
        operational Review's stop of the item's own Attempt as its not-read
        settlement, and refuses one naming a retry answer or another reason.

        Fold level only: no reader Tool contacts a provider, and the reader's Attempt
        failure handler holds the Run on any failure, so no operational Review of a
        reader item can be asked through the product today. The Review events are
        synthetic, appended to a real reading's prefix."""
        from bio_neb.paper_reading.journey import project_paper_readings

        configured = runtime_configuration()
        configured["agent_invocation"]["scopes"] = {
            "role": {"paper_reader": {"model_requests": 1}}
        }
        self.config.write_text(json.dumps(configured), encoding="utf-8")
        model = self.two_areas(
            [("outline_paper", {})],
            [("answer_reading_item", {"findings": [], "not_answered": "Not here."})],
        )
        outcome = self.ask(model, s17_like())
        prefix = Journey.open(Path(outcome.journey_file)).read()
        area = outcome.items[1].attempt_id

        def settled(action, reason, claimed):
            request = {
                "event_id": "operational-request",
                "event_type": "AgentFailureReviewRequested",
                "parent_ids": [prefix[-1]["event_id"]],
                "payload": {"attempt_id": area},
            }
            response = {
                "event_id": "operational-answer",
                "event_type": "AgentFailureReviewResponded",
                "parent_ids": ["operational-request"],
                "payload": {
                    "request_event_id": "operational-request",
                    "response": {
                        "action": action,
                        "by": "researcher",
                        "reason": reason,
                    },
                },
            }
            not_read = {
                "event_id": "not-read",
                "event_type": "ReadingItemNotRead",
                "parent_ids": ["operational-answer"],
                "payload": {
                    "reading_id": outcome.reading_id,
                    "item_id": "area-1",
                    "outcome": "stopped",
                    "review_response_event_id": "operational-answer",
                    **({} if claimed is None else {"reason": claimed}),
                },
            }
            [reading] = project_paper_readings(
                [*prefix, request, response, not_read]
            ).readings
            return reading.settled["area-1"]

        self.assertEqual(
            settled("stop", "Source is down.", "Source is down.").model_dump(),
            {"state": "stopped", "event_id": "not-read", "reason": "Source is down."},
        )
        self.assertEqual(settled("stop", None, None).reason, None)
        for action, claimed in (("retry", "Source is down."), ("stop", "Other.")):
            with (
                self.subTest(action=action, claimed=claimed),
                self.assertRaisesRegex(ValueError, "by its own Review's answer"),
            ):
                settled(action, "Source is down.", claimed)

    def test_a_read_repeated_anywhere_in_its_item_is_refused_whatever_its_cap(self):
        # Recorded repeats, replayed: an s17-list run alternated two zero-row filters (A,
        # B, A, B), and an s15 run sent one search again with another max_hits. The paper
        # does not change, so each repeat is refused, not run.
        rows_a = {"table_id": "t2", "where": {"POS": "99999999"}, "max_rows": 5}
        rows_b = {"table_id": "t3", "where": {"POS": "99999998"}, "max_rows": 5}
        seen = []
        model = reading(
            [
                (
                    "propose_reading_areas",
                    {"reference": None, "areas": [S17_AREA], "not_answered": NO_REF},
                )
            ],
            [("filter_paper_rows", rows_a)],
            [("filter_paper_rows", rows_b)],
            [("filter_paper_rows", rows_a)],
            [("filter_paper_rows", {**rows_b, "max_rows": 10})],
            [("grep_paper", {"pattern": "rs1800562", "max_hits": 10})],
            [("grep_paper", {"pattern": "rs4244285"})],
            [("grep_paper", {"pattern": "rs1800562", "max_hits": 5})],
            [("read_paper", {"at": "main", "offset": 0, "length": 100})],
            [("read_paper", {"at": "main", "offset": 100, "length": 100})],
            [("read_paper", {"at": "main", "offset": 0, "length": 200})],
            [("answer_reading_item", {"findings": [], "not_answered": "Not written."})],
            seen=seen,
        )

        outcome = self.ask(model, s17_like())

        area = outcome.items[1].attempt_id
        events = Journey.open(Path(outcome.journey_file)).read()
        self.assertEqual(
            [
                (e["payload"]["tool_name"], e["payload"]["errors"])
                for e in events
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["attempt_id"] == area
            ],
            [
                ("filter_paper_rows", []),
                ("filter_paper_rows", []),
                ("filter_paper_rows", ["repeated_read"]),
                ("filter_paper_rows", ["repeated_read"]),
                ("grep_paper", []),
                ("grep_paper", []),
                ("grep_paper", ["repeated_read"]),
                # A new offset is a new page of the document; the same page is not.
                ("read_paper", []),
                ("read_paper", []),
                ("read_paper", ["repeated_read"]),
            ],
        )
        self.assertEqual(
            outcome.requests.tool_calls,
            ("filter_paper_rows",) * 2 + ("grep_paper",) * 2 + ("read_paper",) * 2,
        )
        refused = repairs(seen[4])[-1]
        self.assertIn("`filter_paper_rows` was not run", refused)
        self.assertIn("This read already ran in this item", refused)
        self.assertEqual(outcome.stopped, ())

    def test_the_reading_rows_declare_their_result_shaping_arguments(self):
        rows_ = {
            row.tool_name: row.result_shaping
            for row in function_tool_contracts("paper_reader")
        }

        self.assertEqual(
            {name: rows_[name] for name in READING_TOOL_NAMES},
            {
                "outline_paper": (),
                "grep_paper": ("max_hits",),
                "read_paper": ("length",),
                "filter_paper_rows": ("max_rows",),
            },
        )

    def test_each_item_runs_under_the_role_declarations_budgets(self):
        """#446 role declaration: an item's request ceiling and failure rule are the
        global role default's, and a runtime file's role scope tunes the reader alone;
        the reading adds no run total of its own (owner, #439 budgets)."""
        from bio_neb.agentic_journey.role_declaration import ROLE_DEFAULT

        def budgets(outcome):
            return [
                (
                    e["payload"]["effective_invocation_policy"]["model_requests"],
                    e["payload"]["continuation_policy"]["failed_turns"],
                )
                for e in Journey.open(Path(outcome.journey_file)).read()
                if e["event_type"] == "AttemptStarted"
            ]

        def script():
            return reading(
                [
                    (
                        "propose_reading_areas",
                        {
                            "reference": None,
                            "areas": [S17_AREA],
                            "not_answered": NO_REF,
                        },
                    )
                ],
                [("answer_reading_item", {"findings": [], "not_answered": "x"})],
            )

        default = self.ask(script(), s17_like())
        tuned_config = self.root / "tuned.json"
        configured = runtime_configuration()
        configured["agent_invocation"]["scopes"] = {
            "role": {"paper_reader": {"model_requests": 7}}
        }
        tuned_config.write_text(json.dumps(configured), encoding="utf-8")
        tuned = ask(
            self.root,
            s17_like(),
            "What is the most meaningful finding?",
            script(),
            config=tuned_config,
        )

        self.assertEqual(
            budgets(default),
            [(ROLE_DEFAULT.model_requests, ROLE_DEFAULT.failed_turns)] * 2,
        )
        self.assertEqual(budgets(tuned), [(7, ROLE_DEFAULT.failed_turns)] * 2)

    def test_a_listed_id_found_in_code_is_answered_there_and_only_misses_reach_the_model(
        self,
    ):
        seen = []
        missing = "4:99999999 is written nowhere in the paper."
        model = reading(
            [
                (
                    "propose_reading_areas",
                    {
                        "reference": {
                            "name": "EquCab3.0",
                            "span": f"main:p{paragraph(MAIN, 'Reads were mapped')}",
                            "reason": "x",
                        },
                        "areas": [],
                    },
                )
            ],
            [("grep_paper", {"pattern": "99999999"})],
            [("answer_reading_item", {"findings": [], "not_answered": missing})],
            seen=seen,
        )

        outcome = self.ask(
            model,
            s17_like(),
            question="Where are 4:54108297, chr6:15346206 and 4:99999999 reported?",
        )

        paper_item, id_item = (json.loads(seen[n][0].parts[0].content) for n in (0, 1))
        self.assertEqual(
            (paper_item["ids_found_in_code"], paper_item["ids_missed"]),
            (2, ["4:99999999"]),
        )
        self.assertEqual((id_item["item"], id_item["id"]), ("id", "4:99999999"))
        # The model reads the question with the found ids replaced, in every request;
        # the Journey keeps the question as asked.
        self.assertEqual(
            {paper_item["question"], id_item["question"]},
            {"Where are [2 ids found in code] and 4:99999999 reported?"},
        )
        self.assertEqual(len(model.bodies), 3)
        for body in model.bodies:
            for found in ("54108297", "15346206"):
                self.assertNotIn(found, json.dumps(body))
        self.assertEqual(
            [(o.item.kind, o.item.subject) for o in outcome.items],
            [
                (
                    "paper",
                    "Where are [2 ids found in code] and 4:99999999 reported?",
                ),
                ("id", "4:99999999"),
            ],
        )
        # Code's findings for the two found ids: every line holding each, the first
        # as its quote, and the values the id states as that line writes them; code
        # names no table cell REF or ALT.
        chr4, chr6 = outcome.answer.findings
        self.assertEqual(
            [(v.contig, v.position, v.ref, v.alt) for v in chr4.variants],
            [("chr4", 54108297, None, None)],
        )
        self.assertEqual(
            [line.text for line in chr4.passages],
            ["chr4:54108297", RESULTS_LINE, "4\t54108297\tG\tT", S6_RAPGEF5_ROW],
        )
        self.assertEqual(
            [(v.contig, v.position, v.ref, v.alt) for v in chr6.variants],
            [("6", 15346206, None, None)],
        )
        # Code's quote is the paper's text from the first to the last value it kept.
        self.assertEqual(
            [line.text for line in chr6.passages],
            ["6\t15346206", "NC_009149.3\t6\t15346206\tT\tG\t."],
        )
        self.assertEqual(outcome.answer.not_answered, f"4:99999999: {missing}")
        events = Journey.open(Path(outcome.journey_file)).read()
        [opening] = [e for e in events if e["event_type"] == "PaperReadingStarted"]
        self.assertEqual(
            opening["payload"]["question"],
            "Where are 4:54108297, chr6:15346206 and 4:99999999 reported?",
        )
        self.assertEqual(
            [(i["id"], i["found"]) for i in opening["payload"]["requested"]],
            [("4:54108297", True), ("chr6:15346206", True), ("4:99999999", False)],
        )
        # The miss's search finds nothing: a lawful negative result the role declares.
        [searched] = [e for e in events if e["event_type"] == "PaperReadingRecorded"]
        self.assertEqual(searched["payload"]["outcome"], "nothing_found")
        self.assertTrue(negative_domain_result("paper_reader", searched))

    def test_a_code_found_position_states_its_contig_and_alleles_as_its_line_writes(
        self,
    ):
        # Pre-review helper D1: code stated contig "4" where the line writes chr4, and no
        # alleles where the line writes them after the position.
        for line, contig in (
            ("chr4:54,108,297 G>T in KIT.", "chr4"),
            ("Variant 4:54108297 G>T in KIT.", "4"),
        ):
            with self.subTest(line=line):
                paper = Paper(None, [document("main", f"# Q\n{line}")])
                [found] = lookup(paper, ["4:54108297"])

                judged = judge(found_in_code(found, paper), paper)

                self.assertEqual(judged.outcome, "accepted", judged.reasons)
                [variant] = kept_finding(judged).variants
                self.assertEqual(
                    (variant.contig, variant.position, variant.ref, variant.alt),
                    (contig, 54108297, "G", "T"),
                )

    def test_a_code_found_id_keeps_its_quote_across_a_control_character(self):
        # Review r9 (Spec S10): the quote code builds from the paper matches as shown.
        text = "## Results\nGENE1\x07rs12345\n"
        paper = Paper(None, [document("main", text)])
        model = reading(
            [
                (
                    "propose_reading_areas",
                    {"reference": None, "areas": [], "not_answered": NO_REF},
                )
            ]
        )

        outcome = self.ask(
            model, paper, question="What does the paper say about rs12345?"
        )

        [kept] = outcome.answer.findings
        self.assertEqual(
            [p.text for p in kept.passages], ["rs12345", "GENE1\x07rs12345"]
        )
        self.assertEqual(kept.variants[0].rsids, ("rs12345",))

    def test_no_reading_shows_the_model_a_control_character_to_copy(self):
        # The model copied a row's tabs into its quote and the call was lost.
        text = (
            "# Title\n## Results\nKCNH2\x07 is listed in Table 1.\nTable 1\n"
            "Variants found.\nGene\tChange\x07\nKCNH2\tc.1841C>T\x07\n"
        )
        paper = Paper(None, [document("main", text)])
        model = reading(
            [
                ("outline_paper", {}),
                ("grep_paper", {"pattern": "KCNH2"}),
                ("read_paper", {"at": "main"}),
                ("filter_paper_rows", {"where": {"Change": "c.1841C>T"}}),
            ],
            [
                (
                    "propose_reading_areas",
                    {"reference": None, "areas": [], "not_answered": NO_REF},
                )
            ],
        )

        outcome = self.ask(model, paper, question="Which\tSNVs?")

        events = read_evidence(Journey.open(Path(outcome.journey_file)))
        results = [
            e["payload"]["result"]
            for e in events
            if e["event_type"] == "PaperReadingRecorded"
        ]
        [message] = [
            e["payload"]["compatible_adapter_evidence"]["request"]
            for e in events
            if e["event_type"] == "ModelRequestRecorded"
        ][:1]
        # The first request holds the item's message; each result is what a call returned.
        shown_strings = json.dumps(results, ensure_ascii=False) + json.dumps(
            message["messages"], ensure_ascii=False
        )
        self.assertIn("Which SNVs?", shown_strings)
        self.assertNotRegex(shown_strings, r"\\t|\\r|\\u00(0|1)")
        # A column named with a control character is filtered by the name it is shown.
        self.assertEqual(len(results), 4)
        self.assertEqual(results[3]["rows"][0]["cells"]["Change"], "c.1841C>T ")

    def test_every_reading_call_is_recorded_with_its_fact_and_the_journey_reopens(self):
        model = reading(
            [("outline_paper", {}), ("grep_paper", {"pattern": "RAPGEF5"})],
            [
                (
                    "propose_reading_areas",
                    {"reference": None, "areas": [S17_AREA], "not_answered": NO_REF},
                )
            ],
            [
                ("read_paper", {"at": "s2"}),
                ("filter_paper_rows", {"where": {"POS": "54108297"}}),
            ],
            [
                (
                    "answer_reading_item",
                    {"findings": [finding(*S6_ROW_NAMES, reason=WHY_RAPGEF5)]},
                )
            ],
        )

        outcome = self.ask(model, s17_like())

        journey = Journey.open(Path(outcome.journey_file))
        # The Journey's events with its contact journal (observed executions).
        events = read_evidence(journey)
        # Main's reopen of the role's observed Tool executions, with no contact.
        validate_observed_function_tool_role(events, agent_role="paper_reader")
        by_id = {e["event_id"]: e for e in events}
        executed = [
            e["payload"]
            for e in events
            if e["event_type"] == "ObservedToolExecutionSucceeded"
        ]
        facts = [by_id[done["stage_result_event_id"]] for done in executed]
        self.assertEqual(
            [(f["event_type"], f["payload"]["tool_name"]) for f in facts],
            [
                ("PaperReadingRecorded", "outline_paper"),
                ("PaperReadingRecorded", "grep_paper"),
                ("PaperReadingRecorded", "read_paper"),
                ("PaperReadingRecorded", "filter_paper_rows"),
            ],
        )
        for done, fact in zip(executed, facts, strict=True):
            start = by_id[done["started_event_id"]]["payload"]
            validated = by_id[start["validation_event_id"]]["payload"]
            self.assertEqual(
                (validated["status"], validated["tool_call_id"]),
                ("valid", fact["payload"]["tool_call_id"]),
            )
        self.assertEqual(
            facts[3]["payload"]["result"]["rows"][0]["cells"]["INFO"],
            "ANN=T|stop_gained|HIGH|RAPGEF5|c.2624C>A|p.Ser875*",
        )
        # The one assembly offered the stage's reading Tools, the search and the item's
        # output Tool; each Attempt names the role's published instructions.
        requests = [
            e["payload"]["compatible_adapter_evidence"]["request"]
            for e in events
            if e["event_type"] == "ModelRequestRecorded"
        ]
        reading_tools = {
            "outline_paper",
            "grep_paper",
            "read_paper",
            "filter_paper_rows",
            "search_journey",
            "ask_researcher",
        }
        self.assertEqual(
            [
                (
                    {t["name"] for t in r["function_tools"]} == reading_tools,
                    [t["name"] for t in r["output_tools"]],
                )
                for r in requests
            ],
            [(True, ["propose_reading_areas"])] * 2
            + [(True, ["answer_reading_item"])] * 2,
        )
        for item in outcome.items:
            instructions = read_attempt_instructions(journey, item.attempt_id)
            self.assertEqual(
                (instructions.agent_role, instructions.version), ("paper_reader", "v14")
            )

    def test_the_readers_search_declares_mains_refusals_and_points_at_the_paper_tools(
        self,
    ):
        # The paper reader keeps the Journey search to find what earlier reading
        # items recorded; it states that the search never reads the paper, and each
        # refusal main's search can return reaches it explained.
        [search] = [
            row
            for row in function_tool_contracts("paper_reader")
            if row.tool_name == "search_journey"
        ]
        self.assertLessEqual(
            {p.code for p in SEARCH_PRECONDITIONS},
            {p.code for p in search.preconditions},
        )
        # Its purpose is the reader's own block.
        purpose = role_instructions("paper_reader").tool_purpose("search_journey")
        self.assertIn("does not read the paper", purpose)
        self.assertIn("grep_paper", purpose)
        text = role_instructions("paper_reader").text
        self.assertIn("never reads the paper", text)

    def test_nothing_the_model_is_shown_holds_a_gold_paper_s_value(self):
        # Live run 1 L2: c2's own damaged number was an example in the selected
        # instructions and the output schema. Gold values may sit in tests and records,
        # never in what the model reads before it reads the paper: no selected role's
        # instructions in any model's form, output schema, Tool row or the request
        # writer's wording. Only the selected versions ship, so the v3 and v6 texts
        # that held gold examples are archived.
        from bio_neb.agentic_journey.role_declaration import ROLES
        from bio_neb.agentic_journey.role_instructions import _FORMS

        marks = gold_marks()
        # Each role's instructions, message sentences and Tool purposes.
        shown = {
            f"{role}:{form}": "\n".join(
                (
                    selected.text,
                    *(text for _, text in selected.messages),
                    *(text for _, text in selected.tool_purposes),
                )
            )
            for role in ROLES
            for form in _FORMS
            for selected in (role_instructions(role, form=form),)
        }
        from bio_neb.paper_reading.answer import REFUSALS, UNRESOLVED

        shown["schemas, Tool rows, repairs and request writers"] = "\n".join(
            [
                json.dumps(ItemAnswer.model_json_schema()),
                json.dumps(ReadingPlan.model_json_schema()),
                *(repr(row) for row in function_tool_contracts("paper_reader")),
                # The repair wording each refused output reads (every model-
                # visible text the move brought into the product).
                *REFUSALS.values(),
                *UNRESOLVED.values(),
                *(inspect.getsource(writer) for writer in REQUEST_WRITERS),
            ]
        )
        self.assertIn("paper_reader:nemotron_3_5_super", shown)
        self.assertEqual(
            {
                name: found
                for name, text in shown.items()
                if (found := sorted(m for m in marks if m in text))
            },
            {},
        )

    def test_the_text_shown_beside_the_paper_names_no_ticket_or_lane(self):
        shown = "\n".join(
            [
                role_instructions("paper_reader").text,
                json.dumps(ItemAnswer.model_json_schema()),
                json.dumps(ReadingPlan.model_json_schema()),
                *(repr(row) for row in function_tool_contracts("paper_reader")),
                *(inspect.getsource(writer) for writer in REQUEST_WRITERS),
            ]
        )
        self.assertEqual(re.findall(r"#\d+|\blane [A-Z]\w*", shown), [])

    def test_the_reader_is_told_to_cite_span_ids_and_state_values_as_written(self):
        # A recorded h1 run: a change name's bases led the model to state REF/ALT; the
        # model now cites span ids and copies no paragraph.
        text = role_instructions("paper_reader").text
        for told in (
            "a change name's bases are not REF and ALT",
            "A slash pair such as A/C is a genotype",
            "You cite text only by these span ids; never copy a paragraph or row",
            '(written "12,345, 678", number 12345678)',
            "the variant's values are read down that one column",
            "search for these words and read the chunks around them",
            "each with corrects set to the number of the refused finding it corrects",
            # The area rules and the ceiling (paper_reader v8). The recorded name is the
            # assembly's own (paper_reader v9).
            "give in name the assembly's own name, the one a genome database lists it under",
            "and each other name in aliases",
            "never join two names into one",
            f"Propose the areas the question needs, at most {MAX_AREAS},",
            "or the span ids of paragraphs or rows, that hold it",
            "read_paper reads a section, a document, a table or one paragraph or row",
        ):
            self.assertIn(told, text)

    def test_an_item_runs_every_distinct_call_its_model_sends_with_no_scaffold_cap(
        self,
    ):
        patterns = [f"term{n}" for n in range(15)]
        model = reading(
            [
                (
                    "propose_reading_areas",
                    {"reference": None, "areas": [S17_AREA], "not_answered": NO_REF},
                )
            ],
            *([("grep_paper", {"pattern": pattern})] for pattern in patterns),
            [("answer_reading_item", {"findings": [], "not_answered": "x"})],
        )

        outcome = self.ask(model, s17_like())

        # Fifteen searches and the answer in one item: only main's caps bound the loop.
        self.assertEqual(
            [(o.item.kind, o.stopped) for o in outcome.items],
            [("paper", None), ("area", None)],
        )
        self.assertEqual(outcome.requests.tool_calls, ("grep_paper",) * 15)
        self.assertEqual(outcome.requests.model_requests, 17)

    def test_items_naming_one_variant_give_one_finding_with_each_item_s_evidence(self):
        second = {
            **S17_AREA,
            "area": "the RAPGEF5 sentence of the results",
            "views": ["s2"],
        }
        from_text = "The results name it the only HIGH-effect nonsense variant."
        model = reading(
            [
                (
                    "propose_reading_areas",
                    {
                        "reference": None,
                        "areas": [S17_AREA, second],
                        "not_answered": NO_REF,
                    },
                )
            ],
            [
                (
                    "answer_reading_item",
                    {
                        "findings": [
                            finding(
                                value("contig", "4", "t2:r2", column="CHROM"),
                                value(
                                    "position",
                                    "54108297",
                                    "t2:r2",
                                    column="POS",
                                    number=54108297,
                                ),
                                value("ref", "G", "t2:r2", column="REF"),
                                value("alt", "T", "t2:r2", column="ALT"),
                                *S6_ROW_NAMES,
                                S6_ROW_GENE,
                                reason=WHY_RAPGEF5,
                            )
                        ]
                    },
                )
            ],
            # The sentence states the locus without its alleles: it fits the row's.
            [
                (
                    "answer_reading_item",
                    {
                        "findings": [
                            finding(
                                value("contig", "chr4", "main:p3"),
                                value(
                                    "position", "54108297", "main:p3", number=54108297
                                ),
                                value("change", "c.2624C>A", "main:p3"),
                                value("change", "p.Ser875*", "main:p3"),
                                reason=from_text,
                            )
                        ]
                    },
                )
            ],
        )

        outcome = self.ask(model, s17_like())

        # One variant is one finding: the second item's evidence joins the first's.
        [kept] = outcome.answer.findings
        self.assertEqual(
            [(v.contig, v.position, v.ref, v.hgvs) for v in kept.variants],
            [
                ("4", 54108297, "G", ("c.2624C>A", "p.Ser875*")),
                ("chr4", 54108297, None, ("c.2624C>A", "p.Ser875*")),
            ],
        )
        # Each finding's quote (first to last kept value in its change's span), then its
        # line; the second item's evidence joins the first's.
        row_quote = S6_RAPGEF5_ROW[S6_RAPGEF5_ROW.index("4\t54108297") :]
        # The contig the sentence writes twice is kept beside its position.
        text_quote = RESULTS_LINE[
            RESULTS_LINE.index("chr4:54108297") : RESULTS_LINE.index("p.Ser875*)") + 10
        ]
        self.assertEqual(
            [p.text for p in kept.passages],
            [row_quote, S6_RAPGEF5_ROW, text_quote, RESULTS_LINE],
        )
        self.assertEqual(kept.reason, f"{WHY_RAPGEF5} {from_text}")
        self.assertEqual(kept.gene, "RAPGEF5")
        self.assertTrue(verify(outcome.answer, s17_like()).verified)

    def test_a_locus_without_alleles_where_two_pairs_are_stated_joins_neither(self):
        text = (
            "## Results\nBoth chr4:1000 G>T and chr4:1000 G>C were found.\n"
            "The locus chr4:1000 was seen twice.\n"
        )
        paper = Paper(None, [document("main", text)])

        def at(span: str, *alleles: str) -> FindingPointer:
            return FindingPointer.model_validate(
                finding(
                    value("contig", "chr4", span),
                    value("position", "1000", span, number=1000),
                    *(value("alleles", pair, span) for pair in alleles),
                )
            )

        judged = judge_all(
            [at("main:p2", "G>T"), at("main:p2", "G>C"), at("main:p3")], paper
        )

        self.assertEqual([j.outcome for j in judged], ["accepted"] * 3)
        self.assertEqual(len(merged_findings([kept_finding(j) for j in judged])), 3)


if __name__ == "__main__":
    unittest.main()


class ReadingFactTests(unittest.TestCase):
    """The reading's four product facts (owner, #383 Q2): written through the one
    semantic writer (ADR 0032), each document by its stored text's digest."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = self.root / "runtime.json"
        self.config.write_text(json.dumps(runtime_configuration()), encoding="utf-8")

    def read_s17(self, plan: dict, *answers: dict):
        model = reading(
            [("propose_reading_areas", plan)],
            *([("answer_reading_item", answer)] for answer in answers),
        )
        return ask(
            self.root,
            s17_like(),
            "What is the most meaningful finding?",
            model,
            config=self.config,
        )

    def test_each_fact_is_written_by_its_attempts_activity_and_names_stored_texts(self):
        outcome = self.read_s17(
            {"reference": None, "areas": [S17_AREA], "not_answered": NO_REF},
            {"findings": [finding(S6_ROW_GENE, *S6_ROW_NAMES, reason=WHY_RAPGEF5)]},
        )

        journey = Journey.open(Path(outcome.journey_file))
        events = journey.read()
        by_id = {e["event_id"]: e for e in events}
        attempts = {
            e["payload"]["attempt_id"]: e
            for e in events
            if e["event_type"] == "AttemptStarted"
        }
        kinds = set(READING_FACTS) - {"PaperReadingRecorded"}
        facts = [e for e in events if e["event_type"] in kinds]
        self.assertEqual(
            [e["event_type"] for e in facts],
            [
                "PaperReadingStarted",
                "ReadingPlanKept",
                "ReadingItemCompleted",
                "ReadingFindingJudged",
                "ReadingItemCompleted",
                "PaperAnswerRecorded",
            ],
        )
        for fact in facts[:-1]:
            with self.subTest(fact=fact["event_type"]):
                # Each is a step of its own Attempt's activity chain: its one parent is
                # the Attempt's previous fact, never the head a raw record would take.
                parent = by_id[fact["parent_ids"][0]]
                self.assertEqual(
                    parent["payload"].get("attempt_id"), fact["payload"]["attempt_id"]
                )
                self.assertIn(fact["payload"]["attempt_id"], attempts)
        # Each item's completion names its facts and copies none of them.
        plan, paper_done, judged, area_done, answered = facts[1:]
        self.assertEqual(
            [
                (d["payload"]["plan_event_id"], d["payload"]["finding_event_ids"])
                for d in (paper_done, area_done)
            ],
            [(plan["event_id"], []), (None, [judged["event_id"]])],
        )
        # The answer is the coordinator's, recorded once every item settled: no
        # Attempt owns it, and it names each item's settling fact and its findings.
        self.assertNotIn("attempt_id", answered["payload"])
        self.assertEqual(
            (
                answered["payload"]["item_event_ids"],
                answered["payload"]["finding_event_ids"],
            ),
            ([paper_done["event_id"], area_done["event_id"]], [judged["event_id"]]),
        )
        started = facts[0]["payload"]
        [paper] = [e for e in events if e["event_type"] == "PaperRecorded"]
        self.assertEqual(
            [
                (d["document_id"], d["read_event_id"], d["text_sha256"])
                for d in started["documents"]
            ],
            [
                (d["document_id"], d["read_event_id"], d["text_sha256"])
                for d in paper["payload"]["documents"]
            ],
        )
        # Each kept finding points into the stored text: its document's digest, its
        # start:end, and the chunk code copied from exactly there.
        answer = answered["payload"]["answer"]
        digests = {d["document_id"]: d["text_sha256"] for d in answer["documents"]}
        texts = {
            d["document_id"]: journey.read_artifact(
                by_id[d["read_event_id"]]["artifact_references"][0]
            ).decode()
            for d in started["documents"]
        }
        [kept] = answer["findings"]
        for line in kept["passages"]:
            self.assertIn(line["document_id"], digests)
            self.assertEqual(
                texts[line["document_id"]][line["start"] : line["end"]], line["text"]
            )
        self.assertEqual(kept["reason"], WHY_RAPGEF5)
        view = journey.checked_view("paper_readings").reading(outcome.reading_id)
        self.assertEqual(view.answer_event_id, answered["event_id"])
        self.assertEqual(
            {i.item_id: view.state(i.item_id) for i in view.items},
            {"paper": "completed", "area-1": "completed"},
        )

    def test_each_finding_is_recorded_as_its_own_fact_when_code_judges_it(self):
        # Each finding is one fact, recorded when judged, before its item ends; it
        # keeps its pointer, the chunk code copied and the model's reason.
        bad = finding(value("gene", "RAPGEF5", "main:p99"), reason="first try")
        fixed = finding(S6_ROW_GENE, reason="second try", corrects=1)
        good = finding(S6_ROW_GENE, *S6_ROW_NAMES, reason=WHY_RAPGEF5)
        outcome = self.read_s17(
            {"reference": None, "areas": [S17_AREA], "not_answered": NO_REF},
            {"findings": [good, bad]},
            {"findings": [fixed]},
        )

        journey = Journey.open(Path(outcome.journey_file))
        events = journey.read()
        area = outcome.items[1].attempt_id
        steps = [
            e
            for e in events
            if e["event_type"]
            in {"ReadingFindingJudged", "AgentOutputValidated", "ReadingItemCompleted"}
            and e["payload"]["attempt_id"] == area
        ]
        self.assertEqual(
            [
                (
                    e["event_type"],
                    e["payload"].get("outcome") or e["payload"].get("status"),
                )
                for e in steps
            ],
            [
                ("ReadingFindingJudged", "accepted"),
                ("ReadingFindingJudged", "refused"),
                ("AgentOutputValidated", "invalid"),
                ("ReadingFindingJudged", "accepted"),
                ("AgentOutputValidated", "valid"),
                ("ReadingItemCompleted", None),
            ],
        )
        accepted, refused, _, corrected, _, completed = steps
        self.assertEqual(
            (
                refused["payload"]["reasons"],
                refused["payload"]["stated"]["reason"],
                refused["payload"]["quote"],
            ),
            (["unknown_span:main:p99"], "first try", None),
        )
        self.assertEqual(corrected["payload"]["answers"], refused["event_id"])
        self.assertEqual(
            completed["payload"]["finding_event_ids"],
            [accepted["event_id"], refused["event_id"], corrected["event_id"]],
        )
        # The pointer and the chunk: each value by slot and span, each copied from
        # exactly start:end of the stored text whose digest the fact names.
        by_id = {e["event_id"]: e for e in events}
        fact = accepted["payload"]
        [document] = fact["documents"]
        content = journey.read_artifact(
            by_id[document["read_event_id"]]["artifact_references"][0]
        )
        self.assertEqual(document["text_sha256"], hashlib.sha256(content).hexdigest())
        text = content.decode()
        self.assertEqual(
            [(v["slot"], v["span"]) for v in fact["values"]],
            [(v["slot"], v["span"]) for v in good["values"]],
        )
        for line in (
            *fact["lines"],
            fact["quote"],
            *(v["source"] for v in fact["values"]),
        ):
            self.assertEqual(text[line["start"] : line["end"]], line["text"])
        self.assertEqual(fact["lines"][0]["text"], S6_RAPGEF5_ROW)
        self.assertEqual(fact["stated"]["reason"], WHY_RAPGEF5)
        self.assertEqual(
            [f.reason for f in outcome.answer.findings], [WHY_RAPGEF5, "second try"]
        )
        self.assertEqual(outcome.answer.refused, ())
        self.assertEqual(
            outcome.recorded.finding_event_ids,
            (accepted["event_id"], corrected["event_id"]),
        )

    def test_no_reading_code_records_a_raw_journey_fact(self):
        # ADR 0032: an Attempt's reading facts go through its Activity's record; the
        # one direct write is the coordinator's settling facts (an item not read, the
        # answer), through the Journey's semantic writer, in one place.
        import bio_neb.paper_reading as package
        from bio_neb.paper_reading import reading as reading_module

        writes = {
            path.name: len(
                re.findall(
                    r"journey\.record(?:_with_artifact)?\(",
                    path.read_text(encoding="utf-8"),
                )
            )
            for path in Path(package.__file__).parent.glob("*.py")
        }
        self.assertEqual(
            {name: count for name, count in writes.items() if count},
            {"reading.py": 1},
        )
        self.assertRegex(
            inspect.getsource(reading_module._record_settlement),
            r"journey\.record\(",
        )

    def test_a_started_reading_names_only_its_papers_stored_documents(self):
        # ADR 0032 grounding: the fold refuses an opening fact whose documents are not
        # the paper's PaperDocumentRead texts, before it is appended.
        from bio_neb.paper_reading.journey import PaperReadingStarted

        journey, paper_event_id = stored_journey(self.root / "run", s17_like())
        paper = Paper.stored(journey, paper_event_id)
        (recorded,) = [e for e in journey.read() if e["event_id"] == paper_event_id]
        forged = PaperReadingStarted(
            attempt_id="a1",
            reading_id="r1",
            paper_id=recorded["payload"]["paper_id"],
            paper_event_id=paper_event_id,
            question="Q?",
            documents=(
                paper.listing[0].model_copy(update={"text_sha256": "0" * 64}),
                *paper.listing[1:],
            ),
        )

        with self.assertRaisesRegex(ValueError, "names the paper's stored documents"):
            journey.record(
                forged.event_type,
                forged.model_dump(mode="json", exclude={"event_type"}),
                [journey.head_event_id],
            )

    def test_a_stored_paper_is_listed_by_its_intakes_digests_without_hashing_again(
        self,
    ):
        # ADR 0029: the intake admitted each text once and recorded its digest and
        # length; loading the paper (once per item) uses them and hashes nothing.
        from bio_neb.paper_reading import reader as reader_module

        paper = s17_like()
        journey, paper_event_id = stored_journey(self.root / "run", paper)
        reads = {
            e["payload"]["document_id"]: e["event_id"]
            for e in journey.read()
            if e["event_type"] == "PaperDocumentRead"
        }
        with mock.patch.object(reader_module, "hashlib", wraps=hashlib) as hashes:
            first = Paper.stored(journey, paper_event_id)
            second = Paper.stored(journey, paper_event_id)

        self.assertEqual(hashes.sha256.call_count, 0)
        self.assertEqual(first.listing, second.listing)
        self.assertEqual(
            [d.model_dump() for d in first.listing],
            [
                {
                    "document_id": document_id,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "characters": len(text),
                    "read_event_id": reads[document_id],
                }
                for document_id, text in (
                    ("main", paper.documents["main"].text),
                    (S6_ID, paper.documents[S6_ID].text),
                )
            ],
        )

    RAPGEF5_PLAN = {"reference": None, "areas": [S17_AREA], "not_answered": NO_REF}
    RAPGEF5_ANSWER = {
        "findings": [finding(S6_ROW_GENE, *S6_ROW_NAMES, reason=WHY_RAPGEF5)]
    }

    def test_an_answer_that_is_not_the_fold_of_its_facts_is_refused(self):
        """ADR 0032 (a false claim is refused before append; an open re-validates):
        the answer is the fold of the reading's facts (owner, #383 A1b), so one that
        drops a finding they hold is refused on append and on reopen."""
        from bio_neb.paper_reading import reading as reading_module
        from bio_neb.paper_reading.journey import settled_answer

        def forged(reading):
            honest = settled_answer(reading)
            return honest.model_copy(
                update={
                    "answer": honest.answer.model_copy(update={"findings": ()}),
                    "finding_event_ids": (),
                }
            )

        with mock.patch.object(reading_module, "settled_answer", forged):
            with self.assertRaisesRegex(ValueError, "fold of its settled items"):
                self.read_s17(self.RAPGEF5_PLAN, self.RAPGEF5_ANSWER)
        [path] = self.root.glob("run-0/*/*.jsonl")
        refused = Journey.open(path)
        self.assertNotIn(
            "PaperAnswerRecorded", {e["event_type"] for e in refused.read()}
        )
        [view] = refused.checked_view("paper_readings").readings
        self.assertEqual((view.all_settled, view.answer_event_id), (True, None))

        # On reopen: an honest Journey whose recorded answer is rewritten the same way.
        honest = self.read_s17(self.RAPGEF5_PLAN, self.RAPGEF5_ANSWER)
        path = Path(honest.journey_file)
        lines = path.read_bytes().splitlines(keepends=True)
        [number] = [
            n for n, line in enumerate(lines) if b'"PaperAnswerRecorded"' in line
        ]
        event = json.loads(lines[number])
        event["payload"]["answer"]["findings"] = []
        event["payload"]["finding_event_ids"] = []
        lines[number] = canonical_json(event).encode() + b"\n"
        path.write_bytes(b"".join(lines))
        with self.assertRaisesRegex(Exception, "fold of its settled items"):
            Journey.open(path)

    def test_the_reading_view_steps_as_it_folds_and_as_a_fresh_open_reads_it(self):
        # ADR 0032 (tests/test_journey_family_steps.py's rule for a stepped family):
        # the writer folds each reading fact into its held view, equal to a full fold
        # of the prefix at every point and to a fresh open's.
        journey, paper_event_id = stored_journey(self.root / "run", s17_like())
        answer_question(
            journey,
            paper_event_id=paper_event_id,
            question="What is the most meaningful finding?",
            runtime=RuntimeBinding(
                read_runtime_config(self.config).runtime_selections(), paper_event_id
            ),
            environ={"LOCAL_MODEL_KEY": "offline"},
            model_factory=reading(
                [("propose_reading_areas", self.RAPGEF5_PLAN)],
                [
                    (
                        "answer_reading_item",
                        {
                            "findings": [
                                *self.RAPGEF5_ANSWER["findings"],
                                finding(value("gene", "X", "main:p99"), reason="x"),
                            ]
                        },
                    )
                ],
                [("answer_reading_item", {"findings": []})],
            ),
        )
        events = journey.read()
        [family] = [
            f
            for f in semantic_families(frozenset(e["event_type"] for e in events))
            if f.name == "paper_readings"
        ]
        state = family.admit([])
        stepped = 0
        for index, event in enumerate(events):
            if event["event_type"] in family.payloads:
                state = family.admit_next(state, event, events[:index])
                stepped += 1
            self.assertEqual(
                state, family.admit(events[: index + 1]), event["event_type"]
            )
        # The opening, the plan, two findings, two completions and the answer.
        self.assertEqual(stepped, 7)
        self.assertEqual(
            journey.checked_view("paper_readings"),
            Journey.open(journey.path).checked_view("paper_readings"),
        )
        self.assertEqual(len(state.readings[0].findings), 2)

    def test_a_supplied_paper_with_no_ids_gets_its_targets_by_its_paper_id(self):
        # A paper the researcher supplied with no DOI, PMID or PMCID is named by its
        # paper_id and file name, so its targets are built and placement reads them as
        # its record.
        from bio_neb.target_request.paper_targets import PaperTargets

        paper = Paper(None, list(s17_like().documents.values()))
        outcome = ask(
            self.root,
            paper,
            "What is the most meaningful finding?",
            reading(
                [("propose_reading_areas", self.RAPGEF5_PLAN)],
                [("answer_reading_item", self.RAPGEF5_ANSWER)],
            ),
            config=self.config,
        )

        [recorded] = [
            e
            for e in Journey.open(Path(outcome.journey_file)).read()
            if e["event_type"] == "PaperRecorded"
        ]
        paper_id = recorded["payload"]["paper_id"]
        targets = outcome.recorded.targets
        self.assertEqual(
            (targets.paper.paper_id, targets.paper.file_name, targets.paper.doi),
            (paper_id, "supplied.txt", None),
        )
        self.assertEqual(targets.asserted_by, f"paper:paper_id:{paper_id}")
        [target] = targets.targets
        self.assertEqual(target.change.gene, "RAPGEF5")
        self.assertEqual(
            PaperTargets.model_validate_json(targets.model_dump_json()), targets
        )

    def test_the_question_reaches_the_reading_as_the_caller_wrote_it(self):
        # The question follows the researcher's words; nothing in code narrows it to
        # the most significant finding or widens it (ADR 0049: code copies, the model
        # decides which findings the question needs).
        question = "List every finding, the minor ones too, with the reference."
        seen = []
        outcome = ask(
            self.root,
            s17_like(),
            question,
            reading(
                [("propose_reading_areas", self.RAPGEF5_PLAN)],
                [("answer_reading_item", self.RAPGEF5_ANSWER)],
                seen=seen,
            ),
            config=self.config,
        )

        [opening] = [
            e
            for e in Journey.open(Path(outcome.journey_file)).read()
            if e["event_type"] == "PaperReadingStarted"
        ]
        messages = [json.loads(request[0].parts[0].content) for request in seen]
        self.assertEqual(
            [
                opening["payload"]["question"],
                outcome.items[0].item.subject,
                *(m["question"] for m in messages),
                outcome.answer.question,
                outcome.recorded.targets.question,
            ],
            [question] * 6,
        )

    def test_the_organism_is_kept_only_where_its_span_writes_it(self):
        # The paper item names the organism as the paper writes it, by the span that
        # writes it; code keeps it only there, and refuses it elsewhere.
        seen = []
        model = reading(
            [
                (
                    "propose_reading_areas",
                    {
                        "reference": None,
                        "organism": {"written": "horses", "span": "main:p1"},
                        "areas": [],
                        "not_answered": NO_REF,
                    },
                )
            ],
            [
                (
                    "propose_reading_areas",
                    {
                        "reference": None,
                        "organism": {"written": "Thoroughbred", "span": "main:p1"},
                        "areas": [],
                        "not_answered": NO_REF,
                    },
                )
            ],
            seen=seen,
        )

        outcome = ask(
            self.root,
            s17_like(),
            "Which organism is studied?",
            model,
            config=self.config,
        )

        [repair] = repairs(seen[1])
        self.assertIn("organism_not_written:horses in main:p1", repair)
        organism = outcome.answer.organism
        self.assertEqual((organism.written, organism.span), ("Thoroughbred", "main:p1"))
        self.assertEqual(
            MAIN[organism.source.start : organism.source.end], "Thoroughbred"
        )
        self.assertEqual(organism.passage.text, TITLE.rstrip("\n"))
        self.assertEqual(outcome.recorded.answer.organism, organism)
        self.assertEqual(verify(outcome.answer, s17_like()).problems, ())
        # A copied organism that no longer stands in its span fails verification.
        moved = outcome.answer.model_copy(
            update={
                "organism": organism.model_copy(
                    update={"source": organism.passage.model_copy(update={"text": "x"})}
                )
            }
        )
        self.assertIn(
            "organism_not_written:organism:Thoroughbred",
            verify(moved, s17_like()).problems,
        )

    def test_every_roles_output_refusals_take_one_form_from_its_output_type(self):
        # The output session derives its validation from the output type, so every
        # role records one error form: location, then type.
        from bio_neb.agentic_journey.structured_output import (
            normalized_output,
            output_errors,
        )
        from bio_neb.investigation.investigation_agent import INVESTIGATION_OUTPUT
        from bio_neb.investigation.vcf_table_reading import TABLE_READING_OUTPUT

        self.assertEqual(
            output_errors(ReadingPlan, {}), ("reference:missing", "areas:missing")
        )
        self.assertEqual(
            output_errors(ItemAnswer, {"findings": "x"}), ("findings:tuple_type",)
        )
        for output in (INVESTIGATION_OUTPUT, TABLE_READING_OUTPUT):
            with self.subTest(output=output.name):
                errors = output_errors(output.output_type, {})
                self.assertTrue(errors)
                self.assertTrue(all(re.fullmatch(r"[\w.]+:\w+|\w+", e) for e in errors))
        plan = {"reference": None, "areas": []}
        self.assertEqual(output_errors(ReadingPlan, plan), ())
        self.assertEqual(
            normalized_output(ReadingPlan, plan),
            {"reference": None, "organism": None, "areas": [], "not_answered": None},
        )

    def test_the_repeat_identity_drops_only_a_rows_result_shaping_arguments(self):
        # Through the shared admission: a read asked again for more or less of its
        # result is the same read; every other row's identity is the call as called.
        from bio_neb.agentic_journey.action_policy import (
            ProposedAgentAction,
            call_identity,
        )
        from bio_neb.agentic_journey.assignment_context import (
            assignment_repeat_identity,
        )

        def call(tool, **arguments):
            return ProposedAgentAction(tool, "c", arguments)

        self.assertEqual(
            assignment_repeat_identity(call("grep_paper", pattern="x", max_hits=5)),
            assignment_repeat_identity(call("grep_paper", pattern="x")),
        )
        self.assertNotEqual(
            assignment_repeat_identity(call("read_paper", at="main", offset=0)),
            assignment_repeat_identity(call("read_paper", at="main", offset=100)),
        )
        other = call("resolve_taxon", term="Homo sapiens")
        self.assertEqual(assignment_repeat_identity(other), call_identity(other))
        self.assertFalse(
            [
                row.tool_name
                for row in function_tool_contracts()
                # The reading Tools, which the Investigation Planner shares.
                if row.result_shaping and row.tool_name not in READING_TOOLS
            ]
        )
