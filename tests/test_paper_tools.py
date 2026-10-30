"""A paper the researcher's words cite is found and retrieved by the planning Agent (#420).

The scripted model types a Europe PMC query with ``find_paper`` and names one row of
the page with ``retrieve_paper``, from the plan stage on, before it records the plan
(owner redirect, 2026-10-29: "it need to be available from the start"), or in the
paper stage for a ``cited_paper`` the plan declares and has not settled (owner, #420
D1 b); the coordinator, the Tools, the product adapter and
its transport run, and only the opener answers, from recorded-style answers modelled
on `tests/test_paper_sources.py`'s ``hosts``. Europe PMC's options pass verbatim, every
row is copied in its order (ADR 0049: no ranking, filter or cut on a source's answer),
the model's row is checked only by membership, and a dead end asks the paper Review
(owner, #420 P1).
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
import urllib.error
import urllib.parse
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from bio_neb.agentic_journey.action_policy import ProposedAgentAction
from bio_neb.application.advancement import (
    AnswerReview,
    PaperReviewAnswer,
    ResumeRun,
    advance,
    use_run_advancement_composition,
    use_run_advancement_dependencies,
)
from bio_neb.application.queries import ReviewJourneyQuery, review_journey
from bio_neb.failure import ReviewAnswer
from bio_neb.journey import Journey
from bio_neb.paper_intake.tools import paper_tool_errors
from bio_neb.providers.paper_sources import PaperSourceAdapter
from bio_neb.recording import read_evidence
from tests.application_fakes import _registered
from tests.planning_assignment_fakes import PlanningComposition

REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
# A fictional paper: no real paper is named here or in what a model reads.
DOI = "10.5555/fictional.2024.001"
PMCID = "PMC9000001"
CITATION = f"the variant reported in doi:{DOI}"
PROMPT = f"Analyze rs1234 for sample-001 Homo sapiens and produce VCF, {CITATION}."
QUERY = f'DOI:"{DOI}"'
PAPER_ROW = {
    "id": "39000001",
    "source": "MED",
    "pmid": "39000001",
    "pmcid": PMCID,
    "doi": DOI,
    "title": "A fictional variant study",
    "authorString": "Doe J, Roe R.",
    "journalTitle": "J Fict Genet",
    "pubYear": "2024",
    "isOpenAccess": "Y",
}
# Listed first by Europe PMC: code must not move the DOI's row ahead of it.
OTHER_ROW = {
    "id": "PPR100",
    "source": "PPR",
    "doi": "10.5555/other",
    "title": "A preprint",
    "authorString": "Poe A.",
    "pubYear": "2023",
    "isOpenAccess": "N",
}
ARTICLE = (
    b'<article xmlns:xlink="http://www.w3.org/1999/xlink"><body><p>rs1234 is '
    b"reported.</p></body><back><supplementary-material>"
    b'<media xlink:href="table1.csv"/></supplementary-material></back></article>'
)
TABLE = b"rsid,gene\nrs1234,GENE1\n"


def _search_url(query, page_size=25, cursor_mark="*", sort=None):
    parameters = {
        "query": query,
        "format": "json",
        "resultType": "lite",
        "pageSize": str(page_size),
        "cursorMark": cursor_mark,
        **({"sort": sort} if sort else {}),
    }
    return f"{REST}/search?{urllib.parse.urlencode(parameters)}"


def _page(*rows, hit_count=None, cursor="AoE1"):
    return json.dumps(
        {
            "hitCount": len(rows) if hit_count is None else hit_count,
            "nextCursorMark": cursor,
            "resultList": {"result": list(rows)},
        }
    ).encode()


def _listing(*keys):
    contents = "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"{contents}</ListBucketResult>"
    ).encode()


SEARCH = _search_url(QUERY, page_size=10, sort="CITED desc")
RESOLVE = _search_url(f"PMCID:{PMCID}")
LISTING = f"{BUCKET}/?list-type=2&prefix={PMCID}."
JATS = f"{BUCKET}/{PMCID}.1/{PMCID}.1.xml"
TABLE_URL = f"{BUCKET}/{PMCID}.1/table1.csv"


def _answers(**changes):
    """The hosts' answers for the cited paper's search and download."""
    answers = {
        SEARCH: (200, _page(OTHER_ROW, PAPER_ROW, hit_count=7), "application/json"),
        RESOLVE: (200, _page(PAPER_ROW), "application/json"),
        LISTING: (
            200,
            _listing(f"{PMCID}.1/{PMCID}.1.xml", f"{PMCID}.1/table1.csv"),
            "application/xml",
        ),
        JATS: (200, ARTICLE, "binary/octet-stream"),
        TABLE_URL: (200, TABLE, "binary/octet-stream"),
    }
    answers.update(changes)
    return answers


def _hosts(answers):
    """An adapter over recorded answers; a URL never recorded answers 404."""
    asked: list[str] = []

    def opener(request, timeout):
        url = request.full_url
        asked.append(url)
        status, body, kind = answers.get(url, (404, b"", "text/plain"))
        headers = Message()
        headers["Content-Type"] = kind
        if status >= 400:
            raise urllib.error.HTTPError(url, status, "", headers, io.BytesIO(body))
        response = MagicMock(status=status, headers=headers)
        response.__enter__.return_value = response
        response.read.side_effect = lambda size=-1: body if size < 0 else body[:size]
        return response

    return PaperSourceAdapter(opener=opener), asked


KEY = "cited_study"


class _CitingModel(PlanningComposition):
    """The scripted planning model, which also reads the cited paper: each step runs
    when the current planning stage is the step's own, before the scripted planning
    work, and the plan declares ``cited`` as its cited_paper requirements."""

    def __init__(self, receipt, paths, *, steps, cited):
        super().__init__(receipt, paths)
        self.steps = list(steps)
        self.cited_papers = cited

    async def respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        from bio_neb.agentic_journey.planning_assignment import planning_stage

        journey = Journey.open(Path(self.journey.path))
        stage = planning_stage(SimpleNamespace(journey=journey))
        if self.calls and self.steps and self.steps[0][0] == stage:
            name, arguments = self.steps.pop(0)[1](journey)
            self.calls.append(name)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name, arguments, tool_call_id=f"assignment-{len(self.calls)}"
                    )
                ]
            )
        return await super().respond(messages, info)


def _search(journey, *, query=QUERY, page_size=10, sort="CITED desc"):
    return "find_paper", {
        "subject_key": KEY,
        "query": query,
        "quote": CITATION,
        "page_size": page_size,
        "sort": sort,
    }


def _retrieve(candidate_id="MED:39000001"):
    def step(journey):
        search = [e for e in journey.read() if e["event_type"] == "PaperSearchRecorded"]
        return "retrieve_paper", {
            "search_event_id": search[-1]["event_id"],
            "candidate_id": candidate_id,
            "quote": CITATION,
            "reason": "The row's DOI is the one the researcher wrote.",
        }

    return step


def _cited(*, quote=CITATION, retrieved=False):
    """The plan's cited_paper requirement: the researcher's words that cite the
    paper, and its PaperRecorded when it was retrieved before the plan."""

    def requirements(journey):
        papers = [
            e["event_id"] for e in journey.read() if e["event_type"] == "PaperRecorded"
        ]
        return [
            {
                "kind": "cited_paper",
                "subject_key": KEY,
                "domain": "scientific",
                "meaning": "The paper the researcher cites for the variant.",
                "knowledge": {
                    "state": "asserted",
                    "quote": quote,
                    **({"evidence_event_ids": papers} if retrieved else {}),
                },
                "acquisition_route": "validate_with_tool",
                "validation": {
                    "kind": "tool_evidence",
                    "requirement": "the cited paper retrieved",
                },
            }
        ]

    return requirements


# Each step runs in its stage: the plan stage (before the plan) or the paper stage.
BEFORE_PLAN = "proposal_requirements"
PAPER = "paper"


def _stages(events):
    """The planning stage of each recorded model request, in order."""
    return [
        e["payload"]["stage"]
        for e in events
        if e["event_type"] == "ModelRequestRecorded"
    ]


def _of(events, event_type):
    return [e for e in events if e["event_type"] == event_type]


def _unread():
    """A scripted reader for a recorded paper these tests do not read (#383 A2 reads
    every paper a plan cites): its paper item states no reference, organism or area,
    so planning goes on as with the researcher's words alone."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    from tests.model_endpoint_fakes import ScriptedModelEndpoint

    plan = {
        "reference": None,
        "organism": None,
        "areas": [],
        "not_answered": "These tests do not read the paper.",
    }

    async def respond(messages, info):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "propose_reading_areas", plan, tool_call_id=f"read-{len(messages)}"
                )
            ]
        )

    return ScriptedModelEndpoint(respond)


class _Run:
    """One planning Run over a temporary root, with its composition and hosts, and a
    reader that reads nothing of a cited paper (``_unread``)."""

    def __init__(self, *, steps=(), answers=None, prompt=PROMPT, cited=None):
        self.root = Path(tempfile.mkdtemp())
        self.receipt, self.paths = _registered(self.root, prompt_text=prompt)
        self.model = _CitingModel(self.receipt, self.paths, steps=steps, cited=cited)
        self.model.paper_adapter, self.asked = _hosts(
            _answers() if answers is None else answers
        )
        with (
            use_run_advancement_composition(self.model),
            use_run_advancement_dependencies(paper_reading_model_factory=_unread()),
        ):
            self.result = advance(ResumeRun(journey_file=self.receipt.journey_file))
        self.journey = Journey.open(Path(self.receipt.journey_file))
        self.events = read_evidence(self.journey)

    def close(self):
        shutil.rmtree(self.root)

    def answer(self, action, **more):
        request = self.result.current_review.request
        with (
            use_run_advancement_composition(self.model),
            use_run_advancement_dependencies(paper_reading_model_factory=_unread()),
        ):
            self.result = advance(
                AnswerReview(
                    journey_file=self.receipt.journey_file,
                    request_event_id=request.request_event_id,
                    answer=PaperReviewAnswer(
                        response=ReviewAnswer(
                            request_id=request.request.request_id,
                            action=action,
                            by="researcher",
                            **more,
                        )
                    ),
                )
            )
        self.journey = Journey.open(Path(self.receipt.journey_file))
        self.events = read_evidence(self.journey)

    def journal_evidence_ids(self):
        """Every evidence event id the Run's Pass contact journals name."""
        found = set()

        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "evidence_event_id" and isinstance(item, str):
                        found.add(item)
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        for journal in self.journey.run_directory.glob("contacts/*/journal.jsonl"):
            for line in journal.read_text().splitlines():
                walk(json.loads(line))
        return found


class EuropePmcSearchTests(unittest.TestCase):
    """The search port: Europe PMC's answer about the query, of any status with a
    body, reaches the model; only a status that is no answer stops (#458; ADR 0049:
    no status has a fixed meaning)."""

    def test_an_answered_status_is_returned_with_its_body_collected(self):
        from bio_neb.agentic_journey.source_answers import collected_answers
        from bio_neb.providers.paper_sources import search_europe_pmc

        refused = b'{"errMsg": "Syntax error in query"}'
        url = _search_url("TITLE:(unclosed")
        adapter, asked = _hosts({url: (400, refused, "application/json")})

        with collected_answers("find_paper") as got:
            status, page = search_europe_pmc(adapter, "TITLE:(unclosed")

        self.assertEqual((status, page), (400, None))
        self.assertEqual(asked, [url])
        self.assertEqual(
            [(a.request_identity, a.status_code, a.body) for a in got],
            [(url, 400, refused)],
        )


class CitedPaperRunTests(unittest.TestCase):
    """A prompt that cites a paper by DOI: the paper is found, retrieved and recorded
    once in the plan stage, before the plan (owner redirect, 2026-10-29: "it need to
    be available from the start"); the plan cites it, the paper stage is passed, and
    the Run goes on as it does without a paper."""

    @classmethod
    def setUpClass(cls):
        cls.cited = _Run(
            steps=((BEFORE_PLAN, _search), (BEFORE_PLAN, _retrieve())),
            cited=_cited(retrieved=True),
        )
        cls.plain = _Run(
            prompt="Analyze rs1234 for sample-001 Homo sapiens and produce VCF."
        )

    @classmethod
    def tearDownClass(cls):
        cls.cited.close()
        cls.plain.close()

    def test_the_search_sends_the_models_options_and_copies_every_row_in_order(self):
        """Europe PMC's options pass verbatim, and every row comes back as Europe PMC
        lists it, with its hitCount and nextCursorMark (owner, #420 F2 b; ADR 0049)."""
        self.assertEqual(self.cited.asked[0], SEARCH)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(SEARCH).query)
        self.assertEqual(
            {key: value[0] for key, value in query.items()},
            {
                "query": QUERY,
                "format": "json",
                "resultType": "lite",
                "pageSize": "10",
                "cursorMark": "*",
                "sort": "CITED desc",
            },
        )
        (fact,) = _of(self.cited.events, "PaperSearchRecorded")
        result = fact["payload"]["result"]
        self.assertEqual((result["hit_count"], result["next_cursor_mark"]), (7, "AoE1"))
        self.assertEqual(
            [
                (r["candidate_id"], r["doi"], r["pmcid"], r["title"], r["authors"])
                for r in result["rows"]
            ],
            [
                ("PPR:PPR100", "10.5555/other", None, "A preprint", "Poe A."),
                (
                    "MED:39000001",
                    DOI,
                    PMCID,
                    "A fictional variant study",
                    "Doe J, Roe R.",
                ),
            ],
        )
        # The GET's own bytes are retained on its record, which the fact follows.
        (answer,) = fact["payload"]["source_answers"]
        self.assertEqual(
            (answer["operation"], answer["request_identity"], answer["status_code"]),
            ("find_paper", SEARCH, 200),
        )
        (got,) = [
            e
            for e in _of(self.cited.events, "ReferenceSourceRetrieved")
            if e["payload"]["operation"] == "find_paper"
        ]
        self.assertEqual(fact["parent_ids"], [got["event_id"]])
        self.assertEqual(
            self.cited.journey.read_artifact(got["artifact_references"][0]),
            _answers()[SEARCH][1],
        )

    def test_the_retrieved_row_is_recorded_once_with_its_copied_ids_and_contacts(self):
        (paper,) = _of(self.cited.events, "PaperRecorded")
        payload = paper["payload"]
        self.assertEqual(
            (payload["origin"], payload["reference"], payload["quote"]),
            ("citation", "MED:39000001", CITATION),
        )
        self.assertEqual(
            payload["ids"],
            {
                "doi": DOI,
                "pmid": "39000001",
                "pmcid": PMCID,
                "title": "A fictional variant study",
                "authors": "Doe J, Roe R.",
                "journal": "J Fict Genet",
                "year": "2024",
                "open_access": True,
            },
        )
        self.assertEqual(
            [d["document_id"] for d in payload["documents"]],
            ["main", "supp:table1.csv"],
        )
        self.assertEqual(
            [
                e["payload"]["location"]
                for e in _of(self.cited.events, "PaperDocumentRetained")
            ],
            [JATS, TABLE_URL],
        )
        # The download's four GETs, each a physical contact the Pass journal names.
        self.assertEqual(self.cited.asked[1:], [RESOLVE, LISTING, JATS, TABLE_URL])
        contacts = {
            e["event_id"]: e["payload"]["request_identity"]
            for e in _of(self.cited.events, "SourcePhysicalContactStarted")
        }
        self.assertEqual(
            [contacts[i] for i in payload["contact_event_ids"]],
            [RESOLVE, LISTING, JATS, TABLE_URL],
        )
        self.assertLessEqual(set(contacts), self.cited.journal_evidence_ids())
        (retrieval,) = _of(self.cited.events, "PaperRetrievalRecorded")
        self.assertEqual(
            (retrieval["payload"]["status"], retrieval["payload"]["paper_event_id"]),
            ("recorded", paper["event_id"]),
        )

    def test_a_run_citing_a_paper_goes_on_as_one_without_it(self):
        """The two paper calls are the only extra model turns, and a Run that cites
        no paper makes no paper call, fact or contact."""
        self.assertEqual(self.cited.result.status, self.plain.result.status)
        self.assertEqual(
            [
                c
                for c in self.cited.model.calls
                if c not in {"find_paper", "retrieve_paper"}
            ],
            self.plain.model.calls,
        )
        self.assertEqual(
            self.cited.model.calls.count("find_paper")
            + self.cited.model.calls.count("retrieve_paper"),
            2,
        )
        self.assertEqual(self.plain.asked, [])
        self.assertFalse(
            [e for e in self.plain.events if e["event_type"].startswith("Paper")]
        )

    def test_the_plan_cites_the_paper_it_retrieved_and_the_paper_stage_is_passed(self):
        """The search and the retrieval are the plan stage's requests; the plan's
        cited_paper cites the PaperRecorded, so the paper stage is passed: the paper
        is read (#383 A2) and the reference work follows. A Run that cites no paper
        never enters the paper stage (#420 D1 b)."""
        (paper,) = _of(self.cited.events, "PaperRecorded")
        (plan,) = _of(self.cited.events, "ProposalRequirementsIdentified")
        self.assertLess(self.cited.events.index(paper), self.cited.events.index(plan))
        (cited,) = [
            r
            for r in plan["payload"]["plan"]["requirements"]
            if r["kind"] == "cited_paper"
        ]
        self.assertEqual((cited["subject_key"], cited["value"]), (KEY, CITATION))
        self.assertIn(
            {"source_kind": "planning_context", "source_event_id": paper["event_id"]},
            cited["evidence"],
        )
        stages = _stages(self.cited.events)
        self.assertNotIn(PAPER, stages)
        self.assertEqual(
            list(dict.fromkeys(stages))[:3],
            ["proposal_requirements", "paper_reading", "reference"],
        )
        self.assertNotIn(PAPER, _stages(self.plain.events))
        self.assertFalse(
            [
                r
                for r in _of(self.plain.events, "ProposalRequirementsIdentified")[0][
                    "payload"
                ]["plan"]["requirements"]
                if r["kind"] == "cited_paper"
            ]
        )

    def test_the_paper_tools_are_not_the_reference_stages(self):
        """The paper Tools are the plan and paper stages' only (owner, #420 D1 b)."""
        from bio_neb.agentic_journey.tool_execution import stage_tool_names

        for stage, offered in (
            ("proposal_requirements", True),
            ("paper", True),
            ("reference", False),
            ("companions", False),
        ):
            with self.subTest(stage=stage):
                tools = stage_tool_names("planning_agent", stage)
                self.assertEqual(
                    {"find_paper", "retrieve_paper"} <= tools, offered, tools
                )
        self.assertEqual(
            {"find_paper", "retrieve_paper", "request_planning_hitl"},
            stage_tool_names("planning_agent", "paper"),
        )

    def test_a_cited_paper_quote_outside_the_researchers_words_is_refused(self):
        """ADR 0049: a cited_paper's quote is checked only by membership in the
        researcher's words, and its evidence only by membership in the Run's
        recorded citation papers; a value the paper will supply may be to_find
        citing that paper (owner redirect, 2026-10-29)."""
        from bio_neb.agentic_journey.planning_assignment import (
            assignment_action_errors,
        )

        (plan,) = _of(self.cited.events, "ProposalRequirementsIdentified")
        (paper,) = _of(self.cited.events, "PaperRecorded")
        prefix = self.cited.events[: self.cited.events.index(plan)]
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: prefix))
        (admitted,) = [
            e["payload"]["arguments"]
            for e in prefix
            if e["payload"].get("tool_name") == "record_proposal_requirements"
            and "arguments" in e["payload"]
        ]
        draft = admitted.get("draft", admitted)
        others = [r for r in draft["requirements"] if r["kind"] != "cited_paper"]
        (cited,) = [r for r in draft["requirements"] if r["kind"] == "cited_paper"]

        def errors(*requirements):
            return assignment_action_errors(
                deps,
                ProposedAgentAction(
                    "record_proposal_requirements",
                    "call",
                    {"requirements": [*requirements]},
                ),
            )

        self.assertEqual(errors(*others, cited), ())
        self.assertEqual(errors(*others), ())
        unwritten = {
            **cited,
            "knowledge": {**cited["knowledge"], "quote": "a paper nobody wrote"},
        }
        self.assertEqual(
            errors(*others, unwritten), (f"{KEY}:cited_paper_quote_ungrounded",)
        )
        elsewhere = {
            **cited,
            "knowledge": {
                **cited["knowledge"],
                "evidence_event_ids": [plan["payload"]["input_set_id"]],
            },
        }
        self.assertEqual(
            errors(*others, elsewhere), (f"{KEY}:cited_paper_evidence_ungrounded",)
        )
        # The reference the paper will give is to_find, citing the paper, with an
        # existing route: no question to the researcher is forced.
        from_paper = [
            {
                **r,
                "knowledge": {
                    "state": "to_find",
                    "evidence_event_ids": [paper["event_id"]],
                },
                "acquisition_route": "validate_with_tool",
                "validation": {
                    "kind": "tool_evidence",
                    "requirement": f"the assembly {KEY} names",
                },
            }
            if r["kind"] == "reference_assembly_identity"
            else r
            for r in others
        ]
        self.assertEqual(errors(*from_paper, cited), ())

    def test_a_row_not_on_the_cited_page_or_an_unwritten_quote_is_refused(self):
        """A model-named row is checked only by membership (ADR 0049): it is a row of
        the page it cites, and the quote stands in the researcher's words."""
        (search,) = _of(self.cited.events, "PaperSearchRecorded")
        prefix = self.cited.events[: self.cited.events.index(search) + 1]
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: prefix))

        def errors(**arguments):
            return paper_tool_errors(
                deps,
                ProposedAgentAction(
                    "retrieve_paper",
                    "call",
                    {
                        "search_event_id": search["event_id"],
                        "candidate_id": "MED:39000001",
                        "quote": CITATION,
                        "reason": "The DOI matches.",
                        **arguments,
                    },
                ),
            )

        self.assertEqual(errors(), ())
        self.assertEqual(
            errors(candidate_id="MED:11111111"), ("paper_not_on_search_page",)
        )
        self.assertEqual(
            errors(quote="a paper the researcher never cited"),
            ("paper_quote_ungrounded",),
        )


class PaperStageTests(unittest.TestCase):
    """A plan that declares a cited paper it has not retrieved opens the paper stage
    right after the plan: the paper is searched for, retrieved and recorded there,
    and the reference work follows (owner, #420 D1 b)."""

    @classmethod
    def setUpClass(cls):
        cls.staged = _Run(
            steps=((PAPER, _search), (PAPER, _retrieve())), cited=_cited()
        )

    @classmethod
    def tearDownClass(cls):
        cls.staged.close()

    def test_the_paper_stage_follows_the_plan_and_closes_on_the_recorded_paper(self):
        events = self.staged.events
        (plan,) = _of(events, "ProposalRequirementsIdentified")
        (search,) = _of(events, "PaperSearchRecorded")
        (paper,) = _of(events, "PaperRecorded")
        self.assertLess(events.index(plan), events.index(search))
        self.assertEqual(search["payload"]["tool_input"]["subject_key"], KEY)
        stages = _stages(events)
        self.assertEqual(
            list(dict.fromkeys(stages))[:4],
            ["proposal_requirements", "paper", "paper_reading", "reference"],
        )
        # The paper stage's two requests are its search and retrieval.
        self.assertEqual(stages.count(PAPER), 2)
        self.assertEqual(
            [
                c
                for c in self.staged.model.calls
                if c in {"find_paper", "retrieve_paper"}
            ],
            ["find_paper", "retrieve_paper"],
        )
        (retrieval,) = _of(events, "PaperRetrievalRecorded")
        self.assertEqual(retrieval["payload"]["paper_event_id"], paper["event_id"])

    def test_the_paper_stage_findings_show_each_cited_paper_and_its_state(self):
        from bio_neb.agentic_journey.planning_assignment import _stage_facts
        from bio_neb.application.planning_read import current_planning

        events = self.staged.events
        (retrieval,) = _of(events, "PaperRetrievalRecorded")
        prefix = events[: events.index(retrieval)]
        journey = SimpleNamespace(read=lambda: prefix)
        view = current_planning(journey)
        (row,) = [
            f for f in _stage_facts("paper", view, prefix) if f.kind == "cited_paper"
        ]
        self.assertEqual(
            (row.subject_key, row.status, row.value), (KEY, "to_find", CITATION)
        )
        view = current_planning(SimpleNamespace(read=lambda: events))
        (row,) = [
            f for f in _stage_facts("paper", view, events) if f.kind == "cited_paper"
        ]
        self.assertEqual(row.status, "recorded")

    def test_a_search_after_the_plan_names_one_of_its_cited_papers(self):
        """Reopen checks the binding by membership (#420 D1 b): a recorded search
        whose subject_key is no cited_paper of the recorded plan is refused."""
        from bio_neb.paper_intake.journey import project_run_papers

        events = [dict(e) for e in self.staged.events]
        (index,) = [
            i for i, e in enumerate(events) if e["event_type"] == "PaperSearchRecorded"
        ]
        payload = dict(events[index]["payload"])
        payload["tool_input"] = {**payload["tool_input"], "subject_key": "other_paper"}
        events[index] = {**events[index], "payload": payload}
        with self.assertRaisesRegex(ValueError, "names no cited paper of the plan"):
            project_run_papers(events)


class ConfirmationTests(unittest.TestCase):
    """When a page lists several rows, the model asks the researcher before any
    download, through planning's question (owner, #420 F2 b), and the researcher's
    selection binds the row it may retrieve. A search before the plan is asked about
    in the paper stage, on its cited paper's own requirement (#420 D1 b)."""

    def test_the_question_presents_every_row_and_the_selection_binds_retrieval(self):
        from bio_neb.application.advancement import PlanningReviewAnswer
        from bio_neb.parabricks.planning_hitl import SelectPlanningHitlResponse

        run = _Run(
            steps=(
                (BEFORE_PLAN, _search),
                (PAPER, lambda journey: ("request_planning_hitl", {})),
            ),
            cited=_cited(),
        )
        self.addCleanup(run.close)

        self.assertEqual(run.result.status, "planning_hitl_pending")
        (search,) = _of(run.events, "PaperSearchRecorded")
        (asked,) = _of(run.events, "PlanningHitlRequested")
        request = asked["payload"]
        self.assertEqual(request["evidence_event_ids"], [search["event_id"]])
        (plan,) = _of(run.events, "ProposalRequirementsIdentified")
        (cited,) = [
            r
            for r in plan["payload"]["plan"]["requirements"]
            if r["kind"] == "cited_paper"
        ]
        self.assertEqual(request["requirement_id"], cited["requirement_id"])
        self.assertEqual(
            [c["candidate_id"] for c in request["candidates"]],
            ["paper:PPR:PPR100", "paper:MED:39000001"],
        )
        self.assertEqual(
            request["candidates"][1]["display_label"],
            f"A fictional variant study — Doe J, Roe R. — J Fict Genet — 2024 — {DOI}",
        )
        # Nothing is downloaded before the answer.
        self.assertEqual(run.asked, [SEARCH])
        self.assertEqual(_of(run.events, "PaperRecorded"), [])

        # The paper stage resumes on the answer and retrieves the selected row.
        run.model.steps = [(PAPER, _retrieve("PPR:PPR100"))]
        with use_run_advancement_composition(run.model):
            advance(
                AnswerReview(
                    journey_file=run.receipt.journey_file,
                    request_event_id=asked["event_id"],
                    answer=PlanningReviewAnswer(
                        kind="planning",
                        response=SelectPlanningHitlResponse(
                            action="select", candidate_id="paper:PPR:PPR100"
                        ),
                    ),
                )
            )
        events = read_evidence(Journey.open(Path(run.receipt.journey_file)))
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: events))

        def errors(candidate_id):
            return paper_tool_errors(
                deps,
                ProposedAgentAction(
                    "retrieve_paper",
                    "call",
                    {
                        "search_event_id": search["event_id"],
                        "candidate_id": candidate_id,
                        "quote": CITATION,
                        "reason": "The researcher selected it.",
                    },
                ),
            )

        self.assertEqual(errors("MED:39000001"), ("researcher_selected_other_paper",))
        self.assertEqual(errors("PPR:PPR100"), ())
        (retrieval,) = _of(events, "PaperRetrievalRecorded")
        self.assertEqual(
            retrieval["payload"]["tool_input"]["candidate_id"], "PPR:PPR100"
        )


class DeadEndTests(unittest.TestCase):
    """Each dead end of a cited paper records its refusal and asks the paper Review;
    the Attempt ends there (owner, #420 P1: retry, amend, continue or abort)."""

    def refused(self, answers):
        run = _Run(
            steps=((PAPER, _search), (PAPER, _retrieve())),
            answers=answers,
            cited=_cited(),
        )
        self.addCleanup(run.close)
        (refusal,) = _of(run.events, "PaperIntakeRefused")
        self.assertEqual(run.result.status, "paper_review_pending")
        self.assertEqual(run.result.current_review.kind, "paper")
        self.assertEqual(
            [a.action for a in run.result.current_review.legal_actions],
            ["retry", "amend", "continue", "abort"],
        )
        self.assertEqual(
            run.result.current_review.request.request.amendable,
            ("paper_reference", "paper_file"),
        )
        (checkpoint,) = [
            e
            for e in _of(run.events, "SliceCheckpointReached")
            if e["payload"]["checkpoint"] == "paper_review_requested"
        ]
        (review,) = _of(run.events, "PaperReviewRequested")
        self.assertEqual(
            checkpoint["payload"]["paper_review_request_event_id"], review["event_id"]
        )
        self.assertEqual(_of(run.events, "PaperRecorded"), [])
        # The model was not asked again after the Review opened.
        self.assertEqual(run.model.calls[-1], "retrieve_paper")
        return run, refusal["payload"]

    def test_a_dead_end_before_the_plan_ends_the_attempt_at_the_review(self):
        """Live #420 naive a2: the model finds and retrieves the cited paper before it
        records a plan; a stopped download there asks the paper Review, and the
        Attempt ends at it with no plan yet, and the Run reopens."""
        run = _Run(
            steps=((BEFORE_PLAN, _search), (BEFORE_PLAN, _retrieve())),
            answers=_answers(**{LISTING: (403, b"Forbidden", "text/plain")}),
        )
        self.addCleanup(run.close)

        self.assertEqual(_of(run.events, "ProposalRequirementsIdentified"), [])
        self.assertEqual(run.result.status, "paper_review_pending")
        (checkpoint,) = [
            e
            for e in _of(run.events, "SliceCheckpointReached")
            if e["payload"]["checkpoint"] == "paper_review_requested"
        ]
        self.assertIsNone(checkpoint["payload"]["proposal_plan_event_id"])
        self.assertEqual(
            review_journey(
                ReviewJourneyQuery(journey_file=run.receipt.journey_file)
            ).status,
            "paper_review_pending",
        )

    def test_a_paper_with_no_pmcid_has_no_open_copy(self):
        no_pmcid = {k: v for k, v in PAPER_ROW.items() if k != "pmcid"}
        _, refusal = self.refused(
            _answers(
                **{
                    SEARCH: (200, _page(no_pmcid), "application/json"),
                    _search_url(QUERY): (200, _page(no_pmcid), "application/json"),
                }
            )
        )
        self.assertEqual(refusal["code"], "paper_not_open_access")

    def test_a_paper_whose_hosts_hold_no_article_is_refused(self):
        _, refusal = self.refused(
            _answers(**{LISTING: (200, _listing(), "application/xml")})
        )
        self.assertEqual(refusal["code"], "paper_article_missing")

    def test_a_row_europe_pmc_then_names_no_paper_for_is_not_found(self):
        _, refusal = self.refused(
            _answers(**{RESOLVE: (200, _page(), "application/json")})
        )
        self.assertEqual(refusal["code"], "paper_not_found")

    def test_a_stopped_download_closes_its_scope_before_the_review_is_asked(self):
        """The journal admits one failure chain per scope: the stop sends nothing
        more, and the scope finishes before the refusal is recorded."""
        run, refusal = self.refused(
            _answers(**{LISTING: (403, b"Forbidden", "text/plain")})
        )
        self.assertEqual(refusal["code"], "paper_download_stopped")
        self.assertIn("pmc_open_data_listing", refusal["detail"])
        self.assertEqual(run.asked[1:], [RESOLVE, LISTING])
        types = [e["event_type"] for e in run.events]
        finished = [
            i for i, t in enumerate(types) if t == "SourceContactScopeFinished"
        ][-1]
        self.assertLess(finished, types.index("PaperIntakeRefused"))
        (retrieval,) = _of(run.events, "PaperRetrievalRecorded")
        self.assertEqual(retrieval["payload"]["code"], "paper_download_stopped")

    def test_a_page_with_no_row_for_the_citation_names_none_and_asks_the_review(self):
        """#420 SP1: an unfound citation reaches P1's Review through the named
        choice none, with no download contact."""
        run = _Run(
            steps=((PAPER, _search), (PAPER, _retrieve("none"))),
            answers=_answers(**{SEARCH: (200, _page(hit_count=0), "application/json")}),
            cited=_cited(),
        )
        self.addCleanup(run.close)

        (refusal,) = _of(run.events, "PaperIntakeRefused")
        self.assertEqual(refusal["payload"]["code"], "paper_not_found")
        self.assertEqual(run.result.status, "paper_review_pending")
        self.assertEqual(run.asked, [SEARCH])
        (retrieval,) = _of(run.events, "PaperRetrievalRecorded")
        self.assertEqual(retrieval["payload"]["tool_input"]["reference"], None)

    def test_a_stop_after_the_article_keeps_the_paper_with_what_it_could_not_get(self):
        """#420 SP2 (owner P2): the supplement GET is refused after the article came;
        the paper is recorded with the article and the supplement unfetched, and no
        GET follows the stop."""
        run = _Run(
            steps=((PAPER, _search), (PAPER, _retrieve())),
            answers=_answers(**{TABLE_URL: (403, b"Forbidden", "text/plain")}),
            cited=_cited(),
        )
        self.addCleanup(run.close)

        self.assertEqual(_of(run.events, "PaperIntakeRefused"), [])
        (paper,) = _of(run.events, "PaperRecorded")
        self.assertEqual(
            [d["document_id"] for d in paper["payload"]["documents"]], ["main"]
        )
        (unfetched,) = paper["payload"]["unfetched"]
        self.assertEqual(unfetched["name"], "table1.csv")
        self.assertIn("failed", unfetched["reason"])
        self.assertEqual(run.asked[-1], TABLE_URL)

    def test_an_amended_reference_goes_back_to_planning_in_the_researchers_words(self):
        """ADR 0049 re-entry: the amended citation is the researcher's words, which
        the model types again with find_paper, and the quote stands in them."""
        run, _ = self.refused(_answers(**{RESOLVE: (200, _page(), "application/json")}))
        amended = "the paper with PMCID PMC9000002"
        run.model.steps = [
            (
                PAPER,
                lambda journey: (
                    "find_paper",
                    {
                        "subject_key": KEY,
                        "query": "PMCID:PMC9000002",
                        "quote": amended,
                    },
                ),
            )
        ]
        before = len(_stages(run.events))

        run.answer("amend", field="paper_reference", value=amended)

        (answer,) = _of(run.events, "PaperReviewAnswered")
        searches = _of(run.events, "PaperSearchRecorded")
        self.assertEqual(len(searches), 2)
        self.assertEqual(searches[-1]["payload"]["tool_input"]["quote"], amended)
        self.assertLess(run.events.index(answer), run.events.index(searches[-1]))
        # The amended reference sends the stage back.
        self.assertEqual(_stages(run.events)[before], PAPER)
        intake = next(
            i
            for i in run.result.run_papers.intakes
            if i.answer_event_id == answer["event_id"]
        )
        self.assertEqual(intake.state, "reference_due")

    def test_a_review_answered_continue_drops_the_paper_and_closes_the_stage(self):
        """Owner, #420 P1 and D1 b: continue drops the cited paper, the paper stage
        closes on that answer, and the reference work follows."""
        from bio_neb.application.planning_read import current_planning
        from bio_neb.paper_intake.tools import paper_stage_output

        run, _ = self.refused(_answers(**{RESOLVE: (200, _page(), "application/json")}))
        before = len(_stages(run.events))

        run.answer("continue")

        (answer,) = _of(run.events, "PaperReviewAnswered")
        self.assertEqual(_of(run.events, "PaperRecorded"), [])
        self.assertEqual(_stages(run.events)[before], "reference")
        self.assertNotIn(PAPER, _stages(run.events)[before:])
        output = paper_stage_output(
            current_planning(SimpleNamespace(read=lambda: run.events))
        )
        self.assertEqual(output.settled_event_ids, (answer["event_id"],))


class RequestFileCitationTests(unittest.TestCase):
    """#420 SP3: a citation in a registered request file is the researcher's words."""

    def test_a_quote_from_a_request_file_item_stands_and_another_does_not(self):
        from bio_neb.paper_intake.tools import paper_tool_errors

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        request = root / "request.json"
        request.write_text(
            json.dumps(
                {
                    "schema_version": "bio-neb.investigation-request/3",
                    "asserted_by": "researcher",
                    "requests": [
                        {"request": f"rs1234, {CITATION}", "target": {"rsid": "rs1234"}}
                    ],
                }
            )
        )
        receipt, _ = _registered(
            root,
            prompt_text="Analyze rs1234 for sample-001 Homo sapiens and produce VCF.",
            request_files=(request,),
        )
        deps = SimpleNamespace(journey=Journey.open(Path(receipt.journey_file)))

        def errors(quote):
            return paper_tool_errors(
                deps,
                SimpleNamespace(
                    tool_name="find_paper", arguments={"query": QUERY, "quote": quote}
                ),
            )

        self.assertEqual(errors(CITATION), ())
        self.assertEqual(errors("a paper nobody wrote"), ("paper_quote_ungrounded",))


if __name__ == "__main__":
    unittest.main()
