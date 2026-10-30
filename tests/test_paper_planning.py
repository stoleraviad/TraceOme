"""A cited paper is read in planning, and planning takes the paper's values from its
reading (#383 slice A2; owner readiness Q3 a, Q4 a, 2026-10-29).

The prompt is only a citation. The scripted planning model finds and retrieves the paper
(#420) and records a plan whose organism, reference and targets are ``to_find`` from the
recorded paper. The coordinator then reads the paper with the Paper Reader (ADR 0046),
each reading item its own Attempt under a scripted reader model, and planning cites what
the reading kept by membership only (ADR 0049): the taxon is resolved from the organism
the paper writes, the reference row is chosen from NCBI's copied rows quoting the version
the paper writes, and the paper's proposed targets are the Run's targets, so the
coordinator's target question (#342) does not open. The sample stays the researcher's
(owner Q4 a): it is the one question the paper cannot answer.

Only the model endpoints and the source adapters are scripted; the coordinator, the
planning and reading Tools, the paper intake, the reading facts and their folds run.
The paper is fictional: no real paper's values are here or in what a model reads.
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

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.application.advancement import (
    AnswerReview,
    PlanningReviewAnswer,
    ResumeRun,
    advance,
    use_run_advancement_composition,
    use_run_advancement_dependencies,
)
from bio_neb.journey import Journey
from bio_neb.parabricks.planning_hitl import ClarifyPlanningHitlResponse
from bio_neb.providers.paper_sources import PaperSourceAdapter
from bio_neb.recording import read_evidence
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.planning_assignment_fakes import PlanningComposition, configured_planning_run

REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
DOI = "10.5555/fictional.2024.383"
PMCID = "PMC9000383"
TITLE = "A fictional study of one sample"
CITATION = f'In the paper "{TITLE}" by Doe et al. (2024)'
GOAL = "find the most significant SNP and tell me whether my sample carries it."
PROMPT = f"{CITATION}, {GOAL}"
QUERY = f'TITLE:"{TITLE}" AND AUTH:Doe AND PUB_YEAR:2024'
PAPER_ROW = {
    "id": "39000383",
    "source": "MED",
    "pmid": "39000383",
    "pmcid": PMCID,
    "doi": DOI,
    "title": TITLE,
    "authorString": "Doe J, Roe R.",
    "journalTitle": "J Fict Genet",
    "pubYear": "2024",
    "isOpenAccess": "Y",
}
# The paper's own words: the organism, the version its positions are on, and its finding.
ORGANISM = "Homo sapiens"
VERSION = "GRCh38"
METHODS = f"Reads of the {ORGANISM} sample were aligned to the {VERSION} reference."
RESULTS = "The most significant variant was rs1234, carried by every affected sample."


def _article(methods=METHODS):
    return (
        '<article xmlns:xlink="http://www.w3.org/1999/xlink"><body>'
        f"<sec><title>Methods</title><p>{methods}</p></sec>"
        f"<sec><title>Results</title><p>{RESULTS}</p></sec>"
        "</body></article>"
    ).encode()


def _search_url(query, page_size=10):
    parameters = {
        "query": query,
        "format": "json",
        "resultType": "lite",
        "pageSize": str(page_size),
        "cursorMark": "*",
    }
    return f"{REST}/search?{urllib.parse.urlencode(parameters)}"


def _page(*rows):
    return json.dumps(
        {
            "hitCount": len(rows),
            "nextCursorMark": "AoE1",
            "resultList": {"result": list(rows)},
        }
    ).encode()


SEARCH = _search_url(QUERY)
RESOLVE = _search_url(f"PMCID:{PMCID}", page_size=25)
LISTING = f"{BUCKET}/?list-type=2&prefix={PMCID}."
JATS = f"{BUCKET}/{PMCID}.1/{PMCID}.1.xml"


def _hosts(article):
    """Europe PMC and the PMC bucket over recorded-style answers; any other URL 404s."""
    listing = (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<Contents><Key>{PMCID}.1/{PMCID}.1.xml</Key></Contents></ListBucketResult>"
    ).encode()
    answers = {
        SEARCH: (200, _page(PAPER_ROW), "application/json"),
        RESOLVE: (200, _page(PAPER_ROW), "application/json"),
        LISTING: (200, listing, "application/xml"),
        JATS: (200, article, "binary/octet-stream"),
    }
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


def _of(events, event_type):
    return [e for e in events if e["event_type"] == event_type]


def _paper_event(journey):
    return _of(journey.read(), "PaperRecorded")[0]["event_id"]


def _bare_citation_plan(journey, requirements):
    """The plan a model records for a bare citation (owner, #420 D1 b): the research
    goal and the citation quoted from the prompt; the sample to find from the
    researcher; the organism, the reference and the targets to find in the recorded
    paper, citing its PaperRecorded."""
    request = _of(journey.read(), "ResearchRequestRegistered")[0]["event_id"]
    recorded = _of(journey.read(), "PaperRecorded")
    # A plan recorded before its paper stage retrieves the paper (the live model's
    # order) cites the request for the values to find, and its citation names no
    # paper yet.
    paper = recorded[0]["event_id"] if recorded else None
    from_paper = {"state": "to_find", "evidence_event_ids": [paper or request]}
    files = {
        e["event_id"] for e in _of(journey.read(), "InvestigationRequestRegistered")
    }
    planned = []
    for item in requirements:
        kind = item["kind"]
        if set(item["knowledge"].get("evidence_event_ids", ())) & files:
            # A registered request file's own item stays as the shared plan cites
            # it, and the paper's targets are their own item beside it.
            planned.append(item)
            kind = "scientific_investigation"
        if kind == "research_goal":
            item = {**item, "knowledge": {"state": "asserted", "quote": GOAL}}
        elif kind == "sample_identity":
            item = {
                **item,
                "knowledge": {"state": "to_find", "evidence_event_ids": [request]},
                "acquisition_route": "ask_planning_hitl",
            }
        elif kind in {"taxon_identity", "reference_assembly_identity"}:
            item = {
                **item,
                "knowledge": from_paper,
                "acquisition_route": "validate_with_tool",
            }
        elif kind == "scientific_investigation":
            item = {
                **item,
                "subject_key": "most_significant_snp",
                "meaning": "The paper's most significant SNP.",
                "knowledge": from_paper,
                "acquisition_route": "validate_with_tool",
                "investigation_purpose": "literature_claim",
            }
        planned.append(item)
    planned.append(
        {
            "kind": "cited_paper",
            "subject_key": KEY,
            "domain": "scientific",
            "meaning": "The paper the researcher cites.",
            "knowledge": {
                "state": "asserted",
                "quote": CITATION,
                **({"evidence_event_ids": [paper]} if paper else {}),
            },
            "acquisition_route": "validate_with_tool",
            "validation": {
                "kind": "tool_evidence",
                "requirement": "the cited paper retrieved",
            },
        }
    )
    return planned


def _stage(journey):
    from bio_neb.agentic_journey.planning_assignment import planning_stage

    return planning_stage(SimpleNamespace(journey=journey))


class _CitationModel(PlanningComposition):
    """The scripted planning model: it finds and retrieves the cited paper before the
    plan, records the bare-citation plan, and in the reference stage takes the paper's
    values from the stage findings a reading fills, citing them; everything else is
    the shared scripted model's."""

    def __init__(self, receipt, paths):
        super().__init__(receipt, paths)
        self.plan_requirements = _bare_citation_plan
        self.taxon_terms = []
        self.choice_quotes = []
        # The accession the paper writes, which the model types as the supplied
        # reference citing the reading fact that keeps it; None chooses from NCBI's page.
        self.supplied_accession = None
        # True: the plan first, then the paper in the paper stage (the live order).
        self.plan_first = False

    def _call(self, name, arguments):
        self.calls.append(name)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    name, arguments, tool_call_id=f"assignment-{len(self.calls)}"
                )
            ]
        )

    async def respond(self, messages, info):
        from bio_neb.agentic_journey.planning_assignment import planning_stage_findings
        from tests.planning_assignment_fakes import eligible_tools

        journey = Journey.open(Path(self.journey.path))
        events = journey.read()
        search_stage = "paper" if self.plan_first else "proposal_requirements"
        if (self.calls or self.plan_first) and _stage(journey) == search_stage:
            if not _of(events, "PaperSearchRecorded"):
                return self._call(
                    "find_paper",
                    {
                        "subject_key": KEY,
                        "query": QUERY,
                        "quote": CITATION,
                        "page_size": 10,
                    },
                )
            if not _of(events, "PaperRecorded"):
                search = _of(events, "PaperSearchRecorded")[-1]["event_id"]
                return self._call(
                    "retrieve_paper",
                    {
                        "search_event_id": search,
                        "candidate_id": "MED:39000383",
                        "quote": CITATION,
                        "reason": "The row's title, author and year are the cited ones.",
                    },
                )
        tools = eligible_tools(self.journey, info.function_tools)
        findings = planning_stage_findings(SimpleNamespace(journey=journey))
        kept = {
            f.subject_key: f
            for f in findings.findings
            if f.kind == "ReadingPlanKept" and f.status == "kept"
        }
        from bio_neb.application.planning_read import read_planning

        view = read_planning(journey)
        if (
            "resolve_taxon" in tools
            and "organism" in kept
            and view.reference.resolved_taxon is None
        ):
            self.taxon_terms.append(kept["organism"].value)
            return self._call("resolve_taxon", {"term": kept["organism"].value})
        if (
            self.supplied_accession is not None
            and "record_supplied_reference" in tools
            and "genome_version" in kept
            and not _of(events, "ReferenceChoiceRecorded")
        ):
            return self._call(
                "record_supplied_reference",
                {
                    "assembly_accession": self.supplied_accession,
                    "assertion_event_id": kept["genome_version"].source_event_id,
                    "quote": self.supplied_accession,
                    "reason": "The paper names the exact assembly its reads are on.",
                },
            )
        if "record_reference_choice" in tools and "genome_version" in kept:
            page = view.reference.discovery_event_id
            row = _of(events, "ReferenceAssembliesRecorded")[-1]["payload"]["result"][
                "rows"
            ][0]["assembly_accession"]
            self.choice_quotes.append(kept["genome_version"].value)
            return self._call(
                "record_reference_choice",
                {
                    "discovery_event_id": page,
                    "assembly_accession": row,
                    "quote": kept["genome_version"].value,
                    "reason": "NCBI lists this row for the version the paper names.",
                },
            )
        return await super().respond(messages, info)


def _reader(plan, *answers):
    """The scripted reader: the paper item's plan, then each area item's answer."""
    replies = iter([("propose_reading_areas", plan), *answers])

    async def respond(messages, info):
        name, arguments = next(replies)
        return ModelResponse(
            parts=[ToolCallPart(name, arguments, tool_call_id=f"read-{len(messages)}")]
        )

    return ScriptedModelEndpoint(respond)


REFERENCE = {
    "name": VERSION,
    "span": "main:p2",
    "reason": "The reads are aligned to it.",
}
PAPER_ORGANISM = {"written": ORGANISM, "span": "main:p2"}
RESULTS_AREA = {
    "area": "the most significant variant of the results",
    "views": ["s2"],
    "reason": "The question asks for the paper's most significant SNP.",
}
RS1234 = {
    "findings": [
        {
            "values": [{"slot": "rsid", "written": "rs1234", "span": "main:p4"}],
            "reason": "The paper names it its most significant variant.",
        }
    ]
}


def _reading_plan(**changes):
    return {
        "reference": REFERENCE,
        "organism": PAPER_ORGANISM,
        "areas": [RESULTS_AREA],
        **changes,
    }


class _PaperRun:
    """One bare-citation Run over a temporary root: its scripted planning model, its
    scripted reader, and the hosts that hold the paper."""

    def __init__(
        self,
        reader,
        *,
        article=None,
        supplied_accession=None,
        plan_first=False,
        investigation_request=None,
    ):
        self.root = Path(tempfile.mkdtemp())
        self.receipt, self.model, _ = configured_planning_run(
            self.root / "run",
            prompt=PROMPT,
            composition_type=_CitationModel,
            investigation_request=investigation_request,
        )
        self.model.paper_adapter, self.asked = _hosts(article or _article())
        self.model.supplied_accession = supplied_accession
        self.model.plan_first = plan_first
        self.reader = reader
        self.questions = []
        self.advance(ResumeRun(journey_file=self.receipt.journey_file))

    def close(self):
        shutil.rmtree(self.root)

    @property
    def journey(self):
        return Journey.open(Path(self.receipt.journey_file))

    @property
    def events(self):
        return read_evidence(self.journey)

    def advance(self, command):
        with (
            use_run_advancement_composition(self.model),
            use_run_advancement_dependencies(paper_reading_model_factory=self.reader),
        ):
            self.result = advance(command)
        return self.result

    def answer_planning(self, response):
        request = self.result.current_review.request
        self.questions.append(request)
        return self.advance(
            AnswerReview(
                journey_file=self.receipt.journey_file,
                request_event_id=request.request_event_id,
                answer=PlanningReviewAnswer(kind="planning", response=response),
            )
        )

    def answer_until_execution(self):
        """Answer each planning question as the researcher would: the sample by name,
        and a question with candidates by its first one (as the shared scripted
        Run does); stop at any other Review."""
        from bio_neb.parabricks.planning_hitl import SelectPlanningHitlResponse

        while (
            self.result.current_review is not None
            and self.result.current_review.kind == "planning"
        ):
            request = self.result.current_review.request
            if request.candidates:
                self.answer_planning(
                    SelectPlanningHitlResponse(
                        action="select",
                        candidate_id=request.candidates[0].candidate_id,
                    )
                )
            else:
                self.answer_planning(
                    ClarifyPlanningHitlResponse(
                        action="clarify", instruction="My sample is sample-001."
                    )
                )
        return self.result


def _questions(events):
    """Each planning question asked before the Execution Review: its owner and slot."""
    from bio_neb.parabricks.planning_hitl import PlanningHitlRequested

    asked = [
        PlanningHitlRequested.model_validate(e["payload"])
        for e in _of(events, "PlanningHitlRequested")
    ]
    return [(request.owner, request.subject_key) for request in asked]


def _plain_run(root):
    """The same scripted Run with a prompt that states its sample, organism and
    target and cites no paper, answered to its Execution Review."""
    from tests.planning_assignment_fakes import prepare_configured_run

    receipt, review, _ = prepare_configured_run(root)
    return review, read_evidence(Journey.open(Path(receipt.journey_file)))


class BareCitationRunTests(unittest.TestCase):
    """A bare citation reaches the Execution Review with the organism, the reference
    version and the targets taken from the paper's reading (owner, #383 Q3 a, Q8);
    the sample is the one question (Q4 a). A Run whose prompt cites no paper runs as
    before: no reading, no reader Attempt, no paper checkpoint."""

    @classmethod
    def setUpClass(cls):
        cls.paper_run = _PaperRun(
            _reader(_reading_plan(), ("answer_reading_item", RS1234))
        )
        cls.paper_run.answer_until_execution()
        cls.events = cls.paper_run.events
        cls.plain_root = Path(tempfile.mkdtemp())
        cls.plain, cls.plain_events = _plain_run(cls.plain_root / "run")

    @classmethod
    def tearDownClass(cls):
        cls.paper_run.close()
        shutil.rmtree(cls.plain_root)

    def _plan_kept(self):
        (kept,) = _of(self.events, "ReadingPlanKept")
        return kept

    def test_a_cited_recorded_paper_is_read_in_planning_before_the_reference_work(self):
        """The plan's values cite the recorded paper, so the planning Attempt ends
        at the plan and the coordinator reads the paper, for the research goal in
        the researcher's words, before any reference fact (#383 A2)."""
        events = self.events
        (paper,) = _of(events, "PaperRecorded")
        (plan,) = _of(events, "ProposalRequirementsIdentified")
        (started,) = _of(events, "PaperReadingStarted")
        (answer,) = _of(events, "PaperAnswerRecorded")
        (taxon,) = _of(events, "TaxonResolutionRecorded")
        self.assertEqual(
            (started["payload"]["paper_event_id"], started["payload"]["question"]),
            (paper["event_id"], GOAL),
        )
        order = [events.index(e) for e in (plan, started, answer, taxon)]
        self.assertEqual(order, sorted(order))
        (due,) = [
            e
            for e in _of(events, "SliceCheckpointReached")
            if e["payload"]["checkpoint"] == "paper_reading_due"
        ]
        self.assertEqual(due["payload"]["proposal_plan_event_id"], plan["event_id"])
        self.assertLess(events.index(due), events.index(started))
        # Each reading item is its own paper_reader Attempt, the paper item first.
        readers = [
            e["payload"]["member_id"].rsplit(":", 1)[1]
            for e in _of(events, "AttemptStarted")
            if e["payload"]["agent_role"] == "paper_reader"
        ]
        self.assertEqual(readers, ["paper", "area-1"])

    def test_the_taxon_is_resolved_from_the_organism_the_paper_writes(self):
        """resolve_taxon cites the reading fact that keeps the organism's span, and
        its term is the organism the paper writes (ADR 0049: the model names it;
        code checks only that a reading kept one)."""
        kept = self._plan_kept()
        organism = kept["payload"]["organism"]
        self.assertEqual(
            (organism["written"], organism["span"], organism["source"]["text"]),
            (ORGANISM, "main:p2", ORGANISM),
        )
        self.assertEqual(organism["passage"]["text"], METHODS)
        (taxon,) = _of(self.events, "TaxonResolutionRecorded")
        tool_input = taxon["payload"]["tool_input"]
        self.assertEqual(
            (tool_input["query"], tool_input["assertion_event_id"]),
            (ORGANISM, kept["event_id"]),
        )
        self.assertIn(kept["event_id"], tool_input["grounding"]["source_event_ids"])
        self.assertEqual(self.paper_run.model.taxon_terms, [ORGANISM])

    def test_the_reference_row_is_chosen_from_ncbi_quoting_the_papers_version(self):
        """The model chooses a row NCBI listed and quotes the version the paper
        writes; the choice records the reading fact whose kept span holds the
        quote (#383 A2, ADR 0049 membership)."""
        kept = self._plan_kept()
        self.assertEqual(
            kept["payload"]["reference"]["candidate"]["sources"][0]["text"], METHODS
        )
        (page,) = _of(self.events, "ReferenceAssembliesRecorded")
        (choice,) = _of(self.events, "ReferenceChoiceRecorded")
        payload = choice["payload"]
        self.assertEqual(
            (payload["quote"], payload["quote_event_id"]),
            (VERSION, kept["event_id"]),
        )
        self.assertIn(
            payload["assembly_accession"],
            [r["assembly_accession"] for r in page["payload"]["result"]["rows"]],
        )
        (verified,) = _of(self.events, "ReferenceDescriptionRecorded")
        self.assertEqual(
            verified["payload"]["result"]["references"][0]["assembly_accession"],
            payload["assembly_accession"],
        )

    def test_the_papers_targets_are_the_runs_and_no_target_question_opens(self):
        """The reading's proposed target is the Run's request item, in the paper's
        words and asserted by the paper; the coordinator's target question (#342)
        never opens (owner Q4: only the targets' source changes)."""
        from bio_neb.investigation.investigation_request import (
            current_investigation_request,
        )

        (answer,) = _of(self.events, "PaperAnswerRecorded")
        targets = answer["payload"]["targets"]
        request = current_investigation_request(self.paper_run.journey).request
        (item,) = request.items
        self.assertEqual(
            (item.request_text, item.rsid, item.asserted_by),
            (RESULTS, "rs1234", targets["asserted_by"]),
        )
        self.assertEqual(targets["asserted_by"], f"paper:doi:{DOI}")
        self.assertNotIn(
            "requirement_resolution", [o for o, _ in _questions(self.events)]
        )

    def test_the_report_copies_what_the_paper_found_with_the_chunks_it_kept(self):
        """#485 Q3 a: what a report's basis states about the Run's papers is the
        reading's answer as recorded, asserted by the paper, and each finding's
        chunks stand at their place in the text the Run stored for the paper."""
        from bio_neb.paper_reading.report import reported_papers

        (paper,) = reported_papers(self.events)
        (answer,) = _of(self.events, "PaperAnswerRecorded")
        (recorded,) = _of(self.events, "PaperRecorded")
        self.assertEqual(
            (paper.answer_event_id, paper.paper_event_id, paper.asserted_by),
            (answer["event_id"], recorded["event_id"], f"paper:doi:{DOI}"),
        )
        self.assertEqual(paper.paper.title, TITLE)
        (finding,) = paper.answer.findings
        self.assertEqual(finding.reason, RS1234["findings"][0]["reason"])
        quote, *rest = finding.passages
        self.assertEqual(quote.text, "rs1234")
        self.assertIn(RESULTS, [passage.text for passage in rest])
        (read,) = _of(self.events, "PaperDocumentRead")
        stored = self.paper_run.journey.read_artifact(
            read["artifact_references"][0]
        ).decode()
        for passage in finding.passages:
            self.assertEqual(stored[passage.start : passage.end], passage.text)
        self.assertEqual(paper.answer.organism.written, ORGANISM)
        self.assertEqual(paper.answer.reference.candidate.name, VERSION)

    def test_the_investigation_member_receives_its_paper_and_what_planning_kept(self):
        """#485 Q2 a: the member whose target the paper's reading proposed receives
        that paper, the reading's answer it came from and what the reading kept for
        it, each chunk copied from the stored paper; its reading Tools read the
        paper planning read. A member the researcher asserted receives none."""
        from bio_neb.investigation.investigation_request import (
            current_investigation_request,
        )
        from bio_neb.paper_reading.investigation import paper_context, target_paper
        from bio_neb.paper_reading.reader import Paper

        (item,) = current_investigation_request(self.paper_run.journey).request.items
        found = target_paper(self.events, item)
        (answer,) = _of(self.events, "PaperAnswerRecorded")
        (recorded,) = _of(self.events, "PaperRecorded")
        self.assertEqual(
            (found.answer_event_id, found.paper_event_id),
            (answer["event_id"], recorded["event_id"]),
        )
        context = paper_context(
            {e["event_id"]: e for e in self.events}, found.answer_event_id, item
        )
        self.assertEqual(
            (context["title"], context["asserted_by"], context["ids"]["doi"]),
            (TITLE, f"paper:doi:{DOI}", DOI),
        )
        self.assertEqual(context["target_quote"]["text"], RESULTS)
        (finding,) = context["target_findings"]
        self.assertEqual(finding["reason"], RS1234["findings"][0]["reason"])
        self.assertIn(RESULTS, [p["text"] for p in finding["passages"]])
        self.assertEqual(context["genome_version_as_written"]["name"], VERSION)
        self.assertEqual(context["organism_as_written"]["written"], ORGANISM)
        self.assertIn(METHODS, context["organism_as_written"]["passage"]["text"])
        paper = Paper.stored(self.paper_run.journey, found.paper_event_id)
        self.assertIn(RESULTS, paper.documents["main"].text)
        # The researcher's own target: no paper.
        self.assertIsNone(
            target_paper(self.events, item.model_copy(update={"asserted_by": None}))
        )

    def test_a_member_with_no_paper_has_its_reading_tools_wait(self):
        """#485 Q2 a: the reading Tools wait for a member no paper proposed, with
        their reason, and bind the paper's own values for one a paper did."""
        from types import SimpleNamespace

        from bio_neb.agentic_journey.action_policy import PreconditionUnmet
        from bio_neb.investigation.investigation_agent import (
            MEMBER_HAS_NO_PAPER,
            investigation_tool_values,
        )
        from bio_neb.paper_reading.reader import Paper

        (recorded,) = _of(self.events, "PaperRecorded")
        for name in ("outline_paper", "grep_paper", "read_paper", "filter_paper_rows"):
            with self.subTest(name=name):
                with self.assertRaises(PreconditionUnmet) as raised:
                    investigation_tool_values(SimpleNamespace(paper=None), name)
                self.assertEqual(str(raised.exception), MEMBER_HAS_NO_PAPER.code)
        paper = Paper.stored(self.paper_run.journey, recorded["event_id"])
        values = investigation_tool_values(
            SimpleNamespace(paper=paper), "outline_paper"
        )
        self.assertEqual((values.constants, values.enums), ({}, {}))

    def test_a_fork_keeps_the_paper_and_its_reading(self):
        """#485 scope 4: a Fork child of the Run keeps the stored paper and its
        reading in its own run directory: the reader opens the paper from the
        child, the report's papers read the same answer, and the child's member
        receives its paper as the parent's would."""
        from bio_neb.application.fork import ForkRun, fork_run
        from bio_neb.investigation.investigation_request import (
            current_investigation_request,
        )
        from bio_neb.paper_reading.investigation import target_paper
        from bio_neb.paper_reading.reader import Paper
        from bio_neb.paper_reading.report import reported_papers

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        receipt = fork_run(
            ForkRun(
                parent_journey_file=self.paper_run.receipt.journey_file,
                child_run_root=str(root / "child"),
            )
        )
        child = Journey.open(Path(receipt.child_journey_file))
        self.assertNotEqual(child.path.parent, self.paper_run.journey.path.parent)
        events = read_evidence(child)
        (recorded,) = _of(events, "PaperRecorded")
        paper = Paper.stored(child, recorded["event_id"])
        self.assertIn(RESULTS, paper.documents["main"].text)
        self.assertEqual(reported_papers(events), reported_papers(self.events))
        (item,) = current_investigation_request(child).request.items
        self.assertEqual(target_paper(events, item), target_paper(self.events, item))

    def test_the_sample_is_the_only_question_the_paper_leaves(self):
        """Before the Execution Review the Run asks the sample, which only the
        researcher names (owner Q4 a), beside the recipe question the scripted
        planning model asks in every Run, the plain one too; no taxon, reference or
        target question opens."""
        self.assertEqual(self.paper_run.result.current_review.kind, "execution")
        self.assertEqual(self.plain.current_review.kind, "execution")
        self.assertEqual(
            _questions(self.plain_events), [("agent_attempt", "analysis_recipe")]
        )
        self.assertEqual(
            _questions(self.events),
            [
                ("agent_attempt", "sample_identity"),
                ("agent_attempt", "analysis_recipe"),
            ],
        )

    def test_a_run_whose_prompt_cites_no_paper_reads_none(self):
        """No paper, no reading: no reading fact, no reader Attempt and no paper
        checkpoint (#420 D1 b: a Run that cites none never enters the paper stage)."""
        from bio_neb.paper_reading.journey import PAYLOADS

        self.assertFalse([e for e in self.plain_events if e["event_type"] in PAYLOADS])
        self.assertNotIn(
            "paper_reader",
            {
                e["payload"]["agent_role"]
                for e in _of(self.plain_events, "AttemptStarted")
            },
        )
        self.assertNotIn(
            "paper_reading_due",
            {
                e["payload"]["checkpoint"]
                for e in _of(self.plain_events, "SliceCheckpointReached")
            },
        )

    def test_the_journey_reopens_with_the_reading_between_planning_attempts(self):
        """The planning slice composes through the reading's Attempts and its
        settling facts (#383 A1's open reopen point): a fresh open reads the whole
        Run and agrees with the live view."""
        from bio_neb.application.queries import ReviewJourneyQuery, review_journey

        reopened = review_journey(
            ReviewJourneyQuery(journey_file=self.paper_run.receipt.journey_file)
        )
        self.assertEqual(reopened.status, self.paper_run.result.status)
        self.assertEqual(
            reopened.current_review.request.request_event_id,
            self.paper_run.result.current_review.request.request_event_id,
        )


class PlanFirstCitationTests(unittest.TestCase):
    """The live model's order (#383 A2 probe, 2026-10-29): it records the plan with
    the citation, and the paper stage retrieves the paper after it. The plan's
    values cite the request, not the paper; the plan's citation names the paper
    they are found in, so the reading runs and settles them as before."""

    @classmethod
    def setUpClass(cls):
        cls.paper_run = _PaperRun(
            _reader(_reading_plan(), ("answer_reading_item", RS1234)),
            plan_first=True,
        )
        cls.paper_run.answer_until_execution()
        cls.events = cls.paper_run.events

    @classmethod
    def tearDownClass(cls):
        cls.paper_run.close()

    def test_the_plan_precedes_the_paper_and_the_paper_is_read_before_reference(self):
        order = [e["event_type"] for e in self.events]
        self.assertLess(
            order.index("ProposalRequirementsIdentified"), order.index("PaperRecorded")
        )
        self.assertLess(
            order.index("PaperRecorded"), order.index("PaperAnswerRecorded")
        )
        self.assertLess(
            order.index("PaperAnswerRecorded"), order.index("TaxonResolutionRecorded")
        )

    def test_the_values_come_from_the_reading_and_the_sample_is_the_one_question(self):
        from bio_neb.investigation.investigation_request import (
            current_investigation_request,
        )

        (kept,) = _of(self.events, "ReadingPlanKept")
        (taxon,) = _of(self.events, "TaxonResolutionRecorded")
        (choice,) = _of(self.events, "ReferenceChoiceRecorded")
        self.assertEqual(
            taxon["payload"]["tool_input"]["assertion_event_id"], kept["event_id"]
        )
        self.assertEqual(
            (choice["payload"]["quote"], choice["payload"]["quote_event_id"]),
            (VERSION, kept["event_id"]),
        )
        (item,) = current_investigation_request(self.paper_run.journey).request.items
        self.assertEqual((item.request_text, item.rsid), (RESULTS, "rs1234"))
        self.assertEqual(
            _questions(self.events),
            [
                ("agent_attempt", "sample_identity"),
                ("agent_attempt", "analysis_recipe"),
            ],
        )
        self.assertEqual(self.paper_run.result.current_review.kind, "execution")


class CitationBesideRequestFileTests(unittest.TestCase):
    """A cited paper beside a registered request file (lane P landing on M, owner
    2026-10-30: "a yes this papaer is a valide target"): the paper's targets are its
    own item's, the file's item stands for the file's targets alone (#367, #482), and
    the planned request holds both, each once."""

    @classmethod
    def setUpClass(cls):
        cls.paper_run = _PaperRun(
            _reader(_reading_plan(), ("answer_reading_item", RS1234)),
            plan_first=True,
            investigation_request={
                "schema_version": "bio-neb.investigation-request/3",
                "asserted_by": "researcher:owner",
                "requests": [{"request": "Also check rs4149056."}],
            },
        )
        cls.paper_run.answer_until_execution()

    @classmethod
    def tearDownClass(cls):
        cls.paper_run.close()

    def test_the_files_item_carries_none_of_the_papers_targets(self):
        from bio_neb.investigation.investigation_request import (
            current_investigation_request,
        )

        journey = self.paper_run.journey
        (registered,) = _of(journey.read(), "InvestigationRequestRegistered")
        handoff = self.paper_run.result.requirement_resolution.scientific_investigation
        items = {
            item.registered_request_event_id is not None: item for item in handoff.items
        }
        self.assertEqual(len(handoff.items), 2)
        self.assertEqual(
            (items[True].registered_request_event_id, items[True].targets),
            (registered["event_id"], ()),
        )
        self.assertEqual([t.rsid for t in items[False].targets], ["rs1234"])
        request = current_investigation_request(journey).request
        self.assertEqual(
            [(i.request_text, i.rsid) for i in request.items],
            [("Also check rs4149056.", None), (RESULTS, "rs1234")],
        )


class PaperWithoutOrganismTests(unittest.TestCase):
    """A paper that writes no organism settles no taxon: after the sample, the
    stage's taxon question opens as it does today (owner Q3: what the paper does not
    settle goes to the existing question)."""

    @classmethod
    def setUpClass(cls):
        cls.paper_run = _PaperRun(
            _reader(
                _reading_plan(organism=None),
                ("answer_reading_item", RS1234),
            ),
            article=_article(f"Reads were aligned to the {VERSION} reference."),
        )
        cls.paper_run.answer_planning(
            ClarifyPlanningHitlResponse(
                action="clarify", instruction="My sample is sample-001."
            )
        )
        cls.events = cls.paper_run.events

    @classmethod
    def tearDownClass(cls):
        cls.paper_run.close()

    def test_the_taxon_question_opens_after_the_sample(self):
        (answer,) = _of(self.events, "PaperAnswerRecorded")
        self.assertIsNone(answer["payload"]["answer"]["organism"])
        self.assertEqual(
            _questions(self.events),
            [
                ("agent_attempt", "sample_identity"),
                ("agent_attempt", "taxon_identity"),
            ],
        )
        self.assertFalse(_of(self.events, "TaxonResolutionRecorded"))
        self.assertEqual(self.paper_run.result.current_review.kind, "planning")


class PaperAccessionTests(unittest.TestCase):
    """A paper that writes the exact assembly accession: the model types it as the
    supplied reference, citing the reading fact that keeps the span writing it, and
    code checks the quote stands there and writes the accession (ADR 0049
    membership; #472 Q2 a's rule over the paper's words)."""

    ACCESSION = "GCF_000001405.40"

    @classmethod
    def setUpClass(cls):
        methods = (
            f"Reads of the {ORGANISM} sample were aligned to {VERSION} "
            f"({cls.ACCESSION})."
        )
        cls.paper_run = _PaperRun(
            _reader(
                _reading_plan(
                    reference={**REFERENCE, "aliases": [cls.ACCESSION]},
                ),
                ("answer_reading_item", RS1234),
            ),
            article=_article(methods),
            supplied_accession=cls.ACCESSION,
        )
        cls.paper_run.answer_until_execution()
        cls.events = cls.paper_run.events

    @classmethod
    def tearDownClass(cls):
        cls.paper_run.close()

    def test_the_supplied_accession_cites_the_reading_fact_that_keeps_it(self):
        (kept,) = _of(self.events, "ReadingPlanKept")
        (choice,) = _of(self.events, "ReferenceChoiceRecorded")
        payload = choice["payload"]
        self.assertEqual(
            (
                payload["assembly_accession"],
                payload["assertion_event_id"],
                payload["quote"],
                payload.get("discovery_event_id"),
            ),
            (self.ACCESSION, kept["event_id"], self.ACCESSION, None),
        )
        self.assertFalse(_of(self.events, "ReferenceAssembliesRecorded"))
        (verified,) = _of(self.events, "ReferenceDescriptionRecorded")
        self.assertEqual(
            verified["payload"]["result"]["references"][0]["assembly_accession"],
            self.ACCESSION,
        )
        self.assertEqual(self.paper_run.result.current_review.kind, "execution")


class ReadingStopTests(unittest.TestCase):
    """A reading item that stops holds the Run at its Agent work Review: planning
    does not go on, no question opens. Its continue answer re-enters the item, the
    reading records its answer, and planning goes on to the sample (ADR 0039; owner
    #383 Q23)."""

    def test_a_stopped_item_holds_the_run_and_its_continue_resumes_it(self):
        from bio_neb.agentic_journey.agent_work_review import ContinueAgentWorkResponse
        from bio_neb.application.advancement import AgentWorkReviewAnswer

        # Four schema refusals in a row are four failed turns: the area stops.
        run = _PaperRun(
            _reader(
                _reading_plan(),
                *[("answer_reading_item", {})] * 4,
                ("answer_reading_item", RS1234),
            )
        )
        self.addCleanup(run.close)
        self.assertEqual(
            (run.result.status, run.result.current_review.kind),
            ("agent_work_review_pending", "agent_work"),
        )
        events = run.events
        (review,) = _of(events, "AgentWorkReviewRequested")
        stopped = review["payload"]["attempt_id"]
        (area,) = [
            e
            for e in _of(events, "AttemptStarted")
            if e["payload"]["attempt_id"] == stopped
        ]
        self.assertTrue(area["payload"]["member_id"].endswith(":area-1"))
        self.assertFalse(_of(events, "PaperAnswerRecorded"))
        self.assertFalse(_of(events, "PlanningHitlRequested"))
        self.assertEqual(
            [e["payload"]["checkpoint"] for e in _of(events, "SliceCheckpointReached")],
            ["paper_reading_due"],
        )

        run.advance(
            AnswerReview(
                journey_file=run.receipt.journey_file,
                request_event_id=review["event_id"],
                answer=AgentWorkReviewAnswer(
                    kind="agent_work",
                    response=ContinueAgentWorkResponse(action="continue"),
                ),
            )
        )

        events = run.events
        (answer,) = _of(events, "PaperAnswerRecorded")
        self.assertEqual(
            [f["reason"] for f in answer["payload"]["answer"]["findings"]],
            [RS1234["findings"][0]["reason"]],
        )
        resumed = [
            e
            for e in _of(events, "AttemptStarted")
            if e["payload"]["agent_role"] == "paper_reader"
            and e["payload"]["member_id"] == area["payload"]["member_id"]
        ]
        self.assertEqual(len(resumed), 2)
        self.assertEqual(
            run.result.current_review.request.subject_key, "sample_identity"
        )

    def test_a_stopped_item_holds_only_itself_while_its_siblings_are_read(self):
        """ADR 0039 and owner #383 Q23 in the Run's coordinator: a reading item's own
        pending Review holds that item alone, so the reading's next area runs; the
        Review stays pending, and the reading's answer and planning wait for it."""
        from bio_neb.agentic_journey.agent_work_review import ContinueAgentWorkResponse
        from bio_neb.application.advancement import AgentWorkReviewAnswer

        second = {**RESULTS_AREA, "area": "the independent second result"}
        run = _PaperRun(
            _reader(
                _reading_plan(areas=[RESULTS_AREA, second]),
                # Area 1: four schema refusals in a row stop it; area 2 answers.
                *[("answer_reading_item", {})] * 4,
                ("answer_reading_item", RS1234),
                # Area 1 again, re-entered by its continue answer.
                ("answer_reading_item", RS1234),
            )
        )
        self.addCleanup(run.close)

        def members(events):
            return [
                e["payload"]["member_id"].rsplit(":", 1)[1]
                for e in _of(events, "AttemptStarted")
                if e["payload"]["agent_role"] == "paper_reader"
            ]

        events = run.events
        (review,) = _of(events, "AgentWorkReviewRequested")
        self.assertEqual(members(events), ["paper", "area-1", "area-2"])
        self.assertEqual(
            (
                run.result.status,
                run.result.current_review.kind,
                run.result.current_review.request.request_event_id,
            ),
            ("agent_work_review_pending", "agent_work", review["event_id"]),
        )
        self.assertFalse(_of(events, "AgentWorkReviewResponded"))
        self.assertEqual(len(_of(events, "ReadingItemCompleted")), 2)
        self.assertFalse(_of(events, "PaperAnswerRecorded"))
        self.assertFalse(_of(events, "PlanningHitlRequested"))

        run.advance(
            AnswerReview(
                journey_file=run.receipt.journey_file,
                request_event_id=review["event_id"],
                answer=AgentWorkReviewAnswer(
                    kind="agent_work",
                    response=ContinueAgentWorkResponse(action="continue"),
                ),
            )
        )

        events = run.events
        self.assertEqual(members(events), ["paper", "area-1", "area-2", "area-1"])
        self.assertEqual(len(_of(events, "PaperAnswerRecorded")), 1)
        self.assertEqual(
            run.result.current_review.request.subject_key, "sample_identity"
        )


class SharedLineTargetTests(unittest.TestCase):
    """#485 r1 ST1/SP (both reviews' probes): two rsIDs one paragraph writes become
    two targets with the same words; each Investigation member receives its own
    target's reading, matched by its typed target, never its sibling's (ADR 0039).
    A derived input: the fictional paper's results paragraph names two rsIDs, each
    kept as its own finding."""

    @classmethod
    def setUpClass(cls):
        sentence = "The variants rs1234 and rs5678 were enriched among cases."
        answer = {
            "findings": [
                {
                    "values": [{"slot": "rsid", "written": rsid, "span": "main:p4"}],
                    "reason": f"The paper reports {rsid} independently.",
                }
                for rsid in ("rs1234", "rs5678")
            ]
        }
        cls.paper_run = _PaperRun(
            _reader(_reading_plan(), ("answer_reading_item", answer)),
            article=_article().replace(RESULTS.encode(), sentence.encode()),
        )
        cls.addClassCleanup(cls.paper_run.close)
        cls.paper_run.answer_until_execution()
        cls.events = cls.paper_run.events

    def test_each_member_receives_its_own_targets_reading(self):
        from bio_neb.investigation.investigation_request import (
            current_investigation_request,
        )
        from bio_neb.paper_reading.investigation import paper_context, target_paper

        items = current_investigation_request(self.paper_run.journey).request.items
        self.assertEqual([item.rsid for item in items], ["rs1234", "rs5678"])
        self.assertEqual(items[0].request_text, items[1].request_text)
        by_id = {e["event_id"]: e for e in self.events}
        for item in items:
            with self.subTest(rsid=item.rsid):
                found = target_paper(self.events, item)
                self.assertEqual(found.target.request.rsid, item.rsid)
                context = paper_context(by_id, found.answer_event_id, item)
                self.assertEqual(
                    [finding["reason"] for finding in context["target_findings"]],
                    [f"The paper reports {item.rsid} independently."],
                )


class MemberScopedStopTests(unittest.TestCase):
    """#485 Q4 a (the #383 r2/r3 owner gap): a reading item's model-availability
    and effort stops, answered stop through the Run's ``AnswerReview``, settle that
    item alone as not read, and the Run goes on: its sibling is read and the reading
    records its answer (ADR 0031, 0039; the role declares ``settles_member_stops``).
    Promoted from the #383 Spec reviewer's Run probes (r2, `issue383_spec_probes`)."""

    def _limited(self, configured_change):
        from unittest.mock import patch

        from tests.proposal_information_fakes import runtime_configuration

        def runtime(*args, **kwargs):
            configured = runtime_configuration(*args, **kwargs)
            configured_change(configured)
            return configured

        return patch(
            "tests.proposal_information_fakes.runtime_configuration",
            side_effect=runtime,
        )

    def test_an_items_availability_stop_settles_it_alone_and_the_run_goes_on(self):
        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.agentic_journey.model_availability import (
            StopModelAvailabilityResponse,
        )
        from bio_neb.application.advancement import ModelAvailabilityReviewAnswer

        def one_contact(configured):
            configured["failure"] = {
                "model_contact": {"base_delay_seconds": 0, "contacts": 1}
            }

        second = {**RESULTS_AREA, "area": "the independent second result"}
        replies = iter(
            [
                ("propose_reading_areas", _reading_plan(areas=[RESULTS_AREA, second])),
                None,  # area 1's model is unavailable: its Review opens
                ("answer_reading_item", RS1234),  # area 2 is read meanwhile
            ]
        )

        async def respond(messages, info):
            reply = next(replies)
            if reply is None:
                raise ModelHTTPError(503, "scripted/model", "unavailable")
            return ModelResponse(
                parts=[ToolCallPart(*reply, tool_call_id=f"read-{len(messages)}")]
            )

        with self._limited(one_contact):
            run = _PaperRun(ScriptedModelEndpoint(respond))
        self.addCleanup(run.close)
        (review,) = _of(run.events, "ModelAvailabilityReviewRequested")
        self.assertEqual(run.result.current_review.kind, "model_availability")
        self.assertTrue(review["payload"]["member_id"].endswith(":area-1"))

        run.advance(
            AnswerReview(
                journey_file=run.receipt.journey_file,
                request_event_id=review["event_id"],
                answer=ModelAvailabilityReviewAnswer(
                    kind="model_availability",
                    response=StopModelAvailabilityResponse(
                        action="stop", reason="Skip this item."
                    ),
                ),
            )
        )

        events = run.events
        (not_read,) = _of(events, "ReadingItemNotRead")
        self.assertEqual(
            (not_read["payload"]["item_id"], not_read["payload"]["outcome"]),
            ("area-1", "stopped"),
        )
        self.assertEqual(not_read["payload"]["reason"], "Skip this item.")
        (answer,) = _of(events, "PaperAnswerRecorded")
        self.assertEqual(len(answer["payload"]["answer"]["findings"]), 1)
        # The Run's work did not end: planning asks its next question.
        self.assertFalse(_of(events, "FinalizationBasisCompiled"))
        self.assertEqual(
            run.result.current_review.request.subject_key, "sample_identity"
        )

    def test_an_items_effort_stop_settles_it_alone_and_the_run_goes_on(self):
        from bio_neb.agentic_journey.effort import StopAgentEffortResponse
        from bio_neb.application.advancement import AgentEffortReviewAnswer

        def one_request(configured):
            configured["agent_invocation"]["scopes"] = {
                "role": {"paper_reader": {"model_requests": 1}}
            }

        with self._limited(one_request):
            run = _PaperRun(_reader(_reading_plan(), ("outline_paper", {})))
        self.addCleanup(run.close)
        (review,) = _of(run.events, "AgentEffortReviewRequested")

        run.advance(
            AnswerReview(
                journey_file=run.receipt.journey_file,
                request_event_id=review["event_id"],
                answer=AgentEffortReviewAnswer(
                    kind="agent_effort",
                    response=StopAgentEffortResponse(
                        action="stop", reason="Skip this item."
                    ),
                ),
            )
        )

        events = run.events
        (not_read,) = _of(events, "ReadingItemNotRead")
        self.assertEqual(not_read["payload"]["outcome"], "stopped")
        self.assertEqual(len(_of(events, "PaperAnswerRecorded")), 1)
        self.assertFalse(_of(events, "FinalizationBasisCompiled"))

    def test_a_fault_inside_a_reading_tool_holds_the_run_as_its_attempts_failure(self):
        """#485 Q4: a reading Tool reads the stored paper and contacts nothing, so
        it has no operational stop to open: a fault inside it is the reading
        Attempt's failure, and the Run is held under its name (ADR 0031: our bug,
        not a scoped stop). A derived fault: ``outline`` raises."""
        from unittest import mock

        from bio_neb.application.advancement import RunAdvancementRefused
        from bio_neb.paper_reading import tools

        def failing(_paper):
            raise OSError("the stored text could not be read")

        root = self.enterContext(tempfile.TemporaryDirectory())
        with (
            mock.patch.object(tools, "outline", failing),
            # The Run's root, removed here: the held Run never returns its handle.
            mock.patch.object(tempfile, "mkdtemp", return_value=root),
            self.assertRaises(RunAdvancementRefused) as held,
        ):
            _PaperRun(_reader(_reading_plan(), ("outline_paper", {})))
        self.assertEqual(held.exception.code, "paper_reading_attempt_failed")


if __name__ == "__main__":
    unittest.main()
