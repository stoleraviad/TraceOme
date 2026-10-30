"""The planning Agent's paper Tools (#420): find a cited paper, then retrieve it.

A researcher may cite a paper in their words (an id, a link, or a title, author and
year). The planning Agent declares each cited paper as a ``cited_paper`` requirement
of its plan, and a plan with one opens the **paper stage** before the reference work
(owner, #420 D1 b); a Run that cites none never enters it. No code reads the words
for it (ADR 0049): the model types a Europe PMC query with ``find_paper`` for one
cited paper, named by its requirement's ``subject_key``, and Europe PMC's documented options pass verbatim (its query
syntax, ``pageSize``, ``cursorMark`` and ``sort``); every row of the page is copied in
its order, with ``hitCount`` and the next cursor, and each GET's answer is kept beside
them (#458). The model chooses the row with ``retrieve_paper``, and code checks only
membership: the row is one the cited page lists and the quote stands in the
researcher's words. The download runs in the Tool's own source stage, with its 16
contacts (#419); its documents are recorded once as a Run paper (`intake.record_paper`).
A dead end (no open copy, no article, no such paper, a stopped download) records the
refusal and asks the paper Review (owner, #420 P1), and the Attempt ends there.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Annotated, Any, ClassVar

from pydantic import Field, StrictStr

from bio_neb.agentic_journey.action_policy import ChoiceReason, PreconditionUnmet
from bio_neb.agentic_journey.invocation import ToolContext
from bio_neb.agentic_journey.planning_preconditions import (
    CORRECTION_PENDING,
    QUESTION_PENDING,
)
from bio_neb.agentic_journey.source_contacts import (
    observed_source_operation,
    scope_contact_event_ids,
)
from bio_neb.agentic_journey.stage_findings import StageFinding, StageOutput
from bio_neb.agentic_journey.tool_execution import (
    ModelView,
    ToolExecutionAdapter,
    ToolPrecondition,
    function_tool_contract,
)
from bio_neb.application.planning_read import current_planning, planning_evidence
from bio_neb.paper_intake.documents import SourceFile
from bio_neb.paper_intake.intake import record_paper, refuse_paper
from bio_neb.paper_intake.journey import (
    NO_PAPER,
    PaperRetrievalInput,
    PaperRetrievalRecorded,
    PaperSearchInput,
    PaperSearchRecorded,
    PaperSearchResult,
    PaperSearchRow,
    RunPapers,
    UnreadFile,
    new_intake_id,
)
from bio_neb.providers.paper_sources import (
    MAX_SEARCH_PAGE_SIZE,
    PAPER_DOWNLOAD_SURFACES,
    SEARCH_PAGE_SIZE,
    PaperDownloadStopped,
    PaperSourceAdapter,
    download_paper,
    search_europe_pmc,
)
from bio_neb.providers.source_operations import SourceSurface
from bio_neb.reference_discovery.reference_journey import quoted, researcher_text

PAPER_TOOLS = ("find_paper", "retrieve_paper")
# The candidate id a paper question presents a search row by (`paper:<candidate_id>`).
PAPER_CANDIDATE = "paper:"
# A cited paper's intake states that settle it: recorded, or dropped by a paper
# Review answered `continue`.
_SETTLED = frozenset({"recorded", "dropped"})
# The states a search for it may serve: none retrieved yet, or a paper Review sent
# it back (an amended or retried reference, ADR 0049 re-entry).
_TO_FIND = frozenset({"to_find", "reference_due"})

_BOUND_NOTHING = SimpleNamespace(constants={}, enums={})


def run_papers(deps) -> RunPapers:
    """The Run's papers in planning's one read of the current prefix: folded once
    and shared by every paper Tool's availability, admission and question (#420 ST2)."""
    return current_planning(deps.journey).papers


def cited_papers(view) -> tuple:
    """The plan's ``cited_paper`` requirements, in its order: none before the plan
    and on a Run whose words cite no paper."""
    if view.plan is None:
        return ()
    return tuple(r for r in view.plan.plan.requirements if r.kind == "cited_paper")


def _searched(papers: RunPapers) -> dict[str, str]:
    """Each search's cited paper, by its subject key."""
    return {s.source_event_id: s.tool_input.subject_key for s in papers.searches}


def _key_intake(papers: RunPapers, subject_key: str):
    """The intake of the latest retrieval for this cited paper, or None."""
    searched = _searched(papers)
    latest = next(
        (
            r
            for r in reversed(papers.retrievals)
            if searched[r.tool_input.search_event_id] == subject_key
        ),
        None,
    )
    return None if latest is None else papers.intake(latest.intake_id)


def citation_intake(view, requirement):
    """The intake one cited paper rests on: the paper its requirement cites, when
    the paper was retrieved before the plan (owner redirect, 2026-10-29: "it need
    to be available from the start"), else its latest retrieval by subject key;
    None before any retrieval for it."""
    cited = {
        e.source_event_id
        for e in requirement.evidence
        if e.source_kind == "planning_context"
    }
    return next(
        (i for i in view.papers.intakes if i.paper_event_id in cited), None
    ) or _key_intake(view.papers, requirement.subject_key)


def citation_state(view, requirement) -> str:
    """One cited paper's state: ``to_find`` before any retrieval for it, else its
    intake's state (recorded, review_pending, dropped, reference_due, due for an
    amended file, aborted)."""
    intake = citation_intake(view, requirement)
    return "to_find" if intake is None else intake.state


def citations_to_find(view) -> tuple[str, ...]:
    """The subject keys of the cited papers a search may serve now."""
    return tuple(
        r.subject_key for r in cited_papers(view) if citation_state(view, r) in _TO_FIND
    )


def _keys_to_find_before_plan(papers: RunPapers) -> frozenset[str]:
    """Before the plan, the papers searched for and not settled yet, by the subject
    key each search names."""
    return frozenset(
        key
        for key in _searched(papers).values()
        if (intake := _key_intake(papers, key)) is None or intake.state in _TO_FIND
    )


class PaperStageOutput(StageOutput):
    """Each cited paper settled, in the plan's order: its recorded paper, or the
    paper Review's `continue` answer that dropped it (owner, #420 D1 b)."""

    stage: ClassVar[str] = "paper"
    evidence_types: ClassVar = {
        "settled_event_ids": frozenset({"PaperRecorded", "PaperReviewAnswered"})
    }
    settled_event_ids: tuple[StrictStr, ...]


def paper_stage_output(view):
    """The paper stage's typed output once every cited paper is settled, else None.
    A paper retrieved before the plan settles its citation at once; a pending paper
    Review holds it, and an `amend paper_reference` or a retry sends it back
    (``reference_due``). A recorded paper the plan's to_find values cite is read
    first: the stage stays open until its reading records its answer (#383 A2)."""
    from bio_neb.paper_reading.planning import reading_due

    if reading_due(view) is not None:
        return None
    settled = []
    for requirement in cited_papers(view):
        intake = citation_intake(view, requirement)
        if intake is None or intake.state not in _SETTLED:
            return None
        settled.append(
            intake.paper_event_id
            if intake.state == "recorded"
            else intake.source_event_id
        )
    return PaperStageOutput(settled_event_ids=tuple(settled))


def paper_question(deps):
    """The paper confirmation now: the cited paper still to find and the open
    search the question asks about, or None when no cited paper waits (no plan
    yet, or every cited paper settled). A cited paper waiting with no open search
    has no question to ask yet (``paper_question_not_needed``)."""
    view = current_planning(deps.journey)
    keys = citations_to_find(view)
    if not keys:
        return None
    search = open_paper_search(deps)
    if search is None or search.tool_input.subject_key not in keys:
        raise PreconditionUnmet("paper_question_not_needed")
    requirement = next(
        r for r in cited_papers(view) if r.subject_key == search.tool_input.subject_key
    )
    return requirement, search


def open_paper_search(deps):
    """The latest paper search no retrieval and no question has acted on yet: the
    search a planning question asks the researcher about (owner, #420 F2 b). None
    when there is none."""
    papers = run_papers(deps)
    if not papers.searches:
        return None
    latest = papers.searches[-1]
    if any(
        r.tool_input.search_event_id == latest.source_event_id
        for r in papers.retrievals
    ):
        return None
    asked = (
        request
        for request in current_planning(deps.journey).hitl.requests
        if latest.source_event_id in request.evidence_event_ids
    )
    return None if next(asked, None) is not None else latest


def paper_findings(view) -> tuple[StageFinding, ...]:
    """The paper rows of the plan stage and the paper stage: each cited paper of the
    plan with its state and the researcher's words that cite it, each search and
    retrieval as recorded with the cited paper it serves, each recorded citation
    paper (the event a plan's requirement cites), each citation a paper Review sent
    back to planning with the researcher's words (an amended reference) or the
    refused one (a retry), for the model to type again (ADR 0049 re-entry), and
    each one dropped."""
    papers = view.papers
    answers = {r.response_event_id: r.response for r in papers.receipts}
    searched = _searched(papers)
    return (
        *(
            StageFinding(
                kind="cited_paper",
                source_event_id=view.plan_event_id,
                status=citation_state(view, r),
                subject_key=r.subject_key,
                value=r.value,
            )
            for r in cited_papers(view)
        ),
        *(
            StageFinding(
                kind="PaperSearchRecorded",
                source_event_id=s.source_event_id,
                status=s.result.status,
                subject_key=s.tool_input.subject_key,
            )
            for s in papers.searches
        ),
        *(
            StageFinding(
                kind="PaperRetrievalRecorded",
                source_event_id=r.source_event_id,
                status=r.status,
                subject_key=searched[r.tool_input.search_event_id],
            )
            for r in papers.retrievals
        ),
        *(
            StageFinding(
                kind="PaperRecorded",
                source_event_id=r.paper_event_id,
                status="recorded",
                subject_key=searched[r.tool_input.search_event_id],
            )
            for r in papers.retrievals
            if r.paper_event_id is not None
        ),
        *(
            StageFinding(
                kind="PaperReviewAnswered",
                source_event_id=due.answer_event_id,
                status="reference_due",
                subject_key="paper_reference",
                value=answers[due.answer_event_id].value
                if answers[due.answer_event_id].action == "amend"
                else due.reference,
            )
            for due in papers.references_due
        ),
        # A citation the researcher chose to go on without.
        *(
            StageFinding(
                kind="PaperReviewAnswered",
                source_event_id=intake.source_event_id,
                status="dropped",
                subject_key="paper_reference",
                value=intake.reference,
            )
            for intake in papers.intakes
            if intake.state == "dropped" and intake.origin == "citation"
        ),
    )


def _selected_candidate(deps, search_event_id: str) -> str | None:
    """The row the researcher selected in their latest answer about this search."""
    receipts = [
        receipt
        for receipt in current_planning(deps.journey).hitl.receipts
        if search_event_id in receipt.assertion.evidence_event_ids
    ]
    selected = receipts[-1].assertion.selected_candidate_id if receipts else None
    if selected is None or not selected.startswith(PAPER_CANDIDATE):
        return None
    return selected.removeprefix(PAPER_CANDIDATE)


def paper_statements(events) -> tuple[tuple[str, str], ...]:
    """The researcher's words a paper citation may quote, each with its event: their
    request and answers (`researcher_text`), each registered request file's item
    text (#420 SP3), and a paper Review's amended reference, which planning types
    again (ADR 0049 re-entry). The plan's ``cited_paper`` quote and every paper
    Tool's quote are checked against the same words."""
    statements = []
    for event in events:
        payload = event["payload"]
        if (text := researcher_text(event)) is not None:
            statements.append((event["event_id"], text))
        elif event["event_type"] == "InvestigationRequestRegistered":
            statements.extend(
                (event["event_id"], item["request_text"])
                for item in payload["request"]["items"]
            )
        elif (
            event["event_type"] == "PaperReviewAnswered"
            and payload["response"]["action"] == "amend"
            and payload["response"]["field"] == "paper_reference"
        ):
            statements.append((event["event_id"], payload["response"]["value"]))
    return tuple(statements)


def paper_words(events) -> tuple[str, ...]:
    """The researcher's words a paper citation may quote (``paper_statements``)."""
    return tuple(text for _, text in paper_statements(events))


def paper_tool_values(deps, name):
    """A paper Tool's server-bound values now, or the registry reason it waits.

    The paper Tools are offered from the plan stage, so the paper can be found and
    retrieved before the plan is recorded (owner redirect, 2026-10-29), and in the
    paper stage for each cited paper the plan declares and has not settled.
    """
    view = current_planning(deps.journey)
    if view.hitl.pending_requests:
        raise PreconditionUnmet(
            "planning_question_pending", view.hitl.pending_requests[0].request_event_id
        )
    papers = view.papers
    pending = next((r for r in papers.requests if r.pending), None)
    if pending is not None:
        raise PreconditionUnmet("paper_review_pending", pending.request_event_id)
    if view.plan is None:
        # Before the plan: a search names the cited paper's key the plan declares.
        to_find = _keys_to_find_before_plan(papers)
        if name == "find_paper":
            return _BOUND_NOTHING
    else:
        if not cited_papers(view):
            raise PreconditionUnmet("paper_not_cited")
        # Each call names a cited paper of the plan, by membership.
        to_find = frozenset(citations_to_find(view))
        if not to_find:
            raise PreconditionUnmet("cited_papers_settled")
        if name == "find_paper":
            return SimpleNamespace(
                constants={}, enums={"subject_key": citations_to_find(view)}
            )
    if name == "retrieve_paper":
        # A retrieval serves its search's cited paper: only a search for one still
        # to find is offered.
        searches = tuple(
            s.source_event_id
            for s in papers.searches
            if s.tool_input.subject_key in to_find
        )
        if not searches:
            raise PreconditionUnmet("paper_search_absent")
        return SimpleNamespace(constants={}, enums={"search_event_id": searches})
    raise ValueError(f"no paper Tool {name!r}")


def paper_tool_errors(deps, action) -> tuple[str, ...]:
    """A paper call's domain refusals before contact: the row is on the page it
    cites, the researcher's own selection binds it, and the quote stands in the
    researcher's words (ADR 0049 membership)."""
    arguments = action.arguments
    if action.tool_name == "retrieve_paper":
        reason = arguments["reason"]
        if not reason.strip() or reason.strip() != reason:
            return ("paper_reason_invalid",)
        search = run_papers(deps).search(arguments["search_event_id"])
        row = search.result.row(arguments["candidate_id"])
        none = arguments["candidate_id"] == NO_PAPER
        if row is None and not none:
            return ("paper_not_on_search_page",)
        if row is not None and not (row.pmcid or row.doi or row.pmid):
            # The download takes one of its ids: nothing to download.
            return ("paper_row_without_id",)
        selected = _selected_candidate(deps, search.source_event_id)
        if selected is not None and selected != arguments["candidate_id"]:
            return ("researcher_selected_other_paper",)
    if not quoted(paper_words(planning_evidence(deps.journey)), arguments["quote"]):
        return ("paper_quote_ungrounded",)
    return ()


def paper_refusal_facts(deps, action, code) -> dict:
    """The typed facts a paper refusal's sentence names."""
    if code == "researcher_selected_other_paper":
        return {
            "selected": _selected_candidate(deps, action.arguments["search_event_id"])
        }
    return {}


def _search(adapter: PaperSourceAdapter, tool_input: PaperSearchInput):
    """One Europe PMC search page: the rows copied, or Europe PMC's own answer about
    the query with another status."""
    status_code, page = search_europe_pmc(
        adapter,
        tool_input.query,
        tool_input.page_size,
        tool_input.cursor_mark,
        tool_input.sort,
    )
    if page is None:
        return PaperSearchResult(
            status="answered",
            summary=f"Europe PMC answered the query with HTTP {status_code}; its own "
            "words are in source_answers.",
        )
    return PaperSearchResult(
        status="observed",
        summary=f"Europe PMC lists {page.hit_count} papers for the query; this page "
        f"shows {len(page.rows)} of them, in Europe PMC's order.",
        hit_count=page.hit_count,
        next_cursor_mark=page.next_cursor_mark,
        rows=tuple(
            PaperSearchRow(
                candidate_id=row.candidate_id,
                doi=row.ids.doi,
                pmid=row.ids.pmid,
                pmcid=row.ids.pmcid,
                title=row.ids.title,
                authors=row.ids.authors,
                journal=row.ids.journal,
                year=row.ids.year,
                open_access=row.ids.open_access,
            )
            for row in page.rows
        ),
    )


SearchQuery = Annotated[
    str,
    Field(
        min_length=1,
        max_length=1000,
        description="Europe PMC search syntax, sent as you write it: for example "
        'DOI:"<the DOI>", PMCID:<the PMCID>, EXT_ID:<the PMID> AND SRC:MED, or '
        'TITLE:"<words of the title>" AND AUTH:"<an author\'s surname>" AND '
        "PUB_YEAR:<the year>; add AND OPEN_ACCESS:y to list only open copies.",
    ),
]
PaperQuote = Annotated[
    str,
    Field(
        min_length=1,
        max_length=300,
        description="The researcher's exact words that cite the paper, copied on whole "
        "words from their request or answer.",
    ),
]


def find_paper(
    ctx: ToolContext,
    subject_key: Annotated[
        str,
        Field(
            pattern=r"^[a-z][a-z0-9_]{0,79}$",
            description="A lowercase snake-case key for the cited paper this search "
            "is for: the subject_key its cited_paper requirement has, or will have "
            "when you record the plan.",
        ),
    ],
    query: SearchQuery,
    quote: PaperQuote,
    page_size: Annotated[
        int,
        Field(
            ge=1,
            le=MAX_SEARCH_PAGE_SIZE,
            description="Europe PMC's pageSize: rows on this page (1-100).",
        ),
    ] = SEARCH_PAGE_SIZE,
    cursor_mark: Annotated[
        str,
        Field(
            min_length=1,
            max_length=1000,
            description='Europe PMC\'s cursorMark: "*" for the first page, or the '
            "next_cursor_mark a page returned.",
        ),
    ] = "*",
    sort: Annotated[
        str | None,
        Field(
            min_length=1,
            max_length=100,
            description='Europe PMC\'s sort, for example "CITED desc" or "P_PDATE_D '
            "desc\"; omitted, Europe PMC's relevance order.",
        ),
    ] = None,
) -> dict[str, Any]:
    """Search Europe PMC for the paper the researcher's words cite, with Europe PMC's
    own query syntax and paging: every row copied as Europe PMC lists it, with its
    hit_count and next_cursor_mark. Nothing is ranked or cut: which row is the paper
    is your choice, retrieved with `retrieve_paper`.
    """
    from bio_neb.reference_discovery.reference_tools import record_source_fact

    tool_input = PaperSearchInput(
        subject_key=subject_key,
        query=query,
        quote=quote,
        page_size=page_size,
        cursor_mark=cursor_mark,
        sort=sort,
    )
    return record_source_fact(
        ctx,
        "find_paper",
        "find_paper",
        PaperSearchRecorded,
        tool_input,
        lambda: _search(ctx.deps.paper_adapter, tool_input),
        adapter="europe_pmc",
    )


def _intake_id(papers: RunPapers, subject_key: str) -> str:
    """The intake a retrieval for this cited paper settles: the one a paper Review
    sent back to planning (an amended or retried citation, ADR 0049 re-entry), else
    a new one."""
    searched = {s.source_event_id: s.tool_input.subject_key for s in papers.searches}
    due = {i.intake_id for i in papers.references_due}
    return (
        next(
            (
                r.intake_id
                for r in reversed(papers.retrievals)
                if searched[r.tool_input.search_event_id] == subject_key
                and r.intake_id in due
            ),
            None,
        )
        or new_intake_id()
    )


def _settle(activity, row: PaperSearchRow | None, tool_input, intake_id, contacted):
    """Record the retrieved paper, or refuse it and ask its Review: (status, code,
    the paper's or the Review's event)."""
    download, stop, contacts = contacted

    def refuse(code: str, detail: str):
        return (
            "refused",
            code,
            refuse_paper(
                activity.record,
                intake_id=intake_id,
                origin="citation",
                code=code,
                detail=detail,
                reference=tool_input.reference or tool_input.quote,
            ),
        )

    if row is None:
        return refuse(
            "paper_not_found",
            f"the search {tool_input.search_event_id} lists no row for the citation",
        )
    if stop is not None:
        # A stop after the article keeps the paper with what it holds; a stop
        # before it refuses the paper. No GET follows a stop.
        download = stop.download
        held = download.stop
        if download.article is None:
            return refuse(
                "paper_download_stopped",
                f"{held.operation} on {held.host}: {held.failure_class}: {held.detail}",
            )
    if download.resolved is None:
        return refuse("paper_not_found", download.note)
    if not download.resolved.pmcid:
        return refuse("paper_not_open_access", download.note)
    if download.article is None:
        return refuse("paper_article_missing", "no host holds the article's open copy")
    recorded = record_paper(
        activity.record,
        lambda fact, content, media: activity.record_with_artifact(
            fact, content, media_type=media
        ),
        intake_id=intake_id,
        origin="citation",
        reference=tool_input.candidate_id,
        quote=tool_input.quote,
        ids=row.paper_ids(),
        files=tuple(
            SourceFile(d.name, d.url, d.role, d.media_type, d.content)
            for d in download.documents
        ),
        locations=tuple(d.url for d in download.documents),
        unfetched=tuple(
            UnreadFile(name=u.name, reason=u.reason) for u in download.unfetched
        ),
        contact_event_ids=contacts,
    )
    if recorded is None:
        return refuse("paper_article_missing", "no downloaded document holds text")
    return "recorded", None, recorded


def retrieve_paper(
    ctx: ToolContext,
    search_event_id: Annotated[
        str,
        Field(description="The event id of the find_paper search whose page lists it."),
    ],
    candidate_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=300,
            description="The candidate_id of one row of that page, or none when the "
            "page, after any refinement or paging you chose, holds no row for the "
            "citation: the researcher's paper Review is then asked.",
        ),
    ],
    quote: PaperQuote,
    reason: ChoiceReason,
) -> dict[str, Any]:
    """Download the open copy of one row of a find_paper page and record it as this
    Run's paper. A dead end asks the researcher's paper Review, and the Attempt ends
    there."""
    from bio_neb.agentic_journey.attempt_journey import ToolExecutionFailed

    deps = ctx.deps
    papers = run_papers(deps)
    search = papers.search(search_event_id)
    row = search.result.row(candidate_id)
    tool_input = PaperRetrievalInput(
        search_event_id=search_event_id,
        candidate_id=candidate_id,
        quote=quote,
        reason=reason,
        reference=None if row is None else row.pmcid or row.doi or row.pmid,
    )
    intake_id = _intake_id(papers, search.tool_input.subject_key)
    event = None

    def contact():
        # Inside the open scope: the download's GETs, each on its operation's
        # surface, and the contacts it sent. A stopped download sends nothing more;
        # it is returned, never continued, and the scope closes before the paper's
        # Review is asked (the journal's one failure chain per scope).
        try:
            download, stop = (
                download_paper(tool_input.reference, deps.paper_adapter),
                None,
            )
        except PaperDownloadStopped as stopped:
            download, stop = None, stopped
        return download, stop, scope_contact_event_ids()

    def stage():
        nonlocal event
        activity = deps.context.activity.start_child(
            adapter="europe_pmc", stage="retrieve_paper"
        )
        try:
            contacted = (
                # The page holds no row for the citation: nothing to download.
                (None, None, ())
                if row is None
                else observed_source_operation(
                    activity,
                    deps.context.attempt_id,
                    ctx.tool_call_id,
                    "retrieve_paper",
                    contact,
                )()
            )
            status, code, outcome = _settle(
                activity, row, tool_input, intake_id, contacted
            )
            event = activity.record(
                PaperRetrievalRecorded(
                    attempt_id=deps.context.attempt_id,
                    tool_input=tool_input,
                    intake_id=intake_id,
                    status=status,
                    summary=(
                        f"The paper is recorded with {len(outcome['payload']['documents'])} "
                        "documents."
                        if status == "recorded"
                        else f"The paper was not retrieved ({code}); the researcher's "
                        "paper Review is asked."
                    ),
                    paper_event_id=outcome["event_id"]
                    if status == "recorded"
                    else None,
                    review_request_event_id=None
                    if status == "recorded"
                    else outcome["event_id"],
                    code=code,
                )
            )
            activity.succeeded(event)
        except Exception as error:
            failed = activity.record(
                ToolExecutionFailed(
                    attempt_id=deps.context.attempt_id,
                    tool_name="retrieve_paper",
                    error_type=type(error).__name__,
                )
            )
            deps.context.failure_source.capture(failed)
            activity.failed(failed)
            raise
        return event

    return deps.observed_tool_session.execute(
        ctx.tool_call_id,
        stage,
        ToolExecutionAdapter(
            identity="europe_pmc",
            stage_result_event=lambda result: event,
            artifact_digests=lambda result: (),
        ),
    )


def recorded_arguments(tool_name: str, fact: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments a paper call's recorded fact was produced from (replay, ADR
    0034)."""
    tool_input = fact["payload"]["tool_input"]
    if tool_name == "find_paper":
        return {
            key: tool_input[key]
            for key in (
                "subject_key",
                "query",
                "quote",
                "page_size",
                "cursor_mark",
                "sort",
            )
        }
    if tool_name == "retrieve_paper":
        return {
            key: tool_input[key]
            for key in ("search_event_id", "candidate_id", "quote", "reason")
        }
    raise ValueError(f"no paper Tool {tool_name!r}")


def admitted_arguments(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The admitted call's arguments with each omitted Europe PMC option at its
    default."""
    if tool_name == "find_paper":
        return {
            "page_size": SEARCH_PAGE_SIZE,
            "cursor_mark": "*",
            "sort": None,
            **arguments,
        }
    return dict(arguments)


_PAPER_REVIEW_PENDING = ToolPrecondition(
    "paper_review_pending",
    "paper Review",
    "A paper Review is open for the researcher; this Attempt ends there and resumes "
    "after the answer.",
)
_NOT_CITED = ToolPrecondition(
    "paper_not_cited",
    "proposal plan",
    "The recorded plan declares no cited_paper; no paper is to find.",
)
_CITED_SETTLED = ToolPrecondition(
    "cited_papers_settled",
    "cited papers",
    "Every cited_paper of the plan is recorded or dropped; no paper is left to find.",
)
_QUOTE_UNGROUNDED = ToolPrecondition(
    "paper_quote_ungrounded",
    "quote",
    "The quote must copy the researcher's exact words that cite the paper, on whole "
    "words, from their request or one of their answers.",
)
_SOURCE_FAILED = (
    "The source gave no usable answer, as the statement says; call it again, or "
    "continue with another Tool callable now."
)


def _summary(payload: Mapping[str, Any]) -> str:
    return payload["result"]["summary"]


def _retrieval(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in ("paper_event_id", "code")
        if payload[key] is not None
    }


# This domain's Tool rows (ADR 0040): handler, row, bound values and refusal facts
# sit together, and the loop assembles every domain's rows into the one inventory
# (`tool_execution.tool_row_modules`).
TOOL_ROWS = (
    function_tool_contract(
        "find_paper",
        "planning_agent",
        (
            "subject_key: str",
            "query: str (Europe PMC search syntax)",
            "quote: str",
            "page_size: int (1-100, default 25)",
            'cursor_mark: str (default "*")',
            "sort: str | None",
        ),
        "PaperSearchRecorded",
        "europe_pmc",
        "PaperSearchRecorded",
        None,
        None,
        ModelView(
            next_step={
                "observed": "Retrieve the row that is the paper the researcher cites "
                "with `retrieve_paper`, citing this search; when several rows fit "
                "their words or a row disagrees with them, do not retrieve: record "
                "the plan with its cited_paper, and the paper stage asks them with "
                "`request_planning_hitl`. Or read the next page with "
                "next_cursor_mark, or search again with another query.",
                "answered": "Europe PMC answered with its own words about the query, "
                "in source_answers; search again with a corrected query.",
                "*": _SOURCE_FAILED,
            },
            statement=_summary,
            cite="search_event_id",
        ),
        "generic_mechanical_failure_only",
        source_surface=SourceSurface(operations=frozenset({"europe_pmc_search"})),
        stages=("proposal_requirements", "paper"),
        examples=(
            '{"subject_key": "<the cited paper\'s key>", "query": "DOI:\\"<the DOI '
            'the researcher wrote>\\"", "quote": "<their words that cite it>"}',
            '{"subject_key": "<the cited paper\'s key>", "query": "TITLE:\\"<words of '
            'the title>\\" AND AUTH:\\"<an author\'s surname>\\" AND PUB_YEAR:<the '
            'year>", "quote": "<their words that cite it>", "page_size": 10}',
        ),
        # Every page is Europe PMC's answer; no count of rows is a negative result.
        negative_results=(),
        preconditions=(
            CORRECTION_PENDING,
            QUESTION_PENDING,
            _PAPER_REVIEW_PENDING,
            _NOT_CITED,
            _CITED_SETTLED,
            _QUOTE_UNGROUNDED,
        ),
    ),
    function_tool_contract(
        "retrieve_paper",
        "planning_agent",
        (
            "search_event_id: str",
            "candidate_id: str",
            "quote: str",
            "reason: str",
        ),
        "PaperRetrievalRecorded",
        "europe_pmc",
        "PaperRetrievalRecorded",
        None,
        None,
        ModelView(
            next_step={
                "recorded": "The paper is recorded as this Run's paper; continue with "
                "the next cited paper, or the stage closes and the reference work "
                "opens.",
                "refused": "The paper could not be retrieved, as the statement says; "
                "the researcher's paper Review is asked and the Attempt ends there.",
            },
            status=lambda payload: payload["status"],
            statement=lambda payload: payload["summary"],
            result=_retrieval,
        ),
        "generic_mechanical_failure_only",
        # The download's GETs keep their operations' surfaces; one paper takes at most
        # the download's 16 contacts.
        source_surface=PAPER_DOWNLOAD_SURFACES["download_paper"],
        stages=("proposal_requirements", "paper"),
        examples=(
            '{"search_event_id": "<a find_paper search>", "candidate_id": "<one '
            'row\'s candidate_id>", "quote": "<the researcher\'s words that cite '
            'it>", "reason": "The row\'s DOI is the one the researcher wrote."}',
        ),
        preconditions=(
            CORRECTION_PENDING,
            QUESTION_PENDING,
            _PAPER_REVIEW_PENDING,
            _NOT_CITED,
            _CITED_SETTLED,
            ToolPrecondition(
                "paper_search_absent",
                "paper search",
                "Retrieve a paper after `find_paper` lists it; search for it first.",
                met_by=("find_paper",),
            ),
            ToolPrecondition(
                "paper_not_on_search_page",
                "candidate_id",
                "Name the candidate_id of a row of the search page you cite; read "
                "another page with its next_cursor_mark, or search again.",
            ),
            ToolPrecondition(
                "paper_row_without_id",
                "candidate_id",
                "That row names no PMCID, DOI or PMID to download; choose another row, "
                "or search again.",
            ),
            ToolPrecondition(
                "researcher_selected_other_paper",
                "candidate_id",
                "The researcher selected {selected}; retrieve that row, or ask them "
                "with `request_planning_hitl` when it disagrees with what Europe PMC "
                "lists.",
            ),
            _QUOTE_UNGROUNDED,
            ToolPrecondition(
                "paper_reason_invalid",
                "reason",
                "State the reason in at most 2000 characters, not blank and without "
                "surrounding whitespace.",
            ),
        ),
    ),
)


__all__ = [
    "PAPER_CANDIDATE",
    "PAPER_TOOLS",
    "TOOL_ROWS",
    "PaperStageOutput",
    "admitted_arguments",
    "citation_state",
    "cited_papers",
    "citations_to_find",
    "find_paper",
    "open_paper_search",
    "paper_findings",
    "paper_question",
    "paper_stage_output",
    "paper_statements",
    "paper_words",
    "paper_refusal_facts",
    "paper_tool_errors",
    "paper_tool_values",
    "recorded_arguments",
    "retrieve_paper",
    "run_papers",
]
