"""The paper reader's Tools (ADR 0046): each handler beside its registry row (ADR 0040).

Four plain reads of one Run paper (``reader``): ``outline_paper``, ``grep_paper``,
``read_paper`` and ``filter_paper_rows``. The Paper Reader reads with them, and the
Investigation Planner reads its member's paper with the same four (``reading_tool_rows``,
#485 Q2 a). Every call goes through the observed Tool session
and records the one tool-owned fact ``PaperReadingRecorded``; the model reads the row's
model view of it. A call naming no view or column of the paper is refused at admission
(``reading_admission_errors``); the same read again in one item is refused by the shared
admission's repeat identity, its row's result-shaping arguments aside
(``assignment_repeat_identity``); a search or filter that finds nothing is a lawful
negative result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, Any

from pydantic import Field

from bio_neb.agentic_journey.action_policy import PreconditionUnmet
from bio_neb.agentic_journey.invocation import ActionAdmissionSpec, ToolContext
from bio_neb.agentic_journey.lifecycle import Activity
from bio_neb.agentic_journey.researcher_question import researcher_question_row
from bio_neb.agentic_journey.tool_execution import (
    FunctionToolContract,
    ModelView,
    NegativeResult,
    ObservedToolSession,
    ToolExecutionAdapter,
    ToolPrecondition,
    function_tool_contract,
)
from bio_neb.content import shown_to_model
from bio_neb.paper_reading.contracts import ReadingRefused
from bio_neb.paper_reading.journey import PaperReadingRecorded
from bio_neb.paper_reading.reader import (
    MAX_HITS,
    MAX_READ,
    MAX_ROWS,
    Paper,
    grep,
    outline,
    read,
    rows,
    rows_refusal,
)

ROLE = "paper_reader"
STAGE = "paper_reading"
# The shared admission refuses a read this item already ran with this code (ADR 0044's
# repeat identity, the row's result-shaping arguments aside).
REPEATED_READ = "repeated_read"


@dataclass
class ReadingDependencies:
    """What a reading call reads and records with. The Investigation Planner's own
    dependencies carry the same four, its member's paper among them (#485 Q2 a)."""

    paper: Paper
    attempt_id: str
    parent_activity: Activity
    observed_tool_session: ObservedToolSession


def _recorded(
    context: ToolContext[ReadingDependencies],
    tool_name: str,
    arguments: dict[str, Any],
    reading: Callable[[Paper], Any],
    empty: Callable[[Any], bool] = lambda _result: False,
) -> Any:
    """Run one admitted reading through the observed Tool session and record its fact;
    ``empty`` says the reading found nothing, a lawful negative result."""

    deps = context.deps
    event: dict[str, Any] | None = None

    def operation() -> dict[str, Any]:
        nonlocal event
        result = reading(deps.paper)
        if isinstance(result, ReadingRefused):
            # Admission refuses every call the reader would refuse.
            raise RuntimeError(f"admitted {tool_name} call refused: {result.code}")
        event = deps.parent_activity.record(
            PaperReadingRecorded(
                attempt_id=deps.attempt_id,
                tool_call_id=context.tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
                outcome="nothing_found" if empty(result) else "read",
                # What the model is shown: no raw control character to copy.
                result=shown_to_model(result.model_dump(mode="json")),
            )
        )
        return event["payload"]["result"]

    return deps.observed_tool_session.execute(
        context.tool_call_id,
        operation,
        ToolExecutionAdapter(
            identity="retrieved_paper",
            stage_result_event=lambda _result: event,
            artifact_digests=lambda _result: (),
        ),
    )


def outline_paper(context: ToolContext[ReadingDependencies]) -> Any:
    """Takes no arguments."""

    return _recorded(context, "outline_paper", {}, outline)


def grep_paper(
    context: ToolContext[ReadingDependencies],
    pattern: Annotated[
        str,
        Field(
            min_length=1,
            max_length=500,
            description="A case-insensitive regular expression; text that is no expression is searched as written",
        ),
    ],
    max_hits: Annotated[
        int, Field(ge=1, le=MAX_HITS, description="How many hits come back")
    ] = 10,
) -> Any:
    """Every match is counted; the first max_hits come back."""

    return _recorded(
        context,
        "grep_paper",
        {"pattern": pattern, "max_hits": max_hits},
        lambda paper: grep(paper, pattern, max_hits),
        lambda found: not any(found.matches.values()),
    )


def read_paper(
    context: ToolContext[ReadingDependencies],
    at: Annotated[
        str,
        Field(
            min_length=1,
            description="A section id (s1, ...), a document id, or a span id (main:p41, t1:r3)",
        ),
    ],
    offset: Annotated[
        int | None,
        Field(
            ge=0,
            description="Where to start in the document; the section's start by default",
        ),
    ] = None,
    length: Annotated[
        int, Field(ge=1, le=MAX_READ, description="How many characters to read")
    ] = 3000,
) -> Any:
    """Offsets count characters of the whole document."""

    return _recorded(
        context,
        "read_paper",
        {"at": at, "offset": offset, "length": length},
        lambda paper: read(paper, at, offset, length),
    )


def filter_paper_rows(
    context: ToolContext[ReadingDependencies],
    table_id: Annotated[
        str | None, Field(description="A table id (t1, ...) from outline_paper")
    ] = None,
    where: Annotated[
        dict[str, str] | None,
        Field(description="Column name to the value its cell must hold"),
    ] = None,
    columns: Annotated[
        tuple[str, ...],
        Field(description="Only these columns of each row; all by default"),
    ] = (),
    max_rows: Annotated[
        int, Field(ge=1, le=MAX_ROWS, description="How many rows come back")
    ] = 20,
) -> Any:
    """Columns are named exactly as outline_paper lists them."""

    return _recorded(
        context,
        "filter_paper_rows",
        {
            "table_id": table_id,
            "where": dict(where or {}),
            "columns": list(columns),
            "max_rows": max_rows,
        },
        lambda paper: rows(paper, table_id, where, columns, max_rows),
        lambda found: found.matched == 0,
    )


READING_TOOLS = {
    "outline_paper": outline_paper,
    "grep_paper": grep_paper,
    "read_paper": read_paper,
    "filter_paper_rows": filter_paper_rows,
}


def reading_admission_errors(deps: ReadingDependencies, action: Any) -> tuple[str, ...]:
    """A call naming no section, document, table or column of the paper is refused
    before it runs, with its registry row's reason."""

    # Admission runs this after the Tool's own argument check; an optional argument the
    # call leaves out takes the Tool's default (no table, no filter, every column).
    arguments = action.arguments
    if action.tool_name == "read_paper":
        at = arguments["at"]
        known = at in deps.paper.sections or at in deps.paper.documents
        known = known or at in deps.paper.tables
        if not (known or deps.paper.unit(at) is not None):
            return ("unknown_view",)
    if action.tool_name == "filter_paper_rows":
        refused = rows_refusal(
            deps.paper,
            arguments.get("table_id"),
            arguments.get("where"),
            arguments.get("columns") or (),
        )
        if refused:
            return (refused,)
    return ()


def reading_bound_values(deps: ReadingDependencies, tool_name: str) -> Any:
    """The reading Tools bind no value; a Tool with no admissible call on this paper
    waits with its reason (ADR 0040: ``PreconditionUnmet``)."""

    if tool_name == "filter_paper_rows" and not deps.paper.tables:
        raise PreconditionUnmet(NO_TABLE.code)
    return SimpleNamespace(constants={}, enums={})


READING_ADMISSION = ActionAdmissionSpec(
    validator=reading_admission_errors, repeat_error=REPEATED_READ
)


# ---------------------------------------------------------------------------
# The rows (ADR 0040): this domain's Tools, assembled into the one inventory
# ---------------------------------------------------------------------------

_PAPER_READING = "PaperReadingRecorded"
_NOTHING_FOUND = (NegativeResult(_PAPER_READING, "outcome", ("nothing_found",)),)

# A paper does not change, so in one reading item the same read is refused wherever it
# falls, its result-shaping arguments aside: what happened, then where to go; the
# failure module adds the reply's place in its count.
_REPEATED_READ = ToolPrecondition(
    REPEATED_READ,
    "call",
    "This read already ran in this item with these arguments (how many hits, rows or "
    "characters come back aside), and its result is above. Read another place, or "
    "answer with the item's output Tool.",
)
NO_TABLE = ToolPrecondition(
    "paper_has_no_table",
    "table_id",
    "This paper has no table; read its text with `read_paper` or search it with "
    "`grep_paper`.",
)


def _view(**next_step: str) -> ModelView:
    """What the model is shown of a reading: its outcome and the result as recorded.
    A paper's own column names may be any word, so nothing is scrubbed from it."""

    return ModelView(
        next_step=next_step,
        status=lambda payload: payload["outcome"],
        result=lambda payload: payload["result"],
        scrubbed=False,
    )


def reading_tool_rows(
    role: str,
    stage: str,
    *,
    then: str,
    preconditions: tuple[ToolPrecondition, ...] = (),
) -> tuple[FunctionToolContract, ...]:
    """The four reading Tools' rows for one role's stage: ``then`` is the role's next
    step after a read, and ``preconditions`` the role's own refusals of every read."""

    def row(
        tool_name: str,
        typed_input: tuple[str, ...],
        typed_output: str,
        view: ModelView,
        own: tuple[ToolPrecondition, ...] = (),
        **use: Any,
    ) -> FunctionToolContract:
        return function_tool_contract(
            tool_name,
            role,
            typed_input,
            typed_output,
            "retrieved_paper",
            _PAPER_READING,
            None,
            None,
            view,
            "generic_mechanical_failure_only",
            stages=(stage,),
            preconditions=(*own, *preconditions),
            **use,
        )

    return (
        row(
            "outline_paper",
            (),
            "Outline",
            _view(
                read="Read the sections, tables or spans the question needs with "
                "`read_paper`, filter a table's rows with `filter_paper_rows`, or "
                "search with `grep_paper`."
            ),
            examples=("{}",),
        ),
        row(
            "grep_paper",
            ("pattern: str", "max_hits: int"),
            "GrepResult",
            _view(
                read="Read a hit's paragraph or row with `read_paper` at its span id, "
                "or search again with another pattern.",
                nothing_found="Nothing matched: search another spelling or pattern, "
                "or read the outline's sections.",
            ),
            examples=(
                '{"pattern": "rs\\\\d+"}',
                '{"pattern": "BRCA1|TP53"}',
                '{"pattern": "c\\\\.\\\\d+[ACGT]>[ACGT]", "max_hits": 20}',
            ),
            negative_results=_NOTHING_FOUND,
            result_shaping=("max_hits",),
        ),
        row(
            "read_paper",
            ("at: str", "offset: int | None", "length: int"),
            "ReadResult",
            _view(
                read=f"Cite what you read by its span ids, read on from its end, or {then}."
            ),
            (
                ToolPrecondition(
                    "unknown_view",
                    "at",
                    "No section, document, table or span of this paper has this id; `outline_paper` lists the sections, documents and tables, and every reading result names its spans.",
                ),
            ),
            examples=(
                '{"at": "s2"}',
                '{"at": "main", "offset": 1200, "length": 2000}',
                '{"at": "main:p41"}',
            ),
            result_shaping=("length",),
        ),
        row(
            "filter_paper_rows",
            (
                "table_id: str | None",
                "where: dict[str, str]",
                "columns: list[str]",
                "max_rows: int",
            ),
            "RowsResult",
            _view(
                read="Cite each row by its span id and each value by its column, or "
                "filter again.",
                nothing_found="No row matched: filter by another value or column, or "
                "read the table with `read_paper` at its table id.",
            ),
            (
                NO_TABLE,
                ToolPrecondition(
                    "unknown_table",
                    "table_id",
                    "No table of this paper has this id; `outline_paper` lists the tables.",
                ),
                ToolPrecondition(
                    "rows_unscoped",
                    "table_id",
                    "Name a table id, or filters naming the columns to search.",
                ),
                ToolPrecondition(
                    "unknown_filter_column",
                    "where",
                    "A filtered column is not a column of the table, or no table has every filtered column; `outline_paper` lists each table's columns.",
                ),
                ToolPrecondition(
                    "unknown_column",
                    "columns",
                    "A kept column is not a column of the table, or no table has every named column; `outline_paper` lists each table's columns.",
                ),
            ),
            examples=(
                '{"table_id": "t2", "where": {"POS": "1234567"}}',
                '{"where": {"CHROM": "7", "POS": "1234567"}}',
                '{"table_id": "t1", "columns": ["Gene", "Variant"], "max_rows": 50}',
            ),
            negative_results=_NOTHING_FOUND,
            result_shaping=("max_rows",),
        ),
    )


TOOL_ROWS = (
    *reading_tool_rows(
        ROLE,
        STAGE,
        then="answer with the item's output Tool",
        preconditions=(_REPEATED_READ,),
    ),
    # The shared researcher question; the assembly binds its handler.
    researcher_question_row(ROLE, (STAGE,)),
)
