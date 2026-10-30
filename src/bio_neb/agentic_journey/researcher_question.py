"""The researcher question a post-approval role asks (owner, #447 Q8, "as much of
exiting tool as we can").

One shared Tool, ``ask_researcher``: a role that declares its row (the table
reader, the report explainer, the paper reader) offers it, and the Agent assembly
binds the one handler, as it does the Journey search. The call records
``ResearcherQuestionAsked``: what the model found so far, the conflict that stops
it and its one intent question. The Attempt then stops for the existing Agent work
Review (cause ``researcher_question``), which shows the three; the researcher's
continue instruction is the answer, a User assertion, and the same item, chunk or
member starts again as a new Attempt with a fresh count of failed turns.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictStr

from bio_neb.agentic_journey.tool_execution import (
    QUESTION_OPENED,
    FunctionToolContract,
    ToolPrecondition,
    asked_view,
    function_tool_contract,
)
from bio_neb.contracts import FrozenContract

ASK_RESEARCHER = "ask_researcher"
_TEXT = Field(min_length=1, max_length=2000)


class ResearcherQuestionAsked(FrozenContract):
    """The ask Tool's own result: the question the Agent work Review shows."""

    event_type: Literal["ResearcherQuestionAsked"] = "ResearcherQuestionAsked"
    attempt_id: StrictStr
    agent_role: StrictStr
    found_so_far: StrictStr = _TEXT
    conflict: StrictStr = _TEXT
    question: StrictStr = _TEXT


def researcher_question_row(
    agent_role: str, stages: tuple[str, ...]
) -> FunctionToolContract:
    """The role's ``ask_researcher`` row: it opens a question, which ends the
    Attempt; its purpose is the role's own intent-only instruction block."""
    return function_tool_contract(
        ASK_RESEARCHER,
        agent_role,
        ("found_so_far: str", "conflict: str", "question: str"),
        "ResearcherQuestionAsked",
        "researcher_question",
        "ResearcherQuestionAsked",
        None,
        None,
        asked_view(),
        "generic_mechanical_failure_only",
        stages=stages,
        examples=(
            '{"found_so_far": "<what your reads or the rows show so far>", '
            '"conflict": "<what stops you, in one sentence>", '
            '"question": "<the one intent question for the researcher>"}',
        ),
        opens_question=True,
        preconditions=(
            QUESTION_OPENED,
            ToolPrecondition(
                "researcher_question_already_asked",
                "researcher question",
                "This Attempt already asked its question; it ends there.",
            ),
        ),
    )
