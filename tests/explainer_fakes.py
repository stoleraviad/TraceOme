"""A scripted report explainer for flows that reach a published report.

Every published report is explained by default (#460 Q5 b), so a flow that
advances past publication binds this explainer as its report model. Its drafts
name each row's claim as the task shows it and every qualification once.
"""

from __future__ import annotations

import json

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.journey import canonical_json
from tests.model_endpoint_fakes import ScriptedModelEndpoint


def _sent_task(messages) -> dict:
    """The explanation task one Attempt was sent: a member's or the combination's."""
    from pydantic_ai.messages import UserPromptPart

    for message in messages:
        for part in getattr(message, "parts", ()):
            if isinstance(part, UserPromptPart) and '"task":' in str(part.content):
                return json.loads(part.content)
    raise AssertionError("the explanation Agent was sent no task")


def _publication(task: dict) -> str:
    return next(
        item["event_id"]
        for item in task["evidence"]
        if item["event_type"] == "GroundedReportPublished"
    )


def _plain(word: str) -> str:
    return word.replace("_", " ")


def _called(row: dict) -> str:
    return f" with genotype {row['genotype']}" if row.get("genotype") else ""


def member_draft(task: dict) -> dict:
    """A why naming each row's claim in plain words (#460: no state word as
    written) and each genotype a row carries, listed with its quote (#473), and
    one limitation naming every qualification the member was given."""
    reference = _publication(task)
    return {
        "why": f"{task['targets'][0]['subject']}: "
        + "; ".join(
            f"row {row['number']} reads {_plain(row['claim'])}{_called(row)}"
            for row in task["targets"]
        )
        + ".",
        "genotypes": [
            {
                "target": row["number"],
                "genotype": row["genotype"],
                "quote": f"genotype {row['genotype']}",
            }
            for row in task["targets"]
            if row.get("genotype")
        ],
        "why_references": [reference],
        "limitations": [
            {
                "qualifications": [
                    item["qualification"] for item in task["qualifications"]
                ],
                "text": "This request's qualifications apply as the report states.",
                "references": [reference],
            }
        ]
        if task["qualifications"]
        else [],
    }


def combination_draft(task: dict) -> dict:
    """A conclusion naming every request with its member's claim words in plain
    words, or not explained."""
    reference = _publication(task)
    return {
        "conclusion": " ".join(
            f"{entry['subject']}: "
            + (
                ", ".join(_plain(word) for word in entry["claim_words"])
                if entry["status"] == "explained"
                else "not explained"
            )
            + "."
            for entry in task["requests"]
        ),
        "conclusion_references": [reference],
        "would_change": "A call at a request's position would change its answer.",
        "limitations": [
            {
                "qualifications": [
                    item["qualification"] for item in task["qualifications"]
                ],
                "text": "These qualifications apply to every request.",
                "references": [reference],
            }
        ]
        if task["qualifications"]
        else [],
        "not_done": "Every request of the basis is listed with its member's claims.",
        "genotypes": [],
        "title": "Each request of the report with its member's claims",
    }


class ExplainerModel:
    """A scripted explanation Agent, one script per step.

    A member's step is named by its request's subject and the combination's by
    ``"combination"``. Each step's searches run first, then its drafts; once
    they run out it records ``member_draft`` or ``combination_draft``. Per step,
    ``tasks`` keeps what each request sent and ``feedback`` the Tool results and
    corrections it read back; ``offered`` keeps the Tools offered.
    """

    def __init__(self, *, drafts=None, searches=None) -> None:
        self.drafts = {key: list(value) for key, value in (drafts or {}).items()}
        self.searches = {key: list(value) for key, value in (searches or {}).items()}
        self.tasks: dict[str, list[dict]] = {}
        self.feedback: dict[str, list[list]] = {}
        self.offered: dict[str, list[set[str]]] = {}
        self.views: dict[str, list[dict]] = {}
        self.order: list[str] = []
        self.model = ScriptedModelEndpoint(self.respond)

    @staticmethod
    def step(task: dict) -> str:
        return task["targets"][0]["subject"] if "targets" in task else "combination"

    def respond(self, messages, info):  # noqa: ANN001, ANN201
        from pydantic_ai.messages import RetryPromptPart, ToolReturnPart

        task = _sent_task(messages)
        step = self.step(task)
        self.order.append(step)
        self.tasks.setdefault(step, []).append(task)
        self.offered.setdefault(step, []).append(
            {tool.name for tool in (*info.function_tools, *info.output_tools)}
        )
        # The request's view: the Tools callable now and those waiting.
        self.views.setdefault(step, []).append(
            json.loads(info.instructions.rpartition("\n")[2])
        )
        self.feedback.setdefault(step, []).append(
            [
                part.content
                for part in getattr(messages[-1], "parts", ())
                if isinstance(part, ToolReturnPart | RetryPromptPart)
            ]
        )
        call = len(self.order)
        if self.searches.get(step):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "search_journey",
                        canonical_json(self.searches[step].pop(0)(task)),
                        tool_call_id=f"search-{call}",
                    )
                ]
            )
        member = step != "combination"
        script = self.drafts.get(step)
        draft = (
            script.pop(0)(task)
            if script
            else (member_draft(task) if member else combination_draft(task))
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "record_explanation" if member else "record_combined_answer",
                    canonical_json(draft),
                    tool_call_id=f"explanation-{call}",
                )
            ]
        )


def explained_by_default(explainer: ExplainerModel | None = None):
    """Bind the scripted explainer, on the fixture's model credential, for a
    flow whose coordinator publishes a report: it explains it by default (#460)."""
    from bio_neb.application.advancement import use_run_advancement_dependencies
    from tests.observation_input_fakes import TABLE_READING_ENVIRON

    return use_run_advancement_dependencies(
        environ=TABLE_READING_ENVIRON,
        report_model_factory=(explainer or ExplainerModel()).model,
    )
