"""One eval case record, the shared question bank, and the compiled case a run scores against.

A case file holds the public input, the prepared Review answers and the questions for each
layer it covers; shared questions are named by `ref` and the bank supplies their text and
check. Compilation inlines the bank so every compiled case is self-contained. Run

    python -m evals.case compile          # write evals/case.schema.json and evals/compiled/
    python -m evals.case compile --check  # exit 1 when a written file is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EVALS = Path(__file__).resolve().parent
CASES = EVALS / "cases"
COMPILED = EVALS / "compiled"
BANK = EVALS / "question-bank.yaml"
SCHEMA = EVALS / "case.schema.json"

# The checks the scorer implements (evals/score.py); a question names exactly one.
Check = Literal[
    "kinds_present",
    "knowledge_states",
    "first_attempt",
    "no_invented_identifier",
    "terminal",
    "reviews_raised",
    "plan_value",
    "taxon_resolved",
    "reference_verified",
    "reference_from_discovery",
    "investigation_items",
    "ploidy_notice",
]
# The product hard gates every planning run records beside its questions.
Gate = Literal["accepted", "usable"]
# Failure labels, in precedence order.
FailureLabel = Literal[
    "transport",
    "truncation",
    "empty_or_misplaced_output",
    "schema",
    "hallucinated_tool",
    "unauthorized_action",
    "fabrication",
    "grounding",
    "value_not_atomic",
    "repair_stagnation",
    "repeated_call",
    "wrong_tool",
    "premature_final",
    "budget_exhausted",
    "missed_clarification",
    "wrong_terminal",
    "over_clarification",
    "semantic_binding",
    "semantic_coverage",
    "semantic_other",
]
_SENTENCE_END = (".", "?", "!")


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Question(_Record):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9-]*$")
    ask: str
    check: Check
    spec: dict[str, Any] = Field(default_factory=dict)
    gate: Literal["hard", "graded"]
    note: str | None = None


class BankQuestion(Question):
    """A shared question; `params` names the spec keys each citing case supplies."""

    params: tuple[str, ...] = ()


class QuestionRef(_Record):
    ref: str
    params: dict[str, Any] = Field(default_factory=dict)


class PreparedAnswer(_Record):
    """How the runner answers one planning Review, as the researcher would."""

    review: str = Field(description="The Review's subject_key, e.g. analysis_recipe.")
    answer: str = Field(description="The display label of the candidate to select.")
    reason: str


class BadAnswer(_Record):
    summary: str
    must_fail: tuple[str, ...] = Field(min_length=1)
    label: FailureLabel


class Public(_Record):
    """Everything the model may see: the request text and the registered read roles."""

    research_request: str
    fastq_roles: tuple[str, ...] = Field(min_length=1)

    @field_validator("research_request")
    @classmethod
    def unwrapped(cls, value: str) -> str:
        # Product grounding is exact, so a line break inside a sentence breaks a quote.
        for line in value.split("\n")[:-1]:
            if not line.rstrip().endswith(_SENTENCE_END):
                raise ValueError(f"line break inside a sentence after: {line[-40:]!r}")
        return value


class Donor(_Record):
    path: str
    case_id: str
    version: int


class PlanningBlock(_Record):
    prepared_answers: tuple[PreparedAnswer, ...]
    questions: tuple[Union[QuestionRef, Question], ...] = Field(min_length=1)
    good_answer: str
    bad_answers: tuple[BadAnswer, ...] = ()


class EvalCase(_Record):
    schema_version: Literal["bio-neb.eval-case/1"]
    case_id: str = Field(pattern=r"^s[0-9]+-[A-Za-z0-9-]+$")
    family: str
    scenario: str
    # The dev subset holds train cases only; validation and sealed cases never come here.
    split: Literal["train"]
    level: str
    evidence_shape: str
    donor: Donor
    refresh: tuple[str, ...] = Field(
        min_length=1, description="What changed from the donor and why."
    )
    public: Public
    planning: PlanningBlock

    @model_validator(mode="after")
    def no_label_leak(self) -> EvalCase:
        request = self.public.research_request
        for label in (self.case_id, self.family, self.scenario, self.split):
            if label in request:
                raise ValueError(f"case label {label!r} leaks into the request")
        return self


class CompiledLayer(_Record):
    prepared_answers: tuple[PreparedAnswer, ...]
    gates: tuple[Gate, ...]
    questions: tuple[Question, ...]
    good_answer: str
    bad_answers: tuple[BadAnswer, ...]


class CompiledCase(_Record):
    schema_version: Literal["bio-neb.eval-case-compiled/1"] = (
        "bio-neb.eval-case-compiled/1"
    )
    case_id: str
    family: str
    scenario: str
    split: Literal["train"]
    level: str
    evidence_shape: str
    donor: Donor
    public: Public
    planning: CompiledLayer


def load_bank(path: Path = BANK) -> dict[str, BankQuestion]:
    rows = [
        BankQuestion.model_validate(row) for row in yaml.safe_load(path.read_text())
    ]
    bank = {row.id: row for row in rows}
    if len(bank) != len(rows):
        raise ValueError("question bank ids must be unique")
    return bank


def load_case(path: Path) -> EvalCase:
    case = EvalCase.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if path.stem != case.case_id:
        raise ValueError(f"{path.name} holds case {case.case_id}")
    return case


def _resolve(item: QuestionRef | Question, bank: dict[str, BankQuestion]) -> Question:
    if isinstance(item, Question):
        if item.id in bank:
            raise ValueError(f"case question {item.id} shadows a bank question")
        return item
    shared = bank.get(item.ref)
    if shared is None:
        raise ValueError(f"unknown bank question {item.ref}")
    if set(item.params) != set(shared.params):
        raise ValueError(
            f"{item.ref} needs params {sorted(shared.params)}, got {sorted(item.params)}"
        )
    fields = shared.model_dump(exclude={"params"})
    return Question.model_validate({**fields, "spec": {**shared.spec, **item.params}})


def compile_case(case: EvalCase, bank: dict[str, BankQuestion]) -> CompiledCase:
    questions = tuple(_resolve(item, bank) for item in case.planning.questions)
    ids = [question.id for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{case.case_id}: question ids repeat")
    known = {*ids, *Gate.__args__}
    for bad in case.planning.bad_answers:
        unknown = set(bad.must_fail) - known
        if unknown:
            raise ValueError(f"{case.case_id}: bad answer names {sorted(unknown)}")
    layer = case.planning.model_dump(exclude={"questions"})
    return CompiledCase.model_validate(
        {
            **case.model_dump(exclude={"schema_version", "refresh", "planning"}),
            "planning": {**layer, "gates": Gate.__args__, "questions": questions},
        }
    )


def case_paths(layer: str = "planning") -> list[Path]:
    return sorted((CASES / layer).glob("*.yaml"))


def load_compiled(case_id: str, layer: str = "planning") -> CompiledCase:
    path = COMPILED / layer / f"{case_id}.json"
    return CompiledCase.model_validate_json(path.read_text(encoding="utf-8"))


def _dump(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def expected_outputs() -> dict[Path, str]:
    """Every generated file and its current text."""
    bank = load_bank()
    outputs = {SCHEMA: _dump(EvalCase.model_json_schema())}
    for path in case_paths():
        compiled = compile_case(load_case(path), bank)
        target = COMPILED / "planning" / f"{compiled.case_id}.json"
        outputs[target] = _dump(compiled.model_dump(mode="json"))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.case")
    parser.add_argument("command", choices=["compile"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    outputs = expected_outputs()
    stale = sorted(set(COMPILED.rglob("*.json")) - set(outputs))
    for path, text in outputs.items():
        if path.exists() and path.read_text(encoding="utf-8") == text:
            continue
        if args.check:
            stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for path in stale:
        print(f"stale: {path.relative_to(EVALS.parent)}", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
