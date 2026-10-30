"""Versioned instruction content shared by role builders and retained Attempts.

A role's version is one block source of named blocks; a form renders those blocks
as the text a model reads, so a rule changes in one place for every model (R5).
Every word a model reads about its role lives here: the instructions, the
sentences the role's own message states and each function Tool's purpose. This
module is the only reader of a model's form (owner, #439 A3).
"""

import functools
import re
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from importlib.resources import files
from itertools import groupby
from string import Formatter

from bio_neb.agentic_journey.model_settings import (
    BUILD_MODEL_REGISTRY,
    ModelEndpointPolicy,
)
from bio_neb.agentic_journey.role_declaration import ROLES

# The forms a role's selected version gives words of its own
# (``<role>.<version>.<form>.toml``). A declared form text that is missing, or one
# no declaration names, cannot render: Qwen never reads Nemotron's words by a
# missing file (I-65).
_FORM_TEXTS = {
    "planning_agent": frozenset({"qwen_3_5"}),
    "investigation_planner": frozenset({"qwen_3_5"}),
}
# A form is named after its model; each registry entry names its own
# (``ModelEndpointPolicy.instruction_form``).
_FORMS = frozenset(
    entry.instruction_form for entry in BUILD_MODEL_REGISTRY.entries.values()
)
# What a block feeds besides the instruction text: a sentence the role's own
# message states, or a function Tool's purpose (the block named after its Tool).
_USES = ("message", "tool")
# The sentences each role's own message states, with the fields the role fills; its declaration
# names which is its job. A form may name fewer fields (Qwen's job names none). Rendering refuses
# any other message, so a missing or broken sentence is refused before the Attempt's model is
# acquired (I-65).
_MESSAGE_FIELDS: dict[str, dict[str, frozenset[str]]] = {
    "planning_agent": {
        "objective": frozenset(),
        "job": frozenset(),
        "correction_task": frozenset(),
        "target_answer_task": frozenset(),
    },
    "investigation_planner": {
        "job": frozenset({"lookup"}),
        "job_lookup_position": frozenset(),
        "job_lookup_rsid": frozenset(),
        "next_step": frozenset({"assembly", "lookup"}),
        # The replaced target's last lookup, marked replaced.
        "replaced_target": frozenset({"by", "call", "result", "status"}),
    },
    "table_reader": {"job": frozenset()},
    "report_explainer": {"job": frozenset(), "combination_job": frozenset()},
    "paper_reader": {"job": frozenset()},
}


class InstructionsUnrenderable(ValueError):
    """A role's selected instructions cannot render: a missing or broken block
    source or form text, or an unknown form, message or Tool purpose.

    Configuration refuses it before any model is acquired (I-65).
    """

    code = "instructions_unrenderable"

    def __init__(self, detail: str):
        super().__init__(detail)
        self.violations = (detail,)


@dataclass(frozen=True)
class InstructionBlock:
    """One named rule or example; ``stage`` is None for text every stage reads.

    ``text`` is written as every form shows it, with its own line breaks. A block
    with a ``use`` is not instruction text: it is a message sentence or a Tool's
    purpose.
    """

    name: str
    stage: str | None
    text: str
    use: str | None = None


@dataclass(frozen=True)
class RoleInstructions:
    """One role's version in one form; its artifact digest verifies the bytes.

    ``text`` is the instruction text the Attempt records, and with the identity it
    is what two renderings compare by. ``messages`` and ``tool_purposes`` are the
    version's other words in the same form, recorded by its version id; a reader
    of a retained Attempt's artifact has only its text.
    """

    agent_role: str
    version: str
    form: str
    text: str
    messages: tuple[tuple[str, str], ...] = field(default=(), compare=False)
    tool_purposes: tuple[tuple[str, str], ...] = field(default=(), compare=False)

    @property
    def identity(self) -> str:
        return f"{self.agent_role}:{self.version}:{self.form}"

    @property
    def content(self) -> bytes:
        return self.text.encode("utf-8")

    def message(self, name: str, **fields: str) -> str:
        """A sentence the role's own message states, its ``{field}``s filled."""
        try:
            return dict(self.messages)[name].format(**fields)
        except KeyError as error:
            raise InstructionsUnrenderable(
                f"instructions {self.identity} have no message {name} "
                f"or leave its field {error} unfilled"
            ) from None

    def tool_purpose(self, tool_name: str) -> str | None:
        """What the Tool is for, as this model reads it; None when unstated."""
        return dict(self.tool_purposes).get(tool_name)

    def require_tool_purposes(self, tool_names: Iterable[str]) -> None:
        """Refuse unless the blocks state a purpose for exactly these Tools."""
        stated = {name for name, text in self.tool_purposes if text.strip()}
        missing, unknown = set(tool_names) - stated, stated - set(tool_names)
        if missing or unknown:
            raise InstructionsUnrenderable(
                f"instructions {self.identity} state no purpose for "
                f"{sorted(missing)} or one for no Tool of the role: {sorted(unknown)}"
            )


def _published(name: str):
    return files("bio_neb.agentic_journey").joinpath("instructions", name)


def _rows(source, identity: str) -> list[dict[str, str]]:
    """A published file's ``[[block]]`` rows; one that does not parse as rows of
    strings cannot render."""
    try:
        rows = tomllib.loads(source.read_text(encoding="utf-8"))["block"]
    except (tomllib.TOMLDecodeError, KeyError, UnicodeDecodeError) as error:
        raise InstructionsUnrenderable(
            f"{identity} does not parse: {type(error).__name__}"
        ) from error
    if not isinstance(rows, list) or not all(
        isinstance(row, dict) and all(isinstance(v, str) for v in row.values())
        for row in rows
    ):
        raise InstructionsUnrenderable(f"{identity} has a row that is not text")
    return rows


def _checked_version(agent_role: str, version: str | None) -> str:
    # A role's selected version is its declaration's (``roles``); only selected
    # versions ship, every other is archived, so naming one is unknown.
    if agent_role not in ROLES:
        raise InstructionsUnrenderable(f"unknown instruction role: {agent_role}")
    selected = ROLES[agent_role].instruction_version if version is None else version
    if re.fullmatch(r"v[1-9][0-9]*", selected) is None:
        raise InstructionsUnrenderable(
            "instruction version must be v followed by a positive integer"
        )
    return selected


def instruction_blocks(
    agent_role: str, version: str | None = None
) -> tuple[InstructionBlock, ...]:
    """The version's block source; only a selected version ships one."""
    selected = _checked_version(agent_role, version)
    source = _published(f"{agent_role}.{selected}.blocks.toml")
    if not source.is_file():
        raise InstructionsUnrenderable(
            f"unknown instruction version: {agent_role}:{selected}"
        )
    identity = f"{agent_role}:{selected}"
    rows = _rows(source, f"block source {identity}")
    if any(
        set(row)
        not in ({"name", "text"}, {"name", "stage", "text"}, {"name", "use", "text"})
        for row in rows
    ):
        raise InstructionsUnrenderable(f"block source {identity} has an unknown field")
    if any(row.get("use", _USES[0]) not in _USES for row in rows):
        raise InstructionsUnrenderable(f"block source {identity} has an unknown use")
    # A block without a stage or use is shared: every stage's requests read it.
    blocks = tuple(
        InstructionBlock(row["name"], row.get("stage"), row["text"], row.get("use"))
        for row in rows
    )
    # Every name once; of the instruction text, shared blocks first, then each
    # stage's blocks together.
    runs = [
        stage
        for stage, _ in groupby(block.stage for block in blocks if block.use is None)
    ]
    stage_runs = runs[1:] if runs[:1] == [None] else runs
    for broken, reason in (
        (len({block.name for block in blocks}) != len(blocks), "repeats a block name"),
        (None in stage_runs, "has a shared block after a stage"),
        (len(stage_runs) != len(set(stage_runs)), "splits a stage's blocks"),
    ):
        if broken:
            raise InstructionsUnrenderable(f"block source {identity} {reason}")
    return blocks


def _form_blocks(
    agent_role: str,
    version: str,
    form: str,
    blocks: tuple[InstructionBlock, ...],
) -> tuple[InstructionBlock, ...]:
    """The blocks a form reads: each block its form text names, in that form's
    words. The block keeps its place, stage and use; the block source is unchanged.

    Only the selected version's declared forms have form text; any other form
    reads the block source as written.
    """
    source = _published(f"{agent_role}.{version}.{form}.toml")
    identity = f"{agent_role}:{version}:{form}"
    declared = version == ROLES[
        agent_role
    ].instruction_version and form in _FORM_TEXTS.get(agent_role, ())
    if declared != source.is_file():
        raise InstructionsUnrenderable(
            f"form text {identity} is "
            + ("declared but missing" if declared else "published but not declared")
        )
    if not declared:
        return blocks
    rows = _rows(source, f"form text {identity}")
    if any(set(row) != {"name", "text"} for row in rows):
        raise InstructionsUnrenderable(f"form text {identity} has an unknown field")
    own = {row["name"]: row["text"] for row in rows}
    if len(own) != len(rows):
        raise InstructionsUnrenderable(f"form text {identity} repeats a block name")
    for name in own.keys() - {block.name for block in blocks}:
        raise InstructionsUnrenderable(
            f"form text {identity} replaces no block: {name}"
        )
    return tuple(
        replace(block, text=own[block.name]) if block.name in own else block
        for block in blocks
    )


def _render(blocks: Sequence[InstructionBlock]) -> str:
    """The instruction text: the shared blocks, then each stage's section after
    its ``[stage <name>]`` line, every block as written."""
    text = [block for block in blocks if block.use is None]
    sections = ["".join(block.text for block in text if block.stage is None)]
    for stage in dict.fromkeys(b.stage for b in text if b.stage is not None):
        own = "".join(block.text for block in text if block.stage == stage)
        sections.append(f"[stage {stage}]\n" + own)
    return "".join(sections)


def _check_messages(
    identity: str, declared: dict[str, frozenset[str]], messages: dict[str, str]
) -> None:
    """Refuse a missing or undeclared message, or one its role cannot fill.

    Each replacement field is a bare name the role fills with text: no
    conversion, format specification (so no nested field) or attribute access,
    so a message that passes here always renders.
    """
    if messages.keys() != declared.keys():
        raise InstructionsUnrenderable(
            f"instructions {identity} state messages {sorted(messages)}, "
            f"not {sorted(declared)}"
        )
    for name, text in messages.items():
        try:
            fields = [
                (field, spec, conversion)
                for _, field, spec, conversion in Formatter().parse(text)
                if field is not None
            ]
        except ValueError as error:
            raise InstructionsUnrenderable(
                f"instructions {identity} message {name} does not format: {error}"
            ) from None
        if any(spec or conversion for _, spec, conversion in fields):
            raise InstructionsUnrenderable(
                f"instructions {identity} message {name} formats a field: "
                "a field is a bare name"
            )
        if unfilled := {field for field, _, _ in fields} - declared[name]:
            raise InstructionsUnrenderable(
                f"instructions {identity} message {name} names fields its role "
                f"does not fill: {sorted(unfilled)}"
            )


@functools.cache
def role_instructions(
    agent_role: str,
    *,
    version: str | None = None,
    form: str = BUILD_MODEL_REGISTRY.default.instruction_form,
) -> RoleInstructions:
    """Render the selected or explicitly named version in a model's form: its
    blocks, then the form's own words. An Attempt reads its model's
    (``instructions_for``); the default is the build's default model's (owner,
    #441 Q7). Anything that cannot render raises ``InstructionsUnrenderable``.

    Published files are immutable, so each rendering is read once per process
    (owner, #440 A6); ``role_instructions.cache_clear()`` forgets them.
    """
    selected = _checked_version(agent_role, version)
    if form not in _FORMS:
        raise InstructionsUnrenderable(f"unknown instruction form: {form}")
    own = _form_blocks(
        agent_role, selected, form, instruction_blocks(agent_role, selected)
    )
    _check_messages(
        f"{agent_role}:{selected}:{form}",
        _MESSAGE_FIELDS.get(agent_role, {}),
        {b.name: b.text for b in own if b.use == "message"},
    )
    return RoleInstructions(
        agent_role,
        selected,
        form,
        _render(own),
        tuple((b.name, b.text) for b in own if b.use == "message"),
        tuple((b.name, b.text) for b in own if b.use == "tool"),
    )


def instructions_for(agent_role: str, model: ModelEndpointPolicy) -> RoleInstructions:
    """The role's selected instructions in the form ``model`` reads."""
    return role_instructions(agent_role, form=model.instruction_form)


# A line ``[stage <name>]`` opens text only requests in that stage read (v7 on).
_STAGE_SECTION = re.compile(r"^\[stage ([a-z_]+)\]\n", re.MULTILINE)


def stage_text(text: str, stage: str) -> str:
    """The instructions a request in ``stage`` reads: the shared text, then that
    stage's own section, so a request names only the Tools its stage offers.

    Text before the first section is shared; a version with no sections is shared
    whole. Every form keeps the ``[stage <name>]`` lines between its sections.
    """
    shared, *sections = _STAGE_SECTION.split(text)
    own = dict(zip(sections[::2], sections[1::2], strict=True))
    return (
        "\n\n".join(
            part.strip() for part in (shared, own.get(stage, "")) if part.strip()
        )
        + "\n"
    )
