"""Which sources can ever supply a planning requirement's value.

One table owns this for every requirement kind: by knowledge state, remaining
validation kind and acquisition route, the source that supplies the value and what
produces it. Planning admission refuses a requirement no row satisfies, naming the
rows its kind does have, and requirement resolution reads the request's own value and
a requirement's waiting state from the same rows (ADR 0010). Nothing produces
accepted pre-Parabricks evidence yet, so no row uses that route or validation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

# The one vocabulary of a Requirement value's source.
RequirementValueSource = Literal[
    "research_request",
    "review_answer",
    "tool_result",
    "observed_input",
    "accepted_pre_parabricks",
]

_REMAINING_VALIDATION = frozenset(
    {"tool_evidence", "researcher_decision", "deterministic_derivation"}
)
_ROUTES_WHEN_ASSERTED = frozenset(
    {"use_supplied_assertion", "validate_with_tool", "ask_planning_hitl"}
)
_ROUTES_WHEN_TO_FIND = frozenset({"validate_with_tool", "ask_planning_hitl"})
# The planning Tools whose recorded fact or downstream binding supplies each kind.
_TOOLS_BY_KIND = {
    "sample_identity": ("attest_sample",),
    "taxon_identity": ("resolve_taxon",),
    "reference_assembly_identity": ("discover_assemblies", "inspect_reference"),
    "companion_asset": ("describe_companion",),
    "analysis_recipe": ("list_compatible_analysis_recipes",),
    "resource_profile": ("select_resource_candidate",),
}


@dataclass(frozen=True)
class SlotRow:
    """One way a requirement kind gets its value, and what produces it."""

    kind: str
    knowledge: frozenset[str]
    validation: frozenset[str]
    routes: frozenset[str]
    source: RequirementValueSource
    tools: tuple[str, ...] = ()
    producer: str = ""

    @property
    def described_producer(self) -> str:
        return self.producer or f"the {' or '.join(self.tools)} Tool"


def _rows() -> tuple[SlotRow, ...]:
    rows = [
        SlotRow(
            "registered_input",
            frozenset({"observed"}),
            frozenset({"none"}) | _REMAINING_VALIDATION,
            frozenset(
                {"validate_with_tool", "derive_deterministically", "ask_planning_hitl"}
            ),
            "observed_input",
            tools=("inspect_input_metadata",),
        ),
        SlotRow(
            "research_goal",
            frozenset({"asserted"}),
            frozenset({"none"}),
            _ROUTES_WHEN_ASSERTED,
            "research_request",
            producer="the research request's own words",
        ),
        *(
            SlotRow(
                "scientific_investigation",
                frozenset({state}),
                _REMAINING_VALIDATION,
                routes | {"derive_deterministically"},
                "tool_result",
                producer="post-execution Investigation",
            )
            for state, routes in (
                ("asserted", _ROUTES_WHEN_ASSERTED),
                ("to_find", _ROUTES_WHEN_TO_FIND),
            )
        ),
        # A paper the researcher's words cite: the paper stage's Tools retrieve
        # it. Its words are the researcher's; nothing else is stated about the
        # paper.
        SlotRow(
            "cited_paper",
            frozenset({"asserted"}),
            frozenset({"tool_evidence"}),
            frozenset({"validate_with_tool"}),
            "tool_result",
            tools=("find_paper", "retrieve_paper"),
        ),
    ]
    for kind, tools in _TOOLS_BY_KIND.items():
        rows.extend(
            SlotRow(
                kind,
                frozenset({state}),
                frozenset({"tool_evidence"}),
                routes,
                "tool_result",
                tools=tools,
            )
            for state, routes in (
                ("asserted", _ROUTES_WHEN_ASSERTED),
                ("to_find", _ROUTES_WHEN_TO_FIND),
            )
        )
    return tuple(rows)


SLOT_ROWS = _rows()
# A Tool or downstream fact binds exactly one requirement of these kinds.
SINGLE_REQUIREMENT_KINDS = frozenset(
    {
        "sample_identity",
        "taxon_identity",
        "reference_assembly_identity",
        "companion_asset",
    }
)


def slot_rows(
    kind: str, knowledge_state: str, validation_kind: str, route: str
) -> tuple[SlotRow, ...]:
    """The rows that can supply this requirement's value; none when nothing can."""

    return tuple(
        row
        for row in SLOT_ROWS
        if row.kind == kind
        and knowledge_state in row.knowledge
        and validation_kind in row.validation
        and route in row.routes
    )


def _names(values: Iterable[str]) -> str:
    return "/".join(f"'{value}'" for value in sorted(values))


def unsatisfiable_reason(
    kind: str, knowledge_state: str, validation_kind: str, route: str
) -> str | None:
    """Why no source can ever supply this requirement, naming its valid rows."""

    if slot_rows(kind, knowledge_state, validation_kind, route):
        return None
    valid = [
        f"acquisition_route={_names(row.routes)} with validation.kind="
        f"{_names(row.validation)} ({row.described_producer})"
        for row in SLOT_ROWS
        if row.kind == kind and knowledge_state in row.knowledge
    ]
    return (
        f"no Tool, fact or Review can ever supply {kind} with "
        f"knowledge.state='{knowledge_state}', acquisition_route='{route}' and "
        f"validation.kind='{validation_kind}'; "
        + (
            "valid for it: " + "; or ".join(valid)
            if valid
            else f"{kind} cannot be {knowledge_state}"
        )
    )


def repeated_single_requirement_kinds(kinds: Iterable[str]) -> tuple[str, ...]:
    """Kinds recorded more than once although their Tool binds one requirement."""

    counted = list(kinds)
    return tuple(
        sorted(kind for kind in SINGLE_REQUIREMENT_KINDS if counted.count(kind) > 1)
    )


def guidance() -> str:
    """The admission rules the model sees, rendered from the table."""

    shapes: dict[str, str] = {}
    for kind in dict.fromkeys(row.kind for row in SLOT_ROWS):
        shapes[kind] = "; or ".join(
            f"knowledge.state={_names(row.knowledge)} with acquisition_route="
            f"{_names(row.routes)} and validation.kind={_names(row.validation)}"
            for row in SLOT_ROWS
            if row.kind == kind
        )
    # Kinds with the same shapes are listed together.
    grouped: dict[str, list[str]] = {}
    for kind, shape in shapes.items():
        grouped.setdefault(shape, []).append(kind)
    kinds = " ".join(
        f"{', '.join(names)}: {shape}." for shape, names in grouped.items()
    )
    return (
        "Each requirement needs a source that can supply its value. "
        "use_supplied_assertion requires knowledge.state='asserted'; "
        "validation.kind='none' means the value is already in hand. "
        "Prefer acquisition_route='validate_with_tool' with validation.kind='none' "
        "for observed registered_input. When the request names no target, record "
        "one scientific_investigation to_find with acquisition_route="
        "'ask_planning_hitl': the researcher names the targets before execution. "
        "When the researcher's words cite a paper (an id, a link, or a title, author "
        "and year), find and retrieve it with find_paper and retrieve_paper first, "
        "then record one cited_paper per paper, its quote the words that cite it and "
        "its PaperRecorded event in evidence_event_ids once retrieved; a value the "
        "request says comes from a cited paper (the organism, the reference, a "
        "target) is to_find citing that PaperRecorded event (or the request while "
        "the paper is still to find), with acquisition_route='validate_with_tool' "
        "and validation.kind='tool_evidence' naming the cited_paper's subject_key, "
        "not 'ask_planning_hitl'. "
        "Allowed per kind, "
        + kinds
        + " Record at most one each of "
        + ", ".join(sorted(SINGLE_REQUIREMENT_KINDS))
        + "."
    )


__all__ = [
    "SINGLE_REQUIREMENT_KINDS",
    "SLOT_ROWS",
    "SlotRow",
    "RequirementValueSource",
    "guidance",
    "repeated_single_requirement_kinds",
    "slot_rows",
    "unsatisfiable_reason",
]
