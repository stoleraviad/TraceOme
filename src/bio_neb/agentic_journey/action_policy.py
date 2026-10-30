"""Pure generic admission decisions over caller-established facts and role semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field

from bio_neb.content import canonical_json

# Text a model writes that the server keeps as written: not blank, no surrounding
# whitespace. An argument the model chooses carries it in its schema, so the model
# reads the rule before it calls.
CANONICAL_TEXT_PATTERN = r"^\S(?:[\s\S]*\S)?$"

ChoiceReason = Annotated[
    str, Field(min_length=1, max_length=2000, pattern=CANONICAL_TEXT_PATTERN)
]
"""A model's stated reason for a choice it records."""


@dataclass(frozen=True)
class ProposedAgentAction:
    """One function action normalized without framework dispatch state."""

    tool_name: str
    tool_call_id: str
    arguments: Mapping[str, Any]
    arguments_invalid_json: bool = False
    # The model wrote the arguments inside a Markdown fence (``model_turn``).
    arguments_fenced: bool = False
    # The top-level arguments read as the object their JSON text held.
    arguments_decoded: tuple[str, ...] = ()


class PreconditionUnmet(Exception):
    """A Tool's declared precondition does not hold now; admission refuses the call.

    ``code`` names the precondition in the Tool's registry row, which explains it;
    ``value`` is the current value of its slot, shown to the model when known;
    ``facts`` are the typed values the row's sentence names (#372 D2): a mapping,
    or a function that computes it, read only when a refusal is explained, since
    every request checks every stage Tool's precondition.
    """

    def __init__(
        self,
        code: str,
        value: Any = None,
        facts: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.value = value
        self._facts = facts
        self._read: dict[str, Any] | None = None

    @property
    def facts(self) -> dict[str, Any]:
        if self._read is None:
            facts = self._facts() if callable(self._facts) else self._facts
            self._read = dict(facts or {})
        return self._read


def call_identity(action: ProposedAgentAction) -> str:
    """The same call: the same Tool with the same canonical effective arguments.

    It ignores the call id, so two byte-identical calls of any Tool share it, and
    reads the arguments as called, so an omitted, null or blank one is absent.
    """
    return canonical_json(
        {
            "tool_name": action.tool_name,
            "arguments": effective_arguments(action.arguments, {}),
        }
    )


ActionValidator = Callable[[ProposedAgentAction], tuple[str, ...]]
ActionFingerprint = Callable[[ProposedAgentAction], str | None]
ActionPrecondition = Callable[[ProposedAgentAction], str | None]


def effective_arguments(
    arguments: Mapping[str, Any], bound_arguments: Mapping[str, Any]
) -> dict[str, Any]:
    """The model's arguments as called, plus each bound value it omitted.

    As called, an omitted, null or blank argument is one absent value (owner,
    #476 G3): the record, every reader and the repeat identity compare that one
    form. A bound key the model sent keeps the sent value, so admission can refuse
    a contradiction and the record shows what the model said. Nothing is coerced.
    """

    called = {
        name: value
        for name, value in arguments.items()
        if value is not None and value != ""
    }
    return {
        **called,
        **{
            name: value for name, value in bound_arguments.items() if name not in called
        },
    }


def action_admission_errors(
    action: ProposedAgentAction,
    *,
    eligible_tool_names: frozenset[str],
    exposed_tool_names: frozenset[str],
    action_validator: ActionValidator,
    action_fingerprint: ActionFingerprint | None = None,
    admitted_action_fingerprints: frozenset[str] = frozenset(),
    admitted_repeat_error: str | None = None,
    bound_arguments: Mapping[str, Any] | None = None,
    argument_validator: ActionValidator | None = None,
    precondition: ActionPrecondition | None = None,
) -> tuple[str, ...]:
    """Compose admission errors from caller-established facts and role semantics.

    ``action`` carries the effective arguments. The caller owns evidence
    reconstruction and state. Malformed/eligibility/exposure precedence gates the
    Tool's unmet precondition, then the bound values, then the Tool's own argument
    validation, then grounding; a repeat remains additive.
    """

    errors: list[str] = []
    if action.arguments_invalid_json:
        errors.append("tool_arguments_invalid_json")
    elif action.tool_name not in eligible_tool_names:
        errors.append("tool_name_ineligible")
    elif action.tool_name not in exposed_tool_names:
        errors.append("tool_name_unavailable")
    elif (unmet := precondition(action) if precondition else None) is not None:
        errors.append(unmet)
    else:
        mismatched = [
            f"{name}:bound_value_mismatch"
            for name, value in (bound_arguments or {}).items()
            if action.arguments.get(name) != value
        ]
        if mismatched:
            errors.extend(mismatched)
        else:
            invalid = (
                argument_validator(action) if argument_validator is not None else ()
            )
            errors.extend(invalid or action_validator(action))

    fingerprint = action_fingerprint(action) if action_fingerprint is not None else None
    if (
        not action.arguments_invalid_json
        and action.tool_name in eligible_tool_names
        and fingerprint is not None
        and fingerprint in admitted_action_fingerprints
    ):
        if admitted_repeat_error is None:
            raise RuntimeError("admitted action fingerprint lacks repeat policy")
        errors.append(admitted_repeat_error)
    return tuple(errors)
