"""A stage's combined findings and the typed output the server verifies.

A role's stage collects the recorded facts its Tools answer; the model reads them as
one typed view on every request, so one Tool's answer drives the next. A stage closes
on its typed output: values the stage's recorded facts already hold, each cited by
event. The server verifies the citations and values; it never extracts them from
text, and the recorded facts stay the stage evidence (ADR 0040).
"""

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictStr

UNVERIFIED = "stage_output_unverified"


class StageOutput(BaseModel):
    """One stage's typed output: its closing values and the events that hold them.

    ``evidence_types`` names, per citing field, the recorded fact kinds it may cite;
    ``agrees_with`` names, per value field, the citing field whose event must hold it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    stage: ClassVar[str] = ""
    evidence_types: ClassVar[Mapping[str, frozenset[str]]] = {}
    agrees_with: ClassVar[Mapping[str, str]] = {}

    def evidence_event_ids(self) -> tuple[str, ...]:
        cited = []
        for field in self.evidence_types:
            value = getattr(self, field)
            cited.extend(value if isinstance(value, tuple) else (value,))
        return tuple(cited)


class StageFinding(BaseModel):
    """One recorded fact the stage's Tools answered, with its status.

    A slot row names the slot (``subject_key``) and its current ``value``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: StrictStr
    source_event_id: StrictStr
    status: StrictStr | None = None
    subject_key: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    value: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class StageOutputRefusal(BaseModel):
    """Why the server cannot verify a stage output: the slot, its value, the reason."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    code: StrictStr
    slot: StrictStr
    value: JsonValue
    explanation: StrictStr


class VerifiedStageOutput(BaseModel):
    """A closed stage's verified output, as later stages show it."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    stage: StrictStr
    output: dict[str, JsonValue]
    evidence_event_ids: tuple[StrictStr, ...]


class StageFindings(BaseModel):
    """The current stage's combined findings and the state of its typed output."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    stage: StrictStr
    findings: tuple[StageFinding, ...]
    output: Literal["open", "verified", "refused"]
    refusal: StageOutputRefusal | None = None
    verified_outputs: tuple[VerifiedStageOutput, ...] = ()


def _holds(value: Any, expected: Any) -> bool:
    if value == expected:
        return True
    if isinstance(value, Mapping):
        return any(_holds(item, expected) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_holds(item, expected) for item in value)
    return False


def verify_stage_output(
    output: StageOutput, events: Iterable[Mapping[str, Any]]
) -> StageOutputRefusal | None:
    """Refuse a stage output whose citations or values the Run does not hold.

    Each citing field must name a retained event of this Run of an allowed kind,
    and each agreeing value must occur in the event its citing field names.
    """
    by_id = {event["event_id"]: event for event in events}
    for field, kinds in output.evidence_types.items():
        value = getattr(output, field)
        for event_id in value if isinstance(value, tuple) else (value,):
            event = by_id.get(event_id)
            if event is None or event["event_type"] not in kinds:
                return StageOutputRefusal(
                    code=UNVERIFIED,
                    slot=field,
                    value=event_id,
                    explanation=(
                        f"The `{output.stage}` output must cite a retained "
                        + " or ".join(f"`{kind}`" for kind in sorted(kinds))
                        + " of this Run."
                    ),
                )
    for field, citing in output.agrees_with.items():
        value = getattr(output, field)
        if not _holds(by_id[getattr(output, citing)]["payload"], value):
            return StageOutputRefusal(
                code=UNVERIFIED,
                slot=field,
                value=value,
                explanation=(
                    f"The event `{citing}` names does not hold this value; the "
                    f"`{output.stage}` output repeats what its evidence records."
                ),
            )
    return None


def verified_output(output: StageOutput) -> VerifiedStageOutput:
    return VerifiedStageOutput(
        stage=output.stage,
        output=output.model_dump(mode="json"),
        evidence_event_ids=output.evidence_event_ids(),
    )


__all__ = [
    "UNVERIFIED",
    "StageFinding",
    "StageFindings",
    "StageOutput",
    "StageOutputRefusal",
    "VerifiedStageOutput",
    "verified_output",
    "verify_stage_output",
]
