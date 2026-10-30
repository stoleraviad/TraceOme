"""Typed Journey facts for exact Run failure outcomes."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from bio_neb.journey import Journey

RuntimeFailureCode = Literal[
    "runtime_config_invalid",
    "runtime_config_changed",
    "model_credential_unavailable",
    # The role's selected instructions cannot render in its model's form
    # (I-65): a missing or broken block source or form text.
    "instructions_unrenderable",
]
# Only the researcher's own stop or rejection ends a Run (D9); failures are
# recorded where they occur and hold the Run.
RunFailureCode = Literal["rejected_by_user", "required_evidence_unresolved"]


def _canonical_uuid(value: str, field_name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{field_name} must be canonical")
    return value


class _RunFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeConfigurationFailed(_RunFact):
    event_type: Literal["RuntimeConfigurationFailed"] = "RuntimeConfigurationFailed"
    run_id: StrictStr
    code: RuntimeFailureCode
    violations: tuple[StrictStr, ...] = Field(min_length=1)

    @field_validator("run_id")
    @classmethod
    def canonical_run_identity(cls, value: str) -> str:
        return _canonical_uuid(value, "run_id")


class RunFailed(_RunFact):
    event_type: Literal["RunFailed"] = "RunFailed"
    run_id: StrictStr
    code: RunFailureCode
    source_event_id: StrictStr

    @field_validator("run_id", "source_event_id")
    @classmethod
    def canonical_identity(cls, value: str, info: Any) -> str:
        return _canonical_uuid(value, info.field_name)


_FACT_TYPES = {
    fact.model_fields["event_type"].default: fact
    for fact in (RuntimeConfigurationFailed, RunFailed)
}


def validate_run_fact(event: Mapping[str, Any]) -> _RunFact | None:
    owner = _FACT_TYPES.get(event.get("event_type"))
    if owner is None:
        return None
    return owner.model_validate(
        {"event_type": event["event_type"], **event.get("payload", {})}
    )


def record_run_fact(journey: Journey, fact: _RunFact, parent_id: str) -> dict[str, Any]:
    payload = fact.model_dump(mode="json")
    payload.pop("event_type")
    return journey.record(fact.event_type, payload, [parent_id])


def hold_for_configuration_fix(journey: Journey, parent_id: str, error) -> NoReturn:
    """Retain a runtime configuration fault, then hold the Run for a fix (#216, D5).

    Every stage that meets it, planning or Investigation, records it again on
    each dispatch; ``resume`` after the fix retries.
    """
    from bio_neb.agentic_journey.failure_cause import cause_identity
    from bio_neb.failure import Invariant

    leaf = record_run_fact(
        journey,
        RuntimeConfigurationFailed(
            run_id=journey.journey_id, code=error.code, violations=error.violations
        ),
        parent_id,
    )
    raise Invariant(
        error.code,
        "runtime configuration failed",
        cause=cause_identity(error),
        evidence={
            "configuration_failure_event_id": leaf["event_id"],
            "violations": list(error.violations),
        },
    ) from error
