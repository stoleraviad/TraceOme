"""The Run's own health and each Work item's coverage, as Journey facts (#414).

The Investigation's first step once the Run's BAM and VCF are local
(``run_measurement``) records the Run's health once, its key numbers or why it
measured none, and one coverage fact per Work item of the collection. A failed
health measurement asks this module's Review (the failure package's ``Ask``):
``retry`` measures again, ``abort`` records it not measured; the Investigation
then continues. Neither fact nor the Review changes a decision: the report cites
them.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import (
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from bio_neb.contracts import FrozenContract
from bio_neb.failure import Ask, ReviewAnswer
from bio_neb.investigation.qualification_origins import (
    CoverageNotMeasured,
    DepthBelowMinimum,
)
from bio_neb.journey import Journey

_SHA256 = r"^[0-9a-f]{64}$"
STAGE = "run_health"
_ACTIONS = ("retry", "abort")


class HealthNumbers(FrozenContract):
    """The health manifest's digest and its key numbers over the Run's BAM and VCF."""

    manifest_sha256: StrictStr = Field(pattern=_SHA256)
    manifest_byte_count: StrictInt = Field(gt=0)
    reads: StrictInt = Field(ge=0)
    mapped_percent: StrictFloat = Field(ge=0, le=100)
    mean_depth: StrictFloat = Field(ge=0)
    ti_tv: StrictFloat | None = Field(default=None, ge=0)
    # Why there is no Ti/Tv: the VCF holds no transversion, or its preparation
    # failed so it was not read.
    ti_tv_reason: Literal["no_transversion", "vcf_unprepared"] | None = None
    covered_contigs: StrictInt = Field(ge=0)
    covered_bases: StrictInt = Field(ge=0)
    covered_fraction: StrictFloat = Field(ge=0, le=1)

    @model_validator(mode="after")
    def coherent(self) -> "HealthNumbers":
        if (self.ti_tv is None) == (self.ti_tv_reason is None):
            raise ValueError("health numbers state Ti/Tv or why there is none")
        return self


# Why the Run's health was not measured: declared once.
HealthNotMeasuredCode = Literal["bam_unavailable", "measurement_failed"]
# A measuring process's typed fault, as the failure's Review records it.
MeasurementFaultCode = Literal["timeout", "process_failed", "os_error"]


class HealthNotMeasured(FrozenContract):
    """Why the Run's health was not measured."""

    code: HealthNotMeasuredCode
    detail: StrictStr = Field(min_length=1)


class RunHealthMeasured(FrozenContract):
    """The Run's health, measured once, at its first collection (#414)."""

    event_type: Literal["RunHealthMeasured"] = "RunHealthMeasured"
    schema_version: Literal["bio-neb.run-health/2"] = "bio-neb.run-health/2"
    collection_event_id: StrictStr
    # The Run's verified BAM and VCF it measured; None when it held no local BAM.
    bam_sha256: StrictStr | None = Field(default=None, pattern=_SHA256)
    vcf_sha256: StrictStr | None = Field(default=None, pattern=_SHA256)
    measurement: HealthNumbers | None = None
    not_measured: HealthNotMeasured | None = None

    @model_validator(mode="after")
    def coherent(self) -> "RunHealthMeasured":
        if (
            (self.measurement is None) == (self.not_measured is None)
            or (self.bam_sha256 is None) != (self.vcf_sha256 is None)
            or (self.measurement is not None and self.bam_sha256 is None)
        ):
            raise ValueError("run health is incoherent")
        return self


class RunHealthMeasurementFailed(FrozenContract):
    """One health measurement that raised; its Review follows."""

    event_type: Literal["RunHealthMeasurementFailed"] = "RunHealthMeasurementFailed"
    collection_event_id: StrictStr
    code: MeasurementFaultCode
    detail: StrictStr = Field(min_length=1)


class RunHealthReviewRequested(FrozenContract):
    event_type: Literal["RunHealthReviewRequested"] = "RunHealthReviewRequested"
    collection_event_id: StrictStr
    request: Ask


class RunHealthReviewAnswered(FrozenContract):
    event_type: Literal["RunHealthReviewAnswered"] = "RunHealthReviewAnswered"
    request_event_id: StrictStr
    response: ReviewAnswer


FACTS = {
    fact.model_fields["event_type"].default: fact
    for fact in (
        RunHealthMeasured,
        RunHealthMeasurementFailed,
        RunHealthReviewRequested,
        RunHealthReviewAnswered,
    )
}


def validate_run_health_fact(event: Mapping[str, Any]) -> FrozenContract | None:
    from bio_neb.journey import canonical_json

    owner = FACTS.get(event.get("event_type"))
    if owner is None:
        return None
    return owner.model_validate_json(
        canonical_json({"event_type": event["event_type"], **event.get("payload", {})}),
        strict=True,
    )


class RunHealthReviewRequest(FrozenContract):
    """The shared Review view of one failed health measurement."""

    request_event_id: str
    collection_event_id: str
    failure: RunHealthMeasurementFailed
    request: Ask
    pending: bool


class RunHealthReviewReceipt(FrozenContract):
    request_event_id: str
    response_event_id: str
    response: ReviewAnswer


class RunHealthState(FrozenContract):
    """The Run's one health fact, its failed measurements' Reviews and answers."""

    health_event_id: str | None = None
    health: RunHealthMeasured | None = None
    requests: tuple[RunHealthReviewRequest, ...] = ()
    receipts: tuple[RunHealthReviewReceipt, ...] = ()

    @property
    def pending(self) -> bool:
        return any(request.pending for request in self.requests)

    @property
    def latest_action(self) -> str | None:
        """The last answer: ``retry``, ``abort``, or None when none answered."""
        return self.receipts[-1].response.action if self.receipts else None

    @property
    def last_failure(self) -> RunHealthMeasurementFailed | None:
        return self.requests[-1].failure if self.requests else None


def validate_health_answer(request, response) -> None:
    if (
        response.request_id != request.request.request_id
        or response.action not in request.request.legal_actions
    ):
        raise ValueError("run health review changed its exact request or action")


def _refused(code: str, identity: str) -> ValueError:
    return ValueError(f"run health review {code}: {identity}")


def project_run_health(events: Sequence[Mapping[str, Any]]) -> RunHealthState:
    """The Run's health fact and Reviews, each in its exact lineage.

    One health fact per Run, after any failure's answered Review; a failure asks
    its Review at once, and one Review is pending at a time.
    """
    from bio_neb.junction import evaluate_scoped_answer

    health_event_id = health = None
    failures: dict[str, RunHealthMeasurementFailed] = {}
    asked: set[str] = set()
    requests: list[RunHealthReviewRequest] = []
    receipts: list[RunHealthReviewReceipt] = []
    for event in events:
        fact = validate_run_health_fact(event)
        if fact is None:
            continue
        pending = any(request.pending for request in requests)
        if isinstance(fact, RunHealthMeasured):
            if health is not None or pending:
                raise ValueError("run health repeated or preceded its Review's answer")
            health_event_id, health = event["event_id"], fact
        elif isinstance(fact, RunHealthMeasurementFailed):
            if health is not None or pending:
                raise ValueError("run health failure beside a pending Review")
            failures[event["event_id"]] = fact
        elif isinstance(fact, RunHealthReviewRequested):
            failure = failures.get(fact.request.failure_event_id)
            if (
                failure is None
                or event["parent_ids"] != [fact.request.failure_event_id]
                or failure.collection_event_id != fact.collection_event_id
                or fact.request.stage != STAGE
                or fact.request.legal_actions != _ACTIONS
                or fact.request.failure_event_id in asked
            ):
                raise ValueError("run health review changed its failure")
            asked.add(fact.request.failure_event_id)
            requests.append(
                RunHealthReviewRequest(
                    request_event_id=event["event_id"],
                    collection_event_id=fact.collection_event_id,
                    failure=failure,
                    request=fact.request,
                    pending=True,
                )
            )
        else:
            _, replay = evaluate_scoped_answer(
                requests,
                receipts,
                fact.request_event_id,
                fact.response,
                validate=validate_health_answer,
                refusal=_refused,
            )
            if replay is not None or event["parent_ids"] != [fact.request_event_id]:
                raise ValueError("run health review repeated its answer or lineage")
            requests = [
                request.model_copy(update={"pending": False})
                if request.request_event_id == fact.request_event_id
                else request
                for request in requests
            ]
            receipts.append(
                RunHealthReviewReceipt(
                    request_event_id=fact.request_event_id,
                    response_event_id=event["event_id"],
                    response=fact.response,
                )
            )
    return RunHealthState(
        health_event_id=health_event_id,
        health=health,
        requests=tuple(requests),
        receipts=tuple(receipts),
    )


def _record(journey: Journey, fact: FrozenContract, parent_event_id: str):
    return journey.record(
        fact.event_type,
        fact.model_dump(mode="json", exclude={"event_type"}),
        [parent_event_id],
    )


def ask_health_review(
    journey: Journey, collection_event_id: str, code: str, detail: str
) -> dict[str, Any]:
    """Record the failed measurement and ask its Review: retry or abort."""
    failed = _record(
        journey,
        RunHealthMeasurementFailed(
            collection_event_id=collection_event_id, code=code, detail=detail
        ),
        journey.head_event_id,
    )
    return _record(
        journey,
        RunHealthReviewRequested(
            collection_event_id=collection_event_id,
            request=Ask(
                stage=STAGE,
                failure_event_id=failed["event_id"],
                request_id=str(uuid.uuid4()),
                reason=f"The Run's health measurement failed ({code}): {detail}",
                legal_actions=_ACTIONS,
            ),
        ),
        failed["event_id"],
    )


def answer_health_review(
    journey: Journey, request_event_id: str, response: ReviewAnswer
) -> dict[str, Any]:
    """Record the one answer to the pending health Review; replay is inert."""
    from bio_neb.junction import evaluate_scoped_answer

    state = journey.checked_view("run_health")
    _, replay = evaluate_scoped_answer(
        state.requests,
        state.receipts,
        request_event_id,
        response,
        validate=validate_health_answer,
        refusal=_refused,
    )
    if replay is not None:
        return replay
    return _record(
        journey,
        RunHealthReviewAnswered(request_event_id=request_event_id, response=response),
        request_event_id,
    )


class ItemDepthLine(FrozenContract):
    """One measured item's least read depth over its span, as its fact records it."""

    work_item_id: StrictStr
    coverage_event_id: StrictStr
    depth: StrictInt = Field(ge=0)
    mean_depth: StrictFloat = Field(ge=0)
    minimum_depth: StrictInt = Field(ge=0)
    meets_minimum_depth: bool
    batch_flags: tuple[Literal["low_depth"], ...] = ()


class RunHealthSection(FrozenContract):
    """The report's health section: the Run's health fact and each measured item's
    depth line, cited by event; it reports and never changes a decision."""

    health_event_id: StrictStr
    health: RunHealthMeasured
    items: tuple[ItemDepthLine, ...] = ()


@dataclass(frozen=True)
class CollectionMeasurements:
    """What the Run's health and its items' coverage add to a Finalization basis."""

    section: RunHealthSection | None
    # Each limitation's typed origin with the Work item it concerns.
    limitations: tuple[tuple[str, DepthBelowMinimum], ...]
    # Each item with no measured coverage, by Work item, with why.
    not_measured: Mapping[str, CoverageNotMeasured]


def collection_measurements(
    timeline: Sequence[Mapping[str, Any]], item_evidence: Sequence[Any]
) -> CollectionMeasurements:
    """The Run's health fact and each item's coverage field, folded once into its
    item evidence, as report section, limitation and not-measured origins."""
    state = project_run_health(timeline)
    lines: list[ItemDepthLine] = []
    limitations: list[tuple[str, DepthBelowMinimum]] = []
    not_measured: dict[str, CoverageNotMeasured] = {}
    for item in item_evidence:
        coverage = item.coverage
        if coverage.measurement is None:
            not_measured[item.work_item_id] = (
                CoverageNotMeasured()
                if coverage.not_measured is None
                else CoverageNotMeasured(
                    reason_code=coverage.not_measured.code,
                    detail=coverage.not_measured.detail,
                )
            )
            continue
        measurement = coverage.measurement
        minimum = measurement.parameters.minimum_depth
        lines.append(
            ItemDepthLine(
                work_item_id=item.work_item_id,
                coverage_event_id=coverage.coverage_event_id,
                depth=measurement.depth,
                mean_depth=measurement.mean_depth,
                minimum_depth=minimum,
                meets_minimum_depth=measurement.meets_minimum_depth,
                batch_flags=coverage.batch_flags,
            )
        )
        if not measurement.meets_minimum_depth:
            limitations.append(
                (
                    item.work_item_id,
                    DepthBelowMinimum(depth=measurement.depth, minimum=minimum),
                )
            )
    section = (
        RunHealthSection(
            health_event_id=state.health_event_id,
            health=state.health,
            items=tuple(sorted(lines, key=lambda line: line.work_item_id)),
        )
        if state.health is not None
        else None
    )
    return CollectionMeasurements(section, tuple(limitations), not_measured)


__all__ = [
    "STAGE",
    "FACTS",
    "CollectionMeasurements",
    "HealthNotMeasured",
    "HealthNotMeasuredCode",
    "MeasurementFaultCode",
    "HealthNumbers",
    "ItemDepthLine",
    "RunHealthMeasured",
    "RunHealthMeasurementFailed",
    "RunHealthReviewAnswered",
    "RunHealthReviewReceipt",
    "RunHealthReviewRequest",
    "RunHealthReviewRequested",
    "RunHealthSection",
    "RunHealthState",
    "answer_health_review",
    "ask_health_review",
    "collection_measurements",
    "project_run_health",
    "validate_run_health_fact",
]
