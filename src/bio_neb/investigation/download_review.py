"""A grant's file download the failure policy gave up on asks its Review (#363).

The grant's preparation downloads the Run's VCF, then its BAM. When a file's
download is refused after its stage policy's part retries, the Run records the
failure and asks the failure module's Review (owner B, 2026-10-17): ``retry``
downloads that file again from scratch; ``abort`` leaves the BAM unavailable
while the VCF work goes on, and fails the shared preparation for the VCF. The
facts chain from the work frontier, so no observation runs until the answer.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from bio_neb.contracts import FrozenContract
from bio_neb.failure import Ask, ReviewAnswer
from bio_neb.journey import Journey

DownloadRole = Literal["VCF", "BAM"]
DOWNLOAD_STAGES: Mapping[str, str] = {
    "VCF": "vcf_materialization",
    "BAM": "bam_materialization",
}
# The Execution output each downloaded file is.
_EXECUTION_ROLES: Mapping[str, str] = {"VCF": "VCF", "BAM": "aligned_bam"}
DownloadFailureCode = Literal[
    "vcf_materialization_object_read_failed",
    "vcf_materialization_object_read_mismatch",
    "bam_materialization_object_read_failed",
    "bam_materialization_object_read_mismatch",
]
REVIEWED_CODES = frozenset(
    {
        "vcf_materialization_object_read_failed",
        "vcf_materialization_object_read_mismatch",
        "bam_materialization_object_read_failed",
        "bam_materialization_object_read_mismatch",
    }
)
_ACTIONS = ("retry", "abort")


class InputDownloadRefused(Exception):
    """A file's download refused after its stage policy's part retries.

    The failure module asks its Review (owner B, #363) instead of failing the
    preparation.
    """

    def __init__(self, role: str, descriptor, refusal) -> None:
        super().__init__(refusal.code)
        self.role = role
        self.descriptor = descriptor
        self.refusal = refusal


class _Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InvestigationDownloadFailed(_Fact):
    """One file's download, refused after its stage policy's part retries."""

    event_type: Literal["InvestigationDownloadFailed"] = "InvestigationDownloadFailed"
    authority_id: StrictStr = Field(min_length=1)
    role: DownloadRole
    object_key: StrictStr = Field(min_length=1)
    code: DownloadFailureCode
    details: tuple[StrictStr, ...] = ()


class InvestigationDownloadReviewRequested(_Fact):
    event_type: Literal["InvestigationDownloadReviewRequested"] = (
        "InvestigationDownloadReviewRequested"
    )
    authority_id: StrictStr = Field(min_length=1)
    role: DownloadRole
    request: Ask


class InvestigationDownloadReviewAnswered(_Fact):
    event_type: Literal["InvestigationDownloadReviewAnswered"] = (
        "InvestigationDownloadReviewAnswered"
    )
    request_event_id: StrictStr
    response: ReviewAnswer


class InvestigationDownloadReviewRequest(FrozenContract):
    """The shared Review view of one asked download."""

    request_event_id: str
    authority_id: str
    role: DownloadRole
    failure: InvestigationDownloadFailed
    request: Ask
    pending: bool


class InvestigationDownloadReviewReceipt(FrozenContract):
    request_event_id: str
    response_event_id: str
    response: ReviewAnswer


class InvestigationDownloadReviews(FrozenContract):
    requests: tuple[InvestigationDownloadReviewRequest, ...] = ()
    receipts: tuple[InvestigationDownloadReviewReceipt, ...] = ()

    @property
    def pending(self) -> bool:
        return any(request.pending for request in self.requests)

    def latest_action(self, authority_id: str, role: str) -> str | None:
        """The last answer for this grant's file: ``retry``, ``abort`` or none."""
        answered = {receipt.request_event_id: receipt for receipt in self.receipts}
        actions = [
            answered[request.request_event_id].response.action
            for request in self.requests
            if (request.authority_id, request.role) == (authority_id, role)
            and request.request_event_id in answered
        ]
        return actions[-1] if actions else None

    def failure(self, authority_id: str, role: str) -> InvestigationDownloadFailed:
        """The recorded refusal an abort answered, for a VCF's failed preparation."""
        return [
            request.failure
            for request in self.requests
            if (request.authority_id, request.role) == (authority_id, role)
        ][-1]


_PAYLOADS = {
    "InvestigationDownloadFailed": InvestigationDownloadFailed,
    "InvestigationDownloadReviewRequested": InvestigationDownloadReviewRequested,
    "InvestigationDownloadReviewAnswered": InvestigationDownloadReviewAnswered,
}


def _fact(event: Mapping[str, Any]):
    return _PAYLOADS[event["event_type"]].model_validate(
        {"event_type": event["event_type"], **event["payload"]}
    )


def validate_download_answer(request, response) -> None:
    if (
        response.request_id != request.request.request_id
        or response.action not in request.request.legal_actions
    ):
        raise ValueError("download review changed its exact request or legal action")


def _refused(code: str, identity: str) -> ValueError:
    return ValueError(f"download review {code}: {identity}")


def _execution_objects(events: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The Run's verified Execution outputs: role to object key."""
    from bio_neb.journey_path import Verified, recorded_execution
    from bio_neb.providers.object_locations import object_address

    h100 = recorded_execution(events).h100
    if not isinstance(h100, Verified):
        return {}
    return {
        pointer.role: object_address(pointer.location)[1]
        for pointer in h100.pointers
        if pointer.role in _EXECUTION_ROLES.values()
    }


def _check_failure(
    fact: InvestigationDownloadFailed,
    event: Mapping[str, Any],
    prefix: Sequence[Mapping[str, Any]],
    grants: set[str],
) -> None:
    """A failure names a recorded grant, its work frontier and its Execution file."""
    from bio_neb.recording import work_frontier

    if (
        fact.authority_id not in grants
        or event["parent_ids"] != [work_frontier(prefix)["event_id"]]
        or not fact.code.startswith(DOWNLOAD_STAGES[fact.role] + "_")
        or _execution_objects(prefix).get(_EXECUTION_ROLES[fact.role])
        != fact.object_key
    ):
        raise ValueError("download failure changed its grant, frontier or file")


def project_download_reviews(
    events: Sequence[Mapping[str, Any]],
) -> InvestigationDownloadReviews:
    """Each asked download, its exact lineage, and its one answer."""
    from bio_neb.junction import evaluate_scoped_answer

    grants: set[str] = set()
    failures: dict[str, InvestigationDownloadFailed] = {}
    requests: list[InvestigationDownloadReviewRequest] = []
    receipts: list[InvestigationDownloadReviewReceipt] = []
    for index, event in enumerate(events):
        kind = event["event_type"]
        if kind == "InvestigationAuthorityGranted":
            grants.add(event["payload"]["authority"]["authority_id"])
        if kind not in _PAYLOADS:
            continue
        fact = _fact(event)
        if isinstance(fact, InvestigationDownloadFailed):
            if any(request.pending for request in requests):
                raise ValueError("download failure beside a pending download review")
            _check_failure(fact, event, events[:index], grants)
            failures[event["event_id"]] = fact
        elif isinstance(fact, InvestigationDownloadReviewRequested):
            failure_id = fact.request.failure_event_id
            failure = failures.get(failure_id)
            if (
                failure is None
                or event["parent_ids"] != [failure_id]
                or (failure.authority_id, failure.role)
                != (fact.authority_id, fact.role)
                or fact.request.stage != DOWNLOAD_STAGES[fact.role]
                or fact.request.legal_actions != _ACTIONS
                or fact.request.amendable
            ):
                raise ValueError("download review changed its failure or scope")
            if any(request.pending for request in requests):
                raise ValueError("download review duplicated pending work")
            requests.append(
                InvestigationDownloadReviewRequest(
                    request_event_id=event["event_id"],
                    authority_id=fact.authority_id,
                    role=fact.role,
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
                validate=validate_download_answer,
                refusal=_refused,
            )
            if replay is not None or event["parent_ids"] != [fact.request_event_id]:
                raise ValueError(
                    "download review repeated its answer or changed lineage"
                )
            requests = [
                request.model_copy(update={"pending": False})
                if request.request_event_id == fact.request_event_id
                else request
                for request in requests
            ]
            receipts.append(
                InvestigationDownloadReviewReceipt(
                    request_event_id=fact.request_event_id,
                    response_event_id=event["event_id"],
                    response=fact.response,
                )
            )
    return InvestigationDownloadReviews(
        requests=tuple(requests), receipts=tuple(receipts)
    )


def ask_download_review(
    journey: Journey,
    *,
    parent_event_id: str,
    authority_id: str,
    role: str,
    object_key: str,
    code: str,
    details: Sequence[str],
) -> dict[str, Any]:
    """Record the refused download and ask its Review, from the work frontier."""
    failure = InvestigationDownloadFailed(
        authority_id=authority_id,
        role=role,
        object_key=object_key,
        code=code,
        details=tuple(details),
    )
    failed = journey.record(
        failure.event_type,
        failure.model_dump(mode="json", exclude={"event_type"}),
        [parent_event_id],
    )
    requested = InvestigationDownloadReviewRequested(
        authority_id=authority_id,
        role=role,
        request=Ask(
            stage=DOWNLOAD_STAGES[role],
            failure_event_id=failed["event_id"],
            request_id=str(uuid.uuid4()),
            reason=f"{role} download refused: {code}",
            legal_actions=_ACTIONS,
        ),
    )
    return journey.record(
        requested.event_type,
        requested.model_dump(mode="json", exclude={"event_type"}),
        [failed["event_id"]],
    )


def answer_download_review(
    journey: Journey, request_event_id: str, response: ReviewAnswer
) -> dict[str, Any]:
    """Record the one answer to a pending download Review; replay is inert."""
    from bio_neb.junction import evaluate_scoped_answer

    state = journey.checked_view("investigation_download")
    _, replay = evaluate_scoped_answer(
        state.requests,
        state.receipts,
        request_event_id,
        response,
        validate=validate_download_answer,
        refusal=_refused,
    )
    if replay is not None:
        return replay
    answered = InvestigationDownloadReviewAnswered(
        request_event_id=request_event_id, response=response
    )
    return journey.record(
        answered.event_type,
        answered.model_dump(mode="json", exclude={"event_type"}),
        [request_event_id],
    )


__all__ = [
    "DOWNLOAD_STAGES",
    "InputDownloadRefused",
    "REVIEWED_CODES",
    "InvestigationDownloadFailed",
    "InvestigationDownloadReviewAnswered",
    "InvestigationDownloadReviewReceipt",
    "InvestigationDownloadReviewRequest",
    "InvestigationDownloadReviewRequested",
    "InvestigationDownloadReviews",
    "answer_download_review",
    "ask_download_review",
    "project_download_reviews",
]
