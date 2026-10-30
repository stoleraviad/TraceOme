"""The semantic path: typed facts, immutable pointers, and one execution fold.

A writer owns one file for a pass. Effects receive only the contact journal.
The old Journey and mixed stage writers remain T8b migration consumers; this
module is the new path contract, not a second stream beside that Journey.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Sequence

from pydantic import Field, JsonValue, StrictInt, StrictStr, model_validator

from bio_neb.content import CONTENT_ID_PATTERN, ContentId, canonical_json
from bio_neb.contracts import FrozenContract
from bio_neb.failure import Ask, FailureClass, ReviewAnswer


class PathKind(StrEnum):
    DECISION = "decision"
    AUTHORITY = "authority"
    ADMISSION = "admission"
    VERIFIED_ARTIFACT = "verified_artifact"
    REVIEW = "review"
    FAILURE = "failure"
    OUTCOME = "outcome"
    OBSERVATION = "observation"


class Grounding(StrEnum):
    VERIFIED = "verified"
    TOOL_SUCCEEDED = "tool_succeeded"
    AGENT_COMPOSED = "agent_composed"
    AMBIGUOUS = "ambiguous"
    HUMAN = "human"


Name = Annotated[StrictStr, Field(min_length=1)]


class Producer(FrozenContract):
    stage: Name
    tool: Name | None = None
    authority_id: Name | None = None


class ArtifactPointer(FrozenContract):
    role: Name
    content_id: StrictStr = Field(pattern=CONTENT_ID_PATTERN)
    location: Name
    bytes: StrictInt = Field(ge=0)
    producer: Producer
    grounding: Grounding
    proof_ref: StrictStr = Field(pattern=r"^cj:[1-9][0-9]*:[1-9][0-9]*$")


def artifact_location(content_id: str) -> str:
    """The bundle-relative location of a retained local artifact."""
    return "artifacts/sha256/" + ContentId.parse(content_id).digest


class Artifact(FrozenContract):
    """A small retained document; large objects enter the path as pointers only."""

    content: bytes
    role: Name
    producer: Producer
    grounding: Grounding
    proof_ref: StrictStr = Field(pattern=r"^cj:[1-9][0-9]*:[1-9][0-9]*$")

    def pointer(self) -> ArtifactPointer:
        """Describe the immutable bundle location; this performs no write."""
        content_id = str(ContentId.of(self.content))
        return ArtifactPointer(
            role=self.role,
            content_id=content_id,
            location=artifact_location(content_id),
            bytes=len(self.content),
            producer=self.producer,
            grounding=self.grounding,
            proof_ref=self.proof_ref,
        )


class ExecutionBound(FrozenContract):
    execution_id: Name
    authority_id: Name
    proposal_id: Name
    run_id: Name
    approved_input: dict[str, JsonValue]


class Verified(FrozenContract):
    state: Literal["verified"] = "verified"
    pointers: tuple[ArtifactPointer, ...] = Field(min_length=1)
    receipt_id: Name


_SHA256 = r"^[0-9a-f]{64}$"


class ReferenceAcquisition(FrozenContract):
    """The approved reference source's bytes as measured while they streamed in.

    The publisher md5 and size matched the approval; the .gz SHA-256 and the plain
    FASTA's size and SHA-256 are acquired values nobody supplied. Input storage
    records it when the controller streamed the source; otherwise the CPU
    preparation job acquired it and ``ReferencePrepared`` records it.
    """

    source_md5: StrictStr = Field(pattern=r"^[0-9a-f]{32}$")
    source_byte_count: StrictInt = Field(gt=0)
    source_sha256: StrictStr = Field(pattern=_SHA256)
    plain_fasta_byte_count: StrictInt = Field(gt=0)
    plain_fasta_sha256: StrictStr = Field(pattern=_SHA256)


class StagedInputBasis(FrozenContract):
    """Why a staged FASTQ object is trusted: copied in the provider, never uploaded.

    Its sha256 is the staging manifest's when the manifest lists one, otherwise
    one read of the staged object; the md5 is the manifest's, when it has one.
    """

    role: Name
    source_location: StrictStr
    manifest_location: StrictStr | None
    sha256: StrictStr = Field(pattern=_SHA256)
    sha256_basis: Literal["staging_manifest", "source_read"]
    md5: StrictStr | None = Field(pattern=r"^[0-9a-f]{32}$")


class InputsAcquired(Verified):
    reference_acquisition: ReferenceAcquisition | None
    # Present only when an input was a staged object; uploads record none.
    staged_inputs: tuple[StagedInputBasis, ...] = Field(
        default=(), exclude_if=lambda staged: not staged
    )


class Unresolved(FrozenContract):
    state: Literal["unresolved"] = "unresolved"


class Hit(FrozenContract):
    state: Literal["hit"] = "hit"
    pointers: tuple[ArtifactPointer, ...] = Field(min_length=1)
    receipt_id: Name


class Miss(FrozenContract):
    state: Literal["miss"] = "miss"
    reason: Name
    evidence: tuple[ArtifactPointer, ...] = ()


class Drifted(FrozenContract):
    state: Literal["drifted"] = "drifted"
    reason: Name
    evidence: tuple[ArtifactPointer, ...] = Field(min_length=1)


class AdmissionFailed(FrozenContract):
    state: Literal["admission_failed"] = "admission_failed"
    reason: Name
    evidence: tuple[ArtifactPointer, ...] = Field(min_length=1)


class Prepared(FrozenContract):
    state: Literal["prepared"] = "prepared"
    pointers: tuple[ArtifactPointer, ...] = Field(min_length=1)
    receipt_id: Name
    reference_acquisition: ReferenceAcquisition | None


ReferenceResolution = Annotated[
    Hit | Miss | Drifted | AdmissionFailed, Field(discriminator="state")
]
ReferenceState = Annotated[
    Unresolved | Hit | Miss | Prepared | Drifted | AdmissionFailed,
    Field(discriminator="state"),
]


class ReferenceResolved(FrozenContract):
    reference: ReferenceResolution


class Pending(FrozenContract):
    """The CPU preparation Review a reference miss opens: abort or one CPU job.

    It amends nothing (owner, 2026-10-29, #464): a Run's index is the Execution
    Review's bound prebuilt index or a CPU preparation job.
    """

    state: Literal["pending"] = "pending"
    stage: Literal["cpu_preparation"] = "cpu_preparation"
    request_id: Name
    reason: Name
    legal_actions: tuple[Literal["abort", "approve_cpu_preparation"], ...] = Field(
        min_length=1
    )
    action_destinations: dict[str, Literal["stop", "parabricks"]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def exact_destinations(self):
        if self.action_destinations and self.action_destinations != {
            action: "stop" if action == "abort" else "parabricks"
            for action in self.legal_actions
        }:
            raise ValueError("index review destinations disagree with its consequences")
        return self


class Answered(ReviewAnswer):
    action: Literal["abort", "approve_cpu_preparation"]
    field: None = None
    value: None = None


class StageFailure(FrozenContract):
    module: Name
    stage: Name
    step: Name
    failure_class: FailureClass
    detail: Name
    invariant_id: Name | None = None
    context: dict[str, JsonValue] = Field(default_factory=dict)
    evidence: tuple[ArtifactPointer, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def invariant_identity(self) -> StageFailure:
        if self.failure_class == "invariant" and self.invariant_id is None:
            raise ValueError("invariant failure requires its invariant id")
        return self


class RuntimeSubstituted(FrozenContract):
    from_profile: Name
    to_profile: Name
    authority_id: Name
    stage: Name
    reason: Literal["no_capacity"] = "no_capacity"
    proof_ref: StrictStr = Field(pattern=r"^cj:[1-9][0-9]*:[1-9][0-9]*$")


class Outcome(FrozenContract):
    disposition: Literal["incomplete", "failed", "succeeded"]
    contact_journal: Name
    journal_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    journal_bytes: StrictInt | None = Field(default=None, gt=0)
    milestone: Literal["input_storage", "reference_preparation"] | None = None


# Open event names have one registry, grouped by the closed semantic kind.
PATH_REGISTRY: Mapping[PathKind, Mapping[str, type[FrozenContract]]] = {
    PathKind.AUTHORITY: {
        "ExecutionBound": ExecutionBound,
        "CpuPreparationReviewAnswered": Answered,
        "FailureReviewAnswered": ReviewAnswer,
    },
    PathKind.VERIFIED_ARTIFACT: {
        "InputsVerified": InputsAcquired,
        "ReferencePrepared": Prepared,
        "H100ArtifactsVerified": Verified,
    },
    PathKind.DECISION: {
        "ReferenceResolved": ReferenceResolved,
        "RuntimeSubstituted": RuntimeSubstituted,
    },
    PathKind.REVIEW: {
        "CpuPreparationReviewRequested": Pending,
        "FailureReviewRequested": Ask,
    },
    PathKind.FAILURE: {"StageFailed": StageFailure},
    PathKind.OUTCOME: {"ExecutionOutcome": Outcome},
    PathKind.ADMISSION: {},
    PathKind.OBSERVATION: {},
}
_REST_KINDS = {
    PathKind.DECISION,
    PathKind.OUTCOME,
    PathKind.VERIFIED_ARTIFACT,
    PathKind.REVIEW,
}


class PathFact(FrozenContract):
    event_type: Name
    kind: PathKind
    grounding: Grounding
    pass_: StrictInt = Field(alias="pass", ge=1)
    rests_on: tuple[Name, ...] = ()
    payload: (
        ExecutionBound
        | InputsAcquired
        | Verified
        | ReferenceResolved
        | Pending
        | Answered
        | Prepared
        | StageFailure
        | Outcome
        | Ask
        | ReviewAnswer
        | RuntimeSubstituted
    )

    @model_validator(mode="before")
    @classmethod
    def typed_payload(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        kind = PathKind(value.get("kind"))
        contract = PATH_REGISTRY[kind].get(value.get("event_type"))
        if contract is None:
            raise ValueError("event_type is not registered under its kind")
        return {**value, "payload": contract.model_validate(value.get("payload"))}

    @model_validator(mode="after")
    def registered_contract(self) -> PathFact:
        contract = PATH_REGISTRY[self.kind].get(self.event_type)
        if contract is None:
            raise ValueError("event_type is not registered under its kind")
        payload = contract.model_validate(self.payload)
        object.__setattr__(self, "payload", payload)
        dependent = self.kind in _REST_KINDS or self.event_type.endswith(
            "ReviewAnswered"
        )
        # A failure may rest only on the held failure its re-entry pass resolves.
        if not dependent and self.rests_on and self.kind != PathKind.FAILURE:
            raise ValueError(
                "rests_on is only legal on decision/outcome/verified_artifact/review "
                "and applicable authority answers"
            )
        if dependent and not self.rests_on:
            raise ValueError("this path fact requires rests_on")
        if len(set(self.rests_on)) != len(self.rests_on):
            raise ValueError("duplicate rests_on")
        if (
            self.event_type
            in {
                "ExecutionBound",
                "RuntimeSubstituted",
                "CpuPreparationReviewAnswered",
                "FailureReviewAnswered",
            }
            and self.grounding != Grounding.HUMAN
        ):
            raise ValueError("authority and review answers require human grounding")
        if (
            self.kind == PathKind.VERIFIED_ARTIFACT
            and self.grounding != Grounding.VERIFIED
        ):
            raise ValueError("verified artifacts require verified grounding")
        pointers = getattr(self.payload, "pointers", ())
        if isinstance(self.payload, ReferenceResolved) and isinstance(
            self.payload.reference, Hit
        ):
            pointers = self.payload.reference.pointers
        for pointer in pointers:
            expected = (
                Grounding.AGENT_COMPOSED
                if self.event_type == "H100ArtifactsVerified"
                and pointer.role == "execution_result_context"
                else Grounding.VERIFIED
            )
            if pointer.grounding != expected:
                raise ValueError(
                    "verified closure contains an unverified artifact pointer"
                )
        return self


class PathEvent(PathFact):
    event_id: Name


class NotStarted(FrozenContract):
    state: Literal["not_started"] = "not_started"


ArtifactState = Annotated[NotStarted | Verified, Field(discriminator="state")]
InputState = Annotated[NotStarted | InputsAcquired, Field(discriminator="state")]
ReviewState = Annotated[Pending | Answered, Field(discriminator="state")]


class ExecutionState(FrozenContract):
    pass_: StrictInt = Field(default=1, alias="pass", ge=1)
    bound: ExecutionBound | None = None
    inputs: InputState = NotStarted()
    reference: ReferenceState = Unresolved()
    review: ReviewState | None = None
    h100: ArtifactState = NotStarted()
    failures: tuple[StageFailure, ...] = ()
    failure_event_ids: tuple[str, ...] = ()
    resolved_failure_ids: tuple[str, ...] = ()
    review_pass: StrictInt | None = None
    failure_review: (
        Annotated[Ask | ReviewAnswer, Field(discriminator="state")] | None
    ) = None
    failure_review_pass: StrictInt | None = None
    runtime_profile_id: str | None = None
    amendments: dict[str, JsonValue] = Field(default_factory=dict)
    amendment_event_ids: dict[str, str] = Field(default_factory=dict)
    stage_answer_ids: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    job_intent_answer_ids: dict[str, str] = Field(default_factory=dict)
    outcome: Outcome | None = None
    event_ids: tuple[str, ...] = ()
    rests_on: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    latest_event_ids: dict[str, str] = Field(default_factory=dict)

    @property
    def unresolved_failure_ids(self) -> tuple[str, ...]:
        return tuple(
            item
            for item in self.failure_event_ids
            if item not in self.resolved_failure_ids
        )

    @property
    def held_failure_id(self) -> str | None:
        """An unresolved invariant with no outcome holds the Execution for a fix.

        It ends nothing: a ``resume`` re-enters the next pass resting on it (#218).
        """
        if self.outcome is not None:
            return None
        failures = dict(zip(self.failure_event_ids, self.failures, strict=True))
        return next(
            (
                event_id
                for event_id in reversed(self.unresolved_failure_ids)
                if failures[event_id].failure_class == "invariant"
            ),
            None,
        )

    def reentered(self) -> ExecutionState:
        """The held Execution's next pass before its first fact, failure resolved."""
        return self.model_copy(
            update={
                "pass_": self.pass_ + 1,
                "resolved_failure_ids": (
                    *self.resolved_failure_ids,
                    self.held_failure_id,
                ),
            }
        )

    @property
    def disposition(self) -> Literal["incomplete", "failed", "succeeded"]:
        return self.outcome.disposition if self.outcome else "incomplete"

    @property
    def frontier(self) -> str:
        if self.disposition == "failed":
            return "failed"
        if self.held_failure_id is not None:
            return "held"
        if isinstance(self.failure_review, Ask):
            return "failure_review_pending"
        if isinstance(self.review, Pending):
            return "cpu_preparation_review_pending"
        if self.unresolved_failure_ids:
            return "failed"
        if isinstance(self.h100, Verified):
            return "artifacts_verified"
        if isinstance(self.reference, (Prepared, Hit)):
            return "prepared_reference"
        if isinstance(self.reference, (Drifted, AdmissionFailed)):
            return "reference_drifted"
        if isinstance(self.reference, Miss):
            return "cache_miss"
        return "inputs_verified" if isinstance(self.inputs, Verified) else "not_started"

    @property
    def status(self) -> str:
        if (
            self.frontier in {"cache_miss", "reference_drifted"}
            and isinstance(self.review, Answered)
            and self.review.action == "approve_cpu_preparation"
        ):
            return "cpu_preparation_authorized"
        return {
            "failed": "execution_failed",
            "held": "run_held",
            "failure_review_pending": "execution_review_pending",
            "artifacts_verified": "execution_artifacts_verified",
            "cpu_preparation_review_pending": "cpu_preparation_review_pending",
            "prepared_reference": "prepared_reference_cache_hit"
            if isinstance(self.reference, Hit)
            else "prepared_reference_verified",
            "reference_drifted": "prepared_reference_admission_drifted",
            "cache_miss": "prepared_reference_cache_miss",
            "inputs_verified": "execution_inputs_verified",
            "not_started": "execution_inputs_pending",
        }[self.frontier]


def _apply_verified(state: ExecutionState, event: PathEvent) -> dict[str, Any]:
    if event.event_type == "InputsVerified":
        if not isinstance(state.inputs, NotStarted):
            raise ValueError("inputs already verified")
        return {"inputs": event.payload}
    if not isinstance(state.inputs, Verified):
        raise ValueError("reference/H100 verification requires verified inputs")
    if event.event_type == "ReferencePrepared":
        # The Review's bound prebuilt index is the Run's only candidate;
        # without one the Run prepares on CPU.
        reference = state.bound.approved_input["proposal"]["execution_package"][
            "reference"
        ]
        no_candidate = reference.get("prebuilt_index") is None
        if isinstance(state.reference, Unresolved):
            if not no_candidate:
                raise ValueError(
                    "selected candidate requires admission or review approval"
                )
        elif not isinstance(state.reference, (Miss, Drifted, AdmissionFailed)):
            raise ValueError("CPU preparation requires a reference miss or drift")
        if (
            not no_candidate
            or state.review is not None
            or isinstance(state.reference, (Drifted, AdmissionFailed))
        ) and (
            not isinstance(state.review, Answered)
            or state.review.action != "approve_cpu_preparation"
        ):
            raise ValueError("CPU preparation lacks review approval")
        return {"reference": event.payload}
    if not isinstance(state.reference, (Hit, Prepared)):
        raise ValueError("H100 verification requires a prepared reference")
    if isinstance(state.h100, Verified):
        raise ValueError("H100 already verified")
    return {"h100": event.payload}


def _amendment(state, event):
    answer = event.payload
    if answer.action != "amend":
        return {}
    return {
        **({"runtime_profile_id": None} if answer.field == "gpu_class" else {}),
        "amendments": {**state.amendments, answer.field: answer.value},
        "amendment_event_ids": {
            **state.amendment_event_ids,
            answer.field: event.event_id,
        },
    }


def _apply_review(state: ExecutionState, event: PathEvent) -> dict[str, Any]:
    if isinstance(event.payload, Pending):
        request = event.payload
        if isinstance(state.review, Pending) or isinstance(state.failure_review, Ask):
            raise ValueError("review already pending")
        if not isinstance(state.reference, (Miss, Drifted, AdmissionFailed)):
            raise ValueError("CPU review requires a miss or drift")
        if state.unresolved_failure_ids:
            raise ValueError("stage failures require generic failure review")
        return {"review": request, "review_pass": event.pass_}
    _validate_review_answer(state.review, state.review_pass, event, Pending)
    return {"review": event.payload, **_amendment(state, event)}


def _validate_review_answer(request, request_pass, event, request_type):
    """Shared exact HITL validation; each scenario owns its consequences."""
    answer = event.payload
    if not isinstance(request, request_type) or request.request_id != answer.request_id:
        raise ValueError("review answer has no matching request")
    if request_pass is None or event.pass_ != request_pass + 1:
        raise ValueError("review answer requires the next pass after a designed stop")
    if exceeds_declared_authority(request, answer):
        raise ValueError("review answer exceeds declared authority")


def exceeds_declared_authority(request, answer) -> bool:
    """An action the request does not declare, or an amendment of an undeclared field."""
    return answer.action not in request.legal_actions or (
        answer.action == "amend" and answer.field not in request.amendable
    )


def _apply_failure_review(state: ExecutionState, event: PathEvent) -> dict[str, Any]:
    if isinstance(event.payload, Ask):
        request = event.payload
        if isinstance(state.failure_review, Ask) or isinstance(state.review, Pending):
            raise ValueError("review already pending")
        failures = dict(zip(state.failure_event_ids, state.failures, strict=True))
        failure = failures.get(request.failure_event_id)
        if (
            failure is None
            or request.failure_event_id not in state.unresolved_failure_ids
            or failure.stage != request.stage
        ):
            raise ValueError("review requires a matching unresolved stage failure")
        if failure.failure_class == "invariant":
            raise ValueError(
                "an invariant failure holds for a fix and cannot be reviewed away"
            )
        if request.failure_event_id not in event.rests_on:
            raise ValueError("failure review rests_on omits its failure")
        from bio_neb.failure import retry_consequence

        if request.retry_consequence != retry_consequence(failure.context):
            raise ValueError(
                "review consequence disagrees with retained resource evidence"
            )
        if request.amendment_consequences != {
            field: "new_job_intent" for field in request.amendable
        }:
            raise ValueError("review amendment consequence disagrees with its fields")
        return {"failure_review": request, "failure_review_pass": event.pass_}
    request = state.failure_review
    answer = event.payload
    _validate_review_answer(request, state.failure_review_pass, event, Ask)
    resolved = state.resolved_failure_ids
    if answer.action in {"retry", "amend"}:
        resolved += (request.failure_event_id,)
    replaces = (
        answer.action == "retry" and request.retry_consequence == "replace_job"
    ) or (answer.action == "amend" and answer.field == "gpu_class")
    return {
        "job_intent_answer_ids": {
            **state.job_intent_answer_ids,
            **({request.stage: event.event_id} if replaces else {}),
        },
        "failure_review": answer,
        "resolved_failure_ids": resolved,
        "stage_answer_ids": {
            **state.stage_answer_ids,
            request.stage: (
                *state.stage_answer_ids.get(request.stage, ()),
                event.event_id,
            ),
        },
        **_amendment(state, event),
    }


def _apply_runtime(state, event):
    substitution = event.payload
    selection = state.bound.approved_input.get("proposal", {}).get("runtime_selection")
    if selection is None or substitution.authority_id != state.bound.authority_id:
        raise ValueError("runtime substitution lacks approved alternatives")
    from bio_neb.parabricks.capability_policy import ExecutionRuntimeSelection

    selected = ExecutionRuntimeSelection.model_validate(selection).select(
        amended_profile_id=state.amendments.get("gpu_class"),
        substituted_profile_id=state.runtime_profile_id,
    )
    if selected.next_profile_id is None:
        raise ValueError("approved runtime alternatives exhausted")
    if (
        substitution.from_profile != selected.profile_id
        or substitution.to_profile != selected.next_profile_id
    ):
        raise ValueError("runtime substitution must take the next approved alternative")
    if isinstance(state.h100, Verified):
        raise ValueError("runtime cannot change after H100 verification")
    return {"runtime_profile_id": substitution.to_profile}


def _apply_outcome(state, event):
    outcome = event.payload
    if outcome.disposition == "succeeded" and state.unresolved_failure_ids:
        raise ValueError("unresolved stage failure prevents success")
    if outcome.disposition == "succeeded" and not isinstance(state.h100, Verified):
        raise ValueError("success requires verified H100 artifacts")
    if (
        outcome.disposition == "failed"
        and not state.unresolved_failure_ids
        and not (isinstance(state.review, Answered) and state.review.action == "abort")
    ):
        raise ValueError("failed outcome lacks failure or human abort")
    if outcome.disposition == "incomplete" and not (
        isinstance(state.review, Pending)
        or isinstance(state.failure_review, Ask)
        or (
            outcome.milestone == "reference_preparation"
            and isinstance(state.reference, (Hit, Prepared))
        )
        or (
            outcome.milestone == "input_storage"
            and isinstance(state.inputs, Verified)
            and isinstance(state.reference, Unresolved)
            and not state.unresolved_failure_ids
        )
    ):
        raise ValueError("continuation is only legal at a designed review stop")
    if outcome.milestone == "reference_preparation":
        if (
            outcome.disposition != "incomplete"
            or not isinstance(state.reference, (Hit, Prepared))
            or isinstance(state.review, Pending)
            or isinstance(state.failure_review, Ask)
            or state.unresolved_failure_ids
        ):
            raise ValueError("reference milestone requires verified preparation")
        return {"outcome": outcome}
    if outcome.milestone is not None and (
        outcome.disposition != "incomplete"
        or state.review is not None
        or isinstance(state.failure_review, Ask)
        or not isinstance(state.reference, Unresolved)
    ):
        raise ValueError("input milestone is a non-continuable stop")
    return {"outcome": outcome}


def _apply_fact(state: ExecutionState, event: PathEvent) -> dict[str, Any]:
    # Registration admits a payload shape, not an execution interpretation.
    if event.event_type not in {
        "ExecutionBound",
        "InputsVerified",
        "ReferenceResolved",
        "ReferencePrepared",
        "H100ArtifactsVerified",
        "CpuPreparationReviewRequested",
        "CpuPreparationReviewAnswered",
        "FailureReviewRequested",
        "FailureReviewAnswered",
        "RuntimeSubstituted",
        "StageFailed",
        "ExecutionOutcome",
    }:
        raise ValueError("no execution interpretation for this registered event")
    if (
        isinstance(state.failure_review, ReviewAnswer)
        and state.failure_review.action == "abort"
        and event.event_type != "ExecutionOutcome"
    ):
        raise ValueError("human abort requires a terminal outcome")
    if state.unresolved_failure_ids and event.kind not in {
        PathKind.FAILURE,
        PathKind.REVIEW,
        PathKind.AUTHORITY,
        PathKind.OUTCOME,
    }:
        raise ValueError("unresolved stage failure prevents further progress")
    if isinstance(state.review, Pending) and event.event_type not in {
        "ExecutionOutcome",
        "CpuPreparationReviewAnswered",
    }:
        raise ValueError(
            "pending review requires a designed stop before further progress"
        )
    if isinstance(state.failure_review, Ask) and event.event_type not in {
        "ExecutionOutcome",
        "FailureReviewAnswered",
    }:
        raise ValueError(
            "pending review requires a designed stop before further progress"
        )
    match event.kind:
        case PathKind.AUTHORITY:
            if event.event_type == "CpuPreparationReviewAnswered":
                return _apply_review(state, event)
            if event.event_type == "FailureReviewAnswered":
                return _apply_failure_review(state, event)
            if state.bound is not None:
                raise ValueError("execution already bound")
            return {"bound": event.payload}
        case PathKind.VERIFIED_ARTIFACT:
            return _apply_verified(state, event)
        case PathKind.DECISION:
            if isinstance(event.payload, RuntimeSubstituted):
                return _apply_runtime(state, event)
            if isinstance(state.h100, Verified) or not isinstance(
                state.reference, Unresolved
            ):
                raise ValueError("reference resolution requires an unresolved decision")
            if not isinstance(state.inputs, Verified):
                raise ValueError("reference resolution requires verified inputs")
            return {
                "reference": event.payload.reference,
                "review": None,
                "review_pass": None,
            }
        case PathKind.REVIEW:
            if isinstance(event.payload, (Ask, ReviewAnswer)):
                return _apply_failure_review(state, event)
            return _apply_review(state, event)
        case PathKind.FAILURE:
            return {
                "failures": (*state.failures, event.payload),
                "failure_event_ids": (*state.failure_event_ids, event.event_id),
            }
        case PathKind.OUTCOME:
            return _apply_outcome(state, event)
        case _:
            raise ValueError("no execution interpretation for this registered kind")


def _require_dependencies(
    event: PathEvent, latest: Mapping[str, str], state: ExecutionState
) -> None:
    stage = {
        "InputsVerified": "input_storage",
        "ReferenceResolved": "admit_candidate",
        "ReferencePrepared": "cpu_preparation",
        "H100ArtifactsVerified": "h100",
    }.get(event.event_type)
    if not set(state.stage_answer_ids.get(stage, ())) <= set(event.rests_on):
        raise ValueError("rests_on omits the enabling stage answer")
    required = {
        "InputsVerified": ("ExecutionBound",),
        "RuntimeSubstituted": ("ExecutionBound",),
        "ReferenceResolved": ("InputsVerified",),
        "CpuPreparationReviewRequested": ("ReferenceResolved",),
        "CpuPreparationReviewAnswered": ("CpuPreparationReviewRequested",),
        "FailureReviewRequested": ("StageFailed",),
        "FailureReviewAnswered": ("FailureReviewRequested",),
        "ReferencePrepared": (
            "InputsVerified",
            "ExecutionBound"
            if isinstance(state.reference, Unresolved)
            else "ReferenceResolved",
        ),
        "H100ArtifactsVerified": (
            "InputsVerified",
            "ReferencePrepared"
            if "ReferencePrepared" in latest
            else "ReferenceResolved",
        ),
    }.get(event.event_type, ())
    if event.event_type == "RuntimeSubstituted":
        prior = state.amendment_event_ids.get("gpu_class")
        if prior is not None and prior not in event.rests_on:
            raise ValueError("runtime substitution omits its amendment authority")
        if "RuntimeSubstituted" in latest:
            required += ("RuntimeSubstituted",)
    if event.event_type == "ReferencePrepared" and isinstance(state.review, Answered):
        required += ("CpuPreparationReviewAnswered",)
    if event.event_type == "ExecutionOutcome":
        aborted = isinstance(state.review, Answered) and state.review.action == "abort"
        failure_aborted = (
            isinstance(state.failure_review, ReviewAnswer)
            and state.failure_review.action == "abort"
        )
        required = (
            {
                "succeeded": "H100ArtifactsVerified",
                "incomplete": "FailureReviewRequested"
                if isinstance(state.failure_review, Ask)
                else "InputsVerified"
                if event.payload.milestone == "input_storage"
                else (
                    "ReferencePrepared"
                    if isinstance(state.reference, Prepared)
                    else "ReferenceResolved"
                )
                if event.payload.milestone == "reference_preparation"
                else "CpuPreparationReviewRequested",
                "failed": "FailureReviewAnswered"
                if failure_aborted
                else "CpuPreparationReviewAnswered"
                if aborted
                else "StageFailed",
            }[event.payload.disposition],
        )
    if event.event_type.endswith("ReviewAnswered") and event.rests_on != (
        latest.get(required[0]),
    ):
        raise ValueError("review authority requires exactly its current request")
    if any(latest.get(name) not in event.rests_on for name in required):
        raise ValueError("rests_on omits the current prerequisite fact")


def apply_execution_event(
    state: ExecutionState,
    raw: PathEvent | Mapping[str, Any],
) -> ExecutionState:
    """Apply one fact in memory using the same rules as the pass-start fold."""
    event = PathEvent.model_validate(raw)
    seen = set(state.event_ids)
    if event.event_id in seen or not set(event.rests_on) <= seen:
        raise ValueError("duplicate event or rests_on outside the preceding prefix")
    if state.bound is None and (
        event.event_type != "ExecutionBound" or event.pass_ != 1
    ):
        raise ValueError("execution path must begin with ExecutionBound")
    if event.pass_ < state.pass_ or event.pass_ > state.pass_ + 1:
        raise ValueError("execution pass is not sequential")
    if state.outcome is not None and state.outcome.disposition != "incomplete":
        raise ValueError("terminal execution cannot continue")
    if event.pass_ != state.pass_ and state.held_failure_id is not None:
        # After a fix, ``resume`` re-enters a held Execution: the new pass's first
        # fact rests on the held failure and resolves it (Q2).
        if state.held_failure_id not in event.rests_on or (
            event.kind == PathKind.FAILURE
            and event.rests_on != (state.held_failure_id,)
        ):
            raise ValueError("held execution re-entry must rest on its held failure")
        state = state.reentered()
    elif event.pass_ != state.pass_:
        if state.outcome is None or not (
            isinstance(state.review, Pending) or isinstance(state.failure_review, Ask)
        ):
            raise ValueError("new pass requires a designed stop")
        answer_type = (
            "FailureReviewAnswered"
            if isinstance(state.failure_review, Ask)
            else "CpuPreparationReviewAnswered"
        )
        if event.event_type != answer_type:
            raise ValueError("new pass must answer its current review")
        state = state.model_copy(update={"outcome": None})
    elif state.outcome is not None:
        raise ValueError("closed pass cannot append")
    elif event.kind == PathKind.FAILURE and event.rests_on:
        raise ValueError("a failure rests only on the held failure it re-enters")
    updates = _apply_fact(state, event)
    _require_dependencies(event, state.latest_event_ids, state)
    return state.model_copy(
        update={
            **updates,
            "pass_": event.pass_,
            "event_ids": (*state.event_ids, event.event_id),
            "rests_on": {**state.rests_on, event.event_id: event.rests_on},
            "latest_event_ids": {
                **state.latest_event_ids,
                event.event_type: event.event_id,
            },
        }
    )


def project_execution(
    events: Sequence[PathEvent | Mapping[str, Any]],
) -> ExecutionState:
    """Fold a pass-start snapshot once; preserve evidence ids and dependency edges."""
    state = ExecutionState()
    for event in events:
        state = apply_execution_event(state, event)
    return state


def _artifact_pointers(payload: FrozenContract) -> tuple[ArtifactPointer, ...]:
    if isinstance(payload, (Verified, Prepared)):
        return payload.pointers
    if isinstance(payload, StageFailure):
        return payload.evidence
    if isinstance(payload, ReferenceResolved):
        reference = payload.reference
        return reference.pointers if isinstance(reference, Hit) else reference.evidence
    return ()


def _historical_execution_line(value) -> bool:
    """Legacy execution lines carry ``kind``; the semantic writer never wrote them."""
    return isinstance(value, dict) and "kind" in value


def is_historical_execution_file(journey_file) -> bool:
    """Whether a parseable retained file holds legacy execution lines (no owner opens it)."""
    file = Path(journey_file)
    if not file.is_file():
        return False
    try:
        values = [json.loads(line) for line in file.read_bytes().splitlines()]
    except ValueError:
        return False
    return any(_historical_execution_line(value) for value in values)


def has_execution_path(journey_file):
    from bio_neb.journey import FORK_ORIGIN_EVENT_TYPE, Journey

    file = Path(journey_file)
    owner = Journey.processing_owner(file)
    if owner is not None:
        return owner.has_event_type("ExecutionPathRecorded")
    if not file.exists():
        return False
    with file.open() as stream:
        for index, line in enumerate(stream):
            value = json.loads(line)
            if index == 0 and value.get("event_type") == FORK_ORIGIN_EVENT_TYPE:
                # A Fork child's inherited prefix lives in its origin's artifact.
                return Journey.open(file).has_event_type("ExecutionPathRecorded")
            if (
                _historical_execution_line(value)
                or value.get("event_type") == "ExecutionPathRecorded"
            ):
                return True
    return False


def _recorded_execution_fact(event) -> PathEvent:
    """Admit one retained domain fact in its semantic envelope."""
    fact = PathEvent.model_validate(event["payload"])
    if fact.event_id != event["event_id"] or list(fact.rests_on) != event["parent_ids"]:
        raise ValueError("execution fact changed its semantic envelope")
    return fact


def recorded_execution(events):
    """Admit the domain facts retained by the common semantic writer."""
    return project_execution(
        [
            _recorded_execution_fact(event)
            for event in events
            if event["event_type"] == "ExecutionPathRecorded"
        ]
    )


def step_recorded_execution(state: ExecutionState, event, prefix) -> ExecutionState:
    """The writer's step: ``recorded_execution([*prefix, event])`` from its state."""
    if event["event_type"] != "ExecutionPathRecorded":
        return state
    return apply_execution_event(state, _recorded_execution_fact(event))


def execution_contact_identity(event):
    fact = PathEvent.model_validate(event["payload"])
    if not isinstance(fact.payload, ExecutionBound):
        raise ValueError("execution contact owner is not its exact bound operation")
    return fact.payload.execution_id


def validate_execution_contacts(rows):
    """The domain validates its schema and any claimed scope completeness.

    A Pass journal also holds other domains' rows; an Investigation grant's GET
    rows (#107) are its own family's to validate.
    """
    from bio_neb.investigation.observation_inputs import CONTACT_SCHEMAS
    from bio_neb.parabricks.execution_contact_accounting import (
        ExecutionContactJournalRow,
        _ScannedJournal,
    )

    prefix = b""
    for row in rows:
        prefix += (
            canonical_json(row.model_dump(mode="json", exclude_unset=True)) + "\n"
        ).encode()
        domain = row.domain
        if (
            domain is None
            or domain.phase == "started"
            or domain.payload.get("schema_version") in CONTACT_SCHEMAS
        ):
            continue
        detail = ExecutionContactJournalRow.model_validate(domain.payload)
        if detail.execution_id != domain.operation_id:
            raise ValueError("execution contact changed its operation identity")
        closed = detail.row_kind == "final"
        if closed != (domain.phase == "closed") or domain.complete != detail.complete:
            raise ValueError("execution contact changed its scoped closure")
        if (
            closed
            and domain.complete
            and not _ScannedJournal(
                prefix, domain.operation_id, domain.iteration
            ).complete
        ):
            raise ValueError("execution contact falsely claims complete scope")


def _check_bundle_metadata(directory: Path, pointer: ArtifactPointer) -> None:
    expected = artifact_location(pointer.content_id)
    if (
        pointer.location != expected
        or (directory / expected).stat().st_size != pointer.bytes
    ):
        raise ValueError("bundle artifact metadata changed")


def _verified_bundle_bytes(file: Path, pointer: ArtifactPointer) -> bytes:
    content = file.read_bytes()
    if len(content) != pointer.bytes or ContentId.of(content) != ContentId.parse(
        pointer.content_id
    ):
        raise ValueError("artifact integrity changed")
    return content


class JourneyPath:
    """One append-only semantic file, opened once per pass with a prefix digest.

    Prefix validation is an open-time operation; no file polling on each append.
    Callers own the single-writer lifetime. Artifacts are content-addressed and
    immutable; reopening checks local metadata (once per admitted owner), explicit
    reads check integrity (an owner verifies each content address once).
    """

    def __init__(
        self,
        file: Path,
        events: tuple[PathEvent, ...],
        prefix: bytes,
        planning=(),
        planning_owner=None,
    ):
        self.file = file
        self._events = list(events)
        self._prefix = prefix
        self._planning = planning
        self._planning_owner = planning_owner
        self._state: ExecutionState | None = None
        self._write_failed = False
        self._owner = None
        self._historical = False
        self._reentry: tuple[ExecutionState, str] | None = None

    @classmethod
    def create(cls, directory: Path) -> JourneyPath:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / "journey.jsonl"
        file.touch(exist_ok=False)
        return cls(file, (), b"")

    @classmethod
    def open(
        cls, file: Path, *, expected_prefix_sha256: str | None = None
    ) -> JourneyPath:
        file = Path(file)
        from bio_neb.journey import Journey, JourneyEvent

        active_owner = Journey.processing_owner(file)
        prefix = (
            active_owner.retained_prefix
            if active_owner is not None
            else file.read_bytes()
        )
        if (
            expected_prefix_sha256 is not None
            and hashlib.sha256(prefix).hexdigest() != expected_prefix_sha256
        ):
            raise ValueError("Journey prefix changed")
        if prefix and not prefix.endswith(b"\n"):
            raise ValueError("truncated Journey path")
        values = [json.loads(line) for line in prefix.splitlines()]
        if values and not any(_historical_execution_line(value) for value in values):
            owner = active_owner or Journey.open(file)
            return cls.from_journey(owner)

        planning, execution = [], []
        for line in prefix.splitlines():
            value = json.loads(line)
            if _historical_execution_line(value):
                execution.append(PathEvent.model_validate(value))
            elif execution:
                raise ValueError("planning event after execution path began")
            else:
                planning.append(JourneyEvent.model_validate(value))
        planning_owner = None
        if planning:
            # Validate the original sequence and its artifacts without rewriting
            # a byte or creating a second authoritative file.
            planning_owner = Journey(file, planning[0].journey_id, planning)
        events = tuple(execution)
        seen: set[str] = {event.event_id for event in planning}
        for event in events:
            if event.event_id in seen or not set(event.rests_on) <= seen:
                raise ValueError(
                    "duplicate event or rests_on outside the preceding prefix"
                )
            seen.add(event.event_id)
            for pointer in _artifact_pointers(event.payload):
                if pointer.location.startswith("artifacts/"):
                    expected = artifact_location(pointer.content_id)
                    if (
                        pointer.location != expected
                        or (file.parent / expected).stat().st_size != pointer.bytes
                    ):
                        raise ValueError("bundle artifact metadata changed")
        result = cls(file, events, prefix, tuple(planning), planning_owner)
        result._historical = bool(planning and execution)
        return result

    def planning_events(self) -> tuple[dict, ...]:
        """The immutable pre-execution prefix used by the approval bridge."""
        if self._owner is not None:
            events = self._owner.read()
            end = next(
                (
                    index
                    for index, event in enumerate(events)
                    if event["event_type"] == "ExecutionPathRecorded"
                ),
                len(events),
            )
            return tuple(events[:end])
        return tuple(event.model_dump(mode="json") for event in self._planning)

    def planning_evidence(self) -> tuple[dict, ...]:
        """Correlated pre-execution evidence for readers, without authority edges."""
        from bio_neb.recording import read_evidence

        if self._owner is not None:
            events = read_evidence(self._owner)
            end = next(
                (
                    index
                    for index, event in enumerate(events)
                    if event["event_type"] == "ExecutionPathRecorded"
                ),
                len(events),
            )
            return tuple(events[:end])
        return (
            tuple(
                event
                for event in read_evidence(self._planning_owner)
                if event["event_type"] != "ExecutionPathRecorded"
            )
            if self._planning_owner is not None
            else ()
        )

    @property
    def prefix_sha256(self) -> str:
        if self._owner is not None:
            return hashlib.sha256(self._owner.retained_prefix).hexdigest()
        return hashlib.sha256(self._prefix).hexdigest()

    @classmethod
    def from_journey(cls, owner) -> JourneyPath:
        """Read an admitted semantic owner, including an explicitly read-only prefix."""
        result = cls(owner.path, (), owner.retained_prefix, planning_owner=owner)
        result._owner = owner
        # Local metadata is checked once per owner and pointer; an independent
        # open repeats it, and explicit reads still verify the bytes.
        for event_id, pointer in result.local_artifact_pointers():
            owner.derived_once(
                (
                    "bundle_metadata",
                    event_id,
                    pointer.content_id,
                    pointer.location,
                    pointer.bytes,
                ),
                partial(_check_bundle_metadata, owner.path.parent, pointer),
            )
        return result

    def recorded_in(self, journey_file: str) -> bool:
        """Whether ``journey_file`` names a Run whose Journey recorded this Execution.

        A Fork child inherits its parent's Execution facts under the parent's
        Journey identity, whose file lives in the directory that identity names.
        """
        if journey_file == str(self.file):
            return True
        if self._owner is None:
            return False
        inherited = {
            event["journey_id"]
            for event in self._owner.events_view()
            if event["event_type"] == "ExecutionPathRecorded"
        } - {self._owner.journey_id}
        return Path(journey_file).parent.name in inherited

    def local_artifact_pointers(self) -> tuple[tuple[str, ArtifactPointer], ...]:
        """Execution-owned local evidence with its retaining event identity."""
        return tuple(
            (event.event_id, pointer)
            for event in self.read()
            for pointer in _artifact_pointers(event.payload)
            if pointer.location.startswith("artifacts/")
        )

    def read(self) -> tuple[PathEvent, ...]:
        if self._owner is not None:
            return tuple(
                PathEvent.model_validate(event["payload"])
                for event in self._owner.events_view()
                if event["event_type"] == "ExecutionPathRecorded"
            )
        return tuple(
            PathEvent.model_validate_json(e.model_dump_json(by_alias=True))
            for e in self._events
        )

    @property
    def state(self) -> ExecutionState:
        """Fold once on first use, then expose an isolated engine-owned snapshot."""
        if self._owner is not None:
            return self._owner.checked_view("execution")
        if self._state is None:
            self._state = project_execution(self._events)
        return self._state.model_copy(deep=True)

    def preview(self, state: ExecutionState, fact: PathFact) -> ExecutionState:
        """Pure lawful-transition preview for plan and accounting preparation."""
        event, candidate = self._candidate(state, fact)
        if self._owner is not None:
            # The owner validates against its own held prefix; no caller snapshot.
            self._owner._preview_candidate(
                "ExecutionPathRecorded",
                event.model_dump(mode="json", by_alias=True),
                list(event.rests_on),
                event_id=event.event_id,
            )
        return candidate

    def reenter(self, state: ExecutionState) -> ExecutionState:
        """Open the next pass of a held Execution after its fix (#218, Q2).

        The view is that pass before its first fact. This writer rests the first
        fact on the held failure; the fold resolves it on that append.
        """
        held = state.held_failure_id
        if held is None or state != self.state:
            raise ValueError("only the retained held execution re-enters")
        view = state.reentered()
        self._reentry = (view, held)
        return view.model_copy(deep=True)

    def _candidate(self, state: ExecutionState, fact: PathFact):
        if self._historical:
            raise ValueError("historical_execution_is_read_only")
        if self._write_failed:
            raise ValueError("Journey writer is unusable after an I/O failure")
        retained = self.state
        if self._reentry is not None and state == self._reentry[0]:
            view, held = self._reentry
            if fact.pass_ != view.pass_:
                raise ValueError("held execution re-entry writes its next pass")
            if held not in fact.rests_on:
                fact = fact.model_copy(update={"rests_on": (*fact.rests_on, held)})
        elif state != retained:
            raise ValueError(
                "execution state does not match the writer semantic prefix"
            )
        event = PathEvent.model_validate(
            {
                **fact.model_dump(mode="json", by_alias=True),
                "event_id": str(uuid.uuid4()),
            }
        )
        if event.event_id in {item.event_id for item in self._planning}:
            raise ValueError("duplicate event identity in planning prefix")
        return event, apply_execution_event(retained, event)

    def path(self, state: ExecutionState, fact: PathFact) -> tuple[str, ExecutionState]:
        """Record one lawful fact and return its real identity and next state.

        The shared reducer validates before byte persistence. No caller replays
        a fabricated identity; failure never advances the cached semantic prefix.
        A failed write poisons this writer because partial-write recovery is not
        part of the single-writer contract.
        """
        if self._owner is not None and self._owner.read_only:
            raise ValueError("retained Journey view is read-only")
        event, candidate = self._candidate(state, fact)
        if self._owner is not None:
            self._owner.record(
                "ExecutionPathRecorded",
                event.model_dump(mode="json", by_alias=True),
                list(event.rests_on),
                event_id=event.event_id,
            )
            self._reentry = None
            return event.event_id, self.state
        content = (
            canonical_json(event.model_dump(mode="json", by_alias=True)) + "\n"
        ).encode()
        try:
            with self.file.open("ab") as stream:
                if stream.write(content) != len(content):
                    raise OSError("short Journey write")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            self._write_failed = True
            raise
        self._prefix += content
        self._events.append(event.model_copy(deep=True))
        self._state = candidate.model_copy(deep=True)
        self._reentry = None
        return event.event_id, candidate

    def attach(self, artifact: Artifact) -> ArtifactPointer:
        if self._owner is not None and self._owner.read_only:
            raise ValueError("retained Journey view is read-only")
        pointer = artifact.pointer()
        relative = Path(pointer.location)
        destination = self.file.parent / relative
        from bio_neb.persistence import retain_immutable

        retain_immutable(destination, artifact.content)
        return pointer

    def read_artifact(self, pointer: ArtifactPointer) -> bytes:
        """Recover a bundle artifact, verifying its bytes.

        An admitted owner verifies each content address once and reuses those
        bytes (0029); an independent open or an offline read verifies afresh.
        """
        relative = Path(artifact_location(pointer.content_id))
        if pointer.location != relative.as_posix():
            raise ValueError("pointer is not a Journey bundle artifact")
        file = self.file.parent / relative
        if self._owner is not None:
            return self._owner.derived_once(
                ("bundle_artifact", pointer.content_id, pointer.bytes),
                partial(_verified_bundle_bytes, file, pointer),
            )
        return _verified_bundle_bytes(file, pointer)
