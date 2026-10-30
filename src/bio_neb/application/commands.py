"""Internal commands behind public Run advancement and focused deep seams."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import (
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from bio_neb.agentic_journey.continuation import review_answer_parents
from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationOverrides,
)
from bio_neb.agentic_journey.model_availability import (
    ModelAvailabilityReviewResponded,
    ModelAvailabilityReviewResponse,
    ModelAvailabilityReviewResponseReceipt,
    ModelAvailabilityStopped,
    StopModelAvailabilityResponse,
    evaluate_model_availability_review_response,
    project_model_availability_review,
)
from bio_neb.agentic_journey.operational_review import (
    AgentFailureRetryPreset,
    AgentFailureRetryPresetView,
    preset_agent_failure_retry,
)
from bio_neb.agentic_journey.runtime_config import (
    RuntimeConfigError,
    RuntimeSelections,
    read_runtime_config,
)
from bio_neb.application.input_journey import (
    JourneyStarted,
    RegisteredFastqInput,
    ResearchRequestRegistered,
    RunInputsRegistered,
    paired_fastq_roles,
    record_fact,
)
from bio_neb.contracts import FrozenContract
from bio_neb.failure import Invariant
from bio_neb.input_inspection.tool import is_staged_object, registered_locator
from bio_neb.investigation.investigation_request import (
    INVESTIGATION_REQUEST_MEDIA_TYPE,
    InvestigationRequest,
    InvestigationRequestRefused,
    InvestigationRequestRegistered,
    read_investigation_request_document,
)
from bio_neb.journey import FORK_ORIGIN_EVENT_TYPE, Journey, JourneyFile
from bio_neb.locations import local_state_root
from bio_neb.paper_intake.intake import register_paper_files
from bio_neb.parabricks.configuration import (
    ParabricksAccountRefused,
    ParabricksConfigurationRegistered,
    default_configuration,
)

if TYPE_CHECKING:
    pass


_REGISTERED_FASTQ_ROLE_ALIASES = {
    "read1": "read_1",
    "read2": "read_2",
}


def registered_fastq_role(value: str) -> str:
    """The role registration records for a supplied FASTQ role spelling."""
    return _REGISTERED_FASTQ_ROLE_ALIASES.get(value, value)


class RegisterFastqInput(FrozenContract):
    role: StrictStr
    registered_path: StrictStr

    @field_validator("role")
    @classmethod
    def canonical_role(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("role must be a non-empty canonical string")
        return registered_fastq_role(value)

    @field_validator("registered_path")
    @classmethod
    def exact_locator(cls, value: str) -> str:
        return registered_locator(value)


class RegisterRunCommand(FrozenContract):
    run_root: StrictStr
    prompt_file: StrictStr
    # None registers the build's defaults; a file overrides only what it names.
    runtime_config_file: StrictStr | None = None
    inputs: tuple[RegisterFastqInput, ...] = Field(min_length=1)
    # The researcher's SNP request files, each retained in order; none, the Run
    # later reports a request as needed.
    request_files: tuple[StrictStr, ...] = ()
    # The researcher's paper files, each its own Run paper, in order; the
    # coordinator reads them before planning.
    paper_files: tuple[StrictStr, ...] = Field(
        default=(), exclude_if=lambda paths: not paths
    )

    @field_validator("run_root", "prompt_file", "runtime_config_file")
    @classmethod
    def absolute_path(cls, value: str | None) -> str | None:
        if value is not None and (not value or not os.path.isabs(value)):
            raise ValueError("command paths must be absolute")
        return value

    @field_validator("request_files", "paper_files")
    @classmethod
    def absolute_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not path or not os.path.isabs(path) for path in value):
            raise ValueError("command paths must be absolute")
        return value

    @model_validator(mode="after")
    def unique_roles(self) -> RegisterRunCommand:
        roles = [item.role for item in self.inputs]
        if len(roles) != len(set(roles)):
            raise ValueError("registered FASTQ roles must be unique")
        return self


class RespondModelAvailabilityReviewCommand(FrozenContract):
    invocation_overrides: AgentInvocationOverrides = Field(
        default_factory=AgentInvocationOverrides
    )
    journey_file: JourneyFile
    request_event_id: StrictStr
    response: ModelAvailabilityReviewResponse
    # The answer-all shortcut: the same answer for every unanswered item of this
    # item's Review. Without it, only this item.
    every_item: bool = False

    @field_validator("request_event_id")
    @classmethod
    def canonical_request_identity(cls, value: str) -> str:
        try:
            parsed = uuid.UUID(value)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("request_event_id must be a UUID") from error
        if str(parsed) != value:
            raise ValueError("request_event_id must be canonical")
        return value


class RecordRetryPresetCommand(FrozenContract):
    journey_file: JourneyFile
    preset: AgentFailureRetryPreset


class RetryPresetRefused(ValueError):
    """The Run is not at a point a retry preset may be recorded; nothing was written."""

    def __init__(self, code: str, fields: tuple[str, ...]):
        super().__init__(code)
        self.code = code
        self.fields = fields


# The interval records that change no work: a preset may follow them.
_PRESET_NEUTRAL = frozenset({"PassStarted", "PassClosed", "AgentFailureRetryPreset"})


def record_retry_preset(
    command: RecordRetryPresetCommand,
) -> AgentFailureRetryPresetView:
    """Retain a researcher's retry preset for one Agent role's failure Reviews.

    The Journey is read whole first. A preset is recorded only where it changes
    no other work's frontier: before the Run's first work, at a Fork child's
    origin, or while only Agent failure Reviews wait (each answered at the next
    `resume`). Elsewhere it is refused before any write, since the next Review's
    or stage's exact re-entry reads the Journey's head.
    """
    from bio_neb.application.queries import ReviewJourneyQuery, review_journey
    from bio_neb.recording import read_evidence

    observation = review_journey(
        ReviewJourneyQuery(journey_file=str(command.journey_file))
    )
    journey = Journey.open(Path(command.journey_file))
    last_work = next(
        event
        for event in reversed(read_evidence(journey))
        if event["event_type"] not in _PRESET_NEUTRAL
    )
    if not (
        observation.status == "inputs_registered"
        or last_work["event_type"] == FORK_ORIGIN_EVENT_TYPE
        or (
            observation.status == "operational_review_pending"
            and all(review.kind == "junction" for review in observation.pending_reviews)
        )
    ):
        raise RetryPresetRefused(
            "retry_preset_placement_unsupported", ("journey_file",)
        )
    return preset_agent_failure_retry(journey, command.preset)


class ApprovedExecutionCommandRefused(Invariant, ValueError):
    """The compiler's reference admission contradicts retained evidence.

    Planning readiness already checked the supplied source against the plan, so
    this is an integrity fault: the failure module holds the Run (#217).
    """

    def __init__(self, code: str):
        super().__init__("approved_execution_command_refused", code, cause=code)
        self.code = code


class RegistrationReceipt(FrozenContract):
    journey_file: StrictStr
    journey_id: StrictStr
    request_event_id: StrictStr
    registration_event_id: StrictStr


class RegistrationError(ValueError):
    def __init__(self, code: str, fields: tuple[str, ...]):
        super().__init__(code)
        self.code = code
        self.fields = fields

    def facts(self) -> dict[str, Any]:
        return {"code": self.code, "fields": list(self.fields)}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _inside(path: str, directory: Path) -> bool:
    try:
        selected = Path(path).resolve(strict=False)
        root = directory.resolve(strict=True)
        return os.path.normcase(
            os.path.commonpath((selected, root))
        ) == os.path.normcase(str(root))
    except (OSError, ValueError):
        return False


def _read_prompt(path: str) -> str:
    try:
        return Path(path).read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise RegistrationError("prompt_invalid", ("prompt_file",)) from error


def _read_registration_configuration(path: str | None):
    try:
        configuration = read_runtime_config(None if path is None else Path(path))
    except RuntimeConfigError as error:
        fields = tuple(
            "runtime_config_file" if field == "$" else field
            for field in error.violations
        )
        raise RegistrationError("runtime_selections_invalid", fields) from error
    return configuration


def _read_request(path: str) -> tuple[bytes, InvestigationRequest]:
    try:
        return read_investigation_request_document(path)
    except InvestigationRequestRefused as error:
        raise RegistrationError(error.code, error.fields) from error


def _require_model_credentials(
    selections: RuntimeSelections, environ: Mapping[str, str]
) -> None:
    """Every model a role selects names a credential variable ``environ`` sets."""
    missing = tuple(
        f"model_roster.{reference}.credential_env"
        for reference, name in selections.selected_credential_envs().items()
        if not environ.get(name)
    )
    if missing:
        raise RegistrationError("model_credential_unavailable", missing)


def register_run(
    command: RegisterRunCommand,
    *,
    repository_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    parabricks_account: Path | None = None,
) -> RegistrationReceipt:
    """Validate external inputs, then atomically create the initial Journey facts.

    ``environ``, given when a real model adapter will run, must set each selected
    model's credential variable: a missing one is refused by name before any
    Journey exists (owner, #440 Q5). Its value is never read into the Run.
    A configuration with no ``parabricks`` section registers the one composed from
    ``parabricks_account`` and the Run root; a missing or invalid account file
    registers none, so the Run stops at its preparation Review (owner, #454).
    """

    repo = repository_root or _repository_root()
    # Journey evidence stays out of the source tree; the gitignored local-state
    # directory is the one sanctioned place for it inside a checkout.
    if _inside(command.run_root, repo) and not _inside(
        command.run_root, local_state_root()
    ):
        raise RegistrationError("run_root_invalid", ("run_root",))
    prompt = _read_prompt(command.prompt_file)
    configuration = _read_registration_configuration(command.runtime_config_file)
    selections = configuration.runtime_selections()
    if environ is not None:
        _require_model_credentials(selections, environ)
    request_documents = [_read_request(path) for path in command.request_files]
    if not paired_fastq_roles(item.role for item in command.inputs):
        raise RegistrationError("fastq_roles_unsupported", ("inputs",))
    parabricks = configuration.parabricks
    if parabricks is None and parabricks_account is not None:
        try:
            parabricks = default_configuration(parabricks_account, command.run_root)
        except ParabricksAccountRefused:
            parabricks = None
    # A staged object is inspected and copied through the Run's object storage.
    if parabricks is None and any(
        is_staged_object(item.registered_path) for item in command.inputs
    ):
        raise RegistrationError("staged_inputs_need_object_storage", ("inputs",))

    journey = Journey.create(Path(command.run_root))
    started = record_fact(journey, JourneyStarted())
    request = record_fact(
        journey,
        ResearchRequestRegistered(
            prompt=prompt,
            prompt_file=command.prompt_file,
        ),
        (started["event_id"],),
    )
    registration = record_fact(
        journey,
        RunInputsRegistered(
            input_set_id=str(uuid.uuid4()),
            run_root=command.run_root,
            runtime_config_file=command.runtime_config_file,
            runtime_selections=selections,
            inputs=tuple(
                RegisteredFastqInput(
                    role=item.role, registered_path=item.registered_path
                )
                for item in command.inputs
            ),
        ),
        (request["event_id"],),
    )
    for path, (content, investigation_request) in zip(
        command.request_files, request_documents, strict=True
    ):
        captured = InvestigationRequestRegistered(
            request_file=path, request=investigation_request
        )
        # The exact bytes are retained before the fact that names them.
        journey.record_with_artifact(
            captured.event_type,
            captured.model_dump(mode="json", exclude={"event_type"}),
            content,
            INVESTIGATION_REQUEST_MEDIA_TYPE,
            [registration["event_id"]],
        )
    register_paper_files(journey, command.paper_files, registration["event_id"])
    if parabricks is not None:
        configured = ParabricksConfigurationRegistered(configuration=parabricks)
        journey.record(
            configured.event_type,
            configured.model_dump(mode="json", exclude={"event_type"}),
            [registration["event_id"]],
        )
    return RegistrationReceipt(
        journey_file=str(journey.path),
        journey_id=journey.journey_id,
        request_event_id=request["event_id"],
        registration_event_id=registration["event_id"],
    )


def respond_model_availability_review(
    command: RespondModelAvailabilityReviewCommand,
    *,
    environ: Mapping[str, str],
    attempt: Any | None = None,
) -> ModelAvailabilityReviewResponseReceipt:
    """Bind one scoped retry, instruction amendment, or stop direction.

    One keyed Review per cause holds every item with that cause (owner, #476
    alignment G4.1); each item is answered on its own, and ``every_item`` gives
    the same answer to every unanswered item of its Review (owner, #477
    6483312280).
    """

    journey = Journey.open(Path(command.journey_file))
    from bio_neb.recording import read_evidence

    projection = project_model_availability_review(read_evidence(journey))
    answered = next(
        (
            item
            for item in projection.requests
            if item.request_event_id == command.request_event_id
        ),
        None,
    )
    lead = (
        None
        if answered is None
        else answered.joins_event_id or answered.request_event_id
    )
    others = [
        item.request_event_id
        for item in projection.requests
        if command.every_item
        and item.pending
        and item.request_event_id != command.request_event_id
        and (item.joins_event_id or item.request_event_id) == lead
    ]
    retained = evaluate_model_availability_review_response(
        projection, command.request_event_id, command.response
    )
    from bio_neb.application.queries import ReviewJourneyQuery, review_journey

    review_journey(ReviewJourneyQuery(journey_file=str(journey.path)))
    if retained is not None:
        return retained
    pending = next(
        item
        for item in projection.requests
        if item.request_event_id == command.request_event_id and item.pending
    )
    selected = None
    if command.response.action in {"retry", "amend"}:
        from bio_neb.application.agent_reentry import (
            coordinator_assignments,
            reentry_assignment,
        )

        events = read_evidence(journey)
        prior_start = next(
            e
            for e in events
            if e["event_type"] == "AttemptStarted"
            and e["payload"]["attempt_id"] == pending.attempt_id
        )
        retained_overrides = AgentInvocationOverrides.model_validate(
            prior_start["payload"]["invocation_overrides"]
        )
        if (
            command.invocation_overrides.model_fields_set
            and command.invocation_overrides != retained_overrides
        ):
            raise ValueError("provider retry cannot amend pending invocation policy")
        # The checked request binds its retry frontier to the stopped Attempt's
        # own start; the selected assignment re-checks its stage eligibility.
        selected = reentry_assignment(
            events,
            pending.attempt_id,
            (attempt,) if attempt is not None else coordinator_assignments(),
        )
    fact = ModelAvailabilityReviewResponded(
        request_event_id=command.request_event_id,
        response=command.response,
    )
    payload = fact.model_dump(mode="json")
    payload.pop("event_type")
    response_event_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{journey.journey_id}:{command.request_event_id}:model-availability-review-response",
        )
    )
    recorded = journey.record(
        fact.event_type,
        payload,
        review_answer_parents(journey.read(), command.request_event_id),
        event_id=response_event_id,
    )
    receipt = ModelAvailabilityReviewResponseReceipt(
        request_event_id=command.request_event_id,
        response_event_id=recorded["event_id"],
        response=command.response,
    )
    if isinstance(command.response, StopModelAvailabilityResponse):
        rejected = ModelAvailabilityStopped(
            request_event_id=command.request_event_id,
            response_event_id=recorded["event_id"],
            reason=command.response.reason,
        )
        rejected_payload = rejected.model_dump(mode="json")
        rejected_payload.pop("event_type")
        journey.record(rejected.event_type, rejected_payload, [recorded["event_id"]])
    else:
        assert selected is not None
        selected.run(
            journey,
            environ=environ,
            availability_review_response_event_id=recorded["event_id"],
            invocation_overrides=command.invocation_overrides,
        )
    for request_event_id in others:
        respond_model_availability_review(
            command.model_copy(
                update={"request_event_id": request_event_id, "every_item": False}
            ),
            environ=environ,
            attempt=attempt,
        )
    review_journey(ReviewJourneyQuery(journey_file=str(journey.path)))
    return receipt
